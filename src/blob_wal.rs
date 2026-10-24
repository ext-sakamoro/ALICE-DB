//! Write-ahead log for the blob key-value store (v0.2.0-alpha.3).
//!
//! `BlobWal` is an append-only file backing the in-memory
//! [`crate::blob::BlobStorage`]. Every `put_blob` and `delete_blob`
//! invocation writes a checksum-guarded record here before mutating the
//! in-memory map, so a subsequent process can rebuild identical state
//! from the log alone.
//!
//! # Record layout
//!
//! Each record is framed as little-endian bytes. Multi-byte integers are
//! LE so the WAL is portable across host endianness.
//!
//! ```text
//! offset  size  field           description
//! ------  ----  --------------  ------------------------------------------------
//!  0      1     type            0x01 = Put, 0x02 = Delete
//!  1      4     key_len (u32)   payload key length in bytes
//!  5      4     value_len (u32) payload value length; 0 for Delete
//!  9      1     value_kind      0x00 = Raw, 0x01 = Compressed; ignored on Delete
//! 10      N     key             key_len bytes
//! 10+N    M     value           value_len bytes (absent on Delete)
//! trailer 4     crc32           CRC-32 over everything from offset 0 upward
//! ```
//!
//! The checksum is CRC-32 (IEEE 802.3, reflected polynomial `0xEDB88320`,
//! as computed by `crc32fast`), not CRC-32C.
//!
//! # Truncation and recovery
//!
//! During [`BlobWal::replay`]:
//! - A short read at any point past a full record boundary is treated as
//!   a clean end-of-log (e.g. the process crashed mid-write). All fully
//!   framed records preceding the truncation are yielded.
//! - A record whose CRC-32 does not match its payload is treated as
//!   corruption: the record is dropped and replay stops there (later
//!   records may exist but their offsets are no longer trustworthy).
//! - An empty or missing WAL file yields no records; the store opens
//!   with an empty state.
//!
//! [`BlobWal::recover`] (what [`crate::blob::BlobStorage`] calls on open)
//! reads the same records and then makes the file end at the last good
//! record, so the next append is not written behind bytes that replay
//! will stop at again. It first copies the bytes it drops to
//! `<wal>.dropped-<offset>` (written through `.tmp`, fsynced, renamed,
//! the directory fsynced), then cuts the WAL and fsyncs it. A crash
//! between the two steps leaves the copy and the uncut WAL, and the next
//! open does the same cut again, reusing the copy that already holds
//! exactly those bytes. A later recovery at the same offset with other
//! bytes never replaces an earlier copy: it writes
//! `<wal>.dropped-<offset>.<n>` with the lowest free `n`. Stale `.tmp`
//! files of an interrupted copy are removed on open; finished copies are
//! never removed. A WAL that ends on a record boundary is never touched.
//!
//! # Concurrency and durability
//!
//! [`BlobWal::append_put`] / [`BlobWal::append_delete`] serialise the
//! record and write it in a single `write_all`. When and whether the
//! file handle is subsequently `sync_data`'d is controlled by the
//! [`SyncPolicy`] chosen at open time:
//!
//! - [`SyncPolicy::EveryWrite`] (default) — fsync after every record.
//!   Matches alpha-2 behaviour; strongest crash-recovery guarantee, one
//!   fsync per op.
//! - [`SyncPolicy::Batched`] — fsync only once every `max_pending_ops`
//!   records. Trades a bounded window of the newest writes for
//!   throughput. Callers may still force an fsync with
//!   [`BlobWal::flush`].
//! - [`SyncPolicy::Manual`] — never auto-fsync; the caller is fully
//!   responsible for durability via [`BlobWal::flush`]. Intended for
//!   bulk-load pipelines that fsync exactly once at the end.
//!
//! # Cross-process locking
//!
//! On open the WAL takes an exclusive [`fs2`] advisory lock on the file
//! handle. A second process (or a second call in the same process)
//! attempting to open the same WAL file receives a
//! `WouldBlock`-flavoured `io::Error`. Alpha-3 documents this as the
//! canonical way to detect a running writer; multi-writer coordination
//! (leases, coordinator process) is out of scope for alpha.

use alice_serial::bounded::len_within;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use parking_lot::Mutex;

use crate::blob::BlobValue;

/// How aggressively the WAL should `fsync` after each append.
///
/// Introduced in v0.2.0-alpha.3. Alpha-2 was hard-wired to the
/// [`Self::EveryWrite`] behaviour; existing callers of
/// [`BlobWal::open`] and [`crate::blob::BlobStorage::open`] get that
/// default unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub enum SyncPolicy {
    /// `sync_data` after every record. Every append is durable by the
    /// time it returns; matches alpha-2 semantics.
    #[default]
    EveryWrite,
    /// `sync_data` only after `max_pending_ops` unsynced records have
    /// been buffered. A power loss can lose the last
    /// `max_pending_ops - 1` records. Callers can force an fsync
    /// earlier via [`BlobWal::flush`].
    Batched {
        /// Maximum number of unsynced records before an implicit fsync
        /// is issued.
        max_pending_ops: usize,
    },
    /// Never auto-fsync. Durability is entirely delegated to explicit
    /// [`BlobWal::flush`] calls. Suitable for one-shot bulk loading.
    Manual,
}

/// Record type tag: `put` or `delete`.
const RECORD_TYPE_PUT: u8 = 0x01;
const RECORD_TYPE_DELETE: u8 = 0x02;

/// Value kind tag inside a Put record.
const VALUE_KIND_RAW: u8 = 0x00;
const VALUE_KIND_COMPRESSED: u8 = 0x01;

/// Fixed prefix before the variable-length key/value bytes:
///   1 (type) + 4 (`key_len`) + 4 (`value_len`) + 1 (`value_kind`)
const HEADER_LEN: usize = 10;

/// CRC-32 (IEEE) trailer.
const CRC_LEN: usize = 4;

/// A single logical operation reconstructed from the WAL.
///
/// Emitted in file order by [`BlobWal::replay`]; callers apply them
/// left-to-right to arrive at the same in-memory state the original
/// writer would have produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalRecord {
    /// Insert or overwrite `key` with the exact stored representation.
    /// The `BlobValue` variant is preserved verbatim (no re-encoding).
    Put(Vec<u8>, BlobValue),
    /// Tombstone a key. The corresponding in-memory entry becomes
    /// invisible to readers.
    Delete(Vec<u8>),
}

/// Why replay stopped before the end of the WAL file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WalTail {
    /// The file ends inside a record (a write that did not finish).
    Torn,
    /// A complete record whose CRC-32 does not match its bytes. Nothing
    /// after it is read, since its lengths are not trustworthy.
    BadChecksum,
}

/// What [`BlobWal::recover`] cut off the end of the WAL.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WalRecovery {
    /// Length the WAL was cut to: the end of the last good record.
    pub offset: u64,
    /// Number of bytes dropped from `offset` to the old end of the file.
    pub dropped_len: u64,
    /// Why replay stopped at `offset`.
    pub tail: WalTail,
    /// File holding exactly the dropped bytes (`<wal>.dropped-<offset>`,
    /// or `<wal>.dropped-<offset>.<n>` when an earlier copy with other
    /// bytes already has that name).
    pub dropped_copy: PathBuf,
}

/// Append-only write-ahead log for the blob store.
///
/// Wraps a `File` behind a `parking_lot::Mutex` so multi-threaded writers
/// serialise their appends. The `Mutex` also guards the `sync_data`
/// call and the pending-write counter used by
/// [`SyncPolicy::Batched`], so callers of `append_*` observe atomic
/// state transitions.
#[derive(Debug)]
pub struct BlobWal {
    path: PathBuf,
    state: Mutex<WalState>,
    sync_policy: SyncPolicy,
}

/// Whether the WAL handle is opened with the append flag. It is not:
/// Windows maps an append handle to `FILE_APPEND_DATA` without
/// `FILE_WRITE_DATA`, which makes [`BlobWal::truncate`] fail with
/// `ERROR_ACCESS_DENIED`. Kept as a named constant so the open-file
/// audit checks the same value the open call uses.
const OPEN_WITH_APPEND: bool = false;

/// Interior state guarded by the outer `Mutex`. Kept separate so a
/// single `lock()` yields access to both the file handle and the
/// batched-write counter.
#[derive(Debug)]
struct WalState {
    file: File,
    /// Number of appended records not yet reflected on disk via
    /// `sync_data`. `EveryWrite` keeps this at zero because it fsyncs
    /// unconditionally; `Batched` and `Manual` allow it to grow.
    pending_ops: usize,
}

impl BlobWal {
    /// Open (or create) the WAL at `path` with the default
    /// [`SyncPolicy::EveryWrite`].
    ///
    /// Missing parent directories are created. The file is opened in
    /// read + write mode so replay reads, appends, and truncation share
    /// a single handle; every append seeks to the end explicitly (the
    /// handle deliberately lacks the `append` flag so that truncation
    /// also works on Windows). An exclusive `fs2` advisory lock is taken so a
    /// concurrent open sees a `WouldBlock` error.
    ///
    /// # Errors
    /// Returns `io::Error` if directory creation, file open, or lock
    /// acquisition fails.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with_policy(path, SyncPolicy::default())
    }

    /// Same as [`Self::open`] but with an explicit [`SyncPolicy`].
    ///
    /// # Errors
    /// See [`Self::open`].
    pub fn open_with_policy(path: impl AsRef<Path>, sync_policy: SyncPolicy) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Opened read + write, *not* append: on Windows an append handle
        // carries FILE_APPEND_DATA without FILE_WRITE_DATA, and
        // `truncate` (`set_len`) then fails with ERROR_ACCESS_DENIED.
        // Every write seeks to the end first instead (see `write_frame`).
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .append(OPEN_WITH_APPEND)
            .create(true)
            .truncate(false)
            .open(&path)?;
        // Exclusive advisory lock. If another process (or the same
        // process) already holds the lock we bail with a descriptive
        // error so the caller can surface it as "database already open"
        // rather than a mysterious later corruption.
        FileExt::try_lock_exclusive(&file).map_err(|e| {
            // the OS reports contention differently (EWOULDBLOCK on Unix,
            // ERROR_LOCK_VIOLATION on Windows); callers match on WouldBlock
            let kind = if e.raw_os_error().is_some()
                && e.raw_os_error() == fs2::lock_contended_error().raw_os_error()
            {
                io::ErrorKind::WouldBlock
            } else {
                e.kind()
            };
            io::Error::new(
                kind,
                format!(
                    "blob WAL at `{}` is already locked by another writer: {e}",
                    path.display()
                ),
            )
        })?;
        crate::fs_audit::note_open(&path);
        Ok(Self {
            path,
            state: Mutex::new(WalState {
                file,
                pending_ops: 0,
            }),
            sync_policy,
        })
    }

    /// The sync policy in effect for this WAL. Handy for tests and
    /// diagnostic output.
    #[must_use]
    pub fn sync_policy(&self) -> SyncPolicy {
        self.sync_policy
    }

    /// Force an fsync (`sync_data`) of any pending writes.
    ///
    /// A no-op if the file has no unsynced records — callers can invoke
    /// this unconditionally at shutdown without paying a syscall in the
    /// common `EveryWrite` case.
    ///
    /// # Errors
    /// Propagates the underlying `sync_data` error.
    pub fn flush(&self) -> io::Result<()> {
        let mut state = self.state.lock();
        if state.pending_ops == 0 {
            return Ok(());
        }
        crate::fs_audit::sync_data(&state.file, "wal", &self.path)?;
        state.pending_ops = 0;
        Ok(())
    }

    /// Current on-disk size of the WAL in bytes.
    ///
    /// Used by [`crate::blob::BlobStorage`] to decide whether the WAL has
    /// grown past the configured flush threshold. Reflects only bytes
    /// that have already reached the kernel; buffered writes on their
    /// way to `write_all` are not counted (the mutex serialises writes
    /// so this ambiguity never affects a single caller).
    ///
    /// # Errors
    /// Propagates the underlying `metadata` error.
    pub fn size_on_disk(&self) -> io::Result<u64> {
        let state = self.state.lock();
        state.file.metadata().map(|m| m.len())
    }

    /// Truncate the WAL back to zero bytes.
    ///
    /// Intended to be called immediately after a successful `SSTable`
    /// rewrite: the "settled" bytes now live in `blob.sst` and the WAL
    /// should start recording the next diff from empty. The pending
    /// counter is reset because there is nothing to fsync after the
    /// truncate returns.
    ///
    /// # Errors
    /// Propagates any `seek` / `set_len` / `sync_all` error.
    pub fn truncate(&self) -> io::Result<()> {
        let mut state = self.state.lock();
        crate::fs_audit::check_set_len(&self.path, OPEN_WITH_APPEND);
        state.file.seek(SeekFrom::Start(0))?;
        state.file.set_len(0)?;
        // sync_all here so the truncation reaches disk before we
        // consider the WAL "empty" from the caller's point of view.
        crate::fs_audit::sync_all(&state.file, "wal", &self.path)?;
        state.pending_ops = 0;
        Ok(())
    }

    /// The on-disk location of the WAL. Handy for diagnostics.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Serialise a `put` record and durably append it.
    ///
    /// # Errors
    /// Any I/O failure from the underlying write or fsync propagates.
    pub fn append_put(&self, key: &[u8], value: &BlobValue) -> io::Result<()> {
        let (kind, bytes): (u8, &[u8]) = match value {
            BlobValue::Raw(b) => (VALUE_KIND_RAW, b.as_slice()),
            BlobValue::Compressed(b) => (VALUE_KIND_COMPRESSED, b.as_slice()),
            BlobValue::Tombstone => {
                // Callers must convert Tombstone -> Delete before
                // reaching the WAL; hitting this branch is a bug.
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "BlobWal::append_put received a Tombstone value; use append_delete",
                ));
            }
        };
        let frame = serialise_record(RECORD_TYPE_PUT, key, kind, bytes)?;
        self.write_frame(&frame)
    }

    /// Serialise a `delete` record and durably append it.
    ///
    /// # Errors
    /// Any I/O failure from the underlying write or fsync propagates.
    pub fn append_delete(&self, key: &[u8]) -> io::Result<()> {
        let frame = serialise_record(RECORD_TYPE_DELETE, key, VALUE_KIND_RAW, &[])?;
        self.write_frame(&frame)
    }

    fn write_frame(&self, frame: &[u8]) -> io::Result<()> {
        let mut state = self.state.lock();
        // The handle is not opened in append mode (see `open_with_policy`),
        // and `replay` leaves the cursor wherever it stopped reading, so
        // position at the end explicitly. The mutex makes seek + write
        // one step with respect to other writers in this process, and
        // the exclusive file lock rules out writers in other processes.
        state.file.seek(SeekFrom::End(0))?;
        state.file.write_all(frame)?;
        state.pending_ops += 1;
        match self.sync_policy {
            SyncPolicy::EveryWrite => {
                crate::fs_audit::sync_data(&state.file, "wal", &self.path)?;
                state.pending_ops = 0;
            }
            SyncPolicy::Batched { max_pending_ops } => {
                if state.pending_ops >= max_pending_ops {
                    crate::fs_audit::sync_data(&state.file, "wal", &self.path)?;
                    state.pending_ops = 0;
                }
            }
            SyncPolicy::Manual => {
                // Never auto-sync; caller invokes flush() explicitly.
            }
        }
        Ok(())
    }

    /// Read the entire WAL and yield every well-framed record in file order.
    ///
    /// Truncated tails are treated as clean end-of-log; corrupted (bad
    /// CRC) records stop replay early. See the module docs for the
    /// contract.
    ///
    /// # Errors
    /// Any read error (other than end-of-file, which is expected)
    /// propagates.
    pub fn replay(&self) -> io::Result<Vec<WalRecord>> {
        let mut state = self.state.lock();
        // Seek to start; `write_frame` seeks back to the end before each
        // write, so wherever we leave the read cursor does not matter.
        let file_len = state.file.metadata()?.len();
        state.file.seek(SeekFrom::Start(0))?;
        let mut reader = Counting {
            inner: BufReader::new(&state.file),
            pos: 0,
        };

        let mut out = Vec::new();
        while let ReadOutcome::Record(rec) = read_one(&mut reader, file_len)? {
            out.push(rec);
        }
        Ok(out)
    }

    /// Replay the WAL like [`Self::replay`], then cut a torn or corrupt
    /// tail off the file so later appends follow the last good record.
    ///
    /// The dropped bytes are first written to `<wal>.dropped-<offset>`
    /// and made durable (file and directory fsync), then the WAL is cut
    /// and fsynced. Returns the records and, when something was cut, the
    /// [`WalRecovery`] describing it. A WAL that ends on a record
    /// boundary is left as it is.
    ///
    /// # Errors
    /// Propagates read, write, rename, `set_len` and fsync errors. On an
    /// error the WAL is either uncut or cut after the copy is durable.
    pub fn recover(&self) -> io::Result<(Vec<WalRecord>, Option<WalRecovery>)> {
        let mut state = self.state.lock();
        remove_stale_dropped_tmp(&self.path)?;
        let file_len = state.file.metadata()?.len();
        state.file.seek(SeekFrom::Start(0))?;
        let mut reader = Counting {
            inner: BufReader::new(&state.file),
            pos: 0,
        };
        let mut out = Vec::new();
        let mut good_end = 0_u64;
        let tail = loop {
            match read_one(&mut reader, file_len)? {
                ReadOutcome::Record(rec) => {
                    out.push(rec);
                    good_end = reader.pos;
                }
                ReadOutcome::Eof => break None,
                ReadOutcome::Truncated => break Some(WalTail::Torn),
                ReadOutcome::BadChecksum => break Some(WalTail::BadChecksum),
            }
        };
        drop(reader);
        let Some(tail) = tail else {
            return Ok((out, None));
        };

        let mut dropped = Vec::new();
        state.file.seek(SeekFrom::Start(good_end))?;
        state.file.read_to_end(&mut dropped)?;
        let dropped_copy = keep_dropped(&self.path, good_end, &dropped)?;
        crate::fs_audit::crash_point("wal_recover:before_cut")?;

        crate::fs_audit::check_set_len(&self.path, OPEN_WITH_APPEND);
        state.file.set_len(good_end)?;
        crate::fs_audit::sync_all(&state.file, "wal", &self.path)?;
        Ok((
            out,
            Some(WalRecovery {
                offset: good_end,
                dropped_len: dropped.len() as u64,
                tail,
                dropped_copy,
            }),
        ))
    }
}

/// The prefix every dropped copy of `wal` starts with: `<wal>.dropped-`.
fn dropped_prefix(wal: &Path) -> std::ffi::OsString {
    let mut name = wal
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    name.push(".dropped-");
    name
}

/// The candidate names for the copy of the bytes dropped at `offset`, in
/// order: `<wal>.dropped-<offset>`, then `<wal>.dropped-<offset>.<n>` for
/// n = 1, 2, ...
fn dropped_copy_name(wal: &Path, offset: u64, n: u64) -> PathBuf {
    let mut name = dropped_prefix(wal);
    name.push(offset.to_string());
    if n > 0 {
        name.push(format!(".{n}"));
    }
    wal.with_file_name(name)
}

/// Make `bytes` durable as the copy of what recovery drops at `offset`,
/// and return its path. A copy that already holds exactly `bytes` (the
/// same recovery redone after a crash before the cut) is reused; a copy
/// with other bytes (an earlier recovery at the same offset) is never
/// replaced, and the new copy takes the lowest free `.<n>` name. The
/// caller holds the WAL lock, so no other writer picks a name meanwhile.
fn keep_dropped(wal: &Path, offset: u64, bytes: &[u8]) -> io::Result<PathBuf> {
    let mut n = 0;
    loop {
        let path = dropped_copy_name(wal, offset, n);
        match std::fs::read(&path) {
            Ok(existing) if existing == bytes => return Ok(path),
            Ok(_) => n += 1,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                write_durable(&path, bytes)?;
                return Ok(path);
            }
            Err(e) => return Err(e),
        }
    }
}

/// Write `bytes` to the free name `path` through `<path>.tmp` (created
/// with `create_new`): write, fsync, rename, then fsync the directory so
/// the name survives a crash too.
fn write_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp_name = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(bytes)?;
        crate::fs_audit::sync_all(&f, "file", &tmp)?;
    }
    crate::fs_audit::check_replace_target(path);
    std::fs::rename(&tmp, path)?;
    sync_dir(path)
}

/// Remove `<wal>.dropped-*.tmp` left by a copy that was interrupted
/// before its rename. Finished copies (no `.tmp`) are never removed. The
/// caller holds the WAL lock, so no copy is being written meanwhile.
fn remove_stale_dropped_tmp(wal: &Path) -> io::Result<()> {
    let dir = match wal.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let prefix = dropped_prefix(wal);
    let prefix = prefix.to_string_lossy();
    let entries = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        // exactly the lowercase `.tmp` recovery writes (a `.TMP` is not ours)
        if name.starts_with(prefix.as_ref())
            && Path::new(name).extension() == Some(std::ffi::OsStr::new("tmp"))
        {
            let path = entry.path();
            crate::fs_audit::check_remove(&path);
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

/// fsync the directory holding `path` (Unix; Windows has no handle for a
/// directory fsync and makes the rename durable through the file system's
/// own metadata journal).
fn sync_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        let dir = if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        };
        crate::fs_audit::sync_all(&File::open(dir)?, "dir", dir)?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// A reader that counts the bytes it has handed out, so recovery knows
/// where the last good record ends.
struct Counting<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Drop for BlobWal {
    fn drop(&mut self) {
        // Release the advisory lock and flush any pending writes. We
        // deliberately swallow errors here — the process is either
        // exiting normally (where the lock is released implicitly by
        // handle close) or in a panic path (where surfacing a secondary
        // error would obscure the primary panic).
        let state = self.state.lock();
        if state.pending_ops > 0 {
            let _ = crate::fs_audit::sync_data(&state.file, "wal", &self.path);
        }
        let _ = FileExt::unlock(&state.file);
        crate::fs_audit::note_close(&self.path);
    }
}

/// Serialise one record: [header][key][value?][CRC-32].
fn serialise_record(
    record_type: u8,
    key: &[u8],
    value_kind: u8,
    value: &[u8],
) -> io::Result<Vec<u8>> {
    let key_len: u32 = u32::try_from(key.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "key exceeds u32::MAX"))?;
    let value_len: u32 = u32::try_from(value.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "value exceeds u32::MAX"))?;

    let payload_len = HEADER_LEN + key.len() + value.len();
    let mut buf = Vec::with_capacity(payload_len + CRC_LEN);
    buf.push(record_type);
    buf.extend_from_slice(&key_len.to_le_bytes());
    buf.extend_from_slice(&value_len.to_le_bytes());
    buf.push(value_kind);
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&buf);
    let crc = hasher.finalize();
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

/// Outcome of trying to read one record from the WAL.
enum ReadOutcome {
    Record(WalRecord),
    Eof,
    Truncated,
    BadChecksum,
}

/// One record from `reader`, which is at byte `reader.pos` of a WAL of
/// `file_len` bytes. The key and value lengths in the header are claims
/// about the bytes that follow: each is checked against the bytes that
/// remain before anything is allocated for it, and a record that claims
/// more than remains is a torn tail (`Truncated`), like one cut short.
fn read_one<R: Read>(reader: &mut Counting<R>, file_len: u64) -> io::Result<ReadOutcome> {
    let mut header = [0_u8; HEADER_LEN];
    match read_exact_or_eof(reader, &mut header)? {
        ReadResult::Full => {}
        ReadResult::Eof => return Ok(ReadOutcome::Eof),
        ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
    }

    let record_type = header[0];
    let key_claim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]);
    let value_claim = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
    let value_kind = header[9];

    let remaining = usize::try_from(file_len.saturating_sub(reader.pos)).unwrap_or(usize::MAX);
    let Ok(key_len) = len_within(u64::from(key_claim), 1, remaining) else {
        return Ok(ReadOutcome::Truncated);
    };
    let Ok(value_len) = len_within(u64::from(value_claim), 1, remaining - key_len) else {
        return Ok(ReadOutcome::Truncated);
    };

    let mut key = vec![0_u8; key_len];
    match read_exact_or_eof(reader, &mut key)? {
        ReadResult::Full => {}
        ReadResult::Eof | ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
    }

    let mut value = vec![0_u8; value_len];
    if value_len > 0 {
        match read_exact_or_eof(reader, &mut value)? {
            ReadResult::Full => {}
            ReadResult::Eof | ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
        }
    }

    let mut crc_bytes = [0_u8; CRC_LEN];
    match read_exact_or_eof(reader, &mut crc_bytes)? {
        ReadResult::Full => {}
        ReadResult::Eof | ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
    }
    let expected_crc = u32::from_le_bytes(crc_bytes);

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(&key);
    hasher.update(&value);
    if hasher.finalize() != expected_crc {
        return Ok(ReadOutcome::BadChecksum);
    }

    match record_type {
        RECORD_TYPE_PUT => {
            let value = match value_kind {
                VALUE_KIND_RAW => BlobValue::Raw(value),
                VALUE_KIND_COMPRESSED => BlobValue::Compressed(value),
                _ => return Ok(ReadOutcome::BadChecksum),
            };
            Ok(ReadOutcome::Record(WalRecord::Put(key, value)))
        }
        RECORD_TYPE_DELETE => Ok(ReadOutcome::Record(WalRecord::Delete(key))),
        _ => Ok(ReadOutcome::BadChecksum),
    }
}

enum ReadResult {
    Full,
    Eof,
    Truncated,
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<ReadResult> {
    let mut read = 0;
    while read < buf.len() {
        match reader.read(&mut buf[read..]) {
            Ok(0) => {
                return Ok(if read == 0 {
                    ReadResult::Eof
                } else {
                    ReadResult::Truncated
                });
            }
            Ok(n) => read += n,
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(ReadResult::Full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn wal_path(dir: &TempDir) -> PathBuf {
        dir.path().join("blob.wal")
    }

    #[test]
    fn roundtrip_put_and_delete_records() {
        let tmp = TempDir::new().unwrap();
        let wal = BlobWal::open(wal_path(&tmp)).unwrap();
        wal.append_put(b"foo", &BlobValue::Raw(b"hello".to_vec()))
            .unwrap();
        wal.append_put(b"bar", &BlobValue::Compressed(vec![0x78, 0x9c, 0x03, 0x00]))
            .unwrap();
        wal.append_delete(b"foo").unwrap();

        let records = wal.replay().unwrap();
        assert_eq!(
            records,
            vec![
                WalRecord::Put(b"foo".to_vec(), BlobValue::Raw(b"hello".to_vec())),
                WalRecord::Put(
                    b"bar".to_vec(),
                    BlobValue::Compressed(vec![0x78, 0x9c, 0x03, 0x00]),
                ),
                WalRecord::Delete(b"foo".to_vec()),
            ]
        );
    }

    #[test]
    fn append_put_rejects_tombstone_variant() {
        let tmp = TempDir::new().unwrap();
        let wal = BlobWal::open(wal_path(&tmp)).unwrap();
        let err = wal
            .append_put(b"foo", &BlobValue::Tombstone)
            .expect_err("Tombstone must be routed through append_delete");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn empty_wal_yields_no_records() {
        let tmp = TempDir::new().unwrap();
        let wal = BlobWal::open(wal_path(&tmp)).unwrap();
        let records = wal.replay().unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn truncated_tail_is_dropped_cleanly() {
        // Manually build two full records and half of a third, then
        // verify replay yields exactly the two intact ones.
        let tmp = TempDir::new().unwrap();
        let path = wal_path(&tmp);
        let wal = BlobWal::open(&path).unwrap();
        wal.append_put(b"k1", &BlobValue::Raw(b"v1".to_vec()))
            .unwrap();
        wal.append_put(b"k2", &BlobValue::Raw(b"v2".to_vec()))
            .unwrap();
        // Release the exclusive advisory lock before we mutate the file
        // out-of-band and reopen it. Shadowing alone does not drop the
        // previous binding.
        drop(wal);
        // Truncate the file at the size of ~1.5 records.
        let full_size = std::fs::metadata(&path).unwrap().len();
        // Each of our records is HEADER_LEN + 2 (key) + 2 (value) + CRC_LEN = 18 bytes.
        // Two full records = 36 bytes. Truncate to 27 to slice the second record.
        assert_eq!(full_size, 36);
        {
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(27).unwrap();
        }
        let wal = BlobWal::open(&path).unwrap();
        let records = wal.replay().unwrap();
        assert_eq!(
            records,
            vec![WalRecord::Put(
                b"k1".to_vec(),
                BlobValue::Raw(b"v1".to_vec())
            )]
        );
    }

    #[test]
    fn corrupted_checksum_stops_replay_at_that_record() {
        let tmp = TempDir::new().unwrap();
        let path = wal_path(&tmp);
        let wal = BlobWal::open(&path).unwrap();
        wal.append_put(b"good", &BlobValue::Raw(b"ok".to_vec()))
            .unwrap();
        wal.append_put(b"bad", &BlobValue::Raw(b"no".to_vec()))
            .unwrap();
        drop(wal); // release exclusive advisory lock before mutating out-of-band
                   // Corrupt one byte inside the second record's payload region.
        let bytes = std::fs::read(&path).unwrap();
        let first_record_len = HEADER_LEN + 4 + 2 + CRC_LEN; // 4-byte key "good" + 2-byte value "ok"
        let mut corrupted = bytes.clone();
        corrupted[first_record_len + HEADER_LEN + 1] ^= 0xff; // flip a bit inside "bad"
        std::fs::write(&path, corrupted).unwrap();

        let wal = BlobWal::open(&path).unwrap();
        let records = wal.replay().unwrap();
        assert_eq!(
            records,
            vec![WalRecord::Put(
                b"good".to_vec(),
                BlobValue::Raw(b"ok".to_vec())
            )]
        );
    }
}
