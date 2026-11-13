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
//!  9      1     value_kind      0x00 = Raw, 0x01 = Compressed, 0x03 = CompressedSized
//!                               (the decoded length, u64 LE, then the zlib payload); ignored on Delete
//! 10      N     key             key_len bytes
//! 10+N    M     value           value_len bytes (absent on Delete)
//! trailer 4     crc32           CRC-32 over everything from offset 0 upward
//! ```
//!
//! A batch is one record of type 0x03: `key_len` and `value_kind` are 0 and the
//! value is the body below, so one checksum covers every operation and replay
//! yields all of them or none.
//!
//! ```text
//! offset  size  field           description
//! ------  ----  --------------  ------------------------------------------------
//!  0      4     op_count (u32)  number of operations, at least 1
//! then, op_count times:
//!  0      1     type            0x01 = Put, 0x02 = Delete
//!  1      1     value_kind      0x00 = Raw, 0x01 = Compressed, 0x03 = CompressedSized; 0 on Delete
//!  2      4     key_len (u32)
//!  6      4     value_len (u32) 0 on Delete
//! 10      N     key
//! 10+N    M     value
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
//! - A record whose CRC-32 matches but which this build cannot read (an
//!   unknown type or value kind, a batch body that does not parse or holds
//!   no operation) is no torn tail: a newer build wrote it. The open fails
//!   with `InvalidData` and the file stays as it is.
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
//! A second process (or a second call in the same process) attempting to open
//! the same WAL file, by any name, receives a `WouldBlock`-flavoured
//! `io::Error`. Outside Windows the WAL holds an exclusive [`fs2`] advisory lock
//! on its handle. On Windows a lock on a file stops other handles reading it,
//! and a read-only open reads the WAL beside the writer, so the handle is opened
//! with a share mode that lets others read and refuses other writers. Both belong
//! to the file, not its name, and both end when the holder dies. Alpha-3 documents
//! this as the canonical way to detect a running writer; multi-writer coordination
//! (leases, coordinator process) is out of scope for alpha.

use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::blob::BlobValue;
use crate::blob_dir::{BlobDir, BlobFile, StdDir};

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
const RECORD_TYPE_BATCH: u8 = 0x03;

/// Value kind tag inside a Put record.
const VALUE_KIND_RAW: u8 = 0x00;

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
    dir: Arc<dyn BlobDir>,
    state: Mutex<WalState>,
    sync_policy: SyncPolicy,
}

/// Interior state guarded by the outer `Mutex`. Kept separate so a
/// single `lock()` yields access to both the file handle and the
/// batched-write counter.
#[derive(Debug)]
struct WalState {
    file: Box<dyn BlobFile>,
    /// Number of appended records not yet reflected on disk via
    /// `sync_data`. `EveryWrite` keeps this at zero because it fsyncs
    /// unconditionally; `Batched` and `Manual` allow it to grow.
    pending_ops: usize,
    /// Why the WAL refuses writes: a record that failed part-way could not be removed, so a later record would land
    /// behind it and be cut with it at the next open
    poisoned: Option<String>,
}

impl BlobWal {
    /// Open (or create) the WAL at `path` with the default
    /// [`SyncPolicy::EveryWrite`].
    ///
    /// Missing parent directories are created. The file is opened in
    /// read + write mode so replay reads, appends, and truncation share
    /// a single handle; every append seeks to the end explicitly (the
    /// handle deliberately lacks the `append` flag so that truncation
    /// also works on Windows). A second writer of the file sees a `WouldBlock`
    /// error (see the module's cross-process locking section).
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
        Self::open_in(Arc::new(StdDir), path, sync_policy)
    }

    /// Same as [`Self::open_with_policy`], with the files kept in `dir`.
    ///
    /// # Errors
    /// See [`Self::open`].
    pub fn open_in(
        dir: Arc<dyn BlobDir>,
        path: impl AsRef<Path>,
        sync_policy: SyncPolicy,
    ) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            dir.create_dir_all(parent)?;
        }
        // Opened read + write, *not* append (see `blob_dir::OPEN_WITH_APPEND`): every write
        // goes to the end explicitly instead (see `write_frame`).
        let file = dir.open_exclusive(&path)?;
        Ok(Self {
            path,
            dir,
            state: Mutex::new(WalState {
                file,
                pending_ops: 0,
                poisoned: None,
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
        state.file.persist_data("wal")?;
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
        state.file.size()
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
        state.file.set_len(0)?;
        // sync_all here so the truncation reaches disk before we
        // consider the WAL "empty" from the caller's point of view.
        state.file.persist_all("wal")?;
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
        let stored = value.stored();
        let (kind, bytes): (u8, &[u8]) = match &stored {
            Some((kind, bytes)) => (*kind, bytes),
            None => {
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

    /// Serialise a batch as one record and durably append it: replay yields
    /// every operation or none. A [`BlobValue::Tombstone`] value is a delete.
    ///
    /// # Errors
    /// `InvalidInput` when the batch is empty or a length exceeds `u32`, with
    /// nothing written; any I/O failure from the write or fsync propagates.
    pub fn append_batch(&self, ops: &[(&[u8], &BlobValue)]) -> io::Result<()> {
        if ops.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "BlobWal::append_batch received an empty batch",
            ));
        }
        let too_long = |what| move |_| io::Error::new(io::ErrorKind::InvalidInput, what);
        let mut body = u32::try_from(ops.len())
            .map_err(too_long("batch has more than u32::MAX operations"))?
            .to_le_bytes()
            .to_vec();
        for (key, value) in ops {
            let stored = value.stored();
            let (record_type, kind, bytes): (u8, u8, &[u8]) = match &stored {
                Some((kind, bytes)) => (RECORD_TYPE_PUT, *kind, bytes),
                None => (RECORD_TYPE_DELETE, VALUE_KIND_RAW, &[]),
            };
            body.push(record_type);
            body.push(kind);
            let key_len = u32::try_from(key.len()).map_err(too_long("key exceeds u32::MAX"))?;
            let value_len =
                u32::try_from(bytes.len()).map_err(too_long("value exceeds u32::MAX"))?;
            body.extend_from_slice(&key_len.to_le_bytes());
            body.extend_from_slice(&value_len.to_le_bytes());
            body.extend_from_slice(key);
            body.extend_from_slice(bytes);
        }
        let frame = serialise_record(RECORD_TYPE_BATCH, &[], VALUE_KIND_RAW, &body)?;
        self.write_frame(&frame)
    }

    fn write_frame(&self, frame: &[u8]) -> io::Result<()> {
        let mut state = self.state.lock();
        if let Some(reason) = &state.poisoned {
            return Err(io::Error::other(format!(
                "the blob WAL refuses writes: {reason}"
            )));
        }
        // `append` writes at the end whatever was read before. The mutex makes the
        // write one step with respect to other writers in this process, and the
        // exclusive hold on the file rules out writers in other processes.
        let start = state.file.size()?;
        let sync_now = match self.sync_policy {
            SyncPolicy::EveryWrite => true,
            SyncPolicy::Batched { max_pending_ops } => state.pending_ops + 1 >= max_pending_ops,
            SyncPolicy::Manual => false,
        };
        let half = frame.len() / 2;
        let written = state
            .file
            .append(&frame[..half])
            .and_then(|()| crate::fs_audit::crash_point("wal:mid_frame"))
            .and_then(|()| state.file.append(&frame[half..]))
            .and_then(|()| {
                if sync_now {
                    crate::fs_audit::crash_point("wal:before_sync")?;
                    state.file.persist_data("wal")?;
                }
                Ok(())
            });
        if let Err(e) = written {
            // the record is not accepted: put the WAL back to its length before it, so no partial record stays for
            // later records to land behind (and a record whose fsync failed does not come back at the next open)
            let rolled = crate::fs_audit::crash_point("wal:before_rollback")
                .and_then(|()| state.file.set_len(start))
                .and_then(|()| state.file.persist_data("wal"));
            if let Err(r) = rolled {
                state.poisoned = Some(format!(
                    "a record that failed at offset {start} could not be removed ({r}) after: {e}"
                ));
            }
            return Err(e);
        }
        state.pending_ops = if sync_now { 0 } else { state.pending_ops + 1 };
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
        // Read from the start; `write_frame` appends at the end, so where the
        // reading stops does not matter.
        let mut reader = BufReader::new(AtReader {
            file: state.file.as_mut(),
            pos: 0,
        });

        let mut out = Vec::new();
        while let ReadOutcome::Records(records) = read_one(&mut reader)? {
            out.extend(records);
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
        remove_stale_dropped_tmp(self.dir.as_ref(), &self.path)?;
        let mut reader = Counting {
            inner: BufReader::new(AtReader {
                file: state.file.as_mut(),
                pos: 0,
            }),
            pos: 0,
        };
        let mut out = Vec::new();
        let mut good_end = 0_u64;
        let tail = loop {
            match read_one(&mut reader)? {
                ReadOutcome::Records(records) => {
                    out.extend(records);
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
        AtReader {
            file: state.file.as_mut(),
            pos: good_end,
        }
        .read_to_end(&mut dropped)?;
        let dropped_copy = keep_dropped(self.dir.as_ref(), &self.path, good_end, &dropped)?;
        crate::fs_audit::crash_point("wal_recover:before_cut")?;

        state.file.set_len(good_end)?;
        state.file.persist_all("wal")?;
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
fn keep_dropped(dir: &dyn BlobDir, wal: &Path, offset: u64, bytes: &[u8]) -> io::Result<PathBuf> {
    let mut n = 0;
    loop {
        let path = dropped_copy_name(wal, offset, n);
        match dir.read(&path) {
            Ok(existing) if existing == bytes => return Ok(path),
            Ok(_) => n += 1,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                write_durable(dir, &path, bytes)?;
                return Ok(path);
            }
            Err(e) => return Err(e),
        }
    }
}

/// Write `bytes` to the free name `path` through `<path>.tmp` (created
/// with `create_new`): write, fsync, rename, then fsync the directory so
/// the name survives a crash too.
fn write_durable(dir: &dyn BlobDir, path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp_name = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    dir.write_new(&tmp, bytes, "file")?;
    dir.rename(&tmp, path)?;
    dir.sync_parent(path, "dir")
}

/// Remove `<wal>.dropped-*.tmp` left by a copy that was interrupted
/// before its rename. Finished copies (no `.tmp`) are never removed. The
/// caller holds the WAL lock, so no copy is being written meanwhile.
fn remove_stale_dropped_tmp(store: &dyn BlobDir, wal: &Path) -> io::Result<()> {
    let dir = match wal.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let prefix = dropped_prefix(wal);
    let prefix = prefix.to_string_lossy();
    let entries = match store.list(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for path in entries {
        let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
            continue;
        };
        // exactly the lowercase `.tmp` recovery writes (a `.TMP` is not ours)
        if name.starts_with(prefix.as_ref())
            && Path::new(name).extension() == Some(std::ffi::OsStr::new("tmp"))
        {
            match store.remove(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

/// A reader over a [`BlobFile`] from `pos` on.
struct AtReader<'a> {
    file: &'a mut dyn BlobFile,
    pos: u64,
}

impl Read for AtReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.file.read_at(self.pos, buf)?;
        self.pos += n as u64;
        Ok(n)
    }
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
        // The handle's own drop then releases the hold on the file.
        let state = self.state.lock();
        if state.pending_ops > 0 {
            let _ = state.file.persist_data("wal");
        }
    }
}

/// Every well-framed record of the WAL at `path` without taking its lock or
/// changing it. A torn or corrupt tail ends the list, as in [`BlobWal::replay`];
/// it may be a write in progress. `None` when a flush truncated the WAL during
/// the read: the records read then may be a part of what the new `SSTable` holds,
/// and an old value among them would hide a newer one.
///
/// # Errors
/// `NotFound` when there is no file; any read error.
pub fn read_existing(path: &Path) -> io::Result<Option<Vec<WalRecord>>> {
    stable_records(|| StdDir.read(path))
}

/// Read twice: the good records of the first read must still be the start of the
/// second, or the file was cut (and perhaps written again) in between.
fn stable_records(
    mut read: impl FnMut() -> io::Result<Vec<u8>>,
) -> io::Result<Option<Vec<WalRecord>>> {
    let first = read()?;
    let mut reader = Counting {
        inner: first.as_slice(),
        pos: 0,
    };
    let mut out = Vec::new();
    let mut good_end = 0;
    while let ReadOutcome::Records(records) = read_one(&mut reader)? {
        out.extend(records);
        good_end = reader.pos;
    }
    let good = &first[..usize::try_from(good_end).unwrap_or(first.len())];
    Ok(read()?.starts_with(good).then_some(out))
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
    /// One record of the file: one operation, or every operation of a batch
    Records(Vec<WalRecord>),
    Eof,
    Truncated,
    BadChecksum,
}

fn read_one<R: Read>(reader: &mut R) -> io::Result<ReadOutcome> {
    let mut header = [0_u8; HEADER_LEN];
    match read_exact_or_eof(reader, &mut header)? {
        ReadResult::Full => {}
        ReadResult::Eof => return Ok(ReadOutcome::Eof),
        ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
    }

    let record_type = header[0];
    let key_len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]);
    let value_len = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
    let value_kind = header[9];

    let Some(key) = read_len(reader, key_len)? else {
        return Ok(ReadOutcome::Truncated);
    };
    let Some(value) = read_len(reader, value_len)? else {
        return Ok(ReadOutcome::Truncated);
    };

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
            let Some(value) = BlobValue::from_stored(value_kind, value) else {
                return Err(unreadable(&format!("value kind {value_kind:#04x}")));
            };
            Ok(ReadOutcome::Records(vec![WalRecord::Put(key, value)]))
        }
        RECORD_TYPE_DELETE => Ok(ReadOutcome::Records(vec![WalRecord::Delete(key)])),
        RECORD_TYPE_BATCH => parse_batch(&value)
            .map(ReadOutcome::Records)
            .ok_or_else(|| unreadable("batch body")),
        unknown => Err(unreadable(&format!("record type {unknown:#04x}"))),
    }
}

/// `len` bytes, or `None` when fewer remain. The length comes before the checksum
/// that could vouch for it, so the buffer grows with the bytes that arrive and a
/// garbage length in a torn tail reserves nothing.
fn read_len<R: Read>(reader: &mut R, len: u32) -> io::Result<Option<Vec<u8>>> {
    let mut bytes = Vec::new();
    let read = reader
        .by_ref()
        .take(u64::from(len))
        .read_to_end(&mut bytes)?;
    Ok((read == len as usize).then_some(bytes))
}

/// The checksum holds, so this is no torn tail: cutting it would lose it and
/// every record after it.
fn unreadable(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("blob WAL holds a {what} that this build does not read"),
    )
}

/// The operations of a batch body, or `None` when it does not parse exactly.
fn parse_batch(body: &[u8]) -> Option<Vec<WalRecord>> {
    fn take<'a>(rest: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
        if rest.len() < n {
            return None;
        }
        let (head, tail) = rest.split_at(n);
        *rest = tail;
        Some(head)
    }
    fn take_len(rest: &mut &[u8]) -> Option<usize> {
        let bytes: [u8; 4] = take(rest, 4)?.try_into().ok()?;
        usize::try_from(u32::from_le_bytes(bytes)).ok()
    }
    let mut rest = body;
    let count = take_len(&mut rest)?;
    if count == 0 {
        return None;
    }
    let mut out = Vec::new();
    for _ in 0..count {
        let head = take(&mut rest, 2)?;
        let (record_type, kind) = (head[0], head[1]);
        let key_len = take_len(&mut rest)?;
        let value_len = take_len(&mut rest)?;
        let key = take(&mut rest, key_len)?.to_vec();
        let value = take(&mut rest, value_len)?.to_vec();
        out.push(match (record_type, kind) {
            (RECORD_TYPE_PUT, kind) => WalRecord::Put(key, BlobValue::from_stored(kind, value)?),
            (RECORD_TYPE_DELETE, VALUE_KIND_RAW) if value.is_empty() => WalRecord::Delete(key),
            _ => return None,
        });
    }
    rest.is_empty().then_some(out)
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

    #[test]
    fn a_wal_cut_between_the_two_reads_is_not_used() {
        let a = serialise_record(RECORD_TYPE_PUT, b"a", VALUE_KIND_RAW, b"1").unwrap();
        let b = serialise_record(RECORD_TYPE_PUT, b"b", VALUE_KIND_RAW, b"2").unwrap();
        let both = [a.clone(), b.clone()].concat();
        let scripted = |reads: Vec<Vec<u8>>| {
            let mut reads = reads.into_iter();
            stable_records(move || Ok(reads.next().unwrap())).unwrap()
        };
        // unchanged, or only grown: the first read stands
        assert_eq!(
            scripted(vec![a.clone(), a.clone()]).map(|r| r.len()),
            Some(1)
        );
        assert_eq!(
            scripted(vec![a.clone(), both.clone()]).map(|r| r.len()),
            Some(1)
        );
        // a torn tail growing into a whole record is a write in progress
        let torn = both[..both.len() - 1].to_vec();
        assert_eq!(scripted(vec![torn, both.clone()]).map(|r| r.len()), Some(1));
        // cut, or cut and written again
        assert_eq!(scripted(vec![both.clone(), a]).map(|r| r.len()), None);
        assert_eq!(
            scripted(vec![both.clone(), Vec::new()]).map(|r| r.len()),
            None
        );
        assert_eq!(scripted(vec![both, b]).map(|r| r.len()), None);
    }

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
