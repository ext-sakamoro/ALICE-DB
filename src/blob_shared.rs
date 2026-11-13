//! Shared writers: several processes write one blob store ([`crate::blob::BlobStorage::open_shared`]).
//!
//! A store opened shared holds `blob.lock` only for the length of one operation: the lease. Under the lease a
//! process reads what the others wrote since its last lease, does its read or write, and lets go. The lock is
//! the same one an exclusive writer holds for as long as it is open ([`crate::blob_store::StoreLock`]: an
//! advisory lock outside Windows, a share mode on Windows), so the hold ends with the process: a writer killed
//! inside its lease leaves no lease behind and no timer is involved. A lease waits up to
//! [`SharedOptions::lock_wait`] for the lock and is `TimedOut` past it.
//!
//! `blob.lease` holds the lease generation and the epoch of the store's files:
//!
//! ```text
//! offset  size  field
//!  0      8     magic "ALICEDBL"
//!  8      8     generation (u64 LE): raised once by every lease that writes
//! 16      8     epoch (u64 LE): raised by every flush, which replaces the SSTables and empties the WAL
//! 24      4     CRC-32 of bytes 0..24
//! ```
//!
//! It is replaced whole (written beside, fsynced, renamed, the directory fsynced), so a reader sees one
//! version or the next. A WAL record written under a lease carries that lease's generation
//! ([`crate::blob_wal::BlobWal::append_fenced_batch`]); replay refuses a generation that is not above every
//! earlier one, so the writes of a process that wrote without the lease cannot be mixed in silently. The
//! order of the store is the order of the WAL's records, the same on every read of the same files.
//!
//! The lease is reentrant on the thread that holds it, so a read inside a read-modify-write of the same
//! handle does not wait for itself. A file system whose locks do not keep a second handle out (some network
//! file systems accept a lock and enforce nothing) is refused at the open: shared writers need the lock.

use crate::blob_dir::{BlobDir, StdDir};
use crate::blob_store::StoreLock;
use std::cell::RefCell;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// One step of the wait for another process's lease
const WAIT_STEP: Duration = Duration::from_millis(2);

pub(crate) const LEASE_FILE: &str = "blob.lease";
const LEASE_TMP: &str = "blob.lease.tmp";
const LEASE_MAGIC: &[u8; 8] = b"ALICEDBL";
const LEASE_LEN: usize = 8 + 8 + 8 + 4;

/// How a shared writer takes its lease: see [`crate::blob::BlobStorage::open_shared`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SharedOptions {
    /// How long one operation waits for another process's lease before it is `TimedOut` (default 5 s)
    pub lock_wait: Duration,
    /// The most a shared handle holds in memory, counted as the bytes of every key and stored value (a
    /// compressed value counts its stored bytes), default 256 MiB. A shared handle holds the whole store in
    /// memory, plus about 110 bytes per entry for the map (measured: 10 000 and 50 000 entries of 24-byte keys
    /// and 100-byte values held 2.3 MB and 11.6 MB). A write that would take the store past it, and an open
    /// or a refresh that finds the store past it, fail with [`SharedStoreTooLarge`]. Values a read returns
    /// (`get`, `scan_*`) are decoded copies the caller holds: they are not counted.
    pub max_bytes: u64,
}

/// A shared handle would hold more than its [`SharedOptions::max_bytes`]: carried inside an `io::Error` of kind
/// `FileTooLarge`. A refused write wrote nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SharedStoreTooLarge {
    /// The bytes of keys and stored values the store would hold
    pub bytes: u64,
    /// The handle's limit
    pub limit: u64,
}

impl std::fmt::Display for SharedStoreTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the store would hold {} bytes of keys and values, past this shared handle's limit of {}",
            self.bytes, self.limit
        )
    }
}

impl std::error::Error for SharedStoreTooLarge {}

/// Another holder (another process, or another handle of the same store) kept the shared store's lease past
/// this handle's [`SharedOptions::lock_wait`]: carried inside an `io::Error` of kind `TimedOut`. Nothing of the
/// refused operation was written; it can be tried again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct WriterBusy {
    /// How long this handle waited
    pub waited: Duration,
}

impl std::fmt::Display for WriterBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "another holder kept blob.lock past the wait of {:?}",
            self.waited
        )
    }
}

impl std::error::Error for WriterBusy {}

/// `SharedStoreTooLarge` as an `io::Error`, or nothing when `bytes` is within `limit`.
pub(crate) fn check_size(bytes: u64, limit: u64) -> io::Result<()> {
    if bytes > limit {
        return Err(io::Error::new(
            io::ErrorKind::FileTooLarge,
            SharedStoreTooLarge { bytes, limit },
        ));
    }
    Ok(())
}

/// What one entry counts towards [`SharedOptions::max_bytes`]: its key and its stored value.
pub(crate) fn entry_bytes(key: &[u8], value: &crate::blob::BlobValue) -> u64 {
    let stored = value.stored().map_or(0, |(_, bytes)| bytes.len());
    (key.len() + stored) as u64
}

impl Default for SharedOptions {
    fn default() -> Self {
        Self {
            lock_wait: Duration::from_secs(5),
            max_bytes: 256 << 20,
        }
    }
}

impl SharedOptions {
    /// Set [`Self::lock_wait`].
    #[must_use]
    pub const fn with_lock_wait(mut self, lock_wait: Duration) -> Self {
        self.lock_wait = lock_wait;
        self
    }

    /// Set [`Self::max_bytes`].
    #[must_use]
    pub const fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }
}

/// What `blob.lease` holds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lease {
    pub(crate) generation: u64,
    pub(crate) epoch: u64,
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// `blob.lease` of `dir`, `None` when the store has never been opened shared.
///
/// # Errors
/// `InvalidData` for a file that is not a lease record; read errors.
pub(crate) fn read_lease(dir: &Path) -> io::Result<Option<Lease>> {
    let bytes = match StdDir.read(&dir.join(LEASE_FILE)) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if bytes.len() != LEASE_LEN || &bytes[..8] != LEASE_MAGIC {
        return Err(invalid_data(format!("{LEASE_FILE} is not a lease record")));
    }
    let crc = u32::from_le_bytes(bytes[24..28].try_into().unwrap_or([0; 4]));
    if crc32fast::hash(&bytes[..24]) != crc {
        return Err(invalid_data(format!("{LEASE_FILE} fails its CRC-32")));
    }
    let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]));
    Ok(Some(Lease {
        generation: word(8),
        epoch: word(16),
    }))
}

/// Replace `blob.lease` of `dir` with `lease`: written beside it, fsynced, renamed over it, the directory
/// fsynced.
///
/// # Errors
/// Write, sync and rename errors.
pub(crate) fn write_lease(dir: &Path, lease: Lease) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(LEASE_LEN);
    bytes.extend_from_slice(LEASE_MAGIC);
    bytes.extend_from_slice(&lease.generation.to_le_bytes());
    bytes.extend_from_slice(&lease.epoch.to_le_bytes());
    bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
    let tmp = dir.join(LEASE_TMP);
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&bytes)?;
        crate::fs_audit::sync_all(&file, "lease", &tmp)?;
    }
    let target = dir.join(LEASE_FILE);
    StdDir.rename(&tmp, &target)?;
    StdDir.sync_parent(&target, "lease-dir")
}

/// `blob.lock` of `dir`, waiting up to `wait` while another handle holds it.
///
/// The wait is counted in fixed steps of [`WAIT_STEP`] (the crate reads no clock, ALICE-Law G10): about
/// `wait / WAIT_STEP` more attempts after the first.
///
/// # Errors
/// `TimedOut` once the steps are spent; any other open or lock error.
pub(crate) fn acquire(dir: &Path, wait: Duration) -> io::Result<StoreLock> {
    let steps = wait.as_micros() / WAIT_STEP.as_micros();
    let mut attempt: u128 = 0;
    loop {
        match StoreLock::acquire(dir) {
            Ok(lock) => return Ok(lock),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if attempt >= steps {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        WriterBusy { waited: wait },
                    ));
                }
                attempt += 1;
                std::thread::sleep(WAIT_STEP);
            }
            Err(e) => return Err(e),
        }
    }
}

/// With `blob.lock` of `dir` held, whether a second handle is kept out of it, as shared writers need.
///
/// # Errors
/// Open errors other than the lock being held.
pub(crate) fn locks_hold(dir: &Path) -> io::Result<bool> {
    if crate::fs_audit::lock_probe_forced_failure() {
        return Ok(false);
    }
    match StoreLock::acquire(dir) {
        Ok(_second) => Ok(false),
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(true),
        Err(e) => Err(e),
    }
}

/// What one handle knows of the store: which epoch of the files it read, and how many WAL records of that
/// epoch it has applied.
#[derive(Debug, Default)]
pub(crate) struct View {
    pub(crate) epoch: Option<u64>,
    pub(crate) applied: usize,
    /// The bytes of keys and stored values the handle holds (see [`entry_bytes`])
    pub(crate) bytes: u64,
}

/// The lease held by the current thread of one handle: the lock, how deep the nested leases are, and the
/// generation this lease writes under once it has written.
#[derive(Default)]
pub(crate) struct Slot {
    depth: u32,
    lock: Option<StoreLock>,
    generation: Option<u64>,
}

/// The shared part of a [`crate::blob::BlobStorage`] opened shared.
pub(crate) struct Shared {
    pub(crate) dir: PathBuf,
    pub(crate) wal_path: PathBuf,
    pub(crate) wal_name: String,
    pub(crate) wait: Duration,
    pub(crate) max_bytes: u64,
    pub(crate) sync_policy: crate::blob_wal::SyncPolicy,
    pub(crate) slot: parking_lot::ReentrantMutex<RefCell<Slot>>,
    pub(crate) view: parking_lot::Mutex<View>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("dir", &self.dir)
            .field("wal_name", &self.wal_name)
            .field("wait", &self.wait)
            .finish_non_exhaustive()
    }
}

/// A lease held by the current thread; the lock is let go when the outermost one is dropped.
pub(crate) struct LeaseGuard<'a> {
    shared: &'a Shared,
    slot: parking_lot::ReentrantMutexGuard<'a, RefCell<Slot>>,
}

impl Shared {
    /// Take the lease, or join the one this thread already holds.
    ///
    /// # Errors
    /// As [`acquire`].
    pub(crate) fn lease(&self) -> io::Result<LeaseGuard<'_>> {
        let slot = self.slot.lock();
        {
            let mut s = slot.borrow_mut();
            if s.depth == 0 {
                s.lock = Some(acquire(&self.dir, self.wait)?);
                s.generation = None;
            }
            s.depth += 1;
        }
        Ok(LeaseGuard { shared: self, slot })
    }
}

impl LeaseGuard<'_> {
    /// The generation this lease writes under: raised in `blob.lease` the first time the lease writes.
    ///
    /// # Errors
    /// Read and write errors of `blob.lease`; `InvalidData` when it is missing.
    pub(crate) fn generation(&self) -> io::Result<u64> {
        let mut s = self.slot.borrow_mut();
        if let Some(generation) = s.generation {
            return Ok(generation);
        }
        let dir = &self.shared.dir;
        let lease =
            read_lease(dir)?.ok_or_else(|| invalid_data(format!("{LEASE_FILE} is missing")))?;
        let generation = lease.generation + 1;
        write_lease(
            dir,
            Lease {
                generation,
                epoch: lease.epoch,
            },
        )?;
        s.generation = Some(generation);
        Ok(generation)
    }
}

impl LeaseGuard<'_> {
    /// A generation for one more WAL record of this lease: raised in `blob.lease` every time, so two records
    /// of one lease never carry the same generation (replay refuses a generation that does not rise).
    ///
    /// # Errors
    /// As [`Self::generation`].
    pub(crate) fn next_generation(&self) -> io::Result<u64> {
        self.slot.borrow_mut().generation = None;
        self.generation()
    }
}

impl Drop for LeaseGuard<'_> {
    fn drop(&mut self) {
        let mut s = self.slot.borrow_mut();
        s.depth -= 1;
        if s.depth == 0 {
            s.generation = None;
            s.lock = None;
        }
    }
}
