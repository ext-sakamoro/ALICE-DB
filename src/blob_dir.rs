//! Where the blob store keeps its files: the directory operations and the writer's handle.
//!
//! [`crate::blob::BlobStorage`] and its WAL and `SSTables` reach the file system only
//! through [`BlobDir`] (the operations on paths: create, list, read, publish, remove) and
//! [`BlobFile`] (the one writer's handle on the WAL: read at an offset, append, cut, sync).
//! [`StdDir`] is the native implementation over `std::fs`, `fs2` and `memmap2`; it is the
//! only one, and it does what the store did before this layer existed, call for call (the
//! open-file audit of [`crate::fs_audit`] included), so the bytes on disk do not change
//! (`tests/blob_dir_bytes_golden.rs` pins them).
//!
//! The layer is the seam another storage (a browser's origin-private file system) plugs
//! into: every operation the store needs is named here, and nothing outside this module
//! touches `std::fs` for the blob store's files.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[cfg(not(windows))]
use fs2::FileExt;

/// The one writer's handle on a file (the WAL): read anywhere, append at the end, cut,
/// and make durable. Opened by [`BlobDir::open_exclusive`]; while it is open no other
/// writer can open the same file.
pub trait BlobFile: Send + fmt::Debug {
    /// The file's length in bytes.
    ///
    /// # Errors
    /// Any I/O error.
    fn len(&self) -> io::Result<u64>;

    /// Read up to `buf.len()` bytes at `offset`; `Ok(0)` at or past the end.
    ///
    /// # Errors
    /// Any I/O error.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<usize>;

    /// Write all of `bytes` at the end of the file.
    ///
    /// # Errors
    /// Any I/O error.
    fn append(&mut self, bytes: &[u8]) -> io::Result<()>;

    /// Cut (or extend) the file to `len` bytes.
    ///
    /// # Errors
    /// Any I/O error.
    fn set_len(&mut self, len: u64) -> io::Result<()>;

    /// Make the written bytes durable (`what` names the file for the audit).
    ///
    /// # Errors
    /// Any I/O error.
    fn sync_data(&self, what: &str) -> io::Result<()>;

    /// Make the written bytes and the length durable.
    ///
    /// # Errors
    /// Any I/O error.
    fn sync_all(&self, what: &str) -> io::Result<()>;
}

/// The bytes of a published file, held for reading (natively a memory map).
pub trait BlobBytes: Send + Sync + fmt::Debug {
    /// The file's bytes.
    fn bytes(&self) -> &[u8];
}

/// The operations the blob store performs on paths.
pub trait BlobDir: Send + Sync + fmt::Debug {
    /// Create `dir` and its missing parents.
    ///
    /// # Errors
    /// Any I/O error.
    fn create_dir_all(&self, dir: &Path) -> io::Result<()>;

    /// The length of the file at `path` (`NotFound` when there is none).
    ///
    /// # Errors
    /// Any I/O error.
    fn file_len(&self, path: &Path) -> io::Result<u64>;

    /// The whole file at `path` (`NotFound` when there is none).
    ///
    /// # Errors
    /// Any I/O error.
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;

    /// The paths of the entries of `dir`, in the order the system lists them (`NotFound`
    /// when there is no such directory).
    ///
    /// # Errors
    /// Any I/O error.
    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>>;

    /// Remove the file at `path`.
    ///
    /// # Errors
    /// Any I/O error (`NotFound` included).
    fn remove(&self, path: &Path) -> io::Result<()>;

    /// Write `bytes` to a new file at `path` (refused when it exists) and make it durable.
    ///
    /// # Errors
    /// Any I/O error.
    fn write_new(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;

    /// Move `from` over `to` in one step (replacing `to` when it exists).
    ///
    /// # Errors
    /// Any I/O error.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// Make the directory entries of `path`'s parent durable (`what` names it for the audit).
    ///
    /// # Errors
    /// Any I/O error.
    fn sync_parent(&self, path: &Path, what: &str) -> io::Result<()>;

    /// Open (or create) `path` for one writer: reads by others are allowed, a second writer
    /// gets `WouldBlock`, and the hold ends with the handle.
    ///
    /// # Errors
    /// `WouldBlock` when another writer holds it; any I/O error.
    fn open_exclusive(&self, path: &Path) -> io::Result<Box<dyn BlobFile>>;

    /// Hold the bytes of the file at `path` for reading (`NotFound` when there is none).
    ///
    /// # Errors
    /// Any I/O error.
    fn map(&self, path: &Path) -> io::Result<Box<dyn BlobBytes>>;
}

/// The native directory: `std::fs`, an advisory lock (`fs2`, a share mode on Windows) and
/// memory maps (`memmap2`).
#[derive(Debug, Default, Clone, Copy)]
pub struct StdDir;

/// Whether the WAL handle is opened with the append flag. It is not: Windows maps an append
/// handle to `FILE_APPEND_DATA` without `FILE_WRITE_DATA`, which makes a cut fail with
/// `ERROR_ACCESS_DENIED`. Kept as a named constant so the open-file audit checks the same
/// value the open call uses.
pub(crate) const OPEN_WITH_APPEND: bool = false;

/// `FILE_SHARE_READ`: other handles may read the WAL while this writer holds it.
#[cfg(windows)]
const WINDOWS_SHARE_READ: u32 = 0x1;
/// `FILE_SHARE_DELETE`: as in the default share mode, for the rename of a file over the
/// WAL's directory entry. `FILE_SHARE_WRITE` is left out, which refuses another writer of
/// this file by any name.
#[cfg(windows)]
const WINDOWS_SHARE_DELETE: u32 = 0x4;

/// `ERROR_SHARING_VIOLATION`: another handle's share mode refuses this open.
fn is_sharing_violation(e: &io::Error) -> bool {
    cfg!(windows) && e.raw_os_error() == Some(32)
}

fn already_locked(path: &Path, e: &io::Error, kind: io::ErrorKind) -> io::Error {
    io::Error::new(
        kind,
        format!(
            "`{}` is already locked by another writer: {e}",
            path.display()
        ),
    )
}

/// Open `path` for one writer at a time: other handles may read it, a second writer gets
/// `WouldBlock`, and the hold ends with the process.
pub(crate) fn open_exclusive(options: &mut OpenOptions, path: &Path) -> io::Result<File> {
    #[cfg(windows)]
    std::os::windows::fs::OpenOptionsExt::share_mode(
        options,
        WINDOWS_SHARE_READ | WINDOWS_SHARE_DELETE,
    );
    let file = options.open(path).map_err(|e| {
        if is_sharing_violation(&e) {
            already_locked(path, &e, io::ErrorKind::WouldBlock)
        } else {
            e
        }
    })?;
    // Exclusive advisory lock. If another process (or the same process) already holds the
    // lock we bail with a descriptive error so the caller can surface it as "database
    // already open" rather than a mysterious later corruption.
    #[cfg(not(windows))]
    FileExt::try_lock_exclusive(&file).map_err(|e| {
        let kind = if e.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
            io::ErrorKind::WouldBlock
        } else {
            e.kind()
        };
        already_locked(path, &e, kind)
    })?;
    Ok(file)
}

/// The native writer's handle: a file opened read + write (not append) under the lock.
#[derive(Debug)]
pub struct StdFile {
    file: File,
    path: PathBuf,
}

impl BlobFile for StdFile {
    fn len(&self) -> io::Result<u64> {
        self.file.metadata().map(|m| m.len())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read(buf)
    }

    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        // The handle is not opened in append mode (see `OPEN_WITH_APPEND`), so position at
        // the end explicitly before each write.
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(bytes)
    }

    fn set_len(&mut self, len: u64) -> io::Result<()> {
        crate::fs_audit::check_set_len(&self.path, OPEN_WITH_APPEND);
        self.file.seek(SeekFrom::Start(0))?;
        self.file.set_len(len)
    }

    fn sync_data(&self, what: &str) -> io::Result<()> {
        crate::fs_audit::sync_data(&self.file, what, &self.path)
    }

    fn sync_all(&self, what: &str) -> io::Result<()> {
        crate::fs_audit::sync_all(&self.file, what, &self.path)
    }
}

impl Drop for StdFile {
    fn drop(&mut self) {
        #[cfg(not(windows))]
        let _ = FileExt::unlock(&self.file);
        crate::fs_audit::note_close(&self.path);
    }
}

/// A memory map of a published file.
#[derive(Debug)]
pub struct StdBytes(memmap2::Mmap);

impl BlobBytes for StdBytes {
    fn bytes(&self) -> &[u8] {
        &self.0
    }
}

impl BlobDir for StdDir {
    fn create_dir_all(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)
    }

    fn file_len(&self, path: &Path) -> io::Result<u64> {
        std::fs::metadata(path).map(|m| m.len())
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        std::fs::read_dir(dir)?
            .map(|e| e.map(|e| e.path()))
            .collect()
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        crate::fs_audit::check_remove(path);
        std::fs::remove_file(path)
    }

    fn write_new(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
        f.write_all(bytes)?;
        crate::fs_audit::sync_all(&f, "file", path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        crate::fs_audit::check_replace_target(to);
        std::fs::rename(from, to)
    }

    fn sync_parent(&self, path: &Path, what: &str) -> io::Result<()> {
        // Opening a directory as a file to sync it is a Unix idiom; Windows has no
        // equivalent and its rename is durable through the journal.
        #[cfg(unix)]
        if let Some(dir) = path.parent() {
            let dir = if dir.as_os_str().is_empty() {
                Path::new(".")
            } else {
                dir
            };
            crate::fs_audit::sync_all(&File::open(dir)?, what, dir)?;
        }
        #[cfg(not(unix))]
        let _ = (path, what);
        Ok(())
    }

    fn open_exclusive(&self, path: &Path) -> io::Result<Box<dyn BlobFile>> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .append(OPEN_WITH_APPEND)
            .create(true)
            .truncate(false);
        let file = open_exclusive(&mut options, path)?;
        crate::fs_audit::note_open(path);
        Ok(Box::new(StdFile {
            file,
            path: path.to_path_buf(),
        }))
    }

    fn map(&self, path: &Path) -> io::Result<Box<dyn BlobBytes>> {
        let file = File::open(path)?;
        // SAFETY: the store never writes to a published file (it writes a new file and
        // renames it into place), so the mapped bytes do not change while held.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Box::new(StdBytes(mmap)))
    }
}
