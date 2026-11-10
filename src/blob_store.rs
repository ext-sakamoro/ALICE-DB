//! Which directory and which WAL make up a durable blob store.
//!
//! One directory holds one store. Its `SSTables` are `blob.sst` and
//! `blob-000001.sst` whatever the WAL is called. `blob.lock` is held while a
//! writer has the store open and records the name of its WAL, so a second name in
//! the directory stops at the open instead of mixing with the first. A read-only
//! open reads the record and never writes it.
//!
//! The record is `1\n<file name>\n`. A record that ends before its last newline
//! is a write cut short and counts as no record.

use crate::blob_dir::{BlobDir, BlobFile, StdDir};
use crate::blob_sstable::{enumerate_sstables, parse_sstable_seq};
use std::io;
use std::path::{Path, PathBuf};

const LOCK_FILE: &str = "blob.lock";
const WAL_DEFAULT: &str = "blob.wal";
const RECORD_VERSION: &str = "1";

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// The directory of a WAL path; a bare name is in the current directory.
pub(crate) fn store_dir(wal_path: &Path) -> PathBuf {
    match wal_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// The file name of a WAL path; of its target for a symlink, so a link and the
/// file it reaches are one name.
///
/// # Errors
/// `InvalidInput` for a name that is not UTF-8 or holds a newline, and for the
/// name of one of the store's own files, which opening as a WAL would cut.
pub(crate) fn wal_name(wal_path: &Path) -> io::Result<String> {
    let refused = |why: &str| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("blob WAL `{}` {why}", wal_path.display()),
        )
    };
    let is_link = StdDir.is_link(wal_path);
    let real = if is_link {
        StdDir.canonicalize(wal_path)?
    } else {
        wal_path.to_path_buf()
    };
    let name = real
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .filter(|name| !name.contains('\n'))
        .ok_or_else(|| refused("needs a UTF-8 file name without a newline"))?;
    let lower = name.to_ascii_lowercase();
    let is_sstable = Path::new(&lower)
        .extension()
        .is_some_and(|extension| extension == "sst");
    if is_sstable || is_working_file(&lower) {
        return Err(refused("has the name of a file of the store"));
    }
    Ok(name.to_owned())
}

/// A symlink is the same store only beside its target: the files are found next
/// to the path that was given.
fn refuse_symlink_elsewhere(wal_path: &Path, dir: &Path) -> io::Result<()> {
    if !StdDir.is_link(wal_path) {
        return Ok(());
    }
    let target = StdDir.canonicalize(wal_path)?;
    if target.parent() != Some(StdDir.canonicalize(dir)?.as_path()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "blob WAL `{}` is a link to `{}` in another directory, whose files it would not see",
                wal_path.display(),
                target.display()
            ),
        ));
    }
    Ok(())
}

/// Whether two names are one file, as the file system sees it.
fn same_file(a: &Path, b: &Path) -> bool {
    StdDir.same_file(a, b)
}

/// `dir` as a path that means the same after the current directory changes: a
/// store flushes long after it was opened.
///
/// # Errors
/// Any error of creating or resolving the directory.
pub(crate) fn writer_dir(wal_path: &Path) -> io::Result<PathBuf> {
    let dir = store_dir(wal_path);
    StdDir.create_dir_all(&dir)?;
    std::path::absolute(dir)
}

/// The WAL name recorded in `dir`, if there is a whole record.
///
/// # Errors
/// `InvalidData` for a whole record that cannot be read.
pub(crate) fn recorded_wal(dir: &Path) -> io::Result<Option<String>> {
    let path = dir.join(LOCK_FILE);
    let bytes = match StdDir.read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some(body) = bytes.strip_suffix(b"\n") else {
        return Ok(None);
    };
    let unreadable = || invalid_data(format!("`{}` cannot be read", path.display()));
    let text = std::str::from_utf8(body).map_err(|_| unreadable())?;
    if text == RECORD_VERSION {
        return Ok(None);
    }
    let (version, name) = text.split_once('\n').ok_or_else(unreadable)?;
    let plain_name = !name.is_empty() && !name.contains(['/', '\\', '\n']);
    if version != RECORD_VERSION || !plain_name || name == "." || name == ".." {
        return Err(unreadable());
    }
    Ok(Some(name.to_owned()))
}

/// A file the store itself writes, by its lower-cased name: an `SSTable`, an
/// `SSTable` being written (`blob.sst.tmp`, renamed when complete) or the lock.
fn is_working_file(lower: &str) -> bool {
    let published = lower.strip_suffix(".tmp").unwrap_or(lower);
    parse_sstable_seq(published).is_some() || lower == LOCK_FILE
}

/// `<stem>.sst`, which an earlier version wrote for a WAL of another name than
/// `blob.wal`. The file system decides what the name reaches, and a file that is
/// already the store's own under another spelling (`Blob.sst`) is not one.
fn earlier_file(dir: &Path, wal_name: &str) -> io::Result<Option<PathBuf>> {
    let stem = Path::new(wal_name)
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or(wal_name);
    let candidate = dir.join(format!("{stem}.sst"));
    let read = match StdDir.list(dir) {
        Ok(read) => read,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    for path in read {
        // a working file can be the candidate for a moment: a flush renames
        // `blob.sst.tmp` while it is listed here, and the name resolves to the result
        let is_store_file = path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| is_working_file(&name.to_ascii_lowercase()));
        if !is_store_file && same_file(&path, &candidate) {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

/// Every `SSTable` of the store, oldest first, for a reader that moves nothing.
pub(crate) fn source_files(dir: &Path, wal_name: &str) -> io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = earlier_file(dir, wal_name)?.into_iter().collect();
    files.extend(enumerate_sstables(dir)?);
    Ok(files)
}

/// What the open has to do to the record once the WAL is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Record {
    Keep,
    Write,
}

/// Whether `wal_name` may open the store in `dir`.
///
/// # Errors
/// `InvalidData` when the files belong to another WAL or cannot be told apart;
/// `InvalidInput` for a link to another directory.
pub(crate) fn check_open(
    dir: &Path,
    wal_path: &Path,
    wal_name: &str,
    recorded: Option<&str>,
) -> io::Result<Record> {
    refuse_symlink_elsewhere(wal_path, dir)?;
    let numbered = enumerate_sstables(dir)?;
    if let Some(earlier) = earlier_file(dir, wal_name)? {
        if !numbered.is_empty() {
            let names: Vec<String> = numbered.iter().map(|p| p.display().to_string()).collect();
            return Err(invalid_data(format!(
                "`{}` and {} are both in the store: an earlier version wrote the first for `{wal_name}` \
                 and neither replaces the other. Check what each holds and move one out of the directory",
                earlier.display(),
                names.join(", ")
            )));
        }
    }
    let dir_text = dir.display();
    match recorded {
        Some(name) if name == wal_name => Ok(Record::Keep),
        Some(name) => {
            let recorded_path = dir.join(name);
            if recorded_path.exists() {
                if same_file(&recorded_path, wal_path) {
                    return Ok(Record::Keep);
                }
                return Err(invalid_data(format!(
                    "`{dir_text}` is the store of `{name}`, which still exists: open `{name}`, \
                     or move `{wal_name}` and its files to a directory of their own"
                )));
            }
            if numbered.is_empty() {
                return Ok(Record::Write);
            }
            Err(invalid_data(format!(
                "`{dir_text}` holds the files of the store of `{name}`, and `{name}` is gone: \
                 BlobStorage::take_over_store on `{wal_name}` gives them to it, if they are its"
            )))
        }
        None => {
            let owner_is_clear =
                wal_path.exists() || wal_name == WAL_DEFAULT || numbered.is_empty();
            if owner_is_clear {
                return Ok(Record::Write);
            }
            Err(invalid_data(format!(
                "`{dir_text}` holds a store's files and `{wal_name}` is new: \
                 BlobStorage::take_over_store on `{wal_name}` gives them to it, if they are its"
            )))
        }
    }
}

/// `blob.lock`, held for as long as the writer lives.
#[derive(Debug)]
pub(crate) struct StoreLock {
    file: Box<dyn BlobFile>,
}

impl StoreLock {
    /// # Errors
    /// `WouldBlock` while another writer holds the directory.
    pub(crate) fn acquire(dir: &Path) -> io::Result<Self> {
        StdDir.create_dir_all(dir)?;
        let file = StdDir.open_exclusive(&dir.join(LOCK_FILE))?;
        Ok(Self { file })
    }

    /// # Errors
    /// Any write error, or an injected crash under the audit feature.
    pub(crate) fn record(&mut self, wal_name: &str) -> io::Result<()> {
        crate::fs_audit::crash_point("store:before_record")?;
        self.file.set_len(0)?;
        self.file
            .append(format!("{RECORD_VERSION}\n{wal_name}\n").as_bytes())?;
        self.file.persist_all("store-lock")
    }
}

/// Give `<stem>.sst` of an earlier version its name in this one. The rename is one
/// step: a crash leaves the file under one name or the other, and [`source_files`]
/// and the next open read it under either.
///
/// # Errors
/// Rename and sync errors, or an injected crash under the audit feature.
pub(crate) fn move_earlier_file(dir: &Path, wal_name: &str) -> io::Result<()> {
    let Some(earlier) = earlier_file(dir, wal_name)? else {
        return Ok(());
    };
    crate::fs_audit::crash_point("store:before_rename")?;
    let target = dir.join("blob.sst");
    StdDir.rename(&earlier, &target)?;
    crate::fs_audit::crash_point("store:before_dir_sync")?;
    StdDir.sync_parent(&target, "store-dir")
}

/// See [`crate::blob::BlobStorage::take_over_store`].
pub(crate) fn take_over(wal_path: &Path) -> io::Result<()> {
    let name = wal_name(wal_path)?;
    let dir = store_dir(wal_path);
    refuse_symlink_elsewhere(wal_path, &dir)?;
    let mut lock = StoreLock::acquire(&dir)?;
    if let Some(recorded) = recorded_wal(&dir)? {
        let recorded_path = dir.join(&recorded);
        if recorded_path.exists() && !same_file(&recorded_path, wal_path) {
            return Err(invalid_data(format!(
                "`{recorded}` still exists and may hold writes that are not in the files: \
                 open it to flush them, then remove it"
            )));
        }
    }
    lock.record(&name)
}
