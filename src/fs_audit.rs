//! Open-file audit for the file backend (test builds only).
//!
//! Windows refuses to delete a file, to rename another file over it, or
//! to shrink it through a handle that lacks `FILE_WRITE_DATA` while the
//! file is held open or memory-mapped. Unix allows all of these, so a
//! store developed and tested on Linux / macOS can pass every test and
//! still fail on its first Windows compaction.
//!
//! This module makes those Windows rules checkable on any platform. The
//! file backend reports every handle it opens and closes (blob `SSTable`
//! mmaps and the blob WAL) and asks for permission before each rename,
//! delete, and truncate. When the feature `open-file-audit` is enabled
//! a forbidden operation panics with the offending path; without the
//! feature every hook is an empty inline function and the module adds
//! nothing to the build.
//!
//! The feature is enabled for this crate's own tests through a
//! dev-dependency on itself (see `Cargo.toml`), so `cargo test` always
//! runs the blob suites under the audit. It is not meant to be enabled
//! by dependants.
//!
//! Every fsync of the crate goes through `sync_all` / `sync_data`
//! here (`tests/fsync_single_place.rs` fails on a direct call anywhere
//! else in `src/`), and under the feature each one is recorded, so tests
//! can check that a durable step really fsynced the file it names.
//!
//! The module also hosts crash-injection points used by the
//! compaction crash-safety tests: `arm_crash_point` makes the next
//! `crash_point` call with that name on the current thread return an
//! error, which aborts the operation exactly where a crash would.

use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(feature = "open-file-audit")]
mod imp {
    use std::cell::RefCell;
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn open_handles() -> MutexGuard<'static, BTreeMap<PathBuf, usize>> {
        static OPEN: OnceLock<Mutex<BTreeMap<PathBuf, usize>>> = OnceLock::new();
        OPEN.get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The key each opened path got, by the path as given: once the directory is
    /// gone the parent no longer canonicalizes, and the close must still find it.
    fn keys_by_path() -> MutexGuard<'static, BTreeMap<PathBuf, PathBuf>> {
        static KEYS: OnceLock<Mutex<BTreeMap<PathBuf, PathBuf>>> = OnceLock::new();
        KEYS.get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Closes the audit could not match, reported while their thread was already
    /// unwinding (a second panic there would abort the process and hide the first).
    fn unmatched() -> MutexGuard<'static, Vec<PathBuf>> {
        static UNMATCHED: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
        UNMATCHED
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn take_unmatched_closes() -> Vec<PathBuf> {
        std::mem::take(&mut *unmatched())
    }

    thread_local! {
        static ARMED: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
        static SYNCS: RefCell<Vec<(String, PathBuf)>> = const { RefCell::new(Vec::new()) };
    }

    pub fn note_sync(what: &str, path: &Path) {
        SYNCS.with(|s| s.borrow_mut().push((what.to_owned(), key(path))));
    }

    pub fn take_syncs() -> Vec<(String, PathBuf)> {
        SYNCS.with(|s| std::mem::take(&mut *s.borrow_mut()))
    }

    /// Canonical key for `path`: the canonical parent directory joined
    /// with the file name. The parent outlives every file we track, so
    /// the key stays stable even after the file itself is removed.
    fn key(path: &Path) -> PathBuf {
        let name = path.file_name().map(std::ffi::OsStr::to_os_string);
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        match (parent.canonicalize(), name) {
            (Ok(dir), Some(name)) => dir.join(name),
            _ => path.to_path_buf(),
        }
    }

    pub fn note_open(path: &Path) {
        let k = key(path);
        keys_by_path().insert(path.to_path_buf(), k.clone());
        *open_handles().entry(k).or_insert(0) += 1;
    }

    pub fn note_close(path: &Path) {
        let mut open = open_handles();
        let mut k = key(path);
        if !open.contains_key(&k) {
            if let Some(recorded) = keys_by_path().get(path) {
                k.clone_from(recorded);
            }
        }
        match open.get_mut(&k) {
            Some(n) if *n > 1 => *n -= 1,
            Some(_) => {
                open.remove(&k);
                keys_by_path().retain(|_, v| *v != k);
            }
            None if std::thread::panicking() => unmatched().push(path.to_path_buf()),
            None => panic!(
                "open-file audit: close of `{}` which was never recorded as open",
                path.display()
            ),
        }
    }

    pub fn is_open(path: &Path) -> bool {
        open_handles().contains_key(&key(path))
    }

    pub fn check_replace_target(path: &Path) {
        assert!(
            !is_open(path),
            "open-file audit: rename over `{}` while it is still open or mapped \
             (fails with ERROR_ACCESS_DENIED on Windows)",
            path.display()
        );
    }

    pub fn check_remove(path: &Path) {
        assert!(
            !is_open(path),
            "open-file audit: delete of `{}` while it is still open or mapped \
             (fails with ERROR_ACCESS_DENIED on Windows)",
            path.display()
        );
    }

    pub fn check_set_len(path: &Path, append_only_handle: bool) {
        assert!(
            !append_only_handle,
            "open-file audit: set_len on `{}` through a handle opened with append \
             and without write (Windows grants FILE_APPEND_DATA but not \
             FILE_WRITE_DATA, so truncation fails with ERROR_ACCESS_DENIED)",
            path.display()
        );
    }

    /// Crash points armed for whichever thread hits them first (the background flush thread)
    static ARMED_ANY: std::sync::Mutex<BTreeSet<String>> = std::sync::Mutex::new(BTreeSet::new());

    pub fn arm_crash_point(name: &str) {
        ARMED.with(|a| a.borrow_mut().insert(name.to_owned()));
    }

    pub fn arm_crash_point_any_thread(name: &str) {
        ARMED_ANY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.to_owned());
    }

    pub fn is_armed_any_thread(name: &str) -> bool {
        ARMED_ANY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(name)
    }

    pub fn disarm_all_crash_points() {
        ARMED.with(|a| a.borrow_mut().clear());
        ARMED_ANY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    pub fn take_crash_point(name: &str) -> bool {
        ARMED.with(|a| a.borrow_mut().remove(name))
            || ARMED_ANY
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(name)
    }
}

/// Record that a handle (or mmap) on `path` was opened.
#[inline]
pub(crate) fn note_open(path: &Path) {
    #[cfg(feature = "open-file-audit")]
    imp::note_open(path);
    #[cfg(not(feature = "open-file-audit"))]
    let _ = path;
}

/// Record that a handle (or mmap) on `path` was closed.
#[inline]
pub(crate) fn note_close(path: &Path) {
    #[cfg(feature = "open-file-audit")]
    imp::note_close(path);
    #[cfg(not(feature = "open-file-audit"))]
    let _ = path;
}

/// Assert that `path` may be used as the destination of a rename.
#[inline]
pub(crate) fn check_replace_target(path: &Path) {
    #[cfg(feature = "open-file-audit")]
    imp::check_replace_target(path);
    #[cfg(not(feature = "open-file-audit"))]
    let _ = path;
}

/// Assert that `path` may be deleted.
#[inline]
pub(crate) fn check_remove(path: &Path) {
    #[cfg(feature = "open-file-audit")]
    imp::check_remove(path);
    #[cfg(not(feature = "open-file-audit"))]
    let _ = path;
}

/// Assert that a handle on `path` may be truncated with `set_len`.
#[inline]
pub(crate) fn check_set_len(path: &Path, append_only_handle: bool) {
    #[cfg(feature = "open-file-audit")]
    imp::check_set_len(path, append_only_handle);
    #[cfg(not(feature = "open-file-audit"))]
    let _ = (path, append_only_handle);
}

/// Record a completed fsync of `path` (`what` names its kind: `file`,
/// `dir`, `wal`) on the current thread, so tests can check that each
/// durable step actually fsynced. No-op without `open-file-audit`.
#[inline]
pub(crate) fn note_sync(what: &str, path: &Path) {
    #[cfg(feature = "open-file-audit")]
    imp::note_sync(what, path);
    #[cfg(not(feature = "open-file-audit"))]
    let _ = (what, path);
}

/// `sync_all` on `file`, the handle of `path`, recorded as `what`.
///
/// This and [`sync_data`] are the only places in `src/` that call the
/// fsync methods. The record is written after the call returns, from
/// this function, so it shows that the step reached its fsync; it cannot
/// observe the system call itself, so removing the `sync_all` line below
/// would still be recorded. That line is the one fsync the recording
/// cannot check, which is why this function is kept to it alone.
///
/// # Errors
/// Propagates the `sync_all` error.
pub(crate) fn sync_all(file: &File, what: &str, path: &Path) -> io::Result<()> {
    file.sync_all()?;
    note_sync(what, path);
    Ok(())
}

/// `sync_data` on `file`, recorded like [`sync_all`] (same limit).
///
/// # Errors
/// Propagates the `sync_data` error.
pub(crate) fn sync_data(file: &File, what: &str, path: &Path) -> io::Result<()> {
    file.sync_data()?;
    note_sync(what, path);
    Ok(())
}

/// The fsyncs recorded on the current thread since the last call, in
/// order, as `(kind, canonical path)`; clears the record.
#[cfg(feature = "open-file-audit")]
#[must_use]
pub fn take_syncs() -> Vec<(String, std::path::PathBuf)> {
    imp::take_syncs()
}

/// Return an error if the crash point `name` has been armed on this
/// thread (and disarm it). Always `Ok(())` without `open-file-audit`.
///
/// # Errors
/// Returns `io::ErrorKind::Other` when the point fires.
#[inline]
pub(crate) fn crash_point(name: &str) -> io::Result<()> {
    #[cfg(feature = "open-file-audit")]
    if imp::take_crash_point(name) {
        return Err(io::Error::other(format!("injected crash at `{name}`")));
    }
    #[cfg(not(feature = "open-file-audit"))]
    let _ = name;
    Ok(())
}

/// Closes that matched no open and were reported while their thread was already
/// unwinding (recorded instead of a second panic, which would abort); clears the record.
#[cfg(feature = "open-file-audit")]
#[must_use]
pub fn take_unmatched_closes() -> Vec<std::path::PathBuf> {
    imp::take_unmatched_closes()
}

/// Whether some handle on `path` is currently recorded as open.
#[cfg(feature = "open-file-audit")]
#[must_use]
pub fn is_open(path: &Path) -> bool {
    imp::is_open(path)
}

/// Arm the crash point `name` for the next hit on the current thread.
///
/// Names in use: `compact:after_write`, `compact:after_swap`,
/// `compact:before_remove`, `compact:after_first_remove`,
/// `compact:before_wal_truncate`, `wal_recover:before_cut`, `segment:before_write`, `segment:before_sync`,
/// `segment:before_index`, `ts_wal:mid_frame`, `ts_wal:before_sync`, `replay:before_segment_sync`,
/// `replay:before_wal_truncate`, `index:mid_write`, `index:before_sync`, `index:before_rename`,
/// `index:before_dir_sync`.
#[cfg(feature = "open-file-audit")]
pub fn arm_crash_point(name: &str) {
    imp::arm_crash_point(name);
}

/// Arm the crash point `name` for the next hit on any thread (a background thread's work, such as the time-series
/// flush thread).
#[cfg(feature = "open-file-audit")]
pub fn arm_crash_point_any_thread(name: &str) {
    imp::arm_crash_point_any_thread(name);
}

/// Whether the crash point `name` armed for any thread is still waiting for its hit.
#[cfg(feature = "open-file-audit")]
#[must_use]
pub fn is_armed_any_thread(name: &str) -> bool {
    imp::is_armed_any_thread(name)
}

/// Disarm every crash point on the current thread, and every one armed for any thread.
#[cfg(feature = "open-file-audit")]
pub fn disarm_all_crash_points() {
    imp::disarm_all_crash_points();
}

#[cfg(feature = "open-file-audit")]
thread_local! {
    static LOCK_PROBE_FAILS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Make the shared-writer open's lock probe on the current thread report that the file system's locks do
/// not hold between handles (as on a network file system that accepts a lock and enforces nothing), until
/// set back. Thread-local like the crash points, so tests running beside it are not affected. Tests of the
/// refusal path only.
#[cfg(feature = "open-file-audit")]
pub fn force_lock_probe_failure(on: bool) {
    LOCK_PROBE_FAILS.with(|f| f.set(on));
}

/// Whether [`force_lock_probe_failure`] is on. Always `false` without `open-file-audit`.
#[inline]
pub(crate) fn lock_probe_forced_failure() -> bool {
    #[cfg(feature = "open-file-audit")]
    return LOCK_PROBE_FAILS.with(std::cell::Cell::get);
    #[cfg(not(feature = "open-file-audit"))]
    false
}

#[cfg(all(test, feature = "open-file-audit"))]
mod tests {
    use super::{note_close, note_open};
    use std::path::Path;

    /// A destructor that reports a close while its thread is already unwinding must not
    /// turn the first failure into an abort of the whole process.
    #[test]
    fn a_close_while_unwinding_does_not_abort() {
        struct ClosesOnDrop;
        impl Drop for ClosesOnDrop {
            fn drop(&mut self) {
                note_close(Path::new("no-such-directory-for-the-audit/never-opened"));
            }
        }
        let result = std::panic::catch_unwind(|| {
            let _closes = ClosesOnDrop;
            panic!("the first failure");
        });
        assert!(result.is_err(), "the first panic is the one reported");
        assert!(super::take_unmatched_closes()
            .iter()
            .any(|p| p.ends_with("never-opened")));
    }

    /// A handle closed after its directory was removed is the handle that was opened.
    #[test]
    fn a_close_after_the_directory_is_removed_matches_the_open() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("held");
        note_open(&path);
        drop(tmp);
        let result = std::panic::catch_unwind(|| note_close(&path));
        assert!(result.is_ok(), "the close was matched to its open");
    }
}
