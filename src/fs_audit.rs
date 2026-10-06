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
//! The module also hosts crash-injection points used by the
//! compaction crash-safety tests: `arm_crash_point` makes the next
//! `crash_point` call with that name on the current thread return an
//! error, which aborts the operation exactly where a crash would.

use std::io;
use std::path::Path;

#[cfg(feature = "open-file-audit")]
mod imp {
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn open_handles() -> MutexGuard<'static, HashMap<PathBuf, usize>> {
        static OPEN: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
        OPEN.get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    thread_local! {
        static ARMED: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
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
        *open_handles().entry(key(path)).or_insert(0) += 1;
    }

    pub fn note_close(path: &Path) {
        let mut open = open_handles();
        let k = key(path);
        match open.get_mut(&k) {
            Some(n) if *n > 1 => *n -= 1,
            Some(_) => {
                open.remove(&k);
            }
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

    pub fn arm_crash_point(name: &str) {
        ARMED.with(|a| a.borrow_mut().insert(name.to_owned()));
    }

    pub fn disarm_all_crash_points() {
        ARMED.with(|a| a.borrow_mut().clear());
    }

    pub fn take_crash_point(name: &str) -> bool {
        ARMED.with(|a| a.borrow_mut().remove(name))
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
/// `compact:before_wal_truncate`.
#[cfg(feature = "open-file-audit")]
pub fn arm_crash_point(name: &str) {
    imp::arm_crash_point(name);
}

/// Disarm every crash point on the current thread.
#[cfg(feature = "open-file-audit")]
pub fn disarm_all_crash_points() {
    imp::disarm_all_crash_points();
}
