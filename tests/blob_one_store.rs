//! One directory holds one store. Its files are `blob.sst` and `blob-000001.sst`
//! whatever the WAL is called, and a second WAL name in the directory is refused.
//!
//! The stores under `tests/fixtures/old_layout_e5c0d81` were written by the build
//! named in its README.
#![cfg(feature = "fs")]
#![allow(
    clippy::disallowed_methods,
    reason = "one test starts processes from a thread, joined before it returns"
)]

use alice_db::blob::{BlobStorage, BlobStorageConfig};
use alice_db::blob_sstable::FlushMode;
use alice_db::AliceDB;
use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

const MODES: [FlushMode; 2] = [FlushMode::Overwrite, FlushMode::Append];

fn config(flush_mode: FlushMode) -> BlobStorageConfig {
    BlobStorageConfig {
        flush_mode,
        ..BlobStorageConfig::default()
    }
}

fn open(path: &Path, mode: FlushMode) -> BlobStorage {
    BlobStorage::open_with_config(path, config(mode)).unwrap()
}

fn files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().into_string().unwrap(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect()
}

fn fixture(name: &str) -> TempDir {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let tmp = TempDir::new().unwrap();
    let mut copied = 0;
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "README.md" {
            std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
            copied += 1;
        }
    }
    assert!(copied > 0);
    tmp
}

fn get(store: &BlobStorage, key: &str) -> Option<String> {
    store
        .get(key.as_bytes())
        .unwrap()
        .map(|v| String::from_utf8(v).unwrap())
}

fn refused(path: &Path) -> std::io::Error {
    BlobStorage::open(path).expect_err("the open must stop")
}

#[test]
fn flushed_data_of_any_wal_name_is_there_after_a_reopen() {
    for mode in MODES {
        for wal in [
            "x.wal",
            "blob.wal",
            "a.b.wal",
            "no_extension",
            ".wal",
            "x-000001.wal",
        ] {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join(wal);
            {
                let store = open(&path, mode);
                store.put(b"k", b"v").unwrap();
                store.flush_to_sstable().unwrap();
            }
            assert_eq!(
                get(&open(&path, mode), "k"),
                Some("v".to_owned()),
                "{wal} {mode:?}"
            );
            let view = BlobStorage::open_existing_read_only(&path).unwrap();
            assert_eq!(
                get(&view, "k"),
                Some("v".to_owned()),
                "read-only {wal} {mode:?}"
            );
        }
    }
}

#[test]
fn the_file_names_do_not_depend_on_the_wal() {
    for (mode, sst) in [
        (FlushMode::Overwrite, "blob.sst"),
        (FlushMode::Append, "blob-000001.sst"),
    ] {
        let tmp = TempDir::new().unwrap();
        let store = open(&tmp.path().join("x.wal"), mode);
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
        let names = files(tmp.path());
        assert!(names.contains_key(sst), "{names:?}");
        assert!(!names.contains_key("x.sst"), "{names:?}");
    }
}

#[test]
fn a_second_name_in_the_directory_is_refused_while_the_first_is_open() {
    for mode in MODES {
        for (a, b) in [
            ("x.wal", "x.log"),
            ("x.wal", "y.wal"),
            ("x.a", "x.b"),
            ("blob.wal", "blob.log"),
            ("x", "x.wal"),
        ] {
            let tmp = TempDir::new().unwrap();
            let first = open(&tmp.path().join(a), mode);
            first.put(b"k", b"first").unwrap();
            first.flush_to_sstable().unwrap();
            let err = refused(&tmp.path().join(b));
            assert_eq!(err.kind(), ErrorKind::WouldBlock, "{a} {b} {mode:?}: {err}");
            assert!(!tmp.path().join(b).exists(), "{a} {b}: the refused WAL");
            assert_eq!(get(&first, "k"), Some("first".to_owned()));
        }
    }
}

#[test]
fn a_second_name_in_the_directory_is_refused_afterwards_too() {
    for mode in MODES {
        for (a, b) in [
            ("x.wal", "x.log"),
            ("x.wal", "y.wal"),
            ("blob.wal", "x.wal"),
        ] {
            let tmp = TempDir::new().unwrap();
            {
                let first = open(&tmp.path().join(a), mode);
                first.put(b"k", b"first").unwrap();
                first.flush_to_sstable().unwrap();
                first.put(b"w", b"unflushed").unwrap();
            }
            let before = files(tmp.path());
            let err = refused(&tmp.path().join(b));
            assert_eq!(err.kind(), ErrorKind::InvalidData, "{a} {b}: {err}");
            assert!(err.to_string().contains(a), "{err}");
            assert_eq!(files(tmp.path()), before, "{a} {b}: the refused open");
            let first = open(&tmp.path().join(a), mode);
            assert_eq!(get(&first, "k"), Some("first".to_owned()));
            assert_eq!(get(&first, "w"), Some("unflushed".to_owned()));
        }
    }
}

/// The name that is recorded is gone: a rename and a new store look the same.
#[test]
fn a_new_name_over_the_files_of_a_store_whose_wal_is_gone_is_refused() {
    let tmp = TempDir::new().unwrap();
    let x = tmp.path().join("x.wal");
    {
        let store = BlobStorage::open(&x).unwrap();
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
    }
    std::fs::remove_file(&x).unwrap();
    let before = files(tmp.path());
    let err = refused(&tmp.path().join("y.wal"));
    assert_eq!(err.kind(), ErrorKind::InvalidData, "{err}");
    assert!(err.to_string().contains("take_over_store"), "{err}");
    assert_eq!(files(tmp.path()), before);
}

/// Nothing to mix up: the store's name may change.
#[test]
fn a_new_name_is_taken_when_the_recorded_wal_and_the_files_are_gone() {
    let tmp = TempDir::new().unwrap();
    let x = tmp.path().join("x.wal");
    drop(BlobStorage::open(&x).unwrap());
    std::fs::remove_file(&x).unwrap();
    drop(BlobStorage::open(tmp.path().join("y.wal")).unwrap());
    let err = refused(&x);
    assert!(err.to_string().contains("y.wal"), "{err}");
}

/// The WAL of a store with files was removed and the store is opened by the same name.
#[test]
fn the_same_name_opens_when_its_wal_is_gone() {
    let tmp = TempDir::new().unwrap();
    let x = tmp.path().join("x.wal");
    {
        let store = BlobStorage::open(&x).unwrap();
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
    }
    std::fs::remove_file(&x).unwrap();
    assert_eq!(
        get(&BlobStorage::open(&x).unwrap(), "k"),
        Some("v".to_owned())
    );
}

/// `blob.wal` is the name these files have always gone with.
#[test]
fn blob_wal_opens_over_files_with_no_record_when_it_is_gone() {
    let tmp = fixture("old_layout_e5c0d81/blob_wal");
    let path = tmp.path().join("blob.wal");
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        get(&BlobStorage::open(&path).unwrap(), "k"),
        Some("v".to_owned())
    );
}

/// A directory no version of this build opened: a new name is not their store.
#[test]
fn a_new_name_over_files_with_no_record_is_refused() {
    let tmp = fixture("old_layout_e5c0d81/blob_wal");
    std::fs::remove_file(tmp.path().join("blob.wal")).unwrap();
    let before = files(tmp.path());
    let err = refused(&tmp.path().join("y.wal"));
    assert_eq!(err.kind(), ErrorKind::InvalidData, "{err}");
    assert!(err.to_string().contains("take_over_store"), "{err}");
    let mut after = files(tmp.path());
    after.remove("blob.lock");
    assert_eq!(after, before);
}

#[test]
fn a_store_of_an_earlier_version_under_its_own_name_is_read() {
    for mode in MODES {
        // x.sst, which that version wrote and never read
        let tmp = fixture("old_layout_e5c0d81/overwrite");
        let path = tmp.path().join("x.wal");
        let view = BlobStorage::open_existing_read_only(&path).unwrap();
        assert_eq!(get(&view, "k"), Some("v".to_owned()), "read-only {mode:?}");
        assert_eq!(get(&view, "w"), Some("u".to_owned()), "read-only {mode:?}");
        assert!(
            tmp.path().join("x.sst").exists(),
            "a read-only open moved it"
        );
        let store = open(&path, mode);
        assert_eq!(get(&store, "k"), Some("v".to_owned()), "{mode:?}");
        assert_eq!(get(&store, "w"), Some("u".to_owned()), "{mode:?}");
        assert!(!tmp.path().join("x.sst").exists());
        assert!(tmp.path().join("blob.sst").exists());
        drop(store);
        assert_eq!(
            get(&open(&path, mode), "k"),
            Some("v".to_owned()),
            "reopen {mode:?}"
        );

        // blob-000001.sst, under a WAL of another name
        let tmp = fixture("old_layout_e5c0d81/append");
        let path = tmp.path().join("x.wal");
        let view = BlobStorage::open_existing_read_only(&path).unwrap();
        assert_eq!(get(&view, "k"), Some("v".to_owned()), "read-only {mode:?}");
        let store = open(&path, mode);
        assert_eq!(get(&store, "k"), Some("v".to_owned()), "{mode:?}");
        assert_eq!(get(&store, "w"), Some("u".to_owned()), "{mode:?}");
    }
}

/// `x.sst` and `y.sst` each with its WAL: the first open wins, the second stops,
/// and moving the second store away gives it its data.
#[test]
fn of_two_stores_in_one_directory_the_second_stops_and_can_move() {
    for (first, second) in [("x", "y"), ("y", "x")] {
        let tmp = fixture("old_layout_e5c0d81/two_overwrite");
        let wal = |n: &str| tmp.path().join(format!("{n}.wal"));
        let k = |n: &str| format!("k{n}");
        let w = |n: &str| format!("w{n}");

        let opened = BlobStorage::open(wal(first)).unwrap();
        assert_eq!(get(&opened, &k(first)), Some(format!("v{first}")));
        assert_eq!(get(&opened, &w(first)), Some(format!("u{first}")));
        assert_eq!(get(&opened, &k(second)), None);

        let held = refused(&wal(second));
        assert_eq!(held.kind(), ErrorKind::WouldBlock, "{held}");
        drop(opened);
        let wal_bytes = std::fs::read(wal(second)).unwrap();
        let sst = tmp.path().join(format!("{second}.sst"));
        let sst_bytes = std::fs::read(&sst).unwrap();
        let err = refused(&wal(second));
        assert_eq!(err.kind(), ErrorKind::InvalidData, "{err}");
        let view = BlobStorage::open_existing_read_only(wal(second)).unwrap_err();
        assert_eq!(view.kind(), ErrorKind::InvalidData, "{view}");
        assert_eq!(std::fs::read(wal(second)).unwrap(), wal_bytes);
        assert_eq!(std::fs::read(&sst).unwrap(), sst_bytes);

        // The first is still whole after the refused opens.
        let opened = BlobStorage::open(wal(first)).unwrap();
        assert_eq!(get(&opened, &k(first)), Some(format!("v{first}")));
        assert_eq!(get(&opened, &w(first)), Some(format!("u{first}")));
        drop(opened);

        // What CHANGELOG says to do.
        let own = TempDir::new().unwrap();
        let moved = own.path().join(format!("{second}.wal"));
        std::fs::rename(wal(second), &moved).unwrap();
        std::fs::rename(&sst, own.path().join(format!("{second}.sst"))).unwrap();
        let store = BlobStorage::open(&moved).unwrap();
        assert_eq!(get(&store, &k(second)), Some(format!("v{second}")));
        assert_eq!(get(&store, &w(second)), Some(format!("u{second}")));
    }
}

/// Numbered files that an earlier version wrote for two WALs cannot be divided:
/// the first open reads them all, as that version did, and the second stops.
#[test]
fn numbered_files_of_two_wals_are_read_by_the_first_and_the_second_stops() {
    for (first, second) in [("x.wal", "blob.wal"), ("blob.wal", "x.wal")] {
        let tmp = fixture("old_layout_e5c0d81/x_and_blob_append");
        let store = BlobStorage::open(tmp.path().join(first)).unwrap();
        assert_eq!(get(&store, "kx"), Some("vx".to_owned()), "{first}");
        assert_eq!(get(&store, "kb"), Some("vb".to_owned()), "{first}");
        drop(store);
        let err = refused(&tmp.path().join(second));
        assert_eq!(err.kind(), ErrorKind::InvalidData, "{first}: {err}");
    }
}

#[test]
fn a_file_under_the_wal_name_and_numbered_files_stop_the_open_and_stay() {
    let tmp = fixture("old_layout_e5c0d81/overwrite");
    let other = fixture("old_layout_e5c0d81/blob_wal");
    std::fs::copy(
        other.path().join("blob-000001.sst"),
        tmp.path().join("blob-000001.sst"),
    )
    .unwrap();
    let before = files(tmp.path());
    let path = tmp.path().join("x.wal");
    for err in [
        refused(&path),
        BlobStorage::open_existing_read_only(&path).unwrap_err(),
    ] {
        assert_eq!(err.kind(), ErrorKind::InvalidData, "{err}");
        let text = err.to_string();
        assert!(
            text.contains("x.sst") && text.contains("blob-000001.sst"),
            "{text}"
        );
        assert!(!text.contains("rename"), "{text}");
    }
    let mut after = files(tmp.path());
    after.remove("blob.lock");
    assert_eq!(after, before);
}

/// The way a tool outside this crate opens its store, on what that version wrote.
#[test]
fn a_store_of_0_2_0_beta_2_reopens_through_aliceds_open() {
    for mode in MODES {
        let tmp = fixture("store_db_0_2_0_beta_2");
        let db = AliceDB::open_with_blob_config(tmp.path(), config(mode)).unwrap();
        assert_eq!(db.get_blob(b"k").unwrap(), Some(b"v".to_vec()), "{mode:?}");
        assert_eq!(db.get_blob(b"w").unwrap(), Some(b"u".to_vec()), "{mode:?}");
        db.put_blob(b"n", b"m").unwrap();
        db.compact_blob_sstable().unwrap();
        drop(db);
        let db = AliceDB::open(tmp.path()).unwrap();
        for (k, v) in [(b"k", b"v"), (b"w", b"u"), (b"n", b"m")] {
            assert_eq!(db.get_blob(k).unwrap(), Some(v.to_vec()), "{mode:?}");
        }
    }
}

/// The same file under another spelling is the same store, on a file system
/// that treats the two as one name. Linux names are exact, so `Blob.wal` is another
/// WAL there and the test above covers it.
#[cfg(any(target_os = "macos", windows))]
#[test]
fn another_case_of_the_name_is_the_same_store() {
    for mode in MODES {
        for (written, spelled) in [("blob.wal", "Blob.wal"), ("x.wal", "X.WAL")] {
            let tmp = TempDir::new().unwrap();
            {
                let store = open(&tmp.path().join(written), mode);
                store.put(b"k", b"v").unwrap();
                store.flush_to_sstable().unwrap();
            }
            let store = open(&tmp.path().join(spelled), mode);
            assert_eq!(get(&store, "k"), Some("v".to_owned()), "{spelled} {mode:?}");
            let err = refused(&tmp.path().join(written));
            assert_eq!(err.kind(), ErrorKind::WouldBlock, "{err}");
        }
    }
}

/// APFS treats composed and decomposed forms of a name as one.
#[cfg(target_os = "macos")]
#[test]
fn another_normal_form_of_the_name_is_the_same_store() {
    for mode in MODES {
        let tmp = TempDir::new().unwrap();
        {
            let store = open(&tmp.path().join("caf\u{e9}.wal"), mode);
            store.put(b"k", b"v").unwrap();
            store.flush_to_sstable().unwrap();
        }
        let store = open(&tmp.path().join("cafe\u{301}.wal"), mode);
        assert_eq!(get(&store, "k"), Some("v".to_owned()), "{mode:?}");
    }
}

#[cfg(unix)]
#[test]
fn a_symlink_in_the_directory_is_the_same_store() {
    for mode in MODES {
        let tmp = TempDir::new().unwrap();
        let real = tmp.path().join("blob.wal");
        {
            let store = open(&real, mode);
            store.put(b"k", b"v").unwrap();
            store.flush_to_sstable().unwrap();
        }
        let link = tmp.path().join("y.wal");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert_eq!(
            get(&open(&link, mode), "k"),
            Some("v".to_owned()),
            "{mode:?}"
        );
        let view = BlobStorage::open_existing_read_only(&link).unwrap();
        assert_eq!(get(&view, "k"), Some("v".to_owned()), "read-only {mode:?}");
    }
}

/// The files next to the link are not the files next to the WAL.
#[cfg(unix)]
#[test]
fn a_symlink_to_a_wal_in_another_directory_is_refused() {
    let home = TempDir::new().unwrap();
    let real = home.path().join("blob.wal");
    drop(BlobStorage::open(&real).unwrap());
    let elsewhere = TempDir::new().unwrap();
    let link = elsewhere.path().join("y.wal");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    for err in [
        refused(&link),
        BlobStorage::open_existing_read_only(&link).unwrap_err(),
    ] {
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "{err}");
    }
}

#[test]
fn a_hard_link_in_the_directory_is_the_same_store() {
    for mode in MODES {
        let tmp = TempDir::new().unwrap();
        let real = tmp.path().join("blob.wal");
        {
            let store = open(&real, mode);
            store.put(b"k", b"v").unwrap();
            store.flush_to_sstable().unwrap();
        }
        let link = tmp.path().join("y.wal");
        std::fs::hard_link(&real, &link).unwrap();
        let opened = BlobStorage::open_with_config(&link, config(mode));
        // The same file by device and inode.
        #[cfg(unix)]
        assert_eq!(get(&opened.unwrap(), "k"), Some("v".to_owned()), "{mode:?}");
        // Final paths, which differ for a hard link: it stops.
        #[cfg(not(unix))]
        assert_eq!(opened.unwrap_err().kind(), ErrorKind::InvalidData);
    }
}

#[test]
fn a_wal_named_like_a_store_file_is_refused_and_not_cut() {
    for wal in [
        "blob.sst",
        "x.sst",
        "X.SST",
        "blob-000001.sst",
        "blob-000001.sst.tmp",
        "blob.lock",
    ] {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join(wal);
        std::fs::write(&path, b"not a log").unwrap();
        for err in [
            refused(&path),
            BlobStorage::open_existing_read_only(&path).unwrap_err(),
        ] {
            assert_eq!(err.kind(), ErrorKind::InvalidInput, "{wal}: {err}");
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"not a log", "{wal}");
    }
}

#[test]
fn a_read_only_open_creates_no_lock_and_needs_none() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("x.wal");
    {
        let store = BlobStorage::open(&path).unwrap();
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
    }
    std::fs::remove_file(tmp.path().join("blob.lock")).unwrap();
    let before = files(tmp.path());
    let view = BlobStorage::open_existing_read_only(&path).unwrap();
    assert_eq!(get(&view, "k"), Some("v".to_owned()));
    assert_eq!(files(tmp.path()), before);
    let names: Vec<PathBuf> = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert!(names.iter().all(|p| !p.ends_with("blob.lock")));
}

#[cfg(unix)]
#[test]
fn a_read_only_open_works_in_a_directory_it_cannot_write() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("store");
    let path = dir.join("x.wal");
    {
        let store = BlobStorage::open(&path).unwrap();
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
    }
    for remove_lock in [false, true] {
        if remove_lock {
            std::fs::remove_file(dir.join("blob.lock")).unwrap();
        }
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let view = BlobStorage::open_existing_read_only(&path);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(
            get(&view.unwrap(), "k"),
            Some("v".to_owned()),
            "remove_lock {remove_lock}"
        );
    }
}

#[test]
fn a_read_only_open_works_beside_a_writer_holding_the_lock() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("x.wal");
    let writer = BlobStorage::open(&path).unwrap();
    writer.put(b"k", b"v").unwrap();
    let view = BlobStorage::open_existing_read_only(&path).unwrap();
    assert_eq!(get(&view, "k"), Some("v".to_owned()));
    let other = BlobStorage::open_existing_read_only(tmp.path().join("y.wal")).unwrap_err();
    assert_eq!(other.kind(), ErrorKind::NotFound);
}

/// A later open replaces recording files by removing what it knows, and the lock
/// file may stay behind.
#[test]
fn a_lock_file_left_by_removed_store_files_does_not_stop_a_new_store() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"k", b"v").unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);
    for entry in std::fs::read_dir(tmp.path()).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "blob.lock" {
            std::fs::remove_file(entry.path()).unwrap();
        }
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.get_blob(b"k").unwrap(), None);
}

/// A process started by another thread holds a copy of the lock's handle until it
/// execs: closing a store has to release the lock itself.
#[test]
fn a_reopen_is_not_stopped_by_a_process_being_started() {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spawner = {
        let stop = std::sync::Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut started = 0_u32;
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                std::process::Command::new("git")
                    .arg("--version")
                    .output()
                    .unwrap();
                started += 1;
            }
            started
        })
    };
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("x.wal");
    for _ in 0..2_000 {
        drop(BlobStorage::open(&path).unwrap());
    }
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(spawner.join().unwrap() > 0, "no process was started");
}
