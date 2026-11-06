//! A crash at each step of an open that moves a store's file to its new name, and
//! of `take_over_store`, leaves the data readable on the next open.
#![cfg(all(feature = "fs", feature = "open-file-audit"))]

use alice_db::blob::BlobStorage;
use alice_db::fs_audit;
use std::path::Path;
use tempfile::TempDir;

fn old_store() -> TempDir {
    let from =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/old_layout_e5c0d81/overwrite");
    let tmp = TempDir::new().unwrap();
    for name in ["x.sst", "x.wal"] {
        std::fs::copy(from.join(name), tmp.path().join(name)).unwrap();
    }
    tmp
}

fn assert_whole(store: &BlobStorage) {
    assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));
    assert_eq!(store.get(b"w").unwrap(), Some(b"u".to_vec()));
}

#[test]
fn a_crash_at_each_step_of_the_move_loses_nothing() {
    for point in [
        "store:before_rename",
        "store:before_dir_sync",
        "store:before_record",
    ] {
        let tmp = old_store();
        let path = tmp.path().join("x.wal");
        fs_audit::arm_crash_point(point);
        let opened = BlobStorage::open(&path);
        fs_audit::disarm_all_crash_points();
        let err = opened.expect_err(point);
        assert!(err.to_string().contains(point), "{err}");

        // Read-only first: it sees the data at whichever name it has now.
        assert_whole(&BlobStorage::open_existing_read_only(&path).unwrap());
        // The next open finishes it, and one after that has nothing left to do.
        assert_whole(&BlobStorage::open(&path).unwrap());
        assert!(!tmp.path().join("x.sst").exists(), "{point}");
        assert!(tmp.path().join("blob.sst").exists(), "{point}");
        assert_whole(&BlobStorage::open(&path).unwrap());
    }
}

/// Power lost before the directory reached the disk can undo the rename.
#[test]
fn an_undone_move_is_done_again() {
    let tmp = old_store();
    let path = tmp.path().join("x.wal");
    fs_audit::arm_crash_point("store:before_dir_sync");
    BlobStorage::open(&path).expect_err("the armed crash point");
    fs_audit::disarm_all_crash_points();
    std::fs::rename(tmp.path().join("blob.sst"), tmp.path().join("x.sst")).unwrap();
    assert_whole(&BlobStorage::open_existing_read_only(&path).unwrap());
    assert_whole(&BlobStorage::open(&path).unwrap());
}

#[cfg(unix)]
#[test]
fn the_directory_is_synced_after_the_move() {
    let tmp = old_store();
    let path = tmp.path().join("x.wal");
    let _ = fs_audit::take_syncs();
    drop(BlobStorage::open(&path).unwrap());
    let syncs = fs_audit::take_syncs();
    assert!(
        syncs.iter().any(|(what, _)| what == "store-dir"),
        "{syncs:?}"
    );
}

#[test]
fn an_interrupted_take_over_can_be_repeated() {
    let tmp = TempDir::new().unwrap();
    let x = tmp.path().join("x.wal");
    {
        let store = BlobStorage::open(&x).unwrap();
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
    }
    std::fs::remove_file(&x).unwrap();
    let y = tmp.path().join("y.wal");
    fs_audit::arm_crash_point("store:before_record");
    let err = BlobStorage::take_over_store(&y).unwrap_err();
    fs_audit::disarm_all_crash_points();
    assert!(err.to_string().contains("store:before_record"), "{err}");
    BlobStorage::open(&y).expect_err("not taken over yet");
    BlobStorage::take_over_store(&y).unwrap();
    assert_eq!(
        BlobStorage::open(&y).unwrap().get(b"k").unwrap(),
        Some(b"v".to_vec())
    );
}
