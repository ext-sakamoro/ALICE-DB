//! A store's files follow the name of its WAL: `x.wal` uses `x.sst` and
//! `x-000001.sst`, and nothing else in the directory.
#![cfg(feature = "fs")]

use alice_db::blob::{BlobStorage, BlobStorageConfig};
use alice_db::blob_sstable::FlushMode;
use std::io::ErrorKind;
use std::path::Path;
use tempfile::TempDir;

fn config(flush_mode: FlushMode) -> BlobStorageConfig {
    BlobStorageConfig {
        flush_mode,
        ..BlobStorageConfig::default()
    }
}

fn names(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    out.sort();
    out
}

const MODES: [FlushMode; 2] = [FlushMode::Overwrite, FlushMode::Append];

#[test]
fn flushed_data_of_any_wal_name_is_there_after_reopen() {
    for mode in MODES {
        for wal in ["x.wal", "blob.wal", "a.b.wal", "no_extension"] {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join(wal);
            {
                let store = BlobStorage::open_with_config(&path, config(mode)).unwrap();
                store.put(b"k", b"v").unwrap();
                store.flush_to_sstable().unwrap();
            }
            let store = BlobStorage::open_with_config(&path, config(mode)).unwrap();
            assert_eq!(
                store.get(b"k").unwrap().as_deref(),
                Some(&b"v"[..]),
                "{wal} {mode:?}"
            );
            let view = BlobStorage::open_existing_read_only(&path).unwrap();
            assert_eq!(
                view.get(b"k").unwrap().as_deref(),
                Some(&b"v"[..]),
                "read-only {wal} {mode:?}"
            );
        }
    }
}

#[test]
fn the_files_are_named_after_the_wal() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("x.wal");
    let store = BlobStorage::open_with_config(&path, config(FlushMode::Append)).unwrap();
    store.put(b"k", b"v").unwrap();
    store.flush_to_sstable().unwrap();
    drop(store);
    assert_eq!(names(tmp.path()), ["x-000001.sst", "x.wal"]);

    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("x.wal");
    let store = BlobStorage::open_with_config(&path, config(FlushMode::Overwrite)).unwrap();
    store.put(b"k", b"v").unwrap();
    store.flush_to_sstable().unwrap();
    drop(store);
    assert_eq!(names(tmp.path()), ["x.sst", "x.wal"]);
}

#[test]
fn a_blob_wal_keeps_its_file_names() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let store = BlobStorage::open_with_config(&path, config(FlushMode::Append)).unwrap();
    store.put(b"k", b"v").unwrap();
    store.flush_to_sstable().unwrap();
    drop(store);
    assert_eq!(names(tmp.path()), ["blob-000001.sst", "blob.wal"]);

    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let store = BlobStorage::open_with_config(&path, config(FlushMode::Overwrite)).unwrap();
    store.put(b"k", b"v").unwrap();
    store.flush_to_sstable().unwrap();
    drop(store);
    assert_eq!(names(tmp.path()), ["blob.sst", "blob.wal"]);
}

/// Two stores in one directory: neither sees, replaces nor removes the other's data.
#[test]
fn two_stores_in_one_directory_stay_apart() {
    for mode in MODES {
        let tmp = TempDir::new().unwrap();
        let a_path = tmp.path().join("a.wal");
        let blob_path = tmp.path().join("blob.wal");
        let a = BlobStorage::open_with_config(&a_path, config(mode)).unwrap();
        let blob = BlobStorage::open_with_config(&blob_path, config(mode)).unwrap();
        for round in 0..3_u8 {
            a.put(&[b'a', round], b"from a").unwrap();
            a.flush_to_sstable().unwrap();
            blob.put(&[b'b', round], b"from blob").unwrap();
            blob.flush_to_sstable().unwrap();
        }
        a.compact_all_sstables().unwrap();
        blob.compact_all_sstables().unwrap();
        drop(a);
        drop(blob);

        let a_view = BlobStorage::open_existing_read_only(&a_path).unwrap();
        let blob_view = BlobStorage::open_existing_read_only(&blob_path).unwrap();
        for round in 0..3_u8 {
            assert!(a_view.get(&[b'a', round]).unwrap().is_some(), "{mode:?}");
            assert_eq!(a_view.get(&[b'b', round]).unwrap(), None, "{mode:?}");
            assert!(blob_view.get(&[b'b', round]).unwrap().is_some(), "{mode:?}");
            assert_eq!(blob_view.get(&[b'a', round]).unwrap(), None, "{mode:?}");
        }

        let a = BlobStorage::open_with_config(&a_path, config(mode)).unwrap();
        let blob = BlobStorage::open_with_config(&blob_path, config(mode)).unwrap();
        for round in 0..3_u8 {
            let (a_key, b_key) = ([b'a', round], [b'b', round]);
            assert_eq!(
                a.get(&a_key).unwrap().as_deref(),
                Some(&b"from a"[..]),
                "{mode:?}"
            );
            assert_eq!(a.get(&b_key).unwrap(), None, "{mode:?}");
            assert_eq!(
                blob.get(&b_key).unwrap().as_deref(),
                Some(&b"from blob"[..]),
                "{mode:?}"
            );
            assert_eq!(blob.get(&a_key).unwrap(), None, "{mode:?}");
        }
    }
}

/// Each store counts its own numbered files.
#[test]
fn numbering_does_not_depend_on_another_store() {
    let tmp = TempDir::new().unwrap();
    let blob =
        BlobStorage::open_with_config(tmp.path().join("blob.wal"), config(FlushMode::Append))
            .unwrap();
    for _ in 0..3 {
        blob.put(b"k", b"v").unwrap();
        blob.flush_to_sstable().unwrap();
    }
    let x =
        BlobStorage::open_with_config(tmp.path().join("x.wal"), config(FlushMode::Append)).unwrap();
    x.put(b"k", b"v").unwrap();
    x.flush_to_sstable().unwrap();
    assert!(tmp.path().join("x-000001.sst").exists());
    assert!(!tmp.path().join("x-000004.sst").exists());
}

/// A file that a store's flush left under its own name on an earlier build is
/// read: the WAL no longer holds that data.
#[test]
fn a_file_under_the_name_of_the_wal_is_read() {
    let tmp = TempDir::new().unwrap();
    let helper = TempDir::new().unwrap();
    let store = BlobStorage::open(helper.path().join("blob.wal")).unwrap();
    store.put(b"k", b"v").unwrap();
    store.flush_to_sstable().unwrap();
    std::fs::copy(helper.path().join("blob.sst"), tmp.path().join("x.sst")).unwrap();

    let store = BlobStorage::open(tmp.path().join("x.wal")).unwrap();
    assert_eq!(store.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
}

/// Opening one store leaves the unfinished write of another store alone.
#[test]
fn opening_a_store_keeps_the_temporary_file_of_another() {
    let tmp = TempDir::new().unwrap();
    let other = tmp.path().join("blob.sst.tmp");
    std::fs::write(&other, b"being written").unwrap();
    let own = tmp.path().join("x.sst.tmp");
    std::fs::write(&own, b"left by a crash").unwrap();

    drop(BlobStorage::open(tmp.path().join("x.wal")).unwrap());

    assert!(other.exists());
    assert!(!own.exists());
}

/// An earlier build wrote an append-mode store's files as `blob-*.sst` whatever
/// the WAL was called. They cannot be shown to belong to this WAL, and going on
/// without them would look like lost data, so the open stops and says so.
#[test]
fn files_of_the_earlier_layout_stop_the_open_and_are_left_alone() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("x.wal");
    std::fs::write(&path, b"").unwrap();
    let sst = tmp.path().join("blob-000001.sst");
    {
        let helper = TempDir::new().unwrap();
        let store = BlobStorage::open_with_config(
            helper.path().join("blob.wal"),
            config(FlushMode::Append),
        )
        .unwrap();
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
        std::fs::copy(helper.path().join("blob-000001.sst"), &sst).unwrap();
    }
    let before = std::fs::read(&sst).unwrap();

    let err = BlobStorage::open(&path).expect_err("the open must stop");
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("blob-000001.sst"), "{err}");
    let err =
        BlobStorage::open_existing_read_only(&path).expect_err("the read-only open must stop too");
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert_eq!(std::fs::read(&sst).unwrap(), before);

    // Renaming the file as the message says brings the data back.
    std::fs::rename(&sst, tmp.path().join("x-000001.sst")).unwrap();
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(store.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
}

/// With a `blob.wal` beside it, `blob-*.sst` belongs to that store.
#[test]
fn files_of_a_blob_wal_beside_it_do_not_stop_the_open() {
    for mode in MODES {
        let tmp = TempDir::new().unwrap();
        let blob =
            BlobStorage::open_with_config(tmp.path().join("blob.wal"), config(mode)).unwrap();
        blob.put(b"k", b"v").unwrap();
        blob.flush_to_sstable().unwrap();
        let x = BlobStorage::open(tmp.path().join("x.wal")).unwrap();
        assert_eq!(x.get(b"k").unwrap(), None, "{mode:?}");
    }
}

/// `x-000001.wal` would have the file name of the first append file of `x.wal`.
#[test]
fn a_wal_name_that_reads_as_a_numbered_file_is_refused() {
    let tmp = TempDir::new().unwrap();
    for wal in ["x-000001.wal", "blob-000001.wal"] {
        let err = BlobStorage::open(tmp.path().join(wal)).expect_err("the open must stop");
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "{wal}");
        assert!(!tmp.path().join(wal).exists(), "{wal}");
    }
}
