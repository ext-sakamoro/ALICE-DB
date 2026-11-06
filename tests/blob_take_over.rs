//! `BlobStorage::take_over_store`: the explicit step that gives a WAL of another
//! name the files of a store whose WAL is gone.
#![cfg(feature = "fs")]

use alice_db::blob::BlobStorage;
use std::io::ErrorKind;
use tempfile::TempDir;

fn store_without_its_wal() -> (TempDir, std::path::PathBuf) {
    let tmp = TempDir::new().unwrap();
    let x = tmp.path().join("x.wal");
    {
        let store = BlobStorage::open(&x).unwrap();
        store.put(b"k", b"v").unwrap();
        store.flush_to_sstable().unwrap();
    }
    std::fs::remove_file(&x).unwrap();
    let y = tmp.path().join("y.wal");
    (tmp, y)
}

#[test]
fn it_gives_the_files_to_the_named_wal() {
    let (tmp, y) = store_without_its_wal();
    BlobStorage::open(&y).expect_err("refused first");
    BlobStorage::take_over_store(&y).unwrap();
    let store = BlobStorage::open(&y).unwrap();
    assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));
    drop(store);
    let err = BlobStorage::open(tmp.path().join("x.wal")).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData, "{err}");
}

#[test]
fn calling_it_twice_is_the_same_as_once() {
    let (_tmp, y) = store_without_its_wal();
    BlobStorage::take_over_store(&y).unwrap();
    BlobStorage::take_over_store(&y).unwrap();
    let store = BlobStorage::open(&y).unwrap();
    assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));
    drop(store);
    BlobStorage::take_over_store(&y).unwrap();
    assert_eq!(
        BlobStorage::open(&y).unwrap().get(b"k").unwrap(),
        Some(b"v".to_vec())
    );
}

#[test]
fn it_is_refused_while_the_recorded_wal_exists() {
    let tmp = TempDir::new().unwrap();
    let x = tmp.path().join("x.wal");
    let y = tmp.path().join("y.wal");
    let writer = BlobStorage::open(&x).unwrap();
    writer.put(b"k", b"unflushed").unwrap();
    let err = BlobStorage::take_over_store(&y).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock, "{err}");
    drop(writer);
    let err = BlobStorage::take_over_store(&y).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData, "{err}");
    assert_eq!(
        BlobStorage::open(&x).unwrap().get(b"k").unwrap(),
        Some(b"unflushed".to_vec())
    );
}

/// A second name of the recorded WAL's file is the same store.
#[cfg(unix)]
#[test]
fn it_takes_a_second_name_of_the_recorded_wal() {
    let tmp = TempDir::new().unwrap();
    let x = tmp.path().join("x.wal");
    let store = BlobStorage::open(&x).unwrap();
    store.put(b"k", b"v").unwrap();
    drop(store);
    let y = tmp.path().join("y.wal");
    std::fs::hard_link(&x, &y).unwrap();
    BlobStorage::take_over_store(&y).unwrap();
    std::fs::remove_file(&x).unwrap();
    assert_eq!(
        BlobStorage::open(&y).unwrap().get(b"k").unwrap(),
        Some(b"v".to_vec())
    );
}

/// A record cut short by a crash counts as none: nothing opens by mistake and the
/// call can be repeated.
#[test]
fn a_record_cut_short_is_no_record() {
    let (tmp, y) = store_without_its_wal();
    let lock = tmp.path().join("blob.lock");
    let whole = std::fs::read(&lock).unwrap();
    for cut in 0..whole.len() {
        std::fs::write(&lock, &whole[..cut]).unwrap();
        BlobStorage::open(&y).expect_err("still not this WAL's");
        BlobStorage::take_over_store(&y).unwrap();
        assert_eq!(
            BlobStorage::open(&y).unwrap().get(b"k").unwrap(),
            Some(b"v".to_vec())
        );
        std::fs::remove_file(&y).unwrap();
        std::fs::write(&lock, &whole).unwrap();
    }
}

/// A later version's record under this very name, or a path out of the directory,
/// is not taken for this store's record, with or without files to protect.
#[test]
fn a_record_of_another_form_stops_the_open_of_an_empty_directory() {
    for bytes in [&b"2\ny.wal\n"[..], b"1\n../y.wal\n", b"1\nsub/y.wal\n"] {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("blob.lock"), bytes).unwrap();
        let err = BlobStorage::open(tmp.path().join("y.wal")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData, "{bytes:?}: {err}");
    }
}

#[test]
fn an_unreadable_record_stops_the_open() {
    for bytes in [
        &b"2\nx.wal\n"[..],
        b"1\n../x.wal\n",
        b"1\n\n",
        b"nonsense\n",
    ] {
        let (tmp, y) = store_without_its_wal();
        std::fs::write(tmp.path().join("blob.lock"), bytes).unwrap();
        let err = BlobStorage::open(&y).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData, "{bytes:?}: {err}");
    }
}
