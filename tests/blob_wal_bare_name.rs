//! A WAL given by a bare name finds its files in the current directory. The test
//! changes the current directory, so it is alone in its binary.
#![cfg(feature = "fs")]

use alice_db::blob::{BlobStorage, BlobStorageConfig};
use alice_db::blob_sstable::FlushMode;
use tempfile::TempDir;

#[test]
fn a_bare_name_keeps_its_flushed_data() {
    let start = std::env::current_dir().unwrap();
    for flush_mode in [FlushMode::Overwrite, FlushMode::Append] {
        for wal in ["blob.wal", "x.wal"] {
            let tmp = TempDir::new().unwrap();
            std::env::set_current_dir(tmp.path()).unwrap();
            let config = BlobStorageConfig {
                flush_mode,
                ..BlobStorageConfig::default()
            };
            {
                let store = BlobStorage::open_with_config(wal, config).unwrap();
                store.put(b"k", b"v").unwrap();
                store.flush_to_sstable().unwrap();
                store.compact_all_sstables().unwrap();
            }
            let store = BlobStorage::open_with_config(wal, config).unwrap();
            assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()), "{wal}");
            drop(store);
            let view = BlobStorage::open_existing_read_only(wal).unwrap();
            assert_eq!(
                view.get(b"k").unwrap(),
                Some(b"v".to_vec()),
                "read-only {wal}"
            );
            let other = BlobStorage::open(if wal == "x.wal" { "y.wal" } else { "x.wal" });
            assert!(other.is_err(), "{wal}: the record was missed");
            std::env::set_current_dir(&start).unwrap();
        }
    }

    // A store flushes long after it was opened.
    let tmp = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    std::env::set_current_dir(tmp.path()).unwrap();
    let store = BlobStorage::open("moved.wal").unwrap();
    store.put(b"k", b"v").unwrap();
    std::env::set_current_dir(elsewhere.path()).unwrap();
    store.flush_to_sstable().unwrap();
    assert!(tmp.path().join("blob.sst").exists());
    assert!(!elsewhere.path().join("blob.sst").exists());
    drop(store);
    std::env::set_current_dir(&start).unwrap();
}
