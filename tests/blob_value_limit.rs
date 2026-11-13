//! A blob value is at most the store's `max_value_bytes` (default 64 MiB), on the way in and on the way out
//!
//! oracle (from the contract, not read from the code): a value of exactly the limit is stored and read back; one
//! byte more is refused by `put` and by `write_batch` with the typed `ValueTooLarge`, nothing stored; a compressed
//! value already on disk that expands past the limit of the store reading it is refused (the record does not hold
//! its original length, so the store's limit is the ceiling), and so is a scan reaching it.

#![cfg(feature = "fs")]

use alice_db::blob::{
    BlobStorage, BlobStorageConfig, ScanOrder, ValueTooLarge, WriteBatch, DEFAULT_MAX_VALUE_BYTES,
};
use tempfile::TempDir;

fn config(max_value_bytes: usize) -> BlobStorageConfig {
    BlobStorageConfig::default().with_max_value_bytes(max_value_bytes)
}

fn too_large(e: &std::io::Error) -> Option<(usize, usize)> {
    e.get_ref()
        .and_then(|i| i.downcast_ref::<ValueTooLarge>())
        .map(|v| (v.len, v.limit))
}

#[test]
fn the_default_limit_is_64_mib() {
    assert_eq!(DEFAULT_MAX_VALUE_BYTES, 64 << 20);
    assert_eq!(BlobStorageConfig::default().max_value_bytes, 64 << 20);
}

#[test]
fn a_value_past_the_limit_is_refused_on_the_way_in() {
    let tmp = TempDir::new().unwrap();
    let store = BlobStorage::open_with_config(tmp.path().join("blob.wal"), config(1024)).unwrap();
    // compressible and incompressible values alike
    let exact = vec![b'a'; 1024];
    store.put(b"exact", &exact).unwrap();
    assert_eq!(store.get(b"exact").unwrap().as_deref(), Some(&exact[..]));
    let e = store.put(b"over", &vec![b'a'; 1025]).unwrap_err();
    assert_eq!(too_large(&e), Some((1025, 1024)), "{e}");
    let noise: Vec<u8> = (0..1025_u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    assert_eq!(
        too_large(&store.put(b"noise", &noise).unwrap_err()),
        Some((1025, 1024))
    );
    let mut batch = WriteBatch::new();
    batch.put(b"ok", b"small");
    batch.put(b"over", &vec![b'a'; 1025]);
    assert_eq!(
        too_large(&store.write_batch(&batch).unwrap_err()),
        Some((1025, 1024))
    );
    // the whole batch is refused: nothing of it is stored
    assert_eq!(store.get(b"ok").unwrap(), None);
    assert_eq!(store.get(b"over").unwrap(), None);
}

#[test]
fn a_stored_value_past_the_limit_of_the_store_reading_it_is_refused() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let big = vec![0u8; 64 * 1024]; // compresses to a few hundred bytes
    {
        let store = BlobStorage::open_with_config(&wal, config(1 << 20)).unwrap();
        store.put(b"big", &big).unwrap();
        store.put(b"small", b"tiny").unwrap();
    }
    // read back from the WAL (memtable) and, after a flush, from an SSTable
    for flush in [false, true] {
        {
            let store = BlobStorage::open_with_config(&wal, config(1 << 20)).unwrap();
            if flush {
                store.flush_to_sstable().unwrap();
            }
        }
        let store = BlobStorage::open_with_config(&wal, config(1024)).unwrap();
        let e = store.get(b"big").unwrap_err();
        assert!(e.to_string().contains("limit"), "flush {flush}: {e}");
        assert_eq!(store.get(b"small").unwrap().as_deref(), Some(&b"tiny"[..]));
        assert!(
            store.scan_prefix(b"").is_err(),
            "flush {flush}: a scan reaching the value"
        );
        assert!(
            store
                .scan_range(
                    std::ops::Bound::Unbounded,
                    std::ops::Bound::Unbounded,
                    ScanOrder::Ascending,
                    10
                )
                .is_err(),
            "flush {flush}: a range scan reaching the value"
        );
        assert!(
            store.snapshot_to(&mut Vec::new()).is_err(),
            "flush {flush}: a snapshot reaching the value"
        );
    }
}
