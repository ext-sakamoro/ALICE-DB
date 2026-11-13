//! A compressed blob value records its original length, and a record that lies about it is refused
//!
//! oracle (from the format, not read from the code): a value written now is stored as kind 0x03 (u64 LE original
//! length, then the zlib stream) and reads back exactly; a record whose stated length is shorter or longer than
//! what its stream decodes to is refused with `StoredLengthMismatch`, and one stating more than the store's limit
//! with `ValueTooLarge` before anything is decoded; records written by 0.6 (kind 0x01, no length; fixture
//! `tests/data/blob_v06`, written by alice-db 0.6.0) still read, and a flush rewrites them in the new form; a 0.6
//! record that expands past the limit is refused at the limit.

#![cfg(feature = "fs")]

use alice_db::blob::{
    BlobStorage, BlobStorageConfig, BlobValue, StoredLengthMismatch, ValueTooLarge,
};
use alice_db::blob_sstable::BlobSstable;
use alice_db::blob_wal::BlobWal;
use tempfile::TempDir;

fn config(max_value_bytes: usize) -> BlobStorageConfig {
    BlobStorageConfig::default().with_max_value_bytes(max_value_bytes)
}

/// The value an `SSTable` holds for `key`
fn sst_value(sst: &BlobSstable, key: &[u8]) -> Option<BlobValue> {
    sst.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone())
}

fn zlib(data: &[u8]) -> Vec<u8> {
    alice_core::compression::zlib_compress(data, 6).unwrap()
}

#[test]
fn a_value_written_now_records_its_original_length() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let value = vec![b'q'; 50_000];
    {
        let store = BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
        store.put(b"k", &value).unwrap();
        store.flush_to_sstable().unwrap();
        store.put(b"w", &value).unwrap();
    }
    // the WAL record of the put after the flush: kind 0x03 (offset 9), its value starts with the length
    let w = std::fs::read(&wal).unwrap();
    assert_eq!(w[0], 0x01, "a put record");
    assert_eq!(w[9], 0x03, "a sized compressed value in the WAL");
    assert_eq!(u64::from_le_bytes(w[11..19].try_into().unwrap()), 50_000);
    let sst = BlobSstable::open(tmp.path().join("blob.sst"))
        .unwrap()
        .unwrap();
    match sst_value(&sst, b"k") {
        Some(BlobValue::CompressedSized { original_len, .. }) => assert_eq!(original_len, 50_000),
        other => panic!("{other:?}"),
    }
    let store = BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
    assert_eq!(store.get(b"k").unwrap().as_deref(), Some(&value[..]));
    assert_eq!(store.get(b"w").unwrap().as_deref(), Some(&value[..]));
}

#[test]
fn a_record_lying_about_its_length_is_refused() {
    let data = vec![b'z'; 10_000];
    for (stated, mismatch) in [(9_999_u64, true), (10_001, true), (10_000, false)] {
        let tmp = TempDir::new().unwrap();
        let wal = tmp.path().join("blob.wal");
        {
            BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
        }
        BlobWal::open(&wal)
            .unwrap()
            .append_put(
                b"k",
                &BlobValue::CompressedSized {
                    original_len: stated,
                    bytes: zlib(&data),
                },
            )
            .unwrap();
        let store = BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
        let r = store.get(b"k");
        if mismatch {
            let e = r.unwrap_err();
            let m = e
                .get_ref()
                .and_then(|i| i.downcast_ref::<StoredLengthMismatch>());
            assert!(m.is_some(), "stated {stated}: {e}");
        } else {
            assert_eq!(r.unwrap().as_deref(), Some(&data[..]));
        }
    }
    // a stated length past the limit is refused before decoding
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    {
        BlobStorage::open_with_config(&wal, config(1024)).unwrap();
    }
    BlobWal::open(&wal)
        .unwrap()
        .append_put(
            b"k",
            &BlobValue::CompressedSized {
                original_len: 1 << 40,
                bytes: zlib(&data),
            },
        )
        .unwrap();
    let store = BlobStorage::open_with_config(&wal, config(1024)).unwrap();
    let e = store.get(b"k").unwrap_err();
    let v = e
        .get_ref()
        .and_then(|i| i.downcast_ref::<ValueTooLarge>())
        .copied();
    assert_eq!(
        v.map(|v| (v.len as u64, v.limit)),
        Some((1 << 40, 1024)),
        "{e}"
    );
}

fn copy_fixture(to: &std::path::Path) {
    let from = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/blob_v06");
    for f in ["blob.sst", "blob.wal", "blob.lock"] {
        std::fs::copy(from.join(f), to.join(f)).unwrap();
    }
}

#[test]
fn records_written_by_0_6_read_and_a_flush_rewrites_them() {
    let tmp = TempDir::new().unwrap();
    copy_fixture(tmp.path());
    let pattern: Vec<u8> = (0..65536_u32).map(|i| (i % 251) as u8).collect();
    let expect: [(&[u8], Vec<u8>); 4] = [
        (b"sst-compressed", pattern.clone()),
        (b"sst-raw", b"short raw value".to_vec()),
        (b"wal-compressed", vec![7u8; 40000]),
        (b"wal-raw", b"another short one".to_vec()),
    ];
    // the fixture holds the earlier, length-free form
    let sst = BlobSstable::open(tmp.path().join("blob.sst"))
        .unwrap()
        .unwrap();
    assert!(matches!(
        sst_value(&sst, b"sst-compressed"),
        Some(BlobValue::Compressed(_))
    ));
    drop(sst);
    let wal = tmp.path().join("blob.wal");
    {
        let store = BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
        for (k, v) in &expect {
            assert_eq!(
                store.get(k).unwrap().as_deref(),
                Some(&v[..]),
                "{}",
                String::from_utf8_lossy(k)
            );
        }
        store.flush_to_sstable().unwrap();
        store.compact_all_sstables().unwrap();
    }
    // after the flush every compressed value carries its length
    let sst = BlobSstable::open(tmp.path().join("blob.sst"))
        .unwrap()
        .unwrap();
    for k in [&b"sst-compressed"[..], b"wal-compressed"] {
        assert!(
            matches!(sst_value(&sst, k), Some(BlobValue::CompressedSized { .. })),
            "{}: {:?}",
            String::from_utf8_lossy(k),
            sst_value(&sst, k)
        );
    }
    let store = BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
    for (k, v) in &expect {
        assert_eq!(store.get(k).unwrap().as_deref(), Some(&v[..]));
    }
}

#[test]
fn a_0_6_record_past_the_limit_is_refused_at_the_limit() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    {
        BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
    }
    BlobWal::open(&wal)
        .unwrap()
        .append_put(b"bomb", &BlobValue::Compressed(zlib(&vec![0u8; 8 << 20])))
        .unwrap();
    let store = BlobStorage::open_with_config(&wal, config(1 << 20)).unwrap();
    let e = store.get(b"bomb").unwrap_err();
    assert!(e.to_string().contains("limit"), "{e}");
}
