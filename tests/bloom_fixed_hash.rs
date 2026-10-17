//! The Bloom filter's positions are this crate's own hash, and a stored filter of any other
//! derivation is not probed (ALICE-Law G8 / G10)
//!
//! `tests/fixtures/blob_sst_v3/blob.sst` was written by the format v3 writer (0.6.0 before
//! this change, positions from `std::hash::DefaultHasher`): 300 live keys `key-0000` ..
//! `key-0299` and one tombstone `gone`. Its filter's positions came from a hash the standard
//! library does not fix between releases, so a reader must not trust them: both readers
//! rebuild the filter from the file's keys and say so ([`BloomSource::Rebuilt`]).

#![cfg(feature = "fs")]

use alice_db::blob::BlobValue;
use alice_db::blob_sstable::{BlobSstable, BloomRebuild, BloomSource, LoadedSstable};
use alice_db::bloom::BLOOM_HASH_ID;
use std::path::{Path, PathBuf};

fn fixture() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = tmp.path().join("blob.sst");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/blob_sst_v3/blob.sst"),
        &path,
    )
    .expect("copy fixture");
    (tmp, path)
}

fn fixture_keys() -> Vec<Vec<u8>> {
    let mut keys: Vec<Vec<u8>> = (0..300u32)
        .map(|i| format!("key-{i:04}").into_bytes())
        .collect();
    keys.extend([b"gone".to_vec()]);
    keys
}

#[test]
fn the_fixture_is_a_v3_file() {
    let (_tmp, path) = fixture();
    let bytes = std::fs::read(&path).expect("read");
    assert_eq!(&bytes[0..8], b"ALICEBBS");
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        3
    );
}

/// Every key the file holds passes the filter the readers use (0 false negatives), and the
/// filter is the rebuilt one, not the stored `DefaultHasher` positions
#[test]
fn a_v3_filter_is_rebuilt_and_has_no_false_negative() {
    let (_tmp, path) = fixture();
    let loaded = BlobSstable::open(&path).expect("open").expect("present");
    let mapped = LoadedSstable::open_mmap(&path)
        .expect("open")
        .expect("present");
    for source in [loaded.bloom_source(), mapped.bloom_source()] {
        assert_eq!(
            source,
            BloomSource::Rebuilt(BloomRebuild::NoHashId { version: 3 })
        );
    }
    let mut checked = 0;
    for key in fixture_keys() {
        assert!(loaded.bloom().expect("a filter").contains(&key), "{key:?}");
        assert!(mapped.bloom().expect("a filter").contains(&key), "{key:?}");
        checked += 1;
    }
    assert_eq!(checked, 301);
    // the stored bits are not what the readers use: the rebuilt filter differs from them
    assert_ne!(
        loaded.bloom().expect("a filter").as_bits(),
        stored_bits(&std::fs::read(&path).expect("read")).as_slice(),
        "the rebuilt filter equals the stored DefaultHasher filter"
    );
    // every live key reads back through the mmap reader
    for i in 0..300u32 {
        let key = format!("key-{i:04}").into_bytes();
        assert_eq!(
            mapped.get(&key),
            Some(BlobValue::Raw(format!("v{i}").into_bytes())),
            "{i}"
        );
    }
}

/// The bits of a v2 / v3 Bloom section (`num_bits` u64, `num_hashes` u32, `bits_len` u64, bits)
fn stored_bits(file: &[u8]) -> Vec<u8> {
    let n = file.len();
    let bloom_size =
        usize::try_from(u64::from_le_bytes(file[n - 16..n - 8].try_into().unwrap())).expect("size");
    let start = n - 24 - bloom_size;
    let section = &file[start..start + bloom_size];
    let bits_len =
        usize::try_from(u64::from_le_bytes(section[12..20].try_into().unwrap())).expect("len");
    section[20..20 + bits_len].to_vec()
}

#[test]
fn a_new_file_is_v4_and_its_filter_is_used_as_stored() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = tmp.path().join("blob.sst");
    let values: Vec<(Vec<u8>, BlobValue)> = fixture_keys()
        .into_iter()
        .map(|k| (k, BlobValue::Raw(b"x".to_vec())))
        .collect();
    let mut sorted = values;
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    BlobSstable::write_from_iter(&path, sorted.iter().map(|(k, v)| (k.as_slice(), v)))
        .expect("write");
    let bytes = std::fs::read(&path).expect("read");
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        4
    );
    let loaded = BlobSstable::open(&path).expect("open").expect("present");
    let mapped = LoadedSstable::open_mmap(&path)
        .expect("open")
        .expect("present");
    assert_eq!(loaded.bloom_source(), BloomSource::Stored);
    assert_eq!(mapped.bloom_source(), BloomSource::Stored);
    let mut checked = 0;
    for (key, _) in &sorted {
        assert!(loaded.bloom().expect("a filter").contains(key));
        checked += 1;
    }
    assert_eq!(checked, 301);
}

/// A v4 filter under another identifier is rebuilt, and the identifier is reported
#[test]
fn a_foreign_hash_id_is_rebuilt_and_named() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = tmp.path().join("blob.sst");
    let v = BlobValue::Raw(b"x".to_vec());
    BlobSstable::write_from_iter(&path, [(b"a".as_slice(), &v), (b"b".as_slice(), &v)])
        .expect("write");
    let mut bytes = std::fs::read(&path).expect("read");
    let n = bytes.len();
    let bloom_size =
        usize::try_from(u64::from_le_bytes(bytes[n - 16..n - 8].try_into().unwrap())).unwrap();
    let start = n - 24 - bloom_size;
    // the identifier sits after num_bits (8) and num_hashes (4)
    assert_eq!(&bytes[start + 12..start + 44], &BLOOM_HASH_ID);
    let foreign = [0x5a_u8; 32];
    bytes[start + 12..start + 44].copy_from_slice(&foreign);
    let crc_at = start + bloom_size - 4;
    let crc = crc32fast::hash(&bytes[start..crc_at]);
    bytes[crc_at..crc_at + 4].copy_from_slice(&crc.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write");
    let loaded = BlobSstable::open(&path).expect("open").expect("present");
    let mapped = LoadedSstable::open_mmap(&path)
        .expect("open")
        .expect("present");
    for source in [loaded.bloom_source(), mapped.bloom_source()] {
        assert_eq!(
            source,
            BloomSource::Rebuilt(BloomRebuild::ForeignHashId(foreign))
        );
    }
    assert!(loaded.bloom().expect("a filter").contains(b"a"));
    assert!(mapped.bloom().expect("a filter").contains(b"b"));
}
