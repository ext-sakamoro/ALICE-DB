//! Tests for the v0.2.0-alpha.6 `SSTable` format v2 with Bloom filter.
//!
//! Coverage:
//!
//! - New writes produce format v2 with a Bloom filter attached.
//! - The Bloom filter rejects clearly-absent keys and accepts every
//!   inserted key (no false negatives).
//! - Legacy format v1 files (written by v0.2.0-alpha.2 through
//!   v0.2.0-alpha.5) still open successfully — their Bloom is reported
//!   as `None` and callers know to fall back to a records probe.
//! - `AliceDB::open` on a directory with legacy v1 `blob.sst` upgrades
//!   silently and the first flush produces a v2 file.

use alice_db::blob::BlobValue;
use alice_db::blob_sstable::BlobSstable;
use alice_db::AliceDB;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

fn v2_written_sstable(dir: &TempDir) -> BlobSstable {
    let path = dir.path().join("blob.sst");
    let raw = BlobValue::Raw(b"one".to_vec());
    let comp = BlobValue::Compressed(vec![0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
    BlobSstable::write_from_iter(
        &path,
        [(b"alpha".as_slice(), &raw), (b"beta".as_slice(), &comp)],
    )
    .unwrap();
    BlobSstable::open(&path).unwrap().unwrap()
}

#[test]
fn v2_files_carry_a_bloom_filter_that_accepts_every_written_key() {
    let tmp = TempDir::new().unwrap();
    let sst = v2_written_sstable(&tmp);
    let bloom = sst.bloom().expect("v2 file must carry a Bloom filter");

    // No false negatives.
    for k in [b"alpha".as_slice(), b"beta".as_slice()] {
        assert!(
            bloom.contains(k),
            "Bloom must not reject the inserted key {k:?}"
        );
    }
}

#[test]
fn v2_bloom_rejects_clearly_absent_keys() {
    let tmp = TempDir::new().unwrap();
    let sst = v2_written_sstable(&tmp);
    let bloom = sst.bloom().unwrap();

    // We probe 128 unrelated keys. With the 1% target rate and 2 inserted
    // items, false positives will be rare; we only need at least one
    // rejection to prove the filter is doing real work.
    let mut rejected = 0;
    for i in 0..128 {
        let k = format!("never-inserted-{i:03}");
        if !bloom.contains(k.as_bytes()) {
            rejected += 1;
        }
    }
    assert!(
        rejected > 100,
        "Bloom must reject most unrelated keys; got only {rejected} / 128"
    );
}

/// Format v1 file: exactly the layout v0.2.0-alpha.5 produced (no Bloom
/// section, 16-byte footer). Writing this by hand exercises the
/// `open()` back-compat path.
fn write_v1_sstable(path: &std::path::Path) {
    // Header:  magic (8) + version u32 LE = 1 (4) + num_records u64 LE = 1 (8) + reserved u32 = 0 (4)
    // Records: key_len(4) + value_len(4) + value_kind(1) + key(N) + value(M) + crc32c(4)
    // Footer:  records_size u64 LE (8) + magic (8)
    let key = b"legacy-key";
    let value = b"legacy-val";
    let mut record: Vec<u8> = Vec::new();
    record.extend_from_slice(&u32::to_le_bytes(u32::try_from(key.len()).unwrap()));
    record.extend_from_slice(&u32::to_le_bytes(u32::try_from(value.len()).unwrap()));
    record.push(0x00); // Raw
    record.extend_from_slice(key);
    record.extend_from_slice(value);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&record);
    let crc = hasher.finalize();
    record.extend_from_slice(&crc.to_le_bytes());

    let mut file = File::create(path).unwrap();
    file.write_all(b"ALICEBBS").unwrap();
    file.write_all(&1_u32.to_le_bytes()).unwrap(); // version 1
    file.write_all(&1_u64.to_le_bytes()).unwrap(); // num_records
    file.write_all(&0_u32.to_le_bytes()).unwrap(); // reserved
    file.write_all(&record).unwrap();
    let records_size = u64::try_from(record.len()).unwrap();
    file.write_all(&records_size.to_le_bytes()).unwrap();
    file.write_all(b"ALICEEND").unwrap();
    file.sync_all().unwrap();
}

#[test]
fn legacy_v1_files_still_open_and_carry_no_bloom() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.sst");
    write_v1_sstable(&path);

    let sst = BlobSstable::open(&path).unwrap().unwrap();
    assert!(
        sst.bloom().is_none(),
        "v1 files predate the Bloom section — reader must report None"
    );
    assert_eq!(sst.len(), 1);
    let recs: Vec<_> = sst.iter().collect();
    assert_eq!(recs[0].0, b"legacy-key");
    match recs[0].1 {
        BlobValue::Raw(bytes) => assert_eq!(bytes, b"legacy-val"),
        other => panic!("expected Raw, got {other:?}"),
    }
}

#[test]
fn alice_db_open_upgrades_legacy_v1_blob_sst_transparently() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.sst");
    write_v1_sstable(&path);

    // Open through the AliceDB surface. The legacy v1 file is read via
    // the same back-compat path used inside BlobStorage.
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(
        db.get_blob(b"legacy-key").unwrap(),
        Some(b"legacy-val".to_vec())
    );

    // First compact produces a v2 file, replacing the v1.
    db.compact_blob_sstable().unwrap();
    let upgraded = BlobSstable::open(tmp.path().join("blob.sst"))
        .unwrap()
        .unwrap();
    assert!(
        upgraded.bloom().is_some(),
        "the first flush must upgrade the on-disk file to v2"
    );
    assert_eq!(
        db.get_blob(b"legacy-key").unwrap(),
        Some(b"legacy-val".to_vec()),
        "upgrade path must preserve every key"
    );
}
