//! Integration tests for the k-way merge `scan_blob_prefix` shipped in
//! v0.2.0-alpha.8.
//!
//! Coverage:
//!
//! - Multi-`SSTable` prefix scan under `FlushMode::Append` returns
//!   sorted, deduped `(key, value)` pairs with newest-wins semantics.
//! - A tombstone in the memtable (or in a newer `SSTable`) masks any
//!   older value under the same key.
//! - The set of returned pairs matches a hand-computed reference,
//!   proving the heap-based merge respects source priority ordering.
//! - Empty prefix returns nothing when no key matches.
//! - Empty memtable with populated `SSTables` still produces correct
//!   output (regression proof for a common code path).

use std::collections::BTreeMap;

use alice_db::blob::BlobStorageConfig;
use alice_db::blob_sstable::FlushMode;
use alice_db::blob_wal::SyncPolicy;
use alice_db::AliceDB;
use tempfile::TempDir;

fn append_config() -> BlobStorageConfig {
    BlobStorageConfig {
        sync_policy: SyncPolicy::EveryWrite,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
    }
}

#[test]
fn multi_sstable_prefix_scan_returns_sorted_deduped_pairs() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open_with_blob_config(tmp.path(), append_config()).unwrap();

    // SSTable 0: initial seed.
    for i in 0..8 {
        let key = format!("item:{i:02}");
        let value = format!("sst0-{i:02}");
        db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
    }
    db.compact_blob_sstable().unwrap();

    // SSTable 1: overwrite half the items, add new ones.
    for i in 0..4 {
        let key = format!("item:{i:02}");
        let value = format!("sst1-{i:02}");
        db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
    }
    for i in 8..12 {
        let key = format!("item:{i:02}");
        let value = format!("sst1-{i:02}");
        db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
    }
    db.compact_blob_sstable().unwrap();

    // Memtable: one more fresh entry and one overwrite.
    db.put_blob(b"item:00", b"memt-00").unwrap();
    db.put_blob(b"item:12", b"memt-12").unwrap();

    let observed: Vec<(Vec<u8>, Vec<u8>)> = db.scan_blob_prefix(b"item:").unwrap();
    // Result must be sorted by key ascending.
    for pair in observed.windows(2) {
        assert!(pair[0].0 < pair[1].0, "scan_prefix output must be sorted");
    }

    // Compute the expected view by hand.
    let mut expected: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    for i in 0..8 {
        expected.insert(
            format!("item:{i:02}").into_bytes(),
            format!("sst0-{i:02}").into_bytes(),
        );
    }
    for i in 0..4 {
        expected.insert(
            format!("item:{i:02}").into_bytes(),
            format!("sst1-{i:02}").into_bytes(),
        );
    }
    for i in 8..12 {
        expected.insert(
            format!("item:{i:02}").into_bytes(),
            format!("sst1-{i:02}").into_bytes(),
        );
    }
    expected.insert(b"item:00".to_vec(), b"memt-00".to_vec());
    expected.insert(b"item:12".to_vec(), b"memt-12".to_vec());
    let expected_vec: Vec<(Vec<u8>, Vec<u8>)> = expected.into_iter().collect();
    assert_eq!(observed, expected_vec);
}

#[test]
fn tombstone_in_memtable_masks_older_sstable_value_in_scan() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open_with_blob_config(tmp.path(), append_config()).unwrap();

    db.put_blob(b"row:1", b"a").unwrap();
    db.put_blob(b"row:2", b"b").unwrap();
    db.put_blob(b"row:3", b"c").unwrap();
    db.compact_blob_sstable().unwrap();

    // Memtable tombstone must remove `row:2` from the scan output.
    db.delete_blob(b"row:2").unwrap();

    let observed: Vec<(Vec<u8>, Vec<u8>)> = db.scan_blob_prefix(b"row:").unwrap();
    assert_eq!(
        observed,
        vec![
            (b"row:1".to_vec(), b"a".to_vec()),
            (b"row:3".to_vec(), b"c".to_vec()),
        ]
    );
}

#[test]
fn tombstone_in_newer_sstable_masks_older_sstable_value_in_scan() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open_with_blob_config(tmp.path(), append_config()).unwrap();

    db.put_blob(b"row:1", b"a").unwrap();
    db.put_blob(b"row:2", b"b").unwrap();
    db.compact_blob_sstable().unwrap();

    db.delete_blob(b"row:1").unwrap();
    db.compact_blob_sstable().unwrap();

    let observed: Vec<(Vec<u8>, Vec<u8>)> = db.scan_blob_prefix(b"row:").unwrap();
    assert_eq!(
        observed,
        vec![(b"row:2".to_vec(), b"b".to_vec())],
        "the newer SSTable's tombstone must mask the older value in scan_prefix",
    );
}

#[test]
fn empty_result_when_no_key_matches_prefix() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"cat", b"meow").unwrap();
    db.put_blob(b"dog", b"bark").unwrap();
    db.compact_blob_sstable().unwrap();

    let out = db.scan_blob_prefix(b"fish").unwrap();
    assert!(out.is_empty());
}

#[test]
fn scan_prefix_after_reopen_uses_sstable_only_path() {
    // With an empty memtable (post-reopen) every entry surfaces via
    // the mmap'd SSTables. Same content should be visible.
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open_with_blob_config(tmp.path(), append_config()).unwrap();
    for i in 0..16 {
        let key = format!("prefix:{i:03}");
        let value = format!("value-{i:03}");
        db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
    }
    db.compact_blob_sstable().unwrap();
    // Second flush with overwrites to force two sstables.
    for i in 0..8 {
        let key = format!("prefix:{i:03}");
        let value = format!("newer-{i:03}");
        db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
    }
    db.compact_blob_sstable().unwrap();
    drop(db);

    let db = AliceDB::open_with_blob_config(tmp.path(), append_config()).unwrap();
    let mut observed: Vec<(Vec<u8>, Vec<u8>)> = db.scan_blob_prefix(b"prefix:").unwrap();
    observed.sort_by(|a, b| a.0.cmp(&b.0));

    let mut expected: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for i in 0..16 {
        let key = format!("prefix:{i:03}").into_bytes();
        let value = if i < 8 {
            format!("newer-{i:03}").into_bytes()
        } else {
            format!("value-{i:03}").into_bytes()
        };
        expected.push((key, value));
    }
    assert_eq!(observed, expected);
}
