//! The blob store's merge of several `SSTable`s: the newest value or delete of
//! a key wins in `get`, `scan_blob_prefix`, after compaction and after
//! reopening.
//!
//! Fixed sequences on a temporary directory with `FlushMode::Append` and no
//! automatic flush or compaction, so every flush writes one more `SSTable`.
//! The expected result is written out per step (a model is not needed for
//! sequences this short). The fuzzer (`fuzz_index_lookup`) explores
//! sequences; this pins the semantics it checks.
#![cfg(feature = "fs")]

use alice_db::blob::BlobStorageConfig;
use alice_db::blob_sstable::FlushMode;
use alice_db::blob_wal::SyncPolicy;
use alice_db::AliceDB;
use std::path::Path;

fn config() -> BlobStorageConfig {
    BlobStorageConfig {
        sync_policy: SyncPolicy::Manual,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
    }
}

fn open(dir: &Path) -> AliceDB {
    AliceDB::open_with_blob_config(dir, config()).unwrap()
}

fn reopen(db: AliceDB, dir: &Path) -> AliceDB {
    db.flush_blobs().unwrap();
    drop(db);
    open(dir)
}

fn get(db: &AliceDB, k: &[u8]) -> Option<Vec<u8>> {
    db.get_blob(k).unwrap()
}

fn scan(db: &AliceDB) -> Vec<(Vec<u8>, Vec<u8>)> {
    db.scan_blob_prefix(b"").unwrap()
}

fn kv(k: &[u8], v: &[u8]) -> (Vec<u8>, Vec<u8>) {
    (k.to_vec(), v.to_vec())
}

#[test]
fn the_newest_value_of_a_key_wins_across_two_sstables() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.put_blob(b"k", b"v1").unwrap();
    db.compact_blob_sstable().unwrap();
    db.put_blob(b"k", b"v2").unwrap();
    db.compact_blob_sstable().unwrap();
    assert_eq!(db.blob_sstable_count().unwrap(), 2);
    assert_eq!(
        get(&db, b"k").as_deref(),
        Some(&b"v2"[..]),
        "get after two flushes"
    );
    assert_eq!(scan(&db), vec![kv(b"k", b"v2")], "scan after two flushes");

    db.compact_all_blob_sstables().unwrap();
    assert_eq!(db.blob_sstable_count().unwrap(), 1);
    assert_eq!(
        get(&db, b"k").as_deref(),
        Some(&b"v2"[..]),
        "get after compaction"
    );
    assert_eq!(scan(&db), vec![kv(b"k", b"v2")], "scan after compaction");

    let db = reopen(db, dir.path());
    assert_eq!(
        get(&db, b"k").as_deref(),
        Some(&b"v2"[..]),
        "get after reopen"
    );
    assert_eq!(scan(&db), vec![kv(b"k", b"v2")], "scan after reopen");
}

#[test]
fn a_delete_flushed_after_the_value_is_not_resurrected() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.put_blob(b"k", b"v1").unwrap();
    db.put_blob(b"other", b"x").unwrap();
    db.compact_blob_sstable().unwrap();
    db.delete_blob(b"k").unwrap();
    db.compact_blob_sstable().unwrap();
    assert_eq!(get(&db, b"k"), None, "get after the delete is flushed");
    assert_eq!(
        scan(&db),
        vec![kv(b"other", b"x")],
        "scan after the delete is flushed"
    );

    db.compact_all_blob_sstables().unwrap();
    assert_eq!(get(&db, b"k"), None, "get after compaction");
    assert_eq!(scan(&db), vec![kv(b"other", b"x")], "scan after compaction");

    let db = reopen(db, dir.path());
    assert_eq!(get(&db, b"k"), None, "get after reopen");
    assert_eq!(scan(&db), vec![kv(b"other", b"x")], "scan after reopen");
}

#[test]
fn the_memtable_masks_every_sstable_and_a_value_after_a_delete_returns() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    for v in [&b"v1"[..], b"v2", b"v3"] {
        db.put_blob(b"k", v).unwrap();
        db.compact_blob_sstable().unwrap();
    }
    db.delete_blob(b"k").unwrap();
    assert_eq!(
        get(&db, b"k"),
        None,
        "an unflushed delete masks three SSTables"
    );
    assert_eq!(scan(&db), vec![], "and the scan agrees");
    db.put_blob(b"k", b"v4").unwrap();
    assert_eq!(get(&db, b"k").as_deref(), Some(&b"v4"[..]));
    db.compact_blob_sstable().unwrap();
    db.compact_all_blob_sstables().unwrap();
    let db = reopen(db, dir.path());
    assert_eq!(get(&db, b"k").as_deref(), Some(&b"v4"[..]));
    assert_eq!(scan(&db), vec![kv(b"k", b"v4")]);
}

#[test]
fn interleaved_keys_resolve_independently_in_prefix_scans() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.put_blob(b"aa", b"1").unwrap();
    db.put_blob(b"ab", b"1").unwrap();
    db.put_blob(b"b", b"1").unwrap();
    db.compact_blob_sstable().unwrap();
    db.put_blob(b"ab", b"2").unwrap();
    db.delete_blob(b"aa").unwrap();
    db.compact_blob_sstable().unwrap();
    db.put_blob(b"ac", b"3").unwrap();
    let want = vec![kv(b"ab", b"2"), kv(b"ac", b"3")];
    assert_eq!(db.scan_blob_prefix(b"a").unwrap(), want);
    db.compact_all_blob_sstables().unwrap();
    assert_eq!(db.scan_blob_prefix(b"a").unwrap(), want);
    let db = reopen(db, dir.path());
    assert_eq!(db.scan_blob_prefix(b"a").unwrap(), want);
    assert_eq!(db.scan_blob_prefix(b"b").unwrap(), vec![kv(b"b", b"1")]);
}
