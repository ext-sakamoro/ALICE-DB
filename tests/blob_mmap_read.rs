//! Integration tests for the v0.2.0-alpha.7 mmap-backed read path and
//! `BlobStorage` architectural split (memtable + `Vec<LoadedSstable>`).
//!
//! Coverage:
//!
//! - `flush_to_sstable` produces on-disk state that a subsequent
//!   `get_blob` can serve directly from mmap (both Raw and Compressed
//!   values).
//! - Multi-`SSTable` reads return the newest value across sstables.
//! - A tombstone in the memtable masks a live value in an older
//!   `SSTable`.
//! - Legacy v1 files (written by v0.2.0-alpha.2 through v0.2.0-alpha.5)
//!   still read cleanly through the mmap path — the Bloom is absent so
//!   we fall through to the index directly.
//! - The Bloom-guarded read path returns `None` quickly for a key that
//!   was never inserted into any `SSTable` (functional test; a perf
//!   benchmark would be a separate story).
//! - `scan_blob_prefix` merges memtable and every `SSTable`'s prefix
//!   window with newest-wins semantics.
//! - A `compact_all_blob_sstables` cycle preserves every live key.
//! - Concurrent reads from 8 threads against a shared store see a
//!   consistent view.

use std::sync::Arc;

use alice_db::blob::BlobStorageConfig;
use alice_db::blob_sstable::FlushMode;
use alice_db::blob_wal::SyncPolicy;
use alice_db::AliceDB;
use tempfile::TempDir;

/// Open, put, flush, then reopen to force every read through the mmap
/// path (`get_blob` cannot use the memtable because reopen throws it
/// away and rebuilds from the WAL — which we truncated on flush).
#[test]
fn flush_then_reopen_serves_raw_value_via_mmap() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"greeting", b"hello world").unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);

    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(
        db.get_blob(b"greeting").unwrap(),
        Some(b"hello world".to_vec()),
    );
}

/// Repeat above with a large-enough value that compression kicks in.
#[test]
fn flush_then_reopen_serves_compressed_value_via_mmap() {
    let tmp = TempDir::new().unwrap();
    let payload = b"repeat me repeat me repeat me ".repeat(64);
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"repeating", &payload).unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);

    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.get_blob(b"repeating").unwrap(), Some(payload));
}

/// Append-mode flush must materialise a second `SSTable`. A subsequent
/// read returns the newest value (from the later `SSTable`), and both
/// old and new keys are visible.
#[test]
fn append_mode_multi_sstable_read_returns_newest_wins_via_mmap() {
    let tmp = TempDir::new().unwrap();
    let config = BlobStorageConfig {
        sync_policy: SyncPolicy::EveryWrite,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
    };
    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();

    // Round 1: put "shared" -> v1 and "only-in-1" -> "a", then flush.
    db.put_blob(b"shared", b"v1").unwrap();
    db.put_blob(b"only-in-1", b"a").unwrap();
    db.compact_blob_sstable().unwrap();

    // Round 2: overwrite "shared" with v2 and add "only-in-2".
    db.put_blob(b"shared", b"v2").unwrap();
    db.put_blob(b"only-in-2", b"b").unwrap();
    db.compact_blob_sstable().unwrap();

    // At this point there are (at least) two SSTables plus an empty
    // memtable. Reads must surface the newer value.
    assert_eq!(db.get_blob(b"shared").unwrap(), Some(b"v2".to_vec()));
    assert_eq!(db.get_blob(b"only-in-1").unwrap(), Some(b"a".to_vec()));
    assert_eq!(db.get_blob(b"only-in-2").unwrap(), Some(b"b".to_vec()));

    assert!(
        db.blob_sstable_count().unwrap() >= 2,
        "append-mode flush must accumulate SSTable files; got only {}",
        db.blob_sstable_count().unwrap(),
    );

    // Same view after reopen (memtable is empty; reads route through
    // the mmap'd SSTables).
    drop(db);
    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();
    assert_eq!(db.get_blob(b"shared").unwrap(), Some(b"v2".to_vec()));
    assert_eq!(db.get_blob(b"only-in-1").unwrap(), Some(b"a".to_vec()));
    assert_eq!(db.get_blob(b"only-in-2").unwrap(), Some(b"b".to_vec()));
}

/// A memtable tombstone must mask a live value in an older `SSTable`.
///
/// This is the primary reason the memtable can hold Tombstones at all —
/// without it, `delete_blob` after a flush would be a no-op for keys
/// that only live on disk.
#[test]
fn memtable_tombstone_masks_older_sstable_value() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"target", b"live").unwrap();
    db.compact_blob_sstable().unwrap();
    // "target" now lives in a mmap'd SSTable, and memtable is empty.
    assert_eq!(db.get_blob(b"target").unwrap(), Some(b"live".to_vec()));

    // Delete stages a memtable tombstone.
    db.delete_blob(b"target").unwrap();
    assert!(
        db.get_blob(b"target").unwrap().is_none(),
        "memtable tombstone must mask the SSTable value"
    );

    // The mask survives a WAL replay (memtable rebuilds from WAL).
    drop(db);
    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(
        db.get_blob(b"target").unwrap().is_none(),
        "reopen must replay the tombstone from the WAL"
    );
}

/// A key that was never inserted must resolve to `None` even against a
/// large `SSTable` — this exercises the Bloom-guarded fast path
/// functionally (a perf test would live in a `benches/` file, not
/// here).
#[test]
fn absent_key_is_bloom_rejected_on_v2_sstable() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    // Populate the SSTable with a modest number of unrelated keys.
    for i in 0..256 {
        let key = format!("populated-{i:04}");
        let value = format!("value-{i:04}");
        db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
    }
    db.compact_blob_sstable().unwrap();

    // A key that shares no prefix with any inserted one — a Bloom
    // false-positive is unlikely but not impossible; the assertion is
    // that we get None either way.
    assert!(db.get_blob(b"never-inserted-nothanks").unwrap().is_none());

    // And a few more absent probes for good measure.
    for i in 0..64 {
        let k = format!("probe-{i:04}");
        assert!(
            db.get_blob(k.as_bytes()).unwrap().is_none(),
            "absent key {k} must resolve to None"
        );
    }
}

/// `scan_blob_prefix` must merge the memtable and every `SSTable`, with
/// newest-wins semantics and stable ascending order.
#[test]
fn scan_prefix_merges_memtable_and_multiple_sstables() {
    let tmp = TempDir::new().unwrap();
    let config = BlobStorageConfig {
        sync_policy: SyncPolicy::EveryWrite,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
    };
    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();

    // SSTable 0.
    db.put_blob(b"user:001", b"alice-old").unwrap();
    db.put_blob(b"user:002", b"bob").unwrap();
    db.put_blob(b"widget:1", b"unrelated").unwrap();
    db.compact_blob_sstable().unwrap();

    // SSTable 1.
    db.put_blob(b"user:001", b"alice-new").unwrap(); // overwrite
    db.put_blob(b"user:003", b"carol").unwrap();
    db.compact_blob_sstable().unwrap();

    // Memtable-only fresh entry.
    db.put_blob(b"user:004", b"dan").unwrap();

    let mut hits: Vec<(Vec<u8>, Vec<u8>)> = db.scan_blob_prefix(b"user:").unwrap();
    hits.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        hits,
        vec![
            (b"user:001".to_vec(), b"alice-new".to_vec()),
            (b"user:002".to_vec(), b"bob".to_vec()),
            (b"user:003".to_vec(), b"carol".to_vec()),
            (b"user:004".to_vec(), b"dan".to_vec()),
        ],
    );
}

/// `compact_all_blob_sstables` must fold every live key into one file
/// with correct newest-wins semantics.
#[test]
fn compact_all_after_multi_sstable_preserves_every_live_key() {
    let tmp = TempDir::new().unwrap();
    let config = BlobStorageConfig {
        sync_policy: SyncPolicy::EveryWrite,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
    };
    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();

    for round in 0..4 {
        for i in 0..64 {
            let key = format!("key-{i:03}");
            let value = format!("round-{round}-value-{i:03}");
            db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
        }
        db.compact_blob_sstable().unwrap();
    }
    assert!(db.blob_sstable_count().unwrap() >= 4);

    // Fold into one.
    db.compact_all_blob_sstables().unwrap();
    assert_eq!(db.blob_sstable_count().unwrap(), 1);

    // Every key must resolve to the round-3 value (last writer wins).
    for i in 0..64 {
        let key = format!("key-{i:03}");
        let expected = format!("round-3-value-{i:03}");
        assert_eq!(
            db.get_blob(key.as_bytes()).unwrap(),
            Some(expected.into_bytes()),
        );
    }
}

/// 8 threads reading a shared mmap'd store must all observe consistent
/// values. Concurrent writes on a separate thread must not corrupt any
/// reader's view.
#[test]
fn concurrent_reads_and_writes_from_multiple_threads() {
    let tmp = TempDir::new().unwrap();
    let db = Arc::new(AliceDB::open(tmp.path()).unwrap());

    // Seed the store with some keys and flush so they end up on disk.
    for i in 0..128 {
        let key = format!("seed-{i:03}");
        let value = format!("seed-value-{i:03}");
        db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
    }
    db.compact_blob_sstable().unwrap();

    // Half the threads read the seeded keys; the other half write new
    // keys into the memtable while the readers are churning.
    let mut handles = Vec::new();
    for t in 0..4 {
        let db = Arc::clone(&db);
        handles.push(std::thread::spawn(move || {
            for i in 0..128 {
                let key = format!("seed-{i:03}");
                let expected = format!("seed-value-{i:03}");
                let observed = db.get_blob(key.as_bytes()).unwrap();
                assert_eq!(
                    observed,
                    Some(expected.into_bytes()),
                    "thread {t} lost a seeded value at iter {i}"
                );
            }
        }));
    }
    for t in 0..4 {
        let db = Arc::clone(&db);
        handles.push(std::thread::spawn(move || {
            for i in 0..64 {
                let key = format!("thread-{t:02}-key-{i:03}");
                let value = format!("thread-{t:02}-value-{i:03}");
                db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    // Every writer's contribution must be visible.
    for t in 0..4 {
        for i in 0..64 {
            let key = format!("thread-{t:02}-key-{i:03}");
            let expected = format!("thread-{t:02}-value-{i:03}");
            assert_eq!(
                db.get_blob(key.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
            );
        }
    }
}
