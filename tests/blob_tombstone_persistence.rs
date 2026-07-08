//! Integration tests for tombstone persistence in v0.2.0-alpha.8
//! (`FORMAT_VERSION_V3`).
//!
//! Coverage:
//!
//! - `FlushMode::Append` + `delete` + flush + reopen keeps the key
//!   deleted. This closes the v0.2.0-alpha.5 through -alpha.7
//!   limitation where the `SSTable` format could not carry tombstones,
//!   so a delete-then-flush cycle would resurrect the key on reopen.
//! - `compact_all_blob_sstables` after a delete under `Append` folds
//!   the tombstone out — the resulting file has no leftover tombstone
//!   entry and the key stays gone.
//! - Round-trip: put A → flush → delete A → flush → reopen → put A' →
//!   flush → reopen. The final read must see A', not the resurrected
//!   original.
//! - `FlushMode::Overwrite` behaviour is unchanged (regression proof).

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
        // Never auto-compact so we can observe the tombstone masking
        // across multiple SSTables in the tests below.
        max_sstables_before_compaction: usize::MAX,
    }
}

#[test]
fn append_mode_delete_survives_process_restart() {
    let tmp = TempDir::new().unwrap();
    let config = append_config();

    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();
    db.put_blob(b"target", b"payload").unwrap();
    db.compact_blob_sstable().unwrap(); // Append flush #1: writes `target`.
    assert_eq!(db.get_blob(b"target").unwrap(), Some(b"payload".to_vec()));

    db.delete_blob(b"target").unwrap();
    db.compact_blob_sstable().unwrap(); // Append flush #2: writes Tombstone(target).
    assert!(
        db.get_blob(b"target").unwrap().is_none(),
        "delete is immediately visible before restart"
    );

    // Simulate a process restart. The WAL is empty (both flushes
    // truncated it) so the memtable rebuilds to empty. Reads must
    // route through the sstable path and see the tombstone in the
    // newer SSTable, masking the value in the older one.
    drop(db);
    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();
    assert!(
        db.get_blob(b"target").unwrap().is_none(),
        "delete must persist across reopen (v3 SSTables carry tombstones)"
    );
}

#[test]
fn compact_all_after_append_delete_drops_the_tombstone() {
    let tmp = TempDir::new().unwrap();
    let config = append_config();
    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();

    db.put_blob(b"key-A", b"value-A").unwrap();
    db.put_blob(b"key-B", b"value-B").unwrap();
    db.compact_blob_sstable().unwrap();

    db.delete_blob(b"key-A").unwrap();
    db.compact_blob_sstable().unwrap();
    assert!(db.get_blob(b"key-A").unwrap().is_none());
    assert!(db.blob_sstable_count().unwrap() >= 2);

    // The compaction folds Tombstone(key-A) over Raw(key-A) and drops
    // both from the merged file. `key-B` survives unchanged.
    db.compact_all_blob_sstables().unwrap();
    assert_eq!(db.blob_sstable_count().unwrap(), 1);
    assert!(db.get_blob(b"key-A").unwrap().is_none());
    assert_eq!(db.get_blob(b"key-B").unwrap(), Some(b"value-B".to_vec()));

    // Reopen: the single compacted SSTable is v3 but carries no
    // tombstone entry (it was dropped at compaction). `key-A` stays
    // gone; `key-B` stays live.
    drop(db);
    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();
    assert!(db.get_blob(b"key-A").unwrap().is_none());
    assert_eq!(db.get_blob(b"key-B").unwrap(), Some(b"value-B".to_vec()));
}

#[test]
fn overwrite_delete_reopen_cycle_still_works() {
    // Regression proof: `FlushMode::Overwrite` (the default) has
    // always worked for delete + reopen; make sure the new code path
    // does not break it.
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"only", b"original").unwrap();
    db.compact_blob_sstable().unwrap();
    db.delete_blob(b"only").unwrap();
    db.compact_blob_sstable().unwrap();
    assert!(db.get_blob(b"only").unwrap().is_none());
    drop(db);

    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(db.get_blob(b"only").unwrap().is_none());
    assert_eq!(db.blob_sstable_count().unwrap(), 1);
}

#[test]
fn resurrect_and_overwrite_cycle_under_append_mode() {
    // put A → flush → delete A → flush → reopen → put A' → flush →
    // reopen. Final read must see A' (not the resurrected original).
    let tmp = TempDir::new().unwrap();
    let config = append_config();

    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();
    db.put_blob(b"resurrected", b"original").unwrap();
    db.compact_blob_sstable().unwrap();
    db.delete_blob(b"resurrected").unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);

    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();
    assert!(db.get_blob(b"resurrected").unwrap().is_none());
    db.put_blob(b"resurrected", b"replayed").unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);

    let db = AliceDB::open_with_blob_config(tmp.path(), config).unwrap();
    assert_eq!(
        db.get_blob(b"resurrected").unwrap(),
        Some(b"replayed".to_vec()),
        "the newest put must win over both the older value and its tombstone"
    );
}
