//! Integration tests for the v0.2.0-alpha.4 `SSTable` flush pipeline.
//!
//! Coverage:
//!
//! - Manual `compact_blob_sstable` produces a readable `SSTable` and
//!   truncates the WAL.
//! - Auto-flush kicks in once the WAL crosses the configured threshold.
//! - Reopen after flush recovers the same state.
//! - A v0.2.0-alpha.3 database (WAL only, no `SSTable`) opens cleanly and
//!   upgrades transparently on first flush.
//! - Corrupted `SSTable` files surface as `io::Error` on open (fail loud,
//!   preserve the invariants around the tracker's dogfooding contract).

use alice_db::blob::BlobStorageConfig;
use alice_db::blob_wal::SyncPolicy;
use alice_db::AliceDB;
use std::io::Write;
use tempfile::TempDir;

fn wal_size(dir: &TempDir) -> u64 {
    std::fs::metadata(dir.path().join("blob.wal"))
        .map(|m| m.len())
        .unwrap_or(0)
}

fn sst_size(dir: &TempDir) -> u64 {
    std::fs::metadata(dir.path().join("blob.sst"))
        .map(|m| m.len())
        .unwrap_or(0)
}

// -------------------------------------------------------------------------
// Manual flush pipeline
// -------------------------------------------------------------------------

#[test]
fn compact_blob_sstable_materialises_the_sstable_and_truncates_wal() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    for i in 0..16 {
        let key = format!("k{i:02}");
        db.put_blob(key.as_bytes(), format!("v{i}").as_bytes())
            .unwrap();
    }
    assert!(sst_size(&tmp) == 0, "no `SSTable` before compact");
    assert!(wal_size(&tmp) > 0);

    db.compact_blob_sstable().unwrap();

    assert!(sst_size(&tmp) > 0, "`SSTable` now present");
    assert_eq!(
        wal_size(&tmp),
        0,
        "WAL must be truncated after successful compact"
    );

    // In-memory reads still work post-compact.
    for i in 0..16 {
        let key = format!("k{i:02}");
        let expected = format!("v{i}");
        assert_eq!(
            db.get_blob(key.as_bytes()).unwrap(),
            Some(expected.into_bytes())
        );
    }
}

#[test]
fn reopen_after_compact_recovers_full_state_from_sstable_alone() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        for i in 0..50 {
            let key = format!("k{i:02}");
            let value = format!("payload-{i}").repeat(10);
            db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
        }
        db.compact_blob_sstable().unwrap();
        // No further mutations: WAL stays empty, state lives in `SSTable`.
    }
    assert_eq!(wal_size(&tmp), 0);
    assert!(sst_size(&tmp) > 0);

    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.blob_len(), 50);
    for i in 0..50 {
        let key = format!("k{i:02}");
        let expected = format!("payload-{i}").repeat(10);
        assert_eq!(
            db.get_blob(key.as_bytes()).unwrap(),
            Some(expected.into_bytes())
        );
    }
}

#[test]
fn compact_then_wal_diff_reopens_with_sstable_plus_wal_replay() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"a", b"1").unwrap();
        db.put_blob(b"b", b"2").unwrap();
        db.compact_blob_sstable().unwrap();
        // Post-flush diff: settled state is in `blob.sst`, new writes go
        // to the WAL only.
        db.put_blob(b"c", b"3").unwrap();
        db.delete_blob(b"a").unwrap();
    }
    // `SSTable` non-empty, WAL non-empty.
    assert!(sst_size(&tmp) > 0);
    assert!(wal_size(&tmp) > 0);

    let db = AliceDB::open(tmp.path()).unwrap();
    // Deleted key is gone, overwrites are visible.
    assert_eq!(db.get_blob(b"a").unwrap(), None);
    assert_eq!(db.get_blob(b"b").unwrap(), Some(b"2".to_vec()));
    assert_eq!(db.get_blob(b"c").unwrap(), Some(b"3".to_vec()));
    assert_eq!(db.blob_len(), 2);
}

// -------------------------------------------------------------------------
// Auto-flush threshold
// -------------------------------------------------------------------------

#[test]
fn auto_flush_triggers_after_wal_crosses_threshold() {
    let tmp = TempDir::new().unwrap();
    // Absurdly small threshold so a handful of puts crosses it.
    let cfg = BlobStorageConfig {
        sync_policy: SyncPolicy::EveryWrite,
        wal_flush_threshold_bytes: 256,
        ..BlobStorageConfig::default()
    };
    let db = AliceDB::open_with_blob_config(tmp.path(), cfg).unwrap();

    // Each 100-byte-ish record easily crosses 256 bytes within a few puts.
    for i in 0..8 {
        let key = format!("k{i:02}");
        let value = vec![b'x'; 64];
        db.put_blob(key.as_bytes(), &value).unwrap();
    }

    // After all inserts, at least one auto-flush must have run. We
    // observe it via `sst_size > 0` and the WAL either empty or
    // holding only the tail since the most recent flush.
    assert!(sst_size(&tmp) > 0, "auto-flush must produce an `SSTable`");
    // WAL can be non-empty because the very last put comes after the
    // most recent auto-flush; the important guarantee is that it does
    // not grow unbounded.
    assert!(wal_size(&tmp) < 256 + 128);
}

// -------------------------------------------------------------------------
// Backward compat with v0.2.0-alpha.3 databases (WAL only)
// -------------------------------------------------------------------------

#[test]
fn opening_a_pre_v04_wal_only_database_upgrades_transparently() {
    let tmp = TempDir::new().unwrap();
    // Simulate an alpha-3 database: write via the current code but
    // never flush to `SSTable`, then close.
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"legacy-1", b"one").unwrap();
        db.put_blob(b"legacy-2", b"two").unwrap();
    }
    // Confirm no `SSTable` exists yet.
    assert!(sst_size(&tmp) == 0);
    assert!(wal_size(&tmp) > 0);

    // Reopen — `SSTable` path is created but empty; WAL replay fills the map.
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.get_blob(b"legacy-1").unwrap(), Some(b"one".to_vec()));
    assert_eq!(db.get_blob(b"legacy-2").unwrap(), Some(b"two".to_vec()));

    // First compact produces the `SSTable`.
    db.compact_blob_sstable().unwrap();
    assert!(sst_size(&tmp) > 0);
    assert_eq!(wal_size(&tmp), 0);
}

// -------------------------------------------------------------------------
// Corrupted `SSTable` surfaces on open (fail loud)
// -------------------------------------------------------------------------

#[test]
fn corrupted_sstable_footer_is_reported_on_open() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"k", b"v").unwrap();
        db.compact_blob_sstable().unwrap();
    }
    // Overwrite the last byte of the `SSTable` footer to break it.
    let sst_path = tmp.path().join("blob.sst");
    let mut bytes = std::fs::read(&sst_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    let mut f = std::fs::File::create(&sst_path).unwrap();
    f.write_all(&bytes).unwrap();
    drop(f);

    let Err(err) = AliceDB::open(tmp.path()) else {
        panic!("open must fail on corrupted `SSTable`")
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "got: {err}");
}
