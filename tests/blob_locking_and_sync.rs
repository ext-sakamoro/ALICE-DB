//! Tests for the v0.2.0-alpha.3 additions:
//!
//! - Exclusive file lock on the blob WAL.
//! - `SyncPolicy` variants (`EveryWrite`, `Batched`, `Manual`).
//! - Explicit `flush_blobs` API.
//!
//! Cross-process behaviour is verified with an in-process second open
//! (fs2 advisory locks are per-file-handle, so this exercises the same
//! code path a second process would take).

use alice_db::blob_wal::SyncPolicy;
use alice_db::AliceDB;
use std::io::ErrorKind;
use tempfile::TempDir;

// -------------------------------------------------------------------------
// Exclusive file lock
// -------------------------------------------------------------------------

#[test]
fn second_open_on_same_path_fails_while_first_is_alive() {
    let tmp = TempDir::new().unwrap();
    let first = AliceDB::open(tmp.path()).unwrap();
    let Err(err) = AliceDB::open(tmp.path()) else {
        panic!("opening the same data directory twice must fail while the first handle is alive")
    };
    // The concrete kind is WouldBlock on all supported platforms.
    assert_eq!(err.kind(), ErrorKind::WouldBlock, "got: {err}");
    // Keep the first handle alive until here so the drop order is
    // explicit in the test.
    drop(first);
}

#[test]
fn second_open_succeeds_after_first_is_dropped() {
    let tmp = TempDir::new().unwrap();
    {
        let first = AliceDB::open(tmp.path()).unwrap();
        first.put_blob(b"k", b"v").unwrap();
    }
    // Lock released on drop of `first`; reopen succeeds and sees prior state.
    let second = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(second.get_blob(b"k").unwrap(), Some(b"v".to_vec()));
}

// -------------------------------------------------------------------------
// SyncPolicy — Batched
// -------------------------------------------------------------------------

#[test]
fn batched_policy_persists_after_threshold_is_crossed() {
    let tmp = TempDir::new().unwrap();
    let policy = SyncPolicy::Batched { max_pending_ops: 4 };
    {
        let db = AliceDB::open_with_blob_sync_policy(tmp.path(), policy).unwrap();
        for i in 0..8 {
            let key = format!("k{i}");
            db.put_blob(key.as_bytes(), b"v").unwrap();
        }
        // Two full batches of 4 have been fsynced by the time we hit
        // this line, so a subsequent open must see all 8.
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.blob_len(), 8);
}

#[test]
fn batched_policy_uses_flush_to_persist_partial_batch() {
    let tmp = TempDir::new().unwrap();
    let policy = SyncPolicy::Batched {
        max_pending_ops: 32,
    };
    {
        let db = AliceDB::open_with_blob_sync_policy(tmp.path(), policy).unwrap();
        for i in 0..5 {
            let key = format!("k{i}");
            db.put_blob(key.as_bytes(), b"v").unwrap();
        }
        // 5 records is below the threshold; `flush_blobs` forces the fsync.
        db.flush_blobs().unwrap();
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.blob_len(), 5);
}

// -------------------------------------------------------------------------
// SyncPolicy — Manual
// -------------------------------------------------------------------------

#[test]
fn manual_policy_persists_only_via_explicit_flush() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open_with_blob_sync_policy(tmp.path(), SyncPolicy::Manual).unwrap();
        for i in 0..10 {
            let key = format!("k{i}");
            db.put_blob(key.as_bytes(), b"v").unwrap();
        }
        db.flush_blobs().unwrap();
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.blob_len(), 10);
}

// -------------------------------------------------------------------------
// Default policy is EveryWrite — flush is a no-op
// -------------------------------------------------------------------------

#[test]
fn every_write_policy_is_the_default_and_flush_is_a_no_op() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"k", b"v").unwrap();
    // A second flush right after the write must succeed cheaply.
    db.flush_blobs().unwrap();
    db.flush_blobs().unwrap();
    assert_eq!(db.get_blob(b"k").unwrap(), Some(b"v".to_vec()));
}
