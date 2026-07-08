//! Persistence and fault-tolerance tests for the v0.2.0-alpha.2 blob WAL.
//!
//! Every scenario opens `AliceDB` at a `TempDir`, mutates the blob store,
//! drops the handle (which closes the WAL cleanly), then reopens the same
//! directory and verifies the reconstructed state matches expectations.
//!
//! Fault-tolerance tests also poke the WAL file directly to simulate
//! crash / corruption conditions.

use std::io::{Seek, SeekFrom, Write};

use alice_db::AliceDB;
use tempfile::TempDir;

// -------------------------------------------------------------------------
// Persistence — happy paths
// -------------------------------------------------------------------------

#[test]
fn put_then_reopen_recovers_short_value() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"stub-a", b"todo!()").unwrap();
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.get_blob(b"stub-a").unwrap(), Some(b"todo!()".to_vec()));
    assert_eq!(db.blob_len(), 1);
}

#[test]
fn put_then_reopen_recovers_compressed_value() {
    let tmp = TempDir::new().unwrap();
    // Long enough to trigger the compression path.
    let value = b"todo!(\"port from alice-physics::gauss_seidel; XPBD diagonal\")\n".repeat(6);
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"stub-a", &value).unwrap();
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(
        db.get_blob(b"stub-a").unwrap(),
        Some(value.clone()),
        "compressed values must round-trip byte-for-byte after reopen"
    );
}

#[test]
fn delete_then_reopen_stays_deleted() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"stub-a", b"todo!()").unwrap();
        db.put_blob(b"stub-b", b"unimplemented!()").unwrap();
        db.delete_blob(b"stub-a").unwrap();
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.get_blob(b"stub-a").unwrap(), None);
    assert_eq!(
        db.get_blob(b"stub-b").unwrap(),
        Some(b"unimplemented!()".to_vec())
    );
    assert_eq!(db.blob_len(), 1);
}

#[test]
fn overwrite_then_reopen_returns_latest_value() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"key", b"first").unwrap();
        db.put_blob(b"key", b"second").unwrap();
        db.put_blob(b"key", b"third").unwrap();
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.get_blob(b"key").unwrap(), Some(b"third".to_vec()));
    assert_eq!(db.blob_len(), 1);
}

#[test]
fn mixed_put_delete_across_1000_ops_survives_reopen() {
    let tmp = TempDir::new().unwrap();
    // Populate: 500 keys, each written twice, half then deleted.
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        for i in 0..500 {
            let key = format!("k{i:03}");
            db.put_blob(key.as_bytes(), format!("v{i}-first").as_bytes())
                .unwrap();
            db.put_blob(key.as_bytes(), format!("v{i}-second").as_bytes())
                .unwrap();
            if i % 2 == 0 {
                db.delete_blob(key.as_bytes()).unwrap();
            }
        }
    }

    let db = AliceDB::open(tmp.path()).unwrap();
    // Odd-numbered keys survive with their -second value.
    assert_eq!(db.blob_len(), 250);
    for i in (1..500).step_by(2) {
        let key = format!("k{i:03}");
        let expected = format!("v{i}-second");
        assert_eq!(
            db.get_blob(key.as_bytes()).unwrap(),
            Some(expected.into_bytes()),
            "key {key} lost its latest value after reopen"
        );
    }
    // Even-numbered keys are gone.
    for i in (0..500).step_by(2) {
        let key = format!("k{i:03}");
        assert_eq!(db.get_blob(key.as_bytes()).unwrap(), None);
    }
}

#[test]
fn opening_an_empty_directory_yields_an_empty_blob_store() {
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(db.blob_is_empty());
    assert_eq!(db.blob_len(), 0);
    assert_eq!(db.get_blob(b"never").unwrap(), None);
}

// -------------------------------------------------------------------------
// Fault tolerance
// -------------------------------------------------------------------------

/// A partial write (crash mid-record) must not stop earlier fully-framed
/// records from being recovered.
#[test]
fn truncated_wal_tail_drops_partial_record_but_keeps_prior_records() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"k1", b"v1").unwrap();
        db.put_blob(b"k2", b"v2").unwrap();
    }

    // Now truncate the WAL file mid-second-record. We simulate a
    // torn write by lopping a handful of bytes off the end.
    let wal_path = tmp.path().join("blob.wal");
    let full = std::fs::metadata(&wal_path).unwrap().len();
    // Both records are HEADER (10) + 2 (key) + 2 (value) + CRC (4) = 18 bytes each.
    // 36 bytes total; we lop 7 off so the second record loses its CRC + one payload byte.
    assert!(full >= 30);
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&wal_path)
        .unwrap();
    f.set_len(full - 7).unwrap();
    drop(f);

    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.blob_len(), 1);
    assert_eq!(db.get_blob(b"k1").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(db.get_blob(b"k2").unwrap(), None);
}

/// A record whose CRC32C does not match its payload must be dropped, and
/// nothing that comes after it is trusted either.
#[test]
fn corrupted_record_stops_replay_at_that_record() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"good", b"ok").unwrap();
        db.put_blob(b"bad", b"no").unwrap();
        db.put_blob(b"later", b"lost").unwrap();
    }
    // Flip a byte inside the middle record's payload. The exact offset
    // depends on layout: HEADER(10) + key("good"=4) + value("ok"=2) +
    // CRC(4) = 20 bytes for the first record. The middle record's
    // payload begins at offset 20 + HEADER(10) = 30.
    let wal_path = tmp.path().join("blob.wal");
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .read(true)
        .open(&wal_path)
        .unwrap();
    f.seek(SeekFrom::Start(31)).unwrap(); // middle record, one byte into its key
    f.write_all(&[0xff]).unwrap();
    drop(f);

    let db = AliceDB::open(tmp.path()).unwrap();
    // Only the first record survives.
    assert_eq!(db.blob_len(), 1);
    assert_eq!(db.get_blob(b"good").unwrap(), Some(b"ok".to_vec()));
    assert_eq!(db.get_blob(b"bad").unwrap(), None);
    assert_eq!(db.get_blob(b"later").unwrap(), None);
}

/// Opening a data directory that has never seen a blob write must not
/// fail even if only the time-series files exist.
#[test]
fn opening_directory_without_prior_blob_wal_succeeds() {
    let tmp = TempDir::new().unwrap();
    // First open touches the WAL file (creating it empty) — we verify
    // that the store still behaves as an empty one.
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.blob_len(), 0);
    // The WAL file must exist now.
    assert!(tmp.path().join("blob.wal").exists());
}
