//! `AliceDB::get` returns the newest value written for a key
//!
//! Writes sit in the memtable until it fills or `flush` runs. `get` used to
//! look only at flushed segments, so a value that had just been written read
//! back as `None`, and when several segments covered the key it read the
//! first one the index returned rather than the newest. `scan` already saw
//! both; these tests hold `get` to the same answer.
//!
//! Order of precedence: the memtable (latest write first), then segments
//! from the newest (highest segment id) down.

use alice_db::{AliceDB, StorageConfig};

fn db() -> AliceDB {
    AliceDB::in_memory(StorageConfig::default()).unwrap()
}

#[test]
fn put_then_get_without_flush() {
    let db = db();
    db.put(5, 3.25).unwrap();
    assert_eq!(db.get(5).unwrap(), Some(3.25));
    assert_eq!(db.get(6).unwrap(), None);
}

#[test]
fn latest_write_in_the_memtable_wins() {
    let db = db();
    db.put_batch(&[(5, 1.0), (7, 4.0), (5, 2.0)]).unwrap();
    assert_eq!(db.get(5).unwrap(), Some(2.0));
    assert_eq!(db.get(7).unwrap(), Some(4.0));
}

#[test]
fn memtable_shadows_a_flushed_segment() {
    let db = db();
    let old: Vec<(i64, f32)> = (0..10).map(|t| (t, 1.0)).collect();
    db.put_batch(&old).unwrap();
    db.flush().unwrap();
    assert_eq!(db.get(5).unwrap(), Some(1.0));
    db.put(5, 9.5).unwrap();
    assert_eq!(db.get(5).unwrap(), Some(9.5));
    // keys only in the segment still come from the segment
    assert_eq!(db.get(4).unwrap(), Some(1.0));
}

#[test]
fn newest_of_overlapping_segments_wins() {
    let db = db();
    for (round, v) in [1.0f32, 2.0, 3.0].into_iter().enumerate() {
        let data: Vec<(i64, f32)> = (0..10).map(|t| (t, v)).collect();
        db.put_batch(&data).unwrap();
        db.flush().unwrap();
        for t in 0..10 {
            assert_eq!(db.get(t).unwrap(), Some(v), "round {round} key {t}");
        }
    }
}

/// `scan` returns the latest write of a key still in the memtable, once
#[test]
fn scan_keeps_the_latest_write_in_the_memtable() {
    let db = db();
    db.put(5, 1.0).unwrap();
    db.put(7, 4.0).unwrap();
    db.put(5, 2.0).unwrap();
    db.put(5, 3.0).unwrap();
    assert_eq!(db.scan(5, 5).unwrap(), [(5, 3.0)]);
    assert_eq!(db.scan(0, 10).unwrap(), [(5, 3.0), (7, 4.0)]);
    // the same through a batch
    db.put_batch(&[(7, 5.0), (7, 6.0)]).unwrap();
    assert_eq!(db.scan(0, 10).unwrap(), [(5, 3.0), (7, 6.0)]);
}

/// `scan` returns the memtable's value for a key a flushed segment also
/// holds
#[test]
fn scan_prefers_the_memtable_over_a_flushed_segment() {
    let db = db();
    let old: Vec<(i64, f32)> = (0..10).map(|t| (t, 1.0)).collect();
    db.put_batch(&old).unwrap();
    db.flush().unwrap();
    db.put(5, 9.5).unwrap();
    assert_eq!(db.scan(5, 5).unwrap(), [(5, 9.5)]);
    let all = db.scan(0, 9).unwrap();
    assert_eq!(all.len(), 10);
    assert_eq!(all[5], (5, 9.5));
    // the other keys still come from the segment
    assert_eq!(all[4], (4, 1.0));
}
