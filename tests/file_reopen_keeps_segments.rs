//! Writing after reopening a file database must not overwrite the segments
//! already on disk (`seg_<id>.rkyv`): segment ids continue after the largest
//! id in the loaded index instead of starting again from 1.

#![cfg(feature = "fs")]

use alice_db::{AliceDB, FitConfig, StorageConfig};

fn open(dir: &std::path::Path) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        memtable_capacity: 64,
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        ..StorageConfig::default()
    })
    .unwrap()
}

fn value(t: i64) -> f32 {
    // not a low-order polynomial, so every point is stored rather than fitted away
    let r = i16::try_from((t * 7919) % 1009).unwrap();
    f32::from(r) * 0.25 - 100.0
}

fn write(db: &AliceDB, range: std::ops::Range<i64>) {
    for t in range {
        db.put(t, value(t)).unwrap();
    }
    db.flush().unwrap();
}

#[test]
fn reopening_twice_keeps_every_point() {
    let dir = tempfile::tempdir().unwrap();

    let db = open(dir.path());
    write(&db, 0..200);
    db.close().unwrap();

    let db = open(dir.path());
    write(&db, 200..300);
    db.close().unwrap();

    let db = open(dir.path());
    let got = db.scan(0, 300).unwrap();
    let expected: Vec<(i64, u32)> = (0..300).map(|t| (t, value(t).to_bits())).collect();
    let got: Vec<(i64, u32)> = got.into_iter().map(|(t, v)| (t, v.to_bits())).collect();
    assert_eq!(
        got.len(),
        300,
        "points lost across reopen: {} of 300",
        got.len()
    );
    assert_eq!(got, expected);
    db.close().unwrap();
}
