//! Sparse simulation steps written by a physics sink, read back one point
//! at a time and as a range
//!
//! Keys 0, 1, 1000, 1 000 003, 2^40, 2^53 − 1 with values `0.1 + i·1.7`.
//! Before 0.3.0 the plain path with `FitConfig { lossless: true }` returned
//! 0.1 for every key up to 2^40, 8.6 for 2^53 − 1, and one row for the range
//! 0 ..= 2^41 (five written). Both the plain lossless path (which now cuts a
//! segment wherever the spacing changes) and the exact series must return
//! every written point with its bits, in memory and on the file backend.

use alice_db::{AliceDB, FitConfig, StorageConfig};

const STEPS: [i64; 6] = [0, 1, 1000, 1_000_003, 1 << 40, (1 << 53) - 1];

#[allow(clippy::cast_precision_loss, reason = "indices below 8")]
fn value(i: usize) -> f32 {
    0.1 + i as f32 * 1.7
}

fn lossless(config: StorageConfig) -> StorageConfig {
    StorageConfig {
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        ..config
    }
}

/// Every point through `read(s, s)`, and the range 0 ..= 2^41
fn check(read: impl Fn(i64, i64) -> Vec<(i64, f32)>, what: &str) {
    let mut compared = 0;
    for (i, &s) in STEPS.iter().enumerate() {
        let got = read(s, s);
        assert_eq!(got.len(), 1, "{what}: point {s}: {got:?}");
        assert_eq!(got[0].0, s, "{what}: point {s}");
        assert_eq!(
            got[0].1.to_bits(),
            value(i).to_bits(),
            "{what}: point {s}: {} vs {}",
            got[0].1,
            value(i)
        );
        compared += 1;
    }
    let range = read(0, 1 << 41);
    assert_eq!(
        range,
        STEPS[..5]
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, value(i)))
            .collect::<Vec<_>>(),
        "{what}: range 0..=2^41"
    );
    assert_eq!(compared, 6);
}

fn plain(db: &AliceDB, what: &str) {
    for (i, &s) in STEPS.iter().enumerate() {
        db.put(s, value(i)).unwrap();
    }
    db.flush().unwrap();
    check(|a, b| db.scan(a, b).unwrap(), what);
    for (i, &s) in STEPS.iter().enumerate() {
        assert_eq!(
            db.get(s).unwrap().map(f32::to_bits),
            Some(value(i).to_bits())
        );
    }
}

fn series(db: &AliceDB, what: &str) {
    let energy = db.series("energy").unwrap();
    for (i, &s) in STEPS.iter().enumerate() {
        energy.put_f32(s, value(i)).unwrap();
    }
    check(|a, b| energy.scan_f32(a, b).unwrap(), what);
    for (i, &s) in STEPS.iter().enumerate() {
        assert_eq!(
            energy.get_f32(s).unwrap().map(f32::to_bits),
            Some(value(i).to_bits())
        );
    }
}

#[test]
fn plain_lossless_in_memory() {
    plain(
        &AliceDB::in_memory(lossless(StorageConfig::default())).unwrap(),
        "plain, in memory",
    );
}

#[test]
fn series_in_memory() {
    series(
        &AliceDB::in_memory(StorageConfig::default()).unwrap(),
        "series, in memory",
    );
}

#[cfg(feature = "fs")]
#[test]
fn plain_lossless_file_backend_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let config = || {
        lossless(StorageConfig {
            data_dir: dir.path().to_path_buf(),
            ..StorageConfig::default()
        })
    };
    {
        let db = AliceDB::with_config(config()).unwrap();
        plain(&db, "plain, file");
        db.close().unwrap();
    }
    let db = AliceDB::with_config(config()).unwrap();
    check(|a, b| db.scan(a, b).unwrap(), "plain, file, reopened");
}

#[cfg(feature = "fs")]
#[test]
fn series_file_backend_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = AliceDB::open(dir.path()).unwrap();
        series(&db, "series, file");
        db.close().unwrap();
    }
    let db = AliceDB::open(dir.path()).unwrap();
    let energy = db.series("energy").unwrap();
    check(
        |a, b| energy.scan_f32(a, b).unwrap(),
        "series, file, reopened",
    );
}
