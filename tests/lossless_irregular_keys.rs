//! Lossless mode through the plain `put` / `get` / `scan` with irregular keys
//!
//! A physics sink records one value per simulation step with
//! `FitConfig { lossless: true }`. With sparse steps (0, 1, 2^20, 2^40,
//! 2^53 − 1) the model-based store read back the same value for most of
//! them, and a range over half of them returned one row: a segment keeps no
//! per-point keys and reads its points back on the even grid between its
//! first and last key, so a lossless residual was applied to points that
//! were never written.
//!
//! Lossless mode now cuts a segment wherever the spacing changes (each
//! segment holds `start + k·step` for one step) and refuses a key at or
//! below one already written with a typed [`LosslessKeyOrderError`]. These
//! tests require: increasing keys with any gaps read back exactly through
//! `get` and `scan` (before and after a flush, after a reopen), `scan`
//! returns only written keys, an out-of-order or repeated key is refused
//! without writing anything, and evenly spaced data still makes one
//! segment.

use alice_db::{AliceDB, FitConfig, LosslessKeyOrderError, StorageConfig};

const SPARSE: [i64; 5] = [0, 1, 1 << 20, 1 << 40, (1 << 53) - 1];

fn lossless() -> StorageConfig {
    StorageConfig {
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        ..StorageConfig::default()
    }
}

/// Distinct, non-model-like values (the repro wrote 0.1 + step-dependent
/// energies)
#[allow(clippy::cast_precision_loss, reason = "small integers")]
fn value(i: usize) -> f32 {
    0.1 + ((i * 7919) % 613) as f32 * 0.25
}

fn data(keys: &[i64]) -> Vec<(i64, f32)> {
    keys.iter()
        .enumerate()
        .map(|(i, &k)| (k, value(i)))
        .collect()
}

fn assert_exact(db: &AliceDB, keys: &[i64], what: &str) {
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(
            db.get(k).unwrap().map(f32::to_bits),
            Some(value(i).to_bits()),
            "{what}: get({k})"
        );
    }
    let all = db.scan(i64::MIN, i64::MAX).unwrap();
    assert_eq!(all, data(keys), "{what}: full scan");
}

/// The repro: sparse steps, one `put` each, read before and after a flush
#[test]
fn sparse_steps_read_back_exact() {
    let db = AliceDB::in_memory(lossless()).unwrap();
    for (k, v) in data(&SPARSE) {
        db.put(k, v).unwrap();
    }
    assert_exact(&db, &SPARSE, "before flush");
    db.flush().unwrap();
    assert_exact(&db, &SPARSE, "after flush");
    // the range that returned 1 of 5 rows
    assert_eq!(db.scan(0, 1 << 41).unwrap(), data(&SPARSE[..4]));
    // keys between the written ones are not there
    for k in [2, 3, 1 << 21, (1 << 40) + 1] {
        assert_eq!(db.get(k).unwrap(), None, "get({k})");
    }
}

/// Both signs, the ends of `i64`, and a batch that crosses several spacing
/// changes and memtable fills
#[test]
fn both_signs_and_batches_read_back_exact() {
    let keys: Vec<i64> = vec![
        i64::MIN,
        -((1 << 53) - 1),
        -(1 << 40),
        -(1 << 20),
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        10,
        17,
        24,
        1 << 40,
        (1 << 62) + 5,
        i64::MAX,
    ];
    for capacity in [2, 3, 5, 1000] {
        let db = AliceDB::in_memory(StorageConfig {
            memtable_capacity: capacity,
            ..lossless()
        })
        .unwrap();
        db.put_batch(&data(&keys)).unwrap();
        assert_exact(&db, &keys, &format!("capacity {capacity}, buffered"));
        db.flush().unwrap();
        assert_exact(&db, &keys, &format!("capacity {capacity}, flushed"));
    }
}

#[cfg(feature = "fs")]
#[test]
fn sparse_steps_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let config = || StorageConfig {
        data_dir: dir.path().to_path_buf(),
        ..lossless()
    };
    {
        let db = AliceDB::with_config(config()).unwrap();
        for (k, v) in data(&SPARSE[..3]) {
            db.put(k, v).unwrap();
        }
        db.close().unwrap();
    }
    let db = AliceDB::with_config(config()).unwrap();
    assert_exact(&db, &SPARSE[..3], "reopened");
    // continues after the stored keys, refuses below them
    let (k3, v3) = data(&SPARSE)[3];
    db.put(k3, v3).unwrap();
    assert!(db.put(5, 1.0).is_err());
    db.flush().unwrap();
    assert_exact(&db, &SPARSE[..4], "reopened + written");
}

fn order_error(e: &std::io::Error) -> LosslessKeyOrderError {
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    *e.get_ref()
        .and_then(|inner| inner.downcast_ref::<LosslessKeyOrderError>())
        .expect("a LosslessKeyOrderError")
}

/// A key at or below one already written is refused, typed, and nothing of
/// the refused call is stored
#[test]
fn out_of_order_and_repeated_keys_are_refused() {
    let db = AliceDB::in_memory(lossless()).unwrap();
    db.put_batch(&[(10, 1.0), (20, 2.0)]).unwrap();
    let e = db.put(15, 9.0).unwrap_err();
    assert_eq!(order_error(&e), LosslessKeyOrderError { key: 15, last: 20 });
    let e = db.put(20, 9.0).unwrap_err();
    assert_eq!(order_error(&e), LosslessKeyOrderError { key: 20, last: 20 });
    // a batch is checked as a whole: its valid prefix is not written either
    let e = db.put_batch(&[(30, 3.0), (25, 9.0)]).unwrap_err();
    assert_eq!(order_error(&e), LosslessKeyOrderError { key: 25, last: 30 });
    db.flush().unwrap();
    assert_eq!(db.scan(i64::MIN, i64::MAX).unwrap(), [(10, 1.0), (20, 2.0)]);
    // after a flush the order still holds
    assert!(db.put(5, 9.0).is_err());
    db.put(30, 3.0).unwrap();
    assert_eq!(db.get(30).unwrap(), Some(3.0));
}

/// Lossy mode (the default) keeps accepting any key order
#[test]
fn lossy_mode_is_unchanged() {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    db.put_batch(&[(10, 1.0), (5, 2.0), (5, 3.0)]).unwrap();
    assert_eq!(db.get(5).unwrap(), Some(3.0));
}

/// Evenly spaced data still makes a single segment per memtable fill
#[test]
fn even_spacing_is_not_cut() {
    let db = AliceDB::in_memory(lossless()).unwrap();
    let keys: Vec<i64> = (0..900).map(|i| 1000 + i * 3).collect();
    db.put_batch(&data(&keys)).unwrap();
    db.flush().unwrap();
    assert_eq!(db.stats().total_segments, 1);
    assert_exact(&db, &keys, "even");
}
