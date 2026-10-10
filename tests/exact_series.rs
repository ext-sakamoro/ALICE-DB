//! Exact series (`AliceDB::series`): every point reads back with its bits,
//! and a scan returns exactly the keys that were written
//!
//! The keys are the ones that broke the model-based store: sparse step
//! counters (0, 1, 2^20, 2^40, 2^53 − 1), both signs, and the ends of `i64`.
//! The shape follows the two callers the API is built for: a physics sink
//! writing one `f32` per step into three series of one database, and the
//! analytics bridge (`tests/analytics_metrics_exact.rs`).

use alice_db::series::{decode_key, encode_key, series_key, SERIES_KEY_PREFIX};
use alice_db::{AliceDB, SeriesValue, StorageConfig};

const SPARSE: [i64; 5] = [0, 1, 1 << 20, 1 << 40, (1 << 53) - 1];

/// Sparse keys of both signs and the ends of `i64`, deliberately unsorted
fn keys() -> Vec<i64> {
    let mut k: Vec<i64> = SPARSE.iter().rev().copied().collect();
    k.extend(SPARSE.iter().skip(1).map(|&x| -x));
    k.extend([i64::MIN, i64::MAX, -3, 7]);
    k
}

/// A value that differs per key in every bit position class
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bit mixing"
)]
fn f32_at(key: i64) -> f32 {
    f32::from_bits(
        (key as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17) as u32,
    )
}

#[allow(clippy::cast_sign_loss, reason = "bit mixing")]
fn f64_at(key: i64) -> f64 {
    f64::from_bits(
        (key as u64)
            .wrapping_mul(0xD1B5_4A32_D192_ED03)
            .rotate_left(29),
    )
}

fn sorted(mut k: Vec<i64>) -> Vec<i64> {
    k.sort_unstable();
    k
}

/// Writes three series the way a physics step sink does and checks every
/// read path; returns the number of compared values
fn exercise(db: &AliceDB) -> usize {
    let energy = db.series("physics/energy").unwrap();
    let bodies = db.series("physics/bodies").unwrap();
    let wide = db.series("physics/energy64").unwrap();
    for &k in &keys() {
        energy.put_f32(k, f32_at(k)).unwrap();
        bodies.put_f32(k, -f32_at(k)).unwrap();
        wide.put_f64(k, f64_at(k)).unwrap();
    }
    check(db)
}

fn check(db: &AliceDB) -> usize {
    let energy = db.series("physics/energy").unwrap();
    let bodies = db.series("physics/bodies").unwrap();
    let wide = db.series("physics/energy64").unwrap();
    let mut compared = 0;
    for &k in &keys() {
        assert_eq!(
            energy.get_f32(k).unwrap().map(f32::to_bits),
            Some(f32_at(k).to_bits()),
            "energy {k}"
        );
        assert_eq!(
            bodies.get_f32(k).unwrap().map(f32::to_bits),
            Some((-f32_at(k)).to_bits()),
            "bodies {k}"
        );
        assert_eq!(
            wide.get_f64(k).unwrap().map(f64::to_bits),
            Some(f64_at(k).to_bits()),
            "energy64 {k}"
        );
        compared += 3;
    }
    // keys never written
    for k in [2, 1 << 21, -2, i64::MIN + 1, 8] {
        assert_eq!(energy.get_f32(k).unwrap(), None, "{k}");
    }

    // full scan: exactly the written keys, ascending, exact values
    let all = energy.scan_f32(i64::MIN, i64::MAX).unwrap();
    assert_eq!(all.iter().map(|p| p.0).collect::<Vec<_>>(), sorted(keys()));
    for &(k, v) in &all {
        assert_eq!(v.to_bits(), f32_at(k).to_bits());
        compared += 1;
    }
    let all64 = wide.scan_f64(i64::MIN, i64::MAX).unwrap();
    assert_eq!(all64.len(), keys().len());
    for &(k, v) in &all64 {
        assert_eq!(v.to_bits(), f64_at(k).to_bits());
        compared += 1;
    }

    // the range that returned 1 of 5 rows from the model-based store
    let range = energy.scan_f32(0, 1 << 41).unwrap();
    assert_eq!(
        range.iter().map(|p| p.0).collect::<Vec<_>>(),
        [0, 1, 7, 1 << 20, 1 << 40]
    );
    // ranges ending on / starting at stored keys are inclusive
    let mid = energy.scan_f32(-(1 << 20), 1 << 20).unwrap();
    assert_eq!(
        mid.iter().map(|p| p.0).collect::<Vec<_>>(),
        [-(1 << 20), -3, -1, 0, 1, 7, 1 << 20]
    );
    assert!(energy.scan_f32(1, 0).unwrap().is_empty());
    assert!(energy.scan_f32(2, 6).unwrap().is_empty());
    compared
}

#[test]
fn sparse_keys_read_back_exact_in_memory() {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    assert_eq!(exercise(&db), 3 * 13 + 13 + 13);
    // a snapshot carries the series
    let restored = AliceDB::from_bytes(StorageConfig::default(), &db.to_bytes().unwrap()).unwrap();
    assert_eq!(check(&restored), 3 * 13 + 13 + 13);
}

#[cfg(feature = "fs")]
#[test]
fn sparse_keys_read_back_exact_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = AliceDB::open(dir.path()).unwrap();
        assert_eq!(exercise(&db), 65);
        db.close().unwrap();
    }
    let db = AliceDB::open(dir.path()).unwrap();
    assert_eq!(check(&db), 65);
    // and after the blob WAL is rolled into an SSTable
    db.compact_blob_sstable().unwrap();
    drop(db);
    let db = AliceDB::open(dir.path()).unwrap();
    assert_eq!(check(&db), 65);
}

/// NaN payloads, signed zeros, infinities and subnormals keep their bits
#[test]
fn special_values_keep_their_bits() {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    let s = db.series("special").unwrap();
    let f32s = [
        f32::from_bits(0x7FC0_1234),
        f32::from_bits(0xFF80_0001),
        -0.0,
        0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ];
    for (k, v) in (0i64..).zip(f32s) {
        s.put_f32(k, v).unwrap();
    }
    for (k, v) in (100i64..).zip([
        f64::from_bits(0x7FF8_0000_DEAD_BEEF),
        -0.0,
        f64::from_bits(1),
    ]) {
        s.put_f64(k, v).unwrap();
    }
    let got = s.scan(i64::MIN, i64::MAX).unwrap();
    assert_eq!(got.len(), 10);
    for (k, v) in got {
        match (k, v) {
            (0..=6, SeriesValue::F32(x)) => {
                assert_eq!(x.to_bits(), f32s[usize::try_from(k).unwrap()].to_bits());
            }
            (100, SeriesValue::F64(x)) => assert_eq!(x.to_bits(), 0x7FF8_0000_DEAD_BEEF),
            (101, SeriesValue::F64(x)) => assert_eq!(x.to_bits(), (-0.0f64).to_bits()),
            (102, SeriesValue::F64(x)) => assert_eq!(x.to_bits(), 1),
            other => panic!("unexpected {other:?}"),
        }
    }
    // reading with the other width is an error, not a conversion
    assert!(s.get_f64(0).is_err());
    assert!(s.get_f32(100).is_err());
    assert!(s.scan_f32(0, 200).is_err());
}

/// Series are separate namespaces, including names that are prefixes of
/// one another, and do not see the law store or raw blobs
#[test]
fn series_do_not_see_each_other() {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    db.series("a").unwrap().put_f32(5, 1.0).unwrap();
    db.series("ab").unwrap().put_f32(5, 2.0).unwrap();
    db.series("a\u{1}").unwrap().put_f32(5, 3.0).unwrap();
    db.put_blob(b"plain", b"blob").unwrap();
    for (name, v) in [("a", 1.0), ("ab", 2.0), ("a\u{1}", 3.0)] {
        let s = db.series(name).unwrap();
        assert_eq!(s.name(), name);
        assert_eq!(s.scan_f32(i64::MIN, i64::MAX).unwrap(), [(5, v)], "{name}");
    }
    assert_eq!(db.series("b").unwrap().get_f32(5).unwrap(), None);
    assert!(db.series("").is_err());
    assert!(db.series("x\0y").is_err());
}

#[test]
fn overwrite_and_delete() {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    let s = db.series("s").unwrap();
    s.put_f32(3, 1.0).unwrap();
    s.put_f32(3, 2.0).unwrap();
    assert_eq!(s.get_f32(3).unwrap(), Some(2.0));
    s.put_f64(3, 4.0).unwrap();
    assert_eq!(s.get(3).unwrap(), Some(SeriesValue::F64(4.0)));
    s.delete(3).unwrap();
    assert_eq!(s.get(3).unwrap(), None);
    assert!(s.scan(i64::MIN, i64::MAX).unwrap().is_empty());
}

/// The record key layout (prefix, name, NUL, key with the sign bit flipped,
/// big-endian), so byte order is signed key order
#[test]
fn key_layout_is_pinned() {
    assert_eq!(SERIES_KEY_PREFIX, b"\0alice-series\0");
    assert!(!SERIES_KEY_PREFIX.starts_with(alice_db::law_store::LAW_KEY_PREFIX));
    assert!(!alice_db::law_store::LAW_KEY_PREFIX.starts_with(SERIES_KEY_PREFIX));
    assert_eq!(encode_key(0), [0x80, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        encode_key(-1),
        [0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    );
    assert_eq!(encode_key(i64::MIN), [0; 8]);
    assert_eq!(
        encode_key(0x0102_0304_0506_0708),
        [0x81, 2, 3, 4, 5, 6, 7, 8]
    );
    let mut want = b"\0alice-series\0e\0".to_vec();
    want.extend_from_slice(&[0x80, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(series_key("e", 1).unwrap(), want);
    let ks = sorted(keys());
    for w in ks.windows(2) {
        assert!(encode_key(w[0]) < encode_key(w[1]), "{} < {}", w[0], w[1]);
    }
    for &k in &ks {
        assert_eq!(decode_key(encode_key(k)), k);
    }
}
