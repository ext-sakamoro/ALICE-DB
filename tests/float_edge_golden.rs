//! Bit-exact golden for signed zero and cancellation in the public float API
//!
//! `tests/determinism_golden.rs` hashes whole scenarios, so it says that some
//! bit changed but not which one. This file pins single answers of
//! `AliceDB::aggregate`, `AliceDB::downsample`, `AliceDB::get` and
//! `AliceDB::scan` at the IEEE 754 edges where an implementation change is
//! easiest to make without noticing:
//!
//! - **signed zero** — a sum seeded with `+0.0` turns `-0.0 + -0.0` into
//!   `+0.0`; a model evaluated as `a + t·(b − a)` keeps or loses the sign of
//!   zero depending on the operands
//! - **cancellation** — the order in which large values cancel decides whether
//!   a small term survives; the spread of near-equal values is the difference
//!   of two close numbers
//!
//! Every expected value is written as `to_bits()` and every case was checked
//! against the exact result before it was pinned. The aggregates read the
//! memtable (no flush) so that they see the written values themselves; the
//! model cases flush first so that `get` / `scan` evaluate a stored segment.

use alice_db::{Aggregation, AliceDB, FitConfig, StorageConfig};

const NEG_ZERO_F64: u64 = 0x8000_0000_0000_0000;
const POS_ZERO_F64: u64 = 0x0000_0000_0000_0000;
const NEG_ZERO_F32: u32 = 0x8000_0000;
const POS_ZERO_F32: u32 = 0x0000_0000;

fn db() -> AliceDB {
    AliceDB::in_memory(StorageConfig::default()).unwrap()
}

fn lossless_db() -> AliceDB {
    AliceDB::in_memory(StorageConfig {
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        ..StorageConfig::default()
    })
    .unwrap()
}

fn get_bits(db: &AliceDB, key: i64) -> u32 {
    db.get(key).unwrap().expect("key was written").to_bits()
}

/// Near-equal values: 2^24, 2^24 + 2, 2^24 + 4 (consecutive even f32 values,
/// one ulp apart). Exact population variance 8/3, standard deviation √(8/3)
const NEAR_EQUAL: [(i64, f32); 3] = [(0, 16_777_216.0), (1, 16_777_218.0), (2, 16_777_220.0)];
/// 8/3 rounded to nearest f64
const NEAR_EQUAL_VARIANCE: u64 = 0x4005_5555_5555_5555;
/// √(8/3) rounded to nearest f64
const NEAR_EQUAL_STDDEV: u64 = 0x3FFA_20BD_700C_2C3E;

#[test]
fn sum_of_two_negative_zeros_is_negative_zero() {
    // IEEE 754: (-0) + (-0) = -0. A sum seeded with +0.0 would give +0
    let db = db();
    db.put_batch(&[(0, -0.0), (1, -0.0)]).unwrap();
    let sum = db.aggregate(0, 1, Aggregation::Sum).unwrap();
    assert_eq!(sum.to_bits(), NEG_ZERO_F64);
}

#[test]
fn sum_of_opposite_zeros_is_positive_zero() {
    // IEEE 754 (round to nearest): (-0) + (+0) = +0
    let db = db();
    db.put_batch(&[(0, -0.0), (1, 0.0)]).unwrap();
    let sum = db.aggregate(0, 1, Aggregation::Sum).unwrap();
    assert_eq!(sum.to_bits(), POS_ZERO_F64);
}

#[test]
fn avg_of_two_negative_zeros_is_negative_zero() {
    // (-0 + -0) / 2 = -0 / 2 = -0
    let db = db();
    db.put_batch(&[(0, -0.0), (1, -0.0)]).unwrap();
    let avg = db.aggregate(0, 1, Aggregation::Avg).unwrap();
    assert_eq!(avg.to_bits(), NEG_ZERO_F64);
}

#[test]
fn grouped_sum_and_avg_of_negative_zeros_are_negative_zero() {
    // the streaming group aggregator starts from the first value, so a
    // bucket of -0 values sums to -0 and averages to -0
    let db = db();
    db.put_batch(&[(0, -0.0), (1, -0.0)]).unwrap();
    let sum = db.downsample(0, 1, 10, Aggregation::Sum).unwrap();
    assert_eq!(sum.len(), 1);
    assert_eq!(sum[0].0, 0);
    assert_eq!(sum[0].1.to_bits(), NEG_ZERO_F64);
    let avg = db.downsample(0, 1, 10, Aggregation::Avg).unwrap();
    assert_eq!(avg.len(), 1);
    assert_eq!(avg[0].1.to_bits(), NEG_ZERO_F64);
}

#[test]
fn sum_cancels_the_large_pair_before_adding_one() {
    // 1e30 + (-1e30) = +0 exactly, then + 1 = 1: summation runs in key order,
    // and in this order the cancellation loses nothing (exact answer 1)
    let db = db();
    db.put_batch(&[(0, 1e30), (1, -1e30), (2, 1.0)]).unwrap();
    let sum = db.aggregate(0, 2, Aggregation::Sum).unwrap();
    assert_eq!(sum.to_bits(), 1.0f64.to_bits());
}

#[test]
fn variance_and_stddev_of_near_equal_values() {
    // two-pass variance (mean first, then squared deviations): deviations
    // -2, 0, 2 are exact, so the result is 8/3 correctly rounded
    let db = db();
    db.put_batch(&NEAR_EQUAL).unwrap();
    let var = db.aggregate(0, 2, Aggregation::Variance).unwrap();
    assert_eq!(var.to_bits(), NEAR_EQUAL_VARIANCE);
    let std = db.aggregate(0, 2, Aggregation::StdDev).unwrap();
    assert_eq!(std.to_bits(), NEAR_EQUAL_STDDEV);
}

#[test]
fn grouped_welford_variance_and_stddev_of_near_equal_values() {
    // the grouped path uses Welford's online update; on the same input it
    // must produce the same correctly rounded 8/3 and √(8/3)
    let db = db();
    db.put_batch(&NEAR_EQUAL).unwrap();
    let var = db.downsample(0, 2, 10, Aggregation::Variance).unwrap();
    assert_eq!(var.len(), 1);
    assert_eq!(var[0].1.to_bits(), NEAR_EQUAL_VARIANCE);
    let std = db.downsample(0, 2, 10, Aggregation::StdDev).unwrap();
    assert_eq!(std.len(), 1);
    assert_eq!(std[0].1.to_bits(), NEAR_EQUAL_STDDEV);
}

#[test]
fn variance_of_identical_large_values_is_positive_zero() {
    // x − mean = +0 for every point (x − x = +0), in both the two-pass and
    // the Welford path; no rounding residue from the large magnitude
    let db = db();
    db.put_batch(&[(0, 1e8), (1, 1e8), (2, 1e8), (3, 1e8)])
        .unwrap();
    let var = db.aggregate(0, 3, Aggregation::Variance).unwrap();
    assert_eq!(var.to_bits(), POS_ZERO_F64);
    let grouped = db.downsample(0, 3, 10, Aggregation::Variance).unwrap();
    assert_eq!(grouped[0].1.to_bits(), POS_ZERO_F64);
}

#[test]
fn variance_of_identical_inexact_values_is_positive_zero() {
    // 0.1 is not representable; three copies of the same f32 still have an
    // exact mean, so the variance is +0 rather than a rounding residue
    let db = db();
    db.put_batch(&[(0, 0.1), (1, 0.1), (2, 0.1)]).unwrap();
    let var = db.aggregate(0, 2, Aggregation::Variance).unwrap();
    assert_eq!(var.to_bits(), POS_ZERO_F64);
    let grouped = db.downsample(0, 2, 10, Aggregation::Variance).unwrap();
    assert_eq!(grouped[0].1.to_bits(), POS_ZERO_F64);
}

#[test]
fn constant_negative_zero_series_reads_back_negative_zero() {
    // a series of -0 is stored as a constant model of -0; every point read
    // from the segment, by get and by scan, is -0
    let db = db();
    let data: Vec<(i64, f32)> = (0..100).map(|t| (t, -0.0)).collect();
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    assert_eq!(get_bits(&db, 0), NEG_ZERO_F32);
    assert_eq!(get_bits(&db, 50), NEG_ZERO_F32);
    let points = db.scan(0, 99).unwrap();
    assert_eq!(points.len(), 100);
    assert!(points.iter().all(|&(_, v)| v.to_bits() == NEG_ZERO_F32));
}

#[test]
fn linear_through_zero_reads_positive_zero_at_the_crossing() {
    // y = -x on keys -50..=50 (y written as 0.0 - x, so +0 at x = 0); the
    // linear model gives 50 + 0.5·(-100) = 50 - 50 = +0 at the crossing
    let db = db();
    let data: Vec<(i64, f32)> = (-50i16..=50)
        .map(|t| (i64::from(t), 0.0 - f32::from(t)))
        .collect();
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    assert_eq!(get_bits(&db, -50), 50.0f32.to_bits());
    assert_eq!(get_bits(&db, 0), POS_ZERO_F32);
    assert_eq!(get_bits(&db, 50), (-50.0f32).to_bits());
}

#[test]
fn linear_from_positive_zero_starts_at_positive_zero() {
    // start value +0: +0 + 0·(-99) = +0 + (-0) = +0
    let db = db();
    let data: Vec<(i64, f32)> = (0i16..100)
        .map(|t| (i64::from(t), 0.0 - f32::from(t)))
        .collect();
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    assert_eq!(get_bits(&db, 0), POS_ZERO_F32);
    assert_eq!(get_bits(&db, 99), (-99.0f32).to_bits());
}

#[test]
fn linear_from_negative_zero_starts_at_negative_zero() {
    // start value -0: -0 + 0·(-99) = -0 + (-0) = -0, so the sign written at
    // the first key survives the model
    let db = db();
    let data: Vec<(i64, f32)> = (0i16..100)
        .map(|t| (i64::from(t), -(f32::from(t))))
        .collect();
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    assert_eq!(get_bits(&db, 0), NEG_ZERO_F32);
    assert_eq!(get_bits(&db, 99), (-99.0f32).to_bits());
}

#[test]
fn lossless_linear_keeps_negative_zero_at_the_crossing() {
    // y = -x written as -(x), so -0 at x = 0; the model gives +0 there and
    // the lossless residual restores the written -0
    let db = lossless_db();
    let data: Vec<(i64, f32)> = (-50i16..=50)
        .map(|t| (i64::from(t), -(f32::from(t))))
        .collect();
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    assert_eq!(get_bits(&db, -50), 50.0f32.to_bits());
    assert_eq!(get_bits(&db, 0), NEG_ZERO_F32);
    assert_eq!(get_bits(&db, 50), (-50.0f32).to_bits());
}

#[test]
fn lossless_alternating_zeros_keep_each_sign() {
    // -0, +0, -0, ... compare equal, so the fit is a constant; the lossless
    // residual gives every key back the sign it was written with
    let db = lossless_db();
    let data: Vec<(i64, f32)> = (0..100)
        .map(|t| (t, if t % 2 == 0 { -0.0 } else { 0.0 }))
        .collect();
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    for (k, v) in db.scan(0, 99).unwrap() {
        let expected = if k % 2 == 0 {
            NEG_ZERO_F32
        } else {
            POS_ZERO_F32
        };
        assert_eq!(v.to_bits(), expected, "key {k}");
    }
    assert_eq!(get_bits(&db, 0), NEG_ZERO_F32);
    assert_eq!(get_bits(&db, 1), POS_ZERO_F32);
}
