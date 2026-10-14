//! `Aggregation::Min`/`Max` must return the IEEE 754 minimum/maximum for a
//! mixed `-0.0`/`+0.0` pair regardless of argument order, host architecture,
//! or build profile (G1/G2: a numeric law's bits must not depend on the
//! machine it ran on).
//!
//! `f64::min`/`f64::max` (and the `f32` methods) are *not* that: aarch64's
//! `fmin`/`fmax` instructions follow IEEE minimum/maximum (`min(-0, +0) =
//! -0`, `max(-0, +0) = +0`, independent of order), but `x86_64`'s
//! `minsd`/`maxsd` return the *second* operand whenever the two compare
//! equal — so the result silently depends on which operand arrived first.
//! `query_engine.rs` folds over points in timestamp order, so two input
//! streams that are the same multiset in a different order can report a
//! different sign of zero for a query that is ostensibly idempotent to row
//! order, and the two supported architectures can disagree on either one.
//!
//! Fixed by `alice_det_math::{min, max, min64, max64}`, which this crate
//! already depends on, pinning the IEEE-minimum/maximum choice on every
//! architecture.

use alice_db::query_engine::Aggregation;
use alice_db::{AliceDB, StorageConfig};

fn bits(v: f64) -> u64 {
    v.to_bits()
}

const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
const POS_ZERO: u64 = 0x0000_0000_0000_0000;

fn db_with(points: &[(i64, f32)]) -> AliceDB {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    db.put_batch(points).unwrap();
    db
}

fn plain_min_max(points: &[(i64, f32)]) -> (f64, f64) {
    let db = db_with(points);
    let lo = points.iter().map(|&(t, _)| t).min().unwrap();
    let hi = points.iter().map(|&(t, _)| t).max().unwrap();
    (
        db.aggregate(lo, hi, Aggregation::Min).unwrap(),
        db.aggregate(lo, hi, Aggregation::Max).unwrap(),
    )
}

/// `group_by` with an interval wide enough to put every point in one bucket:
/// exercises `BucketState::update`'s running `f32` min/max instead of the
/// plain `fold`.
fn grouped_min_max(points: &[(i64, f32)]) -> (f64, f64) {
    let db = db_with(points);
    let min = db
        .query()
        .range(i64::MIN, i64::MAX)
        .group_by(1_000_000)
        .aggregate(Aggregation::Min)
        .execute()
        .unwrap()
        .into_aggregates();
    let max = db
        .query()
        .range(i64::MIN, i64::MAX)
        .group_by(1_000_000)
        .aggregate(Aggregation::Max)
        .execute()
        .unwrap()
        .into_aggregates();
    assert_eq!(min.len(), 1);
    assert_eq!(max.len(), 1);
    (min[0].1, max[0].1)
}

#[test]
fn plain_aggregate_min_max_of_mixed_zero_is_order_independent() {
    for points in [
        vec![(0_i64, -0.0_f32), (1, 0.0)],
        vec![(0_i64, 0.0_f32), (1, -0.0)],
    ] {
        let (min, max) = plain_min_max(&points);
        assert_eq!(
            bits(min),
            NEG_ZERO,
            "min(-0, +0) must be -0 regardless of order, got {:016x}",
            bits(min)
        );
        assert_eq!(
            bits(max),
            POS_ZERO,
            "max(-0, +0) must be +0 regardless of order, got {:016x}",
            bits(max)
        );
    }
}

#[test]
fn grouped_aggregate_min_max_of_mixed_zero_is_order_independent() {
    for points in [
        vec![(0_i64, -0.0_f32), (1, 0.0)],
        vec![(0_i64, 0.0_f32), (1, -0.0)],
    ] {
        let (min, max) = grouped_min_max(&points);
        assert_eq!(bits(min), NEG_ZERO, "grouped min(-0, +0) must be -0");
        assert_eq!(bits(max), POS_ZERO, "grouped max(-0, +0) must be +0");
    }
}

#[test]
fn mixed_zero_alongside_a_third_point_still_resolves_to_ieee_minimum_maximum() {
    // -0 is still the overall minimum next to a positive value, and +0 the
    // overall maximum next to a negative one; only the *tie between the two
    // zeros themselves* is at risk, not the comparison against other values.
    for points in [
        vec![(0_i64, -0.0_f32), (1, 0.0), (2, 1.0)],
        vec![(0_i64, 0.0_f32), (1, -0.0), (2, 1.0)],
    ] {
        let (min, _) = plain_min_max(&points);
        assert_eq!(bits(min), NEG_ZERO);
    }
    for points in [
        vec![(0_i64, -1.0_f32), (1, -0.0), (2, 0.0)],
        vec![(0_i64, -1.0_f32), (1, 0.0), (2, -0.0)],
    ] {
        let (_, max) = plain_min_max(&points);
        assert_eq!(bits(max), POS_ZERO);
    }
}

#[test]
fn non_zero_inputs_are_unaffected() {
    let (min, max) = plain_min_max(&[(0, 1.0), (1, 2.0), (2, 3.0)]);
    assert_eq!(bits(min), 1.0_f64.to_bits());
    assert_eq!(bits(max), 3.0_f64.to_bits());

    let (min, max) = plain_min_max(&[(0, -3.0), (1, -2.0), (2, -1.0)]);
    assert_eq!(bits(min), (-3.0_f64).to_bits());
    assert_eq!(bits(max), (-1.0_f64).to_bits());
}
