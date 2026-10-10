//! `group_by` / `downsample` buckets are floor buckets: key `t` belongs to
//! `[k·interval, (k+1)·interval)` with `k = ⌊t / interval⌋`, labelled
//! `k·interval`, for every integer `k`
//!
//! Until 0.3.0 the label was `(t / interval) * interval`, which truncates
//! toward zero: with interval 10 the keys −9..=9 all went to bucket 0
//! (19 keys, 2·interval − 1 wide) and negative keys were labelled one
//! bucket too high (−1 → 0, −11 → −10).
//!
//! The lowest bucket can start below `i64::MIN` (`i64::MIN` is
//! −2^63, so its floor bucket starts below it for every interval that does
//! not divide 2^63, e.g. MIN − 6 for interval 7). A label must be an `i64`,
//! so such a bucket is labelled with its lowest key that exists, `i64::MIN`;
//! which keys share the bucket is still decided by the floor rule.
//!
//! The expected values are computed here from the raw points with `i128`
//! floor division and plain accumulation, not by calling the crate.

use alice_db::query_engine::Aggregation;
use alice_db::{AliceDB, StorageConfig};
use std::collections::BTreeMap;

fn db_with(points: &[(i64, f32)]) -> AliceDB {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    db.put_batch(points).unwrap();
    db
}

fn grouped(db: &AliceDB, interval: i64, agg: Aggregation) -> Vec<(i64, f64)> {
    db.query()
        .range(i64::MIN, i64::MAX)
        .group_by(interval)
        .aggregate(agg)
        .execute()
        .unwrap()
        .into_aggregates()
}

/// Reference label: the floor bucket's start, or `i64::MIN` when that start
/// lies below the `i64` range
fn floor_label(t: i64, interval: i64) -> i64 {
    let start = i128::from(t).div_euclid(i128::from(interval)) * i128::from(interval);
    i64::try_from(start).unwrap_or(i64::MIN)
}

/// Values that are exact in `f32` and `f64` sums, with both signs
fn value(t: i64) -> f32 {
    #[allow(clippy::cast_precision_loss, reason = "|r| < 7")]
    let r = t.rem_euclid(7) as f32;
    r.mul_add(1.5, -4.0)
}

/// Reference aggregates per floor bucket, in key order
fn reference(points: &[(i64, f32)], interval: i64) -> BTreeMap<i64, Vec<f32>> {
    let mut buckets: BTreeMap<i64, Vec<f32>> = BTreeMap::new();
    for &(t, v) in points {
        buckets.entry(floor_label(t, interval)).or_default().push(v);
    }
    buckets
}

/// One reference aggregate over a bucket's values
type Fold = fn(&[f32]) -> f64;

/// A small count as `f64` (exact)
fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("small count"))
}

fn points(keys: impl IntoIterator<Item = i64>) -> Vec<(i64, f32)> {
    keys.into_iter().map(|t| (t, value(t))).collect()
}

/// Every bucket holds exactly the keys of `[k·i, (k+1)·i)`: interior
/// buckets hold `interval` keys, including the one around zero
#[test]
fn every_bucket_is_interval_wide() {
    let pts = points(-25..=25);
    let db = db_with(&pts);
    for interval in [1i64, 3, 7, 10] {
        let got = grouped(&db, interval, Aggregation::Count);
        let want: Vec<(i64, f64)> = reference(&pts, interval)
            .into_iter()
            .map(|(label, vs)| (label, count(vs.len())))
            .collect();
        assert_eq!(got, want, "interval {interval}");
        // interior buckets (not cut by −25 or 25) are interval wide
        let width = count(usize::try_from(interval).unwrap());
        for &(label, n) in &got {
            if label >= -25 && label + interval - 1 <= 25 {
                assert_eq!(
                    n.to_bits(),
                    width.to_bits(),
                    "interval {interval}, bucket {label}"
                );
            }
        }
    }
}

/// Labels around zero with interval 10
#[test]
fn labels_floor_toward_negative_infinity() {
    let keys = [-11i64, -10, -9, -1, 0, 9, 10];
    let db = db_with(&points(keys));
    assert_eq!(
        grouped(&db, 10, Aggregation::Count),
        [(-20, 1.0), (-10, 3.0), (0, 2.0), (10, 1.0)]
    );
    for (t, label) in [(-1, -10), (-10, -10), (-11, -20), (9, 0), (10, 10), (0, 0)] {
        assert_eq!(floor_label(t, 10), label, "reference for {t}");
        let got = db
            .query()
            .range(t, t)
            .group_by(10)
            .aggregate(Aggregation::Count)
            .execute()
            .unwrap()
            .into_aggregates();
        assert_eq!(got, [(label, 1.0)], "key {t}");
    }
    // downsample uses the same buckets
    assert_eq!(
        db.downsample(-11, 10, 10, Aggregation::Count).unwrap(),
        [(-20, 1.0), (-10, 3.0), (0, 2.0), (10, 1.0)]
    );
}

/// Sum / avg / count / min / max per bucket over negative keys equal the
/// reference computed from the raw points
#[test]
fn aggregates_per_bucket_match_the_reference() {
    let pts = points(-47..=-1);
    let db = db_with(&pts);
    for interval in [3i64, 10] {
        let reference = reference(&pts, interval);
        let folds: [(Aggregation, Fold); 5] = [
            (Aggregation::Sum, |v| v.iter().map(|&x| f64::from(x)).sum()),
            (Aggregation::Avg, |v| {
                v.iter().map(|&x| f64::from(x)).sum::<f64>() / count(v.len())
            }),
            (Aggregation::Count, |v| count(v.len())),
            (Aggregation::Min, |v| {
                f64::from(v.iter().copied().fold(f32::INFINITY, f32::min))
            }),
            (Aggregation::Max, |v| {
                f64::from(v.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            }),
        ];
        for (agg, fold) in folds {
            let want: Vec<(i64, f64)> = reference.iter().map(|(&l, vs)| (l, fold(vs))).collect();
            assert_eq!(
                grouped(&db, interval, agg),
                want,
                "interval {interval}, {agg:?}"
            );
        }
    }
}

/// Keys at the ends of `i64`: no overflow, every label is an `i64`, and a
/// floor bucket starting below `i64::MIN` is labelled `i64::MIN`
#[test]
fn extreme_keys() {
    let (min, max) = (i64::MIN, i64::MAX);
    let keys = [min, min + 1, -1, 0, max - 1, max];
    let db = db_with(&points(keys));
    let cases: [(i64, &[(i64, f64)]); 4] = [
        (
            1,
            &[
                (min, 1.0),
                (min + 1, 1.0),
                (-1, 1.0),
                (0, 1.0),
                (max - 1, 1.0),
                (max, 1.0),
            ],
        ),
        // 2^63 ≡ 1 mod 7: the bucket of MIN is [MIN − 6, MIN + 1), labelled
        // MIN; MIN + 1 starts the next one. MAX = 7·(2^63 − 1)/7 is a start
        (
            7,
            &[
                (min, 1.0),
                (min + 1, 1.0),
                (-7, 1.0),
                (0, 1.0),
                (max - 7, 1.0),
                (max, 1.0),
            ],
        ),
        // ⌊MIN / MAX⌋ = −2: the bucket starts at −2·MAX = MIN − (MAX − 1)
        (max, &[(min, 1.0), (-max, 2.0), (0, 2.0), (max, 1.0)]),
        // 2^62 divides 2^63: the lowest bucket starts exactly at MIN
        (
            1 << 62,
            &[(min, 2.0), (-(1 << 62), 1.0), (0, 1.0), (1 << 62, 2.0)],
        ),
    ];
    for (interval, want) in cases {
        assert_eq!(
            grouped(&db, interval, Aggregation::Count),
            want,
            "interval {interval}"
        );
        for t in keys {
            assert_eq!(
                want.iter()
                    .filter(|&&(l, _)| l == floor_label(t, interval))
                    .count(),
                1,
                "reference label of {t} for interval {interval}"
            );
        }
    }
    // the clamp is needed exactly where the floor start is below MIN
    for interval in [7i64, 3, 10, max] {
        let start = i128::from(min).div_euclid(i128::from(interval)) * i128::from(interval);
        assert!(start < i128::from(min), "interval {interval}");
    }
    for interval in [1i64, 2, 1 << 62] {
        let start = i128::from(min).div_euclid(i128::from(interval)) * i128::from(interval);
        assert_eq!(start, i128::from(min), "interval {interval}");
    }
}

/// Buckets without points are not emitted (unchanged; the second store has
/// negative keys, so its labels are floor labels)
#[test]
fn empty_buckets_are_skipped() {
    let db = db_with(&[(0, 1.0), (25, 2.0)]);
    assert_eq!(grouped(&db, 10, Aggregation::Sum), [(0, 1.0), (20, 2.0)]);
    let db = db_with(&[(-25, 1.0), (5, 2.0)]);
    assert_eq!(grouped(&db, 10, Aggregation::Sum), [(-30, 1.0), (0, 2.0)]);
}

/// The floor start of the bucket of `i64::MIN` for each interval, and the
/// reported label `max(bucket_start, i64::MIN)`
#[test]
fn lowest_bucket_label_is_clamped_to_min() {
    let min = i64::MIN;
    let cases: [(i64, i128); 4] = [
        (7, i128::from(min) - 6),
        (3, i128::from(min) - 1),
        (10, i128::from(min) - 2),
        (i64::MAX, -(1i128 << 64) + 2),
    ];
    for (interval, start) in cases {
        assert_eq!(
            i128::from(min).div_euclid(i128::from(interval)) * i128::from(interval),
            start,
            "floor start for interval {interval}"
        );
        let db = db_with(&[(min, 1.0)]);
        assert_eq!(
            grouped(&db, interval, Aggregation::Count),
            [(min, 1.0)],
            "interval {interval}"
        );
    }
}

/// Near both ends of `i64`: membership equals the `i128` floor reference,
/// labels are strictly increasing, so no two buckets share a label (the
/// clamped lowest bucket included)
#[test]
fn membership_and_distinct_labels_near_the_extremes() {
    let keys: Vec<i64> = (0..40)
        .map(|d| i64::MIN + d)
        .chain([-1, 0, 1])
        .chain((0..40).rev().map(|d| i64::MAX - d))
        .collect();
    let pts = points(keys.iter().copied());
    let db = db_with(&pts);
    for interval in [3i64, 7, 10, i64::MAX] {
        // reference by the exact i128 floor start (not by the clamped label)
        let mut exact: BTreeMap<i128, Vec<f32>> = BTreeMap::new();
        for &(t, v) in &pts {
            let start = i128::from(t).div_euclid(i128::from(interval)) * i128::from(interval);
            exact.entry(start).or_default().push(v);
        }
        let want: Vec<(i64, f64, f64)> = exact
            .iter()
            .map(|(&start, vs)| {
                (
                    i64::try_from(start.max(i128::from(i64::MIN))).unwrap(),
                    count(vs.len()),
                    vs.iter().map(|&x| f64::from(x)).sum(),
                )
            })
            .collect();
        let count = grouped(&db, interval, Aggregation::Count);
        let sum = grouped(&db, interval, Aggregation::Sum);
        let got: Vec<(i64, f64, f64)> = count
            .iter()
            .zip(&sum)
            .map(|(&(l, c), &(l2, s))| {
                assert_eq!(l, l2);
                (l, c, s)
            })
            .collect();
        assert_eq!(got, want, "interval {interval}");
        assert!(
            got.windows(2).all(|w| w[0].0 < w[1].0),
            "labels not strictly increasing for interval {interval}: {got:?}"
        );
        assert_eq!(got[0].0, i64::MIN, "interval {interval}");
    }
}
