//! `group_by` / `downsample` intervals: a zero or negative interval is
//! refused, and a positive interval works over any key span
//!
//! Until 0.3.0 the grouped aggregation estimated its bucket count as
//! `((last − first) / interval + 1) as usize`. A negative interval made that
//! negative, the cast turned it into a huge capacity and `Vec::with_capacity`
//! panicked ("capacity overflow", found by `fuzz_query_parse`; that input
//! and one with interval 0 are in `fuzz/regressions/fuzz_query_parse/`); a
//! zero interval divided by zero;
//! `last − first` overflowed `i64` for a span above `i64::MAX`. An interval
//! must now be positive (`io::ErrorKind::InvalidInput` otherwise, whatever
//! the data), and the bucket count is computed in `i128` and capped at the
//! number of points (every bucket holds at least one).
//!
//! Expected values are written out by hand from the points.

use alice_db::query_engine::{Aggregation, QueryResult};
use alice_db::{AliceDB, StorageConfig};
use std::io::ErrorKind;

fn db_with(points: &[(i64, f32)]) -> AliceDB {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    db.put_batch(points).unwrap();
    db
}

fn group(db: &AliceDB, interval: i64, agg: Aggregation) -> std::io::Result<QueryResult> {
    db.query()
        .range(i64::MIN, i64::MAX)
        .group_by(interval)
        .aggregate(agg)
        .execute()
}

/// Every interval must be refused with `InvalidInput`, on buffered points
/// (memtable), on flushed points (segment) and on an empty store, through
/// the builder and through `downsample`
fn assert_refused(interval: i64) {
    // the shape of the fuzz input: several keys below zero
    let points: Vec<(i64, f32)> = (-8..=-1).map(|t| (t, -1.0)).collect();
    let flushed = db_with(&points);
    flushed.flush().unwrap();
    for (what, db) in [
        ("buffered", db_with(&points)),
        ("flushed", flushed),
        ("empty", db_with(&[])),
    ] {
        for agg in [Aggregation::Avg, Aggregation::Count, Aggregation::None] {
            let e = group(&db, interval, agg).expect_err("interval must be refused");
            assert_eq!(e.kind(), ErrorKind::InvalidInput, "{what}, {interval}");
        }
        let e = db
            .downsample(i64::MIN, i64::MAX, interval, Aggregation::Avg)
            .expect_err("downsample must refuse the interval too");
        assert_eq!(
            e.kind(),
            ErrorKind::InvalidInput,
            "{what}, downsample {interval}"
        );
    }
}

/// Until 0.3.0: "attempt to divide by zero"
#[test]
fn zero_interval_is_refused() {
    assert_refused(0);
}

/// Until 0.3.0: "capacity overflow" (the bucket count was negative)
#[test]
fn negative_interval_is_refused() {
    assert_refused(-1);
    assert_refused(-7);
}

/// Until 0.3.0 this did not panic on these keys but returned one bucket
/// labelled 0 for keys −8..=−1
#[test]
fn i64_min_interval_is_refused() {
    assert_refused(i64::MIN);
}

/// Keys near ±2^62 and at the ends of `i64`: `last − first` exceeds
/// `i64::MAX`, the bucket count is at most the number of points
#[test]
fn huge_spans_group_without_overflow() {
    let p = 1i64 << 62;
    let cases: [(&[(i64, f32)], i64); 4] = [
        (&[(-p, 1.0), (p, 2.0)], 1),
        (&[(-p, 1.0), (0, 3.0), (p, 2.0)], 1),
        (&[(i64::MIN, 1.0), (i64::MAX, 2.0)], 1),
        (&[(i64::MIN, 1.0), (-1, 4.0), (i64::MAX, 2.0)], i64::MAX),
    ];
    for (points, interval) in cases {
        let db = db_with(points);
        let got = group(&db, interval, Aggregation::Sum)
            .unwrap()
            .into_aggregates();
        // interval 1: one bucket per key; interval i64::MAX: floor buckets
        // starting at −2·MAX (labelled MIN, see `tests/query_group_by_floor.rs`)
        // for MIN, −MAX for −1 and MAX for MAX
        let want: Vec<(i64, f64)> = points
            .iter()
            .map(|&(t, v)| {
                let label = if interval == 1 {
                    t
                } else if t == i64::MIN {
                    i64::MIN
                } else if t < 0 {
                    -i64::MAX
                } else {
                    i64::MAX
                };
                (label, f64::from(v))
            })
            .collect();
        assert_eq!(got, want, "points {points:?}, interval {interval}");
    }
}

/// A positive interval keeps grouping as before
#[test]
fn positive_interval_groups_by_bucket() {
    let points: Vec<(i64, f32)> = (0..10).map(|t| (t, 1.0)).collect();
    let db = db_with(&points);
    let got = group(&db, 5, Aggregation::Count).unwrap().into_aggregates();
    assert_eq!(got, [(0, 5.0), (5, 5.0)]);
    let got = db.downsample(0, 9, 3, Aggregation::Sum).unwrap();
    assert_eq!(got, [(0, 3.0), (3, 3.0), (6, 3.0), (9, 1.0)]);
}
