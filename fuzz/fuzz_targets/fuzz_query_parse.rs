//! Fuzz target: the query builder (`AliceDB::query()`) with arbitrary ranges,
//! aggregations, group-by intervals, limits and offsets over arbitrary points
//!
//! A query must return `Ok` or `Err` without panicking for every parameter
//! combination (reversed ranges, zero or negative intervals, huge limits and
//! offsets). A plain range query must return only points inside the range.

#![no_main]

use alice_db::query_engine::Aggregation;
use alice_db::{AliceDB, StorageConfig};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    points: Vec<(i16, i16)>,
    start: i64,
    end: i64,
    agg: u8,
    group_by: Option<i64>,
    limit: Option<usize>,
    offset: Option<usize>,
}

fn aggregation(tag: u8) -> Aggregation {
    match tag % 11 {
        0 => Aggregation::None,
        1 => Aggregation::Sum,
        2 => Aggregation::Avg,
        3 => Aggregation::Min,
        4 => Aggregation::Max,
        5 => Aggregation::Count,
        6 => Aggregation::First,
        7 => Aggregation::Last,
        8 => Aggregation::StdDev,
        9 => Aggregation::Variance,
        _ => Aggregation::None,
    }
}

fuzz_target!(|input: Input| {
    if input.points.len() > 512 {
        return;
    }
    let db = AliceDB::in_memory(StorageConfig::default()).expect("in-memory db");
    for &(t, v) in &input.points {
        let _ = db.put(i64::from(t), f32::from(v));
    }
    let mut q = db
        .query()
        .range(input.start, input.end)
        .aggregate(aggregation(input.agg));
    if let Some(g) = input.group_by {
        q = q.group_by(g);
    }
    if let Some(n) = input.limit {
        q = q.limit(n);
    }
    if let Some(n) = input.offset {
        q = q.offset(n);
    }
    let _ = q.execute();

    if let Ok(points) = db.scan(input.start, input.end) {
        assert!(
            points
                .iter()
                .all(|&(t, _)| t >= input.start && t <= input.end),
            "scan returned a point outside its range"
        );
    }
});
