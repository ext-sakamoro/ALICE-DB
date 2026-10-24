//! An exactly stored buffer never takes a computed NaN as its residual base
//!
//! A residual is `original ^ model`: the stored bits are recovered by XOR with the model's
//! value at the same key. The bits of a computed NaN are not specified (the default NaN of
//! the architecture: `0x7fc0_0000` on aarch64, `0xffc0_0000` on `x86_64`), so a model that
//! evaluates to NaN would make the residual depend on the machine that wrote it: a store
//! written on one architecture would read back sign-flipped on the other. A fit over
//! non-finite values can land there (`Linear { start: inf, end: 63 }` evaluates `inf - inf`).
//! The flush therefore stores such a buffer with a model that evaluates to a constant, and
//! the residual holds the stored bits.
//!
//! The golden below is a segment written by this crate (alice_serial format 1); every CI OS
//! writes the same bytes and reads the stored bits back from them. The same segment as an older
//! release wrote it (bincode) is read back through a legacy data file.

#![allow(clippy::cast_precision_loss, clippy::unreadable_literal)]

use alice_db::segment::DataSegment;
use alice_db::{FitConfig, MemTable};

/// a NaN with the sign bit and a payload set, opaque to the optimiser
fn payload_nan() -> f32 {
    std::hint::black_box(f32::from_bits(0xffc0_dead))
}

/// `+inf` at key 0, then the line `k` (the fit picks a line through `inf`)
fn inf_then_line() -> Vec<(i64, f32)> {
    (0..64_i64)
        .map(|k| (k, if k == 0 { f32::INFINITY } else { k as f32 }))
        .collect()
}

fn series() -> Vec<(&'static str, Vec<(i64, f32)>)> {
    vec![
        ("inf then line", inf_then_line()),
        (
            "all +inf",
            (0..64_i64).map(|k| (k, f32::INFINITY)).collect(),
        ),
        (
            "alternating +-inf",
            (0..64_i64)
                .map(|k| {
                    (
                        k,
                        if k % 2 == 0 {
                            f32::INFINITY
                        } else {
                            f32::NEG_INFINITY
                        },
                    )
                })
                .collect(),
        ),
        (
            "all payload NaN",
            (0..64_i64).map(|k| (k, payload_nan())).collect(),
        ),
    ]
}

/// one segment holding `points`, written by a lossy (default) and a lossless memtable
fn flushed(points: &[(i64, f32)], lossless: bool) -> DataSegment {
    let table = MemTable::with_config_and_created_at(
        1024,
        FitConfig {
            lossless,
            ..FitConfig::default()
        },
        0,
    );
    assert!(
        table.put_batch(points).is_empty(),
        "no flush before the end"
    );
    table.force_flush().expect("one segment")
}

#[test]
fn the_model_of_an_exact_buffer_evaluates_finite_at_every_key() {
    for (name, points) in series() {
        for lossless in [false, true] {
            let segment = flushed(&points, lossless);
            for &(k, stored) in &points {
                let model = segment.query_point_model_only(k).expect("in range");
                assert!(
                    model.is_finite(),
                    "{name} (lossless {lossless}): the model is {model} at key {k} ({:?})",
                    segment.model
                );
                let read = segment.query_point(k).expect("residual").expect("in range");
                assert_eq!(
                    read.to_bits(),
                    stored.to_bits(),
                    "{name} (lossless {lossless}) key {k}"
                );
            }
        }
    }
}

/// `inf_then_line` through the default (lossy) memtable, as written by this crate
const INF_THEN_LINE_SEGMENT: &str = include_str!("fixtures/residual_base/inf_then_line.hex");

/// The same segment as alice-db wrote it before alice_serial (bincode)
#[cfg(feature = "fs")]
const INF_THEN_LINE_BINCODE: &str =
    include_str!("fixtures/residual_base/inf_then_line.bincode.hex");

#[cfg(feature = "fs")]
#[path = "common/legacy_data_file.rs"]
mod legacy_data_file;

fn unhex(s: &str) -> Vec<u8> {
    let s = s.trim();
    assert!(s.len().is_multiple_of(2), "odd hex length");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

#[test]
fn the_segment_bytes_are_the_same_on_every_architecture() {
    let written = flushed(&inf_then_line(), false).to_bytes().expect("bytes");
    assert_eq!(
        hex(&written),
        INF_THEN_LINE_SEGMENT.trim(),
        "the segment written here differs from the committed one"
    );
}

#[test]
fn the_committed_segment_reads_back_the_stored_bits() {
    let segment = DataSegment::from_bytes(&unhex(INF_THEN_LINE_SEGMENT)).expect("decode");
    let points = inf_then_line();
    let mut compared = 0;
    for &(k, stored) in &points {
        let read = segment.query_point(k).expect("residual").expect("in range");
        assert_eq!(read.to_bits(), stored.to_bits(), "key {k}");
        compared += 1;
    }
    assert_eq!(compared, 64);
}

#[cfg(feature = "fs")]
#[test]
fn the_segment_an_older_release_wrote_reads_back_the_same_bits() {
    let segment = legacy_data_file::read_through_data_file(&unhex(INF_THEN_LINE_BINCODE));
    let new = DataSegment::from_bytes(&unhex(INF_THEN_LINE_SEGMENT)).expect("decode");
    assert_eq!(
        segment.to_bytes().expect("bytes"),
        new.to_bytes().expect("bytes"),
        "the bincode segment and the alice_serial segment are not the same segment"
    );
    let mut compared = 0;
    for &(k, stored) in &inf_then_line() {
        let read = segment.query_point(k).expect("residual").expect("in range");
        assert_eq!(read.to_bits(), stored.to_bits(), "key {k}");
        compared += 1;
    }
    assert_eq!(compared, 64);
}
