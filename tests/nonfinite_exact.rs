//! A stored non-finite value (NaN, ±inf) reads back as stored through a lossy flush
//!
//! The lossy models' error bound is for finite values. A buffer holding a non-finite value is
//! therefore stored exactly whatever `FitConfig::lossless` says (the pinned XOR residual), so
//! `get` / `scan` return the stored value's own bits: a payload NaN, the infinities and -0.0
//! survive the flush. Canonical NaN applies where the crate computes a value (aggregates,
//! model evaluation), not to a stored value read back. The cost: such a segment is not
//! lossy-compressed.

#![cfg(feature = "fs")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]

use alice_db::{AliceDB, FitConfig, ResidualPolicy, StorageConfig};
use std::path::Path;

/// a NaN with the sign bit and a payload set, opaque to the optimiser
fn payload_nan() -> f32 {
    std::hint::black_box(f32::from_bits(0xffc0_dead))
}

fn open(dir: &Path, use_mmap: bool) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_wal: false,
        use_mmap,
        fit_config: FitConfig::default(),
        ..StorageConfig::default()
    })
    .expect("open")
}

/// 64 evenly spaced keys of a line, with the special values at four keys
fn series() -> Vec<(i64, f32)> {
    (0..64_i64)
        .map(|k| {
            let v = match k {
                5 => payload_nan(),
                17 => f32::INFINITY,
                29 => f32::NEG_INFINITY,
                41 => -0.0,
                _ => 0.25 * k as f32,
            };
            (k, v)
        })
        .collect()
}

#[test]
fn the_default_fit_config_is_lossy() {
    assert!(!FitConfig::default().lossless);
}

#[test]
fn non_finite_and_negative_zero_read_back_as_stored_after_a_lossy_flush() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    let points = series();
    {
        let db = open(&dir, false);
        db.put_batch(&points).expect("put");
        db.flush().expect("flush");
        db.close().expect("close");
    }
    let mut compared = 0;
    let mut special = 0;
    for use_mmap in [false, true] {
        let db = open(&dir, use_mmap);
        for &(k, want) in &points {
            let got = db.get(k).expect("read").expect("a stored key");
            assert_eq!(got.to_bits(), want.to_bits(), "mmap {use_mmap} key {k}");
            compared += 1;
            if !want.is_finite() || want.to_bits() == (-0.0f32).to_bits() {
                special += 1;
            }
        }
        let scanned = db.scan(0, 63).expect("scan");
        assert_eq!(scanned.len(), points.len());
        for (&(k, got), &(_, want)) in scanned.iter().zip(&points) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "scan mmap {use_mmap} key {k}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 256);
    assert_eq!(special, 8, "four special values, two readers");
}

/// A line with a payload NaN and -0.0 only (no infinity): a lossy fit approximates such a
/// line, so only the exact storage of a buffer holding a non-finite value returns them
#[test]
fn a_nan_and_negative_zero_on_a_line_read_back_as_stored_after_a_lossy_flush() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    let points: Vec<(i64, f32)> = (0..64_i64)
        .map(|k| {
            // -0.0 where the line is 0: the line still fits, and its model value there is +0
            let v = match k {
                0 => -0.0,
                9 => payload_nan(),
                _ => 0.25 * k as f32,
            };
            (k, v)
        })
        .collect();
    {
        let db = open(&dir, false);
        db.put_batch(&points).expect("put");
        db.flush().expect("flush");
        db.close().expect("close");
    }
    let mut compared = 0;
    for use_mmap in [false, true] {
        let db = open(&dir, use_mmap);
        for &(k, want) in &points {
            let got = db.get(k).expect("read").expect("a stored key");
            assert_eq!(got.to_bits(), want.to_bits(), "mmap {use_mmap} key {k}");
            compared += 1;
        }
    }
    assert_eq!(compared, 128);
}

/// Keys that are not evenly spaced before the first non-finite value: the buffer [0, 5] is
/// sealed as a lossy segment of finite values, and the NaN at 6 starts an exactly stored one.
/// Without that seal the NaN would join [0, 5], and that segment's even grid (0, 3, 6) would
/// read key 5 from the NaN's slot
#[test]
fn a_non_finite_value_after_irregular_keys_still_reads_back_as_stored() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    let mut points: Vec<(i64, f32)> = vec![(0, 0.0), (5, 5.0), (6, payload_nan())];
    points.extend((7..40_i64).map(|k| (k, k as f32)));
    {
        let db = open(&dir, false);
        db.put_batch(&points).expect("put");
        db.flush().expect("flush");
        db.close().expect("close");
    }
    let mut checked = 0;
    for use_mmap in [false, true] {
        let db = open(&dir, use_mmap);
        let got = db.get(6).expect("read").expect("present");
        assert_eq!(got.to_bits(), 0xffc0_dead, "mmap {use_mmap}");
        assert_eq!(db.get(5).expect("read"), Some(5.0), "mmap {use_mmap} key 5");
        for k in 7..40_i64 {
            assert_eq!(
                db.get(k).expect("read"),
                Some(k as f32),
                "mmap {use_mmap} key {k}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 66);
}

/// The boundary between stored and computed: a default read returns the stored NaN's own
/// bits, the model-only read (a computed value) returns the canonical NaN 0x7fc00000
#[test]
fn the_model_only_read_of_a_stored_nan_slot_is_the_canonical_nan() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    {
        let db = open(&dir, false);
        db.put_batch(&series()).expect("put");
        db.flush().expect("flush");
        db.close().expect("close");
    }
    let mut compared = 0;
    for use_mmap in [false, true] {
        let stored = open(&dir, use_mmap).get(5).expect("read").expect("present");
        assert_eq!(
            stored.to_bits(),
            0xffc0_dead,
            "mmap {use_mmap}: the stored bits"
        );
        let model = AliceDB::with_config(StorageConfig {
            data_dir: dir.clone(),
            enable_wal: false,
            use_mmap,
            residual_policy: ResidualPolicy::ModelOnly,
            ..StorageConfig::default()
        })
        .expect("open");
        let v = model.get(5).expect("read").expect("present");
        assert_eq!(
            v.to_bits(),
            0x7fc0_0000,
            "mmap {use_mmap}: the computed value"
        );
        let scanned = model.scan(0, 63).expect("scan");
        for (k, v) in scanned {
            if v.is_nan() {
                assert_eq!(v.to_bits(), 0x7fc0_0000, "mmap {use_mmap} key {k}");
                compared += 1;
            }
        }
    }
    assert!(compared > 0, "no NaN came out of the model-only scan");
}

/// A lossy segment's points are model evaluations (computed): a model that evaluates to a NaN
/// leaves as the canonical 0x7fc00000, in both readers, for a point and a range
#[test]
fn a_lossy_segment_whose_model_is_nan_reads_the_canonical_nan() {
    use alice_db::segment::{DataSegment, SegmentView};
    use alice_db::ModelType;
    let seg = DataSegment::new(
        1,
        0,
        9,
        ModelType::Constant {
            // a NaN with the sign bit and a payload, as a model's arithmetic may leave it
            value: std::hint::black_box(f64::from_bits(0xfff8_0000_0000_dead)),
        },
        10,
        40,
        0,
    );
    let view = SegmentView::from_vec(seg.to_rkyv_bytes().expect("rkyv")).expect("view");
    let mut compared = 0;
    let p = seg.query_point(3).expect("read").expect("in range");
    let v = view.query_point(3).expect("read").expect("in range");
    for (k, x) in seg
        .query_range(0, 9)
        .expect("read")
        .into_iter()
        .chain(view.query_range(0, 9).expect("read"))
        .chain([(3, p), (3, v)])
    {
        assert_eq!(x.to_bits(), 0x7fc0_0000, "key {k}");
        compared += 1;
    }
    assert_eq!(compared, 22);
}
