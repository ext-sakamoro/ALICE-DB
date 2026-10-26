//! A stored segment answers through a `SegmentView` exactly as the segment it decodes to does in memory: one
//! law path. The view of a `RawLzma` segment once answered 0.0 while the in-memory segment held the values
//! (0.2.0-beta.1), because the view evaluated archived models with a copy of the laws of its own.
#![cfg(feature = "fs")]

use alice_db::segment::{DataSegment, SegmentView};
use alice_db::AliceDB;
use std::path::Path;

/// Every point and the whole range, through the view and through the owned segment, bit for bit.
fn assert_same(view: &SegmentView, seg: &DataSegment, what: &str) {
    let (a, b) = (seg.start_time, seg.end_time);
    let span = b.saturating_sub(a).min(4096);
    for t in a..=a + span {
        let v = view.query_point(t).map(|o| o.map(f32::to_bits));
        let m = seg.query_point(t).map(|o| o.map(f32::to_bits));
        assert_eq!(format!("{v:?}"), format!("{m:?}"), "{what}: point {t}");
        assert_eq!(
            view.query_point_model_only(t).map(f32::to_bits),
            seg.query_point_model_only(t).map(f32::to_bits),
            "{what}: model {t}"
        );
    }
    let v = view.query_range(a, a + span).map(|r| {
        r.into_iter()
            .map(|(t, x)| (t, x.to_bits()))
            .collect::<Vec<_>>()
    });
    let m = seg.query_range(a, a + span).map(|r| {
        r.into_iter()
            .map(|(t, x)| (t, x.to_bits()))
            .collect::<Vec<_>>()
    });
    assert_eq!(format!("{v:?}"), format!("{m:?}"), "{what}: range");
    assert_eq!(view.point_count(), seg.metadata.point_count, "{what}");
}

#[test]
fn a_raw_lzma_segment_answers_through_the_view_as_in_memory() {
    // values no procedural model fits, so the flush stores them as RawLzma
    let dir = tempfile::tempdir().unwrap();
    let db = AliceDB::open(dir.path().join("raw")).unwrap();
    let values: Vec<(i64, f32)> = (0..64)
        .map(|i| {
            (
                i,
                f32::from(u16::try_from((i * 7919) % 101).unwrap()) * 0.37 - 5.0,
            )
        })
        .collect();
    db.put_batch(&values).unwrap();
    db.flush().unwrap();
    drop(db);
    let mut segments = 0;
    for entry in std::fs::read_dir(dir.path().join("raw")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rkyv") {
            let bytes = std::fs::read(&path).unwrap();
            let seg = DataSegment::from_stored_bytes(&bytes).unwrap();
            assert!(
                matches!(seg.model, alice_db::model::ModelType::RawLzma { .. }),
                "the values must be stored as RawLzma"
            );
            for &(t, v) in &values {
                assert_eq!(
                    seg.query_point(t).unwrap().map(f32::to_bits),
                    Some(v.to_bits()),
                    "in memory, t={t}"
                );
            }
            assert_same(&SegmentView::open(&path).unwrap(), &seg, "flushed RawLzma");
            segments += 1;
        }
    }
    assert!(segments > 0, "the flush wrote no segment file");
}

#[test]
fn every_stored_fixture_answers_through_the_view_as_in_memory() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut files = Vec::new();
    for store in [
        "store_pre_0_3_0",
        "store_det_math_0_4",
        "store_db_0_5_0",
        "store_db_0_5_1",
        "store_db_0_6_0",
        "store_db_0_6_0_det_math_0_5_2",
    ] {
        files.push(root.join(store).join("seg_1.rkyv"));
    }
    for name in [
        "polynomial",
        "fourier",
        "sine_wave",
        "multi_sine",
        "perlin_noise",
        "constant",
        "linear",
        "raw_lzma",
    ] {
        files.push(
            root.join("segment_format0_variants")
                .join(format!("{name}.rkyv")),
        );
    }
    for file in &files {
        let bytes = std::fs::read(file).unwrap();
        let seg = DataSegment::from_stored_bytes(&bytes).unwrap();
        assert_same(
            &SegmentView::from_vec(bytes.clone()).unwrap(),
            &seg,
            &file.display().to_string(),
        );
        // and the same segment written in the current format
        let current = seg.to_rkyv_bytes().unwrap();
        assert_same(
            &SegmentView::from_vec(current).unwrap(),
            &seg,
            &format!("{} (format 1)", file.display()),
        );
    }
    assert_eq!(files.len(), 14);
}
