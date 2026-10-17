//! Every write path stamps the identifier the read checks with
//!
//! Residuals and law records are stamped with `alice_db::SEMANTICS_ID` (alice-zip's
//! `law::SEMANTICS_ID`) and read through `alice_core::law::check_semantics`. If the stamp and
//! the check came from different crates (as they did before alice-zip 0.11, when the read
//! used alice-det-math's check), fresh data would read back as a previous identifier or as
//! a mismatch. Each path below writes fresh data and requires `Current`: being accepted as
//! some previous identifier is not enough.

#![cfg(feature = "fs")]
#![allow(clippy::cast_precision_loss)]

use alice_db::law_store::RecordSemantics;
use alice_db::segment::ReadStatus;
use alice_db::{AliceDB, FitConfig, StorageConfig};
use alice_det_math::SemanticsCheck;
use std::path::Path;

fn open(dir: &Path, lossless: bool, use_mmap: bool) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_wal: false,
        use_mmap,
        fit_config: FitConfig {
            lossless,
            ..FitConfig::default()
        },
        ..StorageConfig::default()
    })
    .expect("open")
}

/// The number of segment files under `dir` holding `id` as 32 consecutive bytes
fn segments_holding(dir: &Path, id: &[u8; 32]) -> usize {
    std::fs::read_dir(dir)
        .expect("dir")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("seg_"))
        .filter(|e| {
            std::fs::read(e.path())
                .expect("segment")
                .windows(32)
                .any(|w| w == id)
        })
        .count()
}

#[test]
fn the_stamp_is_the_current_identifier_of_the_check() {
    assert_eq!(alice_db::SEMANTICS_ID, alice_core::law::SEMANTICS_ID);
    assert_eq!(
        alice_core::law::check_semantics(&alice_db::SEMANTICS_ID),
        SemanticsCheck::Verified,
        "the stamp must be the check's current identifier, not only an accepted one"
    );
}

/// `(what, lossless, the values written)`: a lossless series, and a lossy series holding a
/// non-finite value (stored exactly with a residual)
/// `(what, lossless, the values written)`
type WritePath = (&'static str, bool, Vec<(i64, f32)>);

fn write_paths() -> [WritePath; 2] {
    [
        (
            "lossless",
            true,
            (0..64_i64)
                .map(|k| (k, 0.5 * k as f32 + 0.125 * ((k * 7) % 5) as f32))
                .collect(),
        ),
        (
            "lossy with a non-finite value",
            false,
            (0..64_i64)
                .map(|k| {
                    (
                        k,
                        if k == 9 {
                            f32::INFINITY
                        } else {
                            0.25 * k as f32
                        },
                    )
                })
                .collect(),
        ),
    ]
}

#[test]
fn fresh_residuals_read_back_as_current() {
    let mut compared = 0;
    for (what, lossless, points) in write_paths() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let dir = tmp.path().join("store");
        {
            let db = open(&dir, lossless, false);
            db.put_batch(&points).expect("put");
            db.flush().expect("flush");
            db.close().expect("close");
        }
        assert!(
            segments_holding(&dir, &alice_db::SEMANTICS_ID) > 0,
            "{what}: no segment is stamped with the current identifier"
        );
        for use_mmap in [false, true] {
            let db = open(&dir, lossless, use_mmap);
            for &(k, v) in &points {
                let (got, status) = db.get_with_status(k).expect("read").expect("present");
                assert_eq!(
                    got.to_bits(),
                    v.to_bits(),
                    "{what} mmap {use_mmap}: key {k}"
                );
                assert_eq!(
                    status,
                    ReadStatus::Current,
                    "{what} mmap {use_mmap}: key {k}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 2 * 2 * 64);
}

#[test]
fn a_fresh_law_record_reads_back_as_current() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    let db = open(&dir, false, false);
    let points: Vec<(f64, f64)> = (0..=8)
        .map(|i| {
            let x = f64::from(i) / 2.0;
            (x, 2.0 * x + 1.0)
        })
        .collect();
    let law = alice_core::law::SignalLaw::fit_polynomial(
        &points,
        1,
        alice_core::law::Provenance::new("test", "line"),
    )
    .expect("fit");
    db.put_law("ohm", &law, &alice_db::SEMANTICS_ID)
        .expect("put law");
    let stored = db.get_law("ohm").expect("read").expect("present");
    assert_eq!(stored.semantics, RecordSemantics::Current);
    let value = db
        .evaluate_by_id(&law.law_id(&alice_db::SEMANTICS_ID), 2.5)
        .expect("found by its law identifier");
    assert_eq!(value.semantics, RecordSemantics::Current);
}
