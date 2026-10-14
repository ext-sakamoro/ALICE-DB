//! Reading a store whose lossless residual was measured against another
//! arithmetic (`tests/fixtures/store_det_math_0_4`: written by alice-db 0.3.0 on
//! alice-det-math 0.4, identifier d2209b30…).
//!
//! The residual stores `original ^ model` under the writer's arithmetic. Under
//! another one it cannot reconstruct the original, and returning the model
//! alone returns a value that is not the original without saying so. The
//! default read refuses with a typed error naming the data; the model is
//! read only when asked for ([`ResidualPolicy::ModelOnly`] /
//! `query_point_model_only`).

#![cfg(feature = "fs")]
// numeric oracles cast freely between the index / float domains and carry
// f32 bit patterns as hex; the crate's pedantic gate is about API code
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::unreadable_literal,
    clippy::too_many_lines
)]

use alice_db::segment::{ResidualStatus, SegmentView};
use alice_db::{AliceDB, ResidualError, ResidualMismatch, ResidualPolicy, StorageConfig};
use std::path::{Path, PathBuf};

const OLD_ID: &str = "d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e";

/// The 64 values the 0.3.0 store was written with, as f32 bits
const ORIGINAL: [u32; 64] = [
    0x3f83b0ec, 0x3fc4eb74, 0x3ff76aa4, 0x400b55e0, 0x400ff207, 0x4009205c, 0x3feef99e, 0x3fb93575,
    0x3f6b838b, 0x3eaa9416, 0xbe85ac98, 0xbf4f46b6, 0xbfa1bdcf, 0xbfca89db, 0xbfde6289, 0xbfdb8203,
    0xbfc22a13, 0xbf949e45, 0xbf2de051, 0xbdea22d8, 0x3ef758b3, 0x3f87a443, 0x3fc8304c, 0x3ff9b639,
    0x400be4cc, 0x400fdd4f, 0x400869dc, 0x3fec69a7, 0x3fb5bd1f, 0x3f6360d0, 0x3e995ed4, 0xbe96485e,
    0xbf5689f0, 0xbfa48711, 0xbfcc3b05, 0xbfded4e8, 0xbfdaab65, 0xbfc01da2, 0xbf918ada, 0xbf263826,
    0xbda650f4, 0x3f0436f7, 0x3f8b9011, 0x3fcb68ef, 0x3ffbf20b, 0x400c6abf, 0x400fbf56, 0x4007aaa4,
    0x3fe9cae1, 0x3fb239f0, 0x3f5b3242, 0x3e88278d, 0xbea6d020, 0xbf5db926, 0xbfa74219, 0xbfcddb10,
    0xbfdf34d9, 0xbfd9c2a0, 0xbfbe00ed, 0xbf8e6a90, 0xbf1e7f6e, 0xbd449808, 0x3f0cbcb4, 0x3f8f742c,
];

fn hex(id: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    id.iter().fold(String::with_capacity(64), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// A private copy of the fixture store (a read may not change the fixture)
fn store_copy() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    std::fs::create_dir_all(&dir).expect("create");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/store_det_math_0_4");
    for e in std::fs::read_dir(&fixture).expect("fixture") {
        let e = e.expect("entry");
        if e.file_name() != "README.md" {
            std::fs::copy(e.path(), dir.join(e.file_name())).expect("copy");
        }
    }
    (tmp, dir)
}

fn open(dir: &Path, policy: ResidualPolicy, use_mmap: bool) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_wal: false,
        use_mmap,
        residual_policy: policy,
        ..StorageConfig::default()
    })
    .expect("open the fixture store")
}

fn mismatch(e: &std::io::Error) -> ResidualMismatch {
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
    match e
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<ResidualError>())
    {
        Some(ResidualError::Mismatch(m)) => m.clone(),
        other => panic!("the error does not carry a ResidualError::Mismatch: {other:?} ({e})"),
    }
}

/// The fixture only says something when this build's arithmetic differs
#[test]
fn this_build_computes_with_another_arithmetic_than_the_fixture() {
    assert_ne!(
        hex(&alice_db::SEMANTICS_ID),
        OLD_ID,
        "the fixture cannot test a mismatch on its own arithmetic"
    );
}

#[test]
fn a_point_read_refuses_and_names_the_data() {
    let (_tmp, dir) = store_copy();
    for use_mmap in [true, false] {
        let db = open(&dir, ResidualPolicy::default(), use_mmap);
        for k in [0_i64, 31, 63] {
            let err = db
                .get(k)
                .expect_err("a residual of another arithmetic must not read back as a value");
            let m = mismatch(&err);
            assert_eq!(hex(&m.stored), OLD_ID, "mmap {use_mmap} key {k}");
            assert_eq!(m.running, alice_db::SEMANTICS_ID);
            assert_eq!(
                (m.start_time, m.end_time),
                (0, 63),
                "the key range of the segment"
            );
            assert_eq!(
                m.series.as_deref(),
                Some(dir.as_path()),
                "the store the segment belongs to"
            );
            let text = err.to_string();
            assert!(
                text.contains(OLD_ID) && text.contains(&dir.display().to_string()),
                "{text}"
            );
        }
        assert_eq!(
            db.get(64).expect("outside the segment"),
            None,
            "a key no segment holds is not an error"
        );
    }
}

#[test]
fn a_range_read_refuses_as_a_whole() {
    let (_tmp, dir) = store_copy();
    for use_mmap in [true, false] {
        let db = open(&dir, ResidualPolicy::default(), use_mmap);
        for (s, e) in [(0, 63), (10, 20), (i64::MIN, i64::MAX), (60, 100)] {
            let err = db
                .scan(s, e)
                .expect_err("no partial or model-valued result");
            assert_eq!(hex(&mismatch(&err).stored), OLD_ID);
        }
        assert_eq!(db.scan(64, 100).expect("no segment in range"), Vec::new());
    }
}

/// The model, read on purpose: the call succeeds, and what it returns is the
/// law's approximation, not one of the stored values
#[test]
fn the_model_only_policy_reads_the_model_knowingly() {
    let (_tmp, dir) = store_copy();
    let view = SegmentView::open(dir.join("seg_1.rkyv")).expect("segment");
    for use_mmap in [true, false] {
        let db = open(&dir, ResidualPolicy::ModelOnly, use_mmap);
        let scanned = db.scan(0, 63).expect("model-only reads succeed");
        assert_eq!(scanned.len(), 64);
        let mut differs = 0;
        for (k, v) in scanned {
            let model = view.query_point_model_only(k).expect("in the segment");
            assert_eq!(v.to_bits(), model.to_bits(), "key {k}: the model value");
            assert_eq!(
                db.get(k).expect("read").map(f32::to_bits),
                Some(model.to_bits())
            );
            differs += usize::from(v.to_bits() != ORIGINAL[usize::try_from(k).expect("key")]);
        }
        assert_eq!(
            differs, 64,
            "the model is not the stored data (else this test would not show the difference)"
        );
    }
}

#[test]
fn the_status_reports_the_mismatch_without_a_read_failing() {
    let (_tmp, dir) = store_copy();
    let view = SegmentView::open(dir.join("seg_1.rkyv")).expect("segment");
    match view.residual_status() {
        ResidualStatus::Mismatch { stored, running } => {
            assert_eq!(hex(&stored), OLD_ID);
            assert_eq!(running, alice_db::SEMANTICS_ID);
        }
        other => panic!("expected a mismatch, got {other:?}"),
    }
}

/// A newer write of every key the old segment holds answers instead of it, so
/// the old residual is never used and the read succeeds; a key only the old
/// segment answers still fails the read
#[test]
fn a_superseded_segment_does_not_fail_a_read_and_a_live_one_does() {
    let (_tmp, dir) = store_copy();
    let db = open(&dir, ResidualPolicy::default(), true);
    for k in 0..32_i64 {
        db.put(k, k as f32).expect("put");
    }
    db.flush().expect("flush");
    let low = db.scan(0, 31).expect("every key here has a newer write");
    assert_eq!(low.len(), 32);
    assert!(low
        .iter()
        .all(|&(k, v)| v.to_bits() == (k as f32).to_bits()));
    assert_eq!(
        db.get(5).expect("newer write").map(f32::to_bits),
        Some(5.0_f32.to_bits())
    );
    assert!(
        db.scan(0, 63).is_err(),
        "keys 32..=63 are still answered only by the old segment"
    );
    assert!(db.get(40).is_err());
}

/// Exact series live in the blob store and never touch a residual: a store
/// holding a mismatched segment reads its series back bit for bit
#[test]
fn exact_series_are_not_affected() {
    let (_tmp, dir) = store_copy();
    let db = open(&dir, ResidualPolicy::default(), true);
    let s = db.series("m").expect("series");
    let values = [0.1_f32, -2.5e-38, f32::MAX, 3.0];
    for (k, &v) in values.iter().enumerate() {
        s.put_f32(i64::try_from(k).expect("key") * 1_000_003, v)
            .expect("put");
    }
    for (k, &v) in values.iter().enumerate() {
        let got = s
            .get_f32(i64::try_from(k).expect("key") * 1_000_003)
            .expect("get")
            .expect("stored");
        assert_eq!(got.to_bits(), v.to_bits());
    }
    assert!(
        db.get(0).is_err(),
        "the model store in the same directory still refuses"
    );
}

/// Damaged or hostile residual blobs: a residual that does not decode to one
/// correction per point is refused as `Corrupt` (never the model presented as
/// the stored value), `residual_status()` reports it, the model-only read
/// still returns the model, and none of it panics
#[test]
fn damaged_residual_blobs_are_refused_as_corrupt_without_panicking() {
    use alice_db::segment::DataSegment;
    let n = 9_usize;
    let model = alice_db::model::ModelType::Constant { value: 1.5 };
    let id = alice_db::SEMANTICS_ID;
    let mut foreign = id;
    foreign[0] ^= 1;
    let with_id = |magic: u8, id: &[u8; 32], tail: &[u8]| {
        let mut v = vec![magic];
        v.extend_from_slice(id);
        v.extend_from_slice(tail);
        v
    };
    let corrupt: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("magic 6 alone", vec![6]),
        ("magic 7, short identifier", vec![7, 1, 2, 3]),
        ("magic 7, identifier, no data", with_id(7, &id, &[])),
        ("magic 6, identifier, not lzma", with_id(6, &id, &[0xff; 7])),
        ("magic 7, identifier, 3 bytes", with_id(7, &id, &[1, 2, 3])),
        ("unknown magic", vec![42, 1, 2, 3, 4, 5]),
        ("legacy magic, garbage", vec![0, 9, 9]),
    ];
    let foreign_garbage = (
        "foreign identifier, garbage",
        with_id(6, &foreign, &[0xff; 5]),
    );
    let mut seen = 0;
    for (name, blob) in corrupt.into_iter().chain(std::iter::once(foreign_garbage)) {
        let seg =
            DataSegment::new(3, 0, (n - 1) as i64, model.clone(), n, n * 4, 0).with_residual(blob);
        let view = SegmentView::from_vec(seg.to_rkyv_bytes().expect("rkyv")).expect("view");
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (
                seg.query_point_model_only(3),
                view.query_range_model_only(0, 8).len(),
                seg.residual_status(),
                view.residual_status(),
                seg.query_point(3),
                view.query_range(0, 8),
            )
        }));
        let (model_point, model_range, seg_status, view_status, point, range) =
            outcome.unwrap_or_else(|_| panic!("{name}: a read panicked"));
        assert_eq!(
            (model_point, model_range),
            (Some(1.5), n),
            "{name}: model-only"
        );
        assert_eq!(seg_status, view_status, "{name}: both paths report alike");
        if name.starts_with("foreign") {
            assert!(
                matches!(seg_status, ResidualStatus::Mismatch { .. }),
                "{name}: {seg_status:?}"
            );
            assert!(
                matches!(point, Err(ResidualError::Mismatch(_))),
                "{name}: {point:?}"
            );
        } else {
            let ResidualStatus::Corrupt {
                needed_bytes,
                found_bytes,
            } = seg_status
            else {
                panic!("{name}: expected Corrupt, got {seg_status:?}");
            };
            assert_eq!(needed_bytes, n * 4, "{name}");
            assert!(found_bytes < needed_bytes, "{name}");
            match point {
                Err(ResidualError::Corrupt(c)) => {
                    assert_eq!(
                        (c.segment_id, c.start_time, c.end_time),
                        (3, 0, (n - 1) as i64),
                        "{name}"
                    );
                    assert_eq!(
                        (c.needed_bytes, c.found_bytes),
                        (needed_bytes, found_bytes),
                        "{name}"
                    );
                }
                other => panic!("{name}: expected Err(Corrupt), got {other:?}"),
            }
            assert!(
                matches!(range, Err(ResidualError::Corrupt(_))),
                "{name}: range {range:?}"
            );
        }
        seen += 1;
    }
    assert_eq!(seen, 9);
}

/// The plain way to open a store refuses: refusing is the default of the
/// store's configuration, not only of the policy type
#[test]
fn opening_with_the_defaults_refuses() {
    assert_eq!(
        StorageConfig::default().residual_policy,
        ResidualPolicy::Refuse
    );
    let (_tmp, dir) = store_copy();
    let db = AliceDB::open(&dir).expect("open with the defaults");
    let err = db.get(10).expect_err("the default read refuses");
    assert_eq!(hex(&mismatch(&err).stored), OLD_ID);
    assert!(db.scan(0, 63).is_err());
}
