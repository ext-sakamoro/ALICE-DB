//! Reading a store whose lossless residual carries no arithmetic identifier
//! (`tests/fixtures/store_pre_0_3_0`: written by alice-db 0.3.0-beta.1, before
//! residuals were stamped)
//!
//! Such a residual stores `original ^ model` under an arithmetic nothing records.
//! Applied under this build's arithmetic it may give a value that is neither the
//! original nor the model, and nothing can tell. The default read refuses it with
//! [`ResidualError::Unpinned`] naming the data; [`ResidualPolicy::AcceptUnpinned`]
//! applies it, and [`AliceDB::get_with_status`] / [`AliceDB::scan_with_status`]
//! report [`ReadStatus::Unpinned`] (ALICE-Law G8: the read carries its status).

#![cfg(feature = "fs")]
// f32 bit patterns are carried as hex
#![allow(clippy::unreadable_literal, clippy::cast_possible_wrap)]

use alice_db::segment::{ResidualStatus, SegmentView};
use alice_db::{AliceDB, ReadStatus, ResidualError, ResidualPolicy, StorageConfig};
use std::path::{Path, PathBuf};

/// The 64 values the store was written with, as f32 bits: `(2 sin(0.3 k + 0.4) + 0.25) as f32`
/// evaluated in f64 by the writer (its platform libm)
const ORIGINAL: [u32; 64] = [
    0x3f83b0ec, 0x3fc4eb73, 0x3ff76aa4, 0x400b55e0, 0x400ff207, 0x4009205c, 0x3feef99f, 0x3fb93578,
    0x3f6b8391, 0x3eaa941f, 0xbe85ac91, 0xbf4f46af, 0xbfa1bdcf, 0xbfca89da, 0xbfde6289, 0xbfdb8204,
    0xbfc22a16, 0xbf949e4a, 0xbf2de056, 0xbdea233b, 0x3ef758ad, 0x3f87a43f, 0x3fc83046, 0x3ff9b637,
    0x400be4cb, 0x400fdd4f, 0x400869dc, 0x3fec69a7, 0x3fb5bd22, 0x3f6360dc, 0x3e995ebb, 0xbe96486b,
    0xbf5689ef, 0xbfa4870f, 0xbfcc3b03, 0xbfded4e9, 0xbfdaab64, 0xbfc01da2, 0xbf918adc, 0xbf263831,
    0xbda65090, 0x3f0436fd, 0x3f8b9011, 0x3fcb68ed, 0x3ffbf207, 0x400c6abe, 0x400fbf56, 0x4007aaa4,
    0x3fe9cae3, 0x3fb239f5, 0x3f5b3254, 0x3e882780, 0xbea6d021, 0xbf5db907, 0xbfa74215, 0xbfcddb12,
    0xbfdf34d8, 0xbfd9c2a0, 0xbfbe00f6, 0xbf8e6a94, 0xbf1e7f63, 0xbd44999b, 0x3f0cbcb5, 0x3f8f741b,
];

/// A private copy of the fixture store (a read may not change the fixture)
fn store_copy() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    std::fs::create_dir_all(&dir).expect("create");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/store_pre_0_3_0");
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

fn unpinned(e: &std::io::Error) -> &alice_db::ResidualUnpinned {
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
    match e
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<ResidualError>())
    {
        Some(ResidualError::Unpinned(u)) => u,
        other => panic!("the error does not carry a ResidualError::Unpinned: {other:?} ({e})"),
    }
}

/// The fixture says something only while its residual really has no identifier
#[test]
fn the_fixture_residual_carries_no_identifier() {
    let view = SegmentView::open(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/store_pre_0_3_0/seg_1.rkyv"),
    )
    .expect("the 0.3.0-beta.1 segment loads");
    assert_eq!(view.residual_status(), ResidualStatus::Unpinned);
}

#[test]
fn the_default_read_refuses_a_residual_without_an_identifier() {
    let (_tmp, dir) = store_copy();
    for use_mmap in [true, false] {
        let db = open(&dir, ResidualPolicy::default(), use_mmap);
        for k in [0_i64, 31, 63] {
            for err in [
                db.get(k).expect_err("get must refuse"),
                db.get_with_status(k)
                    .expect_err("get_with_status must refuse"),
            ] {
                let u = unpinned(&err);
                assert_eq!(
                    (u.start_time, u.end_time),
                    (0, 63),
                    "mmap {use_mmap} key {k}"
                );
                assert_eq!(u.series.as_deref(), Some(dir.as_path()));
                let text = err.to_string();
                assert!(text.contains("AcceptUnpinned"), "{text}");
            }
        }
        for err in [
            db.scan(0, 63).expect_err("scan must refuse"),
            db.scan(10, 20).expect_err("a part of the segment too"),
        ] {
            unpinned(&err);
        }
        unpinned(
            &db.scan_with_status(0, 63)
                .expect_err("scan_with_status must refuse"),
        );
        assert_eq!(db.get(64).expect("outside the segment"), None);
    }
}

#[test]
fn accept_unpinned_applies_it_and_says_so() {
    let (_tmp, dir) = store_copy();
    for use_mmap in [true, false] {
        let db = open(&dir, ResidualPolicy::AcceptUnpinned, use_mmap);
        let mut compared = 0_usize;
        for (k, &want) in ORIGINAL.iter().enumerate() {
            let (got, status) = db
                .get_with_status(k as i64)
                .expect("accepted")
                .expect("a key the segment holds");
            assert_eq!(status, ReadStatus::Unpinned, "mmap {use_mmap} key {k}");
            assert_eq!(got.to_bits(), want, "mmap {use_mmap} key {k}");
            assert_eq!(db.get(k as i64).expect("accepted"), Some(got));
            compared += 1;
        }
        let (points, status) = db.scan_with_status(0, 63).expect("accepted");
        assert_eq!(status, ReadStatus::Unpinned);
        assert_eq!(points.len(), ORIGINAL.len());
        for (i, &(t, v)) in points.iter().enumerate() {
            assert_eq!(t, i as i64);
            assert_eq!(v.to_bits(), ORIGINAL[i], "scan key {i}");
            compared += 1;
        }
        assert_eq!(compared, 128, "compared {compared}");
    }
}

/// A write after opening is current; a range that touches both reports the weaker
#[test]
fn a_range_reports_the_weakest_status_it_touched() {
    let (_tmp, dir) = store_copy();
    let db = open(&dir, ResidualPolicy::AcceptUnpinned, false);
    db.put(100, 1.5).expect("put");
    let (v, status) = db.get_with_status(100).expect("read").expect("present");
    assert_eq!((v, status), (1.5, ReadStatus::Current));
    let (points, status) = db.scan_with_status(100, 100).expect("read");
    assert_eq!((points.len(), status), (1, ReadStatus::Current));
    let (points, status) = db.scan_with_status(60, 100).expect("read");
    assert_eq!(points.len(), 5);
    assert_eq!(
        status,
        ReadStatus::Unpinned,
        "the weaker of Current and Unpinned"
    );
}

/// `ModelOnly` returns the model knowingly, with that status
#[test]
fn model_only_reports_model_only() {
    let (_tmp, dir) = store_copy();
    let db = open(&dir, ResidualPolicy::ModelOnly, false);
    let (_, status) = db.get_with_status(5).expect("read").expect("present");
    assert_eq!(status, ReadStatus::ModelOnly);
    let (_, status) = db.scan_with_status(0, 63).expect("read");
    assert_eq!(status, ReadStatus::ModelOnly);
}
