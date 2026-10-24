//! A segment that exists only in the legacy `data.alice` file (bytes alice-db wrote with
//! bincode, each prefixed by its `u64` LE length) is read on the data-file path and rewritten as
//! a stored `seg_<id>.rkyv` file. Malformed legacy bytes are refused with an error, not a panic,
//! and nothing is written for them.

#![cfg(feature = "fs")]
#![allow(clippy::unreadable_literal)]

use alice_db::model::ModelType;
use alice_db::segment::DataSegment;
use alice_db::{AliceDB, ResidualPolicy, StorageConfig};
use std::path::Path;

/// bincode bytes of a sine segment over keys `0..=8` (the first fixture of
/// `tests/legacy_segment_compat.rs`, written by alice-db `52ca813`)
const OLD_SINE: &str = "00000000000000000800000000000000020000000000404000000040cdcccc3e0000803e012500000000000000030300000003000000030000000300000003000000030000000300000003000000030000000100000000000000c3b24919a10100000900000000000000240000000000000010000000000000000000000000000240";

fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2)
        .map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

/// A store whose only segment (id 1, keys `start..=end`) is `bytes` in `data.alice`
fn legacy_store(dir: &Path, start: i64, end: i64, model: &str, bytes: &[u8]) {
    let mut index = Vec::new();
    index.extend_from_slice(&1u64.to_le_bytes());
    index.extend_from_slice(&1u64.to_le_bytes()); // id
    index.extend_from_slice(&start.to_le_bytes());
    index.extend_from_slice(&end.to_le_bytes());
    index.extend_from_slice(&0u64.to_le_bytes()); // offset in data.alice
    index.extend_from_slice(&(bytes.len() as u64 + 8).to_le_bytes()); // size
    index.extend_from_slice(&1.0f64.to_le_bytes()); // compression ratio
    index.extend_from_slice(&(model.len() as u32).to_le_bytes());
    index.extend_from_slice(model.as_bytes());
    std::fs::write(dir.join("index.alice"), index).expect("index");
    let mut data = (bytes.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(bytes);
    std::fs::write(dir.join("data.alice"), data).expect("data");
}

fn open(dir: &Path) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_wal: false,
        use_mmap: false,
        residual_policy: ResidualPolicy::AcceptUnpinned,
        ..StorageConfig::default()
    })
    .expect("open")
}

#[test]
fn an_old_segment_in_the_data_file_is_read_and_rewritten_as_a_stored_segment() {
    let tmp = tempfile::tempdir().expect("temp");
    legacy_store(tmp.path(), 0, 8, "SineWave", &unhex(OLD_SINE));
    let db = open(tmp.path());
    let values = db.scan(0, 8).expect("scan the legacy segment");
    assert_eq!(values.len(), 9);
    let stored = std::fs::read(tmp.path().join("seg_1.rkyv")).expect("rewritten as rkyv");
    let seg = DataSegment::from_stored_bytes(&stored).expect("stored segment");
    assert_eq!((seg.start_time, seg.end_time), (0, 8));
    assert_eq!(seg.metadata.point_count, 9);
    assert!(
        matches!(seg.model, ModelType::SineWave { .. }),
        "{:?}",
        seg.model
    );
    assert_eq!(seg.residual_blob.as_ref().map(Vec::len), Some(37));
}

fn refused_on_the_data_file_path(bytes: &[u8]) -> std::io::Error {
    let tmp = tempfile::tempdir().expect("temp");
    legacy_store(tmp.path(), 0, 8, "SineWave", bytes);
    let db = open(tmp.path());
    let e = db
        .scan(0, 8)
        .expect_err("malformed legacy bytes must be refused");
    assert!(
        !tmp.path().join("seg_1.rkyv").exists(),
        "nothing is written for refused bytes"
    );
    e
}

#[test]
fn truncated_legacy_bytes_are_refused() {
    let full = unhex(OLD_SINE);
    for cut in [0, 1, 16, 20, full.len() / 2, full.len() - 1] {
        let e = refused_on_the_data_file_path(&full[..cut]);
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "cut {cut}: {e}");
    }
}

#[test]
fn an_oversize_length_is_refused_before_it_sizes_anything() {
    // the residual length (offset 37..45) claims far more bytes than remain
    let mut b = unhex(OLD_SINE);
    b[37..45].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
    let e = refused_on_the_data_file_path(&b);
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
}

#[test]
fn a_bad_model_tag_is_refused() {
    let mut b = unhex(OLD_SINE);
    b[16..20].copy_from_slice(&99u32.to_le_bytes());
    let e = refused_on_the_data_file_path(&b);
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
}

#[test]
fn trailing_bytes_after_a_legacy_segment_are_refused() {
    let mut b = unhex(OLD_SINE);
    b.push(0);
    let e = refused_on_the_data_file_path(&b);
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
}
