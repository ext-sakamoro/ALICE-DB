//! `point_count` sizes allocations when a segment is read, so it is never taken from the bytes
//! past [`MAX_SEGMENT_POINTS`], and a raw payload must decode to exactly `point_count` samples.

#![cfg(feature = "fs")]

use alice_db::model::{DataType, ModelType};
use alice_db::segment::{DataSegment, MAX_SEGMENT_POINTS};
use alice_db::{AliceDB, StorageConfig};
use std::io::ErrorKind;

fn line(points: usize) -> DataSegment {
    let model = ModelType::Polynomial {
        coefficients: vec![1.0, 0.5],
        degree: 1,
        fit_error: 0.0,
    };
    let end = i64::try_from(points).unwrap() - 1;
    DataSegment::new(1, 0, end, model, points, points * 4, 0)
}

fn raw(values: &[f32], points: usize) -> DataSegment {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut compressed = Vec::new();
    lzma_rs::lzma_compress(&mut std::io::Cursor::new(&bytes), &mut compressed).unwrap();
    raw_payload(compressed, points)
}

fn raw_payload(compressed: Vec<u8>, points: usize) -> DataSegment {
    let model = ModelType::RawLzma {
        compressed_data: compressed,
        dtype: DataType::Float32,
        original_size: points * 4,
    };
    let end = i64::try_from(points).unwrap() - 1;
    DataSegment::new(1, 0, end, model, points, points * 4, 0)
}

/// Both stored forms of `segment` read back, or the kind of the refusal.
fn read_back(segment: &DataSegment) -> [Result<(), ErrorKind>; 2] {
    let codec = DataSegment::from_bytes(&segment.to_bytes().unwrap());
    let stored = DataSegment::from_stored_bytes(&segment.to_rkyv_bytes().unwrap());
    [
        codec.map(|_| ()).map_err(|e| e.kind()),
        stored.map(|_| ()).map_err(|e| e.kind()),
    ]
}

#[test]
fn the_point_count_has_a_stated_limit() {
    assert_eq!(MAX_SEGMENT_POINTS, (1 << 30) / 24);
    assert_eq!(read_back(&line(MAX_SEGMENT_POINTS)), [Ok(()), Ok(())]);
    assert_eq!(
        read_back(&line(MAX_SEGMENT_POINTS + 1)),
        [Err(ErrorKind::InvalidData), Err(ErrorKind::InvalidData)]
    );
}

#[test]
fn a_raw_payload_that_does_not_decode_is_refused() {
    let segment = raw_payload(vec![0; 76], 4);
    assert_eq!(
        read_back(&segment),
        [Err(ErrorKind::InvalidData), Err(ErrorKind::InvalidData)]
    );
}

#[test]
fn a_raw_payload_of_another_length_is_refused() {
    let values = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(read_back(&raw(&values, 4)), [Ok(()), Ok(())]);
    assert_eq!(
        read_back(&raw(&values, 5)),
        [Err(ErrorKind::InvalidData), Err(ErrorKind::InvalidData)]
    );
    assert_eq!(
        read_back(&raw(&values, 3)),
        [Err(ErrorKind::InvalidData), Err(ErrorKind::InvalidData)]
    );
}

#[test]
fn a_memtable_that_could_seal_more_does_not_open() {
    let tmp = tempfile::tempdir().unwrap();
    let config = |capacity| StorageConfig {
        data_dir: tmp.path().to_path_buf(),
        memtable_capacity: capacity,
        ..StorageConfig::default()
    };
    assert!(AliceDB::with_config(config(MAX_SEGMENT_POINTS)).is_ok());
    let refused = AliceDB::with_config(config(MAX_SEGMENT_POINTS + 1))
        .err()
        .expect("refused");
    assert_eq!(refused.kind(), ErrorKind::InvalidInput);
}
