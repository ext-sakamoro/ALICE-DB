//! Every model and data type as an older build really wrote it (bincode, see the fixture's
//! README), read the only way the crate reads it: from a legacy `data.alice` file.

#![cfg(feature = "fs")]
#![allow(clippy::unreadable_literal)]

use alice_db::model::{DataType, ModelType};
use alice_db::segment::{DataSegment, SegmentMetadata};

#[path = "common/legacy_data_file.rs"]
mod legacy_data_file;

const FIXTURE: &str = include_str!("fixtures/legacy_bincode/every_model.txt");

fn cases() -> Vec<(&'static str, ModelType, Option<Vec<u8>>)> {
    // nine samples 0..9 of each type, compressed the way the writer does
    let raw = |dtype: DataType| {
        let width = dtype.size();
        let bytes: Vec<u8> = (0..9u8)
            .flat_map(|i| {
                let mut b = vec![0u8; width];
                b[0] = i;
                b
            })
            .collect();
        let mut compressed = Vec::new();
        lzma_rs::lzma_compress(&mut std::io::Cursor::new(&bytes), &mut compressed).unwrap();
        ModelType::RawLzma {
            compressed_data: compressed,
            dtype,
            original_size: bytes.len(),
        }
    };
    vec![
        (
            "polynomial",
            ModelType::Polynomial {
                coefficients: vec![1.5, -2.25],
                degree: 1,
                fit_error: 0.125,
            },
            None,
        ),
        (
            "fourier",
            ModelType::Fourier {
                coefficients: vec![(3, 0.5, -0.25)],
                dc_offset: 1.5,
                sample_count: 16,
            },
            None,
        ),
        (
            "sine",
            ModelType::SineWave {
                frequency: 2.0,
                amplitude: 0.5,
                phase: 0.25,
                offset: -1.0,
            },
            None,
        ),
        (
            "multi_sine",
            ModelType::MultiSine {
                components: vec![(1.0, 0.5, 0.25), (2.0, 0.125, 0.0)],
                dc_offset: 3.0,
            },
            None,
        ),
        (
            "perlin",
            ModelType::PerlinNoise {
                seed: 7,
                scale: 0.5,
                octaves: 3,
                persistence: 0.25,
                lacunarity: 2.0,
            },
            None,
        ),
        ("constant", ModelType::Constant { value: 2.5 }, None),
        (
            "linear",
            ModelType::Linear {
                start_value: 1.0,
                end_value: 9.0,
            },
            None,
        ),
        ("raw_f32", raw(DataType::Float32), None),
        ("raw_f64", raw(DataType::Float64), None),
        ("raw_i32", raw(DataType::Int32), None),
        ("raw_i64", raw(DataType::Int64), None),
        ("raw_u8", raw(DataType::UInt8), None),
        (
            "constant_with_residual",
            ModelType::Constant { value: 2.5 },
            Some(vec![9, 8, 7]),
        ),
    ]
}

fn segment(model: ModelType, residual_blob: Option<Vec<u8>>) -> DataSegment {
    DataSegment {
        start_time: 0,
        end_time: 8,
        model,
        residual_blob,
        metadata: SegmentMetadata {
            id: 1,
            created_at: 2,
            point_count: 9,
            original_size: 36,
            model_size: 4,
            compression_ratio: 9.0,
        },
    }
}

fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2)
        .map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

#[test]
fn every_model_an_older_build_wrote_reads_back_the_same() {
    let mut compared = 0;
    let lines: Vec<&str> = FIXTURE.lines().collect();
    let cases = cases();
    assert_eq!(lines.len(), cases.len());
    for (line, (name, model, residual)) in lines.iter().zip(cases) {
        let (written_name, hex) = line.split_once(' ').expect("name and hex");
        assert_eq!(written_name, name);
        let read = legacy_data_file::read_through_data_file(&unhex(hex));
        let want = segment(model, residual);
        assert_eq!(
            read.to_bytes().expect("bytes"),
            want.to_bytes().expect("bytes"),
            "{name}"
        );
        compared += 1;
    }
    assert_eq!(compared, 13);
}

/// The rewritten store file holds each data type of the fixture under its own tag.
#[test]
fn each_data_type_keeps_its_tag() {
    let mut seen = Vec::new();
    for line in FIXTURE.lines().filter(|l| l.starts_with("raw_")) {
        let (_, hex) = line.split_once(' ').expect("name and hex");
        let read = legacy_data_file::read_through_data_file(&unhex(hex));
        let ModelType::RawLzma { dtype, .. } = read.model else {
            panic!("not a raw model: {line}");
        };
        seen.push(dtype);
    }
    assert_eq!(
        seen,
        [
            DataType::Float32,
            DataType::Float64,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8
        ]
    );
}
