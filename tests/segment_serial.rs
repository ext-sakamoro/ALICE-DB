//! `DataSegment::to_bytes` / `from_bytes` are alice_serial format 1.
//!
//! The layout is written out here as a `Value` tree, independently of the crate's code: a segment
//! is `[kind, start_time, end_time, model, residual, metadata]`, kind `"alice-db/segment/v1"`,
//! residual `Null` or `Bytes`, metadata `[id, created_at, point_count, original_size, model_size,
//! compression_ratio]`, and a model is `[name, fields...]` with lists as packed arrays. Every
//! integer that is a size in the crate is a `U64`. Bytes written by older releases (bincode) are
//! refused by `from_bytes`; they are read only from a legacy data file.

#![allow(clippy::unreadable_literal)]

use alice_db::model::{DataType, ModelType};
use alice_db::segment::{DataSegment, SegmentMetadata};
use alice_serial::{encode, Value};

const KIND: &str = "alice-db/segment/v1";

fn segment(model: ModelType, residual: Option<Vec<u8>>) -> DataSegment {
    DataSegment {
        start_time: -5,
        end_time: 1_000,
        model,
        residual_blob: residual,
        metadata: SegmentMetadata {
            id: 7,
            created_at: 1_761_000_000_000,
            point_count: 9,
            original_size: 36,
            model_size: 20,
            compression_ratio: 1.8,
        },
    }
}

fn s(x: &str) -> Value {
    Value::Str(x.into())
}

fn expected_model(m: &ModelType) -> Value {
    Value::Array(match m {
        ModelType::Polynomial {
            coefficients,
            degree,
            fit_error,
        } => vec![
            s("polynomial"),
            Value::F64Array(coefficients.clone()),
            Value::U64(*degree as u64),
            Value::F64(*fit_error),
        ],
        ModelType::Fourier {
            coefficients,
            dc_offset,
            sample_count,
        } => vec![
            s("fourier"),
            Value::U64Array(coefficients.iter().map(|c| c.0 as u64).collect()),
            Value::F32Array(coefficients.iter().map(|c| c.1).collect()),
            Value::F32Array(coefficients.iter().map(|c| c.2).collect()),
            Value::F32(*dc_offset),
            Value::U64(*sample_count as u64),
        ],
        ModelType::SineWave {
            frequency,
            amplitude,
            phase,
            offset,
        } => vec![
            s("sine_wave"),
            Value::F32(*frequency),
            Value::F32(*amplitude),
            Value::F32(*phase),
            Value::F32(*offset),
        ],
        ModelType::MultiSine {
            components,
            dc_offset,
        } => vec![
            s("multi_sine"),
            Value::F32Array(components.iter().map(|c| c.0).collect()),
            Value::F32Array(components.iter().map(|c| c.1).collect()),
            Value::F32Array(components.iter().map(|c| c.2).collect()),
            Value::F32(*dc_offset),
        ],
        ModelType::PerlinNoise {
            seed,
            scale,
            octaves,
            persistence,
            lacunarity,
        } => vec![
            s("perlin_noise"),
            Value::U64(*seed),
            Value::F32(*scale),
            Value::U64(u64::from(*octaves)),
            Value::F32(*persistence),
            Value::F32(*lacunarity),
        ],
        ModelType::Constant { value } => vec![s("constant"), Value::F64(*value)],
        ModelType::Linear {
            start_value,
            end_value,
        } => vec![
            s("linear"),
            Value::F64(*start_value),
            Value::F64(*end_value),
        ],
        ModelType::RawLzma {
            compressed_data,
            dtype,
            original_size,
        } => vec![
            s("raw_lzma"),
            Value::Bytes(compressed_data.clone()),
            s(match dtype {
                DataType::Float32 => "f32",
                DataType::Float64 => "f64",
                DataType::Int32 => "i32",
                DataType::Int64 => "i64",
                DataType::UInt8 => "u8",
            }),
            Value::U64(*original_size as u64),
        ],
    })
}

fn expected(seg: &DataSegment) -> Vec<u8> {
    let m = &seg.metadata;
    encode(&Value::Array(vec![
        s(KIND),
        Value::I64(seg.start_time),
        Value::I64(seg.end_time),
        expected_model(&seg.model),
        seg.residual_blob.clone().map_or(Value::Null, Value::Bytes),
        Value::Array(vec![
            Value::U64(m.id),
            Value::U64(m.created_at),
            Value::U64(m.point_count as u64),
            Value::U64(m.original_size as u64),
            Value::U64(m.model_size as u64),
            Value::F64(m.compression_ratio),
        ]),
    ]))
    .expect("encode")
}

fn every_model() -> Vec<ModelType> {
    vec![
        ModelType::Polynomial {
            coefficients: vec![1.5, -0.25, f64::from_bits(0x7ff8_0000_0000_0001)],
            degree: 2,
            fit_error: 1e-9,
        },
        ModelType::Fourier {
            coefficients: vec![(1, 0.5, -0.5), (3, 0.125, 2.0)],
            dc_offset: 0.75,
            sample_count: 9,
        },
        ModelType::SineWave {
            frequency: 0.3,
            amplitude: 2.0,
            phase: 0.4,
            offset: -0.0,
        },
        ModelType::MultiSine {
            components: vec![(1.0, 0.5, 0.25), (2.0, 0.125, 0.0)],
            dc_offset: 1.0,
        },
        ModelType::PerlinNoise {
            seed: u64::MAX,
            scale: 0.5,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
        },
        ModelType::Constant { value: 3.25 },
        ModelType::Linear {
            start_value: -1.0,
            end_value: 1.0,
        },
        ModelType::RawLzma {
            compressed_data: vec![0x5d, 0, 0, 0x80, 0],
            dtype: DataType::Int64,
            original_size: 72,
        },
    ]
}

#[test]
fn every_model_is_written_in_the_stated_layout_and_read_back() {
    let mut compared = 0;
    for model in every_model() {
        for residual in [None, Some(vec![6, 1, 2, 3])] {
            let seg = segment(model.clone(), residual);
            let bytes = seg.to_bytes().expect("to_bytes");
            assert_eq!(bytes, expected(&seg), "{}", model.name());
            let back = DataSegment::from_bytes(&bytes).expect("from_bytes");
            assert_eq!(
                back.to_bytes().expect("to_bytes"),
                bytes,
                "{}",
                model.name()
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 16);
}

#[test]
fn the_bytes_of_one_segment_are_pinned() {
    let seg = segment(
        ModelType::SineWave {
            frequency: 0.3,
            amplitude: 2.0,
            phase: 0.4,
            offset: 0.25,
        },
        Some(vec![6, 1]),
    );
    let hex: String = expected(&seg).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, GOLDEN, "{hex}");
    assert_eq!(seg.to_bytes().expect("to_bytes"), expected(&seg));
}

/// alice_serial format 1 of the segment above, checked by hand against `alice_serial::FORMAT_SPEC`
const GOLDEN: &str =
    "414c53520100dd00000006db00000013616c6963652d64622f7365676d656e742f7631d3fffffffffffffffbd300000000000003e8dd00000005db0000000973696e655f77617665ca3e99999aca40000000ca3ecccccdca3e800000c6000000020601dd0000000607cf0000019a03c78a00092414cb3ffccccccccccccd";

/// The first fixture of `tests/legacy_segment_compat.rs`: bytes alice-db wrote with bincode
const OLD_BINCODE_SINE: &str = "00000000000000000800000000000000020000000000404000000040cdcccc3e0000803e012500000000000000030300000003000000030000000300000003000000030000000300000003000000030000000100000000000000c3b24919a10100000900000000000000240000000000000010000000000000000000000000000240";

fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2)
        .map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

#[test]
fn old_bincode_bytes_are_refused_by_from_bytes() {
    let e = DataSegment::from_bytes(&unhex(OLD_BINCODE_SINE)).expect_err("refused");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
    assert!(e.to_string().contains("legacy data file"), "{e}");
}

fn refused(v: &Value) -> std::io::Error {
    DataSegment::from_bytes(&encode(v).expect("encode")).expect_err("refused")
}

fn with_model(model: Value) -> Value {
    let seg = segment(ModelType::Constant { value: 1.0 }, None);
    let Value::Array(mut top) = alice_serial::decode(&expected(&seg)).expect("decode") else {
        unreachable!()
    };
    top[3] = model;
    Value::Array(top)
}

#[test]
fn malformed_segments_are_refused() {
    // another kind
    let seg = segment(ModelType::Constant { value: 1.0 }, None);
    let Value::Array(mut top) = alice_serial::decode(&expected(&seg)).expect("decode") else {
        unreachable!()
    };
    top[0] = s("alice-db/segment/v0");
    assert!(refused(&Value::Array(top.clone()))
        .to_string()
        .contains("kind"));
    // a missing field
    top[0] = s(KIND);
    top.pop();
    let _ = refused(&Value::Array(top));
    // an unknown model
    let _ = refused(&with_model(Value::Array(vec![
        s("spline"),
        Value::F64(1.0),
    ])));
    // a model field of the wrong type
    let _ = refused(&with_model(Value::Array(vec![
        s("constant"),
        Value::U64(1),
    ])));
    // a model with an extra field
    let _ = refused(&with_model(Value::Array(vec![
        s("constant"),
        Value::F64(1.0),
        Value::F64(2.0),
    ])));
    // Fourier columns of different lengths
    let _ = refused(&with_model(Value::Array(vec![
        s("fourier"),
        Value::U64Array(vec![1, 2]),
        Value::F32Array(vec![0.5]),
        Value::F32Array(vec![0.5, 0.25]),
        Value::F32(0.0),
        Value::U64(4),
    ])));
    // octaves past u32
    let _ = refused(&with_model(Value::Array(vec![
        s("perlin_noise"),
        Value::U64(1),
        Value::F32(1.0),
        Value::U64(u64::from(u32::MAX) + 1),
        Value::F32(0.5),
        Value::F32(2.0),
    ])));
    // an unknown data type
    let _ = refused(&with_model(Value::Array(vec![
        s("raw_lzma"),
        Value::Bytes(vec![]),
        s("f16"),
        Value::U64(0),
    ])));
}
