//! [`DataSegment`] as alice_serial format 1: the bytes of [`DataSegment::to_bytes`].
//!
//! A segment is the array `[kind, start_time, end_time, model, residual, metadata]`:
//!
//! - kind: the string [`SEGMENT_KIND`]
//! - start_time / end_time: `I64`
//! - model: `[name, fields...]`, lists as packed arrays (Fourier and multi-sine as one packed array
//!   per column), sizes as `U64`, data types as `"f32" / "f64" / "i32" / "i64" / "u8"`
//! - residual: `Null` or `Bytes`
//! - metadata: `[id, created_at, point_count, original_size, model_size, compression_ratio]`
//!
//! Decoding accepts exactly this shape: a different kind, a missing or extra field, a field of
//! another type, columns of different lengths, or a size that does not fit refuses the bytes.
//! The stored format of a segment file (an rkyv archive, [`crate::segment::SEGMENT_FORMAT`]) is
//! separate and not affected.

use std::io;

use alice_serial::{decode, encode, SerialError, Value};

use crate::model::{DataType, ModelType};
use crate::segment::{DataSegment, SegmentMetadata};

/// The kind string at the start of every encoded segment
pub const SEGMENT_KIND: &str = "alice-db/segment/v1";

fn invalid(what: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.into())
}

/// The alice_serial bytes of `seg`
pub(crate) fn to_bytes(seg: &DataSegment) -> io::Result<Vec<u8>> {
    let m = &seg.metadata;
    let value = Value::Array(vec![
        Value::Str(SEGMENT_KIND.into()),
        Value::I64(seg.start_time),
        Value::I64(seg.end_time),
        model_value(&seg.model),
        seg.residual_blob.clone().map_or(Value::Null, Value::Bytes),
        Value::Array(vec![
            Value::U64(m.id),
            Value::U64(m.created_at),
            Value::U64(m.point_count as u64),
            Value::U64(m.original_size as u64),
            Value::U64(m.model_size as u64),
            Value::F64(m.compression_ratio),
        ]),
    ]);
    encode(&value).map_err(|e| invalid(format!("segment encoding: {e:?}")))
}

/// The segment in `bytes`, which must be [`to_bytes`] output
pub(crate) fn from_bytes(bytes: &[u8]) -> io::Result<DataSegment> {
    let value = decode(bytes).map_err(|e| match e {
        SerialError::BadMagic | SerialError::BufferTooShort if !bytes.starts_with(b"ALSR") => {
            invalid(
                "segment bytes are not alice_serial format 1; segments written by alice-db before \
             this version (bincode) are read only from a legacy data file",
            )
        }
        e => invalid(format!("segment bytes: {e:?}")),
    })?;
    let [kind, start, end, model, residual, metadata] = fields::<6>(value, "segment")?;
    if text(kind, "segment kind")? != SEGMENT_KIND {
        return Err(invalid("segment kind is not alice-db/segment/v1"));
    }
    let [id, created_at, point_count, original_size, model_size, ratio] =
        fields::<6>(metadata, "segment metadata")?;
    Ok(DataSegment {
        start_time: int(start, "start_time")?,
        end_time: int(end, "end_time")?,
        model: model_from(model)?,
        residual_blob: match residual {
            Value::Null => None,
            Value::Bytes(b) => Some(b),
            _ => return Err(invalid("segment residual is neither null nor bytes")),
        },
        metadata: SegmentMetadata {
            id: uint(id, "id")?,
            created_at: uint(created_at, "created_at")?,
            point_count: size(point_count, "point_count")?,
            original_size: size(original_size, "original_size")?,
            model_size: size(model_size, "model_size")?,
            compression_ratio: f64v(ratio, "compression_ratio")?,
        },
    })
}

const fn dtype_name(d: DataType) -> &'static str {
    match d {
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::UInt8 => "u8",
    }
}

fn model_value(m: &ModelType) -> Value {
    let s = |x: &str| Value::Str(x.into());
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
            s(dtype_name(*dtype)),
            Value::U64(*original_size as u64),
        ],
    })
}

fn model_from(v: Value) -> io::Result<ModelType> {
    let Value::Array(items) = v else {
        return Err(invalid("segment model is not an array"));
    };
    let mut it = items.into_iter();
    let name = text(
        it.next()
            .ok_or_else(|| invalid("segment model has no name"))?,
        "model name",
    )?;
    let rest: Vec<Value> = it.collect();
    let model = match name.as_str() {
        "polynomial" => {
            let [c, d, e] = take::<3>(rest, "polynomial")?;
            ModelType::Polynomial {
                coefficients: f64s(c, "coefficients")?,
                degree: size(d, "degree")?,
                fit_error: f64v(e, "fit_error")?,
            }
        }
        "fourier" => {
            let [k, a, b, dc, n] = take::<5>(rest, "fourier")?;
            let (k, a, b) = (u64s(k, "indices")?, f32s(a, "re")?, f32s(b, "im")?);
            if k.len() != a.len() || k.len() != b.len() {
                return Err(invalid("fourier columns have different lengths"));
            }
            let mut coefficients = Vec::with_capacity(k.len());
            for ((k, a), b) in k.into_iter().zip(a).zip(b) {
                coefficients.push((to_usize(k, "fourier index")?, a, b));
            }
            ModelType::Fourier {
                coefficients,
                dc_offset: f32v(dc, "dc_offset")?,
                sample_count: size(n, "sample_count")?,
            }
        }
        "sine_wave" => {
            let [f, a, p, o] = take::<4>(rest, "sine_wave")?;
            ModelType::SineWave {
                frequency: f32v(f, "frequency")?,
                amplitude: f32v(a, "amplitude")?,
                phase: f32v(p, "phase")?,
                offset: f32v(o, "offset")?,
            }
        }
        "multi_sine" => {
            let [f, a, p, dc] = take::<4>(rest, "multi_sine")?;
            let (f, a, p) = (
                f32s(f, "frequency")?,
                f32s(a, "amplitude")?,
                f32s(p, "phase")?,
            );
            if f.len() != a.len() || f.len() != p.len() {
                return Err(invalid("multi_sine columns have different lengths"));
            }
            ModelType::MultiSine {
                components: f
                    .into_iter()
                    .zip(a)
                    .zip(p)
                    .map(|((f, a), p)| (f, a, p))
                    .collect(),
                dc_offset: f32v(dc, "dc_offset")?,
            }
        }
        "perlin_noise" => {
            let [seed, scale, octaves, persistence, lacunarity] = take::<5>(rest, "perlin_noise")?;
            ModelType::PerlinNoise {
                seed: uint(seed, "seed")?,
                scale: f32v(scale, "scale")?,
                octaves: u32::try_from(uint(octaves, "octaves")?)
                    .map_err(|_| invalid("perlin_noise octaves do not fit u32"))?,
                persistence: f32v(persistence, "persistence")?,
                lacunarity: f32v(lacunarity, "lacunarity")?,
            }
        }
        "constant" => {
            let [v] = take::<1>(rest, "constant")?;
            ModelType::Constant {
                value: f64v(v, "value")?,
            }
        }
        "linear" => {
            let [a, b] = take::<2>(rest, "linear")?;
            ModelType::Linear {
                start_value: f64v(a, "start_value")?,
                end_value: f64v(b, "end_value")?,
            }
        }
        "raw_lzma" => {
            let [data, dtype, n] = take::<3>(rest, "raw_lzma")?;
            let Value::Bytes(compressed_data) = data else {
                return Err(invalid("raw_lzma data is not bytes"));
            };
            let dtype = match text(dtype, "dtype")?.as_str() {
                "f32" => DataType::Float32,
                "f64" => DataType::Float64,
                "i32" => DataType::Int32,
                "i64" => DataType::Int64,
                "u8" => DataType::UInt8,
                other => return Err(invalid(format!("raw_lzma data type {other:?} is unknown"))),
            };
            ModelType::RawLzma {
                compressed_data,
                dtype,
                original_size: size(n, "original_size")?,
            }
        }
        other => return Err(invalid(format!("segment model {other:?} is unknown"))),
    };
    Ok(model)
}

fn fields<const N: usize>(v: Value, what: &str) -> io::Result<[Value; N]> {
    let Value::Array(items) = v else {
        return Err(invalid(format!("{what} is not an array")));
    };
    take::<N>(items, what)
}

fn take<const N: usize>(items: Vec<Value>, what: &str) -> io::Result<[Value; N]> {
    let n = items.len();
    <[Value; N]>::try_from(items).map_err(|_| invalid(format!("{what} has {n} fields, not {N}")))
}

fn text(v: Value, what: &str) -> io::Result<String> {
    match v {
        Value::Str(s) => Ok(s),
        _ => Err(invalid(format!("{what} is not a string"))),
    }
}

fn int(v: Value, what: &str) -> io::Result<i64> {
    match v {
        Value::I64(x) => Ok(x),
        _ => Err(invalid(format!("{what} is not an i64"))),
    }
}

fn uint(v: Value, what: &str) -> io::Result<u64> {
    match v {
        Value::U64(x) => Ok(x),
        _ => Err(invalid(format!("{what} is not a u64"))),
    }
}

fn to_usize(x: u64, what: &str) -> io::Result<usize> {
    usize::try_from(x).map_err(|_| invalid(format!("{what} does not fit usize")))
}

fn size(v: Value, what: &str) -> io::Result<usize> {
    to_usize(uint(v, what)?, what)
}

fn f32v(v: Value, what: &str) -> io::Result<f32> {
    match v {
        Value::F32(x) => Ok(x),
        _ => Err(invalid(format!("{what} is not an f32"))),
    }
}

fn f64v(v: Value, what: &str) -> io::Result<f64> {
    match v {
        Value::F64(x) => Ok(x),
        _ => Err(invalid(format!("{what} is not an f64"))),
    }
}

fn f32s(v: Value, what: &str) -> io::Result<Vec<f32>> {
    match v {
        Value::F32Array(x) => Ok(x),
        _ => Err(invalid(format!("{what} is not an f32 array"))),
    }
}

fn f64s(v: Value, what: &str) -> io::Result<Vec<f64>> {
    match v {
        Value::F64Array(x) => Ok(x),
        _ => Err(invalid(format!("{what} is not an f64 array"))),
    }
}

fn u64s(v: Value, what: &str) -> io::Result<Vec<u64>> {
    match v {
        Value::U64Array(x) => Ok(x),
        _ => Err(invalid(format!("{what} is not a u64 array"))),
    }
}
