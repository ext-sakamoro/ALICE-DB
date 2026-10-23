//! A canonical text of a `DataSegment`: every field, floats as their exact bits. Written without rkyv, so the
//! values recorded from one decoder can be compared with another.

use alice_db::model::{DataType, ModelType};
use alice_db::segment::DataSegment;
use std::fmt::Write as _;

fn f64s(v: f64) -> String {
    format!("{:#018x}", v.to_bits())
}

fn f32s(v: f32) -> String {
    format!("{:#010x}", v.to_bits())
}

fn dtype(d: DataType) -> &'static str {
    match d {
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::UInt8 => "UInt8",
    }
}

fn hex(b: &[u8]) -> String {
    let mut o = String::with_capacity(2 * b.len());
    for x in b {
        let _ = write!(o, "{x:02x}");
    }
    o
}

fn model_text(o: &mut String, model: &ModelType) {
    match model {
        ModelType::Polynomial {
            coefficients,
            degree,
            fit_error,
        } => {
            let c: Vec<String> = coefficients.iter().map(|&v| f64s(v)).collect();
            let _ = writeln!(
                o,
                "model Polynomial coefficients [{}] degree {degree} fit_error {}",
                c.join(" "),
                f64s(*fit_error)
            );
        }
        ModelType::Fourier {
            coefficients,
            dc_offset,
            sample_count,
        } => {
            let c: Vec<String> = coefficients
                .iter()
                .map(|&(k, a, b)| format!("({k} {} {})", f32s(a), f32s(b)))
                .collect();
            let _ = writeln!(
                o,
                "model Fourier coefficients [{}] dc_offset {} sample_count {sample_count}",
                c.join(" "),
                f32s(*dc_offset)
            );
        }
        ModelType::SineWave {
            frequency,
            amplitude,
            phase,
            offset,
        } => {
            let _ = writeln!(
                o,
                "model SineWave {} {} {} {}",
                f32s(*frequency),
                f32s(*amplitude),
                f32s(*phase),
                f32s(*offset)
            );
        }
        ModelType::MultiSine {
            components,
            dc_offset,
        } => {
            let c: Vec<String> = components
                .iter()
                .map(|&(a, b, c)| format!("({} {} {})", f32s(a), f32s(b), f32s(c)))
                .collect();
            let _ = writeln!(
                o,
                "model MultiSine components [{}] dc_offset {}",
                c.join(" "),
                f32s(*dc_offset)
            );
        }
        ModelType::PerlinNoise {
            seed,
            scale,
            octaves,
            persistence,
            lacunarity,
        } => {
            let _ = writeln!(*o, "model PerlinNoise seed {seed} scale {} octaves {octaves} persistence {} lacunarity {}", f32s(*scale), f32s(*persistence), f32s(*lacunarity));
        }
        ModelType::Constant { value } => {
            let _ = writeln!(*o, "model Constant {}", f64s(*value));
        }
        ModelType::Linear {
            start_value,
            end_value,
        } => {
            let _ = writeln!(
                o,
                "model Linear {} {}",
                f64s(*start_value),
                f64s(*end_value)
            );
        }
        ModelType::RawLzma {
            compressed_data,
            dtype: d,
            original_size,
        } => {
            let _ = writeln!(
                o,
                "model RawLzma compressed_data {} dtype {} original_size {original_size}",
                hex(compressed_data),
                dtype(*d)
            );
        }
    }
}

/// One line per field, in declaration order.
#[must_use]
pub fn segment_text(s: &DataSegment) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "start_time {}", s.start_time);
    let _ = writeln!(o, "end_time {}", s.end_time);
    model_text(&mut o, &s.model);
    match &s.residual_blob {
        None => {
            let _ = writeln!(o, "residual_blob none");
        }
        Some(b) => {
            let _ = writeln!(o, "residual_blob {}", hex(b));
        }
    }
    let m = &s.metadata;
    let _ = writeln!(o, "metadata id {} created_at {} point_count {} original_size {} model_size {} compression_ratio {}",
        m.id, m.created_at, m.point_count, m.original_size, m.model_size, f64s(m.compression_ratio));
    o
}
