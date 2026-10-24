//! The segments of a legacy `data.alice` file: the bincode 1 layout older releases wrote for a
//! [`DataSegment`], read by hand. It is used only on the data-file path
//! ([`crate::storage_engine`], which rewrites each segment as a stored rkyv file).
//!
//! The layout (bincode 1, fixed-width little-endian): `start_time i64`, `end_time i64`, the model
//! (`u32` variant index, then its fields in declaration order; a `Vec` is a `u64` count then the
//! items, `usize` is `u64`), the residual (`u8` 0 = none, 1 = then a `u64` length and the bytes),
//! then the metadata (`id u64`, `created_at u64`, `point_count`, `original_size`, `model_size` as
//! `u64`, `compression_ratio f64`). Every length is checked against the bytes that remain before
//! anything is sized by it, and bytes after the segment are refused.

use std::io;

use alice_serial::bounded::len_within;

use crate::model::{DataType, ModelType};
use crate::segment::{DataSegment, SegmentMetadata};

fn invalid(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("legacy segment: {what}"),
    )
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn remaining(&self) -> usize {
        self.data.len() - self.at
    }

    fn take<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let end = self
            .at
            .checked_add(N)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| invalid("ends early"))?;
        let mut out = [0u8; N];
        out.copy_from_slice(&self.data[self.at..end]);
        self.at = end;
        Ok(out)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take::<1>()?[0])
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.take()?))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take()?))
    }
    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }
    fn usize(&mut self) -> io::Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid("a size does not fit usize"))
    }

    /// A `Vec` count, checked against the bytes left at `elem_size` bytes per item
    fn count(&mut self, elem_size: usize) -> io::Result<usize> {
        let n = self.u64()?;
        len_within(n, elem_size, self.remaining())
            .map_err(|_| invalid("a length exceeds the bytes"))
    }

    fn bytes(&mut self) -> io::Result<Vec<u8>> {
        let n = self.count(1)?;
        let out = self.data[self.at..self.at + n].to_vec();
        self.at += n;
        Ok(out)
    }
}

/// The segment in `data`, the bincode 1 bytes of one legacy `data.alice` entry
pub(crate) fn decode_segment(data: &[u8]) -> io::Result<DataSegment> {
    let mut r = Reader { data, at: 0 };
    let start_time = r.i64()?;
    let end_time = r.i64()?;
    let model = model(&mut r)?;
    let residual_blob = match r.u8()? {
        0 => None,
        1 => Some(r.bytes()?),
        _ => return Err(invalid("the residual tag is neither 0 nor 1")),
    };
    let metadata = SegmentMetadata {
        id: r.u64()?,
        created_at: r.u64()?,
        point_count: r.usize()?,
        original_size: r.usize()?,
        model_size: r.usize()?,
        compression_ratio: r.f64()?,
    };
    if r.remaining() != 0 {
        return Err(invalid("bytes remain after the segment"));
    }
    Ok(DataSegment {
        start_time,
        end_time,
        model,
        residual_blob,
        metadata,
    })
}

fn model(r: &mut Reader<'_>) -> io::Result<ModelType> {
    Ok(match r.u32()? {
        0 => {
            let n = r.count(8)?;
            let mut coefficients = Vec::with_capacity(n);
            for _ in 0..n {
                coefficients.push(r.f64()?);
            }
            ModelType::Polynomial {
                coefficients,
                degree: r.usize()?,
                fit_error: r.f64()?,
            }
        }
        1 => {
            let n = r.count(16)?;
            let mut coefficients = Vec::with_capacity(n);
            for _ in 0..n {
                coefficients.push((r.usize()?, r.f32()?, r.f32()?));
            }
            ModelType::Fourier {
                coefficients,
                dc_offset: r.f32()?,
                sample_count: r.usize()?,
            }
        }
        2 => ModelType::SineWave {
            frequency: r.f32()?,
            amplitude: r.f32()?,
            phase: r.f32()?,
            offset: r.f32()?,
        },
        3 => {
            let n = r.count(12)?;
            let mut components = Vec::with_capacity(n);
            for _ in 0..n {
                components.push((r.f32()?, r.f32()?, r.f32()?));
            }
            ModelType::MultiSine {
                components,
                dc_offset: r.f32()?,
            }
        }
        4 => ModelType::PerlinNoise {
            seed: r.u64()?,
            scale: r.f32()?,
            octaves: r.u32()?,
            persistence: r.f32()?,
            lacunarity: r.f32()?,
        },
        5 => ModelType::Constant { value: r.f64()? },
        6 => ModelType::Linear {
            start_value: r.f64()?,
            end_value: r.f64()?,
        },
        7 => ModelType::RawLzma {
            compressed_data: r.bytes()?,
            dtype: match r.u32()? {
                0 => DataType::Float32,
                1 => DataType::Float64,
                2 => DataType::Int32,
                3 => DataType::Int64,
                4 => DataType::UInt8,
                _ => return Err(invalid("the data type tag is unknown")),
            },
            original_size: r.usize()?,
        },
        _ => return Err(invalid("the model tag is unknown")),
    })
}
