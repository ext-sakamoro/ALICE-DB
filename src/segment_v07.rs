//! Reader for segment files written before the stored segment format had a header (stored
//! segment format 0): an rkyv 0.7 archive of [`DataSegment`] (default features: 32-bit relative
//! pointers and lengths, little-endian, no `strict` layout).
//!
//! rkyv 0.7 left the archived structs and `ArchivedVec` without a `repr`, so their field order is
//! the one rustc chose; the offsets below are what that archive layout put on disk (measured on
//! the rkyv 0.7.46 archived types). The archived enums are `#[repr(u8)]`, which fixes each
//! variant's layout: the tag byte, then the fields in declaration order with C alignment.
//! `tests/legacy_segment_values.rs` decodes every stored fixture and checks each field against
//! the values recorded by the rkyv 0.7 reader itself.
//!
//! The reader copies everything out into an owned [`DataSegment`]; it never casts the bytes to a
//! type, so it needs no alignment and every offset, length and tag is checked against the input.

use crate::model::{DataType, ModelType};
use crate::segment::{DataSegment, SegmentMetadata};
use std::io;

/// Size of the archived `DataSegment` (the root object, at the end of the archive).
const ROOT_SIZE: usize = 104;

// Archived `DataSegment` fields (offsets from the root).
const SEG_MODEL: usize = 0;
const SEG_START_TIME: usize = 32;
const SEG_END_TIME: usize = 40;
const SEG_METADATA: usize = 48;
const SEG_RESIDUAL: usize = 88;

// Archived `SegmentMetadata` fields (offsets from the metadata).
const META_ID: usize = 0;
const META_CREATED_AT: usize = 8;
const META_COMPRESSION_RATIO: usize = 16;
const META_POINT_COUNT: usize = 24;
const META_ORIGINAL_SIZE: usize = 28;
const META_MODEL_SIZE: usize = 32;

/// `ArchivedVec<T>`: a 32-bit offset relative to its own position, then the 32-bit length.
const VEC_PTR: usize = 0;
const VEC_LEN: usize = 4;

fn bad(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("stored segment (format 0): {what}"),
    )
}

/// Bounds-checked little-endian reads from the archive.
struct Archive<'a>(&'a [u8]);

impl Archive<'_> {
    fn bytes(&self, at: usize, n: usize) -> io::Result<&[u8]> {
        at.checked_add(n)
            .and_then(|end| self.0.get(at..end))
            .ok_or_else(|| bad("a field lies outside the file"))
    }

    fn u8(&self, at: usize) -> io::Result<u8> {
        Ok(self.bytes(at, 1)?[0])
    }

    fn u32(&self, at: usize) -> io::Result<u32> {
        let b = self.bytes(at, 4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&self, at: usize) -> io::Result<i32> {
        let b = self.bytes(at, 4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&self, at: usize) -> io::Result<u64> {
        let b = self.bytes(at, 8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn i64(&self, at: usize) -> io::Result<i64> {
        Ok(self.u64(at)? as i64)
    }

    fn f64(&self, at: usize) -> io::Result<f64> {
        Ok(f64::from_bits(self.u64(at)?))
    }

    fn f32(&self, at: usize) -> io::Result<f32> {
        Ok(f32::from_bits(self.u32(at)?))
    }

    fn usize(&self, at: usize) -> io::Result<usize> {
        Ok(self.u32(at)? as usize)
    }

    /// The (start, len) of the elements of the `ArchivedVec` at `at`, each `elem` bytes; the
    /// whole slice must lie in the file.
    fn vec(&self, at: usize, elem: usize) -> io::Result<(usize, usize)> {
        let offset = i64::from(self.i32(at + VEC_PTR)?);
        let len = self.usize(at + VEC_LEN)?;
        let start = i64::try_from(at)
            .ok()
            .and_then(|a| a.checked_add(offset))
            .and_then(|s| usize::try_from(s).ok())
            .ok_or_else(|| bad("a relative pointer points before the file"))?;
        let bytes = len
            .checked_mul(elem)
            .ok_or_else(|| bad("a length overflows"))?;
        self.bytes(start, bytes)?;
        Ok((start, len))
    }

    fn vec_f64(&self, at: usize) -> io::Result<Vec<f64>> {
        let (start, len) = self.vec(at, 8)?;
        (0..len).map(|i| self.f64(start + 8 * i)).collect()
    }

    fn vec_u8(&self, at: usize) -> io::Result<Vec<u8>> {
        let (start, len) = self.vec(at, 1)?;
        Ok(self.bytes(start, len)?.to_vec())
    }

    /// `Vec<(usize, f32, f32)>`: tuples of (u32, f32, f32), 12 bytes each.
    fn vec_fourier(&self, at: usize) -> io::Result<Vec<(usize, f32, f32)>> {
        let (start, len) = self.vec(at, 12)?;
        (0..len)
            .map(|i| {
                let e = start + 12 * i;
                Ok((self.usize(e)?, self.f32(e + 4)?, self.f32(e + 8)?))
            })
            .collect()
    }

    /// `Vec<(f32, f32, f32)>`, 12 bytes each.
    fn vec_f32x3(&self, at: usize) -> io::Result<Vec<(f32, f32, f32)>> {
        let (start, len) = self.vec(at, 12)?;
        (0..len)
            .map(|i| {
                let e = start + 12 * i;
                Ok((self.f32(e)?, self.f32(e + 4)?, self.f32(e + 8)?))
            })
            .collect()
    }

    fn data_type(&self, at: usize) -> io::Result<DataType> {
        Ok(match self.u8(at)? {
            0 => DataType::Float32,
            1 => DataType::Float64,
            2 => DataType::Int32,
            3 => DataType::Int64,
            4 => DataType::UInt8,
            _ => return Err(bad("unknown data type tag")),
        })
    }

    /// The archived `ModelType` at `at`: `#[repr(u8)]`, variants in declaration order.
    fn model(&self, at: usize) -> io::Result<ModelType> {
        Ok(match self.u8(at)? {
            0 => ModelType::Polynomial {
                coefficients: self.vec_f64(at + 4)?,
                degree: self.usize(at + 12)?,
                fit_error: self.f64(at + 16)?,
            },
            1 => ModelType::Fourier {
                coefficients: self.vec_fourier(at + 4)?,
                dc_offset: self.f32(at + 12)?,
                sample_count: self.usize(at + 16)?,
            },
            2 => ModelType::SineWave {
                frequency: self.f32(at + 4)?,
                amplitude: self.f32(at + 8)?,
                phase: self.f32(at + 12)?,
                offset: self.f32(at + 16)?,
            },
            3 => ModelType::MultiSine {
                components: self.vec_f32x3(at + 4)?,
                dc_offset: self.f32(at + 12)?,
            },
            4 => ModelType::PerlinNoise {
                seed: self.u64(at + 8)?,
                scale: self.f32(at + 16)?,
                octaves: self.u32(at + 20)?,
                persistence: self.f32(at + 24)?,
                lacunarity: self.f32(at + 28)?,
            },
            5 => ModelType::Constant {
                value: self.f64(at + 8)?,
            },
            6 => ModelType::Linear {
                start_value: self.f64(at + 8)?,
                end_value: self.f64(at + 16)?,
            },
            7 => ModelType::RawLzma {
                compressed_data: self.vec_u8(at + 4)?,
                dtype: self.data_type(at + 12)?,
                original_size: self.usize(at + 16)?,
            },
            _ => return Err(bad("unknown model type tag")),
        })
    }
}

/// Decode a stored segment file of format 0 (no header: an rkyv 0.7 archive).
///
/// # Errors
///
/// `InvalidData` when the file is shorter than the archived root, or any offset, length or tag in
/// it does not fit the layout.
pub(crate) fn decode(bytes: &[u8]) -> io::Result<DataSegment> {
    let a = Archive(bytes);
    let root = bytes
        .len()
        .checked_sub(ROOT_SIZE)
        .ok_or_else(|| bad("shorter than the archived segment"))?;
    let meta = root + SEG_METADATA;
    let residual = root + SEG_RESIDUAL;
    let residual_blob = match a.u8(residual)? {
        0 => None,
        1 => Some(a.vec_u8(residual + 4)?),
        _ => return Err(bad("unknown option tag of the residual")),
    };
    Ok(DataSegment {
        start_time: a.i64(root + SEG_START_TIME)?,
        end_time: a.i64(root + SEG_END_TIME)?,
        model: a.model(root + SEG_MODEL)?,
        residual_blob,
        metadata: SegmentMetadata {
            id: a.u64(meta + META_ID)?,
            created_at: a.u64(meta + META_CREATED_AT)?,
            point_count: a.usize(meta + META_POINT_COUNT)?,
            original_size: a.usize(meta + META_ORIGINAL_SIZE)?,
            model_size: a.usize(meta + META_MODEL_SIZE)?,
            compression_ratio: a.f64(meta + META_COMPRESSION_RATIO)?,
        },
    })
}

// The decoder's checks on damaged input. Its values are pinned by tests/legacy_segment_values.rs, whose
// fixtures (every store fixture and one segment per model variant) were recorded by the rkyv 0.7 reader
// itself, together with a cross-check of this decoder against that reader on 400 random segments of
// every variant while rkyv 0.7 was still in the build.
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixtures() -> Vec<Vec<u8>> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut out = Vec::new();
        for store in [
            "store_pre_0_3_0",
            "store_det_math_0_4",
            "store_db_0_5_0",
            "store_db_0_5_1",
            "store_db_0_6_0",
        ] {
            out.push(std::fs::read(root.join(store).join("seg_1.rkyv")).unwrap());
        }
        for name in [
            "polynomial",
            "fourier",
            "sine_wave",
            "multi_sine",
            "perlin_noise",
            "constant",
            "linear",
            "raw_lzma",
        ] {
            out.push(
                std::fs::read(
                    root.join("segment_format0_variants")
                        .join(format!("{name}.rkyv")),
                )
                .unwrap(),
            );
        }
        out
    }

    #[test]
    fn every_fixture_decodes() {
        let all = fixtures();
        assert_eq!(all.len(), 13);
        for bytes in &all {
            decode(bytes).unwrap();
        }
    }

    #[test]
    fn a_damaged_archive_is_an_error_not_a_panic() {
        for bytes in fixtures() {
            for cut in 0..bytes.len() {
                let _ = decode(&bytes[cut..]);
                let _ = decode(&bytes[..cut]);
            }
            for i in 0..bytes.len() {
                for flip in [0x01, 0x80, 0xff] {
                    let mut b = bytes.clone();
                    b[i] ^= flip;
                    let _ = decode(&b);
                }
            }
        }
        assert!(decode(&[]).is_err());
        assert!(decode(&[0u8; ROOT_SIZE - 1]).is_err());
    }

    #[test]
    fn a_tag_outside_the_layout_is_an_error() {
        let bytes = &fixtures()[0];
        let root = bytes.len() - ROOT_SIZE;
        let mut b = bytes.clone();
        b[root + SEG_MODEL] = 8;
        assert!(decode(&b).is_err(), "model tag 8");
        let mut b = bytes.clone();
        b[root + SEG_RESIDUAL] = 2;
        assert!(decode(&b).is_err(), "option tag 2");
    }
}
