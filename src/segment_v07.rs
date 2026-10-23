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

// The decoder against the rkyv 0.7 reader it replaces, while that reader is still in the build.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ArchivedModelType, DataType, ModelType};
    use crate::segment::{
        ArchivedDataSegment, ArchivedSegmentMetadata, DataSegment, SegmentMetadata,
    };
    use std::mem::{offset_of, size_of};

    #[test]
    fn the_offsets_are_the_rkyv_0_7_layout() {
        assert_eq!(size_of::<ArchivedDataSegment>(), ROOT_SIZE);
        assert_eq!(offset_of!(ArchivedDataSegment, model), SEG_MODEL);
        assert_eq!(offset_of!(ArchivedDataSegment, start_time), SEG_START_TIME);
        assert_eq!(offset_of!(ArchivedDataSegment, end_time), SEG_END_TIME);
        assert_eq!(offset_of!(ArchivedDataSegment, metadata), SEG_METADATA);
        assert_eq!(offset_of!(ArchivedDataSegment, residual_blob), SEG_RESIDUAL);
        assert_eq!(offset_of!(ArchivedSegmentMetadata, id), META_ID);
        assert_eq!(
            offset_of!(ArchivedSegmentMetadata, created_at),
            META_CREATED_AT
        );
        assert_eq!(
            offset_of!(ArchivedSegmentMetadata, compression_ratio),
            META_COMPRESSION_RATIO
        );
        assert_eq!(
            offset_of!(ArchivedSegmentMetadata, point_count),
            META_POINT_COUNT
        );
        assert_eq!(
            offset_of!(ArchivedSegmentMetadata, original_size),
            META_ORIGINAL_SIZE
        );
        assert_eq!(
            offset_of!(ArchivedSegmentMetadata, model_size),
            META_MODEL_SIZE
        );
        assert_eq!(size_of::<ArchivedModelType>(), 32);
    }

    /// A small deterministic generator (no dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }
        fn f64(&mut self) -> f64 {
            match self.next() % 8 {
                0 => f64::NAN,
                1 => -0.0,
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                _ => f64::from_bits(self.next() << 11 | self.next() & 0x7ff),
            }
        }
        fn f32(&mut self) -> f32 {
            match self.next() % 8 {
                0 => f32::NAN,
                1 => -0.0,
                _ => f32::from_bits(self.next() as u32),
            }
        }
        fn len(&mut self) -> usize {
            (self.next() % 7) as usize
        }
        fn usize(&mut self) -> usize {
            (self.next() as u32) as usize
        }
    }

    fn model(r: &mut Lcg, variant: u64) -> ModelType {
        match variant {
            0 => ModelType::Polynomial {
                coefficients: (0..r.len()).map(|_| r.f64()).collect(),
                degree: r.usize(),
                fit_error: r.f64(),
            },
            1 => ModelType::Fourier {
                coefficients: (0..r.len())
                    .map(|_| (r.usize(), r.f32(), r.f32()))
                    .collect(),
                dc_offset: r.f32(),
                sample_count: r.usize(),
            },
            2 => ModelType::SineWave {
                frequency: r.f32(),
                amplitude: r.f32(),
                phase: r.f32(),
                offset: r.f32(),
            },
            3 => ModelType::MultiSine {
                components: (0..r.len()).map(|_| (r.f32(), r.f32(), r.f32())).collect(),
                dc_offset: r.f32(),
            },
            4 => ModelType::PerlinNoise {
                seed: r.next(),
                scale: r.f32(),
                octaves: r.next() as u32,
                persistence: r.f32(),
                lacunarity: r.f32(),
            },
            5 => ModelType::Constant { value: r.f64() },
            6 => ModelType::Linear {
                start_value: r.f64(),
                end_value: r.f64(),
            },
            _ => ModelType::RawLzma {
                compressed_data: (0..r.len() * 3).map(|_| r.next() as u8).collect(),
                dtype: [
                    DataType::Float32,
                    DataType::Float64,
                    DataType::Int32,
                    DataType::Int64,
                    DataType::UInt8,
                ][(r.next() % 5) as usize],
                original_size: r.usize(),
            },
        }
    }

    fn segment(r: &mut Lcg, variant: u64) -> DataSegment {
        DataSegment {
            start_time: r.next() as i64,
            end_time: r.next() as i64,
            model: model(r, variant),
            residual_blob: if r.next() % 2 == 0 {
                None
            } else {
                Some((0..r.len() * 5).map(|_| r.next() as u8).collect())
            },
            metadata: SegmentMetadata {
                id: r.next(),
                created_at: r.next(),
                point_count: r.usize(),
                original_size: r.usize(),
                model_size: r.usize(),
                compression_ratio: r.f64(),
            },
        }
    }

    /// Bit-exact comparison through the serde form (floats compared by bits, NaN included).
    fn same(a: &DataSegment, b: &DataSegment) -> bool {
        format!("{a:?}") == format!("{b:?}") && bits(a) == bits(b)
    }

    fn bits(s: &DataSegment) -> Vec<u64> {
        let mut v = vec![s.metadata.compression_ratio.to_bits()];
        match &s.model {
            ModelType::Polynomial {
                coefficients,
                fit_error,
                ..
            } => {
                v.extend(coefficients.iter().map(|c| c.to_bits()));
                v.push(fit_error.to_bits());
            }
            ModelType::Fourier {
                coefficients,
                dc_offset,
                ..
            } => {
                v.extend(
                    coefficients
                        .iter()
                        .flat_map(|&(_, a, b)| [u64::from(a.to_bits()), u64::from(b.to_bits())]),
                );
                v.push(u64::from(dc_offset.to_bits()));
            }
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => {
                v.extend([frequency, amplitude, phase, offset].map(|x| u64::from(x.to_bits())));
            }
            ModelType::MultiSine {
                components,
                dc_offset,
            } => {
                v.extend(
                    components
                        .iter()
                        .flat_map(|&(a, b, c)| [a, b, c].map(|x| u64::from(x.to_bits()))),
                );
                v.push(u64::from(dc_offset.to_bits()));
            }
            ModelType::PerlinNoise {
                scale,
                persistence,
                lacunarity,
                ..
            } => {
                v.extend([scale, persistence, lacunarity].map(|x| u64::from(x.to_bits())));
            }
            ModelType::Constant { value } => v.push(value.to_bits()),
            ModelType::Linear {
                start_value,
                end_value,
            } => v.extend([start_value.to_bits(), end_value.to_bits()]),
            ModelType::RawLzma { .. } => {}
        }
        v
    }

    #[test]
    fn every_variant_decodes_as_the_rkyv_0_7_reader_does() {
        let mut r = Lcg(0x5eed);
        let mut compared = 0;
        for round in 0..400 {
            let seg = segment(&mut r, round % 8);
            let bytes = seg.to_rkyv_bytes().unwrap();
            let by_rkyv = DataSegment::from_stored_bytes(&bytes).unwrap();
            let by_decoder = decode(&bytes).unwrap();
            assert!(
                same(&by_rkyv, &by_decoder),
                "round {round}: {by_rkyv:?} != {by_decoder:?}"
            );
            compared += 1;
        }
        assert_eq!(compared, 400);
    }

    #[test]
    fn a_damaged_archive_is_an_error_not_a_panic() {
        let mut r = Lcg(7);
        for variant in 0..8 {
            let bytes = segment(&mut r, variant).to_rkyv_bytes().unwrap();
            for cut in 0..bytes.len() {
                let _ = decode(&bytes[cut..]);
                let _ = decode(&bytes[..cut]);
            }
            for i in 0..bytes.len() {
                let mut b = bytes.clone();
                b[i] ^= 0xff;
                let _ = decode(&b);
            }
        }
        assert!(decode(&[]).is_err());
        assert!(decode(&[0u8; ROOT_SIZE - 1]).is_err());
    }
}
