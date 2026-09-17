//! Data Segment (Model-Based `SSTable`)
//!
//! The core innovation of ALICE-DB: instead of storing raw data,
//! we store mathematical models that can regenerate data on-demand.
//!
//! # Query Performance
//!
//! Traditional DB: Read from disk → Decompress → Return
//! ALICE-DB: Load model coefficients → Compute f(x) → Return
//!
//! For point queries: O(1) computation, near-zero I/O
//! For range queries: O(n) computation where n = requested points
//!
//! # Zero-Copy I/O (Phase 1)
//!
//! `SegmentView` uses mmap + rkyv for true zero-copy access:
//! - No deserialization overhead
//! - OS handles page caching
//! - Direct pointer access to model coefficients
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::model::{ArchivedDataType, ArchivedModelType, DataType, ModelType};
use alice_core::generators;
use memmap2::Mmap;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;
use wide::f64x4;

/// The model laws — one function per model, shared by the in-memory point
/// query, the in-memory range query, the archived (mmap) point query, the
/// archived range query and `generate_all` (oracle: `tests/analytic_oracle.rs`).
///
/// Every law is a function of the **sample position** `i` (fractional,
/// `0 ..= n − 1`) and the sample count `n`, matching the conventions of the
/// fitters in `alice_core::generators`:
/// - polynomial: `Σ cⱼ·iʲ` (`fit_polynomial` maps its unit-x fit back to
///   integer indices)
/// - Fourier: `dc + Σ w(k)·mag/n · cos(2πk·i/n + phase)`, `w = 1` for DC and
///   Nyquist, `2` otherwise (`generate_from_coefficients`)
/// - sine / multi-sine: `offset + A·sin(2πf·i/n + phase)` (`generate_sine_wave`)
/// - linear: `start + (end − start)·i/(n − 1)`
///
/// History (2026-09-17): four hand-copied evaluators disagreed with these
/// laws and with each other — the point / range paths evaluated the
/// polynomial at `x = i/(n−1)` (coefficients are for `x = i`), used the raw
/// DFT magnitude for Fourier (n/2 × too large: a 1000-sample sine came back
/// with relative MSE 2.5e5) and `i/(n−1)` in the sine argument — while
/// `generate_all` (alice-zip) was right.  Lossless mode hid all of it
/// because the residual was computed against the same wrong value.
mod law {
    use std::f64::consts::PI;

    /// Fractional sample position of `timestamp` in a segment of `n` samples
    /// spanning `start..=end` (uniform spacing).
    #[inline(always)]
    pub(super) fn sample_pos(start: i64, end: i64, n: usize, timestamp: i64) -> f64 {
        let range = (end - start) as f64;
        if n <= 1 || range <= 0.0 {
            return 0.0;
        }
        (timestamp - start) as f64 / range * (n - 1) as f64
    }

    #[inline(always)]
    pub(super) fn polynomial_at(coefficients: impl DoubleEndedIterator<Item = f64>, i: f64) -> f32 {
        let mut result = 0.0f64;
        for c in coefficients.rev() {
            result = result.mul_add(i, c);
        }
        result as f32
    }

    #[inline(always)]
    pub(super) fn fourier_at(
        coefficients: impl Iterator<Item = (usize, f32, f32)>,
        dc_offset: f32,
        n: usize,
        i: f64,
    ) -> f32 {
        if n == 0 {
            return dc_offset;
        }
        let inv_n = 1.0 / n as f64;
        let mut sum = dc_offset as f64;
        for (k, mag, phase) in coefficients {
            if k >= n {
                continue;
            }
            let weight = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
            let angle = 2.0 * PI * k as f64 * i * inv_n + phase as f64;
            sum += weight * mag as f64 * inv_n * angle.cos();
        }
        sum as f32
    }

    #[inline(always)]
    pub(super) fn sine_at(
        frequency: f32,
        amplitude: f32,
        phase: f32,
        offset: f32,
        n: usize,
        i: f64,
    ) -> f32 {
        let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f64 };
        let angle = 2.0 * PI * frequency as f64 * i * inv_n + phase as f64;
        (offset as f64 + amplitude as f64 * angle.sin()) as f32
    }

    #[inline(always)]
    pub(super) fn multisine_at(
        components: impl Iterator<Item = (f32, f32, f32)>,
        dc_offset: f32,
        n: usize,
        i: f64,
    ) -> f32 {
        let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f64 };
        let mut sum = dc_offset as f64;
        for (freq, amp, phase) in components {
            let angle = 2.0 * PI * freq as f64 * i * inv_n + phase as f64;
            sum += amp as f64 * angle.sin();
        }
        sum as f32
    }

    #[inline(always)]
    pub(super) fn linear_at(start_value: f64, end_value: f64, n: usize, i: f64) -> f32 {
        let x = if n <= 1 { 0.0 } else { i / (n - 1) as f64 };
        (start_value + x * (end_value - start_value)) as f32
    }

    /// Materialised (non-analytic) models: nearest stored sample.
    #[inline(always)]
    pub(super) fn sampled_at(data: &[f32], i: f64) -> f32 {
        let idx = i.round().max(0.0) as usize;
        data.get(idx.min(data.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
    }

    /// Fill `results` with `(t, eval(i))` for every uniform sample timestamp in
    /// `query_start..=query_end` (branch on the model once, outside this loop).
    #[inline(always)]
    #[allow(clippy::while_float)]
    pub(super) fn fill_range(
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        start_time: i64,
        n: usize,
        eval: impl Fn(f64) -> f32,
    ) {
        let scale = if n <= 1 || step <= 0.0 {
            0.0
        } else {
            1.0 / step
        };
        let mut t = query_start as f64;
        while t <= query_end as f64 {
            let i = (t - start_time as f64) * scale;
            results.push((t as i64, eval(i)));
            t += step;
        }
    }
}

/// Segment metadata
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
pub struct SegmentMetadata {
    /// Unique segment ID
    pub id: u64,
    /// Creation timestamp (Unix millis)
    pub created_at: u64,
    /// Number of original data points
    pub point_count: usize,
    /// Original data size in bytes
    pub original_size: usize,
    /// Compressed model size in bytes
    pub model_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
}

/// Data Segment: A time range stored as a mathematical model
///
/// This is the fundamental storage unit of ALICE-DB.
/// Each segment covers a contiguous time range and stores
/// a procedural model instead of raw data.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
pub struct DataSegment {
    /// Start timestamp (inclusive)
    pub start_time: i64,
    /// End timestamp (inclusive)
    pub end_time: i64,
    /// The procedural model
    pub model: ModelType,
    /// Optional residual for lossless reconstruction
    /// Stores the difference between model output and actual values
    pub residual_blob: Option<Vec<u8>>,
    /// Segment metadata
    pub metadata: SegmentMetadata,
}

impl DataSegment {
    /// Create a new segment from a model
    #[must_use]
    pub fn new(
        id: u64,
        start_time: i64,
        end_time: i64,
        model: ModelType,
        point_count: usize,
        original_size: usize,
    ) -> Self {
        let model_size = model.estimated_size();
        let compression_ratio = if model_size > 0 {
            original_size as f64 / model_size as f64
        } else {
            1.0
        };

        Self {
            start_time,
            end_time,
            model,
            residual_blob: None,
            metadata: SegmentMetadata {
                id,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_millis() as u64),
                point_count,
                original_size,
                model_size,
                compression_ratio,
            },
        }
    }

    /// Add residual data for lossless reconstruction
    #[must_use]
    pub fn with_residual(mut self, residual: Vec<u8>) -> Self {
        self.residual_blob = Some(residual);
        self
    }

    /// Check if a timestamp falls within this segment
    #[inline]
    #[must_use]
    pub const fn contains(&self, timestamp: i64) -> bool {
        timestamp >= self.start_time && timestamp <= self.end_time
    }

    /// Check if a time range overlaps with this segment
    #[inline]
    #[must_use]
    pub const fn overlaps(&self, start: i64, end: i64) -> bool {
        self.start_time <= end && self.end_time >= start
    }

    /// Query a single value at a specific timestamp
    ///
    /// This is O(1) - just evaluate the mathematical function!
    /// No disk I/O needed beyond loading the segment.
    #[inline(always)]
    #[must_use]
    pub fn query_point(&self, timestamp: i64) -> Option<f32> {
        if !self.contains(timestamp) {
            return None;
        }
        let i = law::sample_pos(
            self.start_time,
            self.end_time,
            self.metadata.point_count,
            timestamp,
        );
        let value = self.evaluate_model_at(i);

        // Apply residual correction if available (decompress on-the-fly)
        if let Some(ref residual) = self.residual_blob {
            let kind = residual_kind(residual);
            let decompressed = decompress_residual(residual);
            let idx = self.timestamp_to_index(timestamp);
            return Some(apply_residual(value, kind, &decompressed, idx));
        }

        Some(value)
    }

    /// Query a range of values (Loop Unswitched - Branch outside loop)
    ///
    /// Returns Vec<(timestamp, value)> for all points in the range.
    /// Computation is O(n) where n = number of requested points.
    ///
    /// # Performance: Loop Unswitching
    ///
    /// Instead of branching inside the loop (killing branch prediction),
    /// we branch ONCE outside, then run a tight branchless loop.
    /// This allows CPU to pipeline SIMD instructions without stalls.
    #[must_use]
    pub fn query_range(&self, start: i64, end: i64) -> Vec<(i64, f32)> {
        let query_start = start.max(self.start_time);
        let query_end = end.min(self.end_time);

        if query_start > query_end {
            return Vec::new();
        }

        // Pre-calculate loop parameters
        let total_range = (self.end_time - self.start_time) as f64;
        let step = if self.metadata.point_count > 1 && total_range > 0.0 {
            total_range / (self.metadata.point_count - 1) as f64
        } else {
            1.0
        };
        let estimated_count = ((query_end - query_start) as f64 / step) as usize + 1;

        // Pre-allocate with exact capacity
        let mut results = Vec::with_capacity(estimated_count.min(self.metadata.point_count));

        // Branch on the model ONCE, then run one tight loop over the law
        let n = self.metadata.point_count;
        let start_time = self.start_time;
        match &self.model {
            ModelType::Polynomial { coefficients, .. } => {
                Self::fill_range_polynomial(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    start_time,
                    n,
                    coefficients,
                );
            }
            ModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| law::fourier_at(coefficients.iter().copied(), *dc_offset, *sample_count, i),
            ),
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| law::sine_at(*frequency, *amplitude, *phase, *offset, n, i),
            ),
            ModelType::MultiSine {
                components,
                dc_offset,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| law::multisine_at(components.iter().copied(), *dc_offset, n, i),
            ),
            ModelType::Constant { value } => {
                let v = *value as f32;
                law::fill_range(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    start_time,
                    n,
                    |_| v,
                );
            }
            ModelType::Linear {
                start_value,
                end_value,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| law::linear_at(*start_value, *end_value, n, i),
            ),
            ModelType::PerlinNoise { .. } | ModelType::RawLzma { .. } => {
                // Non-analytic models: materialise once, then index
                let all = self.generate_all();
                law::fill_range(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    start_time,
                    n,
                    |i| law::sampled_at(&all, i),
                );
            }
        }

        // Apply residuals in separate pass (decompress once, then apply)
        if let Some(ref residual) = self.residual_blob {
            let kind = residual_kind(residual);
            let decompressed = decompress_residual(residual);
            for (timestamp, value) in &mut results {
                let idx = self.timestamp_to_index(*timestamp);
                *value = apply_residual(*value, kind, &decompressed, idx);
            }
        }

        results
    }

    /// Polynomial range loop: 4 sample positions per `f64x4` Horner step, then
    /// a scalar tail — the same law as [`law::polynomial_at`].
    #[inline]
    #[allow(clippy::while_float)]
    fn fill_range_polynomial(
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        start_time: i64,
        n: usize,
        coefficients: &[f64],
    ) {
        let scale = if n <= 1 || step <= 0.0 {
            0.0
        } else {
            1.0 / step
        };
        let start = start_time as f64;
        let mut t = query_start as f64;
        let step4 = step * 4.0;
        while step.mul_add(3.0, t) <= query_end as f64 {
            let ts = [t, t + step, step.mul_add(2.0, t), step.mul_add(3.0, t)];
            let i = f64x4::from([
                (ts[0] - start) * scale,
                (ts[1] - start) * scale,
                (ts[2] - start) * scale,
                (ts[3] - start) * scale,
            ]);
            let v: [f64; 4] = horner_simd(i, coefficients).into();
            for k in 0..4 {
                results.push((ts[k] as i64, v[k] as f32));
            }
            t += step4;
        }
        law::fill_range(results, t as i64, query_end, step, start_time, n, |i| {
            law::polynomial_at(coefficients.iter().copied(), i)
        });
    }

    /// Query a range of values using SIMD acceleration (Phase 3)
    ///
    /// Now unified with `query_range` - this is an alias for compatibility.
    #[inline]
    #[must_use]
    pub fn query_range_simd(&self, start: i64, end: i64) -> Vec<(i64, f32)> {
        // query_range now uses SIMD internally for polynomial
        self.query_range(start, end)
    }

    /// Generate all data points in this segment
    ///
    /// This regenerates the entire dataset from the model.
    /// Used for full segment reads or verification.
    #[must_use]
    pub fn generate_all(&self) -> Vec<f32> {
        let n = self.metadata.point_count;

        match &self.model {
            ModelType::Polynomial { coefficients, .. } => {
                generators::generate_polynomial(n, coefficients)
            }
            ModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => {
                let coefs: Vec<(usize, f32, f32)> = coefficients.clone();
                generators::generate_from_coefficients(*sample_count, &coefs, *dc_offset)
            }
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => generators::generate_sine_wave(n, *frequency, *amplitude, *phase, *offset),
            ModelType::MultiSine {
                components,
                dc_offset,
            } => generators::generate_multi_sine(n, components, *dc_offset),
            ModelType::Constant { value } => {
                vec![*value as f32; n]
            }
            ModelType::Linear {
                start_value,
                end_value,
            } => {
                // Pre-compute reciprocal to avoid repeated division in iterator
                let inv_n_minus_1 = if n > 1 { 1.0 / (n - 1) as f64 } else { 0.0 };
                let delta = end_value - start_value;
                (0..n)
                    .map(|i| {
                        let t = i as f64 * inv_n_minus_1;
                        (start_value + t * delta) as f32
                    })
                    .collect()
            }
            ModelType::PerlinNoise {
                seed,
                scale,
                octaves,
                persistence,
                lacunarity,
            } => {
                // 1D value-noise fBm (alice-zip 0.4 `generate_fbm_1d`, the law
                // alice-zip 0.3 exposed as `generate_perlin_advanced(n, 1, ..)`;
                // sample values are unchanged). Parameters outside the law's
                // domain (`scale <= 0`, `octaves == 0`) can only come from a
                // corrupt model and regenerate as zeros, like a corrupt blob.
                generators::generate_fbm_1d(n, *seed, *scale, *octaves, *persistence, *lacunarity)
                    .unwrap_or_else(|_| vec![0.0; n])
            }
            ModelType::RawLzma {
                compressed_data,
                dtype,
                ..
            } => {
                // Decompress LZMA data
                self.decompress_raw(compressed_data, *dtype, n)
            }
        }
    }

    /// Evaluate the model at (fractional) sample position `i` — see [`law`]
    #[inline(always)]
    fn evaluate_model_at(&self, i: f64) -> f32 {
        let n = self.metadata.point_count;
        match &self.model {
            ModelType::Polynomial { coefficients, .. } => {
                law::polynomial_at(coefficients.iter().copied(), i)
            }
            ModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => law::fourier_at(coefficients.iter().copied(), *dc_offset, *sample_count, i),
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => law::sine_at(*frequency, *amplitude, *phase, *offset, n, i),
            ModelType::MultiSine {
                components,
                dc_offset,
            } => law::multisine_at(components.iter().copied(), *dc_offset, n, i),
            ModelType::Constant { value } => *value as f32,
            ModelType::Linear {
                start_value,
                end_value,
            } => law::linear_at(*start_value, *end_value, n, i),
            ModelType::PerlinNoise { .. } | ModelType::RawLzma { .. } => {
                // Non-analytic models: materialise and index (point queries on
                // these are O(n); range queries materialise once)
                law::sampled_at(&self.generate_all(), i)
            }
        }
    }

    /// Convert timestamp to array index (nearest uniform sample)
    #[inline(always)]
    fn timestamp_to_index(&self, timestamp: i64) -> usize {
        law::sample_pos(
            self.start_time,
            self.end_time,
            self.metadata.point_count,
            timestamp,
        )
        .round() as usize
    }

    /// Decompress raw LZMA data
    #[allow(clippy::unused_self)]
    fn decompress_raw(&self, compressed: &[u8], dtype: DataType, count: usize) -> Vec<f32> {
        use std::io::Cursor;

        let mut decompressed = Vec::new();
        if lzma_rs::lzma_decompress(&mut Cursor::new(compressed), &mut decompressed).is_err() {
            return vec![0.0; count];
        }
        decode_raw_bytes(&decompressed, dtype)
    }

    /// Serialize segment to bytes
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Deserialize segment from bytes
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        bincode::deserialize(data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Write segment to file
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let bytes = self.to_bytes()?;
        writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    /// Read segment from file
    ///
    /// # Errors
    ///
    /// Returns an error if reading or deserialization fails.
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;
        Self::from_bytes(&data)
    }

    /// Serialize to rkyv format (zero-copy compatible)
    ///
    /// # Errors
    ///
    /// Returns an error if rkyv serialization fails.
    pub fn to_rkyv_bytes(&self) -> io::Result<Vec<u8>> {
        rkyv::to_bytes::<_, 256>(self)
            .map(|v| v.to_vec())
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("rkyv serialize: {e:?}"))
            })
    }

    /// Write segment to file in rkyv format
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing to disk fails.
    pub fn write_rkyv<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let bytes = self.to_rkyv_bytes()?;
        std::fs::write(path, bytes)
    }
}

// =============================================================================
// Residual LZMA Compression
// =============================================================================

/// Magic byte: LZMA compressed
const RESIDUAL_LZMA: u8 = 0;
/// Magic byte: raw (uncompressed) additive residual
const RESIDUAL_RAW: u8 = 1;
/// Magic byte: LZMA-compressed **XOR** residual (bit pattern `original ^ model`)
const RESIDUAL_XOR_LZMA: u8 = 2;
/// Magic byte: raw XOR residual
const RESIDUAL_XOR_RAW: u8 = 3;

/// How a residual blob corrects the model output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualKind {
    /// `value + residual` (f32 add; 0.2.0-beta.2 and earlier — not bit exact,
    /// `a + (b − a)` rounds unless `a` and `b` are within a factor of 2)
    Additive,
    /// `f32::from_bits(value.to_bits() ^ residual)` — exact for every bit
    /// pattern (2026-09-17, oracle `tests/analytic_oracle.rs`)
    Xor,
}

/// Residual kind encoded in a blob's magic byte (legacy blobs without a
/// magic byte are additive).
#[must_use]
pub fn residual_kind(blob: &[u8]) -> ResidualKind {
    match blob.first() {
        Some(&RESIDUAL_XOR_LZMA | &RESIDUAL_XOR_RAW) => ResidualKind::Xor,
        _ => ResidualKind::Additive,
    }
}

/// Apply the `idx`-th residual word of a decompressed blob to `value`.
#[inline]
#[must_use]
pub fn apply_residual(value: f32, kind: ResidualKind, decompressed: &[u8], idx: usize) -> f32 {
    let offset = idx * 4;
    let Some(word) = decompressed.get(offset..offset + 4) else {
        return value;
    };
    let bytes: [u8; 4] = word.try_into().unwrap_or([0; 4]);
    match kind {
        ResidualKind::Additive => value + f32::from_le_bytes(bytes),
        ResidualKind::Xor => f32::from_bits(value.to_bits() ^ u32::from_le_bytes(bytes)),
    }
}

/// Compress an XOR residual (`original.to_bits() ^ model.to_bits()` per
/// sample, LE) — the exact-reconstruction format written since 2026-09-17.
///
/// Format: `[1 byte magic] [data]`, raw fallback when LZMA would expand.
#[must_use]
pub fn compress_residual_xor(raw: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::new();
    if lzma_rs::lzma_compress(&mut std::io::Cursor::new(raw), &mut compressed).is_ok()
        && compressed.len() < raw.len()
    {
        let mut out = Vec::with_capacity(1 + compressed.len());
        out.push(RESIDUAL_XOR_LZMA);
        out.extend_from_slice(&compressed);
        out
    } else {
        let mut out = Vec::with_capacity(1 + raw.len());
        out.push(RESIDUAL_XOR_RAW);
        out.extend_from_slice(raw);
        out
    }
}

/// Compress an **additive** residual (`original − model` per sample, f32 LE)
/// — the 0.2.0-beta.2 format; kept so existing blobs and callers keep
/// working, but the memtable writes [`compress_residual_xor`] since
/// 2026-09-17 (an additive residual is not bit exact).
///
/// Format: `[1 byte magic] [data]`, raw fallback when LZMA would expand.
#[must_use]
pub fn compress_residual(raw: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::new();
    if lzma_rs::lzma_compress(&mut std::io::Cursor::new(raw), &mut compressed).is_ok()
        && compressed.len() < raw.len()
    {
        let mut out = Vec::with_capacity(1 + compressed.len());
        out.push(RESIDUAL_LZMA);
        out.extend_from_slice(&compressed);
        out
    } else {
        let mut out = Vec::with_capacity(1 + raw.len());
        out.push(RESIDUAL_RAW);
        out.extend_from_slice(raw);
        out
    }
}

/// SIMD Horner (4 sample positions at once), ascending coefficients — the
/// vector form of [`law::polynomial_at`].
#[inline(always)]
fn horner_simd(x: f64x4, coefficients: &[f64]) -> f64x4 {
    let mut result = f64x4::ZERO;
    for &c in coefficients.iter().rev() {
        result = result * x + f64x4::splat(c);
    }
    result
}

/// Decompress residual blob back to raw f32 LE bytes.
///
/// Handles three formats:
/// - `[0x00][LZMA data]` — LZMA compressed (new format)
/// - `[0x01][raw data]` — raw with magic byte (new format, LZMA expansion fallback)
/// - `[raw f32 LE bytes]` — legacy format (no magic byte)
///
/// Distinguishes new-format LZMA from legacy by attempting LZMA decompression;
/// if it fails, treats the entire blob as legacy raw f32 LE bytes.
#[must_use]
pub fn decompress_residual(blob: &[u8]) -> Vec<u8> {
    if blob.is_empty() {
        return Vec::new();
    }
    match blob[0] {
        RESIDUAL_LZMA => {
            // Try LZMA decompression; if it fails, this is legacy data where
            // the first f32 LE byte happens to be 0x00
            let mut decompressed = Vec::new();
            if lzma_rs::lzma_decompress(&mut std::io::Cursor::new(&blob[1..]), &mut decompressed)
                .is_ok()
            {
                decompressed
            } else {
                // Legacy format fallback
                blob.to_vec()
            }
        }
        RESIDUAL_RAW | RESIDUAL_XOR_RAW => blob[1..].to_vec(),
        RESIDUAL_XOR_LZMA => {
            let mut decompressed = Vec::new();
            if lzma_rs::lzma_decompress(&mut std::io::Cursor::new(&blob[1..]), &mut decompressed)
                .is_ok()
            {
                decompressed
            } else {
                Vec::new()
            }
        }
        _ => {
            // Legacy format: no magic byte, raw f32 LE bytes
            blob.to_vec()
        }
    }
}

// =============================================================================
// SegmentView: Unified Zero-Copy Access (Phase 1)
// =============================================================================

/// Backing storage for Zero-Copy access
///
/// Supports multiple memory sources while maintaining zero-copy semantics.
/// The archived data is accessed directly without deserialization regardless
/// of the underlying storage type.
#[derive(Clone)]
pub enum SegmentSource {
    /// Memory-mapped file (best for large segments, lazy loading).
    /// Holds `File` to keep the advisory shared lock alive until drop.
    Mmap(Arc<Mmap>, Arc<File>),
    /// In-memory bytes (for freshly flushed `MemTable` data)
    Vec(Arc<Vec<u8>>),
    /// Static slice (for embedded/testing)
    Slice(&'static [u8]),
}

impl AsRef<[u8]> for SegmentSource {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Mmap(m, _file) => m.as_ref(),
            Self::Vec(v) => v.as_slice(),
            Self::Slice(s) => s,
        }
    }
}

/// Reinterpret decompressed little-endian bytes as `f32` samples of `dtype`.
fn decode_raw_bytes(decompressed: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => decompressed
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
            .collect(),
        DataType::Float64 => decompressed
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap_or([0; 8])) as f32)
            .collect(),
        DataType::Int32 => decompressed
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap_or([0; 4])) as f32)
            .collect(),
        DataType::Int64 => decompressed
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap_or([0; 8])) as f32)
            .collect(),
        DataType::UInt8 => decompressed.iter().map(|&b| b as f32).collect(),
    }
}

/// Zero-copy segment view (The "Hot" path for all queries)
///
/// This is the unified type for accessing segment data. Whether the data
/// comes from mmap, in-memory Vec, or static slice, queries execute with
/// zero deserialization overhead.
///
/// # Performance
///
/// - Open: O(1) - just validate rkyv header
/// - Query: O(1) per point - compute f(x) from archived coefficients
/// - Memory: Zero-copy - no heap allocation during queries
///
/// # Safety: Field Order Matters!
///
/// Rust drops fields in declaration order (top to bottom).
/// `archived` is a reference into `source`, so we MUST drop `archived` first.
/// By declaring `archived` before `source`, we ensure:
///   1. `archived` is dropped first (reference becomes invalid, but no use-after-free)
///   2. `source` is dropped second (backing memory is freed safely)
///
/// DO NOT reorder these fields without understanding the safety implications!
///
/// # Lifecycle
///
/// ```text
/// MemTable flush → rkyv bytes → SegmentView::from_vec() → Cache
///                             ↓
///                        Write to disk (async)
///                             ↓
/// Next startup → SegmentView::open() → mmap → Cache
/// ```
pub struct SegmentView {
    // ⚠️ SAFETY: archived MUST be declared BEFORE source!
    // Rust drops fields top-to-bottom. archived references source's memory,
    // so archived must be invalidated before source is freed.
    /// Archived segment (zero-copy reference into source) - DROPPED FIRST
    archived: &'static ArchivedDataSegment,
    /// Backing memory (kept alive for archived reference) - DROPPED SECOND
    source: SegmentSource,
}

impl SegmentView {
    /// Open a segment file with zero-copy mmap
    ///
    /// Best for: Reading existing segments from disk.
    /// The OS handles page caching; only accessed pages are loaded.
    ///
    /// # Errors
    ///
    /// Returns `InvalidData` if the file is empty or smaller than the minimum
    /// valid rkyv archive size. An empty mmap is undefined behaviour on some
    /// platforms; we reject it here before the `unsafe` mmap call.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        #[allow(unused_imports)]
        use fs2::FileExt;

        let file = File::open(path)?;

        // Acquire advisory shared lock to prevent concurrent truncation/deletion.
        // The lock is held as long as the File lives (stored in SegmentSource::Mmap).
        // trait 経由で明示 (1.89+ の inherent File::lock_shared と MSRV 1.87 の両立、clippy incompatible_msrv)
        FileExt::lock_shared(&file)?;

        // Validate file size before mapping.
        //
        // A zero-length mmap is undefined behaviour on Linux (mmap(2) returns
        // EINVAL) and Windows. Even a non-zero but truncated file would
        // produce an mmap whose contents are outside the rkyv archive bounds,
        // causing rkyv validation to fail or — if validation were skipped —
        // undefined behaviour. We reject both cases here so that the unsafe
        // Mmap::map call below always operates on a correctly-sized file.
        //
        // rkyv 0.7 archives contain at minimum an 8-byte root offset footer,
        // so any valid archive must be larger than 0 bytes.
        let file_len = file.metadata()?.len();
        if file_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment file is empty (0 bytes); cannot mmap",
            ));
        }
        // A minimal rkyv DataSegment archive is at least a few hundred bytes.
        // Using 8 as the lower bound here matches the rkyv root-offset footer
        // size and prevents the mmap call from succeeding on severely truncated
        // files before rkyv validation catches the corruption.
        if file_len < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "segment file is too small ({file_len} bytes) to contain a valid rkyv archive"
                ),
            ));
        }

        // SAFETY: The file descriptor is valid and the file has been confirmed
        // non-empty above. The resulting Mmap is stored in Arc<Mmap> inside
        // SegmentSource::Mmap and kept alive for the lifetime of the
        // SegmentView, ensuring the mapped pages are not unmapped while any
        // reference into them is live.
        let mmap = unsafe { Mmap::map(&file)? };

        // Pre-fault validation: touch first and last bytes to detect
        // SIGBUS early (file truncated/deleted between stat and mmap)
        {
            let bytes = mmap.as_ref();
            let _ = bytes[0];
            let _ = bytes[bytes.len() - 1];
        }

        let source = SegmentSource::Mmap(Arc::new(mmap), Arc::new(file));
        Self::from_source(source)
    }

    /// Open a segment file by reading into memory (no mmap)
    ///
    /// Use this instead of `open()` to avoid SIGBUS risk when the
    /// underlying file might be deleted or truncated after opening.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or contains invalid rkyv data.
    pub fn open_read<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_vec(data)
    }

    /// Create from in-memory bytes (Zero-Copy, no disk I/O)
    ///
    /// Best for: Freshly flushed `MemTable` data.
    /// Avoids waiting for disk write to complete before caching.
    ///
    /// # Errors
    ///
    /// Returns an error if the rkyv data is invalid.
    ///
    /// # Example
    /// ```ignore
    /// let rkyv_bytes = segment.to_rkyv_bytes()?;
    /// let view = SegmentView::from_vec(rkyv_bytes)?;
    /// // Now queryable immediately, disk write can happen async
    /// ```
    pub fn from_vec(data: Vec<u8>) -> io::Result<Self> {
        let source = SegmentSource::Vec(Arc::new(data));
        Self::from_source(source)
    }

    /// Create from `Arc<Vec<u8>>` (avoids clone if you already have Arc)
    ///
    /// # Errors
    ///
    /// Returns an error if the rkyv data is invalid.
    pub fn from_arc_vec(data: Arc<Vec<u8>>) -> io::Result<Self> {
        let source = SegmentSource::Vec(data);
        Self::from_source(source)
    }

    /// Create from static slice (for embedded/testing)
    ///
    /// # Errors
    ///
    /// Returns an error if the rkyv data is invalid.
    pub fn from_static(data: &'static [u8]) -> io::Result<Self> {
        let source = SegmentSource::Slice(data);
        Self::from_source(source)
    }

    /// Internal: Create `SegmentView` from any `SegmentSource`
    fn from_source(source: SegmentSource) -> io::Result<Self> {
        let data = source.as_ref();

        // Validate rkyv data. `check_archived_root` verifies byte-level
        // validity including alignment, size, and internal rkyv invariants,
        // so the pointer we derive from it is guaranteed to be correctly
        // aligned and to point into initialized, immutable memory within
        // `data`.
        let archived = rkyv::check_archived_root::<DataSegment>(data).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("rkyv validation: {e:?}"),
            )
        })?;

        // SAFETY: Lifetime extension from &'data ArchivedDataSegment to
        // &'static ArchivedDataSegment.
        //
        // Invariants that make this sound:
        //
        // 1. **Alignment & size**: `rkyv::check_archived_root` above has
        //    already verified that `archived` is correctly aligned for
        //    `ArchivedDataSegment` and that all referenced bytes are within
        //    bounds of `data`. No type-punning occurs; we are only changing
        //    the lifetime, not the type or the pointer value.
        //
        // 2. **Backing memory lifetime**: `source` is stored in the same
        //    `SegmentView` struct immediately after this block. Rust's
        //    ownership model guarantees that `source` is not dropped before
        //    `SegmentView` is dropped.
        //
        // 3. **Drop order**: `archived` is declared as the *first* field of
        //    `SegmentView` (see struct definition). Rust drops fields in
        //    declaration order (top to bottom). Therefore `archived` (a plain
        //    reference — no destructor) is "dropped" before `source`. Because
        //    references have no destructor, this ordering ensures no
        //    use-after-free: the reference simply becomes unreachable before
        //    the backing memory is freed. The struct-level doc comment above
        //    the `SegmentView` definition explicitly warns maintainers not to
        //    reorder these fields.
        //
        // 4. **No interior mutability / aliasing**: `SegmentSource` variants
        //    (Mmap, Vec, Slice) all provide shared read-only access to the
        //    underlying bytes. No `&mut` reference to the backing bytes is
        //    ever created while `archived` is live.
        //
        // 5. **No Send/Sync unsoundness**: `Arc<Mmap>` and `Arc<Vec<u8>>`
        //    are `Send + Sync`, so `SegmentView` remains safe to share across
        //    threads. The `&'static` annotation does not introduce additional
        //    aliasing beyond what `Arc` already permits.
        //
        // Alternative considered: storing a raw pointer `*const
        // ArchivedDataSegment` instead of `&'static`. That would be equally
        // safe but would require unsafe derefs at every access site, making
        // the code more verbose without any safety gain.
        let archived: &'static ArchivedDataSegment = unsafe {
            // We cast the pointer, not the reference, to make it explicit that
            // only the lifetime tag changes and no reinterpretation of the
            // pointed-to bits occurs.
            &*std::ptr::from_ref::<ArchivedDataSegment>(archived)
        };

        Ok(Self { archived, source })
    }

    /// Get the underlying source type (for debugging/stats)
    #[must_use]
    pub const fn source_type(&self) -> &'static str {
        match &self.source {
            SegmentSource::Mmap(..) => "mmap",
            SegmentSource::Vec(_) => "vec",
            SegmentSource::Slice(_) => "slice",
        }
    }

    /// Get start timestamp
    #[inline]
    #[must_use]
    pub const fn start_time(&self) -> i64 {
        self.archived.start_time
    }

    /// Get end timestamp
    #[inline]
    #[must_use]
    pub const fn end_time(&self) -> i64 {
        self.archived.end_time
    }

    /// Check if timestamp is in range
    #[inline]
    #[must_use]
    pub const fn contains(&self, timestamp: i64) -> bool {
        timestamp >= self.archived.start_time && timestamp <= self.archived.end_time
    }

    /// Zero-copy point query
    ///
    /// Computes f(x) directly from archived model coefficients.
    /// No deserialization occurs - we read directly from mmap.
    #[inline(always)]
    #[must_use]
    pub fn query_point(&self, timestamp: i64) -> Option<f32> {
        if !self.contains(timestamp) {
            return None;
        }
        let n = self.archived.metadata.point_count as usize;
        let i = law::sample_pos(
            self.archived.start_time,
            self.archived.end_time,
            n,
            timestamp,
        );
        let value = self.evaluate_archived_model(i);

        // Lossless mode stores per-sample residuals; the in-memory path applied
        // them, the mmap path did not until 0.2.0-beta.2.
        if let Some(residual) = self.archived.residual_blob.as_ref() {
            let kind = residual_kind(residual.as_slice());
            let decompressed = decompress_residual(residual.as_slice());
            let idx = self.archived_timestamp_to_index(timestamp);
            return Some(apply_residual(value, kind, &decompressed, idx));
        }
        Some(value)
    }

    /// Sample index of `timestamp` under the segment's uniform-spacing model
    /// (mirror of `DataSegment::timestamp_to_index` for the archived view).
    fn archived_timestamp_to_index(&self, timestamp: i64) -> usize {
        law::sample_pos(
            self.archived.start_time,
            self.archived.end_time,
            self.archived.metadata.point_count as usize,
            timestamp,
        )
        .round() as usize
    }

    /// Zero-copy range query (Loop Unswitched + SIMD for Polynomial)
    ///
    /// # Performance: Same optimization as `DataSegment`
    ///
    /// Branch ONCE outside loop, then run tight branchless loops.
    /// Polynomial uses SIMD (f64x4) for 4x throughput.
    #[must_use]
    pub fn query_range(&self, start: i64, end: i64) -> Vec<(i64, f32)> {
        let query_start = start.max(self.archived.start_time);
        let query_end = end.min(self.archived.end_time);

        if query_start > query_end {
            return Vec::new();
        }

        let total_range = (self.archived.end_time - self.archived.start_time) as f64;
        let point_count = self.archived.metadata.point_count as f64;
        let step = if point_count > 1.0 && total_range > 0.0 {
            total_range / (point_count - 1.0)
        } else {
            1.0
        };
        let estimated_count = ((query_end - query_start) as f64 / step) as usize + 1;

        let mut results = Vec::with_capacity(estimated_count.min(point_count as usize));

        // Branch on the model ONCE, then run one tight loop over the law
        let n = self.archived.metadata.point_count as usize;
        let start_time = self.archived.start_time;
        match &self.archived.model {
            ArchivedModelType::Polynomial { coefficients, .. } => {
                DataSegment::fill_range_polynomial(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    start_time,
                    n,
                    coefficients.as_slice(),
                );
            }
            ArchivedModelType::Constant { value } => {
                let v = *value as f32;
                law::fill_range(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    start_time,
                    n,
                    |_| v,
                );
            }
            ArchivedModelType::Linear {
                start_value,
                end_value,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| law::linear_at(*start_value, *end_value, n, i),
            ),
            ArchivedModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| law::sine_at(*frequency, *amplitude, *phase, *offset, n, i),
            ),
            ArchivedModelType::MultiSine {
                components,
                dc_offset,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| law::multisine_at(components.iter().map(|c| (c.0, c.1, c.2)), *dc_offset, n, i),
            ),
            ArchivedModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => law::fill_range(
                &mut results,
                query_start,
                query_end,
                step,
                start_time,
                n,
                |i| {
                    law::fourier_at(
                        coefficients.iter().map(|c| (c.0 as usize, c.1, c.2)),
                        *dc_offset,
                        *sample_count as usize,
                        i,
                    )
                },
            ),
            ArchivedModelType::PerlinNoise { .. } | ArchivedModelType::RawLzma { .. } => {
                let all = self.materialise_archived_samples();
                law::fill_range(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    start_time,
                    n,
                    |i| law::sampled_at(&all, i),
                );
            }
        }

        // Lossless residuals (see `query_point`)
        if let Some(residual) = self.archived.residual_blob.as_ref() {
            let kind = residual_kind(residual.as_slice());
            let decompressed = decompress_residual(residual.as_slice());
            for (timestamp, value) in &mut results {
                let idx = self.archived_timestamp_to_index(*timestamp);
                *value = apply_residual(*value, kind, &decompressed, idx);
            }
        }

        results
    }

    /// Evaluate the archived model at (fractional) sample position `i` — the
    /// same [`law`] functions as the in-memory path, read zero-copy.
    #[inline(always)]
    fn evaluate_archived_model(&self, i: f64) -> f32 {
        let n = self.archived.metadata.point_count as usize;
        match &self.archived.model {
            ArchivedModelType::Polynomial { coefficients, .. } => {
                law::polynomial_at(coefficients.as_slice().iter().copied(), i)
            }
            ArchivedModelType::Constant { value } => *value as f32,
            ArchivedModelType::Linear {
                start_value,
                end_value,
            } => law::linear_at(*start_value, *end_value, n, i),
            ArchivedModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => law::sine_at(*frequency, *amplitude, *phase, *offset, n, i),
            ArchivedModelType::MultiSine {
                components,
                dc_offset,
            } => law::multisine_at(components.iter().map(|c| (c.0, c.1, c.2)), *dc_offset, n, i),
            ArchivedModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => law::fourier_at(
                coefficients.iter().map(|c| (c.0 as usize, c.1, c.2)),
                *dc_offset,
                *sample_count as usize,
                i,
            ),
            ArchivedModelType::PerlinNoise { .. } | ArchivedModelType::RawLzma { .. } => {
                // Non-analytic models: materialise the sample vector and index it.
                // (0.2.0-beta.1 returned 0.0 here — every value stored through the
                // RawLzma fallback, i.e. any data no procedural model fits, read
                // back as zero from the mmap path while the in-memory path was
                // correct. Found by alice-physics' replay / db_bridge contract
                // tests, 2026-09-15.)
                law::sampled_at(&self.materialise_archived_samples(), i)
            }
        }
    }

    /// Decode the sample vector of a non-analytic archived model (`RawLzma` →
    /// LZMA decompress + dtype reinterpret, `PerlinNoise` → regenerate from the
    /// archived parameters). `point_count` samples; a corrupt blob yields zeros.
    fn materialise_archived_samples(&self) -> Vec<f32> {
        let n = self.archived.metadata.point_count as usize;
        match &self.archived.model {
            ArchivedModelType::RawLzma {
                compressed_data,
                dtype,
                ..
            } => {
                use std::io::Cursor;
                let mut decompressed = Vec::new();
                if lzma_rs::lzma_decompress(
                    &mut Cursor::new(compressed_data.as_slice()),
                    &mut decompressed,
                )
                .is_err()
                {
                    return vec![0.0; n];
                }
                let dtype: DataType = match dtype {
                    ArchivedDataType::Float32 => DataType::Float32,
                    ArchivedDataType::Float64 => DataType::Float64,
                    ArchivedDataType::Int32 => DataType::Int32,
                    ArchivedDataType::Int64 => DataType::Int64,
                    ArchivedDataType::UInt8 => DataType::UInt8,
                };
                decode_raw_bytes(&decompressed, dtype)
            }
            ArchivedModelType::PerlinNoise {
                seed,
                scale,
                octaves,
                persistence,
                lacunarity,
            } => generators::generate_fbm_1d(n, *seed, *scale, *octaves, *persistence, *lacunarity)
                .unwrap_or_else(|_| vec![0.0; n]),
            _ => Vec::new(),
        }
    }

    /// Get compression ratio from metadata
    #[must_use]
    pub const fn compression_ratio(&self) -> f64 {
        self.archived.metadata.compression_ratio
    }

    /// Get point count from metadata
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.archived.metadata.point_count as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_segment() {
        let segment = DataSegment::new(
            1,
            0,
            1000,
            ModelType::Constant { value: 42.0 },
            1001,
            1001 * 4,
        );

        assert!(segment.contains(500));
        assert!(!segment.contains(1001));

        let value = segment.query_point(500).unwrap();
        assert!((value - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_segment() {
        let segment = DataSegment::new(
            1,
            0,
            100,
            ModelType::Linear {
                start_value: 0.0,
                end_value: 100.0,
            },
            101,
            101 * 4,
        );

        let start = segment.query_point(0).unwrap();
        let mid = segment.query_point(50).unwrap();
        let end = segment.query_point(100).unwrap();

        assert!((start - 0.0).abs() < 0.1);
        assert!((mid - 50.0).abs() < 0.1);
        assert!((end - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_polynomial_segment() {
        // y = i² at sample index i (the fitter convention; this test pinned
        // x ∈ [0, 1] until 2026-09-17, which is what `generate_all` never did)
        let segment = DataSegment::new(
            1,
            0,
            100,
            ModelType::Polynomial {
                coefficients: vec![0.0, 0.0, 1.0], // c0 + c1*x + c2*x^2
                degree: 2,
                fit_error: 0.0,
            },
            101,
            101 * 4,
        );

        let at_half = segment.query_point(50).unwrap();
        // i = 50, y = 2500
        assert!((at_half - 2500.0).abs() < 0.01);
        assert!((segment.generate_all()[50] - at_half).abs() < f32::EPSILON);
    }

    #[test]
    fn test_range_query() {
        let segment = DataSegment::new(
            1,
            0,
            100,
            ModelType::Linear {
                start_value: 0.0,
                end_value: 100.0,
            },
            101,
            101 * 4,
        );

        let results = segment.query_range(0, 100);
        assert!(!results.is_empty());

        // First and last should match linear interpolation
        let (_, first_val) = results.first().unwrap();
        let (_, last_val) = results.last().unwrap();
        assert!((first_val - 0.0).abs() < 1.0);
        assert!((last_val - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_segment_serialization() {
        let segment = DataSegment::new(
            1,
            0,
            1000,
            ModelType::Polynomial {
                coefficients: vec![1.0, 2.0, 3.0],
                degree: 2,
                fit_error: 0.001,
            },
            1001,
            1001 * 4,
        );

        let bytes = segment.to_bytes().unwrap();
        let restored = DataSegment::from_bytes(&bytes).unwrap();

        assert_eq!(restored.start_time, segment.start_time);
        assert_eq!(restored.end_time, segment.end_time);
        assert_eq!(restored.metadata.point_count, segment.metadata.point_count);
    }

    #[test]
    fn test_simd_polynomial_query() {
        // y = 2 + 3x + x^2 on [0, 100]
        let segment = DataSegment::new(
            1,
            0,
            100,
            ModelType::Polynomial {
                coefficients: vec![2.0, 3.0, 1.0], // c0 + c1*x + c2*x^2
                degree: 2,
                fit_error: 0.0,
            },
            101,
            101 * 4,
        );

        // Compare scalar vs SIMD results
        let scalar_results = segment.query_range(0, 100);
        let simd_results = segment.query_range_simd(0, 100);

        assert_eq!(scalar_results.len(), simd_results.len());

        // Results should be identical (within floating point tolerance)
        for (s, simd) in scalar_results.iter().zip(simd_results.iter()) {
            assert_eq!(s.0, simd.0); // timestamps match
            assert!(
                (s.1 - simd.1).abs() < 0.001,
                "Mismatch at t={}: scalar={}, simd={}",
                s.0,
                s.1,
                simd.1
            );
        }
    }

    #[test]
    fn test_simd_large_polynomial() {
        // Higher degree polynomial for more thorough testing
        let segment = DataSegment::new(
            1,
            0,
            1000,
            ModelType::Polynomial {
                coefficients: vec![1.0, 0.5, -0.1, 0.01, -0.001], // degree 4
                degree: 4,
                fit_error: 0.0,
            },
            1001,
            1001 * 4,
        );

        let scalar = segment.query_range(0, 1000);
        let simd = segment.query_range_simd(0, 1000);

        assert_eq!(scalar.len(), simd.len());

        // All values should match
        for (s, sim) in scalar.iter().zip(simd.iter()) {
            assert!(
                (s.1 - sim.1).abs() < 0.01,
                "Mismatch: scalar={}, simd={}",
                s.1,
                sim.1
            );
        }
    }

    #[test]
    fn test_contains_boundary_values() {
        let segment = DataSegment::new(1, 10, 20, ModelType::Constant { value: 1.0 }, 11, 44);
        // Inclusive boundaries
        assert!(segment.contains(10));
        assert!(segment.contains(20));
        assert!(segment.contains(15));
        // Outside boundaries
        assert!(!segment.contains(9));
        assert!(!segment.contains(21));
        assert!(!segment.contains(i64::MIN));
        assert!(!segment.contains(i64::MAX));
    }

    #[test]
    fn test_overlaps() {
        let segment = DataSegment::new(1, 100, 200, ModelType::Constant { value: 1.0 }, 101, 404);
        // Overlapping ranges
        assert!(segment.overlaps(50, 150)); // partial overlap left
        assert!(segment.overlaps(150, 250)); // partial overlap right
        assert!(segment.overlaps(100, 200)); // exact match
        assert!(segment.overlaps(120, 180)); // fully inside
        assert!(segment.overlaps(50, 300)); // fully containing
                                            // Non-overlapping ranges
        assert!(!segment.overlaps(0, 99));
        assert!(!segment.overlaps(201, 300));
    }

    #[test]
    fn test_query_point_outside_range_returns_none() {
        let segment = DataSegment::new(
            1,
            0,
            100,
            ModelType::Linear {
                start_value: 0.0,
                end_value: 100.0,
            },
            101,
            404,
        );
        assert!(segment.query_point(-1).is_none());
        assert!(segment.query_point(101).is_none());
    }

    #[test]
    fn test_query_range_non_overlapping() {
        let segment = DataSegment::new(1, 100, 200, ModelType::Constant { value: 5.0 }, 101, 404);
        let results = segment.query_range(0, 50);
        assert!(results.is_empty());

        let results2 = segment.query_range(300, 400);
        assert!(results2.is_empty());
    }

    #[test]
    fn test_single_point_segment() {
        // Segment with start_time == end_time (single point)
        let segment = DataSegment::new(1, 42, 42, ModelType::Constant { value: 99.0 }, 1, 4);
        assert!(segment.contains(42));
        assert!(!segment.contains(41));
        let val = segment.query_point(42).unwrap();
        assert!((val - 99.0).abs() < 0.001);
    }

    #[test]
    fn test_constant_segment_generate_all() {
        let segment = DataSegment::new(1, 0, 99, ModelType::Constant { value: 7.5 }, 100, 400);
        let all = segment.generate_all();
        assert_eq!(all.len(), 100);
        for &v in &all {
            assert!((v - 7.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_linear_generate_all() {
        let segment = DataSegment::new(
            1,
            0,
            99,
            ModelType::Linear {
                start_value: 0.0,
                end_value: 99.0,
            },
            100,
            400,
        );
        let all = segment.generate_all();
        assert_eq!(all.len(), 100);
        assert!((all[0] - 0.0).abs() < 0.1);
        assert!((all[99] - 99.0).abs() < 0.1);
        // Monotonically increasing
        for i in 1..all.len() {
            assert!(all[i] >= all[i - 1] - 0.01);
        }
    }

    #[test]
    fn test_segment_rkyv_roundtrip() {
        let segment = DataSegment::new(
            42,
            100,
            500,
            ModelType::Linear {
                start_value: 10.0,
                end_value: 50.0,
            },
            401,
            1604,
        );
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        assert!(!rkyv_bytes.is_empty());

        // Validate that SegmentView can read the rkyv bytes
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();
        assert_eq!(view.start_time(), 100);
        assert_eq!(view.end_time(), 500);
        assert!(view.contains(300));
        assert!(!view.contains(50));
    }

    #[test]
    fn test_segment_view_source_type() {
        let segment = DataSegment::new(1, 0, 10, ModelType::Constant { value: 1.0 }, 11, 44);
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();
        assert_eq!(view.source_type(), "vec");
    }

    #[test]
    fn test_segment_view_query_point_constant() {
        let segment = DataSegment::new(1, 0, 100, ModelType::Constant { value: 42.0 }, 101, 404);
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();

        let val = view.query_point(50).unwrap();
        assert!((val - 42.0).abs() < 0.001);

        assert!(view.query_point(-1).is_none());
        assert!(view.query_point(101).is_none());
    }

    #[test]
    fn test_segment_view_query_range_linear() {
        let segment = DataSegment::new(
            1,
            0,
            100,
            ModelType::Linear {
                start_value: 0.0,
                end_value: 100.0,
            },
            101,
            404,
        );
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();

        let results = view.query_range(0, 100);
        assert!(!results.is_empty());

        let (_, first_val) = results.first().unwrap();
        let (_, last_val) = results.last().unwrap();
        assert!((first_val - 0.0).abs() < 1.0);
        assert!((last_val - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_segment_view_compression_ratio_and_point_count() {
        let segment = DataSegment::new(1, 0, 99, ModelType::Constant { value: 1.0 }, 100, 400);
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();
        assert!(view.compression_ratio() > 1.0);
        assert_eq!(view.point_count(), 100);
    }

    #[test]
    fn test_segment_write_read_file_roundtrip() {
        let segment = DataSegment::new(
            5,
            0,
            50,
            ModelType::Polynomial {
                coefficients: vec![1.0, -0.5, 0.25],
                degree: 2,
                fit_error: 0.0001,
            },
            51,
            204,
        );

        let mut buf: Vec<u8> = Vec::new();
        segment.write_to(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let restored = DataSegment::read_from(&mut cursor).unwrap();

        assert_eq!(restored.start_time, 0);
        assert_eq!(restored.end_time, 50);
        assert_eq!(restored.metadata.id, 5);
        assert_eq!(restored.metadata.point_count, 51);
    }

    #[test]
    fn test_with_residual() {
        let segment = DataSegment::new(1, 0, 10, ModelType::Constant { value: 5.0 }, 11, 44);
        assert!(segment.residual_blob.is_none());

        let residual_data: Vec<u8> = (0..44).map(|i| i as u8).collect();
        let segment_with_res = segment.with_residual(residual_data);
        assert!(segment_with_res.residual_blob.is_some());
        assert_eq!(segment_with_res.residual_blob.unwrap().len(), 44);
    }

    #[test]
    fn test_segment_metadata_fields() {
        let segment = DataSegment::new(
            99,
            1000,
            2000,
            ModelType::Linear {
                start_value: 0.0,
                end_value: 1.0,
            },
            1001,
            4004,
        );
        assert_eq!(segment.metadata.id, 99);
        assert_eq!(segment.metadata.point_count, 1001);
        assert_eq!(segment.metadata.original_size, 4004);
        assert!(segment.metadata.created_at > 0);
        assert!(segment.metadata.compression_ratio > 1.0);
    }

    #[test]
    fn test_lossless_lzma_roundtrip() {
        // Create a segment with known residuals
        let mut segment = DataSegment::new(1, 0, 99, ModelType::Constant { value: 10.0 }, 100, 400);

        // Build raw residual f32 bytes
        let mut raw_residuals = Vec::with_capacity(100 * 4);
        for i in 0..100 {
            let r = (i as f32) * 0.001;
            raw_residuals.extend_from_slice(&r.to_le_bytes());
        }

        // Compress and attach
        let compressed = super::compress_residual(&raw_residuals);
        // Compressed blob should have magic byte prefix
        assert!(!compressed.is_empty());
        assert!(compressed[0] == super::RESIDUAL_LZMA || compressed[0] == super::RESIDUAL_RAW);
        segment = segment.with_residual(compressed.clone());

        // Verify decompression roundtrip
        let decompressed = super::decompress_residual(&compressed);
        assert_eq!(decompressed.len(), raw_residuals.len());
        assert_eq!(decompressed, raw_residuals);

        // Verify query_point applies residual correction
        let val_at_0 = segment.query_point(0).unwrap();
        // Constant model = 10.0, residual[0] = 0.0 * 0.001 = 0.0
        assert!((val_at_0 - 10.0).abs() < 0.01);

        let val_at_50 = segment.query_point(50).unwrap();
        // Constant model = 10.0, residual[50] = 50 * 0.001 = 0.05
        assert!((val_at_50 - 10.05).abs() < 0.01);
    }

    #[test]
    fn test_compress_decompress_fallback() {
        // Tiny data: LZMA expansion expected → falls back to raw
        let tiny = vec![1u8, 2, 3, 4];
        let compressed = super::compress_residual(&tiny);
        assert_eq!(compressed[0], super::RESIDUAL_RAW);
        let decompressed = super::decompress_residual(&compressed);
        assert_eq!(decompressed, tiny);
    }

    #[test]
    fn test_legacy_residual_no_magic_byte() {
        // Legacy format: no magic byte, just raw f32 LE bytes
        let mut legacy = Vec::new();
        legacy.extend_from_slice(&1.5f32.to_le_bytes());
        legacy.extend_from_slice(&2.5f32.to_le_bytes());

        // Should still work (fallback path)
        let decompressed = super::decompress_residual(&legacy);
        assert_eq!(decompressed, legacy);
    }

    #[test]
    fn test_mmap_file_lock() {
        use fs2::FileExt;

        // Write a valid segment to a temp file
        let segment = DataSegment::new(1, 0, 99, ModelType::Constant { value: 42.0 }, 100, 400);
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_seg.rkyv");
        std::fs::write(&path, &rkyv_bytes).unwrap();

        // Open with SegmentView (acquires shared lock)
        let view = SegmentView::open(&path).unwrap();
        assert_eq!(view.source_type(), "mmap");

        // Another shared lock should succeed (shared locks are compatible)
        let file2 = File::open(&path).unwrap();
        assert!(file2.lock_shared().is_ok());
        file2.unlock().unwrap();

        // Exclusive lock should fail while shared lock is held
        let file3 = File::open(&path).unwrap();
        assert!(
            file3.try_lock_exclusive().is_err(),
            "exclusive lock should fail while shared lock is held"
        );

        // Drop view → shared lock released
        drop(view);

        // Now exclusive lock should succeed
        let file4 = File::open(&path).unwrap();
        assert!(file4.try_lock_exclusive().is_ok());
    }
}
