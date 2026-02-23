//! Data Segment (Model-Based SSTable)
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
//! SegmentView uses mmap + rkyv for true zero-copy access:
//! - No deserialization overhead
//! - OS handles page caching
//! - Direct pointer access to model coefficients
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::model::{ArchivedModelType, DataType, ModelType};
use alice_core::generators;
use memmap2::Mmap;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;
use wide::f64x4;

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
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                point_count,
                original_size,
                model_size,
                compression_ratio,
            },
        }
    }

    /// Add residual data for lossless reconstruction
    pub fn with_residual(mut self, residual: Vec<u8>) -> Self {
        self.residual_blob = Some(residual);
        self
    }

    /// Check if a timestamp falls within this segment
    #[inline]
    pub fn contains(&self, timestamp: i64) -> bool {
        timestamp >= self.start_time && timestamp <= self.end_time
    }

    /// Check if a time range overlaps with this segment
    #[inline]
    pub fn overlaps(&self, start: i64, end: i64) -> bool {
        self.start_time <= end && self.end_time >= start
    }

    /// Query a single value at a specific timestamp
    ///
    /// This is O(1) - just evaluate the mathematical function!
    /// No disk I/O needed beyond loading the segment.
    #[inline(always)]
    pub fn query_point(&self, timestamp: i64) -> Option<f32> {
        if !self.contains(timestamp) {
            return None;
        }

        // Normalize timestamp to [0, 1] range using reciprocal multiply
        let range = (self.end_time - self.start_time) as f64;
        let x = if range > 0.0 {
            let inv_range = 1.0 / range;
            (timestamp - self.start_time) as f64 * inv_range
        } else {
            0.0
        };

        let value = self.evaluate_model_at_x(x);

        // Apply residual correction if available
        if let Some(ref residual) = self.residual_blob {
            let idx = self.timestamp_to_index(timestamp);
            if let Some(correction) = self.get_residual_at(residual, idx) {
                return Some(value + correction);
            }
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
        let inv_range = if total_range > 0.0 {
            1.0 / total_range
        } else {
            0.0
        };
        let estimated_count = ((query_end - query_start) as f64 / step) as usize + 1;

        // Pre-allocate with exact capacity
        let mut results = Vec::with_capacity(estimated_count.min(self.metadata.point_count));

        // ★ LOOP UNSWITCHING: Branch ONCE here, then run tight loop ★
        match &self.model {
            ModelType::Polynomial { coefficients, .. } => {
                self.query_loop_polynomial(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    coefficients,
                );
            }
            ModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => {
                self.query_loop_fourier(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    coefficients,
                    *dc_offset,
                    *sample_count,
                );
            }
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => {
                self.query_loop_sine(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    *frequency,
                    *amplitude,
                    *phase,
                    *offset,
                );
            }
            ModelType::MultiSine {
                components,
                dc_offset,
            } => {
                self.query_loop_multisine(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    components,
                    *dc_offset,
                );
            }
            ModelType::Constant { value } => {
                self.query_loop_constant(&mut results, query_start, query_end, step, *value as f32);
            }
            ModelType::Linear {
                start_value,
                end_value,
            } => {
                self.query_loop_linear(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    *start_value,
                    *end_value,
                );
            }
            ModelType::PerlinNoise { .. } | ModelType::RawLzma { .. } => {
                // Fallback for complex types - generate all and slice
                self.query_loop_fallback(&mut results, query_start, query_end, step, total_range);
            }
        }

        // Apply residuals in separate pass (if needed)
        if let Some(ref residual) = self.residual_blob {
            self.apply_residuals(&mut results, residual);
        }

        results
    }

    // =========================================================================
    // Loop Unswitched Query Implementations (No branches inside loops)
    // =========================================================================

    /// Polynomial query loop with SIMD (4 points at a time)
    #[inline]
    fn query_loop_polynomial(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        coefficients: &[f64],
    ) {
        let mut t = query_start as f64;
        let step4 = step * 4.0;
        let start_time = self.start_time as f64;

        // SIMD path: process 4 points at a time
        while t + step * 3.0 <= query_end as f64 {
            let t0 = t;
            let t1 = t + step;
            let t2 = t + step * 2.0;
            let t3 = t + step * 3.0;

            let x = f64x4::from([
                (t0 - start_time) * inv_range,
                (t1 - start_time) * inv_range,
                (t2 - start_time) * inv_range,
                (t3 - start_time) * inv_range,
            ]);

            // SIMD Horner's method
            let vals = self.horner_simd(x, coefficients);
            let v: [f64; 4] = vals.into();

            results.push((t0 as i64, v[0] as f32));
            results.push((t1 as i64, v[1] as f32));
            results.push((t2 as i64, v[2] as f32));
            results.push((t3 as i64, v[3] as f32));

            t += step4;
        }

        // Scalar tail
        while t <= query_end as f64 {
            let x = (t - start_time) * inv_range;
            let val = self.horner_scalar(x, coefficients);
            results.push((t as i64, val as f32));
            t += step;
        }
    }

    /// Scalar Horner's method (for tail processing)
    #[inline(always)]
    fn horner_scalar(&self, x: f64, coefficients: &[f64]) -> f64 {
        let mut result = 0.0f64;
        for &c in coefficients.iter().rev() {
            result = result * x + c;
        }
        result
    }

    /// Fourier query loop
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_fourier(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        coefficients: &[(usize, f32, f32)],
        dc_offset: f32,
        sample_count: usize,
    ) {
        let mut t = query_start as f64;
        let start_time = self.start_time as f64;
        let n = sample_count as f64;
        let two_pi_over_n = 2.0 * std::f64::consts::PI / n;

        while t <= query_end as f64 {
            let x = (t - start_time) * inv_range;
            let pos = x * n;
            let mut sum = dc_offset;

            for &(freq_idx, mag, phase) in coefficients {
                let angle = two_pi_over_n * freq_idx as f64 * pos;
                sum += mag * (angle as f32 + phase).cos();
            }

            results.push((t as i64, sum));
            t += step;
        }
    }

    /// Sine wave query loop
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_sine(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        frequency: f32,
        amplitude: f32,
        phase: f32,
        offset: f32,
    ) {
        let mut t = query_start as f64;
        let start_time = self.start_time as f64;
        let two_pi_freq = 2.0 * std::f32::consts::PI * frequency;

        while t <= query_end as f64 {
            let x = ((t - start_time) * inv_range) as f32;
            let angle = two_pi_freq * x + phase;
            let val = offset + amplitude * angle.sin();
            results.push((t as i64, val));
            t += step;
        }
    }

    /// Multi-sine query loop
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_multisine(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        components: &[(f32, f32, f32)],
        dc_offset: f32,
    ) {
        let mut t = query_start as f64;
        let start_time = self.start_time as f64;

        while t <= query_end as f64 {
            let x = ((t - start_time) * inv_range) as f32;
            let mut sum = dc_offset;

            for &(freq, amp, phase) in components {
                let angle = 2.0 * std::f32::consts::PI * freq * x + phase;
                sum += amp * angle.sin();
            }

            results.push((t as i64, sum));
            t += step;
        }
    }

    /// Constant query loop (trivial but kept for consistency)
    #[inline]
    fn query_loop_constant(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        value: f32,
    ) {
        let mut t = query_start as f64;
        while t <= query_end as f64 {
            results.push((t as i64, value));
            t += step;
        }
    }

    /// Linear query loop
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_linear(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        start_value: f64,
        end_value: f64,
    ) {
        let mut t = query_start as f64;
        let start_time = self.start_time as f64;
        let delta = end_value - start_value;

        while t <= query_end as f64 {
            let x = (t - start_time) * inv_range;
            let val = start_value + x * delta;
            results.push((t as i64, val as f32));
            t += step;
        }
    }

    /// Fallback query loop for complex types (Perlin, RawLzma)
    #[inline]
    fn query_loop_fallback(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        total_range: f64,
    ) {
        // Generate all data once, then slice
        let all_data = self.generate_all();
        let mut t = query_start as f64;

        // Pre-compute reciprocal to replace division in loop
        let inv_total_range = if total_range > 0.0 {
            1.0 / total_range
        } else {
            0.0
        };
        let start_time = self.start_time;

        while t <= query_end as f64 {
            let timestamp = t as i64;
            let x = (timestamp - start_time) as f64 * inv_total_range;
            let idx = (x * (all_data.len().saturating_sub(1)) as f64).round() as usize;
            let val = all_data.get(idx).copied().unwrap_or(0.0);
            results.push((timestamp, val));
            t += step;
        }
    }

    /// Apply residual corrections in a separate pass
    #[inline]
    fn apply_residuals(&self, results: &mut [(i64, f32)], residual: &[u8]) {
        for (timestamp, value) in results.iter_mut() {
            let idx = self.timestamp_to_index(*timestamp);
            if let Some(correction) = self.get_residual_at(residual, idx) {
                *value += correction;
            }
        }
    }

    /// Query a range of values using SIMD acceleration (Phase 3)
    ///
    /// Now unified with query_range - this is an alias for compatibility.
    #[inline]
    pub fn query_range_simd(&self, start: i64, end: i64) -> Vec<(i64, f32)> {
        // query_range now uses SIMD internally for polynomial
        self.query_range(start, end)
    }

    /// SIMD Horner's method: evaluate polynomial at 4 x-values simultaneously
    #[inline(always)]
    fn horner_simd(&self, x: f64x4, coefficients: &[f64]) -> f64x4 {
        let mut result = f64x4::ZERO;
        for &c in coefficients.iter().rev() {
            let c_vec = f64x4::splat(c);
            result = result * x + c_vec;
        }
        result
    }

    /// Generate all data points in this segment
    ///
    /// This regenerates the entire dataset from the model.
    /// Used for full segment reads or verification.
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
                // Generate 1D Perlin noise by taking a slice
                generators::generate_perlin_advanced(
                    n,
                    1,
                    *seed,
                    *scale,
                    *octaves,
                    *persistence,
                    *lacunarity,
                )
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

    /// Evaluate model at normalized position x ∈ [0, 1]
    #[inline(always)]
    fn evaluate_model_at_x(&self, x: f64) -> f32 {
        match &self.model {
            ModelType::Polynomial { coefficients, .. } => {
                // Horner's method for polynomial evaluation
                let mut result = 0.0f64;
                for &c in coefficients.iter().rev() {
                    result = result * x + c;
                }
                result as f32
            }
            ModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => {
                // Evaluate Fourier series at position
                // Pre-compute inv_n to replace division in the coefficient loop
                let n = *sample_count;
                let inv_n = 1.0 / n as f64;
                let t = x * n as f64;
                let two_pi = 2.0 * std::f64::consts::PI;
                let mut sum = *dc_offset;
                for &(freq_idx, mag, phase) in coefficients {
                    let angle = two_pi * freq_idx as f64 * t * inv_n;
                    sum += mag * (angle as f32 + phase).cos();
                }
                sum
            }
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => {
                let angle = 2.0 * std::f32::consts::PI * frequency * x as f32 + phase;
                offset + amplitude * angle.sin()
            }
            ModelType::MultiSine {
                components,
                dc_offset,
            } => {
                let mut sum = *dc_offset;
                for &(freq, amp, phase) in components {
                    let angle = 2.0 * std::f32::consts::PI * freq * x as f32 + phase;
                    sum += amp * angle.sin();
                }
                sum
            }
            ModelType::Constant { value } => *value as f32,
            ModelType::Linear {
                start_value,
                end_value,
            } => (start_value + x * (end_value - start_value)) as f32,
            ModelType::PerlinNoise { .. } => {
                // For point query, generate and return single value
                // This is less efficient but maintains consistency
                let data = self.generate_all();
                let idx = (x * (data.len() - 1) as f64).round() as usize;
                data.get(idx).copied().unwrap_or(0.0)
            }
            ModelType::RawLzma {
                compressed_data,
                dtype,
                ..
            } => {
                let data = self.decompress_raw(compressed_data, *dtype, self.metadata.point_count);
                let idx = (x * (data.len().saturating_sub(1)) as f64).round() as usize;
                data.get(idx).copied().unwrap_or(0.0)
            }
        }
    }

    /// Convert timestamp to array index
    #[inline(always)]
    fn timestamp_to_index(&self, timestamp: i64) -> usize {
        let range = (self.end_time - self.start_time) as f64;
        if range <= 0.0 || self.metadata.point_count <= 1 {
            return 0;
        }
        let inv_range = 1.0 / range;
        let ratio = (timestamp - self.start_time) as f64 * inv_range;
        (ratio * (self.metadata.point_count - 1) as f64).round() as usize
    }

    /// Get residual correction at index
    fn get_residual_at(&self, residual: &[u8], idx: usize) -> Option<f32> {
        // Residual is stored as quantized LZMA-compressed data
        // For now, assume it's been decompressed and stored as f32
        let offset = idx * 4;
        if offset + 4 <= residual.len() {
            let bytes: [u8; 4] = residual[offset..offset + 4].try_into().ok()?;
            Some(f32::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Decompress raw LZMA data
    fn decompress_raw(&self, compressed: &[u8], dtype: DataType, count: usize) -> Vec<f32> {
        use std::io::Cursor;

        let mut decompressed = Vec::new();
        if lzma_rs::lzma_decompress(&mut Cursor::new(compressed), &mut decompressed).is_err() {
            return vec![0.0; count];
        }

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

    /// Serialize segment to bytes
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Deserialize segment from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        bincode::deserialize(data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Write segment to file
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let bytes = self.to_bytes()?;
        writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    /// Read segment from file
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;
        Self::from_bytes(&data)
    }

    /// Serialize to rkyv format (zero-copy compatible)
    pub fn to_rkyv_bytes(&self) -> io::Result<Vec<u8>> {
        rkyv::to_bytes::<_, 256>(self)
            .map(|v| v.to_vec())
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("rkyv serialize: {:?}", e),
                )
            })
    }

    /// Write segment to file in rkyv format
    pub fn write_rkyv<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let bytes = self.to_rkyv_bytes()?;
        std::fs::write(path, bytes)
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
    /// Memory-mapped file (best for large segments, lazy loading)
    Mmap(Arc<Mmap>),
    /// In-memory bytes (for freshly flushed MemTable data)
    Vec(Arc<Vec<u8>>),
    /// Static slice (for embedded/testing)
    Slice(&'static [u8]),
}

impl AsRef<[u8]> for SegmentSource {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m.as_ref(),
            Self::Vec(v) => v.as_slice(),
            Self::Slice(s) => s,
        }
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
        let file = File::open(path)?;

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
                    "segment file is too small ({} bytes) to contain a valid rkyv archive",
                    file_len
                ),
            ));
        }

        // SAFETY: The file descriptor is valid and the file has been confirmed
        // non-empty above. The resulting Mmap is stored in Arc<Mmap> inside
        // SegmentSource::Mmap and kept alive for the lifetime of the
        // SegmentView, ensuring the mapped pages are not unmapped while any
        // reference into them is live.
        let mmap = unsafe { Mmap::map(&file)? };
        let source = SegmentSource::Mmap(Arc::new(mmap));
        Self::from_source(source)
    }

    /// Create from in-memory bytes (Zero-Copy, no disk I/O)
    ///
    /// Best for: Freshly flushed MemTable data.
    /// Avoids waiting for disk write to complete before caching.
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
    pub fn from_arc_vec(data: Arc<Vec<u8>>) -> io::Result<Self> {
        let source = SegmentSource::Vec(data);
        Self::from_source(source)
    }

    /// Create from static slice (for embedded/testing)
    pub fn from_static(data: &'static [u8]) -> io::Result<Self> {
        let source = SegmentSource::Slice(data);
        Self::from_source(source)
    }

    /// Internal: Create SegmentView from any SegmentSource
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
                format!("rkyv validation: {:?}", e),
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
            &*(archived as *const ArchivedDataSegment)
        };

        Ok(Self { source, archived })
    }

    /// Get the underlying source type (for debugging/stats)
    pub fn source_type(&self) -> &'static str {
        match &self.source {
            SegmentSource::Mmap(_) => "mmap",
            SegmentSource::Vec(_) => "vec",
            SegmentSource::Slice(_) => "slice",
        }
    }

    /// Get start timestamp
    #[inline]
    pub fn start_time(&self) -> i64 {
        self.archived.start_time
    }

    /// Get end timestamp
    #[inline]
    pub fn end_time(&self) -> i64 {
        self.archived.end_time
    }

    /// Check if timestamp is in range
    #[inline]
    pub fn contains(&self, timestamp: i64) -> bool {
        timestamp >= self.archived.start_time && timestamp <= self.archived.end_time
    }

    /// Zero-copy point query
    ///
    /// Computes f(x) directly from archived model coefficients.
    /// No deserialization occurs - we read directly from mmap.
    #[inline(always)]
    pub fn query_point(&self, timestamp: i64) -> Option<f32> {
        if !self.contains(timestamp) {
            return None;
        }

        let range = (self.archived.end_time - self.archived.start_time) as f64;
        let x = if range > 0.0 {
            let inv_range = 1.0 / range;
            (timestamp - self.archived.start_time) as f64 * inv_range
        } else {
            0.0
        };

        Some(self.evaluate_archived_model(x))
    }

    /// Zero-copy range query (Loop Unswitched + SIMD for Polynomial)
    ///
    /// # Performance: Same optimization as DataSegment
    ///
    /// Branch ONCE outside loop, then run tight branchless loops.
    /// Polynomial uses SIMD (f64x4) for 4x throughput.
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
        let inv_range = if total_range > 0.0 {
            1.0 / total_range
        } else {
            0.0
        };
        let estimated_count = ((query_end - query_start) as f64 / step) as usize + 1;

        let mut results = Vec::with_capacity(estimated_count.min(point_count as usize));

        // ★ LOOP UNSWITCHING: Branch ONCE here, then run tight loop ★
        match &self.archived.model {
            ArchivedModelType::Polynomial { coefficients, .. } => {
                self.query_loop_polynomial_archived(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    coefficients.as_slice(),
                );
            }
            ArchivedModelType::Constant { value } => {
                self.query_loop_constant_archived(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    *value as f32,
                );
            }
            ArchivedModelType::Linear {
                start_value,
                end_value,
            } => {
                self.query_loop_linear_archived(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    *start_value,
                    *end_value,
                );
            }
            ArchivedModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => {
                self.query_loop_sine_archived(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    *frequency,
                    *amplitude,
                    *phase,
                    *offset,
                );
            }
            ArchivedModelType::MultiSine {
                components,
                dc_offset,
            } => {
                self.query_loop_multisine_archived(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    components,
                    *dc_offset,
                );
            }
            ArchivedModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => {
                self.query_loop_fourier_archived(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    inv_range,
                    coefficients,
                    *dc_offset,
                    *sample_count,
                );
            }
            ArchivedModelType::PerlinNoise { .. } | ArchivedModelType::RawLzma { .. } => {
                // Fallback for complex types
                self.query_loop_fallback_archived(
                    &mut results,
                    query_start,
                    query_end,
                    step,
                    total_range,
                );
            }
        }

        results
    }

    // =========================================================================
    // Loop Unswitched Query Implementations for Archived Types
    // =========================================================================

    /// Polynomial query loop with SIMD (4 points at a time) - Zero-Copy version
    #[inline]
    fn query_loop_polynomial_archived(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        coefficients: &[f64],
    ) {
        let mut t = query_start as f64;
        let step4 = step * 4.0;
        let start_time = self.archived.start_time as f64;

        // SIMD path: process 4 points at a time
        while t + step * 3.0 <= query_end as f64 {
            let t0 = t;
            let t1 = t + step;
            let t2 = t + step * 2.0;
            let t3 = t + step * 3.0;

            let x = f64x4::from([
                (t0 - start_time) * inv_range,
                (t1 - start_time) * inv_range,
                (t2 - start_time) * inv_range,
                (t3 - start_time) * inv_range,
            ]);

            // SIMD Horner's method
            let vals = Self::horner_simd_static(x, coefficients);
            let v: [f64; 4] = vals.into();

            results.push((t0 as i64, v[0] as f32));
            results.push((t1 as i64, v[1] as f32));
            results.push((t2 as i64, v[2] as f32));
            results.push((t3 as i64, v[3] as f32));

            t += step4;
        }

        // Scalar tail
        while t <= query_end as f64 {
            let x = (t - start_time) * inv_range;
            let val = Self::horner_scalar_static(x, coefficients);
            results.push((t as i64, val as f32));
            t += step;
        }
    }

    /// SIMD Horner's method (static version for SegmentView)
    #[inline(always)]
    fn horner_simd_static(x: f64x4, coefficients: &[f64]) -> f64x4 {
        let mut result = f64x4::ZERO;
        for &c in coefficients.iter().rev() {
            let c_vec = f64x4::splat(c);
            result = result * x + c_vec;
        }
        result
    }

    /// Scalar Horner's method (static version for SegmentView)
    #[inline(always)]
    fn horner_scalar_static(x: f64, coefficients: &[f64]) -> f64 {
        let mut result = 0.0f64;
        for &c in coefficients.iter().rev() {
            result = result * x + c;
        }
        result
    }

    /// Constant query loop - Zero-Copy version
    #[inline]
    fn query_loop_constant_archived(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        value: f32,
    ) {
        let mut t = query_start as f64;
        while t <= query_end as f64 {
            results.push((t as i64, value));
            t += step;
        }
    }

    /// Linear query loop - Zero-Copy version
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_linear_archived(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        start_value: f64,
        end_value: f64,
    ) {
        let mut t = query_start as f64;
        let start_time = self.archived.start_time as f64;
        let delta = end_value - start_value;

        while t <= query_end as f64 {
            let x = (t - start_time) * inv_range;
            let val = start_value + x * delta;
            results.push((t as i64, val as f32));
            t += step;
        }
    }

    /// Sine wave query loop - Zero-Copy version
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_sine_archived(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        frequency: f32,
        amplitude: f32,
        phase: f32,
        offset: f32,
    ) {
        let mut t = query_start as f64;
        let start_time = self.archived.start_time as f64;
        let two_pi_freq = 2.0 * std::f32::consts::PI * frequency;

        while t <= query_end as f64 {
            let x = ((t - start_time) * inv_range) as f32;
            let angle = two_pi_freq * x + phase;
            let val = offset + amplitude * angle.sin();
            results.push((t as i64, val));
            t += step;
        }
    }

    /// Multi-sine query loop - Zero-Copy version
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_multisine_archived(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        components: &rkyv::vec::ArchivedVec<(f32, f32, f32)>,
        dc_offset: f32,
    ) {
        let mut t = query_start as f64;
        let start_time = self.archived.start_time as f64;

        while t <= query_end as f64 {
            let x = ((t - start_time) * inv_range) as f32;
            let mut sum = dc_offset;

            for comp in components.iter() {
                let (freq, amp, phase) = (comp.0, comp.1, comp.2);
                let angle = 2.0 * std::f32::consts::PI * freq * x + phase;
                sum += amp * angle.sin();
            }

            results.push((t as i64, sum));
            t += step;
        }
    }

    /// Fourier query loop - Zero-Copy version
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn query_loop_fourier_archived(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        inv_range: f64,
        coefficients: &rkyv::vec::ArchivedVec<(u32, f32, f32)>,
        dc_offset: f32,
        sample_count: u32,
    ) {
        let mut t = query_start as f64;
        let start_time = self.archived.start_time as f64;
        let n = sample_count as f64;
        let two_pi_over_n = 2.0 * std::f64::consts::PI / n;

        while t <= query_end as f64 {
            let x = (t - start_time) * inv_range;
            let pos = x * n;
            let mut sum = dc_offset;

            for coeff in coefficients.iter() {
                let (freq_idx, mag, phase) = (coeff.0, coeff.1, coeff.2);
                let angle = two_pi_over_n * freq_idx as f64 * pos;
                sum += mag * (angle as f32 + phase).cos();
            }

            results.push((t as i64, sum));
            t += step;
        }
    }

    /// Fallback query loop for complex types - Zero-Copy version
    #[inline]
    fn query_loop_fallback_archived(
        &self,
        results: &mut Vec<(i64, f32)>,
        query_start: i64,
        query_end: i64,
        step: f64,
        total_range: f64,
    ) {
        let mut t = query_start as f64;

        // Pre-compute reciprocal to replace division in loop
        let inv_total_range = if total_range > 0.0 {
            1.0 / total_range
        } else {
            0.0
        };
        let start_time = self.archived.start_time;

        while t <= query_end as f64 {
            let timestamp = t as i64;
            let x = (timestamp - start_time) as f64 * inv_total_range;

            let value = self.evaluate_archived_model(x);
            results.push((timestamp, value));
            t += step;
        }
    }

    /// Evaluate archived model at normalized x
    #[inline(always)]
    fn evaluate_archived_model(&self, x: f64) -> f32 {
        match &self.archived.model {
            ArchivedModelType::Polynomial { coefficients, .. } => {
                // Zero-copy access to coefficient slice
                let coeffs = coefficients.as_slice();
                let mut result = 0.0f64;
                for &c in coeffs.iter().rev() {
                    result = result * x + c;
                }
                result as f32
            }
            ArchivedModelType::Constant { value } => *value as f32,
            ArchivedModelType::Linear {
                start_value,
                end_value,
            } => (start_value + x * (end_value - start_value)) as f32,
            ArchivedModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => {
                let angle = 2.0 * std::f32::consts::PI * frequency * x as f32 + phase;
                offset + amplitude * angle.sin()
            }
            ArchivedModelType::MultiSine {
                components,
                dc_offset,
            } => {
                let mut sum = *dc_offset;
                for comp in components.iter() {
                    let (freq, amp, phase) = (comp.0, comp.1, comp.2);
                    let angle = 2.0 * std::f32::consts::PI * freq * x as f32 + phase;
                    sum += amp * angle.sin();
                }
                sum
            }
            ArchivedModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => {
                // Pre-compute inv_n to replace division in the coefficient loop
                let n = *sample_count;
                let inv_n = 1.0 / n as f64;
                let t = x * n as f64;
                let two_pi = 2.0 * std::f64::consts::PI;
                let mut sum = *dc_offset;
                for coeff in coefficients.iter() {
                    let (freq_idx, mag, phase) = (coeff.0, coeff.1, coeff.2);
                    let angle = two_pi * freq_idx as f64 * t * inv_n;
                    sum += mag * (angle as f32 + phase).cos();
                }
                sum
            }
            ArchivedModelType::PerlinNoise { .. } | ArchivedModelType::RawLzma { .. } => {
                // These require full deserialization, fallback to 0
                // In production, would deserialize and compute
                0.0
            }
        }
    }

    /// Get compression ratio from metadata
    pub fn compression_ratio(&self) -> f64 {
        self.archived.metadata.compression_ratio
    }

    /// Get point count from metadata
    pub fn point_count(&self) -> usize {
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
        // y = x^2 on [0, 1] → values from 0 to 1
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
        // x = 0.5, y = 0.25
        assert!((at_half - 0.25).abs() < 0.01);
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
        let segment_with_res = segment.with_residual(residual_data.clone());
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
}
