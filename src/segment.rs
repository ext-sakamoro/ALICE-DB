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
//! # Stored segments
//!
//! A segment file holds one [`DataSegment`] in the stored segment format
//! ([`SEGMENT_FORMAT`]: a header, then an rkyv archive). [`SegmentView`]
//! decodes a file once into an owned [`DataSegment`] and answers queries with
//! the same methods as an in-memory segment, so both read through one law
//! path. An open view holds the decoded segment (about the file's size).
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::model::{DataType, ModelType};
use alice_core::generators;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
#[cfg(feature = "fs")]
use std::fs::File;
use std::io::{self, Read, Write};
#[cfg(feature = "fs")]
use std::path::Path;
use std::sync::Arc;
use wide::f64x4;

/// The model laws — one function per model, shared by the point query, the
/// range query (in memory and through a [`SegmentView`], which decodes to the
/// same [`DataSegment`]) and `generate_all` (oracle: `tests/analytic_oracle.rs`).
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
    use alice_core::generators;

    /// Fractional sample position of `timestamp` in a segment of `n` samples
    /// spanning `start..=end` (uniform spacing).
    ///
    /// The differences are taken in `i128`: a segment may span more than
    /// `i64::MAX` (keys of both signs), where an `i64` subtraction panics in
    /// debug builds and wraps in release. Below 2^63 the result is the same
    /// value the `i64` form computed.
    #[inline(always)]
    pub(super) fn sample_pos(start: i64, end: i64, n: usize, timestamp: i64) -> f64 {
        let range = (i128::from(end) - i128::from(start)) as f64;
        if n <= 1 || range <= 0.0 {
            return 0.0;
        }
        (i128::from(timestamp) - i128::from(start)) as f64 / range * (n - 1) as f64
    }

    /// First grid point at or after `t` of a segment of `n` samples on the
    /// even grid `start + ⌊k·range/(n−1)⌋`, or `None` past the last one
    pub(super) fn grid_ceil(start: i64, end: i64, n: usize, t: i64) -> Option<i64> {
        if t <= start {
            return Some(start);
        }
        if t > end {
            return None;
        }
        let range = u128::try_from(i128::from(end) - i128::from(start)).unwrap_or(0);
        let nm1 = n.saturating_sub(1) as u128;
        if nm1 == 0 || range == 0 {
            return None;
        }
        // smallest k with ⌊k·range/(n−1)⌋ ≥ d is ⌈d·(n−1)/range⌉
        let d = u128::try_from(i128::from(t) - i128::from(start)).unwrap_or(0);
        let k = (d * nm1).div_ceil(range);
        if k > nm1 {
            return None;
        }
        Some((i128::from(start) + (k * range / nm1) as i128) as i64)
    }

    /// The grid points of a segment inside a query, walked by integer index
    ///
    /// A segment holds `n` samples on an even grid from `start` to `end`.
    /// A range read emits the points `t_k = query_start + ⌊k·range/(n−1)⌋`
    /// for `k = 0, 1, …` while `t_k <= query_end`, so it emits at most `n`
    /// points and always terminates. Until 0.3.0 it stepped an `f64`
    /// timestamp by `range/(n−1)`; above 2^53 a step smaller than half an
    /// ulp left the timestamp unchanged and the loop never ended.
    ///
    /// The value at `t_k` is the model at sample position
    /// `(t_k − start) · (n−1)/range`, computed with the same `f64` operations
    /// as before whenever the old loop was exact (integer step, keys below
    /// 2^53), so the bits read back there do not change
    /// (`tests/determinism_golden.rs`).
    pub(super) struct RangePlan {
        /// First timestamp (already clamped to the segment)
        query_start: i64,
        /// `query_start − start`, ≥ 0
        offset: u128,
        /// `end − start`, ≥ 0
        range: u128,
        /// `n − 1`
        nm1: u128,
        /// `1 / step` (0 for a single point)
        scale: f64,
        /// Number of points emitted
        pub(super) count: u64,
    }

    impl RangePlan {
        /// `query_start..=query_end` must lie inside `start..=end`
        pub(super) fn new(
            start: i64,
            end: i64,
            n: usize,
            query_start: i64,
            query_end: i64,
        ) -> Self {
            let range = u128::try_from(i128::from(end) - i128::from(start)).unwrap_or(0);
            let offset = u128::try_from(i128::from(query_start) - i128::from(start)).unwrap_or(0);
            let span = u128::try_from(i128::from(query_end) - i128::from(query_start)).unwrap_or(0);
            let nm1 = n.saturating_sub(1) as u128;
            let range_f = range as f64;
            let step = if n > 1 && range_f > 0.0 {
                range_f / nm1 as f64
            } else {
                1.0
            };
            let scale = if n <= 1 || step <= 0.0 {
                0.0
            } else {
                1.0 / step
            };
            // ⌊span·(n−1)/range⌋ ≤ n−1; span, n−1 < 2^64 so the product fits
            let count = if n > 1 && range > 0 {
                (span * nm1 / range) as u64 + 1
            } else {
                1
            };
            Self {
                query_start,
                offset,
                range,
                nm1,
                scale,
                count,
            }
        }

        /// Timestamp of grid point `k` (exact)
        #[inline(always)]
        pub(super) fn timestamp(&self, k: u64) -> i64 {
            if self.nm1 == 0 || self.range == 0 {
                return self.query_start;
            }
            // k·range ≤ (n−1)·range < 2^128; the result is ≤ query_end
            let d = u128::from(k) * self.range / self.nm1;
            (i128::from(self.query_start) + d as i128) as i64
        }

        /// Sample position of grid point `k`
        #[inline(always)]
        pub(super) fn pos(&self, k: u64) -> f64 {
            if self.nm1 == 0 || self.range == 0 {
                return 0.0;
            }
            // (t_k − start) as an exact fraction: (offset·(n−1) + k·range)/(n−1),
            // ≤ range·(n−1) < 2^128
            let num = self.offset * self.nm1 + u128::from(k) * self.range;
            (num as f64 / self.nm1 as f64) * self.scale
        }
    }

    /// `Σ cⱼ·iʲ` — delegates to the law in `alice_core`
    ///
    /// ⚠️ Horner with separate multiply and add, **not** `mul_add`. Until
    /// 0.3.0 this was a local copy that used `mul_add`; the fused form rounds
    /// once instead of twice, so it is a different law wearing the same name.
    /// `alice_core::law::SignalLaw::law_id` identifies the non-fused form.
    #[inline(always)]
    pub(super) fn polynomial_at(coefficients: &[f64], i: f64) -> f32 {
        generators::polynomial_at(coefficients, i)
    }

    /// `dc + Σ w(k)·mag/n · cos(2πk·i/n + φ)` — delegates to the law in `alice_core`
    #[inline(always)]
    pub(super) fn fourier_at(
        coefficients: &[(usize, f32, f32)],
        dc_offset: f32,
        n: usize,
        i: f64,
    ) -> f32 {
        generators::fourier_at(n, coefficients, dc_offset, i)
    }

    /// `offset + A·sin(2πf·i/n + φ)` — delegates to the law in `alice_core`
    #[inline(always)]
    pub(super) fn sine_at(
        frequency: f32,
        amplitude: f32,
        phase: f32,
        offset: f32,
        n: usize,
        i: f64,
    ) -> f32 {
        generators::sine_at(n, frequency, amplitude, phase, offset, i)
    }

    /// `dc + Σⱼ Aⱼ·sin(2πfⱼ·i/n + φⱼ)` — delegates to the law in `alice_core`
    #[inline(always)]
    pub(super) fn multisine_at(
        components: &[(f32, f32, f32)],
        dc_offset: f32,
        n: usize,
        i: f64,
    ) -> f32 {
        generators::multi_sine_at(n, components, dc_offset, i)
    }

    #[inline(always)]
    pub(super) fn linear_at(start_value: f64, end_value: f64, n: usize, i: f64) -> f32 {
        let x = if n <= 1 { 0.0 } else { i / (n - 1) as f64 };
        (start_value + x * (end_value - start_value)) as f32
    }

    /// Materialised (non-analytic) models: nearest stored sample.
    #[inline(always)]
    pub(super) fn sampled_at(data: &[f32], i: f64) -> f32 {
        let idx = alice_det_math::max64(i.round(), 0.0) as usize;
        data.get(idx.min(data.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
    }

    /// Fill `results` with `(t, eval(i))` for every grid point of `plan`
    /// (branch on the model once, outside this loop).
    #[inline(always)]
    pub(super) fn fill_range(
        results: &mut Vec<(i64, f32)>,
        plan: &RangePlan,
        first: u64,
        eval: impl Fn(f64) -> f32,
    ) {
        for k in first..plan.count {
            results.push((plan.timestamp(k), eval(plan.pos(k))));
        }
    }
}

/// Segment metadata
/// Magic at the start of a stored segment file of format 1 or later
const SEGMENT_MAGIC: &[u8; 8] = b"ALICESEG";
/// Length of the stored segment header: magic, format (`u32` LE), 4 reserved zero bytes
const SEGMENT_HEADER_LEN: usize = 16;
/// The stored segment format this version writes: 1 = the header, then an rkyv 0.8 archive
/// of [`DataSegment`]. Format 0 (no header, an rkyv 0.7 archive) is what releases before
/// 0.6.0 wrote; it is still read.
pub const SEGMENT_FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[rkyv(derive(Debug))]
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
#[rkyv(derive(Debug))]
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
    ///
    /// `created_at` is a Unix-millis metadata timestamp stamped verbatim
    /// into the stored segment (see [`SegmentMetadata::created_at`]); this
    /// crate never reads the wall clock itself (G10), so the caller supplies
    /// whatever value it wants recorded (or `0`).
    #[must_use]
    pub fn new(
        id: u64,
        start_time: i64,
        end_time: i64,
        model: ModelType,
        point_count: usize,
        original_size: usize,
        created_at: u64,
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
                created_at,
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
    ///
    /// # Errors
    ///
    /// [`ResidualError::Mismatch`] when the segment carries a residual pinned
    /// to a different arithmetic (applying it would give a value that is
    /// neither the original nor the model), [`ResidualError::Corrupt`] when
    /// its residual does not decode to one correction per point; returning
    /// the model alone would give a value that is not the original without
    /// saying so.
    /// [`Self::query_point_model_only`] is the explicit way to read the model.
    #[inline(always)]
    pub fn query_point(&self, timestamp: i64) -> Result<Option<f32>, ResidualError> {
        self.read_point(timestamp, UnpinnedResiduals::Refuse)
            .map(|read| read.map(|(value, _)| value))
    }

    /// [`Self::query_point`] that also says what the value rests on, and whether a residual
    /// written before residuals carried an identifier ([`ResidualSemantics::Unpinned`]) is
    /// applied ([`UnpinnedResiduals::Accept`]) or refused with [`ResidualError::Unpinned`]
    ///
    /// # Errors
    ///
    /// As [`Self::query_point`], and [`ResidualError::Unpinned`] for a residual without an
    /// identifier under [`UnpinnedResiduals::Refuse`].
    pub fn read_point(
        &self,
        timestamp: i64,
        unpinned: UnpinnedResiduals,
    ) -> Result<Option<(f32, ReadStatus)>, ResidualError> {
        let Some(value) = self.model_point(timestamp) else {
            return Ok(None);
        };
        let Some(residual) = checked_residual(
            self.residual_blob.as_deref(),
            self.metadata.point_count,
            self.metadata.id,
            self.start_time,
            self.end_time,
            unpinned,
        )?
        else {
            // no residual: the model's value, a computed one
            return Ok(Some((canon32(value), ReadStatus::Current)));
        };
        let (kind, decompressed, status) = residual;
        let idx = self.timestamp_to_index(timestamp);
        apply_residual(value, kind, &decompressed, idx)
            .map(|value| Some((value, status)))
            .ok_or_else(|| {
                missing_word(
                    self.metadata.id,
                    self.start_time,
                    self.end_time,
                    self.metadata.point_count,
                    decompressed.len(),
                )
            })
    }

    /// The model's value at `timestamp`, with no residual applied
    ///
    /// For a lossless segment this is **not** the stored original: it is the
    /// law's approximation, off from the original by the residual. It is the
    /// explicit read for a segment whose residual belongs to another
    /// arithmetic (see [`Self::query_point`] and [`Self::residual_status`]).
    #[inline(always)]
    #[must_use]
    pub(crate) fn model_point(&self, timestamp: i64) -> Option<f32> {
        if !self.contains(timestamp) {
            return None;
        }
        let i = law::sample_pos(
            self.start_time,
            self.end_time,
            self.metadata.point_count,
            timestamp,
        );
        Some(self.evaluate_model_at(i))
    }

    /// The model's value at `timestamp` with no residual applied, a NaN as the canonical
    /// `f32::NAN` (`0x7fc0_0000`): a computed value (ALICE-Law G4). The residual is applied
    /// to the uncanonicalised value, so a lossless read returns the stored bits
    #[must_use]
    pub fn query_point_model_only(&self, timestamp: i64) -> Option<f32> {
        self.model_point(timestamp).map(canon32)
    }

    /// Whether this segment's values come back exactly, and if not why
    #[must_use]
    pub fn residual_status(&self) -> ResidualStatus {
        residual_status(self.residual_blob.as_deref(), self.metadata.point_count)
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
    ///
    /// # Errors
    ///
    /// [`ResidualError`], as [`Self::query_point`]: the whole range or
    /// nothing, never a shorter or partly corrected result.
    pub fn query_range(&self, start: i64, end: i64) -> Result<Vec<(i64, f32)>, ResidualError> {
        self.read_range(start, end, UnpinnedResiduals::Refuse)
            .map(|(points, _)| points)
    }

    /// [`Self::query_range`] that also says what the values rest on (see [`Self::read_point`])
    ///
    /// # Errors
    ///
    /// As [`Self::read_point`].
    pub fn read_range(
        &self,
        start: i64,
        end: i64,
        unpinned: UnpinnedResiduals,
    ) -> Result<(Vec<(i64, f32)>, ReadStatus), ResidualError> {
        let residual = checked_residual(
            self.residual_blob.as_deref(),
            self.metadata.point_count,
            self.metadata.id,
            self.start_time,
            self.end_time,
            unpinned,
        )?;
        let mut results = self.model_range(start, end);
        let mut status = ReadStatus::Current;
        if let Some((kind, decompressed, residual_status)) = residual {
            status = residual_status;
            for (timestamp, value) in &mut results {
                let idx = self.timestamp_to_index(*timestamp);
                *value = apply_residual(*value, kind, &decompressed, idx).ok_or_else(|| {
                    missing_word(
                        self.metadata.id,
                        self.start_time,
                        self.end_time,
                        self.metadata.point_count,
                        decompressed.len(),
                    )
                })?;
            }
        } else {
            // no residual: the model's values, computed ones
            for (_, value) in &mut results {
                *value = canon32(*value);
            }
        }
        Ok((results, status))
    }

    /// The model's values over `[start, end]`, with no residual applied (see
    /// [`Self::query_point_model_only`])
    #[must_use]
    pub(crate) fn model_range(&self, start: i64, end: i64) -> Vec<(i64, f32)> {
        let query_start = start.max(self.start_time);
        let query_end = end.min(self.end_time);

        if query_start > query_end {
            return Vec::new();
        }

        // Integer grid walk: at most `point_count` points (see `law::RangePlan`)
        let n = self.metadata.point_count;
        let plan = law::RangePlan::new(self.start_time, self.end_time, n, query_start, query_end);

        // Pre-allocate with exact capacity
        let mut results =
            Vec::with_capacity(usize::try_from(plan.count).unwrap_or(n).min(n.max(1)));

        // Branch on the model ONCE, then run one tight loop over the law
        match &self.model {
            ModelType::Polynomial { coefficients, .. } => {
                Self::fill_range_polynomial(&mut results, &plan, coefficients);
            }
            ModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => law::fill_range(&mut results, &plan, 0, |i| {
                law::fourier_at(coefficients, *dc_offset, *sample_count, i)
            }),
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => law::fill_range(&mut results, &plan, 0, |i| {
                law::sine_at(*frequency, *amplitude, *phase, *offset, n, i)
            }),
            ModelType::MultiSine {
                components,
                dc_offset,
            } => law::fill_range(&mut results, &plan, 0, |i| {
                law::multisine_at(components, *dc_offset, n, i)
            }),
            ModelType::Constant { value } => {
                let v = *value as f32;
                law::fill_range(&mut results, &plan, 0, |_| v);
            }
            ModelType::Linear {
                start_value,
                end_value,
            } => law::fill_range(&mut results, &plan, 0, |i| {
                law::linear_at(*start_value, *end_value, n, i)
            }),
            ModelType::PerlinNoise { .. } | ModelType::RawLzma { .. } => {
                // Non-analytic models: materialise once, then index
                let all = self.generate_all_raw();
                law::fill_range(&mut results, &plan, 0, |i| law::sampled_at(&all, i));
            }
        }

        results
    }

    /// The model's values over `[start, end]` with no residual applied, a NaN as the canonical
    /// `f32::NAN` (see [`Self::query_point_model_only`])
    #[must_use]
    pub fn query_range_model_only(&self, start: i64, end: i64) -> Vec<(i64, f32)> {
        self.model_range(start, end)
            .into_iter()
            .map(|(t, v)| (t, canon32(v)))
            .collect()
    }

    /// Polynomial range loop: 4 sample positions per `f64x4` Horner step, then
    /// a scalar tail — the same law as [`law::polynomial_at`].
    #[inline]
    fn fill_range_polynomial(
        results: &mut Vec<(i64, f32)>,
        plan: &law::RangePlan,
        coefficients: &[f64],
    ) {
        let mut k = 0u64;
        while k + 4 <= plan.count {
            let i = f64x4::from([
                plan.pos(k),
                plan.pos(k + 1),
                plan.pos(k + 2),
                plan.pos(k + 3),
            ]);
            let v: [f64; 4] = horner_simd(i, coefficients).into();
            for (j, &value) in (0u64..).zip(v.iter()) {
                results.push((plan.timestamp(k + j), value as f32));
            }
            k += 4;
        }
        law::fill_range(results, plan, k, |i| law::polynomial_at(coefficients, i));
    }

    /// Query a range of values using SIMD acceleration (Phase 3)
    ///
    /// Now unified with `query_range` - this is an alias for compatibility.
    ///
    /// # Errors
    ///
    /// [`ResidualError`], as [`Self::query_range`].
    #[inline]
    pub fn query_range_simd(&self, start: i64, end: i64) -> Result<Vec<(i64, f32)>, ResidualError> {
        // query_range now uses SIMD internally for polynomial
        self.query_range(start, end)
    }

    /// Generate all data points in this segment
    ///
    /// This regenerates the entire dataset from the model.
    /// Used for full segment reads or verification.
    #[must_use]
    pub(crate) fn generate_all_raw(&self) -> Vec<f32> {
        let n = self.metadata.point_count;

        match &self.model {
            // Materialised (non-analytic) models have no law to evaluate at a
            // position; they *are* the samples.
            ModelType::PerlinNoise {
                seed,
                scale,
                octaves,
                persistence,
                lacunarity,
            } => {
                // 1D value-noise fBm (alice-zip `generate_fbm_1d`, the law
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
            } => self.decompress_raw(compressed_data, *dtype, n),

            // Every analytic model is its law evaluated at `i = 0 ..= n-1`.
            //
            // ⚠️ Until 0.3.0 this arm called the *array* generators in
            // `alice_core` while `query_point` / `query_range` called the point
            // law, and the two disagreed: on the same segment 544 of 1206
            // compared samples came back with different bits (`SineWave` 168,
            // `MultiSine` 181, `Fourier` 195). `query_range` and `query_point`
            // agreed with each other, so this arm was the one out of step —
            // and it is `pub`, so the disagreement was observable from outside
            // the crate. `MemTable::seal` measures its residual against
            // `query_point`, which makes the point law the one the stored data
            // is anchored to. `tests/law_single_source.rs` pins all three
            // paths to the same bits.
            _ => (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let pos = i as f64;
                    self.evaluate_model_at(pos)
                })
                .collect(),
        }
    }

    /// Every value of the model, a NaN as the canonical `f32::NAN` (a computed value; see
    /// [`Self::query_point_model_only`])
    #[must_use]
    pub fn generate_all(&self) -> Vec<f32> {
        self.generate_all_raw().into_iter().map(canon32).collect()
    }

    /// Evaluate the model at (fractional) sample position `i` — see [`law`]
    #[inline(always)]
    fn evaluate_model_at(&self, i: f64) -> f32 {
        let n = self.metadata.point_count;
        match &self.model {
            ModelType::Polynomial { coefficients, .. } => law::polynomial_at(coefficients, i),
            ModelType::Fourier {
                coefficients,
                dc_offset,
                sample_count,
            } => law::fourier_at(coefficients, *dc_offset, *sample_count, i),
            ModelType::SineWave {
                frequency,
                amplitude,
                phase,
                offset,
            } => law::sine_at(*frequency, *amplitude, *phase, *offset, n, i),
            ModelType::MultiSine {
                components,
                dc_offset,
            } => law::multisine_at(components, *dc_offset, n, i),
            ModelType::Constant { value } => *value as f32,
            ModelType::Linear {
                start_value,
                end_value,
            } => law::linear_at(*start_value, *end_value, n, i),
            ModelType::PerlinNoise { .. } | ModelType::RawLzma { .. } => {
                // Non-analytic models: materialise and index (point queries on
                // these are O(n); range queries materialise once)
                law::sampled_at(&self.generate_all_raw(), i)
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

    /// The bytes of a stored segment file (stored segment format
    /// [`SEGMENT_FORMAT`]): the 16-byte header (`ALICESEG`, the format as a
    /// little-endian `u32`, 4 reserved zero bytes), then the rkyv 0.8 archive.
    ///
    /// # Errors
    ///
    /// Returns an error if rkyv serialization fails.
    pub fn to_rkyv_bytes(&self) -> io::Result<Vec<u8>> {
        let archive = rkyv::to_bytes::<rkyv::rancor::Error>(self).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("rkyv serialize: {e}"))
        })?;
        let mut out = Vec::with_capacity(SEGMENT_HEADER_LEN + archive.len());
        out.extend_from_slice(SEGMENT_MAGIC);
        out.extend_from_slice(&SEGMENT_FORMAT.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&archive);
        Ok(out)
    }

    /// Decode a stored segment file's bytes into an owned segment: format
    /// [`SEGMENT_FORMAT`] (what [`Self::to_rkyv_bytes`] writes) through
    /// rkyv's checked deserialization, or format 0 (the files of earlier
    /// releases, which start without the header: an rkyv 0.7 archive)
    /// through a built-in reader of that layout.
    ///
    /// Decoding never allocates more than the input's length allows: the
    /// archive is copied once into an aligned buffer of its own size, and
    /// every length inside it is checked against the bytes present.
    ///
    /// # Errors
    ///
    /// Returns `InvalidData` when the bytes are not a valid stored segment,
    /// or carry a format newer than [`SEGMENT_FORMAT`].
    pub fn from_stored_bytes(bytes: &[u8]) -> io::Result<Self> {
        let Some(rest) = bytes.strip_prefix(SEGMENT_MAGIC.as_slice()) else {
            return crate::segment_v07::decode(bytes);
        };
        if rest.len() < SEGMENT_HEADER_LEN - SEGMENT_MAGIC.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stored segment: the header is cut short",
            ));
        }
        let format = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        if format != SEGMENT_FORMAT || rest[4..8] != [0; 4] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "stored segment format {format} is not one this reader knows (it reads 0 and {SEGMENT_FORMAT})"
                ),
            ));
        }
        let mut archive = rkyv::util::AlignedVec::<16>::with_capacity(rest.len() - 8);
        archive.extend_from_slice(&rest[8..]);
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(&archive).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("rkyv validation: {e}"))
        })
    }
    /// Write segment to file in rkyv format
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing to disk fails.
    #[cfg(feature = "fs")]
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
/// Magic byte: LZMA-compressed XOR residual **carrying the arithmetic identifier**
const RESIDUAL_XOR_PINNED_LZMA: u8 = 6;
/// Magic byte: raw XOR residual carrying the arithmetic identifier
const RESIDUAL_XOR_PINNED_RAW: u8 = 7;
/// Length of the arithmetic identifier a pinned residual carries
const SEMANTICS_ID_LEN: usize = 32;

/// Which arithmetic the model values behind a residual were produced with.
///
/// A residual stores `original ^ model`, so it only reconstructs the original
/// when the reader evaluates the law with the **same arithmetic** the writer
/// used. IEEE 754 does not require `sin` / `cos` to be correctly rounded, so
/// "the same law" is not enough — the implementation of the transcendentals is
/// part of what the residual is anchored to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualSemantics {
    /// Written before 0.3.0, with no record of the arithmetic used.
    ///
    /// The model values came from the platform libm of whichever machine wrote
    /// the segment, so the residual reconstructs the original **only on a
    /// machine whose libm agrees with that one**. Nothing in the blob says
    /// which machine that was, so this cannot be checked — reading such a
    /// segment elsewhere can silently return values off in the last place.
    Unpinned,
    /// The model values were produced with the arithmetic this identifier names.
    ///
    /// Check it with `alice_core::law::check_semantics`: [`Verified`] or
    /// [`VerifiedPrevious`] means the residual reconstructs the original
    /// exactly on any target (a release that only adds functions still
    /// reproduces data an earlier release stamped), [`Mismatch`] means it
    /// would not, and that is detectable instead of silent.
    ///
    /// [`Verified`]: alice_det_math::SemanticsCheck::Verified
    /// [`VerifiedPrevious`]: alice_det_math::SemanticsCheck::VerifiedPrevious
    /// [`Mismatch`]: alice_det_math::SemanticsCheck::Mismatch
    Pinned([u8; SEMANTICS_ID_LEN]),
}

/// The arithmetic a residual blob is anchored to — see [`ResidualSemantics`].
#[must_use]
pub fn residual_semantics(blob: &[u8]) -> ResidualSemantics {
    match blob.first() {
        Some(&RESIDUAL_XOR_PINNED_LZMA | &RESIDUAL_XOR_PINNED_RAW)
            if blob.len() > SEMANTICS_ID_LEN =>
        {
            let mut id = [0_u8; SEMANTICS_ID_LEN];
            id.copy_from_slice(&blob[1..=SEMANTICS_ID_LEN]);
            ResidualSemantics::Pinned(id)
        }
        _ => ResidualSemantics::Unpinned,
    }
}

/// Whether the default read ([`UnpinnedResiduals::Refuse`]) applies this residual:
/// whether applying it with the arithmetic in use is known to reconstruct the
/// original, rather than a value that is neither the original nor the model.
///
/// A [`ResidualSemantics::Pinned`] blob answers `true` when its identifier is
/// [`alice_core::law::SEMANTICS_ID`] **or** one of
/// [`alice_core::law::PREVIOUS_SEMANTICS_IDS`] (`alice_core::law::check_semantics`)
/// — a release that only adds functions still reconstructs residuals an
/// earlier release stamped. [`ResidualSemantics::Unpinned`] answers `false`:
/// nothing records the arithmetic it was written under, so the default read
/// refuses it ([`ResidualError::Unpinned`]); [`UnpinnedResiduals::Accept`]
/// applies it on request, reporting [`ReadStatus::Unpinned`].
#[must_use]
pub fn residual_is_applicable(blob: &[u8]) -> bool {
    match residual_semantics(blob) {
        ResidualSemantics::Unpinned => false,
        ResidualSemantics::Pinned(id) => alice_core::law::check_semantics(&id).is_reproducible(),
    }
}

/// A residual pinned to a different arithmetic than the one in use, met while
/// reading values
///
/// The residual stores `original ^ model` under the arithmetic named by
/// `stored`. With another arithmetic the model values may differ, so applying
/// the residual can give a value that is neither the original nor the model,
/// and returning the model alone gives a value that is not the original; a
/// read refuses instead. The fields say which data it is: the segment and the
/// key range it covers.
///
/// The exact values come back only with the arithmetic `stored` names (for a
/// store written with `alice-det-math` 0.4, read it with `alice-db` 0.3 and
/// write the values again). [`DataSegment::query_point_model_only`] and
/// `ResidualPolicy::ModelOnly` read the model's approximation explicitly.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResidualMismatch {
    /// Id of the segment whose residual was refused
    pub segment_id: u64,
    /// First key the segment covers
    pub start_time: i64,
    /// Last key the segment covers
    pub end_time: i64,
    /// Identifier of the arithmetic the residual was measured against
    pub stored: [u8; SEMANTICS_ID_LEN],
    /// Identifier of the arithmetic this build computes with
    pub running: [u8; SEMANTICS_ID_LEN],
    /// The store the segment belongs to (its data directory), when the read
    /// went through a store; `None` for a segment read on its own
    pub series: Option<std::path::PathBuf>,
}

impl std::fmt::Display for ResidualMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hex = |id: &[u8; SEMANTICS_ID_LEN]| {
            id.iter()
                .fold(String::with_capacity(2 * SEMANTICS_ID_LEN), |mut out, b| {
                    let _ = std::fmt::Write::write_fmt(&mut out, format_args!("{b:02x}"));
                    out
                })
        };
        write!(
            f,
            "segment {} (keys {}..={}) holds a residual measured against arithmetic {}, \
             but this build computes with {}: its exact values cannot be reconstructed here",
            self.segment_id,
            self.start_time,
            self.end_time,
            hex(&self.stored),
            hex(&self.running)
        )?;
        if let Some(series) = &self.series {
            write!(f, " (store {})", series.display())?;
        }
        Ok(())
    }
}

impl std::error::Error for ResidualMismatch {}

/// A residual whose bytes cannot be decoded into one correction per point
/// (a payload that does not decompress, a truncated identifier, an unknown
/// format), met while reading values
///
/// Without the corrections a lossless segment can only return the model,
/// which is not what was stored; a read refuses instead.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResidualCorrupt {
    /// Id of the segment whose residual was refused
    pub segment_id: u64,
    /// First key the segment covers
    pub start_time: i64,
    /// Last key the segment covers
    pub end_time: i64,
    /// Decoded residual bytes the segment's points need (4 per point)
    pub needed_bytes: usize,
    /// Decoded residual bytes the blob held
    pub found_bytes: usize,
    /// The store the segment belongs to (its data directory), when the read
    /// went through a store; `None` for a segment read on its own
    pub series: Option<std::path::PathBuf>,
}

impl std::fmt::Display for ResidualCorrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "segment {} (keys {}..={}) holds a residual that decodes to {} bytes, \
             {} are needed: its exact values cannot be reconstructed",
            self.segment_id, self.start_time, self.end_time, self.found_bytes, self.needed_bytes
        )?;
        if let Some(series) = &self.series {
            write!(f, " (store {})", series.display())?;
        }
        Ok(())
    }
}

impl std::error::Error for ResidualCorrupt {}

/// Why a read refused a segment's lossless residual
/// A residual written before residuals carried an identifier (`alice-db` before 0.3.0),
/// met by a read that does not accept one ([`UnpinnedResiduals::Refuse`], the default)
///
/// Such a residual stores `original ^ model` (or `original - model`) under an arithmetic
/// nothing records: applied with this build's arithmetic it may give a value that is
/// neither the original nor the model. A read that wants it anyway says so with
/// [`UnpinnedResiduals::Accept`] (`ResidualPolicy::AcceptUnpinned` for a store), and the
/// result then carries [`ReadStatus::Unpinned`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResidualUnpinned {
    /// The segment that holds the residual
    pub segment_id: u64,
    /// The first key the segment covers
    pub start_time: i64,
    /// The last key the segment covers
    pub end_time: i64,
    /// The store the segment belongs to, when the read came through one
    pub series: Option<std::path::PathBuf>,
}

impl std::fmt::Display for ResidualUnpinned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "segment {} (keys {}..={}) holds a residual written before residuals carried an \
             identifier (alice-db before 0.3.0): its arithmetic is not known, so it is not \
             applied (accept it explicitly with ResidualPolicy::AcceptUnpinned)",
            self.segment_id, self.start_time, self.end_time
        )?;
        if let Some(series) = &self.series {
            write!(f, " (store {})", series.display())?;
        }
        Ok(())
    }
}

impl std::error::Error for ResidualUnpinned {}

/// Whether a read applies a residual that carries no identifier
/// ([`ResidualSemantics::Unpinned`])
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnpinnedResiduals {
    /// Refuse it with [`ResidualError::Unpinned`] (the default)
    #[default]
    Refuse,
    /// Apply it, and say so with [`ReadStatus::Unpinned`]
    Accept,
}

/// What a value read back rests on (ALICE-Law G8: the typed status of a read)
///
/// For a read over several segments, the weakest of theirs ([`ReadStatus::weaker`]):
/// `Current` < `VerifiedPrevious` < `Unpinned` < `ModelOnly`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadStatus {
    /// Exact under this build's arithmetic: no residual, or a residual stamped with
    /// `alice_core::law::SEMANTICS_ID`
    Current,
    /// Exact: the residual is stamped with an earlier identifier that
    /// `alice_core::law::check_semantics` still accepts as computing the same bits
    VerifiedPrevious([u8; SEMANTICS_ID_LEN]),
    /// A residual without an identifier was applied ([`UnpinnedResiduals::Accept`]): exact
    /// only where the writer's arithmetic was this one
    Unpinned,
    /// The model's approximation was returned instead of the stored values
    /// (`ResidualPolicy::ModelOnly`)
    ModelOnly,
}

impl ReadStatus {
    const fn rank(self) -> u8 {
        match self {
            Self::Current => 0,
            Self::VerifiedPrevious(_) => 1,
            Self::Unpinned => 2,
            Self::ModelOnly => 3,
        }
    }

    /// The weaker of the two (for a read that touched both)
    #[must_use]
    pub const fn weaker(self, other: Self) -> Self {
        if other.rank() > self.rank() {
            other
        } else {
            self
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResidualError {
    /// The residual belongs to another arithmetic
    Mismatch(ResidualMismatch),
    /// The residual does not decode to one correction per point
    Corrupt(ResidualCorrupt),
    /// A residual without an identifier, under [`UnpinnedResiduals::Refuse`]
    Unpinned(ResidualUnpinned),
}

impl ResidualError {
    /// Name the store the segment belongs to
    pub(crate) fn in_store(mut self, dir: &std::path::Path) -> Self {
        match &mut self {
            Self::Mismatch(m) => m.series = Some(dir.to_path_buf()),
            Self::Corrupt(c) => c.series = Some(dir.to_path_buf()),
            Self::Unpinned(u) => u.series = Some(dir.to_path_buf()),
        }
        self
    }
}

impl std::fmt::Display for ResidualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mismatch(m) => m.fmt(f),
            Self::Corrupt(c) => c.fmt(f),
            Self::Unpinned(u) => u.fmt(f),
        }
    }
}

impl std::error::Error for ResidualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mismatch(m) => Some(m),
            Self::Corrupt(c) => Some(c),
            Self::Unpinned(u) => Some(u),
        }
    }
}

/// Whether a segment's values come back exactly, and if not why — the same
/// answer the queries act on, without reading a value
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResidualStatus {
    /// A residual measured against the arithmetic in use: reads are exact
    Exact,
    /// No residual: reads return the model (a lossy segment)
    NoResidual,
    /// A residual written before 0.3.0 with no record of its arithmetic: it
    /// is applied, and is exact on a machine whose libm matches the writer's
    Unpinned,
    /// A residual measured against another arithmetic: reads refuse
    Mismatch {
        /// Identifier recorded with the residual
        stored: [u8; SEMANTICS_ID_LEN],
        /// Identifier of the arithmetic in use
        running: [u8; SEMANTICS_ID_LEN],
    },
    /// A residual that does not decode to one correction per point: reads
    /// refuse
    Corrupt {
        /// Decoded residual bytes the segment's points need (4 per point)
        needed_bytes: usize,
        /// Decoded residual bytes the blob held
        found_bytes: usize,
    },
}

/// [`ResidualStatus`] of an optional residual blob for a segment of
/// `point_count` points (the identifier is checked first, then the decoded
/// length)
#[must_use]
pub fn residual_status(blob: Option<&[u8]>, point_count: usize) -> ResidualStatus {
    match decode_residual(blob, point_count) {
        Decoded::None => ResidualStatus::NoResidual,
        Decoded::Mismatch(stored) => ResidualStatus::Mismatch {
            stored,
            running: alice_core::law::SEMANTICS_ID,
        },
        Decoded::Corrupt { needed, found } => ResidualStatus::Corrupt {
            needed_bytes: needed,
            found_bytes: found,
        },
        Decoded::Usable {
            status: ReadStatus::Unpinned,
            ..
        } => ResidualStatus::Unpinned,
        Decoded::Usable { .. } => ResidualStatus::Exact,
    }
}

/// What a residual blob decodes to for a segment of `point_count` points
enum Decoded {
    None,
    Mismatch([u8; SEMANTICS_ID_LEN]),
    Corrupt {
        needed: usize,
        found: usize,
    },
    Usable {
        kind: ResidualKind,
        bytes: Vec<u8>,
        status: ReadStatus,
    },
}

fn decode_residual(blob: Option<&[u8]>, point_count: usize) -> Decoded {
    let Some(blob) = blob else {
        return Decoded::None;
    };
    let status = match residual_semantics(blob) {
        ResidualSemantics::Pinned(id) => match alice_core::law::check_semantics(&id) {
            alice_det_math::SemanticsCheck::Verified => ReadStatus::Current,
            alice_det_math::SemanticsCheck::VerifiedPrevious(previous) => {
                ReadStatus::VerifiedPrevious(previous)
            }
            alice_det_math::SemanticsCheck::Mismatch => return Decoded::Mismatch(id),
        },
        ResidualSemantics::Unpinned => ReadStatus::Unpinned,
    };
    let bytes = decompress_residual(blob);
    let needed = point_count.saturating_mul(4);
    if bytes.len() < needed {
        return Decoded::Corrupt {
            needed,
            found: bytes.len(),
        };
    }
    Decoded::Usable {
        kind: residual_kind(blob),
        bytes,
        status,
    }
}

/// The decoded residual a read may apply: `None` when there is none, the
/// corrections when they belong to the arithmetic in use (or predate the
/// identifier) and cover every point, and [`ResidualError`] otherwise
fn checked_residual(
    blob: Option<&[u8]>,
    point_count: usize,
    segment_id: u64,
    start_time: i64,
    end_time: i64,
    unpinned: UnpinnedResiduals,
) -> Result<Option<(ResidualKind, Vec<u8>, ReadStatus)>, ResidualError> {
    match decode_residual(blob, point_count) {
        Decoded::None => Ok(None),
        Decoded::Usable {
            status: ReadStatus::Unpinned,
            ..
        } if unpinned == UnpinnedResiduals::Refuse => {
            Err(ResidualError::Unpinned(ResidualUnpinned {
                segment_id,
                start_time,
                end_time,
                series: None,
            }))
        }
        Decoded::Usable {
            kind,
            bytes,
            status,
        } => Ok(Some((kind, bytes, status))),
        Decoded::Mismatch(stored) => Err(ResidualError::Mismatch(ResidualMismatch {
            segment_id,
            start_time,
            end_time,
            stored,
            running: alice_core::law::SEMANTICS_ID,
            series: None,
        })),
        Decoded::Corrupt { needed, found } => Err(ResidualError::Corrupt(ResidualCorrupt {
            segment_id,
            start_time,
            end_time,
            needed_bytes: needed,
            found_bytes: found,
            series: None,
        })),
    }
}

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
        Some(
            &RESIDUAL_XOR_LZMA
            | &RESIDUAL_XOR_RAW
            | &RESIDUAL_XOR_PINNED_LZMA
            | &RESIDUAL_XOR_PINNED_RAW,
        ) => ResidualKind::Xor,
        _ => ResidualKind::Additive,
    }
}

/// Apply the `idx`-th residual word of a decompressed blob to `value`
///
/// `None` when the blob holds no word for `idx`: the reader turns that into
/// [`ResidualError::Corrupt`] rather than returning `value` (the model) as if
/// it were the stored original. Crate-internal since 0.4.0; the checked
/// queries are the way to read corrected values.
#[inline]
#[must_use]
pub(crate) fn apply_residual(
    value: f32,
    kind: ResidualKind,
    decompressed: &[u8],
    idx: usize,
) -> Option<f32> {
    let offset = idx.checked_mul(4)?;
    let word: [u8; 4] = decompressed
        .get(offset..offset.checked_add(4)?)?
        .try_into()
        .ok()?;
    Some(match kind {
        ResidualKind::Additive => value + f32::from_le_bytes(word),
        ResidualKind::Xor => f32::from_bits(value.to_bits() ^ u32::from_le_bytes(word)),
    })
}

/// [`ResidualError::Corrupt`] for a residual too short for the point read
fn missing_word(
    segment_id: u64,
    start_time: i64,
    end_time: i64,
    point_count: usize,
    decoded: usize,
) -> ResidualError {
    ResidualError::Corrupt(ResidualCorrupt {
        segment_id,
        start_time,
        end_time,
        needed_bytes: point_count.saturating_mul(4),
        found_bytes: decoded,
        series: None,
    })
}

/// Compress an XOR residual (`original.to_bits() ^ model.to_bits()` per
/// sample, LE) together with the identifier of the arithmetic the model values
/// came from.
///
/// Format: `[1 byte magic] [32 byte arithmetic identifier] [data]`, raw
/// fallback when LZMA would expand.
///
/// The identifier is what makes the blob self-describing. A residual only
/// reconstructs the original when the reader evaluates the law with the same
/// arithmetic, and IEEE 754 leaves the transcendentals free to differ between
/// targets, so without it a reader cannot tell an exact reconstruction from a
/// value that is neither the original nor the model. Blobs written before
/// 0.3.0 carry no identifier ([`ResidualSemantics::Unpinned`]): a read refuses them
/// unless it opts in with [`UnpinnedResiduals::Accept`], and then reports
/// [`ReadStatus::Unpinned`]. Nothing in this crate writes such a blob any more.
#[must_use]
pub fn compress_residual_xor(raw: &[u8], semantics_id: &[u8; SEMANTICS_ID_LEN]) -> Vec<u8> {
    let mut compressed = Vec::new();
    if lzma_rs::lzma_compress(&mut std::io::Cursor::new(raw), &mut compressed).is_ok()
        && compressed.len() < raw.len()
    {
        let mut out = Vec::with_capacity(1 + SEMANTICS_ID_LEN + compressed.len());
        out.push(RESIDUAL_XOR_PINNED_LZMA);
        out.extend_from_slice(semantics_id);
        out.extend_from_slice(&compressed);
        out
    } else {
        let mut out = Vec::with_capacity(1 + SEMANTICS_ID_LEN + raw.len());
        out.push(RESIDUAL_XOR_PINNED_RAW);
        out.extend_from_slice(semantics_id);
        out.extend_from_slice(raw);
        out
    }
}

/// `x`, or the canonical quiet NaN `0x7fc0_0000` when `x` is a NaN: the bits of a computed NaN
/// (a model value) are not specified (ALICE-Law G4). Written as bits: the standard library does
/// not fix the bit pattern of `f32::NAN`
#[inline]
fn canon32(x: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        x
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
        RESIDUAL_XOR_PINNED_RAW => blob.get(1 + SEMANTICS_ID_LEN..).unwrap_or(&[]).to_vec(),
        RESIDUAL_XOR_PINNED_LZMA => {
            let Some(payload) = blob.get(1 + SEMANTICS_ID_LEN..) else {
                return Vec::new();
            };
            let mut decompressed = Vec::new();
            if lzma_rs::lzma_decompress(&mut std::io::Cursor::new(payload), &mut decompressed)
                .is_ok()
            {
                decompressed
            } else {
                Vec::new()
            }
        }
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
// SegmentView: a stored segment, decoded once
// =============================================================================

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

/// A stored segment read for queries.
///
/// The segment file's bytes are decoded once, when the view is made, into an
/// owned [`DataSegment`] (stored segment format 1 through rkyv's checked
/// deserialization, format 0 through the built-in reader of the earlier
/// layout, see [`DataSegment::from_stored_bytes`]); every query then runs the
/// same [`DataSegment`] methods as an in-memory segment, so a stored segment
/// and the segment it was written from answer from one law path.
///
/// # Memory
///
/// An open view holds its decoded segment: about the size of the segment
/// file (the model's coefficients and the residual bytes). The file is not
/// kept mapped after it is read.
pub struct SegmentView {
    segment: DataSegment,
    source: &'static str,
}

impl SegmentView {
    /// Read and decode a segment file.
    ///
    /// The file is read under an advisory shared lock (released when the
    /// read is done), so a concurrent writer cannot truncate it mid-read.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a valid stored
    /// segment.
    #[cfg(feature = "fs")]
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        #[allow(unused_imports)]
        use fs2::FileExt;

        let mut file = File::open(path)?;
        // trait 経由で明示 (1.89+ の inherent File::lock_shared と MSRV の両立、clippy incompatible_msrv)
        FileExt::lock_shared(&file)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        drop(file);
        Self::decode(&data, "file")
    }

    /// Read a segment file without taking the shared lock.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a valid stored
    /// segment.
    #[cfg(feature = "fs")]
    pub fn open_read<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::decode(&data, "file")
    }

    /// Decode stored segment bytes held in memory (a freshly flushed
    /// segment, or a decrypted file).
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not a valid stored segment.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the signature takes the buffer as before; the view keeps only the decoded segment"
    )]
    pub fn from_vec(data: Vec<u8>) -> io::Result<Self> {
        Self::decode(&data, "vec")
    }

    /// As [`Self::from_vec`], from shared bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not a valid stored segment.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the signature takes the buffer as before; the view keeps only the decoded segment"
    )]
    pub fn from_arc_vec(data: Arc<Vec<u8>>) -> io::Result<Self> {
        Self::decode(&data, "vec")
    }

    /// As [`Self::from_vec`], from a static slice (embedded data, tests).
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not a valid stored segment.
    pub fn from_static(data: &'static [u8]) -> io::Result<Self> {
        Self::decode(data, "slice")
    }

    fn decode(data: &[u8], source: &'static str) -> io::Result<Self> {
        Ok(Self {
            segment: DataSegment::from_stored_bytes(data)?,
            source,
        })
    }

    /// Where the bytes were read from: `"file"`, `"vec"` or `"slice"`
    #[must_use]
    pub const fn source_type(&self) -> &'static str {
        self.source
    }

    /// The decoded segment
    #[must_use]
    pub const fn segment(&self) -> &DataSegment {
        &self.segment
    }

    /// Get start timestamp
    #[inline]
    #[must_use]
    pub const fn start_time(&self) -> i64 {
        self.segment.start_time
    }

    /// Get end timestamp
    #[inline]
    #[must_use]
    pub const fn end_time(&self) -> i64 {
        self.segment.end_time
    }

    /// Check if timestamp is in range
    #[inline]
    #[must_use]
    pub const fn contains(&self, timestamp: i64) -> bool {
        self.segment.contains(timestamp)
    }

    /// Whether the segment carries a lossless residual
    #[must_use]
    pub const fn is_lossless(&self) -> bool {
        self.segment.residual_blob.is_some()
    }

    /// The segment's `created_at` metadata (Unix millis, or whatever the
    /// writer passed to [`DataSegment::new`] -- this crate itself never
    /// reads the wall clock, see G10)
    #[must_use]
    pub const fn created_at(&self) -> u64 {
        self.segment.metadata.created_at
    }

    /// First point of the segment's even grid at or after `timestamp`, or
    /// `None` when `timestamp` is past the last point
    ///
    /// The grid is `start + ⌊k·range/(n−1)⌋`, `k = 0 ..= n−1`; for a
    /// lossless segment written by this version those are exactly the keys
    /// that were written (lossless mode cuts segments where the spacing
    /// changes).
    #[must_use]
    pub fn grid_ceil(&self, timestamp: i64) -> Option<i64> {
        law::grid_ceil(
            self.segment.start_time,
            self.segment.end_time,
            self.point_count(),
            timestamp,
        )
    }

    /// Whether `timestamp` is a point of the segment's even grid
    #[must_use]
    pub fn on_grid(&self, timestamp: i64) -> bool {
        self.grid_ceil(timestamp) == Some(timestamp)
    }

    /// Point query, as [`DataSegment::query_point`]
    ///
    /// # Errors
    ///
    /// [`ResidualError`], as [`DataSegment::query_point`].
    pub fn query_point(&self, timestamp: i64) -> Result<Option<f32>, ResidualError> {
        self.segment.query_point(timestamp)
    }

    /// As [`DataSegment::read_point`]
    ///
    /// # Errors
    ///
    /// As [`DataSegment::read_point`].
    pub fn read_point(
        &self,
        timestamp: i64,
        unpinned: UnpinnedResiduals,
    ) -> Result<Option<(f32, ReadStatus)>, ResidualError> {
        self.segment.read_point(timestamp, unpinned)
    }

    /// As [`DataSegment::query_point_model_only`]
    #[must_use]
    pub fn query_point_model_only(&self, timestamp: i64) -> Option<f32> {
        self.segment.query_point_model_only(timestamp)
    }

    /// Whether this segment's values come back exactly, and if not why
    #[must_use]
    pub fn residual_status(&self) -> ResidualStatus {
        self.segment.residual_status()
    }

    /// Range query, as [`DataSegment::query_range`]
    ///
    /// # Errors
    ///
    /// [`ResidualError`], as [`DataSegment::query_range`].
    pub fn query_range(&self, start: i64, end: i64) -> Result<Vec<(i64, f32)>, ResidualError> {
        self.segment.query_range(start, end)
    }

    /// As [`DataSegment::read_range`]
    ///
    /// # Errors
    ///
    /// As [`DataSegment::read_range`].
    pub fn read_range(
        &self,
        start: i64,
        end: i64,
        unpinned: UnpinnedResiduals,
    ) -> Result<(Vec<(i64, f32)>, ReadStatus), ResidualError> {
        self.segment.read_range(start, end, unpinned)
    }

    /// As [`DataSegment::query_range_model_only`]
    #[must_use]
    pub fn query_range_model_only(&self, start: i64, end: i64) -> Vec<(i64, f32)> {
        self.segment.query_range_model_only(start, end)
    }

    /// Get compression ratio from metadata
    #[must_use]
    pub const fn compression_ratio(&self) -> f64 {
        self.segment.metadata.compression_ratio
    }

    /// Get point count from metadata
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.segment.metadata.point_count
    }
}

#[cfg(test)]
mod tests {
    /// A word index past the decoded residual gives `None`, never the model
    /// value standing in for the original (the readers turn it into
    /// `ResidualError::Corrupt`)
    #[test]
    fn apply_residual_without_a_word_for_the_index_is_none() {
        let word = 3_u32.to_le_bytes();
        let one = word.to_vec();
        assert_eq!(
            apply_residual(1.5, ResidualKind::Xor, &one, 0).map(f32::to_bits),
            Some(1.5_f32.to_bits() ^ 3)
        );
        assert_eq!(apply_residual(1.5, ResidualKind::Xor, &one, 1), None);
        assert_eq!(
            apply_residual(1.5, ResidualKind::Additive, &one[..3], 0),
            None
        );
        assert_eq!(apply_residual(1.5, ResidualKind::Xor, &[], 0), None);
        assert_eq!(
            apply_residual(1.5, ResidualKind::Xor, &one, usize::MAX),
            None
        );
        assert_eq!(
            apply_residual(1.5, ResidualKind::Xor, &one, usize::MAX / 4),
            None
        );
    }

    use super::*;

    /// `law::grid_ceil` rounds up to the next point of the even grid
    /// `start + ⌊k·range/(n−1)⌋`; expected values are worked out by hand
    #[test]
    fn grid_ceil_rounds_up_to_the_next_grid_point() {
        use super::law::grid_ceil;
        // 100, 107, …, 233 (n = 20, step 7)
        assert_eq!(grid_ceil(100, 233, 20, 100), Some(100));
        assert_eq!(grid_ceil(100, 233, 20, 101), Some(107));
        assert_eq!(grid_ceil(100, 233, 20, 106), Some(107));
        assert_eq!(grid_ceil(100, 233, 20, 107), Some(107));
        assert_eq!(grid_ceil(100, 233, 20, 227), Some(233));
        assert_eq!(grid_ceil(100, 233, 20, 233), Some(233));
        assert_eq!(grid_ceil(100, 233, 20, 234), None);
        assert_eq!(grid_ceil(100, 233, 20, 99), Some(100));
        assert_eq!(grid_ceil(100, 233, 20, i64::MIN), Some(100));
        assert_eq!(grid_ceil(100, 233, 20, i64::MAX), None);
        // a step that is not an integer: 0, 3, 6, 10 (⌊10k/3⌋)
        assert_eq!(grid_ceil(0, 10, 4, 1), Some(3));
        assert_eq!(grid_ceil(0, 10, 4, 3), Some(3));
        assert_eq!(grid_ceil(0, 10, 4, 4), Some(6));
        assert_eq!(grid_ceil(0, 10, 4, 7), Some(10));
        // negative keys: −70, −63, …, 0
        assert_eq!(grid_ceil(-70, 0, 11, -69), Some(-63));
        assert_eq!(grid_ceil(-70, 0, 11, -63), Some(-63));
        assert_eq!(grid_ceil(-70, 0, 11, -1), Some(0));
        // the whole of i64: n = 2 is the two ends, n = 3 adds
        // MIN + ⌊(2^64 − 1)/2⌋ = −1
        let (min, max) = (i64::MIN, i64::MAX);
        assert_eq!(grid_ceil(min, max, 2, min), Some(min));
        assert_eq!(grid_ceil(min, max, 2, min + 1), Some(max));
        assert_eq!(grid_ceil(min, max, 2, max), Some(max));
        assert_eq!(grid_ceil(min, max, 3, -2), Some(-1));
        assert_eq!(grid_ceil(min, max, 3, -1), Some(-1));
        assert_eq!(grid_ceil(min, max, 3, 0), Some(max));
        // a single point: only the start is on the grid
        assert_eq!(grid_ceil(5, 5, 1, 5), Some(5));
        assert_eq!(grid_ceil(5, 5, 1, 6), None);
        assert_eq!(grid_ceil(0, 10, 1, 3), None);
    }

    #[test]
    fn test_constant_segment() {
        let segment = DataSegment::new(
            1,
            0,
            1000,
            ModelType::Constant { value: 42.0 },
            1001,
            1001 * 4,
            0,
        );

        assert!(segment.contains(500));
        assert!(!segment.contains(1001));

        let value = segment
            .query_point(500)
            .expect("residual of this arithmetic")
            .unwrap();
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
            0,
        );

        let start = segment
            .query_point(0)
            .expect("residual of this arithmetic")
            .unwrap();
        let mid = segment
            .query_point(50)
            .expect("residual of this arithmetic")
            .unwrap();
        let end = segment
            .query_point(100)
            .expect("residual of this arithmetic")
            .unwrap();

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
            0,
        );

        let at_half = segment
            .query_point(50)
            .expect("residual of this arithmetic")
            .unwrap();
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
            0,
        );

        let results = segment
            .query_range(0, 100)
            .expect("residual of this arithmetic");
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
            0,
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
            0,
        );

        // Compare scalar vs SIMD results
        let scalar_results = segment
            .query_range(0, 100)
            .expect("residual of this arithmetic");
        let simd_results = segment
            .query_range_simd(0, 100)
            .expect("residual of this arithmetic");

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
            0,
        );

        let scalar = segment
            .query_range(0, 1000)
            .expect("residual of this arithmetic");
        let simd = segment
            .query_range_simd(0, 1000)
            .expect("residual of this arithmetic");

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
        let segment = DataSegment::new(1, 10, 20, ModelType::Constant { value: 1.0 }, 11, 44, 0);
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
        let segment =
            DataSegment::new(1, 100, 200, ModelType::Constant { value: 1.0 }, 101, 404, 0);
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
            0,
        );
        assert!(segment
            .query_point(-1)
            .expect("residual of this arithmetic")
            .is_none());
        assert!(segment
            .query_point(101)
            .expect("residual of this arithmetic")
            .is_none());
    }

    #[test]
    fn test_query_range_non_overlapping() {
        let segment =
            DataSegment::new(1, 100, 200, ModelType::Constant { value: 5.0 }, 101, 404, 0);
        let results = segment
            .query_range(0, 50)
            .expect("residual of this arithmetic");
        assert!(results.is_empty());

        let results2 = segment
            .query_range(300, 400)
            .expect("residual of this arithmetic");
        assert!(results2.is_empty());
    }

    #[test]
    fn test_single_point_segment() {
        // Segment with start_time == end_time (single point)
        let segment = DataSegment::new(1, 42, 42, ModelType::Constant { value: 99.0 }, 1, 4, 0);
        assert!(segment.contains(42));
        assert!(!segment.contains(41));
        let val = segment
            .query_point(42)
            .expect("residual of this arithmetic")
            .unwrap();
        assert!((val - 99.0).abs() < 0.001);
    }

    #[test]
    fn test_constant_segment_generate_all() {
        let segment = DataSegment::new(1, 0, 99, ModelType::Constant { value: 7.5 }, 100, 400, 0);
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
            0,
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
            0,
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
        let segment = DataSegment::new(1, 0, 10, ModelType::Constant { value: 1.0 }, 11, 44, 0);
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();
        assert_eq!(view.source_type(), "vec");
    }

    #[test]
    fn test_segment_view_query_point_constant() {
        let segment = DataSegment::new(1, 0, 100, ModelType::Constant { value: 42.0 }, 101, 404, 0);
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();

        let val = view
            .query_point(50)
            .expect("residual of this arithmetic")
            .unwrap();
        assert!((val - 42.0).abs() < 0.001);

        assert!(view
            .query_point(-1)
            .expect("residual of this arithmetic")
            .is_none());
        assert!(view
            .query_point(101)
            .expect("residual of this arithmetic")
            .is_none());
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
            0,
        );
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();
        let view = SegmentView::from_vec(rkyv_bytes).unwrap();

        let results = view
            .query_range(0, 100)
            .expect("residual of this arithmetic");
        assert!(!results.is_empty());

        let (_, first_val) = results.first().unwrap();
        let (_, last_val) = results.last().unwrap();
        assert!((first_val - 0.0).abs() < 1.0);
        assert!((last_val - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_segment_view_compression_ratio_and_point_count() {
        let segment = DataSegment::new(1, 0, 99, ModelType::Constant { value: 1.0 }, 100, 400, 0);
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
            0,
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
        let segment = DataSegment::new(1, 0, 10, ModelType::Constant { value: 5.0 }, 11, 44, 0);
        assert!(segment.residual_blob.is_none());

        let residual_data: Vec<u8> = (0..44).map(|i| i as u8).collect();
        let segment_with_res = segment.with_residual(residual_data);
        assert!(segment_with_res.residual_blob.is_some());
        assert_eq!(segment_with_res.residual_blob.unwrap().len(), 44);
    }

    #[test]
    fn test_segment_metadata_fields() {
        // created_at is an explicit input now (G10: the crate never reads
        // the wall clock itself), so the test pins a nonzero literal and
        // checks it flows through verbatim rather than merely being nonzero.
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
            1_700_000_000_000,
        );
        assert_eq!(segment.metadata.id, 99);
        assert_eq!(segment.metadata.point_count, 1001);
        assert_eq!(segment.metadata.original_size, 4004);
        assert_eq!(segment.metadata.created_at, 1_700_000_000_000);
        assert!(segment.metadata.compression_ratio > 1.0);
    }

    /// G10 oracle: two segments built from identical arguments (including
    /// `created_at`) serialize to identical bytes. Before this crate stopped
    /// reading the wall clock, this was false -- the same inputs, called a
    /// few milliseconds apart, produced a different `created_at` and so a
    /// different stored byte stream for the same logical segment.
    #[test]
    fn identical_args_including_created_at_give_identical_bytes() {
        let make = || {
            DataSegment::new(
                1,
                0,
                1000,
                ModelType::Linear {
                    start_value: 0.0,
                    end_value: 1.0,
                },
                1001,
                4004,
                1_700_000_000_000,
            )
        };
        let a = make().to_bytes().expect("serialize a");
        // A real clock read-back would very likely differ across a gap this
        // size (unlike two calls a few microseconds apart, which could
        // coincidentally land in the same millisecond and pass even with
        // the bug this test exists to catch).
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = make().to_bytes().expect("serialize b");
        assert_eq!(a, b, "same inputs must give identical bytes (G10)");
    }

    /// The additive residual blob `alice-db` wrote before 0.3.0 (`[magic][LZMA or raw]`,
    /// no identifier) — the writer is gone; tests build the bytes to read them back
    fn pre_0_3_0_additive_blob(raw: &[u8]) -> Vec<u8> {
        let mut compressed = Vec::new();
        if lzma_rs::lzma_compress(&mut std::io::Cursor::new(raw), &mut compressed).is_ok()
            && compressed.len() < raw.len()
        {
            let mut out = vec![super::RESIDUAL_LZMA];
            out.extend_from_slice(&compressed);
            out
        } else {
            let mut out = vec![super::RESIDUAL_RAW];
            out.extend_from_slice(raw);
            out
        }
    }

    #[test]
    fn test_lossless_lzma_roundtrip() {
        // Create a segment with known residuals
        let mut segment =
            DataSegment::new(1, 0, 99, ModelType::Constant { value: 10.0 }, 100, 400, 0);

        // Build raw residual f32 bytes
        let mut raw_residuals = Vec::with_capacity(100 * 4);
        for i in 0..100 {
            let r = (i as f32) * 0.001;
            raw_residuals.extend_from_slice(&r.to_le_bytes());
        }

        // Compress and attach (the pre-0.3.0 form: no identifier)
        let compressed = pre_0_3_0_additive_blob(&raw_residuals);
        // Compressed blob should have magic byte prefix
        assert!(!compressed.is_empty());
        assert!(compressed[0] == super::RESIDUAL_LZMA || compressed[0] == super::RESIDUAL_RAW);
        segment = segment.with_residual(compressed.clone());

        // Verify decompression roundtrip
        let decompressed = super::decompress_residual(&compressed);
        assert_eq!(decompressed.len(), raw_residuals.len());
        assert_eq!(decompressed, raw_residuals);

        // The default read refuses a residual without an identifier
        assert!(matches!(
            segment.query_point(0),
            Err(ResidualError::Unpinned(_))
        ));

        // Accepted explicitly, it applies and says so
        let (val_at_0, status) = segment
            .read_point(0, UnpinnedResiduals::Accept)
            .expect("accepted")
            .unwrap();
        assert_eq!(status, ReadStatus::Unpinned);
        // Constant model = 10.0, residual[0] = 0.0 * 0.001 = 0.0
        assert!((val_at_0 - 10.0).abs() < 0.01);

        let (val_at_50, _) = segment
            .read_point(50, UnpinnedResiduals::Accept)
            .expect("accepted")
            .unwrap();
        // Constant model = 10.0, residual[50] = 50 * 0.001 = 0.05
        assert!((val_at_50 - 10.05).abs() < 0.01);
    }

    #[test]
    fn test_compress_decompress_fallback() {
        // Tiny data: LZMA expansion expected → falls back to raw
        let tiny = vec![1u8, 2, 3, 4];
        let compressed = pre_0_3_0_additive_blob(&tiny);
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

    #[cfg(feature = "fs")]
    #[test]
    fn test_mmap_file_lock() {
        use fs2::FileExt;

        // Write a valid segment to a temp file
        let segment = DataSegment::new(1, 0, 99, ModelType::Constant { value: 42.0 }, 100, 400, 0);
        let rkyv_bytes = segment.to_rkyv_bytes().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_seg.rkyv");
        std::fs::write(&path, &rkyv_bytes).unwrap();

        // The view reads the file under a shared lock and keeps only the decoded
        // segment, so the file is not held once the view exists
        let view = SegmentView::open(&path).unwrap();
        assert_eq!(view.source_type(), "file");
        let file3 = File::open(&path).unwrap();
        assert!(
            file3.try_lock_exclusive().is_ok(),
            "the view must not hold the file after it is read"
        );
        file3.unlock().unwrap();

        // A writer holding an exclusive lock blocks the read (the shared lock
        // is taken before reading)
        let writer = File::open(&path).unwrap();
        writer.lock_exclusive().unwrap();
        let reader = File::open(&path).unwrap();
        assert!(reader.try_lock_shared().is_err());
        writer.unlock().unwrap();
        drop(view);

        // Now exclusive lock should succeed
        let file4 = File::open(&path).unwrap();
        assert!(file4.try_lock_exclusive().is_ok());
    }

    /// `alice-det-math` 0.5.0's identifier (frozen: this is the exact byte
    /// value that release recorded as `SEMANTICS_ID`, copied verbatim and
    /// never edited — not read from the dependency, which may since have
    /// moved on). A residual stamped with it must keep reading back while the
    /// identifier in use lists it as previous (alice-zip's
    /// `PREVIOUS_SEMANTICS_IDS`, `check_semantics(..).is_reproducible()`), not
    /// only while it happens to equal the current running identifier.
    const SEMANTICS_ID_0_5_0: [u8; SEMANTICS_ID_LEN] = [
        0x13, 0x5c, 0x40, 0x7d, 0x42, 0x49, 0xc1, 0xc2, 0x00, 0xe4, 0xd1, 0xbe, 0x79, 0xa1, 0x2e,
        0x84, 0x8c, 0xe5, 0xc3, 0x55, 0xfc, 0x3a, 0x91, 0x8d, 0xa9, 0x65, 0x84, 0x9e, 0x69, 0x7e,
        0xf0, 0x00,
    ];

    /// A residual stamped with 0.5.0's identifier still applies: the running
    /// identifier has since moved (alice-det-math 0.5.1's, then alice-zip
    /// 0.11's own) and alice-zip lists 0.5.0's in `PREVIOUS_SEMANTICS_IDS`, so
    /// the read reports `VerifiedPrevious` with that identifier. Reverting
    /// `residual_is_applicable` to a raw equality with the running identifier
    /// turns this red.
    #[test]
    fn a_residual_stamped_with_0_5_0s_identifier_still_applies() {
        let raw: Vec<u8> = (0..16u8).collect();
        let blob = compress_residual_xor(&raw, &SEMANTICS_ID_0_5_0);
        assert_eq!(
            residual_semantics(&blob),
            ResidualSemantics::Pinned(SEMANTICS_ID_0_5_0)
        );
        assert!(
            residual_is_applicable(&blob),
            "a residual stamped with a previously-accepted identifier must still apply"
        );
        assert!(matches!(
            decode_residual(Some(&blob), raw.len() / 4),
            Decoded::Usable {
                status: ReadStatus::VerifiedPrevious(id),
                ..
            } if id == SEMANTICS_ID_0_5_0
        ));
    }

    /// An identifier this build neither runs nor lists as a previous one is
    /// refused, not silently treated as a match (the failure mode a plain
    /// `id == SEMANTICS_ID` equality and `check_semantics` both guard
    /// against, so this alone does not distinguish the two — the test above
    /// does, once run against a moved identifier).
    /// The default read does not apply a residual without an identifier
    #[test]
    fn an_unpinned_residual_is_not_applicable_by_default() {
        let raw: Vec<u8> = (0..16u8).collect();
        let blob = pre_0_3_0_additive_blob(&raw);
        assert_eq!(residual_semantics(&blob), ResidualSemantics::Unpinned);
        assert!(!residual_is_applicable(&blob));
    }

    #[test]
    fn an_unknown_semantics_id_is_refused() {
        let unknown = [0xee_u8; SEMANTICS_ID_LEN];
        let raw: Vec<u8> = (0..16u8).collect();
        let blob = compress_residual_xor(&raw, &unknown);
        assert!(!residual_is_applicable(&blob));
        assert!(matches!(
            decode_residual(Some(&blob), raw.len() / 4),
            Decoded::Mismatch(id) if id == unknown
        ));
    }
}
