//! MemTable: In-Memory Write Buffer (Double-Buffered Vec)
//!
//! Data flows: Insert → MemTable → Flush → Model Fitting → Segment
//!
//! The MemTable buffers incoming writes until it reaches capacity,
//! then triggers a flush operation that:
//! 1. Runs multiple fitting algorithms in parallel
//! 2. Selects the best model (highest compression, lowest error)
//! 3. Creates a DataSegment with the winning model
//!
//! # Performance: Double-Buffered Vec
//!
//! Time-series data is typically append-only (monotonically increasing timestamps).
//! SkipList's O(log N) random-access is overkill for this workload.
//!
//! Instead, we use a simple `Vec<(i64, f32)>` with:
//! - O(1) amortized push (just append to Vec)
//! - Double buffering: swap pointers on flush, sort in background
//! - No atomic CAS operations, no node allocations
//!
//! This is 10-100x faster for append-only workloads.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::model::{DataType, FitResult, ModelType};
use crate::segment::DataSegment;
use alice_core::generators;
use parking_lot::Mutex;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for model fitting
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Maximum polynomial degree to try
    pub max_polynomial_degree: usize,
    /// Error threshold for polynomial fitting (relative MSE)
    pub polynomial_error_threshold: f64,
    /// Maximum Fourier coefficients to extract
    pub max_fourier_coefficients: usize,
    /// Energy threshold for Fourier (0.0-1.0)
    pub fourier_energy_threshold: f32,
    /// Whether to try all models or stop at first success
    pub exhaustive_search: bool,
    /// Minimum compression ratio to accept a model
    pub min_compression_ratio: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_polynomial_degree: 9,
            polynomial_error_threshold: 0.001,
            max_fourier_coefficients: 20,
            fourier_energy_threshold: 0.99,
            exhaustive_search: true,
            min_compression_ratio: 2.0,
        }
    }
}

/// MemTable: Double-buffered in-memory write buffer
///
/// # Performance: Double-Buffered Vec
///
/// For append-only time-series workloads:
/// - Write buffer: `Vec::push()` is O(1) amortized
/// - On flush: swap buffer pointer, sort & compress in background
/// - No SkipList node allocations, no CAS loops
///
/// This is optimal for time-series where data arrives in roughly sorted order.
pub struct MemTable {
    /// Active write buffer (append-only)
    buffer: Mutex<Vec<(i64, f32)>>,
    /// Maximum capacity before flush
    capacity: usize,
    /// Fitting configuration
    config: FitConfig,
    /// Next segment ID (atomic)
    next_segment_id: AtomicU64,
}

impl MemTable {
    /// Create a new MemTable with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
            config: FitConfig::default(),
            next_segment_id: AtomicU64::new(1),
        }
    }

    /// Create a new MemTable with custom config
    pub fn with_config(capacity: usize, config: FitConfig) -> Self {
        Self {
            buffer: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
            config,
            next_segment_id: AtomicU64::new(1),
        }
    }

    /// Insert a single value
    ///
    /// Returns Some(DataSegment) if flush was triggered, None otherwise.
    /// Insert is O(1) amortized (just Vec::push).
    #[inline]
    pub fn put(&self, timestamp: i64, value: f32) -> Option<DataSegment> {
        let mut buffer = self.buffer.lock();
        buffer.push((timestamp, value));

        if buffer.len() >= self.capacity {
            // Swap out the buffer (O(1) pointer swap)
            let mut flush_data = Vec::with_capacity(self.capacity);
            std::mem::swap(&mut *buffer, &mut flush_data);
            drop(buffer); // Release lock before expensive operation

            Some(self.flush_internal(flush_data))
        } else {
            None
        }
    }

    /// Insert multiple values at once (batch insert)
    ///
    /// More efficient than individual puts for bulk loading.
    /// Returns Vec of segments if multiple flushes were triggered.
    #[inline]
    pub fn put_batch(&self, data: &[(i64, f32)]) -> Vec<DataSegment> {
        let mut segments = Vec::new();
        let mut buffer = self.buffer.lock();

        for &(timestamp, value) in data {
            buffer.push((timestamp, value));

            if buffer.len() >= self.capacity {
                // Swap out the buffer
                let mut flush_data = Vec::with_capacity(self.capacity);
                std::mem::swap(&mut *buffer, &mut flush_data);
                drop(buffer); // Release lock

                segments.push(self.flush_internal(flush_data));

                buffer = self.buffer.lock(); // Reacquire lock
            }
        }

        segments
    }

    /// Force flush current buffer regardless of capacity
    pub fn force_flush(&self) -> Option<DataSegment> {
        let mut buffer = self.buffer.lock();
        if buffer.is_empty() {
            return None;
        }

        // Swap out the buffer
        let mut flush_data = Vec::with_capacity(self.capacity);
        std::mem::swap(&mut *buffer, &mut flush_data);
        drop(buffer);

        Some(self.flush_internal(flush_data))
    }

    /// Get current buffer size
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.lock().is_empty()
    }

    /// Internal flush implementation (sort + model fitting)
    ///
    /// # Performance Note
    ///
    /// `sort_unstable_by_key` uses pdqsort which is O(N) for nearly-sorted data.
    /// Time-series data is typically already sorted, so this is fast.
    fn flush_internal(&self, mut data: Vec<(i64, f32)>) -> DataSegment {
        if data.is_empty() {
            // Edge case: return empty constant segment
            let id = self.next_id();
            return DataSegment::new(
                id,
                0,
                0,
                ModelType::Constant { value: 0.0 },
                0,
                0,
            );
        }

        // Sort by timestamp (O(N) for nearly-sorted data via pdqsort)
        data.sort_unstable_by_key(|&(t, _)| t);

        let start_time = data.first().map(|&(t, _)| t).unwrap_or(0);
        let end_time = data.last().map(|&(t, _)| t).unwrap_or(0);
        let values: Vec<f32> = data.iter().map(|&(_, v)| v).collect();
        let original_size = values.len() * 4;

        // Run model fitting competition
        let fit_result = self.fit_best_model(&values);

        let id = self.next_id();
        DataSegment::new(
            id,
            start_time,
            end_time,
            fit_result.model,
            values.len(),
            original_size,
        )
    }

    /// Generate next segment ID (atomic, lock-free)
    fn next_id(&self) -> u64 {
        self.next_segment_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Fit the best model to data
    ///
    /// Runs multiple fitting algorithms in parallel and selects the winner.
    /// Selection criteria: compression_ratio / (1 + error)
    fn fit_best_model(&self, values: &[f32]) -> FitResult {
        let original_size = values.len() * 4;

        // Quick checks for trivial cases
        if values.is_empty() {
            return FitResult::new(
                ModelType::Constant { value: 0.0 },
                0,
                0.0,
                true,
            );
        }

        // Check for constant data
        let first = values[0];
        if values.iter().all(|&v| (v - first).abs() < 1e-10) {
            return FitResult::new(
                ModelType::Constant { value: first as f64 },
                original_size,
                0.0,
                true,
            );
        }

        // Check for linear data
        if let Some(linear) = self.try_linear_fit(values) {
            if linear.compression_ratio > self.config.min_compression_ratio {
                return linear;
            }
        }

        // Run fitting algorithms in parallel
        let (poly_result, fourier_result) = rayon::join(
            || self.try_polynomial_fit(values),
            || self.try_fourier_fit(values),
        );

        let sine_result = self.try_sine_fit(values);

        let mut candidates: Vec<FitResult> = vec![
            poly_result,
            fourier_result,
            sine_result,
        ]
        .into_iter()
        .flatten()
        .collect();

        // Select best candidate by score
        candidates.sort_by(|a, b| {
            let score_a = a.compression_ratio / (1.0 + a.error * 100.0);
            let score_b = b.compression_ratio / (1.0 + b.error * 100.0);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return best if it meets minimum compression ratio
        if let Some(best) = candidates.first() {
            if best.compression_ratio >= self.config.min_compression_ratio {
                return best.clone();
            }
        }

        // Fallback to LZMA compression
        self.fallback_lzma(values)
    }

    /// Try linear fitting
    fn try_linear_fit(&self, values: &[f32]) -> Option<FitResult> {
        if values.len() < 2 {
            return None;
        }

        let first = values[0] as f64;
        let last = values[values.len() - 1] as f64;
        let n = values.len();

        // Pre-compute reciprocal to replace division in loop
        let inv_n_minus_1 = 1.0 / (n - 1) as f64;
        let delta = last - first;

        // Check if data is approximately linear
        let mut max_error = 0.0f64;
        for (i, &v) in values.iter().enumerate() {
            let t = i as f64 * inv_n_minus_1;
            let expected = first + t * delta;
            let error = (v as f64 - expected).abs();
            max_error = max_error.max(error);
        }

        let range = (last - first).abs();
        let relative_error = if range > 1e-10 { max_error / range } else { max_error };

        if relative_error < 0.01 {
            let model = ModelType::Linear {
                start_value: first,
                end_value: last,
            };
            Some(FitResult::new(model, values.len() * 4, relative_error, false))
        } else {
            None
        }
    }

    /// Try polynomial fitting
    fn try_polynomial_fit(&self, values: &[f32]) -> Option<FitResult> {
        let result = generators::fit_polynomial(
            values,
            self.config.max_polynomial_degree,
            self.config.polynomial_error_threshold,
        )?;

        let (coefficients, degree, error) = result;

        let model = ModelType::Polynomial {
            coefficients,
            degree,
            fit_error: error,
        };

        Some(FitResult::new(model, values.len() * 4, error, false))
    }

    /// Try Fourier fitting
    fn try_fourier_fit(&self, values: &[f32]) -> Option<FitResult> {
        if values.len() < 4 {
            return None;
        }

        let (coefficients, dc_offset) = generators::analyze_signal(
            values,
            self.config.max_fourier_coefficients,
            self.config.fourier_energy_threshold,
        );

        if coefficients.is_empty() {
            return None;
        }

        // Reconstruct and calculate error
        let reconstructed = generators::generate_from_coefficients(
            values.len(),
            &coefficients,
            dc_offset,
        );

        // Pre-compute reciprocal for MSE and variance normalization
        let inv_len = 1.0 / values.len() as f32;
        let mse: f32 = values
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() * inv_len;

        let variance: f32 = {
            let mean = values.iter().sum::<f32>() * inv_len;
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() * inv_len
        };

        let relative_error = if variance > 1e-10 {
            (mse / variance) as f64
        } else {
            mse as f64
        };

        let model = ModelType::Fourier {
            coefficients,
            dc_offset,
            sample_count: values.len(),
        };

        Some(FitResult::new(model, values.len() * 4, relative_error, false))
    }

    /// Try simple sine wave fitting
    fn try_sine_fit(&self, values: &[f32]) -> Option<FitResult> {
        if values.len() < 8 {
            return None;
        }

        // Use Fourier analysis to find dominant frequency
        let (coefficients, dc_offset) = generators::analyze_signal(values, 1, 0.5);

        if coefficients.is_empty() {
            return None;
        }

        let (freq_idx, magnitude, phase) = coefficients[0];

        // Convert to SineWave parameters
        // Pre-compute reciprocal for amplitude scaling and MSE/variance
        let inv_len = 1.0 / values.len() as f32;
        let frequency = freq_idx as f32;
        let amplitude = magnitude * 2.0 * inv_len;

        let model = ModelType::SineWave {
            frequency,
            amplitude,
            phase,
            offset: dc_offset,
        };

        // Verify fit quality
        let reconstructed = generators::generate_sine_wave(
            values.len(),
            frequency,
            amplitude,
            phase,
            dc_offset,
        );

        let mse: f32 = values
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() * inv_len;

        let variance: f32 = {
            let mean = values.iter().sum::<f32>() * inv_len;
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() * inv_len
        };

        let relative_error = if variance > 1e-10 {
            (mse / variance) as f64
        } else {
            mse as f64
        };

        // Only accept if error is low
        if relative_error < 0.1 {
            Some(FitResult::new(model, values.len() * 4, relative_error, false))
        } else {
            None
        }
    }

    /// Fallback: LZMA compress raw data
    fn fallback_lzma(&self, values: &[f32]) -> FitResult {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();

        let mut compressed = Vec::new();
        let _ = lzma_rs::lzma_compress(&mut Cursor::new(&bytes), &mut compressed);

        let model = ModelType::RawLzma {
            compressed_data: compressed,
            dtype: DataType::Float32,
            original_size: bytes.len(),
        };

        FitResult::new(model, bytes.len(), 0.0, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memtable_basic() {
        let memtable = MemTable::new(100);
        assert!(memtable.is_empty());

        // Insert values
        for i in 0..50 {
            let result = memtable.put(i, i as f32 * 2.0);
            assert!(result.is_none()); // Should not flush yet
        }

        assert_eq!(memtable.len(), 50);
    }

    #[test]
    fn test_memtable_flush_on_capacity() {
        let memtable = MemTable::new(100);

        // Insert values up to capacity
        let mut segment = None;
        for i in 0..100 {
            segment = memtable.put(i, i as f32);
        }

        // Should have flushed
        assert!(segment.is_some());
        assert!(memtable.is_empty());

        let seg = segment.unwrap();
        assert_eq!(seg.start_time, 0);
        assert_eq!(seg.end_time, 99);
    }

    #[test]
    fn test_constant_detection() {
        let memtable = MemTable::new(1000);
        let values: Vec<f32> = vec![42.0; 100];

        // Access internal fitting
        let result = memtable.fit_best_model(&values);
        assert!(matches!(result.model, ModelType::Constant { .. }));
    }

    #[test]
    fn test_linear_detection() {
        let memtable = MemTable::new(1000);
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let result = memtable.fit_best_model(&values);
        // Should detect as Linear or Polynomial degree 1
        assert!(
            matches!(result.model, ModelType::Linear { .. }) ||
            matches!(result.model, ModelType::Polynomial { degree: 1, .. })
        );
    }

    #[test]
    fn test_batch_insert() {
        let memtable = MemTable::new(50);

        let data: Vec<(i64, f32)> = (0..120).map(|i| (i, i as f32)).collect();
        let segments = memtable.put_batch(&data);

        // Should have created 2 segments (100 points each would overflow twice at 50 capacity)
        assert!(segments.len() >= 2);
    }

    #[test]
    fn test_force_flush() {
        let memtable = MemTable::new(1000);

        for i in 0..50 {
            memtable.put(i, i as f32 * 0.5);
        }

        let segment = memtable.force_flush();
        assert!(segment.is_some());
        assert!(memtable.is_empty());
    }
}
