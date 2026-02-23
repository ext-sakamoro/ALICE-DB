//! Model Types for ALICE-DB
//!
//! Defines the procedural models that replace raw data storage.
//! Each model type represents a mathematical function that can regenerate
//! data on-demand with O(1) computation per point.
//!
//! # Philosophy
//!
//! Traditional DB: Store 1000 floats → 4KB on disk
//! ALICE-DB: Store `y = 0.5x + 10` → 16 bytes on disk
//!
//! # Zero-Copy Serialization (Phase 1)
//!
//! Uses rkyv for zero-copy deserialization. When reading from mmap,
//! no parsing or copying occurs - the archived data is accessed directly.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Model type enumeration for procedural data storage
///
/// Each variant stores only the parameters needed to regenerate data.
/// The actual data is never stored - only the "recipe" to create it.
///
/// # Zero-Copy Access
///
/// With rkyv, this enum can be read directly from mmap without deserialization.
/// `ArchivedModelType` provides the same interface for zero-copy access.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
pub enum ModelType {
    /// Polynomial model: `y = c[0] + c[1]*x + c[2]*x^2 + ...`
    ///
    /// Coefficients are stored from lowest to highest degree.
    /// Generated using Horner's method for numerical stability.
    ///
    /// Typical use: Smooth trends, sensor drift, gradual changes
    /// Compression: 1000 points → ~80 bytes (degree 9)
    Polynomial {
        /// Polynomial coefficients [c0, c1, c2, ...] (lowest to highest)
        coefficients: Vec<f64>,
        /// Polynomial degree (redundant but useful for quick checks)
        degree: usize,
        /// Fitting error (relative MSE)
        fit_error: f64,
    },

    /// Fourier series model: sum of sinusoids
    ///
    /// Stores dominant frequency components from FFT analysis.
    /// Regenerates periodic/quasi-periodic signals with high fidelity.
    ///
    /// Typical use: Periodic sensor data, vibration, seasonal patterns
    /// Compression: 1000 points → ~200 bytes (10 components)
    Fourier {
        /// Fourier coefficients: (freq_index, magnitude, phase)
        coefficients: Vec<(usize, f32, f32)>,
        /// DC offset (mean value)
        dc_offset: f32,
        /// Number of samples in original signal (for reconstruction)
        sample_count: usize,
    },

    /// Simple sine wave: y = offset + amplitude * sin(2π * freq * t + phase)
    ///
    /// Special case of Fourier for single-frequency signals.
    /// Extremely compact representation.
    ///
    /// Typical use: AC power monitoring, simple oscillations
    /// Compression: 1000 points → 16 bytes
    SineWave {
        frequency: f32,
        amplitude: f32,
        phase: f32,
        offset: f32,
    },

    /// Multi-sine: sum of multiple sine waves
    ///
    /// More flexible than single sine, more compact than full Fourier.
    ///
    /// Typical use: Mixed frequency signals, harmonics
    /// Compression: 1000 points → 12 bytes per component
    MultiSine {
        /// Components: (frequency, amplitude, phase)
        components: Vec<(f32, f32, f32)>,
        /// DC offset
        dc_offset: f32,
    },

    /// Perlin noise model for stochastic/noise-like data
    ///
    /// Regenerates deterministic pseudo-random patterns.
    ///
    /// Typical use: Texture data, environmental noise
    /// Compression: Any size → 24 bytes
    PerlinNoise {
        seed: u64,
        scale: f32,
        octaves: u32,
        persistence: f32,
        lacunarity: f32,
    },

    /// Constant value (all samples are the same)
    ///
    /// Trivial but important for flat-line detection.
    ///
    /// Compression: Any size → 8 bytes
    Constant { value: f64 },

    /// Linear interpolation between two points
    ///
    /// Simplest non-constant model.
    ///
    /// Compression: Any size → 16 bytes
    Linear { start_value: f64, end_value: f64 },

    /// Raw fallback: LZMA-compressed original data
    ///
    /// Used when no procedural model achieves acceptable error threshold.
    /// Still provides ~2-5x compression via LZMA.
    ///
    /// This is the "defeat" case - we couldn't find a pattern.
    RawLzma {
        /// LZMA-compressed raw bytes
        compressed_data: Vec<u8>,
        /// Original data type
        dtype: DataType,
        /// Uncompressed size in bytes
        original_size: usize,
    },
}

/// Data type for raw storage
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug, PartialEq, Eq))]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
}

impl DataType {
    /// Size of one element in bytes
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::UInt8 => 1,
        }
    }
}

impl ModelType {
    /// Estimate the serialized size of this model in bytes
    pub fn estimated_size(&self) -> usize {
        match self {
            ModelType::Polynomial { coefficients, .. } => {
                // 8 bytes per coefficient + overhead
                coefficients.len() * 8 + 24
            }
            ModelType::Fourier { coefficients, .. } => {
                // 12 bytes per coefficient (usize + f32 + f32) + overhead
                coefficients.len() * 12 + 16
            }
            ModelType::SineWave { .. } => 16,
            ModelType::MultiSine { components, .. } => components.len() * 12 + 8,
            ModelType::PerlinNoise { .. } => 24,
            ModelType::Constant { .. } => 8,
            ModelType::Linear { .. } => 16,
            ModelType::RawLzma {
                compressed_data, ..
            } => compressed_data.len() + 16,
        }
    }

    /// Check if this is a fallback (non-procedural) model
    pub fn is_fallback(&self) -> bool {
        matches!(self, ModelType::RawLzma { .. })
    }

    /// Get a human-readable name for this model type
    pub fn name(&self) -> &'static str {
        match self {
            ModelType::Polynomial { .. } => "Polynomial",
            ModelType::Fourier { .. } => "Fourier",
            ModelType::SineWave { .. } => "SineWave",
            ModelType::MultiSine { .. } => "MultiSine",
            ModelType::PerlinNoise { .. } => "PerlinNoise",
            ModelType::Constant { .. } => "Constant",
            ModelType::Linear { .. } => "Linear",
            ModelType::RawLzma { .. } => "RawLzma",
        }
    }
}

/// Result of model fitting competition
#[derive(Debug, Clone)]
pub struct FitResult {
    /// The winning model
    pub model: ModelType,
    /// Compression ratio achieved (original_size / model_size)
    pub compression_ratio: f64,
    /// Fitting error (relative MSE for lossy, 0 for lossless)
    pub error: f64,
    /// Whether this is a lossless representation
    pub is_lossless: bool,
}

impl FitResult {
    /// Create a new FitResult
    pub fn new(model: ModelType, original_size: usize, error: f64, is_lossless: bool) -> Self {
        let model_size = model.estimated_size();
        let compression_ratio = if model_size > 0 {
            original_size as f64 / model_size as f64
        } else {
            0.0
        };

        Self {
            model,
            compression_ratio,
            error,
            is_lossless,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_size_estimation() {
        let poly = ModelType::Polynomial {
            coefficients: vec![1.0, 2.0, 3.0],
            degree: 2,
            fit_error: 0.001,
        };
        assert!(poly.estimated_size() > 0);
        assert!(!poly.is_fallback());
        assert_eq!(poly.name(), "Polynomial");

        let raw = ModelType::RawLzma {
            compressed_data: vec![0u8; 100],
            dtype: DataType::Float32,
            original_size: 400,
        };
        assert!(raw.is_fallback());
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Float64.size(), 8);
        assert_eq!(DataType::UInt8.size(), 1);
    }

    #[test]
    fn test_data_type_size_all_variants() {
        assert_eq!(DataType::Int32.size(), 4);
        assert_eq!(DataType::Int64.size(), 8);
    }

    #[test]
    fn test_model_name_all_variants() {
        assert_eq!(ModelType::Constant { value: 0.0 }.name(), "Constant");
        assert_eq!(
            ModelType::Linear {
                start_value: 0.0,
                end_value: 1.0
            }
            .name(),
            "Linear"
        );
        assert_eq!(
            ModelType::SineWave {
                frequency: 1.0,
                amplitude: 1.0,
                phase: 0.0,
                offset: 0.0
            }
            .name(),
            "SineWave"
        );
        assert_eq!(
            ModelType::MultiSine {
                components: vec![],
                dc_offset: 0.0
            }
            .name(),
            "MultiSine"
        );
        assert_eq!(
            ModelType::PerlinNoise {
                seed: 0,
                scale: 1.0,
                octaves: 1,
                persistence: 0.5,
                lacunarity: 2.0
            }
            .name(),
            "PerlinNoise"
        );
        assert_eq!(
            ModelType::Fourier {
                coefficients: vec![],
                dc_offset: 0.0,
                sample_count: 0
            }
            .name(),
            "Fourier"
        );
    }

    #[test]
    fn test_is_fallback_non_fallback_variants() {
        assert!(!ModelType::Constant { value: 1.0 }.is_fallback());
        assert!(!ModelType::Linear {
            start_value: 0.0,
            end_value: 1.0
        }
        .is_fallback());
        assert!(!ModelType::SineWave {
            frequency: 1.0,
            amplitude: 1.0,
            phase: 0.0,
            offset: 0.0
        }
        .is_fallback());
        assert!(!ModelType::PerlinNoise {
            seed: 0,
            scale: 1.0,
            octaves: 1,
            persistence: 0.5,
            lacunarity: 2.0
        }
        .is_fallback());
    }

    #[test]
    fn test_estimated_size_constant() {
        let c = ModelType::Constant { value: 42.0 };
        assert_eq!(c.estimated_size(), 8);
    }

    #[test]
    fn test_estimated_size_linear() {
        let l = ModelType::Linear {
            start_value: 0.0,
            end_value: 100.0,
        };
        assert_eq!(l.estimated_size(), 16);
    }

    #[test]
    fn test_estimated_size_sinewave() {
        let s = ModelType::SineWave {
            frequency: 1.0,
            amplitude: 1.0,
            phase: 0.0,
            offset: 0.0,
        };
        assert_eq!(s.estimated_size(), 16);
    }

    #[test]
    fn test_estimated_size_perlin() {
        let p = ModelType::PerlinNoise {
            seed: 0,
            scale: 1.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
        };
        assert_eq!(p.estimated_size(), 24);
    }

    #[test]
    fn test_estimated_size_multisine() {
        let m = ModelType::MultiSine {
            components: vec![(1.0, 0.5, 0.0), (2.0, 0.3, 1.0)],
            dc_offset: 0.0,
        };
        // 2 components * 12 + 8 = 32
        assert_eq!(m.estimated_size(), 32);
    }

    #[test]
    fn test_estimated_size_fourier() {
        let f = ModelType::Fourier {
            coefficients: vec![(1, 0.5, 0.0), (3, 0.3, 1.0), (5, 0.1, 0.5)],
            dc_offset: 0.0,
            sample_count: 1000,
        };
        // 3 coefficients * 12 + 16 = 52
        assert_eq!(f.estimated_size(), 52);
    }

    #[test]
    fn test_estimated_size_raw_lzma() {
        let r = ModelType::RawLzma {
            compressed_data: vec![0u8; 200],
            dtype: DataType::Float32,
            original_size: 1000,
        };
        // 200 + 16 = 216
        assert_eq!(r.estimated_size(), 216);
    }

    #[test]
    fn test_fit_result_compression_ratio() {
        let model = ModelType::Linear {
            start_value: 0.0,
            end_value: 100.0,
        };
        let result = FitResult::new(model, 4000, 0.001, false);
        // 4000 / 16 = 250.0
        assert!((result.compression_ratio - 250.0).abs() < 0.1);
        assert!(!result.is_lossless);
        assert!((result.error - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_fit_result_zero_model_size() {
        // Empty polynomial (no coefficients) has estimated_size = 0*8 + 24 = 24
        let model = ModelType::Polynomial {
            coefficients: vec![],
            degree: 0,
            fit_error: 0.0,
        };
        let result = FitResult::new(model, 0, 0.0, true);
        // original_size=0, model_size=24 => 0/24 = 0.0
        assert!((result.compression_ratio - 0.0).abs() < 0.01);
        assert!(result.is_lossless);
    }
}
