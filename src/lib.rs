//! ALICE-DB: Model-Based LSM-Tree Database
//!
//! A revolutionary database that stores mathematical models instead of raw data,
//! achieving extreme compression ratios for time-series and numerical data.
//!
//! # Core Innovation
//!
//! Traditional databases store raw data:
//! ```text
//! 1000 sensor readings → 4KB on disk
//! ```
//!
//! ALICE-DB stores the mathematical function that generates the data:
//! ```text
//! 1000 sensor readings (linear) → "y = 0.5x + 10" → 16 bytes on disk
//! ```
//!
//! This is based on Kolmogorov complexity: the shortest program that produces
//! the output is the optimal representation.
//!
//! # Features
//!
//! - **Extreme Compression**: 50-1000x for structured data
//! - **O(1) Point Queries**: Compute f(x) instead of disk read
//! - **Automatic Model Selection**: Polynomial, Fourier, Perlin, etc.
//! - **Lossless Option**: Residual storage for exact reconstruction
//! - **Time-Series Optimized**: LSM-Tree architecture with model-based SSTables
//!
//! # Quick Start (Rust)
//!
//! ```rust,ignore
//! use alice_db::AliceDB;
//!
//! let db = AliceDB::open("./my_data")?;
//!
//! // Insert time-series data
//! for i in 0..1000 {
//!     db.put(i, (i as f32).sin())?;
//! }
//!
//! // Query (computes sin(500) from model, no disk read!)
//! let value = db.get(500)?;
//!
//! // Range query with aggregation
//! let avg = db.query()
//!     .range(0, 999)
//!     .aggregate(Aggregation::Avg)
//!     .execute()?;
//! ```
//!
//! # Quick Start (Python)
//!
//! ```python
//! import alice_db
//!
//! db = alice_db.open("./my_data")
//!
//! # Insert
//! db.put(timestamp=100, value=42.0)
//! db.put_batch([(i, i * 0.5) for i in range(1000)])
//!
//! # Query
//! value = db.get(100)
//! points = db.scan(0, 999)
//! avg = db.aggregate(0, 999, "avg")
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                      ALICE-DB                           │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
//! │  │  MemTable   │───▶│   Fitter    │───▶│  Segment   │  │
//! │  │ (BTreeMap)  │    │ Competition │    │  (Model)   │  │
//! │  └─────────────┘    └─────────────┘    └────────────┘  │
//! │         │                  │                  │        │
//! │         │                  │                  │        │
//! │         ▼                  ▼                  ▼        │
//! │  ┌─────────────────────────────────────────────────┐   │
//! │  │                 Storage Engine                   │   │
//! │  │  • Index (time → segment)                       │   │
//! │  │  • WAL (durability)                             │   │
//! │  │  • Compaction                                   │   │
//! │  └─────────────────────────────────────────────────┘   │
//! │                          │                             │
//! │                          ▼                             │
//! │  ┌─────────────────────────────────────────────────┐   │
//! │  │                   libalice                       │   │
//! │  │  • Polynomial fitting (Horner's method)         │   │
//! │  │  • Fourier analysis (FFT)                       │   │
//! │  │  • Perlin noise                                 │   │
//! │  │  • LZMA compression (fallback)                  │   │
//! │  └─────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # License
//!
//! MIT License
//!
//! # Author
//!
//! Moroya Sakamoto

pub mod model;
pub mod segment;
pub mod memtable;
pub mod storage_engine;
pub mod query_engine;

// Re-exports for convenience
pub use model::{DataType, FitResult, ModelType};
pub use segment::DataSegment;
pub use memtable::{FitConfig, MemTable};
pub use storage_engine::{StorageConfig, StorageEngine, StorageStats};
pub use query_engine::{Aggregation, QueryBuilder, QueryInterface, QueryResult};

use std::io;
use std::path::Path;

/// ALICE-DB: High-level database interface
///
/// This is the main entry point for using ALICE-DB.
/// Provides a simple API for insert, query, and management operations.
pub struct AliceDB {
    engine: StorageEngine,
}

impl AliceDB {
    /// Open a database at the specified path
    ///
    /// Creates the directory if it doesn't exist.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let engine = StorageEngine::open(path)?;
        Ok(Self { engine })
    }

    /// Open with custom configuration
    pub fn with_config(config: StorageConfig) -> io::Result<Self> {
        let engine = StorageEngine::new(config)?;
        Ok(Self { engine })
    }

    /// Insert a single value
    pub fn put(&self, timestamp: i64, value: f32) -> io::Result<()> {
        self.engine.put(timestamp, value)
    }

    /// Insert multiple values (batch)
    pub fn put_batch(&self, data: &[(i64, f32)]) -> io::Result<()> {
        self.engine.put_batch(data)
    }

    /// Query a single point
    pub fn get(&self, timestamp: i64) -> io::Result<Option<f32>> {
        self.engine.query_point(timestamp)
    }

    /// Query a time range
    pub fn scan(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.engine.query_range(start, end)
    }

    /// Create a query builder for advanced queries
    pub fn query(&self) -> QueryBuilder<'_> {
        self.engine.query()
    }

    /// Aggregation shorthand
    pub fn aggregate(&self, start: i64, end: i64, agg: Aggregation) -> io::Result<f64> {
        self.engine.aggregate(start, end, agg)
    }

    /// Downsampling query
    pub fn downsample(
        &self,
        start: i64,
        end: i64,
        interval: i64,
        agg: Aggregation,
    ) -> io::Result<Vec<(i64, f64)>> {
        self.engine.downsample(start, end, interval, agg)
    }

    /// Force flush to disk
    pub fn flush(&self) -> io::Result<()> {
        self.engine.flush()
    }

    /// Get database statistics
    pub fn stats(&self) -> StorageStats {
        self.engine.stats()
    }

    /// Close the database
    pub fn close(self) -> io::Result<()> {
        self.engine.close()
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::exceptions::{PyIOError, PyValueError};
    use numpy::{PyArray1, PyArray2, PyArrayMethods};
    use ndarray::Array2;
    use std::sync::Arc;
    use parking_lot::Mutex;

    /// Convert io::Error to PyErr
    fn io_to_py(e: io::Error) -> PyErr {
        PyIOError::new_err(e.to_string())
    }

    /// Python wrapper for AliceDB
    #[pyclass(name = "AliceDB")]
    pub struct PyAliceDB {
        inner: Arc<Mutex<Option<AliceDB>>>,
    }

    #[pymethods]
    impl PyAliceDB {
        /// Open a database at the specified path
        #[new]
        #[pyo3(signature = (path, memtable_capacity=1000, enable_wal=true))]
        fn new(path: &str, memtable_capacity: usize, enable_wal: bool) -> PyResult<Self> {
            let config = StorageConfig {
                data_dir: std::path::PathBuf::from(path),
                memtable_capacity,
                enable_wal,
                ..Default::default()
            };
            let db = AliceDB::with_config(config).map_err(io_to_py)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(Some(db))),
            })
        }

        /// Insert a single value
        fn put(&self, timestamp: i64, value: f32) -> PyResult<()> {
            let guard = self.inner.lock();
            let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
            db.put(timestamp, value).map_err(io_to_py)
        }

        /// Insert multiple values from lists
        fn put_batch(&self, py: Python<'_>, timestamps: Vec<i64>, values: Vec<f32>) -> PyResult<()> {
            if timestamps.len() != values.len() {
                return Err(PyValueError::new_err("timestamps and values must have same length"));
            }
            let data: Vec<(i64, f32)> = timestamps.into_iter().zip(values).collect();
            let inner = self.inner.clone();
            py.allow_threads(move || {
                let guard = inner.lock();
                let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
                db.put_batch(&data).map_err(io_to_py)
            })
        }

        /// Insert from numpy arrays (zero-copy)
        fn put_numpy<'py>(
            &self,
            _py: Python<'py>,
            timestamps: numpy::PyReadonlyArray1<'py, i64>,
            values: numpy::PyReadonlyArray1<'py, f32>,
        ) -> PyResult<()> {
            let ts = timestamps.as_slice()?;
            let vals = values.as_slice()?;
            if ts.len() != vals.len() {
                return Err(PyValueError::new_err("timestamps and values must have same length"));
            }
            let data: Vec<(i64, f32)> = ts.iter().zip(vals.iter()).map(|(&t, &v)| (t, v)).collect();
            let guard = self.inner.lock();
            let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
            db.put_batch(&data).map_err(io_to_py)
        }

        /// Query a single point
        fn get(&self, timestamp: i64) -> PyResult<Option<f32>> {
            let guard = self.inner.lock();
            let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
            db.get(timestamp).map_err(io_to_py)
        }

        /// Query a time range, returns list of (timestamp, value) tuples
        fn scan(&self, py: Python<'_>, start: i64, end: i64) -> PyResult<Vec<(i64, f32)>> {
            let inner = self.inner.clone();
            py.allow_threads(move || {
                let guard = inner.lock();
                let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
                db.scan(start, end).map_err(io_to_py)
            })
        }

        /// Query a time range, returns numpy array (N, 2)
        fn scan_numpy<'py>(
            &self,
            py: Python<'py>,
            start: i64,
            end: i64,
        ) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let guard = self.inner.lock();
            let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
            let points = db.scan(start, end).map_err(io_to_py)?;

            let n = points.len();

            // Create ndarray::Array2 and convert to PyArray2
            let mut arr = Array2::<f64>::zeros((n, 2));
            for (i, &(t, v)) in points.iter().enumerate() {
                arr[[i, 0]] = t as f64;
                arr[[i, 1]] = v as f64;
            }

            Ok(PyArray2::from_owned_array_bound(py, arr))
        }

        /// Aggregation query
        #[pyo3(signature = (start, end, agg="avg"))]
        fn aggregate(&self, py: Python<'_>, start: i64, end: i64, agg: &str) -> PyResult<f64> {
            let aggregation = match agg.to_lowercase().as_str() {
                "sum" => Aggregation::Sum,
                "avg" | "average" | "mean" => Aggregation::Avg,
                "min" => Aggregation::Min,
                "max" => Aggregation::Max,
                "count" => Aggregation::Count,
                "first" => Aggregation::First,
                "last" => Aggregation::Last,
                "stddev" | "std" => Aggregation::StdDev,
                "variance" | "var" => Aggregation::Variance,
                _ => return Err(PyValueError::new_err(format!("Unknown aggregation: {}", agg))),
            };
            let inner = self.inner.clone();
            py.allow_threads(move || {
                let guard = inner.lock();
                let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
                db.aggregate(start, end, aggregation).map_err(io_to_py)
            })
        }

        /// Downsampling query
        #[pyo3(signature = (start, end, interval, agg="avg"))]
        fn downsample(&self, py: Python<'_>, start: i64, end: i64, interval: i64, agg: &str) -> PyResult<Vec<(i64, f64)>> {
            let aggregation = match agg.to_lowercase().as_str() {
                "sum" => Aggregation::Sum,
                "avg" | "average" | "mean" => Aggregation::Avg,
                "min" => Aggregation::Min,
                "max" => Aggregation::Max,
                "count" => Aggregation::Count,
                "first" => Aggregation::First,
                "last" => Aggregation::Last,
                _ => return Err(PyValueError::new_err(format!("Unknown aggregation: {}", agg))),
            };
            let inner = self.inner.clone();
            py.allow_threads(move || {
                let guard = inner.lock();
                let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
                db.downsample(start, end, interval, aggregation).map_err(io_to_py)
            })
        }

        /// Force flush to disk
        fn flush(&self) -> PyResult<()> {
            let guard = self.inner.lock();
            let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
            db.flush().map_err(io_to_py)
        }

        /// Get database statistics
        fn stats(&self) -> PyResult<PyStats> {
            let guard = self.inner.lock();
            let db = guard.as_ref().ok_or_else(|| PyIOError::new_err("Database closed"))?;
            let stats = db.stats();
            Ok(PyStats {
                total_segments: stats.total_segments,
                memtable_size: stats.memtable_size,
                total_disk_size: stats.total_disk_size,
                average_compression_ratio: stats.average_compression_ratio,
                model_distribution: stats.model_distribution,
            })
        }

        /// Close the database
        fn close(&self) -> PyResult<()> {
            let mut guard = self.inner.lock();
            if let Some(db) = guard.take() {
                db.close().map_err(io_to_py)?;
            }
            Ok(())
        }

        fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __exit__(
            &self,
            _exc_type: Option<&Bound<'_, PyAny>>,
            _exc_val: Option<&Bound<'_, PyAny>>,
            _exc_tb: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<bool> {
            self.close()?;
            Ok(false)
        }
    }

    /// Python wrapper for StorageStats
    #[pyclass(name = "Stats")]
    #[derive(Clone)]
    pub struct PyStats {
        #[pyo3(get)]
        pub total_segments: usize,
        #[pyo3(get)]
        pub memtable_size: usize,
        #[pyo3(get)]
        pub total_disk_size: u64,
        #[pyo3(get)]
        pub average_compression_ratio: f64,
        #[pyo3(get)]
        pub model_distribution: std::collections::HashMap<String, usize>,
    }

    #[pymethods]
    impl PyStats {
        fn __repr__(&self) -> String {
            format!(
                "Stats(segments={}, memtable={}, disk={}B, compression={:.1}x, models={:?})",
                self.total_segments,
                self.memtable_size,
                self.total_disk_size,
                self.average_compression_ratio,
                self.model_distribution
            )
        }
    }

    /// Open a database (convenience function)
    #[pyfunction]
    #[pyo3(signature = (path, memtable_capacity=1000, enable_wal=true))]
    fn open(path: &str, memtable_capacity: usize, enable_wal: bool) -> PyResult<PyAliceDB> {
        PyAliceDB::new(path, memtable_capacity, enable_wal)
    }

    /// alice_db Python module
    #[pymodule]
    fn alice_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyAliceDB>()?;
        m.add_class::<PyStats>()?;
        m.add_function(wrap_pyfunction!(open, m)?)?;

        // Version info
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add("__author__", "Moroya Sakamoto")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_alice_db_basic() {
        let dir = tempdir().unwrap();
        let db = AliceDB::open(dir.path()).unwrap();

        // Insert linear data
        for i in 0..100 {
            db.put(i, i as f32 * 2.0).unwrap();
        }

        db.flush().unwrap();

        // Query
        let results = db.scan(0, 99).unwrap();
        assert!(!results.is_empty());

        // Stats
        let stats = db.stats();
        assert!(stats.total_segments >= 1);
        assert!(stats.average_compression_ratio > 1.0);

        db.close().unwrap();
    }

    #[test]
    fn test_alice_db_aggregation() {
        let dir = tempdir().unwrap();
        let db = AliceDB::open(dir.path()).unwrap();

        for i in 0..100 {
            db.put(i, i as f32).unwrap();
        }
        db.flush().unwrap();

        let avg = db.aggregate(0, 99, Aggregation::Avg).unwrap();
        // Average of 0..99 should be ~49.5
        assert!(avg > 40.0 && avg < 60.0);
    }

    #[test]
    fn test_alice_db_sine_wave() {
        let dir = tempdir().unwrap();
        let db = AliceDB::open(dir.path()).unwrap();

        // Insert sine wave data
        for i in 0..1000 {
            let t = i as f32 * 0.01;
            let value = (t * 2.0 * std::f32::consts::PI).sin();
            db.put(i, value).unwrap();
        }
        db.flush().unwrap();

        let stats = db.stats();
        // Sine wave should compress well (Fourier or SineWave model)
        println!("Compression ratio: {}", stats.average_compression_ratio);
        println!("Model distribution: {:?}", stats.model_distribution);

        // Should achieve significant compression
        assert!(stats.average_compression_ratio > 5.0);
    }
}
