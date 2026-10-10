/*
    ALICE-DB
    Copyright (C) 2026 Moroya Sakamoto

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
*/

//! Bridge between ALICE-Analytics streaming aggregation and ALICE-DB persistent storage.
//!
//! Flushes aggregated metrics from [`MetricPipeline`] slots into [`AliceDB`]
//! and reads them back bit for bit.
//!
//! # Architecture
//!
//! ```text
//! MetricPipeline (HLL++, DDSketch streaming)
//!         ↓ flush_metrics_to_db
//! AliceDB blob key-value store (exact: every value reads back bit for bit)
//!         ↑ read_metric / scan_metric
//! ```
//!
//! # Where the values go
//!
//! Each value is one record of the blob key-value store
//! ([`AliceDB::put_blob`]), the same exact path `law_store` uses. Until 0.3.0
//! the bridge wrote into the model-based time-series store
//! ([`AliceDB::put_batch`]); that store keeps a fitted model per segment and
//! assumes evenly spaced keys, while metric keys leave gaps of up to 2^44
//! between metrics, so `get` / `scan` returned values of other keys (840
//! keys written, 0 read back exactly). Values written that way by
//! 0.3.0-beta.x stay in the time-series store and are not migrated; they can
//! still be read approximately with [`AliceDB::get`] / [`AliceDB::scan`] at
//! [`metric_key`], with the inaccuracy described above.
//!
//! # Storage Key Schema
//!
//! Each metric slot produces up to 6 records. The metric key packs the
//! metric identity, the timestamp and the variant:
//! ```text
//! metric_key = (name_hash & 0xFFFFF) << 44 | (timestamp & 0xFF_FFFF_FFFF) << 4 | variant
//! ```
//! and the blob key is [`METRIC_KEY_PREFIX`] followed by `metric_key` as a
//! `u64` in big-endian byte order ([`metric_blob_key`]), so a prefix scan
//! returns one metric's records ordered by `(timestamp, variant)`. The value
//! is the `f32` bit pattern in little-endian byte order (4 bytes).
//!
//! Only the low 20 bits of the name hash and the low 40 bits of the timestamp
//! take part: two names whose hashes agree in the low 20 bits share their
//! records, and timestamps that agree in the low 40 bits address the same
//! record. Readers apply the same masks, so reading with the arguments used
//! for writing finds the record.
//!
//! | Variant | Value | Description |
//! |---------|-------|-------------|
//! | 0 | counter | Aggregated counter |
//! | 1 | gauge | Last gauge value |
//! | 2 | cardinality | HLL++ unique count |
//! | 3 | p50 | DDSketch median |
//! | 4 | p90 | DDSketch 90th percentile |
//! | 5 | p99 | DDSketch 99th percentile |

use crate::AliceDB;
use alice_analytics::pipeline::MetricPipeline;
use std::io;

/// Number of variants stored per metric (counter, gauge, cardinality, p50, p90, p99).
pub const VARIANTS_PER_METRIC: u8 = 6;

/// Blob key prefix reserved for metric records.
///
/// Disjoint from [`crate::law_store::LAW_KEY_PREFIX`] (`"\0alice-law\0"`):
/// the two differ at their eighth byte, so neither is a prefix of the other
/// and a prefix scan of one never returns records of the other.
pub const METRIC_KEY_PREFIX: &[u8] = b"\0alice-metric\0";

/// Length of a metric blob key: [`METRIC_KEY_PREFIX`] plus the 8-byte key.
pub const METRIC_BLOB_KEY_LEN: usize = METRIC_KEY_PREFIX.len() + 8;

/// Mask of the name hash bits that take part in [`metric_key`].
const NAME_HASH_MASK: u64 = 0xF_FFFF;

/// Mask of the timestamp bits that take part in [`metric_key`].
const TIMESTAMP_MASK: i64 = 0xFF_FFFF_FFFF;

/// Compute a composite storage key for a metric variant.
///
/// Packs metric identity (20 bits), timestamp (40 bits), and variant (4 bits)
/// into a single i64 key for non-overlapping storage.
///
/// - Supports up to ~1M distinct metrics
/// - Supports timestamps up to ~34 years at 1-second granularity
/// - Supports up to 16 variants per metric
#[inline]
#[must_use]
pub const fn metric_key(name_hash: u64, timestamp: i64, variant: u8) -> i64 {
    let nh = (name_hash & NAME_HASH_MASK) as i64;
    let ts = timestamp & TIMESTAMP_MASK;
    (nh << 44) | (ts << 4) | (variant as i64)
}

/// Blob key of one metric record: [`METRIC_KEY_PREFIX`] followed by
/// [`metric_key`] as a `u64` in big-endian byte order.
///
/// Big-endian keeps the byte order of the keys equal to the numeric order of
/// `metric_key as u64`, so a prefix scan returns a metric's records ordered
/// by `(timestamp, variant)`.
#[must_use]
pub fn metric_blob_key(name_hash: u64, timestamp: i64, variant: u8) -> [u8; METRIC_BLOB_KEY_LEN] {
    let mut key = [0u8; METRIC_BLOB_KEY_LEN];
    key[..METRIC_KEY_PREFIX.len()].copy_from_slice(METRIC_KEY_PREFIX);
    #[allow(
        clippy::cast_sign_loss,
        reason = "the key is a bit pattern; the top bit belongs to the name hash"
    )]
    let bits = metric_key(name_hash, timestamp, variant) as u64;
    key[METRIC_KEY_PREFIX.len()..].copy_from_slice(&bits.to_be_bytes());
    key
}

/// Metric name hash as `MetricPipeline` uses it (`FnvHasher` over the UTF-8
/// bytes of the name).
#[must_use]
pub fn metric_name_hash(name: &str) -> u64 {
    alice_analytics::sketch::FnvHasher::hash_bytes(name.as_bytes())
}

/// One stored metric value, as returned by [`scan_metric`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricPoint {
    /// Timestamp of the flush cycle (the low 40 bits that the key keeps)
    pub timestamp: i64,
    /// Variant (`0` counter … `5` p99, see the module docs)
    pub variant: u8,
    /// The value exactly as written
    pub value: f32,
}

/// Encode a metric value: the `f32` bit pattern, little-endian.
#[inline]
const fn encode_value(value: f32) -> [u8; 4] {
    value.to_bits().to_le_bytes()
}

/// Decode a stored metric value.
fn decode_value(bytes: &[u8]) -> io::Result<f32> {
    let raw: [u8; 4] = bytes.try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("metric record holds {} bytes, expected 4", bytes.len()),
        )
    })?;
    Ok(f32::from_bits(u32::from_le_bytes(raw)))
}

/// Read one metric value exactly as [`flush_metrics_to_db`] wrote it.
///
/// Returns `Ok(None)` when no record exists for `(name_hash, timestamp,
/// variant)` (for example the quantile variants of a metric without
/// histogram observations).
///
/// # Errors
///
/// Returns the error from [`AliceDB::get_blob`], or `InvalidData` when the
/// stored record is not 4 bytes long.
pub fn read_metric(
    db: &AliceDB,
    name_hash: u64,
    timestamp: i64,
    variant: u8,
) -> io::Result<Option<f32>> {
    db.get(metric_key(name_hash, timestamp, variant))
}

/// Every stored value of one metric, ordered by `(timestamp, variant)`.
///
/// # Errors
///
/// Returns the error from [`AliceDB::scan_blob_prefix`], or `InvalidData`
/// when a stored record is not 4 bytes long.
pub fn scan_metric(db: &AliceDB, name_hash: u64) -> io::Result<Vec<MetricPoint>> {
    let lo = metric_key(name_hash, 0, 0);
    let hi = lo | ((TIMESTAMP_MASK << 4) | 0xF);
    Ok(db
        .scan(lo, hi)?
        .into_iter()
        .map(|(k, value)| MetricPoint {
            timestamp: (k >> 4) & TIMESTAMP_MASK,
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            variant: (k & 0xF) as u8,
            value,
        })
        .collect())
}

/// Flush all non-empty metric slots from a pipeline into the database.
///
/// Each slot's counter, gauge, cardinality, and quantiles are stored as
/// separate records of the blob key-value store under
/// [`metric_blob_key`]`(name_hash, timestamp, variant)`, so every value reads
/// back bit for bit with [`read_metric`] / [`scan_metric`]. Writing the same
/// `(name_hash, timestamp, variant)` again overwrites the record.
///
/// # Arguments
///
/// - `pipeline`: The analytics pipeline to read metrics from
/// - `db`: The database to write into
/// - `timestamp`: The epoch timestamp for this flush cycle
///
/// Returns the number of metric entries written.
///
/// # Errors
///
/// Returns the error from [`AliceDB::put_blob`] when a record cannot be
/// written; records written before the failing one stay written.
pub fn flush_metrics_to_db<const SLOTS: usize, const QUEUE_SIZE: usize>(
    pipeline: &MetricPipeline<SLOTS, QUEUE_SIZE>,
    db: &AliceDB,
    timestamp: i64,
) -> io::Result<usize> {
    let mut batch = Vec::with_capacity(SLOTS * VARIANTS_PER_METRIC as usize);
    for slot in pipeline.iter_slots() {
        if slot.event_count == 0 {
            continue;
        }
        let nh = slot.name_hash;
        batch.push((metric_key(nh, timestamp, 0), slot.counter as f32));
        batch.push((metric_key(nh, timestamp, 1), slot.gauge as f32));
        batch.push((metric_key(nh, timestamp, 2), slot.hll.cardinality() as f32));
        if slot.ddsketch.count() > 0 {
            batch.push((
                metric_key(nh, timestamp, 3),
                slot.ddsketch.quantile(0.50) as f32,
            ));
            batch.push((
                metric_key(nh, timestamp, 4),
                slot.ddsketch.quantile(0.90) as f32,
            ));
            batch.push((
                metric_key(nh, timestamp, 5),
                slot.ddsketch.quantile(0.99) as f32,
            ));
        }
    }
    let count = batch.len();
    if !batch.is_empty() {
        db.put_batch(&batch)?;
    }
    Ok(count)
}

/// Combined Analytics pipeline + DB persistence sink.
///
/// Wraps [`MetricPipeline`] and [`AliceDB`] into a single struct that
/// accumulates streaming metrics and periodically flushes to persistent storage.
///
/// # Usage
///
/// ```rust,ignore
/// use alice_db::analytics_bridge::AnalyticsSink;
///
/// let sink = AnalyticsSink::<128, 512>::open("./metrics_db", 0.05)?;
///
/// // Submit metrics
/// sink.pipeline.submit(MetricEvent::counter(hash, 1.0));
/// sink.pipeline.flush();
///
/// // Periodically persist to disk
/// let written = sink.persist(current_timestamp)?;
/// ```
pub struct AnalyticsSink<const SLOTS: usize, const QUEUE_SIZE: usize> {
    /// Streaming analytics pipeline
    pub pipeline: MetricPipeline<SLOTS, QUEUE_SIZE>,
    /// Persistent storage
    pub db: AliceDB,
    /// Number of flush cycles completed
    pub flush_count: u64,
}

impl<const SLOTS: usize, const QUEUE_SIZE: usize> AnalyticsSink<SLOTS, QUEUE_SIZE> {
    /// Create a new analytics sink.
    ///
    /// - `db`: An opened `AliceDB` instance
    /// - `alpha`: `DDSketch` relative error parameter (e.g., 0.05 for 5%)
    pub fn new(db: AliceDB, alpha: f64) -> Self {
        Self {
            pipeline: MetricPipeline::new(alpha),
            db,
            flush_count: 0,
        }
    }

    /// Open a new analytics sink with a database at the given path.
    ///
    /// - `path`: Database directory path
    /// - `alpha`: `DDSketch` relative error parameter
    ///
    /// # Errors
    ///
    /// Returns the error from [`AliceDB::open`] when the database cannot be opened.
    pub fn open(path: &str, alpha: f64) -> io::Result<Self> {
        let db = AliceDB::open(path)?;
        Ok(Self::new(db, alpha))
    }

    /// Persist all current metric slots to the database.
    ///
    /// Flushes the pipeline's internal queue first, then writes all non-empty
    /// slots to the database using the given timestamp, and syncs the blob
    /// WAL ([`AliceDB::flush_blobs`]). When this returns `Ok`, every value is
    /// readable with [`Self::read`] / [`Self::scan`] and, on a file-backed
    /// database, survives a crash and reopen whatever the blob
    /// [`crate::blob_wal::SyncPolicy`].
    ///
    /// Returns the number of entries written.
    ///
    /// # Errors
    ///
    /// Returns the error from [`flush_metrics_to_db`] or
    /// [`AliceDB::flush_blobs`]; the flush count is not advanced.
    pub fn persist(&mut self, timestamp: i64) -> io::Result<usize> {
        self.pipeline.flush();
        let count = flush_metrics_to_db(&self.pipeline, &self.db, timestamp)?;
        self.db.flush_blobs()?;
        self.flush_count += 1;
        Ok(count)
    }

    /// Read one persisted value (see [`read_metric`]).
    ///
    /// # Errors
    ///
    /// Returns the error from [`read_metric`].
    pub fn read(&self, name_hash: u64, timestamp: i64, variant: u8) -> io::Result<Option<f32>> {
        read_metric(&self.db, name_hash, timestamp, variant)
    }

    /// Every persisted value of one metric, ordered by `(timestamp, variant)`
    /// (see [`scan_metric`]).
    ///
    /// # Errors
    ///
    /// Returns the error from [`scan_metric`].
    pub fn scan(&self, name_hash: u64) -> io::Result<Vec<MetricPoint>> {
        scan_metric(&self.db, name_hash)
    }

    /// Persist and then reset all metric slots for the next aggregation window.
    ///
    /// Returns the number of entries written.
    ///
    /// # Errors
    ///
    /// Returns the error from [`Self::persist`]; the slots are not reset.
    pub fn persist_and_reset(&mut self, timestamp: i64) -> io::Result<usize> {
        let count = self.persist(timestamp)?;
        self.pipeline.reset();
        Ok(count)
    }

    /// Force flush the database to disk: the blob WAL that holds the metrics
    /// ([`AliceDB::flush_blobs`]) and the time-series store
    /// ([`AliceDB::flush`]).
    ///
    /// # Errors
    ///
    /// Returns the error from [`AliceDB::flush_blobs`] or [`AliceDB::flush`].
    pub fn flush_db(&self) -> io::Result<()> {
        self.db.flush_blobs()?;
        self.db.flush()
    }

    /// Number of completed flush cycles.
    #[inline]
    pub const fn flush_count(&self) -> u64 {
        self.flush_count
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alice_analytics::pipeline::MetricEvent;
    use alice_analytics::sketch::FnvHasher;
    use tempfile::tempdir;

    #[test]
    fn test_metric_key_non_overlapping() {
        // Different metrics at same timestamp should produce different keys
        let k1 = metric_key(1, 1000, 0);
        let k2 = metric_key(2, 1000, 0);
        assert_ne!(k1, k2);

        // Same metric, different timestamps
        let k3 = metric_key(1, 1000, 0);
        let k4 = metric_key(1, 1001, 0);
        assert_ne!(k3, k4);

        // Same metric, same timestamp, different variants
        let k5 = metric_key(1, 1000, 0);
        let k6 = metric_key(1, 1000, 1);
        assert_ne!(k5, k6);
    }

    #[test]
    fn test_flush_metrics_to_db() {
        let dir = tempdir().unwrap();
        let db = AliceDB::open(dir.path()).unwrap();

        let mut pipeline = MetricPipeline::<64, 256>::new(0.05);

        let req_hash = FnvHasher::hash_bytes(b"http.requests");
        let lat_hash = FnvHasher::hash_bytes(b"http.latency");

        // Submit metrics
        for _ in 0..100 {
            pipeline.submit(MetricEvent::counter(req_hash, 1.0));
            pipeline.submit(MetricEvent::histogram(lat_hash, 50.0));
        }
        pipeline.flush();

        // Flush to DB
        let count = flush_metrics_to_db(&pipeline, &db, 1000).unwrap();
        // 2 metrics × (3 base + 3 quantiles for histogram, 3 base for counter)
        // req_hash: counter(3 base entries) + lat_hash: histogram(3 base + 3 quantiles)
        assert!(count >= 6, "count = {count}");

        db.flush().unwrap();
        db.close().unwrap();
    }

    #[test]
    fn test_analytics_sink_persist() {
        let dir = tempdir().unwrap();

        let mut sink = AnalyticsSink::<64, 256>::open(dir.path().to_str().unwrap(), 0.05).unwrap();

        let hash = FnvHasher::hash_bytes(b"sensor.temperature");

        // Submit gauge readings
        for v in [20.0, 21.0, 22.0, 23.0, 24.0] {
            sink.pipeline.submit(MetricEvent::gauge(hash, v));
        }

        // Persist
        let count = sink.persist(1000).unwrap();
        assert!(count >= 3, "count = {count}"); // counter, gauge, cardinality at minimum

        assert_eq!(sink.flush_count(), 1);

        sink.flush_db().unwrap();
    }

    #[test]
    fn test_analytics_sink_persist_and_reset() {
        let dir = tempdir().unwrap();

        let mut sink = AnalyticsSink::<64, 256>::open(dir.path().to_str().unwrap(), 0.05).unwrap();

        let hash = FnvHasher::hash_bytes(b"requests");

        // Window 1
        for _ in 0..50 {
            sink.pipeline.submit(MetricEvent::counter(hash, 1.0));
        }
        let c1 = sink.persist_and_reset(1000).unwrap();
        assert!(c1 >= 3);

        // Window 2 — pipeline is reset, counter starts fresh
        for _ in 0..30 {
            sink.pipeline.submit(MetricEvent::counter(hash, 1.0));
        }
        sink.pipeline.flush();

        let slot = sink.pipeline.get_slot(hash).unwrap();
        #[allow(
            clippy::float_cmp,
            reason = "30 additions of 1.0 are exact, so the reset is checked bit for bit"
        )]
        {
            assert_eq!(slot.counter, 30.0); // Reset worked, not 80
        }

        let c2 = sink.persist(2000).unwrap();
        assert!(c2 >= 3);

        assert_eq!(sink.flush_count(), 2);
    }
}
