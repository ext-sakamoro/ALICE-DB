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
//! as time-series data for long-term storage and querying.
//!
//! # Architecture
//!
//! ```text
//! MetricPipeline (HLL++, DDSketch streaming)
//!         ↓ flush_metrics_to_db
//! AliceDB (Model-Based LSM-Tree persistent storage)
//! ```
//!
//! # Storage Key Schema
//!
//! Each metric slot produces up to 6 time-series entries, keyed by:
//! ```text
//! key = (name_hash % 0xFFFFF) << 44 | (timestamp & 0xFFFFFFFFFF) << 4 | variant
//! ```
//!
//! | Variant | Value | Description |
//! |---------|-------|-------------|
//! | 0 | counter | Aggregated counter |
//! | 1 | gauge | Last gauge value |
//! | 2 | cardinality | HLL++ unique count |
//! | 3 | p50 | DDSketch median |
//! | 4 | p90 | DDSketch 90th percentile |
//! | 5 | p99 | DDSketch 99th percentile |

use alice_analytics::pipeline::MetricPipeline;
use crate::AliceDB;
use std::io;

/// Number of variants stored per metric (counter, gauge, cardinality, p50, p90, p99).
pub const VARIANTS_PER_METRIC: u8 = 6;

/// Compute a composite storage key for a metric variant.
///
/// Packs metric identity (20 bits), timestamp (40 bits), and variant (4 bits)
/// into a single i64 key for non-overlapping storage.
///
/// - Supports up to ~1M distinct metrics
/// - Supports timestamps up to ~34 years at 1-second granularity
/// - Supports up to 16 variants per metric
#[inline]
pub fn metric_key(name_hash: u64, timestamp: i64, variant: u8) -> i64 {
    let nh = (name_hash & 0xFFFFF) as i64;
    let ts = timestamp & 0xFF_FFFF_FFFF;
    (nh << 44) | (ts << 4) | (variant as i64)
}

/// Flush all non-empty metric slots from a pipeline into the database.
///
/// Each slot's counter, gauge, cardinality, and quantiles are stored
/// as separate time-series entries keyed by `metric_key(name_hash, timestamp, variant)`.
///
/// # Arguments
///
/// - `pipeline`: The analytics pipeline to read metrics from
/// - `db`: The database to write into
/// - `timestamp`: The epoch timestamp for this flush cycle
///
/// Returns the number of metric entries written.
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

        // Counter
        batch.push((metric_key(nh, timestamp, 0), slot.counter as f32));
        // Gauge
        batch.push((metric_key(nh, timestamp, 1), slot.gauge as f32));
        // Cardinality (HLL)
        batch.push((metric_key(nh, timestamp, 2), slot.hll.cardinality() as f32));

        // Quantiles (DDSketch) — only if observations exist
        if slot.ddsketch.count() > 0 {
            batch.push((metric_key(nh, timestamp, 3), slot.ddsketch.quantile(0.50) as f32));
            batch.push((metric_key(nh, timestamp, 4), slot.ddsketch.quantile(0.90) as f32));
            batch.push((metric_key(nh, timestamp, 5), slot.ddsketch.quantile(0.99) as f32));
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
    /// - `db`: An opened AliceDB instance
    /// - `alpha`: DDSketch relative error parameter (e.g., 0.05 for 5%)
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
    /// - `alpha`: DDSketch relative error parameter
    pub fn open(path: &str, alpha: f64) -> io::Result<Self> {
        let db = AliceDB::open(path)?;
        Ok(Self::new(db, alpha))
    }

    /// Persist all current metric slots to the database.
    ///
    /// Flushes the pipeline's internal queue first, then writes all non-empty
    /// slots to the database using the given timestamp.
    ///
    /// Returns the number of entries written.
    pub fn persist(&mut self, timestamp: i64) -> io::Result<usize> {
        self.pipeline.flush();
        let count = flush_metrics_to_db(&self.pipeline, &self.db, timestamp)?;
        self.flush_count += 1;
        Ok(count)
    }

    /// Persist and then reset all metric slots for the next aggregation window.
    ///
    /// Returns the number of entries written.
    pub fn persist_and_reset(&mut self, timestamp: i64) -> io::Result<usize> {
        let count = self.persist(timestamp)?;
        self.pipeline.reset();
        Ok(count)
    }

    /// Force flush the database to disk.
    pub fn flush_db(&self) -> io::Result<()> {
        self.db.flush()
    }

    /// Number of completed flush cycles.
    #[inline]
    pub fn flush_count(&self) -> u64 {
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
        assert!(count >= 6, "count = {}", count);

        db.flush().unwrap();
        db.close().unwrap();
    }

    #[test]
    fn test_analytics_sink_persist() {
        let dir = tempdir().unwrap();

        let mut sink = AnalyticsSink::<64, 256>::open(
            dir.path().to_str().unwrap(),
            0.05,
        ).unwrap();

        let hash = FnvHasher::hash_bytes(b"sensor.temperature");

        // Submit gauge readings
        for v in [20.0, 21.0, 22.0, 23.0, 24.0] {
            sink.pipeline.submit(MetricEvent::gauge(hash, v));
        }

        // Persist
        let count = sink.persist(1000).unwrap();
        assert!(count >= 3, "count = {}", count); // counter, gauge, cardinality at minimum

        assert_eq!(sink.flush_count(), 1);

        sink.flush_db().unwrap();
    }

    #[test]
    fn test_analytics_sink_persist_and_reset() {
        let dir = tempdir().unwrap();

        let mut sink = AnalyticsSink::<64, 256>::open(
            dir.path().to_str().unwrap(),
            0.05,
        ).unwrap();

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
        assert_eq!(slot.counter, 30.0); // Reset worked, not 80

        let c2 = sink.persist(2000).unwrap();
        assert!(c2 >= 3);

        assert_eq!(sink.flush_count(), 2);
    }
}
