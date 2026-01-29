//! Query Engine: High-level query interface
//!
//! Provides SQL-like query capabilities for ALICE-DB.
//! Optimized for time-series data access patterns.
//!
//! # Key Innovation
//!
//! Queries don't read data from disk - they compute it on the fly!
//!
//! ```text
//! SELECT value FROM sensor WHERE time BETWEEN 100 AND 200
//!
//! Traditional DB: Disk seek → Read pages → Decompress → Filter → Return
//! ALICE-DB:       Load model → Compute f(x) for x∈[100,200] → Return
//! ```
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::storage_engine::StorageEngine;
use std::io;

/// Aggregation functions for time-series queries
#[derive(Debug, Clone, Copy)]
pub enum Aggregation {
    /// No aggregation (return raw points)
    None,
    /// Sum of values
    Sum,
    /// Average value
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of points
    Count,
    /// First value in range
    First,
    /// Last value in range
    Last,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
}

/// Query result
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// Time-series data points
    Points(Vec<(i64, f32)>),
    /// Single aggregated value
    Scalar(f64),
    /// Multiple aggregated values (for GROUP BY)
    Aggregates(Vec<(i64, f64)>),
}

impl QueryResult {
    /// Get as points (panics if not Points variant)
    pub fn into_points(self) -> Vec<(i64, f32)> {
        match self {
            QueryResult::Points(p) => p,
            _ => panic!("Expected Points result"),
        }
    }

    /// Get as scalar (panics if not Scalar variant)
    pub fn into_scalar(self) -> f64 {
        match self {
            QueryResult::Scalar(v) => v,
            _ => panic!("Expected Scalar result"),
        }
    }

    /// Get as aggregates (panics if not Aggregates variant)
    pub fn into_aggregates(self) -> Vec<(i64, f64)> {
        match self {
            QueryResult::Aggregates(a) => a,
            _ => panic!("Expected Aggregates result"),
        }
    }
}

/// Query builder for fluent API
pub struct QueryBuilder<'a> {
    engine: &'a StorageEngine,
    start_time: Option<i64>,
    end_time: Option<i64>,
    aggregation: Aggregation,
    group_by_interval: Option<i64>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl<'a> QueryBuilder<'a> {
    /// Create a new query builder
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self {
            engine,
            start_time: None,
            end_time: None,
            aggregation: Aggregation::None,
            group_by_interval: None,
            limit: None,
            offset: None,
        }
    }

    /// Set time range start
    pub fn from(mut self, start: i64) -> Self {
        self.start_time = Some(start);
        self
    }

    /// Set time range end
    pub fn to(mut self, end: i64) -> Self {
        self.end_time = Some(end);
        self
    }

    /// Set time range (convenience method)
    pub fn range(mut self, start: i64, end: i64) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Set aggregation function
    pub fn aggregate(mut self, agg: Aggregation) -> Self {
        self.aggregation = agg;
        self
    }

    /// Group by time interval (for downsampling)
    pub fn group_by(mut self, interval: i64) -> Self {
        self.group_by_interval = Some(interval);
        self
    }

    /// Limit number of results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Skip first n results
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Execute the query
    pub fn execute(self) -> io::Result<QueryResult> {
        let start = self.start_time.unwrap_or(i64::MIN);
        let end = self.end_time.unwrap_or(i64::MAX);

        // Get raw data points
        let mut points = self.engine.query_range(start, end)?;

        // Apply offset
        if let Some(offset) = self.offset {
            if offset < points.len() {
                points = points[offset..].to_vec();
            } else {
                points.clear();
            }
        }

        // Apply limit
        if let Some(limit) = self.limit {
            points.truncate(limit);
        }

        // Apply grouping if specified
        if let Some(interval) = self.group_by_interval {
            return Ok(self.apply_grouped_aggregation(&points, interval));
        }

        // Apply aggregation
        match self.aggregation {
            Aggregation::None => Ok(QueryResult::Points(points)),
            Aggregation::Sum => Ok(QueryResult::Scalar(
                points.iter().map(|&(_, v)| v as f64).sum()
            )),
            Aggregation::Avg => {
                if points.is_empty() {
                    Ok(QueryResult::Scalar(0.0))
                } else {
                    let sum: f64 = points.iter().map(|&(_, v)| v as f64).sum();
                    Ok(QueryResult::Scalar(sum / points.len() as f64))
                }
            }
            Aggregation::Min => Ok(QueryResult::Scalar(
                points
                    .iter()
                    .map(|&(_, v)| v as f64)
                    .fold(f64::INFINITY, f64::min)
            )),
            Aggregation::Max => Ok(QueryResult::Scalar(
                points
                    .iter()
                    .map(|&(_, v)| v as f64)
                    .fold(f64::NEG_INFINITY, f64::max)
            )),
            Aggregation::Count => Ok(QueryResult::Scalar(points.len() as f64)),
            Aggregation::First => {
                if let Some(&(_, v)) = points.first() {
                    Ok(QueryResult::Scalar(v as f64))
                } else {
                    Ok(QueryResult::Scalar(0.0))
                }
            }
            Aggregation::Last => {
                if let Some(&(_, v)) = points.last() {
                    Ok(QueryResult::Scalar(v as f64))
                } else {
                    Ok(QueryResult::Scalar(0.0))
                }
            }
            Aggregation::StdDev => {
                let variance = self.calculate_variance(&points);
                Ok(QueryResult::Scalar(variance.sqrt()))
            }
            Aggregation::Variance => {
                Ok(QueryResult::Scalar(self.calculate_variance(&points)))
            }
        }
    }

    /// Calculate variance of points
    fn calculate_variance(&self, points: &[(i64, f32)]) -> f64 {
        if points.is_empty() {
            return 0.0;
        }

        let mean: f64 = points.iter().map(|&(_, v)| v as f64).sum::<f64>() / points.len() as f64;
        points
            .iter()
            .map(|&(_, v)| (v as f64 - mean).powi(2))
            .sum::<f64>() / points.len() as f64
    }

    /// Apply aggregation to grouped data (Streaming O(1) space)
    ///
    /// # Performance: Streaming Algorithm
    ///
    /// Instead of BTreeMap<bucket, Vec<f32>> which uses O(n) space,
    /// we use a single-pass streaming algorithm with O(1) space.
    /// Points must be sorted by timestamp (which they are from query_range).
    fn apply_grouped_aggregation(&self, points: &[(i64, f32)], interval: i64) -> QueryResult {
        if points.is_empty() {
            return QueryResult::Aggregates(Vec::new());
        }

        // Streaming aggregator state (O(1) space per bucket)
        struct BucketState {
            sum: f64,
            count: u64,
            min: f32,
            max: f32,
            first: f32,
            last: f32,
            // For variance (Welford's online algorithm)
            mean: f64,
            m2: f64,
        }

        impl BucketState {
            fn new(first_value: f32) -> Self {
                Self {
                    sum: first_value as f64,
                    count: 1,
                    min: first_value,
                    max: first_value,
                    first: first_value,
                    last: first_value,
                    mean: first_value as f64,
                    m2: 0.0,
                }
            }

            fn update(&mut self, value: f32) {
                let v = value as f64;
                self.sum += v;
                self.count += 1;
                self.min = self.min.min(value);
                self.max = self.max.max(value);
                self.last = value;

                // Welford's online algorithm for variance
                let delta = v - self.mean;
                self.mean += delta / self.count as f64;
                let delta2 = v - self.mean;
                self.m2 += delta * delta2;
            }

            fn finalize(&self, agg: Aggregation) -> f64 {
                match agg {
                    Aggregation::None | Aggregation::First => self.first as f64,
                    Aggregation::Sum => self.sum,
                    Aggregation::Avg => self.sum / self.count as f64,
                    Aggregation::Min => self.min as f64,
                    Aggregation::Max => self.max as f64,
                    Aggregation::Count => self.count as f64,
                    Aggregation::Last => self.last as f64,
                    Aggregation::StdDev => {
                        if self.count > 1 {
                            (self.m2 / self.count as f64).sqrt()
                        } else {
                            0.0
                        }
                    }
                    Aggregation::Variance => {
                        if self.count > 1 {
                            self.m2 / self.count as f64
                        } else {
                            0.0
                        }
                    }
                }
            }
        }

        // Estimate result size
        let first_bucket = (points[0].0 / interval) * interval;
        let last_bucket = (points[points.len() - 1].0 / interval) * interval;
        let estimated_buckets = ((last_bucket - first_bucket) / interval + 1) as usize;

        let mut results = Vec::with_capacity(estimated_buckets);
        let mut current_bucket = first_bucket;
        let mut state = BucketState::new(points[0].1);

        // Single-pass streaming: O(n) time, O(1) space per bucket
        for &(t, v) in &points[1..] {
            let bucket = (t / interval) * interval;

            if bucket != current_bucket {
                // Emit completed bucket
                results.push((current_bucket, state.finalize(self.aggregation)));

                // Start new bucket
                current_bucket = bucket;
                state = BucketState::new(v);
            } else {
                state.update(v);
            }
        }

        // Emit final bucket
        results.push((current_bucket, state.finalize(self.aggregation)));

        QueryResult::Aggregates(results)
    }
}

/// High-level query interface
pub trait QueryInterface {
    /// Create a new query builder
    fn query(&self) -> QueryBuilder<'_>;

    /// Shorthand for point query
    fn get(&self, timestamp: i64) -> io::Result<Option<f32>>;

    /// Shorthand for range query
    fn scan(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>>;

    /// Shorthand for aggregation query
    fn aggregate(&self, start: i64, end: i64, agg: Aggregation) -> io::Result<f64>;

    /// Downsampling query
    fn downsample(
        &self,
        start: i64,
        end: i64,
        interval: i64,
        agg: Aggregation,
    ) -> io::Result<Vec<(i64, f64)>>;
}

impl QueryInterface for StorageEngine {
    fn query(&self) -> QueryBuilder<'_> {
        QueryBuilder::new(self)
    }

    fn get(&self, timestamp: i64) -> io::Result<Option<f32>> {
        self.query_point(timestamp)
    }

    fn scan(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.query_range(start, end)
    }

    fn aggregate(&self, start: i64, end: i64, agg: Aggregation) -> io::Result<f64> {
        let result = self
            .query()
            .range(start, end)
            .aggregate(agg)
            .execute()?;
        Ok(result.into_scalar())
    }

    fn downsample(
        &self,
        start: i64,
        end: i64,
        interval: i64,
        agg: Aggregation,
    ) -> io::Result<Vec<(i64, f64)>> {
        let result = self
            .query()
            .range(start, end)
            .group_by(interval)
            .aggregate(agg)
            .execute()?;
        Ok(result.into_aggregates())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage_engine::StorageConfig;
    use tempfile::tempdir;

    fn setup_engine_with_data() -> (tempfile::TempDir, StorageEngine) {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 1000,
            enable_wal: false,
            ..Default::default()
        };

        let engine = StorageEngine::new(config).unwrap();

        // Insert linear data: value = timestamp
        for i in 0..100 {
            engine.put(i, i as f32).unwrap();
        }
        engine.flush().unwrap();

        (dir, engine)
    }

    #[test]
    fn test_query_builder_range() {
        let (_dir, engine) = setup_engine_with_data();

        let result = engine
            .query()
            .range(0, 50)
            .execute()
            .unwrap();

        let points = result.into_points();
        assert!(!points.is_empty());
    }

    #[test]
    fn test_query_aggregations() {
        let (_dir, engine) = setup_engine_with_data();

        // Sum
        let sum = engine.aggregate(0, 99, Aggregation::Sum).unwrap();
        let expected_sum: f64 = (0..100).map(|i| i as f64).sum();
        assert!((sum - expected_sum).abs() < 1.0);

        // Avg
        let avg = engine.aggregate(0, 99, Aggregation::Avg).unwrap();
        let expected_avg = expected_sum / 100.0;
        assert!((avg - expected_avg).abs() < 1.0);

        // Count
        let count = engine.aggregate(0, 99, Aggregation::Count).unwrap();
        assert!(count >= 90.0); // Should be close to 100

        // Min
        let min = engine.aggregate(0, 99, Aggregation::Min).unwrap();
        assert!(min < 5.0); // Should be close to 0

        // Max
        let max = engine.aggregate(0, 99, Aggregation::Max).unwrap();
        assert!(max > 90.0); // Should be close to 99
    }

    #[test]
    fn test_query_downsample() {
        let (_dir, engine) = setup_engine_with_data();

        let downsampled = engine
            .downsample(0, 99, 10, Aggregation::Avg)
            .unwrap();

        // Should have ~10 buckets
        assert!(!downsampled.is_empty());
        assert!(downsampled.len() <= 10);
    }

    #[test]
    fn test_query_limit_offset() {
        let (_dir, engine) = setup_engine_with_data();

        let result = engine
            .query()
            .range(0, 99)
            .limit(10)
            .execute()
            .unwrap();

        let points = result.into_points();
        assert!(points.len() <= 10);
    }
}
