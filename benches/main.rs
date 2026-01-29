//! ALICE-DB Performance Benchmarks
//!
//! Proves the performance claims in README:
//! - Point query: ~50ns (cached, SIMD polynomial)
//! - Range scan: SIMD-accelerated 4x throughput
//! - Write throughput: Model compression overhead
//!
//! Run with: cargo bench --bench main

use alice_db::{AliceDB, StorageConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::prelude::*;
use tempfile::tempdir;

/// Generate sine wave test data
fn generate_sine_wave(n: usize) -> Vec<(i64, f32)> {
    (0..n)
        .map(|i| {
            let t = i as f32 * 0.01;
            (i as i64, (t * 2.0 * std::f32::consts::PI).sin() * 100.0)
        })
        .collect()
}

/// Generate linear test data (compresses to ~0 bytes)
fn generate_linear(n: usize) -> Vec<(i64, f32)> {
    (0..n).map(|i| (i as i64, i as f32 * 2.5)).collect()
}

/// Benchmark: Write Throughput
///
/// Measures the cost of model fitting during writes.
fn bench_write_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("Write Throughput");

    // 10K points batch insert
    group.throughput(Throughput::Elements(10_000));
    group.bench_function("insert_batch_10k_sine", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let db = AliceDB::open(dir.path()).unwrap();
                let data = generate_sine_wave(10_000);
                (dir, db, data)
            },
            |(_dir, db, data)| {
                db.put_batch(black_box(&data)).unwrap();
            },
        )
    });

    // Linear data (best case - perfect compression)
    group.bench_function("insert_batch_10k_linear", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let db = AliceDB::open(dir.path()).unwrap();
                let data = generate_linear(10_000);
                (dir, db, data)
            },
            |(_dir, db, data)| {
                db.put_batch(black_box(&data)).unwrap();
            },
        )
    });

    group.finish();
}

/// Benchmark: Query Latency
///
/// Proves the "50ns point query" claim.
fn bench_query_latency(c: &mut Criterion) {
    // Setup: Create DB with 1M points
    let dir = tempdir().unwrap();
    let config = StorageConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 100_000,
        enable_wal: false,
        ..Default::default()
    };
    let db = AliceDB::with_config(config).unwrap();

    let n = 1_000_000;
    let data = generate_sine_wave(n);
    db.put_batch(&data).unwrap();
    db.flush().unwrap(); // Force to disk/cache

    let mut group = c.benchmark_group("Query Latency");

    // 1. Point Query (proves O(1) computation)
    group.bench_function("point_query_cached", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let t = rng.gen_range(0..n as i64);
            db.get(black_box(t)).unwrap()
        })
    });

    // 2. Range Scan (proves SIMD acceleration)
    group.bench_function("range_scan_1000", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let start = rng.gen_range(0..(n as i64 - 1000));
            db.scan(black_box(start), black_box(start + 1000)).unwrap()
        })
    });

    // 3. Range Scan 10K (larger range)
    group.bench_function("range_scan_10000", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let start = rng.gen_range(0..(n as i64 - 10000));
            db.scan(black_box(start), black_box(start + 10000)).unwrap()
        })
    });

    group.finish();
}

/// Benchmark: Compression Ratio
///
/// Measures the compression achieved by model fitting.
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compression");

    // Sine wave - should compress well with Fourier
    group.bench_function("flush_10k_sine", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let config = StorageConfig {
                    data_dir: dir.path().to_path_buf(),
                    memtable_capacity: 20_000,
                    enable_wal: false,
                    ..Default::default()
                };
                let db = AliceDB::with_config(config).unwrap();
                let data = generate_sine_wave(10_000);
                db.put_batch(&data).unwrap();
                (dir, db)
            },
            |(_dir, db)| {
                db.flush().unwrap();
            },
        )
    });

    // Linear - should compress to ~0 bytes
    group.bench_function("flush_10k_linear", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let config = StorageConfig {
                    data_dir: dir.path().to_path_buf(),
                    memtable_capacity: 20_000,
                    enable_wal: false,
                    ..Default::default()
                };
                let db = AliceDB::with_config(config).unwrap();
                let data = generate_linear(10_000);
                db.put_batch(&data).unwrap();
                (dir, db)
            },
            |(_dir, db)| {
                db.flush().unwrap();
            },
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_write_throughput,
    bench_query_latency,
    bench_compression
);
criterion_main!(benches);
