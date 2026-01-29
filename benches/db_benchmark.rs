//! Benchmarks for ALICE-DB
//!
//! Run with: cargo bench

use alice_db::{AliceDB, Aggregation, StorageConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tempfile::tempdir;

fn benchmark_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let config = StorageConfig {
                        data_dir: dir.path().to_path_buf(),
                        memtable_capacity: size * 2, // Avoid flush during benchmark
                        enable_wal: false,
                        ..Default::default()
                    };
                    (dir, AliceDB::with_config(config).unwrap())
                },
                |(_dir, db)| {
                    for i in 0..size {
                        db.put(i as i64, i as f32).unwrap();
                    }
                    black_box(&db);
                },
            );
        });
    }

    group.finish();
}

fn benchmark_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("flush");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let config = StorageConfig {
                        data_dir: dir.path().to_path_buf(),
                        memtable_capacity: size * 2,
                        enable_wal: false,
                        ..Default::default()
                    };
                    let db = AliceDB::with_config(config).unwrap();

                    // Pre-insert data
                    for i in 0..size {
                        db.put(i as i64, (i as f32).sin()).unwrap();
                    }

                    (dir, db)
                },
                |(_dir, db)| {
                    db.flush().unwrap();
                    black_box(&db);
                },
            );
        });
    }

    group.finish();
}

fn benchmark_query_point(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let config = StorageConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 10000,
        enable_wal: false,
        ..Default::default()
    };
    let db = AliceDB::with_config(config).unwrap();

    // Insert and flush linear data
    for i in 0..10000 {
        db.put(i, i as f32 * 2.0).unwrap();
    }
    db.flush().unwrap();

    c.bench_function("query_point", |b| {
        b.iter(|| {
            // Query from model (should be O(1) computation)
            let value = db.get(black_box(5000)).unwrap();
            black_box(value);
        });
    });
}

fn benchmark_query_range(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let config = StorageConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 10000,
        enable_wal: false,
        ..Default::default()
    };
    let db = AliceDB::with_config(config).unwrap();

    // Insert and flush
    for i in 0..10000 {
        db.put(i, i as f32).unwrap();
    }
    db.flush().unwrap();

    let mut group = c.benchmark_group("query_range");

    for range_size in [100, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(range_size),
            range_size,
            |b, &range_size| {
                b.iter(|| {
                    let results = db.scan(black_box(0), black_box(range_size as i64)).unwrap();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_aggregation(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let config = StorageConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 10000,
        enable_wal: false,
        ..Default::default()
    };
    let db = AliceDB::with_config(config).unwrap();

    // Insert and flush
    for i in 0..10000 {
        db.put(i, i as f32).unwrap();
    }
    db.flush().unwrap();

    let mut group = c.benchmark_group("aggregation");

    for agg in [Aggregation::Sum, Aggregation::Avg, Aggregation::Min, Aggregation::Max].iter() {
        let agg_name = match agg {
            Aggregation::Sum => "sum",
            Aggregation::Avg => "avg",
            Aggregation::Min => "min",
            Aggregation::Max => "max",
            _ => "other",
        };

        group.bench_with_input(BenchmarkId::from_parameter(agg_name), agg, |b, &agg| {
            b.iter(|| {
                let result = db.aggregate(black_box(0), black_box(9999), agg).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn benchmark_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    // Linear data
    group.bench_function("linear_10k", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let config = StorageConfig {
                    data_dir: dir.path().to_path_buf(),
                    memtable_capacity: 10000,
                    enable_wal: false,
                    ..Default::default()
                };
                (dir, AliceDB::with_config(config).unwrap())
            },
            |(_dir, db)| {
                for i in 0..10000 {
                    db.put(i, i as f32 * 0.5 + 10.0).unwrap();
                }
                db.flush().unwrap();
                let stats = db.stats();
                black_box(stats.average_compression_ratio);
            },
        );
    });

    // Sine wave
    group.bench_function("sine_10k", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let config = StorageConfig {
                    data_dir: dir.path().to_path_buf(),
                    memtable_capacity: 10000,
                    enable_wal: false,
                    ..Default::default()
                };
                (dir, AliceDB::with_config(config).unwrap())
            },
            |(_dir, db)| {
                for i in 0..10000 {
                    let value = (i as f32 * 0.01 * std::f32::consts::PI * 2.0).sin();
                    db.put(i, value).unwrap();
                }
                db.flush().unwrap();
                let stats = db.stats();
                black_box(stats.average_compression_ratio);
            },
        );
    });

    // Quadratic
    group.bench_function("quadratic_10k", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let config = StorageConfig {
                    data_dir: dir.path().to_path_buf(),
                    memtable_capacity: 10000,
                    enable_wal: false,
                    ..Default::default()
                };
                (dir, AliceDB::with_config(config).unwrap())
            },
            |(_dir, db)| {
                for i in 0..10000 {
                    let x = i as f32 / 10000.0;
                    let value = x * x + 2.0 * x + 1.0;
                    db.put(i, value).unwrap();
                }
                db.flush().unwrap();
                let stats = db.stats();
                black_box(stats.average_compression_ratio);
            },
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_insert,
    benchmark_flush,
    benchmark_query_point,
    benchmark_query_range,
    benchmark_aggregation,
    benchmark_compression_ratio,
);

criterion_main!(benches);
