//! The analytics bridge reads back every value it wrote, bit for bit
//!
//! `flush_metrics_to_db` writes six values per metric (counter, gauge,
//! cardinality, p50, p90, p99). Until 0.3.0 it wrote them into the
//! model-based time-series store, which fits one model per segment and
//! assumes evenly spaced keys; the metric keys leave gaps of up to 2^44
//! between metrics, so reading them back returned values of other keys
//! (840 keys written, 0 read back exactly). The values now go through the
//! blob key-value store and these tests require every one of them back with
//! the same bits, on the file backend (also after a reopen) and in memory.
//!
//! Each scenario asserts the number of compared values first, so a scenario
//! that stops writing anything cannot pass by comparing nothing.
#![cfg(feature = "analytics")]

use alice_analytics::pipeline::{MetricEvent, MetricPipeline};
use alice_db::analytics_bridge::{
    flush_metrics_to_db, metric_key, metric_name_hash, read_metric, scan_metric, AnalyticsSink,
    MetricPoint, METRICS_SERIES, VARIANTS_PER_METRIC,
};
use alice_db::blob_wal::SyncPolicy;
use alice_db::law_store::LAW_KEY_PREFIX;
use alice_db::series::{series_key, SERIES_KEY_PREFIX};
use alice_db::{AliceDB, StorageConfig};
use std::collections::BTreeSet;

const SLOTS: usize = 256;
const QUEUE: usize = 1024;
const METRICS: usize = 140;
type Pipeline = MetricPipeline<SLOTS, QUEUE>;

/// Runs `f` on a thread with a large stack: a pipeline of 256 slots is
/// built on the stack and does not fit the default test thread stack in
/// debug builds.
fn on_big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(f)
        .expect("spawn test thread")
        .join()
        .expect("test thread");
}

/// 140 realistic metric names whose hashes neither share a pipeline slot
/// (`hash % SLOTS`, a shared slot merges two metrics) nor the 20 bits the
/// key keeps (`hash & 0xFFFFF`).
fn metric_hashes() -> Vec<u64> {
    let services = [
        "api", "auth", "billing", "search", "ingest", "render", "queue",
    ];
    let measures = [
        "request_latency_ms",
        "response_bytes",
        "db_query_ms",
        "cache_hit_ratio",
        "queue_depth",
        "cpu_percent",
        "gc_pause_ms",
        "open_connections",
    ];
    let mut slots = BTreeSet::new();
    let mut keys = BTreeSet::new();
    let mut out = Vec::new();
    'outer: for region in ["ap-northeast-1", "us-east-1", "eu-west-1", "sa-east-1"] {
        for service in services {
            for measure in measures {
                let h = metric_name_hash(&format!("{region}.{service}.{measure}"));
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "the pipeline does the same"
                )]
                let slot = (h as usize) % SLOTS;
                if slots.contains(&slot) || keys.contains(&(h & 0xF_FFFF)) {
                    continue;
                }
                slots.insert(slot);
                keys.insert(h & 0xF_FFFF);
                out.push(h);
                if out.len() == METRICS {
                    break 'outer;
                }
            }
        }
    }
    assert_eq!(out.len(), METRICS, "not enough collision-free names");
    out
}

/// Feeds every metric a cycle-dependent mix of counter, gauge, histogram and
/// unique events, so all six variants are written and differ per cycle.
fn fill(pipeline: &mut Pipeline, hashes: &[u64], cycle: u64) {
    for (m, &h) in hashes.iter().enumerate() {
        let m = m as u64;
        for j in 0..12u64 {
            #[allow(clippy::cast_precision_loss, reason = "small integers")]
            let x = (m * 37 + j * 11 + cycle * 101) as f64;
            assert!(pipeline.submit(MetricEvent::counter(h, 0.5 + x / 7.0)));
            assert!(pipeline.submit(MetricEvent::gauge(h, x * 1.25 - 40.0)));
            assert!(pipeline.submit(MetricEvent::histogram(h, 1.0 + x * 3.7)));
            assert!(pipeline.submit(MetricEvent::unique(h, m * 1000 + j + cycle * 7)));
        }
        pipeline.flush();
    }
}

/// The six values the bridge writes for one slot, recomputed from the slot
fn expected(pipeline: &Pipeline, h: u64) -> [f32; 6] {
    let s = pipeline.get_slot(h).expect("metric has a slot");
    assert!(s.ddsketch.count() > 0, "every metric has histogram data");
    #[allow(clippy::cast_possible_truncation, reason = "the bridge stores f32")]
    [
        s.counter as f32,
        s.gauge as f32,
        s.hll.cardinality() as f32,
        s.ddsketch.quantile(0.50) as f32,
        s.ddsketch.quantile(0.90) as f32,
        s.ddsketch.quantile(0.99) as f32,
    ]
}

/// Reads every value of one cycle back and returns how many matched; panics
/// on the first mismatch with the key and both bit patterns.
fn compare(db: &AliceDB, want: &[(u64, [f32; 6])], ts: i64) -> usize {
    let mut matched = 0;
    for &(h, values) in want {
        for (variant, &v) in (0u8..).zip(values.iter()) {
            let got = read_metric(db, h, ts, variant).expect("read_metric");
            assert_eq!(
                got.map(f32::to_bits),
                Some(v.to_bits()),
                "key {:#018x} (hash {h:#x}, ts {ts}, variant {variant}): got {got:?}, wrote {v}",
                metric_key(h, ts, variant)
            );
            matched += 1;
        }
    }
    matched
}

/// File backend whose blob WAL syncs only on request: these scenarios test
/// what is read back, not durability, and a sync per record makes them slow
fn open_manual(path: &std::path::Path) -> AliceDB {
    AliceDB::open_with_blob_sync_policy(path, SyncPolicy::Manual).unwrap()
}

fn snapshot(pipeline: &Pipeline, hashes: &[u64]) -> Vec<(u64, [f32; 6])> {
    hashes.iter().map(|&h| (h, expected(pipeline, h))).collect()
}

/// 140 metrics × 6 variants on the file backend: 840 values, all exact
#[test]
fn round_trip_840_values_bit_exact() {
    on_big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let db = open_manual(dir.path());
        let hashes = metric_hashes();
        let mut pipeline = Pipeline::new(0.01);
        fill(&mut pipeline, &hashes, 0);
        let want = snapshot(&pipeline, &hashes);

        let written = flush_metrics_to_db(&pipeline, &db, 1000).unwrap();
        assert_eq!(written, METRICS * usize::from(VARIANTS_PER_METRIC));
        let matched = compare(&db, &want, 1000);
        assert_eq!(matched, 840);
    });
}

/// Two flush cycles (ts 1000 and 2000) keep both cycles' values, and a
/// metric scan returns exactly its 12 points in `(timestamp, variant)` order
#[test]
fn two_flush_cycles_keep_both() {
    on_big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let mut sink = AnalyticsSink::<SLOTS, QUEUE>::new(open_manual(dir.path()), 0.01);
        let hashes = metric_hashes();

        fill(&mut sink.pipeline, &hashes, 1);
        let first = snapshot(&sink.pipeline, &hashes);
        assert_eq!(sink.persist_and_reset(1000).unwrap(), 840);

        fill(&mut sink.pipeline, &hashes, 2);
        let second = snapshot(&sink.pipeline, &hashes);
        assert_eq!(sink.persist(2000).unwrap(), 840);
        assert_ne!(
            first[0].1[0].to_bits(),
            second[0].1[0].to_bits(),
            "the two cycles must write different values"
        );

        assert_eq!(compare(&sink.db, &first, 1000), 840);
        assert_eq!(compare(&sink.db, &second, 2000), 840);

        let mut scanned = 0;
        for (&(h, a), &(_, b)) in first.iter().zip(&second) {
            let points = sink.scan(h).unwrap();
            let want: Vec<MetricPoint> = (0u8..6)
                .map(|v| MetricPoint {
                    timestamp: 1000,
                    variant: v,
                    value: a[usize::from(v)],
                })
                .chain((0u8..6).map(|v| MetricPoint {
                    timestamp: 2000,
                    variant: v,
                    value: b[usize::from(v)],
                }))
                .collect();
            assert_eq!(points.len(), 12, "hash {h:#x}");
            for (got, want) in points.iter().zip(&want) {
                assert_eq!(
                    (got.timestamp, got.variant, got.value.to_bits()),
                    (want.timestamp, want.variant, want.value.to_bits()),
                    "hash {h:#x}"
                );
            }
            scanned += points.len();
        }
        assert_eq!(scanned, 140 * 12);
        assert_eq!(sink.flush_count(), 2);
    });
}

/// Persisted values survive closing and reopening the file backend
#[test]
fn reopen_file_backend_reads_back_exact() {
    on_big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let hashes = metric_hashes();
        let want = {
            let mut sink =
                AnalyticsSink::<SLOTS, QUEUE>::open(dir.path().to_str().unwrap(), 0.01).unwrap();
            fill(&mut sink.pipeline, &hashes, 3);
            let want = snapshot(&sink.pipeline, &hashes);
            assert_eq!(sink.persist(1000).unwrap(), 840);
            sink.db.close().unwrap();
            want
        };
        let db = AliceDB::open(dir.path()).unwrap();
        assert_eq!(compare(&db, &want, 1000), 840);
        let total: usize = hashes
            .iter()
            .map(|&h| scan_metric(&db, h).unwrap().len())
            .sum();
        assert_eq!(total, 840);
    });
}

/// The in-memory backend reads back the same bits
#[test]
fn in_memory_backend_reads_back_exact() {
    on_big_stack(|| {
        let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
        let hashes = metric_hashes();
        let mut pipeline = Pipeline::new(0.01);
        fill(&mut pipeline, &hashes, 4);
        let want = snapshot(&pipeline, &hashes);
        assert_eq!(flush_metrics_to_db(&pipeline, &db, 1000).unwrap(), 840);
        assert_eq!(compare(&db, &want, 1000), 840);

        // and after a snapshot round trip
        let restored =
            AliceDB::from_bytes(StorageConfig::default(), &db.to_bytes().unwrap()).unwrap();
        assert_eq!(compare(&restored, &want, 1000), 840);
    });
}

/// The stored record format: series key of `metric_key`, value = `f32`
/// bits little-endian. A reader built from these bytes (another language, a
/// migration tool) depends on exactly this layout.
#[test]
fn record_layout_is_pinned() {
    assert_eq!(METRICS_SERIES, "alice-analytics/metrics");
    let k = metric_key(0xABC_DE123, 0x12_3456_789A, 5);
    // (0xDE123 << 44) | (0x12_3456_789A << 4) | 5, sign bit flipped
    assert_eq!(u64::from_ne_bytes(k.to_ne_bytes()), 0xDE12_3123_4567_89A5);
    let mut want = SERIES_KEY_PREFIX.to_vec();
    want.extend_from_slice(METRICS_SERIES.as_bytes());
    want.push(0);
    want.extend_from_slice(&[0x5E, 0x12, 0x31, 0x23, 0x45, 0x67, 0x89, 0xA5]);
    assert_eq!(series_key(METRICS_SERIES, k).unwrap(), want);

    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    let h = 0xABC_DE123;
    let ts = 0x12_3456_789A;
    db.series(METRICS_SERIES)
        .unwrap()
        .put_f32(k, -0.75)
        .unwrap();
    assert_eq!(
        db.get_blob(&want).unwrap().unwrap(),
        (-0.75f32).to_bits().to_le_bytes()
    );
    assert_eq!(read_metric(&db, h, ts, 5).unwrap(), Some(-0.75));
}

/// A metric scan never returns points of a metric whose hash differs only
/// in its low 4 bits, nor other series
#[test]
fn scan_metric_returns_only_that_metric() {
    let db = AliceDB::in_memory(StorageConfig::default()).unwrap();
    // same top 16 bits of the 20-bit hash, different low 4 bits
    let a = 0x0001_2340_u64;
    let b = 0x0001_2347_u64;
    for (h, v) in [(a, 1.5f32), (b, -2.5f32)] {
        for ts in [5i64, 1, 3] {
            db.series(METRICS_SERIES)
                .unwrap()
                .put_f32(metric_key(h, ts, 2), v)
                .unwrap();
        }
    }
    db.put_blob(&[LAW_KEY_PREFIX, b"x".as_slice()].concat(), b"law")
        .unwrap();
    db.series("other")
        .unwrap()
        .put_f32(metric_key(a, 3, 2), 9.0)
        .unwrap();
    let points = scan_metric(&db, a).unwrap();
    assert_eq!(
        points,
        [1, 3, 5].map(|timestamp| MetricPoint {
            timestamp,
            variant: 2,
            value: 1.5
        })
    );
    assert_eq!(read_metric(&db, b, 3, 2).unwrap(), Some(-2.5));
    assert_eq!(read_metric(&db, b, 4, 2).unwrap(), None);
}
