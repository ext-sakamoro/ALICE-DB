//! Storage backend parity oracles.
//!
//! The in-memory backend (`AliceDB::in_memory`) must store exactly what the
//! file backend stores, so one write sequence has to read back with the
//! same bits from both, and from a `to_bytes` / `from_bytes` round trip.
//! Two independent references sit beside the parity checks so that a defect
//! shared by both backends (they share the fitter) is still caught:
//!
//! - lossless config: every read equals the written `f32` bit-for-bit
//!   (oracle: the written series itself);
//! - lossy config: on the segments whose model has a pointwise acceptance
//!   threshold, every read is within that threshold of the written value,
//!   relative to the value range of the segment (oracle: `MemTable`
//!   accepts a linear model below 1 % max error and a polynomial below
//!   `FitConfig::polynomial_error_threshold` = 0.1 %). Fourier models are
//!   accepted on energy, not pointwise error (a step keeps 35 % Gibbs
//!   overshoot), so those segments are covered by parity only.
//!
//! Every check counts what it compared and fails when the count is zero.

// Test data generation casts small indices and PRNG words to floats / i64.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use alice_db::{Aggregation, AliceDB, FitConfig, StorageConfig};

/// Points per segment (memtable capacity) used by every scenario.
const CAPACITY: usize = 256;

/// Max |read - written| / (segment max - segment min) accepted for lossy
/// reconstruction of a linear segment (`MemTable::try_linear_fit` accepts
/// below 1 %) or polynomial segment (`polynomial_error_threshold` = 0.1 %).
const LOSSY_REL_BOUND: f64 = 0.01;

/// Segments of `series()` whose lossy model has a pointwise bound: the ramp
/// (linear) and the cubic (polynomial). Measured models for the series:
/// Linear 1, Polynomial 1, Fourier 4, `RawLzma` 1.
const BOUNDED_SEGMENTS: [usize; 2] = [0, 3];

/// Deterministic xorshift so the series does not depend on `rand`.
struct Xs(u64);
impl Xs {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
    }
}

/// Six segments of distinct shapes (ramp, sine, two-tone, cubic, step,
/// noise) followed by a partial seventh, so the fitter exercises linear,
/// Fourier / sine, polynomial and LZMA fallback paths.
fn series() -> Vec<(i64, f32)> {
    let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::new();
    let total = CAPACITY * 6 + CAPACITY / 3;
    for i in 0..total {
        let seg = i / CAPACITY;
        let x = (i % CAPACITY) as f32 / CAPACITY as f32;
        let v = match seg {
            0 => 10.0 + 40.0 * x + 0.01 * rng.next_f32(),
            1 => 5.0 * (std::f32::consts::TAU * 3.0 * x).sin(),
            2 => {
                2.0 * (std::f32::consts::TAU * 2.0 * x).sin()
                    + 0.5 * (std::f32::consts::TAU * 7.0 * x).cos()
            }
            3 => 1.0 + 2.0 * x - 3.0 * x * x + 4.0 * x * x * x,
            4 => {
                if x < 0.5 {
                    -1.0
                } else {
                    3.0
                }
            }
            _ => 100.0 * rng.next_f32(),
        };
        // Timestamps start away from zero and have a stride, as real time
        // series do.
        out.push((1_000 + 3 * i as i64, v));
    }
    out
}

fn config(lossless: bool) -> StorageConfig {
    StorageConfig {
        memtable_capacity: CAPACITY,
        fit_config: FitConfig {
            lossless,
            ..FitConfig::default()
        },
        ..StorageConfig::default()
    }
}

/// Apply the same writes to `db`: alternating single puts and batches, plus
/// blobs (short, compressible, incompressible, overwritten, deleted).
fn write_all(db: &AliceDB, data: &[(i64, f32)]) {
    for (n, chunk) in data.chunks(97).enumerate() {
        if n % 2 == 0 {
            db.put_batch(chunk).unwrap();
        } else {
            for &(t, v) in chunk {
                db.put(t, v).unwrap();
            }
        }
    }
    let mut rng = Xs(42);
    let noise: Vec<u8> = (0..600).map(|_| (rng.next_f32() * 255.0) as u8).collect();
    db.put_blob(b"blob/short", b"abc").unwrap();
    db.put_blob(b"blob/zip", &b"alice-db ".repeat(80)).unwrap();
    db.put_blob(b"blob/noise", &noise).unwrap();
    db.put_blob(b"blob/over", b"first").unwrap();
    db.put_blob(b"blob/over", b"second").unwrap();
    db.put_blob(b"blob/gone", b"to be deleted").unwrap();
    db.delete_blob(b"blob/gone").unwrap();
    db.flush().unwrap();
}

/// Everything the read API returns, as raw bits, so that equality is
/// bit equality (NaN-safe, -0.0 distinct from 0.0).
#[derive(Debug, PartialEq, Eq)]
struct Observed {
    points: Vec<(i64, Option<u32>)>,
    scans: Vec<Vec<(i64, u32)>>,
    aggregates: Vec<u64>,
    downsampled: Vec<(i64, u64)>,
    blobs: Vec<(Vec<u8>, Vec<u8>)>,
    blob_gets: Vec<Option<Vec<u8>>>,
}

impl Observed {
    /// Number of individual values compared when two `Observed` are equal.
    fn len(&self) -> usize {
        self.points.len()
            + self.scans.iter().map(Vec::len).sum::<usize>()
            + self.aggregates.len()
            + self.downsampled.len()
            + self.blobs.len()
            + self.blob_gets.len()
    }
}

fn observe(db: &AliceDB, data: &[(i64, f32)]) -> Observed {
    let first = data[0].0;
    let last = data[data.len() - 1].0;
    let points = data
        .iter()
        .map(|&(t, _)| (t, db.get(t).unwrap().map(f32::to_bits)))
        .collect();
    let ranges = [
        (first, last),
        (first - 10, first + 50),
        (first + 700, first + 2_000),
        (last - 100, last + 100),
    ];
    let scans = ranges
        .iter()
        .map(|&(a, b)| {
            db.scan(a, b)
                .unwrap()
                .into_iter()
                .map(|(t, v)| (t, v.to_bits()))
                .collect()
        })
        .collect();
    let aggregates = [
        Aggregation::Sum,
        Aggregation::Avg,
        Aggregation::Min,
        Aggregation::Max,
        Aggregation::Count,
    ]
    .into_iter()
    .map(|agg| db.aggregate(first, last, agg).unwrap().to_bits())
    .collect();
    let downsampled = db
        .downsample(first, last, 150, Aggregation::Avg)
        .unwrap()
        .into_iter()
        .map(|(t, v)| (t, v.to_bits()))
        .collect();
    let blobs = db.scan_blob_prefix(b"").unwrap();
    let blob_gets = [
        &b"blob/short"[..],
        b"blob/zip",
        b"blob/noise",
        b"blob/over",
        b"blob/gone",
    ]
    .iter()
    .map(|k| db.get_blob(k).unwrap())
    .collect();
    Observed {
        points,
        scans,
        aggregates,
        downsampled,
        blobs,
        blob_gets,
    }
}

/// Lossless: every point reads back the written bits.
fn assert_exact(obs: &Observed, data: &[(i64, f32)]) -> usize {
    assert_eq!(obs.points.len(), data.len());
    let mut compared = 0;
    for (&(t, read), &(t0, v)) in obs.points.iter().zip(data) {
        assert_eq!(t, t0);
        assert_eq!(read, Some(v.to_bits()), "t={t}: lossless read differs");
        compared += 1;
    }
    assert!(compared > 0, "lossless reference compared 0 points");
    compared
}

/// Lossy: on `BOUNDED_SEGMENTS`, every point is within `LOSSY_REL_BOUND`
/// of its segment's range; every point of every segment is present.
fn assert_bounded(obs: &Observed, data: &[(i64, f32)]) -> usize {
    assert_eq!(obs.points.len(), data.len());
    assert!(
        obs.points.iter().all(|p| p.1.is_some()),
        "lossy read missing"
    );
    let mut compared = 0;
    for (seg, (seg_points, seg_data)) in obs
        .points
        .chunks(CAPACITY)
        .zip(data.chunks(CAPACITY))
        .enumerate()
    {
        if !BOUNDED_SEGMENTS.contains(&seg) {
            continue;
        }
        let lo = seg_data.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let hi = seg_data
            .iter()
            .map(|p| p.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let range = f64::from(hi - lo).max(1e-6);
        for (&(t, read), &(_, v)) in seg_points.iter().zip(seg_data) {
            let read = f32::from_bits(read.unwrap_or_else(|| panic!("t={t}: lossy read missing")));
            let rel = (f64::from(read) - f64::from(v)).abs() / range;
            assert!(
                rel <= LOSSY_REL_BOUND,
                "t={t}: lossy error {rel:.4} of segment range exceeds {LOSSY_REL_BOUND} \
                 (written {v}, read {read})"
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "lossy reference compared 0 points");
    compared
}

#[cfg(feature = "fs")]
fn file_db(dir: &std::path::Path, lossless: bool) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        ..config(lossless)
    })
    .unwrap()
}

/// File backend vs memory backend vs memory round trip vs file round trip,
/// for one fit configuration.
#[cfg(feature = "fs")]
fn parity_case(lossless: bool) {
    let data = series();
    let dir = tempfile::tempdir().unwrap();
    let file = file_db(dir.path(), lossless);
    let mem = AliceDB::in_memory(config(lossless)).unwrap();
    assert!(!file.is_in_memory());
    assert!(mem.is_in_memory());
    write_all(&file, &data);
    write_all(&mem, &data);

    let from_file = observe(&file, &data);
    let from_mem = observe(&mem, &data);
    let reference = if lossless {
        assert_exact(&from_file, &data)
    } else {
        assert_bounded(&from_file, &data)
    };
    let expected_reference = if lossless {
        data.len()
    } else {
        BOUNDED_SEGMENTS.len() * CAPACITY
    };
    assert_eq!(reference, expected_reference);
    assert_eq!(from_file, from_mem, "file and memory backends differ");

    // Memory → bytes → memory.
    let mem_bytes = mem.to_bytes().unwrap();
    let restored = AliceDB::from_bytes(config(lossless), &mem_bytes).unwrap();
    assert_eq!(
        observe(&restored, &data),
        from_file,
        "memory round trip differs"
    );

    // File → bytes → memory. The two snapshots carry the same index; the
    // segment files differ only in `created_at` (wall clock), so they are
    // compared through their reads.
    let file_bytes = file.to_bytes().unwrap();
    let (file_files, file_blobs) = alice_db::snapshot::decode(&file_bytes).unwrap();
    let (mem_files, mem_blobs) = alice_db::snapshot::decode(&mem_bytes).unwrap();
    assert_eq!(
        file_files.keys().collect::<Vec<_>>(),
        mem_files.keys().collect::<Vec<_>>()
    );
    assert!(file_files.len() > 1, "snapshot holds no segments");
    assert_eq!(file_files.get("index.alice"), mem_files.get("index.alice"));
    assert_eq!(file_blobs, mem_blobs);
    let from_file_bytes = AliceDB::from_bytes(config(lossless), &file_bytes).unwrap();
    assert_eq!(observe(&from_file_bytes, &data), from_file);

    // File reopen still reads the same bits (existing behaviour).
    drop(file);
    let reopened = file_db(dir.path(), lossless);
    assert_eq!(observe(&reopened, &data).points, from_file.points);

    let compared = from_file.len();
    assert!(
        compared > data.len(),
        "parity compared only {compared} values"
    );
    eprintln!(
        "parity lossless={lossless}: {compared} values x 4 comparisons, reference {reference}"
    );
}

#[cfg(feature = "fs")]
#[test]
fn file_and_memory_backends_read_identical_bits_lossless() {
    parity_case(true);
}

#[cfg(feature = "fs")]
#[test]
fn file_and_memory_backends_read_identical_bits_lossy() {
    parity_case(false);
}

/// The memory backend never touches `data_dir`.
#[cfg(feature = "fs")]
#[test]
fn memory_backend_does_not_touch_data_dir() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().join("must-not-exist");
    let db = AliceDB::in_memory(StorageConfig {
        data_dir: data_dir.clone(),
        ..config(false)
    })
    .unwrap();
    write_all(&db, &series());
    let _ = db.to_bytes().unwrap();
    db.close().unwrap();
    assert!(
        !data_dir.exists(),
        "in-memory database created {}",
        data_dir.display()
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

/// Memory-only round trip (runs without the `fs` feature as well).
#[test]
fn memory_round_trip_is_bit_exact_lossless_and_lossy() {
    let data = series();
    for lossless in [true, false] {
        let mem = AliceDB::in_memory(config(lossless)).unwrap();
        write_all(&mem, &data);
        let before = observe(&mem, &data);
        if lossless {
            assert_exact(&before, &data);
        } else {
            assert_bounded(&before, &data);
        }
        let bytes = mem.to_bytes().unwrap();
        let restored = AliceDB::from_bytes(config(lossless), &bytes).unwrap();
        let after = observe(&restored, &data);
        assert!(after.len() > data.len());
        assert_eq!(after, before, "lossless={lossless}: round trip differs");
        // A second export of the restored database is the same buffer.
        assert_eq!(restored.to_bytes().unwrap(), bytes);
    }
}

/// Writes after a restore get fresh segment ids and keep restored data.
#[test]
fn writes_after_restore_do_not_overwrite_restored_segments() {
    let data = series();
    let mem = AliceDB::in_memory(config(true)).unwrap();
    write_all(&mem, &data);
    let restored = AliceDB::from_bytes(config(true), &mem.to_bytes().unwrap()).unwrap();
    let base = data[data.len() - 1].0 + 1_000;
    let extra: Vec<(i64, f32)> = (0..CAPACITY * 2)
        .map(|i| (base + i as i64, i as f32 * 0.25))
        .collect();
    restored.put_batch(&extra).unwrap();
    restored.flush().unwrap();
    // Re-open from bytes so reads come from stored segments, not the cache.
    let again = AliceDB::from_bytes(config(true), &restored.to_bytes().unwrap()).unwrap();
    let all: Vec<(i64, f32)> = data.iter().chain(&extra).copied().collect();
    assert_eq!(assert_exact(&observe(&again, &all), &all), all.len());
}

/// A snapshot that lost, gained or flipped a byte is rejected.
#[test]
fn damaged_snapshot_is_rejected() {
    let mem = AliceDB::in_memory(config(false)).unwrap();
    write_all(&mem, &series());
    let bytes = mem.to_bytes().unwrap();
    let mut rejected = 0;
    for pos in [0, 9, bytes.len() / 2, bytes.len() - 5, bytes.len() - 1] {
        let mut dropped = bytes.clone();
        dropped.remove(pos);
        assert!(AliceDB::from_bytes(config(false), &dropped).is_err());
        let mut flipped = bytes.clone();
        flipped[pos] ^= 0x80;
        assert!(AliceDB::from_bytes(config(false), &flipped).is_err());
        rejected += 2;
    }
    let mut extra = bytes;
    extra.push(0);
    assert!(AliceDB::from_bytes(config(false), &extra).is_err());
    assert_eq!(rejected + 1, 11);
}

/// `to_bytes` flushes first, so points still buffered in the memtable (a
/// partial segment) are part of the snapshot.
#[test]
fn to_bytes_includes_points_not_yet_flushed() {
    let data = series();
    let mem = AliceDB::in_memory(config(true)).unwrap();
    mem.put_batch(&data).unwrap();
    // The last `len % CAPACITY` points are still in the memtable.
    assert_ne!(data.len() % CAPACITY, 0);
    assert!(mem.stats().memtable_size > 0);
    let restored = AliceDB::from_bytes(config(true), &mem.to_bytes().unwrap()).unwrap();
    let compared = assert_exact(&observe_points(&restored, &data), &data);
    assert_eq!(compared, data.len());
}

/// Point reads only, for checks that do not involve blobs.
fn observe_points(db: &AliceDB, data: &[(i64, f32)]) -> Observed {
    Observed {
        points: data
            .iter()
            .map(|&(t, _)| (t, db.get(t).unwrap().map(f32::to_bits)))
            .collect(),
        scans: Vec::new(),
        aggregates: Vec::new(),
        downsampled: Vec::new(),
        blobs: Vec::new(),
        blob_gets: Vec::new(),
    }
}
