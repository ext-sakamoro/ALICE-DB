//! Range reads over segments whose keys are far from zero or span more than
//! `i64::MAX`
//!
//! A segment stores a model plus `start_time` / `end_time` / `point_count`
//! and reads its samples back on an even grid between the two ends (it keeps
//! no per-point timestamps). Two defects lived in that grid arithmetic:
//!
//! - **termination**: `query_range` stepped an `f64` timestamp by the grid
//!   step. Above 2^53 a step smaller than half an ulp leaves the timestamp
//!   unchanged, so the loop never ended and grew its result until memory ran
//!   out (keys near 2^62 have an ulp of 1024, so a step of 1 never moves)
//! - **overflow**: `end_time - start_time` and `timestamp - start_time` were
//!   `i64` subtractions, a panic in debug builds and a silent wrap in release
//!   once a segment spans more than `i64::MAX`
//!
//! The grid is now walked with integer sample indices and each timestamp is
//! computed exactly from its index (`start + ⌊k·range/(n−1)⌋` in 128-bit
//! integers), so a range read returns at most `point_count` points.
//!
//! What a range read of one segment returns (pinned below): the grid points
//! inside the query, starting at the query's first key. When the stored
//! keys are evenly spaced and the query starts on one of them, those are
//! exactly the stored keys, and a lossless segment returns the stored values
//! bit for bit. A single segment does not reproduce keys with gaps: it does
//! not know them, so the grid points between the ends are returned instead
//! (pinned in `uneven_keys_return_the_grid_not_the_stored_keys`). `AliceDB`
//! in lossless mode never builds such a segment: it cuts a new one where
//! the spacing changes (`tests/lossless_irregular_keys.rs`). Keeping uneven
//! keys inside one segment needs per-point timestamps in the segment format
//! (`tests/analytic_oracle.rs`, ignored
//! `lossless_mode_is_exact_for_irregular_timestamps_too`).
//!
//! Each scan runs on a separate thread under a deadline; a scan that does not
//! return in time ends the test process with a failure, so a regression to a
//! non-terminating loop turns CI red instead of hanging it.

use alice_db::segment::{compress_residual_xor, SegmentView};
use alice_db::{AliceDB, DataSegment, FitConfig, MemTable, StorageConfig, SEMANTICS_ID};
use std::sync::mpsc;
use std::time::Duration;

/// A correct read of at most a few hundred points takes microseconds; the
/// broken loop allocates without bound, so the deadline also bounds memory.
const DEADLINE: Duration = Duration::from_secs(3);

/// Runs `f` on its own thread and returns its result, or ends the process
/// with a failure when it does not return before [`DEADLINE`].
fn within_deadline<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(DEADLINE) {
        Ok(v) => {
            handle.join().expect("spawned thread");
            v
        }
        // A panic on the spawned thread drops the sender without a value
        Err(mpsc::RecvTimeoutError::Disconnected) => match handle.join() {
            Err(e) => std::panic::resume_unwind(e),
            Ok(()) => panic!("{what}: no result"),
        },
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "{what}: did not return within {DEADLINE:?}; a range read is not terminating"
            );
            std::process::exit(101);
        }
    }
}

fn lossless() -> FitConfig {
    FitConfig {
        lossless: true,
        ..FitConfig::default()
    }
}

/// Values that no model reproduces, so the lossless residual carries them
#[allow(clippy::cast_precision_loss, reason = "small integers")]
fn value(i: usize) -> f32 {
    ((i * i * 7 + i * 13) % 101) as f32 * 0.37 - 11.0
}

/// One lossless segment holding all of `keys`, whatever their spacing
///
/// Built the way a flush builds a lossless segment (fit, then the XOR
/// residual against the point law), but without the memtable's cut at
/// spacing changes, so the grid arithmetic of a single segment is what is
/// tested here. `AliceDB` in lossless mode cuts segments where the spacing
/// changes (`tests/lossless_irregular_keys.rs`).
fn segment(keys: &[i64]) -> DataSegment {
    let data: Vec<(i64, f32)> = keys
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, value(i)))
        .collect();
    let memtable = MemTable::with_config(data.len() + 1, FitConfig::default());
    assert!(memtable.put_batch(&data).is_empty());
    let seg = memtable.force_flush().expect("one segment");
    let mut residual = Vec::with_capacity(data.len() * 4);
    for &(t, v) in &data {
        let model = seg.query_point(t).expect("inside the segment");
        residual.extend_from_slice(&(v.to_bits() ^ model.to_bits()).to_le_bytes());
    }
    seg.with_residual(compress_residual_xor(&residual, &SEMANTICS_ID))
}

/// Range read on the in-memory segment and on its zero-copy view
fn both_paths(keys: &[i64], start: i64, end: i64) -> [Vec<(i64, f32)>; 2] {
    let keys = keys.to_vec();
    within_deadline("query_range", move || {
        let seg = segment(&keys);
        let view = SegmentView::from_vec(seg.to_rkyv_bytes().unwrap()).unwrap();
        [seg.query_range(start, end), view.query_range(start, end)]
    })
}

/// Asserts `got` holds exactly the stored `(key, value)` pairs, bit for bit
fn assert_stored(got: &[(i64, f32)], keys: &[i64], what: &str) {
    assert_eq!(got.len(), keys.len(), "{what}: point count");
    for (i, (&(t, v), &k)) in got.iter().zip(keys).enumerate() {
        assert_eq!(t, k, "{what}: key {i}");
        assert_eq!(v.to_bits(), value(i).to_bits(), "{what}: value at {k}");
    }
}

/// 100 consecutive keys starting at 2^62: the grid step is 1, below half an
/// ulp of 2^62 in `f64`
fn dense_near_2_62() -> Vec<i64> {
    (0..100).map(|i| (1i64 << 62) + i).collect()
}

/// 9 evenly spaced keys from −2^62 to 2^62: the span is 2^63 > `i64::MAX`
fn spanning_both_signs() -> Vec<i64> {
    (0..9i128)
        .map(|k| i64::try_from(-(1i128 << 62) + k * (1i128 << 60)).expect("within i64"))
        .collect()
}

#[test]
fn dense_keys_near_2_62_terminate_and_read_back_exact() {
    let keys = dense_near_2_62();
    for (path, got) in ["segment", "view"]
        .iter()
        .zip(both_paths(&keys, i64::MIN, i64::MAX))
    {
        assert_stored(&got, &keys, path);
    }
    // a query starting in the middle returns the rest of the grid
    let [seg, view] = both_paths(&keys, keys[40], keys[59]);
    assert_eq!(seg.len(), 20);
    assert_eq!(seg, view);
    assert_eq!(seg[0].0, keys[40]);
    assert_eq!(seg[19].0, keys[59]);
    for (i, &(_, v)) in seg.iter().enumerate() {
        assert_eq!(v.to_bits(), value(i + 40).to_bits());
    }
}

#[test]
fn keys_spanning_both_signs_do_not_overflow() {
    let keys = spanning_both_signs();
    for (path, got) in ["segment", "view"]
        .iter()
        .zip(both_paths(&keys, i64::MIN, i64::MAX))
    {
        assert_stored(&got, &keys, path);
    }
    let seg = within_deadline("query_point", {
        let keys = keys.clone();
        move || segment(&keys)
    });
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(
            seg.query_point(k).map(f32::to_bits),
            Some(value(i).to_bits())
        );
    }
}

/// `i64::MIN`, −1, `i64::MAX`: the span is 2^64 − 1 and the grid step is not
/// an integer ((2^64 − 1) / 2); the middle grid point is
/// `MIN + ⌊(2^64 − 1)/2⌋ = −1`
#[test]
fn full_i64_span_does_not_overflow() {
    let keys = [i64::MIN, -1, i64::MAX];
    for (path, got) in ["segment", "view"]
        .iter()
        .zip(both_paths(&keys, i64::MIN, i64::MAX))
    {
        assert_stored(&got, &keys, path);
    }
    // a query that starts between grid points starts the grid at the query
    // start; the step is still (2^64 − 1)/2, so only one point fits
    let [seg, view] = both_paths(&keys, 0, i64::MAX);
    assert_eq!(seg, view);
    assert_eq!(seg.iter().map(|p| p.0).collect::<Vec<_>>(), [0]);
}

/// Keys with gaps (base + 0, 1, 2, 10): the segment keeps 4 points spanning
/// 10, so it reads back the grid base + ⌊k·10/3⌋ = 0, 3, 6, 10 — two of
/// which were never written. The ends and the count are right; the inner
/// keys are not. Pinned so the behaviour cannot change unnoticed.
#[test]
fn uneven_keys_return_the_grid_not_the_stored_keys() {
    let base = 1i64 << 62;
    let keys = [base, base + 1, base + 2, base + 10];
    let [seg, view] = both_paths(&keys, i64::MIN, i64::MAX);
    assert_eq!(seg, view);
    assert_eq!(
        seg.iter().map(|p| p.0 - base).collect::<Vec<_>>(),
        [0, 3, 6, 10]
    );
    // the ends are stored keys and read back exactly
    assert_eq!(seg[0].1.to_bits(), value(0).to_bits());
    assert_eq!(seg[3].1.to_bits(), value(3).to_bits());
}

/// The same keys through `AliceDB` (in-memory backend, which reads segments
/// through the zero-copy view): `scan` and `get` after a flush
#[test]
fn alice_db_scan_and_get_on_extreme_keys() {
    for keys in [
        dense_near_2_62(),
        spanning_both_signs(),
        vec![i64::MIN, -1, i64::MAX],
    ] {
        let got = within_deadline("AliceDB::scan", {
            let keys = keys.clone();
            move || {
                let db = AliceDB::in_memory(StorageConfig {
                    fit_config: lossless(),
                    ..StorageConfig::default()
                })
                .unwrap();
                let data: Vec<(i64, f32)> = keys
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| (k, value(i)))
                    .collect();
                db.put_batch(&data).unwrap();
                db.flush().unwrap();
                let points: Vec<Option<f32>> = keys.iter().map(|&k| db.get(k).unwrap()).collect();
                (db.scan(i64::MIN, i64::MAX).unwrap(), points)
            }
        });
        assert_stored(&got.0, &keys, "scan");
        for (i, v) in got.1.iter().enumerate() {
            assert_eq!(
                v.map(f32::to_bits),
                Some(value(i).to_bits()),
                "get {}",
                keys[i]
            );
        }
    }
}
