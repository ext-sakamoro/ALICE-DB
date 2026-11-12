//! A segment whose write to the store fails loses no accepted point.
//!
//! oracle (from the rules, not read from the code under test): a `put` that returns `Ok` has accepted its point;
//! an accepted point is readable from then on (`get`, `scan`), whatever happens to the segment that holds it, and it
//! is on disk after a successful `flush` (reopen reads it). A `put` that returns `Err` has accepted nothing: its
//! point is not readable and the same timestamp can be written again once the store works. While a sealed segment
//! cannot be written, `flush` fails and new writes are refused; when the store works again the next write or flush
//! writes it. The store failure is injected at every step of a segment's write (before the bytes, before the fsync,
//! before the index entry), on the writing thread and on the background flush thread; a run where no injected
//! failure fired fails.

#![cfg(all(feature = "fs", feature = "open-file-audit"))]

use alice_db::{fs_audit, AliceDB, FitConfig, StorageConfig};

const STEPS: [&str; 3] = [
    "segment:before_write",
    "segment:before_sync",
    "segment:before_index",
];
/// The memtable's capacity: the CAP-th point seals a segment.
const CAP: i64 = 16;

/// Crash points armed for any thread are process-wide: the tests of this file take turns.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    let g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    fs_audit::disarm_all_crash_points();
    g
}

fn config(dir: &std::path::Path, background: bool) -> StorageConfig {
    StorageConfig {
        data_dir: dir.to_path_buf(),
        memtable_capacity: usize::try_from(CAP).unwrap(),
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        sync_writes: true,
        enable_background_flush: background,
        ..StorageConfig::default()
    }
}

fn value(t: i64) -> f32 {
    // not a low-order polynomial, so every point is stored rather than fitted away
    let r = i16::try_from((t * 7919) % 1009).unwrap();
    f32::from(r) * 0.25 - 100.0
}

fn bits(points: &[(i64, f32)]) -> Vec<(i64, u32)> {
    points.iter().map(|&(t, v)| (t, v.to_bits())).collect()
}

fn expected(range: std::ops::Range<i64>) -> Vec<(i64, u32)> {
    range.map(|t| (t, value(t).to_bits())).collect()
}

fn injected(e: &std::io::Error) -> bool {
    e.to_string().contains("injected crash")
}

/// Every accepted point in 0..n reads back, by key and by range.
fn assert_readable(db: &AliceDB, n: i64, what: &str) {
    assert_eq!(
        bits(&db.scan(0, n - 1).unwrap()),
        expected(0..n),
        "{what}: scan"
    );
    for t in 0..n {
        assert_eq!(
            db.get(t).unwrap().map(f32::to_bits),
            Some(value(t).to_bits()),
            "{what}: get {t}"
        );
    }
}

#[test]
fn a_segment_write_that_fails_on_the_writing_thread_loses_no_accepted_point() {
    let _serial = serial();
    let mut fired = 0;
    for step in STEPS {
        let dir = tempfile::tempdir().unwrap();
        let db = AliceDB::with_config(config(dir.path(), false)).unwrap();
        // the first CAP points fill the memtable; the put that seals it writes the segment, and that write fails
        for t in 0..CAP - 1 {
            db.put(t, value(t)).unwrap();
        }
        fs_audit::arm_crash_point(step);
        let sealing = CAP - 1;
        db.put(sealing, value(sealing)).unwrap();
        fs_audit::disarm_all_crash_points();
        assert_readable(&db, CAP, step);

        // while the segment is not written, flush fails with the store's error and a new write is refused
        fs_audit::arm_crash_point(step);
        let e = db.flush().unwrap_err();
        assert!(injected(&e), "{step}: {e}");
        fired += 1;
        fs_audit::arm_crash_point(step);
        let refused = CAP;
        let e = db.put(refused, value(refused)).unwrap_err();
        assert!(injected(&e), "{step}: {e}");
        fs_audit::disarm_all_crash_points();
        assert_eq!(
            db.get(refused).unwrap(),
            None,
            "{step}: a refused point is not accepted"
        );
        assert_readable(&db, CAP, step);

        // the store works again: the same timestamp is accepted, the segment is written, and all survives a reopen
        db.put(refused, value(refused)).unwrap();
        db.flush().unwrap();
        assert_readable(&db, CAP + 1, step);
        db.close().unwrap();
        let db = AliceDB::with_config(config(dir.path(), false)).unwrap();
        assert_readable(&db, CAP + 1, step);
        db.close().unwrap();
    }
    assert_eq!(
        fired,
        STEPS.len(),
        "an injected failure fired at every step"
    );
}

#[test]
fn a_segment_write_that_fails_on_the_background_thread_loses_no_accepted_point() {
    let _serial = serial();
    let mut fired = 0;
    for step in STEPS {
        let dir = tempfile::tempdir().unwrap();
        let db = AliceDB::with_config(config(dir.path(), true)).unwrap();
        fs_audit::arm_crash_point_any_thread(step);
        for t in 0..CAP {
            db.put(t, value(t)).unwrap();
        }
        // the background write failed: the points stay readable, and flush reports the failure
        assert_readable(&db, CAP, step);
        // the background write has hit the failure (the arm is spent) before the flush is armed (a step count, not
        // a clock: at most 30 000 waits of 1 ms)
        let mut waits = 0;
        while fs_audit::is_armed_any_thread(step) {
            waits += 1;
            assert!(waits < 30_000, "{step}: the background write never ran");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        fs_audit::arm_crash_point_any_thread(step);
        let e = db.flush().unwrap_err();
        assert!(injected(&e), "{step}: {e}");
        fired += 1;
        fs_audit::disarm_all_crash_points();
        assert_readable(&db, CAP, step);
        db.flush().unwrap();
        db.close().unwrap();
        let db = AliceDB::with_config(config(dir.path(), true)).unwrap();
        assert_readable(&db, CAP, step);
        db.close().unwrap();
    }
    assert_eq!(
        fired,
        STEPS.len(),
        "an injected failure fired at every step"
    );
}

#[test]
fn a_flush_whose_segment_write_fails_keeps_the_points_and_succeeds_later() {
    let _serial = serial();
    let mut fired = 0;
    for step in STEPS {
        let dir = tempfile::tempdir().unwrap();
        let db = AliceDB::with_config(config(dir.path(), false)).unwrap();
        for t in 0..5 {
            db.put(t, value(t)).unwrap();
        }
        fs_audit::arm_crash_point(step);
        let e = db.flush().unwrap_err();
        fs_audit::disarm_all_crash_points();
        assert!(injected(&e), "{step}: {e}");
        fired += 1;
        assert_readable(&db, 5, step);
        db.put(5, value(5)).unwrap();
        db.flush().unwrap();
        assert_readable(&db, 6, step);
        db.close().unwrap();
        let db = AliceDB::with_config(config(dir.path(), false)).unwrap();
        assert_readable(&db, 6, step);
        db.close().unwrap();
    }
    assert_eq!(
        fired,
        STEPS.len(),
        "an injected failure fired at every step"
    );
}
