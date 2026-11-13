//! The time-series WAL (`wal.alice`) is framed: a versioned header, then one frame per point (length, CRC-32 of the
//! payload, payload), so a torn or damaged tail is detected and cut at the last whole frame, and a write that fails
//! midway leaves no partial frame for later writes to land behind. A WAL in the earlier unframed formats (12-byte
//! entries, or length-prefixed sealed entries with a key) is still read.
//!
//! oracle (from the format rules, not read from the code under test):
//! - an old unframed WAL is replayed whole, and the WAL written afterwards starts with the new header;
//! - a WAL whose tail is half a frame, or a whole frame with a wrong CRC, replays every whole frame before it, keeps
//!   the dropped bytes in `wal.alice.dropped-<offset>`, and later writes survive the next reopen;
//! - a write that fails partway through a frame, or before its fsync, accepts nothing (the point is not readable, and
//!   not replayed on reopen), and the next write works; every injected failure fired (none = fail).
//!
//! "Crash" is `std::mem::forget` of the time-series engine (`StorageEngine`, which holds no directory lock): nothing is
//! flushed or closed, so the reopen reads only the WAL.

#![cfg(all(feature = "fs", feature = "open-file-audit"))]

use alice_db::storage_engine::StorageEngine;
use alice_db::{fs_audit, FitConfig, StorageConfig};
use std::path::Path;

/// The header of a framed WAL: magic, version 2 (u32 LE), CRC-32 of the 12 bytes before it (u32 LE).
fn header() -> Vec<u8> {
    let mut h = b"ALDBWAL\0".to_vec();
    h.extend_from_slice(&2u32.to_le_bytes());
    let crc = crc32(&h);
    h.extend_from_slice(&crc.to_le_bytes());
    h
}

/// CRC-32 (IEEE, reflected, poly 0xEDB88320), written here from its definition.
fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// One plain frame: length (u32 LE) ‖ CRC-32 of the payload (u32 LE) ‖ payload (timestamp i64 LE ‖ value f32 LE).
fn frame(t: i64, v: f32) -> Vec<u8> {
    let mut payload = t.to_le_bytes().to_vec();
    payload.extend_from_slice(&v.to_le_bytes());
    let mut f = 12u32.to_le_bytes().to_vec();
    f.extend_from_slice(&crc32(&payload).to_le_bytes());
    f.extend_from_slice(&payload);
    f
}

fn config(dir: &Path) -> StorageConfig {
    StorageConfig::default()
        .with_data_dir(dir.to_path_buf())
        .with_memtable_capacity(100_000)
        .with_enable_wal(true)
        .with_sync_writes(true)
        .with_fit_config(FitConfig::default().with_lossless(true))
}

fn value(t: i64) -> f32 {
    let r = i16::try_from((t * 7919) % 1009).unwrap();
    f32::from(r) * 0.25 - 100.0
}

fn bits(points: &[(i64, f32)]) -> Vec<(i64, u32)> {
    points.iter().map(|&(t, v)| (t, v.to_bits())).collect()
}

fn expected(ts: impl IntoIterator<Item = i64>) -> Vec<(i64, u32)> {
    ts.into_iter().map(|t| (t, value(t).to_bits())).collect()
}

fn wal(dir: &Path) -> Vec<u8> {
    std::fs::read(dir.join("wal.alice")).unwrap()
}

#[test]
fn an_old_unframed_wal_is_replayed_and_the_new_wal_is_framed() {
    let dir = tempfile::tempdir().unwrap();
    let mut old = Vec::new();
    for t in 0..10i64 {
        old.extend_from_slice(&t.to_le_bytes());
        old.extend_from_slice(&value(t).to_le_bytes());
    }
    std::fs::write(dir.path().join("wal.alice"), &old).unwrap();
    let db = StorageEngine::new(config(dir.path())).unwrap();
    assert_eq!(bits(&db.query_range(0, 100).unwrap()), expected(0..10));
    db.put(10, value(10)).unwrap();
    std::mem::forget(db);
    // the replayed points went to a segment; the WAL now holds the header and the one new frame
    let mut want = header();
    want.extend_from_slice(&frame(10, value(10)));
    assert_eq!(wal(dir.path()), want);
    let db = StorageEngine::new(config(dir.path())).unwrap();
    assert_eq!(bits(&db.query_range(0, 100).unwrap()), expected(0..11));
}

#[cfg(feature = "crypto")]
#[test]
fn an_old_sealed_wal_is_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let key = alice_crypto::Key::from_bytes([0x5a; 32]);
    let mut old = Vec::new();
    for t in 0..6i64 {
        let mut plain = t.to_le_bytes().to_vec();
        plain.extend_from_slice(&value(t).to_le_bytes());
        let sealed = alice_crypto::seal(&key, &plain).unwrap();
        old.extend_from_slice(&u32::try_from(sealed.len()).unwrap().to_le_bytes());
        old.extend_from_slice(&sealed);
    }
    std::fs::write(dir.path().join("wal.alice"), &old).unwrap();
    let cfg = config(dir.path())
        .with_encryption_key(Some(key))
        .with_use_mmap(false);
    let db = StorageEngine::new(cfg.clone()).unwrap();
    assert_eq!(bits(&db.query_range(0, 100).unwrap()), expected(0..6));
    db.put(6, value(6)).unwrap();
    std::mem::forget(db);
    assert!(
        wal(dir.path()).starts_with(&header()),
        "the new WAL is framed"
    );
    let db = StorageEngine::new(cfg).unwrap();
    assert_eq!(bits(&db.query_range(0, 100).unwrap()), expected(0..7));
}

#[test]
fn a_torn_or_damaged_tail_is_cut_at_the_last_whole_frame_and_later_writes_survive() {
    for damage in ["half a frame", "a frame with a wrong crc"] {
        let dir = tempfile::tempdir().unwrap();
        let db = StorageEngine::new(config(dir.path())).unwrap();
        for t in 0..5 {
            db.put(t, value(t)).unwrap();
        }
        std::mem::forget(db);
        let mut w = wal(dir.path());
        let whole = w.len();
        let mut bad = frame(5, value(5));
        if damage == "half a frame" {
            bad.truncate(9);
        } else {
            let last = bad.len() - 1;
            bad[last] ^= 0x01;
        }
        w.extend_from_slice(&bad);
        std::fs::write(dir.path().join("wal.alice"), &w).unwrap();

        let db = StorageEngine::new(config(dir.path())).unwrap();
        assert_eq!(
            bits(&db.query_range(0, 100).unwrap()),
            expected(0..5),
            "{damage}"
        );
        let kept = std::fs::read(dir.path().join(format!("wal.alice.dropped-{whole}"))).unwrap();
        assert_eq!(kept, bad, "{damage}: the dropped bytes are kept");
        db.put(5, value(5)).unwrap();
        db.put(6, value(6)).unwrap();
        std::mem::forget(db);
        let db = StorageEngine::new(config(dir.path())).unwrap();
        assert_eq!(
            bits(&db.query_range(0, 100).unwrap()),
            expected(0..7),
            "{damage}: later writes survive"
        );
    }
}

#[test]
fn a_write_that_fails_partway_accepts_nothing_and_leaves_no_partial_frame() {
    let mut fired = 0;
    for step in ["ts_wal:mid_frame", "ts_wal:before_sync"] {
        let dir = tempfile::tempdir().unwrap();
        let db = StorageEngine::new(config(dir.path())).unwrap();
        db.put(0, value(0)).unwrap();
        fs_audit::arm_crash_point(step);
        let e = db.put(1, value(1)).unwrap_err();
        fs_audit::disarm_all_crash_points();
        assert!(e.to_string().contains("injected crash"), "{step}: {e}");
        fired += 1;
        assert_eq!(db.query_point(1).unwrap(), None, "{step}: not accepted");
        let mut want = header();
        want.extend_from_slice(&frame(0, value(0)));
        assert_eq!(
            wal(dir.path()),
            want,
            "{step}: nothing of the failed frame stays"
        );
        // the same timestamp is accepted afterwards, and a reopen reads exactly the accepted points
        db.put(1, value(1)).unwrap();
        db.put(2, value(2)).unwrap();
        std::mem::forget(db);
        let db = StorageEngine::new(config(dir.path())).unwrap();
        assert_eq!(
            bits(&db.query_range(0, 100).unwrap()),
            expected(0..3),
            "{step}"
        );
    }
    assert_eq!(fired, 2, "an injected failure fired at every step");
}

#[test]
fn a_crash_at_each_step_of_a_replay_loses_no_point() {
    // the replay writes segments, fsyncs them, writes the index whole (tmp, fsync, rename, directory fsync), and only
    // then empties the WAL: a crash at any step leaves the WAL whole, so the next open replays it again
    let mut fired = 0;
    for step in [
        "replay:before_segment_sync",
        "index:mid_write",
        "index:before_sync",
        "index:before_rename",
        "index:before_dir_sync",
        "replay:before_wal_truncate",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = StorageEngine::new(config(dir.path())).unwrap();
        for t in 0..8 {
            db.put(t, value(t)).unwrap();
        }
        std::mem::forget(db);
        let before = wal(dir.path());
        fs_audit::arm_crash_point(step);
        let e = StorageEngine::new(config(dir.path())).err();
        fs_audit::disarm_all_crash_points();
        let e = e.unwrap_or_else(|| panic!("{step}: the open did not stop at the crash"));
        assert!(e.to_string().contains("injected crash"), "{step}: {e}");
        fired += 1;
        assert_eq!(
            wal(dir.path()),
            before,
            "{step}: the WAL is untouched until the end"
        );
        let db = StorageEngine::new(config(dir.path())).unwrap();
        assert_eq!(
            bits(&db.query_range(0, 100).unwrap()),
            expected(0..8),
            "{step}"
        );
        db.put(8, value(8)).unwrap();
        std::mem::forget(db);
        let db = StorageEngine::new(config(dir.path())).unwrap();
        assert_eq!(
            bits(&db.query_range(0, 100).unwrap()),
            expected(0..9),
            "{step}: and after one more crash"
        );
    }
    assert_eq!(fired, 6, "an injected failure fired at every step");
}

#[cfg(feature = "crypto")]
#[test]
fn an_open_that_cannot_replay_the_wal_refuses_and_leaves_the_wal_whole() {
    let dir = tempfile::tempdir().unwrap();
    let keyed = |b: u8| {
        config(dir.path())
            .with_encryption_key(Some(alice_crypto::Key::from_bytes([b; 32])))
            .with_use_mmap(false)
    };
    let db = StorageEngine::new(keyed(0x5a)).unwrap();
    for t in 0..4 {
        db.put(t, value(t)).unwrap();
    }
    std::mem::forget(db);
    let before = wal(dir.path());
    // another key: the frames' CRCs match but they do not open, which is not a torn write
    let e = StorageEngine::new(keyed(0x11)).err().expect("refused");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
    assert_eq!(
        wal(dir.path()),
        before,
        "nothing flushed, saved or deleted by the failed open"
    );
    let db = StorageEngine::new(keyed(0x5a)).unwrap();
    assert_eq!(bits(&db.query_range(0, 100).unwrap()), expected(0..4));
}
