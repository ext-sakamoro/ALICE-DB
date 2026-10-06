//! Compaction under the Windows file rules, and crash safety at every
//! step of compaction.
//!
//! Windows refuses to rename over, delete, or (through an append-only
//! handle) truncate a file that is open or memory-mapped. The crate's
//! own test builds enable the `open-file-audit` feature (self
//! dev-dependency in `Cargo.toml`), which turns every such operation
//! into a panic on all platforms, so each test in this file — and every
//! other `blob_*` suite — doubles as a check of those rules.
//!
//! The crash tests abort `compact_all_sstables` at each of its steps
//! through the audit's crash points, drop the store as a crashed
//! process would, reopen it, and require the full key set to read back
//! exactly. Expected values come from the sequence of puts and deletes
//! each test performs, not from the store.
#![cfg(feature = "fs")]

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use alice_db::blob::BlobStorageConfig;
use alice_db::blob_sstable::{enumerate_sstables, FlushMode};
use alice_db::blob_wal::SyncPolicy;
use alice_db::fs_audit;
use alice_db::AliceDB;
use tempfile::TempDir;

/// Every crash point inside `compact_all_sstables`, in execution order.
const CRASH_POINTS: [&str; 5] = [
    "compact:after_write",
    "compact:after_swap",
    "compact:before_remove",
    "compact:after_first_remove",
    "compact:before_wal_truncate",
];

fn config(flush_mode: FlushMode) -> BlobStorageConfig {
    BlobStorageConfig {
        sync_policy: SyncPolicy::EveryWrite,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode,
        max_sstables_before_compaction: usize::MAX,
    }
}

/// `(key, expected value)` for every key a scenario touches; `None`
/// means the key must read as absent.
type Expected = Vec<(&'static [u8], Option<&'static [u8]>)>;

fn assert_contents(db: &AliceDB, expected: &Expected, context: &str) {
    for (key, want) in expected {
        let got = db.get_blob(key).unwrap();
        assert_eq!(
            got.as_deref(),
            *want,
            "{context}: key {:?}",
            String::from_utf8_lossy(key)
        );
    }
    let mut want_scan: Vec<(Vec<u8>, Vec<u8>)> = expected
        .iter()
        .filter_map(|(k, v)| v.map(|v| (k.to_vec(), v.to_vec())))
        .collect();
    want_scan.sort();
    assert_eq!(
        db.scan_blob_prefix(b"").unwrap(),
        want_scan,
        "{context}: scan"
    );
}

fn tmp_files(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().ok())
        .filter(|n| {
            Path::new(n)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("tmp"))
        })
        .collect()
}

/// Append mode, three generations of data:
/// - `blob-000001.sst`: a=1, b=1, c=1
/// - `blob-000002.sst`: a=2, b=tombstone (a delete that reached a file)
/// - memtable / WAL:    d=4, c deleted (a delete that only the WAL holds)
fn build_append_scenario(dir: &Path) -> (AliceDB, Expected) {
    let db = AliceDB::open_with_blob_config(dir, config(FlushMode::Append)).unwrap();
    db.put_blob(b"a", b"1").unwrap();
    db.put_blob(b"b", b"1").unwrap();
    db.put_blob(b"c", b"1").unwrap();
    db.compact_blob_sstable().unwrap(); // Append mode: one new numbered file
    db.put_blob(b"a", b"2").unwrap();
    db.delete_blob(b"b").unwrap();
    db.compact_blob_sstable().unwrap(); // Append mode: one new numbered file
    db.put_blob(b"d", b"4").unwrap();
    db.delete_blob(b"c").unwrap();
    assert_eq!(db.blob_sstable_count().unwrap(), 2);
    let expected: Expected = vec![
        (b"a", Some(b"2")),
        (b"b", None),
        (b"c", None),
        (b"d", Some(b"4")),
    ];
    (db, expected)
}

/// Overwrite mode after a stint in Append mode:
/// - `blob.sst`:        a=1, b=1 (mapped when compaction replaces it)
/// - `blob-000001.sst`: e=5      (written while the store ran in Append
///   mode; outranks `blob.sst`, so it must be gone before the WAL goes)
/// - memtable / WAL:    a deleted, c=3
fn build_overwrite_scenario(dir: &Path) -> (AliceDB, Expected) {
    let db = AliceDB::open_with_blob_config(dir, config(FlushMode::Overwrite)).unwrap();
    db.put_blob(b"a", b"1").unwrap();
    db.put_blob(b"b", b"1").unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);
    let db = AliceDB::open_with_blob_config(dir, config(FlushMode::Append)).unwrap();
    db.put_blob(b"e", b"5").unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);
    let db = AliceDB::open_with_blob_config(dir, config(FlushMode::Overwrite)).unwrap();
    db.delete_blob(b"a").unwrap();
    db.put_blob(b"c", b"3").unwrap();
    assert_eq!(db.blob_sstable_count().unwrap(), 2);
    let expected: Expected = vec![
        (b"a", None),
        (b"b", Some(b"1")),
        (b"c", Some(b"3")),
        (b"e", Some(b"5")),
    ];
    (db, expected)
}

fn crash_then_reopen(build: fn(&Path) -> (AliceDB, Expected), flush_mode: FlushMode, point: &str) {
    let tmp = TempDir::new().unwrap();
    let (db, expected) = build(tmp.path());
    assert_contents(&db, &expected, "before compaction");

    fs_audit::arm_crash_point(point);
    let result = db.compact_all_blob_sstables();
    fs_audit::disarm_all_crash_points();
    let err = result.expect_err("armed crash point must abort the compaction");
    assert!(
        err.to_string().contains(point),
        "compaction failed somewhere other than {point}: {err}"
    );
    drop(db);

    let db = AliceDB::open_with_blob_config(tmp.path(), config(flush_mode)).unwrap();
    assert_contents(&db, &expected, &format!("reopen after crash at {point}"));
    assert!(
        tmp_files(tmp.path()).is_empty(),
        "reopen after crash at {point} left {:?}",
        tmp_files(tmp.path())
    );

    // The next compaction absorbs whatever the crash left behind.
    db.compact_all_blob_sstables().unwrap();
    assert_eq!(db.blob_sstable_count().unwrap(), 1);
    assert_eq!(enumerate_sstables(tmp.path()).unwrap().len(), 1);
    assert_contents(&db, &expected, &format!("recompact after crash at {point}"));
    drop(db);

    let db = AliceDB::open_with_blob_config(tmp.path(), config(flush_mode)).unwrap();
    assert_contents(
        &db,
        &expected,
        &format!("final reopen after crash at {point}"),
    );
}

#[test]
fn open_file_audit_is_active_in_test_builds() {
    // Guards against the audit silently switching off (for example if
    // the self dev-dependency is dropped): every other test in the
    // blob suites relies on it to enforce the Windows rules.
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(
        fs_audit::is_open(&wal),
        "audit did not record the WAL handle"
    );
    db.put_blob(b"k", b"v").unwrap();
    db.compact_blob_sstable().unwrap();
    assert!(fs_audit::is_open(&tmp.path().join("blob.sst")));
    drop(db);
    assert!(!fs_audit::is_open(&wal));
    assert!(!fs_audit::is_open(&tmp.path().join("blob.sst")));
}

#[test]
fn append_mode_survives_a_crash_at_every_compaction_step() {
    for point in CRASH_POINTS {
        crash_then_reopen(build_append_scenario, FlushMode::Append, point);
    }
}

#[test]
fn overwrite_mode_survives_a_crash_at_every_compaction_step() {
    for point in CRASH_POINTS {
        crash_then_reopen(build_overwrite_scenario, FlushMode::Overwrite, point);
    }
}

#[test]
fn overwrite_mode_replaces_its_mapped_sstable_repeatedly() {
    // Each round replaces `blob.sst` while the store has it mapped from
    // the previous round, which is the case Windows rejects unless the
    // mapping is released first.
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    for round in 0_u8..5 {
        db.put_blob(&[b'k', round], &[round]).unwrap();
        db.compact_blob_sstable().unwrap();
        for seen in 0..=round {
            assert_eq!(db.get_blob(&[b'k', seen]).unwrap(), Some(vec![seen]));
        }
    }
    assert_eq!(db.blob_sstable_count().unwrap(), 1);
    drop(db);
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.blob_len(), 5);
}

#[test]
fn compaction_while_other_threads_read() {
    // Readers keep probing the SSTables while compactions replace and
    // delete them. Under the audit, any mapping that outlived the
    // list's write lock would turn the delete into a panic.
    let tmp = TempDir::new().unwrap();
    let db =
        Arc::new(AliceDB::open_with_blob_config(tmp.path(), config(FlushMode::Append)).unwrap());
    for i in 0_u8..20 {
        db.put_blob(&[b'k', i], &[i]).unwrap();
        if i % 5 == 4 {
            db.compact_blob_sstable().unwrap();
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for i in 0_u8..20 {
                        assert_eq!(db.get_blob(&[b'k', i]).unwrap(), Some(vec![i]));
                    }
                    assert_eq!(db.scan_blob_prefix(b"k").unwrap().len(), 20);
                }
            })
        })
        .collect();
    for _ in 0..20 {
        db.compact_all_blob_sstables().unwrap();
        db.put_blob(b"extra", b"x").unwrap();
        db.compact_blob_sstable().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
    assert_eq!(
        enumerate_sstables(tmp.path()).unwrap().len(),
        db.blob_sstable_count().unwrap()
    );
}

#[test]
fn wal_truncation_keeps_later_writes_after_reopen() {
    // The WAL handle is not opened in append mode (Windows refuses to
    // truncate through one), so writes must still land at the end after
    // replay moved the cursor and after truncation reset the file.
    let tmp = TempDir::new().unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"one", b"1").unwrap();
    drop(db);
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"two", b"2").unwrap();
    db.compact_blob_sstable().unwrap();
    db.put_blob(b"three", b"3").unwrap();
    drop(db);
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"four", b"4").unwrap();
    drop(db);
    let db = AliceDB::open(tmp.path()).unwrap();
    for (k, v) in [
        (&b"one"[..], &b"1"[..]),
        (b"two", b"2"),
        (b"three", b"3"),
        (b"four", b"4"),
    ] {
        assert_eq!(db.get_blob(k).unwrap().as_deref(), Some(v));
    }
}
