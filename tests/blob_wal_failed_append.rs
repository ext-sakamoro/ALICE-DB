//! A blob write whose WAL append fails part-way, or whose fsync fails, accepts nothing and leaves no partial record
//! for later writes to land behind.
//!
//! oracle (from the rules, not read from the code under test): a `put_blob` that returns `Err` has accepted nothing:
//! its key does not read back, the WAL is the length it was before the write, and a reopen does not replay it; the
//! next write works and survives a reopen with nothing cut from the WAL. When the WAL cannot even be put back, every
//! later write is refused (none lands behind the partial record) and the reopen cuts the partial record, keeping
//! every write acknowledged before it. Every injected failure fired (none = fail).

#![cfg(all(feature = "fs", feature = "open-file-audit"))]

use alice_db::{fs_audit, AliceDB};
use std::path::Path;

fn wal_len(dir: &Path) -> u64 {
    std::fs::metadata(dir.join("blob.wal")).unwrap().len()
}

fn injected(e: &std::io::Error) -> bool {
    e.to_string().contains("injected crash")
}

#[test]
fn a_failed_append_or_fsync_accepts_nothing_and_the_next_write_survives() {
    let mut fired = 0;
    for step in ["wal:mid_frame", "wal:before_sync"] {
        let dir = tempfile::tempdir().unwrap();
        let db = AliceDB::open(dir.path()).unwrap();
        db.put_blob(b"k1", b"one").unwrap();
        let len = wal_len(dir.path());
        fs_audit::arm_crash_point(step);
        let e = db.put_blob(b"k2", b"two").unwrap_err();
        fs_audit::disarm_all_crash_points();
        assert!(injected(&e), "{step}: {e}");
        fired += 1;
        assert_eq!(db.get_blob(b"k2").unwrap(), None, "{step}: not accepted");
        assert_eq!(
            wal_len(dir.path()),
            len,
            "{step}: nothing of the failed record stays"
        );
        db.put_blob(b"k2", b"two").unwrap();
        db.put_blob(b"k3", b"three").unwrap();
        drop(db);
        let db = AliceDB::open(dir.path()).unwrap();
        assert_eq!(
            db.get_blob(b"k1").unwrap().as_deref(),
            Some(&b"one"[..]),
            "{step}"
        );
        assert_eq!(
            db.get_blob(b"k2").unwrap().as_deref(),
            Some(&b"two"[..]),
            "{step}"
        );
        assert_eq!(
            db.get_blob(b"k3").unwrap().as_deref(),
            Some(&b"three"[..]),
            "{step}"
        );
        assert!(
            db.blob_wal_recovery().is_none(),
            "{step}: nothing was cut: {:?}",
            db.blob_wal_recovery()
        );
    }
    assert_eq!(fired, 2, "an injected failure fired at every step");
}

#[test]
fn a_wal_that_cannot_be_put_back_refuses_every_later_write() {
    let dir = tempfile::tempdir().unwrap();
    let db = AliceDB::open(dir.path()).unwrap();
    db.put_blob(b"k1", b"one").unwrap();
    fs_audit::arm_crash_point("wal:mid_frame");
    fs_audit::arm_crash_point("wal:before_rollback");
    let e = db.put_blob(b"k2", b"two").unwrap_err();
    fs_audit::disarm_all_crash_points();
    assert!(injected(&e), "{e}");
    // the partial record could not be removed: nothing may be written behind it
    let e = db.put_blob(b"k3", b"three").unwrap_err();
    assert!(e.to_string().contains("refuses"), "{e}");
    assert_eq!(db.get_blob(b"k3").unwrap(), None);
    drop(db);
    let db = AliceDB::open(dir.path()).unwrap();
    assert_eq!(db.get_blob(b"k1").unwrap().as_deref(), Some(&b"one"[..]));
    assert_eq!(db.get_blob(b"k2").unwrap(), None);
    assert!(
        db.blob_wal_recovery().is_some(),
        "the reopen cut the partial record"
    );
}
