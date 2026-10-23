//! A crash after the dropped bytes are copied and before the WAL is cut loses nothing: the next open cuts at the
//! same record, the copy holds the same bytes, and a write after it survives a reopen.
#![cfg(all(feature = "fs", feature = "open-file-audit"))]

use alice_db::{fs_audit, AliceDB};
use tempfile::TempDir;

#[test]
fn a_crash_between_the_copy_and_the_cut_is_redone_on_the_next_open() {
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"k1", b"v1").unwrap();
        db.put_blob(b"k2", b"v2").unwrap();
    }
    let wal = tmp.path().join("blob.wal");
    let full = std::fs::read(&wal).unwrap();
    let torn = &full[..full.len() - 3];
    std::fs::write(&wal, torn).unwrap();

    fs_audit::arm_crash_point("wal_recover:before_cut");
    let opened = AliceDB::open(tmp.path());
    fs_audit::disarm_all_crash_points();
    let Err(err) = opened else {
        panic!("the armed crash point must stop the open");
    };
    assert!(err.to_string().contains("wal_recover:before_cut"), "{err}");
    let copy = tmp.path().join("blob.wal.dropped-18");
    assert_eq!(
        std::fs::read(&copy).unwrap(),
        torn[18..].to_vec(),
        "the copy is durable before the cut"
    );
    assert_eq!(
        std::fs::read(&wal).unwrap(),
        torn,
        "the crash came before the cut"
    );

    {
        let db = AliceDB::open(tmp.path()).unwrap();
        let rec = db.blob_wal_recovery().expect("the cut is redone");
        assert_eq!(
            (rec.offset, rec.dropped_len),
            (18, (torn.len() - 18) as u64)
        );
        assert_eq!(std::fs::read(&copy).unwrap(), torn[18..].to_vec());
        db.put_blob(b"k3", b"v3").unwrap();
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(db.blob_wal_recovery().is_none());
    assert_eq!(db.get_blob(b"k1").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(db.get_blob(b"k3").unwrap(), Some(b"v3".to_vec()));
    let copies: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("blob.wal.dropped-"))
        .collect();
    assert_eq!(
        copies,
        ["blob.wal.dropped-18"],
        "the redo reused the copy instead of writing another"
    );
}

#[test]
fn the_copy_the_directory_and_the_cut_are_fsynced_in_order() {
    // oracle: the order the module doc states: copy file, then its directory entry, then the cut WAL
    let tmp = TempDir::new().unwrap();
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"k1", b"v1").unwrap();
        db.put_blob(b"k2", b"v2").unwrap();
    }
    let wal = tmp.path().join("blob.wal");
    let full = std::fs::read(&wal).unwrap();
    std::fs::write(&wal, &full[..full.len() - 3]).unwrap();

    let _ = fs_audit::take_syncs();
    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(db.blob_wal_recovery().is_some());
    let syncs = fs_audit::take_syncs();
    let dir = tmp.path().canonicalize().unwrap();
    let file = ("file".to_owned(), dir.join("blob.wal.dropped-18.tmp"));
    let cut = ("wal".to_owned(), dir.join("blob.wal"));
    let expected = if cfg!(unix) {
        vec![file, ("dir".to_owned(), dir.clone()), cut]
    } else {
        vec![file, cut]
    };
    assert_eq!(syncs, expected);
}
