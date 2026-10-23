//! A write acknowledged after a recovery must survive the next reopen.
//!
//! Recovery drops a torn or corrupt tail of the blob WAL. If the bytes of that tail stay in the file,
//! the next append lands behind them, and the following replay stops at the same tail again: the
//! acknowledged write is gone. Oracle: the model is "the records before the tail, then the new writes";
//! it is built from the operations the test performs, not from the store.
#![cfg(feature = "fs")]

use std::io::{Seek, SeekFrom, Write};

use alice_db::AliceDB;
use tempfile::TempDir;

fn write_two(tmp: &TempDir) -> u64 {
    let db = AliceDB::open(tmp.path()).unwrap();
    db.put_blob(b"k1", b"v1").unwrap();
    db.put_blob(b"k2", b"v2").unwrap();
    drop(db);
    std::fs::metadata(tmp.path().join("blob.wal")).unwrap().len()
}

fn assert_after_recovery_write_survives(tmp: &TempDir) {
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        assert_eq!(db.get_blob(b"k1").unwrap(), Some(b"v1".to_vec()));
        db.put_blob(b"k3", b"v3").unwrap();
        assert_eq!(db.get_blob(b"k3").unwrap(), Some(b"v3".to_vec()));
    }
    let db = AliceDB::open(tmp.path()).unwrap();
    assert_eq!(db.get_blob(b"k1").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(
        db.get_blob(b"k3").unwrap(),
        Some(b"v3".to_vec()),
        "a write acknowledged after recovery was lost on the next reopen"
    );
}

#[test]
fn a_write_after_a_torn_tail_survives_the_next_reopen() {
    let tmp = TempDir::new().unwrap();
    let full = write_two(&tmp);
    // every cut inside the second record (18 bytes: header 10 + key 2 + value 2 + crc 4)
    for cut in 1..18 {
        let t = TempDir::new().unwrap();
        for name in ["blob.wal"] {
            std::fs::copy(tmp.path().join(name), t.path().join(name)).unwrap();
        }
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(t.path().join("blob.wal"))
            .unwrap();
        f.set_len(full - cut).unwrap();
        drop(f);
        assert_after_recovery_write_survives(&t);
    }
}

#[test]
fn a_write_after_a_corrupt_record_survives_the_next_reopen() {
    let tmp = TempDir::new().unwrap();
    write_two(&tmp);
    // one byte into the second record's key (first record is 18 bytes, header 10)
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(tmp.path().join("blob.wal"))
        .unwrap();
    f.seek(SeekFrom::Start(18 + 10)).unwrap();
    f.write_all(&[0xff]).unwrap();
    drop(f);
    assert_after_recovery_write_survives(&tmp);
}
