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
    std::fs::metadata(tmp.path().join("blob.wal"))
        .unwrap()
        .len()
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
        std::fs::copy(tmp.path().join("blob.wal"), t.path().join("blob.wal")).unwrap();
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

#[test]
fn the_dropped_bytes_are_kept_and_the_recovery_is_reported() {
    // oracle: the cut keeps the first record (18 bytes) and drops what the test wrote after it
    let tmp = TempDir::new().unwrap();
    let full = write_two(&tmp);
    let wal = tmp.path().join("blob.wal");
    let before = std::fs::read(&wal).unwrap();
    let cut = 5;
    let f = std::fs::OpenOptions::new().write(true).open(&wal).unwrap();
    f.set_len(full - cut).unwrap();
    drop(f);
    let expected_dropped = before[18..usize::try_from(full - cut).unwrap()].to_vec();

    let db = AliceDB::open(tmp.path()).unwrap();
    let rec = db
        .blob_wal_recovery()
        .expect("a torn tail must be reported");
    assert_eq!(rec.offset, 18);
    assert_eq!(rec.dropped_len, expected_dropped.len() as u64);
    assert_eq!(rec.tail, alice_db::blob_wal::WalTail::Torn);
    assert_eq!(rec.dropped_copy, tmp.path().join("blob.wal.dropped-18"));
    assert_eq!(std::fs::read(&rec.dropped_copy).unwrap(), expected_dropped);
    assert_eq!(
        std::fs::metadata(&wal).unwrap().len(),
        18,
        "the WAL ends at the last good record"
    );
}

#[test]
fn a_corrupt_record_is_reported_as_a_bad_checksum() {
    let tmp = TempDir::new().unwrap();
    write_two(&tmp);
    let wal = tmp.path().join("blob.wal");
    let mut bytes = std::fs::read(&wal).unwrap();
    bytes[18 + 10] ^= 0xff;
    std::fs::write(&wal, &bytes).unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    let rec = db
        .blob_wal_recovery()
        .expect("a bad record must be reported");
    assert_eq!(rec.tail, alice_db::blob_wal::WalTail::BadChecksum);
    assert_eq!(
        std::fs::read(&rec.dropped_copy).unwrap(),
        bytes[18..].to_vec()
    );
}

#[test]
fn a_clean_wal_is_never_touched() {
    let tmp = TempDir::new().unwrap();
    write_two(&tmp);
    let wal = tmp.path().join("blob.wal");
    let before = std::fs::read(&wal).unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(db.blob_wal_recovery().is_none());
    drop(db);
    assert_eq!(std::fs::read(&wal).unwrap(), before);
    let extra: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("dropped"))
        .collect();
    assert!(extra.is_empty(), "{extra:?}");
}

/// The copies of what recovery dropped, by file name.
fn dropped_copies(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("blob.wal.dropped-"))
        .collect();
    names.sort();
    names
}

/// Cut `cut` bytes off the end of the WAL and return what lies past the first record (18 bytes).
fn tear(wal: &std::path::Path, cut: usize) -> Vec<u8> {
    let bytes = std::fs::read(wal).unwrap();
    let kept = bytes.len() - cut;
    std::fs::write(wal, &bytes[..kept]).unwrap();
    bytes[18..kept].to_vec()
}

#[test]
fn a_second_recovery_at_the_same_offset_keeps_both_copies() {
    // oracle: each tear's dropped bytes are what the test cut, taken before each open
    let tmp = TempDir::new().unwrap();
    write_two(&tmp);
    let wal = tmp.path().join("blob.wal");
    let first = tear(&wal, 4);
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        assert_eq!(db.blob_wal_recovery().unwrap().offset, 18);
        db.put_blob(b"k3", b"third").unwrap();
    }
    let second = tear(&wal, 2);
    assert_ne!(first, second);
    let db = AliceDB::open(tmp.path()).unwrap();
    let rec = db.blob_wal_recovery().unwrap();
    assert_eq!(rec.offset, 18);
    assert_eq!(rec.dropped_copy, tmp.path().join("blob.wal.dropped-18.1"));
    assert_eq!(
        dropped_copies(tmp.path()),
        ["blob.wal.dropped-18", "blob.wal.dropped-18.1"]
    );
    assert_eq!(
        std::fs::read(tmp.path().join("blob.wal.dropped-18")).unwrap(),
        first
    );
    assert_eq!(
        std::fs::read(tmp.path().join("blob.wal.dropped-18.1")).unwrap(),
        second
    );
}

#[test]
fn a_stale_copy_tmp_is_removed_and_finished_copies_are_kept() {
    let tmp = TempDir::new().unwrap();
    write_two(&tmp);
    let finished = tmp.path().join("blob.wal.dropped-7");
    let stale = tmp.path().join("blob.wal.dropped-18.tmp");
    let stale_n = tmp.path().join("blob.wal.dropped-18.2.tmp");
    let unrelated = tmp.path().join("other.dropped-18.tmp");
    std::fs::write(&finished, b"kept").unwrap();
    std::fs::write(&stale, b"half").unwrap();
    std::fs::write(&stale_n, b"half").unwrap();
    std::fs::write(&unrelated, b"not ours").unwrap();
    let db = AliceDB::open(tmp.path()).unwrap();
    assert!(db.blob_wal_recovery().is_none());
    assert!(
        !stale.exists() && !stale_n.exists(),
        "stale copy .tmp files survive the open"
    );
    assert_eq!(std::fs::read(&finished).unwrap(), b"kept");
    assert!(
        unrelated.exists(),
        "a file that is not a copy of this WAL was removed"
    );
}
