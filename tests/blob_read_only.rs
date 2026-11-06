//! `open_existing_read_only`, `count_prefix` and `snapshot_to`.
//!
//! A read-only open creates nothing, cuts nothing and needs no lock, so it works
//! beside a writer; what it sees is one moment of that writer. The counts and the
//! snapshots are compared with a reference map built from the same operations.
#![cfg(feature = "fs")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a concurrency test: the threads exercise the store's locking and are joined"
)]

use alice_db::blob::{BlobStorage, WriteBatch};
use alice_db::snapshot;
use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tempfile::TempDir;

fn listing(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().into_string().unwrap(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect()
}

#[test]
fn a_missing_store_is_not_found_and_nothing_is_created() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("no/such/dir/blob.wal");
    let err = BlobStorage::open_existing_read_only(&path).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
    assert!(listing(tmp.path()).is_empty(), "the open created something");
    let path = tmp.path().join("blob.wal");
    let err = BlobStorage::open_existing_read_only(&path).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
    assert!(listing(tmp.path()).is_empty(), "the open created something");
}

#[test]
fn it_reads_sstables_and_wal_beside_an_open_writer() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let writer = BlobStorage::open(&path).unwrap();
    writer.put(b"a", b"sst").unwrap();
    writer.put(b"b", b"sst").unwrap();
    writer.put(b"c", b"sst").unwrap();
    writer.flush_to_sstable().unwrap();
    writer.put(b"a", b"wal").unwrap();
    writer.delete(b"b").unwrap();

    let reader = BlobStorage::open_existing_read_only(&path).unwrap();
    assert_eq!(reader.get(b"a").unwrap().unwrap(), b"wal");
    assert_eq!(reader.get(b"b").unwrap(), None);
    assert_eq!(reader.get(b"c").unwrap().unwrap(), b"sst");

    // one moment: later writes are not seen until the next open
    writer.put(b"d", b"later").unwrap();
    assert_eq!(reader.get(b"d").unwrap(), None);
    let again = BlobStorage::open_existing_read_only(&path).unwrap();
    assert_eq!(again.get(b"d").unwrap().unwrap(), b"later");
}

#[test]
fn it_refuses_writes_and_changes_no_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    {
        let writer = BlobStorage::open(&path).unwrap();
        writer.put(b"a", b"1").unwrap();
        writer.flush_to_sstable().unwrap();
        writer.put(b"b", b"2").unwrap();
    }
    // a torn tail: a writer would cut it and keep a copy, a reader leaves it
    let mut wal = std::fs::read(&path).unwrap();
    wal.extend_from_slice(&[1, 9, 0, 0]);
    std::fs::write(&path, &wal).unwrap();
    let before = listing(tmp.path());

    let reader = BlobStorage::open_existing_read_only(&path).unwrap();
    assert_eq!(reader.get(b"b").unwrap().unwrap(), b"2");
    let mut batch = WriteBatch::new();
    batch.put(b"x", b"y");
    for err in [
        reader.put(b"x", b"y").unwrap_err(),
        reader.delete(b"a").unwrap_err(),
        reader.write_batch(&batch).unwrap_err(),
    ] {
        assert_eq!(err.kind(), ErrorKind::PermissionDenied);
    }
    assert_eq!(reader.get(b"x").unwrap(), None);
    assert_eq!(reader.get(b"a").unwrap().unwrap(), b"1");
    // an empty batch is still a write
    let err = reader.write_batch(&WriteBatch::new()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::PermissionDenied);
    reader.flush().unwrap();
    reader.flush_to_sstable().unwrap();
    reader.compact_all_sstables().unwrap();
    drop(reader);
    assert_eq!(
        listing(tmp.path()),
        before,
        "a read-only store changed a file"
    );
}

/// The reader sees whole batches through flushes and compactions, and never goes back in time.
#[test]
fn every_open_sees_one_moment_of_a_busy_writer() {
    const READS: u32 = 100;
    const WRITES: u32 = 1_000;
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let writer = Arc::new(BlobStorage::open(&path).unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let written = Arc::new(AtomicU32::new(0));
    let handle = {
        let (writer, stop, written) =
            (Arc::clone(&writer), Arc::clone(&stop), Arc::clone(&written));
        std::thread::spawn(move || {
            let mut i = 0_u32;
            while !stop.load(Ordering::SeqCst) {
                let v = i.to_be_bytes();
                let mut batch = WriteBatch::new();
                batch.put(b"pair/a", &v);
                batch.put(b"pair/b", &v);
                writer.write_batch(&batch).unwrap();
                match i % 200 {
                    50 => writer.flush_to_sstable().unwrap(),
                    150 => writer.compact_all_sstables().unwrap(),
                    _ => {}
                }
                i += 1;
                written.store(i, Ordering::SeqCst);
            }
        })
    };
    let mut last = [0_u8; 4];
    let mut opened = 0_u32;
    // The writer runs until the reader has seen enough, however the threads are scheduled.
    while opened < READS || written.load(Ordering::SeqCst) < WRITES {
        let reader = BlobStorage::open_existing_read_only(&path).unwrap();
        let got = reader.scan_prefix(b"pair/").unwrap();
        if got.is_empty() {
            continue;
        }
        assert_eq!(got.len(), 2, "one key of a batch: {got:?}");
        assert_eq!(got[0].1, got[1].1, "two batches at once");
        let now: [u8; 4] = got[0].1.as_slice().try_into().unwrap();
        assert!(now >= last, "went back from {last:?} to {now:?}");
        last = now;
        opened += 1;
    }
    stop.store(true, Ordering::SeqCst);
    handle.join().unwrap();
    assert!(
        u32::from_be_bytes(last) > 0,
        "only the first batch was read"
    );
}

/// Operations and a reference map, over a memtable and `SSTables`.
fn store_and_model(path: &Path) -> (BlobStorage, BTreeMap<Vec<u8>, Vec<u8>>) {
    let store = BlobStorage::open(path).unwrap();
    let mut model = BTreeMap::new();
    let mut x = 0x2545_F491_4F6C_DD1D_u64;
    for i in 0..600_u32 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let k = format!(
            "{}/{:02}",
            ["ab", "ac", "b"][(x % 3) as usize],
            (x >> 8) % 50
        )
        .into_bytes();
        if x >> 60 < 4 {
            store.delete(&k).unwrap();
            model.remove(&k);
        } else {
            let v = if (x >> 56).trailing_zeros() >= 2 {
                vec![b'q'; 5000]
            } else {
                x.to_le_bytes().to_vec()
            };
            store.put(&k, &v).unwrap();
            model.insert(k, v);
        }
        if i % 150 == 149 {
            store.flush_to_sstable().unwrap();
        }
    }
    (store, model)
}

#[test]
fn count_prefix_equals_the_reference() {
    let tmp = TempDir::new().unwrap();
    let (store, model) = store_and_model(&tmp.path().join("blob.wal"));
    let mut compared = 0;
    for prefix in [&b""[..], b"a", b"ab", b"ab/", b"ac/0", b"b", b"c", b"\xff"] {
        let want = model.keys().filter(|k| k.starts_with(prefix)).count();
        assert_eq!(store.count_prefix(prefix).unwrap(), want, "{prefix:?}");
        compared += usize::from(want > 0);
    }
    assert!(compared >= 5, "only {compared} non-empty prefixes");
}

/// A prefix of 0xFF bytes has no next key: the range must still end at the last key.
#[test]
fn count_prefix_with_ff_bytes() {
    let store = BlobStorage::new();
    for k in [
        &b"\xfe"[..],
        b"\xff",
        b"\xff\x00",
        b"\xff\xff",
        b"\xff\xff\x01",
    ] {
        store.put(k, b"v").unwrap();
    }
    assert_eq!(store.count_prefix(b"\xff").unwrap(), 4);
    assert_eq!(store.count_prefix(b"\xff\xff").unwrap(), 2);
    assert_eq!(store.count_prefix(b"\xfe").unwrap(), 1);
}

#[test]
fn a_snapshot_decodes_to_the_live_entries() {
    let tmp = TempDir::new().unwrap();
    let (store, model) = store_and_model(&tmp.path().join("blob.wal"));
    assert!(model.len() > 20);
    let mut bytes = Vec::new();
    store.snapshot_to(&mut bytes).unwrap();
    let (files, blobs) = snapshot::decode(&bytes).unwrap();
    assert!(files.is_empty());
    let want: Vec<_> = model.into_iter().collect();
    assert_eq!(blobs, want);
    // the same bytes as the whole-database encoder
    assert_eq!(bytes, snapshot::encode(&files, &want).unwrap());
}

#[test]
fn a_failing_writer_is_reported() {
    struct Full(usize);
    impl std::io::Write for Full {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.0 < buf.len() {
                return Err(std::io::Error::new(ErrorKind::StorageFull, "full"));
            }
            self.0 -= buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let store = BlobStorage::new();
    store.put(b"k", &[7_u8; 4096]).unwrap();
    let err = store.snapshot_to(&mut Full(100)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::StorageFull);
}

#[test]
fn a_snapshot_flushes_its_writer() {
    #[derive(Default)]
    struct Buffered {
        flushed: bool,
    }
    impl std::io::Write for Buffered {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.flushed = false;
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.flushed = true;
            Ok(())
        }
    }
    let store = BlobStorage::new();
    store.put(b"k", b"v").unwrap();
    let mut out = Buffered::default();
    store.snapshot_to(&mut out).unwrap();
    assert!(out.flushed);
}

/// After a compaction a deleted value is in no file of the store.
#[test]
fn a_compaction_leaves_no_bytes_of_a_deleted_value() {
    const GONE: &[u8] = b"a-value-that-was-deleted-and-must-be-gone";
    let tmp = TempDir::new().unwrap();
    let store = BlobStorage::open(tmp.path().join("blob.wal")).unwrap();
    let holds = |dir: &Path| {
        listing(dir)
            .values()
            .any(|bytes| bytes.windows(GONE.len()).any(|w| w == GONE))
    };
    store.put(b"k", GONE).unwrap();
    store.put(b"other", b"stays").unwrap();
    store.flush_to_sstable().unwrap();
    assert!(holds(tmp.path()), "the search must find it first");
    store.delete(b"k").unwrap();
    store.compact_all_sstables().unwrap();
    assert!(!holds(tmp.path()), "still in a file after the compaction");
    assert_eq!(store.get(b"other").unwrap().unwrap(), b"stays");
}

/// A snapshot taken during writes holds whole batches.
#[test]
fn a_snapshot_holds_whole_batches() {
    const SNAPSHOTS: u32 = 200;
    let store = Arc::new(BlobStorage::new());
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (store, stop) = (Arc::clone(&store), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut i = 0_u32;
            while !stop.load(Ordering::SeqCst) {
                let v = i.to_le_bytes();
                let mut batch = WriteBatch::new();
                batch.put(b"a", &v);
                batch.put(b"b", &v);
                store.write_batch(&batch).unwrap();
                i = i.wrapping_add(1);
            }
        })
    };
    // The writer runs until the snapshots are taken, however the threads are scheduled.
    let mut compared = 0;
    let mut values = std::collections::BTreeSet::new();
    while compared < SNAPSHOTS || values.len() < 2 {
        let mut bytes = Vec::new();
        store.snapshot_to(&mut bytes).unwrap();
        let (_, blobs) = snapshot::decode(&bytes).unwrap();
        if blobs.len() == 2 {
            assert_eq!(blobs[0].1, blobs[1].1);
            values.insert(blobs[0].1.clone());
            compared += 1;
        } else {
            assert!(blobs.is_empty());
        }
    }
    stop.store(true, Ordering::SeqCst);
    writer.join().unwrap();
}

const HOLDER_ENV: &str = "ALICE_DB_LOCK_HOLDER";
const HOLDING: &str = "alice-db-lock-holder-ready";

/// Re-entered by the test below with `ALICE_DB_LOCK_HOLDER` set; returns at once
/// without it. Holds the writer until it is killed.
#[test]
fn lock_holder_entry() {
    let Some(path) = std::env::var_os(HOLDER_ENV) else {
        return;
    };
    let writer = BlobStorage::open(Path::new(&path)).unwrap();
    writer.put(b"held", b"1").unwrap();
    println!("{HOLDING}");
    loop {
        std::thread::park();
    }
}

/// A writer in another process: its lock refuses a second writer and leaves
/// reads alone, and the operating system releases it when the process is killed.
#[test]
fn a_killed_writer_releases_its_lock_and_a_read_only_open_works_beside_it() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::time::Duration;

    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "lock_holder_entry",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(HOLDER_ENV, &path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    // libtest prints the test's name without a newline, so the marker ends a line
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let ready = BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
            .any(|line| line.trim_end().ends_with(HOLDING));
        let _ = tx.send(ready);
    });
    let ready = rx.recv_timeout(Duration::from_secs(300));
    if ready != Ok(true) {
        let _ = child.kill();
        panic!("the child did not take the lock: {ready:?}");
    }

    let refused = BlobStorage::open(&path);
    let read = BlobStorage::open_existing_read_only(&path).map(|r| r.get(b"held"));
    let killed = child.kill();
    let status = child.wait().unwrap();
    killed.unwrap();
    assert!(!status.success(), "the child was not killed: {status}");

    let Err(err) = refused else {
        panic!("a second writer must be refused while the first is alive");
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock, "{err}");
    assert_eq!(read.unwrap().unwrap().as_deref(), Some(&b"1"[..]));

    let next = BlobStorage::open(&path).unwrap();
    assert_eq!(next.get(b"held").unwrap().as_deref(), Some(&b"1"[..]));
}

/// A second path to the same WAL does not make a second writer.
#[cfg(unix)]
#[test]
fn a_symlink_to_the_wal_does_not_get_past_the_lock() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let writer = BlobStorage::open(&path).unwrap();
    let other = tmp.path().join("other.wal");
    std::os::unix::fs::symlink(&path, &other).unwrap();
    let Err(err) = BlobStorage::open(&other) else {
        panic!("a second writer through a symlink must be refused");
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock, "{err}");
    drop(writer);
}

/// Nor does a hard link: the second name is the same file.
#[test]
fn a_hard_link_to_the_wal_does_not_get_past_the_lock() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let writer = BlobStorage::open(&path).unwrap();
    let other = tmp.path().join("other.wal");
    std::fs::hard_link(&path, &other).unwrap();
    let Err(err) = BlobStorage::open(&other) else {
        panic!("a second writer through a hard link must be refused");
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock, "{err}");
    drop(writer);
}
