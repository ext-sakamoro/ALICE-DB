//! Two processes write one store at once: `BlobStorage::open_shared`
//!
//! oracle (from the contract, not read from the code): a shared writer holds `blob.lock` only for the length
//! of one write, so two processes can each open the store and write; every write a process saw acknowledged is
//! in the store afterwards, a batch is seen whole or not at all, each process's own writes keep their order, and
//! a process that reads after the other one wrote, flushed and compacted sees the other's data (its view is
//! refreshed, never an earlier file it held). A killed writer releases the lock with its process, so the other
//! one goes on writing at once. A lock held past the wait bound is `TimedOut`, never an endless wait. Every
//! lease bumps a generation recorded in `blob.lease`; the WAL's records carry it.
//!
//! Child processes are this test binary run again with `ROLE_ENV` set to a role and `--exact` naming the
//! `child_entry` test, so the same code runs on every OS of the CI matrix (Windows included).

#![cfg(feature = "fs")]
#![allow(
    clippy::disallowed_methods,
    reason = "the waits are what these tests measure: that a held lock is given up within its bound and that a killed writer's lock is gone at once; nothing the crate computes reads the clock"
)]

use alice_db::blob::{BlobStorage, BlobStorageConfig, SharedOptions};
use alice_db::blob_sstable::FlushMode;
use std::collections::BTreeSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;

const ROLE_ENV: &str = "ALICE_DB_SHARED_WRITER_ROLE";
const WAL_ENV: &str = "ALICE_DB_SHARED_WRITER_WAL";
const COUNT_ENV: &str = "ALICE_DB_SHARED_WRITER_COUNT";
const ACKS_ENV: &str = "ALICE_DB_SHARED_WRITER_ACKS";

/// Batches per writer in the concurrent test
const BATCHES: usize = 150;
/// Keys per batch: a batch is seen whole or not at all
const KEYS_PER_BATCH: usize = 3;

fn options() -> SharedOptions {
    SharedOptions::default().with_lock_wait(Duration::from_secs(20))
}

fn config() -> BlobStorageConfig {
    BlobStorageConfig::default()
        .with_flush_mode(FlushMode::Append)
        .with_wal_flush_threshold_bytes(4 * 1024)
        .with_max_sstables_before_compaction(3)
}

fn open(wal: &Path) -> BlobStorage {
    BlobStorage::open_shared(wal, config(), options()).expect("open shared")
}

fn key(tag: &str, batch: usize, part: usize) -> Vec<u8> {
    format!("{tag}/{batch:05}/{part}").into_bytes()
}

/// One batch of `tag`: `KEYS_PER_BATCH` keys holding the batch number, and the shared key holding `tag:batch`
fn write_batch(store: &BlobStorage, tag: &str, batch: usize) {
    let value = batch.to_string().into_bytes();
    let shared = format!("{tag}:{batch}").into_bytes();
    let mut ops: Vec<(Vec<u8>, Vec<u8>)> = (0..KEYS_PER_BATCH)
        .map(|p| (key(tag, batch, p), value.clone()))
        .collect();
    ops.push((b"shared".to_vec(), shared));
    let mut wb = alice_db::blob::WriteBatch::new();
    for (k, v) in &ops {
        wb.put(k, v);
    }
    store.write_batch(&wb).expect("write batch");
}

/// The batches of `tag` present in `store`; panics on a batch seen in part
fn batches_of(store: &BlobStorage, tag: &str, upto: usize) -> BTreeSet<usize> {
    let mut seen = BTreeSet::new();
    for batch in 0..upto {
        let present: Vec<bool> = (0..KEYS_PER_BATCH)
            .map(|p| store.get(&key(tag, batch, p)).expect("get").is_some())
            .collect();
        assert!(
            present.iter().all(|&p| p) || present.iter().all(|&p| !p),
            "batch {tag}/{batch} is seen in part: {present:?}"
        );
        if present[0] {
            assert_eq!(
                store.get(&key(tag, batch, 0)).unwrap().unwrap(),
                batch.to_string().into_bytes(),
                "batch {tag}/{batch} holds another batch's value"
            );
            seen.insert(batch);
        }
    }
    seen
}

fn spawn(role: &str, wal: &Path, count: usize, acks: Option<&Path>) -> Child {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, role)
        .env(WAL_ENV, wal)
        .env(COUNT_ENV, count.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(acks) = acks {
        cmd.env(ACKS_ENV, acks);
    }
    cmd.spawn().expect("spawn child")
}

/// The child side: runs only when the parent set `ROLE_ENV`
#[test]
fn child_entry() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    let wal = PathBuf::from(std::env::var_os(WAL_ENV).unwrap());
    let count: usize = std::env::var(COUNT_ENV).unwrap().parse().unwrap();
    let store = open(&wal);
    match role.as_str() {
        // `count` batches, each acknowledged write appended to the acks file (one line, then fsync)
        "writer-a" | "writer-b" => {
            let tag = &role[7..];
            let acks = std::env::var_os(ACKS_ENV).map(PathBuf::from);
            for batch in 0..count {
                write_batch(&store, tag, batch);
                if let Some(acks) = &acks {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(acks)
                        .unwrap();
                    writeln!(f, "{batch}").unwrap();
                    f.sync_all().unwrap();
                }
            }
        }
        // writes, then flushes into SSTables and compacts them, so a reader's earlier view of the files is stale
        "compactor" => {
            for batch in 0..count {
                write_batch(&store, "c", batch);
            }
            store.flush_to_sstable().unwrap();
            for batch in count..2 * count {
                write_batch(&store, "c", batch);
                store.flush_to_sstable().unwrap();
            }
            store.compact_all_sstables().unwrap();
        }
        other => panic!("unknown role {other}"),
    }
}

fn wait_ok(mut child: Child, what: &str) {
    let status = child.wait().unwrap();
    assert!(status.success(), "{what} exited with {status}");
}

#[test]
fn two_processes_write_at_once_and_nothing_is_lost_or_torn() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    drop(open(&wal));
    let a = spawn("writer-a", &wal, BATCHES, None);
    let b = spawn("writer-b", &wal, BATCHES, None);
    wait_ok(a, "writer a");
    wait_ok(b, "writer b");

    let store = open(&wal);
    let all: BTreeSet<usize> = (0..BATCHES).collect();
    assert_eq!(
        batches_of(&store, "a", BATCHES),
        all,
        "writer a's acknowledged batches"
    );
    assert_eq!(
        batches_of(&store, "b", BATCHES),
        all,
        "writer b's acknowledged batches"
    );
    // the shared key ends on one writer's last batch: each writer's own writes kept their order
    let last = store.get(b"shared").unwrap().unwrap();
    let last = String::from_utf8(last).unwrap();
    assert!(
        last == format!("a:{}", BATCHES - 1) || last == format!("b:{}", BATCHES - 1),
        "the shared key ends on {last}, not on a writer's last batch"
    );
    // a fresh process reading the same files sees the same
    drop(store);
    let again = open(&wal);
    assert_eq!(batches_of(&again, "a", BATCHES), all);
    assert_eq!(batches_of(&again, "b", BATCHES), all);
}

#[test]
fn a_killed_writer_keeps_what_was_acknowledged_and_the_other_goes_on_at_once() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let acks = tmp.path().join("acks-b");
    drop(open(&wal));
    let mut b = spawn("writer-b", &wal, 1_000_000, Some(&acks));
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let n = std::fs::read_to_string(&acks).map_or(0, |s| s.lines().count());
        if n >= 20 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "writer b acknowledged only {n} batches"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    b.kill().unwrap();
    b.wait().unwrap();
    let acked: BTreeSet<usize> = std::fs::read_to_string(&acks)
        .unwrap()
        .lines()
        .map(|l| l.parse().unwrap())
        .collect();

    // the survivor takes the lease without waiting out a timeout: the kernel released it with the process
    let started = Instant::now();
    let a = spawn("writer-a", &wal, 20, None);
    wait_ok(a, "writer a after the kill");
    assert!(
        started.elapsed() < Duration::from_secs(15),
        "the lock outlived the killed writer"
    );

    let store = open(&wal);
    let max = acked.iter().max().copied().unwrap_or(0) + 2;
    let seen = batches_of(&store, "b", max);
    assert!(
        acked.is_subset(&seen),
        "acknowledged {acked:?} but present {seen:?}"
    );
    // at most the one batch in flight at the kill went in beyond the acknowledged ones
    assert!(
        seen.len() <= acked.len() + 1,
        "present {} for {} acknowledged",
        seen.len(),
        acked.len()
    );
    assert_eq!(
        batches_of(&store, "a", 20),
        (0..20).collect(),
        "the survivor's batches"
    );
}

#[test]
fn a_long_lived_reader_sees_the_others_writes_after_they_were_compacted() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let reader = open(&wal);
    for batch in 0..10 {
        write_batch(&reader, "r", batch);
    }
    reader.flush_to_sstable().unwrap();
    // the reader's view now holds the SSTable it wrote; the other process flushes and compacts it away
    wait_ok(spawn("compactor", &wal, 10, None), "compactor");
    assert_eq!(
        batches_of(&reader, "r", 10),
        (0..10).collect(),
        "the reader's own batches after compaction"
    );
    assert_eq!(
        batches_of(&reader, "c", 20),
        (0..20).collect(),
        "the other process's batches"
    );
    // and the reader can still write on top
    write_batch(&reader, "r", 10);
    assert_eq!(batches_of(&open(&wal), "r", 11), (0..11).collect());
}

#[test]
fn a_lock_held_past_the_wait_bound_is_timed_out() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let store = BlobStorage::open_shared(
        &wal,
        config(),
        SharedOptions::default().with_lock_wait(Duration::from_millis(300)),
    )
    .unwrap();
    // another handle on the lock file holds it, as a stalled process would
    let holder = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(tmp.path().join("blob.lock"))
        .unwrap();
    fs2::FileExt::lock_exclusive(&holder).unwrap();
    let started = Instant::now();
    let err = store.put(b"k", b"v").expect_err("the lock is held");
    assert_eq!(err.kind(), ErrorKind::TimedOut, "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "waited {:?}",
        started.elapsed()
    );
    // closing the handle ends the hold on every OS (on Windows the share mode belongs to the open handle)
    drop(holder);
    store.put(b"k", b"v").expect("the lock is free again");
    assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));
}

#[test]
fn every_lease_bumps_the_generation() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let store = open(&wal);
    let g0 = store.lease_generation().unwrap();
    store.put(b"a", b"1").unwrap();
    store.put(b"b", b"2").unwrap();
    assert_eq!(
        store.lease_generation().unwrap(),
        g0 + 2,
        "two writes, two leases"
    );
    let other = open(&wal);
    other.put(b"c", b"3").unwrap();
    assert_eq!(
        store.lease_generation().unwrap(),
        g0 + 3,
        "the other handle's lease counts too"
    );
}

#[test]
fn an_exclusive_open_of_a_shared_store_is_refused_and_a_held_exclusive_store_times_out_the_shared_writer(
) {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    drop(open(&wal));
    let err = BlobStorage::open(&wal).expect_err("the store is shared");
    assert_eq!(err.kind(), ErrorKind::InvalidInput, "{err}");

    let tmp2 = TempDir::new().unwrap();
    let wal2 = tmp2.path().join("blob.wal");
    let exclusive = BlobStorage::open(&wal2).unwrap();
    let err = BlobStorage::open_shared(
        &wal2,
        config(),
        SharedOptions::default().with_lock_wait(Duration::from_millis(200)),
    )
    .expect_err("an exclusive writer holds the store");
    assert_eq!(err.kind(), ErrorKind::TimedOut, "{err}");
    drop(exclusive);
}

/// A store written by alice-db 0.7.0 (exclusive, records without a generation) opens shared and reads back
#[test]
fn a_store_written_by_0_7_0_opens_shared() {
    let tmp = TempDir::new().unwrap();
    copy_dir(&fixture_dir("blob_v07_exclusive"), tmp.path());
    let wal = tmp.path().join("blob.wal");
    let store = open(&wal);
    for i in 0..5u8 {
        assert_eq!(
            store.get(&[b'k', b'0' + i]).unwrap(),
            Some(vec![b'v', b'0' + i]),
            "k{i}"
        );
    }
    store.put(b"new", b"1").unwrap();
    drop(store);
    assert_eq!(open(&wal).get(b"new").unwrap(), Some(b"1".to_vec()));
}

#[test]
fn a_file_system_whose_locks_do_not_hold_is_refused_for_shared_writers() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    alice_db::fs_audit::force_lock_probe_failure(true);
    let res = BlobStorage::open_shared(&wal, config(), options());
    alice_db::fs_audit::force_lock_probe_failure(false);
    let err = res.expect_err("locks that do not hold");
    assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    assert!(
        !tmp.path().join("blob.lease").exists(),
        "a refused open leaves no lease file"
    );
}

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn copy_dir(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

/// A writer killed in the middle of appending leaves part of a record at the end of the WAL; the next lease
/// cuts it before appending, so the next write is read back (not hidden behind the broken bytes).
#[test]
fn a_record_cut_short_by_a_killed_writer_is_cut_off_before_the_next_append() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let store = open(&wal);
    // both handles are open before the broken bytes appear (an open cuts a torn tail itself)
    let other = open(&wal);
    store.put(b"before", b"1").unwrap();
    // the first bytes of a record header, as a write stopped after them would leave
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
        f.write_all(&[0x04, 0x20, 0, 0, 0, 0x30]).unwrap();
        f.sync_all().unwrap();
    }
    other.put(b"after", b"2").unwrap();
    drop((store, other));
    let again = open(&wal);
    assert_eq!(again.get(b"before").unwrap(), Some(b"1".to_vec()));
    assert_eq!(
        again.get(b"after").unwrap(),
        Some(b"2".to_vec()),
        "the write after the broken bytes"
    );
}

/// A key deleted before a flush stays deleted after the next flush and a reopen: the flush that drops its
/// tombstone removes the earlier files that still hold the key.
#[test]
fn a_key_deleted_and_flushed_does_not_come_back() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let store = open(&wal);
    store.put(b"gone", b"1").unwrap();
    store.put(b"kept", b"2").unwrap();
    store.flush_to_sstable().unwrap();
    store.delete(b"gone").unwrap();
    store.flush_to_sstable().unwrap();
    assert_eq!(store.get(b"gone").unwrap(), None);
    drop(store);
    let again = open(&wal);
    assert_eq!(
        again.get(b"gone").unwrap(),
        None,
        "a deleted key came back after a flush"
    );
    assert_eq!(again.get(b"kept").unwrap(), Some(b"2".to_vec()));
    assert_eq!(
        again.sstable_count().unwrap(),
        1,
        "every flush leaves one SSTable"
    );
}

/// A shared handle holds the whole store in memory, so it holds at most `SharedOptions::max_bytes` of keys and
/// stored values: a write that would take the store past it is refused with a typed error and nothing is
/// written, and a handle whose limit is below what the store already holds refuses to open it.
#[test]
fn a_store_past_the_handles_memory_limit_is_refused_with_a_typed_error() {
    use alice_db::blob::SharedStoreTooLarge;
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let small = SharedOptions::default().with_max_bytes(4096);
    let store = BlobStorage::open_shared(&wal, config(), small).unwrap();
    // incompressible values (xorshift), so the stored bytes are the value's bytes: 3 × 1301 fit in 4096, 4 do not
    let value = |seed: u64| -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..1300)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x.to_le_bytes()[0]
            })
            .collect()
    };
    store.put(b"a", &value(1)).unwrap();
    store.put(b"b", &value(2)).unwrap();
    store.put(b"c", &value(3)).unwrap();
    let err = store.put(b"d", &value(4)).expect_err("past 4096 bytes");
    assert_eq!(err.kind(), ErrorKind::FileTooLarge, "{err}");
    let too_large = err
        .get_ref()
        .and_then(|e| e.downcast_ref::<SharedStoreTooLarge>())
        .expect("typed error");
    assert_eq!(too_large.limit, 4096);
    assert!(too_large.bytes > 4096, "{too_large:?}");
    // nothing of the refused write is in the store
    let other = open(&wal);
    assert_eq!(other.get(b"d").unwrap(), None);
    // replacing a value with one of the same size stays within the limit
    store.put(b"a", &value(9)).unwrap();
    // a handle whose limit is below what the store holds does not open it
    let tiny = SharedOptions::default().with_max_bytes(1024);
    let err = BlobStorage::open_shared(&wal, config(), tiny)
        .expect_err("the store holds more than 1024 bytes");
    assert_eq!(err.kind(), ErrorKind::FileTooLarge, "{err}");
}

/// A flush that stops between raising the epoch and emptying the WAL (a crash point there) leaves every other
/// handle reading the other writer's records: a handle that had applied nothing of the WAL still sees them,
/// and its own next flush keeps them. (Emptying the WAL before raising the epoch would leave that handle at
/// the old epoch with nothing to read, and its next flush would remove the only file that holds them.)
#[test]
fn a_flush_cut_between_its_epoch_and_its_wal_truncate_loses_nothing() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    // `a` opens the empty store: epoch 0, nothing of the WAL applied
    let a = open(&wal);
    let b = open(&wal);
    for batch in 0..5 {
        write_batch(&b, "b", batch);
    }
    alice_db::fs_audit::arm_crash_point("shared_flush:before_wal_truncate");
    let cut = b.flush_to_sstable();
    alice_db::fs_audit::disarm_all_crash_points();
    assert!(cut.is_err(), "the crash point fired");
    assert_eq!(
        batches_of(&a, "b", 5),
        (0..5).collect(),
        "a reads b's batches after b's cut flush"
    );
    a.flush_to_sstable().unwrap();
    drop((a, b));
    assert_eq!(
        batches_of(&open(&wal), "b", 5),
        (0..5).collect(),
        "b's batches survive a's flush"
    );
}
