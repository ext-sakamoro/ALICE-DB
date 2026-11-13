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
        let n = std::fs::read_to_string(&acks)
            .map(|s| s.lines().count())
            .unwrap_or(0);
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
        "acknowledged {:?} but present {:?}",
        acked,
        seen
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
