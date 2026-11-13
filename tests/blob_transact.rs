//! `BlobStorage::transact`: reads and one batch with every other writer kept out
//!
//! oracle (from the contract in the method's docs, not read from the code): what a transaction read still holds
//! when its batch is written, so two processes that each add one to a counter N times leave exactly 2N; the same
//! holds for threads of one process on a store that is not shared. While a transaction holds a shared store's
//! lease another process waits at most its `lock_wait` and gets `TimedOut` carrying `WriterBusy`, never an endless
//! wait. A call on the store from inside its own closure is refused with `Deadlock` carrying
//! `NestedTransaction` at once (a write there would otherwise wait for the gate the transaction holds). A closure
//! that returns `Err` or panics writes nothing, and a panic gives up the lease and the in-transaction mark.
//!
//! Child processes are this test binary run again with `ROLE_ENV` set, as in `blob_shared_writers.rs`.

#![cfg(feature = "fs")]
#![allow(
    clippy::disallowed_methods,
    reason = "the waits are what these tests measure: that a held lease is given up within its bound; nothing the crate computes reads the clock"
)]

use alice_db::blob::{
    BlobStorage, BlobStorageConfig, NestedTransaction, SharedOptions, WriteBatch, WriterBusy,
};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const ROLE_ENV: &str = "ALICE_DB_TRANSACT_ROLE";
const WAL_ENV: &str = "ALICE_DB_TRANSACT_WAL";
const COUNT_ENV: &str = "ALICE_DB_TRANSACT_COUNT";

/// Increments per process (and per thread)
const INCREMENTS: u64 = 150;
const COUNTER: &[u8] = b"counter";
/// The wait of the child that meets a held lease: far below how long the parent holds it
const SHORT_WAIT: Duration = Duration::from_millis(200);

fn open(wal: &Path, wait: Duration) -> BlobStorage {
    BlobStorage::open_shared(
        wal,
        BlobStorageConfig::default(),
        SharedOptions::default().with_lock_wait(wait),
    )
    .expect("open shared")
}

fn read_u64(v: Option<Vec<u8>>) -> u64 {
    v.map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
}

/// One read-increment-write of `COUNTER` in a transaction
fn increment(store: &BlobStorage) -> io::Result<u64> {
    store.transact(|txn| {
        let n = read_u64(txn.get(COUNTER)?) + 1;
        let mut batch = WriteBatch::new();
        batch.put(COUNTER, &n.to_le_bytes());
        Ok((batch, n))
    })
}

fn spawn(role: &str, wal: &Path, count: u64) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, role)
        .env(WAL_ENV, wal)
        .env(COUNT_ENV, count.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn child")
}

fn wait_ok(mut child: Child, what: &str) {
    let status = child.wait().unwrap();
    assert!(status.success(), "{what} exited with {status}");
}

/// The child side: runs only when the parent set `ROLE_ENV`
#[test]
fn child_entry() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    let wal = PathBuf::from(std::env::var_os(WAL_ENV).unwrap());
    let count: u64 = std::env::var(COUNT_ENV).unwrap().parse().unwrap();
    match role.as_str() {
        "incrementer" => {
            let store = open(&wal, Duration::from_secs(20));
            for _ in 0..count {
                increment(&store).unwrap();
            }
        }
        // meets a lease the parent holds: must fail with WriterBusy within its short wait, writing nothing
        "busy" => {
            // the open takes the lease too, so either the open or the put meets it
            let start = Instant::now();
            let err = BlobStorage::open_shared(
                &wal,
                BlobStorageConfig::default(),
                SharedOptions::default().with_lock_wait(SHORT_WAIT),
            )
            .and_then(|store| store.put(b"busy", b"x"))
            .unwrap_err();
            let took = start.elapsed();
            assert_eq!(err.kind(), ErrorKind::TimedOut, "{err}");
            let busy = err
                .get_ref()
                .and_then(|e| e.downcast_ref::<WriterBusy>())
                .unwrap_or_else(|| panic!("not WriterBusy: {err}"));
            assert_eq!(busy.waited, SHORT_WAIT);
            assert!(
                took < Duration::from_secs(5),
                "waited {took:?} for a lease held by another process"
            );
        }
        // takes the lease at once (the parent gave it up) and writes
        "writer" => {
            let store = open(&wal, SHORT_WAIT);
            store.put(b"after", b"x").unwrap();
        }
        other => panic!("unknown role {other}"),
    }
}

/// `body` on its own thread, failed when it does not finish in time (a deadlock shows as a red, not a hang)
fn within(limit: Duration, body: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        body();
        let _ = tx.send(());
    });
    match rx.recv_timeout(limit) {
        Ok(()) => handle.join().unwrap(),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            // the body panicked: surface its message
            handle.join().unwrap();
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!(
                "did not finish within {limit:?}: a call waited on the transaction it ran inside"
            )
        }
    }
}

fn is_nested(r: io::Result<impl std::fmt::Debug>) {
    let err = r.expect_err("a call from inside the closure was let through");
    assert_eq!(err.kind(), ErrorKind::Deadlock, "{err}");
    assert!(
        err.get_ref()
            .and_then(|e| e.downcast_ref::<NestedTransaction>())
            .is_some(),
        "not NestedTransaction: {err}"
    );
}

#[test]
fn two_processes_incrementing_one_counter_lose_no_update() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    drop(open(&wal, Duration::from_secs(20)));
    let a = spawn("incrementer", &wal, INCREMENTS);
    let b = spawn("incrementer", &wal, INCREMENTS);
    wait_ok(a, "incrementer a");
    wait_ok(b, "incrementer b");
    let store = open(&wal, Duration::from_secs(20));
    assert_eq!(read_u64(store.get(COUNTER).unwrap()), 2 * INCREMENTS);
}

#[test]
fn threads_incrementing_one_counter_of_an_unshared_store_lose_no_update() {
    for store in [
        Arc::new(BlobStorage::new()),
        Arc::new(BlobStorage::open(TempDir::new().unwrap().keep().join("blob.wal")).unwrap()),
    ] {
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    for _ in 0..INCREMENTS {
                        increment(&store).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(read_u64(store.get(COUNTER).unwrap()), 4 * INCREMENTS);
    }
}

#[test]
fn a_slow_closure_makes_the_other_process_writer_busy_not_wait_forever() {
    let tmp = TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    let store = open(&wal, Duration::from_secs(20));
    let child_wal = wal.clone();
    // the closure holds the lease for as long as the child takes to give up (the docs forbid this; the test does it
    // on purpose to see the other side's bound)
    store
        .transact(move |_txn| {
            let child = spawn("busy", &child_wal, 0);
            wait_ok(child, "the process that met the held lease");
            Ok((WriteBatch::new(), ()))
        })
        .unwrap();
    assert_eq!(store.get(b"busy").unwrap(), None, "the refused put wrote");
}

#[test]
fn calls_on_the_store_from_inside_its_closure_are_refused_not_deadlocked() {
    within(Duration::from_secs(30), || {
        let tmp = TempDir::new().unwrap();
        let shared = open(&tmp.path().join("blob.wal"), Duration::from_secs(20));
        let unshared = BlobStorage::new();
        for store in [&shared, &unshared] {
            store.put(b"k", b"v").unwrap();
            let read = store
                .transact(|txn| {
                    is_nested(store.transact(|_| Ok((WriteBatch::new(), ()))));
                    is_nested(store.put(b"nested", b"x"));
                    is_nested(store.delete(b"k"));
                    let mut inner = WriteBatch::new();
                    inner.put(b"nested", b"x");
                    is_nested(store.write_batch(&inner));
                    is_nested(store.get(b"k"));
                    is_nested(store.scan_prefix(b""));
                    is_nested(store.count_prefix(b""));
                    // the closure's own reads go through
                    let v = txn.get(b"k")?;
                    assert_eq!(txn.count_prefix(b"")?, 1);
                    Ok((WriteBatch::new(), v))
                })
                .unwrap();
            assert_eq!(read.as_deref(), Some(&b"v"[..]));
            assert_eq!(store.get(b"nested").unwrap(), None, "a refused call wrote");
            assert_eq!(store.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
            // outside the closure the store is usable again
            store.put(b"after", b"x").unwrap();
        }
    });
}

#[test]
fn a_closure_that_returns_err_writes_nothing() {
    let tmp = TempDir::new().unwrap();
    let shared = open(&tmp.path().join("blob.wal"), Duration::from_secs(20));
    for store in [&shared, &BlobStorage::new()] {
        let err = store
            .transact(|_txn| -> io::Result<(WriteBatch, ())> {
                let mut batch = WriteBatch::new();
                batch.put(b"partial", b"x");
                drop(batch);
                Err(io::Error::other("refused by the caller"))
            })
            .unwrap_err();
        assert_eq!(err.to_string(), "refused by the caller");
        assert_eq!(store.get(b"partial").unwrap(), None);
    }
}

#[test]
fn a_panic_in_the_closure_writes_nothing_and_gives_up_the_lease() {
    within(Duration::from_secs(30), || {
        let tmp = TempDir::new().unwrap();
        let wal = tmp.path().join("blob.wal");
        let shared = open(&wal, Duration::from_secs(20));
        let unshared = BlobStorage::new();
        for store in [&shared, &unshared] {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                store.transact(|txn| -> io::Result<(WriteBatch, ())> {
                    let mut batch = WriteBatch::new();
                    batch.put(b"torn", b"x");
                    let _ = txn.get(b"torn")?;
                    panic!("the closure fails half way");
                })
            }));
            assert!(caught.is_err(), "the panic did not reach the caller");
            assert_eq!(store.get(b"torn").unwrap(), None, "a panicked batch wrote");
            // the in-transaction mark is gone: the same handle writes and transacts again
            store.put(b"same", b"x").unwrap();
            assert_eq!(increment(store).unwrap(), 1);
        }
        // the lease is gone: another process takes it within its short wait
        wait_ok(spawn("writer", &wal, 0), "the process after the panic");
        assert_eq!(shared.get(b"after").unwrap().as_deref(), Some(&b"x"[..]));
    });
}
