//! Reopen after a real crash: a child process writes through `AliceDB` with
//! the WAL on, reports that every `put` returned, and is killed (`SIGKILL` on
//! Unix, `TerminateProcess` on Windows, both via `Child::kill`) without
//! `close` or `Drop` running. The parent then reopens the directory with the
//! same configuration and reads every value back bit for bit.
//!
//! The in-process tests simulate the crash with `mem::forget`, which cannot
//! release the advisory lock the way a dead process does, so they cannot reach
//! this path through `AliceDB`. Here the operating system releases the lock and
//! the open file handles exactly as it does after a real crash.
//!
//! Power loss is not covered. A killed process leaves its writes in the
//! operating system's page cache, which survives `SIGKILL` and
//! `TerminateProcess`, so removing an `fsync` (`sync_all`) leaves this test
//! green. Durability across power loss needs a different instrument (a file
//! wrapper that records where `sync_all` is called, or a fault-injecting
//! block device)
//!
//! The child is this same test binary, re-entered through `crash_child_entry`
//! with `ALICE_DB_CRASH_CHILD` set; without that variable the entry returns at
//! once.
#![cfg(feature = "fs")]

use alice_db::{AliceDB, FitConfig, StorageConfig};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const CHILD_ENV: &str = "ALICE_DB_CRASH_CHILD";
const MODE_ENV: &str = "ALICE_DB_CRASH_MODE";
const READY: &str = "alice-db-crash-child-ready";
const N: i64 = 500;

/// A series with no closed form a model fits exactly, so the lossless residual
/// is exercised; integer arithmetic only, the same on every target
fn value(i: i64) -> f32 {
    let r = (i * i + 7 * i) % 97;
    #[allow(
        clippy::cast_precision_loss,
        reason = "r < 97 and i < 500 are exact in f32"
    )]
    {
        (r as f32).mul_add(0.125, i as f32) - 3.0
    }
}

fn config(dir: &Path, encrypted: bool) -> StorageConfig {
    StorageConfig {
        data_dir: dir.to_path_buf(),
        // large enough that nothing is flushed before the crash: every value
        // the parent reads back comes from the WAL replay
        memtable_capacity: 100_000,
        enable_wal: true,
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        #[cfg(feature = "crypto")]
        encryption_key: encrypted.then(|| alice_crypto::Key::from_bytes([0x5a; 32])),
        use_mmap: !encrypted,
        ..StorageConfig::default()
    }
}

#[test]
fn crash_child_entry() {
    let Some(dir) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    let encrypted = std::env::var(MODE_ENV).as_deref() == Ok("encrypted");
    let db = AliceDB::with_config(config(&PathBuf::from(dir), encrypted)).unwrap();
    for i in 0..N {
        db.put(i, value(i)).unwrap();
    }
    println!("{READY}");
    // Wait to be killed. The parent's kill is the only way out; a timeout
    // here would let the child exit normally and run `Drop`, which is not a
    // crash
    loop {
        std::thread::park();
    }
}

fn crash_then_reopen(encrypted: bool) {
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "crash_child_entry",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, dir.path())
        .env(MODE_ENV, if encrypted { "encrypted" } else { "plain" })
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();

    // libtest prints `test crash_child_entry ... ` without a newline, so the
    // marker ends a line rather than being one. A reader thread with a
    // deadline keeps a stuck child from hanging the suite
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let ready = BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
            .any(|line| line.trim_end().ends_with(READY));
        let _ = tx.send(ready);
    });
    let ready = rx.recv_timeout(Duration::from_secs(300));
    if ready != Ok(true) {
        let _ = child.kill();
        panic!("the child did not report every write: {ready:?}");
    }

    child.kill().unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success(), "the child was not killed: {status}");

    let db = AliceDB::with_config(config(dir.path(), encrypted)).unwrap();
    let got = db.scan(0, N - 1).unwrap();
    assert_eq!(
        got.len(),
        usize::try_from(N).unwrap(),
        "values lost in the crash"
    );
    for (i, &(t, v)) in got.iter().enumerate() {
        let i = i64::try_from(i).unwrap();
        assert_eq!(t, i, "timestamp out of order after replay");
        assert_eq!(
            v.to_bits(),
            value(i).to_bits(),
            "value at {i} changed across the crash: {v} != {}",
            value(i)
        );
    }
    db.close().unwrap();
}

#[test]
fn reopen_after_the_writer_process_is_killed() {
    crash_then_reopen(false);
}

#[cfg(feature = "crypto")]
#[test]
fn reopen_after_the_encrypted_writer_process_is_killed() {
    crash_then_reopen(true);
}
