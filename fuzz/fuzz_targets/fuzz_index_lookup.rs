//! Fuzz target: the blob key index (`put_blob` / `get_blob` / `delete_blob` /
//! `scan_blob_prefix`) against a `BTreeMap` model
//!
//! The input is a program, one operation per byte, so that any byte string is
//! a dense sequence of operations and the patterns that matter (the same key
//! written, flushed, written again and flushed) take only a few bytes:
//!
//! - byte 0: the store, odd = a temporary directory with `FlushMode::Append`
//!   and no automatic flush or compaction (several `SSTable`s, the merge of
//!   which is the riskiest code), even = in memory
//! - each following byte `b`: operation `b >> 5` (put, put, get, delete,
//!   prefix scan, flush the memtable into a new `SSTable`, compact all
//!   `SSTable`s, reopen; the last three only on disk) on key `b & 7`, where
//!   keys 0..=6 come from a fixed set of 1- and 2-byte keys and key 7 reads a
//!   raw key from the next bytes (length byte, then the bytes)
//! - a put writes a value unique to its position, so versions always differ
//!
//! After every flush, compaction and reopen, every key ever touched is read
//! back with `get` and the whole store is scanned. Every operation on a valid
//! key must succeed: an `Err` is a defect as much as a disagreement is.
//!
//! What this target can find from an empty corpus within a time budget is not
//! the guarantee: the inputs in `fuzz/regressions/fuzz_index_lookup` (replayed
//! on every run) and the fixed sequences in `tests/blob_merge_semantics.rs`
//! are. The search is for what neither of them lists yet.

#![no_main]

use alice_db::blob::BlobStorageConfig;
use alice_db::blob_sstable::FlushMode;
use alice_db::blob_wal::SyncPolicy;
use alice_db::{AliceDB, StorageConfig};
use libfuzzer_sys::fuzz_target;
use std::collections::{BTreeMap, BTreeSet};

const KEYS: [&[u8]; 7] = [b"a", b"b", b"\x00", b"\xff", b"aa", b"ab", b"ba"];

fn config() -> BlobStorageConfig {
    BlobStorageConfig {
        sync_policy: SyncPolicy::Manual,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
    }
}

/// Every key ever touched read back one by one (a `get` that resolves the
/// merge differently from the scan, a deleted key coming back), then a full
/// scan
fn check_all(db: &AliceDB, model: &BTreeMap<Vec<u8>, Vec<u8>>, touched: &BTreeSet<Vec<u8>>) {
    for k in touched {
        let got = db.get_blob(k).expect("get_blob on a valid key");
        assert_eq!(
            got.as_ref(),
            model.get(k),
            "get_blob disagrees with the model"
        );
    }
    let got = db.scan_blob_prefix(b"").expect("full scan");
    let want: Vec<(Vec<u8>, Vec<u8>)> = model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    assert_eq!(got, want, "full scan disagrees with the model");
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, program)) = data.split_first() else {
        return;
    };
    if program.len() > 64 {
        return;
    }
    let dir = if mode & 1 == 1 {
        Some(tempfile::tempdir().expect("temp dir"))
    } else {
        None
    };
    let open = || match &dir {
        Some(d) => AliceDB::open_with_blob_config(d.path(), config()).expect("open on disk"),
        None => AliceDB::in_memory(StorageConfig::default()).expect("in-memory db"),
    };
    let mut db = open();
    let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    let mut touched: BTreeSet<Vec<u8>> = BTreeSet::new();
    let mut i = 0;
    while i < program.len() {
        let b = program[i];
        let pos = i;
        i += 1;
        let key: Vec<u8> = if b & 7 == 7 {
            let len = usize::from(*program.get(i).unwrap_or(&0) % 8);
            let start = (i + 1).min(program.len());
            let end = (start + len).min(program.len());
            i = end;
            program[start..end].to_vec()
        } else {
            KEYS[usize::from(b & 7)].to_vec()
        };
        let valid = !key.is_empty();
        match b >> 5 {
            0 | 1 if valid => {
                let value = format!("v{pos}").into_bytes();
                db.put_blob(&key, &value).expect("put_blob on a valid key");
                touched.insert(key.clone());
                model.insert(key, value);
            }
            2 if valid => {
                let got = db.get_blob(&key).expect("get_blob on a valid key");
                assert_eq!(
                    got.as_ref(),
                    model.get(&key),
                    "get_blob disagrees with the model"
                );
            }
            3 if valid => {
                db.delete_blob(&key).expect("delete_blob on a valid key");
                touched.insert(key.clone());
                model.remove(&key);
            }
            4 => {
                let got = db.scan_blob_prefix(&key).expect("scan_blob_prefix");
                let want: Vec<(Vec<u8>, Vec<u8>)> = model
                    .iter()
                    .filter(|(k, _)| k.starts_with(&key))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                assert_eq!(got, want, "scan_blob_prefix disagrees with the model");
            }
            5 if dir.is_some() => {
                db.compact_blob_sstable().expect("flush to a new SSTable");
                check_all(&db, &model, &touched);
            }
            6 if dir.is_some() => {
                db.compact_all_blob_sstables()
                    .expect("compact all SSTables");
                check_all(&db, &model, &touched);
            }
            7 if dir.is_some() => {
                db.flush_blobs().expect("sync the WAL");
                drop(db);
                db = open();
                check_all(&db, &model, &touched);
            }
            _ => {}
        }
    }
    check_all(&db, &model, &touched);
});
