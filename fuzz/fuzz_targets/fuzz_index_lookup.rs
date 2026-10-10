//! Fuzz target: the blob key index (`put_blob` / `get_blob` / `delete_blob` /
//! `scan_blob_prefix`) against a `BTreeMap` model
//!
//! Two stores, chosen by the first input field:
//! - in memory: the memtable alone
//! - a temporary directory with `FlushMode::Append` and no automatic flush or
//!   compaction, where the input also flushes the memtable into a new
//!   `SSTable`, compacts all `SSTable`s, and reopens the store. Reads then go
//!   through the k-way merge of several `SSTable`s with the newest value (or
//!   delete) of a key winning, which is the code most likely to resurrect a
//!   deleted key or return a stale value
//!
//! Every operation on a valid key (1..=256 bytes, value <= 4096) must succeed:
//! an `Err` is a defect as much as a disagreement with the model is.

#![no_main]

use alice_db::blob::BlobStorageConfig;
use alice_db::blob_sstable::FlushMode;
use alice_db::blob_wal::SyncPolicy;
use alice_db::{AliceDB, StorageConfig};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;

#[derive(Debug, Arbitrary)]
enum Op {
    Put(Vec<u8>, Vec<u8>),
    Get(Vec<u8>),
    Delete(Vec<u8>),
    Scan(Vec<u8>),
    /// file store only: memtable -> new SSTable
    Flush,
    /// file store only: merge every SSTable into one
    Compact,
    /// file store only: close and open again
    Reopen,
}

#[derive(Debug, Arbitrary)]
struct Input {
    on_disk: bool,
    ops: Vec<Op>,
}

fn valid_key(k: &[u8]) -> bool {
    !k.is_empty() && k.len() <= 256
}

fn config() -> BlobStorageConfig {
    BlobStorageConfig {
        sync_policy: SyncPolicy::Manual,
        wal_flush_threshold_bytes: u64::MAX,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
    }
}

fn check_all(db: &AliceDB, model: &BTreeMap<Vec<u8>, Vec<u8>>) {
    let got = db.scan_blob_prefix(b"").expect("full scan");
    let want: Vec<(Vec<u8>, Vec<u8>)> = model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    assert_eq!(got, want, "full scan disagrees with the model");
}

fuzz_target!(|input: Input| {
    if input.ops.len() > 128 {
        return;
    }
    let dir = if input.on_disk {
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
    for op in input.ops {
        match op {
            Op::Put(k, v) => {
                if !valid_key(&k) || v.len() > 4096 {
                    continue;
                }
                db.put_blob(&k, &v).expect("put_blob on a valid key");
                model.insert(k, v);
            }
            Op::Get(k) => {
                if !valid_key(&k) {
                    continue;
                }
                let got = db.get_blob(&k).expect("get_blob on a valid key");
                assert_eq!(
                    got.as_ref(),
                    model.get(&k),
                    "get_blob disagrees with the model"
                );
            }
            Op::Delete(k) => {
                if !valid_key(&k) {
                    continue;
                }
                db.delete_blob(&k).expect("delete_blob on a valid key");
                model.remove(&k);
            }
            Op::Scan(prefix) => {
                if prefix.len() > 256 {
                    continue;
                }
                let got = db.scan_blob_prefix(&prefix).expect("scan_blob_prefix");
                let want: Vec<(Vec<u8>, Vec<u8>)> = model
                    .iter()
                    .filter(|(k, _)| k.starts_with(&prefix))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                assert_eq!(got, want, "scan_blob_prefix disagrees with the model");
            }
            Op::Flush if dir.is_some() => {
                db.compact_blob_sstable().expect("flush to a new SSTable");
                check_all(&db, &model);
            }
            Op::Compact if dir.is_some() => {
                db.compact_all_blob_sstables()
                    .expect("compact all SSTables");
                check_all(&db, &model);
            }
            Op::Reopen if dir.is_some() => {
                db.flush_blobs().expect("sync the WAL");
                drop(db);
                db = open();
                check_all(&db, &model);
            }
            Op::Flush | Op::Compact | Op::Reopen => {}
        }
    }
    check_all(&db, &model);
});
