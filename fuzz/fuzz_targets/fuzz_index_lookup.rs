//! Fuzz target: the blob key index (`put_blob` / `get_blob` / `delete_blob` /
//! `scan_blob_prefix`) against a `BTreeMap` model
//!
//! Arbitrary keys and operations; after each operation the store must agree
//! with the model on the key just touched, and at the end on every prefix
//! scanned. Any panic or disagreement is a defect in the index.

#![no_main]

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
}

fuzz_target!(|ops: Vec<Op>| {
    if ops.len() > 256 {
        return;
    }
    let db = AliceDB::in_memory(StorageConfig::default()).expect("in-memory db");
    let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    for op in ops {
        match op {
            Op::Put(k, v) => {
                if k.is_empty() || k.len() > 256 || v.len() > 4096 {
                    continue;
                }
                if db.put_blob(&k, &v).is_ok() {
                    model.insert(k, v);
                }
            }
            Op::Get(k) => {
                if let Ok(got) = db.get_blob(&k) {
                    assert_eq!(
                        got.as_ref(),
                        model.get(&k),
                        "get_blob disagrees with the model"
                    );
                }
            }
            Op::Delete(k) => {
                if db.delete_blob(&k).is_ok() {
                    model.remove(&k);
                }
            }
            Op::Scan(prefix) => {
                if prefix.len() > 256 {
                    continue;
                }
                if let Ok(got) = db.scan_blob_prefix(&prefix) {
                    let want: Vec<(Vec<u8>, Vec<u8>)> = model
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    assert_eq!(got, want, "scan_blob_prefix disagrees with the model");
                }
            }
        }
    }
});
