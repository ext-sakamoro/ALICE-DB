//! Fuzz target: arbitrary bytes as a segment, through both readers of `DataSegment` bytes
//!
//! - `DataSegment::from_bytes` (alice_serial format 1) returns `Ok` or `Err` without panicking,
//!   and a segment it accepts is written back to the same bytes
//! - the same bytes as one entry of a legacy `data.alice` file (the bincode layout older releases
//!   wrote), read through the store: the read returns `Ok` or `Err` without panicking or sizing
//!   anything by a length the input cannot hold, and when the entry was accepted it was rewritten
//!   as a stored segment that decodes

#![no_main]

use alice_db::segment::DataSegment;
use alice_db::{AliceDB, ResidualPolicy, StorageConfig};
use libfuzzer_sys::fuzz_target;

fn legacy_store(dir: &std::path::Path, bytes: &[u8]) {
    let mut index = Vec::new();
    index.extend_from_slice(&1u64.to_le_bytes());
    index.extend_from_slice(&1u64.to_le_bytes());
    index.extend_from_slice(&0i64.to_le_bytes());
    index.extend_from_slice(&63i64.to_le_bytes());
    index.extend_from_slice(&0u64.to_le_bytes());
    index.extend_from_slice(&(bytes.len() as u64 + 8).to_le_bytes());
    index.extend_from_slice(&1.0f64.to_le_bytes());
    index.extend_from_slice(&6u32.to_le_bytes());
    index.extend_from_slice(b"legacy");
    std::fs::write(dir.join("index.alice"), index).expect("index");
    let mut data = (bytes.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(bytes);
    std::fs::write(dir.join("data.alice"), data).expect("data");
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    if let Ok(seg) = DataSegment::from_bytes(data) {
        assert_eq!(seg.to_bytes().expect("an accepted segment encodes"), data);
    }

    let tmp = tempfile::tempdir().expect("temp dir");
    legacy_store(tmp.path(), data);
    let db = AliceDB::with_config(StorageConfig {
        data_dir: tmp.path().to_path_buf(),
        enable_wal: false,
        use_mmap: false,
        residual_policy: ResidualPolicy::AcceptUnpinned,
        ..StorageConfig::default()
    })
    .expect("open");
    let _ = db.scan(0, 63);
    drop(db);
    if let Ok(stored) = std::fs::read(tmp.path().join("seg_1.rkyv")) {
        DataSegment::from_stored_bytes(&stored).expect("a rewritten legacy segment decodes");
    }
});
