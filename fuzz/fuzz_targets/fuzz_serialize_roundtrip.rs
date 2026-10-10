//! Fuzz target: the in-memory snapshot (`AliceDB::to_bytes` / `from_bytes`)
//!
//! Two properties, both through the public API:
//! - arbitrary bytes given to `from_bytes` return `Ok` or `Err` without
//!   panicking (a snapshot can come from anywhere)
//! - a database built from arbitrary points and blobs survives `to_bytes` ->
//!   `from_bytes`: the same points scan back and every blob reads back

#![no_main]

use alice_db::{AliceDB, StorageConfig};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    raw: Vec<u8>,
    points: Vec<(i16, u32)>,
    blobs: Vec<(Vec<u8>, Vec<u8>)>,
}

fuzz_target!(|input: Input| {
    if input.raw.len() > 64 * 1024 || input.points.len() > 512 || input.blobs.len() > 64 {
        return;
    }
    let _ = AliceDB::from_bytes(StorageConfig::default(), &input.raw);

    let db = AliceDB::in_memory(StorageConfig::default()).expect("in-memory db");
    for &(t, bits) in &input.points {
        let v = f32::from_bits(bits);
        if v.is_finite() {
            let _ = db.put(i64::from(t), v);
        }
    }
    for (k, v) in &input.blobs {
        if !k.is_empty() && k.len() <= 256 && v.len() <= 4096 {
            let _ = db.put_blob(k, v);
        }
    }
    let Ok(bytes) = db.to_bytes() else { return };
    let back =
        AliceDB::from_bytes(StorageConfig::default(), &bytes).expect("own snapshot restores");
    assert_eq!(
        db.scan(i64::MIN, i64::MAX).ok(),
        back.scan(i64::MIN, i64::MAX).ok(),
        "points changed across to_bytes / from_bytes"
    );
    assert_eq!(
        db.scan_blob_prefix(b"").ok(),
        back.scan_blob_prefix(b"").ok(),
        "blobs changed across to_bytes / from_bytes"
    );
});
