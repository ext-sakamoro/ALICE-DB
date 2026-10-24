//! Bytes alice-db wrote with bincode, read the only way the crate reads them: from a legacy
//! `data.alice` file, which the storage engine rewrites as a stored `seg_1.rkyv` file.

use alice_db::segment::DataSegment;
use alice_db::{AliceDB, ResidualPolicy, StorageConfig};

/// The segment in `bytes` (one legacy `data.alice` entry), as the storage engine rewrites it
pub fn read_through_data_file(bytes: &[u8]) -> DataSegment {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let start = i64::from_le_bytes(bytes[0..8].try_into().expect("start_time"));
    let end = i64::from_le_bytes(bytes[8..16].try_into().expect("end_time"));
    let mut index = Vec::new();
    index.extend_from_slice(&1u64.to_le_bytes());
    index.extend_from_slice(&1u64.to_le_bytes()); // id
    index.extend_from_slice(&start.to_le_bytes());
    index.extend_from_slice(&end.to_le_bytes());
    index.extend_from_slice(&0u64.to_le_bytes()); // offset in data.alice
    index.extend_from_slice(&(bytes.len() as u64 + 8).to_le_bytes()); // size
    index.extend_from_slice(&1.0f64.to_le_bytes()); // compression ratio
    index.extend_from_slice(&6u32.to_le_bytes());
    index.extend_from_slice(b"legacy");
    std::fs::write(dir.join("index.alice"), index).expect("index");
    let mut data = (bytes.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(bytes);
    std::fs::write(dir.join("data.alice"), data).expect("data");

    let db = AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_wal: false,
        use_mmap: false,
        residual_policy: ResidualPolicy::AcceptUnpinned,
        ..StorageConfig::default()
    })
    .expect("open the legacy store");
    // reading the range loads the segment from data.alice and rewrites it; what the read
    // returns (a residual may be refused) does not matter here
    let _ = db.scan(start, end);
    drop(db);
    let stored = std::fs::read(dir.join("seg_1.rkyv"))
        .expect("the legacy segment was rewritten as a stored segment");
    DataSegment::from_stored_bytes(&stored).expect("the rewritten segment decodes")
}
