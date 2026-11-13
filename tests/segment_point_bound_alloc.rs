//! The input the segment fuzzer found (a legacy `data.alice` entry: a raw payload that does not
//! decompress, claiming 4 289 895 731 points) is refused without allocating anything near its
//! point count. Measured with a counting allocator, alone in this binary.

#![cfg(feature = "fs")]
#![allow(unsafe_code, reason = "a counting allocator around the system one")]

use alice_db::{AliceDB, ResidualPolicy, StorageConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The count is of the whole process: a test allocating beside the measurement would be in it.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

struct Counting;
static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call goes to `System` with the arguments it was given.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The fuzz target's legacy store around `bytes` (`fuzz/fuzz_targets/fuzz_segment_bytes.rs`)
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
    std::fs::write(dir.join("index.alice"), index).unwrap();
    let mut data = (bytes.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(bytes);
    std::fs::write(dir.join("data.alice"), data).unwrap();
}

#[test]
fn the_fuzz_input_reserves_nothing_near_its_point_count() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let bytes = include_bytes!("fixtures/fuzz/segment_bytes_point_count_oom.bin");
    let tmp = tempfile::tempdir().unwrap();
    legacy_store(tmp.path(), bytes);
    LARGEST.store(0, Ordering::Relaxed);
    let db = AliceDB::with_config(
        StorageConfig::default()
            .with_data_dir(tmp.path().to_path_buf())
            .with_enable_wal(false)
            .with_use_mmap(false)
            .with_residual_policy(ResidualPolicy::AcceptUnpinned),
    )
    .unwrap();
    let scanned = db.scan(0, 63);
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(largest > 0, "nothing was measured");
    assert!(largest < 64 << 20, "{largest} bytes");
    assert!(scanned.is_err(), "the entry was accepted: {scanned:?}");
    assert!(
        !tmp.path().join("seg_1.rkyv").exists(),
        "a refused entry was rewritten"
    );
}

/// A payload that decompresses to far more than its points stops at its points: its stream says
/// 64 MiB, the segment says 4 samples.
#[test]
fn a_payload_longer_than_its_points_is_read_no_further() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut compressed = Vec::new();
    lzma_rs::lzma_compress(
        &mut std::io::Cursor::new(vec![0u8; 64 << 20]),
        &mut compressed,
    )
    .unwrap();
    let model = alice_db::model::ModelType::RawLzma {
        compressed_data: compressed,
        dtype: alice_db::model::DataType::Float32,
        original_size: 16,
    };
    let segment = alice_db::segment::DataSegment::new(1, 0, 3, model, 4, 16, 0);
    let bytes = segment.to_bytes().unwrap();
    LARGEST.store(0, Ordering::Relaxed);
    let read = alice_db::segment::DataSegment::from_bytes(&bytes);
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(read.is_err(), "a payload of 64 MiB was taken for 4 samples");
    assert!(largest > 0, "nothing was measured");
    // the payload itself is copied out of the bytes (a growing vector may hold twice it); nothing
    // is sized by the 64 MiB the stream claims
    let payload_bound = 2 * bytes.len() + (64 << 10);
    assert!(
        largest <= payload_bound,
        "{largest} bytes, payload bound {payload_bound}"
    );
    assert!(payload_bound < 64 << 20);
}
