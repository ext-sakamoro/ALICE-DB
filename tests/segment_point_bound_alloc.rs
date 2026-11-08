//! The input the segment fuzzer found (a legacy `data.alice` entry: a raw payload that does not
//! decompress, claiming 4 289 895 731 points) is refused without allocating anything near its
//! point count. Measured with a counting allocator, alone in this binary.

#![cfg(feature = "fs")]
#![allow(unsafe_code, reason = "a counting allocator around the system one")]

use alice_db::{AliceDB, ResidualPolicy, StorageConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// The fuzz target's legacy store around `bytes` (fuzz/fuzz_targets/fuzz_segment_bytes.rs)
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
    let bytes = include_bytes!("fixtures/fuzz/segment_bytes_point_count_oom.bin");
    let tmp = tempfile::tempdir().unwrap();
    legacy_store(tmp.path(), bytes);
    LARGEST.store(0, Ordering::Relaxed);
    let db = AliceDB::with_config(StorageConfig {
        data_dir: tmp.path().to_path_buf(),
        enable_wal: false,
        use_mmap: false,
        residual_policy: ResidualPolicy::AcceptUnpinned,
        ..StorageConfig::default()
    })
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
