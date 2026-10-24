//! A segment size read from a damaged or crafted legacy `data.alice` must not decide how much memory
//! the reader reserves: it is checked against the bytes of the file that follow it before the buffer
//! is allocated, and the query that reaches the segment reports the damage instead.
//! Measured with a capped global allocator: a single request over 16 MiB is refused (not allocated)
//! and recorded, so the test never really reserves the claimed size; peak live bytes are counted too.

#![cfg(feature = "fs")]

use alice_db::StorageEngine;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;

/// The largest single request the allocator grants.
const CAP: usize = 16 * 1024 * 1024;

struct Capped;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn admit(size: usize) -> bool {
    LARGEST.fetch_max(size, Ordering::SeqCst);
    size <= CAP
}

fn note(size: usize) {
    let now = LIVE.fetch_add(size, Ordering::SeqCst) + size;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

// SAFETY: forwards every granted call to the system allocator and only records sizes; a refused
// request returns null, which the caller handles as an allocation failure
unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: same layout the caller passed
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    // forwarded so that a zeroed allocation of a claimed length stays lazy (the pages are not touched)
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: same layout the caller passed
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY: `p` came from `System` with this layout
        unsafe { System.dealloc(p, layout) }
    }
}

#[global_allocator]
static GLOBAL: Capped = Capped;

/// Peak live bytes above the level when `f` starts.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    LARGEST.store(0, Ordering::SeqCst);
    let out = f();
    let largest = LARGEST.load(Ordering::SeqCst);
    assert!(largest <= CAP, "a single request of {largest} bytes");
    (out, PEAK.load(Ordering::SeqCst) - base)
}

/// Slack for the engine's own bookkeeping while it serves one query.
const SLACK: usize = 1 << 20;

/// `index.alice` with one well-formed entry pointing at offset 0 of `data.alice`.
fn index_with_one_entry() -> Vec<u8> {
    let mut b = 1_u64.to_le_bytes().to_vec(); // entry count
    b.extend_from_slice(&1_u64.to_le_bytes()); // id
    b.extend_from_slice(&0_i64.to_le_bytes()); // start time
    b.extend_from_slice(&10_i64.to_le_bytes()); // end time
    b.extend_from_slice(&0_u64.to_le_bytes()); // offset in data.alice
    b.extend_from_slice(&16_u64.to_le_bytes()); // size
    b.extend_from_slice(&1.0_f64.to_le_bytes()); // compression ratio
    b.extend_from_slice(&3_u32.to_le_bytes()); // model name length
    b.extend_from_slice(b"pol");
    b
}

#[test]
fn a_claimed_legacy_segment_size_is_not_allocated() {
    for claimed in [1_u64 << 30, u64::from(u32::MAX), u64::MAX] {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("index.alice"), index_with_one_entry()).unwrap();
        // the size claim, then 8 bytes where the claim says gigabytes follow
        let mut data = claimed.to_le_bytes().to_vec();
        data.extend_from_slice(&[0_u8; 8]);
        std::fs::write(tmp.path().join("data.alice"), &data).unwrap();

        let engine = StorageEngine::open(tmp.path()).expect("the index itself is well formed");
        let (answer, peak) = peak_during(|| engine.query_range(0, 10));
        assert!(
            peak <= data.len() + SLACK,
            "size {claimed}: the query reserved {peak} bytes for a {}-byte data.alice",
            data.len()
        );
        let err = answer.expect_err("a segment whose size runs past the file is reported");
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidData,
            "size {claimed}: {err}"
        );
    }
}
