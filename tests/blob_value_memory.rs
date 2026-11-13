//! A stored compressed value far larger than the reading store's limit is refused within a bounded allocation
//!
//! oracle (from the contract, not read from the code): a 64 MiB value of zeros (a few hundred bytes compressed)
//! read back by a store whose `max_value_bytes` is 1 MiB is refused while the live allocation stays within twice
//! the limit plus a small constant; reading the whole value would allocate 64 MiB. The allocator records the peak
//! of live bytes; this file holds a single test so nothing else allocates while it measures.

#![cfg(feature = "fs")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use alice_db::blob::{BlobStorage, BlobStorageConfig};

struct Peak;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the caller's layout to the system allocator.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
}

#[global_allocator]
static A: Peak = Peak;

const LIMIT: usize = 1 << 20;
const SLACK: usize = 256 << 10;

#[test]
fn a_stored_value_past_the_limit_is_refused_within_twice_the_limit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let wal = tmp.path().join("blob.wal");
    {
        let store = BlobStorage::open_with_config(&wal, BlobStorageConfig::default()).unwrap();
        store.put(b"big", &vec![0u8; 64 << 20]).unwrap();
    }
    let store = BlobStorage::open_with_config(
        &wal,
        BlobStorageConfig::default().with_max_value_bytes(LIMIT),
    )
    .unwrap();
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let r = store.get(b"big");
    let peak = PEAK.load(Relaxed) - before;
    assert!(r.is_err(), "a value past the limit was read");
    assert!(peak <= 2 * LIMIT + SLACK, "peak {peak} bytes");
}
