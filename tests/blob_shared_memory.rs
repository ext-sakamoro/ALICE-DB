//! A shared handle holds values as stored (compressed), so its memory follows what `SharedOptions::max_bytes`
//! counts, not the decoded sizes
//!
//! oracle (from the contract of `max_bytes`): the limit counts keys and stored values. If the handle held
//! decoded values, a store of highly compressible values would pass the check and still use far more memory.
//! Twenty values of 1 MiB of zeros (20 MiB decoded, a few KiB stored) are written; a fresh shared handle that
//! loads them must hold under 1 MiB, and a read still returns the full 1 MiB value. The allocator records live
//! bytes; this file holds a single test so nothing else allocates while it measures.

#![cfg(feature = "fs")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use alice_db::blob::{BlobStorage, BlobStorageConfig, SharedOptions};

struct Live;

static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Live {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the caller's layout to the system allocator.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Relaxed);
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
static A: Live = Live;

const VALUES: usize = 20;
const VALUE_LEN: usize = 1 << 20;

#[test]
fn compressible_values_are_held_compressed() {
    let tmp = tempfile::tempdir().unwrap();
    let wal = tmp.path().join("blob.wal");
    let zeros = vec![0_u8; VALUE_LEN];
    {
        let writer =
            BlobStorage::open_shared(&wal, BlobStorageConfig::default(), SharedOptions::default())
                .unwrap();
        for i in 0..VALUES {
            writer.put(format!("v{i:02}").as_bytes(), &zeros).unwrap();
        }
    }
    let before = LIVE.load(Relaxed);
    let reader =
        BlobStorage::open_shared(&wal, BlobStorageConfig::default(), SharedOptions::default())
            .unwrap();
    let held = LIVE.load(Relaxed) - before;
    assert!(
        held < (1 << 20),
        "a handle holding {VALUES} values of {VALUE_LEN} zero bytes uses {held} bytes: the values are held decoded"
    );
    assert_eq!(
        reader.get(b"v07").unwrap().as_deref(),
        Some(zeros.as_slice())
    );
}
