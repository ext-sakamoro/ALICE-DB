//! A residual stream that expands far past its points is refused within a bounded allocation
//!
//! oracle (from the format, not read from the code): the decoder is given `4 * points` as its limit, so
//! refusing a small LZMA stream of 64 MiB whose header declares a 1 GiB dictionary keeps the live allocation
//! within twice the limit plus a small constant, under both LZMA residual forms; a decoder that reserves the
//! declared dictionary makes this red even though it still refuses. The allocator records the peak of live
//! bytes; this file holds a single test so nothing else allocates while it measures. Fixture:
//! `tests/data/make_residual_bomb.py` (liblzma).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use alice_db::segment::decompress_residual;

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

const STREAM: &[u8] = include_bytes!("data/residual_alone_dict1g.lzma");
const LIMIT: usize = 1 << 20;
const SLACK: usize = 256 << 10;

#[test]
fn a_bomb_is_refused_within_twice_the_limit() {
    let bound = 2 * LIMIT + SLACK;
    // 0: the earliest LZMA form, 2: unpinned xor LZMA (both `magic · alone stream`)
    for magic in [0_u8, 2] {
        let mut blob = vec![magic];
        blob.extend_from_slice(STREAM);
        let before = LIVE.load(Relaxed);
        PEAK.store(before, Relaxed);
        let out = decompress_residual(&blob, LIMIT);
        let peak = PEAK.load(Relaxed) - before;
        assert_eq!(out, None, "magic {magic}: refused");
        assert!(peak <= bound, "magic {magic}: peak {peak} bytes > {bound}");
    }
}
