//! Decoding a stored segment allocates in proportion to the bytes it is given, never to a length a damaged
//! header claims: a corrupt file cannot make the reader reserve gigabytes. Measured with a counting global
//! allocator (peak live bytes during the decode).

use alice_db::segment::DataSegment;
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Peak bytes allocated above the starting level while `f` runs.
fn peak_during(f: impl FnOnce()) -> usize {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    f();
    PEAK.load(Ordering::SeqCst) - base
}

#[test]
fn decoding_never_allocates_beyond_what_the_bytes_hold() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let legacy = std::fs::read(root.join("store_db_0_5_1/seg_1.rkyv")).unwrap();

    // a valid format 0 file and the same segment in format 1: the decode costs a small multiple of the file
    let seg = DataSegment::from_stored_bytes(&legacy).unwrap();
    let current = seg.to_rkyv_bytes().unwrap();
    for (what, bytes) in [("format 0", &legacy), ("format 1", &current)] {
        let peak = peak_during(|| {
            DataSegment::from_stored_bytes(bytes).unwrap();
        });
        assert!(
            peak <= 16 * bytes.len() + 4096,
            "{what}: {peak} bytes for a {}-byte file",
            bytes.len()
        );
    }

    // format 0 with the residual's length raised to 2^31 - 1 and every vector length raised alike: refused,
    // with no allocation of that size
    let root_at = legacy.len() - 104;
    for field in [root_at + 88 + 4 + 4, root_at + 4 + 4] {
        let mut b = legacy.clone();
        b[field..field + 4].copy_from_slice(&0x7fff_ffff_u32.to_le_bytes());
        let peak = peak_during(|| {
            assert!(DataSegment::from_stored_bytes(&b).is_err());
        });
        assert!(
            peak <= 64 * 1024,
            "format 0, length at {field}: peak {peak}"
        );
    }

    // format 1 whose archive claims a huge vector: refused, the copy is bounded by the file
    let mut b = current.clone();
    let n = b.len();
    for i in 16..n {
        let mut c = b.clone();
        c[i] = 0xff;
        let peak = peak_during(|| {
            let _ = DataSegment::from_stored_bytes(&c);
        });
        assert!(
            peak <= 16 * n + 4096,
            "format 1, byte {i} set to 0xff: peak {peak}"
        );
    }
    b.truncate(16);
    assert!(DataSegment::from_stored_bytes(&b).is_err());
}

#[test]
fn a_newer_or_unknown_format_is_refused() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let legacy = std::fs::read(root.join("store_db_0_5_1/seg_1.rkyv")).unwrap();
    let mut current = DataSegment::from_stored_bytes(&legacy)
        .unwrap()
        .to_rkyv_bytes()
        .unwrap();
    assert_eq!(&current[..8], b"ALICESEG");
    assert_eq!(
        u32::from_le_bytes(current[8..12].try_into().unwrap()),
        alice_db::segment::SEGMENT_FORMAT
    );
    current[8] = 2;
    let err = DataSegment::from_stored_bytes(&current).unwrap_err();
    assert!(err.to_string().contains("format 2"), "{err}");
    current[8] = 1;
    current[12] = 1;
    assert!(
        DataSegment::from_stored_bytes(&current).is_err(),
        "a reserved byte that is not zero"
    );
}
