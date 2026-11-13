//! An `SSTable` whose header claims far more records than its body holds is refused without reserving memory for
//! the claim
//!
//! oracle (from the format, not read from the code): a record is at least 13 bytes (9-byte header, empty key and
//! value, 4-byte CRC), so a body of `n` bytes holds at most `n / 13` records and a reader never needs to reserve
//! more. The file here is a v1 `SSTable` of one record (under 100 bytes) whose header claims 2^40 records; opening it
//! must fail as corrupt with a peak live allocation far below the claim. A reader that reserves the claimed count
//! asks the allocator for terabytes (recorded here even where the OS overcommits, an abort where it does not).
//! The allocator records the peak of live bytes; this file holds a single test so nothing else allocates while it
//! measures. The same holds for a stored segment, whose stream states its byte length before the bytes.

#![cfg(feature = "fs")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::ErrorKind;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use alice_db::blob_sstable::BlobSstable;
use alice_db::DataSegment;

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

const CLAIMED: u64 = 1 << 40;
const BOUND: usize = 1 << 20;

fn v1_file(num_records: u64) -> Vec<u8> {
    let mut record = Vec::new();
    record.extend_from_slice(&1_u32.to_le_bytes());
    record.extend_from_slice(&1_u32.to_le_bytes());
    record.push(0x00);
    record.extend_from_slice(b"k");
    record.extend_from_slice(b"v");
    let mut h = crc32fast::Hasher::new();
    h.update(&record);
    record.extend_from_slice(&h.finalize().to_le_bytes());
    let mut f = Vec::new();
    f.extend_from_slice(b"ALICEBBS");
    f.extend_from_slice(&1_u32.to_le_bytes());
    f.extend_from_slice(&num_records.to_le_bytes());
    f.extend_from_slice(&0_u32.to_le_bytes());
    f.extend_from_slice(&record);
    f.extend_from_slice(&(record.len() as u64).to_le_bytes());
    f.extend_from_slice(b"ALICEEND");
    f
}

#[test]
fn a_claimed_record_count_is_not_reserved() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blob.sst");
    std::fs::write(&path, v1_file(CLAIMED)).unwrap();

    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let err = BlobSstable::open(&path).expect_err("the claim is refused");
    let peak = PEAK.load(Relaxed) - before;

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(
        err.to_string()
            .contains("header claims 1099511627776 records"),
        "{err}"
    );
    assert!(
        peak < BOUND,
        "opening a <100-byte file reserved {peak} bytes (bound {BOUND})"
    );

    // A stored segment states its length (u64) before its bytes: a stream claiming 2^62 bytes with none behind
    // it is truncated, and reading it reserves no more than what arrived.
    let mut claim = Vec::from((1_u64 << 62).to_le_bytes());
    claim.extend_from_slice(b"short");
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let err =
        DataSegment::read_from(&mut claim.as_slice()).expect_err("the claimed length is refused");
    let peak = PEAK.load(Relaxed) - before;
    assert_eq!(err.kind(), ErrorKind::UnexpectedEof, "{err}");
    assert!(
        peak < BOUND,
        "reading a 13-byte segment stream reserved {peak} bytes (bound {BOUND})"
    );

    // Control: the same file with the true count opens.
    std::fs::write(&path, v1_file(1)).unwrap();
    assert_eq!(BlobSstable::open(&path).unwrap().unwrap().len(), 1);
}
