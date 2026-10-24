//! A length read from a damaged or crafted file must not decide how much memory the reader reserves:
//! the WAL record header (key and value lengths) and the index entry's model-name length are checked
//! against the bytes that remain before anything is allocated for them. Recovery of such a WAL ends on
//! the cut and a second recovery finds nothing to cut (no loop over the same bad record).
//! Measured with a capped global allocator: the peak of live bytes while the reader runs, and the
//! largest single request, which may not exceed 16 MiB (a request above it is refused, not made).

#![cfg(feature = "fs")]

use alice_db::blob_wal::{BlobWal, WalRecord, WalTail};
use alice_db::StorageEngine;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tempfile::TempDir;

/// Largest single request the readers may make in this binary. A request above it is refused (null),
/// so a reader that sizes a buffer by a claimed length stops the process with "memory allocation of N
/// bytes failed" before anything that large is reserved, the same on every OS.
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

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;

/// The tests of this binary measure one process-wide counter, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// Peak bytes allocated above the starting level while `f` runs.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    LARGEST.store(0, Ordering::SeqCst);
    let out = f();
    let largest = LARGEST.load(Ordering::SeqCst);
    assert!(largest <= CAP, "a single request of {largest} bytes");
    (out, PEAK.load(Ordering::SeqCst) - base)
}

/// Fixed cost of a reader apart from what it decodes (buffered reader, records, paths).
const SLACK: usize = 256 * 1024;

/// A length that no file in these tests holds: 1 GiB, far above the cap.
const CLAIMED: u32 = 0x4000_0000;

/// One WAL record frame as the writer lays it out: type, key length, value length, value kind, key,
/// value, CRC-32 of everything before it.
fn frame(record_type: u8, key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut b = vec![record_type];
    b.extend_from_slice(&u32::try_from(key.len()).unwrap().to_le_bytes());
    b.extend_from_slice(&u32::try_from(value.len()).unwrap().to_le_bytes());
    b.push(0x00);
    b.extend_from_slice(key);
    b.extend_from_slice(value);
    let crc = crc32(&b);
    b.extend_from_slice(&crc.to_le_bytes());
    b
}

/// CRC-32 (IEEE, reflected 0xEDB88320), written here so the frame does not come from the crate's writer.
fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xffff_ffff_u32;
    for &b in bytes {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0xedb8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// A good record followed by a header that claims `key_len` / `value_len` and ends right there.
fn wal_with_claimed_tail(key_len: u32, value_len: u32) -> (Vec<u8>, usize) {
    let mut b = frame(0x01, b"good", b"ok");
    let good_end = b.len();
    b.push(0x01);
    b.extend_from_slice(&key_len.to_le_bytes());
    b.extend_from_slice(&value_len.to_le_bytes());
    b.push(0x00);
    b.extend_from_slice(b"xyz");
    (b, good_end)
}

#[test]
fn crc32_matches_the_reference_value() {
    // the check value of CRC-32/ISO-HDLC
    assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
}

#[test]
fn a_claimed_record_length_is_not_allocated_and_recovery_ends_on_the_cut() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut compared = 0;
    for (what, key_len, value_len) in [
        ("key length", CLAIMED, 0),
        ("value length", 3, CLAIMED),
        ("both lengths", CLAIMED, CLAIMED),
    ] {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("blob.wal");
        let (bytes, good_end) = wal_with_claimed_tail(key_len, value_len);
        std::fs::write(&path, &bytes).unwrap();

        let wal = BlobWal::open(&path).unwrap();
        let ((records, cut), peak) = peak_during(|| wal.recover().unwrap());
        assert!(
            peak <= bytes.len() + SLACK,
            "{what}: recovery reserved {peak} bytes for a {}-byte WAL",
            bytes.len()
        );
        assert_eq!(records.len(), 1, "{what}: the good record is replayed");
        assert!(matches!(&records[0], WalRecord::Put(k, _) if k.as_slice() == b"good"));
        let cut = cut.expect("the claimed tail is cut");
        assert_eq!(cut.offset, good_end as u64, "{what}");
        assert_eq!(cut.tail, WalTail::Torn, "{what}");
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            good_end as u64,
            "{what}"
        );
        drop(wal);

        // the cut is final: a second open replays the same record and cuts nothing
        let wal = BlobWal::open(&path).unwrap();
        let ((again, second_cut), peak) = peak_during(|| wal.recover().unwrap());
        assert!(
            peak <= good_end + SLACK,
            "{what}: second recovery reserved {peak} bytes"
        );
        assert_eq!(again.len(), 1, "{what}");
        assert!(second_cut.is_none(), "{what}: a second recovery cut again");

        // replay alone does not reserve the claimed length either
        let (bytes2, _) = wal_with_claimed_tail(key_len, value_len);
        std::fs::write(&path, &bytes2).unwrap();
        drop(wal);
        let wal = BlobWal::open(&path).unwrap();
        let (replayed, peak) = peak_during(|| wal.replay().unwrap());
        assert!(
            peak <= bytes2.len() + SLACK,
            "{what}: replay reserved {peak} bytes"
        );
        assert_eq!(replayed.len(), 1, "{what}");
        compared += 1;
    }
    assert_eq!(compared, 3);
}

/// `index.alice` with one entry whose model-name length is `model_len` and whose bytes end after it.
fn index_with_claimed_model_name(model_len: u32) -> Vec<u8> {
    let mut b = 1_u64.to_le_bytes().to_vec(); // entry count
    b.extend_from_slice(&1_u64.to_le_bytes()); // id
    b.extend_from_slice(&0_i64.to_le_bytes()); // start time
    b.extend_from_slice(&10_i64.to_le_bytes()); // end time
    b.extend_from_slice(&0_u64.to_le_bytes()); // offset
    b.extend_from_slice(&16_u64.to_le_bytes()); // size
    b.extend_from_slice(&1.0_f64.to_le_bytes()); // compression ratio
    b.extend_from_slice(&model_len.to_le_bytes());
    b.extend_from_slice(b"pol");
    b
}

#[test]
fn a_claimed_index_name_length_is_not_allocated() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tmp = TempDir::new().unwrap();
    let index = index_with_claimed_model_name(CLAIMED);
    std::fs::write(tmp.path().join("index.alice"), &index).unwrap();

    let (opened, peak) = peak_during(|| StorageEngine::open(tmp.path()).map(drop));
    assert!(
        peak <= index.len() + 4 * SLACK,
        "opening reserved {peak} bytes for a {}-byte index",
        index.len()
    );
    // whatever the engine decides about the damaged index, it says so instead of reading past its end
    assert!(
        opened.is_err(),
        "an index entry cut inside its model name is reported"
    );
}
