//! A torn or corrupt tail can hold any bytes where the lengths are. They are not
//! checked until the checksum at the end, so reading has to stay within the bytes
//! the file really has. Measured with a counting allocator, alone in this binary.
#![cfg(feature = "fs")]
#![allow(unsafe_code, reason = "a counting allocator around the system one")]

use alice_db::blob::BlobStorage;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;

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

const LIMIT: usize = 1024 * 1024;

fn tail(key_len: u32, value_len: u32) -> Vec<u8> {
    let mut bytes = vec![1_u8];
    bytes.extend_from_slice(&key_len.to_le_bytes());
    bytes.extend_from_slice(&value_len.to_le_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&[0xAB; 64]);
    bytes
}

#[test]
fn a_garbage_length_in_the_tail_reserves_nothing_near_it() {
    let cases = [
        ("key", tail(u32::MAX, 0)),
        ("value", tail(1, u32::MAX)),
        ("both", tail(u32::MAX, u32::MAX)),
        ("large", tail(1 << 30, 1 << 30)),
    ];
    for (name, garbage) in cases {
        for read_only in [false, true] {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("blob.wal");
            {
                let store = BlobStorage::open(&path).unwrap();
                store.put(b"k", b"v").unwrap();
            }
            let mut bytes = std::fs::read(&path).unwrap();
            let good = bytes.len();
            bytes.extend_from_slice(&garbage);
            std::fs::write(&path, &bytes).unwrap();

            LARGEST.store(0, Ordering::Relaxed);
            let store = if read_only {
                BlobStorage::open_existing_read_only(&path).unwrap()
            } else {
                BlobStorage::open(&path).unwrap()
            };
            let largest = LARGEST.load(Ordering::Relaxed);
            assert!(largest > 0, "nothing was measured");
            assert!(
                largest < LIMIT,
                "{name} read_only={read_only}: {largest} bytes"
            );
            assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()), "{name}");
            if !read_only {
                drop(store);
                assert_eq!(
                    std::fs::metadata(&path).unwrap().len(),
                    good as u64,
                    "{name}"
                );
            }
        }
    }
}

/// Real records of any size still read back whole.
#[test]
fn a_record_larger_than_the_limit_above_is_still_read() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let value: Vec<u8> = (0..4 * LIMIT)
        .map(|i| u8::try_from(i * 31 % 251).unwrap())
        .collect();
    {
        let store = BlobStorage::open(&path).unwrap();
        store.put(b"big", &value).unwrap();
    }
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(store.get(b"big").unwrap(), Some(value));
}
