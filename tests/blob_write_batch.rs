//! `write_batch`: every operation of a batch takes effect, or none does.
//!
//! The WAL is cut at every byte and flipped at every byte, and each result is
//! compared with the two states a reader may see: the one before the batch and
//! the one after it. The bytes are compared with a frame built here from the
//! documented layout and an independent CRC-32.
#![cfg(feature = "fs")]

use alice_db::blob::{BlobStorage, WriteBatch};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

type State = BTreeMap<Vec<u8>, Vec<u8>>;

fn state(store: &BlobStorage) -> State {
    store.scan_prefix(b"").unwrap().into_iter().collect()
}

fn expect(pairs: &[(&[u8], &[u8])]) -> State {
    pairs
        .iter()
        .map(|(k, v)| (k.to_vec(), v.to_vec()))
        .collect()
}

/// CRC-32 (IEEE), bit by bit, independent of the crate's checksum.
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A WAL holding base records, then one batch; returns its bytes and the length before the batch.
fn wal_with_batch(dir: &Path) -> (Vec<u8>, usize) {
    let path = dir.join("blob.wal");
    {
        let store = BlobStorage::open(&path).unwrap();
        store.put(b"keep", b"1").unwrap();
        store.put(b"gone", b"2").unwrap();
        store.put(b"over", b"3").unwrap();
    }
    let before = std::fs::metadata(&path).unwrap().len() as usize;
    {
        let store = BlobStorage::open(&path).unwrap();
        let mut batch = WriteBatch::new();
        batch.put(b"new", b"4");
        batch.delete(b"gone");
        batch.put(b"over", b"5");
        store.write_batch(&batch).unwrap();
    }
    (std::fs::read(&path).unwrap(), before)
}

fn before_state() -> State {
    expect(&[(b"keep", b"1"), (b"gone", b"2"), (b"over", b"3")])
}

fn after_state() -> State {
    expect(&[(b"keep", b"1"), (b"new", b"4"), (b"over", b"5")])
}

fn reopen(bytes: &[u8]) -> State {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    std::fs::write(&path, bytes).unwrap();
    state(&BlobStorage::open(&path).unwrap())
}

#[test]
fn a_wal_cut_at_any_byte_of_the_batch_reopens_to_the_state_before_it() {
    let tmp = TempDir::new().unwrap();
    let (bytes, before) = wal_with_batch(tmp.path());
    assert!(bytes.len() > before + 20, "the batch wrote {} bytes", bytes.len() - before);
    let mut compared = 0;
    for cut in before..bytes.len() {
        assert_eq!(reopen(&bytes[..cut]), before_state(), "cut at {cut}");
        compared += 1;
    }
    assert_eq!(compared, bytes.len() - before);
    assert_eq!(reopen(&bytes), after_state());
}

#[test]
fn a_flipped_byte_anywhere_in_the_batch_drops_the_whole_batch() {
    let tmp = TempDir::new().unwrap();
    let (bytes, before) = wal_with_batch(tmp.path());
    let mut compared = 0;
    for at in before..bytes.len() {
        let mut bad = bytes.clone();
        bad[at] ^= 0x40;
        assert_eq!(reopen(&bad), before_state(), "flip at {at}");
        compared += 1;
    }
    assert_eq!(compared, bytes.len() - before);
}

/// The frame of the batch in `wal_with_batch`, from the documented layout.
fn documented_frame() -> Vec<u8> {
    fn op(body: &mut Vec<u8>, kind: u8, key: &[u8], value: &[u8]) {
        body.push(kind);
        body.push(0);
        body.extend_from_slice(&(key.len() as u32).to_le_bytes());
        body.extend_from_slice(&(value.len() as u32).to_le_bytes());
        body.extend_from_slice(key);
        body.extend_from_slice(value);
    }
    let mut body = 3_u32.to_le_bytes().to_vec();
    op(&mut body, 1, b"new", b"4");
    op(&mut body, 2, b"gone", b"");
    op(&mut body, 1, b"over", b"5");
    let mut frame = vec![3_u8];
    frame.extend_from_slice(&0_u32.to_le_bytes());
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.push(0);
    frame.extend_from_slice(&body);
    let crc = crc32_ieee(&frame);
    frame.extend_from_slice(&crc.to_le_bytes());
    frame
}

#[test]
fn the_batch_is_one_frame_with_the_documented_bytes() {
    let tmp = TempDir::new().unwrap();
    let (bytes, before) = wal_with_batch(tmp.path());
    assert_eq!(&bytes[before..], documented_frame().as_slice());
}

/// A frame with a correct CRC whose body does not parse must not apply a part of it.
#[test]
fn a_body_that_does_not_parse_is_dropped_even_with_a_good_checksum() {
    let tmp = TempDir::new().unwrap();
    let (bytes, before) = wal_with_batch(tmp.path());
    let good = documented_frame();
    let body = &good[10..good.len() - 4];
    let reframe = |body: &[u8]| {
        let mut frame = vec![3_u8];
        frame.extend_from_slice(&0_u32.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
        frame.push(0);
        frame.extend_from_slice(body);
        let crc = crc32_ieee(&frame);
        frame.extend_from_slice(&crc.to_le_bytes());
        frame
    };
    let mut more_ops = body.to_vec();
    more_ops[0] = 4; // one more operation than the body holds
    let mut fewer_ops = body.to_vec();
    fewer_ops[0] = 2; // bytes left over after the last operation
    let mut bad_type = body.to_vec();
    bad_type[4] = 9; // an unknown operation
    let mut bad_kind = body.to_vec();
    bad_kind[5] = 9; // an unknown value kind
    let mut long_key = body.to_vec();
    long_key[6..10].copy_from_slice(&u32::MAX.to_le_bytes()); // a key longer than the body
    let zero_ops = 0_u32.to_le_bytes().to_vec(); // the writer never writes an empty batch
    for (name, bad) in [
        ("more", more_ops),
        ("fewer", fewer_ops),
        ("type", bad_type),
        ("kind", bad_kind),
        ("key", long_key),
        ("zero", zero_ops),
    ] {
        let mut wal = bytes[..before].to_vec();
        wal.extend_from_slice(&reframe(&bad));
        assert_eq!(reopen(&wal), before_state(), "{name}");
    }
}

#[test]
fn records_after_a_batch_replay_in_order() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    {
        let store = BlobStorage::open(&path).unwrap();
        let mut batch = WriteBatch::new();
        batch.put(b"a", b"batch");
        batch.put(b"b", b"batch");
        store.write_batch(&batch).unwrap();
        store.put(b"a", b"after").unwrap();
    }
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(state(&store), expect(&[(b"a", b"after"), (b"b", b"batch")]));
    // appends after a reopen land behind the batch
    store.put(b"c", b"later").unwrap();
    drop(store);
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(
        state(&store),
        expect(&[(b"a", b"after"), (b"b", b"batch"), (b"c", b"later")])
    );
}

#[test]
fn within_a_batch_the_later_operation_on_a_key_wins() {
    let store = BlobStorage::new();
    store.put(b"x", b"old").unwrap();
    store.put(b"y", b"old").unwrap();
    let mut batch = WriteBatch::new();
    batch.delete(b"x");
    batch.put(b"x", b"back");
    batch.put(b"y", b"first");
    batch.delete(b"y");
    store.write_batch(&batch).unwrap();
    assert_eq!(state(&store), expect(&[(b"x", b"back")]));
}

#[test]
fn an_empty_batch_writes_nothing() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let store = BlobStorage::open(&path).unwrap();
    let batch = WriteBatch::new();
    assert!(batch.is_empty());
    assert_eq!(batch.len(), 0);
    store.write_batch(&batch).unwrap();
    store.flush().unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
}

#[test]
fn a_value_that_compresses_comes_back_after_reopen() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let value = vec![b'z'; 64 * 1024];
    {
        let store = BlobStorage::open(&path).unwrap();
        let mut batch = WriteBatch::new();
        batch.put(b"big", &value);
        store.write_batch(&batch).unwrap();
        let wal_len = std::fs::metadata(&path).unwrap().len() as usize;
        assert!(wal_len < value.len() / 2, "stored raw: {wal_len} bytes");
    }
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(store.get(b"big").unwrap().unwrap(), value);
}

/// A reader never sees a part of a batch: both keys of each batch hold the same value.
#[test]
fn a_concurrent_reader_sees_all_of_a_batch_or_none() {
    let store = Arc::new(BlobStorage::new());
    let writer = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for i in 0_u32..5_000 {
                let v = i.to_le_bytes();
                let mut batch = WriteBatch::new();
                batch.put(b"pair/a", &v);
                batch.put(b"pair/b", &v);
                store.write_batch(&batch).unwrap();
            }
        })
    };
    let mut seen_pairs = 0_u64;
    while !writer.is_finished() {
        let got = store.scan_prefix(b"pair/").unwrap();
        if got.len() == 2 {
            assert_eq!(got[0].1, got[1].1, "a part of a batch was visible");
            seen_pairs += 1;
        } else {
            assert!(got.is_empty(), "one key of a batch was visible: {got:?}");
        }
    }
    writer.join().unwrap();
    assert!(seen_pairs > 0, "the reader compared nothing");
}

/// Every acknowledged write survives flushes that other threads start.
#[test]
fn writes_during_a_flush_are_not_lost() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let store = Arc::new(BlobStorage::open(&path).unwrap());
    let flusher = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for _ in 0..200 {
                store.flush_to_sstable().unwrap();
            }
        })
    };
    let writers: Vec<_> = (0_u32..4)
        .map(|t| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                for i in 0_u32..500 {
                    let key = format!("w{t}/{i:04}");
                    if i % 2 == 0 {
                        store.put(key.as_bytes(), b"v").unwrap();
                    } else {
                        let mut batch = WriteBatch::new();
                        batch.put(key.as_bytes(), b"v");
                        store.write_batch(&batch).unwrap();
                    }
                }
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    flusher.join().unwrap();
    let live = store.scan_prefix(b"w").unwrap().len();
    assert_eq!(live, 2_000, "acknowledged writes lost in memory");
    drop(store);
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(store.scan_prefix(b"w").unwrap().len(), 2_000, "lost after reopen");
}
