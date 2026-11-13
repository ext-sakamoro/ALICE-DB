//! `write_batch`: every operation of a batch takes effect, or none does.
//!
//! The WAL is cut at every byte and flipped at every byte, and each result is
//! compared with the two states a reader may see: the one before the batch and
//! the one after it. The bytes are compared with a frame built here from the
//! documented layout and an independent CRC-32.
#![cfg(feature = "fs")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a concurrency test: the threads exercise the store's locking and are joined"
)]

use alice_db::blob::{BlobStorage, BlobStorageConfig, WriteBatch};
use alice_db::blob_sstable::FlushMode;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
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
    let before = usize::try_from(std::fs::metadata(&path).unwrap().len()).unwrap();
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
    let store = BlobStorage::open(&path).unwrap();
    state(&store)
}

#[test]
fn a_wal_cut_at_any_byte_of_the_batch_reopens_to_the_state_before_it() {
    let tmp = TempDir::new().unwrap();
    let (bytes, before) = wal_with_batch(tmp.path());
    assert!(
        bytes.len() > before + 20,
        "the batch wrote {} bytes",
        bytes.len() - before
    );
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
        body.extend_from_slice(&u32::try_from(key.len()).unwrap().to_le_bytes());
        body.extend_from_slice(&u32::try_from(value.len()).unwrap().to_le_bytes());
        body.extend_from_slice(key);
        body.extend_from_slice(value);
    }
    let mut body = 3_u32.to_le_bytes().to_vec();
    op(&mut body, 1, b"new", b"4");
    op(&mut body, 2, b"gone", b"");
    op(&mut body, 1, b"over", b"5");
    let mut frame = vec![3_u8];
    frame.extend_from_slice(&0_u32.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
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

/// A frame whose CRC holds is no torn tail. A body this build cannot read stops
/// the open and cuts nothing, and no part of it is applied.
#[test]
fn a_body_that_does_not_parse_stops_the_open_and_cuts_nothing() {
    let tmp = TempDir::new().unwrap();
    let (bytes, before) = wal_with_batch(tmp.path());
    let good = documented_frame();
    let body = &good[10..good.len() - 4];
    let reframe = |body: &[u8]| {
        let mut frame = vec![3_u8];
        frame.extend_from_slice(&0_u32.to_le_bytes());
        frame.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
        frame.push(0);
        frame.extend_from_slice(body);
        let crc = crc32_ieee(&frame);
        frame.extend_from_slice(&crc.to_le_bytes());
        frame
    };
    let good_put = {
        let mut frame = vec![1_u8];
        frame.extend_from_slice(&1_u32.to_le_bytes());
        frame.extend_from_slice(&1_u32.to_le_bytes());
        frame.push(0);
        frame.extend_from_slice(b"pv");
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
    let mut delete_with_value = 1_u32.to_le_bytes().to_vec();
    delete_with_value.extend_from_slice(&[2, 0]);
    delete_with_value.extend_from_slice(&1_u32.to_le_bytes());
    delete_with_value.extend_from_slice(&1_u32.to_le_bytes());
    delete_with_value.extend_from_slice(b"kv");
    let mut delete_compressed = 1_u32.to_le_bytes().to_vec();
    delete_compressed.extend_from_slice(&[2, 1]);
    delete_compressed.extend_from_slice(&1_u32.to_le_bytes());
    delete_compressed.extend_from_slice(&0_u32.to_le_bytes());
    delete_compressed.extend_from_slice(b"k");
    for (name, bad) in [
        ("more", more_ops),
        ("fewer", fewer_ops),
        ("type", bad_type),
        ("kind", bad_kind),
        ("key", long_key),
        ("zero", zero_ops),
        ("delete with a value", delete_with_value),
        ("delete with a kind", delete_compressed),
    ] {
        let mut wal = bytes[..before].to_vec();
        wal.extend_from_slice(&reframe(&bad));
        wal.extend_from_slice(&good_put);
        assert_refused(&wal, name);
    }
}

fn assert_refused(wal: &[u8], name: &str) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    std::fs::write(&path, wal).unwrap();
    let Err(err) = BlobStorage::open(&path) else {
        panic!("{name}: the open went past a record it cannot read");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{name}: {err}");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        wal,
        "{name}: the WAL was changed"
    );
    assert_eq!(
        count_beside_the_lock(tmp.path()),
        1,
        "{name}: a file was added"
    );
}

/// The same for a put whose value kind this build does not know.
#[test]
fn a_put_of_an_unknown_value_kind_stops_the_open_and_cuts_nothing() {
    let mut frame = vec![1_u8];
    frame.extend_from_slice(&1_u32.to_le_bytes());
    frame.extend_from_slice(&1_u32.to_le_bytes());
    frame.push(9);
    frame.extend_from_slice(b"kv");
    let crc = crc32_ieee(&frame);
    frame.extend_from_slice(&crc.to_le_bytes());
    assert_refused(&frame, "kind");
}

/// Only a bad checksum at the end is cut, and its bytes are kept next to the WAL.
#[test]
fn a_bad_checksum_at_the_end_is_still_cut_with_its_bytes_kept() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    drop(BlobStorage::open(&path).unwrap());
    let mut frame = documented_frame();
    let last = frame.len() - 1;
    frame[last] ^= 1;
    std::fs::write(&path, &frame).unwrap();
    let store = BlobStorage::open(&path).unwrap();
    assert!(state(&store).is_empty());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
    assert_eq!(count_beside_the_lock(tmp.path()), 2);
}

/// The files in `dir` other than `blob.lock`, which every writer leaves.
fn count_beside_the_lock(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_name() != "blob.lock")
        .count()
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
        let wal_len = usize::try_from(std::fs::metadata(&path).unwrap().len()).unwrap();
        assert!(wal_len < value.len() / 2, "stored raw: {wal_len} bytes");
    }
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(store.get(b"big").unwrap().unwrap(), value);
}

/// A reader never sees a part of a batch: both keys of each batch hold the same value.
#[test]
fn a_concurrent_reader_sees_all_of_a_batch_or_none() {
    const READS: u64 = 2_000;
    let store = Arc::new(BlobStorage::new());
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (store, stop) = (Arc::clone(&store), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut i = 0_u32;
            while !stop.load(Ordering::SeqCst) {
                let v = i.to_le_bytes();
                let mut batch = WriteBatch::new();
                batch.put(b"pair/a", &v);
                batch.put(b"pair/b", &v);
                store.write_batch(&batch).unwrap();
                i = i.wrapping_add(1);
            }
        })
    };
    // The writer runs until the reads are done, however the threads are scheduled.
    let mut seen_pairs = 0_u64;
    let mut values = std::collections::BTreeSet::new();
    // A writer that stops early ends the loop too, and the join reports why.
    while (seen_pairs < READS || values.len() < 2) && !writer.is_finished() {
        let got = store.scan_prefix(b"pair/").unwrap();
        if got.len() == 2 {
            assert_eq!(got[0].1, got[1].1, "a part of a batch was visible");
            values.insert(got[0].1.clone());
            seen_pairs += 1;
        } else {
            assert!(got.is_empty(), "one key of a batch was visible: {got:?}");
        }
    }
    stop.store(true, Ordering::SeqCst);
    writer.join().unwrap();
    assert!(seen_pairs >= READS);
}

/// Every acknowledged write survives flushes that other threads start, in the
/// mode that replaces the `SSTable` and in the one that adds another.
#[test]
fn writes_during_a_flush_are_not_lost() {
    writes_during_a_flush(FlushMode::Overwrite);
}

#[test]
fn writes_during_an_appending_flush_are_not_lost() {
    writes_during_a_flush(FlushMode::Append);
}

fn writes_during_a_flush(flush_mode: FlushMode) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let config = || BlobStorageConfig::default().with_flush_mode(flush_mode);
    let store = Arc::new(BlobStorage::open_with_config(&path, config()).unwrap());
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
    let store = BlobStorage::open_with_config(&path, config()).unwrap();
    assert_eq!(
        store.scan_prefix(b"w").unwrap().len(),
        2_000,
        "lost after reopen"
    );
}

/// A record whose checksum holds but whose type this build does not know was
/// written by a newer build, and is not a torn tail: the open stops and cuts
/// nothing, here and in a read-only open.
#[test]
fn a_valid_record_of_an_unknown_type_stops_the_open_and_cuts_nothing() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    {
        let store = BlobStorage::open(&path).unwrap();
        store.put(b"before", b"1").unwrap();
    }
    let mut frame = vec![0x7F_u8];
    frame.extend_from_slice(&0_u32.to_le_bytes());
    frame.extend_from_slice(&1_u32.to_le_bytes());
    frame.push(0);
    frame.push(b'x');
    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_le_bytes());
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(&frame);
    std::fs::write(&path, &bytes).unwrap();

    let Err(err) = BlobStorage::open(&path) else {
        panic!("an open that does not know a record must not go past it");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
    let Err(err) = BlobStorage::open_existing_read_only(&path) else {
        panic!("a read-only open must not go past it either");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
    assert_eq!(std::fs::read(&path).unwrap(), bytes, "the WAL was changed");
}
