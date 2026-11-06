//! `scan_range` against a reference `BTreeMap`: both directions, every bound
//! kind and limits, over a memtable and `SSTables` holding overwrites and
//! tombstones. The operations come from a fixed integer sequence, so every run
//! compares the same cases.
#![cfg(feature = "fs")]

use alice_db::blob::{BlobStorage, ScanOrder};
use std::collections::BTreeMap;
use std::ops::Bound;
use tempfile::TempDir;

type Pairs = Vec<(Vec<u8>, Vec<u8>)>;

/// xorshift64, a fixed sequence for the test inputs
struct Seq(u64);
impl Seq {
    fn next(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

fn key(i: u64) -> Vec<u8> {
    format!("k{i:03}").into_bytes()
}

fn bound(seq: &mut Seq) -> Bound<Vec<u8>> {
    match seq.next(3) {
        0 => Bound::Unbounded,
        1 => Bound::Included(key(seq.next(70))),
        _ => Bound::Excluded(key(seq.next(70))),
    }
}

fn as_slice(b: &Bound<Vec<u8>>) -> Bound<&[u8]> {
    match b {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(k) => Bound::Included(k.as_slice()),
        Bound::Excluded(k) => Bound::Excluded(k.as_slice()),
    }
}

fn in_bounds(k: &[u8], start: &Bound<Vec<u8>>, end: &Bound<Vec<u8>>) -> bool {
    let above = match start {
        Bound::Unbounded => true,
        Bound::Included(s) => k >= s.as_slice(),
        Bound::Excluded(s) => k > s.as_slice(),
    };
    let below = match end {
        Bound::Unbounded => true,
        Bound::Included(e) => k <= e.as_slice(),
        Bound::Excluded(e) => k < e.as_slice(),
    };
    above && below
}

/// The reference: filter the whole map, by the definition of the range.
fn reference(
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    start: &Bound<Vec<u8>>,
    end: &Bound<Vec<u8>>,
    order: ScanOrder,
    limit: usize,
) -> Pairs {
    let mut all: Pairs = model
        .iter()
        .filter(|(k, _)| in_bounds(k, start, end))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if matches!(order, ScanOrder::Descending) {
        all.reverse();
    }
    all.truncate(limit);
    all
}

#[test]
fn scans_equal_the_reference_over_memtable_and_sstables() {
    let tmp = TempDir::new().unwrap();
    let store = BlobStorage::open(tmp.path().join("blob.wal")).unwrap();
    let mut model = BTreeMap::new();
    let mut seq = Seq(0x9E37_79B9_7F4A_7C15);
    let mut compared = 0_u32;
    let mut non_empty = 0_u32;
    let mut cut_by_limit = 0_u32;
    for round in 0..40 {
        for _ in 0..30 {
            let k = key(seq.next(60));
            if seq.next(4) == 0 {
                store.delete(&k).unwrap();
                model.remove(&k);
            } else {
                // every third value is long and repetitive, so it is stored compressed
                let n = seq.next(1000);
                let v = if n.is_multiple_of(3) {
                    vec![b'c'; 4096 + usize::try_from(n).unwrap()]
                } else {
                    n.to_le_bytes().to_vec()
                };
                store.put(&k, &v).unwrap();
                model.insert(k, v);
            }
        }
        if round % 4 == 3 {
            store.flush_to_sstable().unwrap();
        }
        for _ in 0..25 {
            let (start, end) = (bound(&mut seq), bound(&mut seq));
            let limit = [0, 1, 2, 7, usize::MAX][usize::try_from(seq.next(5)).unwrap()];
            for order in [ScanOrder::Ascending, ScanOrder::Descending] {
                let got = store
                    .scan_range(as_slice(&start), as_slice(&end), order, limit)
                    .unwrap();
                let want = reference(&model, &start, &end, order, limit);
                assert_eq!(got, want, "{start:?} {end:?} {order:?} {limit}");
                compared += 1;
                non_empty += u32::from(!want.is_empty());
                cut_by_limit += u32::from(want.len() == limit && limit != usize::MAX && limit != 0);
            }
        }
    }
    assert_eq!(compared, 2_000);
    assert!(non_empty > 500, "only {non_empty} non-empty comparisons");
    assert!(
        cut_by_limit > 100,
        "only {cut_by_limit} comparisons cut by the limit"
    );
}

/// The newest observation of a key wins in both directions, a tombstone included.
#[test]
fn the_newest_observation_wins_in_both_directions() {
    let tmp = TempDir::new().unwrap();
    let store = BlobStorage::open(tmp.path().join("blob.wal")).unwrap();
    store.put(b"a", b"old").unwrap();
    store.put(b"b", b"old").unwrap();
    store.put(b"c", b"old").unwrap();
    store.flush_to_sstable().unwrap();
    store.put(b"a", b"new").unwrap();
    store.delete(b"b").unwrap();
    store.flush_to_sstable().unwrap();
    store.put(b"c", b"newest").unwrap();
    let want: Pairs = vec![
        (b"a".to_vec(), b"new".to_vec()),
        (b"c".to_vec(), b"newest".to_vec()),
    ];
    let all = (Bound::Unbounded, Bound::Unbounded);
    let up = store
        .scan_range(all.0, all.1, ScanOrder::Ascending, usize::MAX)
        .unwrap();
    assert_eq!(up, want);
    let mut down = store
        .scan_range(all.0, all.1, ScanOrder::Descending, usize::MAX)
        .unwrap();
    down.reverse();
    assert_eq!(down, want);
    // the limit counts live entries: the tombstone of b is not one of the 2
    let two = store
        .scan_range(all.0, all.1, ScanOrder::Ascending, 2)
        .unwrap();
    assert_eq!(two, want);
}

/// Bounds a `BTreeMap` range would panic on are an empty scan.
#[test]
fn an_inverted_or_empty_range_is_empty() {
    type Range<'a> = (Bound<&'a [u8]>, Bound<&'a [u8]>);
    let store = BlobStorage::new();
    store.put(b"m", b"1").unwrap();
    let cases: [Range<'_>; 4] = [
        (Bound::Included(b"z"), Bound::Included(b"a")),
        (Bound::Excluded(b"m"), Bound::Excluded(b"m")),
        (Bound::Included(b"m"), Bound::Excluded(b"m")),
        (Bound::Excluded(b"m"), Bound::Included(b"m")),
    ];
    for (start, end) in cases {
        for order in [ScanOrder::Ascending, ScanOrder::Descending] {
            assert!(
                store.scan_range(start, end, order, 10).unwrap().is_empty(),
                "{start:?} {end:?}"
            );
        }
    }
    let one = store
        .scan_range(
            Bound::Included(b"m"),
            Bound::Included(b"m"),
            ScanOrder::Descending,
            10,
        )
        .unwrap();
    assert_eq!(one.len(), 1);
}

/// The latest row of one room: the form the descending scan exists for.
#[test]
fn the_latest_rows_under_one_prefix() {
    let store = BlobStorage::new();
    for room in [b"r1", b"r2"] {
        for t in [10_u64, 20, 30] {
            let mut k = room.to_vec();
            k.extend_from_slice(&t.to_be_bytes());
            store.put(&k, &t.to_le_bytes()).unwrap();
        }
    }
    let got = store
        .scan_range(
            Bound::Included(b"r1"),
            Bound::Excluded(b"r2"),
            ScanOrder::Descending,
            2,
        )
        .unwrap();
    let times: Vec<u64> = got
        .iter()
        .map(|(_, v)| u64::from_le_bytes(v.as_slice().try_into().unwrap()))
        .collect();
    assert_eq!(times, [30, 20]);
}
