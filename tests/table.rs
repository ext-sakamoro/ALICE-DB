//! Records with secondary indexes on the blob store.
//!
//! The row and its index entries change together; a unique index never holds two
//! rows; an index never names a row that is gone. Every check that matters is done
//! after a reopen as well as in memory, and the indexes are compared with a
//! reference built from the same operations.
#![cfg(feature = "fs")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a concurrency test: the threads exercise the store's locking and are joined"
)]

use alice_db::blob::{BlobStorage, ScanOrder};
use alice_db::table::{IndexKey, Table, TableError};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;
use std::sync::Arc;
use tempfile::TempDir;

fn handle(h: &[u8]) -> IndexKey<'_> {
    IndexKey::unique("handle", h)
}

fn owner(o: &[u8]) -> IndexKey<'_> {
    IndexKey::non_unique("owner", o)
}

#[test]
fn insert_get_and_refuse_a_second_row_with_the_key() {
    let store = BlobStorage::new();
    let t = Table::new("people").unwrap();
    t.insert(&store, b"p1", b"one", &[]).unwrap();
    assert_eq!(t.get(&store, b"p1").unwrap().unwrap(), b"one");
    assert_eq!(t.get(&store, b"p2").unwrap(), None);
    let err = t.insert(&store, b"p1", b"two", &[]).unwrap_err();
    assert!(matches!(err, TableError::AlreadyExists), "{err:?}");
    assert_eq!(t.get(&store, b"p1").unwrap().unwrap(), b"one");
}

#[test]
fn a_unique_index_holds_one_row_and_a_refused_write_changes_nothing() {
    let store = BlobStorage::new();
    let t = Table::new("people").unwrap();
    t.insert(&store, b"p1", b"one", &[handle(b"alice")])
        .unwrap();
    t.insert(&store, b"p2", b"two", &[handle(b"bob")]).unwrap();

    let err = t
        .insert(&store, b"p3", b"x", &[handle(b"alice")])
        .unwrap_err();
    assert!(
        matches!(&err, TableError::UniqueViolation { index } if index == "handle"),
        "{err:?}"
    );
    assert_eq!(t.get(&store, b"p3").unwrap(), None);

    let err = t
        .upsert(&store, b"p2", b"x", &[handle(b"alice")])
        .unwrap_err();
    assert!(matches!(err, TableError::UniqueViolation { .. }), "{err:?}");
    assert_eq!(t.get(&store, b"p2").unwrap().unwrap(), b"two");
    assert_eq!(
        t.find_unique(&store, "handle", b"bob").unwrap().unwrap(),
        b"p2"
    );

    // a row keeps its own value of the unique index
    t.upsert(&store, b"p2", b"new", &[handle(b"bob")]).unwrap();
    assert_eq!(t.get(&store, b"p2").unwrap().unwrap(), b"new");
}

#[test]
fn an_upsert_moves_the_index_entries_and_a_delete_removes_them() {
    let store = BlobStorage::new();
    let t = Table::new("people").unwrap();
    t.insert(&store, b"p1", b"v", &[handle(b"old"), owner(b"o1")])
        .unwrap();
    t.upsert(&store, b"p1", b"v", &[handle(b"new"), owner(b"o2")])
        .unwrap();
    assert_eq!(t.find_unique(&store, "handle", b"old").unwrap(), None);
    assert_eq!(
        t.find_unique(&store, "handle", b"new").unwrap().unwrap(),
        b"p1"
    );
    assert!(t
        .find(&store, "owner", b"o1", ScanOrder::Ascending, 10)
        .unwrap()
        .is_empty());
    assert_eq!(
        t.find(&store, "owner", b"o2", ScanOrder::Ascending, 10)
            .unwrap(),
        [b"p1".to_vec()]
    );
    // the old value of the unique index is free again
    t.insert(&store, b"p2", b"v", &[handle(b"old")]).unwrap();

    assert!(t.delete(&store, b"p1").unwrap());
    assert!(!t.delete(&store, b"p1").unwrap());
    assert_eq!(t.get(&store, b"p1").unwrap(), None);
    assert_eq!(t.find_unique(&store, "handle", b"new").unwrap(), None);
    assert!(t
        .find(&store, "owner", b"o2", ScanOrder::Ascending, 10)
        .unwrap()
        .is_empty());
    assert_eq!(t.count(&store).unwrap(), 1);
}

/// Index keys and names may hold any byte: neither runs into the next field.
#[test]
fn keys_with_zero_bytes_stay_apart_and_in_order() {
    let store = BlobStorage::new();
    let t = Table::new("t").unwrap();
    let keys: [&[u8]; 6] = [b"a", b"a\0", b"a\0\0", b"a\0b", b"a\x01", b"ab"];
    for (i, k) in keys.iter().enumerate() {
        let pk = [b'p', i as u8];
        t.insert(&store, &pk, b"v", &[IndexKey::non_unique("i", k)])
            .unwrap();
    }
    for (i, k) in keys.iter().enumerate() {
        let found = t.find(&store, "i", k, ScanOrder::Ascending, 10).unwrap();
        assert_eq!(found, [vec![b'p', i as u8]], "{k:?}");
    }
    let all = t
        .index_range(
            &store,
            "i",
            Bound::Unbounded,
            Bound::Unbounded,
            ScanOrder::Ascending,
            100,
        )
        .unwrap();
    let in_order: Vec<&[u8]> = all.iter().map(|(k, _)| k.as_slice()).collect();
    assert_eq!(in_order, keys, "index order is the byte order of the keys");

    // the name of one table or index is not a prefix of another's rows
    let longer = Table::new("t\0").unwrap();
    longer.insert(&store, b"x", b"v", &[]).unwrap();
    assert_eq!(t.get(&store, b"x").unwrap(), None);
    assert_eq!(t.count(&store).unwrap(), 6);
    t.insert(&store, b"q", b"v", &[IndexKey::non_unique("i\0", b"a")])
        .unwrap();
    assert_eq!(
        t.find(&store, "i", b"a", ScanOrder::Ascending, 10).unwrap(),
        [vec![b'p', 0]]
    );
}

/// The newest rows of one owner: a key of owner + big-endian time, descending, limited.
#[test]
fn the_latest_rows_by_an_index_range() {
    let store = BlobStorage::new();
    let t = Table::new("msg").unwrap();
    for (pk, room, time) in [
        (b"m1", b"r1", 10_u64),
        (b"m2", b"r1", 30),
        (b"m3", b"r2", 40),
        (b"m4", b"r1", 20),
    ] {
        let mut key = room.to_vec();
        key.extend_from_slice(&time.to_be_bytes());
        t.insert(&store, pk, b"v", &[IndexKey::non_unique("by_time", &key)])
            .unwrap();
    }
    let got = t
        .index_range(
            &store,
            "by_time",
            Bound::Included(b"r1"),
            Bound::Excluded(b"r2"),
            ScanOrder::Descending,
            2,
        )
        .unwrap();
    let pks: Vec<&[u8]> = got.iter().map(|(_, pk)| pk.as_slice()).collect();
    assert_eq!(pks, [&b"m2"[..], b"m4"]);
}

#[test]
fn scan_returns_rows_without_their_index_lists() {
    let store = BlobStorage::new();
    let t = Table::new("t").unwrap();
    let other = Table::new("u").unwrap();
    other.insert(&store, b"b", b"other", &[]).unwrap();
    t.insert(&store, b"a", b"1", &[handle(b"h1")]).unwrap();
    t.insert(&store, b"b", b"2", &[]).unwrap();
    t.insert(&store, b"c", b"3", &[]).unwrap();
    let up = t
        .scan(
            &store,
            Bound::Included(b"b"),
            Bound::Unbounded,
            ScanOrder::Ascending,
            10,
        )
        .unwrap();
    assert_eq!(
        up,
        [
            (b"b".to_vec(), b"2".to_vec()),
            (b"c".to_vec(), b"3".to_vec())
        ]
    );
    let down = t
        .scan(
            &store,
            Bound::Unbounded,
            Bound::Unbounded,
            ScanOrder::Descending,
            2,
        )
        .unwrap();
    assert_eq!(
        down,
        [
            (b"c".to_vec(), b"3".to_vec()),
            (b"b".to_vec(), b"2".to_vec())
        ]
    );
    assert_eq!(t.count(&store).unwrap(), 3);
}

#[test]
fn every_write_is_atomic_at_every_byte_of_the_wal() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let t = Table::new("t").unwrap();
    {
        let store = BlobStorage::open(&path).unwrap();
        t.insert(&store, b"p", b"old", &[handle(b"h-old"), owner(b"o")])
            .unwrap();
    }
    let before = std::fs::metadata(&path).unwrap().len() as usize;
    {
        let store = BlobStorage::open(&path).unwrap();
        t.upsert(&store, b"p", b"new", &[handle(b"h-new"), owner(b"o")])
            .unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() > before);
    for cut in before..=bytes.len() {
        let dir = TempDir::new().unwrap();
        let copy = dir.path().join("blob.wal");
        std::fs::write(&copy, &bytes[..cut]).unwrap();
        let store = BlobStorage::open(&copy).unwrap();
        let (value, h) = if cut == bytes.len() {
            (&b"new"[..], &b"h-new"[..])
        } else {
            (&b"old"[..], &b"h-old"[..])
        };
        assert_eq!(t.get(&store, b"p").unwrap().unwrap(), value, "cut {cut}");
        assert_eq!(
            t.find_unique(&store, "handle", h).unwrap().unwrap(),
            b"p",
            "cut {cut}"
        );
        let other = if cut == bytes.len() {
            &b"h-old"[..]
        } else {
            &b"h-new"[..]
        };
        assert_eq!(
            t.find_unique(&store, "handle", other).unwrap(),
            None,
            "cut {cut}"
        );
    }
}

/// Threads race for one unique value: exactly one wins, across flushes.
#[test]
fn one_winner_for_a_unique_value_under_threads_and_flushes() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(BlobStorage::open(tmp.path().join("blob.wal")).unwrap());
    let t = Arc::new(Table::new("t").unwrap());
    let mut compared = 0;
    for round in 0_u32..60 {
        let name = format!("name-{round}");
        let threads: Vec<_> = (0_u32..4)
            .map(|i| {
                let (store, t, name) = (Arc::clone(&store), Arc::clone(&t), name.clone());
                std::thread::spawn(move || {
                    let pk = format!("{round}-{i}");
                    t.insert(
                        &store,
                        pk.as_bytes(),
                        b"v",
                        &[IndexKey::unique("handle", name.as_bytes())],
                    )
                    .is_ok()
                })
            })
            .collect();
        let wins = threads
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|w| *w)
            .count();
        assert_eq!(wins, 1, "round {round}");
        compared += 1;
        if round % 10 == 9 {
            store.flush_to_sstable().unwrap();
        }
    }
    assert_eq!(compared, 60);
    assert_eq!(t.count(&store).unwrap(), 60);
}

#[test]
fn indexes_equal_a_reference_after_reopen() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let t = Table::new("t").unwrap();
    let mut rows: BTreeMap<Vec<u8>, (Vec<u8>, Vec<u8>)> = BTreeMap::new(); // pk → (handle, owner)
    let mut x = 0x1234_5678_9ABC_DEF1_u64;
    let (mut refused, mut applied) = (0_u32, 0_u32);
    {
        let store = BlobStorage::open(&path).unwrap();
        for i in 0..1_500_u32 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let pk = format!("p{}", x % 40).into_bytes();
            let h = format!("h{}", (x >> 8) % 60).into_bytes();
            let o = format!("o{}", (x >> 16) % 5).into_bytes();
            if (x >> 24).is_multiple_of(5) {
                assert_eq!(t.delete(&store, &pk).unwrap(), rows.remove(&pk).is_some());
            } else {
                let taken = rows.iter().any(|(p, (rh, _))| *rh == h && *p != pk);
                match t.upsert(&store, &pk, &[1], &[handle(&h), owner(&o)]) {
                    Ok(()) => {
                        assert!(!taken, "{h:?} was taken");
                        rows.insert(pk, (h, o));
                        applied += 1;
                    }
                    Err(TableError::UniqueViolation { .. }) => {
                        assert!(taken, "{h:?} was free");
                        refused += 1;
                    }
                    Err(e) => panic!("{e:?}"),
                }
            }
            if i % 400 == 399 {
                store.flush_to_sstable().unwrap();
            }
        }
    }
    assert!(
        refused > 50 && applied > 50,
        "refused {refused}, applied {applied}"
    );
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(t.count(&store).unwrap(), rows.len());
    for (pk, (h, _)) in &rows {
        assert_eq!(
            t.find_unique(&store, "handle", h).unwrap().as_ref(),
            Some(pk)
        );
    }
    for o in 0..5 {
        let o = format!("o{o}").into_bytes();
        let want: BTreeSet<_> = rows
            .iter()
            .filter(|(_, (_, ro))| *ro == o)
            .map(|(p, _)| p.clone())
            .collect();
        let got: BTreeSet<_> = t
            .find(&store, "owner", &o, ScanOrder::Ascending, 1000)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(got, want, "{o:?}");
    }
}

#[test]
fn a_read_only_store_reads_rows_and_refuses_writes() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let t = Table::new("t").unwrap();
    let writer = BlobStorage::open(&path).unwrap();
    t.insert(&writer, b"p", b"v", &[handle(b"h")]).unwrap();
    let reader = BlobStorage::open_existing_read_only(&path).unwrap();
    assert_eq!(t.get(&reader, b"p").unwrap().unwrap(), b"v");
    assert_eq!(
        t.find_unique(&reader, "handle", b"h").unwrap().unwrap(),
        b"p"
    );
    let err = t.upsert(&reader, b"q", b"v", &[]).unwrap_err();
    assert!(
        matches!(err, TableError::Io(ref e) if e.kind() == std::io::ErrorKind::PermissionDenied),
        "{err:?}"
    );
}

/// Each bound kind at a key that several rows hold, against the definition.
#[test]
fn index_range_bounds_take_or_leave_every_row_of_a_key() {
    let store = BlobStorage::new();
    let t = Table::new("t").unwrap();
    for (pk, key) in [
        (&b"1"[..], &b"a"[..]),
        (b"2", b"b"),
        (b"3", b"b"),
        (b"4", b"b\0"),
        (b"5", b"c"),
    ] {
        t.insert(&store, pk, b"v", &[IndexKey::non_unique("i", key)])
            .unwrap();
    }
    let pks = |start, end| -> Vec<Vec<u8>> {
        t.index_range(&store, "i", start, end, ScanOrder::Ascending, 100)
            .unwrap()
            .into_iter()
            .map(|(_, pk)| pk)
            .collect()
    };
    let list = |want: &[&[u8]]| want.iter().map(|p| p.to_vec()).collect::<Vec<_>>();
    let b: &[u8] = b"b";
    assert_eq!(
        pks(Bound::Included(b), Bound::Unbounded),
        list(&[b"2", b"3", b"4", b"5"])
    );
    assert_eq!(
        pks(Bound::Excluded(b), Bound::Unbounded),
        list(&[b"4", b"5"])
    );
    assert_eq!(
        pks(Bound::Unbounded, Bound::Included(b)),
        list(&[b"1", b"2", b"3"])
    );
    assert_eq!(pks(Bound::Unbounded, Bound::Excluded(b)), list(&[b"1"]));
    assert_eq!(
        pks(Bound::Included(b), Bound::Included(b)),
        list(&[b"2", b"3"])
    );
}

/// A delete is a write on a read-only store even when there is nothing to delete.
#[test]
fn a_read_only_store_refuses_a_delete_of_a_missing_row() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let writer = BlobStorage::open(&path).unwrap();
    writer.put(b"x", b"y").unwrap();
    let reader = BlobStorage::open_existing_read_only(&path).unwrap();
    let t = Table::new("t").unwrap();
    assert!(matches!(t.delete(&reader, b"none"), Err(TableError::Io(_))));
}

#[test]
fn a_table_needs_a_name() {
    assert!(Table::new("").is_err());
}
