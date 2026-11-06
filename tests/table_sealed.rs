//! Sealed tables: values reach no file in the clear, a sealed value opens only for
//! its own row, and a rotation moves every row to the current key.
#![cfg(all(feature = "fs", feature = "crypto"))]

use alice_crypto::keystore::{KeyGeneration, KeyProvider, KeyProviderError};
use alice_crypto::Key;
use alice_db::blob::{BlobStorage, ScanOrder};
use alice_db::table::{IndexKey, SealedTable, TableError};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::Arc;
use tempfile::TempDir;

/// Keys by generation; a retired generation is refused like a key that was destroyed.
#[derive(Default)]
struct Keys {
    keys: Mutex<BTreeMap<KeyGeneration, [u8; 32]>>,
    current: Mutex<KeyGeneration>,
}

impl Keys {
    fn with(gen: KeyGeneration) -> Arc<Self> {
        let keys = Arc::new(Self::default());
        keys.rotate_to(gen);
        keys
    }
    fn rotate_to(&self, gen: KeyGeneration) {
        self.keys
            .lock()
            .insert(gen, [u8::try_from(gen).unwrap(); 32]);
        *self.current.lock() = gen;
    }
    fn retire(&self, gen: KeyGeneration) {
        self.keys.lock().remove(&gen);
    }
}

impl KeyProvider for Keys {
    fn current(&self) -> Result<(KeyGeneration, Key), KeyProviderError> {
        let gen = *self.current.lock();
        Ok((gen, self.by_generation(gen)?))
    }
    fn by_generation(&self, gen: KeyGeneration) -> Result<Key, KeyProviderError> {
        self.keys
            .lock()
            .get(&gen)
            .map(|k| Key::from_bytes(*k))
            .ok_or(KeyProviderError::UnknownGeneration(gen))
    }
}

const MARKER: &[u8] = b"the-words-that-must-not-reach-a-file";

fn files_contain(dir: &std::path::Path, needle: &[u8]) -> bool {
    let mut compared = 0;
    let mut found = false;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        // Windows refuses to read the writer's lock file; it holds nothing
        if path.extension().is_some_and(|x| x == "lock") {
            continue;
        }
        let bytes = std::fs::read(path).unwrap();
        compared += 1;
        found |= bytes.windows(needle.len()).any(|w| w == needle);
    }
    assert!(compared > 0, "no file compared");
    found
}

#[test]
fn a_value_reaches_no_file_in_the_clear_and_comes_back() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let store = BlobStorage::open(&path).unwrap();
    let t = SealedTable::new("t", Keys::with(1)).unwrap();
    t.insert(&store, b"p", MARKER, &[]).unwrap();
    assert_eq!(t.get(&store, b"p").unwrap().unwrap(), MARKER);
    store.flush().unwrap();
    assert!(!files_contain(tmp.path(), MARKER), "in the WAL");
    store.flush_to_sstable().unwrap();
    assert!(!files_contain(tmp.path(), MARKER), "in an SSTable");
    drop(store);
    let store = BlobStorage::open(&path).unwrap();
    assert_eq!(t.get(&store, b"p").unwrap().unwrap(), MARKER);
    // the check of the check: an unsealed value is found by the same search
    store.put(b"plain", MARKER).unwrap();
    store.flush().unwrap();
    assert!(files_contain(tmp.path(), MARKER));
}

/// The stored bytes of one row, copied to another row or table, do not open.
#[test]
fn a_sealed_value_opens_only_for_its_own_row() {
    let store = BlobStorage::new();
    let keys = Keys::with(1);
    let t = SealedTable::new("t", keys.clone()).unwrap();
    t.insert(&store, b"p1", b"secret", &[]).unwrap();
    t.insert(&store, b"p2", b"other", &[]).unwrap();
    let raw: BTreeMap<_, _> = store.scan_prefix(b"").unwrap().into_iter().collect();
    let row1 = raw.iter().find(|(k, _)| k.ends_with(b"p1")).unwrap();
    let row2_key = raw.keys().find(|k| k.ends_with(b"p2")).unwrap();
    store.put(row2_key, row1.1).unwrap();
    assert!(matches!(t.get(&store, b"p2"), Err(TableError::Seal)));

    // the same row bytes under another table
    let other = SealedTable::new("u", keys).unwrap();
    other.insert(&store, b"p1", b"x", &[]).unwrap();
    let other_key: Vec<u8> = store
        .scan_prefix(b"")
        .unwrap()
        .into_iter()
        .map(|(k, _)| k)
        .find(|k| k.ends_with(b"p1") && k != row1.0)
        .unwrap();
    store.put(&other_key, row1.1).unwrap();
    assert!(matches!(other.get(&store, b"p1"), Err(TableError::Seal)));
}

#[test]
fn indexes_scans_and_deletes_work_through_the_seal() {
    let store = BlobStorage::new();
    let t = SealedTable::new("t", Keys::with(1)).unwrap();
    t.insert(&store, b"a", b"1", &[IndexKey::unique("h", b"x")])
        .unwrap();
    t.upsert(&store, b"b", b"2", &[IndexKey::non_unique("o", b"y")])
        .unwrap();
    assert_eq!(
        t.table().find_unique(&store, "h", b"x").unwrap().unwrap(),
        b"a"
    );
    let err = t
        .insert(&store, b"c", b"3", &[IndexKey::unique("h", b"x")])
        .unwrap_err();
    assert!(matches!(err, TableError::UniqueViolation { .. }));
    let rows = t
        .scan(
            &store,
            Bound::Unbounded,
            Bound::Unbounded,
            ScanOrder::Descending,
            10,
        )
        .unwrap();
    assert_eq!(
        rows,
        [
            (b"b".to_vec(), b"2".to_vec()),
            (b"a".to_vec(), b"1".to_vec())
        ]
    );
    assert!(t.table().delete(&store, b"a").unwrap());
    assert_eq!(t.get(&store, b"a").unwrap(), None);
}

#[test]
fn a_rotation_moves_every_row_and_keeps_the_indexes() {
    let tmp = TempDir::new().unwrap();
    let store = BlobStorage::open(tmp.path().join("blob.wal")).unwrap();
    let keys = Keys::with(1);
    let t = SealedTable::new("t", keys.clone()).unwrap();
    for i in 0_u32..700 {
        let pk = format!("p{i:04}");
        let h = format!("h{i}");
        t.insert(
            &store,
            pk.as_bytes(),
            &i.to_le_bytes(),
            &[IndexKey::unique("h", h.as_bytes())],
        )
        .unwrap();
    }
    keys.rotate_to(2);
    t.insert(&store, b"new", b"at-2", &[]).unwrap();
    // old rows still open before the rotation
    assert_eq!(
        t.get(&store, b"p0000").unwrap().unwrap(),
        0_u32.to_le_bytes()
    );

    assert_eq!(
        t.reseal(&store).unwrap(),
        700,
        "the row already at 2 is not rewritten"
    );
    assert_eq!(
        t.reseal(&store).unwrap(),
        0,
        "a second pass has nothing to do"
    );

    keys.retire(1);
    for i in [0_u32, 255, 256, 699] {
        let pk = format!("p{i:04}");
        assert_eq!(
            t.get(&store, pk.as_bytes()).unwrap().unwrap(),
            i.to_le_bytes()
        );
        let h = format!("h{i}");
        assert_eq!(
            t.table()
                .find_unique(&store, "h", h.as_bytes())
                .unwrap()
                .unwrap(),
            pk.as_bytes()
        );
    }
    assert_eq!(t.table().count(&store).unwrap(), 701);
}

/// A key that is gone is an error, never an empty value.
#[test]
fn a_retired_key_is_reported() {
    let store = BlobStorage::new();
    let keys = Keys::with(1);
    let t = SealedTable::new("t", keys.clone()).unwrap();
    t.insert(&store, b"p", b"v", &[]).unwrap();
    keys.rotate_to(2);
    keys.retire(1);
    assert!(matches!(
        t.get(&store, b"p"),
        Err(TableError::Key(KeyProviderError::UnknownGeneration(1)))
    ));
    assert!(matches!(t.reseal(&store), Err(TableError::Key(_))));
}

#[test]
fn a_rotation_refuses_a_read_only_store() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    let keys = Keys::with(1);
    let t = SealedTable::new("t", keys.clone()).unwrap();
    let writer = BlobStorage::open(&path).unwrap();
    t.insert(&writer, b"p", b"v", &[]).unwrap();
    keys.rotate_to(2);
    let reader = BlobStorage::open_existing_read_only(&path).unwrap();
    assert!(matches!(t.reseal(&reader), Err(TableError::Io(_))));
}
