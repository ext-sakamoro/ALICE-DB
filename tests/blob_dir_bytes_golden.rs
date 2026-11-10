//! The bytes a fixed workload leaves on disk, pinned.
//!
//! This is a change detector, not a correctness oracle: it pins the files the blob store
//! writes today (WAL, numbered and canonical `SSTables`, the store record) so that moving the
//! file operations behind another layer can be shown to change no byte. The workload covers
//! every file path the store uses: plain and compressed puts, deletes, a batch, appended
//! flushes, an automatic compaction, an explicit compaction, a reopen with recovery and a
//! canonical overwrite flush.
#![cfg(feature = "fs")]

use alice_db::blob::{BlobStorage, BlobStorageConfig, WriteBatch};
use alice_db::blob_sstable::FlushMode;
use alice_db::blob_wal::SyncPolicy;
use sha2::{Digest, Sha256};
use std::path::Path;

/// sha256 over every file under `dir`, sorted by relative path: name length (u64 LE), name,
/// length (u64 LE), bytes
fn tree_digest(dir: &Path) -> (String, usize) {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    let mut h = Sha256::new();
    for p in &files {
        let name = p
            .strip_prefix(dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = std::fs::read(p).unwrap();
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    (format!("{:x}", h.finalize()), files.len())
}

fn walk(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let n = std::fs::metadata(&p).unwrap().len();
                out.push(format!("{} {n}", p.strip_prefix(dir).unwrap().display()));
            }
        }
    }
    out.sort();
    out
}

fn value(i: u32) -> Vec<u8> {
    // every 3rd value is long and repetitive, so it goes through the compressed path
    if i.is_multiple_of(3) {
        format!("long value {i} ").repeat(40).into_bytes()
    } else {
        format!("v{i}").into_bytes()
    }
}

fn workload(dir: &Path) {
    let wal = dir.join("store").join("blob.wal");
    let append = BlobStorageConfig {
        sync_policy: SyncPolicy::EveryWrite,
        wal_flush_threshold_bytes: 4096,
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: 3,
    };
    {
        let s = BlobStorage::open_with_config(&wal, append).unwrap();
        for i in 0..120u32 {
            s.put(format!("k{i:04}").as_bytes(), &value(i)).unwrap();
        }
        for i in (0..120u32).step_by(7) {
            s.delete(format!("k{i:04}").as_bytes()).unwrap();
        }
        let mut b = WriteBatch::new();
        for i in 200..230u32 {
            b.put(format!("k{i:04}").as_bytes(), &value(i));
        }
        b.delete(b"k0001");
        s.write_batch(&b).unwrap();
        s.flush_to_sstable().unwrap();
        s.compact_all_sstables().unwrap();
        for i in 300..310u32 {
            s.put(format!("k{i:04}").as_bytes(), &value(i)).unwrap();
        }
        // dropped with records left in the WAL
    }
    {
        let s = BlobStorage::open_with_config(&wal, append).unwrap();
        s.put(b"after-reopen", b"x").unwrap();
        s.flush_to_sstable().unwrap();
    }
    {
        let overwrite = BlobStorageConfig {
            sync_policy: SyncPolicy::EveryWrite,
            wal_flush_threshold_bytes: u64::MAX,
            flush_mode: FlushMode::Overwrite,
            max_sstables_before_compaction: usize::MAX,
        };
        let s =
            BlobStorage::open_with_config(dir.join("canon").join("blob.wal"), overwrite).unwrap();
        for i in 0..40u32 {
            s.put(format!("c{i:03}").as_bytes(), &value(i)).unwrap();
        }
        s.flush_to_sstable().unwrap();
        s.put(b"c-tail", b"y").unwrap();
    }
}

/// The digest and file count of the workload's tree on this revision of the store
const PINNED: (&str, usize) = (
    "a774004ab503c1eaac8eb9d9a28fc8a5b85b8c6034980db4dd4dc43f56b8d31e",
    7,
);

#[test]
fn the_files_a_fixed_workload_writes_are_unchanged() {
    let a = tempfile::tempdir().unwrap();
    workload(a.path());
    let got = tree_digest(a.path());
    // the same workload twice gives the same bytes (the pin is meaningful)
    let b = tempfile::tempdir().unwrap();
    workload(b.path());
    assert_eq!(
        tree_digest(b.path()),
        got,
        "the workload is not deterministic"
    );
    assert!(got.1 >= 4, "the workload left too few files: {got:?}");
    if std::env::var_os("BLOB_DIR_BYTES_LIST").is_some() {
        for e in walk(a.path()) {
            eprintln!("{e}");
        }
    }
    assert_eq!((got.0.as_str(), got.1), PINNED, "the on-disk bytes changed");
}
