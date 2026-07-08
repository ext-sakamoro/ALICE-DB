//! Integration tests for the v0.2.0-alpha.5 multi-SSTable path.
#![allow(clippy::case_sensitive_file_extension_comparisons, clippy::useless_vec)]
//!
//! Coverage:
//!
//! - `FlushMode::Append` produces sequentially numbered `SSTables` and
//!   leaves prior ones alone.
//! - Multi-SSTable read on reopen preserves last-write-wins semantics
//!   across sequence order.
//! - Manual `compact_all_sstables` merges every `SSTable` + the in-memory
//!   state into a single new file and deletes the old ones.
//! - Auto-compaction fires once the `SSTable` count reaches
//!   `max_sstables_before_compaction`.
//! - A pre-α-3.3 database (`blob.sst` only) upgrades cleanly under
//!   `FlushMode::Append` — the legacy file is treated as seq 0 and the
//!   next flush writes `blob-000001.sst` alongside it.
//! - Corrupted `SSTable` in the sequence surfaces on open (fail-loud).

use alice_db::blob::{BlobStorage, BlobStorageConfig};
use alice_db::blob_sstable::{
    enumerate_sstables, parse_sstable_seq, sstable_filename_for_seq, FlushMode,
};
use alice_db::AliceDB;
use tempfile::TempDir;

// -------------------------------------------------------------------------
// Filename helpers
// -------------------------------------------------------------------------

#[test]
fn filename_helpers_round_trip_the_sequence_number() {
    for seq in [0_u64, 1, 42, 999_999, 1_000_000, u64::MAX] {
        let name = sstable_filename_for_seq(seq);
        assert_eq!(parse_sstable_seq(&name), Some(seq), "round-trip {seq}");
    }
}

#[test]
fn parse_sstable_seq_recognises_backward_compat_blob_sst_as_seq_0() {
    assert_eq!(parse_sstable_seq("blob.sst"), Some(0));
}

#[test]
fn parse_sstable_seq_rejects_unrelated_filenames() {
    for bad in [
        "notes.txt",
        "blob.sst.tmp",
        "blob-1234.sst",    // too short
        "blob-0000001.sst", // one too many digits — but len() >= 6, still parses as valid! actually 7 digits parses OK
        "readme.md",
    ] {
        // The 7-digit case is intentional: parse_sstable_seq accepts any
        // digit-only stretch as long as it's at least 6 chars. We only
        // reject the truly malformed ones.
        if bad == "blob-0000001.sst" {
            assert_eq!(parse_sstable_seq(bad), Some(1));
            continue;
        }
        assert_eq!(parse_sstable_seq(bad), None, "filename: {bad}");
    }
}

// -------------------------------------------------------------------------
// FlushMode::Append pipeline
// -------------------------------------------------------------------------

fn open_append(tmp: &TempDir) -> AliceDB {
    let cfg = BlobStorageConfig {
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX, // disable auto-compact for this test
        ..BlobStorageConfig::default()
    };
    AliceDB::open_with_blob_config(tmp.path(), cfg).unwrap()
}

#[test]
fn append_mode_creates_numbered_sstables_and_leaves_older_ones_alone() {
    let tmp = TempDir::new().unwrap();
    let db = open_append(&tmp);

    // Three explicit flushes → three SSTable files.
    for round in 0..3 {
        for i in 0..4 {
            let key = format!("round{round}-key{i:02}");
            let value = format!("val-{round}-{i}");
            db.put_blob(key.as_bytes(), value.as_bytes()).unwrap();
        }
        db.compact_blob_sstable().unwrap();
    }

    let files = enumerate_sstables(tmp.path()).unwrap();
    let names: Vec<String> = files
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec![
            "blob-000001.sst".to_string(),
            "blob-000002.sst".to_string(),
            "blob-000003.sst".to_string(),
        ],
        "append mode must produce sequentially numbered SSTables"
    );
}

#[test]
fn reopen_after_multi_flush_recovers_state_with_last_write_wins_semantics() {
    let tmp = TempDir::new().unwrap();
    {
        let db = open_append(&tmp);
        db.put_blob(b"k", b"first").unwrap();
        db.compact_blob_sstable().unwrap();
        db.put_blob(b"k", b"second").unwrap();
        db.compact_blob_sstable().unwrap();
        db.put_blob(b"k", b"third").unwrap();
        db.compact_blob_sstable().unwrap();
    }
    // Reopen. Multi-SSTable load must apply files in ascending order so
    // that the newest ("third") wins.
    let db = open_append(&tmp);
    assert_eq!(db.get_blob(b"k").unwrap(), Some(b"third".to_vec()));
}

// -------------------------------------------------------------------------
// Compaction — manual and automatic
// -------------------------------------------------------------------------

#[test]
fn manual_compaction_merges_every_sstable_into_one() {
    let tmp = TempDir::new().unwrap();
    let db = open_append(&tmp);

    for round in 0..3 {
        for i in 0..2 {
            let key = format!("key-{round}-{i}");
            db.put_blob(key.as_bytes(), format!("v{round}{i}").as_bytes())
                .unwrap();
        }
        db.compact_blob_sstable().unwrap();
    }
    assert_eq!(enumerate_sstables(tmp.path()).unwrap().len(), 3);

    // Compact → single merged SSTable, older ones deleted.
    db.compact_all_blob_sstables().unwrap();
    let after = enumerate_sstables(tmp.path()).unwrap();
    assert_eq!(after.len(), 1, "compaction must collapse to one file");
    let merged_name = after[0].file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        merged_name.starts_with("blob-") && merged_name.ends_with(".sst"),
        "merged name should still follow the append naming convention: {merged_name}"
    );

    // Every key survives the merge.
    for round in 0..3 {
        for i in 0..2 {
            let key = format!("key-{round}-{i}");
            let expected = format!("v{round}{i}");
            assert_eq!(
                db.get_blob(key.as_bytes()).unwrap(),
                Some(expected.into_bytes())
            );
        }
    }
}

#[test]
fn auto_compaction_fires_when_sstable_count_reaches_threshold() {
    let tmp = TempDir::new().unwrap();
    let cfg = BlobStorageConfig {
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: 3,
        wal_flush_threshold_bytes: 64, // tiny so each put drives a flush
        ..BlobStorageConfig::default()
    };
    let db = AliceDB::open_with_blob_config(tmp.path(), cfg).unwrap();

    // Every put crosses the WAL threshold and rolls into a new SSTable.
    // After 3 SSTables, auto-compaction collapses them.
    for i in 0..12 {
        let key = format!("k{i:02}");
        db.put_blob(key.as_bytes(), &vec![b'x'; 128]).unwrap();
    }

    let files = enumerate_sstables(tmp.path()).unwrap();
    assert!(
        files.len() <= 3,
        "auto-compaction must keep SSTable count bounded, got {} files: {:?}",
        files.len(),
        files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
    );

    // All 12 keys are still present.
    for i in 0..12 {
        let key = format!("k{i:02}");
        assert_eq!(db.get_blob(key.as_bytes()).unwrap(), Some(vec![b'x'; 128]));
    }
}

// -------------------------------------------------------------------------
// Backward compat with pre-α-3.3 blob.sst
// -------------------------------------------------------------------------

#[test]
fn opening_a_pre_alpha33_database_under_append_mode_treats_blob_sst_as_seq_0() {
    let tmp = TempDir::new().unwrap();
    // First: write with the α-3.2a-compatible Overwrite mode so we end
    // up with a single blob.sst file.
    {
        let db = AliceDB::open(tmp.path()).unwrap();
        db.put_blob(b"legacy", b"one").unwrap();
        db.compact_blob_sstable().unwrap();
    }
    assert!(tmp.path().join("blob.sst").exists());

    // Reopen with Append. The legacy blob.sst is treated as seq 0 and
    // the next flush produces blob-000001.sst next to it.
    {
        let db = open_append(&tmp);
        assert_eq!(db.get_blob(b"legacy").unwrap(), Some(b"one".to_vec()));
        db.put_blob(b"fresh", b"two").unwrap();
        db.compact_blob_sstable().unwrap();
    }

    let names: Vec<String> = enumerate_sstables(tmp.path())
        .unwrap()
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.contains(&"blob.sst".to_string()),
        "legacy blob.sst must be preserved until the next compaction"
    );
    assert!(
        names.contains(&"blob-000001.sst".to_string()),
        "append flush must have created seq 1: {names:?}"
    );
}

// -------------------------------------------------------------------------
// Fail-loud: corrupted sequential SSTable surfaces on open
// -------------------------------------------------------------------------

#[test]
fn corrupted_sequential_sstable_surfaces_on_open() {
    let tmp = TempDir::new().unwrap();
    {
        let db = open_append(&tmp);
        db.put_blob(b"k", b"v").unwrap();
        db.compact_blob_sstable().unwrap();
    }
    let sst = tmp.path().join("blob-000001.sst");
    assert!(sst.exists());

    // Overwrite the last byte of the footer magic.
    let mut bytes = std::fs::read(&sst).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&sst, &bytes).unwrap();

    let Err(err) = open_append_result(&tmp) else {
        panic!("open must fail on corrupted append-mode SSTable")
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "got: {err}");
}

// Test-only helper: open in append mode returning the Result unwrapped.
fn open_append_result(tmp: &TempDir) -> std::io::Result<AliceDB> {
    let cfg = BlobStorageConfig {
        flush_mode: FlushMode::Append,
        max_sstables_before_compaction: usize::MAX,
        ..BlobStorageConfig::default()
    };
    AliceDB::open_with_blob_config(tmp.path(), cfg)
}

// Silence "unused import" for BlobStorage when the module compiles
// without touching it.
#[allow(dead_code)]
fn __hold_blob_storage_import<S: Into<Option<BlobStorage>>>(_: S) {}
