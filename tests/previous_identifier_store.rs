//! Stores stamped with an earlier identifier read back exact
//!
//! `tests/fixtures/store_db_0_5_1` and `tests/fixtures/store_db_0_5_0` were written by
//! the published `alice-db` 0.5.1 and 0.5.0 (on `alice-zip` 0.10.0, `alice-det-math`
//! 0.5.1 and 0.5.0), so their residuals and law record are stamped with
//! `alice-det-math`'s identifier (`85e08846…`, `135c407d…`), which alice-zip up to 0.10
//! published as its own. This build stamps and checks with alice-zip 0.11's identifier
//! (`51ffeb07…`) and reads those stamps through `alice_core::law::PREVIOUS_SEMANTICS_IDS`:
//! every value comes back with its stored bits and `ReadStatus::VerifiedPrevious`, and the
//! law is found by the law identifier it had under the stamped identifier. A stamp that is
//! neither the current identifier nor a previous one is refused by the default read.
//!
//! `tests/fixtures/store_db_0_6_0` was written by 0.6.0 before its release (`2789c41`, alice-zip
//! 0.11 with alice-det-math 0.5.1, `51ffeb07…`). This build stamps `50a40cf6…` (alice-det-math
//! 0.5.2 at 39507d1b) and alice-zip lists `51ffeb07…` as previous, so its values read back as
//! `VerifiedPrevious` too; its law record is format 2 and carries `51ffeb07…` itself, so it reads
//! as `RecordSemantics::VerifiedPrevious` rather than `Unpinned`.

#![cfg(feature = "fs")]
#![allow(clippy::cast_precision_loss)]

use alice_db::law_store::{LawPointersFoundUnder, LawReadPolicy, RecordSemantics};
use alice_db::segment::ReadStatus;
use alice_db::{AliceDB, StorageConfig};
use std::path::{Path, PathBuf};

/// `(fixture directory, the identifier its writer stamped)`, newest first
const FIXTURES: [(&str, &str); 3] = [
    (
        "store_db_0_6_0",
        "51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e",
    ),
    (
        "store_db_0_5_1",
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
    ),
    (
        "store_db_0_5_0",
        "135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000",
    ),
];

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn unhex32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

/// The value the writer stored at `k` (see the fixture README)
fn stored(k: i64) -> f32 {
    0.5 * k as f32 + 0.125 * ((k * 7) % 5) as f32
}

/// A copy of the fixture in a temporary directory (the store is opened read-write)
fn copy_fixture(name: &str) -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    std::fs::create_dir_all(&dir).expect("create");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    for e in std::fs::read_dir(&fixture).expect("fixture") {
        let e = e.expect("entry");
        if e.file_name() != "README.md" {
            std::fs::copy(e.path(), dir.join(e.file_name())).expect("copy");
        }
    }
    (tmp, dir)
}

fn open(dir: &Path, use_mmap: bool) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_wal: false,
        use_mmap,
        ..StorageConfig::default()
    })
    .expect("open")
}

#[test]
fn the_previous_identifiers_are_not_the_current_one() {
    for (name, stamp) in FIXTURES {
        assert_ne!(unhex32(stamp), alice_db::SEMANTICS_ID, "{name}");
        assert!(
            alice_core::law::PREVIOUS_SEMANTICS_IDS.contains(&unhex32(stamp)),
            "{name}: alice-zip no longer lists {stamp}"
        );
    }
}

#[test]
fn values_stamped_with_a_previous_identifier_read_back_exact_as_verified_previous() {
    let mut compared = 0;
    for (name, stamp) in FIXTURES {
        let (_tmp, dir) = copy_fixture(name);
        for use_mmap in [false, true] {
            let db = open(&dir, use_mmap);
            for k in 0..64_i64 {
                let (value, status) = db
                    .get_with_status(k)
                    .expect("read")
                    .unwrap_or_else(|| panic!("{name} mmap {use_mmap}: key {k} missing"));
                assert_eq!(
                    value.to_bits(),
                    stored(k).to_bits(),
                    "{name} mmap {use_mmap}: key {k}"
                );
                assert_eq!(
                    status,
                    ReadStatus::VerifiedPrevious(unhex32(stamp)),
                    "{name} mmap {use_mmap}: key {k}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, FIXTURES.len() * 2 * 64);
}

/// The stores whose law record is format 1 (written by the 0.5 releases)
const FORMAT_1_LAW_FIXTURES: [&str; 2] = ["store_db_0_5_1", "store_db_0_5_0"];

fn stamp_of(name: &str) -> &'static str {
    FIXTURES
        .iter()
        .find(|(n, _)| *n == name)
        .map_or_else(|| panic!("{name} is not a fixture"), |(_, s)| *s)
}

/// The 0.5 releases wrote law records in format 1, which carries no identifier, so the
/// record reads as `Unpinned` (and a pinned-only read refuses it); the identifier they
/// stamped went into the law-identifier index key, so the law is found by the law
/// identifier it had under that identifier
#[test]
fn a_law_written_before_alice_zip_0_11_is_found_by_its_previous_law_identifier() {
    let mut compared = 0;
    for name in FORMAT_1_LAW_FIXTURES {
        let stamp = stamp_of(name);
        let (_tmp, dir) = copy_fixture(name);
        let db = open(&dir, false);
        let stored = db
            .get_law("ohm")
            .expect("read")
            .unwrap_or_else(|| panic!("{name}: the law record is missing"));
        assert_eq!(stored.semantics, RecordSemantics::Unpinned, "{name}");
        assert!(
            db.get_law_with("ohm", LawReadPolicy::RequirePinned)
                .is_err(),
            "{name}: a pinned-only read refuses a record without an identifier"
        );
        let previous_id = stored.law.law_id(&unhex32(stamp));
        let current_id = stored.law.law_id(&alice_db::SEMANTICS_ID);
        assert_ne!(previous_id, current_id, "{name}");
        let value = db
            .evaluate_by_id(&previous_id, 2.5)
            .unwrap_or_else(|e| panic!("{name}: not found by its previous law identifier: {e}"));
        assert!((value.value - 6.0).abs() < 1e-12, "{name}: {}", value.value);
        compared += 1;
    }
    assert_eq!(compared, FORMAT_1_LAW_FIXTURES.len());
}

/// 0.6.0 writes law records in format 2, which carry the identifier they were written under: the
/// record of `store_db_0_6_0` reads as `VerifiedPrevious` with `51ffeb07…` (also under a
/// pinned-only read), and the law is found by the law identifier it had under that identifier
#[test]
fn a_format_2_law_with_a_previous_identifier_reads_as_verified_previous() {
    let name = "store_db_0_6_0";
    let stamp = unhex32(stamp_of(name));
    let (_tmp, dir) = copy_fixture(name);
    let db = open(&dir, false);
    let stored = db
        .get_law("ohm")
        .expect("read")
        .unwrap_or_else(|| panic!("{name}: the law record is missing"));
    assert_eq!(stored.semantics, RecordSemantics::VerifiedPrevious(stamp));
    let pinned = db
        .get_law_with("ohm", LawReadPolicy::RequirePinned)
        .expect("a pinned record is read under a pinned-only read")
        .expect("present");
    assert_eq!(pinned.semantics, RecordSemantics::VerifiedPrevious(stamp));
    let previous_id = stored.law.law_id(&stamp);
    assert_ne!(previous_id, stored.law.law_id(&alice_db::SEMANTICS_ID));
    let value = db
        .evaluate_by_id(&previous_id, 2.5)
        .unwrap_or_else(|e| panic!("{name}: not found by its previous law identifier: {e}"));
    assert!((value.value - 6.0).abs() < 1e-12, "{name}: {}", value.value);
}

/// The law-identifier index of a 0.5 store was written under the identifier it stamped: a
/// lookup under the current identifier falls through to alice-zip's previous list and finds the
/// stored pointer there, and the reindex adds exactly that pointer under the current
/// identifier (a second reindex adds none)
#[test]
fn law_pointers_fall_through_to_the_previous_identifier_and_reindex_once() {
    let mut compared = 0;
    for (name, stamp) in FIXTURES {
        let (_tmp, dir) = copy_fixture(name);
        let db = open(&dir, false);
        let law = db.get_law("ohm").expect("read").expect("present").law;
        let (pointers, found) = db
            .law_pointers_for(&law, &alice_db::SEMANTICS_ID)
            .expect("lookup");
        assert_eq!(pointers, vec![("ohm".to_owned(), 1)], "{name}");
        assert_eq!(
            found,
            LawPointersFoundUnder::Previous(unhex32(stamp)),
            "{name}"
        );
        assert_eq!(
            db.reindex_laws_under_current_semantics(&alice_db::SEMANTICS_ID)
                .expect("reindex"),
            1,
            "{name}: the one law is reindexed under the current identifier"
        );
        let (pointers, found) = db
            .law_pointers_for(&law, &alice_db::SEMANTICS_ID)
            .expect("lookup");
        assert_eq!(pointers, vec![("ohm".to_owned(), 1)], "{name}");
        assert_eq!(found, LawPointersFoundUnder::Current, "{name}");
        assert_eq!(
            db.reindex_laws_under_current_semantics(&alice_db::SEMANTICS_ID)
                .expect("reindex"),
            0,
            "{name}: a second reindex adds nothing"
        );
        compared += 1;
    }
    assert_eq!(compared, FIXTURES.len());
}

/// The same fixture with its residual stamp replaced by an identifier neither current
/// nor previous: the default read refuses it rather than returning a value
#[test]
fn an_unknown_stamp_in_a_fixture_is_refused() {
    let mut refused = 0;
    for (name, stamp) in FIXTURES {
        let (_tmp, dir) = copy_fixture(name);
        let seg = dir.join("seg_1.rkyv");
        let mut bytes = std::fs::read(&seg).expect("segment");
        let id = unhex32(stamp);
        let at: Vec<usize> = bytes
            .windows(32)
            .enumerate()
            .filter(|(_, w)| *w == id)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(at.len(), 1, "{name}: the stamp occurs once in the segment");
        bytes[at[0]] ^= 0x80;
        std::fs::write(&seg, &bytes).expect("write");
        for use_mmap in [false, true] {
            let db = open(&dir, use_mmap);
            let err = db
                .get_with_status(5)
                .expect_err("an unknown stamp is refused");
            assert!(
                err.to_string().contains(&hex(&{
                    let mut changed = id;
                    changed[0] ^= 0x80;
                    changed
                })),
                "{name} mmap {use_mmap}: {err}"
            );
            refused += 1;
        }
    }
    assert_eq!(refused, FIXTURES.len() * 2);
}
