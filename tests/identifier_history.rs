//! The identifier history: one committed store per arithmetic identifier this crate has stamped
//!
//! Each row is a store written by that release (`tests/fixtures/*/README.md`), in the order the
//! identifiers were introduced. The default read refuses a residual without an identifier and one
//! whose identifier is a change of meaning (`d2209b30…`, alice-det-math 0.4, released as the
//! breaking 0.4.0), reads the identifiers alice-zip still lists as reproducing the same bits as
//! `VerifiedPrevious` with the stored bits, and the current identifier as `Current` (ALICE-Law
//! G9: a meaning change moves the identifier and is a breaking release; a move that reproduces
//! the old bits keeps the old identifier readable).

#![cfg(feature = "fs")]
#![allow(clippy::cast_precision_loss)]

use alice_db::segment::ReadStatus;
use alice_db::{AliceDB, ResidualError, StorageConfig};
use std::path::{Path, PathBuf};

/// What the default read does with a row's store
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Expect {
    /// no identifier: refused (`ResidualError::Unpinned`)
    RefusedUnpinned,
    /// an identifier whose meaning changed: refused (`ResidualError::Mismatch`)
    RefusedMismatch,
    /// an earlier identifier that reproduces the same bits: `ReadStatus::VerifiedPrevious`
    Previous,
    /// the identifier in use: `ReadStatus::Current`
    Current,
}

/// `(fixture, the alice-db release that wrote it, the identifier it stamped, the read)`, oldest
/// first
const HISTORY: [(&str, &str, Option<&str>, Expect); 5] = [
    (
        "store_pre_0_3_0",
        "0.3.0-beta.1",
        None,
        Expect::RefusedUnpinned,
    ),
    (
        "store_det_math_0_4",
        "0.3.0",
        Some("d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e"),
        Expect::RefusedMismatch,
    ),
    (
        "store_db_0_5_0",
        "0.5.0",
        Some("135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"),
        Expect::Previous,
    ),
    (
        "store_db_0_5_1",
        "0.5.1",
        Some("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        Expect::Previous,
    ),
    (
        "store_db_0_6_0",
        "0.6.0",
        Some("51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e"),
        Expect::Current,
    ),
];

fn unhex32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

/// `(major, minor, patch, pre-release or none last)`: a pre-release sorts before its release
fn version_key(v: &str) -> (u64, u64, u64, u8, String) {
    let (core, pre) = v.split_once('-').map_or((v, None), |(c, p)| (c, Some(p)));
    let n: Vec<u64> = core
        .split('.')
        .map(|x| x.parse().expect("number"))
        .collect();
    (
        n[0],
        n[1],
        n[2],
        u8::from(pre.is_none()),
        pre.unwrap_or("").to_owned(),
    )
}

/// The value the 0.5 and later fixtures store at `k` (their README)
fn stored(k: i64) -> f32 {
    0.5 * k as f32 + 0.125 * ((k * 7) % 5) as f32
}

fn copy_fixture(name: &str) -> (tempfile::TempDir, PathBuf) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    assert!(
        fixture.join("seg_1.rkyv").is_file(),
        "{name}: the committed store is missing"
    );
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("store");
    std::fs::create_dir_all(&dir).expect("create");
    for e in std::fs::read_dir(&fixture).expect("fixture") {
        let e = e.expect("entry");
        if e.file_name() != "README.md" {
            std::fs::copy(e.path(), dir.join(e.file_name())).expect("copy");
        }
    }
    (tmp, dir)
}

fn residual_error(e: &std::io::Error) -> &ResidualError {
    e.get_ref()
        .and_then(|inner| inner.downcast_ref::<ResidualError>())
        .unwrap_or_else(|| panic!("the error carries no ResidualError: {e}"))
}

/// The rows are in the order the identifiers were introduced: the writers' versions increase,
/// the reads go from refused to current, the earlier identifiers that stay readable appear in
/// alice-zip's previous list newest first, and the last row is the identifier in use
#[test]
fn the_history_is_in_order_and_ends_at_the_current_identifier() {
    for pair in HISTORY.windows(2) {
        assert!(
            version_key(pair[0].1) < version_key(pair[1].1),
            "{} ({}) is listed before {} ({})",
            pair[0].0,
            pair[0].1,
            pair[1].0,
            pair[1].1
        );
        assert!(
            pair[0].3 <= pair[1].3,
            "{} ({:?}) is listed before {} ({:?})",
            pair[0].0,
            pair[0].3,
            pair[1].0,
            pair[1].3
        );
    }
    let previous_newest_first: Vec<[u8; 32]> = HISTORY
        .iter()
        .rev()
        .filter(|row| row.3 == Expect::Previous)
        .map(|row| unhex32(row.2.expect("a previous row has an identifier")))
        .collect();
    let listed: Vec<[u8; 32]> = alice_core::law::PREVIOUS_SEMANTICS_IDS
        .iter()
        .copied()
        .filter(|id| previous_newest_first.contains(id))
        .collect();
    assert_eq!(
        listed, previous_newest_first,
        "the readable earlier identifiers are not alice-zip's previous list, newest first"
    );
    let last = HISTORY.last().expect("rows");
    assert_eq!(last.3, Expect::Current);
    assert_eq!(unhex32(last.2.expect("stamped")), alice_db::SEMANTICS_ID);
}

#[test]
fn every_identifier_in_the_history_reads_as_its_row_says() {
    let mut rows = 0;
    for (name, release, stamp, expect) in HISTORY {
        let (_tmp, dir) = copy_fixture(name);
        let id = stamp.map(unhex32);
        if let Some(id) = id {
            let seg = std::fs::read(dir.join("seg_1.rkyv")).expect("segment");
            assert_eq!(
                seg.windows(32).filter(|w| *w == id).count(),
                1,
                "{name}: the segment holds the stamp {} once",
                stamp.unwrap_or_default()
            );
        }
        let mut compared = 0;
        for use_mmap in [false, true] {
            let db = AliceDB::with_config(StorageConfig {
                data_dir: dir.clone(),
                enable_wal: false,
                use_mmap,
                ..StorageConfig::default()
            })
            .expect("open");
            for k in 0..64_i64 {
                let what = format!("{name} (alice-db {release}) mmap {use_mmap} key {k}");
                let read = db.get_with_status(k);
                match expect {
                    Expect::RefusedUnpinned => {
                        let e = read.expect_err(&what);
                        assert!(
                            matches!(residual_error(&e), ResidualError::Unpinned(_)),
                            "{what}: {e}"
                        );
                    }
                    Expect::RefusedMismatch => {
                        let e = read.expect_err(&what);
                        let ResidualError::Mismatch(m) = residual_error(&e) else {
                            panic!("{what}: not a mismatch: {e}");
                        };
                        assert_eq!(Some(m.stored), id, "{what}");
                        assert_eq!(m.running, alice_db::SEMANTICS_ID, "{what}");
                    }
                    Expect::Previous | Expect::Current => {
                        let (value, status) = read.expect(&what).expect(&what);
                        assert_eq!(value.to_bits(), stored(k).to_bits(), "{what}");
                        let want = if expect == Expect::Current {
                            ReadStatus::Current
                        } else {
                            ReadStatus::VerifiedPrevious(id.expect("stamped"))
                        };
                        assert_eq!(status, want, "{what}");
                    }
                }
                compared += 1;
            }
        }
        assert_eq!(compared, 2 * 64, "{name}: values compared");
        rows += 1;
    }
    assert_eq!(rows, HISTORY.len());
}
