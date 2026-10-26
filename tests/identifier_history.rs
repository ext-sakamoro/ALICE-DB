//! The identifier history: one committed store per arithmetic identifier this crate has stamped
//!
//! Each row is a store written by that release (`tests/fixtures/*/README.md`), in the order the
//! identifiers were introduced: by the writer's version, and within one version by the position of
//! the writing commit in the first-parent history (two rows may share a version, never a version
//! and a commit). The default read refuses a residual without an identifier and one
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
use std::process::Command;

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

/// The commit whose build wrote a row's store
#[derive(Clone, Copy, Debug)]
enum Writer {
    /// this commit (a release tag's commit, or the commit named in the fixture README)
    At(&'static str),
    /// the commit that adds the fixture directory: its own build wrote the store
    AddingCommit,
}

/// `(fixture, the alice-db version that wrote it, the commit whose build wrote it, the identifier
/// it stamped, the read)`, oldest first
const HISTORY: [(&str, &str, Writer, Option<&str>, Expect); 6] = [
    (
        "store_pre_0_3_0",
        "0.3.0-beta.1",
        Writer::At("52ca813"),
        None,
        Expect::RefusedUnpinned,
    ),
    (
        "store_det_math_0_4",
        "0.3.0",
        Writer::At("v0.3.0"),
        Some("d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e"),
        Expect::RefusedMismatch,
    ),
    (
        "store_db_0_5_0",
        "0.5.0",
        Writer::At("v0.5.0"),
        Some("135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"),
        Expect::Previous,
    ),
    (
        "store_db_0_5_1",
        "0.5.1",
        Writer::At("v0.5.1"),
        Some("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        Expect::Previous,
    ),
    (
        "store_db_0_6_0",
        "0.6.0",
        Writer::At("2789c41"),
        Some("51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e"),
        Expect::Previous,
    ),
    (
        "store_db_0_6_0_det_math_0_5_2",
        "0.6.0",
        Writer::AddingCommit,
        Some("50a40cf64625018acc99c940e544353b901457b4a7a80bf903544ed4ef9af406"),
        Expect::Current,
    ),
];

/// `git` in this repository; the history test needs the full history (CI checks out with
/// `fetch-depth: 0`), so a failing or shallow `git` fails the test rather than skipping it
fn git(args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("utf-8")
        .trim()
        .to_owned()
}

/// The full hash of the commit whose build wrote `fixture`
fn writer_commit(fixture: &str, writer: Writer) -> String {
    let added = || {
        let path = format!("tests/fixtures/{fixture}/seg_1.rkyv");
        let c = git(&[
            "log",
            "--first-parent",
            "--diff-filter=A",
            "--format=%H",
            "HEAD",
            "--",
            &path,
        ]);
        assert!(
            !c.is_empty() && !c.contains('\n'),
            "{fixture}: the commit that adds {path} is not one commit on the first-parent history: {c:?}"
        );
        c
    };
    match writer {
        Writer::At(rev) => {
            let c = git(&["rev-parse", "--verify", &format!("{rev}^{{commit}}")]);
            let first_added = added();
            let ancestor = Command::new("git")
                .args(["merge-base", "--is-ancestor", &c, &first_added])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .status()
                .expect("git merge-base");
            assert!(
                ancestor.success(),
                "{fixture}: the writer {rev} is not an ancestor of the commit that adds the fixture"
            );
            c
        }
        Writer::AddingCommit => added(),
    }
}

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

/// The rows are in the order the identifiers were introduced: ordered by (writer version, the
/// writing commit's position in the first-parent history of `HEAD`), with no two rows sharing
/// both; the reads go from refused to current, the earlier identifiers that stay readable appear
/// in alice-zip's previous list newest first, and the last row is the identifier in use
#[test]
fn the_history_is_in_order_and_ends_at_the_current_identifier() {
    let first_parent: Vec<String> = git(&["rev-list", "--first-parent", "--reverse", "HEAD"])
        .lines()
        .map(str::to_owned)
        .collect();
    assert!(
        first_parent.len() > 1,
        "the checkout has no history (a shallow clone): {} commit(s)",
        first_parent.len()
    );
    let keys: Vec<_> = HISTORY
        .iter()
        .map(|row| {
            let c = writer_commit(row.0, row.2);
            let at = first_parent
                .iter()
                .position(|x| *x == c)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: the writer {c} is not on the first-parent history",
                        row.0
                    )
                });
            (version_key(row.1), at)
        })
        .collect();
    let mut compared = 0;
    for (pair, key) in HISTORY.windows(2).zip(keys.windows(2)) {
        assert!(
            key[0] != key[1],
            "{} and {} share the version {} and the writer commit",
            pair[0].0,
            pair[1].0,
            pair[0].1
        );
        assert!(
            key[0] < key[1],
            "{} ({}, commit {}) is listed before {} ({}, commit {})",
            pair[0].0,
            pair[0].1,
            key[0].1,
            pair[1].0,
            pair[1].1,
            key[1].1
        );
        assert!(
            pair[0].4 <= pair[1].4,
            "{} ({:?}) is listed before {} ({:?})",
            pair[0].0,
            pair[0].4,
            pair[1].0,
            pair[1].4
        );
        compared += 1;
    }
    assert_eq!(compared, HISTORY.len() - 1, "pairs compared");
    let previous_newest_first: Vec<[u8; 32]> = HISTORY
        .iter()
        .rev()
        .filter(|row| row.4 == Expect::Previous)
        .map(|row| unhex32(row.3.expect("a previous row has an identifier")))
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
    assert_eq!(last.4, Expect::Current);
    assert_eq!(unhex32(last.3.expect("stamped")), alice_db::SEMANTICS_ID);
}

#[test]
fn every_identifier_in_the_history_reads_as_its_row_says() {
    let mut rows = 0;
    for (name, release, _, stamp, expect) in HISTORY {
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
