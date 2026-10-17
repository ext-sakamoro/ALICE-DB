//! Content hash of this crate's own arithmetic: [`DB_SEMANTICS_ID`].
//!
//! The recorded-bit tests in `tests/determinism_golden.rs` pin the bits of
//! the decisions this crate computes itself: which model a segment stores and
//! what it reads back, the aggregates over stored values, the size of a
//! bloom filter and the identifiers and values of stored laws. This module folds those digests, plus the semantics
//! identifier of `alice-det-math` (the transcendental functions the models
//! are evaluated with), into one 32-byte value. Two builds with the same
//! identifier compute every pinned scenario to the same bits; a build that
//! computes any of them differently has a different value.
//!
//! [`crate::SEMANTICS_ID`] is a different identifier: the arithmetic a stored
//! residual or law record was measured against (the `alice-det-math`
//! identifier), which is what a reader compares a stored value with.
//!
//! # Hex form (with `alice-det-math` 0.5.1)
//!
//! ```text
//! aab1eb3d22da13c75783c8d306feb7fa5dc8135b5d8625f315f19d68e051162f
//! ```
//!
//! # Fold
//!
//! The procedure of the `SEMANTICS_ID` fold of `alice-det-math`: sort the
//! entries by name (ascending byte order), then SHA-256 over, for each entry
//! in that order, the name length as 4 bytes big-endian (`u32`), the name
//! bytes and the 32 digest bytes.
//!
//! # Independent of cargo features
//!
//! The table and the identifier are constants that no `cfg` gates, so a
//! build with any feature set reports the same identifier.
//!
//! # Staleness
//!
//! [`DB_SEMANTICS_CHECKED_AT`] names the crate version and toolchain the pins
//! were last checked on; a different version or toolchain fails the test
//! below until they are checked again. An `alice-det-math` generation with no
//! row in [`DB_SEMANTICS_BY_DET_MATH`] does not compile.
//!
//! # Checked by
//!
//! `tests/db_semantics_id.rs` reads every digest from the golden test that
//! records it (the source, not a copy), checks that the det-math entry equals
//! `alice_det_math::SEMANTICS_ID` and recomputes the fold. Re-recording order
//! after an intended change: update the golden first, then the entry here,
//! then the identifier.

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a
/// compile error.
const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The entries [`DB_SEMANTICS_ID`] folds: one per recorded-bit scenario of
/// `tests/determinism_golden.rs`, named as that file names it, with its
/// SHA-256 golden, plus `"alice-det-math"` holding
/// `alice_det_math::SEMANTICS_ID`.
pub const DB_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "model selection",
        hex32("d6c3e1e7fe4031262f743c0b326c57b28e34857bafd7e0f3e37e256d304ea160"),
    ),
    (
        "query_point and query_range",
        hex32("84e4d8d5dec997c81798274963b5375e42cd70bd32a9b41bff76e3b46fb3c07c"),
    ),
    (
        "aggregates",
        hex32("113119281a0da813f3d1362e2d6a8a7d87b1bf2c352f0cca5088ea5103ffbb40"),
    ),
    (
        "bloom sizing",
        hex32("9d7933ea59b2d3fad65aa0633af3a39d99c44448aa1af688fdd7913bdc1393bf"),
    ),
    (
        "law_id and evaluate",
        hex32("703f17b165de1699421256302a23582a65cf3904f7ddc14e4c37365b0e444127"),
    ),
    // the resolved alice-det-math's own identifier, not a copy of it: the
    // identifier below is looked up by it
    ("alice-det-math", alice_det_math::SEMANTICS_ID),
];

/// The crate version and toolchain the pins of [`DB_SEMANTICS_PINS`] were last
/// checked against. `tests/db_semantics_id.rs` fails when `Cargo.toml`'s
/// version or `rust-toolchain.toml`'s channel differs, so a release or a
/// toolchain move cannot ship without re-running the recorded-bit tests and
/// then updating these two values (or the pins and the identifier, if a
/// recorded scenario moved).
pub const DB_SEMANTICS_CHECKED_AT: (&str, &str) = ("0.6.0", "1.98.1");

/// [`DB_SEMANTICS_ID`] for each `alice-det-math` generation this crate
/// knows: `(alice_det_math::SEMANTICS_ID, fold of DB_SEMANTICS_PINS with that
/// det-math entry)`. A build against a generation with no row does not
/// compile, rather than reporting an identifier of another generation.
pub const DB_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.1
    (
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        hex32("aab1eb3d22da13c75783c8d306feb7fa5dc8135b5d8625f315f19d68e051162f"),
    ),
];

/// Content hash of this crate's own arithmetic: the fold of
/// [`DB_SEMANTICS_PINS`] described in the [module documentation](self),
/// taken from [`DB_SEMANTICS_BY_DET_MATH`] by the resolved `alice-det-math`.
///
/// Hex (with `alice-det-math` 0.5.1):
/// `aab1eb3d22da13c75783c8d306feb7fa5dc8135b5d8625f315f19d68e051162f`.
pub const DB_SEMANTICS_ID: [u8; 32] = semantics_id_for(&alice_det_math::SEMANTICS_ID);

/// The [`DB_SEMANTICS_ID`] of a build that resolved the `alice-det-math` with
/// identifier `det_math`.
///
/// # Panics
///
/// When [`DB_SEMANTICS_BY_DET_MATH`] has no row for `det_math` (in a const
/// context this stops the compile).
#[must_use]
pub const fn semantics_id_for(det_math: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < DB_SEMANTICS_BY_DET_MATH.len() {
        if same32(&DB_SEMANTICS_BY_DET_MATH[i].0, det_math) {
            return DB_SEMANTICS_BY_DET_MATH[i].1;
        }
        i += 1;
    }
    panic!("alice-det-math SEMANTICS_ID has no row in DB_SEMANTICS_BY_DET_MATH: add its fold");
}

const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
