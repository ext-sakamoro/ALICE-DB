//! Content hash of this crate's own arithmetic: [`DB_SEMANTICS_ID`].
//!
//! The recorded-bit tests in `tests/determinism_golden.rs` pin the bits of
//! the decisions this crate computes itself: which model a segment stores and
//! what it reads back, the aggregates over stored values, the size of a
//! bloom filter and the identifiers and values of stored laws. This module folds those digests, plus the semantics
//! identifier of `alice-zip` (whose models and laws this crate stores and
//! evaluates), into one 32-byte value. Two builds with the same identifier
//! compute every pinned scenario to the same bits; a build that computes any
//! of them differently has a different value.
//!
//! `alice-zip`'s identifier is itself a fold of `alice-det-math`'s (the
//! transcendental functions the models are evaluated with) and of its own
//! pinned outputs, so the `alice-det-math` generation is covered through it:
//! each `alice-det-math` identifier maps to one `alice-zip` identifier
//! (`tests/db_semantics_id.rs` pins that the resolved one is reachable).
//!
//! [`crate::SEMANTICS_ID`] is a different identifier: the arithmetic a stored
//! residual or law record was measured against (`alice-zip`'s), which is what a
//! reader compares a stored value with.
//!
//! # Hex form (with `alice-zip` 0.11 at 19f6f2c1, over `alice-det-math` 0.5.2 at 2d01c63f)
//!
//! ```text
//! 670fa328eed6ddf2e28f96eb5ebabfe602eabd80dd025199edece388241223a7
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
//! below until they are checked again. An `alice-zip` generation with no row
//! in [`DB_SEMANTICS_BY_ZIP`] does not compile.
//!
//! # Checked by
//!
//! `tests/db_semantics_id.rs` reads every digest from the golden test that
//! records it (the source, not a copy), checks that the alice-zip entry equals
//! `alice_core::law::SEMANTICS_ID` and recomputes the fold. Re-recording order
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
/// SHA-256 golden, plus `"alice-zip"` holding `alice_core::law::SEMANTICS_ID`.
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
        hex32("2536bbeb81fd687b2289b3f257f2dcade86d1ebd1ccbcf83d8ca0614a12fb8b9"),
    ),
    // the resolved alice-zip's own identifier, not a copy of it: the
    // identifier below is looked up by it
    ("alice-zip", alice_core::law::SEMANTICS_ID),
];

/// The crate version and toolchain the pins of [`DB_SEMANTICS_PINS`] were last
/// checked against. `tests/db_semantics_id.rs` fails when `Cargo.toml`'s
/// version or `rust-toolchain.toml`'s channel differs, so a release or a
/// toolchain move cannot ship without re-running the recorded-bit tests and
/// then updating these two values (or the pins and the identifier, if a
/// recorded scenario moved).
pub const DB_SEMANTICS_CHECKED_AT: (&str, &str) = ("0.6.0", "1.98.1");

/// [`DB_SEMANTICS_ID`] for each `alice-zip` generation this crate knows:
/// `(alice_core::law::SEMANTICS_ID, fold of DB_SEMANTICS_PINS with that
/// alice-zip entry)`. A build against a generation with no row does not
/// compile, rather than reporting an identifier of another generation.
pub const DB_SEMANTICS_BY_ZIP: &[([u8; 32], [u8; 32])] = &[
    // alice-zip 0.11 (over alice-det-math 0.5.1)
    (
        hex32("51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e"),
        hex32("e1a82c6407b1a880267b354647891271929bf2a5765a93dc48944ff64237c4ce"),
    ),
    // alice-zip at 8065587c (over alice-det-math 0.5.2 at 39507d1b)
    (
        hex32("50a40cf64625018acc99c940e544353b901457b4a7a80bf903544ed4ef9af406"),
        hex32("6e8b456ac72cbfad294beb0f79104762555660ff8f3c40ac61d7d8358e655095"),
    ), // alice-zip at 19f6f2c1 (over alice-det-math 0.5.2 at 2d01c63f)
    (
        hex32("17d0df1d7c0a6e63d1a7380129f12386b677ac96e7ac193dece546805f4fe84c"),
        hex32("670fa328eed6ddf2e28f96eb5ebabfe602eabd80dd025199edece388241223a7"),
    ),
];

/// Content hash of this crate's own arithmetic: the fold of
/// [`DB_SEMANTICS_PINS`] described in the [module documentation](self),
/// taken from [`DB_SEMANTICS_BY_ZIP`] by the resolved `alice-zip`.
///
/// Hex (with `alice-zip` 0.11):
/// `e1a82c6407b1a880267b354647891271929bf2a5765a93dc48944ff64237c4ce`.
pub const DB_SEMANTICS_ID: [u8; 32] = semantics_id_for(&alice_core::law::SEMANTICS_ID);

/// The [`DB_SEMANTICS_ID`] of a build that resolved the `alice-zip` with
/// identifier `zip`.
///
/// # Panics
///
/// When [`DB_SEMANTICS_BY_ZIP`] has no row for `zip` (in a const context this
/// stops the compile).
#[must_use]
pub const fn semantics_id_for(zip: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < DB_SEMANTICS_BY_ZIP.len() {
        if same32(&DB_SEMANTICS_BY_ZIP[i].0, zip) {
            return DB_SEMANTICS_BY_ZIP[i].1;
        }
        i += 1;
    }
    panic!("alice-zip SEMANTICS_ID has no row in DB_SEMANTICS_BY_ZIP: add its fold");
}

pub(crate) const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
