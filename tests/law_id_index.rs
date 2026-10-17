//! Oracles for addressing a stored law by its content identifier
//!
//! `law_store` keys laws by `(name, version)`, which a person assigns and can
//! change. This module adds a second way in: the identifier
//! `SignalLaw::law_id` computes from the law's content, so a stored result can
//! name the law it came from without relying on a name staying put.
//!
//! The guarantee the identifier carries (from `alice-zip`) is one-directional:
//! **equal identifier implies `evaluate` returns the same bits for every `x`**.
//! That is what makes [`AliceDB::evaluate_by_id`] well defined even when one
//! identifier points at several `(name, version)` pairs — every one of them
//! evaluates identically, so picking any of them is the same answer.
//!
//! The converse does not hold, so an identifier is not a deduplication key.
//!
//! Expected values are written here, never read back from the store.

use alice_db::law_store::{
    law_id_key, law_version_key, IngestPolicy, LawPointersFoundUnder, LawStoreError, Provenance,
    ResidualStats, SignalLaw, SignalLawParts, ValidRange, Verdict,
};
use alice_db::{AliceDB, StorageConfig};

/// The identifier this build writes into a law record. A read checks the
/// record's identifier against the build's, so laws these oracles read back
/// are stored under it.
const SEMANTICS_A: [u8; 32] = alice_db::SEMANTICS_ID;
/// A made-up identifier no build reproduces, used only where a law is looked
/// up or indexed under it, never read back through it.
const SEMANTICS_B: [u8; 32] = [0x22; 32];

const POLICY: IngestPolicy = IngestPolicy {
    abs_tolerance: 0.5,
    break_factor: 4.0,
};

fn memory_db() -> AliceDB {
    AliceDB::in_memory(StorageConfig::default()).expect("in-memory database")
}

/// A law with the given coefficients over `[0, 4]`, and whatever evidence the
/// caller wants to attach. `from_parts` takes the coefficients as given, so two
/// calls with the same coefficients and domain describe the same function
/// however the evidence differs.
fn law_with(coefficients: Vec<f64>, evidence: Vec<(f64, f64)>, source: &str) -> SignalLaw {
    SignalLaw::from_parts(SignalLawParts {
        coefficients,
        domain: ValidRange { lo: 0.0, hi: 4.0 },
        evidence,
        residual: ResidualStats {
            n: 0,
            rms: 0.0,
            max_abs: 0.0,
        },
        provenance: Provenance::new(source, "hand-written"),
        oracles: Vec::new(),
    })
    .expect("the fixture satisfies every from_parts precondition")
}

/// `y = 1 + 2x` as a law, with the evidence that line was measured at
fn line_law(source: &str) -> SignalLaw {
    let evidence: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i)))
        .collect();
    // in the normalised variable u = x / 4: y = 1 + 8u
    law_with(vec![1.0, 8.0], evidence, source)
}

/// Every `x` the oracles sample, both endpoints of the domain included
fn grid() -> Vec<f64> {
    let mut xs = vec![0.0, 4.0];
    for i in 1..16_u32 {
        xs.push(f64::from(i) / 4.0);
    }
    xs
}

// ---------------------------------------------------------------------------
// 1. The point of content addressing
// ---------------------------------------------------------------------------

#[test]
fn same_content_under_two_names_shares_one_id_and_evaluates_identically() {
    let db = memory_db();

    // Same coefficients and domain, deliberately different evidence and source.
    let a = line_law("run 1");
    let b = law_with(
        vec![1.0, 8.0],
        vec![(0.5, 2.0), (1.5, 4.0), (2.5, 6.0), (3.5, 8.0)],
        "run 2 on other equipment",
    );

    let id = a.law_id(&SEMANTICS_A);
    assert_eq!(id, b.law_id(&SEMANTICS_A), "same content, one identifier");

    db.put_law("left", &a, &SEMANTICS_A).expect("stored a");
    db.put_law("right", &b, &SEMANTICS_A).expect("stored b");

    let mut pointers = db.law_pointers_by_id(&id).expect("pointers");
    pointers.sort();
    assert_eq!(
        pointers,
        vec![("left".to_string(), 1_u64), ("right".to_string(), 1_u64)],
        "both names must be reachable from the one identifier"
    );

    // The guarantee: whichever pointer the store picks, the answer is the same
    // bits as going through either name.
    for x in grid() {
        let by_id = db.evaluate_by_id(&id, x).expect("evaluate by id").value;
        let via_left = db.evaluate_law("left", x).expect("evaluate left").value;
        let via_right = db.evaluate_law("right", x).expect("evaluate right").value;
        assert_eq!(by_id.to_bits(), via_left.to_bits(), "by id vs left at {x}");
        assert_eq!(
            by_id.to_bits(),
            via_right.to_bits(),
            "by id vs right at {x}"
        );
    }
}

#[test]
fn put_law_round_trips_through_the_id_index() {
    let db = memory_db();
    let law = line_law("run 1");
    let id = law.law_id(&SEMANTICS_A);
    let version = db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    let pointers = db.law_pointers_by_id(&id).expect("pointers");
    assert_eq!(pointers, vec![("line".to_string(), version)]);

    // Restoring through the pointer gives a law with the same identifier.
    let restored = db
        .get_law_version("line", version)
        .expect("read")
        .expect("present")
        .law;
    assert_eq!(
        restored.law_id(&SEMANTICS_A),
        id,
        "identifier survives storage"
    );
}

// ---------------------------------------------------------------------------
// 2. The numeric semantics reach the key
// ---------------------------------------------------------------------------

#[test]
fn a_different_semantics_id_indexes_separately() {
    let db = memory_db();
    let law = line_law("run 1");
    db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    let under_a = law.law_id(&SEMANTICS_A);
    let under_b = law.law_id(&SEMANTICS_B);
    assert_ne!(
        under_a, under_b,
        "premise: the identifier binds the semantics"
    );

    assert_eq!(db.law_pointers_by_id(&under_a).expect("a").len(), 1);
    assert!(
        db.law_pointers_by_id(&under_b).expect("b").is_empty(),
        "a law stored under one numeric semantics must not answer for another"
    );
}

// ---------------------------------------------------------------------------
// 3. Every path that writes a version must index it
// ---------------------------------------------------------------------------

#[test]
fn ingest_evidence_also_indexes_the_version_it_creates() {
    let db = memory_db();
    let law = line_law("run 1");
    db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    // Evidence far enough off the line to force a refit, close enough that the
    // refitted line still explains it: `ingest_evidence` then stores the
    // refitted law as version 2.
    let moved: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.6 + 2.0 * f64::from(i)))
        .collect();
    let verdict = db
        .ingest_evidence("line", &moved, &POLICY, &SEMANTICS_A)
        .expect("ingested");
    let Verdict::ParameterUpdate {
        updated: refitted, ..
    } = &verdict
    else {
        panic!("expected a parameter update, got {verdict:?}");
    };

    let versions = db.law_versions("line").expect("versions");
    assert_eq!(versions, vec![1, 2], "the refit is stored as version 2");

    let new_id = refitted.law_id(&SEMANTICS_A);
    assert_eq!(
        db.law_pointers_by_id(&new_id).expect("pointers"),
        vec![("line".to_string(), 2_u64)],
        "the version `ingest_evidence` wrote must be in the index too, \
         otherwise the index is complete only for `put_law`"
    );
}

// ---------------------------------------------------------------------------
// 4. The index is part of the stored state
// ---------------------------------------------------------------------------

#[test]
fn the_index_survives_to_bytes_and_from_bytes() {
    let db = memory_db();
    let law = line_law("run 1");
    let id = law.law_id(&SEMANTICS_A);
    db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    let bytes = db.to_bytes().expect("serialised");
    let restored = AliceDB::from_bytes(StorageConfig::default(), &bytes).expect("deserialised");

    assert_eq!(
        restored.law_pointers_by_id(&id).expect("pointers"),
        vec![("line".to_string(), 1_u64)],
        "the identifier index must be carried by the serialised form"
    );
    for x in grid() {
        assert_eq!(
            restored
                .evaluate_by_id(&id, x)
                .expect("by id")
                .value
                .to_bits(),
            db.evaluate_by_id(&id, x).expect("by id").value.to_bits(),
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Keys stay apart
// ---------------------------------------------------------------------------

#[test]
fn index_keys_cannot_collide_with_name_keys() {
    // Names are non-empty and hold no NUL, so the byte after the shared prefix
    // is never NUL for a name key and always NUL for an index key.
    let id = [0xAB_u8; 32];
    for name in ["i", "v", "h", "iv", "line", "\u{1F600}"] {
        let idx = law_id_key(&id, name, 1).expect("index key");
        let ver = law_version_key(name, 1).expect("version key");
        assert!(
            !idx.starts_with(&ver),
            "index key {name} contains a version key"
        );
        assert!(
            !ver.starts_with(&idx),
            "version key {name} contains an index key"
        );
        assert_ne!(idx, ver);
    }

    // Two names that differ only in where the boundary falls must not collide:
    // the name is length-prefixed inside the index key.
    let a = law_id_key(&id, "ab", 1).expect("a");
    let b = law_id_key(&id, "a", 1).expect("b");
    assert_ne!(a, b);
    assert!(!a.starts_with(&b), "the name needs a length prefix");
}

// ---------------------------------------------------------------------------
// 6. Degenerate input
// ---------------------------------------------------------------------------

#[test]
fn an_unknown_identifier_is_empty_rather_than_an_error() {
    let db = memory_db();
    db.put_law("line", &line_law("run 1"), &SEMANTICS_A)
        .expect("stored");

    assert!(
        db.law_pointers_by_id(&[0_u8; 32])
            .expect("all zero")
            .is_empty(),
        "an identifier nothing was stored under is absence, not corruption"
    );
    assert!(db
        .law_pointers_by_id(&[0xFF_u8; 32])
        .expect("unknown")
        .is_empty());
    // The kind of failure is part of the contract, not just that it fails: a
    // caller retries a lookup that came up empty, but repairs a store that
    // reports damage. Asserting only `is_err` leaves the two interchangeable.
    assert!(
        matches!(
            db.evaluate_by_id(&[0_u8; 32], 1.0),
            Err(LawStoreError::NotFound)
        ),
        "an identifier with no law is absence, not a damaged store"
    );
}

#[test]
fn law_id_key_rejects_the_names_the_store_rejects() {
    let id = [1_u8; 32];
    assert!(matches!(
        law_id_key(&id, "", 1),
        Err(LawStoreError::InvalidName)
    ));
    assert!(matches!(
        law_id_key(&id, "a\0b", 1),
        Err(LawStoreError::InvalidName)
    ));
}

// ---------------------------------------------------------------------------
// 7. Golden: the key layout itself
//
// Pins the shared prefix, the tag, the field order, the length prefix and the
// byte order, so an independent implementation can reproduce a key.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 8. Reading a law without knowing which semantics identifier wrote it
//
// `law_pointers_for` tries the current identifier, then every one
// `alice_core::law::check_semantics` still accepts as reproducing the same
// bits (`alice_core::law::PREVIOUS_SEMANTICS_IDS`); `reindex_laws_under_current_semantics`
// writes a pointer under the current identifier for whatever it finds that
// way, without touching the pointer it found.
//
// The crate this test runs against pins `alice-det-math` 0.5.0, whose
// `PREVIOUS_SEMANTICS_IDS` is still empty (0.5.0 is the first release of the
// arithmetic these oracles are pinned to), so the fallback loop's body never
// executes here and a mutant that removes it is not caught by this file —
// that case is checked locally against a `[patch.crates-io]` override to the
// unpublished 0.5.1 that populates the list, the same method used for
// `check_semantics`'s own forward-looking test, and is not committed. What
// this file does pin: the current-identifier fast path, the empty-list
// behaviour of the fallback (so a change that panics or errors on an empty
// `PREVIOUS_SEMANTICS_IDS` is caught here), and `reindex`'s additive,
// idempotent write.
// ---------------------------------------------------------------------------

#[test]
fn law_pointers_for_finds_a_fresh_write_under_the_current_identifier() {
    let db = memory_db();
    let law = line_law("run 1");
    let version = db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    let (pointers, found_under) = db.law_pointers_for(&law, &SEMANTICS_A).expect("lookup");
    assert_eq!(pointers, vec![("line".to_string(), version)]);
    assert_eq!(
        found_under,
        LawPointersFoundUnder::Current,
        "a law stored under the identifier it is looked up with is found as Current, \
         never as Previous"
    );
}

#[test]
fn law_pointers_for_misses_when_stored_under_neither_current_nor_any_previous_identifier() {
    let db = memory_db();
    let law = line_law("run 1");
    db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    // SEMANTICS_B is a different made-up identifier, not one
    // `alice_core::law::PREVIOUS_SEMANTICS_IDS` contains (that list only ever
    // holds identifiers the arithmetic crate itself published), so looking the
    // same law up under it must come back empty regardless of how many
    // previous identifiers the running crate happens to carry.
    let (pointers, found_under) = db.law_pointers_for(&law, &SEMANTICS_B).expect("lookup");
    assert!(
        pointers.is_empty(),
        "stored only under SEMANTICS_A, must not answer for SEMANTICS_B or any real previous id"
    );
    assert_eq!(
        found_under,
        LawPointersFoundUnder::Current,
        "the marker is meaningless next to an empty list, but is defined to be Current"
    );
}

#[test]
fn law_pointers_for_tries_the_current_identifier_before_any_previous_one() {
    // Even if a previous identifier also happened to address this law (not
    // constructible here without a real one from `alice_det_math`, see the
    // module doc above), the current identifier must win when it alone is
    // non-empty: cheapest and most specific answer first.
    let db = memory_db();
    let law = line_law("run 1");
    db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    let (_, found_under) = db.law_pointers_for(&law, &SEMANTICS_A).expect("lookup");
    assert_eq!(found_under, LawPointersFoundUnder::Current);
}

#[test]
fn reindex_leaves_a_law_stored_under_an_unaccepted_identifier_alone() {
    // SEMANTICS_A is not the identifier passed as "current" here, and it is
    // not in `alice_core::law::PREVIOUS_SEMANTICS_IDS` either (that list only
    // ever holds identifiers the arithmetic crate itself published as
    // bit-reproducing) — exactly the shape of a `SemanticsCheck::Mismatch`.
    // Reindex must not manufacture a pointer for it: doing so would assert a
    // bit-equivalence between SEMANTICS_A and SEMANTICS_B that nothing backs.
    let db = memory_db();
    let law = line_law("run 1");
    let version = db.put_law("line", &law, &SEMANTICS_A).expect("stored");

    let added = db
        .reindex_laws_under_current_semantics(&SEMANTICS_B)
        .expect("reindex");
    assert_eq!(
        added, 0,
        "SEMANTICS_A is accepted by neither the current id nor any previous one"
    );
    assert!(
        db.law_pointers_by_id(&law.law_id(&SEMANTICS_B))
            .expect("after")
            .is_empty(),
        "no pointer must appear under the identifier reindex was given as current"
    );
    assert_eq!(
        db.law_pointers_by_id(&law.law_id(&SEMANTICS_A))
            .expect("still under A"),
        vec![("line".to_string(), version)],
        "the original pointer is untouched"
    );
}

#[test]
fn reindex_is_a_noop_when_already_reachable_under_the_current_identifier() {
    let db = memory_db();
    db.put_law("line", &line_law("run 1"), &SEMANTICS_A)
        .expect("v1");
    let other = law_with(vec![2.0, -1.0], vec![(0.0, 2.0), (4.0, 1.0)], "run 2");
    db.put_law("other", &other, &SEMANTICS_A).expect("other v1");
    let moved: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.6 + 2.0 * f64::from(i)))
        .collect();
    db.ingest_evidence("line", &moved, &POLICY, &SEMANTICS_A)
        .expect("refit to v2");
    assert_eq!(db.law_versions("line").expect("versions"), vec![1, 2]);

    // Every stored (name, version) is already under SEMANTICS_A, the identity
    // being reindexed to, so nothing is added — and running it again changes
    // nothing either (idempotent on the trivially-already-current case too).
    for attempt in 0..2 {
        let added = db
            .reindex_laws_under_current_semantics(&SEMANTICS_A)
            .unwrap_or_else(|e| panic!("reindex attempt {attempt}: {e:?}"));
        assert_eq!(added, 0, "attempt {attempt}: all three are already current");
    }
}

#[test]
fn law_id_key_layout_golden() {
    let id: [u8; 32] = core::array::from_fn(|i| u8::try_from(i).expect("the index is below 32"));
    let key = law_id_key(&id, "ab", 0x0102_0304_0506_0708).expect("key");

    let mut expected = Vec::new();
    expected.extend_from_slice(b"\0alice-law\0");
    expected.extend_from_slice(b"\0i");
    expected.extend_from_slice(&id);
    expected.extend_from_slice(&2_u32.to_be_bytes());
    expected.extend_from_slice(b"ab");
    expected.extend_from_slice(&0x0102_0304_0506_0708_u64.to_be_bytes());

    assert_eq!(
        key, expected,
        "the key layout is part of the stored format: changing it needs a new tag"
    );
}
