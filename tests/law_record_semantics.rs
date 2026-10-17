//! Oracles for the semantics identifier a law record carries (format 2)
//!
//! A record stores the identifier of the arithmetic it was written under. A
//! read compares it with the identifier of the running build:
//!
//! - equal: [`RecordSemantics::Current`]
//! - an identifier the arithmetic crate verifies as reproducing the same bits:
//!   [`RecordSemantics::VerifiedPrevious`]
//! - anything else: refused with [`LawStoreError::LawRecordMismatch`], naming
//!   both identifiers
//! - a format 1 record (written before the identifier existed):
//!   [`RecordSemantics::Unpinned`], or refused under
//!   [`LawReadPolicy::RequirePinned`]
//!
//! The format 1 record below is a fixture: the bytes the 0.5 encoder wrote for
//! the line `y = 1 + 2x` fitted at `x = 0..=4`, copied here, not produced by
//! this build.

use alice_db::law_store::{
    decode_law_record, encode_law_record, law_version_key, LawReadPolicy, LawStoreError,
    Provenance, RecordSemantics, SignalLaw,
};
use alice_db::{AliceDB, StorageConfig, SEMANTICS_ID};

/// `encode_law_record("ohm", 1, …)` of alice-db 0.5 for [`line_law`]
const V1_RECORD_HEX: &str = "414c41575245430101000000030000006f686d010000000000000002000000040000000000f03fffffffffffff1f400000000000000000000000000000104005000000000000000000000000000000000000000000f03f000000000000f03f00000000000008400000000000000040000000000000144000000000000008400000000000001c400000000000001040000000000000224005000000000000000f494862133dc43c000000000000d03c0b00000062656e63682072756e20310d0000006c6561737420737175617265730000000010aef789";

/// An identifier no build reproduces
const FOREIGN: [u8; 32] = [0x22; 32];

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn line_law() -> SignalLaw {
    let pts: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i)))
        .collect();
    SignalLaw::fit_polynomial(&pts, 1, Provenance::new("bench run 1", "least squares"))
        .expect("five distinct points determine a line")
}

fn memory_db() -> AliceDB {
    AliceDB::in_memory(StorageConfig::default()).expect("in-memory database")
}

/// A store holding [`line_law`] as "ohm" version 1, with the version record
/// replaced by `record`. `put_law` first, so the name and the identifier
/// index exist as they would for a record written by that build.
fn db_with_record(record: &[u8]) -> AliceDB {
    let db = memory_db();
    db.put_law("ohm", &line_law(), &SEMANTICS_ID)
        .expect("stored");
    db.put_blob(&law_version_key("ohm", 1).expect("key"), record)
        .expect("replaced");
    db
}

fn is_mismatch(err: &LawStoreError) -> bool {
    matches!(
        err,
        LawStoreError::LawRecordMismatch { stored, running }
            if *stored == FOREIGN && *running == SEMANTICS_ID
    )
}

// --- format 2 written by this build ---------------------------------------

#[test]
fn a_record_this_build_writes_reads_back_as_current() {
    let db = memory_db();
    db.put_law("ohm", &line_law(), &SEMANTICS_ID)
        .expect("stored");
    for policy in [LawReadPolicy::AcceptUnpinned, LawReadPolicy::RequirePinned] {
        let stored = db
            .get_law_with("ohm", policy)
            .expect("read")
            .expect("present");
        assert_eq!(stored.semantics, RecordSemantics::Current);
        let v = db.evaluate_law_with("ohm", 2.5, policy).expect("evaluated");
        // 1 + 2 · 2.5
        assert!((v.value - 6.0).abs() < 1e-12);
        assert_eq!(v.semantics, RecordSemantics::Current);
    }
    let id = line_law().law_id(&SEMANTICS_ID);
    let by_id = db
        .evaluate_by_id_with(&id, 2.5, LawReadPolicy::RequirePinned)
        .expect("by id");
    assert_eq!(by_id.semantics, RecordSemantics::Current);
}

#[test]
fn the_record_holds_the_identifier_it_was_written_under() {
    let rec = encode_law_record("ohm", 1, &line_law().to_parts(), &SEMANTICS_ID).unwrap();
    // magic 8 + format 4 + name length 4 + "ohm" 3 + version 8 = 27
    assert_eq!(&rec[8..12], &2u32.to_le_bytes(), "format 2");
    assert_eq!(&rec[27..59], &SEMANTICS_ID, "identifier after the version");
    let (_, _, _, semantics) = decode_law_record(&rec).unwrap();
    assert_eq!(semantics, Some(SEMANTICS_ID));
}

// --- format 2 written under an identifier this build does not reproduce ----

#[test]
fn a_record_under_a_foreign_identifier_is_refused_naming_both() {
    let rec = encode_law_record("ohm", 1, &line_law().to_parts(), &FOREIGN).unwrap();
    let db = db_with_record(&rec);
    for policy in [LawReadPolicy::AcceptUnpinned, LawReadPolicy::RequirePinned] {
        let err = db.get_law_with("ohm", policy).expect_err("refused");
        assert!(is_mismatch(&err), "{err:?}");
        let err = db
            .get_law_version_with("ohm", 1, policy)
            .expect_err("refused");
        assert!(is_mismatch(&err), "{err:?}");
        let err = db
            .evaluate_law_with("ohm", 2.5, policy)
            .expect_err("refused");
        assert!(is_mismatch(&err), "{err:?}");
    }
    let id = line_law().law_id(&SEMANTICS_ID);
    let err = db.evaluate_by_id(&id, 2.5).expect_err("refused");
    assert!(is_mismatch(&err), "{err:?}");
    let shown = err.to_string();
    assert!(shown.contains(&hex(&FOREIGN)), "{shown}");
    assert!(shown.contains(&hex(&SEMANTICS_ID)), "{shown}");
}

/// The mismatch is a refusal, not a downgrade: a format 2 record with a wrong
/// identifier must not come back as `Unpinned` (which a lenient caller
/// accepts)
#[test]
fn a_foreign_identifier_is_never_read_as_unpinned() {
    let rec = encode_law_record("ohm", 1, &line_law().to_parts(), &FOREIGN).unwrap();
    let db = db_with_record(&rec);
    assert!(db.get_law("ohm").is_err());
    assert!(db.evaluate_law("ohm", 2.5).is_err());
}

// --- format 1 written before the identifier existed ------------------------

#[test]
fn the_v1_fixture_is_the_v2_layout_without_the_identifier() {
    let v1 = unhex(V1_RECORD_HEX);
    let v2 = encode_law_record("ohm", 1, &line_law().to_parts(), &SEMANTICS_ID).unwrap();
    assert_eq!(&v1[..8], &v2[..8], "same magic");
    assert_eq!(&v1[8..12], &1u32.to_le_bytes(), "format 1");
    assert_eq!(&v1[12..27], &v2[12..27], "name and version");
    // the body after the identifier, without the CRC tails
    assert_eq!(&v1[27..v1.len() - 4], &v2[59..v2.len() - 4]);
}

#[test]
fn a_v1_record_decodes_without_an_identifier() {
    let (name, version, parts, semantics) = decode_law_record(&unhex(V1_RECORD_HEX)).unwrap();
    assert_eq!((name.as_str(), version), ("ohm", 1));
    assert_eq!(parts.coefficients.len(), 2);
    assert_eq!(semantics, None);
}

#[test]
fn a_v1_record_reads_as_unpinned_and_evaluates() {
    let db = db_with_record(&unhex(V1_RECORD_HEX));
    let stored = db.get_law("ohm").expect("read").expect("present");
    assert_eq!(stored.semantics, RecordSemantics::Unpinned);
    let v = db.evaluate_law("ohm", 2.5).expect("evaluated");
    assert!((v.value - 6.0).abs() < 1e-12);
    assert_eq!(v.semantics, RecordSemantics::Unpinned);
    let id = line_law().law_id(&SEMANTICS_ID);
    assert_eq!(
        db.evaluate_by_id(&id, 2.5).expect("by id").semantics,
        RecordSemantics::Unpinned
    );
}

#[test]
fn a_v1_record_is_refused_when_a_pin_is_required() {
    let db = db_with_record(&unhex(V1_RECORD_HEX));
    let p = LawReadPolicy::RequirePinned;
    assert!(matches!(
        db.get_law_with("ohm", p),
        Err(LawStoreError::Unpinned)
    ));
    assert!(matches!(
        db.get_law_version_with("ohm", 1, p),
        Err(LawStoreError::Unpinned)
    ));
    assert!(matches!(
        db.evaluate_law_with("ohm", 2.5, p),
        Err(LawStoreError::Unpinned)
    ));
    let id = line_law().law_id(&SEMANTICS_ID);
    assert!(matches!(
        db.evaluate_by_id_with(&id, 2.5, p),
        Err(LawStoreError::Unpinned)
    ));
}

#[test]
fn the_default_policy_accepts_unpinned() {
    assert_eq!(LawReadPolicy::default(), LawReadPolicy::AcceptUnpinned);
}

#[test]
fn ingest_on_a_v1_record_writes_a_pinned_next_version() {
    let db = db_with_record(&unhex(V1_RECORD_HEX));
    let policy = alice_db::law_store::IngestPolicy {
        abs_tolerance: 0.5,
        break_factor: 4.0,
    };
    // the line plus 0.6: past the 0.5 tolerance, and a refitted line explains
    // it exactly, so a parameter update (as in tests/law_store.rs)
    let shifted: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.6 + 2.0 * f64::from(i)))
        .collect();
    db.ingest_evidence("ohm", &shifted, &policy, &SEMANTICS_ID)
        .expect("judged");
    assert_eq!(db.law_versions("ohm").unwrap(), vec![1, 2]);
    let latest = db
        .get_law_with("ohm", LawReadPolicy::RequirePinned)
        .expect("read")
        .expect("present");
    assert_eq!(latest.semantics, RecordSemantics::Current);
}

// --- format 2 written under a verified earlier identifier -------------------

/// alice-det-math 0.5.0's identifier, which 0.5.1 lists as reproducing the
/// same bits
const PREVIOUS_HEX: &str = "135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000";

#[test]
fn a_record_under_a_verified_earlier_identifier_reads_as_verified_previous() {
    let previous: [u8; 32] = unhex(PREVIOUS_HEX).try_into().expect("32 bytes");
    assert!(
        alice_det_math::PREVIOUS_SEMANTICS_IDS.contains(&previous),
        "the resolved alice-det-math no longer lists the fixture's identifier"
    );
    let rec = encode_law_record("ohm", 1, &line_law().to_parts(), &previous).unwrap();
    let db = db_with_record(&rec);
    for policy in [LawReadPolicy::AcceptUnpinned, LawReadPolicy::RequirePinned] {
        let stored = db
            .get_law_with("ohm", policy)
            .expect("read")
            .expect("present");
        assert_eq!(
            stored.semantics,
            RecordSemantics::VerifiedPrevious(previous)
        );
        let v = db.evaluate_law_with("ohm", 2.5, policy).expect("evaluated");
        assert!((v.value - 6.0).abs() < 1e-12);
        assert_eq!(v.semantics, RecordSemantics::VerifiedPrevious(previous));
    }
}
