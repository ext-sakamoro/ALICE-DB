//! Oracles for the law store (`alice_db::law_store`)
//!
//! Expected values are closed forms written here, never values computed by
//! the store: the evidence is `y = 1 + 2x` at `x = 0, 1, 2, 3, 4`, and each
//! piece of new evidence is that line plus a known perturbation whose effect on
//! a straight-line least-squares fit is worked out in the comment next to it.

use alice_db::law_store::{
    decode_law_record, encode_law_record, law_version_key, IngestPolicy, LawError, LawStoreError,
    Provenance, SignalLaw, SignalLawParts, Verdict, VerdictKind,
};
use alice_db::{AliceDB, StorageConfig};

/// `y = 1 + 2x` at `x = 0..=4`
fn line_points() -> Vec<(f64, f64)> {
    (0..5)
        .map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i)))
        .collect()
}

fn line_law() -> SignalLaw {
    SignalLaw::fit_polynomial(
        &line_points(),
        1,
        Provenance::new("bench run 1", "least squares"),
    )
    .expect("five distinct points determine a line")
}

const POLICY: IngestPolicy = IngestPolicy {
    abs_tolerance: 0.5,
    break_factor: 4.0,
};

fn memory_db() -> AliceDB {
    AliceDB::in_memory(StorageConfig::default()).expect("in-memory database")
}

/// Field-by-field bit comparison of two parts (f64 compared by bit pattern)
fn assert_parts_bit_identical(a: &SignalLawParts, b: &SignalLawParts) {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.coefficients), bits(&b.coefficients), "coefficients");
    assert_eq!(a.domain.lo.to_bits(), b.domain.lo.to_bits(), "domain.lo");
    assert_eq!(a.domain.hi.to_bits(), b.domain.hi.to_bits(), "domain.hi");
    let ev = |v: &[(f64, f64)]| {
        v.iter()
            .map(|(x, y)| (x.to_bits(), y.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(ev(&a.evidence), ev(&b.evidence), "evidence");
    assert_eq!(a.residual.n, b.residual.n, "residual.n");
    assert_eq!(
        a.residual.rms.to_bits(),
        b.residual.rms.to_bits(),
        "residual.rms"
    );
    assert_eq!(
        a.residual.max_abs.to_bits(),
        b.residual.max_abs.to_bits(),
        "residual.max_abs"
    );
    assert_eq!(a.provenance, b.provenance, "provenance");
    assert_eq!(a.oracles.len(), b.oracles.len(), "oracle count");
    for (p, q) in a.oracles.iter().zip(&b.oracles) {
        assert_eq!(p.x.to_bits(), q.x.to_bits());
        assert_eq!(p.expected.to_bits(), q.expected.to_bits());
        assert_eq!(p.tolerance.to_bits(), q.tolerance.to_bits());
        assert_eq!(p.source, q.source);
    }
}

// ---------------------------------------------------------------- round trip

#[test]
fn memory_round_trip_is_bit_identical() {
    let db = memory_db();
    let law = line_law().with_oracle(alice_db::law_store::OracleCase::new(
        2.0, 5.0, 1e-9, "table 1",
    ));
    db.put_law("ohm", &law).unwrap();
    let back = db.get_law("ohm").unwrap().expect("stored law");
    assert_parts_bit_identical(&law.to_parts(), &back.to_parts());
    assert_eq!(db.law_versions("ohm").unwrap(), vec![1]);
    assert_eq!(db.law_names().unwrap(), vec!["ohm".to_string()]);
}

#[test]
fn snapshot_bytes_carry_laws_to_a_memory_database() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    db.ingest_evidence("ohm", &[(2.0, 5.0)], &POLICY).unwrap();
    let bytes = db.to_bytes().unwrap();
    let restored = AliceDB::from_bytes(StorageConfig::default(), &bytes).unwrap();
    assert_parts_bit_identical(
        &line_law().to_parts(),
        &restored.get_law("ohm").unwrap().unwrap().to_parts(),
    );
    assert_eq!(restored.law_history("ohm").unwrap().len(), 1);
}

#[cfg(feature = "fs")]
#[test]
fn file_round_trip_survives_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let law = line_law();
    {
        let db = AliceDB::open(dir.path()).unwrap();
        db.put_law("ohm", &law).unwrap();
        db.close().unwrap();
    }
    {
        let db = AliceDB::open(dir.path()).unwrap();
        let back = db.get_law("ohm").unwrap().expect("law survives reopen");
        assert_parts_bit_identical(&law.to_parts(), &back.to_parts());
        // write after reopen: offset +0.6 → parameter update (see below)
        let shifted: Vec<(f64, f64)> = line_points().iter().map(|&(x, y)| (x, y + 0.6)).collect();
        let v = db.ingest_evidence("ohm", &shifted, &POLICY).unwrap();
        assert!(matches!(v, Verdict::ParameterUpdate { .. }), "{v:?}");
        db.close().unwrap();
    }
    let db = AliceDB::open(dir.path()).unwrap();
    assert_eq!(db.law_versions("ohm").unwrap(), vec![1, 2]);
    assert_parts_bit_identical(
        &law.to_parts(),
        &db.get_law_version("ohm", 1).unwrap().unwrap().to_parts(),
    );
    let hist = db.law_history("ohm").unwrap();
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0].outcome, VerdictKind::ParameterUpdate);
    assert_eq!(hist[0].new_version, Some(2));
    // 1.3 + 2x at x = 2
    assert!((db.evaluate_law("ohm", 2.0).unwrap() - 5.3).abs() < 1e-12);
}

#[cfg(feature = "fs")]
#[test]
fn file_backend_survives_blob_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = AliceDB::open(dir.path()).unwrap();
    db.put_law("ohm", &line_law()).unwrap();
    db.compact_blob_sstable().unwrap();
    drop(db);
    let db = AliceDB::open(dir.path()).unwrap();
    assert_parts_bit_identical(
        &line_law().to_parts(),
        &db.get_law("ohm").unwrap().unwrap().to_parts(),
    );
}

// ---------------------------------------------------------------- evaluation

#[test]
fn evaluate_inside_range_matches_the_line() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    // 1 + 2 · 2.5 = 6 (a condition that was not measured)
    assert!((db.evaluate_law("ohm", 2.5).unwrap() - 6.0).abs() < 1e-12);
    // the closed bounds are inside
    assert!((db.evaluate_law("ohm", 0.0).unwrap() - 1.0).abs() < 1e-12);
    assert!((db.evaluate_law("ohm", 4.0).unwrap() - 9.0).abs() < 1e-12);
}

#[test]
fn evaluate_refuses_out_of_range_and_non_finite() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    for x in [-1e-9, 4.0 + 1e-9, 9.0, f64::NAN, f64::INFINITY] {
        match db.evaluate_law("ohm", x) {
            Err(LawStoreError::Law(LawError::OutOfRange)) => {}
            other => panic!("x = {x}: expected OutOfRange, got {other:?}"),
        }
    }
}

#[test]
fn evaluate_unknown_law_is_not_found() {
    let db = memory_db();
    assert!(matches!(
        db.evaluate_law("none", 1.0),
        Err(LawStoreError::NotFound)
    ));
    assert!(db.get_law("none").unwrap().is_none());
    assert!(db.law_versions("none").unwrap().is_empty());
    assert!(db.law_history("none").unwrap().is_empty());
}

// ---------------------------------------------------------------- verdicts

/// Every verdict kind, in order, each recorded with its evidence count and rms
#[test]
fn each_verdict_is_recorded_in_order() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();

    // 1. no points
    assert_eq!(
        db.ingest_evidence("ohm", &[], &POLICY).unwrap(),
        Verdict::NoEvidence
    );
    // 2. one point at x = 5 outside [0, 4]
    assert_eq!(
        db.ingest_evidence("ohm", &[(1.0, 3.0), (5.0, 11.0)], &POLICY)
            .unwrap(),
        Verdict::OutOfRange { outside: 1 }
    );
    // 3. on the line: rms ≈ 0 ≤ band 0.5
    assert!(matches!(
        db.ingest_evidence("ohm", &[(0.5, 2.0), (3.5, 8.0)], &POLICY)
            .unwrap(),
        Verdict::Supports { .. }
    ));
    // 4. zigzag ±1 at x = 0.5, 1.5, 2.5, 3.5: rms_new = 1 > band 0.5; the refit
    //    on all 9 points leaves SS = 4 − (Σ(x−2)e)²/Sxx = 4 − 4/15, rms =
    //    sqrt((56/15)/9) ≈ 0.644 > 0.5, so no update; 1 ≤ 0.5 · 4 → grew
    let zig = |d: f64| -> Vec<(f64, f64)> {
        [0.5, 1.5, 2.5, 3.5]
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, 1.0 + 2.0 * x + if i % 2 == 0 { d } else { -d }))
            .collect()
    };
    let grew = db.ingest_evidence("ohm", &zig(1.0), &POLICY).unwrap();
    match grew {
        Verdict::ResidualGrew { rms } => assert!((rms - 1.0).abs() < 1e-12, "{rms}"),
        other => panic!("expected ResidualGrew, got {other:?}"),
    }
    // 5. same zigzag ×3: rms_new = 3 > 2, refit rms 3 · 0.644 > 0.5 → breaks
    let broke = db.ingest_evidence("ohm", &zig(3.0), &POLICY).unwrap();
    match broke {
        Verdict::Breaks { rms } => assert!((rms - 3.0).abs() < 1e-12, "{rms}"),
        other => panic!("expected Breaks, got {other:?}"),
    }
    // 6. the line + 0.6 at the same x: rms_new = 0.6 > 0.5; by symmetry the
    //    refit on old + new is 1.3 + 2x with residuals ±0.3 → rms 0.3 ≤ 0.5
    let shifted: Vec<(f64, f64)> = line_points().iter().map(|&(x, y)| (x, y + 0.6)).collect();
    match db.ingest_evidence("ohm", &shifted, &POLICY).unwrap() {
        Verdict::ParameterUpdate {
            previous_rms,
            updated,
        } => {
            assert!((previous_rms - 0.6).abs() < 1e-12, "{previous_rms}");
            assert!((updated.residual().rms - 0.3).abs() < 1e-12);
        }
        other => panic!("expected ParameterUpdate, got {other:?}"),
    }

    let hist = db.law_history("ohm").unwrap();
    let kinds: Vec<VerdictKind> = hist.iter().map(|r| r.outcome).collect();
    assert_eq!(
        kinds,
        vec![
            VerdictKind::NoEvidence,
            VerdictKind::OutOfRange,
            VerdictKind::Supports,
            VerdictKind::ResidualGrew,
            VerdictKind::Breaks,
            VerdictKind::ParameterUpdate,
        ]
    );
    let seqs: Vec<u64> = hist.iter().map(|r| r.seq).collect();
    assert_eq!(seqs, vec![1, 2, 3, 4, 5, 6]);
    let counts: Vec<u64> = hist.iter().map(|r| r.evidence_count).collect();
    assert_eq!(counts, vec![0, 2, 2, 4, 4, 5]);
    // judged against version 1 every time (the update is the last entry)
    assert!(hist.iter().all(|r| r.law_version == 1));
    assert_eq!(hist[0].rms, None);
    assert_eq!(hist[1].rms, None);
    assert_eq!(hist[1].outside, 1);
    assert!(hist[2].rms.unwrap() < 1e-12);
    assert!((hist[3].rms.unwrap() - 1.0).abs() < 1e-12);
    assert!((hist[4].rms.unwrap() - 3.0).abs() < 1e-12);
    assert!((hist[5].rms.unwrap() - 0.6).abs() < 1e-12);
    assert_eq!(hist[5].new_version, Some(2));
    assert!(hist[..5].iter().all(|r| r.new_version.is_none()));
}

#[test]
fn parameter_update_keeps_the_old_version_readable() {
    let db = memory_db();
    let law = line_law();
    db.put_law("ohm", &law).unwrap();
    let shifted: Vec<(f64, f64)> = line_points().iter().map(|&(x, y)| (x, y + 0.6)).collect();
    let Verdict::ParameterUpdate { updated, .. } =
        db.ingest_evidence("ohm", &shifted, &POLICY).unwrap()
    else {
        panic!("expected ParameterUpdate");
    };
    assert_eq!(db.law_versions("ohm").unwrap(), vec![1, 2]);
    // latest = the refitted law, 1.3 + 2x
    let latest = db.get_law("ohm").unwrap().unwrap();
    assert_parts_bit_identical(&updated.to_parts(), &latest.to_parts());
    assert!((latest.evaluate(0.0).unwrap() - 1.3).abs() < 1e-12);
    // version 1 is unchanged, 1 + 2x
    let v1 = db.get_law_version("ohm", 1).unwrap().unwrap();
    assert_parts_bit_identical(&law.to_parts(), &v1.to_parts());
    assert!((v1.evaluate(0.0).unwrap() - 1.0).abs() < 1e-12);
    assert!(db.get_law_version("ohm", 3).unwrap().is_none());
    // the next evidence is judged against version 2: 1.3 + 2x itself supports it
    assert!(matches!(
        db.ingest_evidence("ohm", &[(1.0, 3.3)], &POLICY).unwrap(),
        Verdict::Supports { .. }
    ));
    assert_eq!(db.law_history("ohm").unwrap()[1].law_version, 2);
}

#[test]
fn put_law_on_an_existing_name_adds_a_version() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    let other = SignalLaw::fit_polynomial(
        &[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)],
        2,
        Provenance::new("b", "least squares"),
    )
    .unwrap();
    assert_eq!(db.put_law("ohm", &other).unwrap(), 2);
    assert_eq!(db.law_versions("ohm").unwrap(), vec![1, 2]);
    // x² at 1.5
    assert!((db.evaluate_law("ohm", 1.5).unwrap() - 2.25).abs() < 1e-12);
}

#[test]
fn names_are_independent_even_when_one_is_a_prefix_of_another() {
    let db = memory_db();
    db.put_law("a", &line_law()).unwrap();
    db.put_law("ab", &line_law()).unwrap();
    db.ingest_evidence("ab", &[(1.0, 3.0)], &POLICY).unwrap();
    assert_eq!(db.law_versions("a").unwrap(), vec![1]);
    assert!(db.law_history("a").unwrap().is_empty());
    assert_eq!(db.law_history("ab").unwrap().len(), 1);
    assert_eq!(
        db.law_names().unwrap(),
        vec!["a".to_string(), "ab".to_string()]
    );
    // ordinary blobs are not laws and laws are not mixed into a user prefix
    db.put_blob(b"a", b"x").unwrap();
    assert_eq!(db.scan_blob_prefix(b"a").unwrap().len(), 1);
    assert_eq!(db.law_names().unwrap().len(), 2);
}

// ---------------------------------------------------------------- corruption

#[test]
fn a_flipped_byte_is_rejected_by_the_checksum() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    let key = law_version_key("ohm", 1).unwrap();
    let mut bytes = db.get_blob(&key).unwrap().unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    db.put_blob(&key, &bytes).unwrap();
    assert!(matches!(db.get_law("ohm"), Err(LawStoreError::Corrupt(_))));
    assert!(matches!(
        db.evaluate_law("ohm", 1.0),
        Err(LawStoreError::Corrupt(_))
    ));
}

#[test]
fn a_truncated_record_is_rejected() {
    let parts = line_law().to_parts();
    let rec = encode_law_record("ohm", 1, &parts).unwrap();
    for cut in [0, 1, 8, 12, rec.len() / 2, rec.len() - 1] {
        assert!(
            matches!(
                decode_law_record(&rec[..cut]),
                Err(LawStoreError::Corrupt(_))
            ),
            "cut at {cut}"
        );
    }
    let mut longer = rec;
    longer.push(0);
    assert!(matches!(
        decode_law_record(&longer),
        Err(LawStoreError::Corrupt(_))
    ));
}

/// A record with a valid checksum whose evidence lies outside the stored
/// domain: it decodes, and `SignalLaw::from_parts` refuses it
#[test]
fn a_forged_record_with_evidence_outside_the_domain_is_rejected() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    let mut parts = line_law().to_parts();
    parts.evidence.push((7.0, 15.0)); // domain stays [0, 4]
    let forged = encode_law_record("ohm", 1, &parts).unwrap();
    db.put_blob(&law_version_key("ohm", 1).unwrap(), &forged)
        .unwrap();
    assert!(matches!(
        db.get_law("ohm"),
        Err(LawStoreError::Law(LawError::OutOfRange))
    ));
}

#[test]
fn a_forged_record_with_a_non_finite_coefficient_is_rejected() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    let mut parts = line_law().to_parts();
    parts.coefficients[0] = f64::NAN;
    let forged = encode_law_record("ohm", 1, &parts).unwrap();
    db.put_blob(&law_version_key("ohm", 1).unwrap(), &forged)
        .unwrap();
    assert!(matches!(
        db.get_law("ohm"),
        Err(LawStoreError::Law(LawError::NonFinite))
    ));
}

#[test]
fn a_record_copied_under_another_name_or_version_is_rejected() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    db.put_law("hooke", &line_law()).unwrap();
    let ohm = db
        .get_blob(&law_version_key("ohm", 1).unwrap())
        .unwrap()
        .unwrap();
    db.put_blob(&law_version_key("hooke", 1).unwrap(), &ohm)
        .unwrap();
    assert!(matches!(
        db.get_law("hooke"),
        Err(LawStoreError::Corrupt(_))
    ));
    // the same record placed as version 2 of "ohm"
    db.put_blob(&law_version_key("ohm", 2).unwrap(), &ohm)
        .unwrap();
    assert!(matches!(db.get_law("ohm"), Err(LawStoreError::Corrupt(_))));
}

#[test]
fn stored_residual_is_not_trusted() {
    // a record claiming rms = 0 for evidence off the line restores with the
    // re-measured residual: evidence (0,1) (1,3) (2,5) (3,7) (4,10) about
    // 1 + 2x → one deviation of 1, rms = sqrt(1/5)
    let mut parts = line_law().to_parts();
    parts.evidence[4].1 = 10.0;
    parts.residual.rms = 0.0;
    parts.residual.max_abs = 0.0;
    let rec = encode_law_record("ohm", 1, &parts).unwrap();
    let (name, version, decoded) = decode_law_record(&rec).unwrap();
    assert_eq!((name.as_str(), version), ("ohm", 1));
    let law = SignalLaw::from_parts(decoded).unwrap();
    assert!((law.residual().rms - (0.2_f64).sqrt()).abs() < 1e-12);
    assert!((law.residual().max_abs - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------- degenerate

#[test]
fn invalid_names_are_refused() {
    let db = memory_db();
    for name in ["", "a\0b"] {
        assert!(matches!(
            db.put_law(name, &line_law()),
            Err(LawStoreError::InvalidName)
        ));
        assert!(matches!(db.get_law(name), Err(LawStoreError::InvalidName)));
        assert!(matches!(
            db.evaluate_law(name, 1.0),
            Err(LawStoreError::InvalidName)
        ));
        assert!(matches!(
            db.ingest_evidence(name, &[(1.0, 3.0)], &POLICY),
            Err(LawStoreError::InvalidName)
        ));
    }
    assert!(db.law_names().unwrap().is_empty());
}

#[test]
fn ingest_into_an_unknown_law_records_nothing() {
    let db = memory_db();
    assert!(matches!(
        db.ingest_evidence("none", &[(1.0, 3.0)], &POLICY),
        Err(LawStoreError::NotFound)
    ));
    assert!(db.law_history("none").unwrap().is_empty());
}

#[test]
fn invalid_policies_are_refused_and_not_recorded() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    let bad = [
        IngestPolicy {
            abs_tolerance: f64::NAN,
            break_factor: 4.0,
        },
        IngestPolicy {
            abs_tolerance: -0.1,
            break_factor: 4.0,
        },
        IngestPolicy {
            abs_tolerance: f64::INFINITY,
            break_factor: 4.0,
        },
        IngestPolicy {
            abs_tolerance: 0.5,
            break_factor: f64::NAN,
        },
        IngestPolicy {
            abs_tolerance: 0.5,
            break_factor: 0.5,
        },
        IngestPolicy {
            abs_tolerance: 0.5,
            break_factor: f64::INFINITY,
        },
    ];
    for p in &bad {
        assert!(
            matches!(
                db.ingest_evidence("ohm", &[(1.0, 3.0)], p),
                Err(LawStoreError::InvalidPolicy)
            ),
            "{p:?}"
        );
    }
    assert!(db.law_history("ohm").unwrap().is_empty());
}

#[test]
fn non_finite_evidence_is_out_of_range_and_recorded() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    assert_eq!(
        db.ingest_evidence(
            "ohm",
            &[(1.0, f64::NAN), (f64::NAN, 1.0), (2.0, 5.0)],
            &POLICY
        )
        .unwrap(),
        Verdict::OutOfRange { outside: 2 }
    );
    let h = db.law_history("ohm").unwrap();
    assert_eq!(h.len(), 1);
    assert_eq!(
        (h[0].outcome, h[0].evidence_count, h[0].outside),
        (VerdictKind::OutOfRange, 3, 2)
    );
}

/// CRC-32 (IEEE, reflected 0xEDB88320), bit by bit: an independent reference
/// for re-sealing a forged record
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[test]
fn crc_reference_matches_the_check_value() {
    // CRC-32/ISO-HDLC check value of "123456789"
    assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
}

/// A sealed record whose coefficient count claims `u32::MAX` entries: refused as
/// truncated before anything is allocated for them
#[test]
fn a_forged_huge_count_is_refused_without_allocating() {
    let rec = encode_law_record("ohm", 1, &line_law().to_parts()).unwrap();
    // magic 8 + format 4 + name length 4 + "ohm" 3 + version 8 = 27
    let mut body = rec[..rec.len() - 4].to_vec();
    assert_eq!(
        &body[27..31],
        &2u32.to_le_bytes(),
        "two coefficients of a line"
    );
    body[27..31].copy_from_slice(&u32::MAX.to_le_bytes());
    let crc = crc32_ieee(&body);
    body.extend_from_slice(&crc.to_le_bytes());
    assert!(matches!(
        decode_law_record(&body),
        Err(LawStoreError::Corrupt(_))
    ));

    // evidence count (u64) after 27 + coefficient count 4 + 2 coefficients 16
    // + domain 16 = 63: u64::MAX points of 16 bytes exceed any allocation
    let mut body = rec[..rec.len() - 4].to_vec();
    assert_eq!(&body[63..71], &5u64.to_le_bytes(), "five evidence points");
    body[63..71].copy_from_slice(&u64::MAX.to_le_bytes());
    let crc = crc32_ieee(&body);
    body.extend_from_slice(&crc.to_le_bytes());
    assert!(matches!(
        decode_law_record(&body),
        Err(LawStoreError::Corrupt(_))
    ));
}

#[test]
fn a_history_key_without_a_version_does_not_make_a_name() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    db.put_blob(
        &alice_db::law_store::law_history_key("ghost", 1).unwrap(),
        b"x",
    )
    .unwrap();
    assert_eq!(db.law_names().unwrap(), vec!["ohm".to_string()]);
}

/// Sequence numbers past 255 still come back in numeric order (keys hold
/// big-endian numbers, so byte order is numeric order)
#[test]
fn history_order_holds_past_one_byte() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    for _ in 0..300 {
        db.ingest_evidence("ohm", &[], &POLICY).unwrap();
    }
    let seqs: Vec<u64> = db
        .law_history("ohm")
        .unwrap()
        .iter()
        .map(|r| r.seq)
        .collect();
    assert_eq!(seqs, (1..=300).collect::<Vec<u64>>());
}

/// Concurrent ingests never reuse a sequence number
#[test]
fn concurrent_ingests_record_every_verdict() {
    let db = std::sync::Arc::new(memory_db());
    db.put_law("ohm", &line_law()).unwrap();
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let db = std::sync::Arc::clone(&db);
            std::thread::spawn(move || {
                for _ in 0..50 {
                    db.ingest_evidence("ohm", &[(1.0, 3.0)], &POLICY).unwrap();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let seqs: Vec<u64> = db
        .law_history("ohm")
        .unwrap()
        .iter()
        .map(|r| r.seq)
        .collect();
    assert_eq!(seqs, (1..=400).collect::<Vec<u64>>());
}

/// Replaces bytes `at..at + new.len()` of a record body and re-seals the checksum
fn reseal(rec: &[u8], at: usize, new: &[u8], extra: &[u8]) -> Vec<u8> {
    let mut body = rec[..rec.len() - 4].to_vec();
    body[at..at + new.len()].copy_from_slice(new);
    body.extend_from_slice(extra);
    let crc = crc32_ieee(&body);
    body.extend_from_slice(&crc.to_le_bytes());
    body
}

#[test]
fn sealed_records_with_a_wrong_header_or_tail_are_refused() {
    let rec = encode_law_record("ohm", 1, &line_law().to_parts()).unwrap();
    // the untouched re-seal decodes (the helper itself is sound)
    assert!(decode_law_record(&reseal(&rec, 0, &rec[..1], b"")).is_ok());
    // magic of a verdict record
    assert!(matches!(
        decode_law_record(&reseal(&rec, 0, b"ALAWVRD\x01", b"")),
        Err(LawStoreError::Corrupt(_))
    ));
    // format 2
    assert!(matches!(
        decode_law_record(&reseal(&rec, 8, &2u32.to_le_bytes(), b"")),
        Err(LawStoreError::Corrupt(_))
    ));
    // one byte after the last field
    assert!(matches!(
        decode_law_record(&reseal(&rec, 0, &rec[..1], &[0])),
        Err(LawStoreError::Corrupt(_))
    ));
}

#[test]
fn a_verdict_record_copied_under_another_name_is_rejected() {
    let db = memory_db();
    db.put_law("ohm", &line_law()).unwrap();
    db.put_law("hooke", &line_law()).unwrap();
    db.ingest_evidence("ohm", &[(1.0, 3.0)], &POLICY).unwrap();
    let key = alice_db::law_store::law_history_key;
    let rec = db.get_blob(&key("ohm", 1).unwrap()).unwrap().unwrap();
    db.put_blob(&key("hooke", 1).unwrap(), &rec).unwrap();
    assert!(matches!(
        db.law_history("hooke"),
        Err(LawStoreError::Corrupt(_))
    ));
    // the same record as entry 2 of "ohm"
    db.put_blob(&key("ohm", 2).unwrap(), &rec).unwrap();
    assert!(matches!(
        db.law_history("ohm"),
        Err(LawStoreError::Corrupt(_))
    ));
}
