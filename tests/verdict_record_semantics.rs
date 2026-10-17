//! Verdict records carry the identifier of the arithmetic that judged them
//! (ALICE-Law G8)
//!
//! A verdict's RMS and outcome are this crate's arithmetic (the fit, the RMS,
//! the comparison with the policy). Format 2 stamps [`alice_db::DB_SEMANTICS_ID`]
//! of the judging build; a read reports it ([`RecordSemantics::Current`]) and
//! refuses another identifier. Format 1 (written before the stamp) is read as
//! [`RecordSemantics::Unpinned`], or refused under [`LawReadPolicy::RequirePinned`].
//! The format 1 bytes below were written by `alice-db` before format 2
//! (`main` 66dbb8e, `ingest_evidence` of the line `y = 1 + 2x`).

use alice_db::law_store::{
    IngestPolicy, LawReadPolicy, LawStoreError, Provenance, RecordSemantics, SignalLaw,
    VerdictKind, LAW_KEY_PREFIX, VERDICT_RECORD_FORMAT,
};
use alice_db::{AliceDB, StorageConfig, DB_SEMANTICS_ID};

/// `(seq, bytes)` of two format 1 verdict records of the law `line`
const FORMAT_1: [(u64, &str); 2] = [
    (
        1,
        "414c41575652440101000000040000006c696e65010000000000000001000000000000000300000000000000020000000000000000018330ff3fa42b573f0000000000000000006f884a81",
    ),
    (
        2,
        "414c41575652440101000000040000006c696e6502000000000000000100000000000000030000000000000004000000000000000001819999999999993f0000000000000000000dcfea07",
    ),
];

const POLICY: IngestPolicy = IngestPolicy {
    abs_tolerance: 0.01,
    break_factor: 4.0,
};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

fn history_key(name: &str, seq: u64) -> Vec<u8> {
    let mut k = LAW_KEY_PREFIX.to_vec();
    k.extend_from_slice(name.as_bytes());
    k.extend_from_slice(b"\0h");
    k.extend_from_slice(&seq.to_be_bytes());
    k
}

fn db_with_line() -> AliceDB {
    let db = AliceDB::in_memory(StorageConfig::default()).expect("db");
    let pts: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i)))
        .collect();
    let law =
        SignalLaw::fit_polynomial(&pts, 1, Provenance::new("run 1", "least squares")).expect("fit");
    db.put_law("line", &law, &alice_db::SEMANTICS_ID)
        .expect("put");
    db
}

fn ingest_two(db: &AliceDB) {
    let near = [(0.5, 2.001), (1.5, 3.999), (2.5, 6.002)];
    let off = [(0.5, 2.025), (1.5, 4.025), (2.5, 6.025)];
    for ev in [&near[..], &off[..]] {
        db.ingest_evidence("line", ev, &POLICY, &alice_db::SEMANTICS_ID)
            .expect("ingest");
    }
}

#[test]
fn a_new_verdict_is_stamped_and_read_as_current() {
    let db = db_with_line();
    ingest_two(&db);
    for seq in 1..=2 {
        let bytes = db
            .get_blob(&history_key("line", seq))
            .expect("read")
            .expect("written");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")),
            VERDICT_RECORD_FORMAT
        );
        assert_eq!(VERDICT_RECORD_FORMAT, 2);
        // magic 8 + format 4 + name length 4 + "line"
        assert_eq!(&bytes[20..52], &DB_SEMANTICS_ID, "seq {seq}: the stamp");
    }
    for policy in [LawReadPolicy::AcceptUnpinned, LawReadPolicy::RequirePinned] {
        let history = db.law_history_with("line", policy).expect("read");
        assert_eq!(history.len(), 2);
        for v in &history {
            assert_eq!(v.semantics, RecordSemantics::Current);
        }
        assert_eq!(history[0].record.outcome, VerdictKind::Supports);
        assert_eq!(history[1].record.outcome, VerdictKind::ResidualGrew);
    }
}

/// The real format 1 bytes still read, as `Unpinned`, with every field
#[test]
fn a_format_1_verdict_reads_as_unpinned_and_require_pinned_refuses_it() {
    let db = db_with_line();
    for (seq, hex) in FORMAT_1 {
        db.put_blob(&history_key("line", seq), &unhex(hex))
            .expect("put");
    }
    let history = db
        .law_history_with("line", LawReadPolicy::AcceptUnpinned)
        .expect("read");
    assert_eq!(history.len(), 2);
    let want = [
        (VerdictKind::Supports, 0x3f57_2ba4_3fff_3083_u64),
        (VerdictKind::ResidualGrew, 0x3f99_9999_9999_9981_u64),
    ];
    for (v, (outcome, rms_bits)) in history.iter().zip(want) {
        assert_eq!(v.semantics, RecordSemantics::Unpinned);
        assert_eq!(v.record.outcome, outcome);
        assert_eq!(v.record.law_version, 1);
        assert_eq!(v.record.evidence_count, 3);
        assert_eq!(v.record.rms.map(f64::to_bits), Some(rms_bits));
        assert_eq!(v.record.new_version, None);
    }
    // the status-free read keeps reading them
    assert_eq!(db.law_history("line").expect("read").len(), 2);
    assert!(matches!(
        db.law_history_with("line", LawReadPolicy::RequirePinned),
        Err(LawStoreError::Unpinned)
    ));
}

/// A verdict stamped by another build is refused and names both identifiers
#[test]
fn a_verdict_of_another_db_semantics_is_refused() {
    let db = db_with_line();
    ingest_two(&db);
    let key = history_key("line", 1);
    let mut bytes = db.get_blob(&key).expect("read").expect("written");
    let other = [0x5a_u8; 32];
    bytes[20..52].copy_from_slice(&other);
    let n = bytes.len();
    let crc = crc32fast::hash(&bytes[..n - 4]);
    bytes[n - 4..].copy_from_slice(&crc.to_le_bytes());
    db.put_blob(&key, &bytes).expect("put");
    for policy in [LawReadPolicy::AcceptUnpinned, LawReadPolicy::RequirePinned] {
        match db.law_history_with("line", policy) {
            Err(LawStoreError::VerdictRecordMismatch { stored, running }) => {
                assert_eq!(stored, other);
                assert_eq!(running, DB_SEMANTICS_ID);
            }
            other => panic!("{policy:?}: {other:?}"),
        }
    }
    assert!(matches!(
        db.law_history("line"),
        Err(LawStoreError::VerdictRecordMismatch { .. })
    ));
}
