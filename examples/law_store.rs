//! Store a fitted law, query it, judge new evidence against it, and read the
//! record back after the database is reopened.
//!
//! ```sh
//! cargo run --example law_store
//! ```

use alice_db::law_store::{IngestPolicy, OracleCase, Provenance, SignalLaw, Verdict};
use alice_db::{AliceDB, StorageConfig, SEMANTICS_ID};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y = 1 + 2x measured at x = 0..=4
    let points: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i)))
        .collect();
    let law = SignalLaw::fit_polynomial(&points, 1, Provenance::new("run 1", "least squares"))?
        .with_oracle(OracleCase::new(2.0, 5.0, 1e-9, "reference table"));
    let policy = IngestPolicy {
        abs_tolerance: 0.5,
        break_factor: 4.0,
    };

    let db = AliceDB::in_memory(StorageConfig::default())?;
    run(&db, &law, &policy)?;

    #[cfg(feature = "fs")]
    {
        let dir = tempfile::tempdir()?;
        {
            let db = AliceDB::open(dir.path())?;
            run(&db, &law, &policy)?;
            db.close()?;
        }
        let db = AliceDB::open(dir.path())?;
        println!(
            "[file, reopened] versions {:?}, {} verdicts, f(2) = {}",
            db.law_versions("line")?,
            db.law_history("line")?.len(),
            db.evaluate_law("line", 2.0)?
        );
    }
    Ok(())
}

fn run(
    db: &AliceDB,
    law: &SignalLaw,
    policy: &IngestPolicy,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = if db.is_in_memory() { "memory" } else { "file" };
    db.put_law("line", law, &SEMANTICS_ID)?;
    println!("[{backend}] f(2.5) = {}", db.evaluate_law("line", 2.5)?);
    match db.evaluate_law("line", 9.0) {
        Err(e) => println!("[{backend}] f(9) refused: {e}"),
        Ok(v) => return Err(format!("f(9) = {v} was extrapolated").into()),
    }

    // the same line measured again, then shifted by +0.6
    let again = [(0.5, 2.0), (3.5, 8.0)];
    let shifted: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.6 + 2.0 * f64::from(i)))
        .collect();
    for evidence in [&again[..], &shifted[..]] {
        let verdict = db.ingest_evidence("line", evidence, policy, &SEMANTICS_ID)?;
        let label = match &verdict {
            Verdict::Supports { rms } => format!("supports (rms {rms:.3e})"),
            Verdict::ParameterUpdate { previous_rms, .. } => {
                format!("parameter update (rms before {previous_rms:.3})")
            }
            other => format!("{other:?}"),
        };
        println!("[{backend}] {} points: {label}", evidence.len());
    }
    for r in db.law_history("line")? {
        println!(
            "[{backend}]   #{} against v{}: {:?}, {} points, rms {:?}, new version {:?}",
            r.seq, r.law_version, r.outcome, r.evidence_count, r.rms, r.new_version
        );
    }
    let old = db.get_law_version("line", 1)?.ok_or("version 1 missing")?;
    println!(
        "[{backend}] versions {:?}: v1 f(0) = {}, latest f(0) = {}",
        db.law_versions("line")?,
        old.evaluate(0.0)?,
        db.evaluate_law("line", 0.0)?
    );
    Ok(())
}
