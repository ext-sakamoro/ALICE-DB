//! Analytic oracles — closed-form / independent-reference checks for the
//! numeric laws in ALICE-DB (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms, published test vectors or f64
//! references written in this file — never from the crate function under
//! test.  `StorageConfig::default()` / `FitConfig::default()` are the paths a
//! consumer takes first, so the default (lossy) contract is pinned to its
//! *documented* thresholds and the `lossless: true` contract to bit exactness.
//!
//! Oracle sources:
//! - model evaluation at sample position i = (t − t₀)/(t₁ − t₀)·(n − 1) — the
//!   convention of the `alice_core::generators` fitters that produce the
//!   coefficients: polynomial `Σ cⱼ iʲ`, linear `s + (e − s)·i/(n−1)`, sine
//!   `o + A·sin(2πf·i/n + φ)`, Fourier `dc + Σ w(k)·m/n·cos(2πk·i/n + φ)`
//!   (w = 2 except DC / Nyquist)
//! - lossless residual: put → get is the identity (f32 bit pattern)
//! - default fit: the documented acceptance thresholds (linear max error < 1 %
//!   of range, polynomial relative MSE < 1e-3, Fourier ≥ 99 % energy,
//!   sine relative MSE < 0.1)
//! - CRC-32 check value 0xCBF43926 ("123456789"), XXH64 vectors ("" →
//!   0xEF46DB3751D8E999, "a" → 0xD24EC4F1A98C6E5B, "abc" → 0x44BC2CF5AD770999)
//! - Bloom filter: m = −n ln p / (ln 2)², k = round(m/n · ln 2), no false
//!   negatives, measured false-positive rate ≤ 2p
//! - aggregates: sum / mean / min / max / variance of known data

// numeric oracles cast freely between the index / float domains; the crate's
// pedantic gate is about API code, not about the reference arithmetic here
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use alice_db::bloom::BloomFilter;
use alice_db::checksum::{Checksum, ChecksumAlgorithm};
use alice_db::segment::{compress_residual, decompress_residual};
use alice_db::{Aggregation, AliceDB, DataSegment, DataType, FitConfig, ModelType, StorageConfig};
use std::io::Cursor;

// ───────────────────────── helpers ────────────────────────────────────────

fn segment(model: ModelType, start: i64, end: i64, n: usize) -> DataSegment {
    DataSegment::new(1, start, end, model, n, n * 4)
}

/// Sample position of `t` (fractional index into the n uniform samples)
fn i_of(seg: &DataSegment, t: i64, n: usize) -> f64 {
    (t - seg.start_time) as f64 / (seg.end_time - seg.start_time) as f64 * (n - 1) as f64
}

fn lossless_db(dir: &std::path::Path, mmap: bool) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        use_mmap: mmap,
        ..StorageConfig::default()
    })
    .unwrap()
}

fn default_db(dir: &std::path::Path) -> AliceDB {
    AliceDB::with_config(StorageConfig {
        data_dir: dir.to_path_buf(),
        ..StorageConfig::default()
    })
    .unwrap()
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

// ───────────────────────── model evaluation ───────────────────────────────

#[test]
fn polynomial_linear_constant_models_evaluate_to_their_closed_forms() {
    let coeffs = vec![3.0, -2.0, 0.5, 1.25]; // c0 + c1 x + c2 x² + c3 x³
    let seg = segment(
        ModelType::Polynomial {
            coefficients: coeffs.clone(),
            degree: 3,
            fit_error: 0.0,
        },
        1000,
        1999,
        1000,
    );
    for t in (1000..=1999).step_by(7) {
        let i = i_of(&seg, t, 1000); // = t − 1000 for unit spacing
        let expected: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(j, c)| c * i.powi(j as i32))
            .sum();
        let got = seg.query_point(t).unwrap() as f64;
        assert!(
            (got - expected).abs() < 1e-3 * expected.abs().max(1.0),
            "poly t={t}: {got} vs {expected}"
        );
    }
    let lin = segment(
        ModelType::Linear {
            start_value: -4.0,
            end_value: 12.0,
        },
        0,
        400,
        401,
    );
    for t in 0..=400 {
        let expected = -4.0 + (t as f64 / 400.0) * 16.0;
        assert!(
            (lin.query_point(t).unwrap() as f64 - expected).abs() < 1e-5,
            "lin t={t}"
        );
    }
    assert!(lin.query_point(-1).is_none() && lin.query_point(401).is_none());
    let c = segment(ModelType::Constant { value: 2.5 }, 5, 5, 1);
    assert_eq!(c.query_point(5), Some(2.5));
}

#[test]
fn sine_and_fourier_models_evaluate_to_their_closed_forms() {
    let (f, a, phi, o) = (3.0f32, 2.0f32, 0.7f32, 1.0f32);
    let seg = segment(
        ModelType::SineWave {
            frequency: f,
            amplitude: a,
            phase: phi,
            offset: o,
        },
        0,
        999,
        1000,
    );
    for t in (0..=999).step_by(13) {
        let i = i_of(&seg, t, 1000);
        let expected = o as f64
            + a as f64 * (2.0 * std::f64::consts::PI * f as f64 * i / 1000.0 + phi as f64).sin();
        let got = seg.query_point(t).unwrap() as f64;
        assert!(
            (got - expected).abs() < 1e-4,
            "sine t={t}: {got} vs {expected}"
        );
    }
    // Fourier: dc + Σ w(k)·mag/n · cos(2π k i/n + phase) — `mag` is the raw
    // DFT bin magnitude the fitter stores (alice-zip `analyze_signal`)
    let coefs = vec![(1usize, 750.0f32, 0.2f32), (4usize, 250.0f32, -1.0f32)]; // DFT bins of amplitudes 1.5 / 0.5
    let seg = segment(
        ModelType::Fourier {
            coefficients: coefs.clone(),
            dc_offset: 0.25,
            sample_count: 1000,
        },
        0,
        999,
        1000,
    );
    for t in (0..=999).step_by(11) {
        let i = i_of(&seg, t, 1000);
        let mut expected = 0.25f64;
        for &(k, m, p) in &coefs {
            let w = if k == 0 || 2 * k == 1000 { 1.0 } else { 2.0 };
            expected += w * m as f64 / 1000.0
                * (2.0 * std::f64::consts::PI * k as f64 * i / 1000.0 + p as f64).cos();
        }
        let got = seg.query_point(t).unwrap() as f64;
        assert!(
            (got - expected).abs() < 1e-5,
            "fourier t={t}: {got} vs {expected}"
        );
    }
}

#[test]
fn query_range_is_query_point_sampled_uniformly_over_the_segment() {
    let seg = segment(
        ModelType::Polynomial {
            coefficients: vec![1.0, 2.0, -3.0],
            degree: 2,
            fit_error: 0.0,
        },
        100,
        1099,
        1000,
    );
    let all = seg.query_range(100, 1099);
    assert_eq!(all.len(), 1000, "full range returns point_count samples");
    for (i, (t, v)) in all.iter().enumerate() {
        assert_eq!(*t, 100 + i as i64);
        assert_eq!(v.to_bits(), seg.query_point(*t).unwrap().to_bits(), "t={t}");
    }
    let part = seg.query_range(300, 310);
    assert_eq!(part.len(), 11);
    assert!(part.iter().all(|(t, _)| (300..=310).contains(t)));
    assert!(seg.query_range(2000, 3000).is_empty());
    // generate_all is the same law in bulk
    let bulk = seg.generate_all();
    assert_eq!(bulk.len(), 1000);
    for (i, v) in bulk.iter().enumerate() {
        let expected = seg.query_point(100 + i as i64).unwrap();
        assert!((v - expected).abs() < 1e-5, "generate_all[{i}]");
    }
}

// ───────────────────────── residual / raw round trips ─────────────────────

#[test]
fn residual_blob_and_raw_lzma_round_trip_bit_exactly() {
    let mut seed = 7u64;
    let raw: Vec<f32> = (0..2048).map(|_| lcg(&mut seed) * 1e3).collect();
    let bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let blob = compress_residual(&bytes);
    assert_eq!(decompress_residual(&blob), bytes);
    // incompressible input takes the raw path and still round-trips
    let noise: Vec<u8> = (0..1000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    assert_eq!(decompress_residual(&compress_residual(&noise)), noise);

    // RawLzma model: every value comes back exactly for f32 payloads
    let mut compressed = Vec::new();
    lzma_rs::lzma_compress(&mut Cursor::new(&bytes), &mut compressed).unwrap();
    let seg = segment(
        ModelType::RawLzma {
            compressed_data: compressed,
            dtype: DataType::Float32,
            original_size: bytes.len(),
        },
        0,
        2047,
        2048,
    );
    for (i, v) in raw.iter().enumerate() {
        assert_eq!(
            seg.query_point(i as i64).unwrap().to_bits(),
            v.to_bits(),
            "raw idx {i}"
        );
    }
    let all = seg.generate_all();
    assert_eq!(all.len(), 2048);
    assert!(all
        .iter()
        .zip(&raw)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn lossless_mode_returns_every_put_value_bit_exactly_through_both_read_paths() {
    // three shapes that the fitter maps to different models: a quadratic
    // (polynomial), a noisy sine (Fourier / sine), and arbitrary values (raw)
    let mut seed = 99u64;
    let series: [Vec<(i64, f32)>; 3] = [
        (0..1000)
            .map(|i| (i, 0.002 * (i as f32) * (i as f32) - 0.3 * i as f32 + 7.0))
            .collect(),
        (0..1000)
            .map(|i| {
                (
                    i,
                    (2.0 * std::f32::consts::PI * 3.0 * i as f32 / 1000.0).sin() * 5.0
                        + 0.05 * lcg(&mut seed),
                )
            })
            .collect(),
        (0..1000).map(|i| (i, lcg(&mut seed) * 250.0)).collect(),
    ];
    for mmap in [true, false] {
        for (k, data) in series.iter().enumerate() {
            let dir = tempfile::tempdir().unwrap();
            let db = lossless_db(dir.path(), mmap);
            db.put_batch(data).unwrap();
            db.flush().unwrap();
            for &(t, v) in data {
                let got = db.get(t).unwrap().expect("stored point");
                assert_eq!(
                    got.to_bits(),
                    v.to_bits(),
                    "mmap={mmap} series {k} t={t}: {got} vs {v}"
                );
            }
            let scanned = db.scan(0, 999).unwrap();
            assert_eq!(
                scanned.len(),
                data.len(),
                "mmap={mmap} series {k}: scan count"
            );
            for ((t, got), &(want_t, want)) in scanned.iter().zip(data) {
                assert_eq!(*t, want_t);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "mmap={mmap} series {k} scan t={t}"
                );
            }
        }
    }
}

#[test]
#[ignore = "DataSegment は timestamp を保持せず uniform spacing を仮定する (residual は (t−t₀)/range·(n−1) の丸めで index 化、gap のある系列では別点の residual が当たる、scan は存在しない timestamp を返す) — Backlog ALICE-DB 起票 2026-09-17、segment に timestamp (delta) を持たせる format 変更後に ignore を外す"]
fn lossless_mode_is_exact_for_irregular_timestamps_too() {
    // timestamps with gaps (sensor drop-outs): the residual must be attached
    // to the point that was stored, not to a uniform re-sampling of the range
    let mut seed = 5u64;
    let mut t = 0i64;
    let data: Vec<(i64, f32)> = (0..500)
        .map(|i| {
            t += 1 + (i * 7 % 5) as i64; // gaps of 1..5
            (t, 0.01 * (t as f32) + lcg(&mut seed))
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let db = lossless_db(dir.path(), true);
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    for &(t, v) in &data {
        let got = db.get(t).unwrap().expect("stored point");
        assert_eq!(got.to_bits(), v.to_bits(), "irregular t={t}: {got} vs {v}");
    }
}

#[test]
fn default_lossy_fit_stays_within_its_documented_thresholds() {
    // linear: accepted when max error < 1 % of the range ⇒ every point within 1 %
    let data: Vec<(i64, f32)> = (0..1000).map(|i| (i, 2.0 * i as f32 + 10.0)).collect();
    let dir = tempfile::tempdir().unwrap();
    let db = default_db(dir.path());
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    let range = 2.0 * 999.0f32;
    for &(t, v) in data.iter().step_by(37) {
        let got = db.get(t).unwrap().expect("stored");
        assert!(
            (got - v).abs() <= 0.01 * range + 1e-3,
            "linear t={t}: {got} vs {v}"
        );
    }
    // sine-like: the weakest accepted candidate is the single sine at relative
    // MSE < 0.1 ⇒ RMS error over the segment ≤ √0.1 · σ (the documented law,
    // ~32 % of σ — see Backlog for whether that default is wanted)
    let data: Vec<(i64, f32)> = (0..1000)
        .map(|i| {
            (
                i,
                3.0 + 2.0 * (2.0 * std::f32::consts::PI * 5.0 * i as f32 / 1000.0).sin(),
            )
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let db = default_db(dir.path());
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    let mean = data.iter().map(|&(_, v)| v as f64).sum::<f64>() / 1000.0;
    let var = data
        .iter()
        .map(|&(_, v)| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / 1000.0;
    let mut mse = 0.0f64;
    for &(t, v) in &data {
        let got = db.get(t).unwrap().expect("stored") as f64;
        mse += (got - v as f64).powi(2);
    }
    mse /= 1000.0;
    assert!(
        mse <= 0.1 * var + 1e-9,
        "sine default fit relative MSE {} > 0.1",
        mse / var
    );
}

// ───────────────────────── aggregates ─────────────────────────────────────

#[test]
fn aggregates_match_their_closed_forms_on_a_lossless_store() {
    // 0..=99 ⇒ sum 4950, mean 49.5, min 0, max 99, population variance 833.25
    let data: Vec<(i64, f32)> = (0..100).map(|i| (i, i as f32)).collect();
    let dir = tempfile::tempdir().unwrap();
    let db = lossless_db(dir.path(), true);
    db.put_batch(&data).unwrap();
    db.flush().unwrap();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1.0);
    assert!(close(
        db.aggregate(0, 99, Aggregation::Sum).unwrap(),
        4950.0
    ));
    assert!(close(db.aggregate(0, 99, Aggregation::Avg).unwrap(), 49.5));
    assert!(close(db.aggregate(0, 99, Aggregation::Min).unwrap(), 0.0));
    assert!(close(db.aggregate(0, 99, Aggregation::Max).unwrap(), 99.0));
    assert!(close(
        db.aggregate(0, 99, Aggregation::Count).unwrap(),
        100.0
    ));
    assert!(close(db.aggregate(0, 99, Aggregation::First).unwrap(), 0.0));
    assert!(close(db.aggregate(0, 99, Aggregation::Last).unwrap(), 99.0));
    let var = db.aggregate(0, 99, Aggregation::Variance).unwrap();
    let sd = db.aggregate(0, 99, Aggregation::StdDev).unwrap();
    // population (÷n) = 833.25, sample (÷(n−1)) = 841.67 — either is a
    // defensible convention, but StdDev must be √Variance of the same one
    assert!(
        close(var, 833.25) || close(var, 841.666_666_67),
        "variance {var}"
    );
    assert!(
        close(sd, var.sqrt()),
        "stddev {sd} vs sqrt(variance) {}",
        var.sqrt()
    );
    // sub-range 10..=19 ⇒ sum 145, mean 14.5
    assert!(close(
        db.aggregate(10, 19, Aggregation::Sum).unwrap(),
        145.0
    ));
    assert!(close(db.aggregate(10, 19, Aggregation::Avg).unwrap(), 14.5));
    // downsample with interval 10: bucket means 4.5, 14.5, …
    let ds = db.downsample(0, 99, 10, Aggregation::Avg).unwrap();
    assert_eq!(ds.len(), 10, "10 buckets: {ds:?}");
    for (i, (_, v)) in ds.iter().enumerate() {
        assert!(close(*v, 4.5 + 10.0 * i as f64), "bucket {i}: {v}");
    }
}

// ───────────────────────── checksum test vectors ──────────────────────────

#[test]
fn crc32_and_xxhash64_match_published_test_vectors() {
    // CRC-32 (IEEE 802.3) check value
    assert_eq!(
        Checksum::crc32(b"123456789").to_bytes()[1..5],
        0xCBF4_3926u32.to_le_bytes()
    );
    assert_eq!(Checksum::crc32(b"").to_bytes()[1..5], 0u32.to_le_bytes());
    // XXH64, seed 0 (xxHash reference test vectors)
    let xx =
        |d: &[u8]| u64::from_le_bytes(Checksum::xxhash64(d).to_bytes()[1..9].try_into().unwrap());
    assert_eq!(xx(b""), 0xEF46_DB37_51D8_E999);
    assert_eq!(xx(b"a"), 0xD24E_C4F1_A98C_6E5B);
    assert_eq!(xx(b"abc"), 0x44BC_2CF5_AD77_0999);
    // verify / compute agree and detect a single flipped bit
    for alg in [ChecksumAlgorithm::Crc32, ChecksumAlgorithm::XxHash64] {
        let data: Vec<u8> = (0..4096u32).map(|i| (i * 31 % 251) as u8).collect();
        let c = Checksum::compute(&data, alg);
        assert!(c.verify(&data));
        let mut flipped = data.clone();
        flipped[1234] ^= 0x10;
        assert!(!c.verify(&flipped));
        assert_eq!(Checksum::from_bytes(&c.to_bytes()).unwrap(), c);
    }
}

// ───────────────────────── Bloom filter ───────────────────────────────────

#[test]
fn bloom_filter_sizing_follows_the_closed_form_and_has_no_false_negatives() {
    let (n, p) = (10_000usize, 0.01f64);
    let mut bf = BloomFilter::with_capacity(n, p);
    // oracle: m = −n ln p / (ln 2)² , k = round(m / n · ln 2)
    let m = (-(n as f64) * p.ln() / std::f64::consts::LN_2.powi(2)).ceil();
    let k = (m / n as f64 * std::f64::consts::LN_2).round();
    assert!(
        (bf.num_bits() as f64 - m).abs() <= m * 0.01 + 8.0,
        "num_bits {} vs closed form {m}",
        bf.num_bits()
    );
    assert!(
        (bf.num_hashes() as f64 - k).abs() <= 1.0,
        "num_hashes {} vs {k}",
        bf.num_hashes()
    );
    for i in 0..n {
        bf.insert(format!("key-{i}").as_bytes());
    }
    for i in 0..n {
        assert!(
            bf.contains(format!("key-{i}").as_bytes()),
            "false negative key-{i}"
        );
    }
    let probes = 50_000;
    let fp = (0..probes)
        .filter(|i| bf.contains(format!("other-{i}").as_bytes()))
        .count();
    let rate = fp as f64 / probes as f64;
    assert!(
        rate <= 2.0 * p,
        "false-positive rate {rate} > 2p ({})",
        2.0 * p
    );
}
