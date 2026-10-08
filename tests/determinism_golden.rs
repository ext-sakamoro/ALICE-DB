//! Bit-level golden for everything this crate writes or answers with floating
//! point
//!
//! A segment is written once and read back on whatever machine opens the file
//! later. Three things decide its bits, and all three have to come out the same
//! on every target for the file to mean the same thing everywhere:
//!
//! - **model selection** — which model wins the fitting competition and the
//!   parameters it is stored with (`memtable.rs`, the fit error ranks the
//!   candidates)
//! - **reading back** — `query_point` / `query_range` evaluate the stored model,
//!   and a lossless segment XORs a residual onto exactly those bits
//! - **index sizing** — the bloom filter written next to a blob `SSTable` is
//!   `num_bits` / `num_hashes` sized from a logarithm (`bloom.rs`)
//!
//! The analytic oracles (`tests/analytic_oracle.rs`) cannot see a difference of
//! one ulp: they compare against closed forms with tolerances. This file
//! records bit patterns instead. The expected digests are the same on every
//! target and in every feature set, and CI runs this file on every OS of the
//! test matrix with the default, the full native and no default features, so a
//! disagreement between any two of those runs is a red test.
//!
//! What the layers catch is deliberately disjoint:
//!
//! | layer | catches |
//! |-------|---------|
//! | `tests/analytic_oracle.rs` | the model computing the wrong thing |
//! | this file | the same input producing different bits in different builds |
//! | `clippy.toml` `disallowed-methods` | a platform transcendental coming back |
//!
//! Every value is hashed as `to_bits().to_le_bytes()` (integers as
//! `to_le_bytes()`), so the comparison is over exact IEEE 754 encodings.
//! Updating a digest here is a deliberate act: it means files written under the
//! old arithmetic are no longer what this build would write from the same input.
//!
//! Recorded on aarch64-apple-darwin when bloom sizing moved from the platform
//! `ln` to `alice_det_math::ln64` and the fit error from `powi(2)` to an
//! explicit multiplication. The previous code produces the same five digests
//! in all three feature sets, so the move changed no stored bits there.

use alice_db::bloom::BloomFilter;
use alice_db::law_store::{Provenance, SignalLaw};
use alice_db::segment::decompress_residual;
use alice_db::{DataSegment, FitConfig, MemTable, ModelType};
use sha2::{Digest, Sha256};

/// Collects the bits a scenario produced
#[derive(Default)]
struct Bits {
    hasher: Sha256,
    bytes: usize,
}

impl Bits {
    fn raw(&mut self, b: &[u8]) {
        self.hasher.update(b);
        self.bytes += b.len();
    }

    fn u64(&mut self, v: u64) {
        self.raw(&v.to_le_bytes());
    }

    fn i64(&mut self, v: i64) {
        self.raw(&v.to_le_bytes());
    }

    fn len(&mut self, v: usize) {
        self.u64(u64::try_from(v).expect("lengths fit in u64"));
    }

    fn f32(&mut self, v: f32) {
        self.raw(&v.to_bits().to_le_bytes());
    }

    fn f64(&mut self, v: f64) {
        self.raw(&v.to_bits().to_le_bytes());
    }

    fn tag(&mut self, name: &str) {
        self.len(name.len());
        self.raw(name.as_bytes());
    }
}

/// Compares a scenario against its recorded digest.
///
/// `min_bytes` is the emptiness gate: a comparison of nothing against nothing
/// succeeds, so a scenario that silently stops producing values would keep
/// this file green without it. It is checked before the digest so the failure
/// says which of the two went wrong.
fn assert_golden(scenario: &str, bits: Bits, min_bytes: usize, expected: &str) {
    assert!(
        bits.bytes >= min_bytes,
        "{scenario}: hashed {} bytes, expected at least {min_bytes} — \
         the scenario stopped producing values, so the digest below proves nothing",
        bits.bytes
    );
    let got = hex(&bits.hasher.finalize());
    assert_eq!(
        got, expected,
        "{scenario}: {} bytes hashed to {got}, recorded {expected}",
        bits.bytes
    );
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        write!(out, "{b:02x}").expect("writing to a String does not fail");
        out
    })
}

/// A named series of `(timestamp, value)` points
type Series = (&'static str, Vec<(i64, f32)>);

/// A named series and the segment it was fitted to
type Fitted = (&'static str, Vec<(i64, f32)>, DataSegment);

/// A named law fixture: degree and the function sampled to fit it
type LawFixture<'a> = (&'a str, usize, &'a dyn Fn(f64) -> f64);

/// Every parameter a model is stored with, in declaration order
fn model_bits(bits: &mut Bits, model: &ModelType) {
    bits.tag(model.name());
    match model {
        ModelType::Polynomial {
            coefficients,
            degree,
            fit_error,
        } => {
            bits.len(coefficients.len());
            for &c in coefficients {
                bits.f64(c);
            }
            bits.len(*degree);
            bits.f64(*fit_error);
        }
        ModelType::Fourier {
            coefficients,
            dc_offset,
            sample_count,
        } => {
            bits.len(coefficients.len());
            for &(k, magnitude, phase) in coefficients {
                bits.len(k);
                bits.f32(magnitude);
                bits.f32(phase);
            }
            bits.f32(*dc_offset);
            bits.len(*sample_count);
        }
        ModelType::SineWave {
            frequency,
            amplitude,
            phase,
            offset,
        } => {
            bits.f32(*frequency);
            bits.f32(*amplitude);
            bits.f32(*phase);
            bits.f32(*offset);
        }
        ModelType::MultiSine {
            components,
            dc_offset,
        } => {
            bits.len(components.len());
            for &(f, a, p) in components {
                bits.f32(f);
                bits.f32(a);
                bits.f32(p);
            }
            bits.f32(*dc_offset);
        }
        ModelType::PerlinNoise {
            seed,
            scale,
            octaves,
            persistence,
            lacunarity,
        } => {
            bits.u64(*seed);
            bits.f32(*scale);
            bits.u64(u64::from(*octaves));
            bits.f32(*persistence);
            bits.f32(*lacunarity);
        }
        ModelType::Constant { value } => bits.f64(*value),
        ModelType::Linear {
            start_value,
            end_value,
        } => {
            bits.f64(*start_value);
            bits.f64(*end_value);
        }
        ModelType::RawLzma { original_size, .. } => {
            // The compressed bytes depend on the compressor, not on the
            // arithmetic this file pins; only the fact that the fallback won
            // and how much it holds belong here.
            bits.len(*original_size);
        }
    }
}

/// The fixed series. Each one is built either from integer arithmetic or from
/// `alice_det_math`, so the input itself is the same bits everywhere and a
/// digest change can be attributed to the code under test.
fn series() -> Vec<Series> {
    let ts = |n: usize, f: &dyn Fn(usize) -> f32| -> Vec<(i64, f32)> {
        (0..n)
            .map(|i| (i64::try_from(i).expect("small index") * 10, f(i)))
            .collect()
    };
    #[allow(clippy::cast_precision_loss, reason = "indices below 2^24")]
    let x = |i: usize| i as f32;
    vec![
        ("constant", ts(64, &|_| 42.5)),
        ("linear", ts(100, &|i| 0.25f32.mul_add(x(i), -3.0))),
        (
            "quadratic",
            ts(128, &|i| {
                let t = x(i) / 128.0;
                3.0 * t * t - 2.0 * t + 0.5
            }),
        ),
        (
            "sine",
            ts(256, &|i| {
                2.0f32.mul_add(
                    alice_det_math::sin(core::f32::consts::TAU * 4.0 * x(i) / 256.0 + 0.3),
                    1.0,
                )
            }),
        ),
        (
            "two tones",
            ts(512, &|i| {
                let t = core::f32::consts::TAU * x(i) / 512.0;
                alice_det_math::sin(3.0 * t) + 0.4 * alice_det_math::cos(17.0 * t + 1.0)
            }),
        ),
        (
            "integer noise",
            #[allow(clippy::cast_precision_loss, reason = "values below 2^24")]
            ts(200, &|i| (((i * i + 7 * i) % 23) as f32) - 11.0),
        ),
    ]
}

/// Fits every series with one configuration and returns the segments
fn fit_all(lossless: bool) -> Vec<Fitted> {
    series()
        .into_iter()
        .map(|(name, data)| {
            // one more than the series, so the batch never flushes on its own
            let memtable = MemTable::with_config(
                data.len() + 1,
                FitConfig {
                    lossless,
                    ..FitConfig::default()
                },
            );
            assert!(memtable.put_batch(&data).is_empty());
            let segment = memtable.force_flush().expect("the series is not empty");
            (name, data, segment)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. The arithmetic identifier
//
// A lossless residual records this identifier, and a reader only applies a
// residual whose identifier matches the one it computes with. Pinning it here
// means a change to it cannot arrive as a silent dependency bump.
// ---------------------------------------------------------------------------

#[test]
fn semantics_id_is_the_recorded_value() {
    assert_eq!(
        alice_db::SEMANTICS_ID,
        alice_det_math::SEMANTICS_ID,
        "the identifier the residuals record is not the arithmetic this crate computes with"
    );
    assert_eq!(
        hex(&alice_db::SEMANTICS_ID),
        "d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e",
        "the numeric semantics changed: every residual pinned under the old \
         value is no longer applied on read"
    );
}

// ---------------------------------------------------------------------------
// 2. Model selection
// ---------------------------------------------------------------------------

#[test]
fn model_selection_is_the_recorded_bits() {
    let mut bits = Bits::default();
    let mut points = 0;
    for lossless in [false, true] {
        for (name, data, segment) in fit_all(lossless) {
            bits.tag(name);
            bits.i64(segment.start_time);
            bits.i64(segment.end_time);
            model_bits(&mut bits, &segment.model);
            bits.len(segment.metadata.point_count);
            bits.len(segment.metadata.model_size);
            bits.f64(segment.metadata.compression_ratio);
            match &segment.residual_blob {
                // The decompressed residual is `original ^ model` per sample:
                // the model's bits as the writer evaluated them
                Some(blob) => bits.raw(&decompress_residual(blob)),
                None => bits.u64(0),
            }
            points += data.len();
        }
    }
    assert_golden(
        "model selection",
        bits,
        // the lossless residuals alone are 4 bytes per point
        points / 2 * 4,
        "92b93535061e707389a08115d74493765ce47e48966be71b709ddd3d65059425",
    );
}

// ---------------------------------------------------------------------------
// 3. Reading back
// ---------------------------------------------------------------------------

#[test]
fn query_point_and_query_range_are_the_recorded_bits() {
    let mut bits = Bits::default();
    let mut values = 0;
    for lossless in [false, true] {
        for (name, data, segment) in fit_all(lossless) {
            bits.tag(name);
            // every stored timestamp, and the ones between them
            let end = data.last().expect("not empty").0;
            for t in (0..=end).step_by(5) {
                match segment.query_point(t) {
                    Some(v) => bits.f32(v),
                    None => bits.u64(u64::MAX),
                }
                values += 1;
            }
            for (start, stop) in [(0, end), (end / 3, end / 2), (end / 2, end / 2)] {
                let range = segment.query_range(start, stop);
                bits.len(range.len());
                for (t, v) in range {
                    bits.i64(t);
                    bits.f32(v);
                    values += 1;
                }
            }
        }
    }
    assert_golden(
        "query_point and query_range",
        bits,
        values * 4,
        "2b8604dc4a9af339621c6f157162c2b9d29191f1b9692ed9cdeb6207ee3b25df",
    );
}

// ---------------------------------------------------------------------------
// 4. Bloom filter sizing
// ---------------------------------------------------------------------------

#[test]
fn bloom_sizing_is_the_recorded_bits() {
    let mut bits = Bits::default();
    let rates = [0.5, 0.3, 0.1, 0.05, 0.01, 0.001, 1e-4, 1e-6, 0.779, 0.923];
    let mut sizes = 0;
    for &rate in &rates {
        let mut n = 0_usize;
        while n <= 5_000_000 {
            let filter = BloomFilter::with_capacity(n, rate);
            bits.len(n);
            bits.f64(rate);
            bits.u64(filter.num_bits());
            bits.u64(u64::from(filter.num_hashes()));
            sizes += 1;
            n = if n < 64 { n + 1 } else { n + n / 7 + 1 };
        }
    }
    assert_golden(
        "bloom sizing",
        bits,
        sizes * 32,
        "9d7933ea59b2d3fad65aa0633af3a39d99c44448aa1af688fdd7913bdc1393bf",
    );
}

// ---------------------------------------------------------------------------
// 5. Law identifiers
// ---------------------------------------------------------------------------

#[test]
fn law_id_and_evaluation_are_the_recorded_bits() {
    let fits: [LawFixture; 3] = [
        ("line", 1, &|x| 2.0f64.mul_add(x, 1.0)),
        ("parabola", 2, &|x| 0.5 * x * x - x + 3.0),
        ("cubic", 3, &|x| x * x * x - 2.0 * x),
    ];
    let mut bits = Bits::default();
    for (name, degree, f) in fits {
        let points: Vec<(f64, f64)> = (0..=8)
            .map(|i| {
                let x = f64::from(i) / 2.0;
                (x, f(x))
            })
            .collect();
        let law = SignalLaw::fit_polynomial(&points, degree, Provenance::new("golden", name))
            .expect("the fixture satisfies every fit precondition");
        bits.tag(name);
        bits.raw(&law.law_id(&alice_db::SEMANTICS_ID));
        for i in 0..=40 {
            bits.f64(
                law.evaluate(f64::from(i) / 10.0)
                    .expect("inside the domain"),
            );
        }
    }
    assert_golden(
        "law_id and evaluate",
        bits,
        3 * (32 + 41 * 8),
        "94bf9625675624f07452e43ba434cf7f7584b5abab968ba88a06c46dd6487481",
    );
}
