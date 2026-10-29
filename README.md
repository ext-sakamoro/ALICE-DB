# ALICE-DB

[日本語](README_JP.md)

A model-based LSM-tree database for numeric time series. When the memtable is
flushed, ALICE-DB fits candidate models (polynomial, Fourier, sine, Perlin
noise, constant, linear, with an LZMA fallback) to the buffered points through
[ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) and stores the model that
describes them instead of the samples; a point or range query evaluates the
model. Alongside the time series it has exact series (values stored bit for
bit under any `i64` key, no model), a byte-keyed blob store, and a law store
that keeps fitted laws together with their evidence, valid range and the
verdicts of later evidence.

Storage is either a directory (WAL, segment files, advisory locks; `fs`
feature, on by default) or process memory (`AliceDB::in_memory`, which also
builds for `wasm32-unknown-unknown`). The API is the same for both.

License: LicenseRef-Proprietary

## What it is not for

- **Exact storage by default.** The default `FitConfig` is lossy: a model is
  accepted when its error is under the documented threshold of its kind (for
  example relative MSE < 0.1 for a single sine). Set `lossless: true` in
  `FitConfig` to store an XOR residual and read back the exact bits. A residual
  stores `original ^ model`, so it reconstructs the original only when the
  reader evaluates the law with the same arithmetic; the blob therefore carries
  the identifier of that arithmetic. A read that meets a residual naming a
  different one fails with an `InvalidData` error carrying a
  `ResidualError::Mismatch` (the segment, its key range, the store and both
  identifiers) instead of returning a value: applying the residual would give
  neither the original nor the model, and the model alone is not the
  original. A residual that does not decode to one correction per point is
  refused the same way as `ResidualError::Corrupt`. A range read returns all
  its points or the error, never fewer or partly corrected ones; only the
  segments that still answer a key count, so an old segment whose keys all
  have newer writes does not fail it. `StorageConfig::residual_policy = ResidualPolicy::ModelOnly` (or
  `query_point_model_only` / `query_range_model_only` on a segment) reads the
  model's approximation on purpose, and `residual_status()` reports which
  case a segment is in without reading. **Recovering the exact values** of a
  store written with another arithmetic takes that arithmetic: a store written
  by alice-db 0.3 (alice-det-math 0.4) is read exactly by alice-db 0.3, so read
  it there and write the values again with this version; this version alone
  can give only the model. A **corrupt** residual (`ResidualError::Corrupt`)
  is not recoverable by any version: the corrections are lost, so restore the
  segment from a backup (or accept the model with `ModelOnly`). Residuals
  written before 0.3.0 carry no
  identifier: they are applied, they are exact on the machine that wrote
  them, and that cannot be checked anywhere else. Values
  that must come back exactly under arbitrary keys belong in an
  [exact series](#exact-series).
- **Irregular timestamps in the model store.** A segment keeps no per-point
  timestamps and reads its points back on an even grid between its first and
  last timestamp. In the default (lossy) mode a series with gaps is not
  reproduced point for point. Lossless mode cuts a new segment wherever the
  spacing changes, so increasing timestamps with any gaps read back exactly
  (at the cost of small segments that compress poorly), and it refuses a
  timestamp at or below one already written with
  `LosslessKeyOrderError` instead of storing it inexactly.
- **General SQL or document workloads.** Values are `f32` per `i64`
  timestamp, plus opaque blobs and laws.

## Contents

- [Installation](#installation)
- [Example](#example)
- [Exact series](#exact-series)
- [Laws: evidence, valid range and verdicts](#laws-evidence-valid-range-and-verdicts)
- [Storage backends](#storage-backends)
- [Features](#features)
- [Model types](#model-types)
- [Python](#python)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [Building and testing](#building-and-testing)
- [Related crates](#related-crates)
- [License](#license)

## Installation

```sh
cargo add alice-db
# browser / no filesystem: memory backend only
cargo add alice-db --no-default-features
```

## Example

```rust
use alice_db::{Aggregation, AliceDB, StorageConfig};

// `AliceDB::open("./my_data")` keeps the same data in files (`fs` feature)
let db = AliceDB::in_memory(StorageConfig::default())?;

// y = 0.5 t + 10 at t = 0..1000
for t in 0..1000_i64 {
    db.put(t, 0.5 * t as f32 + 10.0)?;
}
db.flush()?; // fit a model to the buffered points and store it as a segment

// point query, computed from the stored model
let v = db.get(500)?.expect("t = 500 was stored");
assert!((v - 260.0).abs() < 1e-2);

// range aggregation
let avg = db.aggregate(0, 999, Aggregation::Avg)?;
assert!((avg - 259.75).abs() < 1e-2);
# Ok::<(), std::io::Error>(())
```

## Exact series

For records that must read back exactly (metrics, simulation step counters,
sparse or out-of-order keys) use a named series instead of the model store.
Each point is one record of the blob store: `f32` and `f64` values keep their
bits, a scan returns exactly the keys that were written, in key order, and
separate names never see each other's points.

```rust
use alice_db::{AliceDB, StorageConfig};

let db = AliceDB::in_memory(StorageConfig::default())?;
let energy = db.series("energy")?;
for (i, step) in [0i64, 1, 1000, 1_000_003, 1 << 40, (1 << 53) - 1].into_iter().enumerate() {
    energy.put_f32(step, 0.1 + i as f32 * 1.7)?;
}
assert_eq!(energy.get_f32(1000)?, Some(0.1 + 2.0 * 1.7));
assert_eq!(energy.scan_f32(0, 1 << 41)?.len(), 5);
# Ok::<(), std::io::Error>(())
```

The `analytics` feature writes its metrics into the series
`analytics_bridge::METRICS_SERIES`.

## Laws: evidence, valid range and verdicts

`alice_db::law_store` stores `SignalLaw` values from ALICE-Zip (re-exported in
the module): `y = f(x)` as a polynomial, together with the points it was fitted
to, the residual measured over them, the `x` range they cover, provenance and
reference values.

| Method | Effect |
|--------|--------|
| `put_law(name, &law, &semantics_id)` | stores the law as the next version of `name`, and indexes it under its content identifier |
| `get_law(name)` / `get_law_version(name, v)` | restores a version; the residual is measured again from the stored evidence. The result carries `RecordSemantics`: `Current`, `VerifiedPrevious(id)`, or `Unpinned` for a record written before format 2. A record under an identifier this build does not reproduce is refused (`LawRecordMismatch`, both identifiers named); `get_law_with(name, LawReadPolicy::RequirePinned)` also refuses `Unpinned` |
| `evaluate_law(name, x)` | `f(x)` of the latest version with the same `RecordSemantics`; an `x` outside the valid range is refused (`LawError::OutOfRange`), never extrapolated |
| `ingest_evidence(name, points, &policy, &semantics_id)` | judges new points (no evidence / out of range / supports / parameter update / residual grew / breaks), records the verdict, and stores a parameter update as a new version while older versions stay readable |
| `law_pointers_by_id(&id)` / `evaluate_by_id(&id, x)` | every `(name, version)` stored under one content identifier, and `f(x)` through it |
| `law_history(name)` / `law_versions(name)` / `law_names()` | recorded verdicts in order (with evidence count and RMS), stored versions, stored names |

```rust
use alice_db::law_store::{IngestPolicy, Provenance, SignalLaw, Verdict};
use alice_db::{AliceDB, SEMANTICS_ID, StorageConfig};

// `SEMANTICS_ID` identifies the arithmetic the law is evaluated with. It is
// re-exported from the crate that provides the point evaluators, so it is the
// identifier of the arithmetic this build actually uses -- do not substitute a
// constant of your own. It goes into the law's content identifier, so a stored
// result names both the law and the numeric semantics it was computed under.

let db = AliceDB::in_memory(StorageConfig::default())?;
// y = 1 + 2x measured at x = 0..=4
let pts: Vec<(f64, f64)> = (0..5).map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i))).collect();
let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("run 1", "least squares"))?;
db.put_law("line", &law, &SEMANTICS_ID)?;

assert!((db.evaluate_law("line", 2.5)?.value - 6.0).abs() < 1e-12);
assert!(db.evaluate_law("line", 9.0).is_err()); // outside [0, 4]

// The stored law is reachable by its content identifier as well as by name.
let id = law.law_id(&SEMANTICS_ID);
assert_eq!(db.law_pointers_by_id(&id)?, vec![("line".to_string(), 1_u64)]);

let policy = IngestPolicy { abs_tolerance: 0.01, break_factor: 4.0 };
let v = db.ingest_evidence("line", &[(0.5, 2.0), (3.5, 8.0)], &policy, &SEMANTICS_ID)?;
assert!(matches!(v, Verdict::Supports { .. }));
assert_eq!(db.law_history("line")?.len(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Laws are records in the blob store under the reserved key prefix
`"\0alice-law\0"`, so the file backend writes them to the blob WAL / `SSTable`s
and `to_bytes` carries them. Each record is checksummed and names its own key; a
damaged record, one copied under another key, or one that `SignalLaw::from_parts`
refuses (for example evidence outside the stored range) is returned as an error.
The byte layout is documented in the `law_store` module.
`cargo run --example law_store` runs the whole cycle on both backends.


### Addressing a law by its content

A name is something a person assigns and can change. `SignalLaw::law_id`
(ALICE-Zip) is a 32-byte identifier computed from the law itself — the valid
range and the coefficients — together with an identifier for the numeric
semantics it is evaluated under, so a stored result can name the law it came
from without relying on a name staying put.

`put_law` and `ingest_evidence` take that semantics identifier and index every
version they write under it. `law_pointers_by_id` returns every
`(name, version)` for one identifier, and `evaluate_by_id` evaluates it: that
is well defined even for several pointers, because equal identifier implies the
evaluation returns the same bits for every `x`. The converse does not hold — two
laws that evaluate identically can still differ in identifier — so an identifier
is not a deduplication key.

## Storage backends

| Backend | Constructor | Notes |
|---------|-------------|-------|
| Files (`fs`) | `AliceDB::open(path)`, `AliceDB::with_config(config)` | segment files + index, time-series WAL, blob WAL and `SSTable`s, mmap'd `SSTable` reads, exclusive advisory lock on the blob WAL |
| Memory | `AliceDB::in_memory(config)`, `AliceDB::from_bytes(config, &bytes)` | no filesystem; `to_bytes` exports a whole database (either backend) as one checksummed buffer |

A given sequence of writes reads back bit-identically from either backend
(`tests/storage_backend_parity.rs`).

A segment file is decoded once when it is opened, into the same segment type
the writer held, and queried through the same code as an in-memory segment
(`tests/segment_view_parity.rs`). An open segment therefore costs about its
file's size in memory (the model's coefficients and the residual bytes); the
file is not kept mapped. Segment files carry a stored format number (format 1:
a header, then an rkyv 0.8 archive); files written before 0.6.0 (format 0, an
rkyv 0.7 archive without a header) are still read.

On open, the file backend cuts a torn or corrupt tail off the blob WAL at the
last good record, so later writes are not lost behind it. The dropped bytes are
kept in `blob.wal.dropped-<offset>` next to the WAL, written and fsynced before
the cut, and `AliceDB::blob_wal_recovery()` reports what was cut. A later
recovery at the same offset never replaces an earlier copy (it writes
`blob.wal.dropped-<offset>.<n>`), and finished copies are never removed. A WAL
that ends on a record boundary is left as it is.

Recovery keeps the WAL up to its first bad record and nothing after it: a
corrupt record in the middle of the log also drops every acknowledged record
written after it from the live store. Those records are kept only in the
dropped copy; the store does not replay past a record whose lengths it cannot
trust.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `fs` | yes | File storage (`open` / `with_config`: WAL, mmap, advisory locks). Without it only the memory backend is available |
| `ffi` | no | C / C++ / C# FFI (implies `fs`), headers in `bindings/` |
| `python` | no | Python bindings (PyO3 + NumPy, implies `fs`) |
| `analytics` | no | ALICE-Analytics bridge: aggregated metrics to an exact series (implies `fs`) |
| `crypto` | no | ALICE-Crypto encryption at rest (`crypto_bridge::EncryptedDB`, implies `fs`) |
| `sdf` | no | SDF spatial data storage with Morton-code indexing (implies `fs`) |

`analytics` and `crypto` take `alice-analytics` 0.7 and `alice-crypto` 0.6 by git at
a pinned commit of their repositories.

## Model types

| Model | Fits |
|-------|------|
| `Constant` | flat segments |
| `Linear` | ramps |
| `Polynomial` | curves, drift |
| `Fourier` | periodic signals |
| `SineWave` / `MultiSine` | oscillations |
| `PerlinNoise` | noise-like texture |
| `RawLzma` | fallback when no model is accepted |

The model is chosen per segment by fit quality and size. Compression and query
speed depend on the data; `cargo bench` measures them on your hardware.

## Python

```sh
pip install maturin
maturin develop --release --features python
```

```python
import alice_db

db = alice_db.open("./my_timeseries")
timestamps = list(range(1000))
values = [i * 0.5 for i in range(1000)]
db.put_batch(timestamps, values)  # two equal-length sequences
print(db.get(100), db.aggregate(0, 999, "avg"))
db.close()
```

`open` defaults to the lossy model store (`lossless=False`): a series whose
timestamps have gaps reads back on an even grid, so a gapped point can change
after a flush. Pass `lossless=True` to store residuals and read increasing
timestamps back exactly (the same promise as the Rust `FitConfig { lossless: true }`):

```python
db = alice_db.open("./exact", lossless=True)
db.put_batch([0, 1, 2, 10], [10.0, 20.0, 30.0, 31.0])
db.flush()
print([db.get(t) for t in (0, 1, 2, 10)])  # [10.0, 20.0, 30.0, 31.0] — every written point exact
db.close()  # the lossy default drifts on the interior gap (e.g. get(1) reads 10.0, not 20.0)
```

## Minimum supported Rust version

Rust 1.87 (`rust-version` in `Cargo.toml`), checked in CI for the library with
default features, the native feature set and no default features. 1.86 is
refused by dependencies that declare 1.87. Development and CI use the toolchain
pinned in `rust-toolchain.toml`.

## Building and testing

```sh
cargo test                                   # file backend (default)
cargo test --no-default-features             # memory backend only
cargo test --features ffi,sdf                # native feature set
cargo run --example law_store
cargo clippy --all-targets -- -W clippy::pedantic -D warnings
cargo bench
scripts/preflight.sh            # every CI step with the same arguments
scripts/preflight.sh --quick    # static checks, clippy, wasm build, docs and `cargo test --lib`
```

## Related crates

| Crate | Relation |
|-------|----------|
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | model fitting and generators; `law::SignalLaw` |
| [ALICE-DetMath](https://github.com/ext-sakamoro/ALICE-DetMath) | transcendentals that return the same bits on every target (bloom filter sizing, and the laws through ALICE-Zip); `SEMANTICS_ID` |
| [ALICE-Analytics](https://github.com/ext-sakamoro/ALICE-Analytics) | streaming aggregation (`analytics` feature) |
| [ALICE-Crypto](https://github.com/ext-sakamoro/ALICE-Crypto) | encryption at rest (`crypto` feature) |
| [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) | model fitting on embedded devices |

<!-- law-integration:begin law=7f5f4cbde8343a061051bb1addd30f1f0b3bea1e -->
## ALICE-Law integration

Generated by ALICE-Law `scripts/gen-law-readme-block.py` from ALICE-Law `7f5f4cbde834` and this repository's `alice-law-conformance.toml`; do not edit it by hand.

| | |
|---|---|
| Release train | alice-db 0.5.1 (published), law dependency: yes |
| Arithmetic | alice-det-math 0.5.1 in the train, arithmetic identifier `85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315`; this crate requires `alice-det-math = "0.5.2"` |
| End to end check | covered (`alice-db = "0.5"`), digest `9f7cf676708a272e3817ebbf0727c610f391e1547747af4307ee44d5523267c2`, the same on aarch64 and x86_64 |

### Conformance (G1-G14)

| gate | status | note |
|---|---|---|
| G1 no-libm-direct | pass | every banned method of the G1 list is in clippy.toml, every feature is linted with -D warnings (python included, without pedantic: the pyo3 binding trips … |
| G2 ieee-max-min | pass | raw f32 / f64 max, min and clamp are a default-deny in clippy.toml, with no exception and so no raw_* manifest; the helpers come from alice-det-math, the owner … |
| G3 no-outer-fneg-of-fma | pass | the library's only fma calls are the fit competition's model score (not a stored value); the 114 fma calls in the IR otherwise come from alice-det-math and … |
| G4 debug-release-arch-goldens | pass | boundary: a stored value read back (get / scan from the memtable, or from a segment through its residual: lossless, or a lossy-configured segment holding a … |
| G5 release-profile-test-leg | pass | the aarch64 release leg is macos-latest. The repository has no branch protection; a leg is required in the sense that main moves only by a fast-forward after … |
| G6 semantics-id | pass | alice-db computes its own identifier, DB_SEMANTICS_ID (670fa328...1223a7 with alice-zip 0.11 at 19f6f2c1 over alice-det-math 0.5.2 at 2d01c63f), from the bits … |
| G7 handshake-carries-id | pass | the Law-bearing value that crosses this crate's boundary is the stored law record (alice-zip SignalLaw parts), handed over in AliceDB::to_bytes / from_bytes … |
| G8 persisted-semantics-id | pass | every durable form that carries a value computed under an arithmetic carries that arithmetic's identifier: lossless segment residuals since 0.3.0 (the … |
| G9 persisted-meaning-change-is-major | pass | the audit covers the identifier this crate persists and finds no violation, and tests/identifier_history.rs reads a committed store stamped with each earlier … |
| G10 no-run-varying-artifacts | pass | no ambient clock or OS-entropy read is known in a Law-bearing path: `python3 scripts/entropy_guard.py` reports `24 files, 10 patterns, 0 problems` … |
| G11 gate-self-tests | pass | clause (c) passes per crate: every copy this crate carries is at the target ALICE-Law 9cb423c's shared-files table records (scripts/shared_files.sh: 0 … |
| G12 public-repo-text-rules | pass | the exemptions in .public-text.toml are counted per file and class (the documentation linter's own word list and its tests) |
| G13 no-unspecified-order | pass | the order-dependent outputs found earlier (model_distribution, detect_deadlock) were fixed with an oracle each; the remaining HashMap / HashSet sites, … |
| G14 float-environment-invariants | pass | the static scan covers the files of this repository and the dynamic check covers the builds CI runs on ubuntu and macOS; a flag passed by a build outside this … |
| G15 surface-claims | not declared |  |

### Features and the guarantee

| feature | guarantee | note |
|---|---|---|
| `alice-analytics` | not declared |  |
| `analytics` | not declared |  |
| `crypto` | not declared |  |
| `ffi` | not declared |  |
| `fs` | not declared |  |
| `numpy` | not declared |  |
| `open-file-audit` | not declared |  |
| `pyo3` | not declared |  |
| `python` | not declared |  |
| `sdf` | not declared |  |
<!-- law-integration:end -->

## License

`LicenseRef-Proprietary`. The text is [LICENSE-Proprietary](LICENSE-Proprietary).

Enquiries: https://extoria.co.jp/
