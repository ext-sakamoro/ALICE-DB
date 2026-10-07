# ALICE-DB

[日本語](README_JP.md)

A model-based LSM-tree database for numeric time series. When the memtable is
flushed, ALICE-DB fits candidate models (polynomial, Fourier, sine, Perlin
noise, constant, linear, with an LZMA fallback) to the buffered points through
[ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) and stores the model that
describes them instead of the samples; a point or range query evaluates the
model. Alongside the time series it has a byte-keyed blob store, and a law store
that keeps fitted laws together with their evidence, valid range and the
verdicts of later evidence.

Storage is either a directory (WAL, mmap'd segments, advisory locks; `fs`
feature, on by default) or process memory (`AliceDB::in_memory`, which also
builds for `wasm32-unknown-unknown`). The API is the same for both.

License: AGPL-3.0-or-later OR LicenseRef-Commercial

## What it is not for

- **Exact storage by default.** The default `FitConfig` is lossy: a model is
  accepted when its error is under the documented threshold of its kind (for
  example relative MSE < 0.1 for a single sine). Set `lossless: true` in
  `FitConfig` to store an XOR residual and read back the exact bits.
- **Irregular timestamps.** A segment assumes uniform spacing between its first
  and last timestamp; series with gaps are not reproduced point for point.
- **General SQL or document workloads.** Values are `f32` per `i64`
  timestamp, plus opaque blobs and laws.

## Contents

- [Installation](#installation)
- [Example](#example)
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

## Laws: evidence, valid range and verdicts

`alice_db::law_store` stores `SignalLaw` values from ALICE-Zip (re-exported in
the module): `y = f(x)` as a polynomial, together with the points it was fitted
to, the residual measured over them, the `x` range they cover, provenance and
reference values.

| Method | Effect |
|--------|--------|
| `put_law(name, &law)` | stores the law as the next version of `name` |
| `get_law(name)` / `get_law_version(name, v)` | restores a version; the residual is measured again from the stored evidence |
| `evaluate_law(name, x)` | `f(x)` of the latest version; an `x` outside the valid range is refused (`LawError::OutOfRange`), never extrapolated |
| `ingest_evidence(name, points, &policy)` | judges new points (no evidence / out of range / supports / parameter update / residual grew / breaks), records the verdict, and stores a parameter update as a new version while older versions stay readable |
| `law_history(name)` / `law_versions(name)` / `law_names()` | recorded verdicts in order (with evidence count and RMS), stored versions, stored names |

```rust
use alice_db::law_store::{IngestPolicy, Provenance, SignalLaw, Verdict};
use alice_db::{AliceDB, StorageConfig};

let db = AliceDB::in_memory(StorageConfig::default())?;
// y = 1 + 2x measured at x = 0..=4
let pts: Vec<(f64, f64)> = (0..5).map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i))).collect();
let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("run 1", "least squares"))?;
db.put_law("line", &law)?;

assert!((db.evaluate_law("line", 2.5)? - 6.0).abs() < 1e-12);
assert!(db.evaluate_law("line", 9.0).is_err()); // outside [0, 4]

let policy = IngestPolicy { abs_tolerance: 0.01, break_factor: 4.0 };
let v = db.ingest_evidence("line", &[(0.5, 2.0), (3.5, 8.0)], &policy)?;
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

## Storage backends

| Backend | Constructor | Notes |
|---------|-------------|-------|
| Files (`fs`) | `AliceDB::open(path)`, `AliceDB::with_config(config)` | segment files + index, time-series WAL, blob WAL and `SSTable`s, mmap'd reads, exclusive advisory lock on the blob WAL |
| Memory | `AliceDB::in_memory(config)`, `AliceDB::from_bytes(config, &bytes)` | no filesystem; `to_bytes` exports a whole database (either backend) as one checksummed buffer |

A given sequence of writes reads back bit-identically from either backend
(`tests/storage_backend_parity.rs`).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `fs` | yes | File storage (`open` / `with_config`: WAL, mmap, advisory locks). Without it only the memory backend is available |
| `ffi` | no | C / C++ / C# FFI (implies `fs`), headers in `bindings/` |
| `python` | no | Python bindings (PyO3 + NumPy, implies `fs`) |
| `analytics` | no | ALICE-Analytics bridge: aggregated metrics to time series (implies `fs`) |
| `crypto` | no | ALICE-Crypto encryption at rest (`crypto_bridge::EncryptedDB`, implies `fs`) |
| `sdf` | no | SDF spatial data storage with Morton-code indexing (implies `fs`) |

`analytics` and `crypto` depend on the sibling repositories `../ALICE-Analytics`
and `../ALICE-Crypto`.

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
db.put_batch([(i, i * 0.5) for i in range(1000)])
print(db.get(100), db.aggregate(0, 999, "avg"))
db.close()
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
| [ALICE-Analytics](https://github.com/ext-sakamoro/ALICE-Analytics) | streaming aggregation (`analytics` feature) |
| [ALICE-Crypto](https://github.com/ext-sakamoro/ALICE-Crypto) | encryption at rest (`crypto` feature) |
| [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) | model fitting on embedded devices |

## License

`AGPL-3.0-or-later OR LicenseRef-Commercial`: dual-licensed, pick either.

| Option | Terms | Use it when |
|--------|-------|-------------|
| **AGPL-3.0-or-later** | [LICENSE-AGPL](LICENSE-AGPL), free | your project is itself AGPL-compatible open source, or you use it only internally |
| **Commercial License** | [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md), paid, removes the copyleft | closed-source products, proprietary SaaS, firmware or plugin distribution |

AGPL is a strong copyleft: a product, firmware image or service that links
`alice-db` and is distributed or served to users must be released under the
AGPL as well.

Commercial licence enquiries: <contact@extoria.co.jp>
