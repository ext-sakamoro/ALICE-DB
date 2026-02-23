# Contributing to ALICE-DB

## Build

```bash
cargo build
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Optional Features

```bash
# ALICE-Analytics bridge
cargo test --features analytics

# ALICE-Crypto encryption at rest
cargo test --features crypto

# Python bindings (requires Python environment)
cargo build --features python
```

## Design Constraints

- **Model-based storage**: data is stored as mathematical models (Polynomial, Fourier, etc.), not raw values.
- **Kolmogorov compression**: the shortest program that produces the output is the optimal representation.
- **rkyv zero-copy**: segments use zero-copy deserialization via mmap — no parsing or allocation on read.
- **LSM-tree architecture**: MemTable → flush → immutable Segments → compaction.
- **Automatic model selection**: fitter competition picks the best model per segment.
- **LZMA fallback**: when no procedural model meets the error threshold, raw data is LZMA-compressed.
