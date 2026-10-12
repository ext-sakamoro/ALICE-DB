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

## Workflow actions

Every external action in `.github/workflows/` is pinned to a full commit SHA
with the tag (or branch) it came from as a comment, and every workflow sets
`permissions: contents: read` at the top. `scripts/workflow_pin_check.py`
enforces both in CI; with `--online` it also confirms that each SHA is the
commit its comment names.

The pins are moved by hand, once a week or when an action publishes a fix:

```bash
python scripts/workflow_pin_update.py          # shows the diff
python scripts/workflow_pin_update.py --write  # applies it
python scripts/workflow_pin_check.py --online
```

A `vX.Y.Z` comment follows the major tag `vX` (a new major version is a
separate, reviewed change); a `(branch, ...)` comment follows the branch head.
The script does not commit. No update bot is used: commits by a bot account
appear among the repository's contributors.
