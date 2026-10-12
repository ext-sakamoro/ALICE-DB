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
enforces both in CI.

Whether a pinned SHA still matches its tag depends on the action's
repository, not on this one. So CI checks online only the pins a change adds
or edits (`--online --base auto`), and `pin-audit.yml` checks every pin once a
week; its failure is a prompt to update, not a broken build.

Pin to an immutable release tag (`# v2.87.27`). A branch pin
(`# stable (branch, ...)`) is only for a branch that only moves forward
(dtolnay/rust-toolchain); a branch that the action rebuilds on each release
(taiki-e/install-action's per-tool branches) drops the pinned commit, so use
its release tag with `with: tool: <name>`.

The pins are moved by hand, once a week or when an action publishes a fix:

```bash
python scripts/workflow_pin_update.py          # shows the diff, then checks every pin online
python scripts/workflow_pin_update.py --write  # applies it
```

A `vX.Y.Z` comment follows the major tag `vX` (a new major version is a
separate, reviewed change); a `(branch, ...)` comment follows the branch head,
and a branch that no longer contains the pin is reported and left alone. The
script does not commit. No update bot is used: commits by a bot account
appear among the repository's contributors.
