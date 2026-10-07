#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push`: every command below is
# the one .github/workflows/ci.yml, security-audit.yml or fuzz.yml runs, with the
# same arguments. A step this script does not cover is a step that can only fail
# remotely, so a step added to a workflow is added here in the same commit.
#
# usage: scripts/preflight.sh [--quick]
#   (none)   every gate: static checks, clippy, wasm build, docs, MSRV, feature
#            powerset, fuzz build, the full test suites (every feature set),
#            the law_store example, benches compile, and the security jobs
#            (cargo audit / deny / machete)
#   --quick  static checks, clippy, wasm build, docs and `cargo test --lib`;
#            skips the full suites, example, benches, MSRV, powerset, fuzz
#            build and the security jobs
#
# Not reproduced here (runner only): the time-boxed fuzz runs, the coverage and
# semver-checks jobs (informational in CI) and package-integrity.
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
case "${1:-}" in
  --quick) quick=1 ;;
  "") ;;
  *) echo "usage: scripts/preflight.sh [--quick]" >&2; exit 2 ;;
esac
MSRV=1.87

export CARGO_TERM_COLOR=always RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,sdf"

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate root invalidates only this crate's fingerprints.
relint() { touch src/lib.rs; }
add_target() { rustup target list --installed | grep -qx "$1" || rustup target add "$1"; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

need actionlint "brew install actionlint"
need python3 "python 3.9+"

step "ci.yml / actionlint: workflow YAML"
actionlint .github/workflows/*.yml

step "ci.yml / fmt: cargo fmt --check"
cargo fmt -- --check

step "ci.yml / docs-lint: tests + public documents / CHANGELOG structure"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "security-audit.yml / stub-guard"
scripts/stub_guard.sh

step "ci.yml / clippy: default, full native feature set, no default features (pedantic)"
relint
cargo clippy --all-targets -- -W clippy::pedantic -D warnings
relint
cargo clippy --features "$NATIVE_FEATURES" --all-targets -- -W clippy::pedantic -D warnings
relint
cargo clippy --no-default-features --all-targets -- -W clippy::pedantic -D warnings

step "ci.yml / wasm: wasm32-unknown-unknown build + clippy (no default features)"
add_target wasm32-unknown-unknown
cargo build --lib --target wasm32-unknown-unknown --no-default-features
relint
cargo clippy --lib --target wasm32-unknown-unknown --no-default-features -- -W clippy::pedantic -D warnings

step "ci.yml / doc: rustdoc -D warnings (default + full native feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$NATIVE_FEATURES"

if [[ $quick -eq 1 ]]; then
  step "cargo test --lib (quick)"
  cargo test --lib
  echo; echo "preflight --quick OK (full test suites, example, benches, MSRV, powerset, fuzz build and security jobs skipped)"; exit 0
fi

step "ci.yml / test: default, full native feature set, no default features, law_store (memory)"
scripts/run_tests.sh storage_backend_parity -- cargo test
scripts/run_tests.sh storage_backend_parity -- cargo test --features "$NATIVE_FEATURES"
scripts/run_tests.sh storage_backend_parity -- cargo test --no-default-features
scripts/run_tests.sh law_store -- cargo test --no-default-features --test law_store

step "ci.yml / test: law_store example"
cargo run --example law_store

step "ci.yml / test: benches compile"
cargo bench --no-run

step "ci.yml / msrv: rust-version = $MSRV"
if has_toolchain "$MSRV"; then
  cargo +"$MSRV" check --lib
  cargo +"$MSRV" check --lib --features "$NATIVE_FEATURES"
  cargo +"$MSRV" check --lib --no-default-features
else
  echo "toolchain $MSRV not installed (rustup toolchain install $MSRV --profile minimal)" >&2
  exit 1
fi

step "ci.yml / feature-powerset: {fs, ffi, sdf} depth 2"
need cargo-hack "cargo install cargo-hack --locked"
cargo hack check --lib --feature-powerset --depth 2 --exclude-features python,analytics,crypto

step "fuzz.yml: build every fuzz target (nightly)"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (cd fuzz && RUSTFLAGS= cargo +nightly fuzz build)
else
  echo "missing nightly toolchain or cargo-fuzz (cargo install cargo-fuzz --locked)" >&2
  exit 1
fi

step "security-audit.yml: cargo audit / cargo deny / cargo machete"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"
# RUSTSEC-2026-0235: see security-audit.yml
cargo audit --db "${CARGO_TARGET_DIR:-target}/advisory-db" --deny yanked \
  --ignore RUSTSEC-2026-0235 \
  --ignore RUSTSEC-2025-0141 \
  --ignore RUSTSEC-2024-0436 \
  --ignore RUSTSEC-2026-0192 \
  --ignore RUSTSEC-2024-0384 \
  --ignore RUSTSEC-2024-0388 \
  --ignore RUSTSEC-2024-0370 \
  --ignore RUSTSEC-2024-0320
cargo deny --all-features check all
cargo machete

echo; echo "preflight OK"
