#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push`: every command below is
# the one .github/workflows/ci.yml, release-profile.yml, security-audit.yml or
# fuzz.yml runs, with the same arguments. A step this script does not cover is
# a step that can only fail remotely, so a step added to a workflow is added
# here in the same commit.
#
# usage: scripts/preflight.sh [--quick]
#   (none)   every gate: static checks, clippy, wasm build, docs, MSRV, feature
#            powerset, fuzz build, the full test suites (every feature set),
#            the goldens under the release profile,
#            the law_store example, benches compile, and the security jobs
#            (cargo audit / deny / machete)
#   --quick  static checks, clippy, wasm build, docs and `cargo test --lib`;
#            skips the full suites, example, benches, MSRV, powerset, fuzz
#            build and the security jobs
#
# Not reproduced here (runner only): the time-boxed fuzz runs, the coverage job
# (informational in CI) and package-integrity.
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
case "${1:-}" in
  --quick) quick=1 ;;
  "") ;;
  *) echo "usage: scripts/preflight.sh [--quick]" >&2; exit 2 ;;
esac
MSRV=1.87

export CARGO_TERM_COLOR=always RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,sdf,analytics,crypto"

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
python3 scripts/check_py_license.py
python3 scripts/test_workflow_concurrency.py
python3 scripts/workflow_concurrency.py
python3 scripts/test_entropy_guard.py
python3 scripts/entropy_guard.py
python3 scripts/test_libm_dependency_guard.py
python3 scripts/libm_dependency_guard.py
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py
python3 scripts/test_run_tests.py
python3 scripts/test_fuzz_reach.py
python3 scripts/test_tracked_binaries.py
python3 scripts/tracked_binaries.py

step "ci.yml / docs-lint: changed_paths tests + sync (fuzz.yml / security-audit.yml trigger and pattern files)"
python3 scripts/test_changed_paths.py
python3 scripts/changed_paths.py sync

step "ci.yml / docs-lint: workflow pins (SHA, permissions)"
python3 scripts/test_workflow_pin_check.py
# only pins added or edited since origin/main (the weekly pin-audit.yml checks all)
python3 scripts/workflow_pin_check.py --online --base origin/main
python3 scripts/test_workflow_pin_update.py

step "ci.yml / clippy: clippy canary data (generator tests, generated from clippy.toml, no drift)"
python3 scripts/test_gen_clippy_canary_calls.py
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / clippy: clippy canary tests (parsers, accounting, one end-to-end run)"
python3 scripts/test_clippy_canary.py

step "ci.yml / clippy: clippy external coverage (tests, then every public function of a named module is banned or allowed)"
python3 scripts/test_clippy_external_coverage.py
python3 scripts/clippy_external_coverage.py

step "security-audit.yml / stub-guard"
scripts/stub_guard.sh

step "ci.yml / clippy: default, full native feature set, no default features (pedantic), python (lib)"
relint
cargo clippy --all-targets -- -W clippy::pedantic -D warnings
relint
cargo clippy --features "$NATIVE_FEATURES" --all-targets -- -W clippy::pedantic -D warnings
relint
cargo clippy --no-default-features --all-targets -- -W clippy::pedantic -D warnings
relint
cargo clippy --lib --features python -- -D warnings

step "ci.yml / wasm: wasm32-unknown-unknown build + clippy (no default features)"
add_target wasm32-unknown-unknown
cargo build --lib --target wasm32-unknown-unknown --no-default-features
relint
cargo clippy --lib --target wasm32-unknown-unknown --no-default-features -- -W clippy::pedantic -D warnings

step "ci.yml / doc: rustdoc -D warnings (default + full native feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$NATIVE_FEATURES"

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

if [[ $quick -eq 1 ]]; then
  step "cargo test --lib (quick)"
  cargo test --lib
  echo; echo "preflight --quick OK (full test suites, example, benches, MSRV, powerset, fuzz build and security jobs skipped)"; exit 0
fi

step "ci.yml / clippy: clippy canary (every planted banned call is flagged)"
python3 scripts/clippy_canary.py

step "ci.yml / test: default, full native feature set, no default features, law_store (memory)"
scripts/run_tests.sh storage_backend_parity,law_single_source,legacy_segment_compat,residual_arithmetic,determinism_golden,crash_reopen_process,blob_format_pin,blob_merge_semantics,exact_series,segment_range_arith,get_reads_memtable,lossless_irregular_keys,sparse_steps_repro,query_group_by_interval,query_group_by_floor,float_edge_golden,float_environment=2,db_semantics_id,law_record_semantics -- cargo test
scripts/run_tests.sh storage_backend_parity,law_single_source,legacy_segment_compat,residual_arithmetic,determinism_golden,crash_reopen_process,blob_format_pin,blob_merge_semantics,exact_series,segment_range_arith,get_reads_memtable,lossless_irregular_keys,sparse_steps_repro,query_group_by_interval,query_group_by_floor,analytics_metrics_exact,float_edge_golden,float_environment=2,db_semantics_id,law_record_semantics -- cargo test --features "$NATIVE_FEATURES"
scripts/run_tests.sh storage_backend_parity,law_single_source,legacy_segment_compat,determinism_golden,exact_series,segment_range_arith,get_reads_memtable,lossless_irregular_keys,sparse_steps_repro,query_group_by_interval,query_group_by_floor,float_edge_golden,float_environment=2,db_semantics_id,law_record_semantics -- cargo test --no-default-features
scripts/run_tests.sh law_store -- cargo test --no-default-features --test law_store

step "release-profile.yml / release: goldens under the release profile (fail on 0 tests)"
scripts/run_tests.sh determinism_golden,float_edge_golden,float_environment=2 -- cargo test --release --test determinism_golden --test float_edge_golden --test float_environment
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --all-features
python3 scripts/build_flags_check.py -- cargo check --release --lib --no-default-features

step "ci.yml / test: law_store example"
cargo run --example law_store

# README / README_JP の ```rust block は src/lib.rs の cfg(doctest) 経由で
# doctest として取り込んでいる ci.yml の同名 step と対で置く
step "ci.yml / test: doctests (includes both READMEs)"
cargo test --doc

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
cargo hack check --lib --feature-powerset --depth 2 --exclude-features python

step "ci.yml / python: maturin build + README example test (fail on 0 tests)"
need maturin "pip install 'maturin>=1.5,<2.0'"
pf_venv="$(mktemp -d)/venv"; pf_dist="$(mktemp -d)"
python3 -m venv "$pf_venv"
"$pf_venv/bin/pip" install -q numpy pytest
maturin build --out "$pf_dist"              # features from [tool.maturin]; version dynamic from Cargo.toml
"$pf_venv/bin/pip" install -q "$pf_dist"/*.whl
"$pf_venv/bin/python" -c "import alice_db; print('alice_db', alice_db.__version__, 'imported')"
pf_n=$("$pf_venv/bin/python" -m pytest python/tests --collect-only -q 2>/dev/null | grep -c '::')
echo "collected ${pf_n} python test(s)"
[[ "${pf_n}" -gt 0 ]] || { echo "no python tests collected" >&2; exit 1; }
"$pf_venv/bin/python" -m pytest python/tests -v

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

step "semver-checks.yml: semver gate tests + cargo semver-checks against the latest crates.io release"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
python3 scripts/test_semver_gate.py
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

echo; echo "preflight OK"
