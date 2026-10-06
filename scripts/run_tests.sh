#!/usr/bin/env bash
# scripts/run_tests.sh <required-test-binary> -- <cargo test command...>
#
# Runs the given cargo test command and fails unless
#   - the command itself succeeds,
#   - at least one test passed in total, and
#   - the named integration test binary ran and passed at least one test.
# A feature or target mismatch otherwise turns a test step into a silent
# no-op that still reports success.
set -euo pipefail

required="$1"
shift
[[ "${1:-}" == "--" ]] && shift

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

"$@" 2>&1 | tee "$log"

total=0
required_passed=0
current=""
while IFS= read -r line; do
  case "$line" in
    *"Running "*)
      current="$line"
      ;;
    "test result: "*)
      n=$(printf '%s\n' "$line" | sed -E 's/.* ([0-9]+) passed.*/\1/')
      total=$((total + n))
      if [[ "$current" == *"tests/${required}.rs"* || "$current" == *"tests\\${required}.rs"* ]]; then
        required_passed=$((required_passed + n))
      fi
      ;;
  esac
done < <(tr -d '\r' < "$log")

echo "run_tests: ${total} tests passed in total, ${required_passed} in ${required}"
if [[ "$total" -eq 0 ]]; then
  echo "run_tests: no test ran" >&2
  exit 1
fi
if [[ "$required_passed" -eq 0 ]]; then
  echo "run_tests: ${required} did not run any test" >&2
  exit 1
fi
