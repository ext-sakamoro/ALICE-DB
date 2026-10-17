#!/usr/bin/env python3
"""The release legs run the same test binaries as the debug legs (ALICE-Law G5).

.github/workflows/release-profile.yml repeats each `scripts/run_tests.sh <required> -- cargo test
<args>` step of the test job in .github/workflows/ci.yml with `--release`. A binary added to the
ci.yml list only would run in debug and silently not be required in release. This check pairs the
steps by their cargo arguments (with `--release` removed) and fails when a step has no partner,
when the required lists of a pair differ, or when it compared nothing.

usage: scripts/release_lists_check.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STEP_RE = re.compile(r"^\s*run:\s*scripts/run_tests\.sh\s+(?P<list>\S+)\s+--\s+(?P<cmd>cargo test.*?)\s*$")


def steps(text: str, release: bool) -> dict[str, str]:
    """cargo arguments (without --release) -> required list, for every single-line run_tests step"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        args = m["cmd"].split()
        if release != ("--release" in args):
            continue
        key = " ".join(a for a in args if a != "--release")
        if key in out:
            raise ValueError(f"two steps run `{key}`")
        out[key] = m["list"]
    return out


def check(root: Path) -> tuple[list[str], int]:
    debug = steps((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"), release=False)
    rel = steps((root / ".github/workflows/release-profile.yml").read_text(encoding="utf-8"), release=True)
    problems: list[str] = []
    compared = 0
    for key in sorted(set(debug) | set(rel)):
        if key not in rel:
            problems.append(f"ci.yml runs `{key}` with required {debug[key]}, release-profile.yml has no such step")
        elif key not in debug:
            problems.append(f"release-profile.yml runs `{key} --release` with no ci.yml counterpart")
        elif debug[key] != rel[key]:
            problems.append(f"`{key}`: ci.yml requires {debug[key]}, release-profile.yml requires {rel[key]}")
        else:
            compared += 1
    if compared == 0 and not problems:
        problems.append("no run_tests step pair compared (the check would be vacuous)")
    return problems, compared


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    a = ap.parse_args(argv)
    try:
        problems, compared = check(a.root)
    except ValueError as e:
        print(f"release-lists: {e}", file=sys.stderr)
        return 2
    for p in problems:
        print(f"release-lists: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"release-lists: {compared} step pair(s), the release legs require the same binaries")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
