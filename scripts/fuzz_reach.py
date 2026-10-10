#!/usr/bin/env python3
"""A fuzz run must execute inputs and reach code: runs > 0 and libFuzzer's
final coverage at least the floor recorded for the target.

usage: scripts/fuzz_reach.py <target> <fuzz-run.log> [fuzz/coverage-floor.txt]

The floor file has one `target min_cov` per line (`#` comments). A target
without a floor fails, and so does a floor file with no entries: a target that
does not call the crate (a scaffold) or stops at a checksum covers a handful of
edges and cannot meet a floor taken from a real run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = re.compile(r"Done (\d+) runs")
COV = re.compile(r"\bDONE\s+cov: (\d+)")


def floors(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, value = line.split()
        out[name] = int(value)
    return out


def check(target: str, log: str, floor_text: str) -> list[str]:
    table = floors(floor_text)
    if not table:
        return ["the coverage floor file has no entries (compared nothing)"]
    if target not in table:
        return [f"{target} has no coverage floor (add `{target} <min_cov>`)"]
    runs = [int(m) for m in RUNS.findall(log)]
    covs = [int(m) for m in COV.findall(log)]
    problems = []
    if not runs or runs[-1] <= 0:
        problems.append(f"{target} executed 0 inputs")
    if not covs:
        problems.append(f"{target}: no final coverage line in the log")
    elif covs[-1] < table[target]:
        problems.append(
            f"{target} reached cov {covs[-1]} < floor {table[target]} "
            "(the target stops early: a scaffold, a checksum wall, or a regression)"
        )
    return problems


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    target, log_path = sys.argv[1], sys.argv[2]
    floor_path = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "fuzz" / "coverage-floor.txt"
    log = Path(log_path).read_text(encoding="utf-8", errors="replace")
    problems = check(target, log, floor_path.read_text(encoding="utf-8"))
    for p in problems:
        print(f"::error::{p}")
    if not problems:
        runs = RUNS.findall(log)[-1]
        cov = COV.findall(log)[-1]
        print(f"fuzz_reach: {target} runs {runs}, cov {cov} (floor {floors(floor_path.read_text(encoding='utf-8'))[target]})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
