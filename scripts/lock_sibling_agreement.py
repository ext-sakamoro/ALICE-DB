#!/usr/bin/env python3
"""Every tracked Cargo.lock resolves each alice-* crate to the same source.

The fuzz targets and the tools are their own workspaces with their own Cargo.lock. A change that
moves an alice-* dependency in the root lock and not in the others builds the root and leaves the
other workspace on the old commit: on 2026-11-08 the root lock took alice-zip with a new constant,
fuzz/Cargo.lock kept the commit before it, and every fuzz build failed only in CI.

For each alice-* package this reads the `source` of every tracked Cargo.lock that has it and fails
when two of them differ (the lock and the source of each are named). A crate built from this
repository itself has no source and is not compared. Fails when no Cargo.lock is read, or when no
alice-* package appears in two locks (nothing compared).

    python3 scripts/lock_sibling_agreement.py [repo root]
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def tracked_locks(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "Cargo.lock", "*/Cargo.lock"],
        check=True,
        capture_output=True,
    ).stdout
    return [root / p for p in out.decode().split("\0") if p]


def alice_sources(lock: Path) -> dict[str, str]:
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    sources: dict[str, set[str]] = {}
    for package in data.get("package", []):
        name = package.get("name", "")
        source = package.get("source")
        if name.startswith("alice-") and source:
            sources.setdefault(name, set()).add(source)
    return {name: " | ".join(sorted(s)) for name, s in sources.items()}


def check(root: Path) -> tuple[list[str], int, int]:
    """(problems, locks read, crates compared)"""
    locks = tracked_locks(root)
    seen: dict[str, dict[str, str]] = {}
    for lock in locks:
        for name, source in alice_sources(lock).items():
            seen.setdefault(name, {})[str(lock.relative_to(root))] = source
    problems = []
    compared = 0
    for name, by_lock in sorted(seen.items()):
        if len(by_lock) < 2:
            continue
        compared += 1
        if len(set(by_lock.values())) > 1:
            listed = "; ".join(f"{lock}: {source}" for lock, source in sorted(by_lock.items()))
            problems.append(f"{name} resolves differently: {listed}")
    return problems, len(locks), compared


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    problems, locks, compared = check(root)
    if locks == 0:
        print("lock agreement: no tracked Cargo.lock read", file=sys.stderr)
        return 1
    if compared == 0:
        print(f"lock agreement: {locks} locks read, no alice-* crate in two of them (compared nothing)", file=sys.stderr)
        return 1
    for p in problems:
        print(f"lock agreement: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"lock agreement: {locks} locks, {compared} alice-* crates compared, all agree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
