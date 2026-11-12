#!/usr/bin/env python3
"""Every tracked Cargo.lock resolves each alice-* crate to the same source.

The fuzz targets and the tools are their own workspaces with their own Cargo.lock. A change that
moves an alice-* dependency in the root lock and not in the others builds the root and leaves the
other workspace on the old commit: on 2026-11-08 the root lock took alice-zip with a new constant,
fuzz/Cargo.lock kept the commit before it, and every fuzz build failed only in CI.

For each alice-* package this reads the `source` of every tracked Cargo.lock that has it and fails
when two of them differ (the lock and the source of each are named). A crate built from this
repository itself has no source and is not compared. Fails when no tracked Cargo.lock is read. A
repository whose locks share no alice-* package (the other workspaces use path dependencies or no
alice-* crate) passes and says so.

A lock that has to keep another source of one crate on purpose (a reader of an old format kept to
reproduce data written by an old release, for example) is listed in `scripts/lock_agreement.toml`:

    [[exempt]]
    lock = "compat/v0_10_reader/Cargo.lock"
    crate = "alice-det-math"
    reason = "why this lock needs the other source"
    test = "path/of/the/test/that/needs/it.rs"

That pair is left out of the comparison. An entry fails when its lock is not tracked, the crate is
not in it, its test file is not in the repository, its reason is empty, or the lock would agree
without it (an exemption nothing needs).

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


def exemptions(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    """(entries, problems) of scripts/lock_agreement.toml (absent: none)"""
    path = root / "scripts" / "lock_agreement.toml"
    if not path.exists():
        return [], []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries, problems = [], []
    for i, e in enumerate(data.get("exempt", [])):
        missing = [k for k in ("lock", "crate", "reason", "test") if not str(e.get(k, "")).strip()]
        if missing:
            problems.append(f"scripts/lock_agreement.toml exempt #{i + 1}: missing {', '.join(missing)}")
            continue
        if not (root / e["test"]).is_file():
            problems.append(f"scripts/lock_agreement.toml exempt #{i + 1}: test {e['test']} is not a file here")
        entries.append({k: str(e[k]) for k in ("lock", "crate", "reason", "test")})
    return entries, problems


def check(root: Path) -> tuple[list[str], int, int]:
    """(problems, locks read, crates compared)"""
    locks = tracked_locks(root)
    entries, problems = exemptions(root)
    seen: dict[str, dict[str, str]] = {}
    for lock in locks:
        for name, source in alice_sources(lock).items():
            seen.setdefault(name, {})[str(lock.relative_to(root))] = source
    exempt = {(e["lock"], e["crate"]) for e in entries}
    for e in entries:
        by_lock = seen.get(e["crate"], {})
        if e["lock"] not in by_lock:
            problems.append(f"exempt {e['crate']} in {e['lock']}: no such crate in a tracked lock of that path")
            continue
        others = {s for lk, s in by_lock.items() if (lk, e["crate"]) not in exempt}
        if others and {by_lock[e["lock"]]} == others:
            problems.append(f"exempt {e['crate']} in {e['lock']}: the lock agrees without it (remove the entry)")
    compared = 0
    for name, by_lock in sorted(seen.items()):
        kept = {lk: s for lk, s in by_lock.items() if (lk, name) not in exempt}
        if len(kept) < 2:
            continue
        compared += 1
        if len(set(kept.values())) > 1:
            listed = "; ".join(f"{lock}: {source}" for lock, source in sorted(kept.items()))
            problems.append(f"{name} resolves differently: {listed}")
    return problems, len(locks), compared


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    problems, locks, compared = check(root)
    if locks == 0:
        print("lock agreement: no tracked Cargo.lock read", file=sys.stderr)
        return 1
    for p in problems:
        print(f"lock agreement: {p}", file=sys.stderr)
    if problems:
        return 1
    if compared == 0:
        print(f"lock agreement: {locks} locks read, no alice-* crate in two of them (nothing to compare)")
        return 0
    print(f"lock agreement: {locks} locks, {compared} alice-* crates compared, all agree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
