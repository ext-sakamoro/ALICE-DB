#!/usr/bin/env python3
"""A consumer that takes alice-db by git gets a checkout with no sibling repositories next to it, so the
dependencies it uses must resolve from the manifests alone.

- No manifest names a sibling by path (`path = "../…"`), except `alice-analytics` (the `analytics` feature):
  it stays a path dependency until the shared sibling check accepts a repository with no sibling to pin, and
  a consumer that enables `analytics` still needs ../ALICE-Analytics (README, Building).
- A consumer crate next to a copy of the tracked tree placed alone resolves alice-db with the default features
  and with `crypto`.

Usage: python3 scripts/test_git_consumer.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIBLING_PATH = re.compile(r'''^\s*([A-Za-z0-9_-]+)\s*=\s*\{[^}]*\bpath\s*=\s*["']\.\./''')
# the one sibling path dependency left (see the module doc)
ALLOWED = {"alice-analytics"}


def tracked(pattern: str | None = None) -> list[str]:
    args = ["git", "-C", str(ROOT), "ls-files"] + ([pattern] if pattern else [])
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.splitlines()


class GitConsumer(unittest.TestCase):
    def test_no_manifest_names_a_sibling_by_path_but_the_analytics_bridge(self):
        manifests = tracked("*Cargo.toml")
        self.assertTrue(manifests, "no Cargo.toml was read (compared nothing)")
        found, allowed = [], []
        for m in manifests:
            for n, line in enumerate((ROOT / m).read_text(encoding="utf-8").splitlines(), 1):
                hit = SIBLING_PATH.search(line.split("#", 1)[0])
                if hit:
                    (allowed if hit.group(1) in ALLOWED else found).append(f"{m}:{n}: {line.strip()}")
        self.assertEqual(found, [], "sibling path dependencies a git consumer cannot resolve")
        self.assertEqual(len(allowed), 1, allowed)

    def resolve(self, features: list[str]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            alone = Path(d) / "only" / "ALICE-DB"
            for f in tracked():
                src = ROOT / f
                if src.is_file():
                    dst = alone / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            consumer = Path(d) / "consumer"
            (consumer / "src").mkdir(parents=True)
            feats = ", ".join(f'"{f}"' for f in features)
            (consumer / "Cargo.toml").write_text(
                '[package]\nname = "consumer"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\n'
                f'alice-db = {{ path = "../only/ALICE-DB", features = [{feats}] }}\n', encoding="utf-8")
            (consumer / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
            return subprocess.run(["cargo", "metadata", "--format-version", "1"], cwd=consumer, capture_output=True,
                                  text=True, timeout=600)

    def test_a_consumer_with_the_default_features_resolves_alone(self):
        p = self.resolve([])
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])

    def test_a_consumer_with_crypto_resolves_alone(self):
        p = self.resolve(["crypto"])
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
