#!/usr/bin/env python3
"""Tests of scripts/lock_sibling_agreement.py on throwaway git repositories."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock_sibling_agreement as m  # noqa: E402

ZIP_A = "git+https://example.invalid/ALICE-Zip?branch=main#aaaa"
ZIP_B = "git+https://example.invalid/ALICE-Zip?branch=main#bbbb"


def lock(*packages: tuple[str, str | None]) -> str:
    out = "version = 4\n"
    for name, source in packages:
        out += f'\n[[package]]\nname = "{name}"\nversion = "0.1.0"\n'
        if source:
            out += f'source = "{source}"\n'
    return out


class LockAgreement(unittest.TestCase):
    def repo(self, files: dict[str, str]) -> Path:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(d)], check=False))
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        for rel, text in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(d), "add", "."], check=True)
        return d

    def test_the_same_source_in_every_lock_passes(self) -> None:
        d = self.repo({"Cargo.lock": lock(("alice-zip", ZIP_A)), "fuzz/Cargo.lock": lock(("alice-zip", ZIP_A))})
        self.assertEqual(m.main(["x", str(d)]), 0)

    def test_a_lock_left_behind_fails_and_names_it(self) -> None:
        d = self.repo({"Cargo.lock": lock(("alice-zip", ZIP_B)), "fuzz/Cargo.lock": lock(("alice-zip", ZIP_A))})
        problems, locks, compared = m.check(d)
        self.assertEqual((locks, compared), (2, 1))
        self.assertEqual(len(problems), 1)
        self.assertIn("fuzz/Cargo.lock", problems[0])
        self.assertEqual(m.main(["x", str(d)]), 1)

    def test_the_repository_own_crate_is_not_compared(self) -> None:
        d = self.repo({
            "Cargo.lock": lock(("alice-db", None), ("alice-zip", ZIP_A)),
            "fuzz/Cargo.lock": lock(("alice-db", None), ("alice-zip", ZIP_A)),
        })
        self.assertEqual(m.check(d)[2], 1)

    def test_nothing_compared_fails(self) -> None:
        d = self.repo({"Cargo.lock": lock(("alice-zip", ZIP_A)), "fuzz/Cargo.lock": lock(("serde", "registry+x"))})
        self.assertEqual(m.main(["x", str(d)]), 1)

    def test_no_lock_fails(self) -> None:
        d = self.repo({"README.md": "x\n"})
        self.assertEqual(m.main(["x", str(d)]), 1)

    def test_an_untracked_lock_is_not_read(self) -> None:
        d = self.repo({"Cargo.lock": lock(("alice-zip", ZIP_A)), "fuzz/Cargo.lock": lock(("alice-zip", ZIP_A))})
        (d / "tools").mkdir()
        (d / "tools/Cargo.lock").write_text(lock(("alice-zip", ZIP_B)), encoding="utf-8")
        self.assertEqual(m.main(["x", str(d)]), 0)


if __name__ == "__main__":
    unittest.main()
