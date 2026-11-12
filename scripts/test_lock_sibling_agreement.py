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
DM_GIT = "git+https://example.invalid/ALICE-DetMath?branch=main#cccc"
REG = "registry+https://github.com/rust-lang/crates.io-index"
EXEMPT = """[[exempt]]
lock = "compat/Cargo.lock"
crate = "alice-det-math"
reason = "old reader"
test = "compat/tests/old_reader.rs"
"""


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

    def test_locks_that_share_no_alice_crate_pass_and_compare_nothing(self) -> None:
        d = self.repo({"Cargo.lock": lock(("alice-zip", ZIP_A)), "fuzz/Cargo.lock": lock(("serde", "registry+x"))})
        self.assertEqual(m.check(d)[1:], (2, 0))
        self.assertEqual(m.main(["x", str(d)]), 0)

    def test_registry_and_git_sources_disagree(self) -> None:
        d = self.repo({
            "Cargo.lock": lock(("alice-det-math", REG)),
            "fuzz/Cargo.lock": lock(("alice-det-math", DM_GIT)),
        })
        self.assertEqual(m.main(["x", str(d)]), 1)

    def exempt_files(self, entry: str, test_file: bool = True) -> dict[str, str]:
        files = {
            "Cargo.lock": lock(("alice-det-math", DM_GIT)),
            "fuzz/Cargo.lock": lock(("alice-det-math", DM_GIT)),
            "compat/Cargo.lock": lock(("alice-det-math", REG)),
            "scripts/lock_agreement.toml": entry,
        }
        if test_file:
            files["compat/tests/old_reader.rs"] = "// reads data written by the old release\n"
        return files

    def test_an_exempt_lock_is_left_out_of_the_comparison(self) -> None:
        d = self.repo(self.exempt_files(EXEMPT))
        problems, locks, compared = m.check(d)
        self.assertEqual((problems, locks, compared), ([], 3, 1))
        self.assertEqual(m.main(["x", str(d)]), 0)

    def test_without_the_exemption_the_old_source_fails(self) -> None:
        d = self.repo(self.exempt_files(""))
        self.assertEqual(m.main(["x", str(d)]), 1)

    def test_an_exemption_the_lock_does_not_need_fails(self) -> None:
        files = self.exempt_files(EXEMPT)
        files["compat/Cargo.lock"] = lock(("alice-det-math", DM_GIT))
        d = self.repo(files)
        problems = m.check(d)[0]
        self.assertEqual(len(problems), 1)
        self.assertIn("agrees without it", problems[0])

    def test_an_exemption_for_a_crate_not_in_the_lock_fails(self) -> None:
        d = self.repo(self.exempt_files(EXEMPT.replace("alice-det-math", "alice-zip")))
        self.assertTrue(any("no such crate" in p for p in m.check(d)[0]))

    def test_an_exemption_without_its_test_file_fails(self) -> None:
        d = self.repo(self.exempt_files(EXEMPT, test_file=False))
        self.assertTrue(any("is not a file here" in p for p in m.check(d)[0]))

    def test_an_exemption_without_a_reason_fails(self) -> None:
        d = self.repo(self.exempt_files(EXEMPT.replace('reason = "old reader"', 'reason = ""')))
        self.assertTrue(any("missing reason" in p for p in m.check(d)[0]))

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
