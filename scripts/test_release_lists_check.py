#!/usr/bin/env python3
"""tests of scripts/release_lists_check.py (python3 scripts/test_release_lists_check.py)"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_lists_check as chk  # noqa: E402

CI = """jobs:
  test:
    steps:
      - name: Test (default)
        run: scripts/run_tests.sh a,b=2 -- cargo test
      - name: Test (no default)
        run: scripts/run_tests.sh a -- cargo test --no-default-features
      - name: float environment
        run: |
          scripts/run_tests.sh f=2 -- cargo test --test f
"""
REL = """jobs:
  release:
    steps:
      - name: Test (release, default)
        run: scripts/run_tests.sh a,b=2 -- cargo test --release
      - name: Test (release, no default)
        run: scripts/run_tests.sh a -- cargo test --release --no-default-features
"""


class Check(unittest.TestCase):
    def run_on(self, ci: str, rel: str) -> tuple[list[str], int]:
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / ".github" / "workflows"
            w.mkdir(parents=True)
            (w / "ci.yml").write_text(ci, encoding="utf-8")
            (w / "release-profile.yml").write_text(rel, encoding="utf-8")
            return chk.check(Path(d))

    def test_equal_lists_pass(self):
        problems, compared = self.run_on(CI, REL)
        self.assertEqual(problems, [])
        self.assertEqual(compared, 2)

    def test_a_binary_only_in_ci_fails(self):
        problems, _ = self.run_on(CI.replace("a,b=2 -- cargo test\n", "a,b=2,c -- cargo test\n"), REL)
        self.assertEqual(len(problems), 1)
        self.assertIn("ci.yml requires a,b=2,c", problems[0])

    def test_a_count_differing_fails(self):
        problems, _ = self.run_on(CI, REL.replace("a,b=2 -- cargo test --release", "a,b=1 -- cargo test --release"))
        self.assertEqual(len(problems), 1)

    def test_a_ci_step_without_release_partner_fails(self):
        problems, _ = self.run_on(CI, REL.replace("      - name: Test (release, no default)\n        run: scripts/run_tests.sh a -- cargo test --release --no-default-features\n", ""))
        self.assertEqual(len(problems), 1)
        self.assertIn("has no such step", problems[0])

    def test_a_release_step_without_ci_partner_fails(self):
        extra = "      - name: x\n        run: scripts/run_tests.sh z -- cargo test --release --features q\n"
        problems, _ = self.run_on(CI, REL + extra)
        self.assertEqual(len(problems), 1)
        self.assertIn("no ci.yml counterpart", problems[0])

    def test_nothing_compared_fails(self):
        problems, compared = self.run_on("jobs: {}\n", "jobs: {}\n")
        self.assertEqual(compared, 0)
        self.assertEqual(len(problems), 1)
        self.assertIn("vacuous", problems[0])

    def test_this_repository_passes(self):
        problems, compared = chk.check(Path(__file__).resolve().parents[1])
        self.assertEqual(problems, [])
        self.assertGreaterEqual(compared, 4)


if __name__ == "__main__":
    unittest.main()
