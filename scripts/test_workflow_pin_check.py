#!/usr/bin/env python3
"""Tests for scripts/workflow_pin_check.py.

The first case runs the check against this repository. The other cases feed a
workflow with exactly one thing broken and assert that it is reported.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workflow_pin_check as wpc  # noqa: E402

SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
GOOD = f"""name: CI
on: push
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@{SHA} # v5.1.0
      - name: Lint
        uses: reviewdog/action-actionlint@{SHA} # v1.80.0
      - uses: ./.github/actions/local
  lint:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      # the reporter posts a check run
      checks: write
    steps:
      - uses: actions/checkout@{SHA} # v5.1.0
"""


class WorkflowPinCheckTest(unittest.TestCase):
    def test_this_repository_passes(self):
        self.assertEqual(wpc.main(["x"]), 0)

    def test_a_good_workflow_passes(self):
        self.assertEqual(wpc.check_text("ci.yml", GOOD), ([], 3))

    def test_a_tag_reference_fails(self):
        errors, _ = wpc.check_text("ci.yml", GOOD.replace(f"actions/checkout@{SHA} # v5.1.0", "actions/checkout@v5", 1))
        self.assertEqual(errors, ["ci.yml:9: `actions/checkout@v5` is not pinned to a full 40-hex commit SHA"])

    def test_a_short_sha_fails(self):
        errors, _ = wpc.check_text("ci.yml", GOOD.replace(f"@{SHA} # v1.80.0", f"@{SHA[:12]} # v1.80.0"))
        self.assertEqual(len(errors), 1)
        self.assertIn("not pinned", errors[0])

    def test_an_uppercase_or_branch_ref_fails(self):
        for ref in (SHA.upper(), "stable", "main"):
            errors, _ = wpc.check_text("ci.yml", GOOD.replace(f"@{SHA} # v1.80.0", f"@{ref} # v1.80.0"))
            self.assertEqual(len(errors), 1, ref)

    def test_a_pin_without_a_comment_fails(self):
        errors, _ = wpc.check_text("ci.yml", GOOD.replace(f"@{SHA} # v1.80.0", f"@{SHA}"))
        self.assertEqual(errors, [f"ci.yml:11: `reviewdog/action-actionlint@{SHA}` has no `# <tag>` comment"])
        errors, _ = wpc.check_text("ci.yml", GOOD.replace(f"@{SHA} # v1.80.0", f"@{SHA} #"))
        self.assertEqual(len(errors), 1)

    def test_missing_top_level_permissions_fails(self):
        errors, _ = wpc.check_text("ci.yml", GOOD.replace("permissions:\n  contents: read\njobs:", "jobs:"))
        self.assertEqual(errors, ["ci.yml: no top-level `permissions:`"])

    def test_a_write_grant_without_a_reason_fails(self):
        errors, _ = wpc.check_text("ci.yml", GOOD.replace("      # the reporter posts a check run\n", ""))
        self.assertEqual(len(errors), 1)
        self.assertIn("checks: write", errors[0])

    def test_no_workflow_or_no_uses_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wpc.check_dir(d)[0], [f"{d}: no workflow file was compared"])
            with open(os.path.join(d, "a.yml"), "w", encoding="utf-8") as f:
                f.write("on: push\npermissions:\n  contents: read\njobs: {}\n")
            self.assertEqual(wpc.check_dir(d)[0], [f"{d}: no `uses:` was compared"])


if __name__ == "__main__":
    unittest.main()
