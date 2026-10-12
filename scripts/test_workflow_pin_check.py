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
        with:
          persist-credentials: false
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
      - name: Checkout
        uses: actions/checkout@{SHA} # v5.1.0
        with:
          fetch-depth: 0
          persist-credentials: false
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
        self.assertEqual(errors, [f"ci.yml:13: `reviewdog/action-actionlint@{SHA}` has no `# <tag>` comment"])
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

    def test_a_checkout_that_keeps_credentials_fails(self):
        errors, _ = wpc.check_text("ci.yml", GOOD.replace("          persist-credentials: false\n", "", 1))
        self.assertEqual(errors, ["ci.yml:9: actions/checkout without `persist-credentials: false`"])
        errors, _ = wpc.check_text("ci.yml", GOOD.replace("persist-credentials: false", "persist-credentials: true"))
        self.assertEqual(len(errors), 2)

    def test_privileged_triggers_fail(self):
        for trigger in ("pull_request_target", "workflow_run"):
            errors, _ = wpc.check_text("ci.yml", GOOD.replace("on: push", f"on:\n  {trigger}:\n    types: [opened]"))
            self.assertEqual(len(errors), 1, trigger)
            self.assertIn(trigger, errors[0])


class OnlineTest(unittest.TestCase):
    OTHER = "0" * 40

    def run_online(self, text, answers):
        calls = []

        def api(path):
            calls.append(path)
            return answers.get(path)

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "ci.yml"), "w", encoding="utf-8") as f:
                f.write(text)
            return wpc.check_online(d, api), calls

    def test_matching_tag_and_branch_pass(self):
        text = f"steps:\n  - uses: actions/checkout@{SHA} # v5.1.0\n  - uses: dtolnay/rust-toolchain@{SHA} # stable (branch)\n"
        answers = {
            "repos/actions/checkout/commits/v5.1.0": {"sha": SHA},
            f"repos/dtolnay/rust-toolchain/compare/{SHA}...stable": {"status": "ahead"},
        }
        (errors, compared), _ = self.run_online(text, answers)
        self.assertEqual((errors, compared), ([], 2))

    def test_a_comment_naming_another_commit_fails(self):
        text = f"steps:\n  - uses: actions/checkout@{SHA} # v5.1.0\n"
        (errors, _), _ = self.run_online(text, {"repos/actions/checkout/commits/v5.1.0": {"sha": self.OTHER}})
        self.assertEqual(len(errors), 1)
        self.assertIn("is not what tag v5.1.0 resolves to", errors[0])
        (errors, _), _ = self.run_online(text, {})
        self.assertIn("was not found", errors[0])

    def test_a_commit_off_the_branch_fails(self):
        text = f"steps:\n  - uses: dtolnay/rust-toolchain@{SHA} # stable (branch)\n"
        for answer in ({"status": "diverged"}, {"status": "behind"}, None):
            answers = {f"repos/dtolnay/rust-toolchain/compare/{SHA}...stable": answer} if answer else {}
            (errors, _), _ = self.run_online(text, answers)
            self.assertEqual(len(errors), 1, answer)

    def test_identical_pins_are_asked_once_and_nothing_compared_fails(self):
        text = f"steps:\n  - uses: actions/checkout@{SHA} # v5.1.0\n  - uses: actions/checkout@{SHA} # v5.1.0\n"
        (errors, compared), calls = self.run_online(text, {"repos/actions/checkout/commits/v5.1.0": {"sha": SHA}})
        self.assertEqual((errors, compared, len(calls)), ([], 2, 1))
        (errors, compared), _ = self.run_online("steps: []\n", {})
        self.assertEqual(compared, 0)
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
