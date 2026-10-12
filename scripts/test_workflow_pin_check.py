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
        self.assertEqual(wpc.check_text("ci.yml", GOOD), ([], 4))

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

    def test_a_uses_value_with_an_expression_or_spaces_is_not_skipped(self):
        line = f"        uses: reviewdog/action-actionlint@{SHA} # v1.80.0"
        for evasion in ("${{ env.ACT }}@" + SHA + " # v1", "${{ matrix.action }}", "actions/checkout @v5"):
            errors, compared = wpc.check_text("ci.yml", GOOD.replace(line, f"        uses: {evasion}"))
            self.assertEqual(compared, 4, evasion)
            self.assertTrue(any("not pinned" in e for e in errors), (evasion, errors))

    def test_other_spellings_of_uses_are_compared(self):
        line = f"        uses: reviewdog/action-actionlint@{SHA} # v1.80.0"
        for spelling in (f'        uses: "reviewdog/action-actionlint@{SHA}" # v1.80.0',
                         f"        'uses': reviewdog/action-actionlint@{SHA} # v1.80.0",
                         f"        uses:   reviewdog/action-actionlint@{SHA}   # v1.80.0"):
            self.assertEqual(wpc.check_text("ci.yml", GOOD.replace(line, spelling)), ([], 4), spelling)
        for bad in ("        uses:reviewdog/action-actionlint@v1", '        "uses": reviewdog/action-actionlint@v1'):
            errors, _ = wpc.check_text("ci.yml", GOOD.replace(line, bad))
            self.assertTrue(any("not pinned" in e for e in errors), (bad, errors))

    def test_an_unrecognised_uses_line_fails(self):
        line = f"        uses: reviewdog/action-actionlint@{SHA} # v1.80.0"
        errors, _ = wpc.check_text("ci.yml", GOOD.replace(line, "        run: echo; { uses: x/y@v1 }"))
        self.assertTrue(any("does not recognise" in e for e in errors), errors)
        self.assertTrue(any("5 `uses:` lines, 4 compared" in e or "lines," in e for e in errors), errors)

    def test_top_level_write_fails(self):
        errors, _ = wpc.check_text("ci.yml", GOOD.replace("permissions:\n  contents: read\njobs:", "permissions: write-all\njobs:"))
        self.assertEqual(len(errors), 1)
        self.assertIn("write-all", errors[0])
        errors, _ = wpc.check_text("ci.yml", GOOD.replace("permissions:\n  contents: read\njobs:", "permissions:\n  contents: write\njobs:"))
        self.assertEqual(len(errors), 1)
        self.assertIn("contents: write", errors[0])
        reasoned = "permissions:\n  # releases upload assets\n  contents: write\njobs:"
        self.assertEqual(wpc.check_text("ci.yml", GOOD.replace("permissions:\n  contents: read\njobs:", reasoned))[0], [])

    def test_job_level_inline_write_fails(self):
        block = "    permissions:\n      contents: read\n      # the reporter posts a check run\n      checks: write\n"
        for inline in ("    permissions: write-all\n", "    permissions: {contents: write}\n",
                       "    permissions: { checks: write }  # reason here does not count\n"):
            errors, _ = wpc.check_text("ci.yml", GOOD.replace(block, inline))
            self.assertEqual(len(errors), 1, inline)
            self.assertIn("grants write inline", errors[0])
        self.assertEqual(wpc.check_text("ci.yml", GOOD.replace(block, "    permissions: read-all\n"))[0], [])

    def test_local_actions_are_scanned(self):
        action = f"runs:\n  using: composite\n  steps:\n    - uses: actions/checkout@{SHA} # v5.1.0\n      with:\n        persist-credentials: false\n"
        with tempfile.TemporaryDirectory() as root:
            wf = os.path.join(root, "workflows")
            act = os.path.join(root, "actions", "setup")
            os.makedirs(wf)
            os.makedirs(act)
            with open(os.path.join(wf, "ci.yml"), "w", encoding="utf-8") as f:
                f.write(GOOD)
            with open(os.path.join(act, "action.yml"), "w", encoding="utf-8") as f:
                f.write(action)
            actions_dir = os.path.join(root, "actions")
            self.assertEqual(wpc.check_dir(wf, actions_dir), ([], 2, 5))
            with open(os.path.join(act, "action.yml"), "w", encoding="utf-8") as f:
                f.write(action.replace(f"@{SHA} # v5.1.0", "@v5"))
            errors, _, _ = wpc.check_dir(wf, actions_dir)
            self.assertEqual(errors, ["actions/setup/action.yml:4: `actions/checkout@v5` is not pinned to a full 40-hex commit SHA"])


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

    def test_with_a_base_only_new_or_changed_pins_are_asked(self):
        text = f"steps:\n  - uses: actions/checkout@{SHA} # v5.1.0\n  - uses: dtolnay/rust-toolchain@{SHA} # stable (branch)\n"
        known = {("actions/checkout", SHA, "v5.1.0", False)}
        answers = {f"repos/dtolnay/rust-toolchain/compare/{SHA}...stable": {"status": "diverged"}}
        # the unchanged checkout pin is not asked (its upstream may have moved);
        # the new branch pin is, and fails
        (errors, compared), calls = self.run_online_known(text, answers, known)
        self.assertEqual((compared, calls), (1, [f"repos/dtolnay/rust-toolchain/compare/{SHA}...stable"]))
        self.assertEqual(len(errors), 1)

    def test_with_a_base_and_no_changed_pin_nothing_is_asked_and_it_passes(self):
        text = f"steps:\n  - uses: actions/checkout@{SHA} # v5.1.0\n"
        (errors, compared), calls = self.run_online_known(text, {}, {("actions/checkout", SHA, "v5.1.0", False)})
        self.assertEqual((errors, compared, calls), ([], 0, []))

    def test_a_changed_comment_is_a_changed_pin(self):
        text = f"steps:\n  - uses: actions/checkout@{SHA} # v5.2.0\n"
        (errors, compared), _ = self.run_online_known(
            text, {"repos/actions/checkout/commits/v5.2.0": {"sha": self.OTHER}},
            {("actions/checkout", SHA, "v5.1.0", False)})
        self.assertEqual((compared, len(errors)), (1, 1))

    def run_online_known(self, text, answers, known):
        calls = []

        def api(path):
            calls.append(path)
            return answers.get(path)

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "ci.yml"), "w", encoding="utf-8") as f:
                f.write(text)
            return wpc.check_online(d, api, known=known), calls


class BaseTest(unittest.TestCase):
    """`--base` reads the pins of a commit from git (a throwaway repository)"""

    def setUp(self):
        import subprocess
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self.tmp.name
        self.old_root = wpc.ROOT
        wpc.ROOT = self.root
        self.wf = os.path.join(self.root, ".github", "workflows")
        os.makedirs(self.wf)

        def run(*a):
            subprocess.run(["git", *a], cwd=self.root, check=True, capture_output=True,
                           env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
        self.run_git = run
        run("init", "-q")
        self.write(f"steps:\n  - uses: actions/checkout@{SHA} # v5.1.0\n")
        run("add", ".")
        run("commit", "-q", "-m", "a")

    def tearDown(self):
        wpc.ROOT = self.old_root
        self.tmp.cleanup()

    def write(self, text):
        with open(os.path.join(self.wf, "ci.yml"), "w", encoding="utf-8") as f:
            f.write(text)

    def test_pins_at_a_commit_are_read_from_git(self):
        self.write(f"steps:\n  - uses: actions/checkout@{SHA} # v5.1.0\n  - uses: x/y@{SHA} # v1.0.0\n")
        base = wpc.resolve_base("HEAD")
        self.assertEqual(wpc.pins_at(base, self.wf, None), {("actions/checkout", SHA, "v5.1.0", False)})

    def test_an_unknown_base_fails_instead_of_checking_everything(self):
        with self.assertRaises(SystemExit):
            wpc.resolve_base("0" * 40)
        event = os.path.join(self.root, "event.json")
        with open(event, "w", encoding="utf-8") as f:
            f.write('{"before": "' + "0" * 40 + '"}')
        old = os.environ.get("GITHUB_EVENT_PATH")
        os.environ["GITHUB_EVENT_PATH"] = event
        try:
            # no origin/main in this repository and `before` is the zero SHA
            with self.assertRaises(SystemExit):
                wpc.resolve_base("auto")
        finally:
            if old is None:
                del os.environ["GITHUB_EVENT_PATH"]
            else:
                os.environ["GITHUB_EVENT_PATH"] = old

    def test_auto_takes_the_commit_before_the_push(self):
        before = wpc.resolve_base("HEAD")
        self.write("steps: []\n")
        self.run_git("commit", "-qam", "b")
        event = os.path.join(self.root, "event.json")
        with open(event, "w", encoding="utf-8") as f:
            f.write('{"before": "' + before + '"}')
        old = os.environ.get("GITHUB_EVENT_PATH")
        os.environ["GITHUB_EVENT_PATH"] = event
        try:
            self.assertEqual(wpc.resolve_base("auto"), before)
        finally:
            if old is None:
                del os.environ["GITHUB_EVENT_PATH"]
            else:
                os.environ["GITHUB_EVENT_PATH"] = old

    def test_auto_never_compares_the_commit_with_itself(self):
        first = wpc.resolve_base("HEAD")
        self.write("steps: []\n")
        self.run_git("commit", "-qam", "b")
        head = wpc.resolve_base("HEAD")
        event = os.path.join(self.root, "event.json")
        old = os.environ.get("GITHUB_EVENT_PATH")
        os.environ["GITHUB_EVENT_PATH"] = event
        try:
            for payload in ('{"before": "' + head + '"}', '{"pull_request": {"base": {"sha": "' + head + '"}}}'):
                with open(event, "w", encoding="utf-8") as f:
                    f.write(payload)
                self.assertEqual(wpc.resolve_base("auto"), first, payload)
        finally:
            if old is None:
                del os.environ["GITHUB_EVENT_PATH"]
            else:
                os.environ["GITHUB_EVENT_PATH"] = old


if __name__ == "__main__":
    unittest.main()
