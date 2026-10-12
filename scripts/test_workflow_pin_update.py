#!/usr/bin/env python3
"""Tests for scripts/workflow_pin_update.py with a fake GitHub API."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workflow_pin_update as wpu  # noqa: E402

OLD, NEW, BR_OLD, BR_NEW = "a" * 40, "b" * 40, "c" * 40, "d" * 40


def api(path):
    answers = {
        "repos/actions/checkout/commits/v5": {"sha": NEW},
        "repos/actions/checkout/tags?per_page=100": [
            {"name": "v5", "commit": {"sha": NEW}},
            {"name": "v5.1.0", "commit": {"sha": OLD}},
            {"name": "v5.2.0", "commit": {"sha": NEW}},
            {"name": "v5.10.0", "commit": {"sha": NEW}},
            # another major on the same commit must not be taken
            {"name": "v6.0.0", "commit": {"sha": NEW}},
        ],
        "repos/dtolnay/rust-toolchain/commits/stable": {"sha": BR_NEW},
        "repos/old/action/commits/v1": {"sha": NEW},
        "repos/old/action/tags?per_page=100": [{"name": "v1", "commit": {"sha": NEW}}],
    }
    return answers[path]


class UpdateTest(unittest.TestCase):
    def test_a_release_tag_follows_its_major(self):
        text = f"      - uses: actions/checkout@{OLD} # v5.1.0\n"
        new, problems = wpu.update_text(text, api)
        self.assertEqual((new, problems), (f"      - uses: actions/checkout@{NEW} # v5.10.0\n", []))

    def test_a_branch_follows_its_head_and_keeps_the_comment(self):
        text = f"  - uses: dtolnay/rust-toolchain@{BR_OLD} # stable (branch, default toolchain stable)"
        new, _ = wpu.update_text(text, api)
        self.assertEqual(new, f"  - uses: dtolnay/rust-toolchain@{BR_NEW} # stable (branch, default toolchain stable)")

    def test_an_unnamed_major_head_is_reported_and_left(self):
        text = f"  - uses: old/action@{OLD} # v1.2.3"
        new, problems = wpu.update_text(text, api)
        self.assertEqual(new, text)
        self.assertEqual(len(problems), 1)

    def test_other_lines_are_untouched(self):
        text = "name: CI\n  - uses: ./local\n  - run: echo uses: x@y\n"
        self.assertEqual(wpu.update_text(text, api), (text, []))

    def test_a_ref_that_is_neither_tag_nor_branch_is_reported(self):
        text = f"  - uses: actions/checkout@{OLD} # main"
        new, problems = wpu.update_text(text, api)
        self.assertEqual(new, text)
        self.assertEqual(len(problems), 1)


if __name__ == "__main__":
    unittest.main()
