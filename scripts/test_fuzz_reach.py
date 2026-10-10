"""Teeth for scripts/fuzz_reach.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuzz_reach as fr  # noqa: E402

LOG = "#1\tINITED cov: 10\n#2000\tDONE   cov: 482 ft: 914 corp: 185/49Kb\nDone 2000 runs in 61 second(s)\n"
FLOOR = "# target min_cov\nfuzz_law_record 300\nfuzz_query_parse 150\n"


class Reach(unittest.TestCase):
    def test_a_real_run_above_its_floor_passes(self):
        self.assertEqual(fr.check("fuzz_law_record", LOG, FLOOR), [])

    def test_a_scaffold_below_its_floor_fails(self):
        scaffold = "#9\tDONE   cov: 12 ft: 12\nDone 34000000 runs in 61 second(s)\n"
        self.assertTrue(any("floor" in p for p in fr.check("fuzz_law_record", scaffold, FLOOR)))

    def test_zero_runs_fails(self):
        log = "#0\tDONE   cov: 400\nDone 0 runs in 0 second(s)\n"
        self.assertTrue(any("0 inputs" in p for p in fr.check("fuzz_law_record", log, FLOOR)))

    def test_a_target_without_a_floor_fails(self):
        self.assertTrue(any("no coverage floor" in p for p in fr.check("fuzz_new", LOG, FLOOR)))

    def test_an_empty_floor_file_fails(self):
        self.assertTrue(any("compared nothing" in p for p in fr.check("fuzz_law_record", LOG, "# only a comment\n")))

    def test_a_log_without_the_final_line_fails(self):
        self.assertTrue(fr.check("fuzz_law_record", "crashed before DONE\n", FLOOR))


if __name__ == "__main__":
    unittest.main()
