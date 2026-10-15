#!/usr/bin/env python3
"""Tests of scripts/semver_gate.py (python3 scripts/test_semver_gate.py)

No network: crates.io is replaced by a fixture. The recorded outputs are the lines
cargo-semver-checks 0.50 prints; SemverEndToEnd runs the real tool on two small crates
(it fails when cargo-semver-checks is not installed, so the fixtures cannot drift from
the tool unseen). The cases of class Semver are those of ALICE-Law
scripts/test_release_train.py (main eb0779d) for the same functions.

Shared file: holds no repository-specific name; keep every copy byte-identical.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("semver_gate", HERE / "semver_gate.py")
sg = importlib.util.module_from_spec(spec)
sys.modules["semver_gate"] = sg
spec.loader.exec_module(sg)


def semver_output(step: str, total: int, failed: int, lints=(), from_v="1.0.0", to_v="1.1.0",
                  assume=False) -> str:
    """what cargo semver-checks 0.50 prints for one run (the lines the gate reads); with no
    failure it prints no "fail" count at all"""
    counts = f"{total - failed} pass, {failed} fail, 0 warn, 3 skip" if failed else f"{total} pass, 3 skip"
    out = [f"     Building x v{to_v} (current)",
           f"    Checking x v{from_v} -> v{to_v} ({'assume ' if assume else ''}{step} change)",
           f"     Checked [   0.200s] {total} checks: {counts}"]
    out += [f"\n--- failure {lint}: some description ---\n" for lint in lints]
    out.append(f"     Summary {'semver requires new major version: 1 major and 0 minor checks failed' if failed else 'no semver update required'}")
    return "\n".join(out)


def latest_fixture(crate: str, version: str | None):
    def fetch(path: str) -> dict:
        if version is None:
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
        assert path == crate, path
        return {"crate": {"max_stable_version": version, "max_version": version}}
    return fetch


def problems(version: str, latest_v: str | None, run, equal_version: str = "refuse", manifest=None, entry=None):
    return sg.semver_problems(Path("."), "Cargo.toml", "x", version, manifest or {}, entry, latest_fixture("x", latest_v),
                              run=run, equal_version=equal_version)


class Semver(unittest.TestCase):
    def test_feature_specs(self):
        self.assertEqual([l for l, *_ in sg.feature_specs({}, None)], ["default features", "no default features"])
        self.assertEqual([l for l, *_ in sg.feature_specs({"features": {"default": [], "x": []}}, None)][-1],
                         "all features")
        specs = sg.feature_specs({"features": {"x": []}}, {"consumer-features": [["x"], []],
                                                           "consumer-check-features": [["py"]]})
        self.assertEqual([(fs, cmd) for _, _, fs, cmd in specs][2:], [(["x"], "build"), ([], "build"), (["py"], "check")])

    def test_the_step_follows_cargo_compatibility(self):
        for old, new, step in (("3.0.1", "3.1.0", "minor"), ("3.0.1", "3.0.2", "patch"), ("3.0.1", "4.0.0", "major"),
                               ("0.5.0", "0.5.1", "minor"), ("0.5.1", "0.6.0", "major"), ("0.0.1", "0.0.2", "major"),
                               ("1.9.9", "1.10.0", "minor")):
            self.assertEqual(sg.version_step(old, new), step, (old, new))

    def test_a_clean_minor_run_counts_and_a_breaking_one_is_refused(self):
        self.assertEqual(sg.semver_verdict(0, semver_output("minor", 196, 0), "minor", "l"), (None, 1))
        p, n = sg.semver_verdict(1, semver_output("minor", 196, 1, ["enum_discriminants_undefined_non_unit_variant"]),
                                 "minor", "alice-physics 3.1.0 (default features)")
        self.assertEqual(n, 0)
        self.assertIn("a minor version does not allow enum_discriminants_undefined_non_unit_variant", p)
        self.assertIn("semver requires new major version", p)

    def test_zero_checks_a_wrong_step_a_silent_failure_and_no_run_are_refused(self):
        self.assertIn("compared 0 checks", sg.semver_verdict(0, semver_output("minor", 0, 0), "minor", "l")[0])
        self.assertIn("compared a patch step, the versions make a minor step",
                      sg.semver_verdict(0, semver_output("patch", 10, 0), "minor", "l")[0])
        # a nonzero exit with no failed check counted is still refused
        self.assertIsNotNone(sg.semver_verdict(1, semver_output("minor", 10, 0), "minor", "l")[0])
        self.assertIn("did not run (exit 127)", sg.semver_verdict(127, "No such file: cargo", "minor", "l")[0])
        self.assertIn("printed no count", sg.semver_verdict(0, "    Checking x v1.0.0 -> v1.1.0 (minor change)", "minor", "l")[0])
        # colour codes do not hide the lines the gate reads
        coloured = semver_output("minor", 5, 0).replace("Checking", "\x1b[1m\x1b[32mChecking\x1b[0m")
        self.assertEqual(sg.semver_verdict(0, coloured, "minor", "l"), (None, 1))

    def test_the_real_all_pass_line_has_no_fail_field(self):
        # recorded from cargo-semver-checks 0.50.0 on a passing run
        out = ("    Checking alice-db v0.5.1 -> v0.5.1 (assume minor change)\n"
               "     Checked [   0.031s] 223 checks: 223 pass, 31 skip\n"
               "     Summary no semver update required\n")
        self.assertEqual(sg.semver_verdict(0, out, "minor", "l"), (None, 1))

    def test_a_major_step_is_not_compared(self):
        calls = []
        p, n, note = problems("2.0.0", "1.4.0", lambda *a: calls.append(a) or (0, ""))
        self.assertEqual((p, n, calls), ([], 0, []))
        self.assertIn("major step", note)

    def test_a_crate_with_no_release_has_nothing_to_compare(self):
        p, n, note = problems("0.1.0", None, lambda *a: self.fail("ran"))
        self.assertEqual((p, n), ([], 0))
        self.assertIn("no release", note)

    def test_a_version_not_newer_than_the_latest_is_refused(self):
        for latest_v, version in (("1.1.0", "1.1.0"), ("1.2.0", "1.1.0"), ("1.0.0", "1.0.0-rc.1"),
                                  ("1.0.0-rc.2", "1.0.0-rc.1"), ("1.0.0+build.2", "1.0.0+build.1")):
            p, _, _ = problems(version, latest_v, lambda *a: self.fail("ran"))
            self.assertEqual(len(p), 1, (latest_v, version))

    def test_a_pre_release_of_a_later_version_is_compared(self):
        p, n, note = problems("1.1.0-rc.1", "1.0.0", lambda *a: (0, semver_output("minor", 5, 0)))
        self.assertEqual((p, n), ([], 2))
        self.assertIn("(minor)", note)

    def test_precedence_follows_semver(self):
        order = ["0.9.9", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
                 "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.0.1", "1.10.0"]
        keys = [sg.version_precedence(v) for v in order]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(sg.version_precedence("1.0.0+a"), sg.version_precedence("1.0.0"))

    def test_every_feature_set_of_the_consumer_build_is_compared(self):
        seen = []

        def run(repo, manifest_path, baseline, features, target_dir):
            seen.append((tuple(baseline), None if features is None else tuple(features)))
            return 0, semver_output("minor", 50, 0)
        entry = {"consumer-features": [["a", "std"]], "consumer-check-features": [["py"]]}
        p, n, note = problems("1.1.0", "1.0.0", run, manifest={"features": {"a": [], "py": []}}, entry=entry)
        self.assertEqual((p, n), ([], 4))
        base = ("--baseline-version", "1.0.0")
        self.assertEqual(seen, [(base, None), (base, ()), (base, ("a", "std")), (base, ("py",))])
        self.assertEqual(sg.semver_feature_args(None), ["--default-features"])
        self.assertEqual(sg.semver_feature_args([]), ["--only-explicit-features"])
        self.assertEqual(sg.semver_feature_args(["a", "b"]), ["--only-explicit-features", "--features", "a,b"])

    def test_one_refused_feature_set_refuses_the_release(self):
        runs = iter([(0, semver_output("minor", 50, 0)),
                     (1, semver_output("minor", 50, 1, ["enum_variant_added"]))])
        p, n, _ = problems("1.1.0", "1.0.0", lambda *a: next(runs))
        self.assertEqual((len(p), n), (1, 1), p)
        self.assertIn("(no default features)", p[0])

    def test_a_body_without_a_version_is_a_registry_error(self):
        with self.assertRaises(sg.RegistryError):
            sg.semver_problems(Path("."), "Cargo.toml", "x", "1.0.0", {}, None, lambda path: {"crate": {}},
                               run=lambda *a: self.fail("ran"))

    def test_an_http_error_other_than_404_is_raised(self):
        def fetch(path):
            raise urllib.error.HTTPError(path, 403, "Forbidden", None, None)
        with self.assertRaises(urllib.error.HTTPError):
            sg.semver_problems(Path("."), "Cargo.toml", "x", "1.0.0", {}, None, fetch, run=lambda *a: self.fail("ran"))


class EqualVersion(unittest.TestCase):
    """the opt-in "next-smallest-step" mode: the default stays the strict release-path rule"""

    MODE = "next-smallest-step"

    def test_the_default_refuses_an_equal_version(self):
        p, n, _ = problems("0.5.1", "0.5.1", lambda *a: self.fail("ran"))
        self.assertEqual((n, len(p)), (0, 1))
        self.assertIn("is not older than 0.5.1", p[0])

    def test_the_release_path_signature_keeps_the_strict_rule(self):
        # the call release_train.py makes (no equal_version argument) refuses an equal version
        p, n, _ = sg.semver_problems(Path("."), "Cargo.toml", "x", "0.5.1", {}, None, latest_fixture("x", "0.5.1"),
                                     run=lambda *a: self.fail("ran"))
        self.assertEqual((n, len(p)), (0, 1))

    def test_an_equal_version_is_checked_as_the_smallest_next_release(self):
        seen = []

        def run(repo, manifest_path, baseline, features, target_dir):
            seen.append(tuple(baseline))
            return 0, semver_output("minor", 223, 0, from_v="0.5.1", to_v="0.5.1", assume=True)
        p, n, note = problems("0.5.1", "0.5.1", run, self.MODE)
        self.assertEqual((p, n), ([], 2))
        # the step is passed to the tool, so it cannot infer "no change"
        self.assertEqual(set(seen), {("--baseline-version", "0.5.1", "--release-type", "minor")})
        self.assertIn("version equals the published 0.5.1: checking the next release as the smallest step, "
                      "0.5.2 (minor)", note)

    def test_the_smallest_step_follows_cargo_compatibility(self):
        for version, step in (("1.2.3", "patch"), ("0.5.1", "minor"), ("1.0.0", "patch")):
            seen = []

            def run(repo, manifest_path, baseline, features, target_dir, step=step):
                seen.append(baseline[-1])
                return 0, semver_output(step, 9, 0, assume=True)
            p, n, _ = problems(version, version, run, self.MODE)
            self.assertEqual((p, n, set(seen)), ([], 2, {step}), version)

    def test_an_equal_0_0_z_version_makes_a_major_step_and_is_not_compared(self):
        p, n, note = problems("0.0.3", "0.0.3", lambda *a: self.fail("ran"), self.MODE)
        self.assertEqual((p, n), ([], 0))
        self.assertIn("0.0.4 (major)", note)

    def test_an_equal_version_with_a_break_is_refused(self):
        out = semver_output("minor", 223, 1, ["function_missing"], "0.5.1", "0.5.1", assume=True)
        p, n, _ = problems("0.5.1", "0.5.1", lambda *a: (1, out), self.MODE)
        self.assertEqual((n, len(p)), (0, 2))
        self.assertIn("a minor version does not allow function_missing", p[0])

    def test_an_equal_version_whose_run_compared_nothing_is_refused(self):
        out = semver_output("minor", 0, 0, from_v="0.5.1", to_v="0.5.1", assume=True)
        p, n, _ = problems("0.5.1", "0.5.1", lambda *a: (0, out), self.MODE)
        self.assertEqual((n, len(p)), (0, 2))
        self.assertIn("compared 0 checks", p[0])

    def test_an_equal_version_where_the_tool_names_another_step_is_refused(self):
        out = semver_output("patch", 10, 0, from_v="0.5.1", to_v="0.5.1", assume=True)
        p, _, _ = problems("0.5.1", "0.5.1", lambda *a: (0, out), self.MODE)
        self.assertIn("compared a patch step, the versions make a minor step", p[0])

    def test_a_greater_version_takes_the_release_path(self):
        seen = []

        def run(repo, manifest_path, baseline, features, target_dir):
            seen.append(tuple(baseline))
            return 0, semver_output("minor", 50, 0, from_v="0.5.1", to_v="0.5.2")
        p, n, note = problems("0.5.2", "0.5.1", run, self.MODE)
        self.assertEqual((p, n), ([], 2))
        self.assertEqual(set(seen), {("--baseline-version", "0.5.1")})
        self.assertNotIn("equals", note)
        self.assertIn("0.5.1 -> 0.5.2 (minor)", note)

    def test_a_lower_version_is_refused(self):
        for latest_v, version in (("0.5.2", "0.5.1"), ("1.0.0", "1.0.0-rc.1"), ("1.0.0-rc.2", "1.0.0-rc.1")):
            p, n, _ = problems(version, latest_v, lambda *a: self.fail("ran"), self.MODE)
            self.assertEqual((n, len(p)), (0, 1), (latest_v, version))
            self.assertIn("is not older than", p[0])

    def test_an_equal_pre_release_is_refused(self):
        p, n, _ = problems("1.0.0-rc.1", "1.0.0-rc.1", lambda *a: self.fail("ran"), self.MODE)
        self.assertEqual((n, len(p)), (0, 1))
        self.assertIn("pre-release", p[0])

    def test_an_unknown_mode_is_an_error(self):
        with self.assertRaises(ValueError):
            problems("0.5.1", "0.5.1", lambda *a: self.fail("ran"), "skip")


class Cli(unittest.TestCase):
    MANIFEST = '[package]\nname = "x"\nversion = "0.5.1"\n\n[features]\ndefault = ["a"]\na = []\npy = []\n'

    def cli(self, config: str | None, latest_v: str | None, run, manifest: str = MANIFEST) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d, "Cargo.toml")
            m.write_text(manifest, encoding="utf-8")
            argv = ["--manifest", str(m)]
            if config is not None:
                Path(d, "c.toml").write_text(config, encoding="utf-8")
                argv += ["--config", str(Path(d, "c.toml"))]
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = sg.main(argv, fetch=latest_fixture("x", latest_v), run=run)
        return code, out.getvalue()

    CONFIG = ('consumer-features = [["a"]]\nconsumer-features-why = "w"\nconsumer-check-features = [["py"]]\n'
              'equal-version = "next-smallest-step"\n')

    def test_the_mode_is_printed_and_an_additive_change_passes(self):
        code, out = self.cli(self.CONFIG, "0.5.1",
                             lambda *a: (0, semver_output("minor", 9, 0, from_v="0.5.1", to_v="0.5.1", assume=True)))
        self.assertEqual(code, 0, out)
        self.assertIn("equal-version mode: next-smallest-step", out)
        self.assertIn("version equals the published 0.5.1: checking the next release as the smallest step, 0.5.2", out)
        self.assertIn("4 feature set(s) compared", out)

    def test_an_equal_version_prints_the_version_checked(self):
        for version, upcoming, step in (("0.5.1", "0.5.2", "minor"), ("1.2.3", "1.2.4", "patch")):
            manifest = self.MANIFEST.replace('version = "0.5.1"', f'version = "{version}"')
            code, out = self.cli(self.CONFIG, version, lambda *a, step=step: (
                0, semver_output(step, 9, 0, from_v=version, to_v=version, assume=True)), manifest=manifest)
            self.assertEqual(code, 0, out)
            self.assertIn(f"semver gate: x {version} equals the latest release {version}: "
                          f"checking it as the next release {upcoming} ({step} step)\n", out)
            self.assertEqual(sg.smallest_next_release(version, version), (upcoming, step))
        # a newer version, a pre-release and the strict mode print no target
        self.assertIsNone(sg.smallest_next_release("0.5.1", "0.5.2"))
        self.assertIsNone(sg.smallest_next_release("1.0.0-rc.1", "1.0.0-rc.1"))
        code, out = self.cli(None, "0.5.1", lambda *a: self.fail("ran"))
        self.assertNotIn("checking it as the next release", out)

    def test_a_break_exits_1(self):
        code, out = self.cli(self.CONFIG, "0.5.1", lambda *a: (
            1, semver_output("minor", 9, 1, ["function_missing"], "0.5.1", "0.5.1", assume=True)))
        self.assertEqual(code, 1, out)
        self.assertIn("refused: x 0.5.1 (default features): a minor version does not allow function_missing", out)

    def test_the_default_mode_is_the_strict_rule(self):
        code, out = self.cli(None, "0.5.1", lambda *a: self.fail("ran"))
        self.assertEqual(code, 1, out)
        self.assertIn("equal-version mode: refuse", out)

    def test_no_release_passes_with_a_message(self):
        code, out = self.cli(self.CONFIG, None, lambda *a: self.fail("ran"))
        self.assertEqual(code, 0, out)
        self.assertIn("x has no release on crates.io: nothing to compare", out)

    def test_the_tool_missing_exits_1(self):
        code, out = self.cli(self.CONFIG, "0.5.1", lambda *a: (127, "[Errno 2] No such file or directory: 'cargo'"))
        self.assertEqual(code, 1, out)
        self.assertIn("did not run (exit 127)", out)

    def test_a_registry_failure_exits_2(self):
        def fetch(path):
            raise urllib.error.URLError("unreachable")
        with tempfile.TemporaryDirectory() as d:
            m = Path(d, "Cargo.toml")
            m.write_text(self.MANIFEST, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sg.main(["--manifest", str(m)], fetch=fetch, run=lambda *a: self.fail("ran")), 2)

    def test_a_bad_config_or_manifest_exits_2(self):
        never = lambda *a: self.fail("ran")  # noqa: E731
        for config in ('unknown = 1\n', 'consumer-features = [["a"]]\n', 'consumer-features-why = "w"\n',
                       'equal-version = "skip"\n', 'consumer-features = ["a"]\nconsumer-features-why = "w"\n',
                       'consumer-features = [["nope"]]\nconsumer-features-why = "w"\n'):
            self.assertEqual(self.cli(config, "0.5.1", never)[0], 2, config)
        self.assertEqual(self.cli(None, "0.5.1", never, manifest='[package]\nname = "x"\nversion.workspace = true\n')[0], 2)


class SemverEndToEnd(unittest.TestCase):
    """the real cargo semver-checks on two small crates: the baseline 1.0.0 has a
    #[non_exhaustive] unit-only enum and a public function"""

    BASE = ("#[derive(Clone, Copy, Debug)]\n#[non_exhaustive]\npub enum Check {\n    A,\n    B,\n}\n"
            "pub fn probe() -> u32 {\n    1\n}\n")

    def crate(self, root: Path, name: str, version: str, src: str) -> Path:
        d = root / name
        (d / "src").mkdir(parents=True)
        (d / "Cargo.toml").write_text(f'[package]\nname = "semverprobe"\nversion = "{version}"\nedition = "2021"\n'
                                      f'publish = false\n\n[workspace]\n', encoding="utf-8")
        (d / "src" / "lib.rs").write_text(src, encoding="utf-8")
        return d

    def gate(self, version: str, src: str, equal_version: str = "refuse") -> tuple[list[str], int, str]:
        if shutil.which("cargo-semver-checks") is None:
            self.fail("cargo-semver-checks is not installed (CI installs it; locally: cargo install cargo-semver-checks)")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = self.crate(root, "base", "1.0.0", self.BASE)
            new = self.crate(root, "new", version, src)
            return sg.semver_problems(
                new, "Cargo.toml", "semverprobe", version, {}, None, latest_fixture("semverprobe", "1.0.0"),
                run=lambda repo, mp, baseline, features, td: sg.run_semver_checks(
                    repo, mp, ["--baseline-root", str(base), *baseline[2:]], features, td),
                target_dir=root / "target", equal_version=equal_version)

    def test_a_field_variant_under_a_minor_version_is_refused(self):
        src = self.BASE.replace("    B,\n", "    B,\n    C { id: [u8; 32] },\n")
        p, n, _ = self.gate("1.1.0", src)
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 2, p)  # default features and none
        self.assertIn("enum_discriminants_undefined_non_unit_variant", p[0])

    def test_a_unit_variant_under_a_minor_version_passes(self):
        p, n, note = self.gate("1.1.0", self.BASE.replace("    B,\n", "    B,\n    C,\n"))
        self.assertEqual((p, n), ([], 2), p)
        self.assertIn("1.0.0 -> 1.1.0 (minor)", note)

    def test_an_unchanged_api_under_a_patch_version_passes(self):
        self.assertEqual(self.gate("1.0.1", self.BASE)[:2], ([], 2))

    def test_the_same_change_under_a_major_version_is_not_compared(self):
        p, n, note = self.gate("2.0.0", self.BASE.replace("    B,\n", "    B,\n    C { id: [u8; 32] },\n"))
        self.assertEqual((p, n), ([], 0))
        self.assertIn("major step", note)

    def test_an_equal_version_with_an_addition_passes(self):
        p, n, note = self.gate("1.0.0", self.BASE + "pub fn added() -> u32 {\n    2\n}\n", "next-smallest-step")
        self.assertEqual((p, n), ([], 2), p)
        self.assertIn("smallest step, 1.0.1 (patch)", note)

    def test_an_equal_version_with_a_removed_function_is_refused(self):
        p, n, _ = self.gate("1.0.0", self.BASE.replace("pub fn probe", "fn probe"), "next-smallest-step")
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 2, p)
        self.assertIn("a patch version does not allow function_missing", p[0])


if __name__ == "__main__":
    unittest.main()
