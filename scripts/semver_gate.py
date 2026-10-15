#!/usr/bin/env python3
"""cargo semver-checks against the latest crates.io release, once per consumer feature set.

A crate's public API may change only as far as the version in its Cargo.toml allows,
measured from the latest release on crates.io:

  - the step from that release to the Cargo.toml version is the one Cargo's compatibility
    rules name (the leftmost non-zero part is "major": 0.4 -> 0.5 is major, 0.4.1 -> 0.4.2
    is minor, 1.2.3 -> 1.2.4 is patch), see `version_step`
  - `cargo semver-checks check-release --baseline-version <release>` runs once per feature
    set of the consumer build (default features, none, and the declared consumer sets, or
    every feature at once when none are declared), see `feature_specs`
  - a run is refused when the tool did not run, names a different step than the two
    versions make, prints no count of checks, compared 0 checks, or found a change the
    step does not allow (or exited non-zero), see `semver_verdict`
  - a major step is not compared (any change is allowed); a crate with no release has
    nothing to compare; a Cargo.toml version that is not newer than the release (SemVer
    2.0 precedence, so a pre-release below its release is refused) is refused, see
    `semver_problems`

Shared module. The functions `feature_specs`, `version_step`, `version_precedence`,
`semver_feature_args`, `run_semver_checks`, `semver_verdict`, `previous_release` and
`semver_problems` are those of ALICE-Law scripts/release_train.py (main eb0779d), with the
same names, signatures and results, so that file can import them from here. This file and
scripts/test_semver_gate.py hold no repository-specific name or path; the data of a
repository (its consumer feature sets, the equal-version mode) is a small TOML file
passed with --config. Keep every copy byte-identical and change them together.

Differences from release_train.py eb0779d (behaviour, not code layout):
  - `fetch` is typed `Fetch` from this module (release_train.py names it `dc.Fetch`, the
    same Callable[[str], dict]); `latest_release` and `RegistryError` live here, and
    `previous_release` raises this module's RegistryError on a body without a version,
    where release_train.py's `latest` raises its own RegistryError
  - `semver_problems` takes one more keyword argument, `equal_version`. Its default,
    "refuse", is release_train.py's behaviour unchanged: a Cargo.toml version equal to the
    latest release is refused. The release path (the publish gate) keeps that default and
    stays strict: the version being published must be newer than the latest release.
    "next-smallest-step" is an opt-in for a repository that keeps the released version in
    Cargo.toml while changes accumulate before the next release: an equal version is
    checked as the smallest next release (1.2.3 -> 1.2.4 patch, 0.5.1 -> 0.5.2 minor,
    0.0.3 -> 0.0.4 major, which is not compared), with that step passed to the tool as
    `--release-type` so it cannot infer "no change"; a change that needs a larger step
    fails until Cargo.toml carries that bump. A pre-release that equals the latest release
    has no defined smallest next release and is refused in this mode too
  - `http_fetch` does not retry an HTTP 4xx answer (a 404 is the answer "no such crate")

CLI (exit 0 pass, 1 refused, 2 the gate could not decide; every failure path is closed):

  python3 scripts/semver_gate.py [--manifest Cargo.toml] [--config FILE] [--target-dir DIR]

  --config FILE   TOML with the keys of a release-train roster entry that this gate reads:
                  `consumer-features` (cargo build sets), `consumer-check-features`
                  (cargo check sets), `consumer-features-why` (required with either), and
                  `equal-version` ("refuse", the default, or "next-smallest-step");
                  any other key is an error
  The crate name and version are read from [package] of the manifest; the latest release
  from the crates.io API (`max_stable_version`, else `max_version`). A network error, an
  HTTP answer other than 200 / 404, or a body without a version exits 2.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

API = "https://crates.io/api/v1/crates"
USER_AGENT = "semver-gate (https://github.com/ext-sakamoro)"

Fetch = Callable[[str], dict]


class RegistryError(RuntimeError):
    """crates.io answered, but not with the version asked for"""


class ConfigError(ValueError):
    """the --config file or the manifest cannot be used"""


def http_fetch(path: str) -> dict:
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            time.sleep(1)  # crates.io asks for at most one request per second
            return data
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def latest_release(fetch: Fetch, crate: str) -> str:
    c = fetch(crate)["crate"]
    return c.get("max_stable_version") or c["max_version"]


def feature_specs(manifest: dict, entry: dict | None) -> list[tuple[str, str, list[str] | None, str]]:
    """(label, dependency spec, features, cargo command) of each build: the default features,
    none, and either the roster's `consumer-features` sets (`cargo build`) and
    `consumer-check-features` sets (`cargo check`: compiled but not linked, for a feature whose
    library another tool links, such as a Python extension) or every feature at once"""
    entry = entry or {}
    specs = [("default features", "", None, "build"), ("no default features", ", default-features = false", [], "build")]
    declared = [(fs, "build") for fs in entry.get("consumer-features", [])] + \
               [(fs, "check") for fs in entry.get("consumer-check-features", [])]
    if declared:
        for fs, cmd in declared:
            specs.append((f"features {fs}" + (" (cargo check)" if cmd == "check" else ""),
                          ", default-features = false, features = " + json.dumps(fs), fs, cmd))
    elif [f for f in manifest.get("features", {}) if f != "default"]:
        every = sorted(f for f in manifest["features"] if f != "default")
        specs.append(("all features", ", features = " + json.dumps(every), every, "build"))
    return specs


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def version_step(previous: str, version: str) -> str:
    """the step from one release to the next as cargo-semver-checks names it: a change of
    the leftmost non-zero part is "major", of the next one "minor", of the last "patch"
    (Cargo's compatibility: 0.4 -> 0.5 is major, 0.4.1 -> 0.4.2 is minor)"""
    old = [int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", previous).groups()]
    new = [int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups()]
    first = next((i for i, x in enumerate(old) if x), 2)
    for i in range(3):
        if old[i] != new[i]:
            return "major" if i <= first else ("minor" if i == first + 1 else "patch")
    return "patch"


def version_precedence(version: str) -> tuple:
    """a sort key of SemVer 2.0 precedence (build metadata ignored): 1.0.0-rc.1 < 1.0.0,
    1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta, numeric identifiers below alphanumeric ones"""
    core, _, pre = version.split("+")[0].partition("-")
    nums = tuple(int(x) for x in core.split("."))
    if not pre:
        return nums, 1, ()
    ids = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return nums, 0, ids


def semver_feature_args(features: list[str] | None) -> list[str]:
    """None: the default features; a list (possibly empty): exactly those"""
    if features is None:
        return ["--default-features"]
    return ["--only-explicit-features"] + (["--features", ",".join(features)] if features else [])


def run_semver_checks(repo: Path, manifest_path: str, baseline: list[str], features: list[str] | None,
                      target_dir: Path) -> tuple[int, str]:
    """(exit code, stdout + stderr) of `cargo semver-checks check-release` for one feature set"""
    cmd = ["cargo", "semver-checks", "check-release", "--manifest-path", str(repo / manifest_path),
           *baseline, *semver_feature_args(features)]
    env = {**os.environ, "CARGO_TERM_COLOR": "never", "CARGO_TARGET_DIR": str(target_dir)}
    try:
        r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    except FileNotFoundError as e:
        return 127, str(e)
    return r.returncode, r.stdout + r.stderr


def semver_verdict(code: int, output: str, step: str, label: str) -> tuple[str | None, int]:
    """(problem or None, checks counted) of one cargo-semver-checks run whose two versions
    make `step`"""
    text = ANSI.sub("", output)
    claim = re.search(r"Checking \S+ v\S+ -> v\S+ \((?:assume )?(major|minor|patch) change\)", text)
    if claim is None:
        tail = " / ".join(line.strip() for line in text.strip().splitlines()[-3:])
        return f"{label}: cargo semver-checks did not run (exit {code}): {tail}", 0
    if claim.group(1) != step:
        return f"{label}: cargo semver-checks compared a {claim.group(1)} step, the versions make a {step} step", 0
    if step == "major":
        return None, 0
    # "223 checks: 223 pass, 31 skip" when every check passed, "196 checks: 195 pass, 1 fail, ..." otherwise
    checked = re.search(r"Checked \[[^\]]*\]\s+(\d+) checks?: (\d+) pass(?:, (\d+) fail)?", text)
    if checked is None:
        return f"{label}: cargo semver-checks printed no count of checks (exit {code})", 0
    total, failed = int(checked.group(1)), int(checked.group(3) or 0)
    if total == 0:
        return f"{label}: cargo semver-checks compared 0 checks for a {step} step", 0
    lints = re.findall(r"--- failure ([A-Za-z0-9_]+):", text)
    summary = re.search(r"Summary (.+)", text)
    if failed or lints or code != 0:
        return (f"{label}: a {step} version does not allow {', '.join(lints) or f'{failed} failed check(s)'}"
                f" ({summary.group(1).strip() if summary else f'exit {code}'})"), 0
    return None, 1


def previous_release(fetch: Fetch, crate: str) -> str | None:
    """the latest release on crates.io, None when the crate has none"""
    try:
        return latest_release(fetch, crate)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except (KeyError, TypeError) as e:
        raise RegistryError(f"{crate}: crates.io answered without a latest release ({e!r})") from None


EQUAL_VERSION_MODES = ("refuse", "next-smallest-step")


def next_patch(version: str) -> str:
    """the smallest release after `version` (a release, not a pre-release): x.y.z -> x.y.(z+1)"""
    major, minor, patch = (int(x) for x in re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version).groups())
    return f"{major}.{minor}.{patch + 1}"


def semver_problems(repo: Path, manifest_path: str, crate: str, version: str, manifest: dict, entry: dict | None,
                    fetch: Fetch, run=run_semver_checks, target_dir: Path | None = None,
                    equal_version: str = "refuse") -> tuple[list[str], int, str]:
    """(problems, runs compared, what was compared) of cargo semver-checks against the latest
    release, once per feature set of the consumer build"""
    if equal_version not in EQUAL_VERSION_MODES:
        raise ValueError(f"equal_version must be one of {EQUAL_VERSION_MODES}, not {equal_version!r}")
    previous = previous_release(fetch, crate)
    if previous is None:
        return [], 0, f"{crate} has no release on crates.io: nothing to compare"
    baseline = ["--baseline-version", previous]
    note = ""
    if equal_version == "next-smallest-step" and version_precedence(previous) == version_precedence(version):
        if "-" in version.split("+")[0]:
            return [f"{crate}: the version {version} equals the latest release, a pre-release: "
                    f"no smallest next release is defined, set the next version in Cargo.toml"], 0, ""
        upcoming = next_patch(version.split("+")[0])
        step = version_step(previous, upcoming)
        note = (f"version equals the published {previous}: checking the next release as the smallest step, "
                f"{upcoming} ({step}); ")
        if step == "major":
            return [], 0, note + "a major step: any change is allowed, not compared"
        baseline += ["--release-type", step]
    else:
        step = version_step(previous, version)
        if version_precedence(previous) >= version_precedence(version):
            return [f"{crate}: the latest release {previous} is not older than {version}"], 0, ""
        if step == "major":
            return [], 0, f"{previous} -> {version} is a major step: any change is allowed, not compared"
    problems, compared = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, _, features, _ in feature_specs(manifest, entry):
            code, out = run(repo, manifest_path, baseline, features, target_dir or Path(tmp))
            p, n = semver_verdict(code, out, step, f"{crate} {version} ({label})")
            if p:
                problems.append(p)
            compared += n
    # every run either counts or is a problem (a run that compared 0 checks is refused above)
    return problems, compared, note + f"{previous} -> {version} ({step}), {compared} feature set(s) compared"


# ---------------------------------------------------------------- the CLI

CONFIG_KEYS = {"consumer-features", "consumer-check-features", "consumer-features-why", "equal-version"}


def load_config(text: str) -> dict:
    """the --config file: the consumer feature sets and the equal-version mode"""
    c = tomllib.loads(text)
    unknown = sorted(set(c) - CONFIG_KEYS)
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} (known: {sorted(CONFIG_KEYS)})")
    for key in ("consumer-features", "consumer-check-features"):
        sets = c.get(key, [])
        if not isinstance(sets, list) or not all(isinstance(s, list) and all(isinstance(f, str) for f in s)
                                                 for s in sets):
            raise ConfigError(f"`{key}` must be a list of lists of feature names")
    declared = "consumer-features" in c or "consumer-check-features" in c
    if declared and not str(c.get("consumer-features-why", "")).strip():
        raise ConfigError("`consumer-features` needs `consumer-features-why` (how the sets were chosen)")
    if not declared and "consumer-features-why" in c:
        raise ConfigError("`consumer-features-why` without `consumer-features`")
    if c.get("equal-version", "refuse") not in EQUAL_VERSION_MODES:
        raise ConfigError(f"`equal-version` must be one of {EQUAL_VERSION_MODES}")
    return c


def check_features_exist(manifest: dict, entry: dict) -> None:
    known = set(manifest.get("features", {}))
    for key in ("consumer-features", "consumer-check-features"):
        for s in entry.get(key, []):
            missing = sorted(set(s) - known)
            if missing:
                raise ConfigError(f"`{key}` names feature(s) {missing} that the manifest does not define")


def echo_run(run):
    """a `run` that also prints the tool's output, for the job log"""
    def wrapped(repo, manifest_path, baseline, features, target_dir):
        code, out = run(repo, manifest_path, baseline, features, target_dir)
        print(f"::group::cargo semver-checks {' '.join(baseline)} {' '.join(semver_feature_args(features))} "
              f"(exit {code})")
        print(ANSI.sub("", out).rstrip())
        print("::endgroup::")
        return code, out
    return wrapped


def main(argv: list[str] | None = None, fetch: Fetch = http_fetch, run=run_semver_checks) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="Cargo.toml")
    ap.add_argument("--config", help="TOML with the consumer feature sets and the equal-version mode")
    ap.add_argument("--target-dir", help="CARGO_TARGET_DIR of the runs (default: a temporary directory)")
    a = ap.parse_args(argv)
    manifest_path = Path(a.manifest).resolve()
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        pkg = manifest.get("package", {})
        crate, version = pkg.get("name"), pkg.get("version")
        if not isinstance(crate, str) or not isinstance(version, str):
            raise ConfigError(f"{manifest_path}: [package] needs a literal name and version")
        entry = load_config(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
        check_features_exist(manifest, entry)
    except (OSError, tomllib.TOMLDecodeError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    mode = entry.get("equal-version", "refuse")
    print(f"semver gate: {crate} {version}, equal-version mode: {mode}"
          + (" (a version equal to the latest release is refused)" if mode == "refuse"
             else " (a version equal to the latest release is checked as the smallest next release)"))
    for label, _, features, _ in feature_specs(manifest, entry):
        print(f"  feature set: {label}")
    try:
        problems, compared, note = semver_problems(
            manifest_path.parent, manifest_path.name, crate, version, manifest, entry, fetch,
            run=echo_run(run), target_dir=Path(a.target_dir).resolve() if a.target_dir else None,
            equal_version=mode)
    except (urllib.error.URLError, TimeoutError, RegistryError, json.JSONDecodeError) as e:
        print(f"error: the latest release of {crate} could not be read from crates.io: {e}", file=sys.stderr)
        return 2
    for p in problems:
        print(f"refused: {p}")
    if problems:
        return 1
    print(f"semver gate: pass: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
