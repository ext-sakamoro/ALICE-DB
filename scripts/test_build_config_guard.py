#!/usr/bin/env python3
"""Tests for scripts/build_config_guard.py: a clean tree passes, every rule
fires in every file kind it applies to (one mutant per rule and place), a
flag in a comment does not fire while the same text in a string does, a
workflow / script line that sets nothing does not fire, look-alikes and the
target-feature policy pass, the src intrinsics fire, the allowlist covers,
reports and goes stale, an unreadable or unparseable file fails, and a tree
with nothing to read (or no root Cargo.toml) is exit 2."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_config_guard as g  # noqa: E402

PKG = '[package]\nname = "x"\nversion = "0.1.0"\nedition = "2021"\n'


def tree(files: dict[str, str | bytes]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="build-config-guard-"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            p.write_bytes(text)
        else:
            p.write_text(text, encoding="utf-8")
    return root


def run(files: dict[str, str | bytes]) -> tuple[int, str, str]:
    root = tree(files)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = g.main(["--root", str(root)])
    return rc, out.getvalue(), err.getvalue()


# (file, text holding the flag on line 2 or later, expected line)
PLACES = {
    "cargo-config": (".cargo/config.toml", '[build]\nrustflags = ["-C", "{flag}"]\n', 2),
    "cargo-config-legacy": (".cargo/config", '[target.x86_64-unknown-linux-gnu]\nrustflags = ["{flag}"]\n', 2),
    "cargo-config-cfg": (".cargo/config.toml", "[target.'cfg(target_arch = \"x86_64\")']\nrustflags = [\"{flag}\"]\n", 2),
    "cargo-config-string": (".cargo/config.toml", '[build]\nrustflags = "-C {flag}"\n', 2),
    "nested-cargo-config": ("crates/a/.cargo/config.toml", '[env]\nRUSTFLAGS = "{flag}"\n', 2),
    "fuzz-cargo-config": ("fuzz/.cargo/config.toml", '[build]\nrustflags = [\n  "-C",\n  "{flag}",\n]\n', 4),
    "workflow-env": (".github/workflows/ci.yml", 'env:\n  RUSTFLAGS: "-C {flag}"\njobs: {{}}\n', 2),
    "workflow-run": (".github/workflows/x.yaml", 'jobs:\n  a:\n    steps:\n      - run: RUSTFLAGS="{flag}" cargo build\n', 4),
    "action": (".github/actions/setup/action.yml", 'runs:\n  using: composite\n  env:\n    CARGO_ENCODED_RUSTFLAGS: "{flag}"\n', 4),
    "makefile": ("Makefile", 'build:\n\tRUSTFLAGS="{flag}" cargo build\n', 2),
    "mk": ("make/rules.mk", 'X := 1\nRUSTFLAGS += {flag}\n', 2),
    "justfile": ("justfile", 'build:\n    RUSTFLAGS="{flag}" cargo build\n', 2),
    "shell": ("scripts/preflight.sh", '#!/usr/bin/env bash\nexport RUSTFLAGS="{flag}"\n', 2),
    "shell-subshell": ("scripts/run.sh", '#!/usr/bin/env bash\n( export RUSTFLAGS="{flag}"; cargo build )\n', 2),
    "python-script": ("scripts/bench.py", 'import os\nos.environ["RUSTFLAGS"] = "{flag}"\n', 2),
    "build-rs": ("build.rs", 'fn main() {{\n    println!("cargo:rustc-flags={flag}");\n}}\n', 2),
    "build-rs-cc": ("build.rs", 'fn main() {{\n    cc::Build::new().flag_if_supported("{flag}").compile("x");\n}}\n', 2),
    "cargo-profile": ("Cargo.toml", 'cargo-features = ["profile-rustflags"]\n' + PKG
                      + '\n[profile.release]\nrustflags = ["{flag}"]\n', 8),
    "pyproject": ("pyproject.toml", '[tool.maturin]\nrustc-args = ["{flag}"]\n', 2),
}

FLAGS = {
    "target-cpu-native": ["target-cpu=native", "-Ctarget-cpu=native", "target-cpu = native",
                          "-C\\x1ftarget-cpu=native"],
    "llvm-fast-math": ["llvm-args=-enable-unsafe-fp-math", "llvm-args=-enable-no-nans-fp-math",
                       "llvm-args=-enable-no-infs-fp-math", "llvm-args=-enable-no-signed-zeros-fp-math",
                       "llvm-args=-fp-contract=fast", "llvm-args=-denormal-fp-math=preserve-sign"],
    "c-fast-math": ["-march=native", "-mcpu=native", "-mfma", "-ffast-math", "-Ofast",
                    "-funsafe-math-optimizations", "-fassociative-math",
                    "-freciprocal-math", "-ffinite-math-only", "-fno-signed-zeros", "-ffp-contract=fast",
                    "-fdenormal-fp-math=positive-zero", "-mdaz-ftz", "/fp:fast", "-fp-model fast"],
    "x87-float": ["target-feature=-sse2", "target-feature=+avx,-sse"],
}


class Clean(unittest.TestCase):
    def test_minimal_tree_passes(self):
        rc, out, _ = run({"Cargo.toml": PKG})
        self.assertEqual(rc, 0, out)
        self.assertIn("1 files read", out)

    def test_ordinary_flags_and_the_target_feature_policy_pass(self):
        rc, out, _ = run({
            "Cargo.toml": PKG + '\n[profile.release]\nlto = "fat"\nopt-level = 3\n',
            ".github/workflows/ci.yml": 'env:\n  RUSTFLAGS: "-Dwarnings -C target-feature=+avx2,+fma"\n',
            ".cargo/config.toml": '[build]\nrustflags = ["-C", "target-cpu=x86-64-v2"]\n',
            "build.rs": 'fn main() { cc::Build::new().flag("-O2").flag("-march=x86-64-v2").compile("x"); }\n',
            "scripts/a.sh": 'RUSTFLAGS="-C target-feature=-avx2,-avx,-fma,-sse4.1" cargo test\n',
            "scripts/b.sh": 'RUSTFLAGS="-C target-feature=+dotprod" cargo test\n',
            "src/lib.rs": '#[target_feature(enable = "fma")]\npub unsafe fn f(a: f64) -> f64 { a.mul_add(a, a) }\n',
        })
        self.assertEqual(rc, 0, out)

    def test_counts_per_category(self):
        rc, out, _ = run({
            "Cargo.toml": PKG, ".cargo/config.toml": "[build]\n", "pyproject.toml": "[project]\n",
            "build.rs": "fn main() {}\n", ".github/workflows/ci.yml": "jobs: {}\n",
            "scripts/a.sh": "true\n", "src/lib.rs": "\n",
        })
        self.assertEqual(rc, 0, out)
        self.assertIn("7 files read (cargo-config 1, cargo-toml 1, pyproject 1, build-rs 1, "
                      "workflow 1, script 1, src 1)", out)


class EveryRuleEveryPlace(unittest.TestCase):
    def test_each_flag_fires_at_its_line(self):
        for rule, flags in FLAGS.items():
            for flag in flags:
                for place, (rel, body, line) in PLACES.items():
                    if place == "cargo-config-string" and " " in flag:
                        continue
                    files = {"Cargo.toml": PKG}
                    files[rel] = body.format(flag=flag)
                    with self.subTest(rule=rule, flag=flag, place=place):
                        rc, out, _ = run(files)
                        self.assertEqual(rc, 1, out)
                        self.assertIn(f"{rel}:{line}: {rule}:", out)


class Comments(unittest.TestCase):
    def test_hash_comment_does_not_fire(self):
        rc, out, _ = run({
            "Cargo.toml": PKG + '# RUSTFLAGS="-C target-cpu=native" cargo build --release\n',
            ".cargo/config.toml": '[build]\n# rustflags = ["-C", "target-cpu=native"]\n'
                                  '[profile.release]\nopt-level = 3 # target-cpu=native\n',
            ".github/workflows/ci.yml": 'env:\n  RUSTFLAGS: "-Dwarnings"  # never -C target-cpu=native\n'
                                        '  # RUSTFLAGS: -C target-cpu=native\n',
            "scripts/a.sh": '#!/bin/sh\n  # export RUSTFLAGS=-ffast-math\necho ${#x}\n',
        })
        self.assertEqual(rc, 0, out)

    def test_hash_inside_string_is_not_a_comment(self):
        rc, out, _ = run({"Cargo.toml": PKG, "scripts/a.sh": 'RUSTFLAGS="a # b -C target-cpu=native" cargo b\n'})
        self.assertEqual(rc, 1, out)
        self.assertIn("scripts/a.sh:1: target-cpu-native:", out)

    def test_rust_comments_do_not_fire_but_strings_do(self):
        build = (
            'fn main() {\n'
            '    // println!("cargo:rustc-flags=-C target-cpu=native");\n'
            '    /* cc::Build::new().flag("-ffast-math");\n'
            '       /* nested */ still a comment -Ofast */\n'
            '    let s = "// not a comment -ffast-math";\n'
            '    let r = r#"-C target-cpu=native"#;\n'
            '}\n')
        rc, out, _ = run({"Cargo.toml": PKG, "build.rs": build})
        self.assertEqual(rc, 1, out)
        self.assertIn("build.rs:5: c-fast-math:", out)
        self.assertIn("build.rs:6: target-cpu-native:", out)
        self.assertNotIn("build.rs:2:", out)
        self.assertNotIn("build.rs:3:", out)
        self.assertNotIn("build.rs:4:", out)


class FlagContext(unittest.TestCase):
    def test_lines_that_set_nothing_do_not_fire(self):
        wf = ('jobs:\n  a:\n    steps:\n'
              '      - name: no target-cpu=native here\n'
              '        run: echo "this build never uses -C target-cpu=native"\n'
              '      - run: |\n          echo -ffast-math is not used\n')
        rc, out, _ = run({"Cargo.toml": PKG, ".github/workflows/ci.yml": wf,
                          "scripts/a.sh": 'echo "-C target-cpu=native is not used"\n'})
        self.assertEqual(rc, 0, out)

    def test_lines_that_set_flags_fire(self):
        for text, line in [
            ('jobs:\n  a:\n    env:\n      RUSTFLAGS: -C target-cpu=native\n', 4),
            ("jobs:\n  a:\n    steps:\n      - env:\n          RUSTFLAGS: '-C target-cpu=native'\n", 5),
            ('jobs:\n  a:\n    steps:\n      - run: |\n          cargo test\n'
             '          echo "RUSTFLAGS=-C target-cpu=native" >> "$GITHUB_ENV"\n', 6),
            ('jobs:\n  a:\n    steps:\n      - run: cargo rustc --release -- -C target-cpu=native\n', 4),
            ('jobs:\n  a:\n    steps:\n      - run: RUSTDOCFLAGS="-C target-cpu=native" cargo doc\n', 4),
            ('jobs:\n  a:\n    strategy:\n      matrix:\n        rustflags: ["-C target-cpu=native"]\n', 5),
            ('env:\n  CARGO_BUILD_RUSTFLAGS: >-\n    -Dwarnings\n    -C target-cpu=native\njobs: {}\n', 4),
            ('env:\n  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS: "-C target-cpu=native"\n', 2),
        ]:
            with self.subTest(line=line, text=text):
                rc, out, _ = run({"Cargo.toml": PKG, ".github/workflows/ci.yml": text})
                self.assertEqual(rc, 1, out)
                self.assertIn(f".github/workflows/ci.yml:{line}: target-cpu-native:", out)

    def test_block_scalar_ends_at_dedent(self):
        wf = 'env:\n  RUSTFLAGS: |\n    -Dwarnings\n  NOTE: say target-cpu=native\n'
        rc, out, _ = run({"Cargo.toml": PKG, ".github/workflows/ci.yml": wf})
        self.assertEqual(rc, 0, out)

    def test_c_compiler_lines_fire_in_scripts(self):
        for text in ("all:\n\tcc -O2 -ffast-math -c a.c\n", "build:\n\tgcc -Ofast a.c\n",
                     'all:\n\tCFLAGS="-ffast-math" make\n'):
            with self.subTest(text=text):
                rc, out, _ = run({"Cargo.toml": PKG, "Makefile": text})
                self.assertEqual(rc, 1, out)
                self.assertIn("Makefile:2: c-fast-math:", out)


class Source(unittest.TestCase):
    def test_fast_intrinsics_fire(self):
        for text in ("#![feature(core_intrinsics)]\n",
                     "#![feature(float_algebraic)]\n",
                     "let y = unsafe { core::intrinsics::fadd_fast(a, b) };\n",
                     "let y = unsafe { std::intrinsics::fmul_fast(a, b) };\n",
                     "let y = core::intrinsics::fadd_algebraic(a, b);\n",
                     "let y = a.algebraic_mul(b);\n",
                     "use core::intrinsics::{fdiv_fast, frem_fast};\n"):
            with self.subTest(text=text):
                rc, out, _ = run({"Cargo.toml": PKG, "crates/m/src/x.rs": "\n" + text})
                self.assertEqual(rc, 1, out)
                self.assertIn("crates/m/src/x.rs:2: fast-intrinsic:", out)

    def test_comments_and_non_src_do_not_fire(self):
        rc, out, _ = run({"Cargo.toml": PKG,
                          "src/lib.rs": "// fadd_fast is never used\n/* #![feature(core_intrinsics)] */\n",
                          "tests/t.rs": "#![feature(core_intrinsics)]\n",
                          "src/fast.rs": "pub fn add_fast_path() {}\n"})
        self.assertEqual(rc, 0, out)


class Allowlist(unittest.TestCase):
    LEG = ('jobs:\n  t:\n    strategy:\n      matrix:\n'
           '        rustflags: ["", "-C target-cpu=native"]\n')
    REASON = "test leg: results must not depend on the CPU; not a shipped build"

    def files(self, body):
        return {"Cargo.toml": PKG, ".github/workflows/ci.yml": self.LEG,
                "scripts/build_config_allowlist.toml": body}

    def test_allow_covers_and_reports(self):
        rc, out, _ = run(self.files(
            f'[[allow]]\npath = ".github/workflows/ci.yml"\ntext = "target-cpu=native"\nreason = "{self.REASON}"\n'))
        self.assertEqual(rc, 0, out)
        self.assertIn("allowed .github/workflows/ci.yml:5: target-cpu-native:", out)
        self.assertIn("allow entry 1: .github/workflows/ci.yml: 'target-cpu=native': 1 findings", out)
        self.assertIn("1 allowed, 0 problems", out)

    def test_without_allow_the_leg_fails(self):
        rc, out, _ = run({"Cargo.toml": PKG, ".github/workflows/ci.yml": self.LEG})
        self.assertEqual(rc, 1, out)

    def test_stale_or_bad_entries_fail(self):
        cases = {
            "other path": f'[[allow]]\npath = "scripts/a.sh"\ntext = "target-cpu=native"\nreason = "{self.REASON}"\n',
            "empty reason": '[[allow]]\npath = ".github/workflows/ci.yml"\ntext = "target-cpu=native"\nreason = " "\n',
            "no reason": '[[allow]]\npath = ".github/workflows/ci.yml"\ntext = "target-cpu=native"\n',
            "unknown key": (f'[[allow]]\npath = ".github/workflows/ci.yml"\ntext = "native"\n'
                            f'reason = "{self.REASON}"\nscope = "x"\n'),
            "not toml": "[[allow]\n",
            "unknown table": f'[[permit]]\npath = "x"\nreason = "{self.REASON}"\n',
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                rc, out, _ = run(self.files(body))
                self.assertEqual(rc, 1, out)
                self.assertIn("allowlist:", out)

    def test_text_matching_only_a_comment_covers_nothing(self):
        files = self.files(f'[[allow]]\npath = ".github/workflows/ci.yml"\n'
                           f'text = "cpu-independence"\nreason = "{self.REASON}"\n')
        files[".github/workflows/ci.yml"] = self.LEG.replace(
            '"-C target-cpu=native"]', '"-C target-cpu=native"]  # cpu-independence')
        rc, out, _ = run(files)
        self.assertEqual(rc, 1, out)
        self.assertIn("covers no finding", out)
        self.assertIn(".github/workflows/ci.yml:5: target-cpu-native:", out)

    def test_exclude(self):
        fixture = {"Cargo.toml": PKG,
                   "fuzz/.cargo/config.toml": '[build]\nrustflags = ["-C", "target-cpu=native"]\n'}
        rc, out, _ = run(fixture)
        self.assertEqual(rc, 1, out)
        fixture["scripts/build_config_allowlist.toml"] = (
            '[[exclude]]\npath = "fuzz/.cargo/config.toml"\nreason = "fuzz harness only, never shipped"\n')
        rc, out, _ = run(fixture)
        self.assertEqual(rc, 0, out)
        self.assertIn("exclude: fuzz/.cargo/config.toml: not read", out)
        for body in ('[[exclude]]\npath = "fuzz/.cargo/missing.toml"\nreason = "r"\n',
                     '[[exclude]]\npath = "README.md"\nreason = "r"\n',
                     '[[exclude]]\npath = "fuzz/.cargo/config.toml"\nreason = ""\n'):
            with self.subTest(body=body):
                f = dict(fixture, **{"README.md": "x\n", "scripts/build_config_allowlist.toml": body})
                rc, out, _ = run(f)
                self.assertEqual(rc, 1, out)
                self.assertIn("allowlist:", out)

    def test_test_scripts_are_test_data(self):
        rc, out, _ = run({"Cargo.toml": PKG,
                          "scripts/test_preflight.py": 'CASE = \'RUSTFLAGS="-C target-cpu=native" cargo test\'\n'})
        self.assertEqual(rc, 0, out)


class Scope(unittest.TestCase):
    def test_skipped_dirs_and_other_files_are_not_read(self):
        rc, out, _ = run({
            "Cargo.toml": PKG,
            "target/debug/build/x/build.rs": 'fn main() { let _ = "-ffast-math"; }\n',
            "node_modules/a/Makefile": "all:\n\tcc -Ofast\n",
            "docs/notes.md": "never use -C target-cpu=native\n",
            "README.md": 'RUSTFLAGS="-C target-cpu=native" cargo build\n',
            "tests/x.py": 'FLAGS = "-ffast-math"\n',
        })
        self.assertEqual(rc, 0, out)
        self.assertIn("1 files read", out)

    def test_own_files_are_not_read(self):
        rc, out, _ = run({
            "Cargo.toml": PKG,
            "scripts/build_config_guard.py": 'R = "RUSTFLAGS target-cpu=native -ffast-math"\n',
            "scripts/test_build_config_guard.py": 'F = "RUSTFLAGS -Ofast"\n',
        })
        self.assertEqual(rc, 0, out)


class FailClosed(unittest.TestCase):
    def test_unreadable_file_fails(self):
        rc, out, _ = run({"Cargo.toml": PKG, ".cargo/config.toml": b"\xff\xfe rustflags\n"})
        self.assertEqual(rc, 1, out)
        self.assertIn(".cargo/config.toml:0: unreadable:", out)

    def test_unparseable_toml_fails(self):
        for rel in ("Cargo.toml", ".cargo/config.toml", "pyproject.toml", "fuzz/Cargo.toml"):
            with self.subTest(rel=rel):
                files = {"Cargo.toml": PKG, rel: "[build\nrustflags = 1\n"}
                rc, out, _ = run(files)
                self.assertEqual(rc, 1, out)
                self.assertIn(f"{rel}:0: unparseable:", out)

    def test_nothing_read_is_exit_2(self):
        rc, out, err = run({"README.md": "x\n"})
        self.assertEqual(rc, 2, out + err)
        self.assertIn("0 files read", out)

    def test_no_root_manifest_is_exit_2(self):
        rc, out, err = run({"crates/a/Cargo.toml": PKG, ".github/workflows/ci.yml": "jobs: {}\n"})
        self.assertEqual(rc, 2, out + err)


if __name__ == "__main__":
    unittest.main(verbosity=1)
