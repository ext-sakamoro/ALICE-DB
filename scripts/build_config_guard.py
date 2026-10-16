#!/usr/bin/env python3
"""No build configuration sets a flag that changes floating-point results.

ALICE-Law G14 (float-environment-invariants): a Law-bearing crate is not
built with a fast-math-style flag (reassociation, contraction to fma,
flushing subnormals to zero, assuming no NaN / infinity / signed zero) or
with `target-cpu=native` (which selects instructions for whichever machine
builds it). This script reads every place such a flag can come from and
fails on any occurrence that is not deliberate and recorded. The file is
generic and is kept byte-identical across the repositories that carry it.

Files read (anywhere under the root; target/, node_modules/ and .git/ are
skipped), by category:

    cargo-config   .cargo/config, .cargo/config.toml   [build] / [target.*] rustflags, [env]
    cargo-toml     Cargo.toml                          profile rustflags
    pyproject      pyproject.toml                      maturin rustc-args, env
    build-rs       build.rs                            cargo:rustc-flags, cc / cmake flags
    workflow       .github/workflows/*.yml|yaml,
                   .github/actions/**/*.yml|yaml       env at any level, run steps, $GITHUB_ENV
    script         Makefile, makefile, GNUmakefile, *.mk, justfile, Justfile,
                   *.sh, scripts/**/*.py (not scripts/test_*.py, whose strings are
                   test data, and not this file)
    src            *.rs under a src/ directory         float intrinsics, see below

The TOML files (the first three) are parsed with tomllib and must parse. In
them, and in build.rs, every occurrence outside a comment is a finding: they
are configuration or build code throughout. In workflow and script files an
occurrence is a finding when its line sets or passes flags: the line names
RUSTFLAGS / RUSTDOCFLAGS / CARGO_*RUSTFLAGS / rustflags / CFLAGS / CXXFLAGS,
writes $GITHUB_ENV, runs `cargo rustc` or `rustc` or a C compiler (cc, gcc,
clang, .flag(...)), or is inside a YAML block scalar (`KEY: |` / `KEY: >`)
whose key names one of those variables. A step name or an echo that does not
set anything is not a finding.

Comments are blanked first (`#` comments in TOML, YAML, shell, make and
Python; `//` and `/* */` in Rust); string contents are kept, since flags live
in strings.

Rules, one per occurrence:

    target-cpu-native   target-cpu=native in any spelling (-C target-cpu=native,
                        -Ctarget-cpu=native, an array or 0x1f-encoded form)
    llvm-fast-math      LLVM options that relax IEEE 754: enable-unsafe-fp-math,
                        enable-no-nans-fp-math, enable-no-infs-fp-math,
                        enable-no-signed-zeros-fp-math, fp-contract=fast,
                        denormal-fp-math (any value)
    c-fast-math         C / C++ flags of the same kind: -ffast-math, -Ofast,
                        -funsafe-math-optimizations, -fassociative-math,
                        -freciprocal-math, -ffinite-math-only, -fno-signed-zeros,
                        -ffp-contract=fast, -fdenormal-fp-math, -mdaz-ftz,
                        /fp:fast, -fp-model fast, and -march=native / -mcpu=native /
                        -mfma (a C compiler contracts a * b + c into fma by default
                        once fma is available)
    x87-float           a target-feature list that disables sse or sse2 (x86 then
                        computes in x87 extended precision)
    fast-intrinsic      (src only) fadd_fast / fsub_fast / fmul_fast / fdiv_fast /
                        frem_fast, fadd_algebraic and the other *_algebraic
                        intrinsics, the algebraic_add / _sub / _mul / _div / _rem
                        methods, and #![feature(core_intrinsics)] /
                        #![feature(float_algebraic)]
    unparseable         a TOML file that does not parse
    unreadable          a file of these kinds that cannot be read as UTF-8

Not findings, by policy: a Rust target-feature that adds or removes
instructions other than sse / sse2 (+avx2, +fma, +dotprod, -avx2, -fma,
-sse4.1) and #[target_feature(enable = ...)] in source. Rust never contracts
a * b + c into an fma, mul_add is fused whether or not fma is enabled, and the
x86_64 float ABI is SSE2 throughout, so these select different instructions
for the same IEEE 754 results. The tests pin this policy.

Allowlist (per repository, optional): scripts/build_config_allowlist.toml

    [[allow]]          a deliberate finding, e.g. a CI test leg that runs the
    path = "..."       tests under target-cpu=native to show the results do not
    text = "..."       depend on the CPU (not a shipped build): it covers every
    reason = "..."     finding in `path` whose line, with comments blanked,
                       contains `text`
    [[exclude]]        a file that is not read at all (a configuration outside
    path = "..."       the shipped tree, or test data)
    reason = "..."

An entry without a non-empty reason, an allow that covers no finding (text
that matches only inside a comment covers nothing), an exclude whose path is
not a file this guard reads, and an allowlist that is not valid TOML are
problems (rule `allowlist`). Each entry is printed with the number of findings
it covered.

YAML is not parsed here (the standard library has no YAML parser); the
workflow files are parsed by the repository's actionlint job.

Output: `path:line: rule: text` per problem, the allowed findings and entries,
then the counts per category. Exit 0 clean, 1 problems, 2 when nothing was
read or the root has no Cargo.toml.

    python3 scripts/build_config_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

SKIP_DIRS = {"target", "node_modules", ".git"}
SELF = {"scripts/build_config_guard.py", "scripts/test_build_config_guard.py"}
ALLOWLIST = "scripts/build_config_allowlist.toml"
CATEGORIES = ["cargo-config", "cargo-toml", "pyproject", "build-rs", "workflow", "script", "src"]
TOML = {"cargo-config", "cargo-toml", "pyproject"}

RULES = [
    ("target-cpu-native", re.compile(r"target-cpu\s*=\s*[\"']?native\b")),
    ("llvm-fast-math", re.compile(
        r"enable-(?:unsafe|no-nans|no-infs|no-signed-zeros)-fp-math"
        r"|(?<![\w-])-?fp-contract=fast|(?<![\w-])-?denormal-fp-math")),
    ("c-fast-math", re.compile(
        r"(?<![\w-])-(?:ffast-math|Ofast\b|funsafe-math-optimizations|fassociative-math"
        r"|freciprocal-math|ffinite-math-only|fno-signed-zeros|ffp-contract=fast"
        r"|fdenormal-fp-math|mdaz-ftz|march=native\b|mcpu=native\b|mfma\b)"
        r"|/fp:fast|-fp-model[ =]fast")),
    ("x87-float", re.compile(r"target-feature\s*=\s*[\"']?[^\s\"']*-sse2?(?![0-9.\w])")),
]
SOURCE_RULES = [
    ("fast-intrinsic", re.compile(
        r"\b(?:f(?:add|sub|mul|div|rem)_(?:fast|algebraic)|algebraic_(?:add|sub|mul|div|rem))\b"
        r"|feature\s*\(\s*(?:core_intrinsics|float_algebraic)\b")),
]
# a workflow / script line that sets or passes flags
FLAG_CONTEXT = re.compile(
    r"(?i)rust(?:doc)?flags|\bc(?:xx)?flags\b|GITHUB_ENV|\bcargo\s+rustc\b|\brustc\b"
    r"|(?:^|[\s;|&(`$])(?:cc|gcc|g\+\+|clang|clang\+\+)\s|\.flag(?:_if_supported)?\s*\(")
FLAG_KEY_BLOCK = re.compile(
    r"^(\s*)-?\s*[\"']?[\w.-]*(?:FLAGS|flags)[\w.-]*[\"']?\s*:\s*[|>][-+0-9]*\s*$")
MAKE_NAMES = {"Makefile", "makefile", "GNUmakefile", "justfile", "Justfile"}


def category(rel: str) -> str | None:
    """The category of a file this guard reads, or None to skip it."""
    p = Path(rel)
    name, parts = p.name, p.parts
    if name in ("config", "config.toml") and len(parts) >= 2 and parts[-2] == ".cargo":
        return "cargo-config"
    if name == "Cargo.toml":
        return "cargo-toml"
    if name == "pyproject.toml":
        return "pyproject"
    if name == "build.rs":
        return "build-rs"
    if len(parts) >= 3 and parts[0] == ".github" and parts[1] in ("workflows", "actions") \
            and p.suffix in (".yml", ".yaml"):
        return "workflow"
    if name in MAKE_NAMES or p.suffix in (".mk", ".sh"):
        return "script"
    if parts and parts[0] == "scripts" and p.suffix == ".py" and rel not in SELF \
            and not name.startswith("test_"):
        return "script"
    if p.suffix == ".rs" and "src" in parts[:-1]:
        return "src"
    return None


def blank_hash_comments(line: str) -> str:
    """Drop a `#` comment that starts the line or follows whitespace, outside quotes."""
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
        i += 1
    return line


def blank_rust_comments(text: str) -> str:
    """Replace // and /* */ comments with spaces, keeping strings and line numbers."""
    out = []
    i, n = 0, len(text)
    depth = 0
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if depth:
            if c == "/" and nxt == "*":
                depth += 1
                out.append("  ")
                i += 2
            elif c == "*" and nxt == "/":
                depth -= 1
                out.append("  ")
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
            continue
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and nxt == "*":
            depth = 1
            out.append("  ")
            i += 2
        elif c == "r" and re.match(r'r#*"', text[i:]) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'r(#*)"', text[i:])
            close = '"' + m.group(1)
            j = text.find(close, i + len(m.group(0)))
            j = n if j < 0 else j + len(close)
            out.append(text[i:j])
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def code_lines(text: str, cat: str) -> list[str]:
    if cat in ("build-rs", "src"):
        return blank_rust_comments(text).split("\n")
    return [blank_hash_comments(line) for line in text.split("\n")]


def in_flag_context(lines: list[str], cat: str) -> list[bool]:
    """Which lines of a workflow / script file set or pass flags."""
    if cat not in ("workflow", "script"):
        return [True] * len(lines)
    ctx = []
    block_indent = None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if block_indent is not None and stripped and indent <= block_indent:
            block_indent = None
        in_block = block_indent is not None and bool(stripped)
        ctx.append(in_block or bool(FLAG_CONTEXT.search(line)))
        m = FLAG_KEY_BLOCK.match(line) if cat == "workflow" else None
        if m:
            block_indent = len(m.group(1))
    return ctx


def walk(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]) or not path.is_file():
            continue
        yield rel.as_posix(), path


def load_allowlist(root: Path) -> tuple[list[dict], list[dict], list[str]]:
    """The allow and exclude entries, and the problems in the file itself."""
    path = root / ALLOWLIST
    if not path.is_file():
        return [], [], []
    if tomllib is None:
        return [], [], [f"{ALLOWLIST}:0: allowlist: Python 3.11+ (tomllib) is needed to read it"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        return [], [], [f"{ALLOWLIST}:0: allowlist: not readable as TOML ({e.__class__.__name__})"]
    problems = []
    unknown = set(data) - {"allow", "exclude"}
    if unknown:
        problems.append(f"{ALLOWLIST}:0: allowlist: unknown table {sorted(unknown)}")
    entries = {}
    for table, keys in (("allow", ("path", "text", "reason")), ("exclude", ("path", "reason"))):
        rows = data.get(table, [])
        if not isinstance(rows, list):
            problems.append(f"{ALLOWLIST}:0: allowlist: [[{table}]] must be an array of tables")
            rows = []
        good = []
        for i, row in enumerate(rows):
            bad = [k for k in keys if not isinstance(row.get(k), str) or not row[k].strip()]
            extra = sorted(set(row) - set(keys))
            if bad or extra:
                why = "; ".join(x for x in (
                    f"empty or missing {', '.join(bad)}" if bad else "",
                    f"unknown key {', '.join(extra)}" if extra else "") if x)
                problems.append(f"{ALLOWLIST}:0: allowlist: {table} entry {i + 1}: {why}")
                continue
            good.append(row)
        entries[table] = good
    return entries["allow"], entries["exclude"], problems


def scan(root: Path):
    allow, exclude, problems = load_allowlist(root)
    counts = {c: 0 for c in CATEGORIES}
    root_manifest = False
    excluded = {e["path"]: 0 for e in exclude}
    findings = []
    for rel, path in walk(root):
        cat = category(rel)
        if cat is None:
            continue
        if rel in excluded:
            excluded[rel] += 1
            continue
        counts[cat] += 1
        if rel == "Cargo.toml":
            root_manifest = True
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            problems.append(f"{rel}:0: unreadable: {e.__class__.__name__}")
            continue
        if cat in TOML:
            if tomllib is None:
                problems.append(f"{rel}:0: unparseable: Python 3.11+ (tomllib) is needed")
            else:
                try:
                    tomllib.loads(text)
                except tomllib.TOMLDecodeError as e:
                    problems.append(f"{rel}:0: unparseable: {e}")
        lines = code_lines(text, cat)
        ctx = in_flag_context(lines, cat)
        rules = SOURCE_RULES if cat == "src" else RULES
        for lineno, line in enumerate(lines, start=1):
            if not ctx[lineno - 1]:
                continue
            for rule, pattern in rules:
                m = pattern.search(line)
                if m:
                    findings.append((rel, lineno, rule, m.group(0), line))
    used = [0] * len(allow)
    allowed = []
    for rel, lineno, rule, hit, line in findings:
        cover = [i for i, a in enumerate(allow) if a["path"] == rel and a["text"] in line]
        for i in cover:
            used[i] += 1
        msg = f"{rel}:{lineno}: {rule}: {hit}"
        if cover:
            allowed.append(f"allowed {msg} (entry {cover[0] + 1})")
        else:
            problems.append(msg)
    entries = []
    for i, a in enumerate(allow):
        entries.append(f"allow entry {i + 1}: {a['path']}: {a['text']!r}: {used[i]} findings ({a['reason']})")
        if not used[i]:
            problems.append(f"{ALLOWLIST}:0: allowlist: allow entry {i + 1} ({a['path']}: {a['text']!r}) covers no finding")
    for e in exclude:
        entries.append(f"exclude: {e['path']}: {'not read' if excluded[e['path']] else 'no such file'} ({e['reason']})")
        if not excluded[e["path"]]:
            problems.append(f"{ALLOWLIST}:0: allowlist: exclude {e['path']!r} is not a file this guard reads")
    return problems, allowed, entries, counts, root_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=Path(__file__).resolve().parent.parent, type=Path)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    problems, allowed, entries, counts, root_manifest = scan(root)
    read = sum(counts.values())
    for line in problems + allowed + entries:
        print(line)
    per = ", ".join(f"{c} {counts[c]}" for c in CATEGORIES)
    print(f"build_config_guard: {read} files read ({per}), {len(allowed)} allowed, {len(problems)} problems")
    if read == 0 or not root_manifest:
        print("build_config_guard: nothing to check (no root Cargo.toml or no file read)", file=sys.stderr)
        return 2
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
