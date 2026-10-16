#!/usr/bin/env python3
"""No build configuration sets a flag that changes floating-point results.

ALICE-Law G14 (float-environment-invariants): a Law-bearing crate is not
built with a fast-math-style flag (reassociation, contraction to fma,
flushing subnormals to zero, assuming no NaN / infinity / signed zero) or
with `target-cpu=native` (which selects instructions for whichever machine
builds it). This script reads every file a flag can come from and reports
every occurrence of such a flag outside a comment, whatever the context: a
flag can reach a build through a variable, a matrix, an env entry, a YAML
anchor, a continued line or a wrapper, so a mention is not judged by what
surrounds it. An occurrence that is deliberate (a test leg, a message) is
recorded in the allowlist with a reason. The file is generic and is kept
byte-identical across the repositories that carry it.

Files read, anywhere under the root (target/, node_modules/ and .git/ are
skipped), by category:

    cargo-config   .cargo/config, .cargo/config.toml
    toml           every other *.toml (Cargo.toml, pyproject.toml, Makefile.toml,
                   Cross.toml, ...); every TOML file must parse
    build-rs       build.rs
    rust           every other *.rs
    yaml           *.yml, *.yaml (workflows, actions, other CI and compose files)
    json           *.json (devcontainer, editor and tool settings)
    script         *.sh, *.bash, *.zsh, *.py, *.ps1, *.psm1, *.nix, Makefile,
                   makefile, GNUmakefile, *.mk, justfile, Justfile, Dockerfile*,
                   *.dockerfile, .envrc, .env, .env.*, *.env, *.bat, *.cmd
    wrapper        a file named as RUSTC_WRAPPER / RUSTC_WORKSPACE_WRAPPER / RUSTC
                   (env) or rustc-wrapper / rustc-workspace-wrapper / rustc (cargo
                   config) that exists in the tree and is not read already

Comments are blanked first: `#` comments in TOML, YAML and the script kinds
(a `#` that starts the line or follows whitespace, outside quotes), `//` and
`/* */` in Rust and JSON, `rem` / `::` lines in batch files. The value of a
YAML `name:` key is blanked too (a step or job label configures nothing).
String contents are kept, since flags live in strings.

Rules, one per occurrence:

    target-cpu-native      target-cpu=native (rustc accepts target_cpu too)
    target-cpu-unresolved  target-cpu= followed by a variable ($X, ${X}, $(X),
                           ${{ ... }}, %X%): the value cannot be checked here
    target-feature-unresolved  the same for target-feature=
    llvm-fast-math         LLVM options that relax IEEE 754: enable-unsafe-fp-math,
                           enable-no-nans-fp-math, enable-no-infs-fp-math,
                           enable-no-signed-zeros-fp-math, fp-contract=fast,
                           denormal-fp-math (any value)
    c-fast-math            C / C++ flags of the same kind: -ffast-math, -Ofast,
                           -funsafe-math-optimizations, -fassociative-math,
                           -freciprocal-math, -ffinite-math-only, -fno-signed-zeros,
                           -ffp-contract=fast, -ffp-model=fast|aggressive,
                           -fdenormal-fp-math, -mdaz-ftz, /fp:fast, -fp-model fast,
                           -march=native, -mcpu=native, -mfma (a C compiler
                           contracts a * b + c into fma by default once fma is
                           available), -mno-sse, -mno-sse2, -mfpmath=387
    x87-float              a Rust target-feature list that disables sse or sse2
                           (x86 then computes in x87 extended precision)
    fast-intrinsic         (*.rs under a src/ directory) fadd_fast / fsub_fast /
                           fmul_fast / fdiv_fast / frem_fast, the *_algebraic
                           intrinsics, the algebraic_add / _sub / _mul / _div / _rem
                           methods, #![feature(core_intrinsics)] and
                           #![feature(float_algebraic)]
    unparseable            a TOML file that does not parse
    unreadable             a file read here that is not UTF-8

Not findings, by policy: a Rust target-feature that adds or removes
instructions other than sse / sse2 (+avx2, +fma, +dotprod, -avx2, -fma,
-sse4.1) and #[target_feature(enable = ...)] in source. Rust never contracts
a * b + c into an fma, mul_add is fused whether or not fma is enabled, and the
x86_64 float ABI is SSE2 throughout, so these select different instructions
for the same IEEE 754 results. The tests pin this policy.

Allowlist (per repository, optional): scripts/build_config_allowlist.toml
(this file itself is not scanned)

    [[allow]]          a deliberate occurrence: it covers the findings in `path`
    path = "..."       whose comment-blanked line, stripped of surrounding
    line = "..."       whitespace, is exactly `line`; `count` (default 1) must
    count = 1          equal the number of lines it covers, so a new
    reason = "..."     occurrence in the same file is a finding
    [[exclude]]        a file that is not read (this script and its tests, test
    path = "..."       data, a document that quotes flags); `path` must be a
    reason = "..."     file this script would read

An entry without a non-empty reason, with an unknown key or a wrong type, an
allow whose count differs from what it covers (zero included: text that only
matches a comment covers nothing), an exclude whose path is not a file this
script reads, and an allowlist that is not valid TOML are problems (rule
`allowlist`). Each entry is printed with what it covered.

YAML is not parsed here (the standard library has no YAML parser); the
workflow files are parsed by the repository's actionlint job.

Output: `path:line: rule: text` per problem, the allowed occurrences and the
entries, then the counts per category. Exit 0 clean, 1 problems, 2 when
nothing was read or the root has no Cargo.toml.

    python3 scripts/build_config_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

SKIP_DIRS = {"target", "node_modules", ".git"}
ALLOWLIST = "scripts/build_config_allowlist.toml"
CATEGORIES = ["cargo-config", "toml", "build-rs", "rust", "yaml", "json", "script", "wrapper"]

VAR = r"[\"']?(?:\$|%)"
RULES = [
    ("target-cpu-native", re.compile(r"target[-_]cpu\s*=\s*[\"']?native\b")),
    ("target-cpu-unresolved", re.compile(r"target[-_]cpu\s*=\s*" + VAR)),
    ("target-feature-unresolved", re.compile(r"target[-_]feature\s*=\s*" + VAR)),
    ("llvm-fast-math", re.compile(
        r"enable-(?:unsafe|no-nans|no-infs|no-signed-zeros)-fp-math"
        r"|(?<![\w-])-?fp-contract=fast|(?<![\w-])-?denormal-fp-math")),
    ("c-fast-math", re.compile(
        r"(?<![\w-])-(?:ffast-math|Ofast\b|funsafe-math-optimizations|fassociative-math"
        r"|freciprocal-math|ffinite-math-only|fno-signed-zeros|ffp-contract=fast"
        r"|ffp-model=(?:fast|aggressive)\b|fdenormal-fp-math|mdaz-ftz|march=native\b"
        r"|mcpu=native\b|mfma\b|mno-sse2?\b|mfpmath=387\b)"
        r"|/fp:fast|-fp-model[ =]fast")),
    ("x87-float", re.compile(r"target[-_]feature\s*=\s*[\"']?[^\s\"']*-sse2?(?![0-9.\w])")),
]
SOURCE_RULES = [
    ("fast-intrinsic", re.compile(
        r"\b(?:f(?:add|sub|mul|div|rem)_(?:fast|algebraic)|algebraic_(?:add|sub|mul|div|rem))\b"
        r"|feature\s*\(\s*(?:core_intrinsics|float_algebraic)\b")),
]
YAML_NAME = re.compile(r"^(\s*(?:-\s+)?)name\s*:")
WRAPPER_REF = re.compile(
    r"\b(?:RUSTC_WORKSPACE_WRAPPER|RUSTC_WRAPPER|RUSTC|rustc-workspace-wrapper|rustc-wrapper|rustc)"
    r"\s*[:=]\s*[\"']?([^\s\"'$]+)")
SCRIPT_SUFFIXES = {".sh", ".bash", ".zsh", ".py", ".ps1", ".psm1", ".nix", ".mk",
                   ".dockerfile", ".env", ".bat", ".cmd"}
SCRIPT_NAMES = {"Makefile", "makefile", "GNUmakefile", "justfile", "Justfile", ".envrc", ".env"}


def category(rel: str) -> str | None:
    """The category of a file this guard reads, or None to skip it."""
    p = PurePosixPath(rel)
    name, parts, suffix = p.name, p.parts, p.suffix.lower()
    if rel == ALLOWLIST:
        return None
    if name in ("config", "config.toml") and len(parts) >= 2 and parts[-2] == ".cargo":
        return "cargo-config"
    if suffix == ".toml":
        return "toml"
    if name == "build.rs":
        return "build-rs"
    if suffix == ".rs":
        return "rust"
    if suffix in (".yml", ".yaml"):
        return "yaml"
    if suffix == ".json":
        return "json"
    if name in SCRIPT_NAMES or suffix in SCRIPT_SUFFIXES or name.startswith(".env.") \
            or name.startswith("Dockerfile"):
        return "script"
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


def blank_slash_comments(text: str) -> str:
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


def code_lines(rel: str, text: str, cat: str) -> list[str]:
    if cat in ("build-rs", "rust", "json"):
        return blank_slash_comments(text).split("\n")
    lines = text.split("\n")
    if PurePosixPath(rel).suffix.lower() in (".bat", ".cmd"):
        return ["" if re.match(r"\s*(?:@?rem\b|::)", l, re.I) else l for l in lines]
    lines = [blank_hash_comments(l) for l in lines]
    if cat == "yaml":
        lines = [m.group(1) if (m := YAML_NAME.match(l)) else l for l in lines]
    return lines


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
    spec = {"allow": (("path", "line", "reason"), ("count",)), "exclude": (("path", "reason"), ())}
    for table, (required, optional) in spec.items():
        rows = data.get(table, [])
        if not isinstance(rows, list):
            problems.append(f"{ALLOWLIST}:0: allowlist: [[{table}]] must be an array of tables")
            rows = []
        good = []
        for i, row in enumerate(rows):
            why = []
            bad = [k for k in required if not isinstance(row.get(k), str) or not row[k].strip()]
            if bad:
                why.append(f"empty or missing {', '.join(bad)}")
            extra = sorted(set(row) - set(required) - set(optional))
            if extra:
                why.append(f"unknown key {', '.join(extra)}")
            if "count" in row and (not isinstance(row["count"], int) or isinstance(row["count"], bool)
                                   or row["count"] < 1):
                why.append("count must be a positive integer")
            if why:
                problems.append(f"{ALLOWLIST}:0: allowlist: {table} entry {i + 1}: {'; '.join(why)}")
                continue
            good.append(row)
        entries[table] = good
    return entries["allow"], entries["exclude"], problems


def wrapper_targets(rel: str, text: str, root: Path) -> set[str]:
    """Files in the tree named as a rustc wrapper in this file."""
    found = set()
    base = PurePosixPath(rel).parent
    bases = [PurePosixPath(""), base]
    if base.name == ".cargo":
        bases.append(base.parent)
    for m in WRAPPER_REF.finditer(text):
        ref = m.group(1).strip(",;)")
        for b in bases:
            cand = (b / ref).as_posix().lstrip("./") if not ref.startswith("/") else None
            if cand and (root / cand).is_file():
                found.add(cand)
    return found


def scan(root: Path):
    allow, exclude, problems = load_allowlist(root)
    counts = {c: 0 for c in CATEGORIES}
    root_manifest = (root / "Cargo.toml").is_file()
    excluded = {e["path"]: 0 for e in exclude}
    findings = []
    queue = [(rel, path, category(rel)) for rel, path in walk(root)]
    seen = {rel for rel, _, cat in queue if cat}
    while queue:
        rel, path, cat = queue.pop(0)
        if cat is None:
            continue
        if rel in excluded:
            excluded[rel] += 1
            continue
        counts[cat] += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            problems.append(f"{rel}:0: unreadable: {e.__class__.__name__}")
            continue
        if PurePosixPath(rel).suffix == ".toml":
            if tomllib is None:
                problems.append(f"{rel}:0: unparseable: Python 3.11+ (tomllib) is needed")
            else:
                try:
                    tomllib.loads(text)
                except tomllib.TOMLDecodeError as e:
                    problems.append(f"{rel}:0: unparseable: {e}")
        lines = code_lines(rel, text, cat)
        for w in sorted(wrapper_targets(rel, "\n".join(lines), root) - seen):
            seen.add(w)
            queue.append((w, root / w, "wrapper"))
        rules = list(RULES)
        if cat == "rust" and "src" in PurePosixPath(rel).parts[:-1]:
            rules += SOURCE_RULES
        for lineno, line in enumerate(lines, start=1):
            for rule, pattern in rules:
                for m in pattern.finditer(line):
                    findings.append((rel, lineno, rule, m.group(0), line.strip()))
    covered = [0] * len(allow)
    allowed = []
    lines_covered = [set() for _ in allow]
    for rel, lineno, rule, hit, line in findings:
        cover = [i for i, a in enumerate(allow) if a["path"] == rel and a["line"].strip() == line]
        msg = f"{rel}:{lineno}: {rule}: {hit}"
        if cover:
            lines_covered[cover[0]].add(lineno)
            allowed.append(f"allowed {msg} (entry {cover[0] + 1})")
        else:
            problems.append(msg)
    entries = []
    for i, a in enumerate(allow):
        want = a.get("count", 1)
        got = len(lines_covered[i])
        entries.append(f"allow entry {i + 1}: {a['path']}: {got} of {want} lines ({a['reason']})")
        if got != want:
            problems.append(f"{ALLOWLIST}:0: allowlist: allow entry {i + 1} ({a['path']}: {a['line']!r}) "
                            f"covers {got} lines, count is {want}")
    for e in exclude:
        entries.append(f"exclude: {e['path']}: {'not read' if excluded[e['path']] else 'no such file'} ({e['reason']})")
        if not excluded[e["path"]]:
            problems.append(f"{ALLOWLIST}:0: allowlist: exclude {e['path']!r} is not a file this script reads")
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
