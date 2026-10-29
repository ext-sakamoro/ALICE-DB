#!/usr/bin/env python3
"""Check how the workflows reference actions and what their token may do.

For every file in .github/workflows/ (and every local action under
.github/actions/**/action.yml, for its `uses:` and checkouts):

- the workflow sets `permissions:` at the top level (the default token of a
  repository can write; each workflow states what it needs), and no
  `permissions:` (top-level or job) grants `write` inline (`write-all`,
  `{contents: write}`) or in block form without a comment saying why,
- every line with a `uses:` key is recognised and compared (a value with
  spaces, quotes or an expression is not skipped: it fails as unpinned),
- every `uses:` of a remote action names a full 40-hex commit SHA, followed by
  a `# <tag or branch>` comment (a tag or branch can be moved to other code;
  the comment keeps the reference readable),
- a job-level `permissions:` block that grants `write` carries a comment line
  saying why,
- every `actions/checkout` sets `persist-credentials: false` (no job pushes,
  so the token is not left in `.git/config` for later steps),
- no workflow is triggered by `pull_request_target` or `workflow_run` (both
  run with the base repository's token on input from elsewhere),
- no workflow file, or no `uses:` at all, is an error (the check would compare
  nothing).

With `--online` it also asks GitHub (`gh api`, needs `GH_TOKEN` in CI) that
each pinned SHA is the commented ref: for a tag (`# v5.1.0`) the tag must
resolve to that exact commit, for a branch (`# stable (branch, ...)`) the
commit must be on the branch (identical to or behind its head). A comment
that names another commit than the pin is an error.

`--online` alone checks every pin: upstream state, so it belongs in the weekly
audit and in the update script, not in a per-push gate (a branch an action
rewrites would turn the gate red with no change here). Per push, `--online
--base <ref>` checks only the pins that are new or changed since `<ref>`
(unchanged pins were checked when they came in); no changed pin is a pass,
printed as such. `--base auto` takes the commit from scripts/ci_base.py (shared
with the other ALICE crates): on a push to main the commit main stood at before
the push (`github.event.before`), elsewhere the merge base with origin/main. A
base it cannot name, or one that is the commit under test, stops the check
(exit 2); it never falls back to another commit.

Usage: python scripts/workflow_pin_check.py [--online [--base <ref>|auto]] [workflow directory]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import NoReturn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_base  # noqa: E402  (scripts/ci_base.py, a shared file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")
ACTIONS = os.path.join(ROOT, ".github", "actions")

# any line that mentions a `uses` key, outside a comment; every one of them has
# to be recognised below, or the check fails (nothing is skipped silently)
USES_ANY_RE = re.compile(r"""(^|[\s{,'"-])uses['"]?\s*:""")
USES_RE = re.compile(r"""^\s*(?:-\s+)?['"]?uses['"]?\s*:\s*(?P<rest>.*)$""")
PINNED_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^@\s]+)?@[0-9a-f]{40}$")
PERMISSIONS_RE = re.compile(r"^(?P<indent>\s*)permissions\s*:\s*(?P<value>.*)$")
CHECKOUT_RE = re.compile(r"^(\s*)(-\s+)?uses:\s*actions/checkout@")
TRIGGER_RE = re.compile(r"^\s*(pull_request_target|workflow_run)\s*:")


def code_part(line: str) -> str:
    """The line without a trailing ` # comment` (a `#` inside quotes is kept)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def split_uses(rest: str) -> tuple[str, str]:
    """(value, comment) of what follows `uses:`; quotes around the value removed."""
    code = code_part(rest)
    comment = rest[len(code):].strip()
    value = code.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value, comment


def check_uses(name: str, lines: list[str]) -> tuple[list[str], int]:
    errors, compared, total = [], 0, 0
    for no, line in enumerate(lines, 1):
        if not USES_ANY_RE.search(code_part(line)):
            continue
        total += 1
        m = USES_RE.match(line)
        if not m:
            errors.append(f"{name}:{no}: a `uses:` this check does not recognise: `{line.strip()}`")
            continue
        compared += 1
        value, comment = split_uses(m.group("rest"))
        if value.startswith("./"):
            continue
        if not PINNED_RE.match(value):
            errors.append(f"{name}:{no}: `{value}` is not pinned to a full 40-hex commit SHA")
        elif not comment.lstrip("#").strip():
            errors.append(f"{name}:{no}: `{value}` has no `# <tag>` comment")
    if compared != total:
        errors.append(f"{name}: {total} `uses:` lines, {compared} compared")
    return errors, compared


def check_permissions(name: str, lines: list[str], workflow: bool) -> list[str]:
    errors = []
    top = False
    for no, line in enumerate(lines, 1):
        m = PERMISSIONS_RE.match(code_part(line).rstrip())
        if not m:
            continue
        indent, value = len(m.group("indent")), m.group("value").strip()
        if indent == 0:
            top = True
        where = "top-level" if indent == 0 else "job"
        if value:
            # inline forms: `read-all` / `{}` / `{contents: read}` are fine;
            # anything that writes has to be the block form with a reason
            if "write" in value:
                errors.append(f"{name}:{no}: {where} `permissions: {value}` grants write inline "
                              "(use the block form with a comment saying why)")
            continue
        has_comment = False
        for later in lines[no:]:
            if later.strip() and len(later) - len(later.lstrip()) <= indent:
                break
            body = later.strip()
            if body.startswith("#"):
                has_comment = True
            elif re.search(r":\s*write\b", code_part(body)) and not has_comment:
                errors.append(f"{name}:{no}: {where} permissions grant `{code_part(body).strip()}` "
                              "without a comment saying why")
    if workflow and not top:
        errors.append(f"{name}: no top-level `permissions:`")
    return errors


def check_text(name: str, text: str, workflow: bool = True) -> tuple[list[str], int]:
    """(errors, number of `uses:` compared) for one workflow (or, with
    `workflow=False`, one local action.yml: no permissions or triggers)."""
    lines = text.splitlines()
    errors = check_permissions(name, lines, workflow) if workflow else []
    e, compared = check_uses(name, lines)
    errors += e
    for no, line in enumerate(lines, 1):
        m = CHECKOUT_RE.match(line)
        if not m:
            continue
        key = len(m.group(1)) + len(m.group(2) or "")
        step = []
        for later in lines[no:]:
            if later.strip() and (len(later) - len(later.lstrip()) < key or later.lstrip().startswith("- ")):
                break
            step.append(later.strip())
        if "persist-credentials: false" not in step:
            errors.append(f"{name}:{no}: actions/checkout without `persist-credentials: false`")
    if workflow:
        for no, line in enumerate(lines, 1):
            m = TRIGGER_RE.match(line)
            if m:
                errors.append(f"{name}:{no}: `{m.group(1)}` trigger")
    return errors, compared


def pins(text: str) -> list[tuple[int, str, str, str, bool]]:
    """(line, owner/repo, sha, commented ref, ref is a branch) of each pin"""
    out = []
    for no, line in enumerate(text.splitlines(), 1):
        m = USES_RE.match(line)
        if not m:
            continue
        value, comment = split_uses(m.group("rest"))
        comment = comment.lstrip("#").strip()
        if not comment or not PINNED_RE.match(value):
            continue
        action, sha = value.split("@")
        repo = "/".join(action.split("/")[:2])
        out.append((no, repo, sha, comment.split()[0], "(branch" in comment))
    return out


def files(directory: str, actions_dir: str | None) -> list[tuple[str, str, bool]]:
    """(display name, path, is a workflow) of every file the check reads"""
    out = [(n, os.path.join(directory, n), True)
           for n in sorted(os.listdir(directory)) if n.endswith((".yml", ".yaml"))]
    if actions_dir and os.path.isdir(actions_dir):
        for base, _, names in sorted(os.walk(actions_dir)):
            for n in sorted(names):
                if n in ("action.yml", "action.yaml"):
                    path = os.path.join(base, n)
                    rel = os.path.relpath(path, os.path.dirname(actions_dir)).replace(os.sep, "/")
                    out.append((rel, path, False))
    return out


def gh_json(path: str):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def check_online(directory: str, api=gh_json, actions_dir: str | None = None,
                 known: set | None = None) -> tuple[list[str], int]:
    """(errors, pins compared against GitHub); pins whose (repo, sha, ref,
    branch) is in `known` (present at the base) are not asked again."""
    errors, compared, cache = [], 0, {}
    for n, path, _ in files(directory, actions_dir):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for no, repo, sha, ref, branch in pins(text):
            key = (repo, sha, ref, branch)
            if known is not None and key in known:
                continue
            if key not in cache:
                if branch:
                    cmp = api(f"repos/{repo}/compare/{sha}...{ref}")
                    ok = cmp is not None and cmp.get("status") in ("identical", "ahead")
                    why = "is not on branch" if cmp is not None else "could not be compared with branch"
                else:
                    c = api(f"repos/{repo}/commits/{ref}")
                    ok = c is not None and c.get("sha") == sha
                    why = f"is not what tag {ref} resolves to ({c.get('sha') if c else 'not found'})"
                    why = why if c is not None else f"is pinned, but tag {ref} was not found"
                cache[key] = None if ok else why
            compared += 1
            if cache[key]:
                suffix = f" `{ref}`" if branch else ""
                errors.append(f"{n}:{no}: {repo}@{sha} {cache[key]}{suffix}")
    if compared == 0 and known is None:
        errors.append(f"{directory}: no pin was compared online")
    return errors, compared


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def refuse(message: str) -> NoReturn:
    """Stop the check with exit 2 (no judgement: the comparison could not be set up), as
    scripts/ci_base.py does; exit 1 stays for pins that were compared and are wrong"""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def resolve_base(base: str) -> str:
    """A commit to compare with: `base` itself, or for `auto` the one scripts/ci_base.py names from the
    GitHub event (shared with the other ALICE crates): on a push to main the commit main stood at before
    the push, elsewhere the merge-base with origin/main. A base it cannot name, or one that is the commit
    under test, stops the check (exit 2) instead of falling back to another commit."""
    if base != "auto":
        if base and set(base) != {"0"} and git("cat-file", "-e", f"{base}^{{commit}}").returncode == 0:
            return git("rev-parse", base).stdout.strip()
        refuse(f"no base commit to compare with ({base}; fetch the history: fetch-depth 0)")
    before = ""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path, encoding="utf-8") as f:
            before = json.load(f).get("before") or ""
    event, ref = os.environ.get("GITHUB_EVENT_NAME", ""), os.environ.get("GITHUB_REF", "")
    cwd = os.getcwd()
    os.chdir(ROOT)  # ci_base reads git from the working directory
    try:
        commit, rule = ci_base.ci_base(event, ref, before)
    except ValueError as e:
        refuse(f"no base commit to compare with: {e}")
    finally:
        os.chdir(cwd)
    print(f"workflow_pin_check: {ci_base.describe(commit, rule, event, ref)}")
    if commit == git("rev-parse", "HEAD").stdout.strip():
        refuse(f"the base {commit} is the commit under test (nothing would be compared)")
    return commit


def pins_at(commit: str, directory: str, actions_dir: str | None) -> set:
    """(repo, sha, ref, branch) of every pin in the workflow files at `commit`"""
    keys = set()
    roots = [os.path.relpath(directory, ROOT)]
    if actions_dir:
        roots.append(os.path.relpath(actions_dir, ROOT))
    listed = git("ls-tree", "-r", "--name-only", commit, "--", *[r.replace(os.sep, "/") for r in roots])
    for rel in listed.stdout.splitlines():
        name = rel.rsplit("/", 1)[-1]
        if not name.endswith((".yml", ".yaml")):
            continue
        text = git("show", f"{commit}:{rel}").stdout
        keys |= {(repo, sha, ref, branch) for _, repo, sha, ref, branch in pins(text)}
    return keys


def check_dir(directory: str, actions_dir: str | None = None) -> tuple[list[str], int, int]:
    """(errors, files compared, `uses:` compared)"""
    found = files(directory, actions_dir)
    errors, uses = [], 0
    for n, path, workflow in found:
        with open(path, encoding="utf-8") as f:
            e, c = check_text(n, f.read(), workflow)
        errors += e
        uses += c
    if not any(w for _, _, w in found):
        errors.append(f"{directory}: no workflow file was compared")
    elif uses == 0:
        errors.append(f"{directory}: no `uses:` was compared")
    return errors, len(found), uses


def main(argv: list[str]) -> int:
    args = argv[1:]
    online = "--online" in args
    args = [a for a in args if a != "--online"]
    base = None
    if "--base" in args:
        i = args.index("--base")
        base = args[i + 1]
        del args[i:i + 2]
    directory = args[0] if args else WORKFLOWS
    actions_dir = os.path.join(os.path.dirname(directory), "actions")
    errors, n_files, uses = check_dir(directory, actions_dir)
    if online and not errors:
        known = None
        if base is not None:
            commit = resolve_base(base)
            known = pins_at(commit, directory, actions_dir)
        online_errors, compared = check_online(directory, actions_dir=actions_dir, known=known)
        errors += online_errors
        if not errors:
            scope = "" if known is None else f" (new or changed since {commit[:12]})"
            if known is not None and compared == 0:
                print(f"online: no changed pins since {commit[:12]}")
            else:
                print(f"online: {compared} pins{scope} match the commented tag or branch on GitHub")
    for e in errors:
        print(f"error: {e}")
    if errors:
        return 1
    print(f"workflow pins OK: {n_files} files, {uses} `uses:` pinned to a commit SHA")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
