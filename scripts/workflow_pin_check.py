#!/usr/bin/env python3
"""Check how the workflows reference actions and what their token may do.

For every file in .github/workflows/:

- the workflow sets `permissions:` at the top level (the default token of a
  repository can write; each workflow states what it needs),
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
that names another commit than the pin is an error. 0 compared is an error.

Usage: python scripts/workflow_pin_check.py [--online] [workflow directory]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")

USES_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)\s*(#.*)?$")
PINNED_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^@\s]+)?@[0-9a-f]{40}$")
JOB_PERMISSIONS_RE = re.compile(r"^(\s+)permissions:\s*$")
CHECKOUT_RE = re.compile(r"^(\s*)(-\s+)?uses:\s*actions/checkout@")
TRIGGER_RE = re.compile(r"^\s*(pull_request_target|workflow_run)\s*:")


def check_text(name: str, text: str) -> tuple[list[str], int]:
    """(errors, number of `uses:` compared) for one workflow file."""
    errors = []
    lines = text.splitlines()
    if not any(re.match(r"^permissions:", line) for line in lines):
        errors.append(f"{name}: no top-level `permissions:`")
    compared = 0
    for no, line in enumerate(lines, 1):
        m = USES_RE.match(line)
        if not m:
            continue
        ref, comment = m.group(1), m.group(2)
        if ref.startswith("./"):
            continue
        compared += 1
        if not PINNED_RE.match(ref):
            errors.append(f"{name}:{no}: `{ref}` is not pinned to a full 40-hex commit SHA")
        elif not comment or not comment.lstrip("#").strip():
            errors.append(f"{name}:{no}: `{ref}` has no `# <tag>` comment")
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
    for no, line in enumerate(lines, 1):
        m = TRIGGER_RE.match(line)
        if m:
            errors.append(f"{name}:{no}: `{m.group(1)}` trigger")
    for no, line in enumerate(lines, 1):
        m = JOB_PERMISSIONS_RE.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        has_comment = False
        for later in lines[no:]:
            if later.strip() and len(later) - len(later.lstrip()) <= indent:
                break
            if later.strip().startswith("#"):
                has_comment = True
            elif later.strip().endswith(": write") and not has_comment:
                errors.append(f"{name}:{no}: job permissions grant `{later.strip()}` without a comment saying why")
    return errors, compared


def pins(text: str) -> list[tuple[int, str, str, str, bool]]:
    """(line, owner/repo, sha, commented ref, ref is a branch) of each pin"""
    out = []
    for no, line in enumerate(text.splitlines(), 1):
        m = USES_RE.match(line)
        if not m or not m.group(2) or not PINNED_RE.match(m.group(1)):
            continue
        action, sha = m.group(1).split("@")
        repo = "/".join(action.split("/")[:2])
        comment = m.group(2).lstrip("#").strip()
        out.append((no, repo, sha, comment.split()[0], "(branch" in comment))
    return out


def gh_json(path: str):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def check_online(directory: str, api=gh_json) -> tuple[list[str], int]:
    """(errors, pins compared against GitHub)"""
    errors, compared, cache = [], 0, {}
    names = sorted(n for n in os.listdir(directory) if n.endswith((".yml", ".yaml")))
    for n in names:
        with open(os.path.join(directory, n), encoding="utf-8") as f:
            text = f.read()
        for no, repo, sha, ref, branch in pins(text):
            key = (repo, sha, ref, branch)
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
    if compared == 0:
        errors.append(f"{directory}: no pin was compared online")
    return errors, compared


def check_dir(directory: str) -> tuple[list[str], int, int]:
    """(errors, workflows compared, `uses:` compared)"""
    names = sorted(n for n in os.listdir(directory) if n.endswith((".yml", ".yaml")))
    errors, uses = [], 0
    for n in names:
        with open(os.path.join(directory, n), encoding="utf-8") as f:
            e, c = check_text(n, f.read())
        errors += e
        uses += c
    if not names:
        errors.append(f"{directory}: no workflow file was compared")
    elif uses == 0:
        errors.append(f"{directory}: no `uses:` was compared")
    return errors, len(names), uses


def main(argv: list[str]) -> int:
    args = argv[1:]
    online = "--online" in args
    args = [a for a in args if a != "--online"]
    directory = args[0] if args else WORKFLOWS
    errors, files, uses = check_dir(directory)
    if online and not errors:
        online_errors, compared = check_online(directory)
        errors += online_errors
        if not errors:
            print(f"online: {compared} pins match the commented tag or branch on GitHub")
    for e in errors:
        print(f"error: {e}")
    if errors:
        return 1
    print(f"workflow pins OK: {files} workflows, {uses} `uses:` pinned to a commit SHA")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
