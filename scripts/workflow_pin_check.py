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
- no workflow file, or no `uses:` at all, is an error (the check would compare
  nothing).

Usage: python scripts/workflow_pin_check.py [workflow directory]
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")

USES_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)\s*(#.*)?$")
PINNED_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^@\s]+)?@[0-9a-f]{40}$")
JOB_PERMISSIONS_RE = re.compile(r"^(\s+)permissions:\s*$")


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
    directory = argv[1] if len(argv) > 1 else WORKFLOWS
    errors, files, uses = check_dir(directory)
    for e in errors:
        print(f"error: {e}")
    if errors:
        return 1
    print(f"workflow pins OK: {files} workflows, {uses} `uses:` pinned to a commit SHA")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
