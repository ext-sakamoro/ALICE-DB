#!/usr/bin/env python3
"""Move the commit-SHA pins of the workflows to the current release.

Each pin in .github/workflows/ reads `uses: owner/repo@<40-hex sha> # <ref>`.

- `# vX.Y.Z` (a release tag): follows the major tag `vX`. The new pin is the
  commit `vX` points to now, and the comment becomes the full release tag
  that points to the same commit (`vX.Y.Z` or newer). A new major version is
  not taken; that is a change to review by hand.
- `# <name> (branch, ...)`: follows the branch. The new pin is the branch head;
  the comment is kept. A branch whose head no longer contains the old pin was
  rewritten upstream: the line is left as is and reported (pin such an action
  to an immutable release tag instead; branch pins are only for branches that
  only ever move forward).

Prints a unified diff of what would change. `--write` rewrites the files.
Then it runs the full online check (`workflow_pin_check.py --online`, every
pin against GitHub) on the files as they are now. Nothing is committed: review
the diff and commit as usual. (No bot: commits from a bot account show up as
repository contributors.)

Usage: python scripts/workflow_pin_update.py [--write]
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")

PIN_RE = re.compile(
    r"^(?P<lead>\s*(?:-\s+)?uses:\s*)(?P<action>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^@\s]+)?)"
    r"@(?P<sha>[0-9a-f]{40})\s*#\s*(?P<ref>\S+)(?P<rest>.*)$"
)
RELEASE_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def gh_json(path: str):
    """`gh api <path>`; a list endpoint is read across all pages"""
    cmd = ["gh", "api", path]
    paged = "?per_page=" in path
    if paged:
        cmd[2:2] = ["--paginate", "--slurp"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"gh api {path} failed: {r.stderr.strip()}")
    data = json.loads(r.stdout)
    return [item for page in data for item in page] if paged else data


def release_tag_for(repo: str, sha: str, major: str, api) -> str | None:
    """The highest `vX.Y.Z` tag of `major` that points to `sha`."""
    best = None
    for tag in api(f"repos/{repo}/tags?per_page=100"):
        m = RELEASE_RE.match(tag["name"])
        if m and m.group(1) == major and tag["commit"]["sha"] == sha:
            key = tuple(int(x) for x in m.groups())
            if best is None or key > best[0]:
                best = (key, tag["name"])
    return best[1] if best else None


def update_text(text: str, api) -> tuple[str, list[str]]:
    """(new text, problems)"""
    problems, out, cache, reported = [], [], {}, set()
    for line in text.split("\n"):
        m = PIN_RE.match(line)
        if not m:
            out.append(line)
            continue
        action, ref, rest = m.group("action"), m.group("ref"), m.group("rest")
        repo = "/".join(action.split("/")[:2])
        branch = "(branch" in rest
        key = (repo, ref, branch, m.group("sha") if branch else "")
        if key not in cache:
            if branch:
                old = m.group("sha")
                cmp = api(f"repos/{repo}/compare/{old}...{ref}")
                if cmp.get("status") not in ("identical", "ahead"):
                    cache[key] = None
                    problems.append(f"{repo}@{old} # {ref}: the branch no longer contains the pin "
                                    f"({cmp.get('status')}), it was rewritten; pin a release tag instead")
                else:
                    cache[key] = (api(f"repos/{repo}/commits/{ref}")["sha"], ref)
            else:
                rm = RELEASE_RE.match(ref)
                if not rm:
                    cache[key] = None
                else:
                    major = rm.group(1)
                    head = api(f"repos/{repo}/commits/v{major}")["sha"]
                    tag = release_tag_for(repo, head, major, api)
                    cache[key] = (head, tag) if tag else None
                    if not tag:
                        problems.append(f"{repo}: v{major} points to {head}, which no v{major}.x.y tag names")
        new = cache[key]
        if new is None:
            if not RELEASE_RE.match(ref) and not branch and key not in reported:
                reported.add(key)
                problems.append(f"{repo}@{m.group('sha')} # {ref}: neither a vX.Y.Z tag nor a branch, left as is")
            out.append(line)
            continue
        sha, new_ref = new
        out.append(f"{m.group('lead')}{action}@{sha} # {new_ref}{rest}")
    return "\n".join(out), problems


def main(argv: list[str]) -> int:
    write = "--write" in argv[1:]
    names = sorted(n for n in os.listdir(WORKFLOWS) if n.endswith((".yml", ".yaml")))
    if not names:
        print("error: no workflow file")
        return 1
    changed = 0
    problems = []
    for n in names:
        path = os.path.join(WORKFLOWS, n)
        with open(path, encoding="utf-8") as f:
            old = f.read()
        new, p = update_text(old, gh_json)
        problems += [f"{n}: {x}" for x in p]
        if new != old:
            changed += 1
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(True), new.splitlines(True), f"a/.github/workflows/{n}", f"b/.github/workflows/{n}"))
            if write:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(new)
    for p in problems:
        print(f"note: {p}")
    print(f"{changed} of {len(names)} workflows {'rewritten' if write else 'would change'}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import workflow_pin_check as wpc

    errors, compared = wpc.check_online(WORKFLOWS, actions_dir=os.path.join(os.path.dirname(WORKFLOWS), "actions"))
    for e in errors:
        print(f"error: {e}")
    print(f"online (full): {compared} pins checked, {len(errors)} not matching")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
