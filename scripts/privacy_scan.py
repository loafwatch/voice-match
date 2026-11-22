#!/usr/bin/env python3
"""Privacy scan: blocks personal data from being committed to this public repo.

Checks every file (and optionally a commit message) for:
  1. Private terms - names, places, hostnames, entity names - loaded from
     `.private-terms` (local, git-ignored). The list never leaves the
     developer's machine: it is not committed and not stored on GitHub.
  2. Leak patterns - private IP addresses, .local hostnames, email addresses,
     Home Assistant tokens / JWTs, bearer tokens, private keys.

Usage:
  python3 scripts/privacy_scan.py --staged        scan files staged for commit
  python3 scripts/privacy_scan.py --all           scan all tracked files
  python3 scripts/privacy_scan.py --message FILE  scan a commit message file

Exit code 1 if anything is found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
mode = args[0] if args else "--staged"


def load_terms() -> list[str]:
    path = Path(".private-terms")
    if not path.exists():
        return []
    terms = re.split(r"[\n,]", path.read_text(encoding="utf-8"))
    return [t.strip() for t in terms if t.strip() and not t.strip().startswith("#")]


terms = load_terms()
if not terms:
    msg = "privacy-scan: no private terms loaded (.private-terms file)."
    if Path(".private-terms-required").exists():
        print(msg + " Refusing to pass without them.", file=sys.stderr)
        sys.exit(1)
    print("WARNING: " + msg + " Only generic leak patterns will be checked.", file=sys.stderr)

# (label, regex, secret) - secret matches are never printed (CI logs are public).
# Underscores/dashes count as word breaks so entity IDs like sensor.<name>_x match.
checks: list[tuple[str, re.Pattern[str], bool]] = [
    ("private term", re.compile(rf"(^|[^A-Za-z0-9]){re.escape(t)}(?=$|[^A-Za-z0-9])", re.I), True)
    for t in terms
]
checks += [
    ("private IPv4 address", re.compile(r"\b(10\.\d{1,3}|192\.168|172\.(1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"), False),
    (".local hostname", re.compile(r"\b(?!homeassistant\.local\b)[a-z0-9-]+\.local\b", re.I), False),
    (
        "email address",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@(?!example\.(com|org)\b|users\.noreply\.github\.com\b|anthropic\.com\b)"
            r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        False,
    ),
    ("JWT / Home Assistant token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."), False),
    ("bearer token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"), False),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), False),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), False),
]

SKIP = {"scripts/privacy_scan.py"}
BINARY = re.compile(r"\.(png|jpe?g|gif|webp|ico|woff2?|ttf|otf|mp3|wav|flac|ogg|npy|npz|pt|onnx|zip|gz)$", re.I)


def git(*a: str) -> str:
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout


def files_and_contents() -> list[tuple[str, str]]:
    if mode == "--message":
        text = Path(args[1]).read_text(encoding="utf-8")
        return [("commit message", "\n".join(l for l in text.splitlines() if not l.startswith("#")))]
    if mode == "--all":
        names = git("ls-files").splitlines()
    else:
        names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
    out = []
    for name in names:
        if not name or name in SKIP or BINARY.search(name):
            continue
        try:
            content = git("show", f":{name}") if mode == "--staged" else Path(name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, subprocess.CalledProcessError):
            content = ""
        out.append((name, content))
    return out


problems = 0
for name, content in files_and_contents():
    for i, line in enumerate(content.splitlines(), 1):
        for label, regex, secret in checks:
            if regex.search(line):
                problems += 1
                shown = "(matching text hidden)" if secret else line.strip()[:120]
                print(f"  {name}:{i}  {label}  {shown}", file=sys.stderr)

if problems:
    print(
        f"\nprivacy-scan: {problems} problem(s) found. Keep personal details in local config, not the repo.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"privacy-scan: clean ({len(terms)} private terms, {len(checks) - len(terms)} leak patterns checked).")
