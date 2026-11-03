#!/usr/bin/env python3
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check (and with --fix, add) the IBM copyright header of every Python file.

    python tools/verify_headers.py [--fix] [paths ...]

The header must appear in the first lines of the file, after an optional shebang. A shebang line stays first.
"""
import argparse
import sys
from pathlib import Path

HEADER = """# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "__pycache__", "node_modules"}


def python_files(roots):
    for root in roots:
        p = Path(root)
        if p.is_file():
            yield p
            continue
        for f in p.rglob("*.py"):
            if not (set(f.parts) & SKIP_DIRS):
                yield f


def has_header(text: str) -> bool:
    lines = text.splitlines()
    start = 1 if lines and lines[0].startswith("#!") else 0
    return "\n".join(lines[start:start + len(HEADER.splitlines())]) == HEADER.rstrip("\n")


def add_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        return lines[0] + HEADER + "\n" + "".join(lines[1:])
    return HEADER + "\n" + text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("paths", nargs="*", default=["."])
    a = ap.parse_args()
    bad = 0
    for f in sorted(python_files(a.paths)):
        text = f.read_text(encoding="utf8")
        if has_header(text):
            continue
        if a.fix:
            f.write_text(add_header(text), encoding="utf8")
        else:
            print(f"missing or wrong header: {f}")
            bad += 1
    if bad:
        print(f"{bad} file(s) without the IBM header; run with --fix")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
