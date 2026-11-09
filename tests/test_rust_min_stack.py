# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Importing sqd_gqe must enlarge Rust's thread stack before qiskit spawns a thread.

Without it, level-2/3 transpiles segfault on ROQUO's aarch64 nodes (see sqd_gqe/__init__.py).
Checked in a fresh interpreter, since this test process may already have the variable.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(env):
    code = "import sqd_gqe, os; print(os.environ['RUST_MIN_STACK'])"
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_import_sets_a_large_rust_stack():
    env = {k: v for k, v in os.environ.items() if k != "RUST_MIN_STACK"}
    assert int(_run(env)) >= 64 << 20


def test_an_explicit_setting_wins():
    env = dict(os.environ, RUST_MIN_STACK=str(128 << 20))
    assert int(_run(env)) == 128 << 20
