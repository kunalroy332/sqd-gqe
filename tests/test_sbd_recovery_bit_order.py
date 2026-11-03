# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The sbd recovery loop must keep every determinant's orbitals where they are.

Two bit-order mistakes broke ``SbdGpuDiagonalizer.recover_and_diagonalize``:

* ``carryover.bin`` pads each record at the end, and the reader assumed the start.
  At 23 orbitals every carryover determinant came back shifted up one orbital, and
  any with orbital 22 occupied lost an electron. Davidson starts from the first
  determinant in sorted order, which was then a 15-electron string, and 40 of 80
  steps at 46q landed 2.85 Ha above RHF.
* ``recover_configurations`` reads each half of the bitstring matrix in qiskit order
  (column 0 = orbital norb-1), and the loop passed orbital order, so recovery moved
  electrons out of the fullest orbitals.

Driven by a stub binary that reads and writes the same byte layouts as sbd's
``main.cc``, so this runs without a GPU. norb = 10 is deliberately not a multiple of 8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("pyscf")
pytest.importorskip("qiskit_addon_sqd")

NORB = 10
NELEC = (3, 3)
HF = 0b111

# Ported from sbd: DecodeAlphaDetsBits (makedeterminants.h) for the input, and the
# SBD_PREFECT block of apps/chemistry_tpb_selected_basis_diagonalization/main.cc for
# carryover.bin. Every determinant it reads becomes a carryover determinant.
_STUB = '''#!{python}
import numpy as np
norb = {norb}
nbytes = (norb + 7) // 8
raw = open("AlphaDets.bin", "rb").read()
dets = [int.from_bytes(raw[i:i + nbytes], "big") for i in range(0, len(raw), nbytes)]
with open("{log}", "a") as fp:
    fp.write(" ".join(str(d) for d in dets) + "\\n")
open("davidson_energy.txt", "w").write("-1.0\\n")
occ = np.array([[(d >> i) & 1 for i in range(norb)] for d in dets], float).mean(axis=0)
open("occ_a.txt", "w").write("\\n".join(str(float(x)) for x in occ) + "\\n")
with open("carryover.bin", "wb") as fp:
    for d in dets:
        out = bytearray(nbytes)
        for j in range(norb):
            bit = (d >> (norb - 1 - j)) & 1
            out[j // 8] |= bit << (7 - j % 8)
        fp.write(bytes(out))
'''


class _Ham:
    def __init__(self, norb: int):
        self.one_body = np.eye(norb)
        self.two_body = np.zeros((norb,) * 4)
        self.core_energy = 0.0


def _solver(tmp_path: Path):
    from sqd_gqe.sqd.backend_sbd_gpu import SbdGpuDiagonalizer

    log = tmp_path / "calls.txt"
    stub = tmp_path / "fake_diag.py"
    stub.write_text(_STUB.format(python=sys.executable, norb=NORB, log=log))
    stub.chmod(0o755)
    solver = SbdGpuDiagonalizer(_Ham(NORB), NELEC, NORB, binary_path=str(stub),
                                launcher=None, work_dir=str(tmp_path / "run"))
    return solver, log


def _calls(log: Path) -> list[set[int]]:
    return [{int(x) for x in line.split()} for line in log.read_text().splitlines()]


def test_carryover_determinants_come_back_unchanged(tmp_path):
    solver, _ = _solver(tmp_path)
    solver.solver_kwargs["carryover_ratio"] = 0.5
    dets = np.array([0b0000000111, 0b1000000011, 0b0100100001], dtype=np.int64)
    _, _, carry = solver._solve_once(dets)
    assert sorted(carry.tolist()) == sorted(dets.tolist())


def test_every_step_solves_determinants_with_the_right_electron_count(tmp_path):
    """The 46q failure in miniature: a sampled determinant with the top orbital
    occupied goes into the carryover, and must come back with its electrons."""
    solver, log = _solver(tmp_path)
    counts = {
        (HF, HF): 60,
        (0b1000000011, HF): 20,   # HOMO -> top orbital, the one the misread dropped
        (0b0000001101, HF): 20,
    }
    solver.recover_and_diagonalize(counts, max_iterations=3, subspace_dim=50, seed=0)

    calls = _calls(log)
    assert len(calls) == 3
    for dets in calls:
        assert HF in dets
        assert all(bin(d).count("1") == NELEC[0] for d in dets), sorted(dets)
    # carryover is forced into the next step, so nothing a step solved is lost
    assert calls[0] <= calls[1] <= calls[2]


def test_recovery_removes_a_surplus_electron_from_an_empty_orbital(tmp_path):
    """A bit flip put a fourth alpha electron in orbital 9, which the sample says is
    empty. Recovery has to take that one back out, not one of orbitals 0-2."""
    solver, log = _solver(tmp_path)
    counts = {(HF, HF): 99, (HF | 1 << 9, HF): 1}
    solver.recover_and_diagonalize(counts, max_iterations=1, subspace_dim=50, seed=0)

    assert _calls(log) == [{HF}]
