# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File-format helpers for the SBD native GPU diagonalizer binary (``diag-gpu``).

Ported from qcsc-prefect's ``algorithms/sbd/sbd/solver_job.py`` (RHF-only slice: the UHF
interleaved-spin-orbital FCIDUMP writer is intentionally not included here, see
backend_sbd_gpu.py). These are plain functions with zero Prefect coupling in the original
source, so this is a faithful port, not a reimplementation from a spec.

Confirmed empirically on roquo (2026-08-09) against the actual ``diag-gpu`` binary:
  * it reads a fixed filename ``fcidump.txt`` from its current working directory (no CLI flag
    for the path) -- aborts with "Failed to open FCIDUMP file" if absent, and has no --help.
  * RHF needs only ``AlphaDets.bin`` (no beta file -- closed-shell, alpha==beta determinants).
  * Determinant packing: ``(norb+7)//8`` bytes per determinant, big-endian.
  * ``davidson_energy.txt`` is the pure electronic energy (FCIDUMP's ECORE defaults to 0 via
    ``pyscf.tools.fcidump.from_integrals``'s own ``nuc=0`` default) -- callers must add the
    active space's core energy back themselves, same as backend_sqd.py's SqdDiagonalizer does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def write_dets_bin(path: str | Path, det_ints: Sequence[int], norb: int) -> None:
    """Write a determinant list in the format ``diag-gpu``'s ``LoadAlphaDets`` reader expects:
    ``(norb + 7) // 8`` bytes per configuration, big-endian, one flat binary file."""
    dets = np.asarray(det_ints, dtype=np.int64).reshape(-1)
    if np.any(dets < 0):
        raise ValueError("determinants must be non-negative integers.")
    max_ci = 1 << norb
    if np.any(dets >= max_ci):
        raise ValueError(f"determinant is out of range for norb={norb}.")
    bytes_per_config = (norb + 7) // 8
    with Path(path).open("wb") as fp:
        for ci in dets:
            fp.write(int(ci).to_bytes(bytes_per_config, byteorder="big", signed=False))


def read_carryover_bin(path: str | Path, norb: int) -> np.ndarray:
    """Decode an SBD ``carryover.bin`` file into a ``(n, norb)`` bool array.

    Bit layout, matching ``diag-gpu``'s writer in
    ``apps/chemistry_tpb_selected_basis_diagonalization/main.cc``: it writes
    ``bytes_per_config = (norb + 7) // 8`` bytes per determinant and fills bit
    positions ``j = 0 .. norb-1`` counting from the FIRST byte's MSB (``pb = 7 -
    (j % 8)``, ``bb = j // 8``), storing orbital ``norb-1-j`` at position ``j``
    (``sbd::makestring`` order). Any bits past ``j = norb-1`` stay zero.

    So the payload is the LEADING ``norb`` bits, not the trailing ones.
    Slicing ``[-norb:]`` instead (as this function did until 2026-09-21) reads a
    window shifted by ``8*bytes_per_config - norb`` bits into the zero padding,
    silently corrupting every determinant whenever ``norb`` is not a multiple of
    8. Confirmed numerically: for ``norb=10`` the HF determinant 63 (popcount 6,
    bits ``0000111111``) decoded as 960 (popcount 4) -- exactly the wrong-electron
    -count determinants observed poisoning ch2o/co2 recovery subspaces on CCC.
    ``norb=8`` (ch4, n2) and other multiples of 8 are unaffected, which is why
    this only surfaced once the 11-molecule set added ch2o(10)/hcn(11)/c2h4(12)/
    co2(14).
    """
    bytes_per_config = (norb + 7) // 8
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return np.empty((0, norb), dtype=bool)
    if data.size % bytes_per_config != 0:
        raise ValueError(
            f"carryover.bin size {data.size} is not a multiple of bytes_per_config={bytes_per_config}"
        )
    n = data.size // bytes_per_config
    bits = np.unpackbits(data.reshape(n, bytes_per_config), axis=1, bitorder="big")
    return bits[:, :norb].astype(bool)


def read_scalar_txt(path: str | Path) -> float:
    """Read a single scalar from a plain-text file (``davidson_energy.txt``, ``occ_*.txt``)."""
    return float(np.loadtxt(str(path)).item())


def build_solver_args(
    solver_kwargs: dict,
    *,
    adetfile: str = "AlphaDets.bin",
    carryoverfile: str = "carryover.txt",
) -> list[str]:
    """Build the ``diag-gpu`` CLI argument list for a single-GPU RHF solve.

    ``solver_kwargs`` keys (all required, matching the confirmed CLI contract): task_comm_size,
    adet_comm_size, bdet_comm_size, block, iteration, tolerance, carryover_ratio, carryover_type,
    rdm.
    """
    args = [
        "--task_comm_size", str(solver_kwargs["task_comm_size"]),
        "--adet_comm_size", str(solver_kwargs["adet_comm_size"]),
        "--bdet_comm_size", str(solver_kwargs["bdet_comm_size"]),
        "--block", str(solver_kwargs["block"]),
        "--iteration", str(solver_kwargs["iteration"]),
        "--tolerance", str(solver_kwargs["tolerance"]),
        "--carryover_ratio", str(solver_kwargs["carryover_ratio"]),
        "--carryover_type", str(solver_kwargs["carryover_type"]),
        "--rdm", str(solver_kwargs["rdm"]),
        "--adetfile", adetfile,
        "--carryoverfile", carryoverfile,
    ]
    return args
