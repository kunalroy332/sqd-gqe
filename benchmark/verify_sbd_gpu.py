# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Stage-1 standalone validation for the SBD GPU diagonalizer (no pipeline/factory involvement).

Diagonalizes H2's full FCI space (norb=2, dim=4 -- trivially small) three ways: the existing CPU
"sqd" backend, the existing CPU "fci" parity backend, and the new GPU backend
(SbdGpuDiagonalizer). Prints all three energies and the gaps; exits nonzero if the GPU energy
doesn't match the CPU energies to a tight tolerance.

Must run on a roquo compute node with a GPU allocation (see slurm/verify_sbd_gpu.sh) -- the
diag-gpu binary is aarch64/GB200-only and cannot run on the login node.

    python benchmark/verify_sbd_gpu.py --binary-path /path/to/diag-gpu
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary-path", required=True, help="path to the diag-gpu executable")
    ap.add_argument("--launcher-cmd", default="srun --gpu-bind=closest",
                    help="launcher prefix as one shell-quoted string, "
                         "e.g. 'srun --gpu-bind=closest' (empty string to run the binary directly)")
    ap.add_argument("--molecule", default="molecules/small/h2.yaml")
    ap.add_argument("--tolerance-ha", type=float, default=1e-6)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    abs_cfg = str(repo_root / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default")
    cfg.molecule = OmegaConf.load(repo_root / args.molecule)

    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.core.configuration import Configuration, ConfigurationSet
    from sqd_gqe.core.refine import NonOrthogonalSolver
    from sqd_gqe.sqd.backend_sqd import SqdDiagonalizer
    from sqd_gqe.sqd.backend_sbd_gpu import SbdGpuDiagonalizer

    factory = SqdFactory()
    system = factory.build_system(cfg)
    norb, nelec = system.norb, system.nelec
    hamiltonian = system.active_hamiltonian

    from math import comb
    fci_dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    print(f"molecule={args.molecule} norb={norb} nelec={nelec} fci_dim={fci_dim}")

    configs = [Configuration.from_fullci_address(i, norb, nelec) for i in range(fci_dim)]
    config_set = ConfigurationSet(configs)
    strings = np.asarray(config_set.configs, dtype=np.uint64).reshape(-1, 2)

    sqd_diag = SqdDiagonalizer(hamiltonian, nelec, norb)
    energy_sqd, _ = sqd_diag.diagonalize(config_set)

    fci_solver = NonOrthogonalSolver(hamiltonian, nelec, norb)
    energy_fci, _ = fci_solver.diagonalize_subspace(strings)

    launcher = args.launcher_cmd.split() if args.launcher_cmd else []
    sbd_diag = SbdGpuDiagonalizer(hamiltonian, nelec, norb, binary_path=args.binary_path,
                                  launcher=launcher)
    energy_gpu, _ = sbd_diag.diagonalize(config_set)

    gap_cpu = abs(energy_gpu - energy_sqd)
    gap_fci = abs(energy_gpu - energy_fci)
    print(f"energy_sqd={energy_sqd:.10f}")
    print(f"energy_fci={energy_fci:.10f}")
    print(f"energy_gpu={energy_gpu:.10f}")
    print(f"gap vs sqd={gap_cpu:.3e} Ha, gap vs fci={gap_fci:.3e} Ha (tolerance={args.tolerance_ha:.1e})")

    if gap_cpu > args.tolerance_ha or gap_fci > args.tolerance_ha:
        print("FAIL: GPU energy does not match CPU energies within tolerance", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
