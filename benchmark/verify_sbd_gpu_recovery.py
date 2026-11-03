# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Stage-2 standalone validation for SbdGpuDiagonalizer.recover_and_diagonalize.

Builds a synthetic sample-counts dict for H2 (norb=2, dim=4 -- trivially small) by sampling
ffsim's exact ground state, then runs the new GPU-based recovery loop and checks the recovered
energy against the exact CPU FCI answer.

Must run on a roquo compute node with a GPU allocation -- the diag-gpu binary is
aarch64/GB200-only and cannot run on the login node.

    python benchmark/verify_sbd_gpu_recovery.py --binary-path /path/to/diag-gpu
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
    ap.add_argument("--launcher-cmd", default="mpirun -np 1",
                    help="launcher prefix as one shell-quoted string (empty string to run the "
                         "binary directly) -- NOT srun, see backend_sbd_gpu.py's module docstring")
    ap.add_argument("--molecule", default="molecules/small/h2.yaml")
    ap.add_argument("--tolerance-mha", type=float, default=5.0,
                    help="recovery from a small noisy sample won't be exact -- this is a sanity "
                         "bound (chem. acc. is 1.6 mHa; recovery from few shots may not reach it)")
    ap.add_argument("--max-iterations", type=int, default=5)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    abs_cfg = str(repo_root / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default")
    cfg.molecule = OmegaConf.load(repo_root / args.molecule)

    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.core.configuration import Configuration, ConfigurationSet
    from sqd_gqe.core.refine import NonOrthogonalSolver
    from sqd_gqe.sqd.backend_sbd_gpu import SbdGpuDiagonalizer

    factory = SqdFactory()
    system = factory.build_system(cfg)
    norb, nelec = system.norb, system.nelec
    hamiltonian = system.active_hamiltonian
    if nelec[0] != nelec[1]:
        print(f"SKIP: recover_and_diagonalize is RHF-only, molecule has nelec={nelec}",
              file=sys.stderr)
        return 0

    from math import comb
    fci_dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    print(f"molecule={args.molecule} norb={norb} nelec={nelec} fci_dim={fci_dim}")

    # exact reference via the existing CPU FCI parity path
    configs = [Configuration.from_fullci_address(i, norb, nelec) for i in range(fci_dim)]
    strings = np.asarray(ConfigurationSet(configs).configs, dtype=np.uint64).reshape(-1, 2)
    fci_solver = NonOrthogonalSolver(hamiltonian, nelec, norb)
    energy_exact, _ = fci_solver.diagonalize_subspace(strings)
    print(f"energy_exact (full FCI space, CPU)={energy_exact:.10f}")

    # synthetic "sample counts" -- every full-CI determinant equally weighted (norb=2 is small
    # enough that this always contains the exact ground state, so recovery should land on it).
    rng = np.random.default_rng(0)
    counts: dict[tuple[int, int], int] = {}
    for c in configs:
        a, b = int(c[0]), int(c[1])
        counts[(a, b)] = counts.get((a, b), 0) + int(rng.integers(50, 150))
    print(f"synthetic counts: {len(counts)} distinct configs, {sum(counts.values())} total shots")

    launcher = args.launcher_cmd.split() if args.launcher_cmd else []
    sbd_diag = SbdGpuDiagonalizer(hamiltonian, nelec, norb, binary_path=args.binary_path,
                                  launcher=launcher)
    energy_recovered, _, history = sbd_diag.recover_and_diagonalize(
        counts, samples_per_batch=fci_dim, num_batches=1,
        max_iterations=args.max_iterations, seed=0, subspace_dim=fci_dim,
    )
    print(f"energy_recovered={energy_recovered:.10f}")
    print(f"history={history}")

    gap_mha = abs(energy_recovered - energy_exact) * 1e3
    print(f"gap vs exact={gap_mha:.4f} mHa (tolerance={args.tolerance_mha:.1f} mHa)")

    if not np.isfinite(energy_recovered) or gap_mha > args.tolerance_mha:
        print("FAIL: recovered energy does not match the exact answer within tolerance",
              file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
