# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dimensionality sweep on REAL IBM hardware sample counts.

Loads bitstring counts collected from a real Qiskit Runtime job (saved as a
Python dict literal via ``str(counts)``), decodes them into ffsim's
``(alpha_occ, beta_occ)`` convention -- classical bit i = logical qubit i,
no layout permutation needed (confirmed empirically, see
sqd_gqe/gqe/ibm_hardware_sampler.py's docstring/history) -- then runs the
existing SQD subspace-expansion + diagonalization path at increasing
``max_dim``, timing each solve and tracking energy error against the best
available classical reference, stopping once chemical accuracy (1.6 mHa) is
reached.

Not every tier necessarily has usable signal: a real device's returned
bitstring distribution can be dominated by decoherence noise for deep
circuits (e.g. LL's 6,687-depth circuit returned 10,000 distinct bitstrings
across 10,000 shots -- every shot unique, no repeats at all -- which is
uniform noise, not sampling signal, and diagonalizing it would just be
diagonalizing noise). Check the counts' concentration before trusting a
sweep result.
"""

from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHEM_ACC_MHA = 1.6


def decode_counts(raw_counts: dict[str, int], norb: int) -> dict[tuple[int, int], int]:
    """Same decode as IbmHardwareSampler._decode_bitarray: classical bit i ==
    logical qubit i, mask/shift into (alpha_occ, beta_occ). No layout
    permutation -- confirmed unnecessary for a plain measure_all() circuit."""
    n_qubits = 2 * norb
    mask = (1 << norb) - 1
    counts: dict[tuple[int, int], int] = {}
    for bitstring, n in raw_counts.items():
        bits = bitstring.replace(" ", "")
        s = 0
        for i in range(n_qubits):
            bit = int(bits[len(bits) - 1 - i])
            s |= (bit << i)
        alpha = s & mask
        beta = s >> norb
        key = (alpha, beta)
        counts[key] = counts.get(key, 0) + n
    return counts


def signal_concentration(raw_counts: dict[str, int]) -> dict:
    total = sum(raw_counts.values())
    n_distinct = len(raw_counts)
    top_count = max(raw_counts.values())
    return {
        "total_shots": total,
        "n_distinct": n_distinct,
        "top_bitstring_frac": top_count / total,
        "distinct_frac": n_distinct / total,
    }


def sweep_molecule(molecule_yaml: str, counts_file: str, nelec_target: tuple[int, int] | None,
                    max_dims: list[int]):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.core.configuration import ConfigurationSet
    from sqd_gqe.core.system import FciIntractableError

    raw_counts = ast.literal_eval(Path(counts_file).read_text())
    conc = signal_concentration(raw_counts)
    print(f"  signal concentration: {conc['n_distinct']} distinct / {conc['total_shots']} shots "
          f"({100*conc['distinct_frac']:.1f}% distinct), top bitstring = "
          f"{100*conc['top_bitstring_frac']:.1f}% of shots")
    if conc["distinct_frac"] > 0.9:
        print("  WARNING: >90% of shots are distinct bitstrings -- this looks like uniform "
              "noise, not sampling signal. Diagonalization results below are likely meaningless.")

    abs_cfg = str(REPO_ROOT / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default")
    cfg.molecule = OmegaConf.load(molecule_yaml)

    factory = SqdFactory()
    system = factory.build_system(cfg)
    pool = factory.build_operator_pool(cfg)
    norb = pool.norb
    nelec = nelec_target or tuple(system.nelec)

    decoded = decode_counts(raw_counts, norb)
    config_set = ConfigurationSet.from_sample_counts(decoded)
    config_set = config_set.select_by_nelec(nelec)
    print(f"  decoded: {len(decoded)} distinct (alpha,beta) pairs, "
          f"{config_set.size} conserve nelec={nelec}")

    if config_set.size == 0:
        print("  no configurations conserve the target electron count exactly -- trying "
              "recover_and_diagonalize (self-consistent configuration recovery from average "
              "orbital occupancies) instead of giving up.")
        try:
            reference = float(system.casci_energy())
            ref_label = "CASCI"
        except FciIntractableError:
            try:
                reference = float(system.ccsd_t_energy())
                ref_label = "CCSD(T)"
            except Exception:
                reference = float(system.ccsd_energy())
                ref_label = "CCSD"
        print(f"  reference ({ref_label}): {reference:.6f} Ha")

        from sqd_gqe.sqd.backend_sqd import SqdDiagonalizer
        diagonalizer = SqdDiagonalizer(system.active_hamiltonian, nelec, norb)
        t0 = time.perf_counter()
        energy, _, history = diagonalizer.recover_and_diagonalize(
            decoded, samples_per_batch=512, num_batches=3, max_iterations=10, seed=0,
        )
        dt = time.perf_counter() - t0
        err_mha = (energy - reference) * 1000.0
        chem_acc = abs(err_mha) < CHEM_ACC_MHA
        print(f"  recover_and_diagonalize: energy={energy:.6f} Ha  err={err_mha:+.3f} mHa  "
              f"time={dt:.2f}s  iterations={len(history)}  chem_acc={'YES' if chem_acc else 'no'}")
        if history:
            print(f"  convergence history (Ha): {[f'{h:.6f}' for h in history]}")
        return

    try:
        reference = float(system.casci_energy())
        ref_label = "CASCI"
    except FciIntractableError:
        try:
            reference = float(system.ccsd_t_energy())
            ref_label = "CCSD(T)"
        except Exception:
            reference = float(system.ccsd_energy())
            ref_label = "CCSD"
    print(f"  reference ({ref_label}): {reference:.6f} Ha")

    from sqd_gqe.sqd.backend_sqd import SqdDiagonalizer
    diagonalizer = SqdDiagonalizer(system.active_hamiltonian, nelec, norb)

    for max_dim in max_dims:
        expanded = config_set.expand(max_size=max_dim, method="symmetry_completion")
        t0 = time.perf_counter()
        energy, _ = diagonalizer.diagonalize(expanded)
        dt = time.perf_counter() - t0
        err_mha = (energy - reference) * 1000.0
        chem_acc = abs(err_mha) < CHEM_ACC_MHA
        print(f"  max_dim={max_dim:>7d}  dim_used={expanded.size:>7d}  "
              f"energy={energy:.6f} Ha  err={err_mha:+.3f} mHa  "
              f"time={dt:.2f}s  chem_acc={'YES' if chem_acc else 'no'}")
        if chem_acc:
            print(f"  -> chemical accuracy reached at max_dim={max_dim}, stopping sweep.")
            return
    print(f"  -> chemical accuracy NOT reached after max_dim={max_dims[-1]}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", required=True)
    ap.add_argument("--counts-file", required=True)
    ap.add_argument("--max-dims", type=int, nargs="+",
                     default=[500, 2000, 10000, 50000, 100000])
    args = ap.parse_args()

    name = Path(args.molecule).stem
    print(f"=== {name} ===")
    sweep_molecule(args.molecule, args.counts_file, None, args.max_dims)


if __name__ == "__main__":
    main()
