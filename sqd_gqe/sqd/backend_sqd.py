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

"""Diagonalization backend built on ``qiskit-addon-sqd``.

Wraps ``qiskit_addon_sqd.fermion.solve_sci`` to satisfy the pipeline's
``diagonalize(config_set) -> (energy, CIVector)`` contract, so the shared
subspace merger runs on top of the SQD arm unchanged.

Conventions (verified against qiskit-addon-sqd 0.12.x):
  * Integrals are chemist-ordered: ``hcore`` = ``one_body`` (norb, norb),
    ``eri`` = full ``(norb,)*4`` from ``ao2mo.restore(1, ...)``. The system's
    ``active_hamiltonian.two_body`` is already in this layout.
  * ``solve_sci`` returns the active-space electronic energy; the core energy is
    added afterward.
  * Configurations are ``(alpha_occ, beta_occ)`` integer bitmasks (little-endian
    per spin sector) -- the same convention ffsim samples in.
"""

from __future__ import annotations

import numpy as np

from sqd_gqe.core.configuration import ConfigurationSet, CIVector, make_ci_vector


class SqdDiagonalizer:
    def __init__(self, hamiltonian, nelec: tuple[int, int], norb: int, spin_sq: float | None = 0.0,
                 solver_kwargs: dict | None = None):
        self.nelec = tuple(nelec)
        self.norb = int(norb)
        self.spin_sq = spin_sq
        self.solver_kwargs = solver_kwargs or {}
        self._hcore = np.asarray(hamiltonian.one_body, dtype=np.float64, order="C")
        self._eri = np.asarray(hamiltonian.two_body, dtype=np.float64, order="C")
        self._core_energy = float(hamiltonian.core_energy)

    def diagonalize(self, config_set: ConfigurationSet) -> tuple[float, CIVector]:
        from qiskit_addon_sqd.fermion import solve_sci

        if config_set.size == 0:
            return float("inf"), make_ci_vector(np.zeros(0), np.zeros((0, 2), dtype=np.uint64))

        configs = np.asarray(config_set.configs, dtype=np.uint64).reshape(-1, 2)
        # PySCF's selected-CI C kernels operate on signed int64 occupation
        # strings; uint64 breaks their internal bitwise ops. Occupation masks
        # for <=16 orbitals fit in int64, so this cast is lossless.
        strs_a = np.unique(configs[:, 0]).astype(np.int64)
        strs_b = np.unique(configs[:, 1]).astype(np.int64)

        result = solve_sci(
            (strs_a, strs_b),
            self._hcore,
            self._eri,
            self.norb,
            self.nelec,
            spin_sq=self.spin_sq,
            **self.solver_kwargs,
        )
        state = result.sci_state
        energy = float(result.energy) + self._core_energy

        amps = np.asarray(state.amplitudes)
        a = np.asarray(state.ci_strs_a, dtype=np.uint64)
        b = np.asarray(state.ci_strs_b, dtype=np.uint64)
        aa, bb = np.meshgrid(a, b, indexing="ij")
        coeffs = amps.reshape(-1)
        strings = np.stack([aa.reshape(-1), bb.reshape(-1)], axis=1).astype(np.uint64)

        nz = np.abs(coeffs) > 1e-12
        return energy, make_ci_vector(coeffs[nz], strings[nz])

    def recover_and_diagonalize(
        self,
        counts: dict[tuple[int, int], int],
        samples_per_batch: int = 512,
        num_batches: int = 1,
        max_iterations: int = 10,
        seed: int = 0,
    ):
        """Self-consistent configuration recovery + diagonalization.

        Runs qiskit-addon-sqd's ``diagonalize_fermionic_hamiltonian`` loop, which
        repeatedly (i) recovers physically-valid configurations from the average
        orbital occupancies, (ii) subsamples, and (iii) diagonalizes, until the
        energy converges. Returns ``(energy, CIVector, history)`` where
        ``history`` is the list of per-recovery-iteration energies captured via
        the callback -- the data for the "how recovery converged" picture.

        Bitstrings are assembled as a ``(n_samples, 2*norb)`` bool matrix in the
        beta-left / alpha-right convention SQD expects (same as ffsim samples).
        """
        from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

        from qiskit.primitives.containers import BitArray

        norb = self.norb
        n = 2 * norb
        # counts -> {bitstring: count} in beta-left / alpha-right order (SQD's
        # convention). BitArray.from_counts builds the packed array the SQD loop
        # expects (it reads bit_array.array internally).
        str_counts: dict[str, int] = {}
        for (alpha, beta), c in counts.items():
            beta_bits = format(int(beta), f"0{norb}b")   # MSB..LSB, orbital norb-1 first
            alpha_bits = format(int(alpha), f"0{norb}b")
            str_counts[beta_bits + alpha_bits] = str_counts.get(beta_bits + alpha_bits, 0) + int(c)
        if not str_counts:
            return float("inf"), make_ci_vector(np.zeros(0), np.zeros((0, 2), dtype=np.uint64)), []
        bit_array = BitArray.from_counts(str_counts, num_bits=n)

        history: list[float] = []

        def _cb(results):
            # callback receives the list of SCIResult for the iteration's batches
            import sys
            import time
            try:
                best = min(float(r.energy) for r in results)
                history.append(best + self._core_energy)
                print(f"[recover] iteration {len(history)}/{max_iterations}: "
                     f"energy={history[-1]:.6f} ({time.strftime('%H:%M:%S')})",
                     file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[recover] iteration {len(history)+1} callback error: {e}",
                      file=sys.stderr, flush=True)

        total_shots = int(sum(str_counts.values()))
        result = diagonalize_fermionic_hamiltonian(
            self._hcore,
            self._eri,
            bit_array,
            samples_per_batch=min(samples_per_batch, total_shots),
            norb=norb,
            nelec=self.nelec,
            num_batches=num_batches,
            max_iterations=max_iterations,
            callback=_cb,
            seed=seed,
        )
        energy = float(result.energy) + self._core_energy
        state = result.sci_state
        amps = np.asarray(state.amplitudes)
        a = np.asarray(state.ci_strs_a, dtype=np.uint64)
        b = np.asarray(state.ci_strs_b, dtype=np.uint64)
        aa, bb = np.meshgrid(a, b, indexing="ij")
        strings = np.stack([aa.reshape(-1), bb.reshape(-1)], axis=1).astype(np.uint64)
        coeffs = amps.reshape(-1)
        nz = np.abs(coeffs) > 1e-12
        return energy, make_ci_vector(coeffs[nz], strings[nz]), history
