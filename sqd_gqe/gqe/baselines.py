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

"""Fixed-ansatz reference state preparations for benchmarking against GQE.

Two baselines from the source paper (arXiv:2604.09756), both prepared once (no
learned policy) and then fed to the same sample -> subspace -> SQD pipeline:

  * LUCJ  -- apply the CCSD-seeded local unitary cluster Jastrow operator once.
  * SqDRIFT -- randomized (qDRIFT) real-time evolution of the molecular
    Hamiltonian, e^{-iHt}, via ffsim's double-factorized qDRIFT primitive.

Each returns a sampled ``counts`` dict in the same (alpha, beta) integer form as
``FfsimSampler``, so the SQD diagonalizer + subspace utilities are reused
unchanged.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def _ints_to_counts(samples, norb: int) -> dict[tuple[int, int], int]:
    mask = (1 << norb) - 1
    counts: Counter = Counter()
    for s in np.asarray(samples).tolist():
        s = int(s)
        counts[(s & mask, s >> norb)] += 1
    return dict(counts)


def _molecular_hamiltonian(system):
    import ffsim
    h = system.active_hamiltonian
    return ffsim.MolecularHamiltonian(
        np.asarray(h.one_body, dtype=np.float64),
        np.asarray(h.two_body, dtype=np.float64),
        constant=float(h.core_energy),
    )


def lucj_counts(system, shots: int, seed: int = 0, n_reps=None,
                interaction_pairs=None) -> dict[tuple[int, int], int]:
    """Sample from a single CCSD-seeded LUCJ state (the paper's LUCJ baseline)."""
    import ffsim

    norb, nelec = system.norb, tuple(system.nelec)
    if interaction_pairs is None:
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb)]
        interaction_pairs = (pairs_aa, pairs_ab)
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        system.ccsd_amplitudes["t2"], t1=system.ccsd_amplitudes["t1"],
        n_reps=n_reps, interaction_pairs=interaction_pairs,
    )
    vec = ffsim.hartree_fock_state(norb, nelec)
    vec = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)
    samples = ffsim.sample_state_vector(
        vec, norb=norb, nelec=nelec, shots=shots,
        bitstring_type=ffsim.BitstringType.INT, seed=seed,
    )
    return _ints_to_counts(samples, norb)


def sqdrift_counts(system, shots: int, time: float = 1.0, n_samples: int = 500,
                   seed: int = 0) -> dict[tuple[int, int], int]:
    """Sample from a randomized qDRIFT time-evolved state e^{-iHt}|HF>.

    ``n_samples`` is the number of qDRIFT randomizations (paper uses 500);
    ``time`` the evolution time. Uses ffsim's double-factorized qDRIFT.
    """
    import ffsim

    norb, nelec = system.norb, tuple(system.nelec)
    mol_ham = _molecular_hamiltonian(system)
    dfh = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(mol_ham)
    vec = ffsim.hartree_fock_state(norb, nelec)
    vec = ffsim.simulate_qdrift_double_factorized(
        vec, dfh, time, norb=norb, nelec=nelec, n_samples=n_samples, seed=seed,
    )
    # qDRIFT with n_samples>1 returns a batch of states; average the sampled
    # distribution over them (each is one random product-formula realization).
    vecs = np.atleast_2d(vec)
    counts: Counter = Counter()
    per = max(shots // len(vecs), 1)
    for i, v in enumerate(vecs):
        nrm = np.linalg.norm(v)
        if nrm == 0:
            continue
        s = ffsim.sample_state_vector(
            v / nrm, norb=norb, nelec=nelec, shots=per,
            bitstring_type=ffsim.BitstringType.INT, seed=seed + i,
        )
        for k, c in _ints_to_counts(s, norb).items():
            counts[k] += c
    return dict(counts)
