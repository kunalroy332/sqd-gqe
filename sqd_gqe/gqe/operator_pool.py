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

"""ffsim-native operator pool for the GQE policy.

The GQE policy selects a sequence of token indices; each token maps to one
entry of this pool. Every entry is a concrete, individually-applicable
**ffsim unitary** that preserves particle number (and, for the balanced
variants, total spin). The sampler composes the selected
entries on top of the Hartree-Fock state and samples bitstrings.

Design goals:
  * ``pool[0]`` is the identity (a no-op).
  * The pool mixes *chemistry-motivated* layers (LUCJ/UCJ sublayers seeded from
    the molecule's CCSD amplitudes) with *local hardware-efficient* primitives
    (orbital rotations, Givens/hop rotations, diagonal-Coulomb evolutions,
    number-number interactions). This gives the policy the LUCJ "chemistry
    inductive bias" plus enough locality to explore.
  * ``get_gate_count(seq)`` returns an approximate two-qubit / total gate count
    for gate-efficiency comparisons.

All ffsim entry points are looked up lazily so this module can be imported
(for inspection / typing) without ffsim installed; ``build_operator_pool``
requires ffsim at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from sqd_gqe.core.system import MolecularSystem


# --------------------------------------------------------------------------- #
# Pool entries: each is an applicable ffsim unitary with a rough gate count.   #
# --------------------------------------------------------------------------- #
@dataclass
class FfsimOp:
    """One applicable ffsim operator (a single token in the GQE vocabulary)."""

    name: str
    apply_fn: Callable[[np.ndarray, int, tuple[int, int]], np.ndarray]
    #: approximate logical gate counts contributed when this op is applied,
    #: keyed by gate label ("cx", "total", ...). Used only for benchmarking.
    gate_count: Counter = field(default_factory=Counter)
    #: orbital indices this op touches (for the locality metric).
    support: tuple[int, ...] = ()
    #: optional factory returning the Jordan-Wigner qiskit Gate for this op,
    #: ``qiskit_gate_fn(norb) -> qiskit.circuit.Gate``. Used to build a real
    #: circuit and measure transpiled depth. ``None`` for ops without a clean
    #: JW mapping (their cost is taken from ``gate_count`` instead).
    qiskit_gate_fn: Callable[[int], object] | None = None

    def apply(self, vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
        return self.apply_fn(vec, norb, nelec)


class FfsimBasedPool(ABC):
    """Base class for ffsim operator pools (the GQE token vocabulary)."""

    def __init__(self, molecule: MolecularSystem, params: Sequence[float] | None,
                 max_vocab: int | None = None, **kwargs):
        self.molecule = molecule
        self.n_qubits = molecule.norb * 2
        self.norb = molecule.norb
        self.n_electrons = int(sum(molecule.nelec))
        self.nelec = tuple(molecule.nelec)
        self.params = params
        self.pool: list[FfsimOp] = self.build_operator_pool(**kwargs)
        self.n_real = len(self.pool)
        # Pad to a common vocabulary so ONE amortized policy head serves every
        # molecule. Padding tokens are identity no-ops, so a policy trained at
        # max_vocab can act on any smaller molecule; extra indices do nothing.
        if max_vocab is not None and max_vocab > len(self.pool):
            self.pool += [self.identity_op() for _ in range(max_vocab - len(self.pool))]
        self.max_vocab = max_vocab

    def __len__(self):
        return len(self.pool)

    def __iter__(self):
        return iter(self.pool)

    def __getitem__(self, idx):
        # out-of-range indices act as identity (robust to policy overshoot)
        return self.pool[idx] if 0 <= idx < len(self.pool) else self.identity_op()

    def get_vocab_size(self) -> int:
        return len(self.pool)

    @abstractmethod
    def build_operator_pool(self, **kwargs) -> list[FfsimOp]:
        ...

    def get_gate_count(self, seq: Sequence[int]) -> Counter:
        counts: Counter = Counter()
        for i in seq:
            counts.update(self.pool[int(i)].gate_count)
        return counts

    @staticmethod
    def identity_op() -> FfsimOp:
        return FfsimOp(name="I", apply_fn=lambda vec, norb, nelec: vec, gate_count=Counter(), support=())


class FfsimLUCJPool(FfsimBasedPool):
    """LUCJ/UCJ + local-primitive pool.

    Parameters
    ----------
    molecule
        Provides ``norb``, ``nelec`` and CCSD amplitudes to seed the UCJ layers.
    params
        Optional explicit override for the local primitives' angles. If given,
        this wins outright and ``angle_source`` below is ignored.
    angle_source
        How to pick the local primitives' (Givens/diag-Coulomb/num-num) default
        angles when ``params`` is not given. Two CCSD-derived options, both
        measured (N2, 16 qubits) to beat the old fixed ``[pi/4, pi/2]`` baseline
        on at least one real axis -- see ``_ccsd_scaled_angles``'s docstring for
        the exact numbers and how to reproduce them:

          * ``"ffsim_native"`` (default) -- angles from the same real ffsim
            CCSD-amplitude factorization the UCJ seed op itself uses
            (``ffsim.linalg.double_factorized_t2`` + ``orbital_rotation_from_t1_amplitudes``).
            Best accuracy of the two (mean/max error both improve on the fixed
            baseline), plus a real (if smaller) improvement in how often the
            training gate accepts an update -- the more balanced default.
          * ``"rms"`` -- a simpler, cruder heuristic: the RMS magnitude of the
            raw t1/t2 amplitude tensors. Highest training-gate PASS rate of
            the three (more gradient updates get through), at the cost of the
            worst accuracy among the three when it does pass -- useful when
            you specifically want more frequent (if noisier) updates, e.g.
            early in a run or on a system where the gate rarely passes at all.

        Both fall back to the old fixed ``[pi/4, pi/2]`` if CCSD amplitudes are
        missing or the derived scale is numerically degenerate.
    n_reps
        Number of UCJ repetitions used when constructing the seed operator.
    interaction_pairs
        LUCJ locality pattern ``(pairs_aa, pairs_ab)``; if ``None`` a line /
        nearest-neighbour pattern is generated.
    include_orbital_rotations, include_givens, include_diag_coulomb,
    include_num_num
        Toggles for which local primitive families are added to the pool.
    seed
        RNG seed for the random orbital-rotation primitives (kept fixed so the
        pool is deterministic across runs).
    """

    #: default LUCJ repetitions when n_reps is not explicitly overridden.
    #: There is no separate, deeper "UCJ(reps=N)" pool entry any more -- the
    #: auto-selected/ceiling-6 version of that block transpiled to 15k-50k
    #: depth on heavy-hex hardware and dominated every other term in the
    #: reward (see cost_reward.py's real_depths term), so rather than relying
    #: on the reward to train the policy away from it, the option is removed
    #: from the pool: this LUCJ block (reps=2, already restricted to the
    #: nearest-neighbour interaction_pairs below) is now the ONLY chemistry-
    #: seeded op available.
    LUCJ_N_REPS = 2

    #: valid values for angle_source -- checked eagerly in __init__ rather
    #: than only failing lazily inside _ccsd_scaled_angles.
    ANGLE_SOURCES = ("ffsim_native", "rms")

    def __init__(
        self,
        molecule: MolecularSystem,
        params: Sequence[float] | None = None,
        angle_source: str = "ffsim_native",
        n_reps: int | None = None,
        interaction_pairs: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None = None,
        include_orbital_rotations: bool = True,
        include_givens: bool = True,
        include_diag_coulomb: bool = True,
        include_num_num: bool = True,
        n_orbital_rotations: int = 4,
        seed: int = 0,
        max_vocab: int | None = None,
    ):
        if angle_source not in self.ANGLE_SOURCES:
            raise ValueError(f"angle_source must be one of {self.ANGLE_SOURCES}; got {angle_source!r}")
        self.angle_source = angle_source
        self.n_reps = n_reps
        self.interaction_pairs = interaction_pairs
        self.include_orbital_rotations = include_orbital_rotations
        self.include_givens = include_givens
        self.include_diag_coulomb = include_diag_coulomb
        self.include_num_num = include_num_num
        self.n_orbital_rotations = n_orbital_rotations
        self.seed = seed
        super().__init__(molecule, params, max_vocab=max_vocab)

    # -- LUCJ interaction pattern ------------------------------------------- #
    def default_interaction_pairs(self, norb: int):
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]  # nearest-neighbour, upper-tri
        pairs_ab = [(p, p) for p in range(norb)]           # same-orbital opposite-spin rungs
        return pairs_aa, pairs_ab

    #: fallback angle set (radians) used only when CCSD amplitudes are
    #: unavailable/degenerate (see _ccsd_scaled_angles) -- kept as a last
    #: resort, not the intended path any more.
    FALLBACK_ANGLES = [0.25 * np.pi, 0.5 * np.pi]

    def _ccsd_scaled_angles(self) -> list[float]:
        """Derive the local primitives' (Givens/diag-Coulomb/num-num) default
        angles from CCSD, dispatching on ``self.angle_source``. See the class
        docstring's ``angle_source`` entry for what each option is and why
        both exist; see the two ``_angles_from_*`` methods below for the exact
        math. Both return ``[0.5x, 1.5x]`` of a single derived scale, to
        preserve the existing "two angles per primitive family" pool shape.

        Measured on N2 (16 qubits), 20 amortized-training iterations,
        ``--max-dim 200`` (small enough that circuit/angle quality actually
        gates the SQD subspace selection -- at a generous max-dim every choice
        here reaches the exact answer immediately and the comparison is
        meaningless), counting only iterations the training gate accepted
        (see ``benchmark/amortized_train.py``'s ``depth_ceiling_for_qubits``):

        =====================  =========  =================  ================
        angle_source           gate PASS  mean \\|err\\| (mHa)   max \\|err\\| (mHa)
        =====================  =========  =================  ================
        fixed [pi/4, pi/2]      8/20 (40%)  0.0147             0.0756
        ``"rms"``              13/20 (65%)  0.0390             0.3230
        ``"ffsim_native"``     11/20 (55%)  0.0114             0.0630
        =====================  =========  =================  ================

        Reproduce with ``benchmark/amortized_train.py --molecules
        molecules/small/n2.yaml --backend ideal --iters 20 --ngates 8
        --shots 5000 --max-dim 200``, forcing a specific source via
        ``--hydra-override operator_pool.angle_source=rms`` (or
        ``ffsim_native``), or the old fixed baseline via an explicit
        ``--hydra-override "operator_pool.params=[0.7853981633974483,1.5707963267948966]"``.
        """
        if self.angle_source == "rms":
            return self._angles_from_rms()
        return self._angles_from_ffsim_native()

    def _angles_from_ffsim_native(self) -> list[float]:
        """``angle_source="ffsim_native"`` (default): derive the scale from
        the SAME real CCSD-amplitude factorization ffsim's own LUCJ seed op
        already uses, instead of a fixed constant or a plain RMS-of-amplitudes
        heuristic.

        ``ffsim.linalg.double_factorized_t2(t2)`` performs the exact same
        nested-eigenvalue double factorization ``UCJOpSpinBalanced.from_t_amplitudes``
        uses internally, returning real diagonal-Coulomb matrices Z^(k) and
        orbital-rotation matrices U^(k) such that t2 = i sum_k Z^(k) (x) U^(k)
        terms. ``ffsim.variational.util.orbital_rotation_from_t1_amplitudes(t1)``
        builds the analogous one-body rotation exp(t1 - t1^dagger). Both are
        genuine ffsim-native CCSD-derived objects, not an invented scale:

          * diag-Coulomb / num-num angle <- the largest-magnitude eigenvalue of
            the dominant (largest-norm) Z^(k) term -- this IS the coupling
            strength apply_diag_coulomb_evolution's own ``mat`` argument would
            carry for that term, so it's the same physical quantity, just
            reduced from a matrix to a scalar.
          * Givens angle <- the largest rotation angle among U's eigenvalues
            (each eigenvalue of a unitary is e^{i*phi}; phi is a genuine
            single-excitation rotation angle, the same quantity
            orbital_rotation_from_t1_amplitudes turns t1 into internally).

        Falls back to FALLBACK_ANGLES if CCSD amplitudes are missing, the
        factorization yields no terms, or the resulting scale is degenerate
        (below a numerical-noise floor).
        """
        import ffsim
        from ffsim.variational.util import orbital_rotation_from_t1_amplitudes

        amps = self.molecule.ccsd_amplitudes
        t1, t2 = amps["t1"], amps["t2"]

        diag_scale = 0.0
        try:
            diag_mats, _orb_rots = ffsim.linalg.double_factorized_t2(t2)
            if diag_mats.shape[0] > 0:
                norms = np.linalg.norm(diag_mats.reshape(diag_mats.shape[0], -1), axis=1)
                dominant = diag_mats[int(np.argmax(norms))]
                diag_scale = float(np.max(np.abs(np.linalg.eigvalsh(dominant))))
        except Exception:
            pass

        givens_scale = 0.0
        try:
            U = orbital_rotation_from_t1_amplitudes(t1)
            phases = np.angle(np.linalg.eigvals(U))
            givens_scale = float(np.max(np.abs(phases)))
        except Exception:
            pass

        scale = max(diag_scale, givens_scale)
        if not np.isfinite(scale) or scale < 1e-8:
            return list(self.FALLBACK_ANGLES)
        return [0.5 * scale, 1.5 * scale]

    def _angles_from_rms(self) -> list[float]:
        """``angle_source="rms"``: a simpler, cruder heuristic than
        ``"ffsim_native"`` -- the RMS magnitude of the raw t1/t2 amplitude
        tensors, pooled together. Not physically tied to a specific
        primitive's generator the way ``"ffsim_native"``'s eigenvalues are,
        but measured to give the highest training-gate PASS rate of the two
        options (see ``_ccsd_scaled_angles``'s docstring for the numbers) --
        worth trying when more frequent, if noisier, gradient updates matter
        more than per-update accuracy.
        """
        amps = self.molecule.ccsd_amplitudes
        t1, t2 = amps["t1"], amps["t2"]
        rms = float(np.sqrt(np.mean(np.concatenate([t1.ravel(), t2.ravel()]) ** 2)))
        if not np.isfinite(rms) or rms < 1e-8:
            return list(self.FALLBACK_ANGLES)
        return [0.5 * rms, 1.5 * rms]

    def build_operator_pool(self, **kwargs) -> list[FfsimOp]:
        import ffsim  # runtime dependency

        norb = self.norb
        default_angles = list(self.params) if self.params is not None else self._ccsd_scaled_angles()
        pairs_aa, pairs_ab = self.interaction_pairs or self.default_interaction_pairs(norb)

        pool: list[FfsimOp] = [self.identity_op()]

        # 1) LUCJ seed op from CCSD amplitudes -- the chemistry layer, and the
        # ONLY chemistry-seeded op in the pool (see LUCJ_N_REPS docstring for
        # why the deeper "UCJ(reps=N)" entry that used to sit alongside this
        # one was removed rather than just penalized more heavily). Uses the
        # nearest-neighbour / same-orbital interaction_pairs already computed
        # above, at a capped n_reps: an explicit constructor n_reps overrides
        # the default, but is NOT auto-selected from the CCSD
        # double-factorization rank any more (that could pick e.g. 110 for
        # sparser active spaces like ethanol's 10e/16orb).
        t2 = self.molecule.ccsd_amplitudes["t2"]
        t1 = self.molecule.ccsd_amplitudes["t1"]
        n_reps_req = self.n_reps if self.n_reps is not None else self.LUCJ_N_REPS
        lucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2, t1=t1, n_reps=n_reps_req, interaction_pairs=(pairs_aa, pairs_ab),
        )
        n_reps_eff = int(lucj_op.n_reps)

        # Rough two-qubit cost of one LUCJ block: diag-Coulomb (pairs) + two
        # orbital rotations (~norb(norb-1) Givens each). This is a benchmarking
        # estimate, not an exact transpilation.
        n_pairs = len(pairs_aa) + len(pairs_ab)
        orb_cx = norb * (norb - 1)
        lucj_cx = n_reps_eff * (n_pairs + 2 * orb_cx)

        def _apply_lucj(vec, norb, nelec, op=lucj_op):
            return ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec, copy=True)

        def _lucj_gate(norb, op=lucj_op):
            from ffsim.qiskit import UCJOpSpinBalancedJW
            return UCJOpSpinBalancedJW(op)

        pool.append(
            FfsimOp(
                name=f"LUCJ(reps={n_reps_eff})",
                apply_fn=_apply_lucj,
                gate_count=Counter({"cx": lucj_cx, "total": 2 * lucj_cx}),
                support=tuple(range(norb)),
                qiskit_gate_fn=_lucj_gate,
            )
        )

        # 2) Orbital rotations (deterministic Haar-random unitaries) -- these
        #    give the policy expressive, particle-conserving single-body layers.
        if self.include_orbital_rotations:
            for k in range(self.n_orbital_rotations):
                mat = ffsim.random.random_unitary(norb, seed=self.seed + 1000 + k)

                def _apply_orb(vec, norb, nelec, mat=mat):
                    return ffsim.apply_orbital_rotation(vec, mat, norb=norb, nelec=nelec, copy=True)

                def _orb_gate(norb, mat=mat):
                    from ffsim.qiskit import OrbitalRotationJW
                    return OrbitalRotationJW(norb, mat)

                pool.append(
                    FfsimOp(
                        name=f"orb_rot[{k}]",
                        apply_fn=_apply_orb,
                        gate_count=Counter({"cx": orb_cx, "total": 2 * orb_cx}),
                        support=tuple(range(norb)),
                        qiskit_gate_fn=_orb_gate,
                    )
                )

        # 3) Givens / hop rotations on nearest-neighbour orbital pairs.
        if self.include_givens:
            for (p, q) in [(p, p + 1) for p in range(norb - 1)]:
                for theta in default_angles:
                    def _apply_givens(vec, norb, nelec, theta=theta, orbs=(p, q)):
                        return ffsim.apply_givens_rotation(
                            vec, theta, orbs, norb=norb, nelec=nelec, copy=True
                        )

                    def _givens_gate(norb, theta=theta, orbs=(p, q)):
                        from ffsim import GivensAnsatzOp
                        from ffsim.qiskit import GivensAnsatzOpJW
                        op = GivensAnsatzOp(
                            norb, interaction_pairs=[orbs], thetas=np.array([theta]),
                            phis=None, phase_angles=None,
                        )
                        return GivensAnsatzOpJW(op)

                    pool.append(
                        FfsimOp(
                            name=f"givens({p},{q};{theta:.3f})",
                            apply_fn=_apply_givens,
                            gate_count=Counter({"cx": 2, "total": 3}),
                            support=(p, q),
                            qiskit_gate_fn=_givens_gate,
                        )
                    )

        # 4) Diagonal-Coulomb evolutions (two-body, number-conserving).
        if self.include_diag_coulomb:
            rng = np.random.default_rng(self.seed + 7)
            for t in default_angles:
                mat = rng.standard_normal((norb, norb))
                mat = 0.5 * (mat + mat.T)

                def _apply_diag(vec, norb, nelec, mat=mat, t=t):
                    return ffsim.apply_diag_coulomb_evolution(
                        vec, mat, t, norb=norb, nelec=nelec, copy=True
                    )

                def _diag_gate(norb, mat=mat, t=t):
                    from ffsim.qiskit import DiagCoulombEvolutionJW
                    return DiagCoulombEvolutionJW(norb, mat, t)

                pool.append(
                    FfsimOp(
                        name=f"diag_coulomb(t={t:.3f})",
                        apply_fn=_apply_diag,
                        gate_count=Counter({"cx": n_pairs, "total": n_pairs}),
                        support=tuple(range(norb)),
                        qiskit_gate_fn=_diag_gate,
                    )
                )

        # 5) Number-number interactions on nearest-neighbour pairs.
        if self.include_num_num:
            for (p, q) in [(p, p + 1) for p in range(norb - 1)]:
                theta = default_angles[0]

                def _apply_numnum(vec, norb, nelec, theta=theta, orbs=(p, q)):
                    return ffsim.apply_num_num_interaction(
                        vec, theta, orbs, norb=norb, nelec=nelec, copy=True
                    )

                def _numnum_gate(norb, theta=theta, orbs=(p, q)):
                    from ffsim import NumNumAnsatzOpSpinBalanced
                    from ffsim.qiskit import NumNumAnsatzOpSpinBalancedJW
                    # same-spin-only pairs (aa, ab=[]) matches
                    # apply_num_num_interaction's default spin=ALPHA_AND_BETA,
                    # which applies (p,q) within each spin sector, not across
                    # spins -- verified to 0.0 diff numerically for several
                    # theta values before relying on this here.
                    op = NumNumAnsatzOpSpinBalanced(
                        norb, interaction_pairs=([orbs], []),
                        thetas=(np.array([theta]), np.array([])),
                    )
                    return NumNumAnsatzOpSpinBalancedJW(op)

                pool.append(
                    FfsimOp(
                        name=f"num_num({p},{q})",
                        apply_fn=_apply_numnum,
                        gate_count=Counter({"cx": 2, "total": 2}),
                        support=(p, q),
                        qiskit_gate_fn=_numnum_gate,
                    )
                )

        return pool
