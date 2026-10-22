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

"""Accuracy-gated gate-cost penalty for GQE training.

The plain GQE reward is ``-energy``, which gives the policy no reason to be
shallow. To realize the GQE promise --- chemical accuracy at low gate cost ---
we add a depth penalty that only switches on once a circuit is already accurate:

    reward_energy(m) = energy(m)                         if energy(m) - E_ref > chem_acc
                     = energy(m) + lambda * cx_count(m)  otherwise   (within chem. acc.)

Because the penalty raises the *effective energy* (which the group-relative
advantage minimizes), the policy is pushed to find the cheapest circuit that
still reaches chemical accuracy --- never trading away correctness, since above
the accuracy threshold the reward is energy-only.

The ffsim LUCJ pool ops all conserve particle number and spin, so trimming the
circuit only removes valid building blocks --- the chemistry/physics stay intact
by construction.
"""

from __future__ import annotations

from typing import Sequence

CHEM_ACC_HA = 1.6e-3


def cost_adjusted_energies(
    energies: Sequence[float],
    seqs: Sequence[Sequence[int]],
    operator_pool,
    ref_energy: float,
    lam: float,
    chem_acc: float = CHEM_ACC_HA,
    cost_key: str = "cx",
    dims_used: Sequence[int] | None = None,
    lam_dim: float = 0.0,
    real_depths: Sequence[int] | None = None,
    lam_real_depth: float = 0.0,
    higher_rank_fracs: Sequence[float] | None = None,
    lam_excitation: float = 0.0,
    idle_fracs: Sequence[float] | None = None,
    lam_coverage: float = 0.0,
    depth_ceiling: int | None = None,
    lam_overshoot: float = 0.0,
    chem_acc_loose: float | None = None,
    hf_fracs: Sequence[float] | None = None,
    lam_hf: float = 0.0,
    spin_imbalances: Sequence[float] | None = None,
    lam_spin: float = 0.0,
) -> list[float]:
    """Return per-circuit effective reward-energies with the gated depth penalty.

    ``lam`` is the penalty weight (Ha per two-qubit gate) on
    ``operator_pool.get_gate_count(seq)`` -- a CHEAP, precomputed-per-op
    ESTIMATE, not the real device-transpiled depth. This estimate can diverge
    a lot from reality once heavy-hex SWAP routing enters the picture (see
    docs/2026-07-30/presentation_2026-07-30.tex's "why isn't the GPT circuit
    shallow" frame: ethanol's trained policy converged to a confidently
    accurate but ~15000-deep circuit family, because this proxy under-priced
    it). Set ``lam=0`` to recover the plain energy reward.

    ``real_depths``/``lam_real_depth`` add a term on the ACTUAL
    backend-transpiled depth (via circuit_metrics.circuit_depth_on_backend,
    computed by the caller once per sampled circuit -- this function stays
    backend-agnostic) instead of the proxy -- this is what actually closes
    the loop between what's rewarded and what's real. ``lam_real_depth`` is
    Ha per log10(depth) (log-scaled so a 20000-deep outlier doesn't swamp the
    reward by several Ha); pass ``real_depths=None`` (default) to skip it.

    ``dims_used``/``lam_dim`` add a third, identically-gated term on the SQD
    subspace size actually needed to diagonalize that circuit's samples (as
    reported by SqdPipeline.run_adaptive's per-sample ``subspace_size``), so
    cost-aware training pushes circuits toward being easy to diagonalize as
    well as shallow. ``lam_dim`` is Ha per log10(subspace size); pass
    ``dims_used=None`` (default) to skip it.

    ``higher_rank_fracs``/``lam_excitation`` add a fourth, identically-gated
    term rewarding a PHYSICALLY sensible wavefunction, not just an accurate
    one: an accuracy-gated circuit can still be dominated by excitation ranks
    >3 relative to HF (e.g. SnOH3-L zero-shot inference found the accepted
    circuit's recovered state at ~97.5% weight in "higher", 0% in S/D --
    inverted from the CCSD-like S/D-dominated structure a genuine low-energy
    wavefunction should have; see docs/20260921/inference_and_training_report
    .html). ``higher_rank_fracs[i]`` is
    ``excitation_rank_histogram(...)["higher"] / ["combined"]`` for circuit i
    (computed by the caller from that circuit's raw sampled counts --
    SqdSample.counts -- so this stays refine-independent, unlike
    ChemistryReport.excitation_distribution, which needs a merged/refined
    state that never materializes for FCI-intractable molecules). ``term +=
    lam_excitation * higher_rank_fracs[i]`` -- rewards LOW higher-rank weight
    (equivalently, HIGH S+D+T+HF weight) rather than a specific missing rank,
    so it pulls weight down from "higher" without needing a brittle
    per-rank threshold or curriculum stage. ``lam_excitation`` is Ha per unit
    fraction (0..1); pass ``higher_rank_fracs=None`` (default) to skip it.

    ``idle_fracs``/``lam_coverage`` add a fifth, identically-gated term
    rewarding LOW idle-qubit fraction, motivated by the real finding that
    ``--ngates=8`` circuits structurally cap active-qubit count at
    ``8*4=32`` regardless of the molecule's true qubit count (see
    docs/reports/session_status_20260923.html Sec 24-25): without this term,
    nothing in the reward distinguishes a circuit that uses every qubit from
    one that ignores most of them, as long as both are accurate on whichever
    qubits they do touch. ``idle_fracs[i]`` is ``1.0 - active_qubits /
    n_qubits`` for circuit i (computed by the caller from the real backend-
    transpiled circuit -- HardwareBackend.cost()'s own ``active_qubits`` key,
    the same transpile already done for ``real_depths``, no new real-hardware
    cost). ``term += lam_coverage * idle_fracs[i]`` -- rewards LOW idle
    fraction (equivalently HIGH qubit coverage). ``lam_coverage`` is Ha per
    unit idle fraction (0..1); pass ``idle_fracs=None`` (default) to skip it.

    ``depth_ceiling``/``lam_overshoot`` add a sixth term, UNGATED (applies
    regardless of accuracy), specifically counteracting ``lam_coverage``'s
    real failure mode found in production (Sec 26.9-26.10): every other
    depth-related term here (``lam_real_depth``, and ``lam_coverage`` itself)
    is accuracy-gated, so on a hard molecule that never reaches even
    ``chem_acc_loose``, NOTHING in the reward limits depth -- the training
    loop's own separate hard ``depth_ceiling_for_qubits()`` gate then only
    skips the gradient update outright (binary, no signal on how far over),
    letting the coverage term push the policy toward ever-deeper circuits
    with no correcting pressure once it starts failing that gate (observed:
    depth climbing pass-over-pass on repeated molecules, e.g. fe_cn6_14o14e
    542->7873, with 3 straight dW=0.0000 iterations). ``term += lam_overshoot
    * max(0.0, log10(real_depths[i] / depth_ceiling))`` -- zero when at or
    under the ceiling, growing continuously past it, on EVERY circuit
    regardless of energy, so the policy always has a real gradient pulling it
    back under the ceiling even while still inaccurate. ``lam_overshoot`` is
    Ha per log10-unit of overshoot ratio; pass ``depth_ceiling=None`` (default)
    to skip it. Requires ``real_depths`` (shared with ``lam_real_depth``).

    ``chem_acc_loose`` (Ha, optional): a SEPARATE, wider accuracy gate that
    ``lam_real_depth`` and ``lam_excitation`` use instead of ``chem_acc`` --
    ``lam`` and ``lam_dim`` are unaffected and keep the strict gate. Motivated
    by real production monitoring: on the harder molecules in this project's
    curriculum (fe_cn6, co2, c2h4), raw_err_mha routinely sits at 20-290+ mHa,
    far outside the strict chem_acc (1.6 mHa) -- so with only the strict gate,
    lam_excitation essentially never switches on for exactly the molecules
    where the excitation-rank problem is worst, and the depth/excitation
    reward can only ever shape behavior on molecules already solved almost
    exactly. Widening the gate for these two terms specifically (not lam/
    lam_dim, which still require true chemical accuracy before penalizing
    gate count or subspace size) lets shallowness and excitation-sensibility
    be shaped on near-but-not-exact circuits too, without loosening the
    guarantee that lam's cost penalty never fires on a genuinely wrong
    circuit. Defaults to ``chem_acc`` (i.e. no change) when not given.

    All terms are additive and independently gated (only accuracy-gated
    once within chem_acc, or chem_acc_loose for real_depth/excitation) so
    passing none of the optional args recovers the original CX-only reward
    unchanged.
    """
    import math
    # A circuit's sampled counts can fail to conserve the target electron
    # count at all (subspace_size=0 -> energy=inf) -- this is rare with exact
    # ffsim simulation (particle number is exact by construction) but common
    # with noisy/approximate sampling (IbmHardwareSampler's real or MPS-
    # approximated hardware counts). An inf anywhere in the batch poisons
    # GroupRelativeObjective.advantage (mean()/std() both become inf/nan,
    # producing nan gradients for every sample, not just the degenerate one)
    # -- clip to a large-but-finite penalty instead, so the policy still gets
    # a strong "this was bad" signal without destroying the whole batch's
    # training step.
    chem_acc_loose = chem_acc if chem_acc_loose is None else chem_acc_loose
    finite = [e for e in energies if math.isfinite(e)]
    penalty_e = (min(finite) if finite else ref_energy) + 10.0  # +10 Ha, far outside any real chem-acc window
    adjusted = []
    for i, (e, seq) in enumerate(zip(energies, seqs)):
        if not math.isfinite(e):
            adjusted.append(penalty_e)
            continue
        within_acc = (e - ref_energy) <= chem_acc
        within_acc_loose = (e - ref_energy) <= chem_acc_loose
        term = float(e)
        if lam > 0 and within_acc:
            cx = operator_pool.get_gate_count(list(seq)).get(cost_key, 0)
            term += lam * cx
        if lam_real_depth > 0 and within_acc_loose and real_depths is not None:
            term += lam_real_depth * math.log10(max(real_depths[i], 1))
        if lam_dim > 0 and within_acc and dims_used is not None:
            term += lam_dim * math.log10(max(dims_used[i], 1))
        if lam_excitation > 0 and within_acc_loose and higher_rank_fracs is not None:
            term += lam_excitation * higher_rank_fracs[i]
        if lam_hf > 0 and hf_fracs is not None:
            # UNGATED: share of right-spin-sector shots that are exactly the Hartree-Fock determinant. On a noisy device the
            # accuracy gates rarely open, and a circuit that does nothing is the cheapest way to get clean shots.
            term += lam_hf * hf_fracs[i]
        if lam_spin > 0 and spin_imbalances is not None:
            # UNGATED: |alpha-only - beta-only| share of the right-sector shots; a closed-shell state excites both spin channels equally
            term += lam_spin * spin_imbalances[i]
        if lam_coverage > 0 and within_acc_loose and idle_fracs is not None:
            term += lam_coverage * idle_fracs[i]
        if lam_overshoot > 0 and depth_ceiling is not None and real_depths is not None:
            overshoot_ratio = real_depths[i] / depth_ceiling
            if overshoot_ratio > 1.0:
                term += lam_overshoot * math.log10(overshoot_ratio)
        adjusted.append(term)
    return adjusted
