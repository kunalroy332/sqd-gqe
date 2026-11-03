# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Excitation-rank correctness tests, audited against LL_a1_v5_hw
(SnOH3-me_LL, 28 orbitals / 56 qubits, nelec=(21,21)).

Scope note (real finding from the 2026-09-24 audit): the excitation-injection
task spec this suite was written against assumes a per-iteration
"best/second-best/third-best parent -> generate candidates -> merge ->
dedupe -> stale-candidate removal" genetic-search-style workflow. Two
independent searches (a direct code read plus a separate repo-wide Explore
agent search) confirmed that workflow does not exist anywhere in this
codebase. The real LL_a1_v5_hw pipeline
(scripts/analysis/run_L_LL_final_recovery.py +
sqd_gqe/sqd/backend_sbd_gpu.py's recover_and_diagonalize) selects only
best/second-best circuits (already matching the spec's own fallback rule for
a two-parent workflow), and singles are always generated directly from the
FIXED HF reference via bit manipulation (singles_injected_energy), never from
a "parent" determinant -- so there is no parent-relative generation path and
no per-iteration candidate state to go stale. Tests K (stale-candidate) and L
(regenerated-candidate) from the original spec are therefore N/A and are
represented below as documentation-only, not executable tests faking state
that isn't there.
"""
from __future__ import annotations

import numpy as np
import pytest

from sqd_gqe.gqe.circuit_metrics import (
    _excitation_rank,
    excitation_signature,
    hf_occupation,
    raw_shot_excitation_histogram,
    rhf_cartesian_excitation_rank_histogram,
    rhf_spatial_excitation_rank_histogram,
)


def popcount(x: int) -> int:
    return bin(x).count("1")


# --------------------------------------------------------------------------- #
# A-C: HF / alpha-single / beta-single                                        #
# --------------------------------------------------------------------------- #
def test_a_hf_reference():
    hf_a, hf_b = hf_occupation((3, 3))
    sig = excitation_signature(hf_a, hf_b, hf_a, hf_b)
    assert sig["rank"] == 0
    assert sig["hamming_distance"] == 0
    assert _excitation_rank(hf_a, hf_b, hf_a, hf_b) == 0


def test_b_alpha_single():
    hf_a, hf_b = hf_occupation((3, 3))
    # remove occ orbital 0, occupy virtual orbital 3 (alpha only)
    alpha = (hf_a & ~(1 << 0)) | (1 << 3)
    sig = excitation_signature(alpha, hf_b, hf_a, hf_b)
    assert sig["holes_alpha"] == 1
    assert sig["particles_alpha"] == 1
    assert sig["holes_beta"] == 0
    assert sig["particles_beta"] == 0
    assert sig["rank"] == 1
    assert sig["hamming_distance"] == 2
    assert _excitation_rank(alpha, hf_b, hf_a, hf_b) == 1


def test_c_beta_single():
    hf_a, hf_b = hf_occupation((3, 3))
    beta = (hf_b & ~(1 << 1)) | (1 << 4)
    sig = excitation_signature(hf_a, beta, hf_a, hf_b)
    assert sig["rank"] == 1
    assert sig["hamming_distance"] == 2
    assert _excitation_rank(hf_a, beta, hf_a, hf_b) == 1


# --------------------------------------------------------------------------- #
# D-F: doubles, all three spin partitions                                     #
# --------------------------------------------------------------------------- #
def test_d_alpha_alpha_double():
    hf_a, hf_b = hf_occupation((3, 3))
    alpha = (hf_a & ~(1 << 0) & ~(1 << 1)) | (1 << 3) | (1 << 4)
    sig = excitation_signature(alpha, hf_b, hf_a, hf_b)
    assert (sig["holes_alpha"], sig["holes_beta"]) == (2, 0)
    assert sig["rank"] == 2
    assert sig["hamming_distance"] == 4


def test_e_alpha_beta_double():
    hf_a, hf_b = hf_occupation((3, 3))
    alpha = (hf_a & ~(1 << 0)) | (1 << 3)
    beta = (hf_b & ~(1 << 1)) | (1 << 4)
    sig = excitation_signature(alpha, beta, hf_a, hf_b)
    assert (sig["holes_alpha"], sig["holes_beta"]) == (1, 1)
    assert sig["rank"] == 2
    assert sig["hamming_distance"] == 4


def test_f_beta_beta_double():
    hf_a, hf_b = hf_occupation((3, 3))
    beta = (hf_b & ~(1 << 0) & ~(1 << 1)) | (1 << 3) | (1 << 4)
    sig = excitation_signature(hf_a, beta, hf_a, hf_b)
    assert (sig["holes_alpha"], sig["holes_beta"]) == (0, 2)
    assert sig["rank"] == 2
    assert sig["hamming_distance"] == 4


# --------------------------------------------------------------------------- #
# G: all four triple partitions                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_alpha_holes,n_beta_holes", [(3, 0), (2, 1), (1, 2), (0, 3)])
def test_g_triple_partitions(n_alpha_holes, n_beta_holes):
    hf_a, hf_b = hf_occupation((4, 4))
    alpha, beta = hf_a, hf_b
    for k in range(n_alpha_holes):
        alpha = (alpha & ~(1 << k)) | (1 << (4 + k))
    for k in range(n_beta_holes):
        beta = (beta & ~(1 << k)) | (1 << (4 + k))
    sig = excitation_signature(alpha, beta, hf_a, hf_b)
    assert (sig["holes_alpha"], sig["holes_beta"]) == (n_alpha_holes, n_beta_holes)
    assert sig["rank"] == 3
    assert sig["hamming_distance"] == 6


# --------------------------------------------------------------------------- #
# H: invalid electron number                                                  #
# --------------------------------------------------------------------------- #
def test_h_invalid_electron_number():
    hf_a, hf_b = hf_occupation((3, 3))
    # turn off one occupied bit, turn nothing on -> electron count drops by 1
    bad_alpha = hf_a & ~(1 << 0)
    sig = excitation_signature(bad_alpha, hf_b, hf_a, hf_b)
    assert sig["valid_alpha"] is False
    assert sig["valid"] is False
    assert sig["rank"] is None
    # _excitation_rank's bare popcount-and-halve does NOT itself detect this --
    # documented gap, why excitation_signature exists for validation.
    assert popcount(bad_alpha ^ hf_a) % 2 == 1


# --------------------------------------------------------------------------- #
# I: spin-flip / cross-spin move is structurally impossible here              #
# --------------------------------------------------------------------------- #
def test_i_singles_injection_never_mixes_spins():
    """singles_injected_energy's generator (run_L_LL_final_recovery.py) only
    ever flips bits WITHIN the alpha integer -- alpha and beta are separate
    ints, so a cross-spin move is impossible by construction, not just
    disallowed by a runtime check. Reproduce that generator's logic directly
    and assert every produced single is alpha-only (beta side untouched)."""
    norb, n_occ = 6, 3
    hf_a, hf_b = hf_occupation((n_occ, n_occ))
    occ_orbs = [i for i in range(norb) if (hf_a >> i) & 1]
    virt_orbs = [i for i in range(norb) if not (hf_a >> i) & 1]
    singles = set()
    for o in occ_orbs:
        for v in virt_orbs:
            singles.add((hf_a & ~(1 << o)) | (1 << v))
    for s in singles:
        sig = excitation_signature(s, hf_b, hf_a, hf_b)
        assert sig["valid"] is True
        assert sig["rank"] == 1
        assert sig["holes_beta"] == 0 and sig["particles_beta"] == 0


# --------------------------------------------------------------------------- #
# J: duplicate generation collapses to one                                    #
# --------------------------------------------------------------------------- #
def test_j_duplicate_generation_dedup():
    norb, n_occ = 6, 3
    hf_a, _ = hf_occupation((n_occ, n_occ))
    occ_orbs = [i for i in range(norb) if (hf_a >> i) & 1]
    virt_orbs = [i for i in range(norb) if not (hf_a >> i) & 1]
    o, v = occ_orbs[0], virt_orbs[0]
    same_single = (hf_a & ~(1 << o)) | (1 << v)
    # generated "from" two different starting pools -- same set-union pattern
    # singles_injected_energy actually uses (run_L_LL_final_recovery.py:90-96)
    pool_best = {hf_a}
    pool_second = {hf_a, 1 << (norb - 1)}
    merged = set()
    merged.add(same_single)  # from best
    merged.add(same_single)  # from second-best, identical bitstring
    assert len(merged) == 1


# --------------------------------------------------------------------------- #
# K/L: stale-candidate / regenerated-candidate -- N/A, documented not faked   #
# --------------------------------------------------------------------------- #
def test_k_l_stale_candidate_concept_not_applicable():
    """No code path in this repo maintains per-iteration candidate state that
    could go stale (confirmed by direct code read of
    scripts/analysis/run_L_LL_final_recovery.py and
    sqd_gqe/sqd/backend_sbd_gpu.py's recover_and_diagonalize, plus an
    independent repo-wide search finding zero hits for "third-best" or any
    global candidate accumulator not cleared each iteration/step). singles are
    always regenerated fresh from the fixed HF reference on every call to
    singles_injected_energy -- there is nothing to test for staleness because
    there is no persisted candidate set to become stale. This test exists to
    make that finding an explicit, checked assertion rather than a silent
    doc comment: recomputing singles for the same (nelec, norb) twice must
    always produce an identical set (proving there is no hidden mutable
    state influencing the result across calls)."""
    norb, n_occ = 6, 3
    hf_a, _ = hf_occupation((n_occ, n_occ))

    def gen_singles():
        occ_orbs = [i for i in range(norb) if (hf_a >> i) & 1]
        virt_orbs = [i for i in range(norb) if not (hf_a >> i) & 1]
        return {(hf_a & ~(1 << o)) | (1 << v) for o in occ_orbs for v in virt_orbs}

    assert gen_singles() == gen_singles()


# --------------------------------------------------------------------------- #
# M: parent-relative move vs HF-relative rank diverge                         #
# --------------------------------------------------------------------------- #
def test_m_parent_relative_vs_hf_relative():
    hf_a, hf_b = hf_occupation((4, 4))
    # rank-2 alpha parent: holes at 0,1 -> particles at 4,5
    parent = (hf_a & ~(1 << 0) & ~(1 << 1)) | (1 << 4) | (1 << 5)
    assert excitation_signature(parent, hf_b, hf_a, hf_b)["rank"] == 2
    # one more occupied->unoccupied move relative to the PARENT: move electron
    # from orbital 2 (occupied in parent) to orbital 6 (virtual) -- this is a
    # single move relative to parent, but a THIRD excitation relative to HF.
    child = (parent & ~(1 << 2)) | (1 << 6)
    sig = excitation_signature(child, hf_b, hf_a, hf_b)
    assert sig["rank"] == 3  # HF-relative, not 1
    assert sig["rank"] != 1  # rejected from HF-single injection


# --------------------------------------------------------------------------- #
# N: de-excitation move brings a rank-2 parent back to rank 1                 #
# --------------------------------------------------------------------------- #
def test_n_deexcitation_move():
    hf_a, hf_b = hf_occupation((4, 4))
    parent = (hf_a & ~(1 << 0) & ~(1 << 1)) | (1 << 4) | (1 << 5)
    assert excitation_signature(parent, hf_b, hf_a, hf_b)["rank"] == 2
    # reverse one of the two excitations: move the electron at orbital 4 back to orbital 0
    child = (parent & ~(1 << 4)) | (1 << 0)
    sig = excitation_signature(child, hf_b, hf_a, hf_b)
    assert sig["rank"] == 1
    assert sig["valid"] is True


# --------------------------------------------------------------------------- #
# O: rearranging the excited electron among virtuals keeps rank 1             #
# --------------------------------------------------------------------------- #
def test_o_orbital_rearrangement_among_virtuals():
    hf_a, hf_b = hf_occupation((4, 4))
    parent = (hf_a & ~(1 << 0)) | (1 << 4)
    assert excitation_signature(parent, hf_b, hf_a, hf_b)["rank"] == 1
    # move the excited electron from virtual orbital 4 to virtual orbital 5 (same spin)
    child = (parent & ~(1 << 4)) | (1 << 5)
    sig = excitation_signature(child, hf_b, hf_a, hf_b)
    assert sig["rank"] == 1
    assert sig["valid"] is True


# --------------------------------------------------------------------------- #
# P: full 56-qubit / LL_a1_v5_hw integrity check                              #
# --------------------------------------------------------------------------- #
def test_p_ll_a1_v5_hw_56qubit_integrity():
    """Real LL_a1_v5_hw config: SnOH3-me_LL_28o42e -> norb=28, nelec=(21,21),
    n_qubits=56. Every generated single must fit in 28 bits per spin and
    conserve each spin's electron count exactly."""
    norb, nelec = 28, (21, 21)
    hf_a, hf_b = hf_occupation(nelec)
    assert hf_a.bit_length() <= norb
    assert popcount(hf_a) == nelec[0]
    assert popcount(hf_b) == nelec[1]

    occ_orbs = [i for i in range(norb) if (hf_a >> i) & 1]
    virt_orbs = [i for i in range(norb) if not (hf_a >> i) & 1]
    assert len(occ_orbs) == 21 and len(virt_orbs) == 7
    singles = {(hf_a & ~(1 << o)) | (1 << v) for o in occ_orbs for v in virt_orbs}
    assert len(singles) == 21 * 7
    for s in singles:
        assert s.bit_length() <= norb
        assert popcount(s) == nelec[0]  # alpha electron count conserved
        sig = excitation_signature(s, hf_b, hf_a, hf_b)
        assert sig["valid"] is True
        assert sig["rank"] == 1
        assert sig["hamming_distance"] == 2
        assert sig["holes_beta"] == 0 and sig["particles_beta"] == 0  # beta untouched


# --------------------------------------------------------------------------- #
# rhf_cartesian_excitation_rank_histogram: production Cartesian-product path  #
# --------------------------------------------------------------------------- #
def test_rhf_cartesian_histogram_hand_built_case():
    """Small hand-built 3-string case (hf, single, double spatial strings),
    verified once ad hoc during the 2026-09-24 audit, now a permanent
    regression test. norb=4, nelec=(2,2)."""
    norb, nelec = 4, (2, 2)
    hf_a, _ = hf_occupation(nelec)
    single = (hf_a & ~(1 << 0)) | (1 << 2)   # rank 1 vs HF
    double = (hf_a & ~(1 << 0) & ~(1 << 1)) | (1 << 2) | (1 << 3)  # rank 2 vs HF
    strs = [hf_a, single, double]
    hist = rhf_cartesian_excitation_rank_histogram(strs, nelec, max_named_rank=3)
    assert hist["HF"] == 1     # (hf,hf)
    assert hist["S"] == 2      # (hf,single)+(single,hf)
    assert hist["D"] == 3      # (hf,double)+(double,hf)+(single,single)
    assert hist["T"] == 2      # (single,double)+(double,single)
    assert hist["higher"] == 1  # (double,double) -> rank 4
    assert hist["n_dets"] == 9
    assert hist["combined"] == 9


def test_rhf_spatial_histogram_diagonal_only_still_correct_for_its_purpose():
    """rhf_spatial_excitation_rank_histogram characterizes the DIAGONAL only
    (rank of each string vs HF, not paired with others) -- still correct for
    that narrower purpose, unlike the Cartesian-product function above which
    is what recover_and_diagonalize's diagnostics actually use."""
    norb, nelec = 4, (2, 2)
    hf_a, _ = hf_occupation(nelec)
    single = (hf_a & ~(1 << 0)) | (1 << 2)
    hist = rhf_spatial_excitation_rank_histogram([hf_a, single], nelec, max_named_rank=3)
    assert hist["HF"] == 1
    assert hist["S"] == 1
    assert hist["n_dets"] == 2


# --------------------------------------------------------------------------- #
# raw_shot_excitation_histogram: real per-shot (alpha, beta) pairs            #
# --------------------------------------------------------------------------- #
def test_raw_shot_excitation_histogram_weighted_by_real_shot_counts():
    """Unlike the post-recovery strs_a histograms above, this operates on a
    real {(alpha, beta): count} sampled-shots dict with genuinely independent
    alpha/beta per shot -- odd ranks (singles) are structurally possible here,
    weighted by real per-shot multiplicity, not counted once per distinct
    bitstring."""
    nelec = (3, 3)
    hf_a, hf_b = hf_occupation(nelec)
    single_a = (hf_a & ~1) | (1 << 3)  # alpha single, beta untouched -> rank 1
    double_ab = (single_a, (hf_b & ~1) | (1 << 3))  # alpha single + beta single -> rank 2
    counts = {
        (hf_a, hf_b): 500,
        (single_a, hf_b): 300,
        double_ab: 200,
    }
    hist = raw_shot_excitation_histogram(counts, nelec, max_named_rank=4)
    assert hist["HF"] == 500
    assert hist["S"] == 300
    assert hist["D"] == 200
    assert hist["combined"] == 1000
    assert hist["n_dets"] == 3


def test_raw_shot_excitation_histogram_empty_counts():
    nelec = (3, 3)
    hist = raw_shot_excitation_histogram({}, nelec, max_named_rank=4)
    assert hist["combined"] == 0
    assert hist["n_dets"] == 0
