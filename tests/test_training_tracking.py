# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Checks of sqd_gqe.gqe.training_tracking on hand-built examples (no torch, no ffsim needed)."""
from types import SimpleNamespace as NS

from sqd_gqe.gqe.training_tracking import iteration_tracking, sequence_stats, spin_excitation_stats

# 6 orbitals, 2 alpha + 2 beta electrons: HF = 0b0011 per spin, HOMO = orbital 1, LUMO = orbital 2
NELEC, NORB = (2, 2), 6
HF = 0b0011
HL = 0b0101                      # orbital 1 -> orbital 2


def test_valid_and_wrong_sectors():
    counts = {(HF, HF): 50,                 # Hartree-Fock
              (HL, HF): 10,                 # alpha HOMO -> LUMO
              (HF, HL): 10,                 # beta HOMO -> LUMO
              (HL, HL): 5,                  # both
              (0b0111, 0b0001): 15,         # an electron moved from the beta to the alpha channel
              (0b0001, 0b0001): 10}         # an electron lost
    s = spin_excitation_stats(counts, NELEC, NORB)
    assert abs(s["frac_valid"] - 0.75) < 1e-12
    assert abs(s["frac_spin_transfer"] - 0.15) < 1e-12
    assert abs(s["frac_number_error"] - 0.10) < 1e-12
    assert abs(s["valid_hf"] - 50 / 75) < 1e-12
    assert abs(s["valid_homo_lumo_alpha"] - 10 / 75) < 1e-12
    assert abs(s["valid_homo_lumo_beta"] - 10 / 75) < 1e-12
    assert abs(s["valid_homo_lumo_alpha_beta"] - 5 / 75) < 1e-12
    assert abs(s["valid_alpha_only"] - 10 / 75) < 1e-12 and abs(s["valid_beta_only"] - 10 / 75) < 1e-12
    assert abs(s["valid_alpha_and_beta"] - 5 / 75) < 1e-12
    assert s["alpha_beta_asymmetry"] == 0.0


def test_same_spin_vs_opposite_spin_double():
    aa = 0b1100                               # both alpha electrons out of the HF orbitals
    s = spin_excitation_stats({(aa, HF): 1, (HL, HL): 1}, NELEC, NORB)
    assert abs(s["valid_double_alpha_alpha"] - 0.5) < 1e-12
    assert abs(s["valid_double_alpha_beta"] - 0.5) < 1e-12


def _pool():
    names = ["I", "LUCJ(reps=1)", "orb_rot[0]", "givens(0,1;0.1)", "givens(1,2;0.1)", "givens(3,4;0.1)", "diag_coulomb(t=0.100)",
             "num_num(1,2)"]
    return [NS(name=n, support=(1, 2) if n.startswith("num_num") else ()) for n in names]


def test_sequence_roles():
    # (0,1): occupied-occupied, (1,2): HOMO-LUMO, (3,4): virtual-virtual
    st = sequence_stats([3, 4, 4, 5, 0, 1], _pool(), NELEC, NORB)
    assert st["n_givens"] == 4 and st["n_givens_homo_lumo"] == 2
    assert st["n_givens_occ_occ"] == 1 and st["n_givens_virt_virt"] == 1
    assert st["n_identity"] == 1 and st["n_lucj"] == 1
    assert st["n_hf_changing_gates"] == 1 + 2        # LUCJ + the two HOMO-LUMO Givens
    assert st["longest_run"] == 2


def test_iteration_summary():
    counts = [{(HF, HF): 1}, {(HL, HL): 1}]
    summary, per = iteration_tracking([[0, 3, 4], [0, 5, 1]], counts, _pool(), NELEC, NORB, energies=[-1.0, -2.0], depths=[10, 20],
                                      two_qs=[5, 6], idles=[0.0, 0.0], token_entropy=1.5, best_i=1, beta=0.5)
    assert summary["best_idx"] == 1 and len(per) == 2
    assert abs(summary["policy_effective_vocab"] - 4.4816890703380645) < 1e-9
    assert per[0]["seq"] == [3, 4]
    assert summary["exc_best"]["valid_homo_lumo_alpha_beta"] == 1.0
