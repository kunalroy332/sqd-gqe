# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Baseline RHF/MP2 def2-SVP energy for Sn(OH)3-Me, from the geometry in the
Gaussian fchk (molecules/large/SnOH3-me_s0_mp2_def2svp_NO.fchk).

The fchk is a natural-orbitals-only dump (no SCF/MP2 total energy fields), so
there's no Gaussian reference number to diff against directly -- instead we
cross-check the imported NO occupations (from snoh3_fchk_import.py) against
this run's own MP2 natural occupation numbers as an independent consistency
check that the geometry/basis/ECP setup matches what produced the fchk.
"""

from __future__ import annotations

import numpy as np
from pyscf import mp, scf

from benchmark.snoh3_fchk_import import load_snoh3me


def run_mp2_baseline():
    iodata_mol, mol, mo_coeff_fchk, mo_occ_fchk = load_snoh3me()

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"RHF/def2-SVP (+def2-ECP on Sn) energy: {mf.e_tot:.10f} Ha")

    mp2 = mp.MP2(mf)
    mp2.kernel()
    print(f"MP2 correlation energy:                {mp2.e_corr:.10f} Ha")
    print(f"MP2/def2-SVP total energy:              {mf.e_tot + mp2.e_corr:.10f} Ha")

    # Natural occupation numbers from this run's own MP2 1-RDM, for
    # cross-checking against the fchk's imported NO occupations (not a
    # direct energy comparison -- see module docstring).
    dm1 = mp2.make_rdm1(ao_repr=False)
    occ_this_run = np.sort(np.linalg.eigvalsh(dm1))[::-1]
    occ_fchk_sorted = np.sort(mo_occ_fchk)[::-1]
    print("\nTop 5 natural occupations, this run vs fchk:")
    for i in range(5):
        print(f"  this run: {occ_this_run[i]:.6f}   fchk: {occ_fchk_sorted[i]:.6f}")

    return mf, mp2


if __name__ == "__main__":
    run_mp2_baseline()
