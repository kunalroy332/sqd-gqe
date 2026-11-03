# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""S/M/L/LL active spaces exactly as selected in Mitsubishi's MP2-NO-based
report (molecules/large/QC_2026-08-07_calc_MP2-NO-based-AS-Selection-SnOH3Me.pdf,
page 2), built on the imported Gaussian MP2 natural orbitals -- not fresh
pyscf HF canonical MOs (see benchmark/snoh3_make_fcidump.py for that,
smaller, HF-MO-based CAS(8e,8o) space).

PDF MO numbering is 1-indexed and spans MO9-48 (HOMO=MO29, LUMO=MO30);
pyscf's mo_coeff / mo_occ arrays are 0-indexed, so every boundary here is
PDF value minus 1. Verified against this repo's own pyscf HF/def2-svp run,
which reproduces HOMO=29, LUMO=30 (1-indexed) exactly -- see
docs/snoh3_me_fchk_pipeline.ipynb section 1.

    name  (norb, nelec)  MO range (1-idx, incl.)  MO range (0-idx, incl.)
    S     (16, 24)       18-33                    17-32
    M     (19, 24)       18-36                    17-35
    L     (23, 32)       14-36                    13-35
    LL    (28, 42)       9-36                     8-35

All four windows share the same upper bound (0-idx MO 35 = PDF MO-36): the
PDF's caption notes MP2-NO-based LL tops out 3 orbitals below the
HF-MO-based LL (MO-39) -- "a few higher NOs" beyond MO-36 were flagged in
the report as possibly worth adding, not yet included in any of these four.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
from pyscf import ao2mo, mcscf, scf
from pyscf.tools import fcidump

from benchmark.snoh3_fchk_import import load_snoh3me

REPO_ROOT = Path(__file__).resolve().parent.parent

# (norb, nelec, mo_lo_1indexed, mo_hi_1indexed_inclusive)
ACTIVE_SPACES = {
    "S":  (16, 24, 18, 33),
    "M":  (19, 24, 18, 36),
    "L":  (23, 32, 14, 36),
    "LL": (28, 42, 9, 36),
}

# size-bucket placement for SQD-GQE's molecules/{small,medium,large} convention
SIZE_DIR = {"S": "small", "M": "medium", "L": "large", "LL": "large"}

FCI_DIM_LIMIT = 50_000_000


def fci_dimension(norb: int, nelec: int) -> int:
    na = nb = nelec // 2
    return comb(norb, na) * comb(norb, nb)


def build_space(name: str, mf, mo_coeff_no: np.ndarray):
    """CAS one-/two-electron integrals for `name`'s MO window, in the MP2-NO basis.

    `mf` must already be SCF-converged (its mo_coeff is unused here -- only
    mf.mol and mc.get_h1cas/get_h2cas's dependence on the *passed* MO
    coefficients matter, so this works whether mf was converged in the HF-MO
    or MP2-NO basis).
    """
    norb, nelec, mo_lo_1, mo_hi_1 = ACTIVE_SPACES[name]
    lo0, hi0 = mo_lo_1 - 1, mo_hi_1 - 1  # inclusive, 0-indexed
    assert hi0 - lo0 + 1 == norb, f"{name}: MO window size mismatch"

    ncore = lo0  # everything below the window is doubly-occupied core
    na = nb = nelec // 2

    mc = mcscf.CASCI(mf, norb, (na, nb))
    mc.ncore = ncore
    h1, ecore = mc.get_h1cas(mo_coeff_no)
    h2 = ao2mo.restore(1, mc.get_h2cas(mo_coeff_no), norb)

    dim = fci_dimension(norb, nelec)
    return h1, h2, ecore, dim


def write_fcidump_from_ints(h1, h2, ecore, norb, nelec, out_path: Path):
    from pyscf.tools.fcidump import from_integrals

    out_path.parent.mkdir(parents=True, exist_ok=True)
    from_integrals(str(out_path), h1, h2, norb, nelec, nuc=ecore, ms=0)


if __name__ == "__main__":
    iodata_mol, mol, mo_coeff_no, mo_occ_no = load_snoh3me()
    mf = scf.RHF(mol)
    mf.kernel()

    print(f"{'space':4s} {'norb':>5s} {'nelec':>6s} {'MO range (1-idx)':>18s} {'FCI dim':>12s} {'tractable?':>11s} {'dest':>8s}")
    for name, (norb, nelec, lo1, hi1) in ACTIVE_SPACES.items():
        h1, h2, ecore, dim = build_space(name, mf, mo_coeff_no)
        tractable = "yes" if dim <= FCI_DIM_LIMIT else "no"
        dest = REPO_ROOT / "molecules" / SIZE_DIR[name] / f"SnOH3-me_{name}_{norb}o{nelec}e_mp2no_def2svp.fcidump"
        write_fcidump_from_ints(h1, h2, ecore, norb, nelec, dest)
        print(f"{name:4s} {norb:5d} {nelec:6d} {f'{lo1}-{hi1}':>18s} {dim:12.3e} {tractable:>11s} {SIZE_DIR[name]:>8s}")
        print(f"     -> {dest.relative_to(REPO_ROOT)}")
