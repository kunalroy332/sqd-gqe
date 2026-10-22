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

"""Import the Gaussian MP2/def2-SVP natural-orbital fchk for Sn(OH)3-Me and
rebuild the same system in pyscf, for later active-space / FCIDUMP work.

The source fchk (molecules/large/SnOH3-me_s0_mp2_def2svp_NO.fchk) has a
2-word header line ("Guess=OnlyRMP2-FC   def2SVP") instead of the normal
3-word "<command> <lot> <basis>" Gaussian fchk convention, which iodata's
fchk reader requires -- so we patch a 3-word header into a scratch copy
under .cache/fchk_patched/ before loading (original file untouched).

AO ordering: iodata parses fchk's basis into its own convention (pure d
shells as [c0,c1,s1,c2,s2] = m=0,+1,-1,+2,-2). pyscf's def2-svp d shells are
ordered [xy,yz,z2,xz,x2-y2] = m=-2,-1,0,+1,+2 (checked directly via
mol.ao_labels() below) -- same real solid harmonics, different m order, so
importing the Gaussian NO coefficients needs a per-shell permutation, not a
straight copy. This basis (def2-svp on Sn/O/C/H) has no shells past l=2, so
the permutation only ever touches d blocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from iodata import load_one
from iodata.convert import convert_conventions
from pyscf import gto

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_FCHK = REPO_ROOT / "molecules" / "large" / "SnOH3-me_s0_mp2_def2svp_NO.fchk"
PATCHED_FCHK = REPO_ROOT / ".cache" / "fchk_patched" / SRC_FCHK.name

# pyscf's own real-spherical-harmonic AO ordering, keyed the same way iodata
# keys its per-(angmom, kind) conventions. Verified against
# gto.M(atom='O 0 0 0', basis='def2-svp', ...).ao_labels() -- pyscf orders
# pure d as m=-2,-1,0,+1,+2 (xy,yz,z2,xz,x2-y2), s/p/cartesian shells match
# iodata's convention already (no permutation needed there).
PYSCF_CONVENTIONS = {
    (0, "c"): ["1"],
    (1, "c"): ["x", "y", "z"],
    (2, "p"): ["s2", "s1", "c0", "c1", "c2"],
}


def patch_header(src: Path, dst: Path) -> None:
    """Copy fchk with a 3-word 2nd line so iodata's fchk reader accepts it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = src.read_text().splitlines(keepends=True)
    lines[1] = "SP                                                        RMP2      def2SVP\n"
    dst.write_text("".join(lines))


def load_snoh3me():
    """Return (iodata IOData, pyscf gto.Mole) for the Sn(OH)3-Me fchk.

    The Mole is built fresh from the fchk geometry with def2-svp (all
    electron -- no ECP here; that's a separate comparison run, see
    snoh3_hf_ecp.py) so its AO ordering is pyscf's own and directly
    comparable to the permuted MO coefficients returned alongside it.
    """
    if not PATCHED_FCHK.exists() or PATCHED_FCHK.stat().st_mtime < SRC_FCHK.stat().st_mtime:
        patch_header(SRC_FCHK, PATCHED_FCHK)

    iodata_mol = load_one(str(PATCHED_FCHK))

    atom = [
        [int(num), tuple(coord)]
        for num, coord in zip(iodata_mol.atnums, iodata_mol.atcoords)
    ]
    mol = gto.M(
        atom=atom,
        unit="Bohr",  # iodata atcoords are already in bohr (iodata's internal unit)
        basis="def2-svp",
        ecp={"Sn": "def2-svp"},  # matches the fchk's 58-electron count (Sn core replaced)
        charge=int(round(iodata_mol.charge)),
        spin=0,
    )

    permutation, signs = convert_conventions(iodata_mol.obasis, PYSCF_CONVENTIONS)
    mo_coeffs_pyscf_order = iodata_mol.mo.coeffs[permutation] * signs[:, None]
    mo_occs = iodata_mol.mo.occs

    return iodata_mol, mol, mo_coeffs_pyscf_order, mo_occs


if __name__ == "__main__":
    iodata_mol, mol, mo_coeff, mo_occ = load_snoh3me()
    print(f"natm={mol.natm} nao={mol.nao} nelectron={mol.nelectron}")
    print(f"mo_coeff shape={mo_coeff.shape}, occ sum={mo_occ.sum():.1f}")
    print("atom labels:", [mol.atom_symbol(i) for i in range(mol.natm)])
