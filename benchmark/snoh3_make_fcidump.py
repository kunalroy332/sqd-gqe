# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Active-space selection + FCIDUMP export for Sn(OH)3-Me, for SQD.

Active space chosen by inspecting RHF/def2-SVP orbital energies and AO
character around the Fermi level (see session notes / docs notebook):
large HOMO-LUMO gap (~15 eV), HOMO is the Sn-C sigma bond, LUMO/LUMO+1/+2
are Sn-centered empty p ("Rydberg-like", the classic low-lying empty-p
signature of a heavy p-block center), MOs just below HOMO are O
lone-pair / O-Sn sigma character.

CAS(8e,8o): 4 occupied MOs (O lone pairs + Sn-C sigma) + 4 virtual MOs
(Sn empty p manifold) -- a first cut meant to be revisited once SQD results
come back, not a final answer.
"""

from __future__ import annotations

from pathlib import Path

from pyscf import mcscf, scf
from pyscf.tools import fcidump

from benchmark.snoh3_fchk_import import load_snoh3me

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "molecules" / "large"

NCAS = 8
NELECAS = 8


def build_active_space(basis="def2-svp"):
    iodata_mol, mol_svp, _, _ = load_snoh3me()
    mol = mol_svp if basis == "def2-svp" else None
    if mol is None:
        raise NotImplementedError("only def2-svp geometry wired up here; see snoh3_tzvp_compare.py for TZVP")

    mf = scf.RHF(mol)
    mf.kernel()

    nocc = mol.nelectron // 2
    ncore = nocc - NCAS // 2
    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    mc.kernel()
    print(f"CASCI({NELECAS}e,{NCAS}o) energy: {mc.e_tot:.10f} Ha (ncore={ncore})")

    return mf, mc


def write_fcidump(mc, out_path: Path):
    fcidump.from_mcscf(mc, str(out_path))
    print(f"FCIDUMP written to {out_path}")


if __name__ == "__main__":
    mf, mc = build_active_space()
    out_path = OUT_DIR / "SnOH3-me_cas8e8o_def2svp.fcidump"
    write_fcidump(mc, out_path)
