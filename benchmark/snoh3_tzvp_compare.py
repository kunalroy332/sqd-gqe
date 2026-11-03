# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""def2-TZVP HF/MP2 on the Sn(OH)3-Me geometry (same geometry as the def2-SVP
fchk, same def2-ECP on Sn), for basis-set comparison against the def2-SVP
baseline in snoh3_mp2_baseline.py.
"""

from __future__ import annotations

from pyscf import gto, mp, scf

from benchmark.snoh3_fchk_import import load_snoh3me

BASIS = "def2-tzvp"


def run_tzvp():
    iodata_mol, mol_svp, _, _ = load_snoh3me()

    atom = [
        [mol_svp.atom_symbol(i), mol_svp.atom_coord(i)]
        for i in range(mol_svp.natm)
    ]
    mol = gto.M(
        atom=atom,
        unit="Bohr",
        basis=BASIS,
        ecp={"Sn": BASIS},
        charge=mol_svp.charge,
        spin=0,
    )
    print(f"{BASIS}: nao={mol.nao} nelectron={mol.nelectron}")

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"RHF/{BASIS} (+def2-ECP on Sn) energy: {mf.e_tot:.10f} Ha")

    mp2 = mp.MP2(mf)
    mp2.kernel()
    print(f"MP2 correlation energy:              {mp2.e_corr:.10f} Ha")
    print(f"MP2/{BASIS} total energy:             {mf.e_tot + mp2.e_corr:.10f} Ha")

    return mf, mp2


if __name__ == "__main__":
    run_tzvp()
