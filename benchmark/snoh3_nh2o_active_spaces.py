# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""S/M/L/LL active spaces for the Sn(OH)3-Me-nH2O (n=1,2) hydrate fchk's,
exactly as selected in Mitsubishi's MP2-NO-based report
(molecules/large/QC_2026-09-20_calc_MP2-NO-based-AS-Selection-SnOH3Me-nH2O.pdf,
pages 2 and 3), built on the imported Gaussian MP2 natural orbitals -- same
convention as benchmark/snoh3_pdf_active_spaces.py (the no-water molecule),
generalized here to two source fchk's instead of one.

Unlike the original SnOH3-me_s0_mp2_def2svp_NO.fchk (2-word Gaussian header,
needs benchmark/snoh3_fchk_import.py's header patch before iodata will load
it), these two hydrate fchk's already have a normal 3-word header line and
load directly -- no patching needed.

PDF MO numbering is 1-indexed; pyscf's mo_coeff/mo_occ arrays are 0-indexed,
so every boundary here is PDF value minus 1.

    Sn(OH)3-Me-H2O (page 2):
    name  (norb, nelec)  MO range (1-idx, incl.)  MO range (0-idx, incl.)
    S     (19, 30)       20-38                    19-37
    M     (23, 30)       20-42                    19-41
    L     (28, 40)       15-42                    14-41
    LL    (33, 50)       10-42                    9-41

    Sn(OH)3-Me-2H2O (page 3):
    name  (norb, nelec)  MO range (1-idx, incl.)  MO range (0-idx, incl.)
    S     (22, 36)       22-43                    21-42
    M     (29, 36)       22-50                    21-49
    L     (35, 48)       16-50                    15-49
    LL    (40, 58)       11-50                    10-49

FCI dimensions (C(norb,nelec/2)^2): only Sn(OH)3-Me-H2O's S space
(1.5e7) is under FCI_DIM_LIMIT (5e7) -- every other space here (including
2H2O's S, at 5.35e7, just over the limit) needs CCSD(T) or an SQD
self-convergence reference instead of casci_energy(), same situation as the
no-water molecule's L/LL spaces.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
from iodata import load_one
from iodata.convert import convert_conventions
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.tools.fcidump import from_integrals

REPO_ROOT = Path(__file__).resolve().parent.parent
BOHR2ANG = 0.529177210903

# same pyscf pure-d-shell convention as snoh3_fchk_import.py -- this basis
# (def2-svp on Sn/O/C/H) has no shells past l=2, so only d blocks are touched.
PYSCF_CONVENTIONS = {
    (0, "c"): ["1"],
    (1, "c"): ["x", "y", "z"],
    (2, "p"): ["s2", "s1", "c0", "c1", "c2"],
}

# (norb, nelec, mo_lo_1indexed, mo_hi_1indexed_inclusive)
MOLECULES = {
    "SnOH3-me-h2o": {
        "fchk": "SnOH3-me-h2o_s0_mp2_def2svp_FC.fchk",
        "spaces": {
            "S":  (19, 30, 20, 38),
            "M":  (23, 30, 20, 42),
            "L":  (28, 40, 15, 42),
            "LL": (33, 50, 10, 42),
        },
    },
    "SnOH3-me-2h2o": {
        "fchk": "SnOH3-me-2h2o_s0_mp2_def2svp_FC.fchk",
        "spaces": {
            "S":  (22, 36, 22, 43),
            "M":  (29, 36, 22, 50),
            "L":  (35, 48, 16, 50),
            "LL": (40, 58, 11, 50),
        },
    },
}

# size-bucket placement for SQD-GQE's molecules/{small,medium,large} convention
SIZE_DIR = {"S": "small", "M": "medium", "L": "large", "LL": "large"}

FCI_DIM_LIMIT = 50_000_000


def load_fchk(fchk_name: str):
    """Return (iodata IOData, pyscf gto.Mole, mo_coeff_pyscf_order, mo_occ).

    Mirrors benchmark/snoh3_fchk_import.py's load_snoh3me(), generalized to
    an arbitrary fchk filename under molecules/large/ -- no header patch
    needed here (these two files already have a standard 3-word header).
    """
    src = REPO_ROOT / "molecules" / "large" / fchk_name
    iodata_mol = load_one(str(src))

    atom = [
        [int(num), tuple(coord)]
        for num, coord in zip(iodata_mol.atnums, iodata_mol.atcoords)
    ]
    mol = gto.M(
        atom=atom,
        unit="Bohr",  # iodata atcoords are already in bohr
        basis="def2-svp",
        ecp={"Sn": "def2-svp"},
        charge=int(round(iodata_mol.charge)),
        spin=0,
    )

    permutation, signs = convert_conventions(iodata_mol.obasis, PYSCF_CONVENTIONS)
    mo_coeff_pyscf_order = iodata_mol.mo.coeffs[permutation] * signs[:, None]
    mo_occ = iodata_mol.mo.occs

    return iodata_mol, mol, mo_coeff_pyscf_order, mo_occ


def fci_dimension(norb: int, nelec: int) -> int:
    na = nb = nelec // 2
    return comb(norb, na) * comb(norb, nb)


def build_space(mf, mo_coeff_no: np.ndarray, norb: int, nelec: int,
                mo_lo_1: int, mo_hi_1: int):
    lo0, hi0 = mo_lo_1 - 1, mo_hi_1 - 1  # inclusive, 0-indexed
    assert hi0 - lo0 + 1 == norb, f"MO window size mismatch: {hi0 - lo0 + 1} != {norb}"

    ncore = lo0
    na = nb = nelec // 2

    mc = mcscf.CASCI(mf, norb, (na, nb))
    mc.ncore = ncore
    h1, ecore = mc.get_h1cas(mo_coeff_no)
    h2 = ao2mo.restore(1, mc.get_h2cas(mo_coeff_no), norb)

    dim = fci_dimension(norb, nelec)
    return h1, h2, ecore, dim


def write_fcidump_from_ints(h1, h2, ecore, norb, nelec, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from_integrals(str(out_path), h1, h2, norb, nelec, nuc=ecore, ms=0)


def write_yaml(mol_tag: str, name: str, norb: int, nelec: int,
                mo_lo_1: int, mo_hi_1: int, fchk_name: str, mol,
                fci_dim: float, out_path: Path):
    geom_lines = []
    for i, c in enumerate(mol.atom_coords(unit="Angstrom")):
        sym = mol.atom_symbol(i)
        geom_lines.append(f"    ['{sym}', [{c[0]:.8f}, {c[1]:.8f}, {c[2]:.8f}]],")
    geom_block = "\n".join(geom_lines)

    fcidump_rel = f"molecules/{SIZE_DIR[name]}/{mol_tag}_{name}_{norb}o{nelec}e_mp2no_def2svp.fcidump"
    tractable = fci_dim <= FCI_DIM_LIMIT

    if tractable:
        fci_note = f"# FCI dim = {fci_dim:.3e} -- under FCI_DIM_LIMIT (5e7), casci_energy() works."
    else:
        fci_note = (f"# FCI dim = {fci_dim:.3e} -- exceeds FCI_DIM_LIMIT (5e7), so\n"
                     f"# casci_energy() raises FciIntractableError here; use ccsd_t_energy() or an\n"
                     f"# SQD self-convergence reference instead.")

    content = f"""# {mol_tag}, active space "{name}" ({norb}o,{nelec}e) from Mitsubishi's MP2-NO-based
# selection (molecules/large/QC_2026-09-20_calc_MP2-NO-based-AS-Selection-SnOH3Me-nH2O.pdf,
# MO{mo_lo_1}-{mo_hi_1}, 1-indexed). Built on the imported Gaussian MP2 natural
# orbitals from {fchk_name} (not fresh HF canonical MOs) -- see
# benchmark/snoh3_nh2o_active_spaces.py.
#
{fci_note}
_target_: sqd_gqe.core.system.MolecularSystem.from_fcidump
fcidump_path: {fcidump_rel}
geometry:
  type: custom
  geometry: [
{geom_block}
  ]
basis: 'def2-svp'
ecp:
  Sn: 'def2-svp'
spin: 0
charge: 0
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)


if __name__ == "__main__":
    print(f"{'mol':16s} {'space':5s} {'norb':>5s} {'nelec':>6s} "
          f"{'MO range (1-idx)':>18s} {'FCI dim':>12s} {'tractable?':>11s} {'dest':>8s}")
    for mol_tag, info in MOLECULES.items():
        iodata_mol, mol, mo_coeff_no, mo_occ_no = load_fchk(info["fchk"])
        mf = scf.RHF(mol)
        mf.kernel()
        assert mf.converged, f"{mol_tag}: RHF did not converge"

        for name, (norb, nelec, lo1, hi1) in info["spaces"].items():
            h1, h2, ecore, dim = build_space(mf, mo_coeff_no, norb, nelec, lo1, hi1)
            tractable = "yes" if dim <= FCI_DIM_LIMIT else "no"

            fcidump_dest = (REPO_ROOT / "molecules" / SIZE_DIR[name]
                             / f"{mol_tag}_{name}_{norb}o{nelec}e_mp2no_def2svp.fcidump")
            write_fcidump_from_ints(h1, h2, ecore, norb, nelec, fcidump_dest)

            yaml_dest = fcidump_dest.with_suffix(".yaml")
            write_yaml(mol_tag, name, norb, nelec, lo1, hi1, info["fchk"], mol,
                       dim, yaml_dest)

            print(f"{mol_tag:16s} {name:5s} {norb:5d} {nelec:6d} "
                  f"{f'{lo1}-{hi1}':>18s} {dim:12.3e} {tractable:>11s} {SIZE_DIR[name]:>8s}")
            print(f"     -> {fcidump_dest.relative_to(REPO_ROOT)}")
            print(f"     -> {yaml_dest.relative_to(REPO_ROOT)}")
