# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Molecular active-space setup for SQD-GQE (self-contained).

Builds the restricted (RO)HF reference, the CAS one-/two-electron integrals, and
CASCI / CCSD reference energies from PySCF. This is the electronic-structure
front end for the SQD pipeline and carries no quantum-backend dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, cc, lib
from pyscf.tools import fcidump as pyscf_fcidump

#: direct_spin1 needs a dense CI vector of length fci_dimension(); well past
#: this the solve isn't "slow", it's infeasible (hundreds of GB+ for the
#: vector alone) -- but even orders of magnitude below that (1e8-1e9-scale),
#: Davidson iteration count/cost on a dense CI vector this size makes a
#: direct solve impractical, not just memory-infeasible. CO2's 28q CASCI
#: (FCI dim ~9e6) is already a meaningful compute cost in this repo; this
#: ceiling is set with headroom above that, not at the memory wall. Molecules
#: past this ceiling have no from-scratch exact reference here -- see
#: MolecularSystem.casci_energy and the 40q pyridine molecule.
FCI_DIM_LIMIT = 50_000_000


class FciIntractableError(RuntimeError):
    """Raised instead of attempting a direct FCI solve beyond FCI_DIM_LIMIT."""


@dataclass(frozen=True, slots=True)
class ActiveSpaceHamiltonian:
    """One- and two-electron integrals plus the core energy of an active space."""

    one_body: np.ndarray          # (norb, norb)
    two_body: np.ndarray          # (norb, norb, norb, norb), chemist order (pq|rs)
    core_energy: float | np.floating


class MolecularSystem:
    """A molecule with a chosen active space and its electronic-structure refs."""

    def __init__(self, geometry, basis, nelecas, norbcas, spin, charge, ecp=None,
                 num_threads: int | None = None, _skip_active_space_build: bool = False):
        if num_threads is None:
            # Auto-detect: use every CPU Slurm gave us for PySCF's OpenMP'd C
            # kernels (FCI Hc products, CCSD contractions). Direct FCI at
            # tens of millions of determinants is single-threaded-infeasible
            # in practice -- this alone turned a >15min ethanol CASCI at
            # num_threads=1 into a normal-length one. Falls back to 1 outside
            # Slurm (unchanged behavior for local/interactive use).
            import os
            num_threads = int(os.environ.get("PYSCF_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        lib.num_threads(num_threads)
        self.geometry = self._build_geometry(geometry)
        self.basis = basis
        self.ecp = ecp

        n_alpha = (nelecas + spin) // 2
        n_beta = (nelecas - spin) // 2
        self.nelec = (n_alpha, n_beta)
        self.norb = norbcas
        self.spin = spin

        self.mol = gto.M(atom=self.geometry, basis=basis, ecp=ecp, charge=charge, spin=spin, unit="Angstrom")
        self.hf = scf.RHF(self.mol).run() if spin == 0 else scf.ROHF(self.mol).run()
        self.cas = mcscf.CASCI(self.hf, self.norb, self.nelec)

        self._ccsd_amp = None
        self._cache_key = self._cache_id(self.mol.dumps(), nelecas, norbcas)
        self._cache_dir = Path(".cache") / "pyscf"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # from_fcidump() builds the HF/mol above (for reference-energy
        # reporting) but overrides norb/nelec/active_hamiltonian from the
        # FCIDUMP itself rather than a fresh CASCI on self.hf's canonical
        # MOs -- see that classmethod for why (MP2-NO active spaces beat
        # HF-MO ones by tens of mHa, see docs/report/snoh3_active_space_comparison.md).
        if not _skip_active_space_build:
            self.active_indices = self._active_indices()
            self.active_hamiltonian = self._build_active_hamiltonian()

    @classmethod
    def from_fcidump(cls, fcidump_path, geometry, basis, spin, charge, ecp=None,
                      num_threads: int | None = None):
        """Load the active-space Hamiltonian from a pre-computed FCIDUMP
        (e.g. built on Gaussian MP2 natural orbitals instead of fresh HF
        canonical MOs -- see benchmark/snoh3_pdf_active_spaces.py) while
        still building the full-molecule HF/mol from (geometry, basis, ecp)
        for reference-energy reporting (hf.e_tot).

        casci_energy() is still valid here (pyscf's FCI solver only needs
        the active-space h1/h2/norb/nelec, no HF object) but only reflects
        the active space in the FCIDUMP's orbital basis, not self.hf's
        canonical MOs -- by construction, since that's the whole point of
        loading a non-HF-MO active space.

        ccsd_energy()/ccsd_t_energy() also work, but run CCSD entirely
        within the active space (a fresh RHF solved on just the active-space
        integrals, via _run_ccsd_on_active_space()) rather than freezing
        orbitals outside a window in self.hf's canonical MOs -- self.hf's
        frozen-core CCSD doesn't apply here since active_indices is None
        (there's no such contiguous window once the active space comes from
        a different orbital basis than self.hf).
        """
        nelecas_placeholder = 2  # unused once norb/nelec are overridden below; __init__ needs *some* value
        self = cls(geometry, basis, nelecas_placeholder, 1, spin, charge, ecp=ecp,
                   num_threads=num_threads, _skip_active_space_build=True)

        ctx = pyscf_fcidump.read(str(fcidump_path))
        norb = ctx["NORB"]
        nelec = ctx["NELEC"]
        n_alpha = (nelec + spin) // 2
        n_beta = (nelec - spin) // 2

        self.norb = norb
        self.nelec = (n_alpha, n_beta)
        self.active_indices = None  # not meaningful: active space isn't a contiguous slice of self.hf's MOs
        self.active_hamiltonian = ActiveSpaceHamiltonian(
            one_body=np.asarray(ctx["H1"]),
            two_body=ao2mo.restore(1, np.asarray(ctx["H2"]), norb),
            core_energy=ctx["ECORE"],
        )
        self._fcidump_path = str(fcidump_path)
        # __init__ computed _cache_key from the nelecas/norbcas PLACEHOLDER
        # above, before norb/nelec were known -- recompute now with the real
        # active-space size, or every fcidump-loaded molecule sharing this
        # geometry (e.g. Sn(OH)3-Me's S/M/L/LL tiers, which differ only in
        # which FCIDUMP they load) would collide on the same cache key and
        # silently share/overwrite each other's cached CASCI/CCSD amplitudes.
        self._cache_key = self._cache_id(self.mol.dumps(), nelec, norb)
        return self

    # -- caching ------------------------------------------------------------- #
    def _cache_id(self, mol_key, nelecas, norbcas) -> str:
        payload = json.dumps(
            {"mol_key": mol_key, "nelecas": nelecas, "norbcas": norbcas},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- amplitudes / references -------------------------------------------- #
    @property
    def ccsd_amplitudes(self):
        if self._ccsd_amp is None:
            _ = self.ccsd_energy()
        return self._ccsd_amp

    def _build_geometry(self, geometry):
        if geometry.type == "linear_chain":
            return build_linear_chain(geometry.atoms, geometry.bond_length)
        if geometry.type == "custom":
            return geometry.geometry
        raise ValueError(f"Invalid geometry type: {geometry.type}")

    def _active_indices(self):
        ncore = (self.mol.nelectron - sum(self.nelec)) // 2
        idx = list(range(ncore, ncore + self.norb))
        assert int(sum(self.hf.mo_occ[idx])) == sum(self.nelec), \
            "Electron count in active indices does not match nelec."
        return idx

    def _build_active_hamiltonian(self) -> ActiveSpaceHamiltonian:
        mo = self.hf.mo_coeff
        h1, e_core = self.cas.get_h1cas(mo)
        h2 = ao2mo.restore(1, self.cas.get_h2cas(mo), self.norb)
        return ActiveSpaceHamiltonian(one_body=h1, two_body=h2, core_energy=e_core)

    def fci_dimension(self) -> int:
        """Size of the active-space FCI space: C(norb, n_alpha) * C(norb, n_beta)."""
        na, nb = self.nelec
        return comb(self.norb, na) * comb(self.norb, nb)

    def casci_energy(self):
        """Exact active-space FCI energy via direct diagonalization.

        Raises FciIntractableError above FCI_DIM_LIMIT instead of attempting a
        direct_spin1 solve that would need a dense CI vector of that length
        (the 40q pyridine molecule is 3.4e10 -- hundreds of GB, not "slow").
        For molecules past the ceiling, use ccsd_t_energy() (still tractable,
        the standard surrogate gold-standard) or run.py's SQD-self-convergence
        fallback; a DMRG reference is planned as follow-up work for a tighter
        benchmark at that scale (not implemented here).
        """
        dim = self.fci_dimension()
        if dim > FCI_DIM_LIMIT:
            raise FciIntractableError(
                f"FCI dimension {dim:.3e} (norb={self.norb}, nelec={self.nelec}) exceeds "
                f"the direct-solve ceiling ({FCI_DIM_LIMIT:.3e}); no exact reference "
                f"available. Use ccsd_t_energy() or an SQD-self-convergence reference."
            )
        cache = self._cache_dir / f"{self._cache_key}_casci.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                return data["energy"]
        e_fci, _ = self.cas.fcisolver.kernel(
            self.active_hamiltonian.one_body,
            self.active_hamiltonian.two_body,
            self.norb,
            self.nelec,
            ecore=self.active_hamiltonian.core_energy,
        )
        np.savez(cache, energy=e_fci)
        return e_fci

    def _run_ccsd(self):
        if self.active_indices is None:
            return self._run_ccsd_on_active_space()
        nmo = self.hf.mo_coeff.shape[1]
        active = set(self.active_indices)
        frozen = [i for i in range(nmo) if i not in active]
        mycc = cc.RCCSD(self.hf, frozen=frozen)
        mycc.verbose = 0
        mycc.kernel()
        return mycc

    def _run_ccsd_on_active_space(self):
        """CCSD run directly on the active-space integrals (h1/h2/norb/nelec),
        for systems loaded via from_fcidump() where the active space isn't a
        contiguous window of self.hf's canonical MOs (so _run_ccsd()'s
        frozen-core restriction doesn't apply).

        Uses pyscf's standard "fake mean-field" pattern: an empty Mole plus an
        RHF object whose get_hcore/get_ovlp/_eri are overridden to the active
        space's own integrals, so RHF (and then CCSD on top of it) solves
        entirely within the active space -- the same self-contained-integrals
        trick casci_energy() already relies on for its FCI solver, just
        applied to CCSD. This is a fresh SCF solution *within* the active
        space (not the outer self.hf restricted to it), which is the standard
        way to seed post-CASCI correlation methods on a CAS Hamiltonian.
        """
        h = self.active_hamiltonian
        norb = self.norb
        na, nb = self.nelec
        if na != nb:
            raise NotImplementedError("active-space CCSD here assumes a closed-shell (na==nb) active space")

        mol = gto.M()
        mol.nelectron = na + nb
        mol.incore_anyway = True
        mol.build()
        mf_as = scf.RHF(mol)
        mf_as.get_hcore = lambda *args: h.one_body
        mf_as.get_ovlp = lambda *args: np.eye(norb)
        mf_as.energy_nuc = lambda *args: h.core_energy
        mf_as._eri = ao2mo.restore(8, h.two_body, norb)
        mf_as.verbose = 0
        mf_as.kernel()

        mycc = cc.RCCSD(mf_as)
        mycc.verbose = 0
        mycc.kernel()
        return mycc

    def ccsd_energy(self):
        cache = self._cache_dir / f"{self._cache_key}_ccsd.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                self._ccsd_amp = {"t1": data["t1"], "t2": data["t2"]}
                return data["energy"]
        mycc = self._run_ccsd()
        # mycc._scf.e_tot, not self.hf.e_tot: the active-space CCSD path
        # (_run_ccsd_on_active_space) builds its own fake mean-field distinct
        # from self.hf, so self.hf.e_tot would silently mix energies from two
        # different Hamiltonians for from_fcidump()-loaded systems.
        e_tot = mycc._scf.e_tot + mycc.e_corr
        self._ccsd_amp = {"t1": mycc.t1, "t2": mycc.t2}
        np.savez(cache, energy=e_tot, t1=mycc.t1, t2=mycc.t2)
        return e_tot

    def ccsd_t_energy(self):
        """CCSD(T): perturbative triples on top of CCSD. Still polynomial-cost
        (no dense-CI-vector blowup), so it's the reference of choice once
        casci_energy() is infeasible -- see casci_energy()'s docstring."""
        cache = self._cache_dir / f"{self._cache_key}_ccsdt.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                return data["energy"]
        mycc = self._run_ccsd()
        et = mycc.ccsd_t()
        e_tot = mycc._scf.e_tot + mycc.e_corr + et  # see ccsd_energy()'s comment on mycc._scf.e_tot
        np.savez(cache, energy=e_tot)
        return e_tot


def build_linear_chain(atoms: list[str], bond_length: float):
    geometry = [[atoms[0], [0.0, 0.0, 0.0]]]
    dist = 0.0
    for sym in atoms[1:]:
        dist += bond_length
        geometry.append([sym, [0.0, 0.0, dist]])
    return geometry
