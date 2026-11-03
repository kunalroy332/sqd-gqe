# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Proof-of-principle checks for the UFD refinement maps (LiH, 5o/2e, FCI dim 100)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytest.importorskip("qiskit_addon_sqd")

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def system():
    from sqd_gqe.factory import SqdFactory

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs_sqd")):
        cfg = compose(config_name="default")
    cfg.molecule = OmegaConf.load(REPO / "molecules/small/lih.yaml")
    return SqdFactory().build_system(cfg)


@pytest.fixture(scope="module")
def ci_vectors(system):
    """Three overlapping random determinant subsets, each diagonalized with the sqd backend."""
    from math import comb

    from sqd_gqe.core.configuration import Configuration, ConfigurationSet
    from sqd_gqe.sqd.backend_sqd import SqdDiagonalizer

    norb, nelec = system.norb, system.nelec
    fci_dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    rng = np.random.default_rng(0)
    diag = SqdDiagonalizer(system.active_hamiltonian, nelec, norb)
    hf = Configuration.from_fullci_address(0, norb, nelec)
    vecs = []
    for _ in range(3):
        idx = rng.choice(fci_dim, size=12, replace=False)
        configs = [hf] + [Configuration.from_fullci_address(int(i), norb, nelec) for i in idx]
        expanded = ConfigurationSet(configs).expand(max_size=60)
        _, ci = diag.diagonalize(expanded)
        vecs.append(ci)
    return vecs


def _mergers(system):
    from sqd_gqe.core.refine import build_merger

    h, nelec, norb = system.active_hamiltonian, system.nelec, system.norb
    return {m: build_merger(m, h, nelec, norb) for m in ("gevp", "ufd", "tp_ufd")}


def test_variational_ordering(system, ci_vectors):
    mergers = _mergers(system)
    pre = {}
    for name, m in mergers.items():
        m.merge(ci_vectors, max_size=10_000)
        pre[name] = m.last_info["pre_energy"]
    assert pre["tp_ufd"] <= pre["ufd"] + 1e-9
    assert pre["ufd"] <= pre["gevp"] + 1e-9


def test_truncation_and_rediagonalize(system, ci_vectors):
    from sqd_gqe.sqd.backend_sqd import SqdDiagonalizer

    diag = SqdDiagonalizer(system.active_hamiltonian, system.nelec, system.norb)
    for name, m in _mergers(system).items():
        cs = m.merge(ci_vectors, max_size=16)
        assert 0 < cs.size <= 16, name
        energy, ci = diag.diagonalize(cs)
        assert np.isfinite(energy), name
        if name == "tp_ufd":
            # tp_ufd returns the retained tensor basis as |keep| diagonal pairs; the closure
            # is keep x keep and is what diagonalize() actually solves
            assert cs.size == m.last_info["n_alpha"]
            assert m.last_info["retained_closure"] == m.last_info["n_alpha"] ** 2
            assert np.unique(ci.strings[:, 0]).size * np.unique(ci.strings[:, 1]).size == m.last_info["retained_closure"]
            assert m.last_info["rho_tp"] >= 1.0


def test_gevp_regression_matches_subspace_merger(system, ci_vectors):
    from sqd_gqe.core.refine import SubspaceMerger, build_merger

    h, nelec, norb = system.active_hamiltonian, system.nelec, system.norb
    a = SubspaceMerger(h, nelec, norb).merge(ci_vectors, max_size=20)
    b = build_merger("gevp", h, nelec, norb).merge(ci_vectors, max_size=20)
    assert [tuple(map(int, c)) for c in a.configs] == [tuple(map(int, c)) for c in b.configs]


def test_single_vector_ufd_is_identity_energy(system, ci_vectors):
    from sqd_gqe.core.refine import NonOrthogonalSolver, build_merger

    h, nelec, norb = system.active_hamiltonian, system.nelec, system.norb
    v = ci_vectors[0]
    m = build_merger("ufd", h, nelec, norb)
    m.merge([v], max_size=10_000)
    e_ref, _ = NonOrthogonalSolver(h, nelec, norb).diagonalize_subspace(v.strings)
    assert abs(m.last_info["pre_energy"] - e_ref) < 1e-8


def test_invalid_combinations(system):
    from sqd_gqe.core.refine import build_merger

    h, norb = system.active_hamiltonian, system.norb
    with pytest.raises(ValueError):
        build_merger("ufd", h, system.nelec, norb, diag_backend="sbd")
    with pytest.raises(ValueError):
        build_merger("gevp", h, system.nelec, norb, diag_backend="sbd")
    with pytest.raises(ValueError):
        build_merger("tp_ufd", h, system.nelec, norb, diag_backend="fci")
    with pytest.raises(ValueError):
        build_merger("tp_ufd", h, (2, 1), norb)
    with pytest.raises(ValueError):
        build_merger("nope", h, system.nelec, norb)


def test_pretruncation_bounds_solve_size(system, ci_vectors):
    from sqd_gqe.core.refine import build_merger

    h, nelec, norb = system.active_hamiltonian, system.nelec, system.norb
    ufd = build_merger("ufd", h, nelec, norb, dense_limit=8)
    with pytest.warns(UserWarning, match="pre-truncating"):
        cs = ufd.merge(ci_vectors, max_size=100)
    assert ufd.last_info["pre_truncated"] and ufd.last_info["pre_dim"] == 8 and cs.size <= 8

    tp = build_merger("tp_ufd", h, nelec, norb, closure_limit=9)
    with pytest.warns(UserWarning, match="pre-truncating"):
        tp.merge(ci_vectors, max_size=100)
    assert tp.last_info["pre_truncated"] and tp.last_info["n_alpha_union"] == 3

    # pre-truncation + fallback together: the kept weights must stay aligned with A
    from sqd_gqe.core.refine import input_weighted_half_strings

    class FakeSbd:
        def _solve_once(self, strs_a, solver_overrides=None):
            return -1.0, np.zeros(norb), np.empty(0, dtype=np.int64)

    A_all, w_all = input_weighted_half_strings(ci_vectors)
    top3 = set(A_all[np.argsort(w_all)[::-1][:3]].tolist())
    tp2 = build_merger("tp_ufd", h, nelec, norb, sbd=FakeSbd(), diag_backend="sbd", closure_limit=9, n_alpha=2)
    with pytest.warns(UserWarning):
        cs = tp2.merge(ci_vectors, max_size=100)
    kept = {int(c[0]) for c in cs.configs} - {(1 << nelec[0]) - 1}
    assert kept <= top3


def test_tp_ufd_sbd_fallback_when_no_carryover(system, ci_vectors):
    from sqd_gqe.core.refine import build_merger

    class FakeSbd:
        def _solve_once(self, strs_a, solver_overrides=None):
            return -1.0, np.zeros(system.norb), np.empty(0, dtype=np.int64)

    h, nelec, norb = system.active_hamiltonian, system.nelec, system.norb
    m = build_merger("tp_ufd", h, nelec, norb, sbd=FakeSbd(), diag_backend="sbd")
    with pytest.warns(UserWarning, match="no carryover"):
        cs = m.merge(ci_vectors, max_size=16)
    assert m.last_info["compression"] == "input_weights_fallback"
    assert cs.size <= 4 + 1   # n_alpha=isqrt(16)=4 (+ forced HF)


def test_fold_rhf_half_strings_weights():
    from sqd_gqe.core.configuration import fold_rhf_half_strings

    uniq, w = fold_rhf_half_strings(np.array([[3, 5], [5, 3], [3, 3]]), weights=[1.0, 1.0, 2.0])
    assert uniq.tolist() == [3, 5]
    assert np.allclose(w, [3.0, 1.0])
