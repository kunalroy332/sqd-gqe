# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Decode and routing tests for the cuQuantum MPS sampler.

The GPU is not needed and not used: the risky part of ``MpsSampler`` is the path
from a backend's bitstring back to ``(alpha_occ, beta_occ)``, which has two chances
to silently corrupt every energy downstream -- the qubit order (cuQuantum counts
from the opposite end to qiskit) and, with ``route=True``, the SABRE layout
permutation. Both are checked here against qiskit's own ``Statevector``, so the
expected bitstrings come from qiskit rather than from a second copy of the same
assumption.
"""

import pytest

pytest.importorskip("ffsim")
pytest.importorskip("pyscf")
pytest.importorskip("qiskit")

@pytest.fixture(scope="module")
def n2_pool(n2_cfg):
    from sqd_gqe.factory import SqdFactory

    return SqdFactory().build_operator_pool(n2_cfg)


def _sampler(pool, route):
    from sqd_gqe.gqe.mps_sampler import MpsSampler
    s = MpsSampler(pool, shots_count=100, route=route, verbose=False)
    s._bit_order = "qiskit"          # Statevector labels are qubit 0 rightmost
    return s


def _exact_counts(tqc, shots=100):
    """The transpiled circuit's exact outcome distribution, as qiskit-order bitstrings."""
    from qiskit.quantum_info import Statevector

    probs = Statevector(tqc).probabilities_dict()
    return {bits: round(p * shots) for bits, p in probs.items() if p > 1e-12}


# -- helpers, no imports beyond the module under test ------------------------- #
def test_permute_inverts_the_layout():
    from sqd_gqe.gqe.mps_sampler import _permute

    # logical qubit 0 sits at physical position 2, logical 1 at 0, logical 2 at 1
    layout = [2, 0, 1]
    physical = "abc"                       # physical[j] is position j
    assert _permute(physical, layout) == "cab"
    assert _permute(physical, None) == physical


def test_valid_fraction_counts_shots_not_configurations():
    from sqd_gqe.gqe.mps_sampler import _valid_fraction

    nelec = (2, 2)
    counts = {(0b0011, 0b0011): 90,        # 2 + 2 electrons: valid
              (0b0001, 0b0011): 10}        # 1 + 2: truncation leak
    assert _valid_fraction(counts, nelec) == pytest.approx(0.9)
    assert _valid_fraction({}, nelec) == 0.0


# -- full decode chain against qiskit ----------------------------------------- #
@pytest.mark.parametrize("route", [False, True])
def test_hartree_fock_round_trip(n2_pool, route):
    """An empty sequence must decode back to exactly the HF determinant."""
    s = _sampler(n2_pool, route)
    tqc, layout = s.transpile(s.prepare_circuit([]))
    decoded = s._decode(_exact_counts(tqc), layout)

    n_alpha, n_beta = s.nelec
    hf = ((1 << n_alpha) - 1, (1 << n_beta) - 1)
    assert decoded == {hf: 100}


@pytest.mark.parametrize("route", [False, True])
def test_routing_changes_the_circuit_but_not_the_decoded_state(n2_pool, route):
    """Routing inserts SWAPs and permutes the qubits; after decoding, the sampled
    determinants must be identical to the unrouted ones."""
    seq = [i for i in range(len(n2_pool)) if n2_pool[i].name.startswith("givens")][:2]
    assert seq, "pool has no givens ops to exercise"

    out = {}
    for r in (False, True):
        s = _sampler(n2_pool, r)
        tqc, layout = s.transpile(s.prepare_circuit(seq))
        out[r] = s._decode(_exact_counts(tqc, shots=10_000), layout)
    assert out[False].keys() == out[True].keys()


def test_gates_conserve_particle_number(n2_pool):
    """Every pool gate is particle-number conserving, so an exact simulation of the
    translated circuit must put all weight on the right electron count. A failure
    here means the qiskit translation, not the MPS truncation, is wrong."""
    from sqd_gqe.gqe.mps_sampler import _valid_fraction

    seq = [i for i in range(len(n2_pool)) if n2_pool[i].name.startswith(("givens", "num_num"))][:3]
    s = _sampler(n2_pool, route=False)
    tqc, layout = s.transpile(s.prepare_circuit(seq))
    decoded = s._decode(_exact_counts(tqc, shots=10_000), layout)
    assert _valid_fraction(decoded, s.nelec) == pytest.approx(1.0, abs=1e-3)


class _ExactState:
    """Stands in for cuQuantum's NetworkState: same two methods, no truncation."""

    def __init__(self, tqc):
        from qiskit.quantum_info import Statevector
        self._probs = Statevector(tqc).probabilities_dict()

    def compute_sampling(self, nshots, seed=None):
        assert seed is not None and seed > 0, "cuQuantum rejects a non-positive seed"
        return {b: round(p * nshots) for b, p in self._probs.items() if p * nshots >= 1}

    def free(self):
        pass


def test_run_feeds_the_sqd_pipeline(n2_pool):
    """End-to-end without a GPU: stub the MPS contraction with qiskit's exact
    Statevector, then drive the real ``SqdPipeline.run``. Catches wiring mistakes
    (batch shape, decode, counts contract) that the unit tests above cannot."""
    import torch
    from sqd_gqe.gqe.mps_sampler import MpsSampler
    from sqd_gqe.sqd.pipeline import SqdPipeline

    s = MpsSampler(n2_pool, shots_count=10_000, verbose=False)
    s._network_state = _ExactState
    s._bit_order = "qiskit"                 # Statevector labels, not cuQuantum's

    givens = [i for i in range(len(n2_pool)) if n2_pool[i].name.startswith("givens")][:2]
    batch = {"idx": torch.tensor([[0, 0, 0], [0, *givens]], dtype=torch.long)}
    counts = s.run(batch)
    assert len(counts) == 2
    assert all(isinstance(k, tuple) and len(k) == 2 for c in counts for k in c)
    assert s.last_valid_fraction == pytest.approx(1.0, abs=1e-3)

    system = n2_pool.molecule
    pipeline = SqdPipeline(system, n2_pool, s, max_size=200, recover_iterations=0)
    report = pipeline.run(batch)
    assert len(report.samples) == 2
    # the identity circuit is Hartree-Fock, so it must not beat a correlated one
    hf, correlated = (x.energy for x in report.samples)
    assert correlated <= hf + 1e-9
    assert hf == pytest.approx(system.hf.e_tot, abs=1e-6)


def test_max_gates_skips_deep_circuits_without_simulating(n2_pool):
    """Over the cap, the circuit must cost nothing and produce an empty subspace."""
    from sqd_gqe.gqe.mps_sampler import MpsSampler

    seq = [i for i in range(len(n2_pool)) if n2_pool[i].name.startswith("LUCJ")][:1]
    assert seq, "pool has no LUCJ op to exercise"

    s = MpsSampler(n2_pool, shots_count=100, max_gates=10, verbose=False)
    s._bit_order = "qiskit"
    s._network_state = lambda tqc: pytest.fail("a skipped circuit must not be simulated")

    assert s.sample_state(seq) == {}
    assert s.last_valid_fraction == 0.0
    assert s.last_timings["sample"] == 0.0


def test_max_gates_also_skips_mps_topk_without_simulating(n2_pool):
    from sqd_gqe.gqe.mps_topk_sampler import MpsTopKSampler

    seq = [i for i in range(len(n2_pool)) if n2_pool[i].name.startswith("LUCJ")][:1]
    assert seq, "pool has no LUCJ op to exercise"

    s = MpsTopKSampler(
        n2_pool, shots_count=100, top_k=10, max_gates=10, verbose=False,
    )
    s._bit_order = "qiskit"
    s._network_state = lambda tqc: pytest.fail("a skipped circuit must not be simulated")

    assert s.sample_state(seq) == {}
    assert s.last_valid_fraction == 0.0
    assert s.last_topk_exact is False
    assert s.last_captured_weight == 0.0
    assert s.last_timings["build"] == 0.0


def test_max_gates_leaves_circuits_under_the_cap_alone(n2_pool):
    from sqd_gqe.gqe.mps_sampler import MpsSampler

    seq = [i for i in range(len(n2_pool)) if n2_pool[i].name.startswith("givens")][:1]
    s = MpsSampler(n2_pool, shots_count=1000, max_gates=100_000, verbose=False)
    s._network_state = _ExactState
    s._bit_order = "qiskit"

    counts = s.sample_state(seq)
    assert counts and s.last_valid_fraction == pytest.approx(1.0, abs=1e-3)


def test_rejects_a_pool_without_qiskit_gates(n2_pool):
    from sqd_gqe.gqe.mps_sampler import MpsSampler

    class _Blind:
        """Wraps the pool so one op family loses its qiskit_gate_fn."""
        norb, nelec = n2_pool.norb, n2_pool.nelec

        def __iter__(self):
            import dataclasses
            for op in n2_pool:
                yield dataclasses.replace(op, qiskit_gate_fn=None) if op.name.startswith("givens") else op

    with pytest.raises(ValueError, match="qiskit_gate_fn"):
        MpsSampler(_Blind(), shots_count=10)


def test_noisy_mps_flips_bits_and_reports_the_noisy_valid_fraction(n2_pool):
    """Readout flips break particle number, and last_valid_fraction must say so,
    since that is what configuration recovery has to repair."""
    from sqd_gqe.gqe.mps_sampler import MpsSampler, NoisyMpsSampler

    seq = [i for i in range(len(n2_pool)) if n2_pool[i].name.startswith("givens")][:2]
    clean, noisy = (cls(n2_pool, shots_count=2000, verbose=False, **kw)
                    for cls, kw in ((MpsSampler, {}), (NoisyMpsSampler, {"p_readout": 0.05})))
    for s in (clean, noisy):
        s._network_state = _ExactState
        s._bit_order = "qiskit"

    clean.sample_state(seq, seed=7)
    noisy_counts = noisy.sample_state(seq, seed=7)
    assert clean.last_valid_fraction == pytest.approx(1.0, abs=1e-3)
    assert 0.0 < noisy.last_valid_fraction < 0.9
    assert sum(noisy_counts.values()) == pytest.approx(2000, abs=10)
