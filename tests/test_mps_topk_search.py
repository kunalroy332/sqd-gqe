# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The MPS top-K search has to find the true top-K, not merely a plausible set.

Every claim the class makes is checked against brute-force enumeration of a small random
MPS: the right-canonical gauge, the particle-number pruning, the ordering, and the
exactness certificate. No GPU and no cuQuantum -- the search is plain array code and the
tensors are built here, so this runs anywhere.
"""

import itertools

import numpy as np
import pytest

pytest.importorskip("torch")

# Production imports must follow the optional dependency check.
from sqd_gqe.gqe.mps_topk_sampler import MpsTopKSampler  # noqa: E402

NORB = 4
NELEC = (2, 2)
N_SITES = 2 * NORB


class _Op:
    name = "I"
    qiskit_gate_fn = None


class _Pool:
    """The smallest thing MpsSampler.__init__ accepts: identity ops only."""
    norb = NORB
    nelec = NELEC

    def __iter__(self):
        return iter([_Op()])

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return _Op()


def _sampler(top_k=5, beam_width=None, bit_order="qubit0_left"):
    s = MpsTopKSampler(_Pool(), shots_count=1000, top_k=top_k, beam_width=beam_width,
                       verbose=False)
    s._bit_order = bit_order
    return s


def _random_mps(rng, n=N_SITES, chi=4):
    tensors = []
    left = 1
    for k in range(n):
        right = 1 if k == n - 1 else min(chi, 2 ** (k + 1), 2 ** (n - k - 1))
        tensors.append(rng.normal(size=(left, 2, right))
                       + 1j * rng.normal(size=(left, 2, right)))
        left = right
    return tensors


def _amplitudes(tensors):
    """Brute force: every bitstring's amplitude, by contracting the chain."""
    out = {}
    for bits in itertools.product((0, 1), repeat=len(tensors)):
        v = np.ones((1, 1), dtype=complex)
        for k, b in enumerate(bits):
            v = v @ tensors[k][:, b, :]
        out["".join(str(b) for b in bits)] = complex(v[0, 0])
    return out


def _valid(bitstring):
    a, b = bitstring[:NORB], bitstring[NORB:]
    return a.count("1") == NELEC[0] and b.count("1") == NELEC[1]


def test_canonicalisation_makes_every_site_right_isometric():
    tensors = _random_mps(np.random.default_rng(0))
    canon = MpsTopKSampler._right_canonicalize(tensors)
    for k in range(1, len(canon)):
        a = canon[k]
        contracted = sum(a[:, b, :] @ a[:, b, :].conj().T for b in (0, 1))
        assert np.allclose(contracted, np.eye(a.shape[0]), atol=1e-10)


def test_canonicalisation_preserves_the_state_up_to_its_norm():
    """A gauge change plus a normalisation: every amplitude scales by the same factor."""
    tensors = _random_mps(np.random.default_rng(1))
    before = _amplitudes(tensors)
    after = _amplitudes(MpsTopKSampler._right_canonicalize(tensors))
    ratios = [after[b] / before[b] for b in before if abs(before[b]) > 1e-12]
    assert len(ratios) > 1
    for r in ratios[1:]:
        assert r == pytest.approx(ratios[0], rel=1e-9)


def test_canonicalisation_normalises_the_state():
    """cuQuantum does not return a normalised MPS, so without this the captured weight is
    not a fraction and can come back above 1."""
    for seed in (1, 2, 3):
        canon = MpsTopKSampler._right_canonicalize(
            _random_mps(np.random.default_rng(seed)))
        total = sum(abs(a) ** 2 for a in _amplitudes(canon).values())
        assert total == pytest.approx(1.0, rel=1e-9)


def test_captured_weight_cannot_exceed_one():
    """The bug as it appeared in the log, pinned directly."""
    canon = MpsTopKSampler._right_canonicalize(_random_mps(np.random.default_rng(7)))
    _, prob, _ = _sampler(top_k=1000, beam_width=1 << N_SITES)._search(
        canon, np.array([k < NORB for k in range(N_SITES)]))
    assert 0.0 <= prob.sum() <= 1.0 + 1e-9


def test_search_reproduces_the_brute_force_top_k():
    rng = np.random.default_rng(2)
    tensors = MpsTopKSampler._right_canonicalize(_random_mps(rng))
    s = _sampler(top_k=5, beam_width=1 << N_SITES)
    is_alpha = np.array([k < NORB for k in range(N_SITES)])

    strings, prob, exact = s._search(tensors, is_alpha)

    amps = _amplitudes(tensors)
    expected = sorted(((abs(a) ** 2, b) for b, a in amps.items() if _valid(b)), reverse=True)
    assert strings == [b for _, b in expected[:5]]
    assert prob == pytest.approx([p for p, _ in expected[:5]], rel=1e-9)
    assert exact is True


def test_every_returned_bitstring_conserves_particle_number():
    rng = np.random.default_rng(3)
    tensors = MpsTopKSampler._right_canonicalize(_random_mps(rng))
    strings, _, _ = _sampler(top_k=20, beam_width=1 << N_SITES)._search(
        tensors, np.array([k < NORB for k in range(N_SITES)]))
    assert strings                      # the constraint must not empty the result
    assert all(_valid(b) for b in strings)


def test_results_come_back_in_decreasing_probability():
    rng = np.random.default_rng(4)
    tensors = MpsTopKSampler._right_canonicalize(_random_mps(rng))
    _, prob, _ = _sampler(top_k=10, beam_width=1 << N_SITES)._search(
        tensors, np.array([k < NORB for k in range(N_SITES)]))
    assert list(prob) == sorted(prob, reverse=True)


def test_a_narrow_beam_reports_itself_as_unproven():
    """The certificate must not claim exactness it has not earned."""
    rng = np.random.default_rng(5)
    tensors = MpsTopKSampler._right_canonicalize(_random_mps(rng))
    is_alpha = np.array([k < NORB for k in range(N_SITES)])
    _, _, exact = _sampler(top_k=3, beam_width=2)._search(tensors, is_alpha)
    assert exact is False


def test_a_short_result_is_not_certified_when_branches_were_pruned():
    """Fewer than top_k found: a pruned branch could have supplied the missing entries, so
    the usual "bound below the smallest kept" test does not settle it."""
    rng = np.random.default_rng(11)
    tensors = MpsTopKSampler._right_canonicalize(_random_mps(rng))
    is_alpha = np.array([k < NORB for k in range(N_SITES)])
    s = _sampler(top_k=10_000, beam_width=4)          # far more than the space holds
    strings, _, exact = s._search(tensors, is_alpha)
    assert len(strings) < s.top_k
    assert exact is False


def test_a_short_result_is_certified_when_nothing_was_pruned():
    """The same shortfall with a beam wide enough to prune nothing is genuinely exact."""
    rng = np.random.default_rng(11)
    tensors = MpsTopKSampler._right_canonicalize(_random_mps(rng))
    is_alpha = np.array([k < NORB for k in range(N_SITES)])
    s = _sampler(top_k=10_000, beam_width=1 << N_SITES)
    strings, _, exact = s._search(tensors, is_alpha)
    assert len(strings) < s.top_k
    assert exact is True


def test_a_certified_result_really_is_the_true_top_k():
    """Whenever the certificate says exact, brute force must agree -- over many states."""
    for seed in range(12):
        tensors = MpsTopKSampler._right_canonicalize(
            _random_mps(np.random.default_rng(100 + seed)))
        is_alpha = np.array([k < NORB for k in range(N_SITES)])
        strings, _, exact = _sampler(top_k=3, beam_width=8)._search(tensors, is_alpha)
        if not exact:
            continue
        amps = _amplitudes(tensors)
        expected = sorted(((abs(a) ** 2, b) for b, a in amps.items() if _valid(b)),
                          reverse=True)
        assert strings == [b for _, b in expected[:3]], f"seed {seed}"


def test_a_state_with_no_weight_in_the_sector_returns_no_counts():
    """The vacuum has no amplitude with nelec=(2, 2), but the search still returns valid
    bitstrings, all with zero weight. Scaling those to counts would divide by zero."""
    class _Tqc:
        def size(self):
            return 0

    class _State:
        def free(self):
            pass

    vacuum = [np.array([1.0, 0.0], dtype=complex).reshape(1, 2, 1) for _ in range(N_SITES)]
    s = _sampler()
    s._prepare = lambda seq: (_Tqc(), None, 0.0)
    s._network_state = lambda tqc: _State()
    s._mps_tensors = lambda state: vacuum

    assert s.sample_state([]) == {}
    assert s.last_valid_fraction == 0.0
    assert s.last_captured_weight == 0.0


def test_a_hartree_fock_state_returns_only_hartree_fock():
    """Every other determinant has zero weight, so none of them may pad the result: which
    zeros the beam keeps is arbitrary and differs from the exact path's choice."""
    class _Tqc:
        def size(self):
            return 0

    class _State:
        def free(self):
            pass

    occupied = [i < NELEC[0] or NORB <= i < NORB + NELEC[1] for i in range(N_SITES)]
    hf = [np.array([0.0, 1.0] if o else [1.0, 0.0], dtype=complex).reshape(1, 2, 1)
          for o in occupied]
    s = _sampler(top_k=20)
    s._prepare = lambda seq: (_Tqc(), None, 0.0)
    s._network_state = lambda tqc: _State()
    s._mps_tensors = lambda state: hf

    hf_alpha, hf_beta = (1 << NELEC[0]) - 1, (1 << NELEC[1]) - 1
    assert s.sample_state([]) == {(hf_alpha, hf_beta): 1000}


@pytest.mark.parametrize("min_prob, kept", [("auto", 3), (None, 4)])
def test_the_floor_applies_to_mps_topk_too(min_prob, kept):
    """The same floor as ExactTopKSampler: at 1000 shots |c|^2 below 1e-3 is dropped and
    kept counts are prob * shots."""
    class _Tqc:
        def size(self):
            return 0

    class _State:
        def free(self):
            pass

    s = MpsTopKSampler(_Pool(), shots_count=1000, top_k=4, verbose=False, min_prob=min_prob)
    s._bit_order = "qubit0_left"
    s._prepare = lambda seq: (_Tqc(), None, 0.0)
    s._network_state = lambda tqc: _State()
    s._mps_tensors = lambda state: []
    s._right_canonicalize = lambda tensors: tensors
    s._site_spin = lambda n, layout: None
    valid = ["11001100", "11000110", "10101100", "01101010"]   # (2, 2) electrons each
    s._search = lambda tensors, is_alpha: (valid, np.array([0.7, 0.25, 0.0495, 0.0005]), True)
    counts = s.sample_state([])
    assert len(counts) == kept
    if min_prob == "auto":
        assert sorted(counts.values(), reverse=True) == [700, 250, 50]


def test_site_spin_map_follows_the_bit_order():
    """Site k is a position in the sampling string, which qiskit order reverses."""
    left = _sampler(bit_order="qubit0_left")._site_spin(N_SITES, None)
    assert list(left) == [True] * NORB + [False] * NORB

    qiskit = _sampler(bit_order="qiskit")._site_spin(N_SITES, None)
    assert list(qiskit) == [False] * NORB + [True] * NORB


def test_site_spin_map_follows_the_routing_layout():
    """With routing, site k holds whichever logical qubit the layout put at position k."""
    layout = list(reversed(range(N_SITES)))      # logical i sits at position n-1-i
    mapped = _sampler(bit_order="qubit0_left")._site_spin(N_SITES, layout)
    assert list(mapped) == [False] * NORB + [True] * NORB


# -- boundary tensors: cuQuantum drops the dummy bond at each end ------------------- #
from sqd_gqe.gqe.mps_topk_sampler import _as_site_tensor      # noqa: E402


def test_first_site_gains_a_left_bond_of_one():
    a = _as_site_tensor(np.zeros((2, 7)), 0, 5, [])
    assert a.shape == (1, 2, 7)


def test_last_site_gains_a_right_bond_of_one():
    a = _as_site_tensor(np.zeros((7, 2)), 4, 5, [])
    assert a.shape == (7, 2, 1)


def test_a_last_site_whose_left_bond_is_two_is_not_read_as_a_first_site():
    """(2, 2) is ambiguous by shape, so position has to decide."""
    a = _as_site_tensor(np.zeros((2, 2)), 4, 5, [])
    assert a.shape == (2, 2, 1)


def test_a_first_site_whose_right_bond_is_two_stays_a_first_site():
    a = _as_site_tensor(np.zeros((2, 2)), 0, 5, [])
    assert a.shape == (1, 2, 2)


def test_a_rank_two_interior_site_is_an_error():
    with pytest.raises(RuntimeError, match="interior site 2 is rank-2"):
        _as_site_tensor(np.zeros((2, 2)), 2, 5, [])


def test_a_physical_index_that_is_not_in_the_middle_is_an_error():
    with pytest.raises(RuntimeError, match="no physical index in the middle"):
        _as_site_tensor(np.zeros((4, 5, 6)), 2, 5, [])
