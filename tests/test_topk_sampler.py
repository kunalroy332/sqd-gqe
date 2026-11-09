# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The exact top-k reward sampler: deterministic, particle-number conserving, ordered."""

import pytest

pytest.importorskip("ffsim")
pytest.importorskip("pyscf")

@pytest.fixture(scope="module")
def n2_pool(n2_cfg):
    from sqd_gqe.factory import SqdFactory

    f = SqdFactory()
    return f.build_system(n2_cfg), f.build_operator_pool(n2_cfg)


def test_topk_is_deterministic_and_valid(n2_pool):
    from sqd_gqe.gqe.sampler import ExactTopKSampler

    system, pool = n2_pool
    names = [pool[i].name for i in range(len(pool))]
    seq = [next(i for i, n in enumerate(names) if n.startswith("LUCJ")),
           next(i for i, n in enumerate(names) if n.startswith("givens"))]
    sampler = ExactTopKSampler(pool, shots_count=100_000, top_k=200, seed=0, min_prob=None)

    a = sampler.sample_state(seq)
    b = sampler.sample_state(seq)
    assert a == b                                   # no shot noise: repeats are identical
    assert sampler.sample_state(seq, seed=999) == a  # and a seed cannot change it

    na, nb = system.nelec
    assert len(a) == 200
    assert all(alpha.bit_count() == na and beta.bit_count() == nb for alpha, beta in a)


def test_counts_rank_the_determinants_by_weight(n2_pool):
    """`ConfigurationSet.from_sample_counts` orders by count, so the pseudo-counts
    must put the largest-amplitude determinants first."""
    import ffsim
    import numpy as np
    from sqd_gqe.core.configuration import ConfigurationSet
    from sqd_gqe.gqe.sampler import ExactTopKSampler

    system, pool = n2_pool
    names = [pool[i].name for i in range(len(pool))]
    seq = [next(i for i, n in enumerate(names) if n.startswith("LUCJ"))]
    sampler = ExactTopKSampler(pool, shots_count=100_000, top_k=50, seed=0)

    vec = sampler.prepare_state(seq)
    dim_a, dim_b = ffsim.dims(sampler.norb, sampler.nelec)
    a_str = np.asarray(ffsim.addresses_to_strings(np.arange(dim_a), sampler.norb, sampler.nelec[0]),
                       dtype=np.uint64)
    b_str = np.asarray(ffsim.addresses_to_strings(np.arange(dim_b), sampler.norb, sampler.nelec[1]),
                       dtype=np.uint64)
    w = np.abs(np.asarray(vec).reshape(-1)) ** 2
    top = np.argsort(w)[::-1][0]
    ia, ib = divmod(int(top), dim_b)

    first = ConfigurationSet.from_sample_counts(sampler.sample_state(seq)).configs[0]
    assert (int(first[0]), int(first[1])) == (int(a_str[ia]), int(b_str[ib]))


def test_rounding_to_equal_counts_keeps_the_weight_order(n2_pool):
    """With shots=10, weights 0.01, 0.02 and 0.03 all round to a count of 1. The stable
    sort in `from_sample_counts` then keeps the sampler's order, so that order has to be
    by weight, or truncating to two determinants keeps 0.01 or 0.02 instead of 0.03."""
    import numpy as np
    from sqd_gqe.core.configuration import ConfigurationSet
    from sqd_gqe.gqe.sampler import ExactTopKSampler

    _, pool = n2_pool
    sampler = ExactTopKSampler(pool, shots_count=10, top_k=4, seed=0, min_prob=None)
    a_str, b_str, dim_b = sampler._string_tables()
    weights = [0.01, 0.02, 0.03, 0.94]
    vec = np.zeros(len(a_str) * dim_b, dtype=complex)
    vec[:4] = np.sqrt(weights)
    sampler.prepare_state = lambda seq: vec

    configs = ConfigurationSet.from_sample_counts(sampler.sample_state([])).configs
    expected = [divmod(i, dim_b) for i in (3, 2)]          # 0.94, then 0.03
    assert [(int(c[0]), int(c[1])) for c in configs[:2]] == [
        (int(a_str[ia]), int(b_str[ib])) for ia, ib in expected]


def test_zero_weight_determinants_are_not_padded_in(n2_pool):
    """A state with fewer weighted determinants than top_k returns only those. Padding
    with zeros once gave a Hartree-Fock circuit 165 mHa of credit at 28q."""
    import numpy as np
    from sqd_gqe.gqe.sampler import ExactTopKSampler

    _, pool = n2_pool
    sampler = ExactTopKSampler(pool, shots_count=1000, top_k=50, seed=0)
    a_str, b_str, dim_b = sampler._string_tables()
    vec = np.zeros(len(a_str) * dim_b, dtype=complex)
    vec[[0, 7, 30]] = np.sqrt([0.5, 0.3, 0.2])
    vec[40] = 1e-8                                  # weight 1e-16: noise, not a determinant
    sampler.prepare_state = lambda seq: vec

    counts = sampler.sample_state([])
    expected = {(int(a_str[i // dim_b]), int(b_str[i % dim_b])) for i in (0, 7, 30)}
    assert set(counts) == expected


def test_select_top_k_breaks_ties_by_the_lower_index():
    """The cut lands on equal weights (usually exact zeros, since a circuit populates
    far fewer determinants than top_k asks for), so the order has to be defined."""
    import numpy as np
    from sqd_gqe.gqe.sampler import select_top_k

    w = np.array([0.5, 0.0, 0.0, 0.0, 0.3, 0.0])
    np.testing.assert_array_equal(np.sort(select_top_k(np, w, 4)), [0, 1, 2, 4])


def test_select_top_k_beyond_the_array_returns_everything():
    import numpy as np
    from sqd_gqe.gqe.sampler import select_top_k

    w = np.array([0.1, 0.9, 0.4])
    np.testing.assert_array_equal(np.sort(select_top_k(np, w, 10)), [0, 1, 2])


def test_select_top_k_picks_the_k_largest_even_with_many_ties():
    import numpy as np
    from sqd_gqe.gqe.sampler import select_top_k

    rng = np.random.default_rng(0)
    w = np.where(rng.random(500) < 0.7, 0.0, rng.random(500))   # many exact zeros
    chosen = select_top_k(np, w, 50)
    assert len(chosen) == 50
    assert len(set(chosen.tolist())) == 50                      # no index twice
    # the selected weights are exactly the 50 largest, as a multiset
    np.testing.assert_array_equal(np.sort(w[chosen]), np.sort(w)[-50:])


def _sampler_on(n2_pool, weights, **kwargs):
    """An ExactTopKSampler whose state has the given weights on its first determinants."""
    import numpy as np
    from sqd_gqe.gqe.sampler import ExactTopKSampler

    _, pool = n2_pool
    sampler = ExactTopKSampler(pool, **kwargs)
    a_str, b_str, dim_b = sampler._string_tables()
    vec = np.zeros(len(a_str) * dim_b, dtype=complex)
    vec[: len(weights)] = np.sqrt(weights)
    sampler.prepare_state = lambda seq: vec
    keys = [(int(a_str[i // dim_b]), int(b_str[i % dim_b])) for i in range(len(weights))]
    return sampler, keys


def test_the_default_floor_drops_what_these_shots_would_not_draw(n2_pool):
    """auto is 1/shots: at 100 shots a determinant needs |c|^2 >= 0.01, an expected count
    of one. Kept counts are what the shots would give, not renormalised over the kept set."""
    sampler, keys = _sampler_on(n2_pool, [0.6, 0.3, 0.095, 0.004, 0.001],
                                shots_count=100, top_k=5)
    assert sampler.min_prob == 0.01
    assert sampler.sample_state([]) == {keys[0]: 60, keys[1]: 30, keys[2]: 10}


def test_a_numeric_floor_is_taken_as_is(n2_pool):
    sampler, keys = _sampler_on(n2_pool, [0.6, 0.3, 0.095, 0.004, 0.001],
                                shots_count=100, top_k=5, min_prob=0.2)
    assert sampler.sample_state([]) == {keys[0]: 60, keys[1]: 30}


def test_without_a_floor_every_weighted_determinant_is_kept(n2_pool):
    sampler, keys = _sampler_on(n2_pool, [0.6, 0.3, 0.095, 0.004, 0.001],
                                shots_count=100, top_k=5, min_prob=None)
    assert set(sampler.sample_state([])) == set(keys)


def test_a_state_no_run_could_sample_keeps_only_its_largest_determinant(n2_pool):
    sampler, keys = _sampler_on(n2_pool, [0.4, 0.3, 0.3], shots_count=100, top_k=3,
                                min_prob=0.5)              # nothing reaches 0.5
    assert sampler.sample_state([]) == {keys[0]: 40}


@pytest.mark.parametrize("bad", [-0.1, "half", float("nan")])
def test_an_invalid_floor_is_refused(bad):
    from sqd_gqe.gqe.sampler import resolve_min_prob

    with pytest.raises(ValueError):
        resolve_min_prob(bad, 100)
