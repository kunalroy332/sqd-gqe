# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exercise the shared noise transform on counts too wide for a single uint64.

Only the clean sampling engines are stubbed, so no GPU or full CI vector is needed.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyscf")


def _sampler(monkeypatch, kind, norb, counts, **kwargs):
    from sqd_gqe.gqe.mps_sampler import MpsSampler, NoisyMpsSampler
    from sqd_gqe.gqe.noisy_sampler import NoisyFfsimSampler
    from sqd_gqe.gqe.sampler import FfsimSampler

    class Pool(list):
        pass

    pool = Pool([SimpleNamespace(name="I", qiskit_gate_fn=None)])
    pool.norb, pool.nelec = norb, (25, 25) if norb == 33 else (2, 2)
    if kind == "mps":
        base, cls = MpsSampler, NoisyMpsSampler
        kwargs["verbose"] = False
    else:
        base, cls = FfsimSampler, NoisyFfsimSampler
    monkeypatch.setattr(base, "sample_state", lambda self, seq, seed=None: dict(counts))
    return cls(pool, shots_count=sum(counts.values()), **kwargs)


@pytest.mark.parametrize("kind", ["mps", "ffsim"])
@pytest.mark.parametrize("norb", [23, 32, 33, 40, 64])
def test_full_readout_flip_complements_each_spin_mask(monkeypatch, kind, norb):
    from sqd_gqe.core.configuration import ConfigurationSet

    clean = {(1, 2): 7, (1 << (norb - 1), 1 << (norb - 1)): 11}
    sampler = _sampler(monkeypatch, kind, norb, clean, p_readout=1.0)
    counts = sampler.sample_state([], seed=7)
    mask = (1 << norb) - 1
    expected = {(a ^ mask, b ^ mask): n for (a, b), n in clean.items()}
    assert counts == expected
    assert {tuple(map(int, c)) for c in ConfigurationSet.from_sample_counts(counts).configs} == set(expected)


@pytest.mark.parametrize("kind", ["mps", "ffsim"])
def test_66_qubit_hf_readout_noise_round_trips_into_sqd(monkeypatch, kind):
    from sqd_gqe.core.configuration import ConfigurationSet

    hf = (1 << 25) - 1
    sampler = _sampler(monkeypatch, kind, 33, {(hf, hf): 1000}, p_readout=0.01)
    counts = sampler.sample_state([], seed=7)
    mask = (1 << 33) - 1
    assert sum(counts.values()) == 1000
    assert all(0 <= a <= mask and 0 <= b <= mask for a, b in counts)
    assert {tuple(map(int, c)) for c in ConfigurationSet.from_sample_counts(counts).configs} == set(counts)
    if kind == "mps":
        valid = sum(n for (a, b), n in counts.items() if a.bit_count() == b.bit_count() == 25)
        assert sampler.last_valid_fraction == pytest.approx(valid / 1000)
    assert counts == sampler.sample_state([], seed=7)
    assert counts != sampler.sample_state([], seed=8)


@pytest.mark.parametrize("kind", ["mps", "ffsim"])
@pytest.mark.parametrize("norb", [33, 40, 64])
def test_decay_preserves_spin_and_cannot_add_high_orbital_occupations(monkeypatch, kind, norb):
    a, b = (1 << (norb - 1)) | 1, (1 << (norb - 1)) | 2
    backend = SimpleNamespace(err_2q=0.0, amplitude_damping_p=lambda n: 0.3)
    sampler = _sampler(monkeypatch, kind, norb, {(a, b): 2000}, p_readout=0.0,
                       backend=backend, amplitude_damping=True)
    monkeypatch.setattr(sampler, "_n_2q_gates", lambda seq: 10)
    counts = sampler.sample_state([], seed=7)
    assert sum(counts.values()) == 2000
    assert all(0 <= na <= a and na & ~a == 0 and 0 <= nb <= b and nb & ~b == 0
               for na, nb in counts)
    assert any(na == a and nb == b for na, nb in counts)
    assert any(na < a and nb < b for na, nb in counts)


@pytest.mark.parametrize("kind", ["mps", "ffsim"])
def test_empty_counts_stay_empty(monkeypatch, kind):
    sampler = _sampler(monkeypatch, kind, 40, {}, p_readout=0.01)
    assert sampler.sample_state([], seed=7) == {}


@pytest.mark.parametrize("kind", ["mps", "ffsim"])
def test_rejects_more_than_64_orbitals_per_spin(monkeypatch, kind):
    with pytest.raises(ValueError, match="1 to 64 spatial orbitals"):
        _sampler(monkeypatch, kind, 65, {(1, 1): 1}, p_readout=0.01)
