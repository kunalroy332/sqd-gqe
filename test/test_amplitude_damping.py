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

"""Tests for NoisyFfsimSampler's amplitude-damping (T1 relaxation) noise term.

Mirrors test_sampler_decode.py / test_ibm_decode.py's style: exercises the
directional-flip arithmetic directly, plus HardwareBackend.amplitude_damping_p's
formula, without needing ffsim/torch/qiskit installed (NoisyFfsimSampler and
HardwareBackend import cleanly without them -- only methods that use ffsim/
qiskit import those lazily).
"""

import math

import numpy as np


def _apply_symmetric_then_decay(bits: np.ndarray, p_symmetric: float, p_decay: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Standalone copy of NoisyFfsimSampler.sample_state's bit-transform
    logic, isolated from the ffsim-dependent sampling step around it."""
    bits = bits.copy()
    if p_symmetric > 0:
        flips = rng.random(bits.shape) < p_symmetric
        bits ^= flips.astype(np.int8)
    if p_decay > 0:
        decay = (bits == 1) & (rng.random(bits.shape) < p_decay)
        bits[decay] = 0
    return bits


def test_zero_decay_is_a_no_op():
    rng = np.random.default_rng(0)
    bits = np.array([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int8)
    out = _apply_symmetric_then_decay(bits, p_symmetric=0.0, p_decay=0.0, rng=rng)
    assert np.array_equal(out, bits)


def test_decay_never_flips_zero_to_one():
    rng = np.random.default_rng(0)
    bits = np.zeros((1000, 8), dtype=np.int8)  # all zeros
    out = _apply_symmetric_then_decay(bits, p_symmetric=0.0, p_decay=0.9, rng=rng)
    assert np.array_equal(out, bits), "decay-only pass must never turn a 0 into a 1"


def test_decay_only_flips_ones_to_zero_at_expected_rate():
    rng = np.random.default_rng(0)
    n_trials = 20_000
    p_decay = 0.3
    bits = np.ones((n_trials, 1), dtype=np.int8)  # all ones
    out = _apply_symmetric_then_decay(bits, p_symmetric=0.0, p_decay=p_decay, rng=rng)
    observed_decay_rate = 1.0 - out.mean()
    assert abs(observed_decay_rate - p_decay) < 0.02, (
        f"observed decay rate {observed_decay_rate:.3f} too far from p_decay={p_decay}"
    )


def test_symmetric_flip_can_go_either_direction():
    rng = np.random.default_rng(0)
    n_trials = 20_000
    p = 0.5
    bits = np.zeros((n_trials, 1), dtype=np.int8)
    out = _apply_symmetric_then_decay(bits, p_symmetric=p, p_decay=0.0, rng=rng)
    # unlike decay, a symmetric flip on all-zero input CAN produce ones
    assert out.mean() > 0.0, "symmetric flip must be able to turn 0 into 1 (unlike decay)"


def test_amplitude_damping_p_formula():
    from sqd_gqe.hw.backends import HardwareBackend

    backend = HardwareBackend(
        name="test", n_qubits=10, basis_gates=("cz",), heavy_hex_distance=3,
        t1_us=200.0, gate_time_2q_ns=100.0,
    )
    # p = 1 - exp(-(n_2q * gate_time_ns * 1e-3) / t1_us)
    n_2q = 500
    expected = 1.0 - math.exp(-(500 * 100.0 * 1e-3) / 200.0)
    assert abs(backend.amplitude_damping_p(n_2q) - expected) < 1e-12


def test_amplitude_damping_p_zero_without_t1():
    from sqd_gqe.hw.backends import HardwareBackend

    backend = HardwareBackend(
        name="ideal", n_qubits=10, basis_gates=("cz",), heavy_hex_distance=3,
    )
    assert backend.amplitude_damping_p(1000) == 0.0


def test_amplitude_damping_p_zero_at_zero_gates():
    from sqd_gqe.hw.backends import HardwareBackend

    backend = HardwareBackend(
        name="test", n_qubits=10, basis_gates=("cz",), heavy_hex_distance=3,
        t1_us=200.0, gate_time_2q_ns=100.0,
    )
    assert backend.amplitude_damping_p(0) == 0.0


def test_amplitude_damping_p_increases_with_gate_count():
    from sqd_gqe.hw.backends import HardwareBackend

    backend = HardwareBackend(
        name="test", n_qubits=10, basis_gates=("cz",), heavy_hex_distance=3,
        t1_us=200.0, gate_time_2q_ns=100.0,
    )
    p_small = backend.amplitude_damping_p(10)
    p_large = backend.amplitude_damping_p(10_000)
    assert 0.0 < p_small < p_large < 1.0


class _FakePool:
    """Minimal stand-in for FfsimBasedPool -- FfsimSampler.__init__ only
    reads .norb/.nelec, never touches ffsim itself at construction time."""
    norb = 4
    nelec = (2, 2)


def test_noisy_sampler_default_amplitude_damping_off():
    from sqd_gqe.gqe.noisy_sampler import NoisyFfsimSampler

    sampler = NoisyFfsimSampler(operator_pool=_FakePool(), shots_count=100, p_readout=0.0)
    assert sampler.amplitude_damping is False
    assert sampler._amplitude_damping_p(n_2q=1000) == 0.0


def test_noisy_sampler_amplitude_damping_requires_backend():
    from sqd_gqe.gqe.noisy_sampler import NoisyFfsimSampler

    sampler = NoisyFfsimSampler(
        operator_pool=_FakePool(), shots_count=100, p_readout=0.0,
        backend=None, amplitude_damping=True,
    )
    # no backend -> no T1 data -> decay probability stays 0 regardless of the flag
    assert sampler._amplitude_damping_p(n_2q=1000) == 0.0


def test_noisy_sampler_amplitude_damping_uses_backend_t1():
    from sqd_gqe.gqe.noisy_sampler import NoisyFfsimSampler
    from sqd_gqe.hw.backends import HardwareBackend

    backend = HardwareBackend(
        name="test", n_qubits=10, basis_gates=("cz",), heavy_hex_distance=3,
        t1_us=200.0, gate_time_2q_ns=100.0,
    )
    sampler = NoisyFfsimSampler(
        operator_pool=_FakePool(), shots_count=100, p_readout=0.0,
        backend=backend, amplitude_damping=True,
    )
    p = sampler._amplitude_damping_p(n_2q=500)
    assert p == backend.amplitude_damping_p(500)
    assert p > 0.0
