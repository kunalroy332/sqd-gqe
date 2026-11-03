# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Readout-, gate-, and amplitude-damping-noise wrapper around the ideal
ffsim sampler.

ffsim is an exact fermionic statevector simulator with no device noise. This
wrapper models three noise sources on top of ideal sampling, applied as
post-hoc per-bit transforms on the clean sampled counts (finite-shot):

1. **Measurement (readout) noise**: each sampled qubit bit is independently
   flipped with probability ``p_readout`` (symmetric: 0<->1 equally likely).

2. If ``backend`` is given, an *approximate* **gate-noise** term: the
   backend's ``err_2q`` and the circuit's device-native two-qubit-gate count
   (via ``circuit_metrics.circuit_depth_on_backend``) give an aggregate
   circuit failure probability ``1-(1-err_2q)**n_2q``, spread evenly across
   the 2*norb measured bits as an extra SYMMETRIC flip probability. This is a
   PROXY, not a full qiskit-aer statevector-noise simulation (ffsim's fast
   fermionic simulator has no such path, which is exactly what makes ffsim
   fast) -- see docs/presentation_amortized.tex for the explicit caveat, and
   HardwareBackend.noise_model() for the (currently unused for ffsim)
   qiskit-aer NoiseModel this proxy is a cheap stand-in for.

3. If ``backend`` is given AND ``amplitude_damping=True``, a **directional**
   T1-relaxation term: each measured bit currently 1 decays to 0 with
   probability derived from the backend's ``t1_us``/``gate_time_2q_ns`` and
   this circuit's real 2q-gate count (``HardwareBackend.amplitude_damping_p``)
   -- NOT a symmetric flip. This models the dominant real-hardware failure
   mode observed at 38-56 qubit scale: real IBM hardware sampling of trained
   circuits showed 77-100% of shots collapsing to the all-zero bitstring, not
   a roughly-uniform spread of electron counts around the target -- the
   signature of amplitude damping (decay toward |0>), not symmetric
   readout/gate-flip noise. Neither term 1 nor term 2 can produce or teach
   resilience against this specific channel, which is why it's added
   separately rather than folded into the existing symmetric ``p``. Like the
   gate-noise term, this is a PER-CIRCUIT aggregate applied uniformly to
   every measured qubit, not a per-qubit idle-time-aware schedule -- a qubit
   touched only once early in a long circuit decays the same amount here as
   one touched throughout.

Symmetric flips (terms 1+2) are applied first, then the directional decay
(term 3) -- modeling "gate/readout noise scrambles some bits, then relaxation
is observed at measurement time." This ordering is fixed for reproducibility,
not randomized.

Either way, this is exactly the regime SQD's configuration recovery is
designed to tolerate: bit flips break particle-number / spin symmetry, and the
recovery loop repairs them from the average orbital occupancies. (Amplitude
damping's collapse can be severe enough that recovery has no valid
Hamming-weight bitstrings to start from at all -- see
benchmark/dim_sweep_hardware_counts.py's real-hardware findings.)
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from sqd_gqe.gqe.sampler import FfsimSampler


class NoisyFfsimSampler(FfsimSampler):
    def __init__(self, operator_pool, shots_count, p_readout: float = 0.01,
                 seed: int | None = 0, rayon_threads: int | None = None, backend=None,
                 amplitude_damping: bool = False, p_readout_beta: float | None = None,
                 depth_noise: tuple | None = None):
        super().__init__(operator_pool, shots_count, seed=seed, rayon_threads=rayon_threads)
        self.p_readout = float(p_readout)
        # optional separate flip probability for the beta register (None: same as alpha). Real devices put the two registers on
        # different qubits, and the Training-2 hardware runs show a strong alpha/beta asymmetry that symmetric flips cannot produce.
        self.p_readout_beta = self.p_readout if p_readout_beta is None else float(p_readout_beta)
        self.backend = backend
        self.amplitude_damping = bool(amplitude_damping)
        # depth_noise = (d0, tau, p_max): per-bit flip probability p_max * (1 - exp(-max(depth - d0, 0) / tau)) from the circuit's transpiled depth on
        # `backend`, on top of p_readout. Calibrated on 3000 ibm_kobe circuits (23 orbitals): depth 45-60 -> 0.017, 60-100 -> 0.07, >= 300 -> ~0.4-0.5.
        self.depth_noise = None if depth_noise is None else tuple(float(x) for x in depth_noise)
        self._noise_rng = np.random.default_rng((seed or 0) + 20260717)

    def _circ_metrics(self, seq) -> dict:
        if self.backend is None:
            return {}
        from sqd_gqe.gqe import circuit_metrics as cm
        try:
            # optimisation level 1, as in the trainer's own transpile calls (level 3 segfaults in the aarch64 qiskit build on roquo)
            return dict(cm.circuit_depth_on_backend(self.pool, seq, self.backend, optimization_level=1))
        except Exception:
            return {}

    def _n_2q_gates(self, seq) -> int:
        return int(self._circ_metrics(seq).get("two_qubit_gates", 0))

    def _gate_noise_p(self, n_2q: int) -> float:
        if self.backend is None:
            return 0.0
        p_fail_circuit = 1.0 - (1.0 - self.backend.err_2q) ** max(int(n_2q), 0)
        return p_fail_circuit / max(2 * self.norb, 1)

    def _amplitude_damping_p(self, n_2q: int) -> float:
        """Directional P(1->0) for this circuit, from the backend's T1 and
        its real 2q-gate count. 0.0 if amplitude_damping is off or the
        backend has no t1_us/gate_time_2q_ns (e.g. "ideal")."""
        if not self.amplitude_damping or self.backend is None:
            return 0.0
        return self.backend.amplitude_damping_p(n_2q)

    def sample_state(self, seq) -> dict[tuple[int, int], int]:
        # ideal counts first, then apply per-bit flips (readout + gate-noise
        # proxy, both symmetric), then directional amplitude-damping decay.
        clean = super().sample_state(seq)
        m = self._circ_metrics(seq)  # single transpile, shared by all backend-derived terms
        n_2q = int(m.get("two_qubit_gates", 0))
        p_gate = 0.0 if self.depth_noise is not None else self._gate_noise_p(n_2q)
        norb = self.norb
        n = 2 * norb
        p_dep = 0.0
        if self.depth_noise is not None:
            d0, tau, pmax = self.depth_noise
            p_dep = pmax * (1.0 - np.exp(-max(float(m.get("depth", 0)) - d0, 0.0) / tau))
        p_read = np.maximum(np.where(np.arange(n) < norb, self.p_readout, self.p_readout_beta), p_dep)       # bits 0..norb-1 alpha, norb..2norb-1 beta
        p_symmetric = p_read + p_gate - p_read * p_gate  # independent-flip union, per bit
        p_decay = self._amplitude_damping_p(n_2q)
        if float(p_symmetric.max()) <= 0 and p_decay <= 0:
            return clean
        # expand to a per-shot integer array, flip bits vectorized
        words = np.array(
            [((int(b) << norb) | int(a)) for (a, b), c in clean.items() for _ in range(int(c))],
            dtype=np.int64,
        )
        if words.size == 0:
            return {}
        bits = ((words[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)  # (S, n)
        if float(p_symmetric.max()) > 0:
            flips = self._noise_rng.random(bits.shape) < p_symmetric[None, :]
            bits ^= flips.astype(np.int8)
        if p_decay > 0:
            # directional: only bits currently 1 can decay to 0; 0 never becomes 1.
            decay = (bits == 1) & (self._noise_rng.random(bits.shape) < p_decay)
            bits[decay] = 0
        noisy = (bits.astype(np.int64) << np.arange(n)[None, :]).sum(axis=1)
        mask = (1 << norb) - 1
        alpha = (noisy & mask).tolist()
        beta = (noisy >> norb).tolist()
        counts: Counter = Counter(zip(alpha, beta))
        return dict(counts)
