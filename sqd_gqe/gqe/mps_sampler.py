# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""GPU MPS sampling with cuQuantum NetworkState, for CI vectors too large for ffsim.

Cost and accuracy depend on the circuit's bond dimension. SVD truncation can leak
particle number; ``last_valid_fraction`` records the surviving shot fraction.
``MpsTopKSampler`` instead selects valid determinants without shot noise.
Bit order is detected with a Hartree-Fock probe before decoding any result.

Usage and measurements: docs/mps.md.
Keep the cuquantum library and cuquantum-python at the same CUDA major version;
mixed versions can abort with CUTENSORNET_STATUS_INVALID_VALUE.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from typing import Sequence

import numpy as np
import torch

from sqd_gqe.gqe.noisy_sampler import MeasurementNoiseMixin
from sqd_gqe.gqe.operator_pool import FfsimBasedPool

# cuTensorNet MPS requires one- or two-site gates; decompose dense JW operators.
BASIS = ("rz", "ry", "rx", "x", "h", "cx", "cz", "swap")

# Level 3 can segfault on ucj_balanced_jw (see ibm_hardware_sampler).
OPTIMIZATION_LEVEL = 1


class MpsSampler:
    """Sample the pool's circuits from a truncated MPS on the GPU.

    Args:
        operator_pool: pool with ``qiskit_gate_fn`` on every non-identity op.
        shots_count: shots per circuit.
        max_bond: SVD bond limit (``MPSConfig.max_extent``).
        abs_cutoff, rel_cutoff: additional singular-value cutoffs.
        dtype: ``complex64`` or ``complex128``.
        route: transpile onto a nearest-neighbour chain.
        max_gates: skip circuits exceeding this gate count, returning an empty
            subspace. Off by default; see docs/mps.md for the tradeoff.
        gauge_option: ``MPSConfig.gauge_option``. ``"simple"`` (the default here) keeps
            simple-update gauges, so truncation drops true Schmidt weight. ``"free"`` or
            None is cuQuantum's own default and is far less accurate at a fixed chi:
            at 38q, chi=512 reached 0.7 fidelity where ``"simple"`` converged at 256
            (docs/mps.md).
        canonical_center: ``MPSConfig.canonical_center``, the site the final MPS is
            canonicalized around; None leaves it uncanonicalized.
    """

    def __init__(
        self,
        operator_pool: FfsimBasedPool,
        shots_count: int,
        max_bond: int = 256,
        abs_cutoff: float = 1e-5,
        rel_cutoff: float = 1e-5,
        dtype: str = "complex64",
        route: bool = False,
        max_gates: int | None = None,
        gauge_option: str | None = "simple",
        canonical_center: int | None = None,
        seed: int | None = 0,
        seed_transpiler: int = 0,
        verbose: bool = True,
    ):
        offenders = sorted({
            op.name.split("(")[0].split("[")[0]
            for op in operator_pool
            if op.name != "I" and op.qiskit_gate_fn is None
        })
        if offenders:
            raise ValueError(
                f"MpsSampler requires every op to have qiskit_gate_fn set; "
                f"missing for op families: {offenders}"
            )

        self.pool = operator_pool
        self.shots_count = int(shots_count)
        self.norb = operator_pool.norb
        self.nelec = tuple(operator_pool.nelec)
        self.max_bond = int(max_bond)
        self.abs_cutoff = float(abs_cutoff)
        self.rel_cutoff = float(rel_cutoff)
        self.dtype = dtype
        self.route = bool(route)
        self.max_gates = int(max_gates) if max_gates else None
        self.gauge_option = gauge_option
        self.canonical_center = None if canonical_center is None else int(canonical_center)
        self.seed_transpiler = int(seed_transpiler)
        self.verbose = bool(verbose)
        self._rng = np.random.default_rng(seed)
        self._bit_order = None          # detected on first use
        #: diagnostics from the most recent circuit, read by callers that log them
        self.last_valid_fraction = None
        self.last_timings: dict[str, float] = {}

    # -- circuit construction ------------------------------------------------ #
    def prepare_circuit(self, seq: Sequence[int]):
        """Hartree-Fock reference plus the sequence's gates, in ffsim's blocked
        Jordan-Wigner order (alpha on qubits [0,norb), beta on [norb,2*norb))."""
        from qiskit import QuantumCircuit

        n_qubits = 2 * self.norb
        n_alpha, n_beta = self.nelec
        qc = QuantumCircuit(n_qubits)
        for orbital in range(n_alpha):
            qc.x(orbital)
        for orbital in range(n_beta):
            qc.x(self.norb + orbital)
        for idx in seq:
            op = self.pool[int(idx)]
            if op.name == "I":
                continue
            qc.append(op.qiskit_gate_fn(self.norb), range(n_qubits))
        return qc

    def transpile(self, qc):
        """Decompose to BASIS, optionally routing onto a chain.

        Returns ``(transpiled circuit, layout)`` where ``layout[i]`` is the physical
        position holding logical qubit ``i``, or None when not routed.
        """
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap

        cmap = CouplingMap.from_line(qc.num_qubits) if self.route else None
        tqc = transpile(qc, basis_gates=list(BASIS), coupling_map=cmap,
                        optimization_level=OPTIMIZATION_LEVEL,
                        seed_transpiler=self.seed_transpiler)
        layout = None
        if self.route and tqc.layout is not None:
            layout = list(tqc.layout.final_index_layout())
        return tqc, layout

    # -- MPS state ----------------------------------------------------------- #
    def _network_state(self, tqc):
        from cuquantum.tensornet.experimental import MPSConfig, NetworkState

        # None keeps cuQuantum's own default (gauge "free", no canonicalization).
        extra = {k: v for k, v in (("gauge_option", self.gauge_option),
                                   ("canonical_center", self.canonical_center)) if v is not None}
        config = MPSConfig(max_extent=self.max_bond, abs_cutoff=self.abs_cutoff,
                           rel_cutoff=self.rel_cutoff, **extra)
        return NetworkState.from_circuit(tqc, dtype=self.dtype, config=config, backend="cupy")

    def _draw_seed(self) -> int:
        # cuQuantum rejects seed=0 ("seed must be a positive integer")
        return int(self._rng.integers(1, 2**31 - 1))

    # -- bit order ----------------------------------------------------------- #
    def _detect_bit_order(self) -> str:
        """Detect which end of a sampling string holds qubit 0 with an HF probe."""
        tqc, layout = self.transpile(self.prepare_circuit([]))
        state = self._network_state(tqc)
        try:
            counts = state.compute_sampling(100, seed=1)
        finally:
            _free(state)
        if len(counts) != 1:
            raise RuntimeError(
                f"Hartree-Fock probe returned {len(counts)} distinct bitstrings "
                f"(expected 1): {list(counts)[:3]}")
        bits = next(iter(counts))
        n = 2 * self.norb
        expect = "".join("1" if (i < self.nelec[0] or self.norb <= i < self.norb + self.nelec[1])
                         else "0" for i in range(n))
        for order in ("qubit0_left", "qiskit"):
            probe = bits if order == "qubit0_left" else bits[::-1]
            if _permute(probe, layout) == expect:
                return order
        raise RuntimeError(
            f"Hartree-Fock probe bitstring {bits!r} (layout {layout}) matches neither "
            f"qubit order; expected {expect!r} or its reverse.")

    def _decode(self, counts_bits: dict, layout) -> dict[tuple[int, int], int]:
        """cuQuantum bitstrings -> ``{(alpha_occ, beta_occ): count}``, ffsim's convention."""
        mask = (1 << self.norb) - 1
        counts: Counter = Counter()
        for bits, n in counts_bits.items():
            b = bits if self._bit_order == "qubit0_left" else bits[::-1]
            b = _permute(b, layout)           # physical position -> logical qubit
            s = int(b[::-1], 2)               # b[i] is qubit i, the int's bit i
            counts[(s & mask, s >> self.norb)] += int(n)
        return dict(counts)

    def _exceeds_max_gates(self, tqc) -> bool:
        """Skip over-budget circuits; empty subspaces receive +inf energy."""
        if self.max_gates is None or tqc.size() <= self.max_gates:
            return False
        self.last_valid_fraction = 0.0
        self._log(f"{tqc.size()} gates exceeds max_gates={self.max_gates}: skipped, "
                  f"scored as an empty subspace")
        return True

    def _prepare(self, seq: Sequence[int]):
        """Detect bit order once, then return ``(transpiled circuit, layout, seconds)``."""
        if self._bit_order is None:
            self._bit_order = self._detect_bit_order()
            self._log(f"bit order: {self._bit_order}")
        t = time.perf_counter()
        tqc, layout = self.transpile(self.prepare_circuit(seq))
        return tqc, layout, time.perf_counter() - t

    # -- sampling ------------------------------------------------------------ #
    def sample_state(self, seq: Sequence[int], seed: int | None = None) -> dict[tuple[int, int], int]:
        tqc, layout, t_transpile = self._prepare(seq)

        if self._exceeds_max_gates(tqc):
            self.last_timings = {"transpile": t_transpile, "sample": 0.0}
            return {}

        state = self._network_state(tqc)
        try:
            t = time.perf_counter()
            shot_seed = self._draw_seed() if seed is None else (int(seed) % (2**31 - 2)) + 1
            counts_bits = state.compute_sampling(self.shots_count, seed=shot_seed)
            t_sample = time.perf_counter() - t
        finally:
            _free(state)

        counts = self._decode(counts_bits, layout)
        self.last_timings = {"transpile": t_transpile, "sample": t_sample}
        self.last_valid_fraction = _valid_fraction(counts, self.nelec)
        self._log(f"{tqc.size()} gates -> {len(counts)} distinct bitstrings, "
                  f"particle-number-valid {self.last_valid_fraction:.3f} "
                  f"(transpile {t_transpile:.1f}s, contract+sample {t_sample:.1f}s, "
                  f"chi<={self.max_bond})")
        return counts

    @torch.no_grad()
    def run(self, state: dict) -> list[dict[tuple[int, int], int]]:
        idx_output = state["idx"]
        if torch.is_tensor(idx_output):
            idx_output = idx_output.detach().cpu().tolist()
        seeds = [self._draw_seed() for _ in idx_output]
        return [self.sample_state(row, seed=s) for row, s in zip(idx_output, seeds)]

    def close(self):
        """API parity with FfsimSampler's worker-pool teardown; nothing to release."""

    def _log(self, msg: str):
        if self.verbose:
            print(f"[mps] {msg}", file=sys.stderr, flush=True)



class NoisyMpsSampler(MeasurementNoiseMixin, MpsSampler):
    """``NoisyFfsimSampler``'s device noise on top of MPS shot sampling.

    At 46q ffsim cannot build the state, so this is the way to score a circuit the way a
    device would. The noise is the mixin's, shared with the ffsim path: clean shots, then
    per-bit flips scaled by the circuit's transpiled two-qubit count.

    ``last_valid_fraction`` is recomputed on the noisy counts, since that is what
    configuration recovery has to repair.
    """

    def __init__(self, operator_pool, shots_count, p_readout: float = 0.0,
                 seed: int | None = 0, backend=None, amplitude_damping: bool = False,
                 noise_optimization_level: int = 3, p_readout_beta: float | None = None,
                 depth_noise: tuple | None = None, **kwargs):
        super().__init__(operator_pool, shots_count=shots_count, seed=seed, **kwargs)
        self._init_noise(p_readout, backend, amplitude_damping, seed, noise_optimization_level,
                         p_readout_beta=p_readout_beta, depth_noise=depth_noise)

    def sample_state(self, seq, seed: int | None = None):
        counts = super().sample_state(seq, seed=seed)
        self.last_valid_fraction = _valid_fraction(counts, self.nelec)
        return counts

# -- helpers, kept module-level and dependency-free so tests can exercise them -- #
def _permute(bits: str, layout) -> str:
    """``bits[j]`` is physical position j; return the string indexed by logical qubit."""
    if layout is None:
        return bits
    return "".join(bits[layout[i]] for i in range(len(layout)))


def _valid_fraction(counts: dict[tuple[int, int], int], nelec) -> float:
    """Fraction of shots with the target electron count per spin."""
    na, nb = nelec
    total = sum(counts.values())
    if not total:
        return 0.0
    ok = sum(n for (a, b), n in counts.items()
             if int(a).bit_count() == na and int(b).bit_count() == nb)
    return ok / total


def _free(state):
    try:
        state.free()
    except Exception:
        pass
