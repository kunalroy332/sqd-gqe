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

"""Hardware backend specifications for device-aware circuit generation.

Defines target IBM devices (Heron r2 / r3, Nighthawk r1) by their **native gate
set**, **coupling map** (heavy-hex for Heron, square for Nighthawk), and a
depth budget. Circuit generation/transpilation stays fully offline (no IBM
credentials or network), but the err_1q/err_2q/err_readout numbers below are
now real, dated calibration snapshots pulled from the IBM Quantum Platform
device pages (Heron r2 v1.0.0, Heron r3 v1.0.4, Nighthawk r1 v1.0.3; Washington
DC region) rather than estimated placeholders -- see each entry's ``metadata``
for the exact source figures (T1/T2, per-error-type breakdown, physical vs
programmable qubit count).

Heron r2 and r3 are BOTH 156-qubit heavy-hex chips (contrary to this repo's
earlier assumption that r3 was simply "a larger lattice") -- r3 is the same
size with meaningfully better calibrated error rates. Both expose
{cz, id, rx, rz, rzz, sx, x} natively; Nighthawk r1 (120q square lattice, newer
architecture, still ramping up) exposes {cz, id, rx, rz, sx, x} (no native
rzz) and, per the real data, has WORSE 2Q/readout error than either Heron
revision, not better as this repo previously placeholder-guessed -- see
``ibm_nighthawk_r1``'s metadata note. ``err_2q`` is populated from each
device's "layered" 2Q error (IBM's crosstalk-inclusive, real-circuit-condition
estimate) rather than the isolated "best"/"median" figures, since that's the
closest single number to what NoisyFfsimSampler's aggregate per-circuit
failure-probability proxy is trying to approximate; see ``noise_model`` hook
for the (currently unused for ffsim) qiskit-aer NoiseModel these numbers feed.

**Coupling-map caveat found 2026-08-12**: ``coupling_map()``'s idealized
``CouplingMap.from_heavy_hex(heavy_hex_distance)`` construction can only ever
produce qubit counts of 19/57/115/193/291/... (f(distance) for a regular
heavy-hex lattice) -- **156 is not reachable at any distance value**.
``ibm_heron_r2``/``ibm_heron_r3`` (``heavy_hex_distance=9``) have therefore
always transpiled against a 193-qubit/448-edge idealized map, not their real
156-qubit/352-edge topology -- confirmed directly (``CouplingMap.from_heavy_hex
(9).size() == 193``, not 156). This affects every past ``--cost-lambda-real-depth``
training run that used a Heron backend: the ISA-circuit reward was shaped
against a real-but-wrong-sized topology, not the exact device. Left as-is for
``ibm_heron_r2``/``ibm_heron_r3``/``ibm_nighthawk_r1`` (consistent, internally-
comparable approximation); ``ibm_kingston`` (added the same day this was found)
uses the new ``real_coupling_edges`` field instead -- see its docstring below
and ``coupling_map()``'s implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


HERON_BASIS = ["cz", "id", "rx", "rz", "rzz", "sx", "x"]
#: Heron's REAL native basis. ibm_kingston's was confirmed against the live service
#: (target.operation_names) as {cz, id, rz, sx, x} -- no rzz, no rx -- and the same H2
#: LUCJ circuit measured depth 55 under HERON_BASIS against 123 under this one, because
#: HERON_BASIS lets the transpiler take a decomposition the device cannot execute. The
#: r2/r3 entries below use this rather than HERON_BASIS: they are the same 156q Heron
#: generation as kingston, so claiming they natively support rzz while kingston does not
#: would need positive evidence, and depth measured against gates the hardware lacks is
#: not a depth the hardware will honour. Neither r2 nor r3 has been queried live --
#: inferred from same-generation kingston, not confirmed. HERON_BASIS itself is kept
#: unchanged for any caller that wants the old behaviour.
HERON_REAL_BASIS = ["cz", "id", "rz", "sx", "x"]
#: Gate names that act on two qubits, for cost()'s two-qubit count. rzz was missing
#: here originally, and rzz is in HERON_BASIS -- so every operator the transpiler
#: routed through rzz counted as zero two-qubit gates. That silently disqualified 25
#: of the 32 pool entries at N2 (14 givens, 7 num_num, 4 orbital rotations), because
#: the training gate rejects a circuit when two_q == 0. Measured at
#: optimization_level=0, where the transpiler rewrites nothing, one givens carries 54
#: two-qubit gates and one orbital rotation 273 -- they were never empty. Only the
#: three cz-routed operators survived the miscount (LUCJ at depth 652, two
#: diag_coulomb at 475), and all three exceed the depth-300 ceiling, which is why no
#: molecule above 12 qubits ever passed the gate and training collapsed onto H2.
#: Listed by name rather than inferred from qubit count so it stays explicit which
#: basis each backend emits; entries beyond the Heron basis cover other backends and
#: the pre-optimisation forms seen at optimization_level=0.
TWO_QUBIT_GATE_NAMES = ("cz", "cx", "ecr", "rzz", "xx_plus_yy", "cp", "swap",
                        "iswap", "cy", "ch", "crx", "cry", "crz", "csx", "rxx",
                        "ryy", "rzx")
NIGHTHAWK_BASIS = ["cz", "id", "rx", "rz", "sx", "x"]


@dataclass(frozen=True)
class HardwareBackend:
    name: str
    n_qubits: int
    basis_gates: tuple[str, ...]
    #: heavy-hex distance parameter for CouplingMap.from_heavy_hex (odd int);
    #: only used when lattice == "heavy_hex"
    heavy_hex_distance: int
    #: soft two-qubit-depth budget used by the cost reward / feasibility check
    depth_budget: int = 5000
    #: representative gate error rates for an optional aer noise model
    err_1q: float = 2.5e-4
    err_2q: float = 3.0e-3
    err_readout: float = 1.0e-2
    metadata: dict = field(default_factory=dict)
    #: "heavy_hex" (Heron) or "square" (Nighthawk's 12x10 grid)
    lattice: str = "heavy_hex"
    #: (rows, cols) for lattice == "square"; unused for heavy_hex
    grid_shape: tuple[int, int] | None = None
    #: what these numbers describe -- "real" (an actual device's published
    #: specs, even though queried offline here), "noisy_sim" (this entry is
    #: meant to be run through a noisy simulator, e.g. NoisyFfsimSampler's
    #: gate-noise proxy), or "ideal" (no noise at all). Feeds the conditioning
    #: descriptor so the policy can distinguish these regimes -- see
    #: core/conditioning.py's molecule_features().
    sim_kind: str = "real"
    #: T1 relaxation time in microseconds (amplitude-damping timescale) --
    #: promoted from metadata["t1_us"] to a first-class field since
    #: NoisyFfsimSampler's amplitude-damping term now reads it computationally,
    #: not just for reference. None for "ideal" (no decay).
    t1_us: float | None = None
    #: representative two-qubit-gate duration in nanoseconds, used only to
    #: turn a circuit's real 2q-gate count (already computed by cost()) into
    #: an approximate elapsed-time-on-qubit estimate for T1 decay -- not a
    #: real per-qubit schedule (idle time between gates on a given qubit is
    #: not modeled). None for "ideal".
    gate_time_2q_ns: float | None = None
    #: exact, real coupling-map edges (undirected, deduplicated), fetched
    #: once from a live QiskitRuntimeService backend and cached here so
    #: coupling_map() doesn't need live credentials at training/reward time.
    #: When set, coupling_map() uses this EXACT topology instead of the
    #: idealized from_heavy_hex/from_grid construction -- see the module
    #: docstring's "Coupling-map caveat" note for why this exists (heavy-hex
    #: lattices can't reproduce every real qubit count, e.g. 156).
    real_coupling_edges: tuple[tuple[int, int], ...] | None = None

    def amplitude_damping_p(self, n_2q: int) -> float:
        """P(a qubit in |1> decays to |0>) over a circuit with n_2q real
        two-qubit gates, from this backend's T1 and gate_time_2q_ns:
        p = 1 - exp(-t_circuit / T1). Returns 0.0 if t1_us/gate_time_2q_ns
        aren't set (e.g. the "ideal" backend)."""
        if not self.t1_us or not self.gate_time_2q_ns:
            return 0.0
        import math
        elapsed_us = max(int(n_2q), 0) * self.gate_time_2q_ns * 1e-3
        return 1.0 - math.exp(-elapsed_us / self.t1_us)

    def coupling_map(self):
        from qiskit.transpiler import CouplingMap
        if self.real_coupling_edges is not None:
            return CouplingMap(couplinglist=list(self.real_coupling_edges))
        if self.lattice == "square":
            rows, cols = self.grid_shape
            return CouplingMap.from_grid(rows, cols)
        return CouplingMap.from_heavy_hex(self.heavy_hex_distance)

    def transpile(self, circuit, optimization_level: int = 3, seed_transpiler: int = 0):
        """Transpile a circuit to this device's native gates + connectivity.

        seed_transpiler is fixed (not None) by default: qiskit's SABRE
        layout/routing search is randomized, and without a fixed seed the SAME
        logical circuit can transpile to depths differing by ~2x between
        calls (observed directly: one ethanol circuit measured at depth 653
        and 978 across two otherwise-identical runs) -- every depth/2q-count
        comparison in this codebase's reward and analysis scripts is
        meaningless noise without this pinned.
        """
        from qiskit import transpile
        cm = self.coupling_map()
        # if the circuit needs more qubits than the map, fall back to basis-only
        if circuit.num_qubits > cm.size():
            return transpile(circuit, basis_gates=list(self.basis_gates),
                             optimization_level=optimization_level, seed_transpiler=seed_transpiler)
        return transpile(circuit, coupling_map=cm, basis_gates=list(self.basis_gates),
                         optimization_level=optimization_level, seed_transpiler=seed_transpiler)

    def cost(self, circuit, optimization_level: int = 3) -> dict:
        """Device-native depth + two-qubit-gate count for a circuit.

        optimization_level defaults to qiskit's most thorough (routing-aware)
        pass -- level 3's more exhaustive SABRE layout/routing search is what
        actually keeps a logically-local (e.g. per-spin-sector Givens network)
        circuit's SWAP overhead down once mapped onto Heron's heavy-hex graph;
        level 2 leaves real overhead on the table. Pass a lower level (0-1)
        for cheap, approximate, repeated calls -- e.g. per-training-iteration
        diagnostics -- where routing-search cost would dominate wall-time for
        a number that's only ever plotted as a trend.
        """
        tqc = self.transpile(circuit, optimization_level=optimization_level)
        ops = tqc.count_ops()
        two_q = sum(v for k, v in ops.items() if k in TWO_QUBIT_GATE_NAMES)
        # active_qubits counts LOGICAL qubits on the pre-transpile circuit,
        # not post-transpile physical indices: when circuit.num_qubits <=
        # cm.size() (the common case -- every real training run this
        # session used norb<=16, far under Heron's 193-physical-qubit
        # coupling map), SABRE layout/routing at optimization_level>=1 can
        # legitimately spread the circuit across MORE physical qubits than
        # its own logical width (SWAP-based routing widens the physical
        # footprint) -- confirmed directly: a real 56-logical-qubit circuit
        # transpiled to 61 distinct physical qubit indices. Counting
        # post-transpile physical qubits would silently reward routing
        # spread, not genuine logical-qubit usage -- the wrong signal for
        # --cost-lambda-coverage (real bug found 2026-09-30 while drawing
        # zero-shot inference circuits, see
        # scripts/analysis/draw_coverage_56q_circuits.py).
        active_qubits = {
            circuit.find_bit(q).index for instr in circuit.data for q in instr.qubits
        }
        return {
            "depth": int(tqc.depth()),
            "two_qubit_gates": int(two_q),
            "total_gates": int(sum(ops.values())),
            "within_budget": int(tqc.depth()) <= self.depth_budget,
            "fits_qubits": circuit.num_qubits <= self.coupling_map().size(),
            "active_qubits": len(active_qubits),
        }

    def noise_model(self):
        """A simple depolarizing + readout aer NoiseModel from the device error
        rates (offline surrogate for live backend properties)."""
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
        nm = NoiseModel(basis_gates=list(self.basis_gates))
        nm.add_all_qubit_quantum_error(depolarizing_error(self.err_1q, 1), ["rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(self.err_2q, 2), ["cz"])
        p = self.err_readout
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
        return nm


#: 12x10 square grid = 120 qubits, 218 couplers -- matches IBM's published
#: Nighthawk spec exactly (10*(12-1) + (10-1)*12 = 110+108 = 218). See
#: https://www.ibm.com/quantum/blog/large-scale-ftqc and the Nov-2025 IBM
#: Nighthawk/Loon announcement.
NIGHTHAWK_GRID = (10, 12)


def _load_real_coupling_edges(json_filename: str) -> tuple[tuple[int, int], ...]:
    """Load a real device's exact coupling-map edges from a JSON file fetched
    once via QiskitRuntimeService (see sqd_gqe/hw/<device>_coupling_map.json).
    Used where the idealized from_heavy_hex/from_grid construction can't
    reproduce the real qubit count -- see module docstring's coupling-map
    caveat."""
    import json
    path = Path(__file__).resolve().parent / json_filename
    with open(path) as f:
        edges = json.load(f)
    return tuple(tuple(e) for e in edges)


# --- registry of known devices (offline specs) --------------------------- #
_BACKENDS = {
    # Live IBM Quantum Platform snapshot (Washington DC region), QPU v1.0.0.
    # err_1q = SX error (median) 2.445e-4; err_2q = 2Q error (layered) 3.10e-3
    # (CZ error median 1.966e-3, 2Q error median 1.97e-3, best 8.13e-4);
    # err_readout = readout error (median) 9.03e-3.
    "ibm_heron_r2": HardwareBackend(
        name="ibm_heron_r2", n_qubits=156, basis_gates=tuple(HERON_REAL_BASIS),
        heavy_hex_distance=9, depth_budget=5000,
        err_1q=2.445e-4, err_2q=3.10e-3, err_readout=9.03e-3,
        t1_us=243.57,
        # NOT from a live per-device page like the error rates above -- IBM
        # does not publish per-device CZ duration on the same device page;
        # this is a representative figure for the Heron/heavy-hex CZ gate
        # family (commonly cited in the ~120-160ns range), used only as an
        # elapsed-time proxy for NoisyFfsimSampler's amplitude-damping term.
        # Revisit if a real per-device number becomes available.
        gate_time_2q_ns=140.0,
        metadata={"family": "Heron", "revision": "r2", "qpu_version": "1.0.0",
                  "source": "live IBM Quantum Platform device page (Washington DC)",
                  "physical_qubits": 332, "t1_us": 243.57, "t2_us": 132.1,
                  "cz_error_median": 1.966e-3, "2q_error_median": 1.97e-3, "2q_error_best": 8.13e-4,
                  "max_circuits_per_second": 3800},
        sim_kind="real",
    ),
    # Live snapshot, QPU v1.0.4 -- same 156q heavy-hex chip as r2, better
    # calibrated error rates across the board (not a size difference).
    # err_1q = SX error (median) 1.725e-4; err_2q = 2Q error (layered) 2.28e-3
    # (CZ error median 1.456e-3, 2Q error median 1.46e-3, best 5.65e-4);
    # err_readout = readout error (median) 3.66e-3.
    "ibm_heron_r3": HardwareBackend(
        name="ibm_heron_r3", n_qubits=156, basis_gates=tuple(HERON_REAL_BASIS),
        heavy_hex_distance=9, depth_budget=6000,
        err_1q=1.725e-4, err_2q=2.28e-3, err_readout=3.66e-3,
        t1_us=255.27,
        # same representative-figure caveat as ibm_heron_r2 above -- no
        # per-device CZ duration published, this is the Heron-family estimate.
        gate_time_2q_ns=140.0,
        metadata={"family": "Heron", "revision": "r3", "qpu_version": "1.0.4",
                  "source": "live IBM Quantum Platform device page (Washington DC)",
                  "physical_qubits": 332, "t1_us": 255.27, "t2_us": 307.21,
                  "cz_error_median": 1.456e-3, "2q_error_median": 1.46e-3, "2q_error_best": 5.65e-4,
                  "max_circuits_per_second": 3800},
        sim_kind="real",
    ),
    # Kingston: same Heron r2 family/generation as ibm_heron_r2 above, 156q
    # heavy-hex, but this is its OWN device's live calibration snapshot, not
    # a copy of ibm_heron_r2's -- fetched directly via
    # QiskitRuntimeService.backend("ibm_kingston").properties() this session
    # (T1/T2 per-qubit medians, cz/sx/x gate errors, readout error, and --
    # notably -- a REAL measured cz gate_length, not the ~140ns Heron-family
    # estimate every other entry in this file has to fall back to).
    # err_1q = sx/x gate error (median) 2.816e-4; err_2q = cz gate error
    # (median) 2.0234e-3 (best 8.8168e-4); err_readout = readout error
    # (median) 1.001e-2.
    #
    # Unlike every other entry in this file, coupling_map() for this device
    # returns the EXACT real topology (176 undirected edges, cached in
    # ibm_kingston_coupling_map.json), not an idealized from_heavy_hex()
    # approximation -- confirmed heavy_hex_distance=9 cannot reproduce 156
    # qubits at all (only 19/57/115/193/291/... are reachable), so
    # heavy_hex_distance is set here only for documentation/consistency with
    # the other Heron entries; it is never actually used by coupling_map()
    # once real_coupling_edges is set. See module docstring's coupling-map
    # caveat for the full story.
    #
    # SECOND bug found the same day: HERON_BASIS includes rzz/rx, but
    # ibm_kingston's real native basis (confirmed via
    # service.backend("ibm_kingston").target.operation_names) is only
    # {cz, id, rz, sx, x} -- no rzz, no rx. Using HERON_BASIS let the
    # transpiler take a cheaper decomposition path that doesn't exist on the
    # real device (confirmed: depth 55 with the wrong basis vs. depth 123
    # with the real one, on the SAME H2 LUCJ circuit -- the real basis
    # matches a live transpile exactly, the wrong one doesn't). Overridden
    # here to this device's own real basis rather than changing HERON_BASIS
    # itself, since whether ibm_heron_r2/r3 ALSO lack real rzz/rx natively is
    # a separate, not-yet-verified question -- out of scope for this fix.
    "ibm_kingston": HardwareBackend(
        name="ibm_kingston", n_qubits=156, basis_gates=("cz", "id", "rz", "sx", "x"),
        heavy_hex_distance=9, depth_budget=8000,
        err_1q=2.816e-4, err_2q=2.0234e-3, err_readout=1.001e-2,
        t1_us=241.91,
        # REAL measured cz gate duration for this device (not a family
        # estimate) -- confirmed identical (68.0ns) across every edge sampled.
        gate_time_2q_ns=68.0,
        real_coupling_edges=_load_real_coupling_edges("ibm_kingston_coupling_map.json"),
        metadata={"family": "Heron", "revision": "r2", "qpu_version": None,
                  "source": "live IBM Quantum Platform device, fetched via "
                            "QiskitRuntimeService this session (2026-08-12)",
                  "t1_us": 241.91, "t2_us": 142.71,
                  "cz_error_median": 2.0234e-3, "cz_error_best": 8.8168e-4,
                  "cz_gate_length_ns": 68.0,
                  "num_edges": 352},
        sim_kind="real",
    ),
    # Nighthawk r1: 120q square lattice (not heavy-hex), tunable couplers, QPU
    # v1.0.3. Live snapshot -- err_1q = SX error (median) 2.293e-4; err_2q =
    # 2Q error (layered) 1.64e-2 (CZ error median 3.221e-3, 2Q error median
    # 3.22e-3, best 1.15e-3); err_readout = readout error (median) 1.97e-2.
    # This repo previously guessed Nighthawk's error rates would be LOWER than
    # Heron's ("16x more complexity" was a circuit-capacity claim, not a
    # fidelity one); the real numbers are the opposite -- both 2Q and readout
    # error are markedly worse than either Heron revision, consistent with a
    # newer, larger, still-ramping-up architecture prioritizing scale over
    # per-gate fidelity at this stage.
    "ibm_nighthawk_r1": HardwareBackend(
        name="ibm_nighthawk_r1", n_qubits=120, basis_gates=tuple(NIGHTHAWK_BASIS),
        heavy_hex_distance=0, lattice="square", grid_shape=NIGHTHAWK_GRID,
        depth_budget=20000,
        err_1q=2.293e-4, err_2q=1.64e-2, err_readout=1.97e-2,
        t1_us=308.73,
        # unconfirmed placeholder, NOT a cited figure -- Nighthawk's tunable
        # couplers are a newer architecture with no published CZ-duration
        # number available to this repo; reusing the Heron-family estimate as
        # a rough stand-in until a real number is found. Revisit before
        # trusting amplitude_damping_p() numerically for this device.
        gate_time_2q_ns=140.0,
        metadata={"family": "Nighthawk", "revision": "r1", "qpu_version": "1.0.3",
                  "source": "live IBM Quantum Platform device page (Washington DC)",
                  "physical_qubits": 338, "t1_us": 308.73, "t2_us": 216.07,
                  "cz_error_median": 3.221e-3, "2q_error_median": 3.22e-3, "2q_error_best": 1.15e-3,
                  "max_circuits_per_second": 250,
                  "note": "2q_error/readout are measurably WORSE than Heron r2/r3, not better "
                          "as this repo previously placeholder-guessed"},
        sim_kind="real",
    ),
    # ideal statevector target (no hardware constraints) -- for reference runs
    "ideal": HardwareBackend(
        name="ideal", n_qubits=10_000, basis_gates=("rz", "ry", "rx", "cx", "h"),
        heavy_hex_distance=3, depth_budget=10_000_000, sim_kind="ideal",
    ),
}

# noisy-simulator variants: identical specs, but sim_kind="noisy_sim" tells the
# conditioning descriptor and NoisyFfsimSampler's gate-noise proxy that these
# numbers are meant to be run through a simulated noise model here (not "this
# is what the real device reports" -- see HardwareBackend.sim_kind).
for _base_name in ("ibm_heron_r2", "ibm_heron_r3", "ibm_nighthawk_r1"):
    from dataclasses import replace as _replace
    _b = _BACKENDS[_base_name]
    _BACKENDS[f"{_base_name}_noisy_sim"] = _replace(_b, name=f"{_base_name}_noisy_sim", sim_kind="noisy_sim")


def get_backend(name: str) -> HardwareBackend:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Known: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


def list_backends() -> list[str]:
    return sorted(_BACKENDS)
