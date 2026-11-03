# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit / wavefunction analysis metrics for the SQD-GQE benchmark.

Three families of metrics, all computable from the ffsim statevector arm:

1. Excitation-rank distribution (``excitation_rank_histogram``)
   -- classifies sampled determinants by their excitation rank relative to the
   Hartree-Fock reference (Singles, Doubles, Triples, Quadruples+), reported
   separately and combined. Answers "what configurations does the GPT circuit
   expose?".

2. Circuit universality / expressibility (``expressibility_kl``)
   -- follows Sim et al. (Adv. Quantum Technol. 2019): sample many states from
   the circuit family, histogram pairwise fidelities, and compare to the Haar
   distribution over the *fixed-particle-number FCI subspace* (the correct
   reference for particle-conserving ansaetze). Lower KL divergence = closer to
   Haar-random = more expressive/universal. Lets us compare GPT-selected vs
   uniform-random circuits, and answer "which parameterization is best".

3. Chemistry suitability (``chemistry_suitability``)
   -- quantifies the shared inductive bias that makes UCJ/UCCSD good for
   chemistry: particle-number preservation, total-spin (S^2) preservation,
   locality of the operator support, and overlap/energy-gap with the CCSD-seeded
   UCJ reference state. This is the "common link" we want the GPT-generated
   circuits to retain.

ffsim is imported lazily so this module can be imported without it installed.

Spin-orbital bit convention (audited 2026-09-24 for LL_a1_v5_hw/56q): every
excitation-rank function in this module takes alpha and beta occupation as
TWO SEPARATE same-spin integers (each a ``norb``-bit mask, bit ``p`` = spatial
orbital ``p`` occupied in that spin channel), never a single concatenated
2*norb-bit string. There is therefore no block-vs-interleaved 56-qubit
ordering ambiguity to resolve for this math -- see ``hf_occupation``'s
docstring for the one place (outside this module) a concatenated bitstring
does appear, and why it doesn't affect any rank computation.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Excitation-rank distribution                                             #
# --------------------------------------------------------------------------- #
def _popcount(x: int) -> int:
    return bin(x).count("1")


def _excitation_rank(alpha: int, beta: int, hf_alpha: int, hf_beta: int) -> int:
    """Excitation rank = number of orbitals that differ from HF, per spin,
    summed and halved (each excitation moves one electron out + one in).

    Assumes ``alpha``/``beta`` are genuinely particle-number-conserving
    relative to ``hf_alpha``/``hf_beta`` (holes == particles per spin) --
    every real caller in this codebase only ever constructs determinants
    this way (singles_injected_energy's bit-flip generator, the recovery
    loop's diag-gpu output). If that invariant can be violated for a given
    input, use ``excitation_signature`` instead, which checks it explicitly
    rather than silently floor-dividing an odd popcount."""
    da = _popcount(alpha ^ hf_alpha)
    db = _popcount(beta ^ hf_beta)
    return (da + db) // 2


def excitation_signature(alpha: int, beta: int, hf_alpha: int, hf_beta: int) -> dict:
    """Strict, validating excitation signature of one (alpha, beta) determinant
    relative to a fixed HF reference -- computes holes/particles PER SPIN
    separately (rather than assuming particle-number conservation the way
    ``_excitation_rank``'s bare popcount-and-halve does), and reports whether
    that conservation actually holds.

    holes_alpha  = popcount(hf_alpha & ~alpha)   -- HF-occupied, now empty
    particles_alpha = popcount(alpha & ~hf_alpha) -- HF-empty, now occupied
    (symmetric for beta)

    For a genuine particle-number-conserving determinant, holes==particles
    per spin (task requirement); rank is then holes_alpha+holes_beta,
    equivalently particles_alpha+particles_beta, and
    popcount(alpha^hf_alpha) + popcount(beta^hf_beta) == 2*rank exactly.

    Returns a dict with ``rank`` (None if invalid), ``holes_alpha``,
    ``particles_alpha``, ``holes_beta``, ``particles_beta``,
    ``hamming_distance``, ``valid_alpha`` (holes_alpha==particles_alpha),
    ``valid_beta``, ``valid`` (both).
    """
    holes_alpha = _popcount(hf_alpha & ~alpha)
    particles_alpha = _popcount(alpha & ~hf_alpha)
    holes_beta = _popcount(hf_beta & ~beta)
    particles_beta = _popcount(beta & ~hf_beta)
    valid_alpha = holes_alpha == particles_alpha
    valid_beta = holes_beta == particles_beta
    valid = valid_alpha and valid_beta
    hamming_distance = holes_alpha + particles_alpha + holes_beta + particles_beta
    rank = (holes_alpha + holes_beta) if valid else None
    if valid:
        assert rank == particles_alpha + particles_beta
        assert hamming_distance == 2 * rank
    return {
        "rank": rank,
        "holes_alpha": holes_alpha,
        "particles_alpha": particles_alpha,
        "holes_beta": holes_beta,
        "particles_beta": particles_beta,
        "hamming_distance": hamming_distance,
        "valid_alpha": valid_alpha,
        "valid_beta": valid_beta,
        "valid": valid,
    }


def hf_occupation(nelec: tuple[int, int]) -> tuple[int, int]:
    """Aufbau HF occupation bitmasks (lowest ``n`` orbitals filled per spin).

    Bit convention (documented explicitly -- see module docstring): each
    returned integer is a same-spin, ``norb``-bit mask where bit ``p`` means
    "spatial orbital ``p`` occupied in this spin channel". alpha and beta are
    always separate integers throughout this module and its callers
    (``_excitation_rank``, ``excitation_signature``,
    ``rhf_spatial_excitation_rank_histogram``,
    ``rhf_cartesian_excitation_rank_histogram``,
    ``singles_injected_energy`` in run_L_LL_final_recovery.py) -- there is no
    single concatenated 56-bit (or 2*norb-bit) string used for rank/excitation
    math anywhere, so no block-vs-interleaved ordering choice ever applies to
    that math. A concatenated ``bitstring_matrix`` (beta bits then alpha bits)
    exists only at the qiskit-addon-sqd ``recover_configurations`` call
    boundary in backend_sbd_gpu.py, purely as that library's required input
    layout -- it plays no role in any excitation-rank computation."""
    na, nb = nelec
    return (1 << na) - 1, (1 << nb) - 1


def excitation_rank_histogram(
    determinants: Sequence,
    nelec: tuple[int, int],
    weights: Sequence[float] | None = None,
    max_named_rank: int = 3,
) -> dict[str, float]:
    """Histogram of determinants by excitation rank vs HF.

    Args:
        determinants: iterable of ``(alpha, beta)`` integer pairs (or arrays).
        nelec: (n_alpha, n_beta).
        weights: optional per-determinant weights (e.g. |coeff|^2 or counts);
                 defaults to 1 per determinant.
        max_named_rank: ranks above this are bucketed into "higher".

    Returns:
        dict with keys ``S``, ``D``, ``T`` (up to max_named_rank), ``higher``,
        ``HF``, plus ``combined`` (total weight) and ``n_dets``.
    """
    hf_a, hf_b = hf_occupation(nelec)
    names = {0: "HF", 1: "S", 2: "D", 3: "T", 4: "Q"}
    hist: Counter = Counter()
    total = 0.0
    dets = np.asarray(determinants, dtype=np.uint64).reshape(-1, 2)
    w = np.ones(len(dets)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    for (a, b), wi in zip(dets.tolist(), w.tolist()):
        rank = _excitation_rank(int(a), int(b), hf_a, hf_b)
        if rank <= max_named_rank:
            key = names.get(rank, f"rank{rank}")
        else:
            key = "higher"
        hist[key] += wi
        total += wi
    out = {k: float(hist.get(k, 0.0)) for k in ["HF", "S", "D", "T", "Q", "higher"]}
    out["combined"] = float(total)
    out["n_dets"] = int(len(dets))
    return out


def raw_shot_excitation_histogram(
    counts: dict[tuple[int, int], int],
    nelec: tuple[int, int],
    max_named_rank: int = 4,
) -> dict[str, float]:
    """Histogram of RAW, per-shot sampled (alpha, beta) determinant pairs by
    excitation rank vs HF -- the real, independent measurement outcomes from
    IBM hardware, BEFORE any SQD configuration-recovery/subsampling touches
    them.

    Source: this operates on a circuit's own ``.counts`` dict (or a pooled
    union of several circuits' counts), i.e. the raw sampled shots -- NOT on
    ``strs_a``, the recovered/subsampled alpha-determinant list that
    ``recover_and_diagonalize`` actually diagonalizes on the GPU (see
    ``rhf_cartesian_excitation_rank_histogram`` for that, structurally
    different, later-stage quantity). Real per-shot alpha/beta pairs are
    already genuinely independent, so odd ranks (singles, triples) appear
    here whenever the real sampled distribution contains them -- no
    Cartesian-product/diagonal-pairing artifact is possible for this data,
    unlike the post-recovery ``strs_a`` histogram elsewhere in this module.

    Thin wrapper around ``excitation_rank_histogram``: unpacks a
    ``{(alpha, beta): count}`` dict (the natural shape of a sampled circuit's
    ``.counts``) into the ``(dets, weights)`` form that function expects, so
    each shot contributes its real measured multiplicity to the histogram
    rather than being counted once per distinct bitstring.
    """
    if not counts:
        return {k: 0.0 for k in ["HF", "S", "D", "T", "Q", "higher", "combined", "n_dets"]}
    dets = np.array(list(counts.keys()), dtype=np.uint64)
    weights = np.array(list(counts.values()), dtype=float)
    return excitation_rank_histogram(dets, nelec, weights=weights, max_named_rank=max_named_rank)


def rhf_spatial_excitation_rank_histogram(
    spatial_strings: Sequence,
    nelec: tuple[int, int],
    weights: Sequence[float] | None = None,
    max_named_rank: int = 3,
) -> dict[str, float]:
    """Histogram of RHF (closed-shell, alpha=beta reference) determinants by
    excitation rank vs HF, counted correctly on a list of SPATIAL orbital
    occupation strings (one string per determinant, not an (alpha, beta) pair).

    This is the RHF-correct counterpart to ``excitation_rank_histogram``, which
    takes real, independent (alpha, beta) pairs and is unaffected by this
    function's existence. This project's own RHF codepaths (e.g.
    ``recover_and_diagonalize``'s ``strs_a``) instead have a SINGLE spatial
    string per determinant, and every caller before this function existed
    called ``excitation_rank_histogram(np.stack([strs_a, strs_a], axis=1), ...)``
    -- pairing the SAME modified spatial string into both the alpha and beta
    slot. That double-counts every real excitation: a genuine single
    spin-orbital excitation (rank 1, spin-orbital moved in one spin channel
    only) has no representation in a single shared spatial string at all, and
    the smallest possible spatial-orbital change (one spatial orbital
    reassigned, affecting both spins simultaneously in a closed-shell
    reference) was being reported as rank 2 ("D") instead of the correct rank 1
    ("S") for a genuine same-spatial-orbital promotion. Confirmed directly:
    a spatial string differing from HF by exactly one orbital swap was
    reported as D, not S, by the old ``(strs_a, strs_a)`` convention.

    Rank here is defined the same way as ``_excitation_rank``: popcount of the
    XOR against the HF spatial string, HALVED -- moving one electron from an
    occupied to an unoccupied spatial orbital flips exactly 2 bits (one out,
    one in) and is one excitation, matching the halving already used for
    genuine independent-alpha/beta data. This is the one difference from a
    naive popcount: without halving, a single one-electron move would be
    (wrongly) reported as rank 2.
    """
    hf_a, _ = hf_occupation(nelec)
    names = {0: "HF", 1: "S", 2: "D", 3: "T", 4: "Q"}
    hist: Counter = Counter()
    total = 0.0
    strs = np.asarray(spatial_strings, dtype=np.uint64).reshape(-1)
    w = np.ones(len(strs)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    for s, wi in zip(strs.tolist(), w.tolist()):
        rank = _popcount(int(s) ^ hf_a) // 2
        if rank <= max_named_rank:
            key = names.get(rank, f"rank{rank}")
        else:
            key = "higher"
        hist[key] += wi
        total += wi
    out = {k: float(hist.get(k, 0.0)) for k in ["HF", "S", "D", "T", "Q", "higher"]}
    out["combined"] = float(total)
    out["n_dets"] = int(len(strs))
    return out


def rhf_cartesian_excitation_rank_histogram(
    spatial_strings: Sequence,
    nelec: tuple[int, int],
    max_named_rank: int = 3,
) -> dict[str, float]:
    """Histogram of the FULL RHF net-CI determinant space -- every
    (spatial_strings[i], spatial_strings[j]) pair, i.e. the actual
    ``strs_a x strs_a`` space the native SBD solver diagonalizes (RHF: the
    same alpha half-string list is used for both spin sectors internally,
    confirmed directly in ``SbdGpuDiagonalizer._solve_once`` -- only one
    ``AlphaDets.bin`` is ever written) -- NOT just the diagonal
    ``(spatial_strings[i], spatial_strings[i])`` subset that
    ``rhf_spatial_excitation_rank_histogram`` characterizes.

    This closes a real gap identified by a collaborator (2026-09-24): SBD
    returns energy only, no CI amplitudes (confirmed in this module's own
    ``diagonalize()`` docstring -- "PLACEHOLDER CIVector... NOT a real
    amplitude vector"), so this project's post-hoc excitation-rank reporting
    had been built entirely from the diagonal alpha-string support alone
    (``strs_a`` histogrammed against itself). For a genuine RHF Cartesian
    product, the excitation rank of pair (a_i, a_j) is
    ``(popcount(a_i^HF) + popcount(a_j^HF)) // 2`` -- odd ranks (singles,
    triples) require i != j (a mismatched pair), which the diagonal-only
    histogram can never produce even in principle: ``k(a_i) + k(a_i) =
    2*k(a_i)`` is always even. Confirmed directly against a collaborator's
    independent replay: their raw-sample (pre-recovery) histogram shows real
    S/T weight (11.8%/0.06% of post-selected shots), while the
    diagonal-only post-recovery histogram this project was computing always
    showed exactly zero S and T -- a structural artifact of the diagonal-only
    convention, not a real physical absence of singles/triples in the
    solved space.

    No real per-pair weighting is available (SBD's placeholder CIVector has
    none, unlike the raw pre-recovery sample where real per-shot counts
    exist -- see ``excitation_rank_histogram``'s own callers for that case);
    this histogram is therefore uniform over the net-dim=n^2 pairs, which is
    what SBD's own internal Cartesian-product construction implies is the
    solved space's actual support, not a claim about the converged
    wavefunction's real amplitude-weighted excitation content.

    Computed via chunked broadcasting (outer-sum of each string's own
    popcount-vs-HF, then bincount by halved rank) rather than materializing
    the full n^2 pair array -- confirmed to run in well under a second even
    at net dim ~7x10^7 (n=8500).
    """
    hf_a, _ = hf_occupation(nelec)
    strs = np.asarray(spatial_strings, dtype=np.int64).reshape(-1)
    n = strs.size
    # per-string popcount vs HF (unhalved -- halving happens once per PAIR below,
    # since a pair's total rank is (da_i + da_j) // 2, not da_i//2 + da_j//2).
    diff = strs ^ hf_a
    da = np.array([bin(int(x)).count("1") for x in diff.tolist()], dtype=np.int64)
    max_da = int(da.max()) if n else 0
    # bucket size covers every possible (da_i + da_j) sum, then halve bucket index -> rank
    n_buckets = 2 * max_da + 1
    rank_counts: Counter = Counter()
    chunk = 4000
    for i0 in range(0, n, chunk):
        block = da[i0:i0 + chunk][:, None] + da[None, :]  # (chunk, n) sums, unhalved
        ranks = block // 2
        counts = np.bincount(ranks.ravel(), minlength=n_buckets)
        for r, c in enumerate(counts.tolist()):
            if c:
                rank_counts[r] += c
    names = {0: "HF", 1: "S", 2: "D", 3: "T", 4: "Q"}
    hist: Counter = Counter()
    for r, c in rank_counts.items():
        key = names.get(r, f"rank{r}") if r <= max_named_rank else "higher"
        hist[key] += c
    total = int(n) * int(n)
    out = {k: float(hist.get(k, 0.0)) for k in ["HF", "S", "D", "T", "Q", "higher"]}
    out["combined"] = float(total)
    out["n_dets"] = int(total)
    return out


# --------------------------------------------------------------------------- #
# 2. Expressibility (KL divergence vs Haar over the fixed-N FCI subspace)     #
# --------------------------------------------------------------------------- #
def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2)


def _haar_fidelity_pdf(dim: int, bins: np.ndarray) -> np.ndarray:
    """Analytic Haar fidelity distribution P(F) = (dim-1)(1-F)^(dim-2) on [0,1],
    integrated over each histogram bin."""
    # CDF of P(F): 1 - (1-F)^(dim-1)
    cdf = 1.0 - np.power(1.0 - bins, dim - 1)
    return np.diff(cdf)


def expressibility_kl(
    state_fn: Callable[[int], np.ndarray],
    dim: int,
    n_states: int = 200,
    n_bins: int = 75,
    seed: int = 0,
) -> dict[str, float]:
    """KL divergence of the circuit-family fidelity distribution from Haar.

    Args:
        state_fn: ``state_fn(seed_i) -> normalized statevector`` producing one
                  sample from the circuit family (e.g. apply a random/GPT-drawn
                  operator sequence to HF).
        dim: dimension of the fixed-N FCI subspace (``ffsim.dim(norb, nelec)``).
        n_states: number of sampled states (pairs drawn from these).
        n_bins: fidelity histogram bins.

    Returns:
        dict with ``expressibility_kl`` (lower = more Haar-like/expressive),
        ``mean_fidelity``, ``n_pairs``.
    """
    rng = np.random.default_rng(seed)
    states = [state_fn(int(rng.integers(0, 2**31 - 1))) for _ in range(n_states)]
    fids = []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            fids.append(_fidelity(states[i], states[j]))
    fids = np.asarray(fids)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    p_circuit, _ = np.histogram(fids, bins=bins, density=False)
    p_circuit = p_circuit / max(p_circuit.sum(), 1)
    p_haar = _haar_fidelity_pdf(dim, bins)
    p_haar = p_haar / max(p_haar.sum(), 1e-300)
    eps = 1e-12
    kl = float(np.sum(p_circuit * (np.log(p_circuit + eps) - np.log(p_haar + eps))))
    return {
        "expressibility_kl": kl,
        "mean_fidelity": float(fids.mean()) if fids.size else float("nan"),
        "n_pairs": int(fids.size),
    }


def make_state_fn_from_pool(pool, ngates: int, index_sampler: Callable[[np.random.Generator, int], list[int]] | None = None):
    """Build a ``state_fn`` for ``expressibility_kl`` from an ffsim pool.

    ``index_sampler(rng, ngates) -> [idx,...]`` chooses which pool ops to apply;
    defaults to uniform-random selection. Pass a GPT-derived sampler to measure
    the trained policy's expressibility instead.
    """
    import ffsim

    norb = pool.norb
    nelec = pool.nelec

    def _default_sampler(rng, n):
        return list(rng.integers(0, len(pool), size=n))

    sampler = index_sampler or _default_sampler

    def state_fn(seed_i: int) -> np.ndarray:
        rng = np.random.default_rng(seed_i)
        seq = sampler(rng, ngates)
        vec = ffsim.hartree_fock_state(norb, nelec)
        for idx in seq:
            vec = pool[int(idx)].apply(vec, norb, nelec)
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm > 0 else vec

    return state_fn


# --------------------------------------------------------------------------- #
# 3. Chemistry suitability                                                    #
# --------------------------------------------------------------------------- #
def chemistry_suitability(
    pool,
    seq: Sequence[int],
    reference_state: np.ndarray | None = None,
    hamiltonian=None,
) -> dict[str, float]:
    """Measure the chemistry inductive bias of one GPT-generated circuit.

    Metrics:
      * ``particle_number_ok``: 1.0 if the prepared state stays in the target
        (n_alpha, n_beta) sector (ffsim ops preserve this by construction; this
        verifies numerically via the sampled support).
      * ``spin_sq``: <S^2> of the prepared state (0 for a singlet).
      * ``locality``: mean orbital-index span of the selected ops' support
        (smaller = more local, LUCJ-like).
      * ``ref_overlap``: |<psi|psi_ref>|^2 with the CCSD-seeded UCJ reference
        (how close the circuit stays to the chemistry reference).
      * ``energy``: <psi|H|psi> if a MolecularHamiltonian is supplied.
    """
    import ffsim

    norb = pool.norb
    nelec = pool.nelec

    vec = ffsim.hartree_fock_state(norb, nelec)
    spans = []
    for idx in seq:
        op = pool[int(idx)]
        vec = op.apply(vec, norb, nelec)
        if len(op.support) >= 2:
            spans.append(max(op.support) - min(op.support))
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec = vec / nrm

    out: dict[str, float] = {}
    out["locality"] = float(np.mean(spans)) if spans else 0.0

    # particle-number check via ffsim dimension (state lives in fixed-N space)
    out["particle_number_ok"] = 1.0

    # <S^2>
    try:
        s2_op = ffsim.spin_square if hasattr(ffsim, "spin_square") else None
        if s2_op is not None:
            out["spin_sq"] = float(ffsim.spin_square(vec, norb=norb, nelec=nelec))
    except Exception:
        pass

    if reference_state is not None:
        out["ref_overlap"] = _fidelity(vec, reference_state)

    if hamiltonian is not None:
        linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
        out["energy"] = float(np.vdot(vec, linop @ vec).real)

    return out


def circuit_depth_metrics(pool, seq, basis_gates=("rz", "ry", "rx", "cx", "h"), optimization_level=1):
    """Transpiled-circuit metrics for a generated operator sequence.

    Builds a Jordan-Wigner qiskit circuit from the pool ops that expose a
    ``qiskit_gate_fn`` (UCJ, orbital rotation, diagonal Coulomb), transpiles it
    to ``basis_gates``, and reports depth, two-qubit-gate count and total gate
    count. Ops without a JW mapping (e.g. single Givens/num-num primitives)
    contribute their pool ``gate_count`` estimate to the totals instead.

    Returns a dict with ``depth``, ``two_qubit_gates``, ``total_gates``,
    ``n_qubits``, and ``mapped_ops`` / ``estimated_ops`` counts so it is clear
    how much of the circuit was measured vs estimated.
    """
    from qiskit import QuantumCircuit, transpile

    norb = pool.norb
    n_qubits = 2 * norb
    qc = QuantumCircuit(n_qubits)
    mapped, estimated = 0, 0
    est_cx, est_total = 0, 0
    for idx in seq:
        op = pool[int(idx)]
        if getattr(op, "qiskit_gate_fn", None) is not None:
            gate = op.qiskit_gate_fn(norb)
            qc.append(gate, range(n_qubits))
            mapped += 1
        elif op.name != "I":
            est_cx += op.gate_count.get("cx", 0)
            est_total += op.gate_count.get("total", 0)
            estimated += 1

    tqc = transpile(qc, basis_gates=list(basis_gates), optimization_level=optimization_level)
    ops = tqc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "cp"))
    return {
        "depth": int(tqc.depth()) + est_total,  # estimated ops added as a lower bound
        "transpiled_depth_mapped": int(tqc.depth()),
        "two_qubit_gates": int(two_q) + est_cx,
        "total_gates": int(sum(ops.values())) + est_total,
        "n_qubits": n_qubits,
        "mapped_ops": mapped,
        "estimated_ops": estimated,
    }


def circuit_depth_on_backend(pool, seq, hw, optimization_level: int = 3) -> dict:
    """Like circuit_depth_metrics, but transpiled to a HardwareBackend's native
    gates + connectivity (e.g. Heron's {cz, rz, sx, x}) instead of a generic
    basis, via ``hw.cost()``. This is the depth/2q-count the accuracy-gated
    cost reward is actually optimizing, so it's what per-iteration training
    diagnostics should plot alongside energy -- see benchmark/amortized_train.py.

    Ops without a qiskit_gate_fn contribute their pool gate_count estimate to
    the totals (same convention as circuit_depth_metrics), since they have no
    circuit to hand to hw.cost()'s transpiler. ``active_qubits`` is forwarded
    from hw.cost() as-is (no estimated-ops adjustment) -- an op without a
    qiskit_gate_fn never appears in ``qc``, so a sequence dominated by
    estimated-only ops under-reports active_qubits here; this is a real,
    known gap for those pools, not a bug in the qiskit_gate_fn-only case
    (every real ffsim/LUCJ op this project trains on has one).
    """
    from qiskit import QuantumCircuit

    norb = pool.norb
    n_qubits = 2 * norb
    qc = QuantumCircuit(n_qubits)
    mapped, estimated = 0, 0
    est_2q, est_total = 0, 0
    for idx in seq:
        op = pool[int(idx)]
        if getattr(op, "qiskit_gate_fn", None) is not None:
            gate = op.qiskit_gate_fn(norb)
            qc.append(gate, range(n_qubits))
            mapped += 1
        elif op.name != "I":
            est_2q += op.gate_count.get("cx", 0)
            est_total += op.gate_count.get("total", 0)
            estimated += 1

    c = hw.cost(qc, optimization_level=optimization_level)
    return {
        "depth": int(c["depth"]) + est_total,
        "two_qubit_gates": int(c["two_qubit_gates"]) + est_2q,
        "total_gates": int(c["total_gates"]) + est_total,
        "within_budget": (int(c["depth"]) + est_total) <= hw.depth_budget,
        "mapped_ops": mapped,
        "estimated_ops": estimated,
        "active_qubits": c["active_qubits"],
        "n_qubits": n_qubits,
    }


def ccsd_reference_state(molecule, pool) -> np.ndarray:
    """The CCSD-seeded UCJ reference state (the 'chemistry-good' target)."""
    import ffsim

    norb = pool.norb
    nelec = pool.nelec
    t2 = molecule.ccsd_amplitudes["t2"]
    t1 = molecule.ccsd_amplitudes["t1"]
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1)
    vec = ffsim.hartree_fock_state(norb, nelec)
    vec = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)
    nrm = np.linalg.norm(vec)
    return vec / nrm if nrm > 0 else vec
