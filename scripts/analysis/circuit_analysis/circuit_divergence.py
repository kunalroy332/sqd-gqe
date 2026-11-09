# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Ideal (noiseless) divergence of every generated circuit from the LUCJ circuit of the same molecule.

For a generated token sequence g the policy defines |psi_g> = U_g |HF>, U_g = pool[g_L] ... pool[g_1] (FfsimLUCJPool, ffsim exact statevectors, the same
pool and angle source as training/inference). The reference is |phi> = U_LUCJ |HF> with U_LUCJ = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1, n_reps=1,
interaction_pairs=([(p,p+1)], [(p,p)])), i.e. the circuit of infer_lucj_baseline_hw.py (PrepareHartreeFockJW + UCJOpSpinBalancedJW). Everything is computed in the full
N-electron space (alpha, beta strings), so only molecules whose vector fits in memory are feasible (16, 19 and 22 orbitals here).

Metrics per circuit (p = |psi_g|^2, q = |phi|^2 over determinants):
  F       = |<psi_g|phi>|^2                           state (Uhlmann) fidelity
  BC      = sum_x sqrt(p_x q_x)                       Bhattacharyya coefficient;  H = sqrt(1 - BC)  Hellinger distance (Hellinger 1909; Bhattacharyya 1943)
  JS      = H2(m) - (H2(p)+H2(q))/2, m=(p+q)/2        Jensen-Shannon divergence in bits, bounded in [0,1], finite for disjoint support (Lin 1991)
  d_HF    = sqrt(2 - 2|<psi_g|phi>|)                  distance of the two states up to a global phase = min_alpha ||psi_g - e^{i alpha} phi||; a LOWER BOUND of the
                                                      operator distance min_alpha ||U_g - e^{i alpha} U_LUCJ||_op (the operators agree on |HF> at least that well)
  d_HF_raw= sqrt(2 - 2 Re<psi_g|phi>)                 same without the phase minimisation (ffsim's global-phase conventions are arbitrary, so use d_HF)
  exc_gen / exc_lucj : ideal excitation-rank distribution (HF,S,D,T,Q,higher) of p and q relative to the HF determinant, rank = (|a^a_HF|+|b^b_HF|)/2 (bit counts)
Operator-level quantities (small tier only; sno32):
  d_HS    = ||U_g - U_LUCJ||_F / sqrt(D) = sqrt(2 - 2 Re tr(U_g^dag U_LUCJ)/D), by Hutchinson trace estimation with random complex vectors (Hutchinson 1989):
            only forward applications, standard error ~ 1/sqrt(D) per vector. sqrt(2)=1.414 is the value for two unrelated unitaries, 0 for identical ones.
  op_norm = ||U_g - U_LUCJ||_op lower bound = sqrt of the largest Ritz value of A=(U-V)^dag(U-V) after k Lanczos steps (scipy eigsh, matrix-free; a Ritz value never exceeds
            the largest eigenvalue, so the result is a lower bound; the maximum possible value is 2). Needs exact inverses of every pool operator (validated on a toy system).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

MOL = {
    "sno32": "molecules/small/SnOH3-me_S_16o24e_mp2no_def2svp.yaml",
    "h2o_s19": "molecules/small/SnOH3-me-h2o_S_19o30e_mp2no_def2svp.yaml",
    "2h2o_s22": "molecules/small/SnOH3-me-2h2o_S_22o36e_mp2no_def2svp.yaml",
}
TREES = {"hf": "data/hf", "tuned": "data/tuned"}
MAX_VOCAB = 110


# --------------------------------------------------------------------------------------------------------------------------------- operator steps
def ops_to_steps(pool, norb):
    """Introspect each pool op (closure defaults) into a list of ('kind', args) primitive steps; used for forward and exact inverse application."""
    import ffsim
    steps = {}
    for idx, op in enumerate(pool.pool):
        f = op.apply_fn
        d = f.__defaults__ or ()
        n = op.name
        if n == "I":
            steps[idx] = []
        elif n.startswith("LUCJ"):
            steps[idx] = ucj_steps(d[0], norb)
        elif n.startswith("orb_rot"):
            steps[idx] = [("orb", d[0])]
        elif n.startswith("givens"):
            steps[idx] = [("givens", d[0], d[1])]
        elif n.startswith("diag_coulomb"):
            steps[idx] = [("diag", d[0], d[1])]
        elif n.startswith("num_num"):
            steps[idx] = [("numnum", d[0], d[1])]
        else:
            raise ValueError(f"unknown pool op {n}")
    return steps


def ucj_steps(op, norb):
    """Primitive steps of ffsim's UCJOpSpinBalanced._apply_unitary_ (read from the installed ffsim source)."""
    out = []
    cur = np.eye(norb)
    for (aa, ab), orb in zip(op.diag_coulomb_mats, op.orbital_rotations):
        out.append(("orb", orb.T.conj() @ cur))
        out.append(("diag3", (aa, ab, aa), -1.0))
        cur = orb
    if op.final_orbital_rotation is None:
        out.append(("orb", cur))
    else:
        out.append(("orb", op.final_orbital_rotation @ cur))
    return out


def apply_step(vec, st, norb, nelec, inverse=False):
    import ffsim
    k = st[0]
    if k == "orb":
        m = st[1].conj().T if inverse else st[1]
        return ffsim.apply_orbital_rotation(vec, m, norb=norb, nelec=nelec, copy=False)
    if k == "givens":
        th = -st[1] if inverse else st[1]
        return ffsim.apply_givens_rotation(vec, th, st[2], norb=norb, nelec=nelec, copy=False)
    if k == "numnum":
        th = -st[1] if inverse else st[1]
        return ffsim.apply_num_num_interaction(vec, th, st[2], norb=norb, nelec=nelec, copy=False)
    if k == "diag":
        t = -st[2] if inverse else st[2]
        return ffsim.apply_diag_coulomb_evolution(vec, st[1], t, norb=norb, nelec=nelec, copy=False)
    if k == "diag3":
        t = -st[2] if inverse else st[2]
        return ffsim.apply_diag_coulomb_evolution(vec, st[1], t, norb=norb, nelec=nelec, copy=False)
    raise ValueError(k)


def run_steps(vec, step_list, norb, nelec, inverse=False):
    seq = reversed(step_list) if inverse else step_list
    for st in seq:
        vec = apply_step(vec, st, norb, nelec, inverse)
    return vec


def circuit_steps(seq, steps, n_real):
    out = []
    for t in seq:
        t = int(t)
        out += steps[t] if 0 <= t < n_real else []
    return out


def lucj_steps_baseline(system, norb):
    import ffsim
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(system.ccsd_amplitudes["t2"], t1=system.ccsd_amplitudes["t1"], n_reps=1, interaction_pairs=(pairs_aa, pairs_ab))
    return op, ucj_steps(op, norb)


# --------------------------------------------------------------------------------------------------------------------------------- metrics
def hs_rank_classes(norb, nelec):
    """Per-alpha-string and per-beta-string popcount distance from the HF string (alpha strings and beta strings are enumerated in ffsim's order)."""
    import ffsim
    na, nb = nelec
    out = []
    for n in (na, nb):
        strs = ffsim.addresses_to_strings(np.arange(ffsim.dim(norb, n)), norb, n, bitstring_type=ffsim.BitstringType.INT)
        hf = (1 << n) - 1
        x = np.asarray(strs, dtype=np.int64) ^ hf
        pc = np.zeros(len(x), dtype=np.int64)
        y = x.copy()
        while y.any():
            pc += y & 1
            y >>= 1
        out.append(pc)
    return out


def excitation_distribution(prob2d, dist_a, dist_b):
    """prob2d: (dim_a, dim_b) determinant probabilities; rank = (da+db)/2 relative to HF (same convention as sqd_gqe.gqe.circuit_metrics)."""
    ca = np.unique(dist_a)
    cb = np.unique(dist_b)
    hist = {}
    for a in ca:
        row = prob2d[dist_a == a].sum(axis=0)
        for b in cb:
            r = int((a + b) // 2)
            hist[r] = hist.get(r, 0.0) + float(row[dist_b == b].sum())
    names = {0: "HF", 1: "S", 2: "D", 3: "T", 4: "Q"}
    out = {k: 0.0 for k in ["HF", "S", "D", "T", "Q", "higher"]}
    for r, v in hist.items():
        out[names.get(r, "higher")] += v
    return out


def entropy2(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def metrics(psi, phi, dist_a, dist_b, shape):
    ov = np.vdot(phi, psi)
    p = (psi.real ** 2 + psi.imag ** 2)
    q = (phi.real ** 2 + phi.imag ** 2)
    bc = float(np.sqrt(p * q).sum())
    m = 0.5 * (p + q)
    js = entropy2(m) - 0.5 * (entropy2(p) + entropy2(q))
    return {
        "F": float(abs(ov) ** 2), "BC": bc, "H": float(np.sqrt(max(0.0, 1 - bc))), "JS": float(js),
        "d_HF": float(np.sqrt(max(0.0, 2 - 2 * abs(ov)))), "d_HF_raw": float(np.sqrt(max(0.0, 2 - 2 * ov.real))),
        "exc_gen": excitation_distribution(p.reshape(shape), dist_a, dist_b),
    }


# --------------------------------------------------------------------------------------------------------------------------------- operator-level
def hs_distance(sa, sb, norb, nelec, dim, nvec=4, seed=0):
    """Normalised Frobenius distance ||U-V||_F/sqrt(D) by Hutchinson estimation with complex Gaussian vectors (forward applications only)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nvec):
        v = (rng.standard_normal(dim) + 1j * rng.standard_normal(dim)).astype(np.complex128)
        v /= np.linalg.norm(v)
        u = run_steps(v.copy(), sa, norb, nelec)
        w = run_steps(v.copy(), sb, norb, nelec)
        tr = np.vdot(u, w)   # = <v|U^dag V|v>, E[.] = tr(U^dag V)/D
        vals.append(float(np.sqrt(max(0.0, 2 - 2 * tr.real))))
        del u, w
    return {"d_HS": float(np.mean(vals)), "d_HS_std": float(np.std(vals)), "n_vec": nvec}


def op_norm_lanczos(sa, sb, norb, nelec, dim, k=60, seed=1, tol=1e-4):
    """Lower bound of ||U-V||_op: explicit Lanczos (full reorthogonalisation) on A=(U-V)^dag (U-V) (Hermitian PSD, spectrum in [0,4]).
    Ritz values never exceed lambda_max, so sqrt(theta_max) is a lower bound of the operator norm; the residual |beta_k s_k| of the top Ritz pair bounds its distance
    to a true eigenvalue. Stops when the residual < tol or after k matrix-vector products; returns the value after selected steps in `history`."""
    def mv(x):
        d = run_steps(x.copy(), sa, norb, nelec) - run_steps(x.copy(), sb, norb, nelec)
        return run_steps(d.copy(), sa, norb, nelec, inverse=True) - run_steps(d, sb, norb, nelec, inverse=True)

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    V = [v]
    alphas, betas, hist = [], [], []
    t = time.time()
    resid = float("nan")
    for j in range(min(k, dim)):
        w = mv(V[j])
        al = float(np.vdot(V[j], w).real)
        w = w - al * V[j] - (betas[-1] * V[j - 1] if j else 0.0)
        for u in V:                     # full reorthogonalisation
            w = w - np.vdot(u, w) * u
        be = float(np.linalg.norm(w))
        alphas.append(al)
        T = np.diag(alphas)
        if betas:
            T = T + np.diag(betas, 1) + np.diag(betas, -1)
        ev, es = np.linalg.eigh(T)
        theta = float(ev[-1]); resid = float(abs(be * es[-1, -1]))
        hist.append([j + 1, round(float(np.sqrt(max(theta, 0))), 5), resid])
        if be < 1e-12 or resid < tol:
            break
        betas.append(be)
        V.append(w / be)
    return {"op_norm_lb": hist[-1][1], "residual_top_ritz": resid, "lanczos_steps": len(hist), "converged": bool(resid < tol),
            "history": hist[:: max(1, len(hist) // 12)] + [hist[-1]], "seconds": round(time.time() - t, 1)}


# --------------------------------------------------------------------------------------------------------------------------------- toy validation
def toy_validation(norb=4, nelec=(2, 2), seed=3):
    """Check (i) the step executor equals pool.apply, (ii) inverses are exact, (iii) Lanczos == dense spectral norm, on a tiny random molecule."""
    import ffsim
    from sqd_gqe.gqe.operator_pool import FfsimLUCJPool

    class Toy:
        pass
    rng = np.random.default_rng(seed)
    toy = Toy()
    toy.norb = norb; toy.nelec = nelec
    toy.ccsd_amplitudes = {"t1": 0.05 * rng.standard_normal((nelec[0], norb - nelec[0])), "t2": 0.05 * rng.standard_normal((nelec[0],) * 2 + (norb - nelec[0],) * 2)}
    toy.ccsd_amplitudes["t2"] = toy.ccsd_amplitudes["t2"] + toy.ccsd_amplitudes["t2"].transpose(1, 0, 3, 2)
    pool = FfsimLUCJPool(toy, max_vocab=60)
    steps = ops_to_steps(pool, norb)
    dim = ffsim.dim(norb, nelec)
    hf = ffsim.hartree_fock_state(norb, nelec)
    res = {"pool_ops": pool.n_real}
    # (i) each op: executor vs pool op on a random vector; (ii) inverse
    e_fwd, e_inv = 0.0, 0.0
    for i in range(pool.n_real):
        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim); v /= np.linalg.norm(v)
        a = pool.pool[i].apply(v.copy(), norb, nelec)
        b = run_steps(v.copy(), steps[i], norb, nelec)
        c = run_steps(b.copy(), steps[i], norb, nelec, inverse=True)
        e_fwd = max(e_fwd, float(np.linalg.norm(a - b))); e_inv = max(e_inv, float(np.linalg.norm(c - v)))
    res["max_forward_mismatch"] = e_fwd; res["max_inverse_error"] = e_inv
    # dense check of the operator norm for two random sequences vs the n_reps=1 LUCJ baseline
    pa = [1, 5, 9, 14, 20, 7, 3]
    sa = circuit_steps(pa, steps, pool.n_real)
    op, sb = lucj_steps_baseline(toy, norb)
    cols_u = np.stack([run_steps(e.copy(), sa, norb, nelec) for e in np.eye(dim, dtype=complex)], axis=1)
    cols_v = np.stack([run_steps(e.copy(), sb, norb, nelec) for e in np.eye(dim, dtype=complex)], axis=1)
    # baseline executor vs ffsim.apply_unitary
    vv = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    res["lucj_executor_mismatch"] = float(np.linalg.norm(ffsim.apply_unitary(vv.copy(), op, norb=norb, nelec=nelec) - run_steps(vv.copy(), sb, norb, nelec)))
    res["dense_unitarity_error"] = float(np.linalg.norm(cols_u.conj().T @ cols_u - np.eye(dim)))
    dense = float(np.linalg.norm(cols_u - cols_v, 2))
    lz = op_norm_lanczos(sa, sb, norb, nelec, dim, k=min(dim - 1, 36), tol=1e-8)
    res["dense_op_norm"] = dense; res["lanczos_op_norm"] = lz["op_norm_lb"]
    res["dense_HS"] = float(np.linalg.norm(cols_u - cols_v) / np.sqrt(dim))
    res["hutchinson_HS"] = hs_distance(sa, sb, norb, nelec, dim, nvec=64)["d_HS"]
    return res


# --------------------------------------------------------------------------------------------------------------------------------- driver
def load_system(molkey):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    os.chdir(REPO)
    return instantiate(OmegaConf.load(REPO / MOL[molkey]))


def load_selection(tree):
    import glob
    sel = {}
    for f in sorted(glob.glob(str(REPO / tree / "selection_batch*.json"))):
        sel.update(json.loads(Path(f).read_text()))
    return sel


def run_tag(tree, tag, molkey, ctx, op_norm_circuits=(), do_hs=False, kmax=30):
    import ffsim
    system, pool, steps, phi, lucj_steps_, dist_a, dist_b, shape, dim = ctx
    norb, nelec = system.norb, tuple(system.nelec)
    data = json.loads((REPO / tree / "data" / f"{tag}.json").read_text())
    sel = load_selection(tree).get(tag, {})
    for c in data["circuits"]:
        c["is_best"] = c["index"] == sel.get("best")
        c["is_second"] = c["index"] == sel.get("second")
    res = []
    cache = {}
    for c in data["circuits"]:
        key = tuple(int(t) for t in c["seq"])
        t0 = time.time()
        if key not in cache:
            sg = circuit_steps(key, steps, pool.n_real)
            psi = run_steps(ffsim.hartree_fock_state(norb, nelec).astype(np.complex128), sg, norb, nelec)
            cache[key] = (metrics(psi, phi, dist_a, dist_b, shape), sg)
        m, sg = cache[key]
        m = dict(m)
        m.update({"index": c["index"], "is_best": bool(c.get("is_best")), "is_second": bool(c.get("is_second")), "n_real_ops": int(sum(1 for t in key if 0 <= t < pool.n_real and pool.pool[t].name != "I")), "seconds": round(time.time() - t0, 1)})
        if do_hs and (c["index"] in op_norm_circuits):
            m.update(hs_distance(sg, lucj_steps_, norb, nelec, dim, nvec=3))
            m["op_norm_info"] = op_norm_lanczos(sg, lucj_steps_, norb, nelec, dim, k=kmax)
            m["op_norm"] = m["op_norm_info"]["op_norm_lb"]
        res.append(m)
        print(f"[div] {tag} #{c['index']} F={m['F']:.2e} H={m['H']:.3f} JS={m['JS']:.3f} dHF={m['d_HF']:.3f} {m['seconds']}s", flush=True)
    out = REPO / tree / "divergence"
    out.mkdir(exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({"tag": tag, "molecule": molkey, "n_orb": norb, "nelec": list(nelec), "circuits": res}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--tree", choices=list(TREES))
    ap.add_argument("--molecule", choices=list(MOL))
    ap.add_argument("--lineages", default="a1-46q,a5-46q,a1-70q,a5-70q")
    ap.add_argument("--op-norm", action="store_true", help="also HS distance + Lanczos operator-norm lower bound for best/second circuits")
    ap.add_argument("--op-norm-limit", type=int, default=99, help="max number of circuits per run to give the operator-norm treatment")
    ap.add_argument("--kmax", type=int, default=60)
    ap.add_argument("--only-index", default="")
    a = ap.parse_args()
    if a.toy:
        print(json.dumps(toy_validation(), indent=1))
        return
    import ffsim
    from sqd_gqe.gqe.operator_pool import FfsimLUCJPool
    t0 = time.time()
    system = load_system(a.molecule)
    norb, nelec = system.norb, tuple(system.nelec)
    pool = FfsimLUCJPool(system, max_vocab=MAX_VOCAB)
    steps = ops_to_steps(pool, norb)
    op, lsteps = lucj_steps_baseline(system, norb)
    hf = ffsim.hartree_fock_state(norb, nelec).astype(np.complex128)
    phi = ffsim.apply_unitary(hf.copy(), op, norb=norb, nelec=nelec)
    dist_a, dist_b = hs_rank_classes(norb, nelec)
    shape = (len(dist_a), len(dist_b))
    dim = shape[0] * shape[1]
    q = (phi.real ** 2 + phi.imag ** 2).reshape(shape)
    root = REPO / TREES[a.tree] / "divergence"
    root.mkdir(exist_ok=True)
    (root / f"lucj_ideal_{a.molecule}.json").write_text(json.dumps({"molecule": a.molecule, "n_orb": norb, "nelec": list(nelec), "dim": dim, "n_reps": 1, "exc_lucj": excitation_distribution(q, dist_a, dist_b),
                                                                    "p_max": float(q.max()), "setup_seconds": round(time.time() - t0, 1)}, indent=1))
    ctx = (system, pool, steps, phi, lsteps, dist_a, dist_b, shape, dim)
    for lin in a.lineages.split(","):
        tag = f"{a.molecule}_{lin}_trained"
        if not (REPO / TREES[a.tree] / "data" / f"{tag}.json").exists():
            print("[div] no data for", tag); continue
        sel = load_selection(TREES[a.tree]).get(tag, {})
        idx = [i for i in (sel.get("best"), sel.get("second")) if i is not None][: a.op_norm_limit]
        if a.only_index:
            idx = [int(x) for x in a.only_index.split(",")]
        run_tag(TREES[a.tree], tag, a.molecule, ctx, op_norm_circuits=idx, do_hs=a.op_norm, kmax=a.kmax)


if __name__ == "__main__":
    main()
