# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Three-way refinement comparison: GEVP vs determinant UFD vs tensor-product UFD.

Uses identical per-circuit CI vectors (untrained-policy rollouts through the real
SqdPipeline sampler + diagonalizer) and applies each refinement map to the same
inputs, reporting the energy before compression (the union/closure solve), the
energy after compression + rediagonalization, the dimensions involved, the
tensor-expansion ratio rho_TP, and wall time. Optionally repeats for ``--iters``
GQE-style iterations to compare bounded vs cumulative global refinement.

    python benchmark/refine_compare.py --molecule molecules/small/lih.yaml --max-dim 40 --num-circuits 4
    python benchmark/refine_compare.py --molecule molecules/small/n2.yaml --max-dim 500 --num-circuits 8 --iters 5
    # GPU node with the SBD binary: add --sbd-binary-path /path/diag-gpu --launcher-cmd "mpirun -np 1"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent


def rollout(policy, cfg, num_circuits, device):
    state = {"idx": torch.zeros((num_circuits, 1), dtype=torch.long, device=device)}
    with torch.no_grad():
        for _ in range(cfg.ngates):
            nxt = policy.act(state, 1.0)
            state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
    return state


def mp2_proxy_counts(system, num_circuits: int, shots: int, rng: np.random.Generator,
                     temperature_range=(0.7, 1.3)) -> list[dict[tuple[int, int], int]]:
    """Synthetic per-circuit sample counts from an MP2-like excitation distribution.

    Stand-in for circuit sampling where ffsim's statevector is infeasible (38q+:
    40 GB vector, hours per circuit). Doubles are drawn with probability
    proportional to |t2_ijab|^2 with t2 = (ia|jb)/(e_i+e_j-e_a-e_b) built directly in
    the FCIDUMP orbital basis (diagonal Fock energies of the HF determinant), so the
    determinants live in the same basis the solvers use. Each shot applies 0, 1 or 2
    doubles (HF / doubles / quadruples). Each pseudo-circuit uses its own temperature
    p^beta, so the M sets overlap only partially, like real recovered sets do.
    """
    h = system.active_hamiltonian
    norb, (na, nb) = system.norb, system.nelec
    assert na == nb, "mp2 proxy assumes a closed-shell active space"
    h1 = np.asarray(h.one_body); eri = np.asarray(h.two_body)  # chemist (pq|rs)
    occ, vir = np.arange(na), np.arange(na, norb)
    # diagonal Fock of the HF determinant (orbitals 0..na-1 doubly occupied)
    J = np.einsum("pprr->pr", eri)[:, occ].sum(axis=1); K = np.einsum("prrp->pr", eri)[:, occ].sum(axis=1)
    eps = np.diag(h1) + 2 * J - K
    denom = eps[occ][:, None, None, None] + eps[occ][None, :, None, None] - eps[vir][None, None, :, None] - eps[vir][None, None, None, :]
    denom = np.where(np.abs(denom) < 1e-6, -1e-6, denom)
    iajb = eri[np.ix_(occ, vir, occ, vir)]                    # (ia|jb)
    t2 = iajb.transpose(0, 2, 1, 3) / denom                    # t2[i,j,a,b]
    # opposite-spin doubles (i_alpha->a_alpha, j_beta->b_beta): weight |t2|^2
    w_os = (t2 ** 2).ravel()
    # same-spin doubles (i<j -> a<b in one spin sector): weight |t2_ijab - t2_ijba|^2
    ss_idx, w_ss = [], []
    for i in range(na):
        for j in range(i + 1, na):
            for a in range(len(vir)):
                for b in range(a + 1, len(vir)):
                    ss_idx.append((i, j, a, b)); w_ss.append((t2[i, j, a, b] - t2[i, j, b, a]) ** 2)
    w_ss = np.asarray(w_ss); ss_idx = np.asarray(ss_idx, dtype=np.int64).reshape(-1, 4)
    p_all = np.concatenate([w_os, w_ss, w_ss]); p_all /= p_all.sum()   # os, ss-alpha, ss-beta
    n_os = w_os.size; n_ss = w_ss.size
    hf = (1 << na) - 1

    def draw_double(alpha: int, beta: int, k: int):
        if k < n_os:
            i, j, a, b = np.unravel_index(k, t2.shape)
            a, b = vir[a], vir[b]
            if (alpha >> i) & 1 and not (alpha >> a) & 1 and (beta >> j) & 1 and not (beta >> b) & 1:
                return alpha ^ (1 << i) ^ (1 << a), beta ^ (1 << j) ^ (1 << b)
            return None
        k -= n_os; sector = k // n_ss; i, j, a, b = ss_idx[k % n_ss]; a, b = vir[a], vir[b]
        s = alpha if sector == 0 else beta
        if (s >> i) & 1 and (s >> j) & 1 and not (s >> a) & 1 and not (s >> b) & 1:
            s2 = s ^ (1 << i) ^ (1 << j) ^ (1 << a) ^ (1 << b)
            return (s2, beta) if sector == 0 else (alpha, s2)
        return None

    out = []
    for _ in range(num_circuits):
        beta_t = rng.uniform(*temperature_range)
        p = p_all ** beta_t; p /= p.sum()
        n_exc = rng.choice(3, size=shots, p=[0.35, 0.50, 0.15])   # HF / double / quadruple
        counts: dict[tuple[int, int], int] = {}
        for n in n_exc:
            alpha, beta = hf, hf
            for k in rng.choice(p.size, size=int(n), p=p):
                r = draw_double(alpha, beta, int(k))
                if r is not None:
                    alpha, beta = r
            counts[(alpha, beta)] = counts.get((alpha, beta), 0) + 1
        out.append(counts)
    return out


def load_counts_files(paths) -> list[list[dict[tuple[int, int], int]]]:
    """Read IbmHardwareSampler-style counts.json files as one batch per file.

    ``IbmHardwareSampler`` submits one Runtime job per GQE iteration, so a counts.json
    holds exactly the M circuits of one batch. Each file therefore becomes one
    iteration here, which is what the global (cross-iteration) refinement needs.
    """
    batches = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        batch = [{tuple(int(x) for x in k.split(",")): int(v) for k, v in c["decoded_counts"].items()}
                 for c in payload["circuits"]]
        if batch:
            batches.append(batch)
    if not batches:
        raise SystemExit("no circuits found in the given --counts-file(s)")
    return batches


def per_circuit_ci(pipeline, counts_list):
    from sqd_gqe.core.configuration import ConfigurationSet

    vecs, energies = [], []
    for counts in counts_list:
        valid = ConfigurationSet.from_sample_counts(counts).select_by_nelec(pipeline.nelec)
        expanded = valid.expand(max_size=pipeline.max_size, method=pipeline.expand_method)
        e, ci = pipeline.diagonalize(expanded)
        vecs.append(ci)
        energies.append(e)
    return vecs, energies


def apply(merger, ci_vectors, max_dim, diagonalize):
    t0 = time.perf_counter()
    cs = merger.merge(ci_vectors, max_size=max_dim)
    t_merge = time.perf_counter() - t0
    t0 = time.perf_counter()
    e_post, ci = diagonalize(cs)
    t_final = time.perf_counter() - t0
    info = dict(merger.last_info)
    return {
        "pre_energy": info.get("pre_energy"), "pre_dim": info.get("pre_dim"),
        "post_energy": e_post, "retained_dets": cs.size,
        "retained_closure": int(np.unique(ci.strings[:, 0]).size * np.unique(ci.strings[:, 1]).size) if ci.size else 0,
        "rho_tp": info.get("rho_tp"), "n_alpha": info.get("n_alpha"),
        "t_merge_s": t_merge, "t_final_s": t_final,
    }, ci


def err_mha(e, ref):
    return None if (e is None or ref is None) else 1e3 * (e - ref)


def fmt(x, nd=6):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="molecules/small/n2.yaml")
    ap.add_argument("--max-dim", type=int, default=None, help="d_max (default: cfg.sqd.max_dim)")
    ap.add_argument("--num-circuits", type=int, default=None, help="M circuits per batch (default: cfg.trainer.num_samples)")
    ap.add_argument("--iters", type=int, default=1, help="iterations for bounded vs cumulative global refinement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-source", choices=["ffsim", "mp2", "counts"], default="ffsim",
                    help="ffsim: untrained-policy circuits sampled with ffsim (statevector; infeasible at 38q+). "
                         "mp2: synthetic MP2-amplitude-weighted determinant sets (no circuit simulation). "
                         "counts: real-hardware counts.json files (--counts-file), one circuit per entry")
    ap.add_argument("--counts-file", nargs="*", default=[],
                    help="counts.json files saved by IbmHardwareSampler (ibm_save_dir); one file = one batch/iteration")
    ap.add_argument("--shots", type=int, default=None, help="shots per pseudo-circuit for --sample-source mp2 (default cfg.sampler.shots)")
    ap.add_argument("--diag-backend", choices=["sqd", "sbd"], default="sqd",
                    help="per-circuit + final diagonalizer; sbd needs --sbd-binary-path and makes gevp/ufd/tp_ufd(sci) unavailable")
    ap.add_argument("--tp-n-alpha", type=int, default=None,
                    help="half-string budget for tp_ufd (default isqrt(max_dim)); set ~sqrt(closure dim of the "
                         "other methods) for a closure-dimension-matched comparison")
    ap.add_argument("--tp-n-alpha-mode", choices=["dmax", "circuit", "none"], default="dmax",
                    help="half-string budget for tp_ufd: dmax = isqrt(max_dim) (retained dim = max_dim); "
                         "circuit = the largest per-circuit alpha count, so the retained closure matches "
                         "what one circuit's own solve used (the fair comparison against the best single "
                         "circuit on a tensor-basis backend); none = keep the whole union closure")
    ap.add_argument("--tp-match-gevp-closure", action="store_true",
                    help="set the tp_ufd budget per iteration to isqrt(closure dim of the GEVP-retained set)")
    ap.add_argument("--skip", nargs="*", default=[], help="methods to skip, e.g. --skip ufd")
    ap.add_argument("--sbd-binary-path", default=None)
    ap.add_argument("--launcher-cmd", default="mpirun -np 1")
    ap.add_argument("--sbd-tolerance", type=float, default=None,
                    help="SBD Davidson tolerance (default kwargs use 0.01, which leaves ~0.1 mHa; use 1e-5 for energy comparisons)")
    ap.add_argument("--sbd-iteration", type=int, default=None, help="SBD Davidson max iterations (default kwargs: 5)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs_sqd")):
        cfg = compose(config_name="default", overrides=["sqd.recover_iterations=0"])
    cfg.molecule = OmegaConf.load(REPO / args.molecule)
    if args.max_dim is not None:
        cfg.sqd.max_dim = args.max_dim
    if args.num_circuits is not None:
        cfg.trainer.num_samples = args.num_circuits
    cfg.trainer.seed = args.seed
    torch.manual_seed(args.seed)

    from sqd_gqe.core.refine import build_merger
    from sqd_gqe.core.system import FCI_DIM_LIMIT, FciIntractableError
    from sqd_gqe.factory import SqdFactory

    if args.diag_backend == "sbd":
        if not args.sbd_binary_path:
            raise SystemExit("--diag-backend sbd requires --sbd-binary-path")
        cfg.sqd.diag_backend = "sbd"
        cfg.sqd.sbd_binary_path = args.sbd_binary_path
        cfg.sqd.sbd_launcher = args.launcher_cmd.split() if args.launcher_cmd else []
        if args.sbd_tolerance is not None:
            cfg.sqd.sbd_solver_kwargs.tolerance = args.sbd_tolerance
        if args.sbd_iteration is not None:
            cfg.sqd.sbd_solver_kwargs.iteration = args.sbd_iteration
        # gevp/ufd/tp_ufd(sci) need real amplitudes, which the sbd backend does not give
        args.skip = list(args.skip) + ["gevp", "ufd", "tp_ufd (sci"]

    f = SqdFactory()
    system = f.build_system(cfg)
    pipeline = f.build_pipeline(cfg)
    policy = f.build_policy(cfg) if args.sample_source == "ffsim" else None
    h, nelec, norb = system.active_hamiltonian, system.nelec, system.norb
    max_dim = int(cfg.sqd.max_dim)
    try:
        casci = float(system.casci_energy()); ref_label = "casci"
    except FciIntractableError:
        casci = None; ref_label = "ccsd(t)"
        try:
            casci = float(system.ccsd_t_energy())
        except Exception as e:  # noqa: BLE001
            print(f"reference CCSD(T) failed ({e}); errors reported as '-'", flush=True)
    if system.fci_dimension() > FCI_DIM_LIMIT:
        # gevp / det-UFD contract in the full FCI space -> skip automatically
        args.skip = list(args.skip) + ["gevp", "ufd"]
    print(f"molecule={args.molecule} norb={norb} nelec={nelec} fci_dim={system.fci_dimension():.3g} max_dim={max_dim} "
          f"M={cfg.trainer.num_samples} tp_n_alpha={args.tp_n_alpha or 'isqrt(max_dim)'} "
          f"sample_source={args.sample_source} diag_backend={args.diag_backend} {ref_label}={fmt(casci)}", flush=True)
    rng = np.random.default_rng(args.seed)
    counts_batches = None
    if args.sample_source == "counts":
        if not args.counts_file:
            raise SystemExit("--sample-source counts requires --counts-file")
        counts_batches = load_counts_files(args.counts_file)
        print(f"[counts] {len(counts_batches)} batch(es) (one per file), "
              f"{[len(b) for b in counts_batches]} circuits each", flush=True)
        if args.iters > len(counts_batches):
            print(f"[counts] --iters {args.iters} > {len(counts_batches)} batches: batches will be cycled", flush=True)

    mergers = {
        "gevp": build_merger("gevp", h, nelec, norb),
        "ufd (fci dense)": build_merger("ufd", h, nelec, norb, dense_limit=50_000),
        "tp_ufd (sci marginal)": build_merger("tp_ufd", h, nelec, norb, spin_sq=cfg.sqd.spin_sq, n_alpha=args.tp_n_alpha),
    }
    if args.sbd_binary_path:
        from sqd_gqe.sqd.backend_sbd_gpu import SbdGpuDiagonalizer

        sbd_kwargs = {}
        if args.sbd_tolerance is not None:
            sbd_kwargs["tolerance"] = args.sbd_tolerance
        if args.sbd_iteration is not None:
            sbd_kwargs["iteration"] = args.sbd_iteration
        sbd = SbdGpuDiagonalizer(h, nelec, norb, binary_path=args.sbd_binary_path,
                                 launcher=args.launcher_cmd.split() if args.launcher_cmd else [],
                                 solver_kwargs=sbd_kwargs or None)
        mergers["tp_ufd (sbd carryover)"] = build_merger("tp_ufd", h, nelec, norb, sbd=sbd, diag_backend="sbd",
                                                         n_alpha=args.tp_n_alpha)

    mergers = {k: v for k, v in mergers.items() if not any(k.startswith(sk) for sk in args.skip)}
    results = {"molecule": args.molecule, "max_dim": max_dim, "reference": casci, "reference_label": ref_label,
               "sample_source": args.sample_source, "diag_backend": args.diag_backend, "local": [], "global": {}}
    global_state = {name: {"bounded": None, "cumulative": None} for name in mergers}

    for it in range(args.iters):
        if args.sample_source == "counts":
            counts_list = counts_batches[it % len(counts_batches)]
            src = Path(args.counts_file[it % len(counts_batches)]).parent.name
            print(f"[counts] batch {it % len(counts_batches) + 1}/{len(counts_batches)} ({src}): {len(counts_list)} circuits, "
                  f"distinct dets per circuit: {[len(c) for c in counts_list]}", flush=True)
        elif args.sample_source == "mp2":
            t0 = time.perf_counter()
            counts_list = mp2_proxy_counts(system, int(cfg.trainer.num_samples), int(args.shots or cfg.sampler.shots), rng)
            print(f"[mp2 proxy] {len(counts_list)} sets, distinct dets per set: {[len(c) for c in counts_list]} ({time.perf_counter() - t0:.1f}s)", flush=True)
        else:
            state = rollout(policy, cfg, int(cfg.trainer.num_samples), torch.device("cpu"))
            counts_list = pipeline.sampler.run(state)
        t0 = time.perf_counter()
        ci_vectors, energies = per_circuit_ci(pipeline, counts_list)
        circuit_alpha = max((int(np.unique(v.strings[:, 0]).size) for v in ci_vectors if v.size), default=0)
        print(f"[per-circuit diag] {len(ci_vectors)} circuits in {time.perf_counter() - t0:.1f}s "
              f"(max per-circuit alpha strings {circuit_alpha}, closure {circuit_alpha ** 2})", flush=True)
        best_circuit = min(energies)
        print(f"\n== iteration {it + 1}/{args.iters}: best single-circuit energy {best_circuit:.6f}", flush=True)
        print(f"{'method':24s} {'E_pre':>12s} {'err_pre':>8s} {'pre_dim':>8s} {'E_post':>12s} {'err_post':>8s} "
              f"{'ret_dets':>8s} {'closure':>8s} {'rho_TP':>7s} {'t_merge':>8s} {'t_final':>8s}")
        gevp_closure = None
        for name, m in mergers.items():
            if name.startswith("tp_ufd"):
                if args.tp_match_gevp_closure and gevp_closure:
                    m.n_alpha = max(int(np.sqrt(gevp_closure)), 1)
                elif args.tp_n_alpha is None and args.tp_n_alpha_mode == "circuit":
                    m.n_alpha = max(circuit_alpha, 1)
                elif args.tp_n_alpha is None and args.tp_n_alpha_mode == "none":
                    m.n_alpha = None
            try:
                row, local_ci = apply(m, ci_vectors, max_dim, pipeline.diagonalize)
            except Exception as e:  # noqa: BLE001 -- report and keep going (e.g. dense_limit, missing binary)
                print(f"{name:24s} skipped: {e}")
                continue
            if name == "gevp":
                gevp_closure = row["retained_closure"]
            row.update(method=name, iteration=it + 1)
            results["local"].append(row)
            row["err_pre_mha"] = err_mha(row["pre_energy"], casci)
            row["err_post_mha"] = err_mha(row["post_energy"], casci)
            print(f"{name:24s} {fmt(row['pre_energy']):>12s} {fmt(row['err_pre_mha'], 2):>8s} {fmt(row['pre_dim']):>8s} "
                  f"{fmt(row['post_energy']):>12s} {fmt(row['err_post_mha'], 2):>8s} "
                  f"{row['retained_dets']:>8d} {row['retained_closure']:>8d} {fmt(row['rho_tp'], 2):>7s} "
                  f"{row['t_merge_s']:>8.2f} {row['t_final_s']:>8.2f}")
            # global refinement, both modes, using the same merger
            for mode in ("bounded", "cumulative"):
                prev = global_state[name][mode]
                if prev is None:
                    global_state[name][mode] = local_ci
                    continue
                inputs = [prev, *ci_vectors] if mode == "cumulative" else [prev, local_ci]
                cap = 5_000_000 if mode == "cumulative" else max_dim
                # cumulative: let the cap (not a fixed half-string budget) bound tp_ufd,
                # mirroring cfg tp_ufd_n_alpha=null; local/bounded keep the matched budget.
                saved_n_alpha = getattr(m, "n_alpha", None)
                if mode == "cumulative" and hasattr(m, "n_alpha"):
                    m.n_alpha = None
                try:
                    grow, gci = apply(m, inputs, cap, pipeline.diagonalize)
                except Exception as e:  # noqa: BLE001
                    print(f"  global[{mode}] skipped: {e}")
                    continue
                finally:
                    if hasattr(m, "n_alpha"):
                        m.n_alpha = saved_n_alpha
                global_state[name][mode] = gci
                grow.update(method=name, mode=mode, iteration=it + 1)
                grow["err_post_mha"] = err_mha(grow["post_energy"], casci)
                results["global"].setdefault(name, []).append(grow)
                print(f"  global[{mode:10s}] E={grow['post_energy']:.6f} err={fmt(grow['err_post_mha'], 2)} mHa "
                      f"dets={grow['retained_dets']} closure={grow['retained_closure']} "
                      f"t_merge={grow['t_merge_s']:.2f}s t_final={grow['t_final_s']:.2f}s")
        pre = {r["method"]: r["pre_energy"] for r in results["local"] if r["iteration"] == it + 1 and r["pre_energy"] is not None}
        if {"gevp", "ufd (fci dense)", "tp_ufd (sci marginal)"} <= pre.keys():
            # tolerance is the solvers' own convergence noise (solve_sci Davidson vs dense eigh
            # agree to ~1e-9 Ha), still four orders below chemical accuracy
            tol = 1e-7
            d1 = pre["tp_ufd (sci marginal)"] - pre["ufd (fci dense)"]
            d2 = pre["ufd (fci dense)"] - pre["gevp"]
            ok = d1 <= tol and d2 <= tol
            print(f"ordering E_TP-UFD <= E_det-UFD <= E_GEVP (pre-compression): {'OK' if ok else 'VIOLATED'} "
                  f"(TP-det {1e3 * d1:+.2e} mHa, det-GEVP {1e3 * d2:+.2e} mHa, tol {1e3 * tol:.0e} mHa)")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
