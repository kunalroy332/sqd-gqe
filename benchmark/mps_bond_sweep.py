# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sweep MPS bond dimension on fixed circuits through the production SQD pipeline.

Reports energies, timings, valid fractions, ranking stability and determinant overlap.
Use --with-exact for an ffsim reference (where its CI vector fits in memory),
--seeds N to measure shot noise, and --sampler-kind mps_topk for amplitude selection.
Writes docs/report/mps_bond_sweep/<molecule>.json after each measurement.

    python benchmark/mps_bond_sweep.py --molecule n2 --bonds 64 128 256 512 \\
        --sampler-kind mps_topk --with-exact --diag-backend sqd
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "docs" / "report" / "mps_bond_sweep"


def build_sequences(pool, seed: int = 0, n_random: int = 0) -> dict[str, list[int]]:
    """Local/LUCJ cost probes plus random 10-token draws for ranking comparisons."""
    import numpy as np

    names = [pool[i].name for i in range(len(pool))]
    lucj = [i for i, n in enumerate(names) if n.startswith("LUCJ")]
    local = [i for i, n in enumerate(names) if n.startswith(("givens", "num_num"))]
    if not lucj or not local:
        raise ValueError(f"pool lacks LUCJ or local ops; names start: {names[:6]}")
    rng = np.random.default_rng(seed)
    local10 = [int(x) for x in rng.choice(local, 10)]
    mixed10 = list(local10)
    mixed10[3] = lucj[0]          # what a cost-aware policy mostly emits
    out = {"local10": local10, "mixed10": mixed10, "lucj1": [lucj[0]],
           "lucj2": [lucj[0], lucj[0]]}
    # Exclude identity and force one LUCJ block to avoid trivial HF ties.
    choices = [i for i in range(1, len(pool))]
    for k in range(n_random):
        rng_k = np.random.default_rng(1000 + k)
        draw = [int(x) for x in rng_k.choice(choices, 10)]
        draw[int(rng_k.integers(len(draw)))] = int(rng_k.choice(lucj))
        out[f"rand{k}"] = draw
    return out


#: Config keys the sweep sets per run (--sampler-kind and --bonds), not open to --set.
SWEPT_KEYS = ("sampler.kind", "sampler.mps_max_bond")


def make_cfg(args):
    from hydra import compose, initialize_config_dir

    overrides = [f"molecule={args.molecule}", "reference_keys=[]",
                 f"sampler.kind={args.sampler_kind}",
                 f"sampler.shots={args.shots}", f"sqd.max_dim={args.max_dim}",
                 "sqd.recover_iterations=0",
                 f"sqd.diag_backend={args.diag_backend}"]
    if args.sampler_kind == "mps_topk":
        overrides.append("sqd.enlarge_method=None")
    if args.sbd_binary_path:
        overrides.append(f"sqd.sbd_binary_path={args.sbd_binary_path}")
    if getattr(args, "beam_width", None) is not None:
        overrides.append(f"sampler.mps_beam_width={args.beam_width}")
    overrides += list(getattr(args, "set", None) or [])        # last, so they win
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs_sqd")):
        return compose(config_name="default", overrides=overrides)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecule", default="snoh3_l", help="Hydra molecule group, e.g. snoh3_m / snoh3_l")
    ap.add_argument("--bonds", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--sequences", nargs="+", default=["local10", "mixed10", "lucj1"],
                    help="named sequences, plus rand0..randN-1 when --random-sequences is set")
    ap.add_argument("--random-sequences", type=int, default=0, help="add N random 10-token circuits")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--max-dim", type=int, default=1000)
    ap.add_argument("--diag-backend", default="sbd", choices=["sqd", "sbd", "fci"])
    ap.add_argument("--sbd-binary-path", default=None)
    ap.add_argument("--sampler-kind", default="mps", choices=["mps", "mps_topk"],
                    help="mps draws shots; mps_topk selects amplitudes (exact only if certified)")
    ap.add_argument("--beam-width", type=int, default=None,
                    help="mps_topk only: prefixes carried between sites, default 4*max_dim.")
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat each point N times; summaries use finite seed means and ranges")
    ap.add_argument("--with-exact", action="store_true",
                    help="add an ffsim reference (shots or top-K), where its CI vector fits in memory")
    ap.add_argument("--skip-energy", action="store_true",
                    help="sampler only: valid fraction and time, no diagonalization")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="extra Hydra overrides, e.g. sampler.mps_dtype=complex128 "
                         "sampler.mps_abs_cutoff=0; applied last")
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args(argv)
    if any(n <= 0 for n in [args.shots, args.max_dim, args.seeds, *args.bonds]):
        ap.error("shots, max-dim, seeds and bonds must be positive")
    if args.random_sequences < 0 or (args.beam_width is not None and args.beam_width <= 0):
        ap.error("random-sequences must be nonnegative and beam-width must be positive")
    # The sweep sets these itself, per run, so an override would be silently replaced.
    swept = [o for o in args.set if o.split("=", 1)[0].lstrip("+~") in SWEPT_KEYS]
    if swept:
        ap.error(f"--set cannot change {', '.join(swept)}: use --sampler-kind and --bonds")
    return args


def measure(sampler, pipeline, seq, *, chi, name, seed, reference=None):
    """Measure one point; failed/non-finite energies remain null in the report."""
    t = time.perf_counter()
    counts = sampler.sample_state(seq, seed=seed)
    t_sample = time.perf_counter() - t
    keys = set(counts)
    row = {"chi": chi, "sequence": name, "tokens": seq, "seed": seed,
           "recall_vs_exact": len(reference & keys) / len(reference) if reference else None,
           "captured_weight": getattr(sampler, "last_captured_weight", None),
           "topk_exact": getattr(sampler, "last_topk_exact", None),
           "t_sample_s": t_sample, "n_distinct": len(counts),
           "valid_fraction": getattr(sampler, "last_valid_fraction", float(bool(counts))),
           "energy": None, "t_energy_s": None}
    if pipeline is not None:
        if counts:
            t = time.perf_counter()
            try:
                energy = float(pipeline.energy_of_counts(counts))
                if not math.isfinite(energy):
                    raise ValueError(f"non-finite energy: {energy}")
                row["energy"] = energy
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["t_energy_s"] = time.perf_counter() - t
        else:
            row["error"] = "empty sampled subspace"
    return row, keys


def _log_measurement(row):
    label = "exact" if row["chi"] == 0 else str(row["chi"])
    tag = f" seed={row['seed']}"
    if row["recall_vs_exact"] is not None:
        tag += f" recall={row['recall_vs_exact']:.3f}"
    if row["captured_weight"] is not None:
        tag += f" weight={row['captured_weight']:.4f}"
        tag += "" if row["topk_exact"] else " BEAM-LIMITED"
    e = row["energy"]
    print(f"[sweep] chi={label:>5} {row['sequence']:9s}{tag} valid={row['valid_fraction']:.3f} "
          f"distinct={row['n_distinct']:6d} sample={row['t_sample_s']:7.1f}s "
          f"energy={'n/a' if e is None else f'{e:.6f}'} "
          f"{row.get('error', '')}", file=sys.stderr, flush=True)


def main():
    args = _parse_args()

    from sqd_gqe.factory import SqdFactory

    cfg = make_cfg(args)
    factory = SqdFactory()
    system = factory.build_system(cfg)
    pool = factory.build_operator_pool(cfg)
    pipeline = None if args.skip_energy else factory.build_pipeline(cfg)
    sequences = build_sequences(pool, n_random=args.random_sequences)
    if args.random_sequences and not any(n.startswith("rand") for n in args.sequences):
        args.sequences = list(args.sequences) + [f"rand{k}" for k in range(args.random_sequences)]
    unknown = [n for n in args.sequences if n not in sequences]
    if unknown:
        raise SystemExit(f"unknown --sequences {unknown}; available: {sorted(sequences)}")

    seeds = [cfg.trainer.seed + k for k in range(args.seeds)]

    nq = 2 * system.norb
    print(f"[sweep] {args.molecule}: {nq}q norb={system.norb} nelec={tuple(system.nelec)} "
          f"fci_dim={system.fci_dimension():.3g}", file=sys.stderr, flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.molecule}.json"
    # Report what the runs used: the composed config, after every override.
    from omegaconf import OmegaConf
    record = {"molecule": args.molecule, "qubits": nq, "norb": system.norb,
              "nelec": list(system.nelec), "fci_dim": float(system.fci_dimension()),
              "shots": int(cfg.sampler.shots), "max_dim": int(cfg.sqd.max_dim), "seeds": seeds,
              "sampler_kind": cfg.sampler.kind, "beam_width": cfg.sampler.get("mps_beam_width"),
              "overrides": args.set, "diag_backend": cfg.sqd.diag_backend,
              "sampler": OmegaConf.to_container(cfg.sampler, resolve=True),
              "sqd": OmegaConf.to_container(cfg.sqd, resolve=True), "rows": []}

    det_keys: dict = {}          # (chi, sequence, seed) -> the determinants it returned
    runs = [(0, "topk" if args.sampler_kind == "mps_topk" else "ffsim")] if args.with_exact else []
    runs += [(chi, args.sampler_kind) for chi in sorted(set(args.bonds))]
    for chi, kind in runs:
        c = cfg.copy()
        c.sampler.kind = kind
        if chi:
            c.sampler.mps_max_bond = chi            # the one setting the sweep varies
        sampler = factory.build_sampler(c)
        for name in args.sequences:
            for seed in seeds:
                row, keys = measure(sampler, pipeline, sequences[name], chi=chi, name=name,
                                    seed=seed, reference=det_keys.get((0, name, seed)))
                det_keys[(chi, name, seed)] = keys
                record["rows"].append(row)
                out_path.write_text(json.dumps(record, indent=1, allow_nan=False))
                _log_measurement(row)

    print(f"\n[sweep] wrote {out_path}", file=sys.stderr)
    _summarise(record)
    _rank_stability(record)
    _determinant_convergence(det_keys)


def _aggregate_rows(rows):
    """Shared chi/sequence statistics; keep failure counts alongside finite seed means."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["chi"], row["sequence"]), []).append(row)
    by_chi = {}
    for (chi, name), group in grouped.items():
        finite_rows = [r for r in group
                       if r["energy"] is not None and math.isfinite(r["energy"])]
        energies = [r["energy"] for r in finite_rows]
        by_chi.setdefault(chi, {})[name] = {
            "energy": fmean(energies) if energies else None,
            "seed_energies": {r["seed"]: r["energy"] for r in finite_rows},
            "seed_range": max(energies) - min(energies) if len(energies) > 1 else None,
            "n_ok": len(energies), "n_total": len(group),
            **{key: fmean(r[key] for r in group)
               for key in ("valid_fraction", "n_distinct", "t_sample_s")},
        }
    return {chi: by_chi[chi] for chi in sorted(by_chi)}


def _summarise(record):
    """One row per chi, with finite seed means and success counts."""
    stats = _aggregate_rows(record["rows"])
    print(f"\n{record['molecule']} ({record['qubits']}q)")
    for name in dict.fromkeys(r["sequence"] for r in record["rows"]):
        print(f"\n  {name}")
        print(f"    {'chi':>6} {'ok/all':>7} {'valid':>7} {'distinct':>9} {'sample_s':>9} "
              f"{'mean energy':>14} {'d(mean) mHa':>14}")
        prev = None
        for chi, circuits in stats.items():
            if name not in circuits:
                continue
            r = circuits[name]
            e = r["energy"]
            d = "" if (e is None or prev is None) else f"{1e3 * (e - prev):+.3f}"
            ok = f"{r['n_ok']}/{r['n_total']}"
            print(f"    {chi:>6} {ok:>7} {r['valid_fraction']:>7.3f} {r['n_distinct']:>9.1f} "
                  f"{r['t_sample_s']:>9.1f} {'n/a' if e is None else f'{e:>14.6f}'} {d:>14}")
            prev = e


def _determinant_convergence(det_keys):
    """Report determinant overlap across chi, including sizes without an exact reference."""
    by_name: dict = {}
    for (chi, name, seed), keys in det_keys.items():
        if chi:                             # chi=0 is the exact reference, not a rung
            by_name.setdefault(name, {}).setdefault(chi, {})[seed] = keys
    if not by_name:
        return
    chis = sorted({c for d in by_name.values() for c in d})
    rungs = list(zip(chis, chis[1:]))
    if not rungs:
        return

    print("\n  determinant-set overlap between successive chi (seed-averaged)")
    header = "".join(f"{f'{lo}->{hi}':>12}" for lo, hi in rungs)
    print(f"    {'circuit':10}{header}")
    for name, by_chi in by_name.items():
        cells = ""
        for lo, hi in rungs:
            a, b = by_chi.get(lo, {}), by_chi.get(hi, {})
            seeds = a.keys() & b.keys()
            if not seeds:
                cells += f"{'n/a':>12}"
                continue
            vals = [len(a[s] & b[s]) / len(a[s]) if a[s] else 0.0 for s in seeds]
            cells += f"{fmean(vals):>12.3f}"
        print(f"    {name:10}{cells}")


def _kendall_tau(a, b):
    """Rank correlation in [-1, 1], ignoring tied pairs; None if all pairs tie."""
    conc = disc = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    total = conc + disc
    return None if total == 0 else (conc - disc) / total


def _rank_stability(record):
    """Compare complete seed means; separate reference variation from other chi values."""
    stats = _aggregate_rows(record["rows"])
    by_chi = {chi: {name: s["energy"] for name, s in circuits.items()
                    if s["energy"] is not None and s["n_ok"] == s["n_total"]}
              for chi, circuits in stats.items()}
    by_chi = {chi: es for chi, es in by_chi.items() if es}
    if len(by_chi) < 2:
        return

    ref_chi = 0 if 0 in by_chi else max(by_chi)
    ref = by_chi[ref_chi]
    ref_label = "exact" if ref_chi == 0 else f"chi={ref_chi}"

    print(f"\n  ranking stability vs {ref_label}")
    print(f"    {'chi':>6} {'n':>3} {'tau':>6} {'spread mHa':>11} {'seed mHa':>9} order (best first)")
    for chi in sorted(by_chi):
        e = by_chi[chi]
        shared = [k for k in e if k in ref]
        if len(shared) < 2:
            continue
        tau = _kendall_tau([e[k] for k in shared], [ref[k] for k in shared])
        spread = 1e3 * (max(e[k] for k in shared) - min(e[k] for k in shared))
        order = " < ".join(sorted(shared, key=lambda k: e[k]))
        tau_s = "n/a" if tau is None else f"{tau:+.2f}"
        chi_s = "exact" if chi == 0 else str(chi)
        ranges = [stats[chi][k]["seed_range"] for k in shared]
        noise = max((v for v in ranges if v is not None), default=None)
        noise_s = "n/a" if noise is None else f"{1e3 * noise:.1f}"
        print(f"    {chi_s:>6} {len(shared):>3} {tau_s:>6} {spread:>11.1f} {noise_s:>9} {order}")

    spread = 1e3 * (max(ref.values()) - min(ref.values()))
    print(f"\n    spread between circuits at {ref_label}:  {spread:.1f} mHa")

    truncated = [c for c in sorted(by_chi) if c != 0]
    unresolved = False
    if len(truncated) >= 2:
        lo, hi = truncated[-2:]
        shared = by_chi[lo].keys() & by_chi[hi].keys()
        changes = {k: by_chi[hi][k] - by_chi[lo][k] for k in shared}
        if changes:
            shifts = list(changes.values())
            print(f"    mean |dE| from chi={lo} to {hi}: {1e3 * fmean(map(abs, shifts)):.1f} mHa")
            # A common energy offset does not affect group-relative rewards.
            relative_shift = 1e3 * (max(shifts) - min(shifts))
            print(f"    largest change in a circuit-pair gap: {relative_shift:.1f} mHa")
            paired = [k for k in changes if k in ref]
            unresolved = any(abs(changes[a] - changes[b]) > abs(ref[a] - ref[b])
                             for a, b in combinations(paired, 2))
    ranges = [stats[ref_chi][k]["seed_range"] for k in ref]
    noise = max((v for v in ranges if v is not None), default=None)
    if noise is not None:
        print(f"    widest reference spread across sampling seeds: {1e3 * noise:.1f} mHa")

    reversed_pairs = []
    for a, b in combinations(ref, 2):
        ea, eb = (stats[ref_chi][k]["seed_energies"] for k in (a, b))
        gaps = [ea[seed] - eb[seed] for seed in ea.keys() & eb.keys()]
        if gaps and min(gaps) < 0 < max(gaps):
            reversed_pairs.append(f"{a}/{b}")
    if reversed_pairs:
        print("    -> reference ordering reverses across sampling seeds for: "
              + ", ".join(reversed_pairs))
        if record.get("sampler_kind", "mps") == "mps":
            print("       Raise shots, not chi; repeat sampling to check the reference.")
        else:
            print("       These are shot-free measurements; check numerical stability and beam limits.")
    if unresolved:
        print("    -> a circuit-pair gap changes with chi by more than its reference separation;")
        print("       the ranking is not resolved here.")


if __name__ == "__main__":
    main()
