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

"""Per-iteration tracking of what the policy generates and how chemically sensible the sampled states are.

Used by ``benchmark/amortized_train.py --track-circuits FILE`` ("Training 2"). Two kinds of numbers per sampled circuit:

1. Sequence statistics (``sequence_stats``): which operator families the token sequence uses, and how its Givens rotations sit relative
   to the Fermi level. The pool's Givens rotation acts on the neighbouring spatial orbitals (p, p+1) of BOTH spin species, so on the
   Hartree-Fock determinant it changes anything only for the HOMO-LUMO pair (p = n_alpha - 1); a rotation between two occupied or two
   virtual orbitals leaves the Hartree-Fock state unchanged.
2. Sampled-state statistics (``spin_excitation_stats``): the measured (alpha, beta) occupation pairs, split into
   - the right spin sector (n_alpha, n_beta preserved) versus a wrong one: an alpha<->beta transfer (n_alpha +-1, n_beta -+1 with the total
     electron number right) or an electron-number error. Every operator in the pool conserves n_alpha and n_beta, so on a noiseless
     statevector simulator the wrong-sector fraction is exactly zero; on hardware it measures noise;
   - among right-sector pairs, the excitation relative to Hartree-Fock resolved by spin: alpha-only (alpha -> alpha), beta-only
     (beta -> beta) or both (alpha -> alpha together with beta -> beta), and the HOMO -> LUMO excitations in particular.

The bit convention is that of ``circuit_metrics``: bit p of the alpha (beta) integer means spatial orbital p is occupied in that spin
channel, the Hartree-Fock determinant fills the lowest n_alpha (n_beta) orbitals.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_GIVENS = re.compile(r"givens\((\d+),(\d+);")


def _pop(x: int) -> int:
    return bin(int(x)).count("1")


def op_family(name: str) -> str:
    if name == "I":
        return "identity"
    if name.startswith("LUCJ"):
        return "lucj"
    if name.startswith("orb_rot"):
        return "orb_rot"
    if name.startswith("givens"):
        return "givens"
    if name.startswith("diag_coulomb"):
        return "diag_coulomb"
    if name.startswith("num_num"):
        return "num_num"
    return "other"


def sequence_stats(seq, pool, nelec, norb) -> dict:
    """Statistics of one generated token sequence (the leading start token is not part of the circuit and is dropped by the caller)."""
    na = int(nelec[0])
    fam = Counter()
    giv = Counter()
    toks = list(seq)
    covered = set()
    for t in toks:
        try:
            op = pool[int(t)]
        except Exception:                # out-of-range indices act as identity in the pool
            fam["identity"] += 1
            continue
        f = op_family(op.name)
        fam[f] += 1
        if f in ("lucj", "orb_rot"):
            covered.update(range(norb))
        elif f == "givens":
            m = _GIVENS.match(op.name)
            p, q = int(m.group(1)), int(m.group(2))
            covered.update((p, q))
            if p == na - 1 and q == na:
                giv["homo_lumo"] += 1
            elif q < na:
                giv["occ_occ"] += 1
            elif p >= na:
                giv["virt_virt"] += 1
            else:
                giv["other"] += 1
        elif f in ("diag_coulomb", "num_num"):
            covered.update(op.support or ())
    n = max(len(toks), 1)
    cnt = Counter(toks)
    probs = [c / n for c in cnt.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    longest = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    repeats = sum(1 for a, b in zip(toks, toks[1:]) if a == b)
    nonid = n - fam["identity"]
    effective_hf = fam["lucj"] + fam["orb_rot"] + giv["homo_lumo"]
    out = {
        "n_tokens": len(toks), "n_distinct_tokens": len(cnt), "token_entropy_bits": entropy, "longest_run": longest,
        "adjacent_repeat_frac": repeats / max(len(toks) - 1, 1), "orbital_coverage": len(covered) / max(norb, 1),
        "n_identity": fam["identity"], "n_lucj": fam["lucj"], "n_orb_rot": fam["orb_rot"], "n_givens": fam["givens"],
        "n_diag_coulomb": fam["diag_coulomb"], "n_num_num": fam["num_num"],
        "n_givens_homo_lumo": giv["homo_lumo"], "n_givens_occ_occ": giv["occ_occ"], "n_givens_virt_virt": giv["virt_virt"],
        "n_givens_other": giv["other"],
        "frac_identity": fam["identity"] / n,
        # gates that can move weight away from the Hartree-Fock determinant at all
        "n_hf_changing_gates": effective_hf,
        "frac_givens_homo_lumo": giv["homo_lumo"] / max(fam["givens"], 1),
    }
    return out


def spin_excitation_stats(counts, nelec, norb) -> dict:
    """Spin-resolved excitation analysis of one circuit's sampled ``{(alpha, beta): count}`` pairs (all shots, valid or not)."""
    na, nb = int(nelec[0]), int(nelec[1])
    hf_a, hf_b = (1 << na) - 1, (1 << nb) - 1
    hl_a = hf_a ^ (1 << (na - 1)) ^ (1 << na) if na > 0 else hf_a      # alpha HOMO -> LUMO single excitation
    hl_b = hf_b ^ (1 << (nb - 1)) ^ (1 << nb) if nb > 0 else hf_b
    w = Counter()
    ka_dist, kb_dist = Counter(), Counter()
    n_valid_dets = 0
    for (a, b), c in counts.items():
        c = float(c)
        w["total"] += c
        pa, pb = _pop(a), _pop(b)
        if pa == na and pb == nb:
            w["valid"] += c
            n_valid_dets += 1
            ka = _pop(hf_a & ~a)             # alpha electrons moved out of the HF-occupied orbitals
            kb = _pop(hf_b & ~b)
            ka_dist[min(ka, 4)] += c
            kb_dist[min(kb, 4)] += c
            if ka == 0 and kb == 0:
                w["hf"] += c
            elif kb == 0:
                w["alpha_only"] += c
            elif ka == 0:
                w["beta_only"] += c
            else:
                w["alpha_and_beta"] += c
            if a == hl_a and b == hf_b:
                w["homo_lumo_alpha"] += c
            elif a == hf_a and b == hl_b:
                w["homo_lumo_beta"] += c
            elif a == hl_a and b == hl_b:
                w["homo_lumo_alpha_beta"] += c
            if ka == 1 and kb == 0:
                w["single_alpha"] += c
            elif ka == 0 and kb == 1:
                w["single_beta"] += c
            elif ka == 2 and kb == 0:
                w["double_alpha_alpha"] += c
            elif ka == 0 and kb == 2:
                w["double_beta_beta"] += c
            elif ka == 1 and kb == 1:
                w["double_alpha_beta"] += c
        elif pa + pb == na + nb and abs(pa - na) == 1:
            w["spin_transfer"] += c           # an electron moved between the alpha and the beta channel
        elif pa + pb != na + nb:
            w["number_error"] += c
        else:
            w["other_wrong_sector"] += c
    tot = w["total"] or 1.0
    val = w["valid"] or 1.0
    out = {"n_shots": int(w["total"]), "n_valid_dets": n_valid_dets}
    for k in ("valid", "spin_transfer", "number_error", "other_wrong_sector"):
        out[f"frac_{k}"] = w[k] / tot
    for k in ("hf", "alpha_only", "beta_only", "alpha_and_beta", "homo_lumo_alpha", "homo_lumo_beta", "homo_lumo_alpha_beta",
              "single_alpha", "single_beta", "double_alpha_alpha", "double_beta_beta", "double_alpha_beta"):
        out[f"valid_{k}"] = w[k] / val          # fractions of the right-sector shots
    out["valid_homo_lumo_any"] = (w["homo_lumo_alpha"] + w["homo_lumo_beta"] + w["homo_lumo_alpha_beta"]) / val
    out["valid_rank_alpha"] = {str(k): v / val for k, v in sorted(ka_dist.items())}
    out["valid_rank_beta"] = {str(k): v / val for k, v in sorted(kb_dist.items())}
    # mixed-spin imbalance among the excited right-sector shots: alpha-only vs beta-only should be symmetric for a spin-balanced circuit
    ex = w["alpha_only"] + w["beta_only"]
    out["alpha_beta_asymmetry"] = (w["alpha_only"] - w["beta_only"]) / ex if ex else 0.0
    return out


def hf_fraction(counts, nelec) -> float:
    """Share of the right-spin-sector shots that are exactly the Hartree-Fock determinant (1.0 if no shot is in the right sector)."""
    na, nb = int(nelec[0]), int(nelec[1])
    hf_a, hf_b = (1 << na) - 1, (1 << nb) - 1
    valid = hf = 0.0
    for (a, b), c in (counts or {}).items():
        if _pop(a) == na and _pop(b) == nb:
            valid += c
            if a == hf_a and b == hf_b:
                hf += c
    return hf / valid if valid else 1.0


def spin_imbalance(counts, nelec) -> float:
    """|alpha-only - beta-only| weight, as a fraction of the right-spin-sector shots. 0 for a spin-balanced (closed-shell) circuit."""
    na, nb = int(nelec[0]), int(nelec[1])
    hf_a, hf_b = (1 << na) - 1, (1 << nb) - 1
    valid = a_only = b_only = 0.0
    for (a, b), c in (counts or {}).items():
        if _pop(a) == na and _pop(b) == nb:
            valid += c
            if a != hf_a and b == hf_b:
                a_only += c
            elif a == hf_a and b != hf_b:
                b_only += c
    return abs(a_only - b_only) / valid if valid else 0.0


NUMERIC_MEAN = None


def mean_numeric(rows: list[dict]) -> dict:
    """Mean of every plain numeric field over a list of per-circuit dicts (nested dicts are skipped)."""
    out = {}
    if not rows:
        return out
    for k, v in rows[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            out[k] = sum(vals) / len(vals) if vals else None
    return out


def iteration_tracking(seqs, counts_list, pool, nelec, norb, energies, depths, two_qs, idles, token_entropy, best_i, beta) -> tuple[dict, list]:
    """Per-circuit records and an iteration summary (means over the circuits and the values for the best circuit)."""
    per = []
    for i, seq in enumerate(seqs):
        toks = seq[1:] if len(seq) > 1 else seq              # drop the start token
        s = sequence_stats(toks, pool, nelec, norb)
        e = spin_excitation_stats(counts_list[i] or {}, nelec, norb)
        rec = {"idx": i, "seq": [int(t) for t in toks], "energy": float(energies[i]) if energies is not None else None,
               "depth": depths[i] if depths else None, "two_qubit_gates": two_qs[i] if two_qs else None,
               "idle_frac": idles[i] if idles else None, "seq_stats": s, "exc_stats": e}
        per.append(rec)
    summary = {
        "best_idx": int(best_i), "beta": float(beta),
        "policy_entropy_nats": float(token_entropy) if token_entropy is not None else None,
        "policy_effective_vocab": float(math.exp(token_entropy)) if token_entropy is not None else None,
        "seq_mean": mean_numeric([r["seq_stats"] for r in per]), "exc_mean": mean_numeric([r["exc_stats"] for r in per]),
        "seq_best": per[best_i]["seq_stats"] if per else None, "exc_best": per[best_i]["exc_stats"] if per else None,
        "depth_mean": (sum(r["depth"] for r in per if r["depth"] is not None) / max(sum(1 for r in per if r["depth"] is not None), 1))
        if depths else None,
        "n_circuits_with_valid_shots": sum(1 for r in per if r["exc_stats"]["frac_valid"] > 0),
    }
    return summary, per
