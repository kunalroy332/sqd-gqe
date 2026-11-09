# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Regenerate the fresh-shot result tables of the manuscript from circuit_analysis/{rerank,rerecovery}.

Replaces whole \\begin{table}...\\end{table} environments (found by their \\label) in paper.tex.
Re-runnable: cells whose recovery job has not finished are printed as 'pend.'.
Usage: make_tables.py [--tex PATH] [--dry-run]
"""
import argparse
import os
import re
from pathlib import Path

from tables_data import LABEL, LIN_ORDER, MOL_NAME, TIER, load, missing_runs, run_key

LIN = ["a1-46q", "a1-70q", "a5-46q", "a5-70q"]
TEX = Path(__file__).resolve().parents[3] / "docs/report/manuscript/paper.tex"
PEND = "{\\color{red}pend.}"
NA = "n/a"   # not computable: more than 44 qubits, or SnOH3-me M (38q) whose 12+12-electron determinant space has 2.5e9 elements
import json as _json


def div_cells(r):
    """F, H and the operator-norm bound of the best circuit against the idealised LUCJ circuit (circuit_divergence.py); n/a above 44 qubits."""
    from tables_data import ROOT
    f = ROOT / "divergence" / f"{r['tag']}.json"
    if not f.exists():
        return (NA, NA, NA) if (r["nq"] > 44 or r["mol"] == "sno38") else (PEND, PEND, PEND)
    idx = r["t"]["best"]["idx"]
    c = {x["index"]: x for x in _json.loads(f.read_text())["circuits"]}.get(idx)
    if c is None:
        return (PEND, PEND, PEND)
    return (f"{c['F']:.3f}", f"{c['H']:.3f}", f"{c['op_norm']:.2f}" if c.get("op_norm") is not None else "--")



def lin_name(l):
    return l.replace("a1", "A1").replace("a5", "A5")


def rows(d, only_trained=False, l46=None):
    rs = [r for r in d.values() if (r["mol"] == "L46") == l46 and (r["policy"] == "trained" or not only_trained)]
    return sorted(rs, key=run_key)


def err(e, prec=None):
    if not e["done"] and e.get("partial"):
        x = e["p_err"]
        return f"$\\le${x:.2f}$^{{\\dagger}}$" if abs(x) < 1 else f"$\\le${x:.1f}$^{{\\dagger}}$"
    if not e["done"]:
        return PEND
    x = e["err"]
    return f"{x:.2f}" if abs(x) < 1 else f"{x:.1f}"


def with_blocks(rs, fmt):
    out, prev = [], None
    for r in rs:
        if prev is not None and r["mol"] != prev:
            out.append(r"\addlinespace")
        out.append(fmt(r))
        prev = r["mol"]
    return "\n".join(out)


def detail_table(d, l46, label, caption):
    ncol = 7 if l46 else 8

    def fmt(r):
        tgt = "" if l46 else f"{r['label']} & "
        if r.get("missing"):
            return f"{tgt}{lin_name(r['lineage'])} {r['policy']} & \\multicolumn{{6}}{{l}}{{\\textit{{queued: to be run when the cluster is available}}}} \\\\"
        t = r["t"]
        return (f"{tgt}{lin_name(r['lineage'])} {r['policy']} & "
                f"{t['best']['depth']}/{t['second']['depth']} & {t['best']['two_q']}/{t['second']['two_q']} & "
                f"{t['best']['valid_shots']}/{t['second']['valid_shots']} & "
                f"{err(t['best'])}/{err(t['second'])}/{err(t['merged'])} & {err(t['oldbest'])}/{err(t['oldsecond'])} & "
                f"{('%.1f' % t['best']['si']) if t['best']['done'] and t['best']['si'] is not None else PEND} \\\\")
    head = ("Lineage/Policy" if l46 else "Target & Lineage/Policy")
    col = "@{}l" + ("l" if not l46 else "") + "rrrrrr@{}"
    allr = [r for r in d.values() if (r["mol"] == "L46") == l46] + ([] if l46 else missing_runs())
    allr = sorted(allr, key=run_key)
    sec = lambda txt: f"\\multicolumn{{{ncol}}}{{l}}{{\\textbf{{{txt}}}}} \\\\"
    body = (sec("Trained policies") + "\n" + with_blocks([r for r in allr if r["policy"] == "trained"], fmt) +
            "\n\\midrule\n" + sec("Untrained controls") + "\n" + with_blocks([r for r in allr if r["policy"] != "trained"], fmt))
    return (f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
            f"\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{{col}}}\n\\toprule\n{head} & Depth (best/2nd) & 2q (best/2nd) & Valid shots (best/2nd) & "
            f"$\\Delta E$ best/2nd/merged (mHa) & $\\Delta E$ E+D 1/2 (mHa) & $\\Delta E$ best-SI (mHa) \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}")


def long_env(label, caption, header, rows_):
    return (f"{{\\footnotesize\n\\begin{{longtable}}{{@{{}}llrrrrrrr@{{}}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
            f"\\toprule\n{header} \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"
            "\\bottomrule\n\\endfoot\n" + "\n".join(rows_) + "\n\\end{longtable}}")


def exc_table(d, l46, label, caption):
    def one(r, ty):
        e = r["t"][ty]
        nm = f"{r['label']} " if not l46 else ""
        nm += f"{lin_name(r['lineage'])} {r['policy']}"
        if not e["done"]:
            return f"{nm} & {ty} & {PEND} & {PEND} & -- & -- & -- & -- & -- \\\\"
        x = e["exc"]
        return (f"{nm} & {ty} & $-${abs(e['energy']):.6f} & {err(e)} & {int(x['S'])} & {int(x['D'])} & "
                f"{int(x['T'])} & {int(x['Q'])} & {int(x['higher'])} \\\\")

    out, prev = [], None
    for r in rows(d, l46=l46):
        if prev is not None:
            out.append(r"\addlinespace")
        out += [one(r, t) for t in ("best", "second", "merged")]
        prev = r
    return long_env(label, caption, "Target/Lineage & Type & Energy (E$_h$) & $\\Delta E$ (mHa) & S & D & T & Q & Higher", out)


def raw_table(d, label, caption):
    out, prev = [], None
    rs = sorted(d.values(), key=run_key)
    for r in rs:
        if prev is not None:
            out.append(r"\addlinespace")
        for ty in ("best", "second", "merged"):
            e = r["t"][ty]
            nm = f"{r['label']} {lin_name(r['lineage'])} {r['policy']}"
            if not e["done"]:
                out.append(f"{nm} & {ty} & -- & -- & -- & -- & -- & -- & -- \\\\")
                continue
            x = e["raw"]
            out.append(f"{nm} & {ty} & {int(x['HF'])} & {int(x['S'])} & {int(x['D'])} & {int(x['T'])} & {int(x['Q'])} & "
                       f"{int(x['higher'])} & {int(x['combined'])} \\\\")
        prev = r
    return long_env(label, caption, "Target/Lineage & Type & HF & S & D & T & Q & Higher & n", out)


def master_table(d, label, caption):
    def cell(e):
        if not e["done"] and e.get("partial"):
            return f"$\\le-${abs(e['p_energy']):.6f}$^{{\\dagger}}$/{err(e)}"
        return PEND if not e["done"] else f"$-${abs(e['energy']):.6f}/{err(e)}"

    def fmt(r):
        t = r["t"]
        return (f"{r['label']} & {lin_name(r['lineage'])[:2]} ({lin_name(r['lineage'])}) & {cell(t['best'])} & {cell(t['second'])} & "
                f"{cell(t['merged'])} & {err(t['oldbest'])}/{err(t['oldsecond'])} & {t['best']['depth']} & {t['best']['two_q']} \\\\")
    allr = [(run_key(r), "r", r) for r in rows_all_trained(d)]
    miss = [((LABEL[mk][1], LABEL[mk][2], lin), "m", (mk, lin)) for mk, lin in planned_missing(d)] if MISSING_NOTE else []
    out, prev = [], None
    def sk(t):
        if t[1] == "r":
            r = t[2]; return (r["nq"], r["order"], LIN_ORDER.index(r["lineage"]))
        return (LABEL[t[2][0]][1], LABEL[t[2][0]][2], LIN_ORDER.index(t[2][1]))
    for _, kind, x in sorted(allr + miss, key=sk):
        mk = x["mol"] if kind == "r" else x[0]
        if prev is not None and mk != prev:
            out.append(r"\addlinespace")
        if kind == "r":
            out.append(fmt(x))
        else:
            # a planned run with no ranking selection: the batch of ten had a single circuit with valid shots, so no best/second pair exists
            D = "--"
            out.append(f"{LABEL[mk][0]} & {lin_name(x[1])[:2]} ({lin_name(x[1])}) & \\multicolumn{{4}}{{c}}{{only one circuit with valid shots: no best/second pair}} & {D} & {D} \\\\")
        prev = mk
    body = "\n".join(out)
    return (f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{3pt}}\n"
            "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\nMolecule (tier, qubits) & Arch.\\ (lineage) & Best E$_h$/mHa & Second E$_h$/mHa & "
            "Merged E$_h$/mHa & E+D 1/2 (mHa) & Depth & 2q \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


PLANNED = {lin: ["sno32", "h2o_s19", "sno38", "2h2o_s22", "L46", "h2o_m_no23", "llno28", "h2o_l28", "2h2o_m_no29", "h2o_ll33", "2h2o_l35"] for lin in ("a1-46q", "a5-46q", "a1-70q", "a5-70q")}
MISSING_NOTE = os.environ.get("CA_MISSING", "0") == "1"   # the HF-penalty campaign plans all 11 tiers on all four lineages


def planned_missing(d):
    have = {(r["mol"], r["lineage"]) for r in d.values() if r["policy"] == "trained"}
    return [(mk, lin) for lin, mks in PLANNED.items() for mk in mks if (mk, lin) not in have]


def rows_all_trained(d):
    return sorted([r for r in d.values() if r["policy"] == "trained"],
                  key=run_key)


def dmrg_table(d, label, caption):
    order = sorted(LABEL, key=lambda k: (LABEL[k][1], LABEL[k][2]))
    out = []
    for mk in order:
        cand = [((e["err"] if e["done"] else e["p_err"]), (e["energy"] if e["done"] else e["p_energy"]), r, ty, e["done"]) for r in d.values() if r["mol"] == mk
                for ty, e in r["t"].items() if e["done"] or e.get("partial")]
        if not cand:
            continue
        best = min(cand, key=lambda c: (round(c[0], 2), not c[4]))
        ty = best[3]
        tyname = {"oldbest": "E+D 1", "oldsecond": "E+D 2"}.get(ty, ty)
        ref = best[2]["ref"]
        e_s = f"<0.01" if best[0] < 0.01 else (f"{best[0]:.2f}" if best[0] < 1 else f"{best[0]:.1f}")
        npend = sum(1 for r in d.values() if r["mol"] == mk for e in r["t"].values() if not e["done"])
        row = (f"{LABEL[mk][0]} & $-${abs(ref):.6f} & $-${abs(best[1]):.6f} & {e_s} & "
               f"{lin_name(best[2]['lineage'])} {best[2]['policy']} & {tyname}{'$^{*}$' if npend else ''} \\\\")
        if not best[4]:
            row = row.replace(f" & {e_s} & ", f" & $\\le${e_s}$^{{\\dagger}}$ & ", 1)
        out.append(row)
    return (f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n\\setlength{{\\tabcolsep}}{{3pt}}\n"
            "\\begin{tabular}{@{}lrrrll@{}}\n\\toprule\nTarget & DMRG ($\\eh$) & Best SQD ($\\eh$) & $\\Delta E$ (mHa) & Lineage & Type \\\\\n"
            "\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def coverage_table(d, label, caption):
    import json
    from tables_data import ROOT
    out, prev, ncirc, nbad = [], None, 0, 0
    for r in sorted(d.values(), key=run_key):
        cf = ROOT / "connectivity" / f"{r['tag']}.json"
        if not cf.exists():
            continue          # connectivity not computed yet for this run (cluster step): the run is left out of the table
        run = json.loads(cf.read_text())
        n = run["n_logical_qubits"]
        tg = [c["logical"]["qubits_touched_any_gate"] for c in run["circuits"]]
        t2 = [c["logical"]["qubits_in_2q_gates"] for c in run["circuits"]]
        ncirc += len(tg)
        nb = sum(1 for x in tg if x < n)
        nbad += nb
        if prev is not None and prev != r["mol"]:
            out.append(r"\\addlinespace")
        prev = r["mol"]
        out.append(f"{r['label']} & {lin_name(r['lineage'])} {r['policy']} & {n} & {min(tg)} / {max(tg)} & {min(t2)} / {max(t2)} & "
                   f"{n - min(tg)} & {nb}/{len(tg)} \\\\")
    nmiss = sum(1 for r in d.values() if not (ROOT / "connectivity" / f"{r['tag']}.json").exists())
    cap = caption.replace("@NRUN@", str(len(d) - nmiss)).replace("@NCIRC@", str(ncirc)).replace("@NBAD@", str(nbad))
    return (f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{cap}}}\n\\label{{{label}}}\n\\small\n"
            "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nTarget & Lineage/Policy & Target qubits & Qubits with a gate (min / max) & "
            "Qubits in a 2q gate (min / max) & Max idle qubits & Circuits with idle qubits \\\\\n\\midrule\n"
            + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def replace_env(text, label, new_env):
    for kind in ("table", "table*", "longtable"):
        k = re.escape(kind)
        pat = re.compile(r"\\begin\{" + k + r"\}(?:(?!\\begin\{" + k + r"\}|\\end\{" + k + r"\}).)*?\\label\{" + re.escape(label) + r"\}.*?\\end\{" + k + r"\}", re.S)
        mm = list(pat.finditer(text))
        if len(mm) == 1:
            env = new_env
            if kind == "table*" or label == "tab:dmrg-vs-sqd":
                env = env.replace("\\begin{table}", "\\begin{table*}", 1).replace("\\end{table}", "\\end{table*}", 1)
            if label in ("tab:zeroshot-master", "tab:dmrg-vs-sqd"):
                env = env.replace("\\begin{table*}[ht]", "\\begin{table*}[!t]", 1)
            return text[:mm[0].start()] + env + text[mm[0].end():]
    raise AssertionError(label)


def main():
    from captions import CAPTIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(TEX))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = load()
    npend = sum(1 for r in d.values() for e in r["t"].values() if not e["done"])
    pend_note = CAPTIONS["pending"] if npend else ""
    new = {
        "tab:zeroshot-master": master_table(d, "tab:zeroshot-master", CAPTIONS["master"] + pend_note),
        "tab:dmrg-vs-sqd": dmrg_table(d, "tab:dmrg-vs-sqd", CAPTIONS["dmrg"] + ((" An asterisk marks a target with recoveries still pending, so its entry is the lowest of the finished ones." + pend_note) if npend else "")),
        "tab:zeroshot-detail": detail_table(d, True, "tab:zeroshot-detail", CAPTIONS["detail_L"] + pend_note),
        "tab:zeroshot-excitation": exc_table(d, True, "tab:zeroshot-excitation", CAPTIONS["exc_L"] + pend_note),
        "tab:crossnorb-detail": detail_table(d, False, "tab:crossnorb-detail", CAPTIONS["detail_X"] + pend_note),
        "tab:crossnorb-excitation": exc_table(d, False, "tab:crossnorb-excitation", CAPTIONS["exc_X"] + pend_note),
        "tab:coverage-idle": coverage_table(d, "tab:coverage-idle", CAPTIONS["coverage"]),
        "tab:raw-shot-excitation": raw_table(d, "tab:raw-shot-excitation", CAPTIONS["raw"] + pend_note),
    }
    for lab in new:
        if lab in ("tab:zeroshot-master", "tab:dmrg-vs-sqd"):
            continue          # these two are set at a fixed font size (no rescaling), so the text size is the one chosen in the table
        if "resizebox" not in new[lab] and "\\begin{tabular}" in new[lab]:
            new[lab] = new[lab].replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1).replace(
                "\\end{tabular}", "\\end{tabular}}", 1)
    text = Path(a.tex).read_text()
    for lab, env in new.items():
        text = replace_env(text, lab, env)
    if a.dry_run:
        for lab, env in new.items():
            print(f"% ===== {lab}\n{env}\n")
    else:
        Path(a.tex).write_text(text)
    print(f"{len(new)} tables regenerated; {npend} of {sum(len(r['t']) for r in d.values())} result cells pending")


if __name__ == "__main__":
    main()
