# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Training 2: the Stage-1 (ffsim) and Stage-2 (ibm_kobe) trainings of the four lineages (A1/A5 x 46q/70q), 150 + 150 iterations,
with the two warm-start fixes and per-iteration tracking of the generated sequences and of the sampled states.

What differs from Training 1 (the trainings of the first manuscript version; their checkpoints and logs are untouched):
  * one operator vocabulary at both stages (--max-vocab 110, the pool size of the 35-orbital targets), so the token embedding and the
    output head are carried over from Stage 1 to Stage 2 (Training 1: vocabulary 32 -> 74/110, both tensors re-initialised);
  * the prompt encoder is restored from the Stage-1 checkpoint at the Stage-2 warm start (--warmstart-encoder; Training 1 re-initialised it);
  * a fixed 150 iterations in each stage (no early stop; Training 1 stopped itself after 90-132 iterations in Stage 1 and ran 88-150 in Stage 2);
  * --track-circuits: every iteration's token sequences, operator families, Givens rotations relative to the Fermi level, spin-resolved
    excitation statistics of the sampled states (alpha->alpha, beta->beta, wrong spin sector) and the policy's token entropy
    (sqd_gqe/gqe/training_tracking.py); --ibm-save-dir keeps every hardware job's raw counts and transpiled circuits.
Everything else is the Training-1 recipe (the final Stage-2 settings of the 70-qubit runs: depth ceiling 50,000 and energy gate 500 mHa).

Usage on the cluster login node:  python3 make_jobs.py <outdir> ; bash <outdir>/submit_all.sh
Stage 2 of a lineage starts when its Stage 1 has finished successfully (Slurm dependency afterok) and runs a pre-flight check first.
"""
import sys
from pathlib import Path

REPO = "/path/to/sqd-gqe-work"
VENV = "/path/to/sqd-gqe/.venv"
SBD = "/path/to/home/qcsc-prefect/algorithms/sbd/native/diag-gpu"
EXP = f"{REPO}/experiments/training2"
ACCOUNT, PART = "<SLURM_ACCOUNT>", "roquo"
VOCAB, ITERS, SEED = 110, 150, 1234
S1_MOLS = ["small/h2", "small/lih", "small/h2o", "small/beh2", "small/nh3", "small/ch4", "small/n2", "medium/ch2o", "medium/hcn",
           "medium/c2h4", "medium/co2", "dblock/fe_cn6_14o14e_mp2no_def2svp", "dblock/co_nh3_6_15o14e_mp2no_def2svp",
           "dblock/ferrocene_16o16e_mp2no_def2svp"]
ENC = {"a1": "a1_xattn_large_bounded", "a5": "a5_large_bounded"}
# per qubit target: ngates, Stage-1 depth ceiling, Stage-2 orbital tag / depth ceiling / energy gate
TARGET = {"46q": dict(ngates=28, s1_ceiling=10000, orb="23o22e", s2_ceiling=10000, gate=100),
          "70q": dict(ngates=44, s1_ceiling=20000, orb="35o34e", s2_ceiling=50000, gate=500)}
COST = ("--grad-clip 8.0 --cost-lambda-excitation 2.0 --cost-lambda-coverage 10.0 --cost-lambda-overshoot 2.0 --gate-loose-mha 50 "
        "--size-curriculum --warmup-frac 0.25 --focus-frac 0.3 --curriculum-top-frac 0.4")

HEAD = f"""#!/usr/bin/env bash
#SBATCH -A {ACCOUNT}
#SBATCH -p {PART}
#SBATCH -J {{name}}
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time={{time}}
#SBATCH -o {EXP}/logs/{{name}}.out
#SBATCH -e {EXP}/logs/{{name}}.err
module load cuda/13.2 nvhpc/26.5 hpcx/2.50
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
{{creds}}SP=$({VENV}/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:$LD_LIBRARY_PATH"
mkdir -p {EXP}/out {EXP}/ckpt {EXP}/logs {EXP}/track {EXP}/ibm_jobs
cd {REPO}
"""
CREDS = "set -a\nsource /path/to/sqd-gqe/.env.local\nset +a\n"


def mols(paths):
    return " ".join(f"molecules/{p}.yaml" for p in paths)


def stage1(lin, tag):
    t = TARGET[tag]
    name = f"t2_s1_{lin}_{tag}"
    cmd = (f"PYTHONPATH=. {VENV}/bin/python benchmark/amortized_train.py --molecules {mols(S1_MOLS)} --iters {ITERS} --seed {SEED} "
           f"--ngates {t['ngates']} --num-samples 20 --recover-iterations 3 --diag-backend sbd --sbd-binary-path {SBD} --sbd-timeout-s 1500 "
           f"--max-dim 20000 --encoder {ENC[lin]} --max-vocab {VOCAB} --depth-ceiling-max {t['s1_ceiling']} --gate-energy-mha 100 {COST} "
           f"--track-circuits {EXP}/track/{name}.jsonl --out {EXP}/out/{name}.json --ckpt {EXP}/ckpt/{name}.pt\n")
    return name, HEAD.format(name=name, time="72:00:00", creds="") + cmd


def stage2(lin, tag):
    t = TARGET[tag]
    s1 = f"t2_s1_{lin}_{tag}"
    name = f"t2_s2_{lin}_{tag}"
    d = [f"dblock/{m}_{t['orb']}_mp2no_def2svp" for m in ("fe_cn6", "co_nh3_6", "ferrocene")]
    pre = (f"{VENV}/bin/python - <<'PY'\nimport sys, torch\nck = torch.load('{EXP}/ckpt/{s1}_best.pt', map_location='cpu', weights_only=False)\n"
           f"sd = ck.get('state_dict', ck)\nv = int(sd['transformer.wte.weight'].shape[0])\n"
           f"print('pre-flight: Stage-1 checkpoint vocabulary', v, '; prompt encoder state present:', ck.get('prompt_encoder_state_dict') is not None)\n"
           f"sys.exit(0 if v == {VOCAB} and ck.get('prompt_encoder_state_dict') is not None else 1)\nPY\n"
           "[ $? -eq 0 ] || { echo 'pre-flight failed: not starting Stage 2'; exit 1; }\n")
    cmd = (f"PYTHONPATH=. {VENV}/bin/python benchmark/amortized_train.py --molecules {mols(d)} --iters {ITERS} --seed {SEED} "
           f"--ngates {t['ngates']} --num-samples 20 --recover-iterations 3 --diag-backend sbd --sbd-binary-path {SBD} --sbd-timeout-s 1500 "
           f"--max-dim 40000 --encoder {ENC[lin]} --max-vocab {VOCAB} --warmstart {EXP}/ckpt/{s1}_best.pt --warmstart-encoder "
           f"--depth-ceiling-max {t['s2_ceiling']} --gate-energy-mha {t['gate']} {COST} "
           f"--sampler-kind ibm_hardware --ibm-backend-name ibm_kobe --shots 10000 --ibm-save-dir {EXP}/ibm_jobs/{name} "
           f"--track-circuits {EXP}/track/{name}.jsonl --out {EXP}/out/{name}.json --ckpt {EXP}/ckpt/{name}.pt\n")
    return name, HEAD.format(name=name, time="48:00:00", creds=CREDS) + pre + cmd


def main(outdir):
    out = Path(outdir)
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    sub = ["#!/usr/bin/env bash", f"mkdir -p {EXP}/out {EXP}/ckpt {EXP}/logs {EXP}/track {EXP}/ibm_jobs", "set -e", f": > {EXP}/submitted.txt"]
    for lin in ENC:
        for tag in TARGET:
            n1, b1 = stage1(lin, tag)
            n2, b2 = stage2(lin, tag)
            (out / "jobs" / f"{n1}.sbatch").write_text(b1)
            (out / "jobs" / f"{n2}.sbatch").write_text(b2)
            sub.append(f"J1=$(sbatch --parsable {EXP}/jobs/{n1}.sbatch); echo \"{n1} $J1\" >> {EXP}/submitted.txt")
            sub.append(f"J2=$(sbatch --parsable --dependency=afterok:$J1 {EXP}/jobs/{n2}.sbatch); echo \"{n2} $J2 (after $J1)\" >> {EXP}/submitted.txt")
    sub.append(f"cat {EXP}/submitted.txt")
    (out / "submit_all.sh").write_text("\n".join(sub) + "\n")
    print(f"wrote {len(list((out / 'jobs').glob('*.sbatch')))} job scripts and submit_all.sh to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t2_jobs")
