# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Per-run job chain of the Training-2 inference campaign, mirroring missing_runs/submit_missing_runs.sh but with INFER_ROOT set to the arm's own tree:
    ca_<arm>_<tag>   re-sample the 10 stored sequences on ibm_kobe, 100,000 shots each (resample_raw_counts.py)  <- the ONLY hardware step, one Runtime job
    rr_<arm>_<tag>   40000-dimensional ranking energy of every circuit (rerank_circuits.py)
    fin_<arm>_<tag>  selection + the five recovery jobs (select_new.py, rerecover_target.py)
Usage: python3 make_chain.py <outdir> --arm hf|tuned --tags <tag> [<tag> ...]   -> writes the sbatch files and <outdir>/submit_<arm>.sh (hardware steps NOT submitted)"""
import argparse
from pathlib import Path

R = "/path/to/sqd-gqe-work"
V = "/path/to/sqd-gqe/.venv"
ROOT = {"hf": "inference_hf", "tuned": "inference_tuned", "t1": "inference_t1", "hf_seed2": "inference_hf_seed2", "tuned_seed2": "inference_tuned_seed2",
        "hf_s8765": "inference_hf_s8765", "tuned_s8765": "inference_tuned_s8765"}
COMMON = f"""module load cuda/13.2 nvhpc/26.5 hpcx/2.50
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
set -a
source /path/to/sqd-gqe/.env.local
set +a
SP=$({V}/bin/python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:$LD_LIBRARY_PATH"
cd {R}
"""


def head(name, gpus, tasks, cpus, hours):
    return f"""#!/usr/bin/env bash
#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH -J {name}
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks-per-node={tasks}
#SBATCH -N 1
#SBATCH -c {cpus}
#SBATCH --time={hours}:00:00
#SBATCH -o {R}/experiments/inference_logs/{name}.out
#SBATCH -e {R}/experiments/inference_logs/{name}.err
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--arm", required=True, choices=list(ROOT)); ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = f"{R}/experiments/{ROOT[a.arm]}"; sub = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {R}"]
    for tag in a.tags:
        env = f"export INFER_ROOT={root}\nexport PYTHONPATH=.\n"
        ca = head(f"ca_{a.arm}_{tag}", 1, 1, 8, 24) + COMMON + env + f"{V}/bin/python scripts/analysis/circuit_analysis/resample_raw_counts.py --cache {root}/cache/crossnorb70q_{tag}_sample.json\n"
        rr = head(f"rr_{a.arm}_{tag}", 2, 2, 16, 24) + COMMON + env + f"{V}/bin/python scripts/analysis/circuit_analysis/rerank_circuits.py --tag {tag}\n"
        fin = head(f"fin_{a.arm}_{tag}", 1, 1, 4, 1) + COMMON + env + f"""W={root}/work/{tag}
mkdir -p $W/rerank_in
cp {root}/circuit_analysis_rerank/{tag}.json $W/rerank_in/
{V}/bin/python scripts/analysis/circuit_analysis/select_new.py --rerank-dir $W/rerank_in --out $W/selection.json --write-sbatch $W/rc
for f in $(cat $W/rc/SUBMIT_ORDER.txt); do
  sed -i 's#/experiments/production_final/logs/#/experiments/inference_logs/#; s/^#SBATCH --time=.*/#SBATCH --time=36:00:00/; s/^#SBATCH --time-min=.*/#SBATCH --time-min=36:00:00/' $W/rc/$f
  sed -i 's#^PYTHONPATH=\\. #INFER_ROOT={root} PYTHONPATH=. #' $W/rc/$f
  sbatch $W/rc/$f
done
"""
        for n, body in ((f"ca_{a.arm}_{tag}", ca), (f"rr_{a.arm}_{tag}", rr), (f"fin_{a.arm}_{tag}", fin)):
            (out / f"{n}.sbatch").write_text(body)
        sub += [f"J1=$(sbatch --parsable {R}/experiments/inference_logs/chain_{a.arm}/ca_{a.arm}_{tag}.sbatch)",
                f"J2=$(sbatch --parsable --dependency=afterok:$J1 {R}/experiments/inference_logs/chain_{a.arm}/rr_{a.arm}_{tag}.sbatch)",
                f"J3=$(sbatch --parsable --dependency=afterok:$J2 {R}/experiments/inference_logs/chain_{a.arm}/fin_{a.arm}_{tag}.sbatch)",
                f'echo "{a.arm} {tag}: ca=$J1 rr=$J2 fin=$J3"']
    (out / f"submit_{a.arm}.sh").write_text("\n".join(sub) + "\n"); print("wrote chains for", len(a.tags), "runs")


if __name__ == "__main__":
    main()
