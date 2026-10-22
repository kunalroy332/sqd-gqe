# sqd-gqe

> [!NOTE]
> This repository is under active development and the code here should not be considered stable. This is not an officially supported IBM Quantum software.

Sample-based quantum diagonalization (SQD) needs good circuits to sample the electronic configurations that matter, and fixed ansätze such as LUCJ become deep as active spaces grow. `sqd-gqe` trains a transformer policy, a generative quantum eigensolver (GQE), that writes the state-preparation circuit token by token from a pool of particle-number-conserving operators, conditioned on the molecule. The policy is trained first on a noiseless simulator ([ffsim](https://github.com/qiskit-community/ffsim)) and then on an IBM quantum processor, with a reward built from the SQD energy and circuit cost. At inference the trained policy proposes circuits for molecules it has not seen, which are sampled on hardware and diagonalized with [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) and a GPU configuration-recovery step. We use it on 32- to 70-qubit active spaces of tin-organometallic molecules, against DMRG references.

## Publication

*Sample-Based Quantum Diagonalization with a Generative Quantum Eigensolver Scaled Up to 70 Qubits*, K. Kumar *et al.* (manuscript in preparation; a DOI or arXiv reference will be added here once it is available).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10 or newer is required; a GPU is recommended for training. Running on IBM hardware needs credentials: copy `.env.example` to `.env.local` and fill it in (never commit it). See [`tutorial/03_environment_setup.md`](tutorial/03_environment_setup.md) for cluster setup.

## Usage

```bash
python train_sqd.py                                   # train the policy (configuration in configs_sqd/)
python -m sqd_gqe.run --molecule molecules/small/h2o.yaml --backend ideal   # circuit and chemistry report (small molecule, simulator)
pytest -q                                             # unit tests
```

The notebooks in [`tutorial/`](tutorial) walk through training and inference on the simulator and on IBM hardware. The scripts that regenerate the inference analysis are in [`scripts/analysis/`](scripts/analysis) (see [`scripts/analysis/training2/REPRODUCE.md`](scripts/analysis/training2/REPRODUCE.md)); Slurm templates are in [`slurm/`](slurm) and use placeholders (`<SLURM_ACCOUNT>`, `/path/to/...`) that you must replace.

## Repository layout

| Path | Contents |
|---|---|
| `sqd_gqe/` | Library: policy and encoders, operator pool, ffsim sampler, SQD pipeline, IBM hardware sampler, recovery and refinement |
| `configs_sqd/` | Hydra configuration for training |
| `molecules/` | Molecule and active-space definitions (YAML and FCIDUMP) |
| `benchmark/` | Amortized multi-molecule trainer and analysis tools |
| `scripts/analysis/` | Inference and circuit-analysis pipeline |
| `slurm/`, `apptainer.def` | Cluster job templates and container definition |
| `test/`, `tests/` | Unit tests |

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Released under the [Apache-2.0 license](LICENSE).
