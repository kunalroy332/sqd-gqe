# Molecule library

Tiered molecule set for training/evaluating the amortized (generalist) SQD-GQE
policy. Drop a new `MolecularSystem` YAML into the right tier and it becomes
available to the trainer and the point-and-run CLI.

All entries are **RHF singlets** for now. The `MolecularSystem` schema and the
pipeline are structured so a UHF/ROHF path can be added later (open-shell
systems like the Fe–S clusters genuinely need it — see note on `large/fe2s2`).

## Tiers
- **small/** (≤16 qubits, training core): H2, LiH, H2O, NH3, BeH2, N2, HF, CH4.
- **medium/** (industrially relevant, ~20–28 qubits): CO2, C2H4, HCN, CH2O, CH3OH.
- **large/** (published large-scale systems, held out for zero-shot transfer):
  benzene (π active space, 12q), [Fe2S2] core (40q stretch target).

## Training vs evaluation split
- Train the generalist policy on **small + medium**.
- Hold out **large/** entirely for **zero-shot transfer** tests (no retraining).

## Adding a molecule
Copy an existing YAML, set the geometry (`custom` xyz or `linear_chain`),
`basis`, `nelecas`, `norbcas` (qubits = 2·norbcas), `spin`, `charge`. Then:

    sqd-gqe run --molecule molecules/medium/co2.yaml --backend ibm_heron_r2
