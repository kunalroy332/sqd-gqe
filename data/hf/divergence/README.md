# Ideal divergence of the generated circuits from the LUCJ circuit

Produced by `scripts/analysis/circuit_analysis/circuit_divergence.py` (noiseless, exact ffsim statevectors in the full N-electron space; no hardware).

For a generated token sequence g the policy defines |psi_g> = U_g|HF> (U_g = product of the pool operators, ids >= n_real are identity padding). The reference is
|phi> = U_LUCJ|HF>, ffsim `UCJOpSpinBalanced.from_t_amplitudes(t2, t1, n_reps=1, interaction_pairs=([(p,p+1)], [(p,p)]))`, the circuit of the LUCJ hardware baseline.
p = |psi_g|^2 and q = |phi|^2 are the determinant probability distributions.

| key | definition |
|---|---|
| `F` | state fidelity \|<psi_g\|phi>\|^2 |
| `BC`, `H` | Bhattacharyya coefficient sum sqrt(p q); Hellinger distance sqrt(1 - BC) (0 = identical distributions, 1 = disjoint support) |
| `JS` | Jensen-Shannon divergence in bits, H2(m) - (H2(p)+H2(q))/2, m = (p+q)/2; in [0,1] and finite for disjoint support (KL is not reported: it is infinite when supports differ) |
| `d_HF` | sqrt(2 - 2\|<psi_g\|phi>\|), distance of the two states up to a global phase; a lower bound of the operator distance min_alpha \|\|U_g - e^{i alpha} U_LUCJ\|\| |
| `d_HF_raw` | sqrt(2 - 2 Re<psi_g\|phi>), without the phase minimisation |
| `exc_gen` | ideal excitation-rank distribution (HF,S,D,T,Q,higher) of p; the LUCJ one is in `lucj_ideal_<molecule>.json` (`exc_lucj`) |
| `d_HS` | normalised Frobenius (Hilbert-Schmidt) distance \|\|U_g - U_LUCJ\|\|_F / sqrt(D) by Hutchinson trace estimation (3 random vectors; `d_HS_std`); sqrt(2) = 1.414 for unrelated unitaries |
| `op_norm`, `op_norm_info` | lower bound of the operator norm \|\|U_g - U_LUCJ\|\|_op (maximum 2) from Lanczos on (U-V)^dag(U-V) with exact inverses of every pool operator; `residual_top_ritz`, `lanczos_steps`, `history` document the convergence. Only for the best and second circuit of the 16-orbital tier. |

Only tiers whose full N-electron vector fits in memory are computed (16, 19 and 22 orbitals); larger tiers are not available (n/a).
Validation on a toy system (4 orbitals, 2+2 electrons, dense matrices): max inverse error 6e-15, executor = pool op to 0.0, Lanczos 1.99872 vs dense `np.linalg.norm(U-V, 2)` 1.99872.
