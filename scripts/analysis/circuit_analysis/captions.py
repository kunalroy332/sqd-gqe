# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Captions for make_tables.py (kept separate so wording edits do not touch the table logic)."""

SEE = (r"Ranking, selection, shot counts and recovery settings are as in Table~\ref{tab:zeroshot-master}, except for the E+D picks of the untrained controls: "
       r"these eight runs were sampled in the first campaign, and their E+D circuits were chosen with the energy rank of the earlier ranking, which used a cap of "
       r"2{,}000 determinants instead of 40{,}000, added to the rank by distinct bitstrings. Their best, second and merged results use the 40{,}000-determinant ranking.")

PROTOCOL = (
    r"All values come from the fresh-shot re-ranking campaign: every run's ten sampled circuits were re-sampled "
    r"on \code{ibm\_kobe} with 100{,}000 new shots each (Qiskit optimization level 1, no error mitigation), and the "
    r"raw counts and transpiled circuits were stored. Circuits are ranked by a one-shot diagonalization of the "
    r"electron-number-conserving shots in a subspace of up to 200 most-measured alpha strings (net CI dimension "
    r"40{,}000); circuits with no valid shot have no ranking energy and are excluded. ``Best''/``second'' are the two "
    r"lowest-ranking-energy circuits, ``merged'' pools the shots of every circuit with a finite ranking energy, and "
    r"``E+D 1''/``E+D 2'' are the two circuits with the lowest sum of two ranks, the rank of the ranking energy (the same "
    r"40{,}000-determinant ranking) and the rank by number of distinct measured bitstrings, recovered on the same fresh shots. "
    r"Every recovery uses $D=20{,}000$ strings per spin sector and 25 iterations. Errors are relative to the DMRG reference."
)

PENDING = (
    r" A dagger marks a partial value, an upper bound from the iterations completed before the job was interrupted; ``pend.'' (in red) marks a result that is not available yet: a recovery still running or a planned run of the HF-penalty campaign (44 runs: all eleven tiers on all four lineages) not yet finished. \textcolor{red}{These entries are updated when the jobs finish; until then the HF-penalty numbers of this table are those of the finished runs only, and no number in it comes from the Training~1 scheme.}"
)

CAPTIONS = {
    "pending": PENDING,
    "master": (
        r"Zero-shot inference comprehensive summary, trained policies only, every lineage$\times$molecule combination. "
        r"Each entry is energy (E$_h$)/deviation vs.\ DMRG (mHa). Depth/2q are the depth and two-qubit-gate count of the best circuit as transpiled and submitted to "
        r"\code{ibm\_kobe} (live transpilation at Qiskit optimization level 1); the offline estimate on the \code{ibm\_heron\_r2} basis is not used. "
        r"The distance of the generated circuits from the idealised LUCJ circuit is in Section~\ref{sec:divergence}. "
        r"Matched untrained controls are in Appendix~\ref{sec:appendix-inference-detail}. " + PROTOCOL
    ),
    "dmrg": (
        r"Lowest recovered SQD energy vs.\ DMRG reference for every SnOH3-me-family target, with the signed deviation $\Delta E = E_{\mathrm{SQD}} - E_{\mathrm{DMRG}}$ (all positive), taken over all lineages, "
        r"policies and result types (best, second, merged, E+D 1/2) in Tables~\ref{tab:zeroshot-detail} and "
        r"\ref{tab:crossnorb-detail}; the lineage and type that achieved it are noted. DMRG references are from "
        r"Table~\ref{tab:dmrg-reference} (bond dimension 4000, 17 sweeps). Selecting the minimum over many results "
        r"is a best-of-$k$ statistic, not a single pre-declared run."
    ),
    "detail_L": (
        r"Zero-shot inference on the SnOH3-me L tier (46 qubits), per lineage and policy. Depth and 2q are the as-submitted (live) "
        r"cost of the best/second circuit; ``valid shots'' is the number of the 100{,}000 shots with the correct electron "
        r"numbers, i.e.\ the data behind each pick. ``SI'' (singles-injected) adds the single-excitation configurations to "
        r"the best circuit's recovered determinant set and re-diagonalizes. DMRG reference $-479.314216$~E$_h$. "
        + SEE
    ),
    "exc_L": (
        r"Recovered energy and excitation-level composition of the recovered determinant set for the SnOH3-me L tier, per "
        r"lineage/policy and circuit type. \textbf{Caution:} S/D/T/Q/higher are counts of determinant \emph{pairs} "
        r"$(a_i,a_j)$ in the Cartesian product of the recovered string list with itself (the same list serves the alpha and the beta sector of these closed-shell systems; "
        r"Section~\ref{sec:recovery}), not counts out of the 100{,}000 shots. With $n$ recovered strings the product has $n^2$ pairs, and the five columns sum to $n^2-1$ in every row; $n=20{,}000$ ($4\times10^{8}$ pairs) where the cap was reached; the per-shot picture is "
        r"Table~\ref{tab:raw-shot-excitation}. ``Higher'' absorbs every order above quadruple. " + SEE
    ),
    "detail_X": (
        r"Cross-norb ablation and remaining matched-norb targets, per target, lineage and policy: SnOH3-me S (32q), LL (56q), "
        r"SnOH3-me-h2o S/M/LL (38/46/66q) and SnOH3-me-2h2o S/M/L (44/58/70q), with DMRG references from "
        r"Table~\ref{tab:dmrg-reference}. Same columns as Table~\ref{tab:zeroshot-detail}. Where valid shots are few "
        r"(tens to hundreds of 100{,}000) the ranking energy and the pick rest on very little data. Wall times of the "
        r"first campaign are not repeated here. " + SEE
    ),
    "exc_X": (
        r"Recovered energy and pair-counted excitation composition for the targets of Table~\ref{tab:crossnorb-detail}; same "
        r"pair-counting convention and caution as Table~\ref{tab:zeroshot-excitation}. " + SEE
    ),
    "raw": (
        r"Raw-shot excitation-level composition (HF/S/D/T/Q/higher) of the sampled bitstrings, per lineage/policy, for the "
        r"best and second circuit and for the merged pool. Unlike Tables~\ref{tab:zeroshot-excitation} and "
        r"\ref{tab:crossnorb-excitation}, counts are over the measured shots themselves (\code{raw\_shot\_excitation\_histogram}, "
        r"weighted by multiplicity), so each best/second row sums to 100{,}000 (``n''); the merged row sums to the pooled "
        r"shots of all finite-energy circuits. " + SEE
    ),
    "coverage": (
        r"Qubit coverage of the fresh-shot sampled circuits, per run, over all ten circuits of each of the @NRUN@ runs (@NCIRC@ circuits). "
        r"Each generated circuit was translated to \{rz, sx, cx\} without a coupling map and the qubits carrying at least one gate "
        r"(or at least one two-qubit gate) were counted; ``idle'' means no gate at all after translation. @NBAD@ circuits leave at least one qubit idle. "
        r"The earlier \code{active\_qubits} field stored with each circuit counts the qubits named by any instruction of the pre-transpile circuit, "
        r"including composite gates that translate to nothing on some qubits; it equals the register width for every circuit and so does not detect these idle qubits. "
        r"Targets of a different norb than the lineage's training norb are the cross-norb ablation of Section~\ref{sec:crossnorb}."
    ),
}
