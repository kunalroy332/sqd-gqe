# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Plot the held-out --dynamic-resources verification: ResourceHead's predicted
starting max_dim vs the dim actually used after run_adaptive's doubling loop
corrects it, ideal vs a noisy backend, on a molecule outside the sweep.

Numbers are hardcoded from the benzene.yaml --dynamic-resources runs (see
session log) since sqd_gqe.run doesn't emit a machine-readable predicted-dim
field today -- this is a small, one-off verification plot, not a sweep.

Output: docs/2026-07-30/figures/dynamic_resources_benzene.png
"""

from __future__ import annotations

from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "docs" / "2026-07-30" / "figures"

# from `python -m sqd_gqe.run --molecule molecules/large/benzene.yaml
#   --checkpoint full8_policy.pt --dynamic-resources --max-dim-cap 200000`
ROWS = [
    {"backend": "ideal", "predicted_dim": 14893, "dim_used": 29786, "err_mha": 0.0000,
     "depth": 202, "two_q": 392},
    {"backend": "ibm_heron_r2_noisy_sim", "predicted_dim": 2630, "dim_used": 5260, "err_mha": 0.0000,
     "depth": 141, "two_q": 258},
]


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["backend"] for r in ROWS]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    ax[0].bar(x - width / 2, [r["predicted_dim"] for r in ROWS], width,
              label="ResourceHead predicted (start)", color="#4C78A8")
    ax[0].bar(x + width / 2, [r["dim_used"] for r in ROWS], width,
              label="actual dim used (after run_adaptive)", color="#F58518")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=15, ha="right")
    ax[0].set_ylabel("SQD subspace dim")
    ax[0].set_title("benzene.yaml (held-out, zero-shot): predicted vs. actual dim")
    ax[0].legend(fontsize=8)
    for i, r in enumerate(ROWS):
        ax[0].annotate(f"err={r['err_mha']:.4f}mHa", (i, max(r["predicted_dim"], r["dim_used"]) * 1.05),
                       ha="center", fontsize=8)

    ax[1].bar(x - width / 2, [r["depth"] for r in ROWS], width, label="depth", color="#54A24B")
    ax[1].bar(x + width / 2, [r["two_q"] for r in ROWS], width, label="2q-gates", color="#B279A2")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=15, ha="right")
    ax[1].set_title("Device-transpiled circuit cost")
    ax[1].legend(fontsize=8)

    fig.suptitle("Dynamic-resources inference, both halves working:\n"
                 "prediction differs by backend, adaptive doubling corrects it, both reach exact CASCI")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "dynamic_resources_benzene.png", dpi=130)
    print(f"wrote {FIG / 'dynamic_resources_benzene.png'}")


if __name__ == "__main__":
    main()
