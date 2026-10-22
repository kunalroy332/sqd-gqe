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

"""Train ResourceHead by plain regression on benchmark/build_resource_dataset.py's
labeled (molecule, backend) -> (max_dim, shots) dataset.

Independent of the GRPO/RL loop: this is a small MLP fit with ordinary MSE,
fast enough to run on CPU. See core/conditioning.py's ResourceHead docstring
for why this is a sibling model rather than a branch of the GPT-2 policy.

Output: experiments/amortized/resource_head.pt + resource_head_fit.png
(predicted vs. actual log10(dim)/log10(shots), leave-one-molecule-out).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(EXP / "resource_dataset.json"))
    ap.add_argument("--out", default=str(EXP / "resource_head.pt"))
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sqd_gqe.core.conditioning import ResourceHead, FEATURE_DIM

    rows = json.loads(Path(args.dataset).read_text())
    if len(rows) < 3:
        raise SystemExit(f"need >=3 molecules in {args.dataset}, got {len(rows)}")

    X = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.asarray([[np.log10(max(r["label_dim"], 1)), np.log10(max(r["label_shots"], 1))] for r in rows],
                   dtype=np.float32)
    names = [r["molecule"] for r in rows]
    assert X.shape[1] == FEATURE_DIM, f"dataset feature_dim={X.shape[1]} != current FEATURE_DIM={FEATURE_DIM}"

    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    head = ResourceHead(feature_dim=FEATURE_DIM)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        opt.zero_grad()
        pred = head(Xt)
        loss = torch.mean((pred - yt) ** 2)
        loss.backward()
        opt.step()
        if epoch % 200 == 0 or epoch == args.epochs - 1:
            print(f"[TRAIN-RESOURCE-HEAD] epoch={epoch:5d} mse={float(loss):.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "molecules": names, "feature_dim": FEATURE_DIM}, args.out)
    print(f"saved -> {args.out}")

    # leave-one-out fit quality (dataset is small; a held-in fit would just
    # show memorization) -- refit N times, each time holding out one molecule.
    loo_pred = np.zeros_like(y)
    for i in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool); mask[i] = False
        h = ResourceHead(feature_dim=FEATURE_DIM)
        o = torch.optim.Adam(h.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            o.zero_grad()
            loss = torch.mean((h(Xt[mask]) - yt[mask]) ** 2)
            loss.backward(); o.step()
        with torch.no_grad():
            loo_pred[i] = h(Xt[i:i + 1]).numpy()[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, (label, ax_) in enumerate(zip(["log10(max_dim)", "log10(shots)"], ax)):
        ax_.scatter(y[:, j], loo_pred[:, j], color="#0F62FE")
        lo, hi = y[:, j].min() - 0.3, y[:, j].max() + 0.3
        ax_.plot([lo, hi], [lo, hi], "--", color="gray")
        for i, n in enumerate(names):
            ax_.annotate(n, (y[i, j], loo_pred[i, j]), fontsize=7)
        ax_.set_xlabel(f"actual {label}"); ax_.set_ylabel(f"predicted {label} (leave-one-out)")
        ax_.grid(alpha=.3)
    fig.suptitle("ResourceHead: predicted vs. actual (leave-one-molecule-out)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(EXP / "resource_head_fit.png", dpi=130); plt.close(fig)
    print(f"wrote {EXP/'resource_head_fit.png'}")


if __name__ == "__main__":
    main()
