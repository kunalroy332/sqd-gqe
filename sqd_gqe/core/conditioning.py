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

"""Molecule + backend conditioning for the amortized (generalist) policy.

To generalize across molecules without retraining, the transformer policy is
conditioned on a fixed-length descriptor of (molecule, backend). We build that
descriptor from cheap, transferable features and project it to the model's
embedding dimension so it can be injected as a prefix/conditioning token.

Features are deliberately size-invariant (ratios, per-orbital summaries,
reference energies) so a policy trained on small molecules transfers to larger
ones. Nothing here depends on the quantum backend library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def molecule_features(system, backend=None) -> np.ndarray:
    """A compact, size-invariant descriptor of a molecular system (+ backend).

    Returns a 1-D float32 feature vector. All features are intensive or
    normalized so they transfer across system sizes.
    """
    norb = int(system.norb)
    na, nb = system.nelec
    nelec = na + nb
    h = system.active_hamiltonian
    h1 = np.asarray(h.one_body, dtype=np.float64)
    h2 = np.asarray(h.two_body, dtype=np.float64)

    # HF orbital energies (Fock eigenvalues) restricted to the active window
    try:
        mo_e = np.asarray(system.hf.mo_energy, dtype=np.float64)
        act = system.active_indices
        eps = mo_e[act]
        homo_lumo = float(eps[na] - eps[na - 1]) if 0 < na < len(eps) else 0.0
        eps_mean, eps_std = float(eps.mean()), float(eps.std())
    except Exception:
        homo_lumo = eps_mean = eps_std = 0.0

    hf_e = float(getattr(system.hf, "e_tot", 0.0))
    t2_norm_per_pair = 0.0
    try:
        ccsd_e = float(system.ccsd_energy())
        corr = ccsd_e - hf_e            # correlation energy (intensive-ish)
        t2 = system.ccsd_amplitudes.get("t2") if system.ccsd_amplitudes else None
        if t2 is not None:
            n_pairs = max(min(na, nb), 1)
            t2_norm_per_pair = float(np.linalg.norm(t2)) / n_pairs
    except Exception:
        ccsd_e = hf_e
        corr = 0.0

    # Problem-size features for the resource (max_dim/shots) predictor: these
    # correlate with how big a subspace SQD needs, independent of whether an
    # exact FCI reference is even computable (see MolecularSystem.fci_dimension
    # / FciIntractableError -- this descriptor is defined for every molecule,
    # including ones where casci_energy() would raise).
    # fci_dimension() is a plain Python int (math.comb) that can exceed int64
    # range well before any real molecule becomes intractable for other
    # reasons -- confirmed at norb=35 (fe_cn6_35o34e: dim~2.06e19 > 2**63),
    # where np.log10 on the raw int raises a TypeError (numpy's ufunc loop
    # can't cast an int this large). float() first loses no precision that
    # matters going into log10.
    log_fci_dim = float(np.log10(max(float(system.fci_dimension()), 1.0)))
    virt_fraction = 1.0 - nelec / max(2 * norb, 1)

    feats = [
        norb / 32.0,                       # size (scaled)
        nelec / 64.0,
        (na - nb),                         # spin polarization
        float(system.spin),
        na / max(norb, 1),                 # alpha filling fraction
        nb / max(norb, 1),
        float(np.abs(h1).mean()),          # one-body scale
        float(np.abs(h1).max()),
        float(np.abs(h2).mean()),          # two-body scale
        float(np.abs(h2).max()),
        float(h.core_energy) / 100.0,
        corr / max(abs(hf_e), 1.0),        # relative correlation strength
        homo_lumo,
        eps_mean, eps_std,
        log_fci_dim / 12.0,                # problem-size proxy (scaled log10 FCI dim)
        t2_norm_per_pair,                  # correlation-entropy proxy
        virt_fraction,                     # fraction of active space that's virtual
    ]

    # backend descriptor, incl. a sim_kind one-hot (ideal / noisy_sim / real) so
    # the policy can tell "these numbers are a live device" from "these numbers
    # are meant to be run through a simulated noise model" -- see
    # hw.backends.HardwareBackend.sim_kind.
    if backend is not None:
        sim_kind = getattr(backend, "sim_kind", "real")
        one_hot = [1.0 if sim_kind == k else 0.0 for k in ("ideal", "noisy_sim", "real")]
        feats += [
            backend.n_qubits / 256.0,
            float(len(backend.basis_gates)) / 8.0,
            np.log10(max(backend.err_1q, 1e-6)),
            np.log10(max(backend.err_2q, 1e-6)),
            np.log10(max(backend.err_readout, 1e-6)),
            backend.depth_budget / 10000.0,
            *one_hot,
        ]
    else:
        feats += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return np.asarray(feats, dtype=np.float32)


FEATURE_DIM = 27  # len(molecule_features(...))


class ConditioningEncoder(nn.Module):
    """MLP that maps the molecule+backend descriptor to an embedding-sized vector,
    used as a conditioning/prefix token for the transformer policy."""

    def __init__(self, embed_dim: int, feature_dim: int = FEATURE_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class ResourceHead(nn.Module):
    """Predicts (max_dim, shots) from the same size-invariant descriptor used
    by ConditioningEncoder -- a sibling model, not a branch of the GPT-2 token
    path. Dimension/shots are scalar post-hoc resource choices for the SQD
    diagonalization step, not circuit structure, so they don't belong in the
    policy's embedding bias; this reuses the descriptor without perturbing the
    token embeddings GPT-2 conditions on. Trained by plain regression on
    labeled sweep data (see benchmark/build_resource_dataset.py), independent
    of the GRPO/RL loop -- see SqdPipeline.solve_adaptive for the runtime
    robustness net around whatever this predicts.

    Outputs (log10 max_dim, log10 shots); use predict() for the exponentiated,
    clipped integers actually passed to the pipeline.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 64,
                 log_dim_range: tuple[float, float] = (2.5, 6.5),
                 log_shots_range: tuple[float, float] = (3.5, 6.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.log_dim_range = log_dim_range
        self.log_shots_range = log_shots_range

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

    @torch.no_grad()
    def predict(self, feats: torch.Tensor) -> tuple[int, int]:
        out = self.forward(feats.float().unsqueeze(0) if feats.dim() == 1 else feats.float())
        log_dim, log_shots = out[0, 0].item(), out[0, 1].item()
        log_dim = min(max(log_dim, self.log_dim_range[0]), self.log_dim_range[1])
        log_shots = min(max(log_shots, self.log_shots_range[0]), self.log_shots_range[1])
        return int(round(10 ** log_dim)), int(round(10 ** log_shots))

    @classmethod
    def load(cls, path: str, feature_dim: int = FEATURE_DIM) -> "ResourceHead":
        head = cls(feature_dim=feature_dim)
        state = torch.load(path, map_location="cpu", weights_only=False)
        head.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        head.eval()
        return head


@dataclass
class ConditioningContext:
    """Bundles the (molecule, backend) descriptor for a run."""
    features: np.ndarray

    @classmethod
    def build(cls, system, backend=None) -> "ConditioningContext":
        return cls(features=molecule_features(system, backend))

    def tensor(self, device=None) -> torch.Tensor:
        t = torch.from_numpy(self.features)
        return t.to(device) if device is not None else t
