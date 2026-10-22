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

"""Prompt encoders for the cross-attention conditioned policy.

Every encoder here satisfies one contract:

    encoder(system, backend) -> Tensor (L, n_embd)      # L is free per encoder

``CrossAttnPolicy.set_prompt`` takes that tensor directly, so encoders can be
swapped without touching the policy or the training loop. This is the point of
the cross-attention route: the additive path (``ConditioningEncoder``) forces
L == 1, which means any encoder with a natural per-orbital or per-atom output has
to be pooled first, and a measured difference then conflates "which encoder" with
"how it was pooled".

Encoders implemented here:

* ``NullPromptEncoder``      (A0)        -- L=1, no molecular information at all
* ``ScalarPromptEncoder``    (A1-xattn)  -- L=1, the existing 27-dim descriptor
* ``OrbitalSetPromptEncoder``(A2)        -- L=norb, one token per active orbital
* ``PosEgnnPromptEncoder``   (A5)        -- L=n_atoms+1, POS-EGNN + descriptor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sqd_gqe.core.conditioning import FEATURE_DIM, molecule_features


class NullPromptEncoder(nn.Module):
    """A0: one learned vector, identical for every molecule.

    The lower-bound arm. Deliberately *learnable* rather than a frozen constant:

    * A zero vector would make the cross-attention Key/Value all-zero, which
      degenerates the attention denominator -- numerically fragile.
    * A frozen random vector would make the baseline seed-dependent, so the
      "floor" would not reproduce across runs.
    * A learned vector still carries zero molecular information, so it fulfils
      the baseline's role, and it answers the stricter question: what is the best
      energy reachable *without* knowing which molecule this is?

    Initialised at the same scale as the token embeddings (std=0.02) and fixed by
    ``--seed``, so it reproduces.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.null_prompt = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def forward(self, system=None, backend=None) -> torch.Tensor:
        return self.null_prompt          # (1, E) -- independent of the molecule


class NormScalarPromptEncoder(nn.Module):
    """A1'': the 27-dim descriptor, LayerNormed like A2's, then projected to one token.

    Exists to separate two things the A1-vs-A2 comparison currently conflates.
    ``ScalarPromptEncoder`` feeds the raw descriptor to its first Linear, while
    ``OrbitalSetPromptEncoder`` LayerNorms first -- so a measured difference between
    the fixed-length and variable-length arms carries both "prompt length" and
    "input normalisation", and neither can be attributed on its own.

    That the missing normalisation actually matters here was measured on the four
    evaluation molecules (ch2o/hcn/ch3oh/c2h4): the descriptor's largest column is
    7.17 against a median of 0.056 (129x), its top two columns hold 93% of the
    squared norm, and only 16 of the 27 columns vary between molecules at all. So
    the first Linear is dominated by one or two columns and the rest land below the
    downstream GELUs' resolution. The eps repair (A1') makes this worse rather than
    better -- restoring eps_mean from 0.0 to -5.49 raises the top-two share to 94%.

    Everything except the LayerNorm is identical to ``ScalarPromptEncoder`` (same
    features, same 3-layer shape, same hidden width), so A1'' minus A1-xattn is the
    normalisation and nothing else.
    """

    def __init__(self, embed_dim: int, feature_dim: int = FEATURE_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (feature_dim,) or (1, feature_dim) -> (1, embed_dim)."""
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self.net(self.norm(feats))


class ScalarPromptEncoder(nn.Module):
    """A1-xattn: the existing 27-dim descriptor, projected to one prompt token.

    Same input features as ``ConditioningEncoder`` (so A1-add and A1-xattn differ
    only in *how* the conditioning reaches the decoder, not in what it contains),
    and the same 3-layer MLP shape, so a difference between the two arms is
    attributable to the injection route rather than to encoder capacity.

    ``bound_output``: optional ``tanh``-based clamp on the final projection,
    ``bound * tanh(x / bound)`` (so small activations pass through ~linearly and
    only large ones get compressed). Off by default -- every existing arm's
    numbers were measured unbounded, so this only changes behaviour when a new
    ``_bounded`` arm variant asks for it. Exists because neither this class nor
    ``PosEgnnPromptEncoder`` has any output normalisation today, so the
    cross-attention K/V the decoder reads can, in principle, take on an
    arbitrarily large scale as training moves the encoder's weights -- this is
    the one piece of the original proposal's "latent-space bounding" that maps
    onto this architecture with no translation needed.
    """

    def __init__(self, embed_dim: int, feature_dim: int = FEATURE_DIM, hidden: int = 128,
                 bound_output: float | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.bound_output = bound_output

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (feature_dim,) or (1, feature_dim) -> (1, embed_dim)."""
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        out = self.net(feats)           # (1, E)
        if self.bound_output is not None:
            out = self.bound_output * torch.tanh(out / self.bound_output)
        return out


def molecule_features_epsfix(system, backend=None) -> np.ndarray:
    """A1': the 27-dim descriptor with its three orbital-energy features repaired.

    ``molecule_features`` reads ``system.hf.mo_energy[system.active_indices]``, but
    ``from_fcidump``-loaded systems set ``active_indices = None`` (the active space
    is not a contiguous window of this HF's canonical MOs). ``mo_e[None]`` does not
    raise -- numpy treats it as ``np.newaxis`` and returns shape ``(1, nmo)`` -- so
    ``len(eps) == 1``, the ``0 < na < len(eps)`` guard is false, ``homo_lumo``
    silently becomes 0.0, and ``eps[na]`` then raises IndexError, which the bare
    ``except`` swallows into all three features being 0.

    Measured on the Sn tiers:

        feature      S(32q) buggy   M(38q) buggy   S fixed      M fixed
        homo_lumo    0.000000       0.000000       0.000099     0.000122
        eps_mean     1.131804       1.131804      -6.276856    -6.158471
        eps_std      6.827642       6.827642       0.661614     0.806305

    All three were identical across tiers, i.e. they carried no size information at
    all -- the exact thing the cross-size task needs them for.

    The fix diagonalises the active-space one-body Hamiltonian instead of indexing
    the full-molecule HF: ``h1`` comes from the FCIDUMP, so it is defined for every
    tier and never routes through ``self.hf``.

    Kept as a separate function rather than a patch to ``molecule_features``: the
    additive arm (A1-add) must keep its current behaviour so that A1-add vs A1' is
    a controlled comparison of the bug's effect.
    """
    feats = molecule_features(system, backend).copy()

    h1 = np.asarray(system.active_hamiltonian.one_body, dtype=np.float64)
    eps = np.linalg.eigh(h1)[0]
    na, _ = system.nelec

    # Indices 12/13/14 are homo_lumo, eps_mean, eps_std -- see the `feats = [...]`
    # literal in molecule_features.
    feats[12] = float(eps[na] - eps[na - 1]) if 0 < na < len(eps) else 0.0
    feats[13] = float(eps.mean())
    feats[14] = float(eps.std())
    return feats.astype(np.float32)


class PromptContext:
    """Holds whatever a prompt encoder needs, computed once per molecule.

    Mirrors ``ConditioningContext`` but stays deliberately separate: that class's
    ``.tensor()`` contract (a 1-D descriptor) is relied on by six call sites in
    benchmark/, and the orbital-set encoder will need a (norb, F) tensor instead.
    Keeping them apart means the additive path is untouched.
    """

    def __init__(self, features: np.ndarray):
        self.features = features

    @classmethod
    def build(cls, system, backend=None, *, epsfix: bool = False) -> "PromptContext":
        """``epsfix=True`` selects the A1' descriptor (see molecule_features_epsfix)."""
        fn = molecule_features_epsfix if epsfix else molecule_features
        return cls(features=fn(system, backend))

    def tensor(self, device=None) -> torch.Tensor:
        t = torch.from_numpy(self.features)
        return t.to(device) if device is not None else t


ORBITAL_FEATURE_DIM = 12


def orbital_features(system) -> np.ndarray:
    """A2: one feature vector per active orbital -> (norb, ORBITAL_FEATURE_DIM).

    The point of this encoder. The 27-dim descriptor compresses the two-electron
    integrals to ``|h2|.mean()`` and ``|h2|.max()`` -- two numbers for what is
    19**4 = 130,321 elements at the M tier -- so "which orbital pairs are strongly
    correlated" is not recoverable from it, and that is exactly what picking
    excitation operators needs. Here each orbital keeps its own row.

    Every entry comes from ``h1``/``h2`` alone, so it is defined at every tier with
    no CCSD call. That matters because the LL tier's CCSD cost is unknown, and a
    descriptor that degrades at one size would break the size-independence claim
    this arm exists to test.

    ``sum_qrs |h2[p,q,r,s]|`` is divided by ``norb**3`` to keep it intensive, in the
    same spirit as the 27-dim descriptor's ``norb/32`` and ``nelec/64`` scalings --
    otherwise the feature would grow with tier size and the encoder would key off
    magnitude rather than structure.

    Orbital energies come from ``eigh(h1)``, not from ``system.hf`` -- see
    ``molecule_features_epsfix`` for why the HF route is unusable here.
    """
    h1 = np.asarray(system.active_hamiltonian.one_body, dtype=np.float64)
    h2 = np.asarray(system.active_hamiltonian.two_body, dtype=np.float64)
    norb = int(system.norb)
    na, _ = system.nelec

    eps = np.linalg.eigh(h1)[0]
    abs_h1 = np.abs(h1)
    exch = np.einsum("pqqp->pq", h2)          # h2[p,q,q,p] -- exchange
    coul = np.einsum("ppqq->pq", h2)          # h2[p,p,q,q] -- Coulomb
    abs_h2_p = np.abs(h2).reshape(norb, -1)   # (norb, norb**3)
    # norb**1.5, not norb**3. Dividing a sum over norb**3 elements by norb**3 is just
    # their mean, and h2 is mostly near-zero, so the fraction of near-zero entries
    # grows with norb and the mean falls with it -- the opposite of intensive.
    # Measured, train (norb 2-8) vs held-out (norb 10,12): /norb**3 lands the
    # held-out molecules at 0.21x the training scale, /norb**1.5 at 0.84x. The
    # exponent is empirical, chosen from that sweep (norb**2 gives 0.57x, no
    # division 1.95x), not from an analytic argument.
    scale = max(norb ** 1.5, 1.0)

    feats = np.stack([
        np.diag(h1),                          # diagonal one-body
        abs_h1.sum(axis=1),                   # total one-body coupling
        abs_h1.max(axis=1),                   # strongest one-body coupling
        (np.arange(norb) < na).astype(np.float64),   # reference occupancy
        eps,                                  # orbital energy (from eigh(h1))
        np.einsum("pppp->p", h2),             # self-Coulomb
        exch.mean(axis=1),
        exch.max(axis=1),
        coul.mean(axis=1),
        coul.max(axis=1),
        abs_h2_p.sum(axis=1) / scale,         # correlation strength (see scale above)
        # Mean of the norb largest |h2| entries for this orbital, not a fixed
        # quantile. A 0.9 quantile over norb**3 entries slides toward zero as norb
        # grows, for the same reason the mean does: measured 0.18x the training
        # scale on the held-out molecules, against 0.87x for this. Taking norb
        # entries keeps the count tied to the orbital rather than to norb**3.
        np.sort(abs_h2_p, axis=1)[:, -norb:].mean(axis=1),
    ], axis=1)
    assert feats.shape == (norb, ORBITAL_FEATURE_DIM), feats.shape
    return feats.astype(np.float32)


class OrbitalSetPromptEncoder(nn.Module):
    """A2: embeds each orbital independently -> prompt of length norb.

    Weights are shared across orbitals, so the same module handles norb=16 (S),
    19 (M) and 23 (L) without any change -- which is the whole reason this arm is
    routed through cross-attention rather than the additive path. The additive path
    would force a pool down to one vector, and a measured difference would then
    conflate "orbital-resolved features" with "how they were pooled".

    No attention pooling, no positional encoding: cross-attention already treats
    the prompt as a set, and adding position would make the output depend on
    orbital ordering, which carries no physical meaning here.
    """

    def __init__(self, embed_dim: int, feature_dim: int = ORBITAL_FEATURE_DIM,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (norb, feature_dim) -> (norb, embed_dim).

        LayerNorm over the feature axis first: the raw columns span very different
        scales (diagonal h1 is O(1) Hartree while the size-normalised h2 sum is
        O(1e-3)), and without it the largest column dominates the first layer.
        """
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self.net(self.norm(feats))


class OrbitalSetTransformerEncoder(nn.Module):
    """A2-large: the same per-orbital features, but orbitals attend to each other.

    ``OrbitalSetPromptEncoder`` applies one shared 3-layer MLP to each orbital
    independently, so an orbital's embedding never sees any other orbital -- the
    prompt is norb tokens computed in isolation. The active space is defined by how
    its orbitals couple (the h2 tensor is the whole point), so that architecture can
    represent "which orbitals are present" but not "how they interact".

    Two things motivate the larger arm concretely. First, GQCO (arXiv:2501.16986)
    conditions its decoder on a 12-layer, 8-head graph-transformer encoder of ~127M
    parameters, against 0.117M here -- three orders of magnitude, and their encoder
    passes messages between nodes while ours does not. Second, in our own 24q
    measurement A2 finished last (post-recovery median +9.48 mHa against +0.04 for
    the existing additive path), so "orbital-resolved conditioning does not help" is
    currently confounded with "this encoder is too small to express the coupling".

    Self-attention, not graph convolution: the orbital set has no canonical adjacency
    to convolve over, and attention over the full set is the natural analogue. Also
    no positional encoding -- cross-attention downstream treats the prompt as a set,
    and orbital index carries no physical meaning, so adding position would make the
    output depend on an arbitrary ordering. Permutation equivariance is therefore
    preserved: permuting the input orbitals permutes the output tokens identically.
    """

    def __init__(self, embed_dim: int, feature_dim: int = ORBITAL_FEATURE_DIM,
                 hidden: int = 256, n_layers: int = 4, n_heads: int = 8,
                 ffn_mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * ffn_mult,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(hidden, embed_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (norb, feature_dim) -> (norb, embed_dim)."""
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        x = self.proj(self.norm(feats)).unsqueeze(0)   # (1, norb, hidden)
        x = self.blocks(x)
        return self.out(x.squeeze(0))                  # (norb, embed_dim)


class OrbitalSetContext:
    """Per-orbital features, computed once per molecule.

    Separate from ``PromptContext`` because the tensor shape differs: (norb, F)
    versus (27,). Keeping them apart means neither the additive path's
    ``ConditioningContext`` nor the scalar prompt path needs to know about this one.
    """

    def __init__(self, features: np.ndarray):
        self.features = features

    @classmethod
    def build(cls, system, backend=None) -> "OrbitalSetContext":
        # backend is accepted for signature parity with the other contexts; the
        # orbital features are a property of the Hamiltonian, not the device.
        return cls(features=orbital_features(system))

    def tensor(self, device=None) -> torch.Tensor:
        t = torch.from_numpy(self.features)
        return t.to(device) if device is not None else t


#: The arm names used in the handover table, mapped to the module names the code and the
#: checkpoints carry. ``baseline`` and ``a1_xattn`` take the same 27-dim descriptor and
#: differ only in how it reaches the decoder (added at every position, versus attended to
#: as a prompt token), which is exactly what ``add`` versus ``scalar27`` failed to say.
#: The original names stay valid.
ARM_ALIASES = {
    "baseline": "add",
    "a1_xattn": "scalar27",
    "a1_xattn_large": "scalar27_large",
    "a1_xattn_norm": "scalar27_norm",
    "a1_xattn_epsfix": "scalar27_epsfix",
    "a2": "orbital_set",
    "a2_large": "orbital_set_large",
    "a2_xl": "orbital_set_xl",
    "a5": "posegnn",
    "a5_large": "posegnn_large",
    "a0": "null",
    # tanh-bounded encoder-output variants of the two focus arms (see
    # ScalarPromptEncoder/PosEgnnPromptEncoder's bound_output) -- separate arm
    # names rather than a flag on the existing ones, so a1_xattn_large/a5_large's
    # already-measured unbounded numbers stay exactly reproducible.
    "a1_xattn_large_bounded": "scalar27_large_bounded",
    "a5_large_bounded": "posegnn_large_bounded",
    # A6: MACE-MP-0 in place of POS-EGNN as the 3D encoder, same descriptor +
    # per-atom-token design as A5 -- see MaceContext.
    "a6_mace": "mace",
    "a6_mace_large": "mace_large",
    "a6_mace_large_bounded": "mace_large_bounded",
}


def canonical_arm(kind: str) -> str:
    """Resolve an arm name to the module name used internally and in checkpoints."""
    return ARM_ALIASES.get(kind, kind)


def build_prompt_encoder(kind: str, embed_dim: int, **kwargs) -> nn.Module:
    """Factory used by the training entry point to pick an arm by name.

    Accepts the arm names from the handover table (``a1_xattn``, ``a2_large``, ``a5``,
    ``baseline``, ``a0``) or the original module names (``scalar27``, ``orbital_set``,
    ``posegnn``, ``add``, ``null``). See :data:`ARM_ALIASES`.
    """
    kind = canonical_arm(kind)
    if kind == "null":
        return NullPromptEncoder(embed_dim)
    if kind in ("scalar27", "scalar27_epsfix"):
        return ScalarPromptEncoder(embed_dim, **kwargs)
    if kind == "scalar27_norm":
        return NormScalarPromptEncoder(embed_dim, **kwargs)
    if kind == "scalar27_large":
        # A1-xattn at A2-large's capacity, so that "A2-large beats the additive
        # baseline" is attributable to the orbital-resolved design rather than to a
        # parameter count. Only A2 had a large variant, which reproduces in reverse
        # the confound that made A2 look worst when it was the small one: at 0.117M
        # against 3.36M the comparison measures capacity, not injection route or
        # prompt shape. hidden=1500 gives 3.446M against orbital_set_large's 3.360M
        # (1.03x) -- the closest this architecture's single width knob reaches. Same
        # descriptor, same 3-layer shape; only the width changes.
        kwargs.setdefault("hidden", 1500)
        return ScalarPromptEncoder(embed_dim, **kwargs)
    if kind == "scalar27_large_bounded":
        # a1_xattn_large's exact capacity (hidden=1500), plus a tanh-bounded
        # output -- see ScalarPromptEncoder.bound_output. bound=3.0 is a few
        # times the token embedding's own init std (0.02 * sqrt(n_embd) is
        # O(0.5-1) in practice), loose enough not to saturate a well-behaved
        # encoder but tight enough to cap a runaway one.
        kwargs.setdefault("hidden", 1500)
        kwargs.setdefault("bound_output", 3.0)
        return ScalarPromptEncoder(embed_dim, **kwargs)
    if kind == "orbital_set":
        return OrbitalSetPromptEncoder(embed_dim, **kwargs)
    if kind == "orbital_set_large":
        return OrbitalSetTransformerEncoder(embed_dim, **kwargs)
    if kind == "orbital_set_xl":
        # Same architecture, one size up: 8 layers x 512 wide instead of 4 x 256.
        # Justified by measurement rather than by hoping bigger is better -- at 3.36M
        # the smallest between-molecule cosine similarity is 0.225 (against 0.900 at
        # 0.117M, alongside a 45x error drop at 24q), but same-size molecules are
        # still unseparated: hcn and ch3oh, both norb=11, sit at 0.99952. Whatever
        # capacity distinguishes molecules of equal active-space size has not been
        # reached yet.
        kwargs.setdefault("hidden", 512)
        kwargs.setdefault("n_layers", 8)
        return OrbitalSetTransformerEncoder(embed_dim, **kwargs)
    if kind == "posegnn_large":
        # Same reasoning for A5: hold capacity at A2-large's level so the comparison
        # reads design rather than parameter count. hidden=1100 gives 3.216M against
        # orbital_set_large's 3.360M (0.96x); 900 was tried first and lands at 2.452M
        # (0.73x), which is near enough to call similar but not near enough to hold
        # capacity fixed. Only the two projection heads widen -- the frozen POS-EGNN
        # embedding is untouched.
        kwargs.setdefault("hidden", 1100)
        return PosEgnnPromptEncoder(embed_dim, **kwargs)
    if kind == "posegnn_large_bounded":
        # a5_large's exact capacity (hidden=1100), plus the same tanh bound as
        # scalar27_large_bounded -- see PosEgnnPromptEncoder.bound_output.
        kwargs.setdefault("hidden", 1100)
        kwargs.setdefault("bound_output", 3.0)
        return PosEgnnPromptEncoder(embed_dim, **kwargs)
    if kind == "posegnn":
        return PosEgnnPromptEncoder(embed_dim, **kwargs)
    if kind == "mace":
        # A6: same architecture as A5 (posegnn), same 27-dim descriptor + per-atom
        # 3D tokens, only the frozen 3D backbone differs (MACE-MP-0 vs POS-EGNN).
        return PosEgnnPromptEncoder(embed_dim, **kwargs)
    if kind == "mace_large":
        # Capacity-matched to a5_large/posegnn_large (hidden=1100, 3.216M) so this
        # is a controlled "which 3D foundation model" comparison, not a capacity
        # confound -- same reasoning as posegnn_large's own comment above.
        kwargs.setdefault("hidden", 1100)
        return PosEgnnPromptEncoder(embed_dim, **kwargs)
    if kind == "mace_large_bounded":
        kwargs.setdefault("hidden", 1100)
        kwargs.setdefault("bound_output", 3.0)
        return PosEgnnPromptEncoder(embed_dim, **kwargs)
    raise ValueError(
        f"unknown prompt encoder {kind!r}; expected one of: "
        "null, scalar27, scalar27_epsfix, scalar27_norm, orbital_set, "
        "orbital_set_large, orbital_set_xl, scalar27_large, scalar27_large_bounded, "
        "posegnn, posegnn_large, posegnn_large_bounded, "
        "mace, mace_large, mace_large_bounded"
    )


# --------------------------------------------------------------------------- #
# A5: POS-EGNN (3D foundation model) concatenated with the active-space descriptor
# --------------------------------------------------------------------------- #

_ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "posegnn"
# Searched in order; the first file containing a matching norb wins. multimol
# covers the benchmark set (H2 through C2H4, one distinct geometry each), sn covers
# the Sn(OH)3-Me tiers (all four sharing one geometry).
POSEGNN_ASSETS = (_ASSET_DIR / "multimol_posegnn_embeddings.pt",
                  _ASSET_DIR / "sn_posegnn_embeddings.pt")
POSEGNN_ASSET = POSEGNN_ASSETS[1]      # kept for callers that referenced it


def _match_entry(tiers: dict, system, blob: dict) -> str | None:
    """Find the stored entry for ``system``, or None. Asset-agnostic -- used by both
    PosEgnnContext and MaceContext, since both store the same {norb, symbols,
    embedding} shape per entry.

    norb alone is not a key: in this benchmark set NH3 and BeH2 both have norb=7,
    and N2 and CH4 both have norb=8. A norb-only lookup silently returned the wrong
    molecule's geometry for two of the seven training molecules -- caught because
    NH3 (4 atoms) came back with a 3-atom embedding. Matching on (norb, n_atoms)
    separates every pair in this set; the chemical symbols are compared too, so a
    future collision fails loudly instead of conditioning on the wrong structure.
    """
    norb = int(system.norb)
    try:
        symbols = [system.hf.mol.atom_symbol(i) for i in range(system.hf.mol.natm)]
    except Exception:
        symbols = None          # from_fcidump systems may not expose a usable mol

    hits = [k for k, e in tiers.items() if e.get("norb") is not None
            and int(e["norb"]) == norb]
    if not hits:
        # Older asset without a norb field (the Sn tiers): fall back to its index.
        by_norb = blob.get("by_norb") or {16: "S", 19: "M", 23: "L", 28: "LL"}
        return by_norb.get(norb)
    if symbols is None:
        return hits[0] if len(hits) == 1 else None
    exact = [k for k in hits if sorted(tiers[k].get("symbols", [])) == sorted(symbols)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"ambiguous POS-EGNN entry for norb={norb} symbols={sorted(symbols)}: "
            f"{exact} all match. Pass tier= explicitly."
        )
    raise ValueError(
        f"POS-EGNN asset has entries at norb={norb} ({hits}) but none with symbols "
        f"{sorted(symbols)} -- the stored geometry is a different molecule. "
        "Precompute this molecule with scripts/precompute_posegnn.py."
    )


class PosEgnnContext:
    """A5 input: precomputed POS-EGNN per-atom embeddings + the 27-dim descriptor.

    The embeddings are read from a file rather than produced by running POS-EGNN
    here, and that is deliberate. POS-EGNN pins ``torch==2.5.1`` with
    ``torch_geometric``/``torch_scatter`` compiled against it, while the training
    environment runs torch 2.11; installing both in one place risks breaking the
    environment the experiments depend on. Precomputing keeps those dependencies
    out of it entirely -- what ships is an 11 KB tensor.

    This is sound here only because the embedding is a constant for this benchmark:
    the S/M/L/LL tiers share a byte-identical geometry, so POS-EGNN returns the
    same (11, 256) for all four -- verified to max|diff| = 0.0. Nothing changes
    between iterations or tiers, so there is nothing to recompute.

    Not speed: POS-EGNN inference is 0.014s against 226-492s for one
    diagonalization, so calling it every iteration would be free. The reason is
    purely dependency isolation.

    LIMITATION: a system whose coordinates differ from the stored geometry cannot
    use this file. ``build`` raises rather than conditioning on the wrong
    structure, so the failure is loud. Such a case needs POS-EGNN installed in the
    training environment (the isolated-venv approach stops working there).
    """

    def __init__(self, features: np.ndarray, embedding: np.ndarray):
        self.features = features            # (27,)   active-space descriptor
        self.embedding = embedding          # (n_atoms, 256)  POS-EGNN

    @classmethod
    def build(cls, system, backend=None, *, tier: str | None = None,
              asset_path=None, epsfix: bool = True) -> "PosEgnnContext":
        """``tier`` selects which stored geometry to use; inferred from norb if None.

        ``epsfix`` defaults to True here: A5's whole premise is that the descriptor
        carries the size information the 3D side cannot, so it would be
        self-defeating to pair it with the descriptor variant whose three
        orbital-energy features are dead (see molecule_features_epsfix).
        """
        candidates = ([Path(asset_path)] if asset_path is not None
                      else [p for p in POSEGNN_ASSETS if p.exists()])
        if not candidates:
            raise FileNotFoundError(
                f"No POS-EGNN embeddings found in {_ASSET_DIR}. Generate them in an "
                "isolated venv (torch 2.5.1 + torch_geometric) -- see "
                "scripts/precompute_posegnn.py. They are NOT computed at training "
                "time on purpose, to keep POS-EGNN's pinned torch off this env."
            )
        norb = int(system.norb)
        seen = {}
        errors = {}
        emb = None
        for path in candidates:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            tiers = blob["tiers"]
            key = tier
            if key is None:
                # _match_entry raises when THIS file has a norb collision with a
                # different molecule (e.g. multimol's ethanol/ferrocene both sit at
                # norb=16) -- that must not block checking the next candidate file,
                # since the real match (e.g. the Sn tiers' norb=None entries) can
                # live there instead. Record and keep going; only re-raise if no
                # candidate file matches at all.
                try:
                    key = _match_entry(tiers, system, blob)
                except ValueError as exc:
                    errors[path.name] = str(exc)
                    seen[path.name] = sorted(tiers)
                    continue
            if key is not None and key in tiers:
                entry = tiers[key]
                # A stored norb that disagrees with the system means the lookup
                # matched the wrong molecule -- refuse rather than condition on it.
                stored_norb = entry.get("norb")
                if stored_norb is not None and int(stored_norb) != norb:
                    raise ValueError(
                        f"asset {path.name} entry {key!r} has norb={stored_norb}, "
                        f"but this system has norb={norb}"
                    )
                emb = np.asarray(entry["embedding"], dtype=np.float32)
                break
            seen[path.name] = sorted(tiers)
        if emb is None:
            detail = f" Errors: {errors}." if errors else ""
            raise ValueError(
                f"no stored POS-EGNN embedding for norb={norb} (tier={tier!r}). "
                f"Available: {seen}.{detail} Precompute it with "
                "scripts/precompute_posegnn.py in the isolated venv -- these "
                "embeddings are geometry-specific, so a molecule that is not in the "
                "asset genuinely has no entry."
            )
        fn = molecule_features_epsfix if epsfix else molecule_features
        return cls(features=fn(system, backend), embedding=emb)

    def tensors(self, device=None):
        """-> (embedding (n_atoms, 256), features (27,))."""
        e = torch.from_numpy(self.embedding)
        f = torch.from_numpy(self.features)
        if device is not None:
            e, f = e.to(device), f.to(device)
        return e, f


_MACE_ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "mace"
MACE_ASSETS = (_MACE_ASSET_DIR / "multimol_mace_embeddings.pt",)


class MaceContext:
    """A6 input: precomputed MACE-MP-0 per-atom embeddings + the 27-dim descriptor.

    Structurally identical to PosEgnnContext (same {norb, symbols, embedding} asset
    shape, same _match_entry lookup, same 256-dim per-atom output) -- MACE-MP-0 is a
    second, independently-pretrained equivariant GNN foundation model
    (Batatia et al., "A foundation model for atomistic materials chemistry",
    mace-torch, MIT-licensed), added to test whether POS-EGNN's specific pretraining
    matters for this task or any modern equivariant 3D foundation model would do
    about as well -- not because POS-EGNN was found lacking.

    Precomputed offline (scripts/precompute_mace.py, isolated venv) for the same
    dependency-isolation reason as POS-EGNN, even though mace-torch has no strict
    torch-version pin conflicting with the training venv: keeping e3nn/matscipy/
    torch-ema etc. out of the actual training environment, not working around an
    incompatibility.

    NOT the same experiment as this file's existing note (on PosEgnnPromptEncoder)
    that "the paper's Appendix A.1.2 found pooled MACE embeddings indistinguishable
    from no molecular information" -- that finding was about POOLING a MACE
    embedding into a single vector for additive conditioning. This uses MACE's
    per-atom tokens with cross-attention, the same architecture A5 already uses
    successfully with POS-EGNN, not the pooled-additive approach that failed.
    """

    def __init__(self, features: np.ndarray, embedding: np.ndarray):
        self.features = features            # (27,)   active-space descriptor
        self.embedding = embedding          # (n_atoms, 256)  MACE-MP-0

    @classmethod
    def build(cls, system, backend=None, *, tier: str | None = None,
              asset_path=None, epsfix: bool = True) -> "MaceContext":
        candidates = ([Path(asset_path)] if asset_path is not None
                      else [p for p in MACE_ASSETS if p.exists()])
        if not candidates:
            raise FileNotFoundError(
                f"No MACE embeddings found in {_MACE_ASSET_DIR}. Generate them in an "
                "isolated venv (pip install mace-torch) -- see "
                "scripts/precompute_mace.py. Not computed at training time, to keep "
                "mace-torch's dependency tree (e3nn, matscipy, torch-ema) off this env."
            )
        norb = int(system.norb)
        seen = {}
        errors = {}
        emb = None
        for path in candidates:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            tiers = blob["tiers"]
            key = tier
            if key is None:
                # Same per-file short-circuit fix as PosEgnnContext.build above: a
                # norb collision in ONE candidate file (e.g. multimol) must not
                # block checking the next file, where the real match may live.
                try:
                    key = _match_entry(tiers, system, blob)
                except ValueError as exc:
                    errors[path.name] = str(exc)
                    seen[path.name] = sorted(tiers)
                    continue
            if key is not None and key in tiers:
                entry = tiers[key]
                stored_norb = entry.get("norb")
                if stored_norb is not None and int(stored_norb) != norb:
                    raise ValueError(
                        f"asset {path.name} entry {key!r} has norb={stored_norb}, "
                        f"but this system has norb={norb}"
                    )
                emb = np.asarray(entry["embedding"], dtype=np.float32)
                break
            seen[path.name] = sorted(tiers)
        if emb is None:
            detail = f" Errors: {errors}." if errors else ""
            raise ValueError(
                f"no stored MACE embedding for norb={norb} (tier={tier!r}). "
                f"Available: {seen}.{detail} Precompute it with "
                "scripts/precompute_mace.py in the isolated venv -- these "
                "embeddings are geometry-specific, so a molecule that is not in the "
                "asset genuinely has no entry."
            )
        fn = molecule_features_epsfix if epsfix else molecule_features
        return cls(features=fn(system, backend), embedding=emb)

    def tensors(self, device=None):
        """-> (embedding (n_atoms, 256), features (27,))."""
        e = torch.from_numpy(self.embedding)
        f = torch.from_numpy(self.features)
        if device is not None:
            e, f = e.to(device), f.to(device)
        return e, f


class PosEgnnPromptEncoder(nn.Module):
    """A5: POS-EGNN per-atom embeddings and the descriptor as separate prompt tokens.

    Prompt length is ``n_atoms + 1``: one token per atom from POS-EGNN, plus one
    from the 27-dim descriptor. They are kept as separate tokens rather than
    concatenated into one, because that is the point of the cross-attention route
    -- the additive path would force a pool to a single vector, and the paper's
    Appendix A.1.2 found pooled MACE embeddings indistinguishable from using no
    molecular information at all.

    Two independent projections rather than one shared: the inputs are unrelated
    (a 256-dim learned geometric representation versus 27 hand-built active-space
    statistics), and sharing weights would force one layer to serve both scales.

    What this arm can and cannot show: the 3D side is *constant* across the S/M/L/LL
    tiers (byte-identical geometry, verified), so all size information reaching the
    decoder comes from the descriptor token. If A5 matches A1', the honest reading
    is "the 3D embedding contributed nothing here" -- which is a result worth
    reporting, since the paper only ever tested 3D embeddings alone, never combined
    with an active-space descriptor.
    """

    def __init__(self, embed_dim: int, egnn_dim: int = 256,
                 feature_dim: int = FEATURE_DIM, hidden: int = 128,
                 bound_output: float | None = None):
        super().__init__()
        self.egnn_norm = nn.LayerNorm(egnn_dim)
        self.egnn_proj = nn.Sequential(
            nn.Linear(egnn_dim, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        # See ScalarPromptEncoder's bound_output docstring -- same rationale,
        # applied to both token streams so neither can dominate the other in
        # cross-attention purely by having a larger unbounded scale.
        self.bound_output = bound_output

    def forward(self, inputs) -> torch.Tensor:
        """inputs: (embedding (n_atoms, egnn_dim), features (feature_dim,))
        -> (n_atoms + 1, embed_dim)."""
        emb, feats = inputs
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        atom_tokens = self.egnn_proj(self.egnn_norm(emb))    # (n_atoms, E)
        desc_token = self.feat_proj(feats)                   # (1, E)
        out = torch.cat([atom_tokens, desc_token], dim=0)    # (n_atoms + 1, E)
        if self.bound_output is not None:
            out = self.bound_output * torch.tanh(out / self.bound_output)
        return out
