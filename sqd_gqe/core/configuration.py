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

"""Slater-determinant configurations and their subspaces (self-contained).

A ``Configuration`` is a single Slater determinant encoded as a pair of uint64
occupation bitmasks ``[alpha, beta]``. A ``ConfigurationSet`` is an ordered
collection of them with the operations the SQD pipeline needs: electron-number
post-selection and total-spin symmetry completion. ``CIVector`` carries CI
amplitudes alongside their configuration strings for the refinement step.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from pyscf import fci


# --------------------------------------------------------------------------- #
# bit helpers                                                                 #
# --------------------------------------------------------------------------- #
def _iter_set_bits(x: int):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


def _popcount(x: int) -> int:
    return bin(int(x)).count("1")


# --------------------------------------------------------------------------- #
# CI vector                                                                   #
# --------------------------------------------------------------------------- #
class CIVector(np.ndarray):
    """A complex amplitude array tagged with its (alpha, beta) string table."""

    def __array_finalize__(self, obj):
        self.strings = getattr(obj, "strings", None)


def fold_rhf_half_strings(strings, weights=None):
    """RHF: alpha and beta half-strings are interchangeable, so fold both columns
    of an ``(N, 2)`` (alpha, beta) string table into one sorted unique list.

    With ``weights`` (length N, e.g. ``|c|^2`` or sampling probabilities), each
    weight is split equally over the pair's two half-strings and summed per
    unique string. Returns ``(uniq int64, uniq_weights | None)``.
    """
    strings = np.asarray(strings).reshape(-1, 2).astype(np.int64)
    mixed = np.concatenate([strings[:, 0], strings[:, 1]])
    uniq, inv = np.unique(mixed, return_inverse=True)
    if weights is None:
        return uniq, None
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    uniq_w = np.bincount(inv.ravel(), weights=np.concatenate([w, w]) / 2.0, minlength=uniq.size)
    return uniq, uniq_w


def make_ci_vector(coeffs, strings) -> CIVector:
    vec = np.asarray(coeffs).view(CIVector)
    vec.strings = np.asarray(strings, dtype=np.uint64)
    return vec


# --------------------------------------------------------------------------- #
# single determinant                                                          #
# --------------------------------------------------------------------------- #
class Configuration(np.ndarray):
    """One Slater determinant as ``[alpha_occ, beta_occ]`` uint64 bitmasks."""

    def __new__(cls, config: np.ndarray | list[int]):
        arr = np.asarray(config, dtype=np.uint64).reshape(-1)
        if arr.size != 2:
            raise ValueError(
                f"Configuration must be [alpha, beta] (size 2); got size={arr.size}, val={config!r}"
            )
        return arr.view(cls)

    def conserves_nelec(self, nelec: tuple[int, int]) -> bool:
        return _popcount(int(self[0])) == nelec[0] and _popcount(int(self[1])) == nelec[1]

    @classmethod
    def from_fullci_address(cls, idx: int, norb: int, nelec: tuple[int, int]) -> "Configuration":
        dim_b = math.comb(norb, nelec[1])
        alpha = fci.cistring.addr2str(norb, nelec[0], idx // dim_b)
        beta = fci.cistring.addr2str(norb, nelec[1], idx % dim_b)
        return cls([alpha, beta])

    def spin_symmetry_group(self) -> np.ndarray:
        """Enumerate spin assignments of open-shell orbitals (fixed #alpha-open)."""
        a, b = int(self[0]), int(self[1])
        doubly = a & b
        open_ = a ^ b
        if open_ == 0:
            return np.asarray(self, dtype=np.uint64).reshape(1, 2)
        open_orbs = list(_iter_set_bits(open_))
        n_alpha_open = _popcount(a & open_)
        seen: set[tuple[int, int]] = set()
        rows: list[list[np.uint64]] = []
        for choice in itertools.combinations(open_orbs, n_alpha_open):
            a_new = b_new = doubly
            chosen = set(choice)
            for orb in open_orbs:
                if orb in chosen:
                    a_new |= 1 << orb
                else:
                    b_new |= 1 << orb
            key = (a_new, b_new)
            if key in seen:
                continue
            seen.add(key)
            rows.append([np.uint64(a_new), np.uint64(b_new)])
        return np.asarray(rows, dtype=np.uint64)


# --------------------------------------------------------------------------- #
# subspace                                                                    #
# --------------------------------------------------------------------------- #
class ConfigurationSet:
    """An ordered set of determinants defining the SQD subspace."""

    def __init__(self, configs: Sequence[Configuration]):
        self.configs = configs

    @property
    def size(self) -> int:
        return len(self.configs)

    @classmethod
    def from_sample_counts(cls, counts: Mapping[tuple[int, int], int]) -> "ConfigurationSet":
        """Build from ``{(alpha_occ, beta_occ): count}`` (ffsim/qiskit samples),
        ordered by descending count. No interleaving/endianness handling needed:
        the sampler already emits integer occupation bitmasks per spin sector."""
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        configs = [Configuration([np.uint64(a), np.uint64(b)]) for (a, b), _ in ordered]
        return cls(configs)

    def select_by_nelec(self, nelec: tuple[int, int]) -> "ConfigurationSet":
        return ConfigurationSet([c for c in self.configs if c.conserves_nelec(nelec)])

    def expand(self, max_size: int, method: str = "symmetry_completion") -> "ConfigurationSet":
        if method == "symmetry_completion":
            return self._complete_spin_symmetry(max_size)
        if method == "None":
            return ConfigurationSet(self.configs[:max_size])
        raise ValueError(f"Unknown expand method: {method}")

    def _complete_spin_symmetry(self, max_size: int) -> "ConfigurationSet":
        expanded: list[Configuration] = []
        seen: set[tuple[int, int]] = set()
        for config in self.configs:
            group = np.asarray(config.spin_symmetry_group(), dtype=np.uint64)
            if group.ndim == 1:
                group = group.reshape(1, 2)
            elif group.ndim == 2 and group.shape[1] != 2:
                raise ValueError(f"Invalid group shape: expected (N,2), got {group.shape}")
            for a, b in group:
                key = (int(a), int(b))
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(Configuration([a, b]))
                if len(expanded) >= int(max_size):
                    return ConfigurationSet(expanded)
        return ConfigurationSet(expanded)
