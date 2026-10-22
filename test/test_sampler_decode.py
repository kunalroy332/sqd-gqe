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

"""Endianness / decode round-trip tests for the ffsim sampler.

These guard the single most error-prone integration point: decoding ffsim's
concatenated integer samples (beta high bits, alpha low bits) into
``(alpha_occ, beta_occ)`` occupation bitmasks, and feeding them into
``ConfigurationSet.from_sample_counts``. Pure-Python, no ffsim/pyscf needed.
"""

import numpy as np


def _decode(s: int, norb: int) -> tuple[int, int]:
    mask = (1 << norb) - 1
    return s & mask, s >> norb


def test_decode_beta_high_alpha_low():
    norb = 4
    # beta occupies orbitals {0,2} -> 0b0101 = 5; alpha occupies {1,3} -> 0b1010 = 10
    alpha, beta = 0b1010, 0b0101
    s = (beta << norb) | alpha
    dec_alpha, dec_beta = _decode(s, norb)
    assert dec_alpha == alpha
    assert dec_beta == beta


def test_hf_state_decode():
    # Hartree-Fock for (n_alpha, n_beta) = (2, 2), norb=4: lowest 2 orbitals filled
    norb = 4
    hf_alpha = hf_beta = 0b0011
    s = (hf_beta << norb) | hf_alpha
    a, b = _decode(s, norb)
    assert a == hf_alpha and b == hf_beta
    assert bin(a).count("1") == 2 and bin(b).count("1") == 2


def test_configuration_set_from_counts_orders_by_count():
    from sqd_gqe.core.configuration import ConfigurationSet

    counts = {(0b0011, 0b0011): 10, (0b0101, 0b0011): 3, (0b0011, 0b0101): 7}
    cs = ConfigurationSet.from_sample_counts(counts)
    assert cs.size == 3
    # highest count first
    first = cs.configs[0]
    assert int(first[0]) == 0b0011 and int(first[1]) == 0b0011


def test_nelec_selection():
    from sqd_gqe.core.configuration import ConfigurationSet

    counts = {(0b0011, 0b0011): 5, (0b0111, 0b0011): 2}  # second has 3 alpha electrons
    cs = ConfigurationSet.from_sample_counts(counts).select_by_nelec((2, 2))
    assert cs.size == 1
    assert int(cs.configs[0][0]) == 0b0011
