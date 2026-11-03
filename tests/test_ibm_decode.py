# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Decode round-trip tests for IbmHardwareSampler's bitstring -> occupation
mask conversion.

Guards the single highest-risk correctness spot for hardware sampling. A
qiskit bitstring's classical bit i corresponds to LOGICAL qubit i, NOT
physical qubit i after SABRE routing -- confirmed empirically (job 30063:
flipping one logical qubit at a time via ``x(i)`` and checking which
classical bit position moved in the returned counts). ``measure_all()`` adds
one clbit per qubit of the PRE-transpile (logical) circuit, and that
correspondence is preserved through transpilation regardless of which
physical qubits ``final_index_layout()`` says the circuit landed on -- so no
layout-based permutation is needed to decode a plain ``measure_all()``
circuit's counts. (An earlier version of this module assumed the opposite --
that the layout permutation had to be undone first -- based on a
pre-execution reading of the Stage 0 plan; this was corrected once real
sampling was exercised end-to-end.) Pure-Python, no qiskit-ibm-runtime/
hardware needed.
"""

from collections import Counter


def _decode_bitarray(counts_by_bitstring: dict, layout, norb: int) -> dict[tuple[int, int], int]:
    """Standalone copy of IbmHardwareSampler._decode_bitarray's logic, so this
    test has no import-time dependency on qiskit/ffsim being installed.
    ``layout`` is accepted but unused (see module docstring)."""
    n_qubits = 2 * norb
    mask = (1 << norb) - 1
    counts: Counter = Counter()
    for bitstring, n in counts_by_bitstring.items():
        bits = bitstring.replace(" ", "")
        logical_bits = [int(bits[len(bits) - 1 - i]) for i in range(n_qubits)]
        s = 0
        for i, b in enumerate(logical_bits):
            s |= (b << i)
        alpha = s & mask
        beta = s >> norb
        counts[(alpha, beta)] += n
    return dict(counts)


def test_decode_beta_high_alpha_low():
    # norb=2 -> 4 qubits. alpha occupies orbital 0 (bit 0), beta occupies
    # orbital 1 (bit 3): bitstring '1001' -> bit0=1, bit1=0, bit2=0, bit3=1
    norb = 2
    counts = {"1001": 7}
    decoded = _decode_bitarray(counts, None, norb)
    assert decoded == {(0b01, 0b10): 7}


def test_layout_argument_is_ignored():
    # layout must NOT affect the decode -- any layout value gives the same result.
    norb = 2
    counts = {"1001": 3}
    decoded_none = _decode_bitarray(counts, None, norb)
    decoded_identity = _decode_bitarray(counts, [0, 1, 2, 3], norb)
    decoded_permuted = _decode_bitarray(counts, [2, 0, 3, 1], norb)
    assert decoded_none == decoded_identity == decoded_permuted == {(0b01, 0b10): 3}


def test_hf_state_all_zero():
    norb = 4
    counts = {"00000000": 100}
    decoded = _decode_bitarray(counts, None, norb)
    assert decoded == {(0, 0): 100}


def test_multiple_bitstrings_accumulate_counts():
    norb = 2
    decoded = _decode_bitarray({"0011": 5, "1100": 3}, None, norb)
    # '0011' -> bits [1,1,0,0] -> alpha=0b11, beta=0b00
    # '1100' -> bits [0,0,1,1] -> alpha=0b00, beta=0b11
    assert decoded == {(0b11, 0b00): 5, (0b00, 0b11): 3}


def test_bitstring_with_spaces_stripped():
    # qiskit sometimes groups classical registers with a space separator;
    # a plain measure_all() on one register shouldn't have one, but strip
    # defensively since the decode must not silently misparse if it does.
    norb = 2
    decoded_spaced = _decode_bitarray({"10 01": 4}, None, norb)
    decoded_plain = _decode_bitarray({"1001": 4}, None, norb)
    assert decoded_spaced == decoded_plain
