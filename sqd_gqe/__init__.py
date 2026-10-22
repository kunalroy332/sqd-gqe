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

"""sqd_gqe: a Generative Quantum Eigensolver for Sample-based Quantum Diagonalization.

A transformer policy learns to design state-preparation circuits whose sampled
Slater determinants form a compact, high-quality subspace for diagonalizing a
molecular Hamiltonian. Circuits are simulated with ffsim (fast fermionic
statevector simulation) and the subspace is diagonalized with qiskit-addon-sqd;
the policy is trained by a group-relative policy-gradient objective on the SQD
subspace energy. The package is fully self-contained.
"""
