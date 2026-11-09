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

import os as _os

# qiskit runs transpiler passes such as CommutativeCancellation on Rust threads, whose
# default stack is 2 MB. On ROQUO's aarch64 nodes that overflows at optimization_level
# 2 and 3 and the process dies with SIGSEGV, from 32q up (ROQUO job 169486). A larger
# stack is the only fix among the ones tried: one Rayon thread, QISKIT_PARALLEL=FALSE and
# three OpenBLAS settings all still crash. Rust reads this once, when it spawns its first
# thread, so it has to be set before qiskit does any work. That is why it lives here and
# not at the noise model's level-3 transpile: an earlier qiskit call (an MPS or IBM
# transpile, a circuit-depth metric) would already have started the threads.
_os.environ.setdefault("RUST_MIN_STACK", str(64 << 20))
