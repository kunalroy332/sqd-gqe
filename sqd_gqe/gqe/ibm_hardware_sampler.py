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

"""IBM Quantum (real or fake) hardware sampler for the GQE + SQD pipeline.

Drop-in replacement for ``FfsimSampler`` (``sqd_gqe/gqe/sampler.py``) --
matches its ``__init__``/``run()`` shape (duck-typed, not a subclass, since it
shares no simulation logic). Builds a real ``QuantumCircuit`` from the
operator-pool sequence (every op must have ``qiskit_gate_fn`` set -- see
``sqd_gqe/gqe/operator_pool.py``'s Givens/num-num closures), transpiles it to
a target backend, submits it through Qiskit Runtime's ``SamplerV2``, and
decodes the returned bitstrings back into ffsim's ``(alpha_occ, beta_occ)``
convention.

Real hardware returns bitstrings ordered by *physical* qubit index after
SABRE routing; ffsim never permutes, so decoding must first undo that
permutation via the transpiled circuit's ``final_index_layout()`` (a plain
``list[logical_qubit_idx] = physical_qubit_idx`` map) before applying the
same mask/shift decode ``FfsimSampler`` uses.

``fake_backend=True`` runs against a ``qiskit_ibm_runtime.fake_provider``
backend wrapped in ``AerSimulator`` instead of real Qiskit Runtime -- no
credentials, no queue, no cost. At M/L/LL's 38-56 qubit scale, exact
statevector simulation is infeasible (``2**56`` amplitudes for LL); the fake
path therefore always uses ``method="matrix_product_state"``, an approximate
tensor-network method that scales with circuit entanglement instead of qubit
count. This validates sampler/decode/pipeline *plumbing* at scale, not
physical correctness of the resulting energies -- that's what real hardware
(or small systems where exact simulation stays tractable) checks.

Any script importing this alongside ``torch``/``sqd_gqe.factory`` (which
transitively imports ``ffsim``/``pyscf``) MUST
``export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152`` in the shell
before starting Python -- combining these native-extension-heavy packages
exhausts aarch64 glibc's default static-TLS pool otherwise (confirmed via
direct empirical testing, not an import-order issue).

``optimization_level`` defaults to 1, not Qiskit's usual 3 -- found the hard
way (Stage E, 2026-08-13): transpiling this project's ``ucj_balanced_jw`` op
(a dense custom unitary, not a standard gate) at ``optimization_level=3``
segfaults Qiskit's native transpiler outright (SIGSEGV, no Python traceback,
reproduced identically against both the live Runtime backend object and the
offline ``HardwareBackend`` coupling map -- so it's the optimization level,
not the live-vs-offline backend object, that triggers it). Every other
transpile call in this project (the offline ISA-circuit reward path) already
used ``optimization_level=1``; this sampler's real-hardware submission path
was the one place still defaulting to 3.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch

from sqd_gqe.gqe.operator_pool import FfsimBasedPool


class IbmHardwareSampler:
    def __init__(
        self,
        operator_pool: FfsimBasedPool,
        shots_count: int,
        backend_name: str,
        seed: int | None = 0,
        channel: str | None = "ibm_quantum_platform",
        instance: str | None = None,
        token: str | None = None,
        optimization_level: int = 1,
        seed_transpiler: int = 0,
        max_shots_per_job: int | None = None,
        dry_run: bool = False,
        fake_backend: bool = False,
        save_dir: str | None = None,
        dynamical_decoupling: bool = False,
        dd_sequence_type: str = "XpXm",
        init_qubits: bool = True,
        session_mode: bool = False,
        session_max_time: int | str | None = None,
    ):
        offenders = sorted({
            op.name.split("(")[0].split("[")[0]
            for op in operator_pool
            if op.name != "I" and op.qiskit_gate_fn is None
        })
        if offenders:
            raise ValueError(
                f"IbmHardwareSampler requires every op to have qiskit_gate_fn set; "
                f"missing for op families: {offenders}"
            )

        self.pool = operator_pool
        self.shots_count = int(shots_count)
        self.norb = operator_pool.norb
        self.nelec = tuple(operator_pool.nelec)
        self.backend_name = backend_name
        self.channel = channel
        self.instance = instance
        self.token = token
        self.optimization_level = optimization_level
        self.seed_transpiler = seed_transpiler
        self.max_shots_per_job = max_shots_per_job
        self.dry_run = dry_run
        self.fake_backend = fake_backend
        self.save_dir = save_dir
        self.dynamical_decoupling = dynamical_decoupling
        self.dd_sequence_type = dd_sequence_type
        self.init_qubits = init_qubits
        self.session_mode = session_mode
        self.session_max_time = session_max_time
        self._rng = np.random.default_rng(seed)

        # lazily constructed on first use -- so instantiating for config-wiring
        # tests never touches the network or builds an expensive Aer backend.
        self._backend = None
        # A real Runtime Session is also opened lazily.  In particular, all
        # circuits are transpiled before the first submission so that work does
        # not consume the session's finite active window.
        self._session = None
        # Side-channel: transpiled depth of each circuit in the most recent run()/
        # _sample_batch() call, in the same order as its return value. Populated
        # AFTER transpile (so it's known) but the samples/counts are still fetched
        # for every circuit regardless of depth -- callers that want to skip the
        # expensive diagonalization step for a deep outlier (while still keeping
        # its hardware counts for later inspection) read this list themselves;
        # this sampler does not filter or drop anything on its own.
        self.last_depths: list[int] = []
        self.last_two_qubit_gates: list[int] = []
        # Set by SqdPipeline (if it was constructed with max_depth_for_diag) purely so
        # _sample_batch's post-transpile log line can report the would-be-rejected count
        # right there, before diagonalize() -- this sampler never reads it to change its
        # own behavior (it always samples/decodes every circuit regardless of depth).
        self.max_depth_for_diag_for_log: int | None = None

    # -- backend construction (lazy) ----------------------------------------- #
    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        if self.fake_backend:
            import qiskit_ibm_runtime.fake_provider as fake_provider
            from qiskit_aer import AerSimulator

            fake_cls = getattr(fake_provider, self.backend_name)
            fake = fake_cls()
            if self.token is not None and self.instance is not None:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService(
                    channel=self.channel, token=self.token, instance=self.instance,
                )
                fake.refresh(service)
            self._backend = AerSimulator.from_backend(fake, method="matrix_product_state")
            self._transpile_target = fake
        else:
            import sys
            print(f"[sampler] connecting to QiskitRuntimeService "
                  f"(channel={self.channel})...", file=sys.stderr, flush=True)
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(
                channel=self.channel, token=self.token, instance=self.instance,
            )
            if self.backend_name == "least_busy":
                print("[sampler] resolving least_busy backend...", file=sys.stderr, flush=True)
                # sorted by pending_jobs ascending; operational=True excludes
                # backends currently down for maintenance/calibration.
                candidates = [b for b in service.backends(operational=True)]
                if not candidates:
                    raise RuntimeError("no operational backends available under this instance")
                backend = min(candidates, key=lambda b: b.status().pending_jobs)
                print(f"[sampler] least_busy resolved to {backend.name} "
                      f"({backend.status().pending_jobs} pending jobs)", file=sys.stderr, flush=True)
            else:
                print(f"[sampler] fetching backend {self.backend_name}...",
                      file=sys.stderr, flush=True)
                backend = service.backend(self.backend_name)
            self._backend = backend
            self._transpile_target = backend
            print(f"[sampler] backend ready: {backend.name}", file=sys.stderr, flush=True)
        return self._backend

    def _get_execution_mode(self):
        """Return the backend (job mode) or the sampler's persistent Session."""
        backend = self._get_backend()
        if not self.session_mode:
            return backend
        if self.fake_backend:
            raise ValueError("IBM Runtime session mode cannot be used with a fake backend")
        if self._session is None:
            import sys
            from qiskit_ibm_runtime import Session

            print(
                f"[sampler] opening Runtime session on {self.backend_name} "
                f"(max_time={self.session_max_time!r})...",
                file=sys.stderr,
                flush=True,
            )
            self._session = Session(backend=backend, max_time=self.session_max_time)
        return self._session

    def close(self) -> None:
        """Stop accepting work in the Runtime Session, if one was opened."""
        if self._session is None:
            return
        import sys

        session, self._session = self._session, None
        print("[sampler] closing Runtime session...", file=sys.stderr, flush=True)
        session.close()

    # -- SamplerV2 options (error mitigation) ---------------------------------- #
    def _configure_sampler_options(self, sampler) -> None:
        """Apply dynamical decoupling + active qubit reset to a SamplerV2 instance.

        Neither was ever configured before (both defaulted to Runtime's own
        defaults, DD off) -- found while investigating Stage F's 46q real-hardware
        collapse (most sampled circuits had zero electron-count-conserving
        bitstrings). DD idles qubits with an echo sequence during long gate-free
        windows instead of leaving them exposed to dephasing; init_qubits=True
        makes explicit what was already Runtime's default (active reset before
        each shot) so it's not silently relying on an unconfirmed default.
        """
        if self.dynamical_decoupling:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = self.dd_sequence_type
        sampler.options.execution.init_qubits = self.init_qubits

    # -- circuit construction -------------------------------------------------- #
    def prepare_circuit(self, seq: Sequence[int]):
        from qiskit import QuantumCircuit

        n_qubits = 2 * self.norb
        n_alpha, n_beta = self.nelec
        n_qubits = 2 * self.norb
        qc = QuantumCircuit(n_qubits)
        # Match ffsim.hartree_fock_state:
        # alpha orbitals occupy qubits [0, norb)
        # beta orbitals occupy qubits [norb, 2*norb)
        for orbital in range(n_alpha):
            qc.x(orbital)
        for orbital in range(n_beta):
            qc.x(self.norb + orbital)

        for idx in seq:
            op = self.pool[int(idx)]
            if op.name == "I":
                continue
            gate = op.qiskit_gate_fn(self.norb)
            qc.append(gate, range(n_qubits))
        return qc

    def transpile_circuit(self, qc):
        from qiskit import transpile

        self._get_backend()
        tqc = transpile(
            qc, backend=self._transpile_target,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed_transpiler,
        )
        layout = tqc.layout.final_index_layout() if tqc.layout is not None else None
        return tqc, layout

    # -- decode ---------------------------------------------------------------- #
    def _decode_bitarray(self, counts_by_bitstring: dict, layout, norb: int) -> dict[tuple[int, int], int]:
        """Apply ffsim's mask/shift decode to qiskit's returned bitstrings.

        ``counts_by_bitstring`` keys are qiskit-order bitstrings (classical
        bit i = LOGICAL qubit i, confirmed empirically -- job 30063: flipping
        one logical qubit at a time and checking which classical bit moved).
        ``measure_all()`` adds one clbit per qubit of the PRE-transpile
        (logical) circuit, and that clbit<->logical-qubit correspondence is
        preserved through transpilation regardless of the physical
        qubits/SABRE routing `final_index_layout()` describes -- so no
        permutation is needed here. ``layout`` is accepted but unused; kept in
        the signature since a future multi-register or partial-measurement
        circuit shape might need it, and to keep this method's shape matching
        ``transpile_circuit``'s two return values.
        """
        n_qubits = 2 * norb
        mask = (1 << norb) - 1
        counts: Counter = Counter()
        for bitstring, n in counts_by_bitstring.items():
            bits = bitstring.replace(" ", "")
            # leftmost char = highest classical/logical bit index; logical
            # qubit i's bit is at string index (len(bits) - 1 - i).
            logical_bits = [int(bits[len(bits) - 1 - i]) for i in range(n_qubits)]
            s = 0
            for i, b in enumerate(logical_bits):
                s |= (b << i)
            alpha = s & mask
            beta = s >> norb
            counts[(alpha, beta)] += n
        return dict(counts)

    # -- persistence (reproducibility) ----------------------------------------- #
    def _save_job_data(self, job_id: str, tqcs: list, raw_counts: list[dict],
                        decoded_counts: list[dict[tuple[int, int], int]], shots: int) -> None:
        """Save raw+decoded counts and transpiled circuits (QPY) keyed by
        backend name and shot count, so a real-hardware result can be
        re-diagonalized (e.g. on a different diag_backend) or its circuits
        visualized later without resubmitting -- job_id ties it back to the
        exact Runtime job for cross-checking against the IBM dashboard."""
        if self.save_dir is None:
            return
        import json
        import sys
        from pathlib import Path
        from qiskit import qpy

        run_dir = Path(self.save_dir) / f"{self.backend_name}_shots{shots}" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "circuits.qpy", "wb") as f:
            qpy.dump(tqcs, f)

        counts_payload = {
            "job_id": job_id,
            "backend_name": self.backend_name,
            "shots": shots,
            "circuits": [
                {
                    "index": i,
                    "depth": tqcs[i].depth(),
                    "raw_counts": raw_counts[i],
                    "decoded_counts": {f"{a},{b}": n for (a, b), n in decoded_counts[i].items()},
                }
                for i in range(len(tqcs))
            ],
        }
        with open(run_dir / "counts.json", "w") as f:
            json.dump(counts_payload, f, indent=2)
        print(f"[sampler] saved {len(tqcs)} circuits + counts to {run_dir}",
              file=sys.stderr, flush=True)

    # -- sampling ---------------------------------------------------------------- #
    def sample_state(self, seq: Sequence[int]) -> dict[tuple[int, int], int]:
        import sys
        from qiskit_ibm_runtime import SamplerV2

        print(f"[sampler] preparing circuit ({len(seq)} gates)...", file=sys.stderr, flush=True)
        qc = self.prepare_circuit(seq)
        qc.measure_all()
        print(f"[sampler] transpiling to {self.backend_name} "
              f"(optimization_level={self.optimization_level})...", file=sys.stderr, flush=True)
        tqc, layout = self.transpile_circuit(qc)
        print(f"[sampler] transpiled: depth={tqc.depth()}", file=sys.stderr, flush=True)

        if self.dry_run:
            return {}

        sampler = SamplerV2(mode=self._get_execution_mode())
        self._configure_sampler_options(sampler)
        shots = self.shots_count
        if self.max_shots_per_job is not None:
            shots = min(shots, self.max_shots_per_job)
        print(f"[sampler] submitting Runtime job to {self.backend_name} "
              f"({shots} shots)...", file=sys.stderr, flush=True)
        job = sampler.run([tqc], shots=shots)
        print(f"[sampler] job submitted, id={job.job_id()} -- waiting for result "
              f"(this blocks on the real device's queue + execution)...", file=sys.stderr, flush=True)
        result = job.result()
        print(f"[sampler] job {job.job_id()} finished, decoding counts...",
              file=sys.stderr, flush=True)
        counts_by_bitstring = result[0].data.meas.get_counts()
        decoded = self._decode_bitarray(counts_by_bitstring, layout, self.norb)
        print(f"[sampler] decoded {len(decoded)} distinct bitstrings from "
              f"{sum(counts_by_bitstring.values())} shots", file=sys.stderr, flush=True)
        self._save_job_data(job.job_id(), [tqc], [counts_by_bitstring], [decoded], shots)
        return decoded

    def _sample_batch(self, seqs: Sequence[Sequence[int]]) -> list[dict[tuple[int, int], int]]:
        """Sample every sequence in ONE Runtime job -- all circuits submitted
        as separate pubs in a single SamplerV2.run([...]) call, instead of one
        job per circuit. Used by run() during training: a batch of
        num_samples circuits per training iteration becomes one real job per
        iteration, not num_samples jobs -- real hardware queue/job overhead
        is paid once per batch, not once per circuit."""
        import sys
        from qiskit_ibm_runtime import SamplerV2

        print(f"[sampler] about to sample+transpile {len(seqs)} circuits "
              f"({[len(s) for s in seqs]} gates each) for {self.backend_name} "
              f"(optimization_level={self.optimization_level})...",
              file=sys.stderr, flush=True)
        tqcs = []
        layouts = []
        for seq in seqs:
            qc = self.prepare_circuit(seq)
            qc.measure_all()
            tqc, layout = self.transpile_circuit(qc)
            tqcs.append(tqc)
            layouts.append(layout)
        depths = [t.depth() for t in tqcs]
        twoq_counts = [t.num_nonlocal_gates() for t in tqcs]
        self.last_depths = depths
        self.last_two_qubit_gates = twoq_counts
        max_depth_for_diag = self.max_depth_for_diag_for_log
        n_would_reject = (sum(1 for d in depths if d > max_depth_for_diag)
                          if max_depth_for_diag is not None else None)
        print(f"[sampler] all {len(seqs)} circuits transpiled: "
              f"depth range=[{min(depths)}, {max(depths)}] "
              f"2q-gate range=[{min(twoq_counts)}, {max(twoq_counts)}] "
              f"(depths: {depths}, 2q-gates: {twoq_counts})"
              + (f" -- {n_would_reject}/{len(seqs)} exceed max_depth_for_diag={max_depth_for_diag} "
                 "and will be skipped before diagonalize()" if n_would_reject is not None else ""),
              file=sys.stderr, flush=True)

        if self.dry_run:
            return [{} for _ in seqs]

        sampler = SamplerV2(mode=self._get_execution_mode())
        self._configure_sampler_options(sampler)
        shots = self.shots_count
        if self.max_shots_per_job is not None:
            shots = min(shots, self.max_shots_per_job)
        print(f"[sampler] submitting ONE Runtime job with {len(tqcs)} pubs to "
              f"{self.backend_name} ({shots} shots each)...", file=sys.stderr, flush=True)
        job = sampler.run(tqcs, shots=shots)
        print(f"[sampler] job submitted, id={job.job_id()} -- waiting for result "
              f"(this blocks on the real device's queue + execution)...", file=sys.stderr, flush=True)
        result = job.result()
        print(f"[sampler] job {job.job_id()} finished, decoding {len(seqs)} circuits' counts...",
              file=sys.stderr, flush=True)
        raw_counts = [result[i].data.meas.get_counts() for i in range(len(seqs))]
        decoded = [
            self._decode_bitarray(raw_counts[i], layouts[i], self.norb)
            for i in range(len(seqs))
        ]
        print(f"[sampler] decoded all {len(seqs)} circuits "
              f"({[len(d) for d in decoded]} distinct bitstrings each)",
              file=sys.stderr, flush=True)
        self._save_job_data(job.job_id(), tqcs, raw_counts, decoded, shots)
        return decoded

    @torch.no_grad()
    def run(self, state: dict) -> list[dict[tuple[int, int], int]]:
        """Sample every circuit in the batch, as ONE Runtime job (see
        _sample_batch). Same contract as FfsimSampler.run."""
        idx_output = state["idx"]
        if torch.is_tensor(idx_output):
            idx_output = idx_output.detach().cpu().tolist()
        return self._sample_batch(idx_output)
