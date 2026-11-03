#!/usr/bin/env python3
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check IBM Quantum connectivity without submitting a Runtime job.

The check authenticates using ``IBM_API_KEY`` and ``IBM_CRN``, resolves the
requested backend, and fetches its status.  A successful check confirms that
DNS/network access, credentials, the configured instance, and backend access
all work from the local machine.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


class CheckTimedOut(RuntimeError):
    """Raised when the connectivity check exceeds its wall-clock deadline."""


def _timeout_handler(_signum, _frame) -> None:
    raise CheckTimedOut("connectivity check timed out")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="least_busy",
        help='IBM backend name, or "least_busy" (default: %(default)s)',
    )
    parser.add_argument(
        "--channel",
        default="ibm_quantum_platform",
        help="Qiskit Runtime channel (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="maximum seconds for the complete check (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    missing = [name for name in ("IBM_API_KEY", "IBM_CRN") if not os.environ.get(name)]
    if missing:
        print(f"FAIL: missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        print(f"FAIL: qiskit-ibm-runtime is not installed: {exc}", file=sys.stderr)
        return 2

    # This script targets the Unix cluster used by the submit files. SIGALRM
    # bounds calls that would otherwise hang on DNS, firewall, or proxy issues.
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout)
    started = time.monotonic()

    try:
        print(f"Connecting through channel {args.channel!r} ...", flush=True)
        service = QiskitRuntimeService(
            channel=args.channel,
            token=os.environ["IBM_API_KEY"],
            instance=os.environ["IBM_CRN"],
        )

        if args.backend == "least_busy":
            print("Fetching operational backends ...", flush=True)
            candidates = list(service.backends(operational=True))
            if not candidates:
                raise RuntimeError("the configured instance has no operational backends")
            statuses = [(backend, backend.status()) for backend in candidates]
            backend, status = min(statuses, key=lambda item: item[1].pending_jobs)
        else:
            print(f"Fetching backend {args.backend!r} ...", flush=True)
            backend = service.backend(args.backend)
            status = backend.status()

        elapsed = time.monotonic() - started
        print(
            f"PASS: authenticated and reached backend {backend.name!r} in {elapsed:.1f}s\n"
            f"      operational={status.operational}, "
            f"pending_jobs={status.pending_jobs}, status={status.status_msg!r}\n"
            "      No Runtime job was submitted."
        )
        return 0
    except CheckTimedOut:
        print(
            f"FAIL: no complete response within {args.timeout}s; check the machine's "
            "outbound HTTPS/DNS/proxy policy.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Check outbound HTTPS access, IBM_API_KEY, IBM_CRN, and backend access.",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
