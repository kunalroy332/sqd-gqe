# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Create the QiskitRuntimeService for the fetch scripts without ever printing credentials.

If IBM_API_KEY and IBM_CRN are both set in your shell, they are used (this selects the same account and instance the cluster used);
otherwise the account saved with QiskitRuntimeService.save_account is used.
"""
import os


def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    key, crn = os.environ.get("IBM_API_KEY"), os.environ.get("IBM_CRN")
    if key and crn:
        print("using account from IBM_API_KEY / IBM_CRN (values not printed)")
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=key, instance=crn)
    print("using the saved IBM account (set IBM_API_KEY and IBM_CRN to override)")
    return QiskitRuntimeService()


def diagnose(service, backend="ibm_kobe", n=5):
    """Print what this account can see: instance names and its most recent jobs (ids, backend, creation time) -- no secrets."""
    try:
        for inst in service.instances():
            print("visible instance:", inst.get("name"))
    except Exception as e:  # noqa: BLE001
        print("could not list instances:", type(e).__name__)
    try:
        jobs = service.jobs(limit=n, backend_name=backend)
        print(f"{len(jobs)} most recent {backend} jobs visible to this account:")
        for j in jobs:
            print("  ", j.job_id(), j.creation_date)
    except Exception as e:  # noqa: BLE001
        print("could not list jobs:", type(e).__name__, str(e)[:200])
