# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""sampler.noise_optimization_level reaches the noisy samplers.

The noise model transpiles per circuit to count two-qubit gates, and qiskit's default
level 3 segfaults on aarch64 -- a segfault the sampler's own except cannot catch, so the
knob has to be reachable from the config or a ROQUO run dies with no traceback.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("hydra")


def _cfg(*overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs_sqd")):
        return compose(config_name="default", overrides=list(overrides))


def test_the_default_is_qiskits_level_three():
    assert _cfg().sampler.noise_optimization_level == 3


def test_the_override_is_readable_by_the_factory_branch():
    cfg = _cfg("sampler.noise_optimization_level=1")
    assert int(cfg.sampler.get("noise_optimization_level", 3)) == 1


def test_the_noisy_samplers_accept_it():
    """Both noisy samplers must take the keyword the factory now passes."""
    import inspect

    from sqd_gqe.gqe.mps_sampler import NoisyMpsSampler
    from sqd_gqe.gqe.noisy_sampler import NoisyFfsimSampler

    for cls in (NoisyFfsimSampler, NoisyMpsSampler):
        sig = inspect.signature(cls.__init__)
        assert "noise_optimization_level" in sig.parameters, cls.__name__


def test_the_level_reaches_the_transpile_that_counts_gates(monkeypatch):
    """Not only accepted: the configured level is what the 2q-gate count transpiles with."""
    pytest.importorskip("ffsim")
    pytest.importorskip("pyscf")
    from omegaconf import OmegaConf

    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.gqe import circuit_metrics
    from sqd_gqe.hw.backends import get_backend

    levels = []

    def fake_depth(pool, seq, hw, optimization_level=3):
        levels.append(optimization_level)
        return {"two_qubit_gates": 10}

    monkeypatch.setattr(circuit_metrics, "circuit_depth_on_backend", fake_depth)
    cfg = _cfg("sampler.noise_optimization_level=1", "sampler.shots=100")
    cfg.molecule = OmegaConf.load(str(ROOT / "molecules" / "small" / "n2.yaml"))
    sampler = SqdFactory().build_sampler(cfg, get_backend("ibm_heron_r2_noisy_sim"))
    sampler.sample_state([0], seed=1)
    assert levels == [1]
