# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An unhandled sampler.kind must fail, not fall back to ffsim.

A run once asked for `sampler.kind=mps` against a factory whose mps branch had been
overwritten. Nothing raised: the kind fell through to the final `else` and built a plain
FfsimSampler, so a 38q run spent an hour on exact ffsim circuits while the logs showed
neither an error nor the sampler's own output.
"""

import pytest

pytest.importorskip("ffsim")
pytest.importorskip("pyscf")

def _sampler(cfg, kind):
    from sqd_gqe.factory import SqdFactory
    cfg = cfg.copy()
    cfg.sampler.kind = kind
    return SqdFactory().build_sampler(cfg)


def test_unknown_kind_raises(n2_cfg):
    with pytest.raises(ValueError, match="Unknown sampler.kind"):
        _sampler(n2_cfg, "definitely-not-a-sampler")


def test_a_kind_whose_branch_is_missing_raises_rather_than_falling_back(n2_cfg):
    """The exact shape of the incident: a plausible kind the factory does not implement."""
    with pytest.raises(ValueError, match="sampler.kind"):
        _sampler(n2_cfg, "mps_exact")


def test_known_kinds_still_build(n2_cfg):
    from sqd_gqe.gqe.sampler import ExactTopKSampler, FfsimSampler
    from sqd_gqe.gqe.mps_sampler import MpsSampler
    from sqd_gqe.gqe.mps_topk_sampler import MpsTopKSampler

    assert isinstance(_sampler(n2_cfg, "ffsim"), FfsimSampler)
    assert isinstance(_sampler(n2_cfg, "topk"), ExactTopKSampler)
    assert isinstance(_sampler(n2_cfg, "mps"), MpsSampler)
    assert isinstance(_sampler(n2_cfg, "mps_topk"), MpsTopKSampler)


def test_mps_topk_is_not_silently_the_shot_based_sampler(n2_cfg):
    """The two kinds share a branch and a constructor; only one draws shots."""
    from sqd_gqe.gqe.mps_topk_sampler import MpsTopKSampler

    assert not isinstance(_sampler(n2_cfg, "mps"), MpsTopKSampler)


def test_mps_topk_defaults_its_k_to_max_dim(n2_cfg):
    """Returning fewer determinants than the subspace asks for would shrink it silently."""
    cfg = n2_cfg.copy()
    cfg.sqd.max_dim = 777
    cfg.sampler.top_k = None
    assert _sampler(cfg, "mps_topk").top_k == 777


def test_one_top_k_sets_both_top_k_kinds(n2_cfg):
    cfg = n2_cfg.copy()
    cfg.sampler.top_k = 123
    assert _sampler(cfg, "topk").top_k == 123
    assert _sampler(cfg, "mps_topk").top_k == 123


def test_mps_settings_reach_both_kinds(n2_cfg):
    cfg = n2_cfg.copy()
    cfg.sampler.mps_max_bond = 321
    assert _sampler(cfg, "mps").max_bond == 321
    assert _sampler(cfg, "mps_topk").max_bond == 321


def test_mps_topk_pipeline_rejects_default_postprocessing(n2_cfg):
    from sqd_gqe.factory import SqdFactory

    cfg = n2_cfg.copy()
    cfg.sampler.kind = "mps_topk"
    with pytest.raises(ValueError, match="sqd.enlarge_method=None.*sqd.recover_iterations=0"):
        SqdFactory().build_pipeline(cfg)


def test_mps_topk_pipeline_accepts_compatible_postprocessing(n2_cfg):
    from sqd_gqe.factory import SqdFactory

    cfg = n2_cfg.copy()
    cfg.sampler.kind = "mps_topk"
    cfg.sqd.enlarge_method = "None"
    cfg.sqd.recover_iterations = 0
    pipeline = SqdFactory().build_pipeline(cfg)
    assert pipeline.expand_method == "None"
    assert pipeline.recover_iterations == 0


def test_noisy_mps_draws_shots_through_the_noise_model(n2_cfg):
    from sqd_gqe.gqe.mps_sampler import NoisyMpsSampler

    cfg = n2_cfg.copy()
    cfg.sampler.p_readout = 0.01
    sampler = _sampler(cfg, "mps")
    assert isinstance(sampler, NoisyMpsSampler) and sampler.p_readout == 0.01


@pytest.mark.parametrize("kind", ["topk", "mps_topk"])
def test_top_k_kinds_refuse_device_noise(n2_cfg, kind):
    """They read amplitudes, so a noisy config would return a noiseless score."""
    cfg = n2_cfg.copy()
    cfg.sampler.p_readout = 0.01
    with pytest.raises(ValueError, match="cannot carry"):
        _sampler(cfg, kind)


@pytest.mark.parametrize("kind", ["mps", "mps_topk"])
def test_mps_gauge_and_canonical_center_reach_the_sampler(n2_cfg, kind):
    sampler = _sampler(n2_cfg, kind)
    assert sampler.gauge_option == "simple" and sampler.canonical_center is None
    cfg = n2_cfg.copy()
    cfg.sampler.mps_gauge_option = None                  # back to cuQuantum's "free"
    cfg.sampler.mps_canonical_center = 3
    sampler = _sampler(cfg, kind)
    assert sampler.gauge_option is None and sampler.canonical_center == 3


@pytest.mark.parametrize("kind", ["topk", "mps_topk"])
def test_a_predicted_max_dim_is_the_default_k(n2_cfg, monkeypatch, kind):
    """dynamic_resources can predict a max_dim other than the config's. K must follow it,
    or a larger prediction is capped at sqd.max_dim and a smaller one searches for nothing."""
    from sqd_gqe.factory import SqdFactory

    monkeypatch.setattr(SqdFactory, "_predict_resources", lambda self, cfg, system, backend: (4000, 100))
    cfg = n2_cfg.copy()
    cfg.dynamic_resources = True
    cfg.sampler.kind = kind
    cfg.sqd.enlarge_method = "None"
    cfg.sqd.recover_iterations = 0
    assert cfg.sqd.max_dim != 4000
    assert SqdFactory().build_pipeline(cfg).sampler.top_k == 4000
    cfg.sampler.top_k = 123                              # an explicit top_k still wins
    assert SqdFactory().build_pipeline(cfg).sampler.top_k == 123


@pytest.mark.parametrize("kind", ["topk", "mps_topk"])
def test_the_top_k_floor_defaults_to_one_over_shots(n2_cfg, kind):
    cfg = n2_cfg.copy()
    cfg.sampler.shots = 1000
    assert _sampler(cfg, kind).min_prob == pytest.approx(1e-3)
    cfg.sampler.min_prob = 2e-4
    assert _sampler(cfg, kind).min_prob == pytest.approx(2e-4)
    cfg.sampler.min_prob = None
    assert _sampler(cfg, kind).min_prob is None
