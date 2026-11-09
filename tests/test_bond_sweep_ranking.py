# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""GPU-free checks of sweep measurements, seed aggregation and ranking diagnostics."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "mps_bond_sweep", ROOT / "benchmark" / "mps_bond_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def _cfg_args(sampler_kind):
    return Namespace(
        molecule="n2", sampler_kind=sampler_kind, shots=100, max_dim=10,
        diag_backend="sqd", sbd_binary_path=None,
    )


def test_make_cfg_uses_the_requested_sampler_kind():
    assert sweep.make_cfg(_cfg_args("mps")).sampler.kind == "mps"
    assert sweep.make_cfg(_cfg_args("mps_topk")).sampler.kind == "mps_topk"


def test_make_cfg_disables_expansion_for_mps_topk_only():
    assert sweep.make_cfg(_cfg_args("mps")).sqd.enlarge_method == "symmetry_completion"
    assert sweep.make_cfg(_cfg_args("mps_topk")).sqd.enlarge_method == "None"


def test_identical_orderings_give_tau_one():
    assert sweep._kendall_tau([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0


def test_reversed_orderings_give_tau_minus_one():
    assert sweep._kendall_tau([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == -1.0


def test_one_swap_out_of_three_pairs():
    # (a,b) and (a,c) agree, (b,c) is inverted -> (2 - 1) / 3
    assert sweep._kendall_tau([1.0, 2.0, 3.0], [10.0, 30.0, 20.0]) == 1 / 3


def test_all_ties_give_none():
    assert sweep._kendall_tau([1.0, 1.0], [5.0, 5.0]) is None


def _record(rows):
    return _seeded([(c, s, 0, e) for c, s, e in rows])


def test_warns_when_the_chi_shift_exceeds_the_spread(capsys):
    """The 46q case: circuits 4 mHa apart, moved 16 mHa by the last doubling of chi."""
    sweep._rank_stability(_record([
        (512, "a", -479.2553), (512, "b", -479.2560),
        (1024, "a", -479.2718), (1024, "b", -479.2673),
    ]))
    out = capsys.readouterr().out
    assert "not resolved here" in out


def test_silent_when_the_circuits_separate_cleanly(capsys):
    """Circuits 50 mHa apart, chi moving them 1 mHa: the ordering is real."""
    sweep._rank_stability(_record([
        (512, "a", -100.000), (512, "b", -100.050),
        (1024, "a", -100.001), (1024, "b", -100.051),
    ]))
    out = capsys.readouterr().out
    assert "not resolved here" not in out
    assert "+1.00" in out           # ordering preserved


def test_a_single_chi_produces_no_summary(capsys):
    sweep._rank_stability(_record([(512, "a", -1.0), (512, "b", -2.0)]))
    assert capsys.readouterr().out == ""


def test_rows_without_energies_are_skipped(capsys):
    """A sweep whose diagonalizer failed must not be summarised as if it had worked."""
    record = _record([(512, "a", None), (1024, "b", None)])
    sweep._rank_stability(record)
    assert capsys.readouterr().out == ""


def _seeded(rows):
    return {"molecule": "x", "qubits": 38, "rows": [
        {"chi": c, "sequence": s, "seed": sd, "energy": e, "valid_fraction": 1.0,
         "n_distinct": 1, "t_sample_s": 0.0} for c, s, sd, e in rows]}


def test_seed_noise_larger_than_the_spread_is_called_out(capsys):
    """38q's open question: circuits 3 mHa apart, one moving 8 mHa between seeds."""
    sweep._rank_stability(_seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.008),
        (0, "b", 0, -100.003), (0, "b", 1, -100.003),
        (512, "a", 0, -100.000), (512, "b", 0, -100.003),
    ]))
    assert "Raise shots, not chi" in capsys.readouterr().out


def test_small_seed_noise_is_reported_without_the_warning(capsys):
    sweep._rank_stability(_seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.0001),
        (0, "b", 0, -100.050), (0, "b", 1, -100.050),
        (512, "a", 0, -100.000), (512, "b", 0, -100.050),
    ]))
    out = capsys.readouterr().out
    assert "across sampling seeds" in out
    assert "Raise shots, not chi" not in out


@pytest.mark.parametrize("ref_chi", [0, 1024])
def test_reference_rank_reversal_is_not_hidden_by_a_distant_circuit(ref_chi, capsys):
    sweep._rank_stability(_seeded([
        (ref_chi, "a", 0, -100.000), (ref_chi, "a", 1, -100.004),
        (ref_chi, "b", 1, -100.001), (ref_chi, "b", 0, -100.001),
        (ref_chi, "c", 0, -100.100), (ref_chi, "c", 1, -100.100),
        (512, "a", 0, -100.002), (512, "b", 0, -100.001),
        (512, "c", 0, -100.100),
    ]))
    out = capsys.readouterr().out
    assert "reference ordering reverses across sampling seeds for: a/b" in out
    assert "Raise shots, not chi" in out


def test_common_seed_offset_does_not_make_reference_ranking_uncertain(capsys):
    sweep._rank_stability(_seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.020),
        (0, "b", 1, -100.021), (0, "b", 0, -100.001),
        (512, "a", 0, -100.010), (512, "b", 0, -100.011),
    ]))
    out = capsys.readouterr().out
    assert "across sampling seeds: 20.0 mHa" in out
    assert "reference ordering reverses" not in out
    assert "Raise shots" not in out


def test_reference_rank_reversal_requires_matching_seed_ids(capsys):
    sweep._rank_stability(_seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.004),
        (0, "b", 1, -100.005), (0, "b", 2, -100.001),
        (512, "a", 0, -100.002), (512, "b", 0, -100.003),
    ]))
    out = capsys.readouterr().out
    assert "reference ordering reverses" not in out
    assert "Raise shots" not in out


def test_failed_reference_seed_group_does_not_get_a_rank_reversal_warning(capsys):
    sweep._rank_stability(_seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.004), (0, "a", 2, None),
        (0, "b", 0, -100.001), (0, "b", 1, -100.001), (0, "b", 2, -100.001),
        (0, "c", 0, -100.100), (0, "c", 1, -100.100), (0, "c", 2, -100.100),
        (512, "b", 0, -100.001), (512, "c", 0, -100.100),
    ]))
    out = capsys.readouterr().out
    assert "reference ordering reverses" not in out
    assert "Raise shots" not in out


def test_energies_are_averaged_over_seeds_before_ranking(capsys):
    """Ranking must use the seed mean, not whichever draw happened to come last."""
    sweep._rank_stability(_seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.000),
        (0, "b", 0, -100.010), (0, "b", 1, -100.010),
        (512, "a", 0, -100.000), (512, "a", 1, -100.040),
        (512, "b", 0, -100.010), (512, "b", 1, -100.010),
    ]))
    out = capsys.readouterr().out
    assert "a < b" in out          # mean(a) = -100.020 beats b = -100.010


# -- determinant-set convergence: the only such measure where no exact reference exists -- #

def test_overlap_of_one_when_the_set_stops_changing(capsys):
    keys = {(1, 2), (3, 4), (5, 6)}
    sweep._determinant_convergence({
        (256, "a", 0): set(keys), (512, "a", 0): set(keys),
    })
    out = capsys.readouterr().out
    assert "256->512" in out
    assert "1.000" in out


def test_overlap_of_zero_when_the_set_is_replaced(capsys):
    sweep._determinant_convergence({
        (256, "a", 0): {(1, 2), (3, 4)}, (512, "a", 0): {(5, 6), (7, 8)},
    })
    assert "0.000" in capsys.readouterr().out


def test_partial_overlap_is_a_fraction_of_the_lower_chi_set(capsys):
    sweep._determinant_convergence({
        (256, "a", 0): {(1, 2), (3, 4), (5, 6), (7, 8)},
        (512, "a", 0): {(1, 2), (3, 4), (9, 9)},
    })
    assert "0.500" in capsys.readouterr().out


def test_the_exact_reference_is_not_treated_as_a_rung(capsys):
    """chi=0 is exact selection, not the smallest bond dimension."""
    sweep._determinant_convergence({
        (0, "a", 0): {(1, 2)}, (256, "a", 0): {(1, 2)}, (512, "a", 0): {(1, 2)},
    })
    out = capsys.readouterr().out
    assert "0->256" not in out
    assert "256->512" in out


def test_a_single_chi_produces_no_convergence_table(capsys):
    sweep._determinant_convergence({(256, "a", 0): {(1, 2)}})
    assert capsys.readouterr().out == ""


def test_seeds_are_averaged(capsys):
    """One seed keeps its set, the other replaces it: the pair averages to 0.500."""
    sweep._determinant_convergence({
        (256, "a", 0): {(1, 2)}, (512, "a", 0): {(1, 2)},
        (256, "a", 1): {(3, 4)}, (512, "a", 1): {(9, 9)},
    })
    assert "0.500" in capsys.readouterr().out


def test_opposite_chi_shifts_do_not_cancel(capsys):
    sweep._rank_stability(_record([
        (512, "a", -100.000), (512, "b", -100.038),
        (1024, "a", -100.020), (1024, "b", -100.018),
    ]))
    out = capsys.readouterr().out
    assert "20.0 mHa" in out
    assert "not resolved here" in out


def test_close_pair_is_not_hidden_by_a_distant_circuit(capsys):
    sweep._rank_stability(_record([
        (512, "a", -100.000), (512, "b", -100.002), (512, "c", -100.100),
        (1024, "a", -100.004), (1024, "b", -100.002), (1024, "c", -100.100),
    ]))
    assert "not resolved here" in capsys.readouterr().out


def test_common_energy_shift_does_not_change_ranking(capsys):
    sweep._rank_stability(_record([
        (512, "a", -100.000), (512, "b", -100.002),
        (1024, "a", -100.020), (1024, "b", -100.022),
    ]))
    out = capsys.readouterr().out
    assert "20.0 mHa" in out
    assert "not resolved here" not in out


def test_low_chi_variation_does_not_invalidate_the_exact_reference(capsys):
    sweep._rank_stability(_seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.000),
        (0, "b", 0, -100.002), (0, "b", 1, -100.002),
        (64, "a", 0, -99.980), (64, "a", 1, -99.960),
        (64, "b", 0, -99.990), (64, "b", 1, -99.990),
    ]))
    out = capsys.readouterr().out
    assert "20.0" in out
    assert "reference ordering reverses" not in out
    assert "Raise shots" not in out


def test_topk_variation_does_not_recommend_more_shots(capsys):
    record = _seeded([
        (0, "a", 0, -100.000), (0, "a", 1, -100.008),
        (0, "b", 0, -100.003), (0, "b", 1, -100.003),
        (512, "a", 0, -100.000), (512, "b", 0, -100.003),
    ])
    record["sampler_kind"] = "mps_topk"
    sweep._rank_stability(record)
    out = capsys.readouterr().out
    assert "numerical" in out
    assert "Raise shots" not in out


def test_summary_deltas_compare_seed_means_between_chis(capsys):
    sweep._summarise(_seeded([
        (256, "a", 0, -100.000), (256, "a", 1, -100.010),
        (512, "a", 0, -100.001), (512, "a", 1, -100.011),
    ]))
    out = capsys.readouterr().out
    assert "-100.005000" in out and "-100.006000" in out
    assert "-1.000" in out
    assert "-10.000" not in out and "+9.000" not in out


def test_incomplete_seed_groups_are_marked_and_excluded_from_ranking(capsys):
    record = _seeded([
        (0, "a", 0, -100.000), (0, "b", 0, -100.001), (0, "c", 0, -100.002),
        (512, "a", 0, -101.0), (512, "a", 1, None),
        (512, "b", 0, -100.001), (512, "c", 0, -100.002),
    ])
    sweep._summarise(record)
    assert "1/2" in capsys.readouterr().out
    sweep._rank_stability(record)
    line = next(line.split() for line in capsys.readouterr().out.splitlines()
                if line.split() and line.split()[0] == "512")
    assert line[1] == "2"
    assert "a" not in line


@pytest.mark.parametrize("energy", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurements_are_reported_as_failures(energy):
    sampler = SimpleNamespace(sample_state=lambda seq, seed: {(1, 1): 1})
    pipeline = SimpleNamespace(energy_of_counts=lambda counts: energy)
    row, keys = sweep.measure(sampler, pipeline, [0], chi=64, name="a", seed=0)
    assert row["energy"] is None
    assert "non-finite energy" in row["error"]
    assert keys == {(1, 1)}
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("energy", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rows_never_get_a_rank_correlation(energy, capsys):
    record = _record([
        (0, "a", -100.002), (0, "b", -100.000),
        (512, "a", energy), (512, "b", energy),
    ])
    sweep._rank_stability(record)
    assert capsys.readouterr().out == ""
    sweep._summarise(record)
    assert "n/a" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["mps", "mps_topk"])
def test_failed_measurement_does_not_stop_the_sweep(kind, tmp_path, monkeypatch):
    from sqd_gqe import factory

    calls, kinds = [], []

    class FakeFactory:
        def build_system(self, cfg):
            return SimpleNamespace(norb=1, nelec=(1, 1), fci_dimension=lambda: 1)

        def build_operator_pool(self, cfg):
            return []

        def build_pipeline(self, cfg):
            def energy_of_counts(counts):
                calls.append(counts)
                return float("nan") if len(calls) == 1 else -1.0
            return SimpleNamespace(energy_of_counts=energy_of_counts)

        def build_sampler(self, cfg):
            kinds.append(cfg.sampler.kind)
            return SimpleNamespace(sample_state=lambda seq, seed: {(1, 1): 1})

    monkeypatch.setattr(factory, "SqdFactory", FakeFactory)
    monkeypatch.setattr(sweep, "build_sequences", lambda *a, **kw: {"a": [0]})
    monkeypatch.setattr(sys, "argv", [
        "mps_bond_sweep.py", "--molecule", "n2", "--diag-backend", "sqd",
        "--sampler-kind", kind, "--with-exact", "--seeds", "2",
        "--bonds", "64", "128", "--sequences", "a", "--out", str(tmp_path),
    ])
    sweep.main()
    report = json.loads((tmp_path / "n2.json").read_text())
    assert report["sampler_kind"] == kind
    assert len(report["rows"]) == 6
    assert report["rows"][0]["energy"] is None
    assert all(row["energy"] == -1.0 for row in report["rows"][1:])
    assert kinds == ["topk" if kind == "mps_topk" else "ffsim", kind, kind]


def test_determinant_overlap_matches_seed_ids_not_insertion_order(capsys):
    sweep._determinant_convergence({
        (256, "a", 0): {(1, 2)}, (256, "a", 1): {(3, 4)},
        (512, "a", 1): {(3, 4)}, (512, "a", 0): {(1, 2)},
    })
    assert "1.000" in capsys.readouterr().out


def test_set_overrides_reach_the_config_and_win():
    pytest.importorskip("hydra")
    args = Namespace(molecule="n2", sampler_kind="mps", shots=10, max_dim=10,
                     diag_backend="sqd", sbd_binary_path=None,
                     set=["sampler.mps_dtype=complex128", "sampler.mps_abs_cutoff=0",
                          "sqd.recover_iterations=2"])
    cfg = sweep.make_cfg(args)
    assert cfg.sampler.mps_dtype == "complex128"
    assert float(cfg.sampler.mps_abs_cutoff) == 0.0
    assert cfg.sqd.recover_iterations == 2          # beats the sweep's own recover_iterations=0


def _run_capturing_configs(tmp_path, monkeypatch, *extra):
    """Run main() on a fake factory and return (configs passed to build_sampler, report)."""
    from sqd_gqe import factory

    seen = []

    class FakeFactory:
        def build_system(self, cfg):
            return SimpleNamespace(norb=1, nelec=(1, 1), fci_dimension=lambda: 1)

        def build_operator_pool(self, cfg):
            return []

        def build_pipeline(self, cfg):
            return SimpleNamespace(energy_of_counts=lambda counts: -1.0)

        def build_sampler(self, cfg):
            seen.append(cfg.copy())
            return SimpleNamespace(sample_state=lambda seq, seed: {(1, 1): 1})

    monkeypatch.setattr(factory, "SqdFactory", FakeFactory)
    monkeypatch.setattr(sweep, "build_sequences", lambda *a, **kw: {"a": [0]})
    monkeypatch.setattr(sys, "argv", [
        "mps_bond_sweep.py", "--molecule", "n2", "--diag-backend", "sqd",
        "--sampler-kind", "mps_topk", "--bonds", "64", "128", "--sequences", "a",
        "--out", str(tmp_path), *extra,
    ])
    sweep.main()
    return seen, json.loads((tmp_path / "n2.json").read_text())


def test_set_overrides_survive_every_run_and_reach_the_report(tmp_path, monkeypatch):
    """--set sampler.mps_beam_width=32000 used to be reset to None per chi, and the report
    recorded the CLI's shots instead of the composed config's."""
    seen, report = _run_capturing_configs(
        tmp_path, monkeypatch, "--set", "sampler.mps_beam_width=32000", "sampler.shots=10",
        "sqd.max_dim=77")
    assert [c.sampler.mps_max_bond for c in seen] == [64, 128]          # only chi is swept
    assert all(c.sampler.mps_beam_width == 32000 and c.sampler.shots == 10 for c in seen)
    assert (report["shots"], report["max_dim"], report["beam_width"]) == (10, 77, 32000)
    assert report["sampler"]["mps_beam_width"] == 32000


def test_beam_width_flag_applies_and_set_wins_over_it(tmp_path, monkeypatch):
    seen, _ = _run_capturing_configs(tmp_path, monkeypatch, "--beam-width", "7")
    assert all(c.sampler.mps_beam_width == 7 for c in seen)
    seen, _ = _run_capturing_configs(tmp_path, monkeypatch, "--beam-width", "7",
                                     "--set", "sampler.mps_beam_width=9")
    assert all(c.sampler.mps_beam_width == 9 for c in seen)


@pytest.mark.parametrize("key", ["sampler.kind=mps", "sampler.mps_max_bond=32"])
def test_set_refuses_the_keys_the_sweep_varies(key):
    with pytest.raises(SystemExit):
        sweep._parse_args(["--set", key])
