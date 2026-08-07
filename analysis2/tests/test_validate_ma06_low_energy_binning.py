from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis2.cache import CacheStore
from analysis2.config import PRODUCTION
from analysis2.lifetime_template_banks import LifetimeTemplateBank, save_bank_artifacts
from analysis2.profiled_statistics import PROFILED_ACCURACY_COLUMNS
from analysis2.workflows import validate_ma06_low_energy_binning as workflow


def _toy_production_bank(mass_gev: float = 0.6) -> LifetimeTemplateBank:
    edges = np.geomspace(mass_gev, 400.0, 21)
    photon = np.vstack(
        (
            np.linspace(20.0, 1.0, 20),
            np.linspace(18.0, 2.0, 20),
        )
    )
    su2 = np.vstack(
        (
            np.linspace(1.0, 20.0, 20),
            np.linspace(2.0, 18.0, 20),
        )
    )
    photon /= photon.sum(axis=1, keepdims=True)
    su2 /= su2.sum(axis=1, keepdims=True)
    return LifetimeTemplateBank(
        mass_gev=mass_gev,
        energy_edges_gev=edges,
        minimum_bin_n_eff=100.0,
        jeffreys_alpha=0.5,
        event_threshold=10.0,
        template_seed_offset=0,
        template_base_seed=54_321,
        photon_ctau_m=np.array([6.0, 60.0]),
        photon_probabilities=photon,
        photon_n_events=np.array([20.0, 10.0]),
        photon_n_events_before_ecal=np.array([40.0, 20.0]),
        photon_epsilon_ecal_weighted=np.array([0.5, 0.5]),
        photon_total_n_eff=np.array([1000.0, 1000.0]),
        photon_interval_m=np.array([6.0, 60.0]),
        su2_ctau_m=np.array([6.0, 240.0]),
        su2_probabilities=su2,
        su2_n_events=np.array([20.0, 10.0]),
        su2_n_events_before_ecal=np.array([40.0, 20.0]),
        su2_epsilon_ecal_weighted=np.array([0.5, 0.5]),
        su2_total_n_eff=np.array([1000.0, 1000.0]),
        su2_interval_m=np.array([6.0, 240.0]),
        profile="production",
        selection_name="diphoton_ecal",
    )


def _write_bank(path: Path, mass_gev: float = 0.6) -> None:
    save_bank_artifacts(
        _toy_production_bank(mass_gev),
        bank_path=path,
        summary_path=path.parent / "template_summary.csv",
        probability_path=path.parent / "template_probabilities.csv",
    )


def _fake_profiled_seed(
    bank,
    bank_path,
    config,
    seed,
    event_counts,
    cache,
    *,
    force,
):
    del bank_path, cache, force
    factor = config.profiled_likelihood.rebin_factor
    rows = []
    seed_penalty = 1.0e-4 * config.profiled_likelihood.seeds.index(seed)
    for truth_model, lifetimes in (
        ("photon", bank.photon_ctau_m),
        ("su2", bank.su2_ctau_m),
    ):
        for truth_index, ctau_m in enumerate(lifetimes):
            truth_bonus = 0.003 * (truth_model == "su2") + 0.002 * truth_index
            for number_of_events in event_counts:
                factor_penalty = {1: 0.0, 2: 0.01, 4: 0.07}[factor]
                accuracy = min(
                    0.999,
                    0.82
                    + 0.06 * int(number_of_events)
                    - factor_penalty
                    + truth_bonus
                    - seed_penalty,
                )
                rows.append(
                    {
                        "mass_GeV": bank.mass_gev,
                        "seed": seed,
                        "truth_model": truth_model,
                        "truth_lifetime_index": truth_index,
                        "truth_ctau_m": ctau_m,
                        "number_of_events": number_of_events,
                        "number_of_pseudoexperiments": 100_000,
                        "correct_fraction": accuracy,
                        "selected_photon_fraction": 0.5,
                        "selected_su2_fraction": 0.5,
                        "tie_fraction": 0.0,
                        "mean_profile_statistic_T": 0.0,
                        "std_profile_statistic_T": 1.0,
                    }
                )
    return pd.DataFrame(rows, columns=PROFILED_ACCURACY_COLUMNS)


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_ma06_validation_is_finite_isolated_and_reports_threshold_changes(
    tmp_path,
    monkeypatch,
):
    production_root = tmp_path / "outputs" / "production"
    bank_path = production_root / "banks" / "template_bank_ma_0p6.npz"
    _write_bank(bank_path)
    before = _snapshot_files(production_root)
    output_dir = tmp_path / "outputs" / "validation" / "ma06_low_energy_binning"
    monkeypatch.setattr(workflow, "cached_profiled_seed", _fake_profiled_seed)

    with pytest.raises(ValueError, match="production output tree"):
        workflow.run_validation(
            bank_path=bank_path,
            output_dir=production_root / "forbidden_validation",
            production_output_root=production_root,
        )

    result = workflow.run_validation(
        bank_path=bank_path,
        output_dir=output_dir,
        cache=CacheStore("production", tmp_path / "cache"),
        production_output_root=production_root,
    )

    assert set(result.summary["mass_GeV"]) == {0.6}
    assert result.summary["rebin_factor"].tolist() == [1, 2, 4]
    assert result.summary["number_of_final_adaptive_energy_bins"].tolist() == [
        20,
        10,
        5,
    ]
    numeric = result.summary.select_dtypes(include=np.number)
    assert np.isfinite(numeric.to_numpy()).all()
    assert result.summary["persistent_90_required_event_count"].tolist() == [
        2,
        2,
        3,
    ]
    for row in result.summary.itertuples(index=False):
        if row.required_event_threshold_stable:
            assert row.required_event_difference_from_nominal == 0
            assert row.threshold_change_report == "stable"
        else:
            assert row.required_event_difference_from_nominal != 0
            assert row.threshold_change_report.startswith("changed by")
    assert not result.summary["overall_nominal_conclusion_robust"].any()
    assert _snapshot_files(production_root) == before
    assert {path.name for path in output_dir.iterdir()} == {
        "ma06_low_energy_binning_summary.csv",
        "manifest.json",
    }

    wrong_mass_path = tmp_path / "template_bank_ma_0p7.npz"
    _write_bank(wrong_mass_path, mass_gev=0.7)
    with pytest.raises(ValueError, match="only m_a=0.6 GeV"):
        workflow.run_validation(
            bank_path=wrong_mass_path,
            output_dir=tmp_path / "wrong_mass_output",
            cache=CacheStore("production", tmp_path / "wrong_mass_cache"),
            production_output_root=production_root,
        )
