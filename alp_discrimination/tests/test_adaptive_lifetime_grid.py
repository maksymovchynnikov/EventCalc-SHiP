from pathlib import Path

import numpy as np
import pandas as pd

from alp_discrimination.statistics.adaptive_grid import (
    AdaptiveLifetimeSettings,
    AdaptivePseudoexperimentSettings,
    AdaptiveScanSettings,
    audit_omitted_truths,
    binning_is_stable,
    distance_screening_truth_indices,
    event_grid_specification,
    final_event_grid_from_bracket,
    initial_adaptive_lifetime_grid,
    monte_carlo_threshold_diagnostics,
    propose_lifetime_refinement,
    rangefinder_bracket,
    select_hard_truth_indices,
    threshold_history_is_stable,
    total_variation_matrix,
)
from alp_discrimination.templates.lifetime_banks import LifetimeTemplateBank
from alp_discrimination.statistics.reduction import (
    build_conservative_seed_envelope,
    build_seed_worst_case_table,
)
from alp_discrimination.workflows.lifetime_bank_builder import (
    parse_arguments,
    settings_from_arguments,
)
from alp_discrimination.workflows.plot_n90_comparison import (
    plot_n90_comparison,
)
from alp_discrimination.workflows.validate_adaptive_lifetime_grid import (
    calibrate_dense_bank,
)


def adaptive_toy_bank() -> LifetimeTemplateBank:
    photon = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.68, 0.24, 0.08],
            [0.51, 0.35, 0.14],
            [0.50, 0.35, 0.15],
        ]
    )
    su2 = np.array(
        [
            [0.52, 0.34, 0.14],
            [0.51, 0.34, 0.15],
            [0.50, 0.35, 0.15],
            [0.49, 0.35, 0.16],
        ]
    )
    common = dict(
        mass_gev=0.3,
        energy_edges_gev=np.array([0.3, 2.0, 20.0, 400.0]),
        minimum_bin_n_eff=100.0,
        jeffreys_alpha=0.5,
        event_threshold=2.3,
        template_seed_offset=0,
        template_base_seed=54_321,
        photon_ctau_m=np.array([0.01, 0.1, 10.0, 100.0]),
        photon_probabilities=photon,
        photon_n_events=np.full(4, 100.0),
        photon_n_events_before_ecal=np.full(4, 200.0),
        photon_epsilon_ecal_weighted=np.full(4, 0.5),
        photon_total_n_eff=np.full(4, 1000.0),
        photon_interval_m=np.array([0.01, 100.0]),
        photon_interval_index=np.array([0, 0, 1, 1]),
        photon_allowed_intervals_m=np.array([[0.01, 0.1], [10.0, 100.0]]),
        su2_ctau_m=np.array([0.02, 0.2, 20.0, 200.0]),
        su2_probabilities=su2,
        su2_n_events=np.full(4, 100.0),
        su2_n_events_before_ecal=np.full(4, 200.0),
        su2_epsilon_ecal_weighted=np.full(4, 0.5),
        su2_total_n_eff=np.full(4, 1000.0),
        su2_interval_m=np.array([0.02, 200.0]),
        su2_interval_index=np.array([0, 0, 1, 1]),
        su2_allowed_intervals_m=np.array([[0.02, 0.2], [20.0, 200.0]]),
        profile="validation",
        selection_name="diphoton_ecal",
    )
    return LifetimeTemplateBank(**common)


def _toy_grid_from_bank(bank: LifetimeTemplateBank) -> pd.DataFrame:
    rows = []
    for model, ctaus, intervals in (
        ("ALP-photon-combined", bank.photon_ctau_m, bank.photon_interval_index),
        ("ALP-SU2L", bank.su2_ctau_m, bank.su2_interval_index),
    ):
        for ctau, interval in zip(ctaus, intervals):
            rows.append(
                {
                    "model": model,
                    "mass_GeV": bank.mass_gev,
                    "interval_index": int(interval),
                    "ctau_m": float(ctau),
                }
            )
    return pd.DataFrame(rows)


def test_calibrated_lifetime_defaults_are_width_scaled_and_batched():
    settings = AdaptiveLifetimeSettings()
    assert settings.initial_points_per_decade == 4.0
    assert settings.maximum_log_gap_decades == 0.25
    assert settings.maximum_rounds == 8
    assert settings.maximum_new_points_per_model_per_round == 16
    assert settings.maximum_soft_priority_at_convergence == 6.0

    args = parse_arguments(["--masses", "0.3"])
    cli_settings = settings_from_arguments(args).lifetime
    assert cli_settings == settings


def test_soft_only_curvature_does_not_block_stable_convergence():
    bank = adaptive_toy_bank()
    distances = total_variation_matrix(bank)
    minimum = float(distances.min())
    settings = AdaptiveLifetimeSettings(
        maximum_log_gap_decades=2.0,
        minimum_region_log_gap_decades=2.0,
        maximum_adjacent_template_tv=0.02,
        maximum_log_interpolation_tv=1.0,
        maximum_adjacent_distance_change=1.0,
        maximum_soft_priority_at_convergence=10.0,
    )
    decision = propose_lifetime_refinement(
        bank,
        distances,
        _toy_grid_from_bank(bank),
        settings,
        round_index=2,
        previous_minimum_distance=minimum,
    )
    assert decision.converged
    assert decision.additions.empty
    diagnostics = decision.diagnostics
    assert diagnostics["exceeds_nominal_tolerance"].any()
    assert not diagnostics["required_for_convergence"].any()


def test_extreme_soft_curvature_still_requires_refinement():
    bank = adaptive_toy_bank()
    distances = total_variation_matrix(bank)
    minimum = float(distances.min())
    settings = AdaptiveLifetimeSettings(
        maximum_log_gap_decades=2.0,
        minimum_region_log_gap_decades=2.0,
        maximum_adjacent_template_tv=0.02,
        maximum_log_interpolation_tv=1.0,
        maximum_adjacent_distance_change=1.0,
        maximum_soft_priority_at_convergence=5.0,
        maximum_new_points_per_model_per_round=1,
    )
    decision = propose_lifetime_refinement(
        bank,
        distances,
        _toy_grid_from_bank(bank),
        settings,
        round_index=2,
        previous_minimum_distance=minimum,
    )
    assert not decision.converged
    assert not decision.additions.empty
    selected = decision.diagnostics.loc[
        decision.diagnostics["selected_for_refinement"]
    ]
    assert selected["required_for_convergence"].all()
    assert selected.groupby("truth_model").size().max() <= 1


def test_initial_lifetime_grid_scales_with_log_width_and_retains_components():
    domains = pd.DataFrame(
        [
            {
                "model": "ALP-photon-combined",
                "mass_GeV": 0.3,
                "interval_index": 0,
                "ctau_min_m": 1.0e-3,
                "ctau_max_m": 1.0e3,
            },
            {
                "model": "ALP-photon-combined",
                "mass_GeV": 0.3,
                "interval_index": 1,
                "ctau_min_m": 2.0,
                "ctau_max_m": 4.0,
            },
            {
                "model": "ALP-SU2L",
                "mass_GeV": 0.3,
                "interval_index": 0,
                "ctau_min_m": 1.0,
                "ctau_max_m": 10.0,
            },
        ]
    )
    settings = AdaptiveLifetimeSettings(
        initial_points_per_decade=4.0,
        minimum_points_per_interval=3,
    )
    grid = initial_adaptive_lifetime_grid(domains, 0.3, settings)
    counts = grid.groupby(["model", "interval_index"]).size()
    assert counts[("ALP-photon-combined", 0)] > counts[("ALP-SU2L", 0)]
    assert counts[("ALP-photon-combined", 1)] == 3
    for _, group in grid.groupby(["model", "interval_index"]):
        ordered = group.sort_values("ctau_m")
        assert bool(ordered.iloc[0]["is_interval_endpoint"])
        assert bool(ordered.iloc[-1]["is_interval_endpoint"])


def test_refinement_is_axis_sensitive_and_never_bridges_gaps():
    bank = adaptive_toy_bank()
    distances = total_variation_matrix(bank)
    grid_rows = []
    for model, ctaus, intervals in (
        ("ALP-photon-combined", bank.photon_ctau_m, bank.photon_interval_index),
        ("ALP-SU2L", bank.su2_ctau_m, bank.su2_interval_index),
    ):
        for ctau, interval in zip(ctaus, intervals):
            grid_rows.append(
                {
                    "model": model,
                    "mass_GeV": 0.3,
                    "interval_index": int(interval),
                    "ctau_m": float(ctau),
                }
            )
    settings = AdaptiveLifetimeSettings(
        maximum_log_gap_decades=2.0,
        minimum_region_log_gap_decades=2.0,
        maximum_adjacent_template_tv=0.05,
        maximum_adjacent_distance_change=0.05,
        maximum_rounds=3,
    )
    decision = propose_lifetime_refinement(
        bank,
        distances,
        pd.DataFrame(grid_rows),
        settings,
        round_index=0,
    )
    additions = decision.additions
    assert not additions.empty
    # The photon templates vary much more rapidly than the SU(2)_L templates.
    counts = additions.groupby("model").size().to_dict()
    assert counts.get("ALP-photon-combined", 0) > counts.get("ALP-SU2L", 0)
    for row in additions.itertuples(index=False):
        if row.model == "ALP-photon-combined":
            allowed = bank.photon_allowed_intervals_m[row.interval_index]
        else:
            allowed = bank.su2_allowed_intervals_m[row.interval_index]
        assert allowed[0] < row.ctau_m < allowed[1]


def test_lifetime_size_cap_is_not_misreported_as_convergence():
    bank = adaptive_toy_bank()
    rows = []
    for model, ctaus, intervals in (
        ("ALP-photon-combined", bank.photon_ctau_m, bank.photon_interval_index),
        ("ALP-SU2L", bank.su2_ctau_m, bank.su2_interval_index),
    ):
        for ctau, interval in zip(ctaus, intervals):
            rows.append(
                {
                    "model": model,
                    "mass_GeV": 0.3,
                    "interval_index": int(interval),
                    "ctau_m": float(ctau),
                }
            )
    settings = AdaptiveLifetimeSettings(
        maximum_log_gap_decades=0.01,
        maximum_total_lifetimes_per_model=4,
        maximum_rounds=2,
    )
    decision = propose_lifetime_refinement(
        bank,
        total_variation_matrix(bank),
        pd.DataFrame(rows),
        settings,
        round_index=0,
    )
    assert decision.additions.empty
    assert decision.reached_size_limit
    assert not decision.converged


def test_distance_screening_includes_minimum_neighbours_and_interval_endpoints():
    bank = adaptive_toy_bank()
    distances = total_variation_matrix(bank)
    settings = AdaptivePseudoexperimentSettings(
        screening_truths_per_model=1,
        screening_neighbourhood=1,
    )
    selected = distance_screening_truth_indices(bank, distances, settings)
    for model, intervals in (
        ("photon", bank.photon_interval_index),
        ("su2", bank.su2_interval_index),
    ):
        for interval in np.unique(intervals):
            indices = np.flatnonzero(intervals == interval)
            assert int(indices[0]) in selected[model]
            assert int(indices[-1]) in selected[model]



def test_rangefinder_supports_single_observed_event():
    settings = AdaptivePseudoexperimentSettings()
    assert settings.rangefinder_minimum_events == 1
    curve = pd.DataFrame(
        {
            "number_of_events": [1, 2, 3],
            "worst_case_correct_fraction": [0.91, 0.94, 0.97],
        }
    )
    bracket = rangefinder_bracket(curve, settings)
    assert bracket.lower_failing_events == 0
    assert bracket.upper_passing_events == 1
    grid = final_event_grid_from_bracket(bracket, settings)
    assert grid[0] == 1

def test_rangefinder_builds_a_single_cache_stable_unit_window_and_tail():
    curve = pd.DataFrame(
        {
            "number_of_events": [50, 100, 200, 400],
            "worst_case_correct_fraction": [0.60, 0.82, 0.94, 0.99],
        }
    )
    settings = AdaptivePseudoexperimentSettings()
    bracket = rangefinder_bracket(curve, settings)
    assert bracket.lower_failing_events == 100
    assert bracket.upper_passing_events == 200
    grid = final_event_grid_from_bracket(bracket, settings)
    assert bracket.lower_failing_events in grid
    assert bracket.upper_passing_events in grid
    estimated = int(round(bracket.estimated_crossing_events))
    assert estimated - 1 in grid
    assert estimated in grid
    assert grid[-1] > bracket.upper_passing_events
    specification = event_grid_specification(grid)
    assert ":" in specification


def _toy_detailed_table() -> pd.DataFrame:
    rows = []
    probabilities = {
        ("photon", 0): [0.89, 0.91, 0.94],
        ("photon", 1): [0.95, 0.97, 0.99],
        ("su2", 0): [0.90, 0.92, 0.95],
        ("su2", 1): [0.96, 0.98, 0.995],
    }
    for seed in (10, 20):
        for (model, index), values in probabilities.items():
            for event, value in zip((10, 11, 20), values):
                rows.append(
                    {
                        "mass_GeV": 0.3,
                        "seed": seed,
                        "truth_model": model,
                        "truth_lifetime_index": index,
                        "truth_interval_index": index,
                        "truth_ctau_m": float(index + 1),
                        "number_of_events": event,
                        "number_of_pseudoexperiments": 2000,
                        "correct_fraction": value - (0.002 if seed == 20 else 0.0),
                        "selected_photon_fraction": 0.5,
                        "selected_su2_fraction": 0.5,
                        "tie_fraction": 0.0,
                        "mean_profile_statistic_T": 0.0,
                        "std_profile_statistic_T": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_hard_truth_selection_and_omitted_audit_are_statistical_not_fixed_counts():
    bank = adaptive_toy_bank()
    detailed = _toy_detailed_table()
    curve = build_conservative_seed_envelope(build_seed_worst_case_table(detailed))
    settings = AdaptivePseudoexperimentSettings(
        hard_truth_accuracy_gap=0.02,
        minimum_hard_truths_per_model=1,
        maximum_hard_truth_fraction_per_model=1.0,
        screening_truths_per_model=1,
        screening_neighbourhood=0,
    )
    selected, ranking = select_hard_truth_indices(
        bank,
        detailed,
        curve,
        total_variation_matrix(bank),
        [10, 11, 20],
        settings,
    )
    assert set(selected) == {"photon", "su2"}
    assert ranking["selected_for_high_statistics"].any()

    omitted = detailed.loc[
        (detailed["truth_model"] == "photon")
        & (detailed["truth_lifetime_index"] == 1)
    ]
    audit = audit_omitted_truths(
        omitted,
        curve,
        total_truth_count=4,
        number_of_seeds=2,
        global_alpha=0.01,
    )
    assert audit.simultaneous_bounds == 4 * 2 * 3
    assert audit.adjusted_z > 0.0
    assert not audit.truth_summary.empty


def test_threshold_diagnostics_and_stability_flags():
    detailed = _toy_detailed_table()
    curve = build_conservative_seed_envelope(build_seed_worst_case_table(detailed))
    diagnostics = monte_carlo_threshold_diagnostics(
        detailed,
        curve,
        target_accuracy=0.90,
        global_alpha=0.01,
        total_truth_count=4,
        number_of_seeds=2,
    )
    assert diagnostics.point_estimate == 11
    assert diagnostics.previous_tested_events == 10
    assert diagnostics.local_sigma_events is not None
    settings = AdaptivePseudoexperimentSettings(
        threshold_stability_events=1,
        required_stable_transitions=1,
    )
    assert threshold_history_is_stable([15, 14], settings)
    assert not threshold_history_is_stable([20, 14], settings)


def test_binning_stability_requires_same_connected_components():
    settings = AdaptiveScanSettings()
    assert binning_is_stable(0.05, 0.049, (1, 0), (1, 0), settings)
    assert not binning_is_stable(0.05, 0.049, (1, 0), (0, 0), settings)


def test_dense_bank_calibration_uses_fewer_or_equal_templates():
    bank = adaptive_toy_bank()
    summary, rounds = calibrate_dense_bank(
        bank,
        AdaptiveLifetimeSettings(
            initial_points_per_decade=1.0,
            minimum_points_per_interval=2,
            maximum_log_gap_decades=2.0,
            maximum_rounds=2,
        ),
    )
    row = summary.iloc[0]
    assert row["adaptive_photon_lifetimes"] <= row["dense_photon_lifetimes"]
    assert row["adaptive_su2_lifetimes"] <= row["dense_su2_lifetimes"]
    assert row["relative_minimum_D_TV_error"] >= 0.0
    assert not rounds.empty


def test_adaptive_cli_and_blue_orange_plot(tmp_path: Path):
    args = parse_arguments(["--masses", "0.3", "1.0"])
    assert args.masses == [0.3, 1.0]
    assert args.selections == ["diphoton_ecal", "diphoton_ecal_e1gev"]
    results = pd.DataFrame(
        {
            "mass_GeV": [0.3, 1.0, 0.3, 1.0],
            "selection_name": [
                "diphoton_ecal",
                "diphoton_ecal",
                "diphoton_ecal_e1gev",
                "diphoton_ecal_e1gev",
            ],
            "N90": [142, 620, 122, 500],
            "N90_mc_lower": [-1, -1, -1, -1],
            "N90_mc_upper": [-1, -1, -1, -1],
            "convergence_status": ["converged"] * 4,
        }
    )
    pdf, png = plot_n90_comparison(results, tmp_path / "comparison")
    assert pdf.is_file()
    assert png.is_file()


def test_resumable_controller_stops_at_first_stable_level_not_below_10k(
    tmp_path: Path,
    monkeypatch,
):
    import alp_discrimination.workflows.lifetime_bank_builder as controller

    bank = adaptive_toy_bank()
    distances = total_variation_matrix(bank)
    distance_summary = pd.Series(
        {
            "minimum_D_TV": float(distances.min()),
            "minimum_photon_interval_index": 1,
            "minimum_su2_interval_index": 1,
        }
    )
    bank_dir = tmp_path / "fake_bank"
    (bank_dir / "template_banks").mkdir(parents=True)

    monkeypatch.setattr(
        controller,
        "_adaptive_bank",
        lambda **kwargs: (
            bank_dir,
            bank,
            distance_summary,
            2,
            "lifetime_grid_converged",
        ),
    )
    monkeypatch.setattr(
        controller,
        "_run_rangefinder",
        lambda **kwargs: (
            np.array([10, 11, 12, 20]),
            {
                "lower_failing_events": 10,
                "upper_passing_events": 11,
                "estimated_crossing_events": 10.5,
            },
        ),
    )

    detailed = _toy_detailed_table()
    curve = build_conservative_seed_envelope(build_seed_worst_case_table(detailed))
    summary = pd.Series(
        {
            "threshold_reached": True,
            "minimum_persistent_events": 11,
        }
    )
    called_levels = []

    def fake_profiled_stage(**kwargs):
        called_levels.append(kwargs["pseudoexperiments"])
        output = tmp_path / f"profile_{kwargs['pseudoexperiments']}"
        output.mkdir(parents=True, exist_ok=True)
        return output, summary, detailed, curve

    monkeypatch.setattr(controller, "_profiled_stage", fake_profiled_stage)
    monkeypatch.setattr(
        controller,
        "select_hard_truth_indices",
        lambda *args, **kwargs: (
            {"photon": np.array([0, 1]), "su2": np.array([0, 1])},
            pd.DataFrame(
                {
                    "truth_model": ["photon", "su2"],
                    "truth_lifetime_index": [0, 0],
                }
            ),
        ),
    )

    settings = AdaptiveScanSettings(
        pseudoexperiments=AdaptivePseudoexperimentSettings(
            full_domain_pilot_pseudoexperiments=2000,
            minimum_final_pseudoexperiments=10000,
            pseudoexperiment_ladder=(5000, 10000, 20000),
            final_seeds=2,
            minimum_hard_truths_per_model=1,
        )
    )
    domain_path = tmp_path / "domains.csv"
    domain_path.write_text("model,mass_GeV,interval_index,ctau_min_m,ctau_max_m\n")
    result = controller.run_point(
        mass_gev=0.3,
        selection_name="diphoton_ecal",
        profile="validation",
        domain_path=domain_path,
        domains=pd.DataFrame(),
        output_dir=tmp_path / "output",
        settings=settings,
        workers=1,
        stop_after="final",
        skip_conditional_binning_check=False,
        diagnostic_plots=False,
    )
    assert result is not None
    assert result["N90"] == 11
    assert result["convergence_status"] == "converged"
    # Full-domain 2k, then 5k, then 10k.  It must not stop at 5k even though
    # the threshold is already stable, and it must not waste the 20k stage.
    assert called_levels == [2000, 5000, 10000]
    final_path = (
        tmp_path
        / "output"
        / "per_mass"
        / "ma_0p3"
        / "geom"
        / "final_result.json"
    )
    assert final_path.is_file()
