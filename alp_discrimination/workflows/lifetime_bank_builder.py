"""Run a resumable adaptive lifetime-profiled N90 scan for arbitrary masses and selections.

The controller keeps the validated EventCalc/template/profiled-likelihood
kernels unchanged.  It adapts only the lifetime grid, event-count grid, truth
set and pseudoexperiment statistics, checkpointing every expensive stage.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from math import ceil
from pathlib import Path
from time import perf_counter
import traceback
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from alp_discrimination.statistics.adaptive_grid import (
    AdaptiveLifetimeSettings,
    AdaptivePseudoexperimentSettings,
    AdaptiveScanSettings,
    DOMAIN_MODEL_LABELS,
    SELECTIONS,
    TRUTH_MODELS,
    audit_omitted_truths,
    binning_is_stable,
    distance_screening_truth_indices,
    estimate_event_scale_from_distance,
    event_grid_specification,
    final_event_grid_from_bracket,
    initial_adaptive_lifetime_grid,
    lifetime_grid_from_bank,
    merge_truth_indices,
    monte_carlo_threshold_diagnostics,
    omitted_truth_indices,
    propose_lifetime_refinement,
    rangefinder_bracket,
    rangefinder_event_grid,
    result_row,
    select_hard_truth_indices,
    should_run_fine_binning_check,
    threshold_history_is_stable,
    total_variation_matrix,
    truth_subset_table,
)
from alp_discrimination.cache import CacheStore, atomic_output_path, canonical_json
from alp_discrimination.config import AnalysisConfig, get_config
from alp_discrimination.templates.lifetime_banks import LifetimeTemplateBank, load_template_bank
from alp_discrimination.paths import OUTPUT_ROOT, portable_path
from alp_discrimination.statistics.reduction import minimum_persistent_events
from alp_discrimination.workflows import float_token, write_dataframe
from alp_discrimination.workflows.plot_n90_comparison import (
    plot_n90_comparison,
)


# Legacy cache/workflow key retained so existing expensive checkpoints remain reusable.
WORKFLOW_NAME = "adaptive_week8_scan"
RESULT_COLUMNS = (
    "mass_GeV",
    "selection_name",
    "N90",
    "N90_mc_lower",
    "N90_mc_upper",
    "local_mc_sigma_events",
    "convergence_status",
    "final_PE_count",
    "number_of_selected_photon_truths",
    "number_of_selected_su2_truths",
    "number_of_omitted_photon_truths",
    "number_of_omitted_su2_truths",
    "number_of_photon_profile_lifetimes",
    "number_of_su2_profile_lifetimes",
    "number_of_energy_bins",
    "minimum_D_TV",
    "limiting_truth_model",
    "limiting_truth_lifetime_index",
    "limiting_truth_ctau_m",
    "limiting_seed",
    "accuracy_at_threshold",
    "audit_simultaneous_bounds",
    "minimum_omitted_lower_margin",
    "lifetime_refinement_rounds",
    "runtime_seconds",
)


class AdaptivePointError(RuntimeError):
    """A recoverable failure of one mass-selection point."""


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--masses", nargs="+", type=float)
    parser.add_argument(
        "--selections",
        nargs="+",
        choices=SELECTIONS,
        default=list(SELECTIONS),
    )
    parser.add_argument(
        "--profile",
        choices=("validation", "production", "quick", "smoke"),
        default="validation",
    )
    parser.add_argument(
        "--domain-path",
        type=Path,
        default=(
            OUTPUT_ROOT
            / "production"
            / "week8_domains"
            / "allowed_ctau_domains.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "production" / "week8_adaptive_scan",
    )
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Explicitly request checkpoint reuse. Resumption is always safe and "
            "enabled; this flag is accepted for readable batch commands."
        ),
    )
    parser.add_argument("--initial-energy-bins", type=int, default=200)
    parser.add_argument("--minimum-bin-n-eff", type=float, default=100.0)
    parser.add_argument("--maximum-lifetime-rounds", type=int, default=8)
    parser.add_argument("--maximum-lifetimes-per-model", type=int, default=120)
    parser.add_argument(
        "--maximum-new-lifetimes-per-model-per-round",
        type=int,
        default=16,
    )
    parser.add_argument("--lifetime-points-per-decade", type=float, default=4.0)
    parser.add_argument(
        "--minimum-lifetime-points-per-interval", type=int, default=5
    )
    parser.add_argument("--maximum-log-ctau-gap", type=float, default=0.25)
    parser.add_argument("--maximum-adjacent-template-tv", type=float, default=0.018)
    parser.add_argument(
        "--maximum-log-interpolation-tv", type=float, default=0.004
    )
    parser.add_argument(
        "--maximum-adjacent-distance-change", type=float, default=0.035
    )
    parser.add_argument(
        "--maximum-soft-priority-at-convergence",
        type=float,
        default=6.0,
    )
    parser.add_argument("--pilot-pseudoexperiments", type=int, default=2000)
    parser.add_argument(
        "--pseudoexperiment-ladder",
        nargs="+",
        type=int,
        default=[5000, 10000, 20000],
    )
    parser.add_argument("--rangefinder-pseudoexperiments", type=int, default=1000)
    parser.add_argument("--rangefinder-seeds", type=int, default=2)
    parser.add_argument("--final-seeds", type=int, default=5)
    parser.add_argument("--minimum-final-pseudoexperiments", type=int, default=10000)
    parser.add_argument("--hard-truth-gap", type=float, default=0.030)
    parser.add_argument("--audit-alpha", type=float, default=0.01)
    parser.add_argument("--rangefinder-maximum-events", type=int, default=20000)
    parser.add_argument("--unit-window-half-width", type=int, default=30)
    parser.add_argument("--maximum-unit-window-points", type=int, default=241)
    parser.add_argument("--persistence-tail-factor", type=float, default=1.8)
    parser.add_argument(
        "--conditional-fine-binning-bins", type=int, default=400
    )
    parser.add_argument(
        "--fine-binning-distance-threshold", type=float, default=0.08
    )
    parser.add_argument(
        "--fine-binning-relative-tolerance", type=float, default=0.05
    )
    parser.add_argument(
        "--skip-conditional-binning-check",
        action="store_true",
        help="Skip the cached 400-initial-bin check even for fragile banks.",
    )
    parser.add_argument(
        "--diagnostic-plots",
        action="store_true",
        help="Also write per-mass distance/profile plots; off by default for speed.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("bank", "distance", "rangefinder", "pilot", "final"),
        default="final",
    )
    parser.add_argument(
        "--import-result-json",
        nargs="*",
        type=Path,
        default=(),
        help="Import already validated final point JSON files into the master plot.",
    )
    parser.add_argument(
        "--import-results-csv",
        nargs="*",
        type=Path,
        default=(),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--rerun-final-points",
        action="store_true",
        help="Recompute points already marked converged/imported in the master CSV.",
    )
    return parser.parse_args(argv)


def settings_from_arguments(args) -> AdaptiveScanSettings:
    lifetime = AdaptiveLifetimeSettings(
        initial_points_per_decade=args.lifetime_points_per_decade,
        minimum_points_per_interval=args.minimum_lifetime_points_per_interval,
        maximum_rounds=args.maximum_lifetime_rounds,
        maximum_total_lifetimes_per_model=args.maximum_lifetimes_per_model,
        maximum_new_points_per_model_per_round=(
            args.maximum_new_lifetimes_per_model_per_round
        ),
        maximum_log_gap_decades=args.maximum_log_ctau_gap,
        maximum_adjacent_template_tv=args.maximum_adjacent_template_tv,
        maximum_log_interpolation_tv=args.maximum_log_interpolation_tv,
        maximum_adjacent_distance_change=(
            args.maximum_adjacent_distance_change
        ),
        maximum_soft_priority_at_convergence=(
            args.maximum_soft_priority_at_convergence
        ),
    )
    pseudo = AdaptivePseudoexperimentSettings(
        rangefinder_pseudoexperiments=args.rangefinder_pseudoexperiments,
        rangefinder_seeds=args.rangefinder_seeds,
        full_domain_pilot_pseudoexperiments=args.pilot_pseudoexperiments,
        minimum_final_pseudoexperiments=args.minimum_final_pseudoexperiments,
        final_seeds=args.final_seeds,
        pseudoexperiment_ladder=tuple(args.pseudoexperiment_ladder),
        hard_truth_accuracy_gap=args.hard_truth_gap,
        audit_global_alpha=args.audit_alpha,
        rangefinder_maximum_events=args.rangefinder_maximum_events,
        unit_window_minimum_half_width=args.unit_window_half_width,
        maximum_unit_window_points=args.maximum_unit_window_points,
        persistence_tail_factor=args.persistence_tail_factor,
    )
    return AdaptiveScanSettings(
        lifetime=lifetime,
        pseudoexperiments=pseudo,
        initial_energy_bins=args.initial_energy_bins,
        minimum_bin_n_eff=args.minimum_bin_n_eff,
        conditional_fine_binning_bins=(
            args.conditional_fine_binning_bins
        ),
        fine_binning_distance_threshold=(
            args.fine_binning_distance_threshold
        ),
        fine_binning_relative_tolerance=(
            args.fine_binning_relative_tolerance
        ),
    )


def _write_json(payload: Mapping, path: Path) -> Path:
    with atomic_output_path(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _settings_fingerprint(settings: AdaptiveScanSettings, profile: str) -> str:
    payload = {"profile": profile, "settings": settings.as_dict()}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _selection_token(selection_name: str) -> str:
    return "geom" if selection_name == "diphoton_ecal" else "e1gev"


def _point_dir(output_dir: Path, mass_gev: float, selection_name: str) -> Path:
    return (
        output_dir
        / "per_mass"
        / f"ma_{float_token(mass_gev)}"
        / _selection_token(selection_name)
    )


def _stage_complete(directory: Path, required_relative: Sequence[str]) -> bool:
    return all((directory / relative).is_file() for relative in required_relative)


def _resolved_stage_dir(
    base: Path,
    required_relative: Sequence[str],
) -> tuple[Path, bool]:
    candidates = [base, *sorted(base.parent.glob(base.name + "_retry*"))]
    for candidate in reversed(candidates):
        if _stage_complete(candidate, required_relative):
            return candidate, True
    if not base.exists():
        return base, False
    index = 1
    while True:
        retry = base.parent / f"{base.name}_retry{index:02d}"
        if not retry.exists():
            return retry, False
        index += 1


def _available_masses(domains: pd.DataFrame) -> tuple[float, ...]:
    return tuple(sorted(float(value) for value in domains["mass_GeV"].unique()))


def _resolve_masses(requested: Sequence[float], domains: pd.DataFrame) -> tuple[float, ...]:
    available = _available_masses(domains)
    selected: list[float] = []
    for value in requested:
        matches = [
            mass
            for mass in available
            if np.isclose(float(value), mass, rtol=0.0, atol=1.0e-12)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Mass {value:g} GeV is absent from the Week-8 domain table. "
                f"Available: {available}"
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    return tuple(selected)


def _base_config(
    profile: str,
    selection_name: str,
    settings: AdaptiveScanSettings,
) -> AnalysisConfig:
    config = get_config(profile)
    return replace(
        config,
        selection_name=selection_name,
        templates=replace(
            config.templates,
            initial_energy_bins=settings.initial_energy_bins,
            minimum_bin_n_eff=settings.minimum_bin_n_eff,
        ),
    )


def _profile_config(
    base: AnalysisConfig,
    *,
    pseudoexperiments: int,
    number_of_seeds: int,
    maximum_events: int,
) -> AnalysisConfig:
    return replace(
        base,
        profiled_likelihood=replace(
            base.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=int(pseudoexperiments),
            number_of_seeds=int(number_of_seeds),
            maximum_observed_events=int(maximum_events),
            chunk_size=min(
                int(base.profiled_likelihood.chunk_size),
                max(200, int(pseudoexperiments)),
            ),
            truth_lifetime_grid="all",
            profile_lifetime_grid="all",
            rebin_factor=1,
        ),
    )


def _build_bank_stage(
    *,
    config: AnalysisConfig,
    domain_path: Path,
    mass_gev: float,
    lifetime_grid_path: Path,
    output_base: Path,
    initial_energy_bins: int,
) -> tuple[Path, LifetimeTemplateBank]:
    required = (
        "manifest.json",
        f"template_banks/template_bank_ma_{float_token(mass_gev)}.npz",
    )
    output_dir, completed = _resolved_stage_dir(output_base, required)
    bank_path = (
        output_dir
        / "template_banks"
        / f"template_bank_ma_{float_token(mass_gev)}.npz"
    )
    if not completed:
        from alp_discrimination.eventcalc.adapter import EventCalcAdapter
        from alp_discrimination.workflows.lifetime_blind_discrimination import (
            run_template_bank_workflow,
        )

        stage_config = replace(
            config,
            templates=replace(
                config.templates,
                initial_energy_bins=int(initial_energy_bins),
            ),
        )
        cache = CacheStore(stage_config.name)
        adapter = EventCalcAdapter(stage_config, cache=cache, force=False)
        run_template_bank_workflow(
            config=stage_config,
            adapter=adapter,
            domain_path=domain_path,
            output_dir=output_dir,
            requested_masses=[mass_gev],
            overwrite=False,
            lifetime_grid_path=lifetime_grid_path,
        )
    return output_dir, load_template_bank(bank_path)


def _distance_stage(
    *,
    config: AnalysisConfig,
    bank_dir: Path,
    mass_gev: float,
    output_base: Path,
    workers: int,
    make_plots: bool,
) -> tuple[Path, pd.Series]:
    del workers  # Distance post-processing is deterministic and cheap.
    required = ("manifest.json", "distance_map_summary.csv")
    output_dir, completed = _resolved_stage_dir(output_base, required)
    if not completed:
        from alp_discrimination.workflows.lifetime_blind_distance_maps import (
            run_distance_map_workflow,
        )

        run_distance_map_workflow(
            config=config,
            cache=CacheStore(config.name),
            input_dir=bank_dir / "template_banks",
            output_dir=output_dir,
            requested_masses=[mass_gev],
            overwrite=False,
            force=False,
            make_plots=make_plots,
        )
    summary = pd.read_csv(output_dir / "distance_map_summary.csv").iloc[0]
    return output_dir, summary


def _profiled_stage(
    *,
    base_config: AnalysisConfig,
    bank_dir: Path,
    mass_gev: float,
    output_base: Path,
    event_counts: np.ndarray,
    pseudoexperiments: int,
    number_of_seeds: int,
    workers: int,
    truth_subset_path: Path | None,
    make_plots: bool,
) -> tuple[Path, pd.Series, pd.DataFrame, pd.DataFrame]:
    required = (
        "manifest.json",
        "profiled_threshold_summary.csv",
        f"tables/profiled_accuracy_ma_{float_token(mass_gev)}.csv",
        f"tables/profiled_conservative_curve_ma_{float_token(mass_gev)}.csv",
    )
    output_dir, completed = _resolved_stage_dir(output_base, required)
    if not completed:
        from alp_discrimination.workflows.lifetime_blind_profiled_likelihood import (
            run_workflow,
        )

        config = _profile_config(
            base_config,
            pseudoexperiments=pseudoexperiments,
            number_of_seeds=number_of_seeds,
            maximum_events=int(event_counts[-1]),
        )
        run_workflow(
            config,
            input_dir=bank_dir / "template_banks",
            output_dir=output_dir,
            masses=[mass_gev],
            cache=CacheStore(config.name),
            force=False,
            make_plots=make_plots,
            event_counts=event_counts,
            workers=workers,
            truth_subset_path=truth_subset_path,
        )
    summary = pd.read_csv(output_dir / "profiled_threshold_summary.csv").iloc[0]
    detailed = pd.read_csv(
        output_dir / "tables" / f"profiled_accuracy_ma_{float_token(mass_gev)}.csv"
    )
    curve = pd.read_csv(
        output_dir
        / "tables"
        / f"profiled_conservative_curve_ma_{float_token(mass_gev)}.csv"
    )
    return output_dir, summary, detailed, curve


def _write_subset(
    bank: LifetimeTemplateBank,
    indices: Mapping[str, Sequence[int]],
    path: Path,
) -> Path:
    table = truth_subset_table(bank, indices)
    if path.is_file():
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(
            existing.reset_index(drop=True),
            table.reset_index(drop=True),
            check_dtype=False,
            rtol=1.0e-12,
            atol=0.0,
        )
    else:
        write_dataframe(table, path)
    return path


def _filter_truths(
    detailed: pd.DataFrame,
    indices: Mapping[str, Sequence[int]],
) -> pd.DataFrame:
    frames = []
    for model in TRUTH_MODELS:
        frames.append(
            detailed.loc[
                (detailed["truth_model"] == model)
                & detailed["truth_lifetime_index"].astype(int).isin(
                    np.asarray(indices[model], dtype=int)
                )
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _threshold_from_summary(summary: pd.Series) -> int | None:
    return (
        int(summary["minimum_persistent_events"])
        if bool(summary["threshold_reached"])
        else None
    )


def _critical_event_counts(curve: pd.DataFrame, threshold: int) -> np.ndarray:
    """Return a local crossing window for high-statistics truth ranking.

    Ranking over the entire high-N persistence tail is counterproductive: once
    every truth is near unit accuracy, tiny discrete PE differences can make
    many harmless truths look artificially close to the envelope. The omitted-
    truth audit still checks the complete persistence tail later.
    """
    events = curve["number_of_events"].to_numpy(dtype=int)
    lower = max(int(events.min()), int(threshold) - 10)
    upper = int(threshold) + max(30, int(ceil(0.25 * int(threshold))))
    selected = events[(events >= lower) & (events <= upper)]
    if len(selected) == 0:
        raise AdaptivePointError("No event counts remain in the ranking window.")
    return selected


def _unit_crossing_is_resolved(curve: pd.DataFrame, threshold: int) -> bool:
    events = set(curve["number_of_events"].astype(int))
    return threshold in events and (threshold == min(events) or threshold - 1 in events)


def _final_limiting_row(curve: pd.DataFrame, threshold: int) -> pd.Series:
    rows = curve.loc[curve["number_of_events"].astype(int) == int(threshold)]
    if len(rows) != 1:
        raise AdaptivePointError("Final curve does not contain a unique threshold row.")
    return rows.iloc[0]


def _load_or_create_state(point_dir: Path, payload: dict) -> dict:
    state_path = point_dir / "state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("settings_fingerprint") != payload["settings_fingerprint"]:
            raise AdaptivePointError(
                "Existing adaptive point uses different settings. Choose a new "
                "--output-dir rather than mixing cache/output provenance."
            )
        return state
    point_dir.mkdir(parents=True, exist_ok=True)
    _write_json(payload, state_path)
    return payload


def _update_state(point_dir: Path, state: dict, **updates) -> dict:
    result = dict(state)
    result.update(updates)
    result["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(result, point_dir / "state.json")
    return result


def _adaptive_bank(
    *,
    point_dir: Path,
    config: AnalysisConfig,
    domain_path: Path,
    domains: pd.DataFrame,
    mass_gev: float,
    settings: AdaptiveScanSettings,
    workers: int,
    diagnostic_plots: bool,
) -> tuple[Path, LifetimeTemplateBank, pd.Series, int, str]:
    grid_dir = point_dir / "lifetime_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / "round_00.csv"
    if not grid_path.is_file():
        write_dataframe(
            initial_adaptive_lifetime_grid(domains, mass_gev, settings.lifetime),
            grid_path,
        )

    previous_minimum: float | None = None
    final_status = "lifetime_grid_converged"
    final_bank_dir: Path | None = None
    final_bank: LifetimeTemplateBank | None = None
    final_distance_summary: pd.Series | None = None
    rounds_used = 0

    for round_index in range(settings.lifetime.maximum_rounds):
        rounds_used = round_index + 1
        bank_dir, bank = _build_bank_stage(
            config=config,
            domain_path=domain_path,
            mass_gev=mass_gev,
            lifetime_grid_path=grid_path,
            output_base=point_dir / "banks" / f"round_{round_index:02d}",
            initial_energy_bins=settings.initial_energy_bins,
        )
        distance_dir, distance_summary = _distance_stage(
            config=config,
            bank_dir=bank_dir,
            mass_gev=mass_gev,
            output_base=point_dir / "distance_maps" / f"round_{round_index:02d}",
            workers=workers,
            make_plots=False,
        )
        del distance_dir
        distances = total_variation_matrix(bank)
        current_grid = lifetime_grid_from_bank(
            bank, adaptive_round_added=round_index, reason="built_bank"
        )
        decision = propose_lifetime_refinement(
            bank,
            distances,
            current_grid,
            settings.lifetime,
            round_index=round_index,
            previous_minimum_distance=previous_minimum,
        )
        write_dataframe(
            decision.diagnostics,
            grid_dir / f"round_{round_index:02d}_diagnostics.csv",
        )
        if not decision.additions.empty:
            write_dataframe(
                decision.additions,
                grid_dir / f"round_{round_index:02d}_additions.csv",
            )
        nominal_unresolved = int(
            decision.diagnostics["exceeds_nominal_tolerance"].sum()
        )
        required_unresolved = int(
            decision.diagnostics["required_for_convergence"].sum()
        )
        maximum_soft_priority = float(
            decision.diagnostics["soft_priority"].max()
            if not decision.diagnostics.empty
            else 0.0
        )
        _write_json(
            {
                "round": round_index,
                "minimum_D_TV": decision.minimum_distance,
                "previous_minimum_D_TV": decision.previous_minimum_distance,
                "relative_minimum_D_TV_change": (
                    decision.relative_minimum_distance_change
                ),
                "number_of_new_points": int(len(decision.additions)),
                "number_of_nominal_unresolved_pairs": nominal_unresolved,
                "number_of_required_unresolved_pairs": required_unresolved,
                "maximum_soft_priority": maximum_soft_priority,
                "maximum_soft_priority_at_convergence": (
                    settings.lifetime.maximum_soft_priority_at_convergence
                ),
                "number_of_photon_lifetimes": len(bank.photon_ctau_m),
                "number_of_su2_lifetimes": len(bank.su2_ctau_m),
                "converged": decision.converged,
                "reached_round_limit": decision.reached_round_limit,
                "reached_size_limit": decision.reached_size_limit,
            },
            grid_dir / f"round_{round_index:02d}_summary.json",
        )
        final_bank_dir = bank_dir
        final_bank = bank
        final_distance_summary = distance_summary
        if decision.converged:
            break
        if decision.additions.empty:
            # No point could be inserted. Distinguish a genuine numerical
            # stability warning from exhaustion of the configured grid cap.
            final_status = (
                "lifetime_grid_size_limit"
                if decision.reached_size_limit
                else "lifetime_grid_distance_unstable"
            )
            break
        if round_index + 1 >= settings.lifetime.maximum_rounds:
            final_status = "lifetime_grid_round_limit"
            break
        next_path = grid_dir / f"round_{round_index + 1:02d}.csv"
        if not next_path.is_file():
            write_dataframe(decision.grid, next_path)
        grid_path = next_path
        previous_minimum = decision.minimum_distance

    if final_bank_dir is None or final_bank is None or final_distance_summary is None:
        raise AdaptivePointError("No adaptive template bank was produced.")

    # Conditional binning refinement is a cached rehistogram.  It is run only
    # for fragile banks and uses the finer stable bank for the final likelihood.
    if should_run_fine_binning_check(
        final_bank, float(final_distance_summary["minimum_D_TV"]), settings
    ):
        checks: list[dict] = []
        previous_bank = final_bank
        previous_summary = final_distance_summary
        stable = False
        for check_index in range(1, settings.maximum_binning_refinement_rounds + 1):
            initial_bins = (
                settings.conditional_fine_binning_bins
                * settings.binning_refinement_factor ** (check_index - 1)
            )
            check_dir, check_bank = _build_bank_stage(
                config=config,
                domain_path=domain_path,
                mass_gev=mass_gev,
                lifetime_grid_path=grid_path,
                output_base=point_dir / "banks" / f"binning_{initial_bins}",
                initial_energy_bins=initial_bins,
            )
            _, check_summary = _distance_stage(
                config=config,
                bank_dir=check_dir,
                mass_gev=mass_gev,
                output_base=point_dir / "distance_maps" / f"binning_{initial_bins}",
                workers=workers,
                make_plots=False,
            )
            previous_intervals = (
                int(previous_summary["minimum_photon_interval_index"]),
                int(previous_summary["minimum_su2_interval_index"]),
            )
            check_intervals = (
                int(check_summary["minimum_photon_interval_index"]),
                int(check_summary["minimum_su2_interval_index"]),
            )
            stable = binning_is_stable(
                float(previous_summary["minimum_D_TV"]),
                float(check_summary["minimum_D_TV"]),
                previous_intervals,
                check_intervals,
                settings,
            )
            checks.append(
                {
                    "comparison_index": check_index,
                    "coarser_initial_bins": (
                        settings.initial_energy_bins
                        if check_index == 1
                        else settings.conditional_fine_binning_bins
                        * settings.binning_refinement_factor ** (check_index - 2)
                    ),
                    "coarser_final_bins": previous_bank.number_of_energy_bins,
                    "coarser_minimum_D_TV": float(previous_summary["minimum_D_TV"]),
                    "finer_initial_bins": initial_bins,
                    "finer_final_bins": check_bank.number_of_energy_bins,
                    "finer_minimum_D_TV": float(check_summary["minimum_D_TV"]),
                    "coarser_minimum_intervals": previous_intervals,
                    "finer_minimum_intervals": check_intervals,
                    "stable": stable,
                }
            )
            # Once calculated, retain the finer representation for the final
            # likelihood.  This reproduces the validated m_a=0.3 convention.
            final_bank_dir = check_dir
            final_bank = check_bank
            final_distance_summary = check_summary
            previous_bank = check_bank
            previous_summary = check_summary
            if stable:
                final_status = (
                    "fine_binning_converged"
                    if final_status == "lifetime_grid_converged"
                    else final_status + "+fine_binning_converged"
                )
                break
        if not stable:
            final_status = (
                "binning_refinement_limit"
                if final_status == "lifetime_grid_converged"
                else final_status + "+binning_refinement_limit"
            )
        _write_json(
            {
                "triggered": True,
                "comparisons": checks,
                "final_bank_dir": portable_path(final_bank_dir),
                "stable": stable,
            },
            point_dir / "binning_check.json",
        )

    if diagnostic_plots:
        _distance_stage(
            config=config,
            bank_dir=final_bank_dir,
            mass_gev=mass_gev,
            output_base=point_dir / "distance_maps" / "final_with_plots",
            workers=workers,
            make_plots=True,
        )
    return (
        final_bank_dir,
        final_bank,
        final_distance_summary,
        rounds_used,
        final_status,
    )


def _run_rangefinder(
    *,
    point_dir: Path,
    config: AnalysisConfig,
    bank_dir: Path,
    bank: LifetimeTemplateBank,
    distances: np.ndarray,
    minimum_distance: float,
    mass_gev: float,
    settings: AdaptiveScanSettings,
    workers: int,
) -> tuple[np.ndarray, dict]:
    indices = distance_screening_truth_indices(
        bank, distances, settings.pseudoexperiments
    )
    subset_path = _write_subset(
        bank, indices, point_dir / "truth_subsets" / "rangefinder.csv"
    )
    estimate = estimate_event_scale_from_distance(
        minimum_distance, settings.pseudoexperiments
    )
    grid = rangefinder_event_grid(estimate, settings.pseudoexperiments)
    history = []
    bracket = None
    for attempt in range(3):
        output_dir, summary, _, curve = _profiled_stage(
            base_config=config,
            bank_dir=bank_dir,
            mass_gev=mass_gev,
            output_base=point_dir / "profiled" / f"rangefinder_{attempt:02d}",
            event_counts=grid,
            pseudoexperiments=(
                settings.pseudoexperiments.rangefinder_pseudoexperiments
            ),
            number_of_seeds=settings.pseudoexperiments.rangefinder_seeds,
            workers=workers,
            truth_subset_path=subset_path,
            make_plots=False,
        )
        del output_dir, summary
        bracket = rangefinder_bracket(curve, settings.pseudoexperiments)
        history.append(
            {
                "attempt": attempt,
                "event_counts": grid.tolist(),
                "threshold_reached": bracket.threshold_reached,
                "lower_failing_events": bracket.lower_failing_events,
                "upper_passing_events": bracket.upper_passing_events,
                "estimated_crossing_events": bracket.estimated_crossing_events,
            }
        )
        if bracket.threshold_reached:
            break
        if bracket.upper_passing_events >= settings.pseudoexperiments.rangefinder_maximum_events:
            break
        grid = rangefinder_event_grid(
            bracket.upper_passing_events, settings.pseudoexperiments
        )
    if bracket is None or not bracket.threshold_reached:
        raise AdaptivePointError(
            "Range finder did not bracket a persistent 90% crossing. Increase "
            "--pseudoexperiment-ladder range or inspect this mass manually."
        )
    final_grid = final_event_grid_from_bracket(
        bracket, settings.pseudoexperiments
    )
    _write_json(
        {
            "minimum_D_TV": minimum_distance,
            "distance_scale_estimate": estimate,
            "history": history,
            "final_event_counts": final_grid.tolist(),
            "final_event_grid_specification": event_grid_specification(final_grid),
        },
        point_dir / "rangefinder_plan.json",
    )
    write_dataframe(
        pd.DataFrame({"number_of_events": final_grid}),
        point_dir / "final_event_grid.csv",
    )
    return final_grid, history[-1]


def _import_result_json(path: Path) -> dict:
    data = _read_json(path)
    mass = float(data["mass_GeV"])
    selection = str(data["selection_name"])
    threshold = int(
        data.get("N90", data.get("candidate_persistent_events"))
    )
    return {
        "mass_GeV": mass,
        "selection_name": selection,
        "N90": threshold,
        "N90_mc_lower": int(data.get("N90_mc_lower", -1)),
        "N90_mc_upper": int(data.get("N90_mc_upper", -1)),
        "local_mc_sigma_events": float(data.get("local_mc_sigma_events", np.nan)),
        "convergence_status": "imported_validated",
        "final_PE_count": int(data.get("selected_truth_statistics", 0)),
        "number_of_selected_photon_truths": int(
            data.get("number_of_selected_photon_truths", 0)
        ),
        "number_of_selected_su2_truths": int(
            data.get("number_of_selected_su2_truths", 0)
        ),
        "number_of_omitted_photon_truths": int(
            data.get("number_of_omitted_photon_truths", 0)
        ),
        "number_of_omitted_su2_truths": int(
            data.get("number_of_omitted_su2_truths", 0)
        ),
        "number_of_photon_profile_lifetimes": int(
            data.get("number_of_photon_profile_lifetimes", 0)
        ),
        "number_of_su2_profile_lifetimes": int(
            data.get("number_of_su2_profile_lifetimes", 0)
        ),
        "number_of_energy_bins": int(data.get("number_of_energy_bins", 0)),
        "minimum_D_TV": float(data.get("minimum_D_TV", np.nan)),
        "limiting_truth_model": str(data.get("limiting_truth_model", "")),
        "limiting_truth_lifetime_index": int(
            data.get("limiting_truth_lifetime_index", -1)
        ),
        "limiting_truth_ctau_m": float(
            data.get("limiting_truth_ctau_m", np.nan)
        ),
        "limiting_seed": int(data.get("limiting_seed", -1)),
        "accuracy_at_threshold": float(
            data.get("candidate_accuracy", data.get("accuracy_at_threshold", np.nan))
        ),
        "audit_simultaneous_bounds": int(
            data.get("number_of_simultaneous_one_sided_bounds", 0)
        ),
        "minimum_omitted_lower_margin": float(
            data.get("minimum_omitted_global99_lower_margin", np.nan)
        ),
        "lifetime_refinement_rounds": int(
            data.get("lifetime_refinement_rounds", 0)
        ),
        "runtime_seconds": float(data.get("runtime_seconds", 0.0)),
    }


def _load_master_results(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "adaptive_n90_results.csv"
    if not path.is_file():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    table = pd.read_csv(path)
    for column in RESULT_COLUMNS:
        if column not in table:
            table[column] = np.nan
    return table.loc[:, RESULT_COLUMNS]


def _upsert_result(output_dir: Path, row: Mapping) -> pd.DataFrame:
    table = _load_master_results(output_dir)
    mask = (
        np.isclose(
            pd.to_numeric(table["mass_GeV"], errors="coerce").to_numpy(dtype=float),
            float(row["mass_GeV"]),
            rtol=0.0,
            atol=1.0e-12,
            equal_nan=False,
        )
        & (table["selection_name"].astype(str) == str(row["selection_name"]))
    ) if not table.empty else np.asarray([], dtype=bool)
    if len(mask):
        table = table.loc[~mask]
    new_row = pd.DataFrame([row], columns=RESULT_COLUMNS)
    table = (
        new_row
        if table.empty
        else pd.concat([table, new_row], ignore_index=True)
    ).sort_values(["mass_GeV", "selection_name"], ignore_index=True)
    write_dataframe(table, output_dir / "adaptive_n90_results.csv")
    return table


def _write_master_plot(output_dir: Path) -> None:
    table = _load_master_results(output_dir)
    positive = pd.to_numeric(table["N90"], errors="coerce") > 0
    final_status = table["convergence_status"].astype(str).isin(
        {"converged", "imported_validated"}
    )
    if not (positive & final_status).any():
        return
    plot_n90_comparison(
        table,
        output_dir / "week8_n90_comparison",
        logarithmic_y=True,
        show_mc_intervals=True,
    )


def _master_point_is_final(
    output_dir: Path, mass_gev: float, selection_name: str
) -> bool:
    table = _load_master_results(output_dir)
    if table.empty:
        return False
    mass = pd.to_numeric(table["mass_GeV"], errors="coerce").to_numpy(dtype=float)
    matches = table.loc[
        np.isclose(mass, float(mass_gev), rtol=0.0, atol=1.0e-12)
        & (table["selection_name"].astype(str) == selection_name)
        & (pd.to_numeric(table["N90"], errors="coerce") > 0)
        & table["convergence_status"].astype(str).isin(
            {"converged", "imported_validated"}
        )
    ]
    return not matches.empty


def _failure_row(
    mass_gev: float,
    selection_name: str,
    runtime_seconds: float,
    status: str,
) -> dict:
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "mass_GeV": float(mass_gev),
            "selection_name": selection_name,
            "N90": -1,
            "N90_mc_lower": -1,
            "N90_mc_upper": -1,
            "convergence_status": status,
            "final_PE_count": 0,
            "number_of_selected_photon_truths": 0,
            "number_of_selected_su2_truths": 0,
            "number_of_omitted_photon_truths": 0,
            "number_of_omitted_su2_truths": 0,
            "number_of_photon_profile_lifetimes": 0,
            "number_of_su2_profile_lifetimes": 0,
            "number_of_energy_bins": 0,
            "limiting_truth_model": "",
            "limiting_truth_lifetime_index": -1,
            "limiting_seed": -1,
            "audit_simultaneous_bounds": 0,
            "lifetime_refinement_rounds": 0,
            "runtime_seconds": float(runtime_seconds),
        }
    )
    return row


def run_point(
    *,
    mass_gev: float,
    selection_name: str,
    profile: str,
    domain_path: Path,
    domains: pd.DataFrame,
    output_dir: Path,
    settings: AdaptiveScanSettings,
    workers: int,
    stop_after: str,
    skip_conditional_binning_check: bool,
    diagnostic_plots: bool,
) -> dict | None:
    started = perf_counter()
    point_dir = _point_dir(output_dir, mass_gev, selection_name)
    effective_settings = settings
    if skip_conditional_binning_check:
        effective_settings = replace(
            settings,
            fine_binning_distance_threshold=np.finfo(float).tiny,
            fine_binning_minimum_final_bins=2,
        )
    fingerprint = _settings_fingerprint(effective_settings, profile)
    state = _load_or_create_state(
        point_dir,
        {
            "workflow": WORKFLOW_NAME,
            "mass_GeV": float(mass_gev),
            "selection_name": selection_name,
            "profile": profile,
            "domain_path": portable_path(domain_path),
            "settings_fingerprint": fingerprint,
            "settings": effective_settings.as_dict(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "started",
        },
    )
    final_json = point_dir / "final_result.json"
    if final_json.is_file():
        return _read_json(final_json)

    config = _base_config(profile, selection_name, effective_settings)

    bank_dir, bank, distance_summary, lifetime_rounds, bank_status = _adaptive_bank(
        point_dir=point_dir,
        config=config,
        domain_path=domain_path,
        domains=domains,
        mass_gev=mass_gev,
        settings=effective_settings,
        workers=workers,
        diagnostic_plots=diagnostic_plots,
    )
    state = _update_state(
        point_dir,
        state,
        status="bank_complete",
        bank_dir=portable_path(bank_dir),
        bank_status=bank_status,
        number_of_photon_lifetimes=len(bank.photon_ctau_m),
        number_of_su2_lifetimes=len(bank.su2_ctau_m),
        number_of_energy_bins=bank.number_of_energy_bins,
        minimum_D_TV=float(distance_summary["minimum_D_TV"]),
    )
    if stop_after == "bank":
        return None
    if stop_after == "distance":
        return None

    distances = total_variation_matrix(bank)
    final_event_counts, rangefinder_record = _run_rangefinder(
        point_dir=point_dir,
        config=config,
        bank_dir=bank_dir,
        bank=bank,
        distances=distances,
        minimum_distance=float(distance_summary["minimum_D_TV"]),
        mass_gev=mass_gev,
        settings=effective_settings,
        workers=workers,
    )
    state = _update_state(
        point_dir,
        state,
        status="rangefinder_complete",
        rangefinder=rangefinder_record,
        final_event_grid=final_event_counts.tolist(),
    )
    if stop_after == "rangefinder":
        return None

    pilot_dir = None
    pilot_summary = pilot_detailed = pilot_curve = None
    pilot_threshold = None
    for pilot_attempt in range(3):
        pilot_dir, pilot_summary, pilot_detailed, pilot_curve = _profiled_stage(
            base_config=config,
            bank_dir=bank_dir,
            mass_gev=mass_gev,
            output_base=(
                point_dir / "profiled" / f"full_domain_2k_plan_{pilot_attempt:02d}"
            ),
            event_counts=final_event_counts,
            pseudoexperiments=(
                effective_settings.pseudoexperiments.
                full_domain_pilot_pseudoexperiments
            ),
            number_of_seeds=(
                effective_settings.pseudoexperiments.final_seeds
            ),
            workers=workers,
            truth_subset_path=None,
            make_plots=False,
        )
        pilot_threshold = _threshold_from_summary(pilot_summary)
        if (
            pilot_threshold is not None
            and _unit_crossing_is_resolved(pilot_curve, pilot_threshold)
        ):
            break

        # The compact range finder can miss the full-domain limiter. Use the
        # completed full-domain pilot itself to recenter or extend the one final
        # cache-stable event grid, rather than failing the overnight scan.
        pilot_bracket = rangefinder_bracket(
            pilot_curve, effective_settings.pseudoexperiments
        )
        replanned = final_event_grid_from_bracket(
            pilot_bracket, effective_settings.pseudoexperiments
        )
        if np.array_equal(replanned, final_event_counts):
            break
        final_event_counts = replanned
        write_dataframe(
            pd.DataFrame({"number_of_events": final_event_counts}),
            point_dir / f"final_event_grid_pilot_retry_{pilot_attempt + 1:02d}.csv",
        )
        state = _update_state(
            point_dir,
            state,
            status="full_domain_pilot_replanned",
            final_event_grid=final_event_counts.tolist(),
            pilot_replan_attempt=pilot_attempt + 1,
        )

    if pilot_threshold is None:
        raise AdaptivePointError(
            "Full-domain pilot did not reach 90% after adaptive event-grid "
            "extension. Raise --rangefinder-maximum-events for this point."
        )
    if not _unit_crossing_is_resolved(pilot_curve, pilot_threshold):
        raise AdaptivePointError(
            "Full-domain pilot crossing remains outside the unit-spaced window "
            "after adaptive replanning. Increase --unit-window-half-width."
        )
    critical_counts = _critical_event_counts(pilot_curve, pilot_threshold)
    selected_indices, ranking = select_hard_truth_indices(
        bank,
        pilot_detailed,
        pilot_curve,
        distances,
        critical_counts,
        settings.pseudoexperiments,
    )
    truth_dir = point_dir / "truth_subsets"
    truth_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(ranking, truth_dir / "full_domain_2k_ranking.csv")
    selected_subset_path = _write_subset(
        bank, selected_indices, truth_dir / "selected_stage_00.csv"
    )
    state = _update_state(
        point_dir,
        state,
        status="full_domain_pilot_complete",
        pilot_dir=portable_path(pilot_dir),
        pilot_threshold=pilot_threshold,
        selected_photon_truths=len(selected_indices["photon"]),
        selected_su2_truths=len(selected_indices["su2"]),
    )
    if stop_after == "pilot":
        return None

    thresholds: list[int | None] = [pilot_threshold]
    final_level = effective_settings.pseudoexperiments.full_domain_pilot_pseudoexperiments
    final_detailed = pilot_detailed
    final_curve = pilot_curve
    final_audit = None
    final_selected = selected_indices
    convergence_status = "maximum_PE_reached_unstable"

    for stage_index, level in enumerate(
        effective_settings.pseudoexperiments.pseudoexperiment_ladder, start=1
    ):
        stage_selected = final_selected
        promotion_round = 0
        while True:
            selected_subset_path = _write_subset(
                bank,
                stage_selected,
                truth_dir
                / f"selected_stage_{stage_index:02d}_promotion_{promotion_round:02d}.csv",
            )
            stage_dir, stage_summary, stage_detailed, stage_curve = _profiled_stage(
                base_config=config,
                bank_dir=bank_dir,
                mass_gev=mass_gev,
                output_base=(
                    point_dir
                    / "profiled"
                    / f"selected_{level}_promotion_{promotion_round:02d}"
                ),
                event_counts=final_event_counts,
                pseudoexperiments=level,
                number_of_seeds=effective_settings.pseudoexperiments.final_seeds,
                workers=workers,
                truth_subset_path=selected_subset_path,
                make_plots=False,
            )
            stage_threshold = _threshold_from_summary(stage_summary)
            if stage_threshold is None:
                raise AdaptivePointError(
                    f"Selected-truth {level}-PE stage did not reach 90%."
                )
            if not _unit_crossing_is_resolved(stage_curve, stage_threshold):
                raise AdaptivePointError(
                    f"The {level}-PE crossing lies outside the unit window."
                )

            omitted = omitted_truth_indices(bank, stage_selected)
            omitted_detailed = _filter_truths(pilot_detailed, omitted)
            # Auditing at sub-final PE levels can promote many truths merely
            # because the selected envelope is still noisy. Delay promotions
            # until the first level that is eligible to become final.
            audit_is_due = (
                level
                >= effective_settings.pseudoexperiments.minimum_final_pseudoexperiments
            )
            if omitted_detailed.empty or not audit_is_due:
                audit = None
                promotions = {
                    model: np.asarray([], dtype=int) for model in TRUTH_MODELS
                }
            else:
                persistence_curve = stage_curve.loc[
                    stage_curve["number_of_events"].astype(int)
                    >= int(stage_threshold)
                ].copy()
                audit = audit_omitted_truths(
                    omitted_detailed,
                    persistence_curve,
                    total_truth_count=(
                        len(bank.photon_ctau_m) + len(bank.su2_ctau_m)
                    ),
                    number_of_seeds=(
                        effective_settings.pseudoexperiments.final_seeds
                    ),
                    global_alpha=(
                        effective_settings.pseudoexperiments.audit_global_alpha
                    ),
                )
                audit_dir = (
                    point_dir
                    / "audits"
                    / f"pe_{level}_promotion_{promotion_round:02d}"
                )
                write_dataframe(audit.point_table, audit_dir / "audit_points.csv")
                write_dataframe(audit.truth_summary, audit_dir / "audit_summary.csv")
                promotions = {
                    model: audit.overlapping_truths.loc[
                        audit.overlapping_truths["truth_model"] == model,
                        "truth_lifetime_index",
                    ].to_numpy(dtype=int)
                    for model in TRUTH_MODELS
                }

            if all(len(promotions[model]) == 0 for model in TRUTH_MODELS):
                final_audit = audit
                final_selected = stage_selected
                final_level = int(level)
                final_detailed = stage_detailed
                final_curve = stage_curve
                thresholds.append(stage_threshold)
                state = _update_state(
                    point_dir,
                    state,
                    status=f"pe_{level}_complete",
                    current_profiled_dir=portable_path(stage_dir),
                    current_threshold=stage_threshold,
                    threshold_history=thresholds,
                    selected_photon_truths=len(final_selected["photon"]),
                    selected_su2_truths=len(final_selected["su2"]),
                )
                break

            new_selected = merge_truth_indices(stage_selected, promotions)
            if all(
                np.array_equal(new_selected[model], stage_selected[model])
                for model in TRUTH_MODELS
            ):
                raise AdaptivePointError("Omitted-truth promotion made no progress.")
            stage_selected = new_selected
            promotion_round += 1
            if promotion_round > 4:
                raise AdaptivePointError("Too many omitted-truth promotion rounds.")

        if (
            level >= effective_settings.pseudoexperiments.minimum_final_pseudoexperiments
            and threshold_history_is_stable(
                thresholds, effective_settings.pseudoexperiments
            )
        ):
            convergence_status = "converged"
            break

    final_threshold = minimum_persistent_events(
        final_curve,
        accuracy_column="worst_case_correct_fraction",
        target_accuracy=effective_settings.pseudoexperiments.target_accuracy,
    )
    if final_threshold is None:
        raise AdaptivePointError("Final selected-truth curve has no persistent crossing.")
    omitted = omitted_truth_indices(bank, final_selected)
    if final_audit is not None and not final_audit.overlapping_truths.empty:
        convergence_status = "omitted_truth_overlap"
    if bank_status not in {"lifetime_grid_converged", "fine_binning_converged"}:
        convergence_status = bank_status + "+" + convergence_status

    diagnostics = monte_carlo_threshold_diagnostics(
        final_detailed,
        final_curve,
        target_accuracy=effective_settings.pseudoexperiments.target_accuracy,
        global_alpha=effective_settings.pseudoexperiments.audit_global_alpha,
        total_truth_count=len(bank.photon_ctau_m) + len(bank.su2_ctau_m),
        number_of_seeds=effective_settings.pseudoexperiments.final_seeds,
    )
    limiting = _final_limiting_row(final_curve, final_threshold)
    row = result_row(
        mass_gev=mass_gev,
        selection_name=selection_name,
        status=convergence_status,
        threshold=final_threshold,
        diagnostics=diagnostics,
        final_pseudoexperiments=final_level,
        selected_truths=final_selected,
        omitted_truths=omitted,
        number_of_energy_bins=bank.number_of_energy_bins,
        minimum_distance=float(distance_summary["minimum_D_TV"]),
        limiting_row=limiting,
        runtime_seconds=perf_counter() - started,
        lifetime_rounds=lifetime_rounds,
        profile_lifetime_counts={
            "photon": len(bank.photon_ctau_m),
            "su2": len(bank.su2_ctau_m),
        },
        audit=final_audit,
    )
    final_payload = {
        **row,
        "workflow": WORKFLOW_NAME,
        "profile": profile,
        "domain_path": portable_path(domain_path),
        "bank_dir": portable_path(bank_dir),
        "event_counts": final_event_counts.tolist(),
        "event_grid_specification": event_grid_specification(final_event_counts),
        "threshold_history": thresholds,
        "selected_photon_truth_indices": final_selected["photon"].tolist(),
        "selected_su2_truth_indices": final_selected["su2"].tolist(),
        "omitted_photon_truth_indices": omitted["photon"].tolist(),
        "omitted_su2_truth_indices": omitted["su2"].tolist(),
        "settings": effective_settings.as_dict(),
    }
    _write_json(final_payload, final_json)
    failure_path = point_dir / "failure.json"
    if failure_path.is_file():
        failure_path.unlink()
    _update_state(
        point_dir,
        state,
        status=convergence_status,
        final_result=portable_path(final_json),
    )
    return final_payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    settings = settings_from_arguments(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    settings_path = output_dir / "adaptive_settings.json"
    settings_payload = {
        "profile": args.profile,
        "settings": settings.as_dict(),
        "settings_fingerprint": _settings_fingerprint(settings, args.profile),
    }
    if settings_path.is_file():
        existing = _read_json(settings_path)
        if existing != settings_payload:
            raise ValueError(
                "Output directory contains different adaptive settings. Choose "
                "a new --output-dir instead of mixing provenance."
            )
    else:
        _write_json(settings_payload, settings_path)

    for path in args.import_result_json:
        _upsert_result(output_dir, _import_result_json(path))
    for path in args.import_results_csv:
        imported = pd.read_csv(path)
        for row in imported.to_dict(orient="records"):
            _upsert_result(output_dir, row)

    if args.plot_only:
        _write_master_plot(output_dir)
        print(output_dir / "adaptive_n90_results.csv")
        return

    if not args.domain_path.is_file():
        raise FileNotFoundError(f"Week-8 domain table not found: {args.domain_path}")
    if not args.masses:
        raise ValueError("--masses is required unless --plot-only is used.")
    domains = pd.read_csv(args.domain_path)
    masses = _resolve_masses(args.masses, domains)
    selections = tuple(dict.fromkeys(args.selections))

    initial_grid_plan = []
    for mass_gev in masses:
        initial_grid = initial_adaptive_lifetime_grid(
            domains, mass_gev, settings.lifetime
        )
        for model, group in initial_grid.groupby("model", sort=False):
            domain_rows = domains.loc[
                np.isclose(
                    domains["mass_GeV"].to_numpy(dtype=float),
                    mass_gev,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                & (domains["model"] == model)
            ]
            width_decades = float(
                np.log10(
                    domain_rows["ctau_max_m"].to_numpy(dtype=float)
                    / domain_rows["ctau_min_m"].to_numpy(dtype=float)
                ).sum()
            )
            initial_grid_plan.append(
                {
                    "mass_GeV": mass_gev,
                    "model": model,
                    "number_of_connected_intervals": int(
                        group["interval_index"].nunique()
                    ),
                    "total_log10_ctau_width": width_decades,
                    "initial_lifetime_points": int(len(group)),
                }
            )

    plan = {
        "masses_GeV": list(masses),
        "selections": list(selections),
        "profile": args.profile,
        "workers": args.workers,
        "domain_path": portable_path(args.domain_path),
        "output_dir": portable_path(output_dir),
        "stop_after": args.stop_after,
        "automatic_resume": True,
        "initial_lifetime_grid_plan": initial_grid_plan,
        "settings": settings.as_dict(),
    }
    _write_json(plan, output_dir / "latest_run_plan.json")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    # Run both selections for one mass consecutively. The second selection can
    # then reuse every selection-independent EventCalc proposal just produced
    # for the first, and an interrupted overnight run still yields paired points.
    for mass_gev in masses:
        for selection_name in selections:
            if (
                not args.rerun_final_points
                and _master_point_is_final(
                    output_dir, mass_gev, selection_name
                )
            ):
                print(
                    f"Skipping finalized point m_a={mass_gev:g} GeV, "
                    f"selection={selection_name}"
                )
                continue
            started = perf_counter()
            print("\n" + "=" * 78)
            print(
                f"Adaptive Week-8 point: m_a={mass_gev:g} GeV, "
                f"selection={selection_name}"
            )
            print("=" * 78)
            try:
                result = run_point(
                    mass_gev=mass_gev,
                    selection_name=selection_name,
                    profile=args.profile,
                    domain_path=args.domain_path,
                    domains=domains,
                    output_dir=output_dir,
                    settings=settings,
                    workers=args.workers,
                    stop_after=args.stop_after,
                    skip_conditional_binning_check=(
                        args.skip_conditional_binning_check
                    ),
                    diagnostic_plots=args.diagnostic_plots,
                )
                if result is not None:
                    table = _upsert_result(output_dir, result)
                    _write_master_plot(output_dir)
                    print(table.tail(10).to_string(index=False))
                else:
                    print(f"Stopped after requested stage: {args.stop_after}")
            except Exception as error:  # one failed mass must not kill overnight scan
                failure_dir = _point_dir(output_dir, mass_gev, selection_name)
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_payload = {
                    "mass_GeV": mass_gev,
                    "selection_name": selection_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_json(failure_payload, failure_dir / "failure.json")
                row = _failure_row(
                    mass_gev,
                    selection_name,
                    perf_counter() - started,
                    status=f"failed:{type(error).__name__}",
                )
                _upsert_result(output_dir, row)
                _write_master_plot(output_dir)
                print(f"FAILED: {error}")
                if args.fail_fast:
                    raise

    _write_master_plot(output_dir)
    print("\nAdaptive Week-8 scan outputs:")
    print(f"  {output_dir / 'adaptive_n90_results.csv'}")
    print(f"  {output_dir / 'week8_n90_comparison.pdf'}")
    print(f"  {output_dir / 'week8_n90_comparison.png'}")


if __name__ == "__main__":
    main()
