"""Run the ECAL-aware, independently lifetime-profiled frozen-reference classifier."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from analysis2.cache import CacheStore
from analysis2.config import AnalysisConfig, get_config
from analysis2.lifetime_template_banks import LifetimeTemplateBank, load_template_bank
from analysis2.paths import portable_path, profile_output_dir
from analysis2.profiled_reduction import (
    build_conservative_seed_envelope,
    build_seed_worst_case_table,
    minimum_persistent_events,
)
from analysis2.profiled_statistics import lifetime_grid_indices
from analysis2.workflows import (
    add_profile_cache_arguments,
    float_token,
    write_dataframe,
    write_manifest,
)
from analysis2.workflows.profiled_likelihood_cache import (
    cached_profiled_seed,
    input_fingerprint,
)


THRESHOLD_SUMMARY_COLUMNS = (
    "mass_GeV",
    "rebin_factor",
    "number_of_energy_bins",
    "jeffreys_alpha",
    "stored_jeffreys_alpha",
    "truth_grid",
    "profile_grid",
    "number_of_photon_truth_lifetimes",
    "number_of_su2_truth_lifetimes",
    "number_of_photon_profile_lifetimes",
    "number_of_su2_profile_lifetimes",
    "pseudoexperiments_per_truth_and_seed",
    "number_of_seeds",
    "target_accuracy",
    "threshold_reached",
    "minimum_persistent_events",
    "maximum_tested_events",
    "worst_case_accuracy_at_maximum_events",
    "accuracy_at_threshold",
    "limiting_seed_at_threshold",
    "limiting_truth_model_at_threshold",
    "limiting_truth_lifetime_index_at_threshold",
    "limiting_truth_ctau_m_at_threshold",
)


@dataclass(frozen=True)
class ProfiledWorkflowResult:
    summary: pd.DataFrame
    manifest_path: Path
    artifacts: tuple[Path, ...]
    elapsed_seconds: float
    cache_stats: dict[str, int]


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Template-bank directory; defaults to the selected profile output.",
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        help="Optional subset of configured masses.",
    )
    return parser.parse_args(argv)


def profiled_run_axes(config: AnalysisConfig) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the frozen tested event counts and pseudoexperiment seeds."""
    settings = config.profiled_likelihood
    event_counts = np.arange(1, settings.maximum_observed_events + 1, dtype=int)
    return event_counts, settings.seeds


def _validate_settings(config: AnalysisConfig) -> None:
    settings = config.profiled_likelihood
    if not settings.shape_only or not settings.independent_lifetime_profiling:
        raise ValueError("This workflow requires shape-only independent profiling.")
    if settings.persistent_criterion != "all_larger_tested_event_counts":
        raise ValueError("Unknown persistent-threshold convention.")
    if settings.rebin_factor != 1:
        raise ValueError("The frozen workflow uses the saved common bank binning.")
    if settings.truth_lifetime_grid not in {"all", "even", "odd"}:
        raise ValueError("Unknown truth lifetime grid.")
    if settings.profile_lifetime_grid not in {"all", "even", "odd"}:
        raise ValueError("Unknown profile lifetime grid.")


def _configured_masses(
    config: AnalysisConfig,
    requested: Sequence[float] | None,
) -> tuple[float, ...]:
    if requested is None:
        return config.masses_gev
    selected = []
    for value in requested:
        matches = [
            mass
            for mass in config.masses_gev
            if np.isclose(value, mass, rtol=0.0, atol=1.0e-12)
        ]
        if len(matches) != 1:
            raise ValueError(f"Mass {value:g} GeV is not in profile {config.name!r}.")
        if matches[0] not in selected:
            selected.append(matches[0])
    if not selected:
        raise ValueError("At least one configured mass is required.")
    return tuple(selected)


def resolve_bank_paths(
    config: AnalysisConfig,
    *,
    input_dir: Path | None = None,
    masses: Sequence[float] | None = None,
) -> tuple[Path, ...]:
    directory = input_dir or (
        profile_output_dir(config.name, "lifetime_blind_discrimination")
        / "template_banks"
    )
    paths = tuple(
        directory / f"template_bank_ma_{float_token(mass)}.npz"
        for mass in _configured_masses(config, masses)
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing template banks in {directory}: {names}")
    return paths


def summarize_mass_threshold(
    bank: LifetimeTemplateBank,
    conservative_curve: pd.DataFrame,
    config: AnalysisConfig,
) -> dict:
    """Build the legacy-compatible persistent-threshold record."""
    settings = config.profiled_likelihood
    threshold = minimum_persistent_events(
        conservative_curve,
        accuracy_column="worst_case_correct_fraction",
        target_accuracy=settings.target_accuracy,
    )
    maximum_events = int(conservative_curve["number_of_events"].max())
    maximum_row = conservative_curve.loc[
        conservative_curve["number_of_events"] == maximum_events
    ].iloc[0]
    summary = {
        "mass_GeV": bank.mass_gev,
        "rebin_factor": settings.rebin_factor,
        "number_of_energy_bins": bank.number_of_energy_bins,
        "jeffreys_alpha": bank.jeffreys_alpha,
        "stored_jeffreys_alpha": bank.jeffreys_alpha,
        "truth_grid": settings.truth_lifetime_grid,
        "profile_grid": settings.profile_lifetime_grid,
        "number_of_photon_truth_lifetimes": len(
            lifetime_grid_indices(
                len(bank.photon_ctau_m), settings.truth_lifetime_grid
            )
        ),
        "number_of_su2_truth_lifetimes": len(
            lifetime_grid_indices(len(bank.su2_ctau_m), settings.truth_lifetime_grid)
        ),
        "number_of_photon_profile_lifetimes": len(
            lifetime_grid_indices(
                len(bank.photon_ctau_m), settings.profile_lifetime_grid
            )
        ),
        "number_of_su2_profile_lifetimes": len(
            lifetime_grid_indices(
                len(bank.su2_ctau_m), settings.profile_lifetime_grid
            )
        ),
        "pseudoexperiments_per_truth_and_seed": (
            settings.pseudoexperiments_per_truth_and_seed
        ),
        "number_of_seeds": settings.number_of_seeds,
        "target_accuracy": settings.target_accuracy,
        "threshold_reached": threshold is not None,
        "minimum_persistent_events": threshold if threshold is not None else -1,
        "maximum_tested_events": maximum_events,
        "worst_case_accuracy_at_maximum_events": float(
            maximum_row["worst_case_correct_fraction"]
        ),
    }
    if threshold is None:
        summary.update(
            {
                "accuracy_at_threshold": np.nan,
                "limiting_seed_at_threshold": -1,
                "limiting_truth_model_at_threshold": "not_reached",
                "limiting_truth_lifetime_index_at_threshold": -1,
                "limiting_truth_ctau_m_at_threshold": np.nan,
            }
        )
    else:
        row = conservative_curve.loc[
            conservative_curve["number_of_events"] == threshold
        ].iloc[0]
        summary.update(
            {
                "accuracy_at_threshold": float(row["worst_case_correct_fraction"]),
                "limiting_seed_at_threshold": int(row["limiting_seed"]),
                "limiting_truth_model_at_threshold": str(row["limiting_truth_model"]),
                "limiting_truth_lifetime_index_at_threshold": int(
                    row["limiting_truth_lifetime_index"]
                ),
                "limiting_truth_ctau_m_at_threshold": float(
                    row["limiting_truth_ctau_m"]
                ),
            }
        )
    return summary


def run_workflow(
    config: AnalysisConfig,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    masses: Sequence[float] | None = None,
    cache: CacheStore | None = None,
    force: bool = False,
    make_plots: bool = True,
) -> ProfiledWorkflowResult:
    """Run all configured seeds, reductions, tables, plots, and manifest."""
    _validate_settings(config)
    started = perf_counter()
    bank_paths = resolve_bank_paths(config, input_dir=input_dir, masses=masses)
    destination = output_dir or profile_output_dir(
        config.name, "lifetime_blind_profiled_likelihood"
    )
    table_dir = destination / "tables"
    plot_dir = destination / "plots"
    cache = cache or CacheStore(config.name)
    if cache.profile != config.name:
        raise ValueError("Cache and analysis profiles differ.")
    event_counts, seeds = profiled_run_axes(config)
    artifacts: list[Path] = []
    summary_rows = []

    for bank_path in bank_paths:
        bank = load_template_bank(bank_path)
        expected_mass = float(bank_path.stem.removeprefix("template_bank_ma_").replace("p", "."))
        if not np.isclose(bank.mass_gev, expected_mass, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Template-bank mass disagrees with {bank_path.name}.")
        detailed = pd.concat(
            [
                cached_profiled_seed(
                    bank,
                    bank_path,
                    config,
                    seed,
                    event_counts,
                    cache,
                    force=force,
                )
                for seed in seeds
            ],
            ignore_index=True,
        )
        seed_worst = build_seed_worst_case_table(detailed)
        conservative = build_conservative_seed_envelope(seed_worst)
        threshold_summary = summarize_mass_threshold(bank, conservative, config)
        threshold = (
            int(threshold_summary["minimum_persistent_events"])
            if threshold_summary["threshold_reached"]
            else None
        )
        token = float_token(bank.mass_gev)
        outputs = (
            (detailed, table_dir / f"profiled_accuracy_ma_{token}.csv"),
            (seed_worst, table_dir / f"profiled_worst_case_by_seed_ma_{token}.csv"),
            (conservative, table_dir / f"profiled_conservative_curve_ma_{token}.csv"),
            (
                pd.DataFrame([threshold_summary], columns=THRESHOLD_SUMMARY_COLUMNS),
                table_dir / f"profiled_threshold_ma_{token}.csv",
            ),
        )
        for table, path in outputs:
            write_dataframe(table, path)
            artifacts.append(path)
        if make_plots:
            from analysis2.lifetime_blind_plotting import plot_profiled_accuracy

            artifacts.extend(
                plot_profiled_accuracy(
                    conservative,
                    mass_gev=bank.mass_gev,
                    target_accuracy=config.profiled_likelihood.target_accuracy,
                    threshold=threshold,
                    output_stem=plot_dir / f"profiled_accuracy_ma_{token}",
                )
            )
        summary_rows.append(threshold_summary)

    summary = pd.DataFrame(
        summary_rows,
        columns=THRESHOLD_SUMMARY_COLUMNS,
    ).sort_values("mass_GeV", ignore_index=True)
    summary_path = destination / "profiled_threshold_summary.csv"
    write_dataframe(summary, summary_path)
    artifacts.append(summary_path)
    if make_plots:
        from analysis2.lifetime_blind_plotting import plot_profiled_thresholds

        artifacts.extend(
            plot_profiled_thresholds(
                summary,
                output_stem=plot_dir / "profiled_minimum_events_vs_mass",
            )
        )
    elapsed = perf_counter() - started
    cache_stats = cache.counter_snapshot()
    manifest_path = write_manifest(
        config,
        "lifetime_blind_profiled_likelihood",
        destination,
        elapsed_seconds=elapsed,
        cache_stats=cache_stats,
        artifacts=artifacts,
        extra={
            "input_banks": [portable_path(path) for path in bank_paths],
            "input_fingerprints": [input_fingerprint(path) for path in bank_paths],
            "event_counts": event_counts.tolist(),
            "pseudoexperiment_seeds": list(seeds),
        },
    )
    return ProfiledWorkflowResult(
        summary=summary,
        manifest_path=manifest_path,
        artifacts=tuple(artifacts),
        elapsed_seconds=elapsed,
        cache_stats=cache_stats,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = get_config(args.profile)
    cache = CacheStore(config.name, enabled=not args.no_cache)
    result = run_workflow(
        config,
        input_dir=args.input_dir,
        masses=args.masses,
        cache=cache,
        force=args.force,
    )
    print(result.summary.to_string(index=False))
    print(f"Saved profiled likelihood outputs to {result.manifest_path.parent}")
    print(f"Runtime: {result.elapsed_seconds:.3f} s; cache: {result.cache_stats}")


if __name__ == "__main__":
    main()
