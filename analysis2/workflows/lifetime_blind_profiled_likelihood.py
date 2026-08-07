"""Run the ECAL-aware, independently lifetime-profiled frozen-reference classifier."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from analysis2.cache import CacheStore, file_fingerprint
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
    TRUTH_MODELS,
    cached_profiled_seed,
    input_fingerprint,
    validated_truth_indices,
)


PROFILED_ACCURACY_WITH_DOMAIN_COLUMNS = (
    "mass_GeV",
    "seed",
    "truth_model",
    "truth_lifetime_index",
    "truth_interval_index",
    "truth_ctau_m",
    "number_of_events",
    "number_of_pseudoexperiments",
    "correct_fraction",
    "selected_photon_fraction",
    "selected_su2_fraction",
    "tie_fraction",
    "mean_profile_statistic_T",
    "std_profile_statistic_T",
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
    "number_of_photon_allowed_intervals",
    "number_of_su2_allowed_intervals",
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
    "limiting_truth_interval_index_at_threshold",
    "limiting_truth_ctau_m_at_threshold",
)


@dataclass(frozen=True)
class ProfiledWorkflowResult:
    summary: pd.DataFrame
    manifest_path: Path
    artifacts: tuple[Path, ...]
    elapsed_seconds: float
    cache_stats: dict[str, int]


@dataclass(frozen=True)
class _ProfiledSeedTask:
    bank_path: Path
    config: AnalysisConfig
    seed: int
    event_counts: tuple[int, ...]
    cache_profile: str
    cache_root: Path
    cache_enabled: bool
    force: bool
    truth_indices: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class _ProfiledSeedResult:
    seed: int
    detailed: pd.DataFrame
    cache_stats: dict[str, int]
    pseudoexperiment_ranges: tuple[tuple[int, int], ...]


def _run_profiled_seed_task(task: _ProfiledSeedTask) -> _ProfiledSeedResult:
    """Run one independent mass/seed task in a spawned worker process."""
    bank = load_template_bank(task.bank_path)
    worker_cache = CacheStore(
        task.cache_profile,
        root=task.cache_root,
        enabled=task.cache_enabled,
    )
    detailed = cached_profiled_seed(
        bank,
        task.bank_path,
        task.config,
        task.seed,
        np.asarray(task.event_counts, dtype=int),
        worker_cache,
        force=task.force,
        truth_indices={
            model: np.asarray(indices, dtype=int)
            for model, indices in task.truth_indices
        },
    )
    ranges = tuple(
        (int(lower), int(upper))
        for lower, upper in detailed.attrs.get(
            "pseudoexperiment_ranges",
            [[0, task.config.profiled_likelihood.pseudoexperiments_per_truth_and_seed]],
        )
    )
    return _ProfiledSeedResult(
        seed=task.seed,
        detailed=detailed,
        cache_stats=worker_cache.counter_snapshot(),
        pseudoexperiment_ranges=ranges,
    )


_CACHE_COUNTER_NAMES = ("hits", "misses", "writes", "rejected")


def _add_cache_stats(
    total: dict[str, int],
    increment: dict[str, int],
) -> None:
    for name in _CACHE_COUNTER_NAMES:
        total[name] = int(total.get(name, 0)) + int(increment.get(name, 0))


def _run_profiled_seed_jobs(
    *,
    bank: LifetimeTemplateBank,
    bank_path: Path,
    config: AnalysisConfig,
    seeds: tuple[int, ...],
    event_counts: np.ndarray,
    cache: CacheStore,
    force: bool,
    workers: int,
    truth_indices: Mapping[str, np.ndarray],
) -> tuple[list[pd.DataFrame], dict[str, int]]:
    """Run mass-seed jobs serially or with at most two spawned workers.

    Results are returned in the configured seed order, independent of worker
    completion order. Each worker owns its CacheStore counters and writes only
    unique seed-and-truth cache keys using the existing atomic cache adapter.
    """
    if workers == 1:
        frames = [
            cached_profiled_seed(
                bank,
                bank_path,
                config,
                seed,
                event_counts,
                cache,
                force=force,
                truth_indices=truth_indices,
            )
            for seed in seeds
        ]
        return frames, cache.counter_snapshot()

    tasks = [
        _ProfiledSeedTask(
            bank_path=bank_path,
            config=config,
            seed=int(seed),
            event_counts=tuple(int(value) for value in event_counts),
            cache_profile=cache.profile,
            cache_root=cache.root,
            cache_enabled=cache.enabled,
            force=force,
            truth_indices=_truth_selection_payload(truth_indices),
        )
        for seed in seeds
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        results = list(executor.map(_run_profiled_seed_task, tasks))

    returned_seeds = tuple(result.seed for result in results)
    if returned_seeds != seeds:
        raise RuntimeError("Parallel profiled results returned in a different seed order.")

    cache_stats = {name: 0 for name in _CACHE_COUNTER_NAMES}
    frames = []
    for result in results:
        _add_cache_stats(cache_stats, result.cache_stats)
        result.detailed.attrs["pseudoexperiment_ranges"] = [
            [lower, upper]
            for lower, upper in result.pseudoexperiment_ranges
        ]
        frames.append(result.detailed)
    return frames, cache_stats


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Template-bank directory; defaults to the selected profile output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to the selected profile output.",
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        help="Optional subset of configured masses.",
    )
    event_group = parser.add_mutually_exclusive_group()
    event_group.add_argument("--maximum-observed-events", type=int)
    event_group.add_argument(
        "--event-count-grid",
        type=str,
        help=(
            "Explicit observed-event grid. Use comma-separated integers and "
            "inclusive ranges START:STOP[:STEP], for example "
            "'1:300,350:1000:10'."
        ),
    )
    parser.add_argument("--pseudoexperiments-per-truth-and-seed", type=int)
    parser.add_argument("--number-of-seeds", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument(
        "--truth-subset-path",
        type=Path,
        help=(
            "CSV selecting truth hypotheses by mass_GeV, truth_model and "
            "truth_lifetime_index. Profile lifetime grids remain complete. "
            "Subset thresholds are screening diagnostics, not final domain-wide results."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Number of independent mass-seed worker processes. "
            "Restricted to one or two on the intended laptop."
        ),
    )
    return parser.parse_args(argv)


def parse_event_count_grid(specification: str) -> np.ndarray:
    """Parse a compact positive-integer event-count grid.

    Tokens are comma separated.  A token is either one integer or an inclusive
    ``START:STOP[:STEP]`` range.  The returned grid is sorted and deduplicated.
    """
    if not isinstance(specification, str) or not specification.strip():
        raise ValueError("--event-count-grid must not be empty")

    counts: set[int] = set()
    for raw_token in specification.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("--event-count-grid contains an empty token")
        parts = token.split(":")
        try:
            values = [int(part) for part in parts]
        except ValueError as error:
            raise ValueError(
                f"Invalid event-count token {token!r}; expected integers."
            ) from error

        if len(values) == 1:
            start = stop = values[0]
            step = 1
        elif len(values) in {2, 3}:
            start, stop = values[:2]
            step = values[2] if len(values) == 3 else 1
        else:
            raise ValueError(
                f"Invalid event-count token {token!r}; use N or START:STOP[:STEP]."
            )
        if start < 1 or stop < 1:
            raise ValueError("Observed event counts must be positive")
        if stop < start:
            raise ValueError(
                f"Event-count range {token!r} has STOP < START"
            )
        if step < 1:
            raise ValueError("Event-count range steps must be positive")
        counts.update(range(start, stop + 1, step))
        counts.add(stop)

    return np.asarray(sorted(counts), dtype=int)



_TRUTH_SUBSET_REQUIRED_COLUMNS = {
    "mass_GeV",
    "truth_model",
    "truth_lifetime_index",
}


def load_truth_subset_table(path: Path) -> pd.DataFrame:
    """Load a portable explicit truth-hypothesis selection."""
    if not path.is_file():
        raise FileNotFoundError(f"Truth-subset CSV not found: {path}")
    table = pd.read_csv(path)
    missing = _TRUTH_SUBSET_REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(
            f"Truth-subset CSV is missing columns: {sorted(missing)}"
        )
    if table.empty:
        raise ValueError("Truth-subset CSV contains no rows.")

    result = table.copy()
    result["mass_GeV"] = pd.to_numeric(result["mass_GeV"], errors="raise")
    raw_indices = pd.to_numeric(
        result["truth_lifetime_index"],
        errors="raise",
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(raw_indices)):
        raise ValueError("Truth-subset indices must be finite.")
    integer_indices = np.rint(raw_indices).astype(int)
    if not np.array_equal(raw_indices, integer_indices):
        raise ValueError("Truth-subset indices must be integers.")
    if np.any(integer_indices < 0):
        raise ValueError("Truth-subset indices cannot be negative.")
    result["truth_lifetime_index"] = integer_indices
    result["truth_model"] = result["truth_model"].astype(str)
    unknown = sorted(set(result["truth_model"]) - set(TRUTH_MODELS))
    if unknown:
        raise ValueError(
            "Unknown truth models in subset: " + ", ".join(unknown)
        )
    if result.duplicated(
        ["mass_GeV", "truth_model", "truth_lifetime_index"]
    ).any():
        raise ValueError("Truth-subset CSV contains duplicate hypotheses.")
    return result.sort_values(
        ["mass_GeV", "truth_model", "truth_lifetime_index"],
        ignore_index=True,
    )


def resolve_truth_subset_indices(
    bank: LifetimeTemplateBank,
    config: AnalysisConfig,
    subset_table: pd.DataFrame | None,
) -> dict[str, np.ndarray]:
    """Resolve and validate global bank indices for one mass."""
    if subset_table is None:
        return validated_truth_indices(bank, config)

    mass_values = subset_table["mass_GeV"].to_numpy(dtype=float)
    selected = subset_table.loc[
        np.isclose(
            mass_values,
            bank.mass_gev,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()
    if selected.empty:
        raise ValueError(
            f"Truth-subset CSV has no rows for m_a={bank.mass_gev:g} GeV."
        )

    indices: dict[str, np.ndarray] = {}
    for model in TRUTH_MODELS:
        rows = selected.loc[selected["truth_model"] == model]
        if rows.empty:
            raise ValueError(
                f"Truth-subset CSV must retain at least one {model} truth."
            )
        indices[model] = rows["truth_lifetime_index"].to_numpy(dtype=int)

    resolved = validated_truth_indices(bank, config, indices)
    optional_checks = (
        ("truth_ctau_m", "ctau_m"),
        ("truth_interval_index", "interval_index"),
    )
    for column, quantity in optional_checks:
        if column not in selected:
            continue
        for row in selected.itertuples(index=False):
            model = str(row.truth_model)
            index = int(row.truth_lifetime_index)
            if quantity == "ctau_m":
                expected = float(
                    bank.photon_ctau_m[index]
                    if model == "photon"
                    else bank.su2_ctau_m[index]
                )
                actual = float(getattr(row, column))
                if not np.isclose(actual, expected, rtol=1.0e-12, atol=0.0):
                    raise ValueError(
                        f"Truth-subset lifetime disagrees with {model} index {index}."
                    )
            else:
                expected = int(
                    bank.photon_interval_index[index]
                    if model == "photon"
                    else bank.su2_interval_index[index]
                )
                actual = int(getattr(row, column))
                if actual != expected:
                    raise ValueError(
                        f"Truth-subset interval disagrees with {model} index {index}."
                    )
    return resolved


def _truth_selection_payload(
    truth_indices: Mapping[str, np.ndarray],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            model,
            tuple(int(index) for index in truth_indices[model]),
        )
        for model in TRUTH_MODELS
    )

def _validated_event_counts(event_counts: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(event_counts)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("At least one observed event count is required")
    integer_values = values.astype(int)
    if not np.array_equal(values, integer_values):
        raise ValueError("Observed event counts must be integers")
    if np.any(integer_values < 1):
        raise ValueError("Observed event counts must be positive")
    return np.unique(integer_values)


def profiled_run_axes(
    config: AnalysisConfig,
    event_counts: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the tested event counts and pseudoexperiment seeds."""
    settings = config.profiled_likelihood
    resolved = (
        np.arange(1, settings.maximum_observed_events + 1, dtype=int)
        if event_counts is None
        else _validated_event_counts(event_counts)
    )
    return resolved, settings.seeds


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


def _discover_bank_paths(directory: Path) -> tuple[tuple[float, Path], ...]:
    """Discover unique template banks and order them by their stored mass."""
    candidates = sorted(directory.glob("template_bank_ma_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No template banks found in {directory}.")

    discovered: list[tuple[float, Path]] = []
    for path in candidates:
        bank = load_template_bank(path)
        mass = float(bank.mass_gev)
        if any(np.isclose(mass, other, rtol=0.0, atol=1.0e-12) for other, _ in discovered):
            raise ValueError(
                f"Multiple template banks in {directory} contain m_a={mass:g} GeV."
            )
        discovered.append((mass, path))
    return tuple(sorted(discovered, key=lambda item: item[0]))


def resolve_bank_paths(
    config: AnalysisConfig,
    *,
    input_dir: Path | None = None,
    masses: Sequence[float] | None = None,
) -> tuple[Path, ...]:
    """Resolve banks from the input directory, not the legacy config mass list."""
    directory = input_dir or (
        profile_output_dir(config.name, "lifetime_blind_discrimination")
        / "template_banks"
    )
    discovered = _discover_bank_paths(directory)
    if masses is None:
        return tuple(path for _, path in discovered)

    selected: list[Path] = []
    for requested in masses:
        matches = [
            path
            for mass, path in discovered
            if np.isclose(float(requested), mass, rtol=0.0, atol=1.0e-12)
        ]
        if len(matches) != 1:
            available = ", ".join(f"{mass:g}" for mass, _ in discovered)
            raise ValueError(
                f"Mass {float(requested):g} GeV is not available in {directory}. "
                f"Available masses: {available}."
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    if not selected:
        raise ValueError("At least one available template-bank mass is required.")
    return tuple(selected)


def _truth_interval_index(
    bank: LifetimeTemplateBank,
    truth_model: str,
    truth_lifetime_index: int,
) -> int:
    if truth_model == "photon":
        values = bank.photon_interval_index
    elif truth_model == "su2":
        values = bank.su2_interval_index
    else:
        raise ValueError(f"Unknown truth model: {truth_model}")
    index = int(truth_lifetime_index)
    if index < 0 or index >= len(values):
        raise ValueError(
            f"Truth lifetime index {index} lies outside the {truth_model} grid."
        )
    return int(values[index])


def add_truth_interval_provenance(
    detailed: pd.DataFrame,
    bank: LifetimeTemplateBank,
) -> pd.DataFrame:
    """Attach the connected allowed-domain component to each truth row."""
    result = detailed.copy()
    result.insert(
        result.columns.get_loc("truth_lifetime_index") + 1,
        "truth_interval_index",
        [
            _truth_interval_index(bank, str(model), int(index))
            for model, index in zip(
                result["truth_model"],
                result["truth_lifetime_index"],
            )
        ],
    )
    return result.loc[:, PROFILED_ACCURACY_WITH_DOMAIN_COLUMNS]


def add_limiting_interval_provenance(
    conservative_curve: pd.DataFrame,
    bank: LifetimeTemplateBank,
) -> pd.DataFrame:
    """Attach the allowed-domain component of the conservative limiting truth."""
    result = conservative_curve.copy()
    result.insert(
        result.columns.get_loc("limiting_truth_lifetime_index") + 1,
        "limiting_truth_interval_index",
        [
            _truth_interval_index(bank, str(model), int(index))
            for model, index in zip(
                result["limiting_truth_model"],
                result["limiting_truth_lifetime_index"],
            )
        ],
    )
    return result


def _validate_disconnected_domain_settings(
    bank: LifetimeTemplateBank,
    config: AnalysisConfig,
) -> None:
    """Require the full saved grid when a model has disconnected domains."""
    has_disconnected_domain = (
        len(bank.photon_allowed_intervals_m) > 1
        or len(bank.su2_allowed_intervals_m) > 1
    )
    settings = config.profiled_likelihood
    if has_disconnected_domain and (
        settings.truth_lifetime_grid != "all"
        or settings.profile_lifetime_grid != "all"
    ):
        raise ValueError(
            "Disconnected Week-8 lifetime domains require truth_grid='all' "
            "and profile_grid='all'."
        )


def summarize_mass_threshold(
    bank: LifetimeTemplateBank,
    conservative_curve: pd.DataFrame,
    config: AnalysisConfig,
    *,
    truth_indices: Mapping[str, np.ndarray] | None = None,
    truth_grid_label: str | None = None,
) -> dict:
    """Build the legacy-compatible persistent-threshold record."""
    settings = config.profiled_likelihood
    selected_truth = validated_truth_indices(bank, config, truth_indices)
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
        "truth_grid": truth_grid_label or settings.truth_lifetime_grid,
        "profile_grid": settings.profile_lifetime_grid,
        "number_of_photon_truth_lifetimes": len(selected_truth["photon"]),
        "number_of_su2_truth_lifetimes": len(selected_truth["su2"]),
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
        "number_of_photon_allowed_intervals": len(
            bank.photon_allowed_intervals_m
        ),
        "number_of_su2_allowed_intervals": len(bank.su2_allowed_intervals_m),
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
                "limiting_truth_interval_index_at_threshold": -1,
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
                "limiting_truth_interval_index_at_threshold": (
                    _truth_interval_index(
                        bank,
                        str(row["limiting_truth_model"]),
                        int(row["limiting_truth_lifetime_index"]),
                    )
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
    event_counts: Sequence[int] | np.ndarray | None = None,
    workers: int = 1,
    truth_subset_path: Path | None = None,
) -> ProfiledWorkflowResult:
    """Run all configured seeds, reductions, tables, plots, and manifest."""
    _validate_settings(config)
    started = perf_counter()
    bank_paths = resolve_bank_paths(config, input_dir=input_dir, masses=masses)
    if truth_subset_path is not None and output_dir is None:
        raise ValueError(
            "--truth-subset-path requires an explicit output directory."
        )
    destination = output_dir or profile_output_dir(
        config.name, "lifetime_blind_profiled_likelihood"
    )
    truth_subset_table = (
        None
        if truth_subset_path is None
        else load_truth_subset_table(truth_subset_path)
    )
    table_dir = destination / "tables"
    plot_dir = destination / "plots"
    cache = cache or CacheStore(config.name)
    if cache.profile != config.name:
        raise ValueError("Cache and analysis profiles differ.")
    event_counts, seeds = profiled_run_axes(config, event_counts)
    if workers not in {1, 2}:
        raise ValueError("workers must be one or two")
    artifacts: list[Path] = []
    summary_rows = []
    domain_provenance: list[dict] = []
    truth_selection_provenance: list[dict] = []
    pseudoexperiment_range_provenance: list[dict] = []
    parallel_cache_stats = {name: 0 for name in _CACHE_COUNTER_NAMES}

    for bank_path in bank_paths:
        bank = load_template_bank(bank_path)
        _validate_disconnected_domain_settings(bank, config)
        expected_mass = float(bank_path.stem.removeprefix("template_bank_ma_").replace("p", "."))
        if not np.isclose(bank.mass_gev, expected_mass, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Template-bank mass disagrees with {bank_path.name}.")
        truth_indices = resolve_truth_subset_indices(
            bank,
            config,
            truth_subset_table,
        )
        seed_frames, mass_cache_stats = _run_profiled_seed_jobs(
            bank=bank,
            bank_path=bank_path,
            config=config,
            seeds=seeds,
            event_counts=event_counts,
            cache=cache,
            force=force,
            workers=workers,
            truth_indices=truth_indices,
        )
        for seed, frame in zip(seeds, seed_frames):
            ranges = frame.attrs.get(
                "pseudoexperiment_ranges",
                [[0, config.profiled_likelihood.pseudoexperiments_per_truth_and_seed]],
            )
            pseudoexperiment_range_provenance.append(
                {
                    "mass_GeV": bank.mass_gev,
                    "seed": int(seed),
                    "contributing_ranges": [
                        [int(lower), int(upper)]
                        for lower, upper in ranges
                    ],
                }
            )
        detailed = pd.concat(seed_frames, ignore_index=True)
        if workers == 2:
            _add_cache_stats(parallel_cache_stats, mass_cache_stats)
        seed_worst = build_seed_worst_case_table(detailed)
        conservative = build_conservative_seed_envelope(seed_worst)
        detailed_output = add_truth_interval_provenance(detailed, bank)
        conservative = add_limiting_interval_provenance(conservative, bank)
        threshold_summary = summarize_mass_threshold(
            bank,
            conservative,
            config,
            truth_indices=truth_indices,
            truth_grid_label=(
                "custom_subset"
                if truth_subset_table is not None
                else None
            ),
        )
        threshold = (
            int(threshold_summary["minimum_persistent_events"])
            if threshold_summary["threshold_reached"]
            else None
        )
        token = float_token(bank.mass_gev)
        outputs = (
            (detailed_output, table_dir / f"profiled_accuracy_ma_{token}.csv"),
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
        domain_provenance.append(
            {
                "mass_GeV": bank.mass_gev,
                "event_level_geom_only": bank.event_threshold,
                "photon_allowed_intervals_m": (
                    bank.photon_allowed_intervals_m.tolist()
                ),
                "su2_allowed_intervals_m": bank.su2_allowed_intervals_m.tolist(),
            }
        )
        truth_selection_provenance.append(
            {
                "mass_GeV": bank.mass_gev,
                "photon_truth_indices": truth_indices["photon"].tolist(),
                "su2_truth_indices": truth_indices["su2"].tolist(),
                "number_of_photon_truths": len(truth_indices["photon"]),
                "number_of_su2_truths": len(truth_indices["su2"]),
            }
        )

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
    cache_stats = (
        cache.counter_snapshot()
        if workers == 1
        else parallel_cache_stats
    )
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
            "truth_selection_mode": (
                "custom_subset"
                if truth_subset_table is not None
                else "configured_full_grid"
            ),
            "truth_subset_path": (
                portable_path(truth_subset_path)
                if truth_subset_path is not None
                else None
            ),
            "truth_subset_fingerprint": (
                file_fingerprint(truth_subset_path)
                if truth_subset_path is not None
                else None
            ),
            "truth_selection_by_mass": truth_selection_provenance,
            "progressive_truth_level_pseudoexperiment_caching": True,
            "pseudoexperiment_ranges_by_mass_seed": (
                pseudoexperiment_range_provenance
            ),
            "classification_fraction_range_combination": (
                "exact_half_integer_numerators"
            ),
            "profile_statistic_range_combination": (
                "population_first_and_second_moments"
            ),
            "complete_truth_domain_coverage": truth_subset_table is None,
            "threshold_is_screening_only": truth_subset_table is not None,
            "profile_lifetime_grid_reduced_by_truth_subset": False,
            "workers": workers,
            "shape_only": True,
            "conditioned_on_observed_event_count": True,
            "expected_event_rates_used_in_likelihood": False,
            "independent_lifetime_profiling_by_model": True,
            "disconnected_domains_profiled_as_saved_template_unions": True,
            "allowed_lifetime_domains": domain_provenance,
        },
    )
    return ProfiledWorkflowResult(
        summary=summary,
        manifest_path=manifest_path,
        artifacts=tuple(artifacts),
        elapsed_seconds=elapsed,
        cache_stats=cache_stats,
    )

def apply_cli_overrides(config: AnalysisConfig, args) -> AnalysisConfig:
    settings = config.profiled_likelihood
    updates: dict[str, int] = {}

    if args.event_count_grid is not None:
        updates["maximum_observed_events"] = int(
            parse_event_count_grid(args.event_count_grid)[-1]
        )

    if args.maximum_observed_events is not None:
        if args.maximum_observed_events < 1:
            raise ValueError("--maximum-observed-events must be positive")
        updates["maximum_observed_events"] = args.maximum_observed_events

    if args.pseudoexperiments_per_truth_and_seed is not None:
        if args.pseudoexperiments_per_truth_and_seed < 1:
            raise ValueError(
                "--pseudoexperiments-per-truth-and-seed must be positive"
            )
        updates["pseudoexperiments_per_truth_and_seed"] = (
            args.pseudoexperiments_per_truth_and_seed
        )

    if args.number_of_seeds is not None:
        if args.number_of_seeds < 1:
            raise ValueError("--number-of-seeds must be positive")
        updates["number_of_seeds"] = args.number_of_seeds

    if args.chunk_size is not None:
        if args.chunk_size < 1:
            raise ValueError("--chunk-size must be positive")
        updates["chunk_size"] = args.chunk_size

    return replace(
        config,
        profiled_likelihood=replace(settings, **updates),
    )

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = apply_cli_overrides(get_config(args.profile), args)
    cache = CacheStore(config.name, enabled=not args.no_cache)
    event_counts = (
        None
        if args.event_count_grid is None
        else parse_event_count_grid(args.event_count_grid)
    )
    result = run_workflow(
        config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        masses=args.masses,
        cache=cache,
        force=args.force,
        event_counts=event_counts,
        workers=args.workers,
        truth_subset_path=args.truth_subset_path,
    )
    print(result.summary.to_string(index=False))
    print(f"Saved profiled likelihood outputs to {result.manifest_path.parent}")
    print(f"Runtime: {result.elapsed_seconds:.3f} s; cache: {result.cache_stats}")


if __name__ == "__main__":
    main()
