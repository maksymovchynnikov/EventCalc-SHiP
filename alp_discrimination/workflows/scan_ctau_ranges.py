"""Scan the exact ECAL-aware observable lifetime domains.

The saved scan contains the legacy coarse grid and every fixed-step bisection
evaluation.  The returned bisection midpoint is a diagnostic only and is not
evaluated or appended to ``ctau_scan.csv``; it is written only to the named
diagnostic tables.  Production template endpoints instead come from local
log--log interpolation of the final saved bracket, followed by the approved
0.2% inward shift in log lifetime.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd

from alp_discrimination.config import AnalysisConfig, lower_ctau_m
from alp_discrimination.plotting.lifetime import plot_lifetime_scan
from alp_discrimination.physics.lifetimes import geometric_coarse_grid, threshold_brackets
from alp_discrimination.physics.models import MODELS, ModelDefinition
from alp_discrimination.physics.observable_domains import (
    ObservableLifetimeDomain,
    collect_observable_domains,
    domain_table,
    padded_lifetime_grid,
)
from alp_discrimination.paths import portable_path, profile_output_dir
from alp_discrimination.workflows import (
    add_profile_cache_arguments,
    config_and_adapter,
    write_dataframe,
    write_manifest,
)

if TYPE_CHECKING:
    from alp_discrimination.eventcalc.adapter import EventCalcAdapter


MAXIMUM_ALLOWED_RELATIVE_INCREASE = 2.0e-3
STAGE_NAME = "scan_ctau_ranges"


def parse_arguments(arguments: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    return parser.parse_args(arguments)


def fixed_step_log_bisection_midpoint(
    evaluate_rate: Callable[[float], float],
    left_m: float,
    right_m: float,
    *,
    threshold: float,
    steps: int,
    left_passes: bool,
    right_passes: bool,
) -> float:
    """Perform exactly ``steps`` evaluations and return an unevaluated midpoint."""
    if left_m <= 0.0 or right_m <= left_m or threshold <= 0.0 or steps < 1:
        raise ValueError("invalid fixed-step logarithmic-bisection arguments")
    if left_passes == right_passes:
        raise ValueError("the supplied interval does not bracket the event threshold")
    left = float(left_m)
    right = float(right_m)
    for _ in range(steps):
        middle = float(np.sqrt(left * right))
        middle_passes = bool(evaluate_rate(middle) >= threshold)
        if middle_passes == left_passes:
            left = middle
        else:
            right = middle
    return float(np.sqrt(left * right))


def _preselection_events(spectrum) -> float:
    value = getattr(spectrum, "preselection_expected_events", None)
    return float(spectrum.expected_events if value is None else value)


def _scan_row(
    config: AnalysisConfig,
    model: ModelDefinition,
    mass_gev: float,
    ctau_m: float,
    model_seed: int,
    spectrum,
) -> dict:
    source_events = spectrum.source_expected_events
    before_ecal = _preselection_events(spectrum)
    expected_events = float(spectrum.expected_events)
    return {
        "profile": config.name,
        "selection_name": config.selection_name,
        "model": model.legacy_name,
        "model_id": model.identifier,
        "mass_GeV": float(mass_gev),
        "ctau_m": float(ctau_m),
        "ctau_min_m": lower_ctau_m(mass_gev),
        "coupling_squared": float(spectrum.coupling_squared_gev_inv2),
        "N_events": expected_events,
        "N_events_before_ECAL": before_ecal,
        "epsilon_ECAL_weighted": (
            expected_events / before_ecal if before_ecal > 0.0 else 0.0
        ),
        "passes_event_cut": expected_events >= config.lifetimes.event_threshold,
        "N_events_primary": source_events.get("primary", np.nan),
        "N_events_cascade": source_events.get("cascade", np.nan),
        "N_events_inclusive": source_events.get("inclusive", np.nan),
        "cascade_event_fraction": (
            source_events.get("cascade", 0.0) / expected_events
            if expected_events > 0.0
            else np.nan
        ),
        "valid_mother_samples": int(
            getattr(spectrum, "preselection_samples", spectrum.accepted_samples)
        ),
        "samples_passing_ECAL": int(spectrum.accepted_samples),
        "template_model_seed": int(model_seed),
        "spectrum_cache_key": spectrum.cache_key,
    }


def scan_model_mass(
    config: AnalysisConfig,
    adapter: "EventCalcAdapter",
    model: ModelDefinition,
    mass_gev: float,
) -> tuple[list[dict], list[float]]:
    """Evaluate one model/mass, returning saved rows and diagnostic midpoints."""
    settings = config.lifetimes
    model_seed = config.seed_policy.model_seed(mass_gev, model.identifier)
    evaluated: dict[float, dict] = {}

    def evaluate(ctau_m: float) -> dict:
        key = float(ctau_m)
        if key not in evaluated:
            spectrum = adapter.evaluate_model(
                model.identifier,
                mass_gev,
                key,
                model_seed,
                "ctau",
            )
            evaluated[key] = _scan_row(
                config,
                model,
                mass_gev,
                key,
                model_seed,
                spectrum,
            )
        return evaluated[key]

    coarse_grid = geometric_coarse_grid(
        lower_ctau_m(mass_gev),
        settings.maximum_ctau_m,
        settings.coarse_factor,
    )
    coarse_rates = np.asarray(
        [evaluate(ctau_m)["N_events"] for ctau_m in coarse_grid],
        dtype=float,
    )
    relative_increase = np.diff(coarse_rates) / np.maximum(
        coarse_rates[:-1],
        1.0e-300,
    )
    if np.any(relative_increase > MAXIMUM_ALLOWED_RELATIVE_INCREASE):
        offending = np.flatnonzero(
            relative_increase > MAXIMUM_ALLOWED_RELATIVE_INCREASE
        )
        details = ", ".join(
            f"{coarse_grid[index]:.6g}->{coarse_grid[index + 1]:.6g} m"
            for index in offending
        )
        raise RuntimeError(
            "ECAL-accepted event rate is not monotonically decreasing for "
            f"{model.legacy_name}, m_a={mass_gev:g} GeV: {details}"
        )

    coarse_states = coarse_rates >= settings.event_threshold
    midpoints: list[float] = []
    brackets = threshold_brackets(
        coarse_grid,
        coarse_rates,
        settings.event_threshold,
    )
    for left_m, right_m in brackets:
        left_index = int(np.flatnonzero(coarse_grid == left_m)[0])
        before = len(evaluated)
        midpoint = fixed_step_log_bisection_midpoint(
            lambda value: evaluate(value)["N_events"],
            left_m,
            right_m,
            threshold=settings.event_threshold,
            steps=settings.bisection_steps,
            left_passes=bool(coarse_states[left_index]),
            right_passes=bool(coarse_states[left_index + 1]),
        )
        if len(evaluated) - before != settings.bisection_steps:
            raise RuntimeError("bisection did not produce the configured evaluations")
        if midpoint in evaluated:
            raise RuntimeError("diagnostic bisection midpoint was unexpectedly evaluated")
        midpoints.append(midpoint)

    coarse_values = set(coarse_grid.tolist())
    for ctau_m, row in evaluated.items():
        row["scan_point_kind"] = (
            "coarse" if ctau_m in coarse_values else "bisection_evaluation"
        )
    return sorted(evaluated.values(), key=lambda row: row["ctau_m"]), midpoints


def _model_id(model_name: str) -> str:
    for model in MODELS:
        if model.legacy_name == model_name:
            return model.identifier
    raise KeyError(f"unknown legacy model name {model_name!r}")


def build_domain_output(
    config: AnalysisConfig,
    domains: dict[tuple[str, float], ObservableLifetimeDomain],
) -> pd.DataFrame:
    """Add the actual padded template-grid endpoints to the domain table."""
    table = domain_table(
        domains,
        log_padding_fraction=config.templates.log_endpoint_padding_fraction,
    )
    if table.empty:
        return table
    model_ids = []
    grid_lowers = []
    grid_uppers = []
    for row in table.itertuples(index=False):
        domain = domains[(str(row.model), float(row.mass_GeV))]
        grid = padded_lifetime_grid(
            domain,
            config.templates.lifetime_points_per_model,
            config.templates.log_endpoint_padding_fraction,
        )
        model_ids.append(_model_id(str(row.model)))
        grid_lowers.append(float(grid[0]))
        grid_uppers.append(float(grid[-1]))
    table.insert(1, "model_id", model_ids)
    table["template_grid_lower_m"] = grid_lowers
    table["template_grid_upper_m"] = grid_uppers
    table["number_of_lifetime_templates"] = (
        config.templates.lifetime_points_per_model
    )
    table["template_endpoint_convention"] = (
        config.templates.observable_endpoint_convention
    )
    table["diagnostic_endpoint_convention"] = (
        config.lifetimes.diagnostic_endpoint_convention
    )
    return table


def _diagnostic_tables(domains: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    individual = []
    for (model_name, mass_gev), domain in sorted(
        domains.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        individual.append(
            {
                "model": model_name,
                "model_id": _model_id(model_name),
                "mass_GeV": mass_gev,
                "bisection_diagnostic_lower_m": domain.bisection_lower_m,
                "bisection_diagnostic_upper_m": domain.bisection_upper_m,
            }
        )
    common = []
    first_name, second_name = (model.legacy_name for model in MODELS)
    masses = sorted({mass for _, mass in domains})
    for mass_gev in masses:
        first = domains.get((first_name, mass_gev))
        second = domains.get((second_name, mass_gev))
        if first is None or second is None:
            continue
        lower = max(first.bisection_lower_m, second.bisection_lower_m)
        upper = min(first.bisection_upper_m, second.bisection_upper_m)
        if lower < upper:
            common.append(
                {
                    "mass_GeV": mass_gev,
                    "common_bisection_diagnostic_lower_m": lower,
                    "common_bisection_diagnostic_upper_m": upper,
                }
            )
    return pd.DataFrame(individual), pd.DataFrame(common)


def run_scan_ctau_ranges(
    config: AnalysisConfig,
    adapter: "EventCalcAdapter",
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run the reusable scan and atomically write all portable artifacts."""
    started = perf_counter()
    output_dir = output_dir or profile_output_dir(config.name, STAGE_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    returned_midpoints: dict[tuple[str, float], list[float]] = {}
    for model in MODELS:
        for mass_gev in config.masses_gev:
            model_rows, midpoints = scan_model_mass(
                config,
                adapter,
                model,
                float(mass_gev),
            )
            rows.extend(model_rows)
            returned_midpoints[(model.legacy_name, float(mass_gev))] = midpoints
            print(
                f"{model.legacy_name}, m_a={mass_gev:g} GeV: "
                f"{len(model_rows)} saved scan rows, {len(midpoints)} crossing(s)"
            )

    scan = pd.DataFrame(rows).sort_values(
        ["mass_GeV", "model", "ctau_m"],
        ignore_index=True,
    )
    domains = collect_observable_domains(
        scan,
        threshold=config.lifetimes.event_threshold,
        allow_truncated=False,
    )
    for key, midpoints in returned_midpoints.items():
        domain = domains.get(key)
        if domain is not None and len(midpoints) == 1:
            if domain.bisection_upper_m != midpoints[0]:
                raise RuntimeError("saved bracket does not reproduce bisection diagnostic")

    domain_output = build_domain_output(config, domains)
    diagnostic, common_diagnostic = _diagnostic_tables(domains)
    artifacts = {
        "scan": output_dir / "ctau_scan.csv",
        "domains": output_dir / "observable_lifetime_domains.csv",
        "bisection_diagnostics": output_dir / "bisection_diagnostic_ranges.csv",
        "common_bisection_diagnostics": (
            output_dir / "common_bisection_diagnostic_ranges.csv"
        ),
        "plot": output_dir / "ctau_scan_all_masses.png",
    }
    write_dataframe(scan, artifacts["scan"])
    write_dataframe(domain_output, artifacts["domains"])
    write_dataframe(diagnostic, artifacts["bisection_diagnostics"])
    write_dataframe(common_diagnostic, artifacts["common_bisection_diagnostics"])
    plot_lifetime_scan(
        scan,
        event_threshold=config.lifetimes.event_threshold,
        output_path=artifacts["plot"],
    )
    manifest_path = write_manifest(
        config,
        STAGE_NAME,
        output_dir,
        elapsed_seconds=perf_counter() - started,
        cache_stats=adapter.cache.counter_snapshot(),
        artifacts=list(artifacts.values()),
        extra={
            "output_directory": portable_path(output_dir),
            "cache_directory": portable_path(adapter.cache.root),
            "maximum_allowed_relative_rate_increase": (
                MAXIMUM_ALLOWED_RELATIVE_INCREASE
            ),
            "returned_bisection_midpoints_saved_to_scan_table": False,
            "returned_bisection_midpoints_saved_to_diagnostic_table": True,
            "production_template_endpoint_convention": (
                config.templates.observable_endpoint_convention
            ),
            "template_log_endpoint_padding_fraction": (
                config.templates.log_endpoint_padding_fraction
            ),
            "bisection_diagnostic_endpoint_convention": (
                config.lifetimes.diagnostic_endpoint_convention
            ),
        },
    )
    artifacts["manifest"] = manifest_path
    return artifacts


def main() -> None:
    args = parse_arguments()
    config, adapter = config_and_adapter(args)
    artifacts = run_scan_ctau_ranges(config, adapter)
    print(f"Saved {STAGE_NAME} artifacts to {artifacts['scan'].parent}")


if __name__ == "__main__":
    main()
