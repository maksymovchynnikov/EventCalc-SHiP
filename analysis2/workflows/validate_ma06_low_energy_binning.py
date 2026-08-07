"""Validate the m_a=0.6 GeV profiled result against legacy bin coarsening.

This workflow is intentionally narrow.  It reads the accepted production
template bank, never calls EventCalc, and repeats the frozen profiled
pseudoexperiments after exactly the rebinning used by the legacy validation:
merge consecutive final adaptive bins in groups of 1, 2, or 4, with any
remainder retained in the last bin.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from analysis2.cache import CacheStore
from analysis2.config import AnalysisConfig, PRODUCTION
from analysis2.lifetime_template_banks import LifetimeTemplateBank, load_template_bank
from analysis2.paths import OUTPUT_ROOT, portable_path
from analysis2.profiled_reduction import (
    build_conservative_seed_envelope,
    build_seed_worst_case_table,
)
from analysis2.workflows import write_dataframe, write_manifest
from analysis2.workflows.lifetime_blind_profiled_likelihood import (
    profiled_run_axes,
    summarize_mass_threshold,
)
from analysis2.workflows.profiled_likelihood_cache import cached_profiled_seed


MASS_GEV = 0.6
REBIN_FACTORS = (1, 2, 4)
LOW_EDGE_COUNT = 5
DEFAULT_BANK_PATH = (
    OUTPUT_ROOT
    / "production"
    / "lifetime_blind_discrimination"
    / "template_banks"
    / "template_bank_ma_0p6.npz"
)
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "validation" / "ma06_low_energy_binning"
PRODUCTION_OUTPUT_ROOT = OUTPUT_ROOT / "production"


@dataclass(frozen=True)
class BinningValidationResult:
    summary: pd.DataFrame
    summary_path: Path
    manifest_path: Path
    elapsed_seconds: float
    cache_stats: dict[str, int]
    cache_stats_by_rebin_factor: dict[int, dict[str, int]]


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank-path",
        type=Path,
        default=DEFAULT_BANK_PATH,
        help="Accepted production m_a=0.6 template bank.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Isolated validation output directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional production-profile cache root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the isolated validation directory.",
    )
    return parser.parse_args(argv)


def rebin_template_bank(
    bank: LifetimeTemplateBank,
    factor: int,
) -> LifetimeTemplateBank:
    """Apply the legacy exact coarse graining to a typed template bank."""
    if factor < 1:
        raise ValueError("Rebin factor must be a positive integer.")

    number_of_bins = bank.number_of_energy_bins
    starts = np.arange(0, number_of_bins, factor, dtype=int)
    edges = np.concatenate((bank.energy_edges_gev[starts], bank.energy_edges_gev[-1:]))
    photon = np.add.reduceat(bank.photon_probabilities, starts, axis=1)
    su2 = np.add.reduceat(bank.su2_probabilities, starts, axis=1)

    # This mirrors the legacy convention and only removes summation round-off.
    photon /= photon.sum(axis=1, keepdims=True)
    su2 /= su2.sum(axis=1, keepdims=True)
    return replace(
        bank,
        energy_edges_gev=edges,
        photon_probabilities=photon,
        su2_probabilities=su2,
    )


def _ensure_isolated_output(output_dir: Path, production_output_root: Path) -> None:
    destination = output_dir.resolve()
    production = production_output_root.resolve()
    if destination == production or production in destination.parents:
        raise ValueError(
            "Low-energy-binning validation outputs cannot be written under "
            f"the production output tree: {production}"
        )


def _validate_inputs(config: AnalysisConfig, bank: LifetimeTemplateBank) -> None:
    if config != PRODUCTION:
        raise ValueError("The m_a=0.6 validation requires the frozen production profile.")
    if not np.isclose(bank.mass_gev, MASS_GEV, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"This validation processes only m_a={MASS_GEV:g} GeV; "
            f"the input bank contains {bank.mass_gev:g} GeV."
        )
    if bank.profile != config.name:
        raise ValueError("The input template bank is not a production-profile bank.")
    if bank.selection_name != config.selection_name:
        raise ValueError("The input bank does not use the frozen ECAL selection.")
    if not np.isclose(
        bank.jeffreys_alpha,
        config.templates.jeffreys_alpha,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("The input bank does not use the frozen Jeffreys smoothing.")
    if not np.isclose(
        bank.event_threshold,
        config.lifetimes.event_threshold,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("The input bank does not use the frozen lifetime domain cut.")


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {name: int(after[name] - before.get(name, 0)) for name in after}


def _base_summary_row(
    *,
    bank: LifetimeTemplateBank,
    factor: int,
    conservative: pd.DataFrame,
    config: AnalysisConfig,
) -> dict:
    factor_config = replace(
        config,
        profiled_likelihood=replace(config.profiled_likelihood, rebin_factor=factor),
    )
    threshold = summarize_mass_threshold(bank, conservative, factor_config)
    required = int(threshold["minimum_persistent_events"])
    limiting_event_count = (
        required
        if bool(threshold["threshold_reached"])
        else int(threshold["maximum_tested_events"])
    )
    limiting = conservative.loc[
        conservative["number_of_events"] == limiting_event_count
    ].iloc[0]
    row = {
        "mass_GeV": bank.mass_gev,
        "rebin_factor": factor,
        "binning": "nominal" if factor == 1 else f"rebin_{factor}",
        "number_of_final_adaptive_energy_bins": bank.number_of_energy_bins,
        "lowest_five_energy_bin_edges_GeV": ";".join(
            f"{edge:.12g}" for edge in bank.energy_edges_gev[:LOW_EDGE_COUNT]
        ),
        "persistent_90_required_event_count": required,
        "threshold_reached": bool(threshold["threshold_reached"]),
        "limiting_event_count": limiting_event_count,
        "limiting_truth_model": str(limiting["limiting_truth_model"]),
        "limiting_truth_lifetime_index": int(
            limiting["limiting_truth_lifetime_index"]
        ),
        "limiting_truth_ctau_m": float(limiting["limiting_truth_ctau_m"]),
    }
    for record in conservative.itertuples(index=False):
        row[f"worst_case_accuracy_N{int(record.number_of_events)}"] = float(
            record.worst_case_correct_fraction
        )
    return row


def _add_nominal_comparisons(
    rows: list[dict],
    event_counts: np.ndarray,
) -> pd.DataFrame:
    nominal = rows[0]
    nominal_required = int(nominal["persistent_90_required_event_count"])
    thresholds_stable = all(
        int(row["persistent_90_required_event_count"]) == nominal_required
        for row in rows
    )
    for row in rows:
        difference = int(row["persistent_90_required_event_count"]) - nominal_required
        row["required_event_difference_from_nominal"] = difference
        row["required_event_threshold_stable"] = difference == 0
        row["threshold_change_report"] = (
            "stable"
            if difference == 0
            else f"changed by {difference:+d} tested event(s)"
        )
        row["overall_nominal_conclusion_robust"] = thresholds_stable
        for event_count in event_counts:
            column = f"worst_case_accuracy_N{int(event_count)}"
            row[f"accuracy_difference_from_nominal_N{int(event_count)}"] = (
                float(row[column]) - float(nominal[column])
            )
    return pd.DataFrame(rows).sort_values("rebin_factor", ignore_index=True)


def run_validation(
    *,
    config: AnalysisConfig = PRODUCTION,
    bank_path: Path = DEFAULT_BANK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache: CacheStore | None = None,
    overwrite: bool = False,
    production_output_root: Path = PRODUCTION_OUTPUT_ROOT,
) -> BinningValidationResult:
    """Run only the production m_a=0.6 bin-coarsening comparison."""
    started = perf_counter()
    _ensure_isolated_output(output_dir, production_output_root)
    summary_path = output_dir / "ma06_low_energy_binning_summary.csv"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (summary_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Validation output already exists; pass --overwrite to replace only "
            f"the isolated validation artifacts: {', '.join(map(str, existing))}"
        )
    if not bank_path.is_file():
        raise FileNotFoundError(
            "The accepted m_a=0.6 production template bank is missing; refusing "
            f"to regenerate EventCalc samples: {bank_path}"
        )

    bank = load_template_bank(bank_path)
    _validate_inputs(config, bank)
    event_counts, seeds = profiled_run_axes(config)
    cache = cache or CacheStore(config.name)
    if cache.profile != config.name:
        raise ValueError("Cache and analysis profiles differ.")

    rows: list[dict] = []
    per_factor_cache: dict[int, dict[str, int]] = {}
    initial_counters = cache.counter_snapshot()
    for factor in REBIN_FACTORS:
        rebinned = rebin_template_bank(bank, factor)
        factor_config = replace(
            config,
            profiled_likelihood=replace(
                config.profiled_likelihood,
                rebin_factor=factor,
            ),
        )
        before = cache.counter_snapshot()
        detailed = pd.concat(
            [
                cached_profiled_seed(
                    rebinned,
                    bank_path,
                    factor_config,
                    seed,
                    event_counts,
                    cache,
                    force=False,
                )
                for seed in seeds
            ],
            ignore_index=True,
        )
        after = cache.counter_snapshot()
        per_factor_cache[factor] = _counter_delta(after, before)
        conservative = build_conservative_seed_envelope(
            build_seed_worst_case_table(detailed)
        )
        rows.append(
            _base_summary_row(
                bank=rebinned,
                factor=factor,
                conservative=conservative,
                config=config,
            )
        )

    summary = _add_nominal_comparisons(rows, event_counts)
    write_dataframe(summary, summary_path)
    elapsed = perf_counter() - started
    cache_stats = _counter_delta(cache.counter_snapshot(), initial_counters)
    manifest_path = write_manifest(
        config,
        "validate_ma06_low_energy_binning",
        output_dir,
        elapsed_seconds=elapsed,
        cache_stats=cache_stats,
        artifacts=[summary_path],
        extra={
            "mass_GeV": MASS_GEV,
            "input_bank": portable_path(bank_path),
            "production_template_bank_reused": True,
            "eventcalc_invocations": 0,
            "rebin_factors": list(REBIN_FACTORS),
            "rebin_convention": (
                "merge consecutive saved adaptive bins; retain final remainder bin"
            ),
            "event_counts": event_counts.tolist(),
            "pseudoexperiment_seeds": list(seeds),
            "cache_stats_by_rebin_factor": {
                str(factor): stats for factor, stats in per_factor_cache.items()
            },
        },
    )
    return BinningValidationResult(
        summary=summary,
        summary_path=summary_path,
        manifest_path=manifest_path,
        elapsed_seconds=elapsed,
        cache_stats=cache_stats,
        cache_stats_by_rebin_factor=per_factor_cache,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    result = run_validation(
        bank_path=args.bank_path,
        output_dir=args.output_dir,
        cache=CacheStore("production", root=args.cache_dir),
        overwrite=args.overwrite,
    )
    columns = [
        "rebin_factor",
        "number_of_final_adaptive_energy_bins",
        "persistent_90_required_event_count",
        "required_event_difference_from_nominal",
        "limiting_truth_model",
        "limiting_truth_ctau_m",
        "overall_nominal_conclusion_robust",
    ]
    print(result.summary.loc[:, columns].to_string(index=False))
    print(f"Saved validation outputs to {result.manifest_path.parent}")
    print(f"Runtime: {result.elapsed_seconds:.3f} s; cache: {result.cache_stats}")
    print(f"Cache by rebin factor: {result.cache_stats_by_rebin_factor}")
    print("EventCalc invocations: 0 (accepted production bank reused)")


if __name__ == "__main__":
    main()
