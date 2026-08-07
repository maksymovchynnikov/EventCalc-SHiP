"""Calibrate the adaptive lifetime-grid planner against an existing dense bank.

This workflow performs no EventCalc generation and no pseudoexperiments.  It
subsamples a previously validated dense template bank, applies the same
interval-aware refinement diagnostics as the adaptive controller, and compares
the resulting minimum distance with the full dense-bank reference.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis2.adaptive_week8 import (
    AdaptiveLifetimeSettings,
    DOMAIN_MODEL_LABELS,
    TRUTH_MODELS,
    initial_adaptive_lifetime_grid,
    lifetime_grid_from_bank,
    propose_lifetime_refinement,
    total_variation_matrix,
)
from analysis2.lifetime_template_banks import LifetimeTemplateBank, load_template_bank
from analysis2.workflows import write_dataframe


_MODEL_PREFIX = {"photon": "photon", "su2": "su2"}
_MODEL_FROM_LABEL = {value: key for key, value in DOMAIN_MODEL_LABELS.items()}


def _domain_table_from_bank(bank: LifetimeTemplateBank) -> pd.DataFrame:
    rows = []
    for model in TRUTH_MODELS:
        intervals = getattr(bank, f"{model}_allowed_intervals_m")
        for interval_index, (lower, upper) in enumerate(intervals):
            rows.append(
                {
                    "model": DOMAIN_MODEL_LABELS[model],
                    "mass_GeV": float(bank.mass_gev),
                    "interval_index": int(interval_index),
                    "ctau_min_m": float(lower),
                    "ctau_max_m": float(upper),
                }
            )
    return pd.DataFrame(rows)


def _nearest_dense_index(
    bank: LifetimeTemplateBank,
    model: str,
    interval_index: int,
    target_ctau_m: float,
    selected: set[int],
) -> int | None:
    ctaus = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
    intervals = np.asarray(
        getattr(bank, f"{model}_interval_index"), dtype=int
    )
    candidates = np.flatnonzero(intervals == int(interval_index))
    candidates = np.asarray(
        [index for index in candidates if int(index) not in selected], dtype=int
    )
    if len(candidates) == 0:
        return None
    distances = np.abs(np.log(ctaus[candidates]) - np.log(float(target_ctau_m)))
    return int(candidates[int(np.argmin(distances))])


def _initial_dense_indices(
    bank: LifetimeTemplateBank,
    settings: AdaptiveLifetimeSettings,
) -> dict[str, np.ndarray]:
    targets = initial_adaptive_lifetime_grid(
        _domain_table_from_bank(bank), bank.mass_gev, settings
    )
    result: dict[str, np.ndarray] = {}
    for model in TRUTH_MODELS:
        selected: set[int] = set()
        rows = targets.loc[targets["model"] == DOMAIN_MODEL_LABELS[model]]
        for row in rows.itertuples(index=False):
            index = _nearest_dense_index(
                bank,
                model,
                int(row.interval_index),
                float(row.ctau_m),
                selected,
            )
            if index is not None:
                selected.add(index)
        dense_intervals = np.asarray(
            getattr(bank, f"{model}_interval_index"), dtype=int
        )
        for interval in np.unique(dense_intervals):
            indices = np.flatnonzero(dense_intervals == interval)
            selected.update({int(indices[0]), int(indices[-1])})
        result[model] = np.asarray(sorted(selected), dtype=int)
    return result


def _slice_bank(
    bank: LifetimeTemplateBank,
    indices: dict[str, np.ndarray],
) -> LifetimeTemplateBank:
    updates = {}
    for model in TRUTH_MODELS:
        chosen = np.asarray(indices[model], dtype=int)
        for suffix in (
            "ctau_m",
            "probabilities",
            "n_events",
            "n_events_before_ecal",
            "epsilon_ecal_weighted",
            "total_n_eff",
            "interval_index",
        ):
            values = np.asarray(getattr(bank, f"{model}_{suffix}"))
            updates[f"{model}_{suffix}"] = values[chosen]
    return replace(bank, **updates)


def _map_additions_to_dense(
    dense_bank: LifetimeTemplateBank,
    additions: pd.DataFrame,
    selected: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {
        model: set(int(index) for index in selected[model])
        for model in TRUTH_MODELS
    }
    for row in additions.sort_values(
        ["model", "interval_index", "ctau_m"], ignore_index=True
    ).itertuples(index=False):
        model = _MODEL_FROM_LABEL[str(row.model)]
        index = _nearest_dense_index(
            dense_bank,
            model,
            int(row.interval_index),
            float(row.ctau_m),
            result[model],
        )
        if index is not None:
            result[model].add(index)
    return {
        model: np.asarray(sorted(result[model]), dtype=int)
        for model in TRUTH_MODELS
    }


def calibrate_dense_bank(
    bank: LifetimeTemplateBank,
    settings: AdaptiveLifetimeSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dense_distances = total_variation_matrix(bank)
    dense_min_flat = int(np.argmin(dense_distances))
    dense_min_indices = np.unravel_index(dense_min_flat, dense_distances.shape)
    dense_minimum = float(dense_distances[dense_min_indices])

    selected = _initial_dense_indices(bank, settings)
    previous_minimum = None
    round_rows = []

    for round_index in range(settings.maximum_rounds):
        subbank = _slice_bank(bank, selected)
        distances = total_variation_matrix(subbank)
        decision = propose_lifetime_refinement(
            subbank,
            distances,
            lifetime_grid_from_bank(
                subbank,
                adaptive_round_added=round_index,
                reason="dense_bank_calibration",
            ),
            settings,
            round_index=round_index,
            previous_minimum_distance=previous_minimum,
        )
        proposed = len(decision.additions)
        updated = _map_additions_to_dense(bank, decision.additions, selected)
        added = sum(
            len(np.setdiff1d(updated[model], selected[model]))
            for model in TRUTH_MODELS
        )
        sub_minimum = float(distances.min())
        round_rows.append(
            {
                "round": round_index,
                "number_of_photon_lifetimes": len(selected["photon"]),
                "number_of_su2_lifetimes": len(selected["su2"]),
                "minimum_D_TV": sub_minimum,
                "relative_error_vs_dense_minimum": abs(
                    sub_minimum - dense_minimum
                )
                / max(abs(dense_minimum), np.finfo(float).eps),
                "proposed_midpoints": proposed,
                "new_dense_indices": added,
                "planner_converged": bool(decision.converged),
                "reached_size_limit": bool(decision.reached_size_limit),
            }
        )
        if decision.converged or added == 0:
            break
        selected = updated
        previous_minimum = decision.minimum_distance

    final_bank = _slice_bank(bank, selected)
    final_distances = total_variation_matrix(final_bank)
    final_min_flat = int(np.argmin(final_distances))
    final_min_indices = np.unravel_index(final_min_flat, final_distances.shape)

    summary = pd.DataFrame(
        [
            {
                "mass_GeV": float(bank.mass_gev),
                "selection_name": bank.selection_name,
                "dense_photon_lifetimes": len(bank.photon_ctau_m),
                "adaptive_photon_lifetimes": len(final_bank.photon_ctau_m),
                "dense_su2_lifetimes": len(bank.su2_ctau_m),
                "adaptive_su2_lifetimes": len(final_bank.su2_ctau_m),
                "dense_minimum_D_TV": dense_minimum,
                "adaptive_minimum_D_TV": float(final_distances[final_min_indices]),
                "relative_minimum_D_TV_error": abs(
                    float(final_distances[final_min_indices]) - dense_minimum
                )
                / max(abs(dense_minimum), np.finfo(float).eps),
                "dense_minimum_photon_ctau_m": float(
                    bank.photon_ctau_m[dense_min_indices[0]]
                ),
                "adaptive_minimum_photon_ctau_m": float(
                    final_bank.photon_ctau_m[final_min_indices[0]]
                ),
                "dense_minimum_su2_ctau_m": float(
                    bank.su2_ctau_m[dense_min_indices[1]]
                ),
                "adaptive_minimum_su2_ctau_m": float(
                    final_bank.su2_ctau_m[final_min_indices[1]]
                ),
                "dense_minimum_photon_interval_index": int(
                    bank.photon_interval_index[dense_min_indices[0]]
                ),
                "adaptive_minimum_photon_interval_index": int(
                    final_bank.photon_interval_index[final_min_indices[0]]
                ),
                "dense_minimum_su2_interval_index": int(
                    bank.su2_interval_index[dense_min_indices[1]]
                ),
                "adaptive_minimum_su2_interval_index": int(
                    final_bank.su2_interval_index[final_min_indices[1]]
                ),
                "rounds_used": len(round_rows),
            }
        ]
    )
    return summary, pd.DataFrame(round_rows)


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--banks", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lifetime-points-per-decade", type=float, default=4.0)
    parser.add_argument("--minimum-points-per-interval", type=int, default=5)
    parser.add_argument("--maximum-log-gap", type=float, default=0.25)
    parser.add_argument("--maximum-rounds", type=int, default=8)
    parser.add_argument("--maximum-lifetimes-per-model", type=int, default=120)
    parser.add_argument(
        "--maximum-new-lifetimes-per-model-per-round",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--maximum-soft-priority-at-convergence",
        type=float,
        default=6.0,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    settings = AdaptiveLifetimeSettings(
        initial_points_per_decade=args.lifetime_points_per_decade,
        minimum_points_per_interval=args.minimum_points_per_interval,
        maximum_log_gap_decades=args.maximum_log_gap,
        maximum_rounds=args.maximum_rounds,
        maximum_total_lifetimes_per_model=args.maximum_lifetimes_per_model,
        maximum_new_points_per_model_per_round=(
            args.maximum_new_lifetimes_per_model_per_round
        ),
        maximum_soft_priority_at_convergence=(
            args.maximum_soft_priority_at_convergence
        ),
    )
    summaries = []
    rounds = []
    for bank_path in args.banks:
        bank = load_template_bank(bank_path)
        summary, history = calibrate_dense_bank(bank, settings)
        summary.insert(0, "bank_path", str(bank_path))
        history.insert(0, "bank_path", str(bank_path))
        summaries.append(summary)
        rounds.append(history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_table = pd.concat(summaries, ignore_index=True)
    history_table = pd.concat(rounds, ignore_index=True)
    write_dataframe(summary_table, args.output_dir / "adaptive_dense_bank_summary.csv")
    write_dataframe(history_table, args.output_dir / "adaptive_dense_bank_rounds.csv")
    print(summary_table.to_string(index=False))
    print()
    print(history_table.to_string(index=False))


if __name__ == "__main__":
    main()
