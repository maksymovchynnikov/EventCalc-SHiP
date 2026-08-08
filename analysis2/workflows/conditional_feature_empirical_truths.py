#!/usr/bin/env python3
"""Prepare a targeted difficult-truth set for empirical feature validation.

The input is a completed selected-5k audit directory. The script selects the
most difficult truth lifetimes around the persistent N90 crossing and expands
them by same-interval lifetime neighbours. It does not run pseudoexperiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-gap", type=float, default=0.01)
    parser.add_argument("--top-per-group", type=int, default=2)
    parser.add_argument("--global-top-per-count", type=int, default=10)
    parser.add_argument("--neighbour-radius", type=int, default=1)
    return parser.parse_args()


def unique_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def main() -> None:
    args = parse_args()
    selected_dir = args.selected_dir.expanduser().resolve()
    bank_path = args.bank_path.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if args.hard_gap < 0.0:
        raise ValueError("--hard-gap must be non-negative")
    if args.top_per_group <= 0 or args.global_top_per_count <= 0:
        raise ValueError("Top-count arguments must be positive")
    if args.neighbour_radius < 0:
        raise ValueError("--neighbour-radius must be non-negative")

    summary_path = unique_file(selected_dir, "selected_5k_audit_summary_ma_*.json")
    detailed_path = unique_file(selected_dir, "selected_5k_detailed_accuracy_ma_*.csv")
    limiting_path = unique_file(selected_dir, "selected_5k_limiting_rows_ma_*.csv")

    summary = json.loads(summary_path.read_text())
    threshold = summary.get("persistent_thresholds", {}).get("selected_5k")
    if threshold is None:
        raise RuntimeError("The selected-5k summary has no persistent threshold")
    threshold = int(threshold)

    detailed = pd.read_csv(detailed_path)
    limiting = pd.read_csv(limiting_path)
    required = {
        "truth_model",
        "truth_lifetime_index",
        "truth_ctau_m",
        "seed",
        "number_of_events",
        "correct_fraction",
    }
    missing = required - set(detailed.columns)
    if missing:
        raise ValueError(f"Detailed table is missing columns: {sorted(missing)}")

    available_counts = set(detailed["number_of_events"].astype(int))
    decision_counts = sorted(
        count for count in {threshold - 1, threshold, threshold + 1}
        if count in available_counts and count > 0
    )
    decision = detailed[
        detailed["number_of_events"].astype(int).isin(decision_counts)
    ].copy()
    if decision.empty:
        raise RuntimeError("No rows exist around the selected-5k threshold")

    reasons: dict[tuple[str, int], set[str]] = {}

    def add_rows(frame: pd.DataFrame, reason: str) -> None:
        for row in frame.itertuples(index=False):
            key = (str(row.truth_model), int(row.truth_lifetime_index))
            reasons.setdefault(key, set()).add(reason)

    for count, frame in decision.groupby("number_of_events"):
        minimum = float(frame["correct_fraction"].min())
        add_rows(
            frame[frame["correct_fraction"] <= minimum + float(args.hard_gap)],
            f"within_{args.hard_gap:g}_of_N{int(count)}_minimum",
        )
        add_rows(
            frame.nsmallest(int(args.global_top_per_count), "correct_fraction"),
            f"global_top_N{int(count)}",
        )

    grouped_rows = []
    for _, frame in decision.groupby(
        ["number_of_events", "seed", "truth_model"],
        sort=True,
    ):
        grouped_rows.append(
            frame.nsmallest(int(args.top_per_group), "correct_fraction")
        )
    if grouped_rows:
        add_rows(
            pd.concat(grouped_rows, ignore_index=True),
            "per_seed_model_limiter",
        )

    add_rows(limiting, "selected_5k_limiting_table")

    limiter = summary.get("limiting_truth_at_selected_5k_threshold", {})
    if limiter:
        key = (
            str(limiter["truth_model"]),
            int(limiter["truth_lifetime_index"]),
        )
        reasons.setdefault(key, set()).add("summary_threshold_limiter")

    with np.load(bank_path, allow_pickle=False) as bank:
        mass = float(np.asarray(bank["mass_GeV"]).item())
        selection = scalar_text(bank["selection_name"])
        model_arrays = {
            model: {
                "ctau": np.asarray(bank[f"{model}_ctau_m"], dtype=float),
                "interval": np.asarray(
                    bank[f"{model}_interval_index"], dtype=int
                ),
            }
            for model in ("photon", "su2")
        }

    original_keys = list(reasons)
    for model, index in original_keys:
        arrays = model_arrays[model]
        intervals = arrays["interval"]
        if index < 0 or index >= len(intervals):
            raise IndexError(f"Invalid {model} truth index {index}")
        interval = int(intervals[index])
        for neighbour in range(
            max(0, index - int(args.neighbour_radius)),
            min(len(intervals), index + int(args.neighbour_radius) + 1),
        ):
            if int(intervals[neighbour]) != interval:
                continue
            key = (model, int(neighbour))
            reasons.setdefault(key, set()).add(f"same_interval_neighbour_of_{index}")

    rows = []
    for model, index in sorted(reasons, key=lambda item: (item[0], item[1])):
        arrays = model_arrays[model]
        rows.append(
            {
                "mass_GeV": mass,
                "selection_name": selection,
                "truth_model": model,
                "truth_lifetime_index": int(index),
                "truth_ctau_m": float(arrays["ctau"][index]),
                "truth_interval_index": int(arrays["interval"][index]),
                "selection_reason": ";".join(sorted(reasons[(model, index)])),
                "selected_5k_threshold": threshold,
            }
        )

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = pd.read_csv(output)
        compare_columns = ["truth_model", "truth_lifetime_index"]
        if not existing[compare_columns].equals(result[compare_columns]):
            raise FileExistsError(
                f"Refusing to overwrite a different truth set: {output}"
            )
    else:
        result.to_csv(output, index=False)

    counts = result.groupby("truth_model").size().to_dict()
    print(
        json.dumps(
            {
                "mass_GeV": mass,
                "selection_name": selection,
                "selected_5k_threshold": threshold,
                "decision_event_counts": decision_counts,
                "number_of_empirical_truths": {
                    "photon": int(counts.get("photon", 0)),
                    "su2": int(counts.get("su2", 0)),
                    "total": int(len(result)),
                },
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
