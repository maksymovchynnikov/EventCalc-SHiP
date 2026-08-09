"""Conservative reductions of lifetime-profiled pseudoexperiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


SEED_WORST_CASE_COLUMNS = (
    "mass_GeV",
    "seed",
    "number_of_events",
    "photon_truth_worst_accuracy",
    "photon_limiting_lifetime_index",
    "photon_limiting_ctau_m",
    "su2_truth_worst_accuracy",
    "su2_limiting_lifetime_index",
    "su2_limiting_ctau_m",
    "worst_case_correct_fraction",
    "limiting_truth_model",
    "limiting_truth_lifetime_index",
    "limiting_truth_ctau_m",
)

CONSERVATIVE_ENVELOPE_COLUMNS = (
    "mass_GeV",
    "number_of_events",
    "photon_truth_worst_accuracy",
    "su2_truth_worst_accuracy",
    "worst_case_correct_fraction",
    "limiting_seed",
    "limiting_truth_model",
    "limiting_truth_lifetime_index",
    "limiting_truth_ctau_m",
)


def _limiting_row(group: pd.DataFrame) -> pd.Series:
    """Select a deterministic minimum, breaking ties by lifetime index."""
    return group.sort_values(
        ["correct_fraction", "truth_lifetime_index"],
        kind="mergesort",
    ).iloc[0]


def build_seed_worst_case_table(detailed: pd.DataFrame) -> pd.DataFrame:
    """Reduce truth lifetimes and truth models independently for each seed."""
    rows: list[dict] = []
    for (mass_gev, seed, number_of_events), group in detailed.groupby(
        ["mass_GeV", "seed", "number_of_events"],
        sort=True,
    ):
        photon_group = group.loc[group["truth_model"] == "photon"]
        su2_group = group.loc[group["truth_model"] == "su2"]
        if photon_group.empty or su2_group.empty:
            raise RuntimeError("Both truth models are required for aggregation.")

        photon_row = _limiting_row(photon_group)
        su2_row = _limiting_row(su2_group)
        global_row = _limiting_row(group)
        rows.append(
            {
                "mass_GeV": float(mass_gev),
                "seed": int(seed),
                "number_of_events": int(number_of_events),
                "photon_truth_worst_accuracy": float(
                    photon_row["correct_fraction"]
                ),
                "photon_limiting_lifetime_index": int(
                    photon_row["truth_lifetime_index"]
                ),
                "photon_limiting_ctau_m": float(photon_row["truth_ctau_m"]),
                "su2_truth_worst_accuracy": float(su2_row["correct_fraction"]),
                "su2_limiting_lifetime_index": int(
                    su2_row["truth_lifetime_index"]
                ),
                "su2_limiting_ctau_m": float(su2_row["truth_ctau_m"]),
                "worst_case_correct_fraction": float(
                    global_row["correct_fraction"]
                ),
                "limiting_truth_model": str(global_row["truth_model"]),
                "limiting_truth_lifetime_index": int(
                    global_row["truth_lifetime_index"]
                ),
                "limiting_truth_ctau_m": float(global_row["truth_ctau_m"]),
            }
        )
    table = pd.DataFrame(rows, columns=SEED_WORST_CASE_COLUMNS)
    return table.sort_values(
        ["mass_GeV", "seed", "number_of_events"],
        ignore_index=True,
    )


def build_conservative_seed_envelope(seed_table: pd.DataFrame) -> pd.DataFrame:
    """Take the minimum accuracy over all validation seeds at every count."""
    rows: list[dict] = []
    for (mass_gev, number_of_events), group in seed_table.groupby(
        ["mass_GeV", "number_of_events"],
        sort=True,
    ):
        limiting = group.sort_values(
            ["worst_case_correct_fraction", "seed"],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "mass_GeV": float(mass_gev),
                "number_of_events": int(number_of_events),
                "photon_truth_worst_accuracy": float(
                    group["photon_truth_worst_accuracy"].min()
                ),
                "su2_truth_worst_accuracy": float(
                    group["su2_truth_worst_accuracy"].min()
                ),
                "worst_case_correct_fraction": float(
                    group["worst_case_correct_fraction"].min()
                ),
                "limiting_seed": int(limiting["seed"]),
                "limiting_truth_model": str(limiting["limiting_truth_model"]),
                "limiting_truth_lifetime_index": int(
                    limiting["limiting_truth_lifetime_index"]
                ),
                "limiting_truth_ctau_m": float(
                    limiting["limiting_truth_ctau_m"]
                ),
            }
        )
    table = pd.DataFrame(rows, columns=CONSERVATIVE_ENVELOPE_COLUMNS)
    return table.sort_values(
        ["mass_GeV", "number_of_events"],
        ignore_index=True,
    )


def minimum_persistent_events(
    curve: pd.DataFrame,
    *,
    accuracy_column: str,
    target_accuracy: float,
) -> int | None:
    """First tested count from which every larger tested count passes."""
    ordered = curve.sort_values("number_of_events")
    values = ordered[accuracy_column].to_numpy(dtype=float)
    event_counts = ordered["number_of_events"].to_numpy(dtype=int)
    passing = values >= target_accuracy
    persistent = np.logical_and.accumulate(passing[::-1])[::-1]
    indices = np.flatnonzero(persistent)
    return None if len(indices) == 0 else int(event_counts[indices[0]])
