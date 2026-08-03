"""High-statistics validation using saved same-lifetime probability templates."""

from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from analysis2.config import PROFILES, get_config, validation_seed
from analysis2.paths import profile_output_dir
from analysis2.plotting import plot_validated_thresholds
from analysis2.statistics import (
    finite_threshold_summary, minimum_persistent_events, simulate_shape_discrimination,
)
from analysis2.templates import SavedTemplatePair, load_saved_template_pair
from analysis2.workflows import (
    require_columns, write_dataframe, write_manifest,
)


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    return parser.parse_args()


def selected_templates(grid_dir) -> list[SavedTemplatePair]:
    point_path = grid_dir / "selected_lifetime_points.csv"
    points = pd.read_csv(point_path).sort_values(["mass_GeV", "log_interval_fraction"])
    require_columns(points, {"mass_GeV", "ctau_m", "lifetime_label", "log_interval_fraction"}, point_path)
    available = [
        load_saved_template_pair(path)
        for path in sorted((grid_dir / "templates").glob("probability_templates_*.csv"))
    ]
    selected = []
    for point in points.itertuples(index=False):
        matches = [item for item in available if item.lifetime_label == point.lifetime_label and np.isclose(
            item.mass_gev, point.mass_GeV, rtol=0.0, atol=1e-12
        ) and np.isclose(item.ctau_m, point.ctau_m, rtol=1e-12, atol=1e-12)]
        if len(matches) != 1:
            raise ValueError(
                f"expected one saved template for m={point.mass_GeV:g}, "
                f"{point.lifetime_label}; found {len(matches)}"
            )
        selected.append(matches[0])
    if len(available) > len(selected):
        print(f"Ignoring {len(available) - len(selected)} stale template file(s)")
    return selected


def threshold_column(target: float) -> str:
    return f"minimum_events_for_{100 * target:.0f}pct_worst_case_accuracy"


def summarize_points(thresholds: pd.DataFrame, maximum_spread: int) -> pd.DataFrame:
    rows = []
    group_columns = ["mass_GeV", "ctau_m", "lifetime_label", "number_of_bins", "target_accuracy"]
    for keys, group in thresholds.groupby(group_columns, sort=True):
        photon = finite_threshold_summary(group["photon_threshold"].to_numpy(float))
        su2 = finite_threshold_summary(group["su2_threshold"].to_numpy(float))
        worst = finite_threshold_summary(group["worst_case_threshold"].to_numpy(float))
        row = dict(zip(group_columns, keys))
        row.update({
            "number_of_validation_seeds": len(group),
            "photon_all_seeds_reached": photon["all_reached"],
            "photon_threshold_min": photon["minimum"], "photon_threshold_median": photon["median"],
            "photon_threshold_max": photon["maximum"],
            "su2_all_seeds_reached": su2["all_reached"], "su2_threshold_min": su2["minimum"],
            "su2_threshold_median": su2["median"], "su2_threshold_max": su2["maximum"],
            "worst_case_all_seeds_reached": worst["all_reached"],
            "worst_case_threshold_min": worst["minimum"],
            "worst_case_threshold_median": worst["median"],
            "worst_case_threshold_max": worst["maximum"], "worst_case_threshold_spread": worst["spread"],
            "threshold_stable": bool(worst["all_reached"] and worst["spread"] <= maximum_spread),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mass_GeV", "target_accuracy", "ctau_m"])


def summarize_masses(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mass_gev, target), group in points.groupby(["mass_GeV", "target_accuracy"], sort=True):
        minima = group["worst_case_threshold_min"].to_numpy(float)
        medians = group["worst_case_threshold_median"].to_numpy(float)
        maxima = group["worst_case_threshold_max"].to_numpy(float)
        finite_min, finite_med, finite_max = (
            values[np.isfinite(values)] for values in (minima, medians, maxima)
        )
        conservative = float(finite_max.max()) if len(finite_max) else np.nan
        limiting = group[np.isclose(group["worst_case_threshold_max"], conservative)].sort_values("ctau_m")
        rows.append({
            "mass_GeV": mass_gev, "target_accuracy": target,
            "target_accuracy_percent": 100 * target, "number_of_lifetimes": len(group),
            "all_lifetimes_reached": len(finite_max) == len(group),
            "minimum_events_over_lifetimes_and_seeds": float(finite_min.min()) if len(finite_min) else np.nan,
            "median_events_over_lifetimes": float(np.median(finite_med)) if len(finite_med) else np.nan,
            "conservative_required_events": conservative,
            "limiting_lifetime_label": "" if limiting.empty else limiting.iloc[0]["lifetime_label"],
            "limiting_ctau_m": np.nan if limiting.empty else limiting.iloc[0]["ctau_m"],
            "all_seed_thresholds_stable": bool(group["threshold_stable"].all()),
        })
    return pd.DataFrame(rows).sort_values(["target_accuracy", "mass_GeV"])


def main() -> None:
    args = parse_arguments()
    config = get_config(args.profile)
    grid_dir = profile_output_dir(config.name, "same_lifetime_discrimination")
    templates = selected_templates(grid_dir)
    if not templates:
        raise FileNotFoundError(f"No probability templates found in {grid_dir / 'templates'}")
    accuracy_frames, threshold_rows = [], []
    for point_index, template in enumerate(templates):
        for seed_index in range(config.discrimination.validation_seeds):
            seed = validation_seed(point_index, seed_index)
            simulation = simulate_shape_discrimination(
                template.photon, template.su2,
                config.discrimination.validation_maximum_events,
                config.discrimination.validation_pseudoexperiments, seed,
            )
            accuracy = pd.DataFrame(simulation.records())
            for column, value in reversed((
                ("mass_GeV", template.mass_gev), ("ctau_m", template.ctau_m),
                ("lifetime_label", template.lifetime_label), ("validation_seed", seed),
            )):
                accuracy.insert(0, column, value)
            accuracy_frames.append(accuracy)
            for target in config.discrimination.target_accuracies:
                thresholds = [
                    minimum_persistent_events(simulation.number_of_events, values, target)
                    for values in (
                        simulation.photon_correct_fraction, simulation.su2_correct_fraction,
                        simulation.worst_case_correct_fraction,
                    )
                ]
                if thresholds[0] is not None and thresholds[1] is not None:
                    if thresholds[2] != max(thresholds[:2]):
                        raise RuntimeError("worst-case validation threshold is inconsistent")
                threshold_rows.append({
                    "mass_GeV": template.mass_gev, "ctau_m": template.ctau_m,
                    "lifetime_label": template.lifetime_label,
                    "number_of_bins": template.number_of_bins, "validation_seed": seed,
                    "target_accuracy": target, "photon_threshold": thresholds[0],
                    "su2_threshold": thresholds[1], "worst_case_threshold": thresholds[2],
                })
    output_dir = profile_output_dir(config.name, "validation")
    combined = pd.concat(accuracy_frames, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    points = summarize_points(thresholds, config.discrimination.maximum_threshold_spread)
    grid_summary = pd.read_csv(grid_dir / "discrimination_grid_summary.csv")
    points["original_grid_threshold"] = np.nan
    points["original_threshold_within_validation_range"] = False
    for index, row in points.iterrows():
        match = grid_summary[
            np.isclose(grid_summary["mass_GeV"], row.mass_GeV)
            & np.isclose(grid_summary["ctau_m"], row.ctau_m, rtol=1e-10, atol=1e-10)
            & (grid_summary["lifetime_label"] == row.lifetime_label)
        ]
        original = float(match.iloc[0][threshold_column(row.target_accuracy)])
        points.at[index, "original_grid_threshold"] = original
        points.at[index, "original_threshold_within_validation_range"] = (
            row.worst_case_threshold_min <= original <= row.worst_case_threshold_max
        )
    masses = summarize_masses(points)
    write_dataframe(combined, output_dir / "classification_accuracy_validation_all.csv")
    write_dataframe(thresholds, output_dir / "thresholds_by_seed.csv")
    write_dataframe(points, output_dir / "threshold_validation_by_point.csv")
    write_dataframe(masses, output_dir / "threshold_validation_by_mass.csv")
    plot_validated_thresholds(masses, output_dir / "plots" / "minimum_events_vs_mass_validated.pdf")
    write_manifest(config, "validate_same_lifetime_discrimination", output_dir)
    print(f"Saved high-statistics validation to {output_dir}")


if __name__ == "__main__":
    main()
