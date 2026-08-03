"""Compare available analysis2 production results with committed analysis/ outputs."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis2.config import PROFILES, get_config
from analysis2.lifetimes import lifetime_point_records
from analysis2.paths import LEGACY_ANALYSIS_ROOT, profile_output_dir
from analysis2.workflows import write_dataframe


@dataclass(frozen=True)
class Comparison:
    category: str
    status: str
    compared_rows: int = 0
    maximum_absolute_difference: float = np.nan
    maximum_relative_difference: float = np.nan
    location: str = ""
    detail: str = ""


def compare_frames(
    category: str, legacy: pd.DataFrame, current: pd.DataFrame, keys: list[str],
    *, exact_columns: list[str] = (), float_columns: list[str] = (),
    rtol: float = 1e-10, atol: float = 1e-12,
) -> Comparison:
    exact_columns, float_columns = list(exact_columns), list(float_columns)
    missing_columns = (set(keys) | set(exact_columns) | set(float_columns)) - (
        set(legacy.columns) & set(current.columns)
    )
    if missing_columns:
        return Comparison(category, "genuine_mismatch", detail=f"missing columns: {sorted(missing_columns)}")
    for name, frame in (("legacy", legacy), ("new", current)):
        duplicates = frame.duplicated(keys, keep=False)
        if duplicates.any():
            return Comparison(
                category, "genuine_mismatch",
                detail=f"{name} contains {int(duplicates.sum())} rows with duplicate keys",
            )
    merged = legacy[keys + exact_columns + float_columns].merge(
        current[keys + exact_columns + float_columns], on=keys, how="outer",
        suffixes=("_legacy", "_new"), indicator=True, validate="one_to_one",
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        return Comparison(
            category, "genuine_mismatch", len(merged) - int(unmatched.sum()),
            detail=f"{int(unmatched.sum())} unmatched key rows",
        )
    for column in exact_columns:
        left, right = merged[f"{column}_legacy"], merged[f"{column}_new"]
        equal = left.eq(right) | (left.isna() & right.isna())
        if not equal.all():
            index = int(np.flatnonzero(~equal.to_numpy())[0])
            return Comparison(
                category, "genuine_mismatch", len(merged), location=str(merged.loc[index, keys].to_dict()),
                detail=f"exact column {column!r} differs",
            )
    largest_absolute = largest_relative = 0.0
    location = ""
    all_exact = True
    for column in float_columns:
        left = pd.to_numeric(merged[f"{column}_legacy"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(merged[f"{column}_new"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(left) & np.isnan(right)
        finite = np.isfinite(left) & np.isfinite(right)
        if not np.all(both_nan | finite):
            return Comparison(category, "genuine_mismatch", len(merged), detail=f"finite state differs in {column}")
        if not np.array_equal(left[finite], right[finite]):
            all_exact = False
        difference = np.abs(left[finite] - right[finite])
        relative = difference / np.maximum(np.abs(left[finite]), 1e-300)
        if len(difference) and float(difference.max()) >= largest_absolute:
            local_index = np.flatnonzero(finite)[int(np.argmax(difference))]
            largest_absolute = float(difference.max())
            location = f"{merged.loc[local_index, keys].to_dict()}, column={column}"
        if len(relative):
            largest_relative = max(largest_relative, float(relative.max()))
        if not np.all(np.isclose(left[finite], right[finite], rtol=rtol, atol=atol)):
            return Comparison(
                category, "genuine_mismatch", len(merged), largest_absolute,
                largest_relative, location, f"tolerance rtol={rtol:g}, atol={atol:g} exceeded",
            )
    if all_exact:
        return Comparison(category, "exact_agreement", len(merged), 0.0, 0.0)
    return Comparison(
        category, "floating_point_agreement", len(merged),
        largest_absolute, largest_relative, location,
    )


def stochastic_frame_comparison(
    category: str, legacy: pd.DataFrame, current: pd.DataFrame, keys: list[str],
    columns: list[str], pseudoexperiments: int,
) -> Comparison:
    structure = compare_frames(category, legacy[keys], current[keys], keys)
    if structure.status != "exact_agreement":
        return structure
    merged = legacy[keys + columns].merge(
        current[keys + columns], on=keys, suffixes=("_legacy", "_new"), validate="one_to_one"
    )
    if all(np.array_equal(merged[f"{column}_legacy"], merged[f"{column}_new"]) for column in columns):
        return Comparison(category, "exact_agreement", len(merged))
    maximum_pull, location = 0.0, ""
    for column in columns:
        first, second = merged[f"{column}_legacy"].to_numpy(float), merged[f"{column}_new"].to_numpy(float)
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            return Comparison(category, "genuine_mismatch", len(merged), detail=f"non-finite {column}")
        standard_error = np.sqrt(
            first * (1.0 - first) / pseudoexperiments
            + second * (1.0 - second) / pseudoexperiments
        )
        pull = np.abs(first - second) / np.maximum(standard_error, 1.0 / pseudoexperiments)
        if float(pull.max()) >= maximum_pull:
            index = int(np.argmax(pull))
            maximum_pull = float(pull[index])
            location = f"{merged.loc[index, keys].to_dict()}, column={column}"
    if maximum_pull <= 5.0:
        return Comparison(category, "stochastic_agreement", len(merged), location=location,
                          detail=f"maximum conservative pull={maximum_pull:.3g}")
    return Comparison(category, "genuine_mismatch", len(merged), location=location,
                      detail=f"maximum conservative pull={maximum_pull:.3g} > 5")


def load_templates(directory: Path) -> pd.DataFrame | None:
    paths = sorted(directory.glob("probability_templates_*.csv"))
    return None if not paths else pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def missing(category: str, legacy: Path | None, current: Path | None) -> Comparison | None:
    if legacy is not None and not legacy.exists():
        return Comparison(category, "missing_legacy_counterpart", detail=str(legacy))
    if current is not None and not current.exists():
        return Comparison(category, "missing_new_counterpart", detail=str(current))
    return None


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    args = parser.parse_args()
    config = get_config(args.profile)
    new_root = profile_output_dir(config.name)
    legacy_grid = LEGACY_ANALYSIS_ROOT / "energy_spectrum_discrimination_grid"
    new_grid = new_root / "same_lifetime_discrimination"
    comparisons: list[Comparison] = [Comparison(
        "individual_lifetime_intervals", "missing_legacy_counterpart",
        detail="legacy scan computed intervals in memory but did not save a summary table",
    )]

    table_specs = [
        (
            "common_lifetime_intervals", LEGACY_ANALYSIS_ROOT / "ctau_scan/common_ctau_ranges.csv",
            new_root / "ctau_ranges/common_ctau_ranges.csv", ["mass_GeV"],
            ["upper_extends_beyond_scan"], ["ctau_lower_m", "ctau_upper_m"], 1e-6, 0.0,
        ),
        (
            "selected_lifetime_points", legacy_grid / "selected_lifetime_points.csv",
            new_grid / "selected_lifetime_points.csv", ["mass_GeV", "lifetime_label"],
            [], ["log_interval_fraction", "ctau_m", "ctau_lower_m", "ctau_upper_m"], 1e-10, 1e-12,
        ),
        (
            "expected_event_counts", legacy_grid / "discrimination_grid_summary.csv",
            new_grid / "discrimination_grid_summary.csv", ["mass_GeV", "lifetime_label"], [],
            ["ctau_m", "photon_expected_events", "su2_expected_events"], 1e-6, 0.0,
        ),
        (
            "spectrum_distances", legacy_grid / "discrimination_grid_summary.csv",
            new_grid / "discrimination_grid_summary.csv", ["mass_GeV", "lifetime_label"], [],
            ["model_total_variation_distance", "KL_photon_to_su2_per_event", "KL_su2_to_photon_per_event"],
            1e-10, 1e-12,
        ),
        (
            "required_event_counts", legacy_grid / "discrimination_grid_summary.csv",
            new_grid / "discrimination_grid_summary.csv", ["mass_GeV", "lifetime_label"],
            [f"minimum_events_for_{percent}pct_worst_case_accuracy" for percent in (90, 95, 99)],
            [], 0.0, 0.0,
        ),
        (
            "validation_thresholds", LEGACY_ANALYSIS_ROOT / "energy_spectrum_discrimination_validation/threshold_validation_by_mass.csv",
            new_root / "validation/threshold_validation_by_mass.csv", ["mass_GeV", "target_accuracy"],
            ["number_of_lifetimes", "all_lifetimes_reached", "conservative_required_events",
             "limiting_lifetime_label", "all_seed_thresholds_stable"],
            ["limiting_ctau_m"], 1e-10, 1e-12,
        ),
    ]
    for category, legacy_path, current_path, keys, exact, floating, rtol, atol in table_specs:
        absent = missing(category, legacy_path, current_path)
        comparisons.append(absent or compare_frames(
            category, pd.read_csv(legacy_path), pd.read_csv(current_path), keys,
            exact_columns=exact, float_columns=floating, rtol=rtol, atol=atol,
        ))

    legacy_templates, new_templates = load_templates(legacy_grid / "templates"), load_templates(new_grid / "templates")
    if legacy_templates is None:
        comparisons.append(Comparison("probability_templates", "missing_legacy_counterpart"))
    elif new_templates is None:
        comparisons.append(Comparison("probability_templates", "missing_new_counterpart"))
    else:
        comparisons.append(compare_frames(
            "probability_templates", legacy_templates, new_templates,
            ["mass_GeV", "lifetime_label", "bin_index"],
            float_columns=["ctau_m", "energy_low_GeV", "energy_high_GeV",
                           "photon_probability", "su2_probability", "log_su2_over_photon"],
        ))

    legacy_accuracy = legacy_grid / "classification_accuracy_grid_all.csv"
    new_accuracy = new_grid / "classification_accuracy_grid_all.csv"
    absent = missing("classification_probabilities", legacy_accuracy, new_accuracy)
    comparisons.append(absent or stochastic_frame_comparison(
        "classification_probabilities", pd.read_csv(legacy_accuracy), pd.read_csv(new_accuracy),
        ["mass_GeV", "lifetime_label", "number_of_events"],
        ["photon_correct_fraction", "su2_correct_fraction", "balanced_accuracy",
         "worst_case_correct_fraction"], config.discrimination.pseudoexperiments,
    ))

    legacy_stability = LEGACY_ANALYSIS_ROOT / "energy_spectra_lifetime_scan/lifetime_stability_summary.csv"
    current_stability = new_root / "lifetime_spectra/lifetime_stability_summary.csv"
    absent = missing("lifetime_stability_distances", legacy_stability, current_stability)
    if absent:
        comparisons.append(absent)
    else:
        legacy_data, current_data = pd.read_csv(legacy_stability), pd.read_csv(current_stability)
        keys = ["model", "mass_GeV", "ctau_m"]
        common_keys = legacy_data[keys].merge(current_data[keys], on=keys).drop_duplicates()
        comparisons.append(compare_frames(
            "lifetime_stability_distances_overlap", legacy_data.merge(common_keys, on=keys),
            current_data.merge(common_keys, on=keys), keys,
            float_columns=["total_variation_distance_to_reference", "binned_cdf_max_distance_to_reference"],
        ))
        missing_legacy = len(current_data) - len(current_data.merge(legacy_data[keys].drop_duplicates(), on=keys))
        missing_new = len(legacy_data) - len(legacy_data.merge(current_data[keys].drop_duplicates(), on=keys))
        comparisons.append(Comparison(
            "lifetime_stability_distance_coverage",
            "missing_legacy_counterpart" if missing_legacy else (
                "missing_new_counterpart" if missing_new else "exact_agreement"
            ), len(common_keys), detail=f"new-only rows={missing_legacy}; legacy-only rows={missing_new}",
        ))

    # Deterministic regression available before a new EventCalc production run.
    legacy_ranges = pd.read_csv(LEGACY_ANALYSIS_ROOT / "ctau_scan/common_ctau_ranges.csv")
    expected_points = pd.read_csv(legacy_grid / "selected_lifetime_points.csv")
    calculated_points = pd.DataFrame(lifetime_point_records(
        legacy_ranges.to_dict("records"), config.discrimination.lifetime_points
    ))
    comparisons.append(compare_frames(
        "legacy_input_point_selection_algorithm", expected_points, calculated_points,
        ["mass_GeV", "lifetime_label"],
        float_columns=["log_interval_fraction", "ctau_m", "ctau_lower_m", "ctau_upper_m"],
    ))
    report = pd.DataFrame(asdict(item) for item in comparisons)
    output_dir = new_root / "regression"
    write_dataframe(report, output_dir / "legacy_comparison_report.csv")
    print(report.to_string(index=False))
    print(f"Saved regression report to {output_dir}")
    if (report["status"] == "genuine_mismatch").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
