from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from analysis.energy_spectrum_discrimination import (
        simulate_shape_discrimination,
    )
except ModuleNotFoundError:
    from energy_spectrum_discrimination import (
        simulate_shape_discrimination,
    )


ANALYSIS_DIR = Path(__file__).resolve().parent
GRID_OUTPUT_DIR = ANALYSIS_DIR / "energy_spectrum_discrimination_grid"
TEMPLATE_DIR = GRID_OUTPUT_DIR / "templates"
GRID_SUMMARY_PATH = GRID_OUTPUT_DIR / "discrimination_grid_summary.csv"
OUTPUT_DIR = ANALYSIS_DIR / "energy_spectrum_discrimination_validation"
PLOT_DIR = OUTPUT_DIR / "plots"


NUMBER_OF_PSEUDOEXPERIMENTS = 100_000
MAXIMUM_OBSERVED_EVENTS = 15
NUMBER_OF_VALIDATION_SEEDS = 5
BASE_VALIDATION_SEED = 20260724

TARGET_ACCURACIES = (
    0.90,
    0.95,
    0.99,
)

# A variation of at most one observed event between independent
# pseudoexperiment seeds is considered numerically stable.
MAXIMUM_ALLOWED_THRESHOLD_SPREAD = 1


def threshold_column(target_accuracy: float) -> str:
    """Return the threshold-column name used by the grid summary."""
    return f"minimum_events_for_{100.0 * target_accuracy:.0f}pct_worst_case_accuracy"


def unique_value(data: pd.DataFrame, column: str, path: Path):
    """Read one metadata value that must be constant in a template."""
    values = data[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one value of {column!r} in:\n"
            f"  {path}\n"
            f"Found:\n{values.to_string(index=False)}"
        )

    return values.iloc[0]


def load_probability_template(path: Path) -> dict:
    """Load and validate one saved probability-template table."""
    data = pd.read_csv(path)

    required_columns = {
        "mass_GeV",
        "ctau_m",
        "lifetime_label",
        "bin_index",
        "energy_low_GeV",
        "energy_high_GeV",
        "photon_probability",
        "su2_probability",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in {path}:\n  {sorted(missing_columns)}")

    data = data.sort_values("bin_index").reset_index(drop=True)

    mass_gev = float(unique_value(data,"mass_GeV", path))
    ctau_m = float(unique_value(data, "ctau_m", path))
    lifetime_label = str(unique_value(data, "lifetime_label", path))

    bin_indices = data["bin_index"].to_numpy(dtype=int)
    expected_bin_indices = np.arange(len(data))

    if not np.array_equal(
        bin_indices,
        expected_bin_indices,
    ):
        raise ValueError(f"The bin indices are not consecutive in:\n  {path}")

    energy_low = data["energy_low_GeV"].to_numpy(dtype=float)
    energy_high = data["energy_high_GeV"].to_numpy(dtype=float)

    photon_probabilities = data["photon_probability"].to_numpy(dtype=float)
    su2_probabilities = data["su2_probability"].to_numpy(dtype=float)

    arrays_to_check = {
        "energy_low_GeV": energy_low,
        "energy_high_GeV": energy_high,
        "photon_probability": photon_probabilities,
        "su2_probability": su2_probabilities,
    }

    for name, values in arrays_to_check.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values found in {name!r}:\n  {path}")

    if mass_gev <= 0.0 or ctau_m <= 0.0:
        raise ValueError(f"The mass and lifetime must be positive:\n  {path}")

    if np.any(energy_high <= energy_low):
        raise ValueError(f"At least one energy bin has non-positive width:\n  {path}")

    if len(data) > 1 and not np.allclose(
        energy_high[:-1],
        energy_low[1:],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"The energy bins are not contiguous:\n  {path}")

    for (model_name, probabilities) in (
        ("ALP-photon-combined", photon_probabilities),
        ("ALP-SU2L", su2_probabilities),
    ):
        if np.any(probabilities <= 0.0):
            raise ValueError(
                "All smoothed template probabilities must be "
                f"strictly positive for {model_name}:\n"
                f"  {path}"
            )

        probability_sum = float(np.sum(probabilities))

        if not np.isclose(
            probability_sum,
            1.0,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                f"{model_name} probabilities do not sum to one:\n"
                f"  file = {path}\n"
                f"  sum = {probability_sum:.16g}"
            )

    if "log_su2_over_photon" in data.columns:
        stored_log_ratio = data["log_su2_over_photon"].to_numpy(dtype=float)

        calculated_log_ratio = np.log(su2_probabilities / photon_probabilities)

        if not np.allclose(
            stored_log_ratio,
            calculated_log_ratio,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "The stored log-likelihood ratios are inconsistent "
                "with the probabilities:\n"
                f"  {path}"
            )

    return {
        "path": path,
        "mass_GeV": mass_gev,
        "ctau_m": ctau_m,
        "lifetime_label": lifetime_label,
        "number_of_bins": len(data),
        "photon_probabilities": (photon_probabilities),
        "su2_probabilities": (su2_probabilities),
    }


def minimum_persistent_events(
    accuracy_table: pd.DataFrame,
    accuracy_column: str,
    target_accuracy: float,
) -> int | None:
    """
    Return the first event count from which the accuracy remains
    above the requested target for every larger tested event count.

    This is more conservative than accepting a single upward
    statistical fluctuation across the target line.
    """
    values = accuracy_table[accuracy_column].to_numpy(dtype=float)
    event_counts = accuracy_table["number_of_events"].to_numpy(dtype=int)

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {accuracy_column!r}.")

    passing = values >= target_accuracy
    passing_from_here_onward = np.logical_and.accumulate(passing[::-1])[::-1]
    passing_indices = np.flatnonzero(passing_from_here_onward)

    if len(passing_indices) == 0:
        return None

    return int(event_counts[passing_indices[0]])


def run_validation_seed(
    *,
    template: dict,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run one high-statistics pseudoexperiment validation."""
    accuracy = simulate_shape_discrimination(
        photon_probabilities=(template["photon_probabilities"]),
        su2_probabilities=(template["su2_probabilities"]),
        maximum_events=(MAXIMUM_OBSERVED_EVENTS),
        number_of_pseudoexperiments=(NUMBER_OF_PSEUDOEXPERIMENTS),
        seed=seed,
    )

    accuracy.insert(0, "validation_seed", seed)
    accuracy.insert(0, "lifetime_label", template["lifetime_label"])
    accuracy.insert(0, "ctau_m", template["ctau_m"])
    accuracy.insert(0, "mass_GeV", template["mass_GeV"])

    threshold_rows = []
    for target_accuracy in TARGET_ACCURACIES:
        photon_threshold = minimum_persistent_events(
            accuracy_table=accuracy,
            accuracy_column=("photon_correct_fraction"),
            target_accuracy=(target_accuracy),
        )

        su2_threshold = minimum_persistent_events(
            accuracy_table=accuracy,
            accuracy_column=("su2_correct_fraction"),
            target_accuracy=(target_accuracy),
        )

        worst_case_threshold = minimum_persistent_events(
            accuracy_table=accuracy,
            accuracy_column=("worst_case_correct_fraction"),
            target_accuracy=(target_accuracy),
        )

        if photon_threshold is not None and su2_threshold is not None:
            expected_worst_case = max(photon_threshold, su2_threshold,)

            if worst_case_threshold != expected_worst_case:
                raise RuntimeError(
                    "The worst-case threshold is inconsistent with "
                    "the two conditional thresholds:\n"
                    f"m_a = {template['mass_GeV']:g} GeV\n"
                    f"c_tau = {template['ctau_m']:.6g} m\n"
                    f"target = {target_accuracy:.3f}\n"
                    f"photon threshold = {photon_threshold}\n"
                    f"SU(2)_L threshold = {su2_threshold}\n"
                    f"worst-case threshold = "
                    f"{worst_case_threshold}"
                )

        threshold_rows.append(
            {
                "mass_GeV": (template["mass_GeV"]),
                "ctau_m": (template["ctau_m"]),
                "lifetime_label": (template["lifetime_label"]),
                "number_of_bins": (template["number_of_bins"]),
                "validation_seed": seed,
                "target_accuracy": (target_accuracy),
                "photon_threshold": (photon_threshold),
                "su2_threshold": (su2_threshold),
                "worst_case_threshold": (worst_case_threshold),
            }
        )

    return (accuracy, threshold_rows)


def finite_summary(values: pd.Series) -> dict:
    """Summarize threshold values while recording missing results."""
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    all_reached = len(finite_values) == len(numeric_values)

    if len(finite_values) == 0:
        return {
            "all_reached": False,
            "minimum": np.nan,
            "median": np.nan,
            "maximum": np.nan,
            "spread": np.nan,
        }

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    return {
        "all_reached": all_reached,
        "minimum": minimum,
        "median": float(np.median(finite_values)),
        "maximum": maximum,
        "spread": (maximum - minimum),
    }


def summarize_points(thresholds_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed dependence separately at each grid point."""
    rows = []

    group_columns = [
        "mass_GeV",
        "ctau_m",
        "lifetime_label",
        "number_of_bins",
        "target_accuracy",
    ]

    for (group_values, group) in thresholds_by_seed.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            mass_gev,
            ctau_m,
            lifetime_label,
            number_of_bins,
            target_accuracy,
        ) = group_values

        photon = finite_summary(group["photon_threshold"])
        su2 = finite_summary(group["su2_threshold"])

        worst_case = finite_summary(group["worst_case_threshold"])
        threshold_stable = worst_case["all_reached"] and worst_case["spread"] <= (
            MAXIMUM_ALLOWED_THRESHOLD_SPREAD
        )

        rows.append(
            {
                "mass_GeV": mass_gev,
                "ctau_m": ctau_m,
                "lifetime_label": (lifetime_label),
                "number_of_bins": (number_of_bins),
                "target_accuracy": (target_accuracy),
                "number_of_validation_seeds": (len(group)),
                "photon_all_seeds_reached": (photon["all_reached"]),
                "photon_threshold_min": (photon["minimum"]),
                "photon_threshold_median": (photon["median"]),
                "photon_threshold_max": (photon["maximum"]),
                "su2_all_seeds_reached": (su2["all_reached"]),
                "su2_threshold_min": (su2["minimum"]),
                "su2_threshold_median": (su2["median"]),
                "su2_threshold_max": (su2["maximum"]),
                "worst_case_all_seeds_reached": (worst_case["all_reached"]),
                "worst_case_threshold_min": (worst_case["minimum"]),
                "worst_case_threshold_median": (worst_case["median"]),
                "worst_case_threshold_max": (worst_case["maximum"]),
                "worst_case_threshold_spread": (worst_case["spread"]),
                "threshold_stable": (threshold_stable),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "mass_GeV",
            "target_accuracy",
            "ctau_m",
        ],
        ignore_index=True,
    )


def add_original_grid_thresholds(
    point_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare validation thresholds with the original grid result."""
    point_summary = point_summary.copy()
    point_summary["original_grid_threshold"] = np.nan
    point_summary["original_threshold_within_validation_range"] = False

    if not GRID_SUMMARY_PATH.exists():
        print()
        print("Warning: the original grid summary was not found:")
        print(f"  {GRID_SUMMARY_PATH}")
        print("The validation will continue without the direct " "threshold comparison.")

        return point_summary

    grid_summary = pd.read_csv(GRID_SUMMARY_PATH)
    for row_index, row in point_summary.iterrows():
        target_accuracy = float(row["target_accuracy"])
        original_column = threshold_column(target_accuracy)

        if original_column not in grid_summary.columns:
            raise ValueError(
                f"The original grid summary is missing the column:\n  {original_column}"
            )

        matches = grid_summary.loc[
            np.isclose(
                grid_summary["mass_GeV"],
                float(row["mass_GeV"]),
                rtol=0.0,
                atol=1.0e-12,
            )
            & np.isclose(
                grid_summary["ctau_m"],
                float(row["ctau_m"]),
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            & (grid_summary["lifetime_label"].astype(str) == str(row["lifetime_label"]))
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Could not uniquely match a validation point to "
                "the original grid summary:\n"
                f"m_a = {row['mass_GeV']:g} GeV\n"
                f"c_tau = {row['ctau_m']:.12g} m\n"
                f"label = {row['lifetime_label']}\n"
                f"matches = {len(matches)}"
            )

        original_value = pd.to_numeric(
            matches.iloc[0][original_column],
            errors="coerce",
        )

        point_summary.at[
            row_index,
            "original_grid_threshold",
        ] = original_value

        validation_min = float(row["worst_case_threshold_min"])
        validation_max = float(row["worst_case_threshold_max"])

        reproduced = (
            np.isfinite(original_value)
            and np.isfinite(validation_min)
            and np.isfinite(validation_max)
            and validation_min <= original_value <= validation_max
        )

        point_summary.at[
            row_index,
            "original_threshold_within_validation_range",
        ] = reproduced

    return point_summary


def summarize_masses(
    point_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the final mass-level result.

    The conservative requirement is the maximum threshold over:
      1. all three lifetime points;
      2. all validation seeds.
    """
    rows = []

    for ((mass_gev, target_accuracy), group) in point_summary.groupby(
        ["mass_GeV", "target_accuracy"], sort=True,
    ):
        minima = pd.to_numeric(group["worst_case_threshold_min"], errors="coerce").to_numpy(dtype=float)
        medians = pd.to_numeric(group["worst_case_threshold_median"], errors="coerce").to_numpy(dtype=float)
        maxima = pd.to_numeric(group["worst_case_threshold_max"], errors="coerce").to_numpy(dtype=float)

        finite_minima = minima[np.isfinite(minima)]
        finite_medians = medians[np.isfinite(medians)]
        finite_maxima = maxima[np.isfinite(maxima)]
        all_lifetimes_reached = len(finite_maxima) == len(group)

        if len(finite_maxima) == 0:
            conservative_events = np.nan
            limiting_lifetime_label = ""
            limiting_ctau_m = np.nan

        else:
            conservative_events = float(np.max(finite_maxima))

            limiting_candidates = group.loc[
                np.isclose(
                    group["worst_case_threshold_max"],
                    conservative_events,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ].sort_values("ctau_m")

            limiting_row = limiting_candidates.iloc[0]
            limiting_lifetime_label = str(limiting_row["lifetime_label"])
            limiting_ctau_m = float(limiting_row["ctau_m"])

        rows.append(
            {
                "mass_GeV": mass_gev,
                "target_accuracy": (target_accuracy),
                "target_accuracy_percent": (100.0 * target_accuracy),
                "number_of_lifetimes": (len(group)),
                "all_lifetimes_reached": (all_lifetimes_reached),
                "minimum_events_over_lifetimes_and_seeds": (
                    float(np.min(finite_minima)) if len(finite_minima) > 0 else np.nan
                ),
                "median_events_over_lifetimes": (
                    float(np.median(finite_medians)) if len(finite_medians) > 0 else np.nan
                ),
                "conservative_required_events": (conservative_events),
                "limiting_lifetime_label": (limiting_lifetime_label),
                "limiting_ctau_m": (limiting_ctau_m),
                "all_seed_thresholds_stable": bool(group["threshold_stable"].all()),
            }
        )

    return pd.DataFrame(rows).sort_values(["target_accuracy", "mass_GeV"], ignore_index=True)


def plot_mass_summary(mass_summary: pd.DataFrame) -> Path:
    """Plot the validated conservative event requirement versus mass."""
    figure, axis = plt.subplots(figsize=(8.0, 5.8))
    plotted_values = []

    for target_accuracy in TARGET_ACCURACIES:
        target_data = mass_summary.loc[
            np.isclose(
                mass_summary["target_accuracy"],
                target_accuracy,
                rtol=0.0,
                atol=1.0e-12,
            )
        ].sort_values("mass_GeV")

        masses = target_data["mass_GeV"].to_numpy(dtype=float)
        lower = target_data["minimum_events_over_lifetimes_and_seeds"].to_numpy(dtype=float)
        conservative = target_data["conservative_required_events"].to_numpy(dtype=float)
        valid = np.isfinite(masses) & np.isfinite(lower) & np.isfinite(conservative)

        (line,) = axis.plot(
            masses[valid],
            conservative[valid],
            marker="o",
            linewidth=2.0,
            label=(f"{100.0 * target_accuracy:.0f}% " "worst-case accuracy"),
        )

        axis.fill_between(
            masses[valid],
            lower[valid],
            conservative[valid],
            alpha=0.15,
            color=line.get_color(),
        )

        plotted_values.extend(conservative[valid].tolist())

    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel("Required observed ALP decays")
    axis.set_title("Shape-only identification of the ALP coupling model")
    axis.grid(True, alpha=0.3)
    axis.legend(title=("Line: conservative maximum\n" "Band: lifetime and seed range"))

    unique_masses = np.sort(mass_summary["mass_GeV"].unique())
    axis.set_xticks(unique_masses)

    if plotted_values:
        maximum = float(np.max(plotted_values))
        axis.set_ylim(0.5, max(3.5, maximum + 1.0))
        if maximum <= 20.0:
            axis.set_yticks(np.arange(1, int(np.ceil(maximum)) + 2))

    figure.tight_layout()
    output_stem = PLOT_DIR / "minimum_events_vs_mass_validated"
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    return pdf_path


def main() -> None:
    """Validate the saved lifetime-dependent discrimination templates."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    template_paths = sorted(TEMPLATE_DIR.glob("probability_templates_*.csv"))

    if not template_paths:
        raise FileNotFoundError(
            "No saved probability templates were found:\n"
            f"  {TEMPLATE_DIR}\n"
            "Run energy_spectrum_discrimination_grid.py first."
        )

    print()
    print("=" * 80)
    print("High-statistics validation of " "energy-spectrum discrimination")
    print("This is pure post-processing: EventCalc is not launched.")
    print(f"Pseudoexperiments per hypothesis and seed: {NUMBER_OF_PSEUDOEXPERIMENTS:,}")
    print(f"Independent validation seeds: {NUMBER_OF_VALIDATION_SEEDS}")
    print(f"Maximum observed events tested: {MAXIMUM_OBSERVED_EVENTS}")
    print("=" * 80)

    all_accuracy_tables = []
    all_threshold_rows = []
    for (point_index, template_path) in enumerate(template_paths):
        template = load_probability_template(template_path)
        print()
        print("#" * 80)
        print(
            f"m_a = {template['mass_GeV']:g} GeV, "
            f"c_tau = {template['ctau_m']:.6g} m, "
            f"label = {template['lifetime_label']}"
        )
        print(f"Adaptive energy bins: {template['number_of_bins']}")
        print("#" * 80)

        for seed_index in range(NUMBER_OF_VALIDATION_SEEDS):
            seed = BASE_VALIDATION_SEED + 100_000 * point_index + 1_000 * seed_index
            accuracy, threshold_rows = run_validation_seed(template=template, seed=seed)

            all_accuracy_tables.append(accuracy)
            all_threshold_rows.extend(threshold_rows)

            threshold_text = []
            for threshold_row in threshold_rows:
                target_percent = 100.0 * threshold_row["target_accuracy"]
                result = threshold_row["worst_case_threshold"]
                if result is None:
                    result_text = f">{MAXIMUM_OBSERVED_EVENTS}"
                else:
                    result_text = str(int(result))

                threshold_text.append(f"{target_percent:.0f}%: {result_text}")

            print(f"Seed {seed}: " + ", ".join(threshold_text))

    combined_accuracy = pd.concat(all_accuracy_tables, ignore_index=True)
    thresholds_by_seed = pd.DataFrame(all_threshold_rows)

    accuracy_path = OUTPUT_DIR / "classification_accuracy_validation_all.csv"
    thresholds_by_seed_path = OUTPUT_DIR / "thresholds_by_seed.csv"

    combined_accuracy.to_csv(accuracy_path, index=False)
    thresholds_by_seed.to_csv(thresholds_by_seed_path, index=False)

    point_summary = summarize_points(thresholds_by_seed)
    point_summary = add_original_grid_thresholds(point_summary)
    point_summary_path = OUTPUT_DIR / "threshold_validation_by_point.csv"
    point_summary.to_csv(point_summary_path, index=False)

    mass_summary = summarize_masses(point_summary)
    mass_summary_path = OUTPUT_DIR / "threshold_validation_by_mass.csv"
    mass_summary.to_csv(mass_summary_path, index=False)

    plot_pdf_path = plot_mass_summary(mass_summary)

    unstable = point_summary.loc[~point_summary["threshold_stable"]]
    unreproduced = point_summary.loc[
        np.isfinite(point_summary["original_grid_threshold"])
        & ~point_summary["original_threshold_within_validation_range"]
    ]

    print()
    print("=" * 80)
    print("Validation finished")
    print("=" * 80)

    display_columns = [
        "mass_GeV",
        "target_accuracy_percent",
        "conservative_required_events",
        "limiting_lifetime_label",
        "limiting_ctau_m",
        "all_seed_thresholds_stable",
    ]

    print()
    print(mass_summary[display_columns].to_string(index=False))

    print()
    print(f"Accuracy tables saved to:\n  {accuracy_path}")
    print(f"Seed thresholds saved to:\n  {thresholds_by_seed_path}")
    print(f"Point-level validation saved to:\n  {point_summary_path}")
    print(f"Mass-level validation saved to:\n  {mass_summary_path}")
    print(f"Final plot saved to: {plot_pdf_path}")

    if unstable.empty:
        print()
        print(f"All thresholds are stable within {MAXIMUM_ALLOWED_THRESHOLD_SPREAD} event.")
    else:
        print()
        print(
            "Warning: the following thresholds vary by more than "
            f"{MAXIMUM_ALLOWED_THRESHOLD_SPREAD} event between seeds:"
        )

        print(
            unstable[
                [
                    "mass_GeV",
                    "ctau_m",
                    "lifetime_label",
                    "target_accuracy",
                    "worst_case_threshold_min",
                    "worst_case_threshold_max",
                ]
            ].to_string(index=False)
        )

    if unreproduced.empty:
        print()
        print(
            "All finite original grid thresholds are reproduced "
            "within the high-statistics validation range."
        )
    else:
        print()
        print(
            "Warning: some original grid thresholds are outside "
            "the range obtained from the validation seeds:"
        )

        print(
            unreproduced[
                [
                    "mass_GeV",
                    "ctau_m",
                    "lifetime_label",
                    "target_accuracy",
                    "original_grid_threshold",
                    "worst_case_threshold_min",
                    "worst_case_threshold_max",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
