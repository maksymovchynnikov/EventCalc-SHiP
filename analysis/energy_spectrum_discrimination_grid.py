from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from analysis.compare_energy_spectra_grid import (
        MINIMUM_EVENTS,
        load_benchmarks,
    )
    from analysis.energy_spectrum_discrimination import (
        MINIMUM_BIN_N_EFF,
        NUMBER_OF_PSEUDOEXPERIMENTS,
        PSEUDOEXPERIMENT_SEED,
        TARGET_ACCURACIES,
        calculate_templates,
        float_token,
        minimum_events_for_accuracy,
        simulate_shape_discrimination,
    )
except ModuleNotFoundError:
    from compare_energy_spectra_grid import (
        MINIMUM_EVENTS,
        load_benchmarks,
    )
    from energy_spectrum_discrimination import (
        MINIMUM_BIN_N_EFF,
        NUMBER_OF_PSEUDOEXPERIMENTS,
        PSEUDOEXPERIMENT_SEED,
        TARGET_ACCURACIES,
        calculate_templates,
        float_token,
        minimum_events_for_accuracy,
        simulate_shape_discrimination,
    )


ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "energy_spectrum_discrimination_grid"
PLOT_DIR = OUTPUT_DIR / "plots"
TEMPLATE_DIR = OUTPUT_DIR / "templates"
ACCURACY_DIR = OUTPUT_DIR / "accuracy_tables"

# CONFIGURATION
TARGET_MASSES_GEV = (
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
)

LOG_INTERVAL_FRACTIONS = (
    ("low", 0.10), # Not on boundary to avoid tiny numerical changes
    ("mid", 0.50),
    ("high", 0.90), # Not on boundary to avoid tiny numerical changes
)

MAXIMUM_OBSERVED_EVENTS = 100

SELECTION_ONLY = False # After checking the selected points, change it to False.


def threshold_column(target_accuracy: float) -> str:
    """Return the summary-column name for one target accuracy."""
    return f"minimum_events_for_{100 * target_accuracy:.0f}pct_worst_case_accuracy"


def select_grid_points(common_ranges: pd.DataFrame) -> pd.DataFrame:
    """
    Select three logarithmically spaced representative lifetimes for every requested mass.

    No D_TV stability requirement is imposed here. The purpose is to 
    measure how discrimination changes across the full common N_events >= 10 interval.
    """
    rows = []

    for target_mass in TARGET_MASSES_GEV:
        matches = common_ranges.loc[
            np.isclose(
                common_ranges["mass_GeV"],
                target_mass,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]

        if matches.empty:
            print(f"Warning: no common N_events >= 10 interval for m_a = {target_mass:g} GeV")

            continue

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one common lifetime interval for m_a = {target_mass:g} GeV."
            )

        row = matches.iloc[0]
        ctau_lower_m = float(row["ctau_lower_m"])
        ctau_upper_m = float(row["ctau_upper_m"])
        log_lower = np.log(ctau_lower_m)
        log_upper = np.log(ctau_upper_m)

        for (lifetime_label, fraction) in LOG_INTERVAL_FRACTIONS:
            ctau_m = float(np.exp(log_lower + fraction * (log_upper - log_lower)))

            rows.append(
                {
                    "mass_GeV": target_mass,
                    "lifetime_label": (lifetime_label),
                    "log_interval_fraction": (fraction),
                    "ctau_m": ctau_m,
                    "ctau_lower_m": (ctau_lower_m),
                    "ctau_upper_m": (ctau_upper_m),
                }
            )

    if not rows:
        raise RuntimeError("No lifetime grid points were selected.")

    return pd.DataFrame(rows)


def write_templates(
    *,
    mass_gev: float,
    ctau_m: float,
    lifetime_label: str,
    energy_edges: np.ndarray,
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
) -> Path:
    """Save the probability templates used at one grid point."""
    table = pd.DataFrame(
        {
            "mass_GeV": mass_gev,
            "ctau_m": ctau_m,
            "lifetime_label": (lifetime_label),
            "bin_index": np.arange(len(photon_probabilities)),
            "energy_low_GeV": (energy_edges[:-1]),
            "energy_high_GeV": (energy_edges[1:]),
            "photon_probability": (photon_probabilities),
            "su2_probability": (su2_probabilities),
            "log_su2_over_photon": np.log(su2_probabilities / photon_probabilities),
        }
    )

    path = TEMPLATE_DIR / (
        "probability_templates"
        f"_ma_{float_token(mass_gev)}"
        f"_ctau_{float_token(ctau_m)}"
        f"_{lifetime_label}"
        ".csv"
    )

    table.to_csv(path, index=False)

    return path


def plot_mass_thresholds(mass_summary: pd.DataFrame) -> Path:
    """
    Plot the required number of events versus lifetime for one ALP mass.
    """
    mass_summary = mass_summary.sort_values("ctau_m")
    mass_gev = float(mass_summary.iloc[0]["mass_GeV"])

    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    finite_values = []
    for target_accuracy in TARGET_ACCURACIES:
        values = pd.to_numeric(
            mass_summary[threshold_column(target_accuracy)],
            errors="coerce",
        )

        finite_values.extend(values[np.isfinite(values)].tolist())

        axis.plot(
            mass_summary["ctau_m"],
            values,
            marker="o",
            label=(f"{100 * target_accuracy:.0f}% " "worst-case accuracy"),
        )

    axis.set_xscale("log")
    axis.set_xlabel(r"$c\tau_a$ [m]")
    axis.set_ylabel("Required observed ALP decays")
    axis.set_title(rf"Lifetime-dependent shape discrimination: $m_a={mass_gev:g}$ GeV")
    axis.grid(True, alpha=0.3)
    axis.legend()

    if finite_values:
        maximum = max(finite_values)
        axis.set_ylim(0.5, max(3.5, maximum + 1.0))
        if maximum <= 20:
            axis.set_yticks(np.arange(1, int(np.ceil(maximum)) + 2))

    figure.tight_layout()

    output_stem = PLOT_DIR / f"minimum_events_vs_ctau_ma_{float_token(mass_gev)}"
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    return pdf_path


def plot_combined_95pct(summary: pd.DataFrame) -> Path:
    """Plot the 95% threshold for all analysed masses."""
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    column = threshold_column(0.95)
    finite_values = []

    for (mass_gev, mass_summary) in summary.groupby("mass_GeV", sort=True):
        mass_summary = mass_summary.sort_values("ctau_m")
        values = pd.to_numeric(mass_summary[column], errors="coerce")
        finite_values.extend(values[np.isfinite(values)].tolist())
        axis.plot(
            mass_summary["ctau_m"],
            values,
            marker="o",
            label=(rf"$m_a={mass_gev:g}$ GeV"),
        )

    axis.set_xscale("log")
    axis.set_xlabel(r"$c\tau_a$ [m]")
    axis.set_ylabel("Events required for 95% " "worst-case accuracy")
    axis.set_title("Lifetime-dependent " "shape discrimination")
    axis.grid(True, alpha=0.3)
    axis.legend()

    if finite_values:
        maximum = max(finite_values)
        axis.set_ylim(0.5, max(3.5, maximum + 1.0))
        if maximum <= 20:
            axis.set_yticks(np.arange(1, int(np.ceil(maximum)) + 2))

    figure.tight_layout()
    output_stem = PLOT_DIR / ("minimum_events_95pct" "_all_masses")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return pdf_path


def main() -> None:
    """Run the lifetime-dependent discrimination grid."""
    for directory in (
        OUTPUT_DIR,
        PLOT_DIR,
        TEMPLATE_DIR,
        ACCURACY_DIR,
    ):
        directory.mkdir(parents=True,exist_ok=True)

    common_ranges = load_benchmarks()
    selected_points = select_grid_points(common_ranges) # Selection of LOG_INTERVAL_FRACTIONS

    selection_path = OUTPUT_DIR / "selected_lifetime_points.csv"
    selected_points.to_csv(selection_path, index=False)

    print()
    print("=" * 80)
    print("Lifetime-dependent " "energy-spectrum discrimination")
    print(
        "Selection: three interior points in each common "
        f"N_events >= {MINIMUM_EVENTS:g} lifetime interval."
    )
    print("No D_TV stability cut is imposed " "in this script.")
    print("=" * 80)

    print(selected_points.to_string(index=False))

    print()
    print(f"Selected points saved to: {selection_path}")
    if SELECTION_ONLY:
        print()
        print(
            "SELECTION_ONLY = True, so EventCalc was not launched. Change it to False to run the grid."
        )
        return

    unique_masses = sorted(selected_points["mass_GeV"].unique())
    mass_seed_index = {mass: index for (index, mass) in enumerate(unique_masses)}

    all_accuracy_tables = []
    summary_rows = []

    for (point_index, row) in enumerate(selected_points.itertuples(index=False)):
        mass_gev = float(row.mass_GeV)
        ctau_m = float(row.ctau_m)
        lifetime_label = str(row.lifetime_label)

        print()
        print("#" * 80)
        print(f"m_a = {mass_gev:g} GeV, c_tau = {ctau_m:.6g} m ({lifetime_label})")
        print("#" * 80)

        # Reusing the same mass seed at all three lifetimes
        # reduces irrelevant Monte Carlo fluctuations when
        # comparing the lifetime dependence.
        templates = calculate_templates(
            mass_gev=mass_gev,
            ctau_m=ctau_m,
            mass_index=(mass_seed_index[mass_gev]),
        )

        photon_probabilities = np.asarray(templates["photon_probabilities"], dtype=float)
        su2_probabilities = np.asarray(templates["su2_probabilities"], dtype=float)
        energy_edges = np.asarray(templates["energy_edges"], dtype=float)

        photon_events = float(templates["photon_n_events"])
        su2_events = float(templates["su2_n_events"])

        if photon_events < MINIMUM_EVENTS or su2_events < MINIMUM_EVENTS:
            raise RuntimeError(
                "A selected point does not pass N_events >= 10 in the current EventCalc rerun:\n"
                f"m_a = {mass_gev:g} GeV, "
                f"c_tau = {ctau_m:.6g} m\n"
                f"photon = {photon_events:.6g}, "
                f"SU(2)_L = {su2_events:.6g}"
            )

        template_path = write_templates(
            mass_gev=mass_gev,
            ctau_m=ctau_m,
            lifetime_label=(lifetime_label),
            energy_edges=(energy_edges),
            photon_probabilities=(photon_probabilities),
            su2_probabilities=(su2_probabilities),
        )

        model_tv = 0.5 * float(np.sum(np.abs(photon_probabilities - su2_probabilities)))
        photon_kl_su2 = float(
            np.sum(photon_probabilities * np.log(photon_probabilities / su2_probabilities))
        )
        su2_kl_photon = float(
            np.sum(su2_probabilities * np.log(su2_probabilities / photon_probabilities))
        )

        accuracy = simulate_shape_discrimination(
            photon_probabilities=(photon_probabilities),
            su2_probabilities=(su2_probabilities),
            maximum_events=(MAXIMUM_OBSERVED_EVENTS),
            number_of_pseudoexperiments=(NUMBER_OF_PSEUDOEXPERIMENTS),
            seed=(PSEUDOEXPERIMENT_SEED + 10_000 * point_index),
        )

        accuracy.insert(0, "lifetime_label", lifetime_label)
        accuracy.insert(0, "ctau_m", ctau_m)
        accuracy.insert(0, "mass_GeV", mass_gev)

        accuracy_path = ACCURACY_DIR / (
            "classification_accuracy"
            f"_ma_{float_token(mass_gev)}"
            f"_ctau_{float_token(ctau_m)}"
            f"_{lifetime_label}"
            ".csv"
        )

        accuracy.to_csv(accuracy_path, index=False)
        all_accuracy_tables.append(accuracy)

        thresholds = {
            target: minimum_events_for_accuracy(accuracy, target)
            for target in (TARGET_ACCURACIES)
        }

        summary_row = {
            "mass_GeV": mass_gev,
            "lifetime_label": (lifetime_label),
            "log_interval_fraction": float(row.log_interval_fraction),
            "ctau_m": ctau_m,
            "ctau_lower_m": float(row.ctau_lower_m),
            "ctau_upper_m": float(row.ctau_upper_m),
            "photon_expected_events": (photon_events),
            "su2_expected_events": (su2_events),
            "number_of_adaptive_bins": (len(energy_edges) - 1),
            "minimum_required_bin_N_eff": (MINIMUM_BIN_N_EFF),
            "photon_template_total_N_eff": float(templates["photon_total_n_eff"]),
            "su2_template_total_N_eff": float(templates["su2_total_n_eff"]),
            "model_total_variation_distance": (model_tv),
            "KL_photon_to_su2_per_event": (photon_kl_su2),
            "KL_su2_to_photon_per_event": (su2_kl_photon),
        }

        for target in TARGET_ACCURACIES:
            summary_row[threshold_column(target)] = thresholds[target]

        summary_rows.append(summary_row)

        # Updated after every expensive EventCalc point.
        checkpoint_path = OUTPUT_DIR / ("discrimination_grid_" "summary_checkpoint.csv")
        pd.DataFrame(summary_rows).to_csv(checkpoint_path, index=False)

        print(f"Adaptive energy bins: {len(energy_edges) - 1}")
        print(f"Model-to-model D_TV: {model_tv:.6g}")
        print(
            "Expected events, not used "
            "in classification: "
            f"photon = {photon_events:.6g}, "
            f"SU(2)_L = {su2_events:.6g}"
        )

        for target in TARGET_ACCURACIES:
            result = thresholds[target]
            result_text = (
                f"{result} events"
                if result is not None
                else f"not reached by {MAXIMUM_OBSERVED_EVENTS} events"
            )
            print(f"{100 * target:.0f}% worst-case accuracy: {result_text}")

        print(f"Template saved to: {template_path}")
        print(f"Accuracy table saved to: {accuracy_path}")

    summary = pd.DataFrame(summary_rows).sort_values(["mass_GeV", "ctau_m"], ignore_index=True)
    summary_path = OUTPUT_DIR / "discrimination_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    combined_accuracy = pd.concat(all_accuracy_tables, ignore_index=True)
    combined_accuracy_path = OUTPUT_DIR / ("classification_accuracy_" "grid_all.csv")
    combined_accuracy.to_csv(combined_accuracy_path, index=False)

    plot_paths = []
    for (_, mass_summary) in summary.groupby("mass_GeV", sort=True):
        plot_paths.append(plot_mass_thresholds(mass_summary))

    plot_paths.append(plot_combined_95pct(summary))

    print()
    print("=" * 80)
    print("Lifetime-dependent " "discrimination grid finished.")
    print(f"Grid points analysed: {len(summary)}")
    print(f"Summary saved to: {summary_path}")
    print(f"Combined accuracy table saved to: {combined_accuracy_path}")
    for path in plot_paths:
        print(f"Plot saved to: {path}")


if __name__ == "__main__":
    main()
