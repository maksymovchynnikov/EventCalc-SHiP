# The important function in this script is simulate_shape_discrimination

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from analysis.compare_energy_spectra import (
        BASE_SEED,
        MODEL_CONFIGS,
        NUMBER_OF_ENERGY_BINS,
        calculate_model_spectrum,
    )
    from analysis.compare_energy_spectra_grid import (
        ENERGY_MAX_GEV,
        MAXIMUM_TV_DISTANCE,
        MINIMUM_EVENTS,
        load_benchmarks,
        select_stable_benchmarks,
    )
except ModuleNotFoundError:
    from compare_energy_spectra import (
        BASE_SEED,
        MODEL_CONFIGS,
        NUMBER_OF_ENERGY_BINS,
        calculate_model_spectrum,
    )
    from compare_energy_spectra_grid import (
        ENERGY_MAX_GEV,
        MAXIMUM_TV_DISTANCE,
        MINIMUM_EVENTS,
        load_benchmarks,
        select_stable_benchmarks,
    )


ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "energy_spectrum_discrimination"
PLOT_DIR = OUTPUT_DIR / "plots"
TEMPLATE_DIR = OUTPUT_DIR / "templates"


# Effective Monte Carlo statistics required in each final bin.
MINIMUM_BIN_N_EFF = 100.0


# Jeffreys-prior pseudocount used to avoid exactly zero model probabilities caused by finite Monte Carlo statistics.
JEFFREYS_ALPHA = 0.5


# These pseudoexperiments are cheap compared with EventCalc.
NUMBER_OF_PSEUDOEXPERIMENTS = 20_000
MAXIMUM_OBSERVED_EVENTS = 10
TARGET_ACCURACIES = (
    0.90,
    0.95,
    0.99,
)
PSEUDOEXPERIMENT_SEED = 20260723


def float_token(value: float,) -> str:
    """Convert a number to a filename-safe string."""
    return f"{value:.12g}".replace(".", "p").replace("-", "m").replace("+", "")


def histogram_moments(spectrum: dict, energy_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate sum(w) and sum(w^2) in the requested bins."""
    energies = np.asarray(spectrum["energies"], dtype=float)
    weights = np.asarray(spectrum["weights"], dtype=float)

    sum_weights, _ = np.histogram(
        energies,
        bins=energy_edges,
        weights=weights,
    )

    sum_squared_weights, _ = np.histogram(
        energies,
        bins=energy_edges,
        weights=weights**2,
    )

    return (sum_weights, sum_squared_weights)


def find_first_problem_bin(
    spectra: dict[str, dict],
    energy_edges: np.ndarray,
    minimum_n_eff: float,
) -> int | None:
    """
    Find the first bin that is either empty for both models or has
    insufficient effective statistics in a contributing model.
    """
    sum_weights_by_model = []

    low_statistics = np.zeros(len(energy_edges) - 1, dtype=bool)

    for spectrum in spectra.values():
        sum_weights, sum_squared_weights = histogram_moments(
            spectrum,
            energy_edges,
        )

        sum_weights_by_model.append(sum_weights)

        n_eff = np.divide(
            sum_weights**2,
            sum_squared_weights,
            out=np.zeros_like(sum_weights),
            where=(sum_squared_weights > 0.0),
        )

        low_statistics |= (sum_weights > 0.0) & (n_eff < minimum_n_eff)

    sum_weights_matrix = np.vstack(sum_weights_by_model)

    empty_in_either_model = np.any(sum_weights_matrix == 0.0, axis=0)

    problem_bins = np.flatnonzero(low_statistics | empty_in_either_model)

    if len(problem_bins) == 0:
        return None

    return int(problem_bins[0])


def make_common_adaptive_energy_edges(
    spectra: dict[str, dict],
    initial_energy_edges: np.ndarray,
    minimum_n_eff: float,
) -> np.ndarray:
    """
    Merge neighbouring bins until the common model templates have
    sufficient Monte Carlo statistics.
    The same final binning is used for both hypotheses.
    """
    energy_edges = np.asarray(initial_energy_edges, dtype=float).copy()

    if energy_edges.ndim != 1 or len(energy_edges) < 2 or np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError(
            "initial_energy_edges must be a strictly " "increasing one-dimensional array."
        )

    if minimum_n_eff <= 0.0:
        raise ValueError("minimum_n_eff must be positive.")

    while len(energy_edges) > 2:
        problem_bin = find_first_problem_bin(
            spectra=spectra,
            energy_edges=energy_edges,
            minimum_n_eff=minimum_n_eff,
        )

        if problem_bin is None:
            break

        number_of_bins = len(energy_edges) - 1

        if problem_bin == 0:
            # Merge the first bin with its right neighbour.
            edge_to_remove = 1

        elif problem_bin == (number_of_bins - 1):
            # Merge the final bin with its left neighbour.
            edge_to_remove = len(energy_edges) - 2

        else:
            # Choose the merge that creates the smaller interval in logarithmic energy.
            left_merged_log_width = np.log(
                energy_edges[problem_bin + 1] / energy_edges[problem_bin - 1]
            )

            right_merged_log_width = np.log(
                energy_edges[problem_bin + 2] / energy_edges[problem_bin]
            )

            if left_merged_log_width <= right_merged_log_width:
                # Merge with the bin to the left.
                edge_to_remove = problem_bin
            else:
                # Merge with the bin to the right.
                edge_to_remove = problem_bin + 1

        energy_edges = np.delete(energy_edges, edge_to_remove)

    remaining_problem_bin = find_first_problem_bin(
        spectra=spectra,
        energy_edges=energy_edges,
        minimum_n_eff=minimum_n_eff,
    )

    if remaining_problem_bin is not None:
        raise RuntimeError("Could not construct a statistically reliable " "common binning.")

    return energy_edges


def smoothed_bin_probabilities(
    spectrum: dict,
    energy_edges: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """
    Construct one normalized model-probability template.

    The Jeffreys prior prevents an infinite likelihood ratio when a
    bin happens to be empty in one finite Monte Carlo sample.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    sum_weights, _ = histogram_moments(
        spectrum,
        energy_edges,
    )

    weights = np.asarray(
        spectrum["weights"],
        dtype=float,
    )

    total_weight = float(np.sum(weights))
    total_squared_weight = float(np.sum(weights**2))

    if total_weight <= 0.0 or total_squared_weight <= 0.0:
        raise RuntimeError("Cannot construct a probability template " "from zero total weight.")

    raw_probabilities = sum_weights / np.sum(sum_weights)
    total_n_eff = total_weight**2 / total_squared_weight
    effective_counts = total_n_eff * raw_probabilities

    probabilities = (effective_counts + alpha) / (total_n_eff + alpha * len(raw_probabilities))
    probabilities /= np.sum(probabilities)

    return (probabilities, float(total_n_eff))


def write_probability_templates(
    *,
    mass_gev: float,
    ctau_m: float,
    energy_edges: np.ndarray,
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
) -> Path:
    """Write the model templates used by the likelihood test."""
    template_table = pd.DataFrame(
        {
            "mass_GeV": mass_gev,
            "ctau_m": ctau_m,
            "bin_index": np.arange(len(photon_probabilities)),
            "energy_low_GeV": (energy_edges[:-1]),
            "energy_high_GeV": (energy_edges[1:]),
            "photon_probability": (photon_probabilities),
            "su2_probability": (su2_probabilities),
            "log_su2_over_photon": np.log(su2_probabilities / photon_probabilities),
        }
    )

    output_path = TEMPLATE_DIR / (
        f"probability_templates_ma_{float_token(mass_gev)}_ctau_{float_token(ctau_m)}.csv"
    )

    template_table.to_csv(output_path, index=False,)

    return output_path


def calculate_templates(*, mass_gev: float, ctau_m: float, mass_index: int) -> dict:
    """
    Run EventCalc for the two models and construct statistically
    reliable common probability templates.
    """
    initial_energy_edges = np.geomspace(
        mass_gev,
        ENERGY_MAX_GEV,
        NUMBER_OF_ENERGY_BINS + 1,
    )

    spectra = {}

    for model_index, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
        seed = BASE_SEED + 10_000 * mass_index + 100 * model_index

        spectra[model_name] = calculate_model_spectrum(
            model_name=model_name,
            config=config,
            mass_gev=mass_gev,
            ctau_m=ctau_m,
            energy_edges=initial_energy_edges,
            seed=seed,
        )

    energy_edges = make_common_adaptive_energy_edges(
        spectra=spectra,
        initial_energy_edges=(initial_energy_edges),
        minimum_n_eff=(MINIMUM_BIN_N_EFF),
    )

    photon_probabilities, photon_total_n_eff = smoothed_bin_probabilities(
        spectrum=spectra["ALP-photon-combined"],
        energy_edges=energy_edges,
        alpha=JEFFREYS_ALPHA,
    )

    su2_probabilities, su2_total_n_eff = smoothed_bin_probabilities(
        spectrum=spectra["ALP-SU2L"],
        energy_edges=energy_edges,
        alpha=JEFFREYS_ALPHA,
    )

    return {
        "energy_edges": energy_edges,
        "photon_probabilities": (photon_probabilities),
        "su2_probabilities": (su2_probabilities),
        "photon_n_events": float(spectra["ALP-photon-combined"]["n_events"]),
        "su2_n_events": float(spectra["ALP-SU2L"]["n_events"]),
        "photon_total_n_eff": (photon_total_n_eff),
        "su2_total_n_eff": (su2_total_n_eff),
    }


def simulate_shape_discrimination(
    *,
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
    maximum_events: int,
    number_of_pseudoexperiments: int,
    seed: int,
) -> pd.DataFrame:
    """
    Perform equal-prior shape-only pseudoexperiments.

    The test statistic is

        log[L(SU2) / L(photon)].

    Positive values select ALP-SU2L; negative values select
    ALP-photon-combined.
    """
    if maximum_events < 1:
        raise ValueError("maximum_events must be at least one.")

    if number_of_pseudoexperiments < 1:
        raise ValueError("number_of_pseudoexperiments must be positive.")

    photon_probabilities = np.asarray(photon_probabilities, dtype=float)
    su2_probabilities = np.asarray(su2_probabilities, dtype=float)

    if photon_probabilities.shape != su2_probabilities.shape:
        raise ValueError("The probability templates must have " "identical shapes.")

    # For each bin, if ALP-SU2L is more likely this is positive. If ALP-photon is more likely this is negative
    log_likelihood_ratio_per_bin = np.log(su2_probabilities / photon_probabilities)

    photon_rng = np.random.default_rng(seed)
    su2_rng = np.random.default_rng(seed + 1)

    # Assume ALP-photon is correct and simulate 
    photon_bins = photon_rng.choice(
        len(photon_probabilities), # Number of bins
        size = (number_of_pseudoexperiments, maximum_events),
        p = photon_probabilities, # Probability of choosing from a certain bin
    )

    # Assume ALP-SU2L is correct and simulate 
    su2_bins = su2_rng.choice(
        len(su2_probabilities),
        size = (number_of_pseudoexperiments, maximum_events),
        p = su2_probabilities,
    )

    # Takes log_likelihood_ratio_per_bin for each simulated event and makes the rows cumulated
    photon_llr = np.cumsum(log_likelihood_ratio_per_bin[photon_bins], axis=1) 
    su2_llr = np.cumsum(log_likelihood_ratio_per_bin[su2_bins], axis=1)

    rows = []
    for event_index in range(maximum_events):
        number_of_events = event_index + 1

        photon_values = photon_llr[:, event_index] # Takes the last element which is the total log_likelihood_ratio_per_bin
        su2_values = su2_llr[:, event_index]

        # P (ALP-photon result from simulation | ALP-photon probabilility distribution correct)
        # P (ALP-SU2L result from simulation | ALP-SU2L probabilility distribution correct)
        photon_correct = float(np.mean(photon_values < 0.0) + 0.5 * np.mean(photon_values == 0.0))
        su2_correct = float(np.mean(su2_values > 0.0) + 0.5 * np.mean(su2_values == 0.0))

        rows.append(
            {
                "number_of_events": number_of_events,
                "photon_correct_fraction": photon_correct,
                "su2_correct_fraction": su2_correct,
                "balanced_accuracy": 0.5 * (photon_correct + su2_correct),
                "worst_case_correct_fraction": min(photon_correct, su2_correct),
                "photon_llr_median": float(np.median(photon_values)),
                "su2_llr_median": float(np.median(su2_values)),
            }
        )

    return pd.DataFrame(rows)


def minimum_events_for_accuracy(accuracy_table: pd.DataFrame, target_accuracy: float) -> int | None:
    """
    Find the first event count for which both hypotheses are identified correctly at the requested probability.
    """
    passing = accuracy_table.loc[accuracy_table["worst_case_correct_fraction"] >= target_accuracy]

    if passing.empty:
        return None

    return int(passing.iloc[0]["number_of_events"])


def plot_accuracy(*, accuracy_table: pd.DataFrame, mass_gev: float, ctau_m: float) -> Path:
    """Plot correct-classification probability versus event count."""
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    number_of_events = accuracy_table["number_of_events"]

    axis.plot(
        number_of_events,
        accuracy_table["photon_correct_fraction"],
        label=("Correct if photon is true"),
    )

    axis.plot(
        number_of_events,
        accuracy_table["su2_correct_fraction"],
        label=(r"Correct if $SU(2)_L$ is true"),
    )

    axis.plot(
        number_of_events,
        accuracy_table["worst_case_correct_fraction"],
        linewidth=2.5,
        label="Worst-case accuracy",
    )

    axis.axhline(
        0.90,
        linestyle="--",
        linewidth=1.0,
        label="90% target",
    )

    axis.axhline(
        0.95,
        linestyle="--",
        linewidth=1.0,
        label="95% target",
    )

    axis.axhline(
        0.99,
        linestyle="--",
        linewidth=1.0,
        label="99% target",
    )

    axis.set_xlim(1, MAXIMUM_OBSERVED_EVENTS)
    axis.set_ylim(0.85, 1.01)
    axis.set_xlabel("Number of observed ALP decays")
    axis.set_ylabel("Correct-classification probability")
    axis.set_title(rf"Shape-only discrimination: $m_a={mass_gev:g}$ GeV, $c\tau={ctau_m:g}$ m")
    axis.set_xticks(np.arange(1, MAXIMUM_OBSERVED_EVENTS + 1))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_stem = PLOT_DIR / (
        f"classification_accuracy_ma_{float_token(mass_gev)}_ctau_{float_token(ctau_m)}"
    )

    pdf_path = output_stem.with_suffix(".pdf")

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return pdf_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    common_ranges = load_benchmarks()
    benchmarks = select_stable_benchmarks(common_ranges)

    if benchmarks.empty:
        raise RuntimeError("No stable common benchmarks were selected.")

    all_accuracy_tables = []
    summary_rows = []
    print()
    print("=" * 76)
    print("Shape-only energy-spectrum discrimination")
    print("The expected event rates are reported, " "but are not used in the likelihood ratio.")
    print(
        "Benchmark requirements: "
        f"N_events >= {MINIMUM_EVENTS:g}, "
        f"D_TV <= {MAXIMUM_TV_DISTANCE:g}"
    )
    print("=" * 76)

    for mass_index, row in enumerate(benchmarks.itertuples(index=False)):
        mass_gev = float(row.mass_GeV)

        ctau_m = float(row.ctau_benchmark_m)

        print()
        print("#" * 76)
        print(f"m_a = {mass_gev:g} GeV, c_tau = {ctau_m:.6g} m")
        print("#" * 76)

        templates = calculate_templates(
            mass_gev=mass_gev,
            ctau_m=ctau_m,
            mass_index=mass_index,
        )

        photon_probabilities = templates["photon_probabilities"]
        su2_probabilities = templates["su2_probabilities"]
        energy_edges = templates["energy_edges"]

        template_path = write_probability_templates(
            mass_gev=mass_gev,
            ctau_m=ctau_m,
            energy_edges=energy_edges,
            photon_probabilities=(photon_probabilities),
            su2_probabilities=(su2_probabilities),
        )

        model_tv_distance = 0.5 * float(np.sum(np.abs(photon_probabilities - su2_probabilities)))

        photon_kl_su2 = float(
            np.sum(photon_probabilities * np.log(photon_probabilities / su2_probabilities))
        )

        su2_kl_photon = float(
            np.sum(su2_probabilities * np.log(su2_probabilities / photon_probabilities))
        )

        accuracy_table = simulate_shape_discrimination(
            photon_probabilities=(photon_probabilities),
            su2_probabilities=(su2_probabilities),
            maximum_events=(MAXIMUM_OBSERVED_EVENTS),
            number_of_pseudoexperiments=(NUMBER_OF_PSEUDOEXPERIMENTS),
            seed=(PSEUDOEXPERIMENT_SEED + 10_000 * mass_index),
        )

        accuracy_table.insert(0, "ctau_m", ctau_m)

        accuracy_table.insert(0, "mass_GeV", mass_gev)

        accuracy_path = OUTPUT_DIR / (
            "classification_accuracy"
            f"_ma_{float_token(mass_gev)}"
            f"_ctau_{float_token(ctau_m)}"
            ".csv"
        )

        accuracy_table.to_csv(accuracy_path, index=False)
        all_accuracy_tables.append(accuracy_table)

        pdf_path = plot_accuracy(
            accuracy_table=accuracy_table,
            mass_gev=mass_gev,
            ctau_m=ctau_m,
        )

        threshold_results = {}

        for target_accuracy in TARGET_ACCURACIES:
            threshold_results[target_accuracy] = minimum_events_for_accuracy(
                accuracy_table, 
                target_accuracy,
            )

        summary_row = {
            "mass_GeV": mass_gev,
            "ctau_m": ctau_m,
            "number_of_adaptive_bins": (len(energy_edges) - 1),
            "minimum_bin_N_eff": (MINIMUM_BIN_N_EFF),
            "photon_expected_events": (templates["photon_n_events"]),
            "su2_expected_events": (templates["su2_n_events"]),
            "photon_template_total_N_eff": (templates["photon_total_n_eff"]),
            "su2_template_total_N_eff": (templates["su2_total_n_eff"]),
            "model_total_variation_distance": (model_tv_distance),
            "KL_photon_to_su2_per_event": (photon_kl_su2),
            "KL_su2_to_photon_per_event": (su2_kl_photon),
        }

        for target_accuracy in TARGET_ACCURACIES:
            column_name = (
                f"minimum_events_for_{100 * target_accuracy:.0f}pct_worst_case_accuracy"
            )
            summary_row[column_name] = threshold_results[target_accuracy]

        summary_rows.append(summary_row)

        print(f"Adaptive energy bins: {len(energy_edges) - 1}")
        print(f"Model-to-model total variation distance: {model_tv_distance:.6g}")
        print("Expected event rates " "(not used in classification):")
        print(f"  ALP-photon-combined: {templates['photon_n_events']:.6g}")
        print(f"  ALP-SU2L: {templates['su2_n_events']:.6g}")

        for target_accuracy in TARGET_ACCURACIES:
            minimum_events = threshold_results[target_accuracy]

            if minimum_events is None:
                print(
                    f"  {100 * target_accuracy:.0f}% "
                    "worst-case accuracy: "
                    f"not reached by "
                    f"{MAXIMUM_OBSERVED_EVENTS} events"
                )
            else:
                print(
                    f"  {100 * target_accuracy:.0f}% "
                    "worst-case accuracy: "
                    f"{minimum_events} events"
                )

        print(f"Template table saved to: {template_path}")
        print(f"Accuracy table saved to: {accuracy_path}")
        print(f"Accuracy plots saved to: {pdf_path}")

    combined_accuracy = pd.concat(all_accuracy_tables, ignore_index=True)
    combined_accuracy_path = OUTPUT_DIR / "classification_accuracy_all_masses.csv"
    combined_accuracy.to_csv(combined_accuracy_path, index=False)

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "discrimination_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 76)
    print("Discrimination analysis finished.")
    print(f"Benchmarks analysed: {len(summary)}")
    print(f"Summary saved to: {summary_path}")
    print(f"Combined accuracy table saved to: {combined_accuracy_path}")


if __name__ == "__main__":
    main()