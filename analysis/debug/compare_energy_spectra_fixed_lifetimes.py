from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from analysis.compare_energy_spectra import (
        BASE_SEED,
        MODEL_CONFIGS,
        N_EFF_WARNING_THRESHOLD,
        NUMBER_OF_ENERGY_BINS,
        calculate_model_spectrum,
        numerical_summary,
    )
except ModuleNotFoundError:
    # Also allow:
    # python analysis/compare_energy_spectra_lifetime_scan.py
    from compare_energy_spectra import (
        BASE_SEED,
        MODEL_CONFIGS,
        N_EFF_WARNING_THRESHOLD,
        NUMBER_OF_ENERGY_BINS,
        calculate_model_spectrum,
        numerical_summary,
    )


ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "energy_spectra_lifetime_scan"
PLOT_DIR = OUTPUT_DIR / "plots" / "debug"
CTAU_PATH = ANALYSIS_DIR / "ctau_scan" / "debug" / "common_ctau_ranges.csv"


MASSES_GEV = [0.3, 1.0]
FIXED_CTAU_VALUES_M = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
REFERENCE_CTAU_M = 1000.0
ENERGY_MAX_GEV = 400.0


def load_mass_ranges() -> pd.DataFrame:
    """
    Load masses for which both models have a finite common
    N_events >= 10 lifetime interval.
    """
    data = pd.read_csv(CTAU_PATH)

    required_columns = {
        "mass_GeV",
        "ctau_lower_m",
        "ctau_upper_m",
        "upper_extends_beyond_scan",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in common lifetime table: {sorted(missing_columns)}")

    numeric_columns = [
        "mass_GeV",
        "ctau_lower_m",
        "ctau_upper_m",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not np.all(np.isfinite(data[numeric_columns].to_numpy(dtype=float))):
        raise ValueError("The common lifetime table contains " "non-finite numerical values.")

    flag_text = data["upper_extends_beyond_scan"].astype(str).str.strip().str.lower()

    if not flag_text.isin(["true", "false"]).all():
        raise ValueError("upper_extends_beyond_scan must contain " "only True or False.")

    data["upper_extends_beyond_scan"] = flag_text == "true"

    if data["upper_extends_beyond_scan"].any():
        raise RuntimeError("At least one common lifetime interval has " "no finite upper boundary.")

    if np.any(data["ctau_upper_m"] <= data["ctau_lower_m"]):
        raise ValueError("Every upper lifetime must exceed " "the corresponding lower lifetime.")

    return data.sort_values("mass_GeV").reset_index(drop=True)


def build_ctau_values() -> tuple[float, ...]:
    """
    Construct a logarithmic lifetime grid inside the common
    event-rate interval and add one long-lifetime reference.
    """
    scan_values = FIXED_CTAU_VALUES_M

    all_values = np.concatenate([scan_values, [REFERENCE_CTAU_M]])

    # Rounding prevents effectively identical floating-point
    # values from appearing twice.
    unique_values = np.unique(np.round(all_values, decimals=12))

    return tuple(float(value) for value in unique_values)


def distribution_distances(
    spectrum: dict,
    reference: dict,
) -> tuple[float, float]:
    """
    Compare two normalized spectra on their common histogram grid.

    Returns
    -------
    total_variation_distance:
        0.5 * sum_i |p_i - q_i|.

    maximum_cdf_distance:
        Maximum absolute difference between the two binned CDFs.

    Both quantities are zero for identical distributions and increase
    when their shapes become different. They are descriptive measures,
    not yet the final model-discrimination test.
    """
    energy_edges = np.asarray(spectrum["energy_edges"], dtype=float)

    reference_energy_edges = np.asarray(reference["energy_edges"], dtype=float)

    if not np.array_equal(energy_edges, reference_energy_edges):
        raise ValueError("The spectrum and reference spectrum must use " "identical energy-bin edges.")

    weights = np.asarray(spectrum["sum_weights_per_bin"], dtype=float)
    reference_weights = np.asarray(reference["sum_weights_per_bin"], dtype=float)
    total_weight = float(np.sum(weights))
    reference_total_weight = float(np.sum(reference_weights))

    if total_weight <= 0.0 or reference_total_weight <= 0.0:
        raise RuntimeError("Cannot compare spectra with zero total weight.")

    probabilities = weights / total_weight
    reference_probabilities = reference_weights / reference_total_weight
    total_variation_distance = 0.5 * float(np.sum(np.abs(probabilities - reference_probabilities)))
    maximum_cdf_distance = float(
        np.max(np.abs(np.cumsum(probabilities) - np.cumsum(reference_probabilities)))
    )

    return (total_variation_distance, maximum_cdf_distance)


def plot_model_scan(
    model_name: str,
    mass_gev: float,
    ctau_values_m: tuple[float, ...],
    spectra_by_ctau: dict[float, dict],
) -> Path:
    """
    Plot the normalized energy distributions at all lifetimes
    for one ALP model.
    """
    figure, axis = plt.subplots(
        figsize=(8.5, 6.0),
    )

    for ctau_m in ctau_values_m:
        spectrum = spectra_by_ctau[ctau_m]

        density = np.asarray(spectrum["density"], dtype=float)
        density_error = np.asarray(spectrum["density_error"], dtype=float)
        energy_edges = np.asarray(spectrum["energy_edges"], dtype=float)
        energy_centres = np.asarray(spectrum["energy_centres"], dtype=float)
        sum_weights = np.asarray(spectrum["sum_weights_per_bin"], dtype=float) 
        effective_samples = np.asarray(spectrum["effective_samples_per_bin"], dtype=float)

        low_statistics_mask = (sum_weights > 0.0) & (effective_samples < N_EFF_WARNING_THRESHOLD)

        # Do not connect poorly resolved non-empty bins to the
        # statistically reliable part of the curve.
        reliable_density = density.copy()
        reliable_density[low_statistics_mask] = np.nan

        is_reference = np.isclose(ctau_m, REFERENCE_CTAU_M)

        label = rf"$c\tau={ctau_m:g}\,\mathrm{{m}}$"

        if is_reference:
            label += " (reference)"

        stairs = axis.stairs(
            reliable_density,
            energy_edges,
            label=label,
            linewidth=(3.0 if is_reference else 1.6),
            zorder=(10 if is_reference else 2),
        )

        # Show low-statistics bins as separate points with
        # Monte Carlo error bars.
        if np.any(low_statistics_mask):
            axis.errorbar(
                energy_centres[low_statistics_mask],
                density[low_statistics_mask],
                yerr=density_error[low_statistics_mask],
                fmt="x",
                color=stairs.get_edgecolor(),
                markersize=5,
                capsize=2,
                linestyle="none",
            )

    model_label = MODEL_CONFIGS[model_name]["plot_label"]

    axis.set_xscale("log")

    axis.set_xlim(
        mass_gev,
        ENERGY_MAX_GEV,
    )

    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(r"$E_a$ [GeV]")
    axis.set_ylabel(r"$(1/N_{\rm events})\," r"dN_{\rm events}/dE_a$ " r"[GeV$^{-1}$]")
    axis.set_title(rf"{model_label}: lifetime dependence, $m_a={mass_gev:g}$ GeV")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(title="Lifetime")
    figure.tight_layout()

    model_string = model_name.lower().replace("-", "_")
    mass_string = f"{mass_gev:g}".replace(".", "p")
    output_stem = PLOT_DIR / f"lifetime_dependence_{model_string}_ma_{mass_string}"
    pdf_path = output_stem.with_suffix(".pdf")

    figure.savefig(pdf_path,bbox_inches="tight",)
    plt.close(figure)

    return pdf_path


def make_summary(
    mass_gev: float,
    ctau_values_m: tuple[float, ...],
    all_spectra: dict[str, dict[float, dict]],
) -> pd.DataFrame:
    """
    Build one summary row for every model and lifetime.
    """
    rows = []
    for (model_name, spectra_by_ctau) in all_spectra.items():
        reference_spectrum = spectra_by_ctau[REFERENCE_CTAU_M]

        for ctau_m in ctau_values_m:
            spectrum = spectra_by_ctau[ctau_m]

            # Reuse the statistics already defined in
            # compare_energy_spectra.py.
            summary_row = (
                numerical_summary(
                    {model_name: spectrum}, mass_gev = mass_gev, ctau_m = ctau_m,
                ).iloc[0].to_dict()
            )

            total_variation_distance, maximum_cdf_distance = distribution_distances(
                spectrum, reference_spectrum,
            )

            weighted_mean_energy = float(summary_row["weighted_mean_energy_GeV"])
            weighted_mean_gamma = weighted_mean_energy / mass_gev

            nonempty_low_statistics_bins = int(
                np.count_nonzero(
                    (np.asarray(spectrum["sum_weights_per_bin"], dtype=float) > 0.0)
                    & (np.asarray(spectrum["effective_samples_per_bin"], dtype=float)
                        < N_EFF_WARNING_THRESHOLD
                    )
                )
            )

            summary_row.update(
                {
                    "reference_ctau_m": (REFERENCE_CTAU_M),
                    "is_reference": bool(np.isclose(ctau_m, REFERENCE_CTAU_M)),
                    "weighted_mean_gamma": (weighted_mean_gamma),
                    "ctau_times_weighted_mean_gamma_m": (ctau_m * weighted_mean_gamma),
                    "N_events_ge_10": bool(spectrum["n_events"] >= 10.0),
                    "total_variation_distance_to_reference": (total_variation_distance),
                    "binned_cdf_max_distance_to_reference": (maximum_cdf_distance),
                    "nonempty_bins_with_low_N_eff": (nonempty_low_statistics_bins),
                }
            )

            rows.append(summary_row)

    return pd.DataFrame(rows).sort_values(["model", "ctau_m"], ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    plot_paths = []

    for mass_index, mass_gev in enumerate(MASSES_GEV):
        ctau_values_m = build_ctau_values()

        if not any(
            np.isclose(ctau_m, REFERENCE_CTAU_M)
            for ctau_m in ctau_values_m
        ):
            raise RuntimeError("The reference lifetime is missing from " "the lifetime scan.")

        print()
        print("#" * 76)
        print(f"Mass: {mass_gev:g} GeV")
        print("Sampled lifetimes: " + ", ".join(f"{value:.6g}" for value in ctau_values_m) + " m")
        print("#" * 76)

        energy_edges = np.geomspace(
            mass_gev,
            ENERGY_MAX_GEV,
            NUMBER_OF_ENERGY_BINS + 1,
        )

        all_spectra = {}

        for model_index, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
            print()
            print("=" * 72)
            print(f"Lifetime scan for {model_name}")
            print(f"m_a = {mass_gev:g} GeV")
            print("=" * 72)

            # Same samples at every lifetime for one model and
            # mass, but independent samples between masses.
            model_seed = (
                BASE_SEED
                + 1000 * mass_index
                + 100 * model_index
            )

            spectra_by_ctau = {}

            for ctau_m in ctau_values_m:
                spectra_by_ctau[ctau_m] = calculate_model_spectrum(
                    model_name=model_name,
                    config=config,
                    mass_gev=mass_gev,
                    ctau_m=ctau_m,
                    energy_edges=energy_edges,
                    seed=model_seed,
                )

            all_spectra[model_name] = spectra_by_ctau

            pdf_path = plot_model_scan(
                model_name=model_name,
                mass_gev=mass_gev,
                ctau_values_m=ctau_values_m,
                spectra_by_ctau=spectra_by_ctau,
            )

            plot_paths.append(pdf_path)

        mass_summary = make_summary(
            mass_gev=mass_gev,
            ctau_values_m=ctau_values_m,
            all_spectra=all_spectra,
        )

        summary_frames.append(mass_summary)

    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = OUTPUT_DIR / "lifetime_stability_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 76)
    print("Lifetime-stability scan finished.")
    print(f"Masses analysed: {summary['mass_GeV'].nunique()}")
    print(f"Summary saved to: {summary_path}")
    print(f"Plot saved to: {pdf_path}")


if __name__ == "__main__":
    main()
