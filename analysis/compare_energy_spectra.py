from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#
# parents[0] = analysis/
# parents[1] = EventCalc-SHiP/
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


os.chdir(REPO_ROOT)

from funcs.initLLP import LLP
from funcs.kinematics import Grids
from funcs.ship_setup import theta_max_dec_vol


# Configuration

N_EFF_WARNING_THRESHOLD = 20.0

N_POT = 6.0e20

MASS_GEV = 0.3
CTAU_M = 100

RESAMPLE_SIZE = 1_000_000
N_INTERPOLATION_POINTS = 10 * RESAMPLE_SIZE

NUMBER_OF_ENERGY_BINS = 50

BASE_SEED = 54321
ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "energy_spectra"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_CONFIGS = {
    "ALP-photon-primary": {
        "plot_label": "ALP-photon, primary",
        "particle_selection": {
            "LLP_name": "ALP-photon",
            "particle_path": str(
                REPO_ROOT / "Distributions" / "ALP-photon"
            ),
        },
        "alp_production_mode": "primary",
    },
    "ALP-SU2L": {
        "plot_label": r"ALP-$SU(2)_L$",
        "particle_selection": {
            "LLP_name": "ALP-SU2L",
            "particle_path": str(
                REPO_ROOT / "Distributions" / "ALP-SU2L"
            ),
        },
        "alp_production_mode": None,
    },
}


def normalized_event_energy_spectrum(
    mother_particle_results: np.ndarray,
    energy_edges: np.ndarray,
) -> dict:
    """
        (1 / N_events) dN_events / dE_a
    using the decay probability P_decay as event weight.

    EventCalc columns used:
       3: LLP energy E_a.
       6: Decay probability P_decay.
    """
    results = np.asarray(
        mother_particle_results,
        dtype=float,
    )

    if results.ndim != 2 or results.shape[1] <= 6:
        raise ValueError(
            "mother_particle_results must be a 2D array "
            "with at least seven columns."
        )

    if len(results) == 0:
        raise RuntimeError(
            "No accepted mother-particle samples were generated."
        )

    energies = results[:, 3]
    weights = results[:, 6]

    # Remove any non-finite entries.
    valid = (
        np.isfinite(energies)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )

    energies = energies[valid]
    weights = weights[valid]

    if len(energies) == 0:
        raise RuntimeError(
            "No valid energy-weight pairs remain."
        )

    total_weight_before_histogram = float(
        np.sum(weights)
    )

    if total_weight_before_histogram <= 0.0:
        raise RuntimeError(
            "The total decay-probability weight is zero."
        )

    in_histogram_range = (
        (energies >= energy_edges[0])
        & (energies <= energy_edges[-1])
    )

    histogram_energies = energies[in_histogram_range]
    histogram_weights = weights[in_histogram_range]

    if len(histogram_energies) == 0:
        raise RuntimeError(
            "No events lie inside the requested energy range."
        )

    sum_weights, _ = np.histogram(
        histogram_energies,
        bins=energy_edges,
        weights=histogram_weights,
    )

    sum_squared_weights, _ = np.histogram(
        histogram_energies,
        bins=energy_edges,
        weights=histogram_weights**2,
    )

    bin_widths = np.diff(energy_edges)
    total_histogram_weight = float(
        np.sum(histogram_weights)
    )

    if total_histogram_weight <= 0.0:
        raise RuntimeError(
            "The energy histogram has zero total weight."
        )

    density = sum_weights / (
        total_histogram_weight * bin_widths
    )

    density_error = np.sqrt(sum_squared_weights) / (
        total_histogram_weight * bin_widths
    )

    effective_samples_per_bin = np.divide(
        sum_weights**2,
        sum_squared_weights,
        out=np.zeros_like(sum_weights),
        where=sum_squared_weights > 0.0,
    )

    energy_centres = np.sqrt(
        energy_edges[:-1] * energy_edges[1:]
    )

    normalization = float(
        np.sum(density * bin_widths)
    )

    if not np.isclose(
        normalization,
        1.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Spectrum normalization failed: "
            f"integral = {normalization}"
        )

    range_coverage = (
        total_histogram_weight
        / total_weight_before_histogram
    )

    if not np.isclose(
        range_coverage,
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "The energy range does not contain the full "
            f"weighted sample: coverage={range_coverage:.12g}"
        )

    return {
        "energy_edges": energy_edges,
        "energy_centres": energy_centres,
        "density": density,
        "bin_widths": bin_widths,
        "normalization": normalization,
        "range_coverage": range_coverage,
        "number_of_samples": len(histogram_energies),

        # Event-level information.
        "energies": histogram_energies,
        "weights": histogram_weights,
        "total_weight": total_histogram_weight,

        # Histogram-level diagnostic information.
        "sum_weights_per_bin": sum_weights,
        "sum_squared_weights_per_bin": (
            sum_squared_weights
        ),
        "density_error": density_error,
        "effective_samples_per_bin": (
            effective_samples_per_bin
        ),
    }


def calculate_model_spectrum(
    model_name: str,
    config: dict,
    mass_gev: float,
    ctau_m: float,
    energy_edges: np.ndarray,
    seed: int,
):
    """
    Generate accepted EventCalc samples and construct the normalized
    event-energy spectrum for one model.
    """
    print()
    print(f"Processing {model_name}")
    print(f"m_a   = {mass_gev} GeV")
    print(f"c_tau = {ctau_m} m")

    llp = LLP(
        mass=None,
        particle_selection=config["particle_selection"],
        mixing_pattern=None,
        uncertainty=None,
        alp_production_mode=config["alp_production_mode"],
    )

    llp.set_mass(mass_gev)
    llp.compute_mass_dependent_properties()
    llp.set_c_tau(ctau_m)
    np.random.seed(seed)

    kin = Grids(
        llp.Distr,
        llp.Energy_distr,
        N_INTERPOLATION_POINTS,
        llp.mass,
        ctau_m,
        theta_max_sim=theta_max_dec_vol,
    )

    kin.interpolate(False)
    kin.resample(RESAMPLE_SIZE, False)

    # Second seed: true decay positions and geometric acceptance.
    np.random.seed(seed + 1)
    kin.true_samples(False)
    mother_particle_results = kin.get_kinematics()

    print(
        "Sampling seeds: "
        f"{seed} for resampling, "
        f"{seed + 1} for true samples"
    )

    spectrum = normalized_event_energy_spectrum(
        mother_particle_results=mother_particle_results,
        energy_edges=energy_edges,
    )

    br_visible = float(
        np.sum(llp.BrRatios_distr)
    )

    coupling_squared = float(
        llp.c_tau_int / ctau_m
    )

    n_llp_total = (
        N_POT
        * float(llp.Yield)
        * coupling_squared
    )

    weights = spectrum["weights"]

    n_events_from_spectrum = (
        n_llp_total
        * float(kin.epsilon_polar)
        * np.sum(weights) / RESAMPLE_SIZE
        * br_visible
    )

    # Equivalent factorized form used in the lifetime scan.
    epsilon_azimuthal = (
        len(weights) / RESAMPLE_SIZE
    )

    mean_decay_probability = float(
        np.mean(weights)
    )

    n_events_factorized = (
        n_llp_total
        * float(kin.epsilon_polar)
        * epsilon_azimuthal
        * mean_decay_probability
        * br_visible
    )

    if not np.isclose(
        n_events_from_spectrum,
        n_events_factorized,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise RuntimeError(
            "The two event-rate calculations disagree."
        )

    spectrum.update(
        {
            "seed": seed,
            "coupling_squared": coupling_squared,
            "n_llp_total": n_llp_total,
            "epsilon_polar": float(
                kin.epsilon_polar
            ),
            "epsilon_azimuthal": epsilon_azimuthal,
            "mean_decay_probability": (
                mean_decay_probability
            ),
            "br_visible": br_visible,
            "n_events": n_events_from_spectrum,
        }
    )

    print(
        "Accepted samples: "
        f"{spectrum['number_of_samples']}"
    )
    print(
        "Histogram range coverage: "
        f"{spectrum['range_coverage']:.8f}"
    )
    print(
        "Normalization integral: "
        f"{spectrum['normalization']:.12f}"
    )
    print(
        "Estimated event rate: "
        f"{n_events_from_spectrum:.6g}"
    )

    nonempty_low_statistics_bins = (
        (spectrum["density"] > 0.0)
        & (spectrum["effective_samples_per_bin"] < 20.0)
    )

    print(
        "Non-empty energy bins with N_eff < 20: "
        f"{np.count_nonzero(nonempty_low_statistics_bins)}"
    )

    return spectrum


def plot_spectra(
    spectra: dict[str, dict],
    *,
    mass_gev: float,
    ctau_m: float,
    output_dir: Path,
) -> Path:
    """
    Plot the normalized event-energy spectra.

    Bins with N_eff below the warning threshold are not connected
    to the well-resolved spectrum. They are shown separately with
    Monte Carlo error bars.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 6.0),
    )

    low_statistics_label_used = False

    for model_name, spectrum in spectra.items():
        plot_label = MODEL_CONFIGS[model_name]["plot_label"]
        density = np.asarray(spectrum["density"], dtype=float)
        density_error = np.asarray(spectrum["density_error"], dtype=float)
        energy_edges = np.asarray(spectrum["energy_edges"], dtype=float)
        energy_centres = np.asarray(spectrum["energy_centres"], dtype=float)
        sum_weights = np.asarray(spectrum["sum_weights_per_bin"], dtype=float)
        n_eff = np.asarray(spectrum["effective_samples_per_bin"], dtype=float)

        low_statistics_mask = (
            (sum_weights > 0.0)
            & (
                n_eff
                < N_EFF_WARNING_THRESHOLD
            )
        )

        # Keep zero-density bins, but break the curve across
        # non-empty bins whose statistical precision is inadequate.
        reliable_density = density.copy()
        reliable_density[
            low_statistics_mask
        ] = np.nan

        stairs = axis.stairs(
            reliable_density,
            energy_edges,
            label=plot_label,
            linewidth=2.0,
        )

        if np.any(low_statistics_mask):
            low_statistics_label = None

            if not low_statistics_label_used:
                low_statistics_label = (
                    rf"Bins with "
                    rf"$\left(\sum_i w_i \right)^{2} / \sum_i w_i^{2}"
                    rf"<"
                    rf"{N_EFF_WARNING_THRESHOLD:g}$"
                )

                low_statistics_label_used = True

            axis.errorbar(
                energy_centres[
                    low_statistics_mask
                ],
                density[
                    low_statistics_mask
                ],
                yerr=density_error[
                    low_statistics_mask
                ],
                fmt="x",
                color=stairs.get_edgecolor(),
                markersize=7,
                capsize=3,
                linestyle="none",
                label=low_statistics_label,
            )

    axis.set_xscale("log")
    axis.set_xlabel(
        r"$E_a$ [GeV]"
    )
    axis.set_ylabel(
        r"$(1/N_{\rm events})\,"
        r"dN_{\rm events}/dE_a$ "
        r"[GeV$^{-1}$]"
    )
    axis.set_title(
        rf"$m_a={mass_gev:g}$ GeV, "
        rf"$c\tau={ctau_m:g}$ m"
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )
    axis.legend()
    figure.tight_layout()

    mass_string = str(
        mass_gev
    ).replace(".", "p")

    ctau_string = str(
        ctau_m
    ).replace(".", "p")

    output_path = (
        output_dir
        / (
            "normalized_event_energy_spectra"
            f"_ma_{mass_string}"
            f"_ctau_{ctau_string}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """
    Calculate weighted quantiles.

    Parameters
    ----------
    values:
        Values whose quantiles are requested.

    weights:
        Non-negative weights associated with the values.

    quantiles:
        Quantiles between zero and one.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError(
            "values and weights must be one-dimensional."
        )

    if len(values) != len(weights):
        raise ValueError(
            "values and weights must have the same length."
        )

    if np.any(weights < 0.0):
        raise ValueError(
            "weights must be non-negative."
        )

    if np.any(
        (quantiles < 0.0)
        | (quantiles > 1.0)
    ):
        raise ValueError(
            "quantiles must lie between zero and one."
        )

    positive = weights > 0.0
    values = values[positive]
    weights = weights[positive]

    if len(values) == 0:
        raise RuntimeError(
            "All statistical weights are zero."
        )

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]

    total_weight = float(
        np.sum(sorted_weights)
    )

    # Weighted CDF evaluated at the centre of each weight.
    cumulative_probability = (
        np.cumsum(sorted_weights)
        - 0.5 * sorted_weights
    ) / total_weight

    return np.interp(
        quantiles,
        cumulative_probability,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def numerical_summary(
    spectra: dict[str, dict],
    *,
    mass_gev: float,
    ctau_m: float,
) -> pd.DataFrame:
    """Write weighted spectrum statistics to a CSV file."""
    summary_rows = []

    for model_name, spectrum in spectra.items():
        energies = np.asarray(
            spectrum["energies"],
            dtype=float,
        )
        weights = np.asarray(
            spectrum["weights"],
            dtype=float,
        )

        total_weight = float(
            np.sum(weights)
        )

        if total_weight <= 0.0:
            raise RuntimeError(
                f"{model_name}: total weight is zero."
            )

        weighted_mean = float(
            np.average(
                energies,
                weights=weights,
            )
        )

        q16, weighted_median, q84 = (
            weighted_quantiles(
                values=energies,
                weights=weights,
                quantiles=np.array(
                    [0.16, 0.50, 0.84]
                ),
            )
        )

        fraction_below_1_gev = float(
            np.sum(
                weights[energies < 1.0]
            )
            / total_weight
        )

        sum_squared_weights = float(
            np.sum(weights**2)
        )

        effective_weighted_sample_size = (
            total_weight**2
            / sum_squared_weights
        )

        summary_rows.append(
            {
                "model": model_name,
                "plot_label": (
                    MODEL_CONFIGS[model_name][
                        "plot_label"
                    ]
                ),
                "mass_GeV": mass_gev,
                "ctau_m": ctau_m,
                "weighted_mean_energy_GeV": (
                    weighted_mean
                ),
                "weighted_median_energy_GeV": (
                    float(weighted_median)
                ),
                "energy_q16_GeV": float(q16),
                "energy_q84_GeV": float(q84),
                "fraction_below_1_GeV": (
                    fraction_below_1_gev
                ),
                "effective_sample_size": (
                    effective_weighted_sample_size
                ),
                "accepted_samples": spectrum[
                    "number_of_samples"
                ],
                "normalization": spectrum[
                    "normalization"
                ],
                "range_coverage": spectrum[
                    "range_coverage"
                ],
                "N_LLP_total": spectrum[
                    "n_llp_total"
                ],
                "epsilon_polar": spectrum[
                    "epsilon_polar"
                ],
                "epsilon_azimuthal": spectrum[
                    "epsilon_azimuthal"
                ],
                "mean_P_decay": spectrum[
                    "mean_decay_probability"
                ],
                "visible_Br": spectrum[
                    "br_visible"
                ],
                "N_events": spectrum[
                    "n_events"
                ],
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary



def main() -> None:
    # Begin with a relatively fine common grid.
    energy_edges = np.geomspace(
        MASS_GEV,
        400.0,
        NUMBER_OF_ENERGY_BINS + 1,
    )

    spectra = {}

    for model_index, (
        model_name,
        config,
    ) in enumerate(MODEL_CONFIGS.items()):
        spectra[model_name] = calculate_model_spectrum(
            model_name=model_name,
            config=config,
            mass_gev=MASS_GEV,
            ctau_m=CTAU_M,
            energy_edges=energy_edges,
            seed=BASE_SEED + 100 * model_index,
        )

    print()
    print(
        "Number of  energy bins: "
        f"{len(energy_edges) - 1}"
    )
    print()
    plot_path = plot_spectra(
        spectra,
        mass_gev=MASS_GEV,
        ctau_m=CTAU_M,
        output_dir=OUTPUT_DIR,
    )

    summary = numerical_summary(
        spectra,
        mass_gev=MASS_GEV,
        ctau_m=CTAU_M,
    )

    summary_path = (
        OUTPUT_DIR
        / "energy_spectra_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )
    print()
    print("Numerical summary")

    for model_name, spectrum in spectra.items():
        print(
            f"  {model_name}: "
            f"N_events = "
            f"{spectrum['n_events']:.6g}"
        )

    print()
    print("Finished.")
    print(f"Plot saved to: {plot_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()