from __future__ import annotations

"""
Quick diagnostic of the daughter-level ECAL acceptance.

Run from the repository root with

    python -m analysis.diagnose_ecal_acceptance

The script evaluates three sources at one benchmark point:

    * ALP-photon primary
    * ALP-photon cascade
    * ALP-SU(2)_L inclusive

For each source it

    1. generates an EventCalc mother-particle sample,
    2. samples a -> gamma gamma and projects both photons to the ECAL plane,
    3. checks four-momentum conservation and photon masslessness,
    4. prints unweighted and P_decay-weighted ECAL acceptances,
    5. saves one hit-position plot and one energy-bias plot.

The default ECAL geometry is the simplified 4 m x 6 m plane at z = 95 m
used in the current analysis.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

from funcs.initLLP import LLP
from funcs.kinematics import Grids
from funcs.ship_setup import theta_max_dec_vol
from analysis.ECAL import (
    DEFAULT_ECAL,
    DiphotonECALResult,
    diphoton_ecal_acceptance,
    weighted_ecal_acceptance,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MASS_GEV = 0.3
CTAU_M = 100.0

# This is intentionally smaller than the production analysis sample.
# Increase it if the diagnostic plots look statistically sparse.
RESAMPLE_SIZE = 250_000
N_INTERPOLATION_POINTS = 10 * RESAMPLE_SIZE

MAX_SCATTER_POINTS_PER_CATEGORY = 35_000
NUMBER_OF_ENERGY_BINS = 60

BASE_SEED = 64321

OUTPUT_DIR = Path(__file__).resolve().parent / "ecal_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SOURCE_CONFIGS = {
    "ALP-photon-primary": {
        "plot_label": "ALP-photon primary",
        "particle_selection": {
            "LLP_name": "ALP-photon",
            "particle_path": str(REPO_ROOT / "Distributions" / "ALP-photon"),
        },
        "alp_production_mode": "primary",
        "seed_offset": 0,
    },
    "ALP-photon-cascade": {
        "plot_label": "ALP-photon cascade",
        "particle_selection": {
            "LLP_name": "ALP-photon",
            "particle_path": str(REPO_ROOT / "Distributions" / "ALP-photon"),
        },
        "alp_production_mode": "cascade",
        "seed_offset": 10,
    },
    "ALP-SU2L": {
        "plot_label": r"ALP-$SU(2)_L$",
        "particle_selection": {
            "LLP_name": "ALP-SU2L",
            "particle_path": str(REPO_ROOT / "Distributions" / "ALP-SU2L"),
        },
        "alp_production_mode": None,
        "seed_offset": 100,
    },
}


def generate_mother_sample(
    *,
    config: dict,
    seed: int,
) -> np.ndarray:
    """Generate one EventCalc mother-particle sample."""

    llp = LLP(
        mass=None,
        particle_selection=config["particle_selection"],
        mixing_pattern=None,
        uncertainty=None,
        alp_production_mode=config["alp_production_mode"],
    )

    llp.set_mass(MASS_GEV)
    llp.compute_mass_dependent_properties()
    llp.set_c_tau(CTAU_M)

    np.random.seed(seed)

    kin = Grids(
        llp.Distr,
        llp.Energy_distr,
        N_INTERPOLATION_POINTS,
        llp.mass,
        CTAU_M,
        theta_max_sim=theta_max_dec_vol,
    )

    kin.interpolate(False)
    kin.resample(RESAMPLE_SIZE, False)

    np.random.seed(seed + 1)
    kin.true_samples(False)

    results = np.asarray(kin.get_kinematics(), dtype=float)

    if results.ndim != 2 or results.shape[1] < 10:
        raise RuntimeError(
            "EventCalc returned an invalid mother-particle array. "
            "At least ten columns are required."
        )

    valid = (
        np.all(np.isfinite(results[:, :10]), axis=1)
        & (results[:, 3] > 0.0)
        & (results[:, 4] > 0.0)
        & (results[:, 6] >= 0.0)
    )
    results = results[valid]

    if len(results) == 0:
        raise RuntimeError("No valid EventCalc mother particles were generated.")

    return results


def sample_indices(
    mask: np.ndarray,
    *,
    maximum_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return all matching indices or a reproducible random subset."""

    indices = np.flatnonzero(mask)

    if len(indices) <= maximum_size:
        return indices

    return np.sort(
        rng.choice(
            indices,
            size=maximum_size,
            replace=False,
        )
    )


def validate_kinematics(
    mother_results: np.ndarray,
    ecal_result: DiphotonECALResult,
) -> tuple[float, float]:
    """Validate the generated two-photon four-momenta."""

    daughter_sum = (
        ecal_result.photon_1_four_momentum
        + ecal_result.photon_2_four_momentum
    )
    mother_four_momentum = mother_results[:, :4]

    maximum_conservation_residual = float(
        np.max(np.abs(daughter_sum - mother_four_momentum))
    )

    photon_1 = ecal_result.photon_1_four_momentum
    photon_2 = ecal_result.photon_2_four_momentum

    photon_1_mass_squared = (
        photon_1[:, 3] ** 2
        - np.sum(photon_1[:, :3] ** 2, axis=1)
    )
    photon_2_mass_squared = (
        photon_2[:, 3] ** 2
        - np.sum(photon_2[:, :3] ** 2, axis=1)
    )

    maximum_photon_mass_squared = float(
        max(
            np.max(np.abs(photon_1_mass_squared)),
            np.max(np.abs(photon_2_mass_squared)),
        )
    )

    if maximum_conservation_residual > 1.0e-8:
        raise RuntimeError(
            "Diphoton four-momentum conservation failed: "
            f"maximum residual = {maximum_conservation_residual:.6g}."
        )

    if maximum_photon_mass_squared > 1.0e-8:
        raise RuntimeError(
            "A generated photon is not numerically massless: "
            f"maximum |E^2-p^2| = {maximum_photon_mass_squared:.6g} GeV^2."
        )

    return maximum_conservation_residual, maximum_photon_mass_squared


def validate_hit_mask(
    ecal_result: DiphotonECALResult,
) -> None:
    """Check that every accepted event has both photons inside the rectangle."""

    accepted = ecal_result.event_mask

    if not np.array_equal(
        accepted,
        ecal_result.photon_1_hit_mask & ecal_result.photon_2_hit_mask,
    ):
        raise RuntimeError(
            "The event-level ECAL mask is inconsistent with the photon masks."
        )

    if not np.any(accepted):
        raise RuntimeError("No events pass the ECAL requirement.")

    x_min = DEFAULT_ECAL.centre_x_m - DEFAULT_ECAL.half_width_x_m
    x_max = DEFAULT_ECAL.centre_x_m + DEFAULT_ECAL.half_width_x_m
    y_min = DEFAULT_ECAL.centre_y_m - DEFAULT_ECAL.half_height_y_m
    y_max = DEFAULT_ECAL.centre_y_m + DEFAULT_ECAL.half_height_y_m

    accepted_coordinates = (
        ecal_result.photon_1_x_ecal_m[accepted],
        ecal_result.photon_1_y_ecal_m[accepted],
        ecal_result.photon_2_x_ecal_m[accepted],
        ecal_result.photon_2_y_ecal_m[accepted],
    )

    x1, y1, x2, y2 = accepted_coordinates

    inside = (
        (x1 >= x_min)
        & (x1 <= x_max)
        & (x2 >= x_min)
        & (x2 <= x_max)
        & (y1 >= y_min)
        & (y1 <= y_max)
        & (y2 >= y_min)
        & (y2 <= y_max)
    )

    if not np.all(inside):
        raise RuntimeError(
            "At least one accepted event has a photon outside the ECAL rectangle."
        )


def plot_hit_positions(
    *,
    source_name: str,
    plot_label: str,
    ecal_result: DiphotonECALResult,
    seed: int,
) -> Path:
    """Plot projected photon positions together with the ECAL rectangle."""

    rng = np.random.default_rng(seed)

    photon_1_finite = (
        np.isfinite(ecal_result.photon_1_x_ecal_m)
        & np.isfinite(ecal_result.photon_1_y_ecal_m)
    )
    photon_2_finite = (
        np.isfinite(ecal_result.photon_2_x_ecal_m)
        & np.isfinite(ecal_result.photon_2_y_ecal_m)
    )

    accepted = ecal_result.event_mask

    photon_1_all_indices = sample_indices(
        photon_1_finite,
        maximum_size=MAX_SCATTER_POINTS_PER_CATEGORY,
        rng=rng,
    )
    photon_2_all_indices = sample_indices(
        photon_2_finite,
        maximum_size=MAX_SCATTER_POINTS_PER_CATEGORY,
        rng=rng,
    )
    accepted_indices = sample_indices(
        accepted,
        maximum_size=MAX_SCATTER_POINTS_PER_CATEGORY,
        rng=rng,
    )

    figure, axis = plt.subplots(figsize=(7.5, 7.0))

    axis.scatter(
        ecal_result.photon_1_x_ecal_m[photon_1_all_indices],
        ecal_result.photon_1_y_ecal_m[photon_1_all_indices],
        s=4,
        alpha=0.12,
        label=r"Photon 1 reaching $z_{\rm ECAL}$",
    )
    axis.scatter(
        ecal_result.photon_2_x_ecal_m[photon_2_all_indices],
        ecal_result.photon_2_y_ecal_m[photon_2_all_indices],
        s=4,
        alpha=0.12,
        label=r"Photon 2 reaching $z_{\rm ECAL}$",
    )

    # Accepted events are overlaid to make the populated region inside the
    # rectangle immediately visible.
    axis.scatter(
        ecal_result.photon_1_x_ecal_m[accepted_indices],
        ecal_result.photon_1_y_ecal_m[accepted_indices],
        s=5,
        alpha=0.25,
        marker="x",
        label="Photons in accepted diphoton events",
    )
    axis.scatter(
        ecal_result.photon_2_x_ecal_m[accepted_indices],
        ecal_result.photon_2_y_ecal_m[accepted_indices],
        s=5,
        alpha=0.25,
        marker="x",
    )

    rectangle = Rectangle(
        (
            DEFAULT_ECAL.centre_x_m - DEFAULT_ECAL.half_width_x_m,
            DEFAULT_ECAL.centre_y_m - DEFAULT_ECAL.half_height_y_m,
        ),
        DEFAULT_ECAL.width_x_m,
        DEFAULT_ECAL.height_y_m,
        fill=False,
        linewidth=2.0,
        label="ECAL boundary",
    )
    axis.add_patch(rectangle)

    axis.set_xlabel(r"$x_{\rm ECAL}$ [m]")
    axis.set_ylabel(r"$y_{\rm ECAL}$ [m]")
    axis.set_title(
        rf"{plot_label}: photon intersections, "
        rf"$m_a={MASS_GEV:g}$ GeV, $c\tau={CTAU_M:g}$ m"
    )
    axis.set_aspect("equal", adjustable="box")

    # Show the detector and a moderate surrounding region rather than allowing
    # rare far-away intersections to determine the plotting range.
    axis.set_xlim(
        DEFAULT_ECAL.centre_x_m - 1.8 * DEFAULT_ECAL.width_x_m,
        DEFAULT_ECAL.centre_x_m + 1.8 * DEFAULT_ECAL.width_x_m,
    )
    axis.set_ylim(
        DEFAULT_ECAL.centre_y_m - 1.3 * DEFAULT_ECAL.height_y_m,
        DEFAULT_ECAL.centre_y_m + 1.3 * DEFAULT_ECAL.height_y_m,
    )

    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()

    output_path = OUTPUT_DIR / f"ecal_hits_{source_name}.png"
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    return output_path


def plot_energy_bias(
    *,
    source_name: str,
    plot_label: str,
    mother_results: np.ndarray,
    ecal_result: DiphotonECALResult,
) -> Path:
    """Compare the P_decay-weighted ALP spectrum before and after ECAL."""

    energies = mother_results[:, 3]
    weights = mother_results[:, 6]
    accepted = ecal_result.event_mask

    energy_edges = np.geomspace(
        MASS_GEV,
        400.0,
        NUMBER_OF_ENERGY_BINS + 1,
    )

    before_sum, _ = np.histogram(
        energies,
        bins=energy_edges,
        weights=weights,
    )
    after_sum, _ = np.histogram(
        energies[accepted],
        bins=energy_edges,
        weights=weights[accepted],
    )

    bin_widths = np.diff(energy_edges)

    before_normalization = float(np.sum(before_sum))
    after_normalization = float(np.sum(after_sum))

    if before_normalization <= 0.0 or after_normalization <= 0.0:
        raise RuntimeError("The weighted energy histogram has zero normalization.")

    before_density = before_sum / (before_normalization * bin_widths)
    after_density = after_sum / (after_normalization * bin_widths)

    figure, axis = plt.subplots(figsize=(8.0, 5.8))

    axis.stairs(
        before_density,
        energy_edges,
        linewidth=2.0,
        label="Before ECAL requirement",
    )
    axis.stairs(
        after_density,
        energy_edges,
        linewidth=2.0,
        label="Both photons hit ECAL",
    )

    axis.set_xscale("log")
    axis.set_xlabel(r"$E_a$ [GeV]")
    axis.set_ylabel(
        r"Normalized $P_{\rm decay}$-weighted density [GeV$^{-1}$]"
    )
    axis.set_title(
        rf"{plot_label}: ECAL-induced energy bias, "
        rf"$m_a={MASS_GEV:g}$ GeV, $c\tau={CTAU_M:g}$ m"
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path = OUTPUT_DIR / f"ecal_energy_bias_{source_name}.png"
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    return output_path


def diagnose_source(
    *,
    source_name: str,
    config: dict,
) -> dict:
    """Run all diagnostics for one production source."""

    seed = BASE_SEED + int(config["seed_offset"])
    plot_label = str(config["plot_label"])

    print()
    print(f"Processing {source_name}")
    print(f"m_a   = {MASS_GEV:g} GeV")
    print(f"c_tau = {CTAU_M:g} m")
    print(
        f"Sampling seeds: {seed} for resampling, "
        f"{seed + 1} for true samples, {seed + 2} for a -> gamma gamma"
    )

    mother_results = generate_mother_sample(
        config=config,
        seed=seed,
    )

    ecal_result = diphoton_ecal_acceptance(
        mother_results,
        geometry=DEFAULT_ECAL,
        seed=seed + 2,
        return_details=True,
    )

    maximum_conservation_residual, maximum_photon_mass_squared = (
        validate_kinematics(
            mother_results,
            ecal_result,
        )
    )
    validate_hit_mask(ecal_result)

    event_mask = ecal_result.event_mask
    event_weights = mother_results[:, 6]

    unweighted_acceptance = float(np.mean(event_mask))
    weighted_acceptance = weighted_ecal_acceptance(
        event_mask,
        event_weights,
    )

    hit_plot_path = plot_hit_positions(
        source_name=source_name,
        plot_label=plot_label,
        ecal_result=ecal_result,
        seed=seed + 3,
    )
    energy_plot_path = plot_energy_bias(
        source_name=source_name,
        plot_label=plot_label,
        mother_results=mother_results,
        ecal_result=ecal_result,
    )

    print(f"Mother-level samples: {len(mother_results)}")
    print(f"ECAL-accepted samples: {np.count_nonzero(event_mask)}")
    print(f"Unweighted ECAL acceptance: {unweighted_acceptance:.6f}")
    print(f"Weighted ECAL acceptance: {weighted_acceptance:.6f}")
    print(
        "Maximum diphoton four-momentum residual: "
        f"{maximum_conservation_residual:.6e}"
    )
    print(
        "Maximum photon |E^2-p^2|: "
        f"{maximum_photon_mass_squared:.6e} GeV^2"
    )
    print(f"Hit-position plot: {hit_plot_path}")
    print(f"Energy-bias plot: {energy_plot_path}")

    return {
        "source": source_name,
        "plot_label": plot_label,
        "mass_GeV": MASS_GEV,
        "ctau_m": CTAU_M,
        "mother_samples": len(mother_results),
        "ecal_accepted_samples": int(np.count_nonzero(event_mask)),
        "epsilon_ecal_unweighted": unweighted_acceptance,
        "epsilon_ecal_weighted": weighted_acceptance,
        "maximum_four_momentum_residual": maximum_conservation_residual,
        "maximum_photon_mass_squared_GeV2": maximum_photon_mass_squared,
        "hit_plot": str(hit_plot_path),
        "energy_bias_plot": str(energy_plot_path),
    }


def main() -> None:
    summary_rows = []

    for source_name, config in SOURCE_CONFIGS.items():
        summary_rows.append(
            diagnose_source(
                source_name=source_name,
                config=config,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "ecal_diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("Finished.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()