from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------

# parents[0] = analysis/
# parents[1] = EventCalc-SHiP/
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Some EventCalc paths are interpreted relative to the repository root.
os.chdir(REPO_ROOT)

from funcs.initLLP import LLP
from funcs.kinematics import Grids
from funcs.ship_setup import theta_max_dec_vol


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

N_POT = 6.0e20

# smaller than the final production scan.
RESAMPLE_SIZE = 100_000
N_INTERPOLATION_POINTS = 10 * RESAMPLE_SIZE


EVENT_LEVELS = (
    2.3,
    3.0,
    10.0,
    30.0,
    100.0,
)

# Use an extremely long lifetime when constructing the reusable
# theta-energy sample.
#
# This forces EventCalc to sample the full energy interval starting
# at E_a = m_a. The physical lifetime is inserted later for every
# coupling point.
SAMPLING_CTAU_M = 1.0e99

# Fixed seeds make the scan reproducible.
BASE_SEED = 24680

ANALYSIS_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = ANALYSIS_DIR

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PHOTON_MINIMUM_TABLE_MASS = 2.000000e-02
PHOTON_MAXIMUM_TABLE_MASS = 4.0

SU2_MINIMUM_TABLE_MASS = 0.01
SU2_MAXIMUM_TABLE_MASS = 5.1

photon_base_masses = np.geomspace(
    PHOTON_MINIMUM_TABLE_MASS,
    PHOTON_MAXIMUM_TABLE_MASS,
    50,
)

su2l_base_masses = np.geomspace(
    SU2_MINIMUM_TABLE_MASS,
    SU2_MAXIMUM_TABLE_MASS,
    50,
)


AUTO_REFINE_ENDPOINTS = True

# Add this many new masses inside every insufficiently
# resolved endpoint interval.
ENDPOINT_REFINEMENT_POINTS = 3

# Stop adding real EventCalc masses when the relative
# width of the endpoint bracket is below 0.5%.
ENDPOINT_RELATIVE_WIDTH_TOLERANCE = 5.0e-3


PHOTON_PRIMARY_MODEL = "ALP-photon-primary"
PHOTON_CASCADE_MODEL = "ALP-photon-cascades"
PHOTON_COMBINED_MODEL = "ALP-photon-combined"
SU2_MODEL = "ALP-SU2L"

PHOTON_SOURCE_MODELS = (
    PHOTON_PRIMARY_MODEL,
    PHOTON_CASCADE_MODEL,
)

PHYSICAL_MODEL_NAMES = (
    PHOTON_COMBINED_MODEL,
    SU2_MODEL,
)

PHYSICAL_MODEL_LABELS = {
    PHOTON_COMBINED_MODEL: ("ALP-photon, primary + cascades"),
    SU2_MODEL: r"ALP-$SU(2)_L$",
}


# These are the independently tabulated EventCalc sources.
#
# The primary ALP-photon tables already contain the prompt
# pi0/eta, photon-fusion, and primary Primakov contributions.
# The cascade tables contain Primakov production from secondary
# shower photons. Their event rates must be calculated separately
# and then added point by point.
SOURCE_MODEL_CONFIGS = {
    PHOTON_PRIMARY_MODEL: {
        "plot_label": "ALP-photon, primary",
        "masses": photon_base_masses.copy(),
        "particle_selection": {
            "LLP_name": "ALP-photon",
            "particle_path": str(REPO_ROOT / "Distributions" / "ALP-photon"),
        },
        "alp_production_mode": "primary",
        "couplings": np.geomspace(
            1.0e-10,
            1.0e-2,
            111,
        ),
        # Preserve the seed convention of the old primary scan.
        "seed_offset": 0,
    },
    PHOTON_CASCADE_MODEL: {
        "plot_label": "ALP-photon, cascades",
        "masses": photon_base_masses.copy(),
        "particle_selection": {
            "LLP_name": "ALP-photon",
            "particle_path": str(REPO_ROOT / "Distributions" / "ALP-photon"),
        },
        # initLLP.py accepts both "cascade" and "cascades".
        "alp_production_mode": "cascades",
        "couplings": np.geomspace(
            1.0e-10,
            1.0e-2,
            111,
        ),
        "seed_offset": 2000,
    },
    SU2_MODEL: {
        "plot_label": r"ALP-$SU(2)_L$",
        "masses": su2l_base_masses.copy(),
        "particle_selection": {
            "LLP_name": "ALP-SU2L",
            "particle_path": str(REPO_ROOT / "Distributions" / "ALP-SU2L"),
        },
        "alp_production_mode": None,
        "couplings": np.geomspace(
            1.0e-8,
            3.0,
            111,
        ),
        # Preserve the seed convention of the old SU(2)_L scan.
        "seed_offset": 1000,
    },
}


def stable_float_key(
    value: float,
) -> str:
    """Stable key for values written to and read from CSV."""
    return f"{float(value):.12e}"


def deduplicate_scan_data(
    scan_data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove repeated mass-coupling points."""
    scan_data = scan_data.copy()

    scan_data["_mass_key"] = scan_data["mass_GeV"].map(stable_float_key)

    scan_data["_coupling_key"] = scan_data["coupling_GeV_inv"].map(stable_float_key)

    scan_data = scan_data.drop_duplicates(
        subset=[
            "model",
            "_mass_key",
            "_coupling_key",
        ],
        keep="last",
    )

    return scan_data.drop(
        columns=[
            "_mass_key",
            "_coupling_key",
        ]
    )


def synchronize_source_mass_grids(
    source_scan_data: pd.DataFrame,
) -> None:
    """
    Ensure that both photon sources are evaluated at the same masses.

    This also imports any endpoint-refinement masses already present
    in a legacy primary-only checkpoint, so the existing expensive
    primary calculation can be reused instead of repeated.
    """
    if source_scan_data.empty:
        return

    for model_name, model_data in source_scan_data.groupby(
        "model",
        sort=False,
    ):
        if model_name not in SOURCE_MODEL_CONFIGS:
            continue

        existing_masses = model_data["mass_GeV"].to_numpy(
            dtype=float,
        )

        SOURCE_MODEL_CONFIGS[model_name]["masses"] = np.unique(
            np.concatenate(
                [
                    SOURCE_MODEL_CONFIGS[model_name]["masses"],
                    existing_masses,
                ]
            )
        )

    photon_masses = []

    for model_name in PHOTON_SOURCE_MODELS:
        photon_masses.extend(SOURCE_MODEL_CONFIGS[model_name]["masses"])

    common_photon_masses = np.unique(
        np.asarray(
            photon_masses,
            dtype=float,
        )
    )

    for model_name in PHOTON_SOURCE_MODELS:
        SOURCE_MODEL_CONFIGS[model_name]["masses"] = common_photon_masses.copy()


def build_physical_scan_data(
    source_scan_data: pd.DataFrame,
    *,
    require_complete_photon_pairing: bool,
) -> pd.DataFrame:
    """
    Build the physical scan used for contours.

    The photophilic-ALP event rate is

        N_combined = N_primary + N_cascades.

    No fixed cascade/primary multiplier is introduced: each source
    is evaluated with its own yield and angle-energy distribution.
    """
    source_scan_data = deduplicate_scan_data(source_scan_data)

    su2_data = source_scan_data.loc[source_scan_data["model"] == SU2_MODEL].copy()

    primary = source_scan_data.loc[source_scan_data["model"] == PHOTON_PRIMARY_MODEL].copy()

    cascades = source_scan_data.loc[source_scan_data["model"] == PHOTON_CASCADE_MODEL].copy()

    if primary.empty or cascades.empty:
        if require_complete_photon_pairing:
            raise RuntimeError(
                "Cannot construct ALP-photon-combined: " "primary or cascade scan rows are missing."
            )

        return su2_data.reset_index(drop=True)

    for frame in (
        primary,
        cascades,
    ):
        frame["_mass_key"] = frame["mass_GeV"].map(stable_float_key)

        frame["_coupling_key"] = frame["coupling_GeV_inv"].map(stable_float_key)

    primary_keys = set(
        zip(
            primary["_mass_key"],
            primary["_coupling_key"],
        )
    )

    cascade_keys = set(
        zip(
            cascades["_mass_key"],
            cascades["_coupling_key"],
        )
    )

    if require_complete_photon_pairing and primary_keys != cascade_keys:
        primary_only = primary_keys - cascade_keys

        cascade_only = cascade_keys - primary_keys

        raise RuntimeError(
            "Primary and cascade photon scans do not contain "
            "the same mass-coupling points.\n"
            f"Primary-only points: {len(primary_only)}\n"
            f"Cascade-only points: {len(cascade_only)}\n"
            "Run the script again; completed source points will "
            "be reused from event_density_scan_sources.csv."
        )

    merged = primary.merge(
        cascades,
        on=[
            "_mass_key",
            "_coupling_key",
        ],
        how="inner",
        suffixes=(
            "_primary",
            "_cascades",
        ),
        validate="one_to_one",
    )

    if merged.empty:
        if require_complete_photon_pairing:
            raise RuntimeError("No matching primary/cascade photon points found.")

        return su2_data.reset_index(drop=True)

    common_checks = (
        "mass_GeV",
        "coupling_GeV_inv",
        "coupling_squared_GeV_inv2",
        "ctau_m",
        "unit_coupling_ctau_m",
        "visible_Br",
    )

    for column in common_checks:
        left = merged[f"{column}_primary"].to_numpy(
            dtype=float,
        )

        right = merged[f"{column}_cascades"].to_numpy(
            dtype=float,
        )

        if not np.allclose(
            left,
            right,
            rtol=1.0e-10,
            atol=1.0e-14,
        ):
            maximum_difference = float(np.max(np.abs(left - right)))

            raise RuntimeError(
                "Inconsistent common ALP-photon quantity "
                "between primary and cascade sources:\n"
                f"column = {column}\n"
                f"maximum absolute difference = "
                f"{maximum_difference:.6e}"
            )

    n_events_primary = merged["N_events_primary"].to_numpy(
        dtype=float,
    )

    n_events_cascades = merged["N_events_cascades"].to_numpy(
        dtype=float,
    )

    n_events_combined = n_events_primary + n_events_cascades

    cascade_fraction = np.divide(
        n_events_cascades,
        n_events_combined,
        out=np.zeros_like(n_events_combined),
        where=(n_events_combined > 0.0),
    )

    cascade_to_primary_ratio = np.divide(
        n_events_cascades,
        n_events_primary,
        out=np.full_like(
            n_events_cascades,
            np.nan,
        ),
        where=(n_events_primary > 0.0),
    )

    combined = pd.DataFrame(
        {
            "model": PHOTON_COMBINED_MODEL,
            "mass_GeV": merged["mass_GeV_primary"],
            "coupling_GeV_inv": merged["coupling_GeV_inv_primary"],
            "coupling_squared_GeV_inv2": merged["coupling_squared_GeV_inv2_primary"],
            "ctau_m": merged["ctau_m_primary"],
            "unit_coupling_ctau_m": merged["unit_coupling_ctau_m_primary"],
            "yield_per_PoT_per_coupling_squared": (
                merged["yield_per_PoT_per_coupling_squared_primary"]
                + merged["yield_per_PoT_per_coupling_squared_cascades"]
            ),
            "N_LLP_total": (merged["N_LLP_total_primary"] + merged["N_LLP_total_cascades"]),
            # A single effective epsilon is not well defined for a
            # mixture of independently sampled source distributions.
            "epsilon_polar": np.nan,
            "epsilon_azimuthal": np.nan,
            "mean_P_decay": np.nan,
            "sum_P_decay": np.nan,
            "visible_Br": merged["visible_Br_primary"],
            "sampled_inside_volume": (
                merged["sampled_inside_volume_primary"] + merged["sampled_inside_volume_cascades"]
            ),
            "N_events": n_events_combined,
            "N_events_primary": (n_events_primary),
            "N_events_cascades": (n_events_cascades),
            "cascade_event_fraction": (cascade_fraction),
            "cascade_to_primary_event_ratio": (cascade_to_primary_ratio),
            "yield_primary": merged["yield_per_PoT_per_coupling_squared_primary"],
            "yield_cascades": merged["yield_per_PoT_per_coupling_squared_cascades"],
            "epsilon_polar_primary": merged["epsilon_polar_primary"],
            "epsilon_polar_cascades": merged["epsilon_polar_cascades"],
            "epsilon_azimuthal_primary": merged["epsilon_azimuthal_primary"],
            "epsilon_azimuthal_cascades": merged["epsilon_azimuthal_cascades"],
            "mean_P_decay_primary": merged["mean_P_decay_primary"],
            "mean_P_decay_cascades": merged["mean_P_decay_cascades"],
        }
    )

    physical = pd.concat(
        [
            combined,
            su2_data,
        ],
        ignore_index=True,
        sort=False,
    )

    physical = physical.sort_values(
        [
            "model",
            "mass_GeV",
            "coupling_GeV_inv",
        ]
    ).reset_index(drop=True)

    return deduplicate_scan_data(physical)


def add_refinement_masses_to_sources(
    *,
    physical_model_name: str,
    extra_masses: np.ndarray,
) -> None:
    """Map physical contour-refinement masses to source scans."""
    if physical_model_name == PHOTON_COMBINED_MODEL:
        source_models = PHOTON_SOURCE_MODELS

    elif physical_model_name == SU2_MODEL:
        source_models = (SU2_MODEL,)

    else:
        raise ValueError(
            f"Unknown physical model for endpoint refinement: {physical_model_name}"
        )

    for source_model_name in source_models:
        SOURCE_MODEL_CONFIGS[source_model_name]["masses"] = np.unique(
            np.concatenate(
                [
                    SOURCE_MODEL_CONFIGS[source_model_name]["masses"],
                    extra_masses,
                ]
            )
        )


# ---------------------------------------------------------------------
# LLP initialization
# ---------------------------------------------------------------------


def make_llp(config: dict) -> LLP:
    """Load one EventCalc LLP model."""
    return LLP(
        mass=None,
        particle_selection=(config["particle_selection"]),
        mixing_pattern=None,
        uncertainty=None,
        alp_production_mode=(config["alp_production_mode"]),
    )


# ---------------------------------------------------------------------
# Reusable theta-energy sample
# ---------------------------------------------------------------------

NEGATIVE_WEIGHT_FRACTION_TOLERANCE = 1.0e-3


def validate_and_sanitize_interpolation(
    *,
    kin: Grids,
    model_name: str,
    mass_gev: float,
) -> None:
    """
    Validate the interpolated angle-energy distribution.

    Physical distribution densities and sampling weights must be
    non-negative. Tiny negative values may arise numerically near
    interpolation boundaries and are clipped only when their total
    contribution is negligible.
    """
    mass_grid_min = float(np.min(kin.grid_x))

    mass_grid_max = float(np.max(kin.grid_x))

    mass_tolerance = 1.0e-12 * max(
        1.0,
        abs(mass_grid_min),
        abs(mass_grid_max),
    )

    if mass_gev < mass_grid_min - mass_tolerance or mass_gev > mass_grid_max + mass_tolerance:
        raise RuntimeError(
            "Requested mass lies outside the tabulated "
            "distribution range:\n"
            f"model = {model_name}\n"
            f"requested m_a = {mass_gev:.6e} GeV\n"
            f"supported range = "
            f"[{mass_grid_min:.6e}, "
            f"{mass_grid_max:.6e}] GeV"
        )

    interpolated_values = np.asarray(
        kin.interpolated_values,
        dtype=float,
    )

    sampled_energy = np.asarray(
        kin.energy,
        dtype=float,
    )

    # Grids.interpolate() samples down to E = m_a for an effectively
    # infinite lifetime. For light ALPs this can lie below the minimum
    # energy represented by the tabulated distribution. Trilinear
    # interpolation outside the table then becomes an extrapolation and
    # can produce sizeable unphysical negative densities, especially for
    # the soft cascade spectrum.
    #
    # EventCalc has no information outside the tabulated energy support.
    # Treat the tabulated density as zero there instead of extrapolating
    # it. Keeping these proposal points with zero weight gives the correct
    # Monte-Carlo integral over the supported table domain and avoids
    # changing the shared Grids implementation.
    energy_grid_min = float(np.min(kin.grid_z))

    energy_grid_max = float(np.max(kin.grid_z))

    energy_grid_scale = max(
        1.0,
        abs(energy_grid_min),
        abs(energy_grid_max),
    )

    energy_support_tolerance = 1.0e-12 * energy_grid_scale

    below_energy_grid = sampled_energy < energy_grid_min - energy_support_tolerance

    above_energy_grid = sampled_energy > energy_grid_max + energy_support_tolerance

    outside_energy_grid = below_energy_grid | above_energy_grid

    number_outside_energy_grid = int(np.sum(outside_energy_grid))

    if number_outside_energy_grid > 0:
        interpolated_values = interpolated_values.copy()

        interpolated_values[outside_energy_grid] = 0.0

        kin.interpolated_values = interpolated_values

        print(
            "  proposal points outside tabulated "
            "energy support = "
            f"{number_outside_energy_grid}/"
            f"{len(sampled_energy)} "
            f"({number_outside_energy_grid / len(sampled_energy):.6e})"
        )

        print(
            "  below table minimum = "
            f"{int(np.sum(below_energy_grid))}, "
            "above table maximum = "
            f"{int(np.sum(above_energy_grid))}"
        )

        print("  Setting their distribution density to zero " "instead of extrapolating the table.")
    else:
        print("  proposal points outside tabulated " "energy support = 0")

    max_energy = np.asarray(
        kin.max_energy,
        dtype=float,
    )

    min_energy = np.asarray(
        kin.e_min_sampling,
        dtype=float,
    )

    energy_widths = max_energy - min_energy

    print("Interpolation ranges:")

    print(
        "  distribution mass grid = "
        f"[{float(np.min(kin.grid_x)):.6e}, "
        f"{float(np.max(kin.grid_x)):.6e}] GeV"
    )

    print(
        "  distribution energy grid = "
        f"[{float(np.min(kin.grid_z)):.6e}, "
        f"{float(np.max(kin.grid_z)):.6e}] GeV"
    )

    print(
        "  sampled energy range = "
        f"[{float(np.min(kin.energy)):.6e}, "
        f"{float(np.max(kin.energy)):.6e}] GeV"
    )

    print(
        "  interpolated density range = "
        f"[{float(np.min(interpolated_values)):.6e}, "
        f"{float(np.max(interpolated_values)):.6e}]"
    )

    if np.any(~np.isfinite(interpolated_values)):
        raise RuntimeError(
            "Non-finite interpolated distribution values:\n"
            f"model = {model_name}\n"
            f"m_a = {mass_gev} GeV"
        )

    if np.any(~np.isfinite(energy_widths)):
        raise RuntimeError(
            f"Non-finite sampling energy widths:\nmodel = {model_name}\nm_a = {mass_gev} GeV"
        )

    energy_scale = max(
        1.0,
        float(np.max(np.abs(max_energy))),
        float(np.max(np.abs(min_energy))),
    )

    width_tolerance = 1.0e-12 * energy_scale

    minimum_width = float(np.min(energy_widths))

    if minimum_width < -width_tolerance:
        raise RuntimeError(
            "Substantially negative energy interval found:\n"
            f"model = {model_name}\n"
            f"m_a = {mass_gev} GeV\n"
            f"minimum E_max - E_min = "
            f"{minimum_width:.6e} GeV"
        )

    # Remove only round-off-level negative energy widths.
    kin.max_energy = np.maximum(
        max_energy,
        min_energy,
    )

    energy_widths = kin.max_energy - kin.e_min_sampling

    negative_mask = interpolated_values < 0.0

    number_negative = int(np.sum(negative_mask))

    if number_negative == 0:
        print("  negative interpolation values = 0")
        return

    positive_values = np.clip(
        interpolated_values,
        0.0,
        None,
    )

    negative_magnitudes = np.clip(
        -interpolated_values,
        0.0,
        None,
    )

    positive_weight = float(np.sum(positive_values * energy_widths))

    negative_weight_magnitude = float(np.sum(negative_magnitudes * energy_widths))

    if positive_weight <= 0.0:
        raise RuntimeError(
            "Interpolated distribution has no positive weight:\n"
            f"model = {model_name}\n"
            f"m_a = {mass_gev} GeV"
        )

    negative_weight_fraction = negative_weight_magnitude / positive_weight

    negative_point_fraction = number_negative / len(interpolated_values)

    print(
        "  negative interpolation points = "
        f"{number_negative}/"
        f"{len(interpolated_values)} "
        f"({negative_point_fraction:.6e})"
    )

    print(f"  negative weight fraction = {negative_weight_fraction:.6e}")

    if negative_weight_fraction > NEGATIVE_WEIGHT_FRACTION_TOLERANCE:
        raise RuntimeError(
            "Negative interpolation contribution is too "
            "large to clip safely:\n"
            f"model = {model_name}\n"
            f"m_a = {mass_gev} GeV\n"
            "negative weight fraction = "
            f"{negative_weight_fraction:.6e}\n"
            "This likely indicates interpolation outside a "
            "well-supported table region."
        )

    print("  Clipping negligible negative " "interpolation values to zero.")

    kin.interpolated_values = positive_values


def prepare_kinematic_sample(
    *,
    llp: LLP,
    mass_gev: float,
    seed: int,
) -> tuple[Grids, float]:
    """
    Generate one reusable theta-energy sample.

    The production distribution does not depend on the coupling.
    Therefore, theta and energy are sampled only once for each
    model and mass.

    Decay positions and decay probabilities are recalculated for
    every coupling point.
    """
    llp.set_mass(mass_gev)
    llp.compute_mass_dependent_properties()

    # Use the very long dummy lifetime only while constructing the
    # full theta-energy proposal sample.
    llp.set_c_tau(SAMPLING_CTAU_M)

    visible_br = float(np.sum(llp.BrRatios_distr))

    if not np.isfinite(visible_br) or visible_br <= 0.0:
        raise RuntimeError(
            "Invalid visible branching ratio:\n"
            f"LLP = {llp.LLP_name}\n"
            f"m_a = {mass_gev} GeV\n"
            f"Br_visible = {visible_br}"
        )

    np.random.seed(seed)

    kin = Grids(
        llp.Distr,
        llp.Energy_distr,
        N_INTERPOLATION_POINTS,
        llp.mass,
        SAMPLING_CTAU_M,
        theta_max_sim=theta_max_dec_vol,
    )

    kin.interpolate(False)

    # The reusable sample must cover the full physical energy range.
    # At every theta point, the lower energy limit should be m_a.
    if not np.allclose(
        kin.e_min_sampling,
        mass_gev,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        minimum = float(np.min(kin.e_min_sampling))

        maximum = float(np.max(kin.e_min_sampling))

        raise RuntimeError(
            "The reusable sample does not cover the full "
            "energy range.\n"
            f"LLP = {llp.LLP_name}\n"
            f"m_a = {mass_gev} GeV\n"
            "e_min_sampling range = "
            f"[{minimum}, {maximum}] GeV"
        )

    validate_and_sanitize_interpolation(
        kin=kin,
        model_name=llp.LLP_name,
        mass_gev=mass_gev,
    )

    kin.resample(
        RESAMPLE_SIZE,
        False,
    )

    return kin, visible_br


# ---------------------------------------------------------------------
# One mass-coupling point
# ---------------------------------------------------------------------


def evaluate_coupling(
    *,
    model_name: str,
    llp: LLP,
    kin: Grids,
    mass_gev: float,
    coupling_gev_inv: float,
    visible_br: float,
    true_sample_seed: int,
) -> dict:
    """
    Calculate the EventCalc event rate for one mass-coupling point.
    """
    coupling_gev_inv = float(coupling_gev_inv)

    if coupling_gev_inv <= 0.0:
        raise ValueError("The coupling must be positive.")

    coupling_squared = coupling_gev_inv**2

    # c_tau_int is EventCalc's c*tau at unit coupling:
    #
    #     g_ref = 1 GeV^-1.
    #
    # Therefore:
    #
    #     c*tau(g) = c_tau_int / g^2.
    unit_coupling_ctau_m = float(llp.c_tau_int)

    ctau_m = unit_coupling_ctau_m / coupling_squared

    if not np.isfinite(ctau_m) or ctau_m <= 0.0:
        raise RuntimeError(
            "Invalid lifetime obtained from coupling:\n"
            f"model = {model_name}\n"
            f"m_a = {mass_gev} GeV\n"
            f"coupling = {coupling_gev_inv} GeV^-1\n"
            f"c_tau = {ctau_m} m"
        )

    llp.set_c_tau(ctau_m)

    # The yield table is normalized to unit coupling squared.
    n_llp_total = N_POT * float(llp.Yield) * coupling_squared

    # Keep the same sampled theta and energy values, but insert the
    # physical lifetime corresponding to this coupling.
    kin.c_tau = ctau_m

    # Reusing the same random numbers at every coupling reduces
    # artificial point-to-point Monte Carlo fluctuations.
    np.random.seed(true_sample_seed)

    kin.true_samples(False)

    mother_particle_results = kin.get_kinematics()

    number_inside_volume = len(mother_particle_results)

    epsilon_polar = float(kin.epsilon_polar)

    epsilon_azimuthal = number_inside_volume / RESAMPLE_SIZE

    if number_inside_volume == 0:
        mean_decay_probability = 0.0
        summed_decay_probability = 0.0
        n_events = 0.0

    else:
        # EventCalc column 6 contains P_decay.
        decay_probabilities = np.asarray(
            mother_particle_results[:, 6],
            dtype=float,
        )

        if np.any(~np.isfinite(decay_probabilities)):
            raise RuntimeError("Non-finite decay probabilities found.")

        if np.any(decay_probabilities < 0.0):
            raise RuntimeError("Negative decay probabilities found.")

        mean_decay_probability = float(np.mean(decay_probabilities))

        summed_decay_probability = float(np.sum(decay_probabilities))

        # Equivalent to:
        #
        # N_LLP_total
        # * epsilon_polar
        # * epsilon_azimuthal
        # * <P_decay>
        # * Br_visible.
        n_events = (
            n_llp_total * epsilon_polar * summed_decay_probability / RESAMPLE_SIZE * visible_br
        )

    return {
        "model": model_name,
        "mass_GeV": mass_gev,
        "coupling_GeV_inv": (coupling_gev_inv),
        "coupling_squared_GeV_inv2": (coupling_squared),
        "ctau_m": ctau_m,
        "unit_coupling_ctau_m": (unit_coupling_ctau_m),
        "yield_per_PoT_per_coupling_squared": (float(llp.Yield)),
        "N_LLP_total": n_llp_total,
        "epsilon_polar": epsilon_polar,
        "epsilon_azimuthal": (epsilon_azimuthal),
        "mean_P_decay": (mean_decay_probability),
        "sum_P_decay": (summed_decay_probability),
        "visible_Br": visible_br,
        "sampled_inside_volume": (number_inside_volume),
        "N_events": n_events,
    }


# ---------------------------------------------------------------------
# Scan one model and mass
# ---------------------------------------------------------------------


def scan_model_mass(
    *,
    model_name: str,
    config: dict,
    mass_gev: float,
    seed: int,
) -> list[dict]:
    """Scan all requested couplings for one model and mass."""
    print()
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Mass:  {mass_gev:g} GeV")
    print("=" * 70)

    llp = make_llp(config)

    kin, visible_br = prepare_kinematic_sample(
        llp=llp,
        mass_gev=mass_gev,
        seed=seed,
    )

    print(f"Unit-coupling lifetime: {float(llp.c_tau_int):.6e} m")

    print(f"Yield per PoT per coupling squared: {float(llp.Yield):.6e}")

    print(f"Visible branching ratio: {visible_br:.6e}")

    rows = []

    couplings = config["couplings"]

    for coupling_index, coupling in enumerate(couplings):
        row = evaluate_coupling(
            model_name=model_name,
            llp=llp,
            kin=kin,
            mass_gev=mass_gev,
            coupling_gev_inv=coupling,
            visible_br=visible_br,
            true_sample_seed=seed + 1,
        )

        rows.append(row)

        print(
            f"[{coupling_index + 1:2d}/"
            f"{len(couplings):2d}] "
            f"g = {coupling:.4e} GeV^-1, "
            f"c_tau = {row['ctau_m']:.4e} m, "
            f"N_events = {row['N_events']:.6g}"
        )

    rates = np.array(
        [row["N_events"] for row in rows],
        dtype=float,
    )

    peak_index = int(np.argmax(rates))

    print()
    print("Largest event rate in this scan:")

    print(f"  g = {rows[peak_index]['coupling_GeV_inv']:.6e} GeV^-1")

    print(f"  c_tau = {rows[peak_index]['ctau_m']:.6e} m")

    print(f"  N_events = {rows[peak_index]['N_events']:.6g}")

    return rows


# ---------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------


def safe_filename(text: str) -> str:
    """Convert a model name into a safe filename component."""
    return text.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def plot_diagnostic_curves(
    scan_data: pd.DataFrame,
) -> list[Path]:
    """
    Plot N_events as a function of coupling for each diagnostic mass.
    """
    output_paths = []

    for model_name, model_data in scan_data.groupby(
        "model",
        sort=False,
    ):
        figure, axis = plt.subplots(
            figsize=(8.5, 6.0),
        )

        for mass_gev, mass_data in model_data.groupby(
            "mass_GeV",
            sort=True,
        ):
            mass_data = mass_data.sort_values("coupling_GeV_inv")

            couplings = mass_data["coupling_GeV_inv"].to_numpy(
                dtype=float,
            )

            rates = mass_data["N_events"].to_numpy(
                dtype=float,
            )

            MIN_PLOTTED_RATE = 1.0e-1
            plot_rates = np.where(
                rates >= MIN_PLOTTED_RATE,
                rates,
                np.nan,
            )

            axis.plot(
                couplings,
                plot_rates,
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                label=(rf"$m_a={mass_gev:g}$ GeV"),
            )

        event_line_styles = {
            2.3: (0, (1, 1)),
            3.0: ":",
            10.0: "--",
            30.0: "-.",
            100.0: "-",
        }

        for event_level in EVENT_LEVELS:
            axis.axhline(
                event_level,
                linestyle=event_line_styles[event_level],
                linewidth=1.2,
                label=rf"$N_{{\rm events}}={event_level:g}$",
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(
            1.0e-1,
            1.0e10,
        )

        axis.set_xlabel(r"Coupling [GeV$^{-1}$]")

        axis.set_ylabel(r"$N_{\rm events}$")

        axis.set_title(PHYSICAL_MODEL_LABELS[model_name])

        axis.grid(
            True,
            which="both",
            alpha=0.3,
        )

    #axis.legend(
        #    fontsize=9,
    #)

        figure.tight_layout()

        output_path = PLOT_DIR / f"event_rate_vs_coupling_{safe_filename(model_name)}.pdf"

        figure.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close(figure)

        output_paths.append(output_path)

    return output_paths


def find_level_crossings(
    mass_data: pd.DataFrame,
    event_level: float,
) -> list[float]:
    """
    Find couplings where N_events crosses one event level.

    Interpolation is linear in log10(g) and log10(N_events).
    """
    mass_data = mass_data.sort_values("coupling_GeV_inv")

    couplings = mass_data["coupling_GeV_inv"].to_numpy(
        dtype=float,
    )

    event_rates = mass_data["N_events"].to_numpy(
        dtype=float,
    )

    # Avoid log10(0). This floor is only used when identifying
    # intervals containing a crossing.
    event_rates_safe = np.maximum(
        event_rates,
        1.0e-300,
    )

    log_couplings = np.log10(couplings)

    log_rate_difference = np.log10(event_rates_safe) - np.log10(event_level)

    crossings = []

    for index in range(len(log_couplings) - 1):
        difference_left = log_rate_difference[index]

        difference_right = log_rate_difference[index + 1]

        if difference_left == 0.0:
            crossings.append(couplings[index])

        if difference_left * difference_right < 0.0:
            fraction = -difference_left / (difference_right - difference_left)

            log_crossing = log_couplings[index] + fraction * (
                log_couplings[index + 1] - log_couplings[index]
            )

            crossings.append(10.0**log_crossing)

    return crossings


def build_boundary_table(
    scan_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build lower and upper coupling boundaries."""
    rows = []

    grouped_data = scan_data.groupby(
        [
            "model",
            "mass_GeV",
        ],
        sort=False,
    )

    for (
        model_name,
        mass_gev,
    ), mass_data in grouped_data:
        mass_data = mass_data.sort_values("coupling_GeV_inv")

        couplings = mass_data["coupling_GeV_inv"].to_numpy(dtype=float)

        rates = mass_data["N_events"].to_numpy(dtype=float)

        peak_index = int(np.argmax(rates))

        maximum_rate = float(rates[peak_index])

        peak_coupling = float(couplings[peak_index])

        first_rate = float(rates[0])
        last_rate = float(rates[-1])

        for event_level in EVENT_LEVELS:
            crossings = find_level_crossings(
                mass_data,
                event_level,
            )

            number_of_crossings = len(crossings)

            if maximum_rate < event_level:
                status = "outside_mass_reach"

            elif number_of_crossings == 2:
                status = "resolved"

            elif number_of_crossings > 2:
                status = "multiple_crossings"

            elif number_of_crossings == 1:
                first_above = first_rate >= event_level

                last_above = last_rate >= event_level

                if not first_above and last_above:
                    status = "upper_boundary_above_scan"

                elif first_above and not last_above:
                    status = "lower_boundary_below_scan"

                else:
                    status = "one_crossing_unclassified"

            elif first_rate >= event_level and last_rate >= event_level:
                status = "both_boundaries_outside_scan"

            else:
                status = "unresolved_numerically"

            rows.append(
                {
                    "model": model_name,
                    "mass_GeV": mass_gev,
                    "event_level": event_level,
                    "status": status,
                    "number_of_crossings": (number_of_crossings),
                    "maximum_N_events": (maximum_rate),
                    "peak_coupling_GeV_inv": (peak_coupling),
                    "lower_coupling_GeV_inv": (
                        crossings[0] if number_of_crossings >= 1 else np.nan
                    ),
                    "upper_coupling_GeV_inv": (
                        crossings[-1] if number_of_crossings >= 2 else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def find_endpoint_refinement_masses(
    *,
    boundary_data: pd.DataFrame,
    scan_data: pd.DataFrame,
    points_per_bracket: int,
    relative_width_tolerance: float,
) -> dict[str, np.ndarray]:
    """
    Find additional masses needed to refine contour endpoints.

    For each model and event level, identify:

        last mass with two resolved crossings
        first larger mass outside the mass reach

    If the relative width of this bracket is still too large,
    place equally spaced masses inside it.
    """
    refinement_masses = {model_name: [] for model_name in PHYSICAL_MODEL_NAMES}

    existing_masses = {
        model_name: model_data["mass_GeV"].to_numpy(dtype=float)
        for model_name, model_data in scan_data.groupby(
            "model",
            sort=False,
        )
    }

    grouped_data = boundary_data.groupby(
        [
            "model",
            "event_level",
        ],
        sort=False,
    )

    for (
        model_name,
        event_level,
    ), level_data in grouped_data:
        level_data = level_data.sort_values("mass_GeV")

        resolved_data = level_data[level_data["status"] == "resolved"]

        if resolved_data.empty:
            continue

        # Last mass for which the contour still has
        # both a lower and an upper crossing.
        left_row = resolved_data.loc[resolved_data["mass_GeV"].idxmax()]

        mass_left = float(left_row["mass_GeV"])

        outside_data = level_data[
            (level_data["status"] == "outside_mass_reach") & (level_data["mass_GeV"] > mass_left)
        ]

        # Example: SU(2)_L, N_events = 3.
        # The contour remains sensitive up to the
        # maximum supported table mass, so no endpoint
        # bracket exists.
        if outside_data.empty:
            print(f"No endpoint bracket for {model_name}, N_events = {event_level:g}.")
            continue

        right_row = outside_data.loc[outside_data["mass_GeV"].idxmin()]

        mass_right = float(right_row["mass_GeV"])

        bracket_midpoint = 0.5 * (mass_left + mass_right)

        relative_width = (mass_right - mass_left) / bracket_midpoint

        print(
            f"Endpoint bracket: {model_name}, "
            f"N_events = {event_level:g}, "
            f"[{mass_left:.8g}, "
            f"{mass_right:.8g}] GeV, "
            f"relative width = "
            f"{relative_width:.4%}"
        )

        if relative_width <= relative_width_tolerance:
            print("  Bracket is sufficiently narrow; " "no new EventCalc masses added.")
            continue

        candidates = np.linspace(
            mass_left,
            mass_right,
            points_per_bracket + 2,
        )[1:-1]

        model_existing_masses = existing_masses.get(
            model_name,
            np.array([], dtype=float),
        )

        for candidate in candidates:
            already_scanned = np.any(
                np.isclose(
                    model_existing_masses,
                    candidate,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )

            if not already_scanned:
                refinement_masses[model_name].append(float(candidate))

    return {
        model_name: np.unique(
            np.asarray(
                masses,
                dtype=float,
            )
        )
        for model_name, masses in refinement_masses.items()
    }


def add_interpolated_closing_points(
    boundary_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append one interpolated closing point to every bracketed contour.

    The closing mass is found by interpolating log(N_peak)
    between:

        last resolved mass,
        first mass outside the sensitivity reach.

    The peak coupling is interpolated logarithmically.
    """
    output_data = boundary_data.copy()

    output_data["is_interpolated"] = False

    closing_rows = []

    grouped_data = boundary_data.groupby(
        [
            "model",
            "event_level",
        ],
        sort=False,
    )

    for (
        model_name,
        event_level,
    ), level_data in grouped_data:
        level_data = level_data.sort_values("mass_GeV")

        resolved_data = level_data[level_data["status"] == "resolved"]

        if resolved_data.empty:
            continue

        resolved_row = resolved_data.loc[resolved_data["mass_GeV"].idxmax()]

        mass_left = float(resolved_row["mass_GeV"])

        outside_data = level_data[
            (level_data["status"] == "outside_mass_reach") & (level_data["mass_GeV"] > mass_left)
        ]

        # There is no demonstrated closing point within
        # the available table range.
        #
        # This is currently the case for:
        # ALP-SU2L, N_events = 3.
        if outside_data.empty:
            print(
                "No interpolated closing point for "
                f"{model_name}, "
                f"N_events = {event_level:g}: "
                "the contour reaches the maximum "
                "available mass."
            )
            continue

        outside_row = outside_data.loc[outside_data["mass_GeV"].idxmin()]

        mass_right = float(outside_row["mass_GeV"])

        rate_left = float(resolved_row["maximum_N_events"])

        rate_right = float(outside_row["maximum_N_events"])

        coupling_left = float(resolved_row["peak_coupling_GeV_inv"])

        coupling_right = float(outside_row["peak_coupling_GeV_inv"])

        if not (rate_left >= event_level and rate_right < event_level):
            raise RuntimeError(
                "Invalid endpoint bracket:\n"
                f"model = {model_name}\n"
                f"N_events = {event_level:g}\n"
                f"left: m = {mass_left}, "
                f"N_peak = {rate_left}\n"
                f"right: m = {mass_right}, "
                f"N_peak = {rate_right}"
            )

        # Solve:
        #
        # log(N_peak(m_closing))
        #     = log(event_level)
        #
        # using linear interpolation in mass.
        interpolation_fraction = (np.log(event_level) - np.log(rate_left)) / (
            np.log(rate_right) - np.log(rate_left)
        )

        closing_mass = mass_left + interpolation_fraction * (mass_right - mass_left)

        # Coupling is naturally treated logarithmically
        # because the coupling axis spans many orders
        # of magnitude.
        closing_log_coupling = np.log(coupling_left) + interpolation_fraction * (
            np.log(coupling_right) - np.log(coupling_left)
        )

        closing_coupling = float(np.exp(closing_log_coupling))

        closing_rows.append(
            {
                "model": model_name,
                "mass_GeV": closing_mass,
                "event_level": event_level,
                "status": ("interpolated_closing_point"),
                # This is not an independently evaluated
                # EventCalc mass.
                "number_of_crossings": np.nan,
                "maximum_N_events": event_level,
                "peak_coupling_GeV_inv": (closing_coupling),
                "lower_coupling_GeV_inv": (closing_coupling),
                "upper_coupling_GeV_inv": (closing_coupling),
                "is_interpolated": True,
            }
        )

        print(
            "Interpolated closing point: "
            f"{model_name}, "
            f"N_events = {event_level:g}, "
            f"m_a = {closing_mass:.8g} GeV, "
            f"g = {closing_coupling:.8e} GeV^-1"
        )

    if closing_rows:
        output_data = pd.concat(
            [
                output_data,
                pd.DataFrame(closing_rows),
            ],
            ignore_index=True,
        )

    output_data = output_data.sort_values(
        [
            "model",
            "event_level",
            "mass_GeV",
        ]
    ).reset_index(drop=True)

    return output_data


# ---------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------


def main() -> None:
    source_csv_path = OUTPUT_DIR / "event_density_scan_sources.csv"

    physical_csv_path = OUTPUT_DIR / "event_density_scan_coarse.csv"

    # Reuse the old primary-only checkpoint automatically.
    #
    # Before cascade support, event_density_scan_coarse.csv contained
    # ALP-photon-primary and ALP-SU2L rows. On the first run of this
    # version, those rows become the source-level checkpoint, and only
    # the missing cascade source is evaluated.
    if source_csv_path.exists():
        existing_source_data = pd.read_csv(source_csv_path)

    elif physical_csv_path.exists():
        legacy_data = pd.read_csv(physical_csv_path)

        existing_source_data = legacy_data.loc[
            legacy_data["model"].isin(SOURCE_MODEL_CONFIGS)
        ].copy()

        if not existing_source_data.empty:
            print("Migrating legacy primary-only checkpoint to:")
            print(f"  {source_csv_path}")

    else:
        existing_source_data = pd.DataFrame()

    if not existing_source_data.empty:
        existing_source_data = deduplicate_scan_data(existing_source_data)

        synchronize_source_mass_grids(existing_source_data)

        completed_points = {
            (
                model_name,
                stable_float_key(mass_gev),
            )
            for model_name, mass_gev in zip(
                existing_source_data["model"],
                existing_source_data["mass_GeV"],
            )
        }

        all_source_rows = existing_source_data.to_dict(orient="records")

    else:
        completed_points = set()
        all_source_rows = []

    current_source_data = pd.DataFrame(all_source_rows)

    if not current_source_data.empty:
        current_physical_data = build_physical_scan_data(
            current_source_data,
            require_complete_photon_pairing=False,
        )
    else:
        current_physical_data = pd.DataFrame()

    if AUTO_REFINE_ENDPOINTS and not current_physical_data.empty:
        previous_boundary_data = build_boundary_table(current_physical_data)

        automatic_refinement_masses = find_endpoint_refinement_masses(
            boundary_data=(previous_boundary_data),
            scan_data=(current_physical_data),
            points_per_bracket=(ENDPOINT_REFINEMENT_POINTS),
            relative_width_tolerance=(ENDPOINT_RELATIVE_WIDTH_TOLERANCE),
        )

        for (
            physical_model_name,
            extra_masses,
        ) in automatic_refinement_masses.items():
            if len(extra_masses) == 0:
                continue

            add_refinement_masses_to_sources(
                physical_model_name=(physical_model_name),
                extra_masses=extra_masses,
            )

            print(
                f"Automatically added "
                f"{len(extra_masses)} endpoint "
                f"masses for {physical_model_name}:"
            )

            print(extra_masses)

    # Keep primary and cascade photon mass grids identical after any
    # automatic endpoint refinement.
    if not current_source_data.empty:
        synchronize_source_mass_grids(current_source_data)
    else:
        photon_masses = np.unique(
            np.concatenate(
                [
                    SOURCE_MODEL_CONFIGS[PHOTON_PRIMARY_MODEL]["masses"],
                    SOURCE_MODEL_CONFIGS[PHOTON_CASCADE_MODEL]["masses"],
                ]
            )
        )

        for source_model_name in PHOTON_SOURCE_MODELS:
            SOURCE_MODEL_CONFIGS[source_model_name]["masses"] = photon_masses.copy()

    for (
        source_model_name,
        config,
    ) in SOURCE_MODEL_CONFIGS.items():
        for mass_index, mass_gev in enumerate(config["masses"]):
            seed = BASE_SEED + int(config["seed_offset"]) + 100 * mass_index

            point_key = (
                source_model_name,
                stable_float_key(mass_gev),
            )

            if point_key in completed_points:
                print(
                    "Skipping completed source point: "
                    f"{source_model_name}, "
                    f"m_a = {mass_gev:g} GeV"
                )
                continue

            rows = scan_model_mass(
                model_name=(source_model_name),
                config=config,
                mass_gev=float(mass_gev),
                seed=seed,
            )

            all_source_rows.extend(rows)

            current_source_data = pd.DataFrame(all_source_rows)

            current_source_data = current_source_data.sort_values(
                [
                    "model",
                    "mass_GeV",
                    "coupling_GeV_inv",
                ]
            ).reset_index(drop=True)

            current_source_data = deduplicate_scan_data(current_source_data)

            current_source_data.to_csv(
                source_csv_path,
                index=False,
            )

            current_physical_data = build_physical_scan_data(
                current_source_data,
                require_complete_photon_pairing=False,
            )

            current_physical_data.to_csv(
                physical_csv_path,
                index=False,
            )

            print("Source checkpoint saved to:")
            print(f"  {source_csv_path}")

            print("Physical combined checkpoint saved to:")
            print(f"  {physical_csv_path}")

    source_scan_data = pd.DataFrame(all_source_rows)

    source_scan_data = source_scan_data.sort_values(
        [
            "model",
            "mass_GeV",
            "coupling_GeV_inv",
        ]
    ).reset_index(drop=True)

    source_scan_data = deduplicate_scan_data(source_scan_data)

    source_scan_data.to_csv(
        source_csv_path,
        index=False,
    )

    scan_data = build_physical_scan_data(
        source_scan_data,
        require_complete_photon_pairing=True,
    )

    scan_data.to_csv(
        physical_csv_path,
        index=False,
    )

    plot_paths = plot_diagnostic_curves(scan_data)

    raw_boundary_data = build_boundary_table(scan_data)

    raw_boundary_path = OUTPUT_DIR / "event_contour_boundaries_raw.csv"

    raw_boundary_data.to_csv(
        raw_boundary_path,
        index=False,
    )

    boundary_data = add_interpolated_closing_points(raw_boundary_data)

    boundary_path = OUTPUT_DIR / "event_contour_boundaries.csv"

    boundary_data.to_csv(
        boundary_path,
        index=False,
    )

    print(f"Source scan saved to: {source_csv_path}")

    print(f"Physical combined scan saved to: {physical_csv_path}")

    print(f"Raw boundary table saved to: {raw_boundary_path}")

    print(f"Final boundary table saved to: {boundary_path}")

    photon_data = scan_data.loc[scan_data["model"] == PHOTON_COMBINED_MODEL]

    if not photon_data.empty:
        fractions = photon_data["cascade_event_fraction"].to_numpy(
            dtype=float,
        )

        print()
        print("Cascade fraction among parent-level " "ALP-photon decay events:")

        print(f"  minimum = {float(np.nanmin(fractions)):.6g}")

        print(f"  maximum = {float(np.nanmax(fractions)):.6g}")

    print()
    print("=" * 70)
    print("Finished event-density scan with " "primary + cascade ALP-photon production")
    print("=" * 70)

    for plot_path in plot_paths:
        print(f"Diagnostic plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
