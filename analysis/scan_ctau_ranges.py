from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Make imports and relative EventCalc paths work when this file is run as
#     python analysis/scan_ctau_ranges.py
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

from funcs.initLLP import LLP
from funcs.kinematics import Grids
from funcs.ship_setup import theta_max_dec_vol


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

N_POT = 6.0e20

RESAMPLE_SIZE = 200_000
N_INTERPOLATION_POINTS = 10 * RESAMPLE_SIZE

MASSES_GEV = np.array([0.3, 0.4, 0.5, 0.75, 1.0])

EVENT_THRESHOLD = 10.0
MAX_CTAU_M = 1.0e3

# Consecutive coarse scan points differ by this factor.
COARSE_FACTOR = 1.7

# Number of log-bisection iterations near N_events = 10.
BISECTION_STEPS = 14

# Fixed seeds make the scan reproducible and reduce artificial
# fluctuations between neighbouring lifetime points.
BASE_SEED = 12345

ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "ctau_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_CONFIGS = {
    "ALP-photon-primary": {
        "particle_selection": {
            "LLP_name": "ALP-photon",
            "particle_path": str(
                REPO_ROOT / "Distributions" / "ALP-photon"
            ),
        },
        "alp_production_mode": "primary",
    },
    "ALP-SU2L": {
        "particle_selection": {
            "LLP_name": "ALP-SU2L",
            "particle_path": str(
                REPO_ROOT / "Distributions" / "ALP-SU2L"
            ),
        },
        "alp_production_mode": None,
    },
}


# ---------------------------------------------------------------------
# Basic definitions
# ---------------------------------------------------------------------

def lower_ctau_limit(m_a: float) -> float:
    """
    Lower lifetime limit in metres.

    Parameters
    ----------
    m_a:
        ALP mass in GeV.
    """
    return 3.0 * (m_a / 0.3)


def make_llp(config: dict) -> LLP:
    """Load one EventCalc LLP model."""
    return LLP(
        mass=None,
        particle_selection=config["particle_selection"],
        mixing_pattern=None,
        uncertainty=None,
        alp_production_mode=config["alp_production_mode"],
    )


def make_coarse_ctau_grid(
    ctau_min: float,
    ctau_max: float,
    factor: float,
) -> np.ndarray:
    """Construct a geometrically spaced grid using a fixed ratio."""
    if ctau_min <= 0.0:
        raise ValueError("ctau_min must be positive.")

    if ctau_max <= ctau_min:
        raise ValueError("ctau_max must be larger than ctau_min.")

    values = [float(ctau_min)]

    while values[-1] < ctau_max:
        next_value = min(values[-1] * factor, ctau_max)

        if next_value <= values[-1]:
            break

        values.append(next_value)

    return np.asarray(values)


# ---------------------------------------------------------------------
# Kinematic sampling
# ---------------------------------------------------------------------

def prepare_kinematic_sample(
    llp: LLP,
    m_a: float,
    ctau_min: float,
    seed: int,
) -> tuple[Grids, float]:
    """
    Generate the theta-energy sample once for one model and one mass.

    In the lifetime range considered here, EventCalc's lower sampled
    energy should be exactly m_a. Therefore, the theta-energy proposal
    distribution is independent of c*tau and can be reused.

    Decay positions and decay probabilities must still be recalculated
    separately for every lifetime.
    """
    llp.set_mass(m_a)
    llp.compute_mass_dependent_properties()
    llp.set_c_tau(ctau_min)

    br_visible = float(np.sum(llp.BrRatios_distr))

    if not np.isfinite(br_visible) or br_visible <= 0.0:
        raise RuntimeError(
            f"Invalid visible branching ratio for "
            f"{llp.LLP_name}, m={m_a} GeV: {br_visible}"
        )

    np.random.seed(seed)

    kin = Grids(
        llp.Distr,
        llp.Energy_distr,
        N_INTERPOLATION_POINTS,
        llp.mass,
        ctau_min,
        theta_max_sim=theta_max_dec_vol,
    )

    kin.interpolate(False)

    # EventCalc uses
    #
    # e_min = max[m_a, min(2.133*m_a/c_tau, E_max/2)].
    #
    # For c_tau >= 3 m * m_a/(0.3 GeV), the first term should
    # dominate for all requested masses.
    if not np.allclose(
        kin.e_min_sampling,
        m_a,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        minimum = float(np.min(kin.e_min_sampling))
        maximum = float(np.max(kin.e_min_sampling))

        raise RuntimeError(
            "The theta-energy sample cannot safely be reused across "
            "lifetimes because e_min_sampling is not equal to m_a.\n"
            f"Model: {llp.LLP_name}\n"
            f"m_a: {m_a} GeV\n"
            f"e_min_sampling range: [{minimum}, {maximum}] GeV"
        )

    kin.resample(RESAMPLE_SIZE, False)

    return kin, br_visible


def evaluate_ctau(
    *,
    model_name: str,
    llp: LLP,
    kin: Grids,
    m_a: float,
    ctau: float,
    ctau_min: float,
    br_visible: float,
    true_sample_seed: int,
) -> dict:
    """
    Calculate the EventCalc event rate for one lifetime.

    The same random numbers are used at every lifetime for a fixed
    model and mass. This reduces Monte Carlo noise in the lifetime scan.
    """
    ctau = float(ctau)

    if ctau <= 0.0:
        raise ValueError("ctau must be positive.")

    llp.set_c_tau(ctau)

    coupling_squared = float(llp.c_tau_int / ctau)

    n_llp_total = (
        N_POT
        * float(llp.Yield)
        * coupling_squared
    )

    # Reuse theta and energy, but recalculate decay positions and
    # decay probabilities for this lifetime.
    kin.c_tau = ctau

    # Common random numbers for all lifetimes at fixed mass/model.
    np.random.seed(true_sample_seed)
    kin.true_samples(False)

    mother_particle_results = kin.get_kinematics()
    final_events = len(mother_particle_results)

    epsilon_polar = float(kin.epsilon_polar)
    epsilon_azimuthal = final_events / RESAMPLE_SIZE

    if final_events == 0:
        p_decay_averaged = 0.0
        n_events = 0.0
    else:
        # Column 6 is P_decay, as in simulate.py.
        p_decay_averaged = float(
            mother_particle_results[:, 6].mean()
        )

        n_events = (
            n_llp_total
            * epsilon_polar
            * epsilon_azimuthal
            * p_decay_averaged
            * br_visible
        )

    return {
        "model": model_name,
        "mass_GeV": m_a,
        "ctau_m": ctau,
        "ctau_min_m": ctau_min,
        "coupling_squared": coupling_squared,
        "yield_per_PoT_per_coupling_squared": float(llp.Yield),
        "N_LLP_total": n_llp_total,
        "epsilon_polar": epsilon_polar,
        "epsilon_azimuthal": epsilon_azimuthal,
        "P_decay_averaged": p_decay_averaged,
        "visible_Br": br_visible,
        "N_events": n_events,
        "passes_event_cut": n_events >= EVENT_THRESHOLD,
        "sampled_inside_volume": final_events,
    }


# ---------------------------------------------------------------------
# Crossing refinement
# ---------------------------------------------------------------------

def refine_threshold_crossing(
    evaluate,
    ctau_left: float,
    ctau_right: float,
) -> float:
    """
    Refine a bracket containing a change across N_events = threshold.

    The midpoint is geometric because c*tau is scanned logarithmically.
    """
    row_left = evaluate(ctau_left)
    row_right = evaluate(ctau_right)

    state_left = bool(row_left["passes_event_cut"])
    state_right = bool(row_right["passes_event_cut"])

    if state_left == state_right:
        raise ValueError(
            "The supplied interval does not bracket a threshold crossing."
        )

    left = float(ctau_left)
    right = float(ctau_right)

    for _ in range(BISECTION_STEPS):
        middle = np.sqrt(left * right)
        row_middle = evaluate(middle)
        state_middle = bool(row_middle["passes_event_cut"])

        if state_middle == state_left:
            left = middle
        else:
            right = middle

    # Central estimate of the remaining logarithmic bracket.
    return float(np.sqrt(left * right))


def intervals_from_crossings(
    ctau_min: float,
    crossings: list[float],
    starts_above_threshold: bool,
) -> list[tuple[float, float | None]]:
    """
    Convert threshold crossings into intervals satisfying N_events >= 10.

    An upper value of None means that the interval extends beyond
    MAX_CTAU_M.
    """
    intervals = []

    state = starts_above_threshold
    interval_start = ctau_min if state else None

    for crossing in sorted(crossings):
        if state:
            intervals.append((float(interval_start), float(crossing)))
            interval_start = None
        else:
            interval_start = float(crossing)

        state = not state

    if state:
        intervals.append((float(interval_start), None))

    return intervals



def scan_model_mass(
    *,
    model_name: str,
    llp: LLP,
    m_a: float,
    seed: int,
) -> tuple[list[dict], list[tuple[float, float | None]]]:
    ctau_min = lower_ctau_limit(m_a)

    kin, br_visible = prepare_kinematic_sample(
        llp=llp,
        m_a=m_a,
        ctau_min=ctau_min,
        seed=seed,
    )

    cache: dict[float, dict] = {}

    def evaluate(ctau: float) -> dict:
        # The float itself is safe as a cache key here because identical
        # values are passed internally during each bisection.
        key = float(ctau)

        if key not in cache:
            cache[key] = evaluate_ctau(
                model_name=model_name,
                llp=llp,
                kin=kin,
                m_a=m_a,
                ctau=key,
                ctau_min=ctau_min,
                br_visible=br_visible,
                true_sample_seed=seed + 1,
            )

        return cache[key]

    coarse_ctaus = make_coarse_ctau_grid(
        ctau_min=ctau_min,
        ctau_max=MAX_CTAU_M,
        factor=COARSE_FACTOR,
    )

    coarse_rows = [evaluate(ctau) for ctau in coarse_ctaus]
    
    coarse_rates = np.array([row["N_events"] for row in coarse_rows])
    relative_increase = np.diff(coarse_rates) / np.maximum(coarse_rates[:-1], 1.0e-300)
    if np.any(relative_increase > 1.0e-3):
        indices = np.where(relative_increase > 1.0e-3)[0]
        print(
            f"WARNING: non-monotonic event rate for "
            f"{model_name}, m_a={m_a:g} GeV at "
            f"indices {indices.tolist()}."
        )

    crossings = []

    for left_row, right_row in zip(
        coarse_rows[:-1],
        coarse_rows[1:],
    ):
        left_passes = bool(left_row["passes_event_cut"])
        right_passes = bool(right_row["passes_event_cut"])

        if left_passes != right_passes:
            crossing = refine_threshold_crossing(
                evaluate=evaluate,
                ctau_left=left_row["ctau_m"],
                ctau_right=right_row["ctau_m"],
            )
            crossings.append(crossing)

            # Add the central crossing estimate to the stored output.
            evaluate(crossing)

    starts_above = bool(coarse_rows[0]["passes_event_cut"])

    intervals = intervals_from_crossings(
        ctau_min=ctau_min,
        crossings=crossings,
        starts_above_threshold=starts_above,
    )

    if len(crossings) > 1:
        print(
            f"WARNING: {model_name}, m={m_a} GeV has "
            f"{len(crossings)} threshold crossings. "
            "Inspect the curve for Monte Carlo fluctuations or "
            "non-monotonic behaviour."
        )

    rows = sorted(
        cache.values(),
        key=lambda row: row["ctau_m"],
    )

    return rows, intervals


# ---------------------------------------------------------------------
# Interval intersection
# ---------------------------------------------------------------------

def intersect_intervals(
    intervals_a: list[tuple[float, float | None]],
    intervals_b: list[tuple[float, float | None]],
) -> list[tuple[float, float | None]]:
    """Find lifetimes accepted by both models."""
    intersections = []

    for lower_a, upper_a in intervals_a:
        for lower_b, upper_b in intervals_b:
            lower = max(lower_a, lower_b)

            numerical_upper_a = (
                np.inf if upper_a is None else upper_a
            )
            numerical_upper_b = (
                np.inf if upper_b is None else upper_b
            )

            upper = min(numerical_upper_a, numerical_upper_b)

            if lower < upper:
                intersections.append(
                    (
                        float(lower),
                        None if np.isinf(upper) else float(upper),
                    )
                )

    return intersections


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def format_interval(
    interval: tuple[float, float | None],
) -> str:
    lower, upper = interval

    if upper is None:
        return f"[{lower:.6g}, > {MAX_CTAU_M:.6g}] m"

    return f"[{lower:.6g}, {upper:.6g}] m"


def plot_all_scans(dataframe: pd.DataFrame) -> None:
    """
    Plot all masses and both models in the same figure.

    Colour identifies the ALP mass.
    Line style identifies the model.
    """
    figure, axis = plt.subplots(figsize=(9.0, 6.0))

    mass_values = sorted(dataframe["mass_GeV"].unique())
    colours = plt.cm.viridis(
        np.linspace(0.05, 0.95, len(mass_values))
    )

    model_styles = {
        "ALP-photon-primary": "-",
        "ALP-SU2L": "--",
    }

    for colour, m_a in zip(colours, mass_values):
        mass_data = dataframe[
            np.isclose(dataframe["mass_GeV"], m_a)
        ]

        for model_name, model_data in mass_data.groupby("model"):
            model_data = model_data.sort_values("ctau_m")

            axis.loglog(
                model_data["ctau_m"],
                model_data["N_events"],
                linestyle=model_styles.get(model_name, "-"),
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                color=colour,
                label=(
                    rf"$m_a={m_a:g}\,\mathrm{{GeV}}$, "
                    f"{model_name}"
                ),
            )

    axis.axhline(
        EVENT_THRESHOLD,
        linestyle=":",
        linewidth=1.3,
        color="black",
        label=r"$N_{\rm events}=10$",
    )

    axis.set_xlabel(r"$c\tau$ [m]")
    axis.set_ylabel(r"$N_{\rm events}$")
    axis.set_title(
        r"Event rate versus lifetime for ALP-photon and "
        r"ALP-$SU(2)_L$"
    )

    axis.grid(True, which="both", alpha=0.25)
    axis.legend(
        fontsize=8,
        ncol=2,
    )

    figure.tight_layout()

    output_path = OUTPUT_DIR / "ctau_scan_all_masses.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    llp_models = {
        model_name: make_llp(config)
        for model_name, config in MODEL_CONFIGS.items()
    }

    all_rows = []
    model_intervals = {}

    for model_index, (model_name, llp) in enumerate(
        llp_models.items()
    ):
        print(f"\nScanning {model_name}")

        for mass_index, m_a in enumerate(MASSES_GEV):
            seed = (
                BASE_SEED
                + 10_000 * model_index
                + 100 * mass_index
            )

            print(
                f"  m_a = {m_a:g} GeV, "
                f"c_tau_min = {lower_ctau_limit(m_a):g} m"
            )

            rows, intervals = scan_model_mass(
                model_name=model_name,
                llp=llp,
                m_a=float(m_a),
                seed=seed,
            )

            all_rows.extend(rows)
            model_intervals[(model_name, float(m_a))] = intervals

            if intervals:
                for interval in intervals:
                    print(
                        "    N_events >= 10 for "
                        f"{format_interval(interval)}"
                    )
            else:
                print(
                    "    No lifetime interval with "
                    "N_events >= 10 was found."
                )

    results = pd.DataFrame(all_rows)
    results.sort_values(
        ["mass_GeV", "model", "ctau_m"],
        inplace=True,
    )

    results_path = OUTPUT_DIR / "ctau_scan.csv"
    results.to_csv(results_path, index=False)

    summary_rows = []

    print("\nCommon ranges for both models")

    photon_name = "ALP-photon-primary"
    su2l_name = "ALP-SU2L"

    for m_a in MASSES_GEV:
        photon_intervals = model_intervals[
            (photon_name, float(m_a))
        ]
        su2l_intervals = model_intervals[
            (su2l_name, float(m_a))
        ]

        common_intervals = intersect_intervals(
            photon_intervals,
            su2l_intervals,
        )

        print(f"\n  m_a = {m_a:g} GeV")

        if not common_intervals:
            print("    No common interval.")
        else:
            for lower, upper in common_intervals:
                print(
                    f"    {format_interval((lower, upper))}"
                )

                summary_rows.append(
                    {
                        "mass_GeV": m_a,
                        "ctau_lower_m": lower,
                        "ctau_upper_m": (
                            np.nan if upper is None else upper
                        ),
                        "upper_extends_beyond_scan": upper is None,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "common_ctau_ranges.csv"
    summary.to_csv(summary_path, index=False)

    plot_all_scans(results)

    plot_path = OUTPUT_DIR / "ctau_scan_all_masses.png"

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()