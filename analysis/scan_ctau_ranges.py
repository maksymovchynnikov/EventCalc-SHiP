"""Scan ECAL-accepted observable ALP lifetime ranges for the final mass study.

The scan uses the same EventCalc source preparation, deterministic seed
hierarchy, event weights, and two-photon ECAL requirement as
``analysis.lifetime_blind_discrimination``.  The threshold therefore applies
to the detector-level rate that is actually used to build the likelihood
templates:

    N_events(after ECAL) >= 10.

Run from the repository root with

    python -m analysis.scan_ctau_ranges

The default masses stop at 1.20 GeV.  A previous mother-level scan already
showed that m_a >= 1.25 GeV does not reach ten events even before applying the
ECAL requirement, so those masses cannot become observable after ECAL.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from analysis.ECAL import DEFAULT_ECAL, diphoton_ecal_acceptance
from analysis.compare_energy_spectra import (
    BASE_SEED,
    MODEL_CONFIGS,
    N_POT,
    RESAMPLE_SIZE,
)
from analysis.lifetime_blind_discrimination import prepare_model_sources


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MASSES_GEV = np.array(
    [
        0.30,
        0.40,
        0.50,
        0.60,
        0.75,
        0.90,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
        1.30,
        1.40,
        1.50,
    ],
    dtype=float,
)

PHOTON_MINIMUM_TABLE_MASS_GEV = 0.02
PHOTON_MAXIMUM_TABLE_MASS_GEV = 4.0
SU2_MINIMUM_TABLE_MASS_GEV = 0.01
SU2_MAXIMUM_TABLE_MASS_GEV = 5.1
COMMON_MINIMUM_TABLE_MASS_GEV = max(
    PHOTON_MINIMUM_TABLE_MASS_GEV,
    SU2_MINIMUM_TABLE_MASS_GEV,
)
COMMON_MAXIMUM_TABLE_MASS_GEV = min(
    PHOTON_MAXIMUM_TABLE_MASS_GEV,
    SU2_MAXIMUM_TABLE_MASS_GEV,
)

MODEL_NAMES = (
    "ALP-photon-combined",
    "ALP-SU2L",
)

EVENT_THRESHOLD = 10.0
MAX_CTAU_M = 1.0e3
COARSE_FACTOR = 1.7
BISECTION_STEPS = 14
MAXIMUM_ALLOWED_RELATIVE_INCREASE = 2.0e-3

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "ctau_scan"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


# -----------------------------------------------------------------------------
# CLI and basic definitions
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan model-specific ECAL-accepted N_events >= 10 lifetime "
            "intervals for the combined photophilic and ALP-SU(2)_L models."
        )
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Masses in GeV. By default, scan the final candidate grid from "
            "0.30 to 1.20 GeV."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-ctau",
        type=float,
        default=MAX_CTAU_M,
        help=f"Largest scanned proper lifetime in metres (default: {MAX_CTAU_M:g}).",
    )
    return parser.parse_args()


def resolve_masses(requested: list[float] | None) -> np.ndarray:
    masses = (
        DEFAULT_MASSES_GEV.copy()
        if requested is None
        else np.asarray(requested, dtype=float)
    )
    masses = np.unique(masses)

    if masses.ndim != 1 or len(masses) == 0:
        raise ValueError("At least one mass is required.")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("All masses must be finite and positive.")
    if np.any(masses < COMMON_MINIMUM_TABLE_MASS_GEV - 1.0e-12):
        raise ValueError(
            "A requested mass lies below the common EventCalc table range: "
            f"{COMMON_MINIMUM_TABLE_MASS_GEV:g} GeV."
        )
    if np.any(masses > COMMON_MAXIMUM_TABLE_MASS_GEV + 1.0e-12):
        raise ValueError(
            "A requested mass lies above the common EventCalc table range. "
            f"The photophilic table ends at {PHOTON_MAXIMUM_TABLE_MASS_GEV:g} GeV, "
            f"while the SU(2)_L table ends at {SU2_MAXIMUM_TABLE_MASS_GEV:g} GeV."
        )

    return np.sort(masses.astype(float))


def lower_ctau_limit(mass_gev: float) -> float:
    """Use the Week-6 lower bound c*tau = 3 m * (m_a / 0.3 GeV)."""
    return 3.0 * (mass_gev / 0.3)


def make_coarse_ctau_grid(
    *,
    ctau_min: float,
    ctau_max: float,
    factor: float,
) -> np.ndarray:
    if ctau_min <= 0.0:
        raise ValueError("ctau_min must be positive.")
    if ctau_max <= ctau_min:
        raise ValueError("ctau_max must exceed ctau_min.")
    if factor <= 1.0:
        raise ValueError("The coarse-grid factor must exceed one.")

    values = [float(ctau_min)]
    while values[-1] < ctau_max:
        next_value = min(values[-1] * factor, ctau_max)
        if next_value <= values[-1]:
            break
        values.append(next_value)
    return np.asarray(values, dtype=float)


# -----------------------------------------------------------------------------
# Detector-level event-rate evaluation
# -----------------------------------------------------------------------------


def evaluate_prepared_source_allow_empty(
    prepared,
    *,
    model_name: str,
    mass_gev: float,
    ctau_m: float,
) -> dict:
    """Evaluate one source exactly as the bank builder, allowing zero ECAL events."""
    if ctau_m <= 0.0:
        raise ValueError("c*tau must be positive.")

    llp = prepared.llp
    kinematics = prepared.kinematics

    llp.set_c_tau(float(ctau_m))
    coupling_squared = float(llp.c_tau_int / ctau_m)
    n_llp_total = N_POT * float(llp.Yield) * coupling_squared

    kinematics.c_tau = float(ctau_m)
    np.random.seed(prepared.true_sample_seed)
    kinematics.true_samples(False)
    results = np.asarray(kinematics.get_kinematics(), dtype=float)

    if results.ndim != 2 or results.shape[1] <= 6:
        raise RuntimeError(
            f"Invalid EventCalc output for {model_name}, {prepared.source_label}, "
            f"m_a={mass_gev:g} GeV, c*tau={ctau_m:g} m."
        )

    valid = (
        np.isfinite(results[:, 3])
        & np.isfinite(results[:, 6])
        & (results[:, 6] >= 0.0)
    )
    results = results[valid]

    epsilon_polar = float(kinematics.epsilon_polar)
    event_weight_scale = (
        n_llp_total
        * epsilon_polar
        * prepared.visible_branching_ratio
        / RESAMPLE_SIZE
    )

    if len(results) == 0:
        return {
            "source_label": prepared.source_label,
            "coupling_squared": coupling_squared,
            "n_events": 0.0,
            "n_events_before_ecal": 0.0,
            "epsilon_ecal_weighted": 0.0,
            "number_of_valid_mothers": 0,
            "number_passing_ecal": 0,
        }

    decay_probabilities = np.asarray(results[:, 6], dtype=float)
    event_weights_before_ecal = event_weight_scale * decay_probabilities
    n_events_before_ecal = float(np.sum(event_weights_before_ecal))

    ecal_mask = diphoton_ecal_acceptance(
        results,
        geometry=DEFAULT_ECAL,
        seed=prepared.ecal_seed,
    )
    n_events = float(np.sum(event_weights_before_ecal[ecal_mask]))

    return {
        "source_label": prepared.source_label,
        "coupling_squared": coupling_squared,
        "n_events": n_events,
        "n_events_before_ecal": n_events_before_ecal,
        "epsilon_ecal_weighted": (
            n_events / n_events_before_ecal
            if n_events_before_ecal > 0.0
            else 0.0
        ),
        "number_of_valid_mothers": int(len(results)),
        "number_passing_ecal": int(np.count_nonzero(ecal_mask)),
    }


def scan_model_mass(
    *,
    model_name: str,
    mass_gev: float,
    mass_index: int,
    model_index: int,
    maximum_ctau_m: float,
) -> tuple[list[dict], list[tuple[float, float | None]]]:
    """Scan one model and mass using the exact template-generation seed hierarchy."""
    ctau_min = lower_ctau_limit(mass_gev)
    model_seed = BASE_SEED + 10_000 * mass_index + 100 * model_index

    prepared_sources = prepare_model_sources(
        model_name=model_name,
        mass_gev=mass_gev,
        preparation_ctau_m=ctau_min,
        seed=model_seed,
    )

    cache: dict[float, dict] = {}

    def evaluate(ctau_m: float) -> dict:
        key = float(ctau_m)
        if key in cache:
            return cache[key]

        source_results = {
            label: evaluate_prepared_source_allow_empty(
                prepared,
                model_name=model_name,
                mass_gev=mass_gev,
                ctau_m=key,
            )
            for label, prepared in prepared_sources.items()
        }

        n_events_by_source = {
            label: float(result["n_events"])
            for label, result in source_results.items()
        }
        before_by_source = {
            label: float(result["n_events_before_ecal"])
            for label, result in source_results.items()
        }
        n_events = float(sum(n_events_by_source.values()))
        n_events_before_ecal = float(sum(before_by_source.values()))

        coupling_values = np.asarray(
            [result["coupling_squared"] for result in source_results.values()],
            dtype=float,
        )
        if not np.allclose(
            coupling_values,
            coupling_values[0],
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RuntimeError(
                f"{model_name}, m_a={mass_gev:g} GeV: source coupling "
                "normalizations disagree."
            )

        row = {
            "model": model_name,
            "mass_GeV": float(mass_gev),
            "ctau_m": key,
            "ctau_min_m": float(ctau_min),
            "coupling_squared": float(coupling_values[0]),
            "N_events": n_events,
            "N_events_before_ECAL": n_events_before_ecal,
            "epsilon_ECAL_weighted": (
                n_events / n_events_before_ecal
                if n_events_before_ecal > 0.0
                else 0.0
            ),
            "passes_event_cut": bool(n_events >= EVENT_THRESHOLD),
            "N_events_primary": n_events_by_source.get("primary", np.nan),
            "N_events_cascade": n_events_by_source.get("cascade", np.nan),
            "N_events_inclusive": n_events_by_source.get("inclusive", np.nan),
            "N_events_before_ECAL_primary": before_by_source.get("primary", np.nan),
            "N_events_before_ECAL_cascade": before_by_source.get("cascade", np.nan),
            "N_events_before_ECAL_inclusive": before_by_source.get("inclusive", np.nan),
            "valid_mother_samples": int(
                sum(result["number_of_valid_mothers"] for result in source_results.values())
            ),
            "samples_passing_ECAL": int(
                sum(result["number_passing_ecal"] for result in source_results.values())
            ),
            "template_model_seed": int(model_seed),
        }
        cache[key] = row
        return row

    coarse_ctaus = make_coarse_ctau_grid(
        ctau_min=ctau_min,
        ctau_max=maximum_ctau_m,
        factor=COARSE_FACTOR,
    )
    coarse_rows = [evaluate(ctau) for ctau in coarse_ctaus]
    coarse_rates = np.asarray([row["N_events"] for row in coarse_rows], dtype=float)

    relative_increase = np.diff(coarse_rates) / np.maximum(
        coarse_rates[:-1],
        1.0e-300,
    )
    if np.any(relative_increase > MAXIMUM_ALLOWED_RELATIVE_INCREASE):
        indices = np.flatnonzero(
            relative_increase > MAXIMUM_ALLOWED_RELATIVE_INCREASE
        )
        details = ", ".join(
            (
                f"{coarse_ctaus[index]:.6g}->{coarse_ctaus[index + 1]:.6g} m: "
                f"{coarse_rates[index]:.6g}->{coarse_rates[index + 1]:.6g}"
            )
            for index in indices
        )
        raise RuntimeError(
            f"The ECAL-accepted event-rate curve is not monotonically decreasing "
            f"for {model_name}, m_a={mass_gev:g} GeV: {details}"
        )

    crossings: list[float] = []
    states = coarse_rates >= EVENT_THRESHOLD
    for index in range(len(coarse_ctaus) - 1):
        if states[index] == states[index + 1]:
            continue
        crossings.append(
            refine_threshold_crossing(
                evaluate,
                coarse_ctaus[index],
                coarse_ctaus[index + 1],
            )
        )

    intervals = intervals_from_crossings(
        ctau_min=ctau_min,
        crossings=crossings,
        starts_above_threshold=bool(states[0]),
    )

    rows = sorted(cache.values(), key=lambda row: row["ctau_m"])
    return rows, intervals


# -----------------------------------------------------------------------------
# Threshold crossings and interval helpers
# -----------------------------------------------------------------------------


def refine_threshold_crossing(
    evaluate: Callable[[float], dict],
    ctau_left: float,
    ctau_right: float,
) -> float:
    row_left = evaluate(ctau_left)
    row_right = evaluate(ctau_right)
    state_left = bool(row_left["passes_event_cut"])
    state_right = bool(row_right["passes_event_cut"])

    if state_left == state_right:
        raise ValueError("The supplied interval does not bracket a threshold crossing.")

    left = float(ctau_left)
    right = float(ctau_right)
    for _ in range(BISECTION_STEPS):
        middle = float(np.sqrt(left * right))
        state_middle = bool(evaluate(middle)["passes_event_cut"])
        if state_middle == state_left:
            left = middle
        else:
            right = middle

    return float(np.sqrt(left * right))


def intervals_from_crossings(
    *,
    ctau_min: float,
    crossings: list[float],
    starts_above_threshold: bool,
) -> list[tuple[float, float | None]]:
    intervals: list[tuple[float, float | None]] = []
    state = starts_above_threshold
    interval_start: float | None = float(ctau_min) if state else None

    for crossing in sorted(crossings):
        if state:
            if interval_start is None:
                raise RuntimeError("Internal error while closing an interval.")
            intervals.append((float(interval_start), float(crossing)))
            interval_start = None
        else:
            interval_start = float(crossing)
        state = not state

    if state:
        if interval_start is None:
            raise RuntimeError("Internal error while extending an interval.")
        intervals.append((float(interval_start), None))

    return intervals


def intersect_intervals(
    intervals_a: list[tuple[float, float | None]],
    intervals_b: list[tuple[float, float | None]],
) -> list[tuple[float, float | None]]:
    intersections: list[tuple[float, float | None]] = []
    for lower_a, upper_a in intervals_a:
        for lower_b, upper_b in intervals_b:
            lower = max(lower_a, lower_b)
            numerical_upper_a = np.inf if upper_a is None else upper_a
            numerical_upper_b = np.inf if upper_b is None else upper_b
            upper = min(numerical_upper_a, numerical_upper_b)
            if lower < upper:
                intersections.append(
                    (float(lower), None if np.isinf(upper) else float(upper))
                )
    return intersections


def format_interval(
    interval: tuple[float, float | None],
    *,
    maximum_ctau_m: float,
) -> str:
    lower, upper = interval
    if upper is None:
        return f"[{lower:.6g}, > {maximum_ctau_m:.6g}] m"
    return f"[{lower:.6g}, {upper:.6g}] m"


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def build_observable_mass_summary(
    results: pd.DataFrame,
    model_intervals: dict[tuple[str, float], list[tuple[float, float | None]]],
    masses: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    photon_name = "ALP-photon-combined"
    su2_name = "ALP-SU2L"

    for mass_gev in masses:
        photon_data = results.loc[
            np.isclose(results["mass_GeV"], mass_gev)
            & (results["model"] == photon_name)
        ]
        su2_data = results.loc[
            np.isclose(results["mass_GeV"], mass_gev)
            & (results["model"] == su2_name)
        ]
        photon_intervals = model_intervals[(photon_name, float(mass_gev))]
        su2_intervals = model_intervals[(su2_name, float(mass_gev))]
        common = intersect_intervals(photon_intervals, su2_intervals)

        rows.append(
            {
                "mass_GeV": float(mass_gev),
                "photon_maximum_N_events": float(photon_data["N_events"].max()),
                "su2_maximum_N_events": float(su2_data["N_events"].max()),
                "photon_maximum_N_events_before_ECAL": float(
                    photon_data["N_events_before_ECAL"].max()
                ),
                "su2_maximum_N_events_before_ECAL": float(
                    su2_data["N_events_before_ECAL"].max()
                ),
                "photon_has_observable_interval": bool(photon_intervals),
                "su2_has_observable_interval": bool(su2_intervals),
                "both_models_have_observable_interval": bool(
                    photon_intervals and su2_intervals
                ),
                "has_common_observable_interval": bool(common),
            }
        )

    return pd.DataFrame(rows)


def plot_all_scans(dataframe: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(11.0, 7.2))
    mass_values = np.asarray(sorted(dataframe["mass_GeV"].unique()), dtype=float)
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(mass_values)))
    colour_by_mass = dict(zip(mass_values, colours))
    model_styles = {
        "ALP-photon-combined": "-",
        "ALP-SU2L": "--",
    }

    for mass_gev in mass_values:
        mass_data = dataframe.loc[np.isclose(dataframe["mass_GeV"], mass_gev)]
        for model_name, model_data in mass_data.groupby("model"):
            model_data = model_data.sort_values("ctau_m")
            axis.loglog(
                model_data["ctau_m"],
                model_data["N_events"],
                linestyle=model_styles.get(model_name, "-"),
                marker="o",
                markersize=2.2,
                linewidth=1.15,
                color=colour_by_mass[mass_gev],
            )

    axis.axhline(EVENT_THRESHOLD, linestyle=":", linewidth=1.3, color="black")
    axis.set_xlabel(r"$c\tau_a$ [m]")
    axis.set_ylabel(r"ECAL-accepted $N_{\mathrm{events}}$")
    axis.set_title(
        r"ECAL-accepted event rate versus lifetime for ALP-photon and "
        r"ALP-$SU(2)_L$"
    )
    axis.grid(True, which="both", alpha=0.25)

    model_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            label="ALP-photon, primary + cascade",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label=r"ALP-$SU(2)_L$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            label=r"$N_{\rm events}=10$",
        ),
    ]
    first_legend = axis.legend(handles=model_handles, loc="upper right", fontsize=8)
    axis.add_artist(first_legend)

    mass_handles = [
        Line2D(
            [0],
            [0],
            color=colour_by_mass[mass_gev],
            marker="o",
            linestyle="none",
            markersize=4,
            label=rf"${mass_gev:g}$",
        )
        for mass_gev in mass_values
    ]
    axis.legend(
        handles=mass_handles,
        title=r"$m_a$ [GeV]",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7.5,
    )

    figure.tight_layout()
    output_path = OUTPUT_DIR / "ctau_scan_all_masses.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_arguments()
    masses = resolve_masses(args.masses)
    maximum_ctau_m = float(args.max_ctau)
    if not np.isfinite(maximum_ctau_m) or maximum_ctau_m <= float(masses[-1]) * 10.0:
        raise ValueError("--max-ctau must be finite and comfortably above the lower bounds.")

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("ECAL-accepted ALP lifetime-range scan")
    print(f"Masses: {masses.tolist()}")
    print(f"EventCalc resample size: {RESAMPLE_SIZE:,}")
    print(f"N_events threshold after ECAL: {EVENT_THRESHOLD:g}")
    print(f"Maximum scanned lifetime: {maximum_ctau_m:g} m")
    print("Seed hierarchy matches lifetime_blind_discrimination.py.")
    print("=" * 80)

    all_rows: list[dict] = []
    model_intervals: dict[
        tuple[str, float],
        list[tuple[float, float | None]],
    ] = {}

    for model_index, model_name in enumerate(MODEL_NAMES):
        if model_name not in MODEL_CONFIGS:
            raise KeyError(f"Missing MODEL_CONFIGS entry for {model_name}.")
        print(f"\nScanning {model_name}")

        for mass_index, mass_gev in enumerate(masses):
            model_seed = BASE_SEED + 10_000 * mass_index + 100 * model_index
            print(
                f"  m_a={mass_gev:g} GeV, "
                f"c*tau_min={lower_ctau_limit(float(mass_gev)):g} m, "
                f"seed={model_seed}"
            )

            rows, intervals = scan_model_mass(
                model_name=model_name,
                mass_gev=float(mass_gev),
                mass_index=mass_index,
                model_index=model_index,
                maximum_ctau_m=maximum_ctau_m,
            )
            all_rows.extend(rows)
            model_intervals[(model_name, float(mass_gev))] = intervals

            if intervals:
                for interval in intervals:
                    print(
                        "    ECAL-accepted N_events >= 10 for "
                        + format_interval(
                            interval,
                            maximum_ctau_m=maximum_ctau_m,
                        )
                    )
            else:
                print("    No ECAL-accepted interval with N_events >= 10.")

    results = pd.DataFrame(all_rows).sort_values(
        ["mass_GeV", "model", "ctau_m"],
        ignore_index=True,
    )
    results_path = OUTPUT_DIR / "ctau_scan.csv"
    results.to_csv(results_path, index=False)

    photon_name = "ALP-photon-combined"
    su2_name = "ALP-SU2L"
    common_rows: list[dict] = []

    print("\nCommon ECAL-accepted ranges for both models")
    for mass_gev in masses:
        common_intervals = intersect_intervals(
            model_intervals[(photon_name, float(mass_gev))],
            model_intervals[(su2_name, float(mass_gev))],
        )
        print(f"\n  m_a={mass_gev:g} GeV")
        if not common_intervals:
            print("    No common interval.")
        else:
            for lower, upper in common_intervals:
                print(
                    "    "
                    + format_interval(
                        (lower, upper),
                        maximum_ctau_m=maximum_ctau_m,
                    )
                )
                common_rows.append(
                    {
                        "mass_GeV": float(mass_gev),
                        "ctau_lower_m": lower,
                        "ctau_upper_m": np.nan if upper is None else upper,
                        "upper_extends_beyond_scan": upper is None,
                    }
                )

    common_summary = pd.DataFrame(
        common_rows,
        columns=[
            "mass_GeV",
            "ctau_lower_m",
            "ctau_upper_m",
            "upper_extends_beyond_scan",
        ],
    )
    common_path = OUTPUT_DIR / "common_ctau_ranges.csv"
    common_summary.to_csv(common_path, index=False)

    mass_summary = build_observable_mass_summary(
        results,
        model_intervals,
        masses,
    )
    mass_summary_path = OUTPUT_DIR / "observable_mass_summary.csv"
    mass_summary.to_csv(mass_summary_path, index=False)

    plot_path = plot_all_scans(results)

    valid_masses = mass_summary.loc[
        mass_summary["both_models_have_observable_interval"],
        "mass_GeV",
    ].tolist()

    truncated = [
        (model_name, mass_gev)
        for (model_name, mass_gev), intervals in model_intervals.items()
        if any(upper is None for _, upper in intervals)
    ]

    print("\nMasses with a model-specific ECAL-accepted interval for both hypotheses:")
    print(valid_masses)

    if truncated:
        print("\nWARNING: at least one accepted interval reaches --max-ctau:")
        for model_name, mass_gev in truncated:
            print(f"  {model_name}, m_a={mass_gev:g} GeV")
        print("Increase --max-ctau before building the final template banks.")

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {common_path}")
    print(f"  {mass_summary_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()