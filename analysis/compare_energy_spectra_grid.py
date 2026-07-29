from pathlib import Path

import numpy as np
import pandas as pd

try:
    from analysis.compare_energy_spectra import (
        BASE_SEED,
        MODEL_CONFIGS,
        NUMBER_OF_ENERGY_BINS,
        calculate_model_spectrum,
        numerical_summary,
        plot_spectra,
    )
except ModuleNotFoundError:
    from compare_energy_spectra import (
        BASE_SEED,
        MODEL_CONFIGS,
        NUMBER_OF_ENERGY_BINS,
        calculate_model_spectrum,
        numerical_summary,
        plot_spectra,
    )


ANALYSIS_DIR = Path(__file__).resolve().parent
CTAU_PATH = ANALYSIS_DIR / "ctau_scan" / "common_ctau_ranges.csv"
STABILITY_PATH = ANALYSIS_DIR / "energy_spectra_lifetime_scan" / "lifetime_stability_summary.csv"
OUTPUT_DIR = ANALYSIS_DIR / "energy_spectra_grid"
PLOT_DIR = OUTPUT_DIR / "plots"

MINIMUM_EVENTS = 10.0
MAXIMUM_TV_DISTANCE = 1.0e-2
ENERGY_MAX_GEV = 400.0


def load_benchmarks() -> pd.DataFrame:
    """Load and validate the common lifetime intervals."""
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

    flag_text = data["upper_extends_beyond_scan"].astype(str).str.strip().str.lower()
    valid_flags = flag_text.isin(["true", "false"])

    if not valid_flags.all():
        raise ValueError("upper_extends_beyond_scan must contain " "only True or False.")

    data["upper_extends_beyond_scan"] = flag_text == "true"

    if data["upper_extends_beyond_scan"].any():
        raise RuntimeError(
            "At least one common interval extends beyond "
            "the lifetime scan. A finite logarithmic "
            "midpoint cannot be chosen automatically."
        )

    numeric_columns = [
        "mass_GeV",
        "ctau_lower_m",
        "ctau_upper_m",
    ]

    if not np.all(np.isfinite(data[numeric_columns].to_numpy(dtype=float))):
        raise ValueError("The benchmark table contains non-finite " "mass or lifetime values.")

    if data[numeric_columns].le(0.0).any().any():
        raise ValueError("Masses and lifetimes must be positive.")

    if np.any(data["ctau_upper_m"] <= data["ctau_lower_m"]):
        raise ValueError("Every upper lifetime must exceed " "the corresponding lower lifetime.")

    return data.sort_values("mass_GeV").reset_index(drop=True)


def select_stable_benchmarks(
    common_ranges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the smallest sampled lifetime at each mass for which
    both models satisfy the event-rate and lifetime-stability
    requirements.
    """
    if not STABILITY_PATH.exists():
        raise FileNotFoundError(
            "Lifetime-stability summary not found:\n"
            f"{STABILITY_PATH}\n"
            "Run compare_energy_spectra_lifetime_scan.py first."
        )

    stability_data = pd.read_csv(STABILITY_PATH)

    required_columns = {
        "model",
        "mass_GeV",
        "ctau_m",
        "N_events",
        "total_variation_distance_to_reference",
    }

    missing_columns = required_columns - set(stability_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in lifetime-stability summary: {sorted(missing_columns)}"
        )

    numeric_columns = [
        "mass_GeV",
        "ctau_m",
        "N_events",
        "total_variation_distance_to_reference",
    ]

    for column in numeric_columns:
        stability_data[column] = pd.to_numeric(
            stability_data[column],
            errors="raise",
        )

    if not np.all(np.isfinite(stability_data[numeric_columns].to_numpy(dtype=float))):
        raise ValueError("The lifetime-stability summary contains " "non-finite numerical values.")

    expected_models = set(MODEL_CONFIGS)

    available_models = set(stability_data["model"])

    missing_models = expected_models - available_models

    if missing_models:
        raise ValueError(
            f"The lifetime-stability summary is missing the models: {sorted(missing_models)}"
        )

    stable_points = stability_data.loc[
        stability_data["model"].isin(expected_models)
        & (stability_data["N_events"] >= MINIMUM_EVENTS)
        & (stability_data["total_variation_distance_to_reference"] <= MAXIMUM_TV_DISTANCE)
    ].copy()

    if stable_points.empty:
        raise RuntimeError(
            "No sampled lifetime satisfies both " "the event-rate and stability requirements."
        )

    # Use a rounded key so that masses read from separate CSV files
    # can be matched robustly.
    stable_points["_mass_key"] = stable_points["mass_GeV"].round(12)

    model_counts = stable_points.groupby(
        [
            "_mass_key",
            "ctau_m",
        ]
    )["model"].nunique()

    common_stable_points = model_counts[model_counts == len(expected_models)].reset_index()

    if common_stable_points.empty:
        raise RuntimeError(
            "No mass-lifetime point satisfies the " "requirements simultaneously for both models."
        )

    # At each mass, select the smallest lifetime that is stable
    # and has at least ten events for both models.
    selected_lifetimes = (
        common_stable_points.sort_values(
            [
                "_mass_key",
                "ctau_m",
            ]
        )
        .groupby(
            "_mass_key",
            as_index=False,
        )
        .first()
        .rename(
            columns={
                "ctau_m": "ctau_benchmark_m",
            }
        )
    )

    benchmarks = common_ranges.copy()

    benchmarks["_mass_key"] = benchmarks["mass_GeV"].round(12)

    benchmarks = benchmarks.merge(
        selected_lifetimes[
            [
                "_mass_key",
                "ctau_benchmark_m",
            ]
        ],
        on="_mass_key",
        how="inner",
        validate="one_to_one",
    )

    benchmarks = benchmarks.drop(columns="_mass_key")

    if benchmarks.empty:
        raise RuntimeError(
            "No masses are present in both the common-lifetime "
            "table and the lifetime-stability summary."
        )

    lifetime_outside_interval = (benchmarks["ctau_benchmark_m"] < benchmarks["ctau_lower_m"]) | (
        benchmarks["ctau_benchmark_m"] > benchmarks["ctau_upper_m"]
    )

    if lifetime_outside_interval.any():
        invalid_rows = benchmarks.loc[
            lifetime_outside_interval,
            [
                "mass_GeV",
                "ctau_lower_m",
                "ctau_benchmark_m",
                "ctau_upper_m",
            ],
        ]

        raise RuntimeError(
            "A selected benchmark lies outside its common "
            "N_events >= 10 interval:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    return benchmarks.sort_values("mass_GeV").reset_index(drop=True)


def print_benchmark_selection_report(
    common_ranges: pd.DataFrame,
    selected_benchmarks: pd.DataFrame,
) -> None:
    """
    Print whether each mass has a common sampled benchmark
    satisfying the event-rate and stability requirements.
    """
    stability_data = pd.read_csv(STABILITY_PATH)

    selected_by_mass = {
        round(float(row.mass_GeV), 12): float(row.ctau_benchmark_m)
        for row in selected_benchmarks.itertuples(index=False)
    }

    print()
    print("=" * 76)
    print("Benchmark-selection report")
    print("=" * 76)

    for row in common_ranges.itertuples(index=False):
        mass_gev = float(row.mass_GeV)

        mass_key = round(
            mass_gev,
            12,
        )

        mass_data = stability_data.loc[
            np.isclose(
                stability_data["mass_GeV"],
                mass_gev,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]

        if mass_key in selected_by_mass:
            print(
                f"m_a = {mass_gev:g} GeV: "
                f"selected c_tau = "
                f"{selected_by_mass[mass_key]:.6g} m"
            )
            continue

        print(f"m_a = {mass_gev:g} GeV: " "no common sampled benchmark")

        for model_name in MODEL_CONFIGS:
            model_data = mass_data.loc[mass_data["model"] == model_name].copy()

            event_allowed = model_data.loc[model_data["N_events"] >= MINIMUM_EVENTS]

            if event_allowed.empty:
                print(
                    f"  {model_name}: no sampled point with N_events >= {MINIMUM_EVENTS:g}"
                )
                continue

            best_row = event_allowed.loc[
                event_allowed["total_variation_distance_to_reference"].idxmin()
            ]

            print(
                f"  {model_name}: best available "
                f"D_TV = "
                f"{best_row['total_variation_distance_to_reference']:.6g} "
                f"at c_tau = "
                f"{best_row['ctau_m']:.6g} m, "
                f"N_events = "
                f"{best_row['N_events']:.6g}"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    common_ranges = load_benchmarks()

    benchmarks = select_stable_benchmarks(common_ranges)

    print_benchmark_selection_report(
        common_ranges,
        benchmarks,
    )

    summary_frames = []
    plot_paths = []

    for mass_index, row in enumerate(benchmarks.itertuples(index=False)):
        mass_gev = float(row.mass_GeV)

        ctau_lower_m = float(row.ctau_lower_m)

        ctau_upper_m = float(row.ctau_upper_m)

        ctau_benchmark_m = float(row.ctau_benchmark_m)

        print()
        print("=" * 70)
        print(f"Mass: {mass_gev:g} GeV")
        print(f"Common lifetime interval: [{ctau_lower_m:.6g}, {ctau_upper_m:.6g}] m")
        print(f"Chosen benchmark lifetime: {ctau_benchmark_m:.6g} m")
        print(
            "Selection requirements: "
            f"N_events >= {MINIMUM_EVENTS:g}, "
            f"D_TV <= {MAXIMUM_TV_DISTANCE:g}"
        )
        print("=" * 70)

        energy_edges = np.geomspace(
            mass_gev,
            ENERGY_MAX_GEV,
            NUMBER_OF_ENERGY_BINS + 1,
        )

        spectra = {}

        for model_index, (
            model_name,
            config,
        ) in enumerate(MODEL_CONFIGS.items()):
            seed = BASE_SEED + 10_000 * mass_index + 100 * model_index

            spectra[model_name] = calculate_model_spectrum(
                model_name=model_name,
                config=config,
                mass_gev=mass_gev,
                ctau_m=ctau_benchmark_m,
                energy_edges=energy_edges,
                seed=seed,
            )

        plot_path = plot_spectra(
            spectra,
            mass_gev=mass_gev,
            ctau_m=ctau_benchmark_m,
            output_dir=PLOT_DIR,
        )

        plot_paths.append(plot_path)

        mass_summary = numerical_summary(
            spectra,
            mass_gev=mass_gev,
            ctau_m=ctau_benchmark_m,
        )

        mass_summary["ctau_lower_m"] = ctau_lower_m

        mass_summary["ctau_upper_m"] = ctau_upper_m

        mass_summary["benchmark_ctau_m"] = ctau_benchmark_m

        mass_summary["minimum_required_events"] = MINIMUM_EVENTS

        mass_summary["maximum_allowed_tv_distance"] = MAXIMUM_TV_DISTANCE

        summary_frames.append(mass_summary)

        print()
        print("Event rates:")

        for model_name, spectrum in spectra.items():
            print(f"  {model_name}: N_events = {spectrum['n_events']:.6g}")

        print(f"Plot saved to: {plot_path}")

    full_summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    summary_path = OUTPUT_DIR / "energy_spectra_grid_summary.csv"

    full_summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"Combined summary saved to: {summary_path}")

    print()
    print("=" * 70)
    print("Grid analysis finished.")
    print(f"Number of benchmark plots: {len(plot_paths)}")


if __name__ == "__main__":
    main()
