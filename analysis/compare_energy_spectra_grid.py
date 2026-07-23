from pathlib import Path

import numpy as np
import pandas as pd

from compare_energy_spectra import (
    BASE_SEED,
    MODEL_CONFIGS,
    NUMBER_OF_ENERGY_BINS,
    calculate_model_spectrum,
    numerical_summary,
    plot_spectra,
)


ANALYSIS_DIR = Path(__file__).resolve().parent

CTAU_PATH = (
    ANALYSIS_DIR
    / "ctau_scan"
    / "common_ctau_ranges.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "energy_spectra_grid"
)

PLOT_DIR = OUTPUT_DIR / "plots"

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

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in common lifetime table: "
            f"{sorted(missing_columns)}"
        )

    flag_text = (
        data["upper_extends_beyond_scan"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_flags = flag_text.isin(
        ["true", "false"]
    )

    if not valid_flags.all():
        raise ValueError(
            "upper_extends_beyond_scan must contain "
            "only True or False."
        )

    data["upper_extends_beyond_scan"] = (
        flag_text == "true"
    )

    if data[
        "upper_extends_beyond_scan"
    ].any():
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

    if not np.all(
        np.isfinite(
            data[numeric_columns].to_numpy(
                dtype=float
            )
        )
    ):
        raise ValueError(
            "The benchmark table contains non-finite "
            "mass or lifetime values."
        )

    if (
        data[numeric_columns]
        .le(0.0)
        .any()
        .any()
    ):
        raise ValueError(
            "Masses and lifetimes must be positive."
        )

    if np.any(
        data["ctau_upper_m"]
        <= data["ctau_lower_m"]
    ):
        raise ValueError(
            "Every upper lifetime must exceed "
            "the corresponding lower lifetime."
        )

    return data.sort_values(
        "mass_GeV"
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmarks = load_benchmarks()

    summary_frames = []
    plot_paths = []

    for mass_index, row in enumerate(
        benchmarks.itertuples(index=False)
    ):
        mass_gev = float(
            row.mass_GeV
        )

        ctau_lower_m = float(
            row.ctau_lower_m
        )

        ctau_upper_m = float(
            row.ctau_upper_m
        )

        # Logarithmic midpoint of the common allowed interval.
        ctau_benchmark_m = float(
            np.sqrt(
                ctau_lower_m
                * ctau_upper_m
            )
        )

        print()
        print("=" * 70)
        print(
            f"Mass: {mass_gev:g} GeV"
        )
        print(
            "Common lifetime interval: "
            f"[{ctau_lower_m:.6g}, "
            f"{ctau_upper_m:.6g}] m"
        )
        print(
            "Chosen benchmark lifetime: "
            f"{ctau_benchmark_m:.6g} m"
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
        ) in enumerate(
            MODEL_CONFIGS.items()
        ):
            seed = (
                BASE_SEED
                + 10_000 * mass_index
                + 100 * model_index
            )

            spectra[model_name] = (
                calculate_model_spectrum(
                    model_name=model_name,
                    config=config,
                    mass_gev=mass_gev,
                    ctau_m=ctau_benchmark_m,
                    energy_edges=energy_edges,
                    seed=seed,
                )
            )

        plot_path = plot_spectra(
            spectra,
            mass_gev=mass_gev,
            ctau_m=ctau_benchmark_m,
            output_dir=PLOT_DIR,
        )

        plot_paths.append(
            plot_path
        )

        mass_summary = numerical_summary(
            spectra,
            mass_gev=mass_gev,
            ctau_m=ctau_benchmark_m,
        )

        mass_summary["ctau_lower_m"] = (
            ctau_lower_m
        )

        mass_summary["ctau_upper_m"] = (
            ctau_upper_m
        )

        summary_frames.append(
            mass_summary
        )

        print()
        print("Event rates:")

        for model_name, spectrum in spectra.items():
            print(
                f"  {model_name}: "
                f"N_events = "
                f"{spectrum['n_events']:.6g}"
            )

        print(
            f"Plot saved to: {plot_path}"
        )
    
    full_summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "energy_spectra_grid_summary.csv"
    )

    full_summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"Combined summary saved to: "
        f"{summary_path}"
    )

    print()
    print("=" * 70)
    print("Grid analysis finished.")
    print(
        f"Number of benchmark plots: "
        f"{len(plot_paths)}"
    )

if __name__ == "__main__":
    main()