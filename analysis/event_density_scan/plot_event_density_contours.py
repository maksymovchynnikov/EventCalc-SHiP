from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
BOUNDARY_PATH = ANALYSIS_DIR / "event_contour_boundaries.csv"
PLOT_DIR = ANALYSIS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVELS = (
    3.0,
    10.0,
    30.0,
    100.0,
)

MODEL_LABELS = {
    "ALP-photon-primary": "ALP-photon, primary",
    "ALP-SU2L": r"ALP-$SU(2)_L$",
}

TABLE_LIMITS_GEV = {
    "ALP-photon-primary": 4.0,
    "ALP-SU2L": 5.1,
}

Y_LABELS = {
    "ALP-photon-primary": (
        r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]"
    ),
    "ALP-SU2L": (
        r"$c_W/f_a$ [GeV$^{-1}$]"
    ),
}

LINE_STYLES = {
    3.0: ":",
    10.0: "--",
    30.0: "-.",
    100.0: "-",
}


def safe_filename(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

def draw_event_contours(
    axis: plt.Axes,
    model_data: pd.DataFrame,
) -> None:
    for event_level in EVENT_LEVELS:
            level_data = model_data[
                np.isclose(
                    model_data["event_level"],
                    event_level,
                )
            ].sort_values("mass_GeV")

            masses = level_data[
                "mass_GeV"
            ].to_numpy(dtype=float)

            lower = level_data[
                "lower_coupling_GeV_inv"
            ].to_numpy(dtype=float)

            upper = level_data[
                "upper_coupling_GeV_inv"
            ].to_numpy(dtype=float)

            valid_lower = np.isfinite(lower)
            valid_upper = np.isfinite(upper)

            lower_line, = axis.plot(
                masses[valid_lower],
                lower[valid_lower],
                linestyle=LINE_STYLES[event_level],
                linewidth=2.0,
                label=(
                    rf"$N_{{\rm events}}={event_level:g}$"
                ),
            )

            axis.plot(
                masses[valid_upper],
                upper[valid_upper],
                linestyle=LINE_STYLES[event_level],
                linewidth=2.0,
                color=lower_line.get_color(),
            )


def main() -> None:
    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            "Could not find boundary table:\n"
            f"  {BOUNDARY_PATH}\n"
            "Run scan_event_density.py first."
        )

    boundary_data = pd.read_csv(BOUNDARY_PATH)

    for model_name, model_data in boundary_data.groupby(
        "model",
        sort=False,
    ):
        problem_statuses = {
            "upper_boundary_above_scan",
            "lower_boundary_below_scan",
            "both_boundaries_outside_scan",
            "multiple_crossings",
            "one_crossing_unclassified",
            "unresolved_numerically",
        }

        problematic = model_data[
            model_data["status"].isin(
                problem_statuses
            )
        ]

        if not problematic.empty:
            print()
            print(
                f"Warning: scan-limited or numerically "
                f"unresolved contours for {model_name}:"
            )

            print(
                problematic[
                    [
                        "mass_GeV",
                        "event_level",
                        "status",
                        "number_of_crossings",
                        "maximum_N_events",
                    ]
                ].to_string(index=False)
            )

        figure, axis = plt.subplots(
            figsize=(8.5, 6.5),
        )

        draw_event_contours(
            axis,
            model_data
        )

        axis.set_xscale("log")
        axis.set_yscale("log")

        axis.set_xlabel(r"$m_a$ [GeV]")
        axis.set_ylabel(
            Y_LABELS.get(
                model_name,
                r"Coupling [GeV$^{-1}$]",
            )
        )
        
        axis.set_title(
            MODEL_LABELS.get(
                model_name,
                model_name,
            )
        )

        table_limit = TABLE_LIMITS_GEV.get(model_name)
        
        if table_limit is not None:
            axis.axvline(
                table_limit,
                linewidth=1.0,
                linestyle=":",
                color="dimgray",
                alpha=0.7,
            )

            axis.text(
                table_limit / 1.04,
                0.58,
                "table limit",
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="center",
                ha="right",
                color="dimgray",
            )

        axis.grid(
            True,
            which="both",
            alpha=0.3,
        )
        axis.legend()
        figure.tight_layout()

        output_path = (
            PLOT_DIR
            / (
                "event_density_contours_"
                f"{safe_filename(model_name)}.pdf"
            )
        )

        figure.savefig(
            output_path,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

