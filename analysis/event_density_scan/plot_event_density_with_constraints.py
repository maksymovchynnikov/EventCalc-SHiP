from __future__ import annotations
from matplotlib.patches import Polygon
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_event_density_contours import safe_filename
from ..constraints.plot_photon_constraints import (
    draw_photon_constraints,
)
from ..constraints.plot_su2_constraints import (
    draw_su2_constraints,
)

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
    "ALP-photon-combined": ("ALP-photon, primary + cascades"),
    "ALP-SU2L": r"ALP-$SU(2)_L$",
}

Y_LABELS = {
    "ALP-photon-combined": (r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]"),
    "ALP-SU2L": (r"$c_W/f_a$ [GeV$^{-1}$]"),
}

TABLE_LIMITS_GEV = {
    "ALP-photon-combined": 4.0,
    "ALP-SU2L": 5.1,
}

COMBINED_LIMITS = {
    "ALP-photon-combined": {
        "x": (1.5e-2, 5),
        "y": (1.0e-8, 1.2e-2),
    },
    "ALP-SU2L": {
        "x": (5.0e-2, 7),
        "y": (8.0e-7, 6.0e-1),
    },
}

BBOX_TO_ANCHOR = {
    "ALP-photon-combined": (0.95, 0.7),
    "ALP-SU2L": (0.9, 0.83),
}

LINE_STYLES = {
    3.0: ":",
    10.0: "--",
    30.0: "-.",
    100.0: "-",
}

PHOTON_COMBINED_LABEL_POSITIONS_AXES = {
    "bounds_SN1987.txt": (
        0.34,
        0.06,
        0,
        "left",
        "center",
    ),
    "bounds_NuCal.txt": (
        0.34,
        0.49,
        -15,
        "center",
        "center",
    ),
    "bounds_PrimEx.txt": (
        0.378,
        0.87,
        90,
        "center",
        "center",
    ),
    "bounds_BESIII_2024.txt": (
        0.445,
        0.807,
        0,
        "center",
        "center",
    ),
    "bounds_Belle2.txt": (
        0.51,
        0.880,
        0,
        "center",
        "center",
    ),
    "bounds_BESIII_2022.txt": (
        0.453,
        0.747,
        0,
        "center",
        "center",
    ),
    "bounds_E137.txt": (
        0.34,
        0.375,
        -5,
        "center",
        "center",
    ),
}

SU2_COMBINED_LABEL_POSITIONS_AXES = {
    "bounds_CDF.txt": (
        0.035,
        0.85,
        -12,
        "left",
        "center",
    ),
    "bounds_E949_displ.txt": (
        0.03,
        0.705,
        -9,
        "left",
        "center",
    ),
    "bounds_KOTO.txt": (
        0.095,
        0.62,
        0,
        "center",
        "center",
    ),
    "bounds_NA62_2.txt": (
        0.03,
        0.535,
        2,
        "left",
        "center",
    ),
    "bounds_E949_prompt.txt": (
        0.03,
        0.455,
        -5,
        "left",
        "center",
    ),
    "bounds_E137.txt": (
        0.34,
        0.375,
        -5,
        "center",
        "center",
    ),
    "bounds_SN1987.txt": (
        0.02,
        0.2,
        0,
        "left",
        "center",
    ),
    "bounds_KTEV.txt": (
        0.275,
        0.870,
        90,
        "center",
        "center",
    ),
    "bounds_NA62_1.txt": (
        0.34,
        0.870,
        90,
        "center",
        "center",
    ),
    "bounds_NA6264.txt": (
        0.365,
        0.895,
        90,
        "center",
        "center",
    ),
    "bounds_BaBar.txt": (
        0.36,
        0.635,
        0,
        "center",
        "center",
    ),
    "bounds_LEP.txt": (
        0.52,
        0.89,
        0,
        "center",
        "center",
    ),
}


def draw_constraints_for_model(
    axis: plt.Axes,
    model_name: str,
) -> None:
    if model_name == "ALP-photon-combined":
        draw_photon_constraints(
            axis,
            draw_labels=True,
            label_positions_axes=(PHOTON_COMBINED_LABEL_POSITIONS_AXES),
            label_fontsize=9.5,
        )

    elif model_name == "ALP-SU2L":
        draw_su2_constraints(
            axis,
            draw_labels=True,
            label_positions_axes=(SU2_COMBINED_LABEL_POSITIONS_AXES),
            label_fontsize=9.5,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def draw_ship_event_contours(
    axis: plt.Axes,
    model_name: str,
    model_data: pd.DataFrame,
) -> None:
    problem_statuses = {
        "upper_boundary_above_scan",
        "lower_boundary_below_scan",
        "both_boundaries_outside_scan",
        "multiple_crossings",
        "one_crossing_unclassified",
        "unresolved_numerically",
    }

    problematic = model_data[model_data["status"].isin(problem_statuses)]

    if not problematic.empty:
        print()
        print(f"Warning: scan-limited or numerically unresolved contours for {model_name}:")
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

    for event_level in EVENT_LEVELS:
        level_data = model_data[
            np.isclose(
                model_data["event_level"],
                event_level,
            )
        ].sort_values("mass_GeV")

        if level_data.empty:
            continue

        masses = level_data["mass_GeV"].to_numpy(dtype=float)

        lower = level_data["lower_coupling_GeV_inv"].to_numpy(dtype=float)

        upper = level_data["upper_coupling_GeV_inv"].to_numpy(dtype=float)

        valid_lower = np.isfinite(lower)
        valid_upper = np.isfinite(upper)

        label = rf"$N_{{\rm events}} = {int(event_level)}$"

        axis.plot(
            masses[valid_lower],
            lower[valid_lower],
            linestyle=LINE_STYLES[event_level],
            linewidth=2.0,
            color="C0",
            label=label,
            zorder=10,
        )

        axis.plot(
            masses[valid_upper],
            upper[valid_upper],
            linestyle=LINE_STYLES[event_level],
            linewidth=2.0,
            color="C0",
            label="_nolegend_",
            zorder=10,
        )

    axis.axvline(
        TABLE_LIMITS_GEV[model_name],
        color="black",
        linewidth=1.5,
        linestyle="-",
        label="table limit",
        zorder=11,
    )


def configure_axis(
    axis: plt.Axes,
    model_name: str,
) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")

    axis.set_xlim(*COMBINED_LIMITS[model_name]["x"])
    axis.set_ylim(*COMBINED_LIMITS[model_name]["y"])

    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel(Y_LABELS[model_name])

    axis.set_title(f"Existing constraints and SHiP event contours:\n{MODEL_LABELS[model_name]}")

    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    axis.grid(False)

    legend = axis.legend(
        bbox_to_anchor=BBOX_TO_ANCHOR[model_name],
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="whitesmoke",
        edgecolor="gray",
    )

    legend.get_frame().set_linewidth(0.8)


def add_su2_top_left_excluded_patch(
    axis: plt.Axes,
) -> None:
    patch = Polygon(
        [
            (0.00, 1.00),
            (0.00, 0.935),
            (0.135, 1.00),
        ],
        closed=True,
        transform=axis.transAxes,
        facecolor="gainsboro",
        edgecolor="none",
        zorder=-150,
        clip_on=True,
    )

    axis.add_patch(patch)


def make_plot_for_model(
    model_name: str,
    boundary_data: pd.DataFrame,
) -> Path:
    model_data = boundary_data[boundary_data["model"] == model_name].copy()

    if model_data.empty:
        raise ValueError(f"No contour data found for {model_name}")

    figure, axis = plt.subplots(
        figsize=(8.5, 6.5),
    )

    draw_constraints_for_model(
        axis,
        model_name,
    )

    if model_name == "ALP-SU2L":
        add_su2_top_left_excluded_patch(axis)

    draw_ship_event_contours(
        axis,
        model_name,
        model_data,
    )

    configure_axis(
        axis,
        model_name,
    )

    figure.tight_layout()

    output_path = PLOT_DIR / f"event_density_with_constraints_{safe_filename(model_name)}.pdf"

    figure.savefig(output_path)
    plt.close(figure)

    print(f"Saved {output_path}")

    return output_path


def main() -> None:
    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            "Could not find boundary table:\n"
            f"  {BOUNDARY_PATH}\n"
            "Run scan_event_density.py first."
        )

    boundary_data = pd.read_csv(BOUNDARY_PATH)

    for model_name in (
        "ALP-photon-combined",
        "ALP-SU2L",
    ):
        make_plot_for_model(
            model_name,
            boundary_data,
        )


if __name__ == "__main__":
    main()
