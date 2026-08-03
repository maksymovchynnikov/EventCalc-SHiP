"""Plot saved event-density contours, optionally over downloaded constraints."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon

from analysis2.config import PROFILES, get_config
from analysis2.constraints.plotting import (
    LABEL_CONFIG_PATH, PHOTON_SPECS, SU2_SPECS, draw_constraints, load_label_config,
)
from analysis2.paths import profile_output_dir
from analysis2.plot_style import (
    EVENT_DENSITY_OVERLAY_LAYOUT,
    PLOT_CONFIG,
    style_axis,
    use_report_style,
)
from analysis2.plotting import draw_event_contours
from analysis2.workflows import require_columns

Y_LABELS = {
    "ALP-photon-combined": r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]",
    "ALP-SU2L": r"$c_W/f_a$ [GeV$^{-1}$]",
}
CONSTRAINT_INPUTS = {
    "ALP-photon-combined": ("raw/alp_photon", PHOTON_SPECS, "alp_photon"),
    "ALP-SU2L": ("converted/alp_su2l", SU2_SPECS, "alp_su2l"),
}
def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument("--with-constraints", action="store_true")
    parser.add_argument("--label-config", type=Path, default=LABEL_CONFIG_PATH)
    args = parser.parse_args()
    use_report_style()
    config = get_config(args.profile)
    event_dir = profile_output_dir(config.name, "event_density")
    path = event_dir / "event_contour_boundaries.csv"
    data = pd.read_csv(path)
    require_columns(data, {
        "model", "mass_GeV", "event_level", "status",
        "lower_coupling_GeV_inv", "upper_coupling_GeV_inv",
    }, path)
    unsupported = set(data["model"]) - set(Y_LABELS)
    if unsupported:
        raise ValueError(f"unsupported event-density models: {sorted(unsupported)}")
    levels = tuple(level for level in config.event_density.event_levels if level != 2.3)
    constraint_dir = profile_output_dir(config.name, "constraints")
    label_config = load_label_config(args.label_config) if args.with_constraints else None
    for model, model_data in data.groupby("model", sort=True):
        figure, axis = plt.subplots(figsize=PLOT_CONFIG.event_density_figsize)
        if args.with_constraints:
            subdirectory, specs, constraint_model = CONSTRAINT_INPUTS[model]
            directory = constraint_dir / subdirectory
            if not directory.exists():
                raise FileNotFoundError(f"constraint directory not found: {directory}")
            draw_constraints(
                axis, directory, specs, model=constraint_model,
                context="event_density_overlay", config=label_config,
            )
            overlay = EVENT_DENSITY_OVERLAY_LAYOUT[model]
            if overlay.corner_polygon_axes:
                axis.add_patch(Polygon(
                    overlay.corner_polygon_axes,
                    closed=True,
                    transform=axis.transAxes, facecolor="gainsboro", edgecolor="none",
                    zorder=-150, clip_on=True,
                ))
        draw_event_contours(axis, model_data, levels, color="C0" if args.with_constraints else None)
        axis.set(xscale="log", yscale="log", xlabel=r"$m_a$ [GeV]",
                 ylabel=Y_LABELS[model])
        if args.with_constraints:
            overlay = EVENT_DENSITY_OVERLAY_LAYOUT[model]
            axis.set(xlim=overlay.xlim, ylim=overlay.ylim)
            axis.axvline(
                overlay.table_mass_limit_gev,
                color="black",
                linewidth=PLOT_CONFIG.event_density_table_line_width,
                label="table limit",
                zorder=11,
            )
            axis.tick_params(which="both", direction="in", top=True, right=True)
            axis.grid(False)
            legend = axis.legend(
                bbox_to_anchor=overlay.legend_anchor,
                frameon=True,
                fancybox=False,
                framealpha=1.0, facecolor="whitesmoke", edgecolor="gray",
            )
            legend.get_frame().set_linewidth(
                PLOT_CONFIG.event_density_legend_frame_width
            )
        else:
            axis.grid(
                True,
                which="both",
                alpha=PLOT_CONFIG.event_density_grid_alpha,
            )
            axis.legend()
        style_axis(axis)
        figure.tight_layout()
        stem = (f"event_density_with_constraints_{model.lower()}" if args.with_constraints
                else f"event_density_{model.lower()}")
        output = event_dir / "plots" / f"{stem}.pdf"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
