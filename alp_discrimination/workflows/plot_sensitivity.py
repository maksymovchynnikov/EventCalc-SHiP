"""Create the report-ready EventCalc validation and sensitivity figure.

This module reads existing analysis products and does not rerun the full scan.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import pandas as pd

from alp_discrimination.config import PROFILES, get_config
from alp_discrimination.constraints.bc9 import draw_bc9_constraints
from alp_discrimination.constraints.plotting import (
    LABEL_CONFIG_PATH,
    SU2_SPECS,
    draw_constraints,
    load_label_config,
)
from alp_discrimination.paths import PACKAGE_ROOT, profile_output_dir
from alp_discrimination.plotting.style import EVENT_DENSITY_OVERLAY_LAYOUT, PLOT_CONFIG
from alp_discrimination.plotting.common import draw_event_contours
from alp_discrimination.plotting.reference_curves import (
    REFERENCE_FILENAMES,
    SensitivityReference,
    load_eventcalc_branches,
    load_reference,
)
from alp_discrimination.workflows import require_columns


# Two-column figure, native size.
FIGSIZE = (7.15, 4.15)

FONT_SIZE = 8.6
LABEL_SIZE = 9.2
TICK_SIZE = 8.0
LEGEND_SIZE = 7.1
ANNOTATION_SIZE = 7.2

EVENTCALC_COLOR = "C0"
REFERENCE_COLOR = "C1"
EVENTCALC_LINEWIDTH = 2.05
REFERENCE_LINEWIDTH = 1.00

SU2_CONTOUR_LINEWIDTH = 0.92

SELECTION_STYLES = {
    "geom_only": "--",
    "epsilon_dec_1": "-",
}


def use_report_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.title_fontsize": LEGEND_SIZE,
            "axes.linewidth": 0.8,
        }
    )


def finish_axis(axis: plt.Axes) -> None:
    axis.set_title("")
    axis.grid(False)
    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_SIZE,
    )
    axis.xaxis.label.set_fontsize(LABEL_SIZE)
    axis.yaxis.label.set_fontsize(LABEL_SIZE)
    axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)


def remove_constraint_text(axis: plt.Axes) -> None:
    for artist in list(axis.texts):
        artist.remove()


def draw_table_limit(axis: plt.Axes, mass_limit: float) -> None:
    axis.axvline(
        mass_limit,
        color="0.35",
        linewidth=0.75,
        linestyle=":",
        zorder=30,
    )


def validate_mother_level_provenance(event_dir: Path) -> None:
    manifest_path = event_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Event-density manifest not found: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_name = payload.get("selection_name")
    if selection_name != "mother_level":
        raise ValueError(
            "The epsilon_dec=1 EventCalc contour is expected to come from "
            "selection_name='mother_level', but the manifest contains "
            f"{selection_name!r}."
        )


def load_saved_eventcalc_boundaries(
    epsilon_dec_1_path: Path,
    geom_only_path: Path,
) -> dict[str, pd.DataFrame]:
    if epsilon_dec_1_path.resolve() == geom_only_path.resolve():
        raise ValueError(
            "epsilon_dec=1 and geom-only EventCalc contours must be different files."
        )

    tables: dict[str, pd.DataFrame] = {}
    for selection_name, path in (
        ("epsilon_dec_1", epsilon_dec_1_path),
        ("geom_only", geom_only_path),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{selection_name} EventCalc contour table not found: {path}"
            )

        table = pd.read_csv(path)
        required = {
            "model",
            "mass_GeV",
            "event_level",
            "lower_coupling_GeV_inv",
            "upper_coupling_GeV_inv",
        }
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )
        tables[selection_name] = table

    return tables


def load_photon_references() -> list[SensitivityReference]:
    references = [
        load_reference(PACKAGE_ROOT / "reference_data" / "photon_sensitivity" / filename, name)
        for name, filename in REFERENCE_FILENAMES.items()
    ]
    names = {reference.name for reference in references}
    expected = {"geom_only", "epsilon_dec_1"}
    if names != expected:
        raise ValueError(
            "Expected photon references "
            f"{sorted(expected)}, found {sorted(names)}"
        )
    return references


def draw_eventcalc_pair(
    axis: plt.Axes,
    branches: dict[str, pd.DataFrame],
    selection_name: str,
) -> None:
    lower = branches["lower"]
    upper = branches["upper"]

    style = dict(
        color=EVENTCALC_COLOR,
        linestyle=SELECTION_STYLES[selection_name],
        linewidth=EVENTCALC_LINEWIDTH,
        zorder=22,
    )

    axis.plot(lower["mass_GeV"], lower["coupling_GeV_inv"], **style)
    axis.plot(upper["mass_GeV"], upper["coupling_GeV_inv"], **style)

    if not lower.empty and not upper.empty:
        lower_first = lower.sort_values("mass_GeV").iloc[0]
        upper_first = upper.sort_values("mass_GeV").iloc[0]
        if abs(float(lower_first["mass_GeV"]) - float(upper_first["mass_GeV"])) <= 1e-10:
            x = float(lower_first["mass_GeV"])
            axis.plot(
                [x, x],
                [
                    float(lower_first["coupling_GeV_inv"]),
                    float(upper_first["coupling_GeV_inv"]),
                ],
                **style,
            )


def photon_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=EVENTCALC_COLOR, linestyle="--", linewidth=EVENTCALC_LINEWIDTH, label="EventCalc (geom only)"),
        Line2D([0], [0], color=EVENTCALC_COLOR, linestyle="-", linewidth=EVENTCALC_LINEWIDTH, label=r"EventCalc ($\epsilon_{\rm dec}=1$)"),
        Line2D([0], [0], color=REFERENCE_COLOR, linestyle="--", linewidth=REFERENCE_LINEWIDTH, label="Reference (geom only)"),
        Line2D([0], [0], color=REFERENCE_COLOR, linestyle="-", linewidth=REFERENCE_LINEWIDTH, label=r"Reference ($\epsilon_{\rm dec}=1$)"),
    ]


def format_level(level: float) -> str:
    if float(level).is_integer():
        return str(int(level))
    return f"{level:g}"


def su2_legend_handles(levels: tuple[float, ...]) -> list[Line2D]:
    styles = list(PLOT_CONFIG.event_density_line_styles)
    # If fewer styles than levels, matplotlib will have already cycled in draw_event_contours.
    if len(styles) < len(levels):
        styles = styles * (len(levels) // len(styles) + 1)
    return [
        Line2D(
            [0], [0],
            color="C0",
            linestyle=styles[idx],
            linewidth=SU2_CONTOUR_LINEWIDTH,
            label=rf"$N={format_level(level)}$",
        )
        for idx, level in enumerate(levels)
    ]


def plot_combined_figure(
    event_dir: Path,
    geom_only_path: Path,
    data: pd.DataFrame,
    levels: tuple[float, ...],
    constraint_root: Path,
    label_config,
    output_dir: Path,
    *,
    show_su2_constraint_labels: bool,
) -> tuple[Path, Path]:
    use_report_style()

    epsilon_path = event_dir / "event_contour_boundaries.csv"
    validate_mother_level_provenance(event_dir)

    saved = load_saved_eventcalc_boundaries(epsilon_path, geom_only_path)
    eventcalc = {name: load_eventcalc_branches(table) for name, table in saved.items()}
    references = {item.name: item for item in load_photon_references()}

    figure, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)

    # ------------------------------------------------------------------
    # Left panel: photophilic comparison
    # ------------------------------------------------------------------
    draw_bc9_constraints(ax_left)

    for selection_name in ("geom_only", "epsilon_dec_1"):
        draw_eventcalc_pair(ax_left, eventcalc[selection_name], selection_name)
        reference = references[selection_name]
        ax_left.plot(
            reference.points[:, 0],
            reference.points[:, 1],
            color=REFERENCE_COLOR,
            linestyle=SELECTION_STYLES[selection_name],
            linewidth=REFERENCE_LINEWIDTH,
            alpha=0.95,
            zorder=24,
        )

    ax_left.set(
        xscale="log",
        yscale="log",
        xlim=(1.5e-2, 5.0),
        ylim=(1.0e-8, 1.2e-2),
        xlabel=r"$m_a$ [GeV]",
        ylabel=r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]",
    )
    draw_table_limit(ax_left, 4.0)
    finish_axis(ax_left)
    ax_left.text(
        0.5,
        -0.27,
        r"(a) Photophilic ALPs",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    ax_left.legend(
        handles=photon_legend_handles(),
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.35,
        handletextpad=0.42,
        columnspacing=0.72,
        labelspacing=0.22,
        borderaxespad=0.0,
    )

    # ------------------------------------------------------------------
    # Right panel: SU(2)_L sensitivity
    # ------------------------------------------------------------------
    model = "ALP-SU2L"
    model_data = data.loc[data["model"] == model].copy()
    if model_data.empty:
        raise ValueError("No ALP-SU2L rows found in the production contour table.")

    directory = constraint_root / "converted/alp_su2l"
    if not directory.exists():
        raise FileNotFoundError(
            f"SU(2)_L constraint directory not found: {directory}"
        )

    draw_constraints(
        ax_right,
        directory,
        SU2_SPECS,
        model="alp_su2l",
        context="event_density_overlay",
        config=label_config,
    )

    if not show_su2_constraint_labels:
        remove_constraint_text(ax_right)

    overlay = EVENT_DENSITY_OVERLAY_LAYOUT[model]
    if overlay.corner_polygon_axes:
        ax_right.add_patch(
            Polygon(
                overlay.corner_polygon_axes,
                closed=True,
                transform=ax_right.transAxes,
                facecolor="gainsboro",
                edgecolor="none",
                zorder=-150,
                clip_on=True,
            )
        )

    before = len(ax_right.lines)
    draw_event_contours(ax_right, model_data, levels, color="C0")
    # Make the event contours a bit slimmer for better separation.
    for line in ax_right.lines[before:]:
        line.set_linewidth(SU2_CONTOUR_LINEWIDTH)

    ax_right.set(
        xscale="log",
        yscale="log",
        xlim=overlay.xlim,
        ylim=overlay.ylim,
        xlabel=r"$m_a$ [GeV]",
        ylabel=r"$g_W$ [GeV$^{-1}$]",
    )

    if show_su2_constraint_labels:
        for annotation in ax_right.texts:
            annotation.set_fontsize(ANNOTATION_SIZE)

    finish_axis(ax_right)
    ax_right.text(
        0.5,
        -0.27,
        r"(b) $SU(2)_L$-coupled ALPs",
        transform=ax_right.transAxes,
        ha="center",
        va="top",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    ax_right.legend(
        handles=su2_legend_handles(levels),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.25,
        handletextpad=0.40,
        columnspacing=0.62,
        labelspacing=0.22,
        borderaxespad=0.0,
    )

    figure.tight_layout(w_pad=0.85, rect=(0.0, 0.07, 1.0, 0.90))

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "sensitivity_panels.pdf"
    png = output_dir / "sensitivity_panels.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return pdf, png


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument(
        "--geom-only-boundaries",
        type=Path,
        required=True,
        help=(
            "Saved ECAL geom-only photophilic EventCalc contour table."
        ),
    )
    parser.add_argument("--label-config", type=Path, default=LABEL_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Default: analysis2/outputs/<profile>/report",
    )
    parser.add_argument(
        "--show-su2-constraint-labels",
        action="store_true",
        help=(
            "Show experiment names on the SU(2)_L exclusions. They are hidden "
            "by default for readability."
        ),
    )
    args = parser.parse_args()

    config = get_config(args.profile)
    event_dir = profile_output_dir(config.name, "event_density")
    contour_path = event_dir / "event_contour_boundaries.csv"

    data = pd.read_csv(contour_path)
    require_columns(
        data,
        {
            "model",
            "mass_GeV",
            "event_level",
            "status",
            "lower_coupling_GeV_inv",
            "upper_coupling_GeV_inv",
        },
        contour_path,
    )

    levels = tuple(config.event_density.event_levels)
    constraint_root = profile_output_dir(config.name, "constraints")
    label_config = load_label_config(args.label_config)

    output_dir = args.output_dir or (event_dir.parent / "report")

    print("Inputs:")
    print(f"  production event-density table: {contour_path}")
    print(f"  photophilic geom-only table:    {args.geom_only_boundaries}")
    print(f"  photon exclusions:              BC9")
    print(f"  SU(2)_L exclusions:             {constraint_root / 'converted/alp_su2l'}")
    print()

    pdf, png = plot_combined_figure(
        event_dir,
        args.geom_only_boundaries,
        data,
        levels,
        constraint_root,
        label_config,
        output_dir,
        show_su2_constraint_labels=args.show_su2_constraint_labels,
    )

    print("Saved combined figure:")
    print(f"  {pdf}")
    print(f"  {png}")


if __name__ == "__main__":
    main()
