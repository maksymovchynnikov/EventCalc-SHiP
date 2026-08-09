"""Compare the two saved EventCalc ALP-photon contours with Figure 13.

The final comparison contains four matched curves at N_events = 2.3:

* EventCalc with daughter-level diphoton ECAL geometry ("Geom only"),
  loaded from the ECAL-updated legacy analysis output;
* the bundled Figure-13 geom-only reference;
* EventCalc before daughter-level selection (epsilon_dec = 1),
  loaded from the accepted analysis2 production event-density output;
* the bundled Figure-13 epsilon_dec = 1 reference.

No EventCalc scan is rerun.  The workflow only reads the two already-produced
boundary tables and compares each one with the matching reference curve.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from alp_discrimination.config import PROFILES
from alp_discrimination.constraints.plotting import (
    LABEL_CONFIG_PATH,
    PHOTON_SPECS,
    draw_constraints,
    load_label_config,
)
from alp_discrimination.paths import PACKAGE_ROOT, profile_output_dir
from alp_discrimination.plot_style import style_axis, use_report_style
from alp_discrimination.reference_curves import (
    EVENT_LEVEL,
    REFERENCE_FILENAMES,
    SensitivityReference,
    load_eventcalc_branches,
    load_reference,
    make_distance_summary,
    make_pointwise_comparison,
    make_reference_summary,
    split_reference_branches,
)
from alp_discrimination.workflows import write_dataframe


SELECTION_ORDER = ("geom_only", "epsilon_dec_1")

SELECTION_LABELS = {
    "geom_only": r"Geom only",
    "epsilon_dec_1": r"$\epsilon_{\mathrm{dec}}=1$",
}

EVENTCALC_COLOR = "C0"
REFERENCE_COLOR = "C1"

SELECTION_STYLES = {
    "geom_only": {
        "linestyle": "--",
        "label": "Geom only",
    },
    "epsilon_dec_1": {
        "linestyle": "-",
        "label": r"$\epsilon_{\mathrm{dec}}=1$",
    },
}

EVENTCALC_LINEWIDTH = 2.4
REFERENCE_LINEWIDTH = 1.25

# Native size for a single column in the two-column report.
COMPARISON_FIGSIZE = (3.45, 3.05)
COMPARISON_LABELSIZE = 8.5
COMPARISON_TICKSIZE = 7.5
COMPARISON_LEGENDSIZE = 6.7

LOG_DISTANCE_STYLES = {
    "geom_only": {
        "color": "C0",
        "linestyle": "-",
    },
    "epsilon_dec_1": {
        "color": "C1",
        "linestyle": "--",
    },
}


def _validate_analysis2_provenance(event_dir: Path) -> None:
    """Ensure the accepted analysis2 contour is the mother-level epsilon_dec=1 result."""
    manifest_path = event_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Event-density manifest not found: {manifest_path}. "
            "The epsilon_dec=1 curve cannot be labelled safely without provenance."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_name = payload.get("selection_name")
    if selection_name != "mother_level":
        raise ValueError(
            "Expected the saved analysis2 event-density contour to use "
            f"selection_name='mother_level', found {selection_name!r}."
        )


def load_saved_eventcalc_boundaries(
    epsilon_dec_1_path: Path,
    geom_only_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load the two independently produced EventCalc contour tables."""
    epsilon_dec_1_path = epsilon_dec_1_path.resolve()
    geom_only_path = geom_only_path.resolve()
    if epsilon_dec_1_path == geom_only_path:
        raise ValueError(
            "epsilon_dec=1 and geom-only EventCalc contours must be different files"
        )

    tables = {}
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


def _load_references() -> list[SensitivityReference]:
    references = [
        load_reference(PACKAGE_ROOT / "reference_curves" / filename, name)
        for name, filename in REFERENCE_FILENAMES.items()
    ]
    names = {item.name for item in references}
    if names != set(SELECTION_ORDER):
        raise ValueError(
            "REFERENCE_FILENAMES must contain exactly "
            f"{list(SELECTION_ORDER)}, found {sorted(names)}"
        )
    if references[0].production_modes != references[1].production_modes:
        raise ValueError("bundled reference curves list different production modes")
    return references


def _pairwise_comparison(
    eventcalc_by_selection: dict[str, dict[str, pd.DataFrame]],
    reference_branches: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Compare each reference only with the EventCalc curve of the same selection."""
    frames = []
    for selection_name in SELECTION_ORDER:
        frame = make_pointwise_comparison(
            eventcalc_by_selection[selection_name],
            {selection_name: reference_branches[selection_name]},
        )
        frame["eventcalc_selection"] = selection_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _draw_closed_eventcalc_curve(
    axis: plt.Axes,
    branches: dict[str, pd.DataFrame],
    selection_name: str,
) -> None:
    for branch_name in ("lower", "upper"):
        branch = branches[branch_name]
        axis.plot(
            branch["mass_GeV"],
            branch["coupling_GeV_inv"],
            color=EVENTCALC_COLOR,
            linestyle=SELECTION_STYLES[selection_name]["linestyle"],
            linewidth=EVENTCALC_LINEWIDTH,
            label="_nolegend_",
            zorder=22,
        )


def _remove_constraint_labels(axis: plt.Axes) -> None:
    """Remove experiment-name labels drawn by draw_constraints()."""
    for artist in list(axis.texts):
        artist.remove()

def _draw_comparison(
    references: list[SensitivityReference],
    eventcalc_by_selection: dict[str, dict[str, pd.DataFrame]],
    constraint_dir: Path,
    label_config,
    output_pdf: Path,
    output_png: Path,
) -> None:
    figure, axis = plt.subplots(figsize=COMPARISON_FIGSIZE)
    draw_constraints(
        axis,
        constraint_dir,
        PHOTON_SPECS,
        model="alp_photon",
        context="event_density_overlay",
        config=label_config,
    )

    _remove_constraint_labels(axis)

    reference_by_name = {item.name: item for item in references}
    for selection_name in SELECTION_ORDER:
        _draw_closed_eventcalc_curve(
            axis,
            eventcalc_by_selection[selection_name],
            selection_name,
        )
        reference = reference_by_name[selection_name]
        axis.plot(
            reference.points[:, 0],
            reference.points[:, 1],
            color=REFERENCE_COLOR,
            linestyle=SELECTION_STYLES[selection_name]["linestyle"],
            linewidth=REFERENCE_LINEWIDTH,
            alpha=0.95,
            label="_nolegend_",
            zorder=24,
        )

    axis.axvline(
        4.0,
        color="0.3",
        linewidth=1.2,
        linestyle=":",
        zorder=20,
    )

    axis.text(
        4.0,
        0.03,
        "EventCalc table limit",
        transform=axis.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=COMPARISON_TICKSIZE,
        color="0.3",
    )

    axis.set(
        xscale="log",
        yscale="log",
        xlim=(1.5e-2, 5.0),
        ylim=(1.0e-8, 1.2e-2),
        xlabel=r"$m_a$ [GeV]",
        ylabel=r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]",
    )
    axis.grid(False)
    legend_handles = [
        Line2D(
            [0], [0],
            color=EVENTCALC_COLOR,
            linestyle=SELECTION_STYLES["geom_only"]["linestyle"],
            linewidth=EVENTCALC_LINEWIDTH,
            label="EventCalc (geom only)",
        ),
        Line2D(
            [0], [0],
            color=EVENTCALC_COLOR,
            linestyle=SELECTION_STYLES["epsilon_dec_1"]["linestyle"],
            linewidth=EVENTCALC_LINEWIDTH,
            label=r"EventCalc ($\epsilon_{\mathrm{dec}}=1$)",
        ),
        Line2D(
            [0], [0],
            color=REFERENCE_COLOR,
            linestyle=SELECTION_STYLES["geom_only"]["linestyle"],
            linewidth=REFERENCE_LINEWIDTH,
            label="Reference (geom only)",
        ),
        Line2D(
            [0], [0],
            color=REFERENCE_COLOR,
            linestyle=SELECTION_STYLES["epsilon_dec_1"]["linestyle"],
            linewidth=REFERENCE_LINEWIDTH,
            label=r"Reference ($\epsilon_{\mathrm{dec}}=1$)",
        ),
    ]

    style_axis(axis)
    axis.xaxis.label.set_size(COMPARISON_LABELSIZE)
    axis.yaxis.label.set_size(COMPARISON_LABELSIZE)
    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=COMPARISON_TICKSIZE,
    )
    axis.legend(
        handles=legend_handles,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=COMPARISON_LEGENDSIZE,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _draw_log_distance(
    pointwise: pd.DataFrame,
    branch_name: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    selected_branch = pointwise.loc[pointwise["branch"] == branch_name]

    for selection_name in SELECTION_ORDER:
        selected = selected_branch.loc[
            selected_branch["reference"] == selection_name
        ]
        style = LOG_DISTANCE_STYLES[selection_name]
        axis.plot(
            selected["mass_GeV"],
            selected["log10_reference_over_eventcalc_dex"],
            label=SELECTION_LABELS[selection_name],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
        )
    y_limit = max(
        0.02,
        1.1 * float(selected_branch["absolute_log_distance_dex"].max()),
    )
    axis.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="Exact agreement",
    )
    axis.set(
        xscale="log",
        xlim=(
            selected_branch["mass_GeV"].min(),
            selected_branch["mass_GeV"].max(),
        ),
        ylim=(-y_limit, y_limit),
        xlabel=r"$m_a$ [GeV]",
        ylabel=(
            r"$\Delta\log_{10}g="
            r"\log_{10}(g_{\rm ref}/g_{\rm EventCalc})$ [dex]"
        ),
    )
    relation = r"\geq 0" if branch_name == "lower" else r"\leq 0"
    axis.text(
        0.02,
        0.04,
        rf"Reference inside matching EventCalc curve: "
        rf"$\Delta\log_{{10}}g{relation}$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
            "pad": 4.0,
        },
    )
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="whitesmoke",
        edgecolor="gray",
    )
    axis.grid(False)
    style_axis(axis)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="production",
    )
    parser.add_argument(
        "--constraint-dir",
        type=Path,
        help="Override analysis2/outputs/<profile>/constraints/raw/alp_photon",
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        default=LABEL_CONFIG_PATH,
    )
    parser.add_argument(
        "--geom-only-boundaries",
        type=Path,
        required=True,
        help=(
            "ECAL-updated EventCalc geom-only boundary table."
        ),
    )
    args = parser.parse_args()

    use_report_style()
    event_dir = profile_output_dir(args.profile, "event_density")
    epsilon_boundary_path = event_dir / "event_contour_boundaries.csv"
    _validate_analysis2_provenance(event_dir)

    constraint_dir = args.constraint_dir or (
        profile_output_dir(args.profile, "constraints") / "raw/alp_photon"
    )
    if not constraint_dir.exists():
        raise FileNotFoundError(
            f"constraint directory not found: {constraint_dir}"
        )

    references = _load_references()
    reference_branches = {
        item.name: split_reference_branches(item.points)
        for item in references
    }

    saved_boundaries = load_saved_eventcalc_boundaries(
        epsilon_boundary_path,
        args.geom_only_boundaries,
    )
    eventcalc_by_selection = {
        selection_name: load_eventcalc_branches(boundaries)
        for selection_name, boundaries in saved_boundaries.items()
    }
    pointwise = _pairwise_comparison(
        eventcalc_by_selection,
        reference_branches,
    )
    distance_summary = make_distance_summary(pointwise)

    write_dataframe(
        make_reference_summary(references),
        event_dir / "photon_sensitivity_reference_summary.csv",
    )
    write_dataframe(
        pointwise,
        event_dir / "photon_sensitivity_reference_log_distances.csv",
    )
    write_dataframe(
        distance_summary,
        event_dir / "photon_sensitivity_reference_distance_summary.csv",
    )

    plot_dir = event_dir / "plots"
    label_config = load_label_config(args.label_config)
    comparison_pdf = (
        plot_dir / "photon_sensitivity_reference_comparison.pdf"
    )
    comparison_png = (
        plot_dir / "photon_sensitivity_reference_comparison.png"
    )
    lower_pdf = (
        plot_dir / "photon_sensitivity_reference_log_distance_lower.pdf"
    )
    upper_pdf = (
        plot_dir / "photon_sensitivity_reference_log_distance_upper.pdf"
    )

    _draw_comparison(
        references,
        eventcalc_by_selection,
        constraint_dir,
        label_config,
        comparison_pdf,
        comparison_png,
    )
    _draw_log_distance(pointwise, "lower", lower_pdf)
    _draw_log_distance(pointwise, "upper", upper_pdf)

    print("Saved matched EventCalc/reference comparison:")
    print(f"  epsilon_dec=1 EventCalc input: {epsilon_boundary_path}")
    print(f"  geom-only EventCalc input:     {args.geom_only_boundaries}")
    for output in (
        comparison_pdf,
        comparison_png,
        lower_pdf,
        upper_pdf,
    ):
        print(f"  {output}")


if __name__ == "__main__":
    main()