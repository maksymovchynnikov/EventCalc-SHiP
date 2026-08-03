"""Compare combined EventCalc photon sensitivity with bundled SHiP references."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis2.config import PROFILES
from analysis2.constraints.plotting import (
    LABEL_CONFIG_PATH, PHOTON_SPECS, draw_constraints, load_label_config,
)
from analysis2.paths import PACKAGE_ROOT, profile_output_dir
from analysis2.plot_style import style_axis, use_report_style
from analysis2.reference_curves import (
    EVENT_LEVEL, REFERENCE_FILENAMES, SensitivityReference, load_eventcalc_branches,
    load_reference, make_distance_summary, make_pointwise_comparison,
    make_reference_summary, split_reference_branches,
)
from analysis2.workflows import write_dataframe

REFERENCE_STYLES = {
    "geom_only": ("Reference: Geom only", "C1", "-."),
    "baseline": ("Reference: Baseline", "C2", "--"),
}


def _draw_comparison(
    references: list[SensitivityReference], eventcalc: dict[str, pd.DataFrame],
    constraint_dir: Path, label_config, output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 6.7))
    draw_constraints(
        axis, constraint_dir, PHOTON_SPECS, model="alp_photon",
        context="event_density_overlay", config=label_config,
    )
    for branch_name in ("lower", "upper"):
        branch = eventcalc[branch_name]
        axis.plot(
            branch["mass_GeV"], branch["coupling_GeV_inv"], color="C0", linestyle="-",
            linewidth=2.6, label=("EventCalc: Geom only"
                                  if branch_name == "lower" else "_nolegend_"), zorder=20,
        )
    for reference in references:
        label, colour, linestyle = REFERENCE_STYLES[reference.name]
        axis.plot(
            reference.points[:, 0], reference.points[:, 1], color=colour,
            linestyle=linestyle, linewidth=2.4,
            label=label, zorder=21,
        )
    axis.axvline(4.0, color="black", linewidth=1.5, linestyle=":", label="EventCalc table limit", zorder=22)
    axis.set(xscale="log", yscale="log", xlim=(1.5e-2, 5.0), ylim=(1e-8, 1.2e-2),
             xlabel=r"$m_a$ [GeV]", ylabel=r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]")
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.grid(False)
    legend = axis.legend(bbox_to_anchor=(0.555, 0.75), frameon=True, fancybox=False,
                         framealpha=1.0, facecolor="whitesmoke", edgecolor="gray")
    legend.get_frame().set_linewidth(0.8)
    style_axis(axis)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _draw_log_distance(pointwise: pd.DataFrame, branch_name: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    selected_branch = pointwise.loc[pointwise["branch"] == branch_name]
    for reference_name, (label, colour, linestyle) in REFERENCE_STYLES.items():
        selected = selected_branch.loc[selected_branch["reference"] == reference_name]
        axis.plot(selected["mass_GeV"], selected["log10_reference_over_eventcalc_dex"],
                  label=label, color=colour, linestyle=linestyle, linewidth=2.2)
    y_limit = max(0.02, 1.1 * float(selected_branch["absolute_log_distance_dex"].max()))
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle=":",
                 label="Exact agreement with EventCalc")
    axis.set(xscale="log", xlim=(selected_branch["mass_GeV"].min(),
                                 selected_branch["mass_GeV"].max()),
             ylim=(-y_limit, y_limit), xlabel=r"$m_a$ [GeV]",
             ylabel=r"$\Delta\log_{10}g=\log_{10}(g_{\rm ref}/g_{\rm EventCalc})$ [dex]")
    relation = r"\geq 0" if branch_name == "lower" else r"\leq 0"
    axis.text(0.02, 0.04, rf"Inside EventCalc: $\Delta\log_{{10}}g{relation}$",
              transform=axis.transAxes, ha="left", va="bottom",
              bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9, "pad": 4.0})
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.legend(frameon=True, framealpha=1.0, facecolor="whitesmoke", edgecolor="gray")
    axis.grid(False)
    style_axis(axis)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument("--constraint-dir", type=Path,
                        help="Override analysis2/outputs/<profile>/constraints/raw/alp_photon")
    parser.add_argument("--label-config", type=Path, default=LABEL_CONFIG_PATH)
    args = parser.parse_args()
    use_report_style()
    event_dir = profile_output_dir(args.profile, "event_density")
    boundary_path = event_dir / "event_contour_boundaries.csv"
    if not boundary_path.exists():
        raise FileNotFoundError(f"EventCalc contour table not found: {boundary_path}")
    constraint_dir = args.constraint_dir or profile_output_dir(
        args.profile, "constraints") / "raw/alp_photon"
    if not constraint_dir.exists():
        raise FileNotFoundError(f"constraint directory not found: {constraint_dir}")

    references = [load_reference(PACKAGE_ROOT / "reference_curves" / filename, name)
                  for name, filename in REFERENCE_FILENAMES.items()]
    if references[0].production_modes != references[1].production_modes:
        raise ValueError("bundled reference curves list different production modes")
    eventcalc = load_eventcalc_branches(pd.read_csv(boundary_path))
    reference_branches = {item.name: split_reference_branches(item.points) for item in references}
    pointwise = make_pointwise_comparison(eventcalc, reference_branches)
    distance_summary = make_distance_summary(pointwise)
    write_dataframe(make_reference_summary(references),
                    event_dir / "photon_sensitivity_reference_summary.csv")
    write_dataframe(pointwise, event_dir / "photon_sensitivity_reference_log_distances.csv")
    write_dataframe(distance_summary,
                    event_dir / "photon_sensitivity_reference_distance_summary.csv")

    plot_dir = event_dir / "plots"
    label_config = load_label_config(args.label_config)
    outputs = [
        plot_dir / "photon_sensitivity_reference_comparison.pdf",
        plot_dir / "photon_sensitivity_reference_log_distance_lower.pdf",
        plot_dir / "photon_sensitivity_reference_log_distance_upper.pdf",
    ]
    _draw_comparison(references, eventcalc, constraint_dir, label_config, outputs[0])
    _draw_log_distance(pointwise, "lower", outputs[1])
    _draw_log_distance(pointwise, "upper", outputs[2])
    print("Saved reference summaries and plots:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
