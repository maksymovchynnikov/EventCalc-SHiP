"""Plot N90 points for the two diphoton selections."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alp_discrimination.plot_style import PLOT_CONFIG, style_axis, use_report_style


FINAL_RESULT_STATUSES = {"converged", "imported_validated"}


SELECTION_PRESENTATION = {
    "diphoton_ecal": {
        "label": r"ECAL geometry only",
        "color": "tab:blue",
        "marker": "o",
    },
    "diphoton_ecal_e1gev": {
        "label": r"ECAL geometry + $E_{\gamma_1},E_{\gamma_2}\geq1$ GeV",
        "color": "tab:orange",
        "marker": "s",
    },
}


def plot_n90_comparison(
    results: pd.DataFrame,
    output_stem: Path,
    *,
    logarithmic_y: bool = True,
    show_mc_intervals: bool = True,
    include_nonconverged: bool = False,
) -> tuple[Path, Path]:
    required = {"mass_GeV", "selection_name", "N90", "convergence_status"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Adaptive result table is missing columns: {sorted(missing)}")

    use_report_style()
    figure, axis = plt.subplots(figsize=PLOT_CONFIG.profiled_figsize)
    plotted = False
    for selection_name, presentation in SELECTION_PRESENTATION.items():
        status_mask = (
            np.ones(len(results), dtype=bool)
            if include_nonconverged
            else results["convergence_status"].astype(str).isin(
                FINAL_RESULT_STATUSES
            ).to_numpy()
        )
        selected = results.loc[
            (results["selection_name"] == selection_name)
            & (pd.to_numeric(results["N90"], errors="coerce") > 0)
            & status_mask
        ].copy()
        if selected.empty:
            continue
        selected = selected.sort_values("mass_GeV")
        x = selected["mass_GeV"].to_numpy(dtype=float)
        y = selected["N90"].to_numpy(dtype=float)

        yerr = None
        if show_mc_intervals and {
            "N90_mc_lower",
            "N90_mc_upper",
        }.issubset(selected.columns):
            lower = selected["N90_mc_lower"].to_numpy(dtype=float)
            upper = selected["N90_mc_upper"].to_numpy(dtype=float)
            finite = (lower > 0) & (upper > 0) & (lower <= y) & (upper >= y)
            if np.any(finite):
                lower_error = np.where(finite, y - lower, 0.0)
                upper_error = np.where(finite, upper - y, 0.0)
                yerr = np.vstack([lower_error, upper_error])

        axis.errorbar(
            x,
            y,
            yerr=yerr,
            color=presentation["color"],
            marker=presentation["marker"],
            linestyle="-",
            linewidth=1.6,
            markersize=6.5,
            capsize=3.0 if yerr is not None else 0.0,
            label=presentation["label"],
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        raise ValueError("No positive N90 values are available to plot.")

    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel(r"Observed events required for 90\% classification")
    if logarithmic_y:
        axis.set_yscale("log")
    axis.grid(True, which="both", alpha=PLOT_CONFIG.grid_alpha)
    axis.legend(frameon=True)
    style_axis(axis)
    figure.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=PLOT_CONFIG.png_dpi, bbox_inches="tight")
    plt.close(figure)
    return pdf, png


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--linear-y", action="store_true")
    parser.add_argument("--no-error-bars", action="store_true")
    parser.add_argument(
        "--include-nonconverged",
        action="store_true",
        help="Also plot positive screening values that are not final.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    results = pd.read_csv(args.results_csv)
    paths = plot_n90_comparison(
        results,
        args.output_stem,
        logarithmic_y=not args.linear_y,
        show_mc_intervals=not args.no_error_bars,
        include_nonconverged=args.include_nonconverged,
    )
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
