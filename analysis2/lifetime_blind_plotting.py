"""Plots for the ECAL-aware lifetime-blind analysis.

All numerical inputs are already-computed tables or template banks.  Figure
sizes and other adjustable presentation values live in :mod:`plot_style`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .lifetime_template_banks import LifetimeTemplateBank
from .plot_style import PLOT_CONFIG, style_axis, use_report_style


def _save_pdf_png(figure: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=PLOT_CONFIG.png_dpi, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def plot_lifetime_scan(
    scan: pd.DataFrame,
    *,
    event_threshold: float,
    output_path: Path,
) -> Path:
    """Preserve the current ECAL event-rate scan layout."""
    use_report_style()
    figure, axis = plt.subplots(figsize=PLOT_CONFIG.lifetime_scan_figsize)
    masses = np.asarray(sorted(scan["mass_GeV"].unique()), dtype=float)
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(masses)))
    colour_by_mass = dict(zip(masses, colours))
    line_style = {"ALP-photon-combined": "-", "ALP-SU2L": "--"}
    for mass_gev in masses:
        mass_data = scan.loc[np.isclose(scan["mass_GeV"], mass_gev)]
        for model, model_data in mass_data.groupby("model"):
            ordered = model_data.sort_values("ctau_m")
            axis.loglog(
                ordered["ctau_m"],
                ordered["N_events"],
                linestyle=line_style.get(str(model), "-"),
                marker="o",
                markersize=2.2,
                linewidth=1.15,
                color=colour_by_mass[mass_gev],
            )
    axis.axhline(event_threshold, linestyle=":", linewidth=1.3, color="black")
    axis.set_xlabel(r"$c\tau_a$ [m]")
    axis.set_ylabel(r"ECAL-accepted $N_{\mathrm{events}}$")
    axis.grid(True, which="both", alpha=PLOT_CONFIG.grid_alpha)
    model_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="ALP-photon, primary + cascade"),
        Line2D([0], [0], color="black", linestyle="--", label=r"ALP-$SU(2)_L$"),
        Line2D([0], [0], color="black", linestyle=":", label=r"$N_{\rm events}=10$"),
    ]
    model_legend = axis.legend(handles=model_handles, loc="upper right", fontsize=8)
    axis.add_artist(model_legend)
    mass_handles = [
        Line2D(
            [0], [0], color=colour_by_mass[mass], marker="o", linestyle="none",
            markersize=4, label=rf"${mass:g}$",
        )
        for mass in masses
    ]
    axis.legend(
        handles=mass_handles,
        title=r"$m_a$ [GeV]",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7.5,
    )
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=PLOT_CONFIG.png_dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def logarithmic_cell_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or len(centres) < 2:
        raise ValueError("At least two lifetime centres are required.")
    if np.any(centres <= 0.0) or np.any(np.diff(centres) <= 0.0):
        raise ValueError("Lifetime centres must be positive and increasing.")
    log_centres = np.log(centres)
    log_edges = np.empty(len(centres) + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_centres[:-1] + log_centres[1:])
    log_edges[0] = log_centres[0] - 0.5 * (log_centres[1] - log_centres[0])
    log_edges[-1] = log_centres[-1] + 0.5 * (log_centres[-1] - log_centres[-2])
    return np.exp(log_edges)


def plot_distance_map(
    bank: LifetimeTemplateBank,
    distances: np.ndarray,
    summary: dict,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    use_report_style()
    figure, axis = plt.subplots(figsize=PLOT_CONFIG.distance_map_figsize)
    mesh = axis.pcolormesh(
        logarithmic_cell_edges(bank.photon_ctau_m),
        logarithmic_cell_edges(bank.su2_ctau_m),
        np.asarray(distances).T,
        shading="flat",
        vmin=PLOT_CONFIG.distance_vmin,
        vmax=PLOT_CONFIG.distance_vmax,
        cmap=PLOT_CONFIG.distance_cmap,
    )
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(r"$D_{\mathrm{TV}}$")
    axis.scatter(
        [summary["minimum_photon_ctau_m"]],
        [summary["minimum_su2_ctau_m"]],
        marker="*",
        s=PLOT_CONFIG.distance_minimum_marker_size,
        edgecolors="black",
        linewidths=0.9,
        label=r"Minimum $D_{\mathrm{TV}}$" + f" = {summary['minimum_D_TV']:.4f}",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Photophilic $c\tau_a$ [m]")
    axis.set_ylabel(r"$SU(2)_L$ $c\tau_a$ [m]")
    axis.legend(loc="best")
    style_axis(axis)
    figure.tight_layout()
    return _save_pdf_png(figure, output_stem)


def plot_minimum_pair_spectra(
    bank: LifetimeTemplateBank,
    summary: dict,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    photon_index = int(summary["minimum_photon_lifetime_index"])
    su2_index = int(summary["minimum_su2_lifetime_index"])
    use_report_style()
    figure, axis = plt.subplots(figsize=PLOT_CONFIG.spectrum_figsize)
    axis.stairs(
        bank.photon_probabilities[photon_index],
        bank.energy_edges_gev,
        linewidth=PLOT_CONFIG.spectrum_line_width,
        label="ALP-photon, " + rf"$c\tau_a={summary['minimum_photon_ctau_m']:.3g}\,$m",
    )
    axis.stairs(
        bank.su2_probabilities[su2_index],
        bank.energy_edges_gev,
        linewidth=PLOT_CONFIG.spectrum_line_width,
        label=r"ALP-$SU(2)_L$, " + rf"$c\tau_a={summary['minimum_su2_ctau_m']:.3g}\,$m",
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"$E_a$ [GeV]")
    axis.set_ylabel("Probability per adaptive energy bin")
    axis.grid(True, alpha=PLOT_CONFIG.grid_alpha)
    axis.legend(
        title=rf"$m_a={bank.mass_gev:g}\,$GeV, "
        + rf"$D_{{\mathrm{{TV}}}}={summary['minimum_D_TV']:.4f}$"
    )
    style_axis(axis)
    figure.tight_layout()
    return _save_pdf_png(figure, output_stem)


def plot_profiled_accuracy(
    curve: pd.DataFrame,
    *,
    mass_gev: float,
    target_accuracy: float,
    threshold: int | None,
    output_stem: Path,
) -> tuple[Path, Path]:
    use_report_style()
    figure, axis = plt.subplots(figsize=PLOT_CONFIG.profiled_figsize)
    events = curve["number_of_events"].to_numpy(dtype=int)
    axis.plot(events, curve["photon_truth_worst_accuracy"], marker="o", label="Worst photophilic truth lifetime")
    axis.plot(events, curve["su2_truth_worst_accuracy"], marker="s", label=r"Worst $SU(2)_L$ truth lifetime")
    axis.plot(events, curve["worst_case_correct_fraction"], marker="^", linewidth=2.2, label="Overall worst case")
    axis.axhline(target_accuracy, linestyle="--", linewidth=1.4, label=f"Target = {100 * target_accuracy:.0f}%")
    if threshold is not None:
        axis.axvline(threshold, linestyle=":", linewidth=1.4, label=f"Persistent threshold: N = {threshold}")
    axis.set_xlabel("Observed ALP decays, $N$")
    axis.set_ylabel("Correct-classification probability")
    axis.set_xticks(events)
    axis.set_ylim(0.45, 1.01)
    axis.grid(True, alpha=PLOT_CONFIG.grid_alpha)
    axis.legend(loc="lower right")
    style_axis(axis)
    figure.tight_layout()
    return _save_pdf_png(figure, output_stem)


def plot_profiled_thresholds(
    summary: pd.DataFrame,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    use_report_style()
    figure, axis = plt.subplots(figsize=PLOT_CONFIG.profiled_figsize)
    masses = summary["mass_GeV"].to_numpy(dtype=float)
    reached = summary["threshold_reached"].to_numpy(dtype=bool)
    maximum_tested = int(summary["maximum_tested_events"].max())
    requirements = np.where(
        reached,
        summary["minimum_persistent_events"].to_numpy(dtype=float),
        maximum_tested + 1.0,
    )
    axis.plot(masses, requirements, marker="o", linewidth=1.8)
    for mass, reached_here, value in zip(masses, reached, requirements):
        if not reached_here:
            axis.annotate(f">{maximum_tested}", (mass, value), textcoords="offset points", xytext=(0, 6), ha="center")
    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel("Minimum persistent observed events")
    axis.set_xscale("log")
    finite = requirements[reached]
    upper = max(5.5, float(np.max(finite)) + 1.0) if len(finite) else maximum_tested + 1.8
    axis.set_ylim(0.5, upper)
    axis.set_yticks(np.arange(1, int(np.ceil(upper)), dtype=int))
    axis.grid(True, alpha=PLOT_CONFIG.grid_alpha)
    style_axis(axis)
    figure.tight_layout()
    return _save_pdf_png(figure, output_stem)
