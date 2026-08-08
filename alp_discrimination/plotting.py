"""Shared plotting only; no EventCalc calls or scientific mutation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import PLOT_CONFIG, style_axis, use_report_style
from .spectra import HistogramSpectrum


def plot_ctau_rates(data: pd.DataFrame, threshold: float, output_path: Path) -> Path:
    use_report_style()
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    masses = sorted(data["mass_GeV"].unique())
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(masses)))
    styles = {"alp_photon_combined": "-", "alp_su2l": "--"}
    for colour, mass_gev in zip(colours, masses):
        mass_data = data[np.isclose(data["mass_GeV"], mass_gev)]
        for model_id, model_data in mass_data.groupby("model_id"):
            model_data = model_data.sort_values("ctau_m")
            axis.loglog(
                model_data["ctau_m"], model_data["N_events"], styles.get(model_id, "-"),
                marker="o", markersize=2.5, linewidth=1.3, color=colour,
                label=rf"$m_a={mass_gev:g}$ GeV, {model_id}",
            )
    axis.axhline(threshold, linestyle=":", color="black", label=rf"$N_{{events}}={threshold:g}$")
    axis.set(xlabel=r"$c\tau$ [m]", ylabel=r"$N_{events}$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_lifetime_spectra(
    spectra: Mapping[float, HistogramSpectrum], model_label: str, mass_gev: float,
    reference_ctau_m: float, n_eff_warning: float, output_path: Path,
) -> Path:
    use_report_style()
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    for ctau_m in sorted(spectra):
        spectrum = spectra[ctau_m]
        centres = np.sqrt(spectrum.energy_edges_gev[:-1] * spectrum.energy_edges_gev[1:])
        low = (spectrum.sum_weights_per_bin > 0.0) & (
            spectrum.effective_samples_per_bin < n_eff_warning
        )
        density = spectrum.density_per_gev.copy()
        density[low] = np.nan
        reference = np.isclose(ctau_m, reference_ctau_m)
        label = rf"$c\tau={ctau_m:g}$ m" + (" (reference)" if reference else "")
        stairs = axis.stairs(
            density, spectrum.energy_edges_gev, label=label,
            linewidth=3.0 if reference else 1.5, zorder=10 if reference else 2,
        )
        if np.any(low):
            axis.errorbar(
                centres[low], spectrum.density_per_gev[low],
                yerr=spectrum.density_error_per_gev[low], fmt="x",
                color=stairs.get_edgecolor(), markersize=5, capsize=2,
            )
    axis.set_xscale("log")
    axis.set(xlim=(mass_gev, max(item.energy_edges_gev[-1] for item in spectra.values())),
             ylim=(0.0, None), xlabel=r"$E_a$ [GeV]",
             ylabel=r"$(1/N_{events})\,dN_{events}/dE_a$ [GeV$^{-1}$]")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(title="Lifetime")
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_accuracy(data: pd.DataFrame, mass_gev: float, ctau_m: float, output_path: Path) -> Path:
    use_report_style()
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    axis.plot(data["number_of_events"], data["photon_correct_fraction"], label="Photon true")
    axis.plot(data["number_of_events"], data["su2_correct_fraction"], label=r"$SU(2)_L$ true")
    axis.plot(
        data["number_of_events"], data["worst_case_correct_fraction"],
        linewidth=2.5, label="Worst-case accuracy",
    )
    for target in (0.90, 0.95, 0.99):
        axis.axhline(target, linestyle="--", linewidth=0.8, label=f"{100 * target:.0f}% target")
    axis.set(xlabel="Number of observed ALP decays", ylabel="Correct-classification probability",
             ylim=(0.45, 1.01))
    axis.grid(True, alpha=0.3)
    axis.legend()
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_validated_thresholds(data: pd.DataFrame, output_path: Path) -> Path:
    use_report_style()
    figure, axis = plt.subplots(figsize=(8.0, 5.8))
    for target, group in data.groupby("target_accuracy", sort=True):
        group = group.sort_values("mass_GeV")
        axis.plot(
            group["mass_GeV"], group["conservative_required_events"], marker="o",
            label=f"{100 * target:.0f}% worst case",
        )
        axis.fill_between(
            group["mass_GeV"], group["minimum_events_over_lifetimes_and_seeds"],
            group["conservative_required_events"], alpha=0.15,
        )
    axis.set(xlabel=r"$m_a$ [GeV]", ylabel="Required observed ALP decays")
    axis.grid(True, alpha=0.3)
    axis.legend()
    style_axis(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def draw_event_contours(
    axis: plt.Axes, data: pd.DataFrame, event_levels, *, color: str | None = None,
) -> None:
    for level, style in zip(event_levels, PLOT_CONFIG.event_density_line_styles):
        points = data[np.isclose(data["event_level"], level)].sort_values("mass_GeV")
        masses = points["mass_GeV"].to_numpy(float)
        lower, upper = (points[name].to_numpy(float) for name in (
            "lower_coupling_GeV_inv", "upper_coupling_GeV_inv"
        ))
        line, = axis.plot(
            masses[np.isfinite(lower)], lower[np.isfinite(lower)], linestyle=style,
            linewidth=PLOT_CONFIG.event_density_line_width,
            color=color,
            label=rf"$N_{{events}}={level:g}$",
        )
        axis.plot(
            masses[np.isfinite(upper)], upper[np.isfinite(upper)], linestyle=style,
            linewidth=PLOT_CONFIG.event_density_line_width,
            color=line.get_color(),
        )


def plot_event_rate_curves(data: pd.DataFrame, event_levels, output_dir: Path) -> list[Path]:
    use_report_style()
    paths = []
    for model, model_data in data.groupby("model", sort=True):
        figure, axis = plt.subplots(figsize=PLOT_CONFIG.event_rate_figsize)
        for mass_gev, mass_data in model_data.groupby("mass_GeV", sort=True):
            mass_data = mass_data.sort_values("coupling_GeV_inv")
            rates = mass_data["N_events"].where(mass_data["N_events"] >= 0.1)
            axis.plot(mass_data["coupling_GeV_inv"], rates, linewidth=1.0,
                      label=rf"$m_a={mass_gev:g}$ GeV")
        for level in event_levels:
            axis.axhline(level, linestyle=":", linewidth=0.8)
        axis.set(xscale="log", yscale="log", ylim=(0.1, None), xlabel=r"Coupling [GeV$^{-1}$]",
                 ylabel=r"$N_{events}$")
        axis.grid(True, which="both", alpha=PLOT_CONFIG.event_density_grid_alpha)
        style_axis(axis)
        figure.tight_layout()
        path = output_dir / f"event_rate_vs_coupling_{model.lower()}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
