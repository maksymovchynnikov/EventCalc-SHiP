"""Create report-ready accepted energy-spectrum figures.

This module reads existing analysis products and does not rerun the full scan.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alp_discrimination.cache import CacheStore
from alp_discrimination.config import PROFILES, AnalysisConfig, get_config
from alp_discrimination.eventcalc.adapter import EventCalcAdapter
from alp_discrimination.physics.models import ALP_PHOTON_COMBINED, ALP_SU2L, MODELS
from alp_discrimination.paths import profile_output_dir
from alp_discrimination.plotting.style import PLOT_CONFIG, style_axis, use_report_style
from alp_discrimination.eventcalc.selections import MotherLevelSelection
from alp_discrimination.physics.spectra import HistogramSpectrum, WeightedSpectrum, normalized_weighted_spectrum
from alp_discrimination.workflows import write_dataframe, write_manifest


DEFAULT_MASSES_GEV = (0.3, 1.0)
MODEL_LINESTYLES = {
    ALP_PHOTON_COMBINED.identifier: "-",
    ALP_SU2L.identifier: "--",
}
PANEL_LABELS = {
    "mother_level": "Before ECAL requirement",
    "diphoton_ecal": "After diphoton ECAL requirement",
}


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="production",
        help="Cache/output namespace. The report should normally use production.",
    )
    parser.add_argument(
        "--masses",
        type=float,
        nargs="+",
        default=list(DEFAULT_MASSES_GEV),
        metavar="GEV",
        help="Masses to plot; default: 0.3 1.0.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute requested cache entries instead of loading them.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Neither read nor write EventCalc caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; default: analysis2/outputs/<profile>/report.",
    )
    return parser.parse_args()


def _mass_rows(data: pd.DataFrame, mass_gev: float) -> pd.DataFrame:
    return data[np.isclose(data["mass_GeV"].to_numpy(float), mass_gev, rtol=0.0, atol=1.0e-12)]


def _stored_midpoint_path(config: AnalysisConfig) -> Path:
    return (
        profile_output_dir(config.name, "same_lifetime_discrimination")
        / "selected_lifetime_points.csv"
    )


def _stored_midpoint_from_summary(config: AnalysisConfig, mass_gev: float) -> float | None:
    """Return the saved equal-lifetime midpoint when that workflow has been run."""
    candidates = (
        _stored_midpoint_path(config),
        profile_output_dir(config.name, "same_lifetime_discrimination")
        / "discrimination_grid_summary.csv",
    )
    for path in candidates:
        if not path.exists():
            continue
        data = pd.read_csv(path)
        required = {"mass_GeV", "ctau_m"}
        if not required.issubset(data.columns):
            continue
        rows = _mass_rows(data, mass_gev)
        if "lifetime_label" in rows.columns:
            rows = rows[rows["lifetime_label"].astype(str).str.lower() == "mid"]
        if len(rows) == 1:
            value = float(rows.iloc[0]["ctau_m"])
            if np.isfinite(value) and value > 0.0:
                return value
    return None


def _common_observable_midpoint(config: AnalysisConfig, mass_gev: float) -> float:
    """Geometric midpoint of the two-model common observable lifetime interval."""
    path = (
        profile_output_dir(config.name, "scan_ctau_ranges")
        / "observable_lifetime_domains.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the production lifetime-domain scan first."
        )

    data = pd.read_csv(path)
    required = {
        "mass_GeV",
        "model_id",
        "template_domain_lower_m",
        "template_domain_upper_m",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    rows = _mass_rows(data, mass_gev)
    expected_models = {model.identifier for model in MODELS}
    found_models = set(rows["model_id"].astype(str))
    if found_models != expected_models:
        raise ValueError(
            f"{path} has models {sorted(found_models)} at m_a={mass_gev:g} GeV; "
            f"expected {sorted(expected_models)}"
        )

    lower_m = float(rows["template_domain_lower_m"].max())
    upper_m = float(rows["template_domain_upper_m"].min())
    if not (np.isfinite(lower_m) and np.isfinite(upper_m) and 0.0 < lower_m < upper_m):
        raise ValueError(
            f"No finite common observable lifetime interval at m_a={mass_gev:g} GeV"
        )
    return float(np.sqrt(lower_m * upper_m))


def benchmark_lifetime(config: AnalysisConfig, mass_gev: float) -> tuple[float, str]:
    stored = _stored_midpoint_from_summary(config, mass_gev)
    if stored is not None:
        return stored, "stored_same_lifetime_midpoint"
    return _common_observable_midpoint(config, mass_gev), "common_interval_geometric_midpoint"


def _validate_requested_masses(config: AnalysisConfig, masses_gev: tuple[float, ...]) -> None:
    if not masses_gev:
        raise ValueError("At least one mass is required.")
    if len(set(masses_gev)) != len(masses_gev):
        raise ValueError("Masses must not be repeated.")
    for mass_gev in masses_gev:
        if not np.isfinite(mass_gev) or mass_gev <= 0.0:
            raise ValueError("Masses must be finite and positive.")
        if not any(
            np.isclose(mass_gev, configured, rtol=0.0, atol=1.0e-12)
            for configured in config.seed_policy.mass_order_gev
        ):
            raise ValueError(
                f"m_a={mass_gev:g} GeV is not in the immutable production mass order "
                f"{config.seed_policy.mass_order_gev}"
            )


def _evaluate(
    adapter: EventCalcAdapter,
    config: AnalysisConfig,
    model_id: str,
    mass_gev: float,
    ctau_m: float,
) -> WeightedSpectrum:
    model_seed = config.seed_policy.model_seed(
        mass_gev,
        model_id,
        seed_offset=config.templates.seed_offset,
    )
    spectrum = adapter.evaluate_model(
        model_id,
        mass_gev,
        ctau_m,
        model_seed,
        "spectrum",
    )
    if spectrum.expected_events <= 0.0:
        raise RuntimeError(
            f"Non-positive event weight for {model_id}, m_a={mass_gev:g} GeV, "
            f"c tau={ctau_m:g} m"
        )
    return spectrum


def _histogram(
    spectrum: WeightedSpectrum,
    config: AnalysisConfig,
) -> HistogramSpectrum:
    edges = np.geomspace(
        spectrum.mass_gev,
        config.energy_max_gev,
        config.initial_energy_bins + 1,
    )
    return normalized_weighted_spectrum(spectrum, edges)


def _table_rows(
    *,
    profile: str,
    benchmark_source: str,
    spectrum: WeightedSpectrum,
    histogram: HistogramSpectrum,
    model_label: str,
) -> list[dict]:
    rows: list[dict] = []
    for bin_index, (
        energy_low_gev,
        energy_high_gev,
        probability,
        density,
        density_error,
        bin_n_eff,
    ) in enumerate(
        zip(
            histogram.energy_edges_gev[:-1],
            histogram.energy_edges_gev[1:],
            histogram.bin_probabilities,
            histogram.density_per_gev,
            histogram.density_error_per_gev,
            histogram.effective_samples_per_bin,
        )
    ):
        rows.append(
            {
                "profile": profile,
                "benchmark_source": benchmark_source,
                "selection_name": spectrum.selection_name,
                "panel_label": PANEL_LABELS[spectrum.selection_name],
                "mass_GeV": spectrum.mass_gev,
                "ctau_m": spectrum.ctau_m,
                "model_id": spectrum.model_id,
                "model_label": model_label,
                "source": spectrum.source,
                "bin_index": bin_index,
                "energy_low_GeV": energy_low_gev,
                "energy_high_GeV": energy_high_gev,
                "bin_probability": probability,
                "density_per_GeV": density,
                "density_error_per_GeV": density_error,
                "bin_N_eff": bin_n_eff,
                "expected_events": spectrum.expected_events,
                "expected_events_before_ECAL": spectrum.preselection_expected_events,
                "weighted_ECAL_efficiency": spectrum.selection_efficiency_weighted,
                "total_N_eff": spectrum.total_n_eff,
                "accepted_samples": spectrum.accepted_samples,
                "cache_key": spectrum.cache_key,
            }
        )
    return rows


def _mass_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _output_stem(masses_gev: tuple[float, ...]) -> str:
    return "energy_spectra_before_after_ecal_ma_" + "_".join(
        _mass_token(value) for value in masses_gev
    )


def _draw_figure(
    histograms: dict[tuple[str, float, str], HistogramSpectrum],
    benchmarks: dict[float, tuple[float, str]],
    masses_gev: tuple[float, ...],
    output_dir: Path,
) -> tuple[Path, Path]:
    use_report_style()
    n_masses = len(masses_gev)
    figure, axes = plt.subplots(
        2,
        n_masses,
        figsize=(6.2 * n_masses, 8.4),
        squeeze=False,
        sharex="col",
    )

    selection_order = ("mother_level", "diphoton_ecal")
    selection_titles = {
        "mother_level": "Without ECAL requirement",
        "diphoton_ecal": "With diphoton ECAL requirement",
    }

    for column, mass_gev in enumerate(masses_gev):
        ctau_m, _ = benchmarks[mass_gev]
        for row, selection_name in enumerate(selection_order):
            axis = axes[row, column]
            for model in MODELS:
                histogram = histograms[(selection_name, mass_gev, model.identifier)]
                axis.stairs(
                    histogram.density_per_gev,
                    histogram.energy_edges_gev,
                    linewidth=PLOT_CONFIG.spectrum_line_width,
                    linestyle=MODEL_LINESTYLES[model.identifier],
                    label=model.plot_label,
                )

            axis.set_xscale("log")
            axis.set_xlim(
                mass_gev,
                max(item.energy_edges_gev[-1] for item in histograms.values()),
            )
            axis.set_ylim(0.0, None)
            axis.grid(True, which="both", alpha=PLOT_CONFIG.grid_alpha)

            # Clear panel label
            axis.text(
                0.03,
                0.95,
                selection_titles[selection_name],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="0.7"),
            )

            # Report titles contain only the mass and plotted observable.
            if row == 0:
                axis.set_title(
                    rf"$m_a={mass_gev:g}\,\mathrm{{GeV}},\ E_a$"
                )
                axis.text(
                    0.97,
                    0.06,
                    rf"$c\tau_a={ctau_m:.3g}\,\mathrm{{m}}$",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                )

            if row == 1:
                axis.set_xlabel(r"$E_a$ [GeV]")

            style_axis(axis)

    axes[0, 0].set_ylabel(
        r"$(1/N_{\mathrm{events}})\,dN_{\mathrm{events}}/dE_a$ "
        r"[GeV$^{-1}$]"
    )
    axes[1, 0].set_ylabel(
        r"$(1/N_{\mathrm{events}})\,dN_{\mathrm{events}}/dE_a$ "
        r"[GeV$^{-1}$]"
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.015),
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(masses_gev)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=PLOT_CONFIG.png_dpi, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    args = parse_arguments()
    config = get_config(args.profile)
    masses_gev = tuple(float(value) for value in args.masses)
    _validate_requested_masses(config, masses_gev)

    output_dir = args.output_dir or profile_output_dir(config.name, "report")
    cache = CacheStore(config.name, enabled=not args.no_cache)

    ecal_adapter = EventCalcAdapter(
        config,
        cache=cache,
        force=args.force,
    )
    mother_config = replace(config, selection_name="mother_level")
    mother_adapter = EventCalcAdapter(
        mother_config,
        cache=cache,
        force=args.force,
        selection=MotherLevelSelection(),
    )

    benchmarks = {
        mass_gev: benchmark_lifetime(config, mass_gev)
        for mass_gev in masses_gev
    }
    histograms: dict[tuple[str, float, str], HistogramSpectrum] = {}
    table_rows: list[dict] = []

    for mass_gev in masses_gev:
        ctau_m, benchmark_source = benchmarks[mass_gev]
        print(
            f"m_a={mass_gev:g} GeV: c tau={ctau_m:.12g} m "
            f"({benchmark_source})"
        )
        for model in MODELS:
            for adapter in (mother_adapter, ecal_adapter):
                spectrum = _evaluate(
                    adapter,
                    config,
                    model.identifier,
                    mass_gev,
                    ctau_m,
                )
                histogram = _histogram(spectrum, config)
                histograms[
                    (spectrum.selection_name, mass_gev, model.identifier)
                ] = histogram
                table_rows.extend(
                    _table_rows(
                        profile=config.name,
                        benchmark_source=benchmark_source,
                        spectrum=spectrum,
                        histogram=histogram,
                        model_label=model.plot_label,
                    )
                )
                print(
                    f"  {spectrum.selection_name:14s} "
                    f"{model.identifier:20s} "
                    f"N={spectrum.expected_events:.6g}, "
                    f"N_eff={spectrum.total_n_eff:.6g}"
                )

    table = pd.DataFrame(table_rows).sort_values(
        ["mass_GeV", "selection_name", "model_id", "bin_index"]
    )
    csv_path = write_dataframe(
        table,
        output_dir / f"{_output_stem(masses_gev)}.csv",
    )
    pdf_path, png_path = _draw_figure(
        histograms,
        benchmarks,
        masses_gev,
        output_dir,
    )
    manifest_path = write_manifest(
        config,
        "plot_report_energy_spectra",
        output_dir,
        cache_stats=cache.counter_snapshot(),
        artifacts=[csv_path, pdf_path, png_path],
        extra={
            "masses_GeV": list(masses_gev),
            "benchmark_lifetimes": {
                f"{mass_gev:g}": {
                    "ctau_m": benchmarks[mass_gev][0],
                    "source": benchmarks[mass_gev][1],
                }
                for mass_gev in masses_gev
            },
            "normalization": "each model and panel independently normalized to unit area",
        },
    )

    print(f"Saved table:    {csv_path}")
    print(f"Saved PDF:      {pdf_path}")
    print(f"Saved PNG:      {png_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Cache summary:  {cache.counter_snapshot()}")


if __name__ == "__main__":
    main()