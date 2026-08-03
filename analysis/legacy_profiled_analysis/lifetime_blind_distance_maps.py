"""Build 2D lifetime-distance maps from the saved frozen-reference template banks.

This is the second post-processing step of the lifetime-blind ALP analysis.
For every mass, it compares every photophilic lifetime template with every
SU(2)_L lifetime template using the total-variation distance

    D_TV(p, q) = 1/2 * sum_i |p_i - q_i|.

The script does not launch EventCalc. It reads the compact ``.npz`` banks
created by ``analysis.lifetime_blind_discrimination`` and writes

* one long-form distance table per mass;
* one combined summary table;
* a 2D lifetime-distance heatmap per mass;
* an overlay of the least-distinguishable spectrum pair per mass; and
* a bin-by-bin table for that least-distinguishable pair.

Run from the repository root with

    python -m analysis.lifetime_blind_distance_maps

Useful examples:

    python -m analysis.lifetime_blind_distance_maps --masses 0.3
    python -m analysis.lifetime_blind_distance_maps --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from analysis.plot_style import style_axis, use_report_style


ANALYSIS_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ANALYSIS_DIR / "lifetime_blind_discrimination" / "template_banks"
OUTPUT_DIR = ANALYSIS_DIR / "lifetime_blind_discrimination" / "distance_maps"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

REQUIRED_KEYS = {
    "mass_GeV",
    "energy_edges_GeV",
    "photon_ctau_m",
    "photon_probabilities",
    "photon_n_events",
    "su2_ctau_m",
    "su2_probabilities",
    "su2_n_events",
}


# -----------------------------------------------------------------------------
# Validation and distance calculation
# -----------------------------------------------------------------------------


def _validate_probability_matrix(
    probabilities: np.ndarray,
    *,
    number_of_lifetimes: int,
    number_of_bins: int,
    label: str,
    path: Path,
) -> None:
    """Validate one model's probability-template matrix."""
    if probabilities.shape != (number_of_lifetimes, number_of_bins):
        raise ValueError(
            f"Unexpected shape for {label} probabilities in {path}:\n"
            f"  expected {(number_of_lifetimes, number_of_bins)}\n"
            f"  found    {probabilities.shape}"
        )
    if np.any(~np.isfinite(probabilities)):
        raise ValueError(f"Non-finite {label} probabilities in {path}.")
    if np.any(probabilities <= 0.0):
        raise ValueError(f"Non-positive {label} probabilities in {path}.")
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(f"The {label} templates are not normalized in {path}.")


def load_template_bank(path: Path) -> dict[str, np.ndarray | float | Path]:
    """Load and strictly validate one compact template bank."""
    with np.load(path) as raw:
        missing = REQUIRED_KEYS - set(raw.files)
        if missing:
            raise ValueError(f"Missing keys in {path}: {sorted(missing)}")

        bank: dict[str, np.ndarray | float | Path] = {
            "path": path,
            "mass_GeV": float(np.asarray(raw["mass_GeV"]).item()),
            "energy_edges_GeV": np.asarray(raw["energy_edges_GeV"], dtype=float),
            "photon_ctau_m": np.asarray(raw["photon_ctau_m"], dtype=float),
            "photon_probabilities": np.asarray(
                raw["photon_probabilities"], dtype=float
            ),
            "photon_n_events": np.asarray(raw["photon_n_events"], dtype=float),
            "su2_ctau_m": np.asarray(raw["su2_ctau_m"], dtype=float),
            "su2_probabilities": np.asarray(raw["su2_probabilities"], dtype=float),
            "su2_n_events": np.asarray(raw["su2_n_events"], dtype=float),
        }

    mass_gev = float(bank["mass_GeV"])
    energy_edges = np.asarray(bank["energy_edges_GeV"], dtype=float)
    photon_ctau = np.asarray(bank["photon_ctau_m"], dtype=float)
    su2_ctau = np.asarray(bank["su2_ctau_m"], dtype=float)
    photon_probabilities = np.asarray(bank["photon_probabilities"], dtype=float)
    su2_probabilities = np.asarray(bank["su2_probabilities"], dtype=float)
    photon_n_events = np.asarray(bank["photon_n_events"], dtype=float)
    su2_n_events = np.asarray(bank["su2_n_events"], dtype=float)

    if not np.isfinite(mass_gev) or mass_gev <= 0.0:
        raise ValueError(f"Invalid mass in {path}: {mass_gev}")
    if energy_edges.ndim != 1 or len(energy_edges) < 2:
        raise ValueError(f"Invalid energy edges in {path}.")
    if np.any(~np.isfinite(energy_edges)) or np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError(f"Energy edges are not finite and increasing in {path}.")

    for label, lifetimes, event_rates in (
        ("photon", photon_ctau, photon_n_events),
        ("su2", su2_ctau, su2_n_events),
    ):
        if lifetimes.ndim != 1 or len(lifetimes) < 2:
            raise ValueError(f"Invalid {label} lifetime grid in {path}.")
        if np.any(~np.isfinite(lifetimes)) or np.any(lifetimes <= 0.0):
            raise ValueError(f"Invalid {label} lifetimes in {path}.")
        if np.any(np.diff(lifetimes) <= 0.0):
            raise ValueError(f"The {label} lifetime grid is not increasing in {path}.")
        if event_rates.shape != lifetimes.shape:
            raise ValueError(f"The {label} event-rate array has the wrong shape in {path}.")
        if np.any(~np.isfinite(event_rates)) or np.any(event_rates < 0.0):
            raise ValueError(f"Invalid {label} event rates in {path}.")

    number_of_bins = len(energy_edges) - 1
    _validate_probability_matrix(
        photon_probabilities,
        number_of_lifetimes=len(photon_ctau),
        number_of_bins=number_of_bins,
        label="photon",
        path=path,
    )
    _validate_probability_matrix(
        su2_probabilities,
        number_of_lifetimes=len(su2_ctau),
        number_of_bins=number_of_bins,
        label="SU(2)_L",
        path=path,
    )

    return bank


def total_variation_matrix(
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
) -> np.ndarray:
    """Return D_TV for every photon/SU(2)_L lifetime pair."""
    distances = 0.5 * np.sum(
        np.abs(
            photon_probabilities[:, np.newaxis, :]
            - su2_probabilities[np.newaxis, :, :]
        ),
        axis=2,
    )
    if np.any(~np.isfinite(distances)):
        raise RuntimeError("The total-variation matrix contains non-finite values.")
    if np.any(distances < -1.0e-14) or np.any(distances > 1.0 + 1.0e-14):
        raise RuntimeError("A total-variation distance lies outside [0, 1].")
    return np.clip(distances, 0.0, 1.0)


def distance_table(bank: dict, distances: np.ndarray) -> pd.DataFrame:
    """Create one long-form row per lifetime pair."""
    mass_gev = float(bank["mass_GeV"])
    photon_ctau = np.asarray(bank["photon_ctau_m"], dtype=float)
    su2_ctau = np.asarray(bank["su2_ctau_m"], dtype=float)
    photon_n_events = np.asarray(bank["photon_n_events"], dtype=float)
    su2_n_events = np.asarray(bank["su2_n_events"], dtype=float)

    photon_indices, su2_indices = np.indices(distances.shape)
    return pd.DataFrame(
        {
            "mass_GeV": mass_gev,
            "photon_lifetime_index": photon_indices.ravel(),
            "photon_ctau_m": photon_ctau[photon_indices.ravel()],
            "photon_N_events": photon_n_events[photon_indices.ravel()],
            "su2_lifetime_index": su2_indices.ravel(),
            "su2_ctau_m": su2_ctau[su2_indices.ravel()],
            "su2_N_events": su2_n_events[su2_indices.ravel()],
            "D_TV": distances.ravel(),
        }
    ).sort_values(
        ["photon_lifetime_index", "su2_lifetime_index"],
        ignore_index=True,
    )


def summarize_distance_matrix(bank: dict, distances: np.ndarray) -> dict:
    """Return the least- and most-distinguishable lifetime pairs."""
    photon_ctau = np.asarray(bank["photon_ctau_m"], dtype=float)
    su2_ctau = np.asarray(bank["su2_ctau_m"], dtype=float)
    photon_n_events = np.asarray(bank["photon_n_events"], dtype=float)
    su2_n_events = np.asarray(bank["su2_n_events"], dtype=float)

    minimum_flat = int(np.argmin(distances))
    maximum_flat = int(np.argmax(distances))
    minimum_indices = np.unravel_index(minimum_flat, distances.shape)
    maximum_indices = np.unravel_index(maximum_flat, distances.shape)
    photon_min_index, su2_min_index = map(int, minimum_indices)
    photon_max_index, su2_max_index = map(int, maximum_indices)

    return {
        "mass_GeV": float(bank["mass_GeV"]),
        "number_of_energy_bins": len(np.asarray(bank["energy_edges_GeV"])) - 1,
        "number_of_photon_lifetimes": len(photon_ctau),
        "number_of_su2_lifetimes": len(su2_ctau),
        "minimum_D_TV": float(distances[photon_min_index, su2_min_index]),
        "minimum_photon_lifetime_index": photon_min_index,
        "minimum_photon_ctau_m": float(photon_ctau[photon_min_index]),
        "minimum_photon_N_events": float(photon_n_events[photon_min_index]),
        "minimum_su2_lifetime_index": su2_min_index,
        "minimum_su2_ctau_m": float(su2_ctau[su2_min_index]),
        "minimum_su2_N_events": float(su2_n_events[su2_min_index]),
        "maximum_D_TV": float(distances[photon_max_index, su2_max_index]),
        "maximum_photon_lifetime_index": photon_max_index,
        "maximum_photon_ctau_m": float(photon_ctau[photon_max_index]),
        "maximum_su2_lifetime_index": su2_max_index,
        "maximum_su2_ctau_m": float(su2_ctau[su2_max_index]),
    }


# -----------------------------------------------------------------------------
# Plotting and saved diagnostics
# -----------------------------------------------------------------------------


def logarithmic_cell_edges(centres: np.ndarray) -> np.ndarray:
    """Construct positive pcolormesh edges from increasing log-grid centres."""
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or len(centres) < 2:
        raise ValueError("At least two lifetime centres are required.")
    if np.any(centres <= 0.0) or np.any(np.diff(centres) <= 0.0):
        raise ValueError("Lifetime centres must be positive and increasing.")

    log_centres = np.log(centres)
    log_edges = np.empty(len(centres) + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_centres[:-1] + log_centres[1:])
    log_edges[0] = log_centres[0] - 0.5 * (log_centres[1] - log_centres[0])
    log_edges[-1] = log_centres[-1] + 0.5 * (
        log_centres[-1] - log_centres[-2]
    )
    return np.exp(log_edges)


def plot_distance_map(
    bank: dict,
    distances: np.ndarray,
    summary: dict,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Plot one 2D D_TV map and mark its global minimum."""
    photon_ctau = np.asarray(bank["photon_ctau_m"], dtype=float)
    su2_ctau = np.asarray(bank["su2_ctau_m"], dtype=float)
    photon_edges = logarithmic_cell_edges(photon_ctau)
    su2_edges = logarithmic_cell_edges(su2_ctau)

    use_report_style()
    figure, axis = plt.subplots(figsize=(8.2, 6.4))
    mesh = axis.pcolormesh(
        photon_edges,
        su2_edges,
        distances.T,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(r"$D_{\mathrm{TV}}$")

    axis.scatter(
        [summary["minimum_photon_ctau_m"]],
        [summary["minimum_su2_ctau_m"]],
        marker="*",
        s=180,
        edgecolors="black",
        linewidths=0.9,
        label=(
            r"Minimum $D_{\mathrm{TV}}$"
            + f" = {summary['minimum_D_TV']:.4f}"
        ),
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Photophilic $c\tau_a$ [m]")
    axis.set_ylabel(r"$SU(2)_L$ $c\tau_a$ [m]")
    axis.legend(loc="best")
    style_axis(axis)
    figure.tight_layout()

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def least_distinguishable_bin_table(bank: dict, summary: dict) -> pd.DataFrame:
    """Create a bin-by-bin table for the global-minimum pair."""
    photon_index = int(summary["minimum_photon_lifetime_index"])
    su2_index = int(summary["minimum_su2_lifetime_index"])
    energy_edges = np.asarray(bank["energy_edges_GeV"], dtype=float)
    photon = np.asarray(bank["photon_probabilities"], dtype=float)[photon_index]
    su2 = np.asarray(bank["su2_probabilities"], dtype=float)[su2_index]

    absolute_difference = np.abs(photon - su2)
    return pd.DataFrame(
        {
            "mass_GeV": float(bank["mass_GeV"]),
            "photon_ctau_m": float(summary["minimum_photon_ctau_m"]),
            "su2_ctau_m": float(summary["minimum_su2_ctau_m"]),
            "bin_index": np.arange(len(photon), dtype=int),
            "energy_low_GeV": energy_edges[:-1],
            "energy_high_GeV": energy_edges[1:],
            "photon_probability": photon,
            "su2_probability": su2,
            "absolute_probability_difference": absolute_difference,
            "D_TV_bin_contribution": 0.5 * absolute_difference,
        }
    )


def plot_least_distinguishable_spectra(
    bank: dict,
    summary: dict,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Overlay the two templates at the global D_TV minimum."""
    photon_index = int(summary["minimum_photon_lifetime_index"])
    su2_index = int(summary["minimum_su2_lifetime_index"])
    energy_edges = np.asarray(bank["energy_edges_GeV"], dtype=float)
    photon = np.asarray(bank["photon_probabilities"], dtype=float)[photon_index]
    su2 = np.asarray(bank["su2_probabilities"], dtype=float)[su2_index]

    use_report_style()
    figure, axis = plt.subplots(figsize=(8.2, 5.8))
    axis.stairs(
        photon,
        energy_edges,
        linewidth=2.0,
        label=(
            "ALP-photon, "
            + rf"$c\tau_a={summary['minimum_photon_ctau_m']:.3g}\,$m"
        ),
    )
    axis.stairs(
        su2,
        energy_edges,
        linewidth=2.0,
        label=(
            r"ALP-$SU(2)_L$, "
            + rf"$c\tau_a={summary['minimum_su2_ctau_m']:.3g}\,$m"
        ),
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"$E_a$ [GeV]")
    axis.set_ylabel("Probability per adaptive energy bin")
    axis.grid(True, alpha=0.25)
    axis.legend(
        title=(
            rf"$m_a={float(bank['mass_GeV']):g}\,$GeV, "
            + rf"$D_{{\mathrm{{TV}}}}={summary['minimum_D_TV']:.4f}$"
        )
    )
    style_axis(axis)
    figure.tight_layout()

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def mass_token(mass_gev: float) -> str:
    """Return the filename token used by the template-bank builder."""
    return f"{mass_gev:g}".replace("-", "m").replace(".", "p")


def ensure_output_paths(paths: list[Path], *, overwrite: bool) -> None:
    """Protect existing outputs unless replacement was requested."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Distance-map output already exists. Use --overwrite to replace it:\n"
            + formatted
        )


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 2D total-variation-distance maps from the saved "
            "lifetime-blind ALP template banks."
        )
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        default=None,
        help="Subset of masses in GeV. By default, process every bank.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Template-bank directory (default: {INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing distance tables and plots.",
    )
    return parser.parse_args()


def select_bank_paths(input_dir: Path, requested_masses: list[float] | None) -> list[Path]:
    """Find all requested template banks and reject ambiguous mass matching."""
    paths = sorted(input_dir.glob("template_bank_ma_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No template banks found in:\n  {input_dir}\n"
            "Run `python -m analysis.lifetime_blind_discrimination` first."
        )

    if requested_masses is None:
        return paths

    selected: list[Path] = []
    available: list[tuple[Path, float]] = []
    for path in paths:
        with np.load(path) as raw:
            available.append((path, float(np.asarray(raw["mass_GeV"]).item())))

    for requested in requested_masses:
        matches = [
            path
            for path, mass in available
            if np.isclose(mass, requested, rtol=0.0, atol=1.0e-12)
        ]
        if len(matches) != 1:
            available_text = ", ".join(f"{mass:g}" for _, mass in available)
            raise ValueError(
                f"Could not uniquely match m_a={requested:g} GeV. "
                f"Available masses: {available_text}"
            )
        selected.append(matches[0])

    return selected


def main() -> None:
    args = parse_arguments()

    global TABLE_DIR, PLOT_DIR
    output_dir = args.output_dir.resolve()
    TABLE_DIR = output_dir / "tables"
    PLOT_DIR = output_dir / "plots"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    bank_paths = select_bank_paths(args.input_dir.resolve(), args.masses)

    print()
    print("=" * 80)
    print("Lifetime-blind 2D total-variation-distance maps")
    print("This is pure post-processing: EventCalc is not launched.")
    print(f"Template banks: {len(bank_paths)}")
    print("=" * 80)

    summary_rows: list[dict] = []

    for bank_path in bank_paths:
        bank = load_template_bank(bank_path)
        mass_gev = float(bank["mass_GeV"])
        token = mass_token(mass_gev)
        distances = total_variation_matrix(
            np.asarray(bank["photon_probabilities"], dtype=float),
            np.asarray(bank["su2_probabilities"], dtype=float),
        )
        summary = summarize_distance_matrix(bank, distances)
        summary_rows.append(summary)

        distance_csv = TABLE_DIR / f"distance_map_ma_{token}.csv"
        minimum_pair_csv = TABLE_DIR / f"minimum_pair_spectra_ma_{token}.csv"
        heatmap_stem = PLOT_DIR / f"distance_map_ma_{token}"
        spectra_stem = PLOT_DIR / f"minimum_pair_spectra_ma_{token}"
        output_paths = [
            distance_csv,
            minimum_pair_csv,
            heatmap_stem.with_suffix(".pdf"),
            heatmap_stem.with_suffix(".png"),
            spectra_stem.with_suffix(".pdf"),
            spectra_stem.with_suffix(".png"),
        ]
        ensure_output_paths(output_paths, overwrite=args.overwrite)

        distance_table(bank, distances).to_csv(distance_csv, index=False)
        least_distinguishable_bin_table(bank, summary).to_csv(
            minimum_pair_csv,
            index=False,
        )
        heatmap_pdf, _ = plot_distance_map(
            bank,
            distances,
            summary,
            output_stem=heatmap_stem,
        )
        spectra_pdf, _ = plot_least_distinguishable_spectra(
            bank,
            summary,
            output_stem=spectra_stem,
        )

        print()
        print(f"m_a = {mass_gev:g} GeV")
        print(
            "  minimum D_TV = "
            f"{summary['minimum_D_TV']:.6f} at "
            f"c_tau(photon) = {summary['minimum_photon_ctau_m']:.6g} m, "
            f"c_tau(SU2) = {summary['minimum_su2_ctau_m']:.6g} m"
        )
        print(f"  heatmap: {heatmap_pdf}")
        print(f"  minimum-pair spectra: {spectra_pdf}")

    summary_table = pd.DataFrame(summary_rows).sort_values(
        "mass_GeV",
        ignore_index=True,
    )
    summary_path = output_dir / "distance_map_summary.csv"
    ensure_output_paths([summary_path], overwrite=args.overwrite)
    summary_table.to_csv(summary_path, index=False)

    display_columns = [
        "mass_GeV",
        "minimum_D_TV",
        "minimum_photon_ctau_m",
        "minimum_su2_ctau_m",
        "maximum_D_TV",
    ]
    print()
    print("=" * 80)
    print("Distance-map generation finished")
    print("=" * 80)
    print(summary_table[display_columns].to_string(index=False))
    print()
    print(f"Summary: {summary_path}")
    print("The next step is the independently profiled likelihood analysis.")


if __name__ == "__main__":
    main()
