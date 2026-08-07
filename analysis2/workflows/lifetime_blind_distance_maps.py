"""Compute exact frozen-reference total-variation maps from lifetime template banks."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analysis2.cache import CacheStore, file_fingerprint
from analysis2.config import AnalysisConfig, get_config
from analysis2.distance_statistics import (
    build_distance_table,
    minimum_pair_bin_table,
    summarize_distance_matrix,
    total_variation_matrix,
)
from analysis2.lifetime_blind_plotting import (
    plot_distance_map,
    plot_minimum_pair_spectra,
)
from analysis2.lifetime_template_banks import LifetimeTemplateBank, load_template_bank
from analysis2.paths import portable_path, profile_output_dir
from analysis2.workflows import (
    add_profile_cache_arguments,
    float_token,
    write_dataframe,
    write_manifest,
)
from analysis2.workflows.lifetime_blind_discrimination import (
    resolve_requested_masses,
)


WORKFLOW_NAME = "lifetime_blind_distance_maps"
DISTANCE_CACHE_VERSION = 1


@dataclass(frozen=True)
class DistanceProducts:
    distances: np.ndarray
    distance_table: pd.DataFrame
    summary: dict
    minimum_pair_table: pd.DataFrame


def parse_arguments(arguments: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    parser.add_argument("--masses", nargs="+", type=float, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def discover_template_bank_masses(input_dir: Path) -> tuple[float, ...]:
    """Return masses stored in the template banks found in ``input_dir``."""
    candidates = sorted(input_dir.glob("template_bank_ma_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No template banks found in {input_dir}.")

    masses: list[float] = []
    for path in candidates:
        mass = float(load_template_bank(path).mass_gev)
        if any(
            np.isclose(mass, other, rtol=0.0, atol=1.0e-12)
            for other in masses
        ):
            raise ValueError(
                f"Multiple template banks in {input_dir} contain "
                f"m_a={mass:g} GeV."
            )
        expected_name = f"template_bank_ma_{float_token(mass)}.npz"
        if path.name != expected_name:
            raise ValueError(
                f"Template-bank filename and stored mass disagree: {path}. "
                f"Expected filename {expected_name}."
            )
        masses.append(mass)

    return tuple(sorted(masses))


def select_bank_paths(
    input_dir: Path,
    masses: tuple[float, ...],
) -> list[Path]:
    paths = [
        input_dir / f"template_bank_ma_{float_token(mass)}.npz"
        for mass in masses
    ]
    if missing := [path for path in paths if not path.is_file()]:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError("Missing requested template banks:\n" + listing)
    return paths


def distance_products(
    bank: LifetimeTemplateBank,
    distances: np.ndarray,
) -> DistanceProducts:
    summary = summarize_distance_matrix(
        mass_gev=bank.mass_gev,
        energy_edges_gev=bank.energy_edges_gev,
        photon_ctau_m=bank.photon_ctau_m,
        photon_expected_events=bank.photon_n_events,
        su2_ctau_m=bank.su2_ctau_m,
        su2_expected_events=bank.su2_n_events,
        distances=distances,
        photon_interval_index=bank.photon_interval_index,
        su2_interval_index=bank.su2_interval_index,
    )
    table = build_distance_table(
        mass_gev=bank.mass_gev,
        photon_ctau_m=bank.photon_ctau_m,
        photon_expected_events=bank.photon_n_events,
        su2_ctau_m=bank.su2_ctau_m,
        su2_expected_events=bank.su2_n_events,
        distances=distances,
        photon_interval_index=bank.photon_interval_index,
        su2_interval_index=bank.su2_interval_index,
    )
    photon_index = int(summary["minimum_photon_lifetime_index"])
    su2_index = int(summary["minimum_su2_lifetime_index"])
    minimum = minimum_pair_bin_table(
        mass_gev=bank.mass_gev,
        energy_edges_gev=bank.energy_edges_gev,
        photon_ctau_m=bank.photon_ctau_m[photon_index],
        photon_probabilities=bank.photon_probabilities[photon_index],
        su2_ctau_m=bank.su2_ctau_m[su2_index],
        su2_probabilities=bank.su2_probabilities[su2_index],
        photon_interval_index=int(bank.photon_interval_index[photon_index]),
        su2_interval_index=int(bank.su2_interval_index[su2_index]),
    )
    return DistanceProducts(distances, table, summary, minimum)


def cached_distance_matrix(
    *,
    cache: CacheStore,
    bank_path: Path,
    bank: LifetimeTemplateBank,
    force: bool = False,
) -> np.ndarray:
    """Cache the cheap map separately from the immutable input bank."""
    identity = {
        "distance_cache_version": DISTANCE_CACHE_VERSION,
        "definition": "D_TV=0.5*sum(abs(photon-su2))",
        "profile": cache.profile,
        "template_bank": file_fingerprint(bank_path),
    }
    expected_shape = (
        len(bank.photon_ctau_m),
        len(bank.su2_ctau_m),
    )

    def validate(arrays: dict[str, np.ndarray], metadata: dict) -> None:
        distances = arrays["distances"]
        if distances.shape != expected_shape or np.any(~np.isfinite(distances)):
            raise ValueError("cached distance matrix has invalid shape or values")
        if np.any((distances < 0.0) | (distances > 1.0)):
            raise ValueError("cached total-variation distances lie outside [0, 1]")

    if not force:
        loaded = cache.load("lifetime_distance_map", identity, validate)
        if loaded:
            return loaded[0]["distances"]
    elif cache.enabled:
        _, _, key = cache.paths("lifetime_distance_map", identity)
        print(f"CACHE FORCED   [lifetime_distance_map] {key[:12]}")
    distances = total_variation_matrix(
        bank.photon_probabilities,
        bank.su2_probabilities,
    )
    cache.save(
        "lifetime_distance_map",
        identity,
        {"distances": distances},
        {"mass_gev": bank.mass_gev},
    )
    return distances


def distance_output_paths(
    output_dir: Path,
    mass_gev: float,
    *,
    include_plots: bool,
) -> dict[str, Path]:
    token = float_token(mass_gev)
    paths = {
        "distance_table": output_dir / "tables" / f"distance_map_ma_{token}.csv",
        "minimum_table": (
            output_dir / "tables" / f"minimum_pair_spectra_ma_{token}.csv"
        ),
    }
    if include_plots:
        for label, stem in (
            ("distance", output_dir / "plots" / f"distance_map_ma_{token}"),
            ("minimum", output_dir / "plots" / f"minimum_pair_spectra_ma_{token}"),
        ):
            paths[f"{label}_pdf"] = stem.with_suffix(".pdf")
            paths[f"{label}_png"] = stem.with_suffix(".png")
    return paths


def _protect_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Distance-map output already exists; use --overwrite:\n" + listing
        )


def run_distance_map_workflow(
    *,
    config: AnalysisConfig,
    cache: CacheStore,
    input_dir: Path,
    output_dir: Path,
    requested_masses: Iterable[float] | None = None,
    overwrite: bool = False,
    force: bool = False,
    make_plots: bool = True,
) -> pd.DataFrame:
    """Run cached pure post-processing and return the combined summary."""
    started = perf_counter()
    available_masses = discover_template_bank_masses(input_dir)
    masses = resolve_requested_masses(requested_masses, available_masses)
    bank_paths = select_bank_paths(input_dir, masses)
    path_sets = [
        distance_output_paths(output_dir, mass, include_plots=make_plots)
        for mass in masses
    ]
    summary_path = output_dir / "distance_map_summary.csv"
    protected = [summary_path, output_dir / "manifest.json"]
    protected.extend(path for paths in path_sets for path in paths.values())
    _protect_outputs(protected, overwrite)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    if make_plots:
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    summaries = []
    artifacts: list[Path] = []
    for mass_gev, bank_path, paths in zip(masses, bank_paths, path_sets):
        bank = load_template_bank(bank_path)
        if not np.isclose(bank.mass_gev, mass_gev, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Bank filename and stored mass disagree: {bank_path}")
        distances = cached_distance_matrix(
            cache=cache,
            bank_path=bank_path,
            bank=bank,
            force=force,
        )
        products = distance_products(bank, distances)
        write_dataframe(products.distance_table, paths["distance_table"])
        write_dataframe(products.minimum_pair_table, paths["minimum_table"])
        artifacts.extend([paths["distance_table"], paths["minimum_table"]])
        if make_plots:
            plot_distance_map(
                bank,
                distances,
                products.summary,
                output_stem=paths["distance_pdf"].with_suffix(""),
            )
            plot_minimum_pair_spectra(
                bank,
                products.summary,
                output_stem=paths["minimum_pdf"].with_suffix(""),
            )
            artifacts.extend(
                path for name, path in paths.items()
                if name.endswith("_pdf") or name.endswith("_png")
            )
        summaries.append(products.summary)

    summary = pd.DataFrame(summaries).sort_values("mass_GeV", ignore_index=True)
    write_dataframe(summary, summary_path)
    artifacts.append(summary_path)
    write_manifest(
        config,
        WORKFLOW_NAME,
        output_dir,
        elapsed_seconds=perf_counter() - started,
        cache_stats=cache.counter_snapshot(),
        artifacts=artifacts,
        extra={
            "input_template_banks": [portable_path(path) for path in bank_paths],
            "distance_definition": "D_TV=0.5*sum(abs(photon-su2))",
            "interval_aware_domains": True,
        },
    )
    return summary


def main() -> None:
    args = parse_arguments()
    config = get_config(args.profile)
    cache = CacheStore(config.name, enabled=not args.no_cache)
    input_dir = args.input_dir or (
        profile_output_dir(config.name, "lifetime_blind_discrimination")
        / "template_banks"
    )
    output_dir = args.output_dir or profile_output_dir(
        config.name, "lifetime_blind_distance_maps"
    )
    summary = run_distance_map_workflow(
        config=config,
        cache=cache,
        input_dir=input_dir,
        output_dir=output_dir,
        requested_masses=args.masses,
        overwrite=args.overwrite,
        force=args.force,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
