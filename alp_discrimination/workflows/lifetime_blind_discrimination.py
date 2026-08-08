"""Build ECAL-aware, independently lifetime-profiled Week-8 template banks."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from alp_discrimination.cache import CacheStore, file_fingerprint
from alp_discrimination.config import AnalysisConfig, get_config
from alp_discrimination.eventcalc_proposals import EVENTCALC_FULL_SUPPORT_CTAU_M
from alp_discrimination.lifetime_template_banks import (
    LifetimeTemplateBank,
    build_lifetime_template_bank,
    load_template_bank,
    save_bank_artifacts,
)
from alp_discrimination.models import MODELS
from alp_discrimination.paths import portable_path, profile_output_dir
from alp_discrimination.lifetime_domains import (
    available_lifetime_domain_masses,
    build_lifetime_grid,
    load_allowed_ctau_domains,
)
from alp_discrimination.workflows import (
    add_profile_cache_arguments,
    float_token,
    write_dataframe,
    write_manifest,
)

if TYPE_CHECKING:
    from alp_discrimination.eventcalc_adapter import EventCalcAdapter


WORKFLOW_NAME = "lifetime_blind_discrimination"
WEEK8_DOMAIN_EVENT_LEVEL = 2.3


def _minimum_photon_energy_gev(
    adapter: object,
    selection_name: str,
) -> float | None:
    # Real EventCalc adapters expose ``selection``. Lightweight workflow
    # test doubles historically expose only ``config`` and ``evaluate_model``.
    # The frozen Week-8 selection name is therefore the authoritative fallback.
    selection = getattr(adapter, "selection", None)
    threshold = getattr(selection, "minimum_photon_energy_gev", None)
    if threshold is not None:
        return float(threshold)
    if selection_name == "diphoton_ecal_e1gev":
        return 1.0
    return None


def proposal_lifetime_for_target(ctau_m: float) -> float:
    """Choose the efficient EventCalc proposal for one target lifetime.

    Short-lived templates retain EventCalc's lifetime-dependent lower-energy
    sampling bound and therefore need an exact-lifetime proposal.  At longer
    lifetimes the bound is E_a >= m_a, so one common full-support proposal can
    be reused.
    """
    ctau_m = float(ctau_m)
    if not np.isfinite(ctau_m) or ctau_m <= 0.0:
        raise ValueError("ctau_m must be finite and positive")
    return (
        ctau_m
        if ctau_m < EVENTCALC_FULL_SUPPORT_CTAU_M
        else EVENTCALC_FULL_SUPPORT_CTAU_M
    )


def parse_arguments(arguments: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    parser.add_argument("--masses", nargs="+", type=float, default=None)
    parser.add_argument("--domain-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--selection-name",
        choices=("diphoton_ecal", "diphoton_ecal_e1gev"),
        default=None,
        help=(
            "Detector selection used for accepted spectra. The Week-8 lifetime "
            "domain remains the geometry-only N_events>=2.3 domain."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--lifetime-points-per-interval",
        type=int,
        default=None,
        help="Number of log-spaced templates in each connected lifetime interval.",
    )
    parser.add_argument(
        "--initial-energy-bins",
        type=int,
        default=None,
        help="Number of initial logarithmic ALP-energy bins before adaptive merging.",
    )
    parser.add_argument(
        "--minimum-bin-n-eff",
        type=float,
        default=None,
        help="Minimum effective sample size required in every final adaptive bin.",
    )
    parser.add_argument(
        "--energy-edges-from-bank",
        type=Path,
        default=None,
        help=(
            "Use the exact saved energy edges from one template-bank NPZ. "
            "This requires a single requested mass and never merges the edges."
        ),
    )
    parser.add_argument(
        "--lifetime-grid-path",
        type=Path,
        default=None,
        help=(
            "CSV with explicit model, mass_GeV, interval_index and ctau_m rows. "
            "Every connected interval must retain both endpoints."
        ),
    )
    return parser.parse_args(arguments)


def apply_cli_overrides(config: AnalysisConfig, args) -> AnalysisConfig:
    """Return a profile with explicitly requested template settings replaced."""
    updates: dict[str, int | float] = {}

    if args.lifetime_points_per_interval is not None:
        if args.lifetime_points_per_interval < 2:
            raise ValueError("--lifetime-points-per-interval must be at least 2")
        updates["lifetime_points_per_model"] = args.lifetime_points_per_interval

    if args.initial_energy_bins is not None:
        if args.initial_energy_bins < 1:
            raise ValueError("--initial-energy-bins must be positive")
        updates["initial_energy_bins"] = args.initial_energy_bins

    if args.minimum_bin_n_eff is not None:
        value = float(args.minimum_bin_n_eff)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("--minimum-bin-n-eff must be finite and positive")
        updates["minimum_bin_n_eff"] = value

    return replace(
        config,
        selection_name=(
            config.selection_name
            if args.selection_name is None
            else args.selection_name
        ),
        templates=replace(config.templates, **updates),
    )


def _template_overrides_requested(args) -> bool:
    return any(
        value is not None
        for value in (
            args.selection_name,
            args.lifetime_points_per_interval,
            args.initial_energy_bins,
            args.minimum_bin_n_eff,
            args.energy_edges_from_bank,
            args.lifetime_grid_path,
        )
    )


def resolve_template_output_dir(config: AnalysisConfig, args) -> Path:
    """Protect the default validated output from convergence overrides."""
    if args.output_dir is not None:
        return args.output_dir
    if _template_overrides_requested(args):
        raise ValueError(
            "Output-affecting overrides require an explicit --output-dir so "
            "validated template banks cannot be overwritten accidentally."
        )
    return profile_output_dir(
        config.name,
        "lifetime_blind_discrimination_week8",
    )


def resolve_requested_masses(
    requested: Iterable[float] | None,
    available: Iterable[float],
) -> tuple[float, ...]:
    """Resolve a requested subset while retaining the CSV mass order."""
    configured = tuple(sorted(set(float(value) for value in available)))
    if not configured:
        raise ValueError("No masses are available in the Week-8 domain table.")
    if requested is None:
        return configured
    selected: set[float] = set()
    for value in requested:
        matches = [
            mass
            for mass in configured
            if np.isclose(value, mass, rtol=0.0, atol=1.0e-12)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Requested mass {value:g} GeV is not in the Week-8 masses "
                f"{configured}."
            )
        selected.add(matches[0])
    return tuple(mass for mass in configured if mass in selected)


def _mass_seed_indices(
    config: AnalysisConfig,
    available_masses: Iterable[float],
) -> dict[float, int]:
    """Preserve frozen seed indices and append new Week-8 masses deterministically."""
    frozen = tuple(float(value) for value in config.seed_policy.mass_order_gev)
    extra = [
        mass
        for mass in sorted(set(float(value) for value in available_masses))
        if not any(np.isclose(mass, old, rtol=0.0, atol=1.0e-12) for old in frozen)
    ]
    result = {mass: index for index, mass in enumerate(frozen)}
    result.update(
        {
            mass: len(frozen) + index
            for index, mass in enumerate(extra)
        }
    )
    return result


def _model_domain_rows(
    domains: pd.DataFrame,
    *,
    model_label: str,
    mass_gev: float,
) -> pd.DataFrame:
    selected = domains[
        (domains["model"] == model_label)
        & np.isclose(
            domains["mass_GeV"].to_numpy(dtype=float),
            float(mass_gev),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].sort_values("ctau_min_m", ignore_index=True)
    if selected.empty:
        raise ValueError(
            f"No allowed Week-8 interval for {model_label}, "
            f"m_a={mass_gev:g} GeV."
        )
    return selected


def build_template_lifetime_grid_table(
    domains: pd.DataFrame,
    masses: Iterable[float],
    points_per_interval: int,
) -> pd.DataFrame:
    """Tabulate every lifetime template and its connected allowed interval."""
    frames = []
    for mass_gev in masses:
        for model in MODELS:
            grid = build_lifetime_grid(
                domains,
                model=model.legacy_name,
                mass_gev=mass_gev,
                points_per_interval=points_per_interval,
            ).copy()
            grid.insert(1, "model_id", model.identifier)
            frames.append(grid)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["mass_GeV", "model", "ctau_m"],
        ignore_index=True,
    )


_CUSTOM_LIFETIME_GRID_REQUIRED_COLUMNS = {
    "model",
    "mass_GeV",
    "interval_index",
    "ctau_m",
}


def load_custom_lifetime_grid(
    path: Path,
    *,
    domains: pd.DataFrame,
    masses: Iterable[float],
) -> pd.DataFrame:
    """Load an explicit interval-aware lifetime grid with full-domain coverage.

    Every requested model-mass-interval must contain at least two distinct
    lifetimes and retain both saved domain endpoints.  Points may be added only
    inside their declared connected interval; excluded gaps are never bridged.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Custom lifetime-grid CSV not found: {path}")

    data = pd.read_csv(path)
    missing = _CUSTOM_LIFETIME_GRID_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"{path} is missing custom lifetime-grid columns: {sorted(missing)}"
        )
    data = data.loc[:, sorted(_CUSTOM_LIFETIME_GRID_REQUIRED_COLUMNS)].copy()
    for column in ("mass_GeV", "interval_index", "ctau_m"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    numeric = data[["mass_GeV", "interval_index", "ctau_m"]].to_numpy(float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("Custom lifetime-grid values must be finite")
    if np.any(data["mass_GeV"].to_numpy(float) <= 0.0) or np.any(
        data["ctau_m"].to_numpy(float) <= 0.0
    ):
        raise ValueError("Custom lifetime-grid masses and lifetimes must be positive")

    raw_interval = data["interval_index"].to_numpy(float)
    interval_index = np.rint(raw_interval).astype(int)
    if not np.allclose(raw_interval, interval_index, rtol=0.0, atol=1.0e-12) or np.any(
        interval_index < 0
    ):
        raise ValueError("Custom lifetime-grid interval_index values must be non-negative integers")
    data["interval_index"] = interval_index

    model_id_by_legacy_name = {
        model.legacy_name: model.identifier
        for model in MODELS
    }
    unknown = sorted(set(data["model"].astype(str)) - set(model_id_by_legacy_name))
    if unknown:
        raise ValueError(
            "Unknown models in custom lifetime grid: " + ", ".join(unknown)
        )
    data["model"] = data["model"].astype(str)

    requested_masses = tuple(float(value) for value in masses)
    selected = data[
        data["mass_GeV"].apply(
            lambda value: any(
                np.isclose(float(value), mass, rtol=0.0, atol=1.0e-12)
                for mass in requested_masses
            )
        )
    ].copy()
    if selected.empty:
        raise ValueError("Custom lifetime grid contains none of the requested masses")

    duplicate = selected.duplicated(
        ["model", "mass_GeV", "interval_index", "ctau_m"],
        keep=False,
    )
    if duplicate.any():
        raise ValueError("Custom lifetime grid contains duplicate lifetime rows")

    frames: list[pd.DataFrame] = []
    tolerance = 1.0e-12
    for mass_gev in requested_masses:
        for model in MODELS:
            domain_rows = _model_domain_rows(
                domains,
                model_label=model.legacy_name,
                mass_gev=mass_gev,
            ).sort_values("interval_index")
            model_rows = selected[
                (selected["model"] == model.legacy_name)
                & np.isclose(
                    selected["mass_GeV"].to_numpy(float),
                    mass_gev,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ].copy()
            interval_frames: list[pd.DataFrame] = []
            for domain_row in domain_rows.itertuples(index=False):
                index = int(domain_row.interval_index)
                lower = float(domain_row.ctau_min_m)
                upper = float(domain_row.ctau_max_m)
                interval_rows = model_rows[
                    model_rows["interval_index"] == index
                ].sort_values("ctau_m", ignore_index=True)
                if len(interval_rows) < 2:
                    raise ValueError(
                        f"Custom grid requires at least two points for "
                        f"{model.legacy_name}, m_a={mass_gev:g}, interval {index}."
                    )
                lifetimes = interval_rows["ctau_m"].to_numpy(float)
                if np.any(np.diff(lifetimes) <= 0.0):
                    raise ValueError("Custom lifetime-grid points must be strictly increasing")
                if np.any(lifetimes < lower * (1.0 - tolerance)) or np.any(
                    lifetimes > upper * (1.0 + tolerance)
                ):
                    raise ValueError(
                        f"Custom lifetime lies outside {model.legacy_name}, "
                        f"m_a={mass_gev:g}, interval {index}."
                    )
                if not np.isclose(lifetimes[0], lower, rtol=tolerance, atol=0.0) or not np.isclose(
                    lifetimes[-1], upper, rtol=tolerance, atol=0.0
                ):
                    raise ValueError(
                        f"Custom grid must retain both endpoints for "
                        f"{model.legacy_name}, m_a={mass_gev:g}, interval {index}."
                    )
                interval_rows = interval_rows.copy()
                interval_rows["lifetime_index_within_interval"] = np.arange(
                    len(interval_rows), dtype=int
                )
                interval_rows["is_interval_endpoint"] = False
                interval_rows.loc[
                    interval_rows.index[[0, len(interval_rows) - 1]],
                    "is_interval_endpoint",
                ] = True
                interval_frames.append(interval_rows)

            known_indices = set(domain_rows["interval_index"].astype(int))
            extra_indices = set(model_rows["interval_index"].astype(int)) - known_indices
            if extra_indices:
                raise ValueError(
                    f"Custom grid refers to unknown interval indices {sorted(extra_indices)} "
                    f"for {model.legacy_name}, m_a={mass_gev:g}."
                )
            combined = pd.concat(interval_frames, ignore_index=True).sort_values(
                ["ctau_m", "interval_index"],
                ignore_index=True,
            )
            combined.insert(
                3,
                "global_lifetime_index",
                np.arange(len(combined), dtype=int),
            )
            combined.insert(1, "model_id", model.identifier)
            frames.append(combined)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["mass_GeV", "model", "ctau_m"],
        ignore_index=True,
    )


def _validate_template_spectrum(model_id: str, mass_gev: float, spectrum) -> None:
    """Reject only numerically unusable selected spectra, never low event rates."""
    rate = float(spectrum.expected_events)
    if not np.isfinite(rate) or rate <= 0.0:
        raise RuntimeError(
            f"No usable selected spectrum for {model_id}, m_a={mass_gev:g} GeV, "
            f"c*tau={spectrum.ctau_m:.6g} m: N_events={rate:.6g}."
        )
    energies = np.asarray(spectrum.energies_gev, dtype=float)
    weights = np.asarray(spectrum.absolute_event_weights, dtype=float)
    if (
        energies.ndim != 1
        or weights.shape != energies.shape
        or len(energies) == 0
        or np.any(~np.isfinite(energies))
        or np.any(~np.isfinite(weights))
        or np.sum(weights) <= 0.0
    ):
        raise RuntimeError(
            f"Numerically unusable selected spectrum for {model_id}, "
            f"m_a={mass_gev:g} GeV, c*tau={spectrum.ctau_m:.6g} m."
        )


def build_mass_bank(
    *,
    config: AnalysisConfig,
    adapter: "EventCalcAdapter",
    mass_gev: float,
    domains: pd.DataFrame,
    mass_seed_index: int,
    lifetime_grid_table: pd.DataFrame | None = None,
    fixed_energy_edges_gev: np.ndarray | None = None,
) -> LifetimeTemplateBank:
    """Generate selected spectra for all connected Week-8 lifetime intervals."""
    spectra: dict[str, dict[float, object]] = {}
    lifetime_grids: dict[str, pd.DataFrame] = {}
    allowed_intervals_m: dict[str, np.ndarray] = {}

    for model in MODELS:
        domain_rows = _model_domain_rows(
            domains,
            model_label=model.legacy_name,
            mass_gev=mass_gev,
        )
        if lifetime_grid_table is None:
            grid = build_lifetime_grid(
                domains,
                model=model.legacy_name,
                mass_gev=mass_gev,
                points_per_interval=config.templates.lifetime_points_per_model,
            )
        else:
            grid = lifetime_grid_table[
                (lifetime_grid_table["model_id"] == model.identifier)
                & np.isclose(
                    lifetime_grid_table["mass_GeV"].to_numpy(float),
                    mass_gev,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ].copy()
            if grid.empty:
                raise ValueError(
                    f"Lifetime grid has no rows for {model.identifier}, "
                    f"m_a={mass_gev:g} GeV."
                )
        lifetimes = grid["ctau_m"].to_numpy(dtype=float)
        lifetime_grids[model.identifier] = grid
        allowed_intervals_m[model.identifier] = domain_rows.sort_values(
            "interval_index"
        )[["ctau_min_m", "ctau_max_m"]].to_numpy(dtype=float)

        model_seed = config.seed_policy.model_seed_from_indices(
            mass_seed_index,
            config.seed_policy.model_index(model.identifier),
            seed_offset=config.templates.seed_offset,
        )
        spectra[model.identifier] = {}
        for ctau_m in lifetimes:
            proposal_ctau_m = proposal_lifetime_for_target(float(ctau_m))
            spectrum = adapter.evaluate_model(
                model.identifier,
                mass_gev,
                float(ctau_m),
                model_seed,
                "spectrum",
                proposal_ctau_m=proposal_ctau_m,
            )
            _validate_template_spectrum(model.identifier, mass_gev, spectrum)
            spectra[model.identifier][float(ctau_m)] = spectrum

    initial_edges = (
        np.asarray(fixed_energy_edges_gev, dtype=float)
        if fixed_energy_edges_gev is not None
        else np.geomspace(
            mass_gev,
            config.templates.energy_max_gev,
            config.templates.initial_energy_bins + 1,
        )
    )
    return build_lifetime_template_bank(
        mass_gev=mass_gev,
        spectra=spectra,
        lifetime_grids=lifetime_grids,
        allowed_intervals_m=allowed_intervals_m,
        initial_energy_edges_gev=initial_edges,
        minimum_bin_n_eff=config.templates.minimum_bin_n_eff,
        fixed_energy_edges_gev=fixed_energy_edges_gev,
        jeffreys_alpha=config.templates.jeffreys_alpha,
        event_threshold=WEEK8_DOMAIN_EVENT_LEVEL,
        template_base_seed=(
            config.seed_policy.base_seed + config.templates.seed_offset
        ),
        template_seed_offset=config.templates.seed_offset,
        profile=config.name,
        selection_name=config.selection_name,
        minimum_photon_energy_gev=_minimum_photon_energy_gev(
            adapter,
            config.selection_name,
        ),
    )


def bank_paths(output_dir: Path, mass_gev: float) -> tuple[Path, Path, Path]:
    token = float_token(mass_gev)
    return (
        output_dir / "template_banks" / f"template_bank_ma_{token}.npz",
        output_dir / "tables" / f"template_summary_ma_{token}.csv",
        output_dir / "tables" / f"probability_templates_ma_{token}.csv",
    )


def _protect_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Template-bank output already exists; use --overwrite:\n" + listing
        )


def run_template_bank_workflow(
    *,
    config: AnalysisConfig,
    adapter: "EventCalcAdapter",
    domain_path: Path,
    output_dir: Path,
    requested_masses: Iterable[float] | None = None,
    overwrite: bool = False,
    energy_edges_from_bank: Path | None = None,
    lifetime_grid_path: Path | None = None,
) -> pd.DataFrame:
    """Run the Week-8 template-bank workflow and return its mass summary."""
    started = perf_counter()
    domains = load_allowed_ctau_domains(
        domain_path,
        expected_event_level=WEEK8_DOMAIN_EVENT_LEVEL,
    )
    available_masses = tuple(available_lifetime_domain_masses(domains))
    masses = resolve_requested_masses(requested_masses, available_masses)
    seed_indices = _mass_seed_indices(config, available_masses)

    fixed_energy_edges_by_mass: dict[float, np.ndarray] = {}
    fixed_edge_source_bank: LifetimeTemplateBank | None = None
    if energy_edges_from_bank is not None:
        if len(masses) != 1:
            raise ValueError(
                "--energy-edges-from-bank requires exactly one requested mass"
            )
        fixed_edge_source_bank = load_template_bank(energy_edges_from_bank)
        mass_gev = masses[0]
        if not np.isclose(
            fixed_edge_source_bank.mass_gev,
            mass_gev,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Fixed-edge source-bank mass does not match the requested mass"
            )
        if fixed_edge_source_bank.selection_name != config.selection_name:
            raise ValueError(
                "Fixed-edge source-bank selection does not match the active selection"
            )
        fixed_energy_edges_by_mass[mass_gev] = (
            fixed_edge_source_bank.energy_edges_gev.copy()
        )

    domain_copy_path = output_dir / "tables" / "week8_allowed_ctau_domains.csv"
    lifetime_grid_output_path = (
        output_dir / "tables" / "week8_template_lifetime_grid.csv"
    )
    bank_manifest_path = output_dir / "template_bank_manifest.csv"
    artifact_sets = [bank_paths(output_dir, mass) for mass in masses]
    protected = [
        domain_copy_path,
        lifetime_grid_output_path,
        bank_manifest_path,
        output_dir / "manifest.json",
    ]
    protected.extend(path for paths in artifact_sets for path in paths)
    _protect_outputs(protected, overwrite)

    selected_domains = domains[
        domains["mass_GeV"].apply(
            lambda value: any(
                np.isclose(value, mass, rtol=0.0, atol=1.0e-12)
                for mass in masses
            )
        )
    ].copy()
    write_dataframe(selected_domains, domain_copy_path)
    lifetime_grid = (
        build_template_lifetime_grid_table(
            domains,
            masses,
            config.templates.lifetime_points_per_model,
        )
        if lifetime_grid_path is None
        else load_custom_lifetime_grid(
            lifetime_grid_path,
            domains=domains,
            masses=masses,
        )
    )
    write_dataframe(lifetime_grid, lifetime_grid_output_path)

    artifacts = [domain_copy_path, lifetime_grid_output_path]
    selection_minimum_photon_energy_gev = _minimum_photon_energy_gev(
        adapter,
        config.selection_name,
    )

    summary_rows = []
    for mass_gev, paths in zip(masses, artifact_sets):
        bank = build_mass_bank(
            config=config,
            adapter=adapter,
            mass_gev=mass_gev,
            domains=domains,
            mass_seed_index=seed_indices[mass_gev],
            lifetime_grid_table=lifetime_grid,
            fixed_energy_edges_gev=fixed_energy_edges_by_mass.get(mass_gev),
        )
        save_bank_artifacts(
            bank,
            bank_path=paths[0],
            summary_path=paths[1],
            probability_path=paths[2],
        )
        artifacts.extend(paths)
        summary_rows.append(
            {
                "mass_GeV": mass_gev,
                "template_bank_path": portable_path(paths[0]),
                "template_summary_path": portable_path(paths[1]),
                "probability_table_path": portable_path(paths[2]),
                "number_of_energy_bins": bank.number_of_energy_bins,
                "number_of_photon_lifetimes": len(bank.photon_ctau_m),
                "number_of_photon_intervals": len(bank.photon_allowed_intervals_m),
                "number_of_su2_lifetimes": len(bank.su2_ctau_m),
                "number_of_su2_intervals": len(bank.su2_allowed_intervals_m),
                "minimum_bin_N_eff": bank.minimum_bin_n_eff,
                "jeffreys_alpha": bank.jeffreys_alpha,
                "selection_name": bank.selection_name,
                "minimum_photon_energy_GeV": (
                    bank.minimum_photon_energy_gev
                ),
                "mass_seed_index": seed_indices[mass_gev],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mass_GeV", ignore_index=True)
    write_dataframe(summary, bank_manifest_path)
    artifacts.append(bank_manifest_path)
    elapsed = perf_counter() - started
    write_manifest(
        config,
        WORKFLOW_NAME,
        output_dir,
        elapsed_seconds=elapsed,
        cache_stats=adapter.cache.counter_snapshot(),
        artifacts=artifacts,
        extra={
            "domain_path": portable_path(domain_path),
            "domain_definition": (
                "geom-only N_events>=2.3 sensitivity minus existing exclusions"
            ),
            "domain_event_level": WEEK8_DOMAIN_EVENT_LEVEL,
            "proposal_strategy": (
                "exact-lifetime adaptive-Emin below the EventCalc full-support "
                "threshold; one shared full-Ea>=mass proposal above it"
            ),
            "eventcalc_full_support_ctau_m": EVENTCALC_FULL_SUPPORT_CTAU_M,
            "requested_masses_GeV": list(masses),
            "lifetime_grid_mode": (
                "generated_log_grid"
                if lifetime_grid_path is None
                else "custom_csv"
            ),
            "lifetime_points_per_connected_interval": (
                config.templates.lifetime_points_per_model
                if lifetime_grid_path is None
                else None
            ),
            "configured_lifetime_points_per_connected_interval": (
                config.templates.lifetime_points_per_model
            ),
            "custom_lifetime_grid_path": (
                None
                if lifetime_grid_path is None
                else portable_path(lifetime_grid_path)
            ),
            "custom_lifetime_grid_fingerprint": (
                None
                if lifetime_grid_path is None
                else file_fingerprint(lifetime_grid_path)
            ),
            "energy_binning_mode": (
                "adaptive_common"
                if energy_edges_from_bank is None
                else "fixed_from_bank"
            ),
            "fixed_energy_edges_source_bank": (
                None
                if energy_edges_from_bank is None
                else portable_path(energy_edges_from_bank)
            ),
            "fixed_energy_edges_source_fingerprint": (
                None
                if energy_edges_from_bank is None
                else file_fingerprint(energy_edges_from_bank)
            ),
            "number_of_fixed_energy_bins": (
                None
                if fixed_edge_source_bank is None
                else fixed_edge_source_bank.number_of_energy_bins
            ),
            "initial_energy_bins": (
                config.templates.initial_energy_bins
                if energy_edges_from_bank is None
                else None
            ),
            "configured_initial_energy_bins": (
                config.templates.initial_energy_bins
            ),
            "minimum_bin_N_eff": config.templates.minimum_bin_n_eff,
            "selection_name": config.selection_name,
            "minimum_photon_energy_GeV": (
                selection_minimum_photon_energy_gev
            ),
            "photon_energy_threshold_inclusive": (
                selection_minimum_photon_energy_gev is not None
            ),
            "photon_separation_cut_applied": False,
            "lifetime_domain_selection_name": "diphoton_ecal",
            "allowed_lifetime_domains_recomputed_for_selection": False,
            "post_ECAL_event_rate_is_diagnostic_only": True,
            "old_N_events_ge_10_cut_applied": False,
            "old_mass_scaled_ctau_lower_cut_applied": False,
        },
    )
    return summary


def main() -> None:
    args = parse_arguments()
    config = apply_cli_overrides(get_config(args.profile), args)
    cache = CacheStore(config.name, enabled=not args.no_cache)
    from alp_discrimination.eventcalc_adapter import EventCalcAdapter

    adapter = EventCalcAdapter(config, cache=cache, force=args.force)
    domain_path = args.domain_path or (
        profile_output_dir(config.name, "week8_domains")
        / "allowed_ctau_domains.csv"
    )
    output_dir = resolve_template_output_dir(config, args)
    summary = run_template_bank_workflow(
        config=config,
        adapter=adapter,
        domain_path=domain_path,
        output_dir=output_dir,
        requested_masses=args.masses,
        overwrite=args.overwrite,
        energy_edges_from_bank=args.energy_edges_from_bank,
        lifetime_grid_path=args.lifetime_grid_path,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
