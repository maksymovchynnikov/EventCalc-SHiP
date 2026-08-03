"""Build ECAL-aware, independently lifetime-profiled template banks."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from analysis2.config import AnalysisConfig
from analysis2.lifetime_template_banks import (
    LifetimeTemplateBank,
    build_lifetime_template_bank,
    save_bank_artifacts,
)
from analysis2.models import MODELS
from analysis2.observable_domains import (
    ObservableLifetimeDomain,
    collect_observable_domains,
    domain_table,
    load_lifetime_scan,
    padded_lifetime_grid,
)
from analysis2.paths import portable_path, profile_output_dir
from analysis2.workflows import (
    add_profile_cache_arguments,
    config_and_adapter,
    float_token,
    write_dataframe,
    write_manifest,
)

if TYPE_CHECKING:
    from analysis2.eventcalc_adapter import EventCalcAdapter


WORKFLOW_NAME = "lifetime_blind_discrimination"
EVENT_RATE_RELATIVE_TOLERANCE = 0.05


def parse_arguments(arguments: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    parser.add_argument("--masses", nargs="+", type=float, default=None)
    parser.add_argument("--scan-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def resolve_requested_masses(
    requested: Iterable[float] | None,
    configured: tuple[float, ...],
) -> tuple[float, ...]:
    """Resolve a subset while retaining the immutable configured mass order."""
    if requested is None:
        return configured
    selected: set[float] = set()
    for value in requested:
        matches = [
            mass for mass in configured
            if np.isclose(value, mass, rtol=0.0, atol=1.0e-12)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Requested mass {value:g} GeV is not in profile masses {configured}."
            )
        selected.add(matches[0])
    return tuple(mass for mass in configured if mass in selected)


def _validate_scan_identity(scan: pd.DataFrame, config: AnalysisConfig) -> None:
    for column, expected in (
        ("profile", config.name),
        ("selection_name", config.selection_name),
    ):
        if column not in scan:
            continue
        values = set(scan[column].dropna().astype(str))
        if values != {expected}:
            raise ValueError(
                f"Lifetime scan {column} values {sorted(values)} do not match {expected!r}."
            )


def collect_profile_domains(
    scan: pd.DataFrame,
    config: AnalysisConfig,
    masses: tuple[float, ...],
) -> dict[tuple[str, float], ObservableLifetimeDomain]:
    """Normalize legacy scan labels to stable model identifiers."""
    _validate_scan_identity(scan, config)
    raw = collect_observable_domains(
        scan,
        threshold=config.lifetimes.event_threshold,
        allow_truncated=False,
    )
    normalized: dict[tuple[str, float], ObservableLifetimeDomain] = {}
    for mass_gev in masses:
        for model in MODELS:
            matches = [
                domain
                for (label, available_mass), domain in raw.items()
                if label in {model.identifier, model.legacy_name}
                and np.isclose(available_mass, mass_gev, rtol=0.0, atol=1.0e-12)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one observable domain for {model.identifier}, "
                    f"m_a={mass_gev:g} GeV; found {len(matches)}."
                )
            normalized[(model.identifier, mass_gev)] = matches[0]
    return normalized


def profile_domain_table(
    domains: Mapping[tuple[str, float], ObservableLifetimeDomain],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Record raw, diagnostic, and actual padded template-grid endpoints."""
    legacy_domains = {
        (model.legacy_name, mass): domains[(model.identifier, mass)]
        for mass in config.masses_gev
        if (MODELS[0].identifier, mass) in domains
        for model in MODELS
    }
    table = domain_table(
        legacy_domains,
        log_padding_fraction=config.templates.log_endpoint_padding_fraction,
    )
    grid_lower, grid_upper = [], []
    for row in table.itertuples(index=False):
        domain = legacy_domains[(row.model, float(row.mass_GeV))]
        grid = padded_lifetime_grid(
            domain,
            config.templates.lifetime_points_per_model,
            config.templates.log_endpoint_padding_fraction,
        )
        grid_lower.append(float(grid[0]))
        grid_upper.append(float(grid[-1]))
    table["template_grid_lower_m"] = grid_lower
    table["template_grid_upper_m"] = grid_upper
    return table


def _validate_template_rate(
    model_id: str,
    mass_gev: float,
    spectrum,
    event_threshold: float,
) -> None:
    rate = spectrum.expected_events
    minimum_rate = event_threshold * (1.0 - EVENT_RATE_RELATIVE_TOLERANCE)
    if not np.isfinite(rate) or rate < minimum_rate:
        raise RuntimeError(
            f"Template lies outside the intended observable domain: {model_id}, "
            f"m_a={mass_gev:g} GeV, c*tau={spectrum.ctau_m:.6g} m gives "
            f"N_events={rate:.6g}."
        )
    if rate < event_threshold:
        print(
            "WARNING: endpoint-level Monte Carlo difference: "
            f"{model_id}, m_a={mass_gev:g} GeV, "
            f"c*tau={spectrum.ctau_m:.6g} m, N_events={rate:.6g}."
        )


def build_mass_bank(
    *,
    config: AnalysisConfig,
    adapter: "EventCalcAdapter",
    mass_gev: float,
    domains: Mapping[tuple[str, float], ObservableLifetimeDomain],
) -> LifetimeTemplateBank:
    """Generate selected spectra, then construct one exact common-binning bank."""
    spectra = {}
    model_domains = {}
    for model in MODELS:
        domain = domains[(model.identifier, mass_gev)]
        model_domains[model.identifier] = domain
        lifetimes = padded_lifetime_grid(
            domain,
            config.templates.lifetime_points_per_model,
            config.templates.log_endpoint_padding_fraction,
        )
        model_seed = config.seed_policy.model_seed(
            mass_gev,
            model.identifier,
            seed_offset=config.templates.seed_offset,
        )
        proposal_ctau_m = float(lifetimes[0])
        spectra[model.identifier] = {}
        for ctau_m in lifetimes:
            spectrum = adapter.evaluate_model(
                model.identifier,
                mass_gev,
                float(ctau_m),
                model_seed,
                "spectrum",
                proposal_ctau_m=proposal_ctau_m,
            )
            _validate_template_rate(
                model.identifier,
                mass_gev,
                spectrum,
                config.lifetimes.event_threshold,
            )
            spectra[model.identifier][float(ctau_m)] = spectrum

    initial_edges = np.geomspace(
        mass_gev,
        config.templates.energy_max_gev,
        config.templates.initial_energy_bins + 1,
    )
    return build_lifetime_template_bank(
        mass_gev=mass_gev,
        spectra=spectra,
        domains=model_domains,
        initial_energy_edges_gev=initial_edges,
        minimum_bin_n_eff=config.templates.minimum_bin_n_eff,
        jeffreys_alpha=config.templates.jeffreys_alpha,
        event_threshold=config.lifetimes.event_threshold,
        template_base_seed=(
            config.seed_policy.base_seed + config.templates.seed_offset
        ),
        template_seed_offset=config.templates.seed_offset,
        profile=config.name,
        selection_name=config.selection_name,
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
    scan_path: Path,
    output_dir: Path,
    requested_masses: Iterable[float] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run the reusable template-bank workflow and return its mass summary."""
    started = perf_counter()
    masses = resolve_requested_masses(requested_masses, config.masses_gev)
    scan = load_lifetime_scan(scan_path)
    domains = collect_profile_domains(scan, config, masses)
    domain_path = output_dir / "tables" / "observable_lifetime_domains.csv"
    bank_manifest_path = output_dir / "template_bank_manifest.csv"
    artifact_sets = [bank_paths(output_dir, mass) for mass in masses]
    protected = [domain_path, bank_manifest_path, output_dir / "manifest.json"]
    protected.extend(path for paths in artifact_sets for path in paths)
    _protect_outputs(protected, overwrite)

    domain_frame = profile_domain_table(domains, config)
    write_dataframe(domain_frame, domain_path)
    artifacts = [domain_path]
    summary_rows = []
    for mass_gev, paths in zip(masses, artifact_sets):
        bank = build_mass_bank(
            config=config,
            adapter=adapter,
            mass_gev=mass_gev,
            domains=domains,
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
                "number_of_su2_lifetimes": len(bank.su2_ctau_m),
                "minimum_bin_N_eff": bank.minimum_bin_n_eff,
                "jeffreys_alpha": bank.jeffreys_alpha,
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
            "scan_path": portable_path(scan_path),
            "production_endpoint_convention": (
                config.templates.observable_endpoint_convention
            ),
            "diagnostic_endpoint_convention": (
                config.lifetimes.diagnostic_endpoint_convention
            ),
            "event_rate_relative_tolerance": EVENT_RATE_RELATIVE_TOLERANCE,
        },
    )
    return summary


def main() -> None:
    args = parse_arguments()
    config, adapter = config_and_adapter(args)
    scan_path = args.scan_path or (
        profile_output_dir(config.name, "scan_ctau_ranges") / "ctau_scan.csv"
    )
    output_dir = args.output_dir or profile_output_dir(
        config.name, "lifetime_blind_discrimination"
    )
    summary = run_template_bank_workflow(
        config=config,
        adapter=adapter,
        scan_path=scan_path,
        output_dir=output_dir,
        requested_masses=args.masses,
        overwrite=args.overwrite,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
