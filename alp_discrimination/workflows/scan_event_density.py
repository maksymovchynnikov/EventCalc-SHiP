"""Resumable mother-level event-density scan and coupling contour construction."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alp_discrimination.cache import cache_key, canonical_json
from alp_discrimination.config import event_density_seed
from alp_discrimination.physics.event_density import (
    SOURCE_SCANS, add_interpolated_closing_points, build_boundary_table,
    combine_photon_sources, endpoint_refinement_masses, stable_float_key,
)
from alp_discrimination.physics.models import get_model
from alp_discrimination.paths import profile_output_dir
from alp_discrimination.plotting.common import plot_event_rate_curves
from alp_discrimination.workflows import (
    add_profile_cache_arguments, config_and_adapter, write_dataframe, write_manifest,
)


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    return parser.parse_args()


def scan_configuration_key(config) -> str:
    """Identify rows that were produced with the current numerical configuration."""
    return cache_key({"workflow": "scan_event_density", "configuration": asdict(config)})


def _legacy_final_is_compatible(output_dir: Path, config) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("workflow") == "scan_event_density"
        and manifest.get("profile") == config.name
        and manifest.get("selection_name") == config.selection_name
        and canonical_json(manifest.get("configuration")) == canonical_json(asdict(config))
    )


def _complete_source_groups(data: pd.DataFrame, config) -> pd.DataFrame:
    required = {"profile", "selection_name", "model", "mass_GeV", "coupling_GeV_inv"}
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)
    accepted = []
    for definition in SOURCE_SCANS:
        expected_couplings = np.geomspace(
            definition.coupling_min_gev_inv, definition.coupling_max_gev_inv,
            config.event_density.coupling_points,
        )
        candidates = data[
            (data["profile"] == config.name)
            & (data["selection_name"] == config.selection_name)
            & (data["model"] == definition.identifier)
        ]
        for _, group in candidates.groupby("mass_GeV", sort=False):
            actual = np.sort(group["coupling_GeV_inv"].to_numpy(float))
            if len(group) == len(expected_couplings) and np.allclose(
                actual, expected_couplings, rtol=1e-12, atol=0.0,
            ):
                accepted.append(group)
    return pd.concat(accepted, ignore_index=True) if accepted else pd.DataFrame(columns=data.columns)


def load_resumable_source_rows(output_dir: Path, config) -> pd.DataFrame:
    """Load only complete mass scans with configuration-compatible provenance."""
    current_key = scan_configuration_key(config)
    frames = []
    candidates = (
        output_dir / "event_density_scan_sources.csv",
        output_dir / "event_density_scan_sources_checkpoint.csv",
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if "scan_configuration_key" in frame:
            frame = frame[frame["scan_configuration_key"] == current_key]
        elif path.name != "event_density_scan_sources.csv" or not _legacy_final_is_compatible(
            output_dir, config
        ):
            continue
        frame = _complete_source_groups(frame, config)
        if not frame.empty:
            frame["scan_configuration_key"] = current_key
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        ["model", "mass_GeV", "coupling_GeV_inv"], keep="last"
    )
    return _complete_source_groups(combined, config)


def main() -> None:
    args = parse_arguments()
    config, adapter = config_and_adapter(args)
    output_dir = profile_output_dir(config.name, "event_density")
    photon_name, su2_name = "ALP-photon-combined", "ALP-SU2L"
    photon_scan = next(item for item in SOURCE_SCANS if item.model_id == "alp_photon_combined")
    su2_scan = next(item for item in SOURCE_SCANS if item.model_id == "alp_su2l")
    mass_grids = {
        photon_name: np.geomspace(
            photon_scan.mass_min_gev, photon_scan.mass_max_gev, config.event_density.photon_masses
        ),
        su2_name: np.geomspace(
            su2_scan.mass_min_gev, su2_scan.mass_max_gev, config.event_density.su2_masses
        ),
    }
    resumed = pd.DataFrame() if args.force else load_resumable_source_rows(output_dir, config)
    source_rows = resumed.to_dict("records")
    completed = {definition.identifier: set() for definition in SOURCE_SCANS}
    for definition in SOURCE_SCANS:
        existing = resumed[resumed["model"] == definition.identifier] if not resumed.empty else resumed
        completed[definition.identifier].update(existing["mass_GeV"].map(stable_float_key))
    if source_rows:
        print(f"Resuming from {len(source_rows)} validated source rows")
    configuration_key = scan_configuration_key(config)

    def scan_pending_masses() -> None:
        for definition in SOURCE_SCANS:
            model = get_model(definition.model_id)
            source = next(item for item in model.sources if item.identifier == definition.source_id)
            physical_name = photon_name if definition.model_id == "alp_photon_combined" else su2_name
            couplings = np.geomspace(
                definition.coupling_min_gev_inv, definition.coupling_max_gev_inv,
                config.event_density.coupling_points,
            )
            for mass_index, mass_gev in enumerate(mass_grids[physical_name]):
                mass_key = stable_float_key(mass_gev)
                if mass_key in completed[definition.identifier]:
                    continue
                seed = event_density_seed(definition.seed_offset, mass_index)
                proposal = adapter.prepare_kinematic_proposal(
                    model, source, float(mass_gev), seed, "event_density"
                )
                for coupling in couplings:
                    coupling_squared = float(coupling**2)
                    ctau_m = proposal.unit_coupling_ctau_m / coupling_squared
                    spectrum = adapter.evaluate_spectrum(
                        proposal, ctau_m, seed + 1, coupling_squared, cache_result=False,
                    )
                    source_rows.append({
                        "profile": config.name, "selection_name": config.selection_name,
                        "scan_configuration_key": configuration_key,
                        "model": definition.identifier, "mass_GeV": mass_gev,
                        "coupling_GeV_inv": coupling, "coupling_squared_GeV_inv2": coupling_squared,
                        "ctau_m": ctau_m, "unit_coupling_ctau_m": proposal.unit_coupling_ctau_m,
                        "yield_per_PoT_per_coupling_squared": proposal.yield_per_pot_per_coupling_squared,
                        "N_LLP_total": spectrum.n_llp_total, "epsilon_polar": spectrum.epsilon_polar,
                        "epsilon_azimuthal": spectrum.epsilon_azimuthal,
                        "mean_P_decay": spectrum.mean_decay_probability,
                        "sum_P_decay": spectrum.mean_decay_probability * spectrum.accepted_samples,
                        "visible_Br": spectrum.visible_br, "sampled_inside_volume": spectrum.accepted_samples,
                        "N_events": spectrum.expected_events, "proposal_seed": seed,
                        "true_sample_seed": seed + 1, "proposal_cache_key": proposal.cache_key,
                        "spectrum_cache_key": spectrum.cache_key,
                    })
                completed[definition.identifier].add(mass_key)
                write_dataframe(
                    pd.DataFrame(source_rows).sort_values(["model", "mass_GeV", "coupling_GeV_inv"]),
                    output_dir / "event_density_scan_sources_checkpoint.csv",
                )

    for refinement_round in range(12):
        scan_pending_masses()
        sources = pd.DataFrame(source_rows)
        scan = combine_photon_sources(sources)
        boundaries = build_boundary_table(scan, config.event_density.event_levels)
        additions = endpoint_refinement_masses(
            boundaries, config.event_density.endpoint_refinement_points,
            config.event_density.endpoint_relative_width,
        )
        changed = False
        for model_name, masses in additions.items():
            combined = np.unique(np.concatenate([mass_grids[model_name], masses]))
            changed |= len(combined) > len(mass_grids[model_name])
            mass_grids[model_name] = combined
        if not changed:
            break
    else:
        raise RuntimeError("event-density endpoint refinement did not converge in 12 rounds")

    final_boundaries = add_interpolated_closing_points(boundaries)
    write_dataframe(sources, output_dir / "event_density_scan_sources.csv")
    write_dataframe(scan, output_dir / "event_density_scan_coarse.csv")
    write_dataframe(boundaries, output_dir / "event_contour_boundaries_raw.csv")
    write_dataframe(final_boundaries, output_dir / "event_contour_boundaries.csv")
    plot_event_rate_curves(scan, config.event_density.event_levels, output_dir / "plots")
    write_manifest(config, "scan_event_density", output_dir)
    print(f"Saved event-density scan to {output_dir}")


if __name__ == "__main__":
    main()
