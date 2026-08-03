"""Cache adapter for deterministic per-seed profiled pseudoexperiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis2.cache import CacheStore, file_fingerprint
from analysis2.config import AnalysisConfig
from analysis2.lifetime_template_banks import LifetimeTemplateBank
from analysis2.profiled_statistics import (
    PROFILED_ACCURACY_COLUMNS,
    lifetime_grid_indices,
    run_profiled_seed,
)


WORKFLOW_FORMAT_VERSION = 1


def input_fingerprint(bank_path: Path) -> dict:
    """Fingerprint a bank and its producing manifest when one is present."""
    fingerprint = {"bank": file_fingerprint(bank_path)}
    manifest_path = bank_path.parent.parent / "manifest.json"
    if manifest_path.is_file():
        fingerprint["manifest"] = file_fingerprint(manifest_path)
    return fingerprint


def cached_profiled_seed(
    bank: LifetimeTemplateBank,
    bank_path: Path,
    config: AnalysisConfig,
    seed: int,
    event_counts: np.ndarray,
    cache: CacheStore,
    *,
    force: bool,
) -> pd.DataFrame:
    """Load or calculate one mass/seed detailed pseudoexperiment table."""
    settings = config.profiled_likelihood
    identity = {
        "workflow_format_version": WORKFLOW_FORMAT_VERSION,
        "input": input_fingerprint(bank_path),
        "mass_gev": bank.mass_gev,
        "seed": int(seed),
        "settings": asdict(settings),
        "event_counts": event_counts.tolist(),
    }
    truth_count = len(
        lifetime_grid_indices(len(bank.photon_ctau_m), settings.truth_lifetime_grid)
    ) + len(lifetime_grid_indices(len(bank.su2_ctau_m), settings.truth_lifetime_grid))
    expected_rows = truth_count * len(event_counts)

    def validate(arrays: dict[str, np.ndarray], metadata: dict) -> None:
        if tuple(metadata.get("columns", ())) != PROFILED_ACCURACY_COLUMNS:
            raise ValueError("Cached profiled table has different columns.")
        for column in PROFILED_ACCURACY_COLUMNS:
            if column not in arrays or arrays[column].shape != (expected_rows,):
                raise ValueError(f"Cached profiled column {column!r} has the wrong shape.")
        if not np.all(arrays["seed"] == seed):
            raise ValueError("Cached profiled table has a different seed.")
        if not np.allclose(
            arrays["mass_GeV"], bank.mass_gev, rtol=0.0, atol=0.0,
        ):
            raise ValueError("Cached profiled table has a different mass.")
        if not set(np.unique(arrays["truth_model"])).issubset({"photon", "su2"}):
            raise ValueError("Cached profiled table has an invalid truth model.")

    if not force:
        loaded = cache.load("profiled_pseudoexperiments", identity, validate)
        if loaded is not None:
            arrays, _ = loaded
            return pd.DataFrame(
                {column: arrays[column] for column in PROFILED_ACCURACY_COLUMNS},
                columns=PROFILED_ACCURACY_COLUMNS,
            )

    detailed = run_profiled_seed(
        mass_gev=bank.mass_gev,
        photon_ctau_m=bank.photon_ctau_m,
        photon_probabilities=bank.photon_probabilities,
        su2_ctau_m=bank.su2_ctau_m,
        su2_probabilities=bank.su2_probabilities,
        event_counts=event_counts,
        number_of_pseudoexperiments=settings.pseudoexperiments_per_truth_and_seed,
        seed=seed,
        chunk_size=settings.chunk_size,
        tie_tolerance=settings.tie_tolerance,
        truth_grid=settings.truth_lifetime_grid,
        profile_grid=settings.profile_lifetime_grid,
    )
    arrays = {
        column: detailed[column].to_numpy(
            dtype=str if column == "truth_model" else None
        )
        for column in PROFILED_ACCURACY_COLUMNS
    }
    cache.save(
        "profiled_pseudoexperiments",
        identity,
        arrays,
        {"columns": PROFILED_ACCURACY_COLUMNS, "rows": len(detailed)},
    )
    return detailed
