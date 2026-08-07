"""Cache adapter for deterministic per-seed profiled pseudoexperiments."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from analysis2.cache import CacheStore, canonical_json, file_fingerprint
from analysis2.config import AnalysisConfig
from analysis2.lifetime_template_banks import LifetimeTemplateBank
from analysis2.profiled_statistics import (
    PROFILED_ACCURACY_COLUMNS,
    combine_profiled_truth_tables,
    lifetime_grid_indices,
    simulate_truth_template,
)


WORKFLOW_FORMAT_VERSION = 1
CACHE_KIND = "profiled_pseudoexperiments"

TRUTH_CACHE_FORMAT_VERSION = 1
TRUTH_CACHE_KIND = "profiled_truth_pseudoexperiments"
TRUTH_MODELS = ("photon", "su2")


_IDENTITY_INDEX: dict[tuple[str, str], list[dict]] = {}


def _cached_identities(cache: CacheStore, kind: str) -> list[dict]:
    """Parse one cache directory once per process."""
    key = (str(cache.root), kind)
    if key not in _IDENTITY_INDEX:
        identities: list[dict] = []
        directory = cache.root / kind
        if directory.is_dir():
            for metadata_path in directory.glob("*.json"):
                try:
                    metadata = json.loads(metadata_path.read_text())
                    identity = metadata["identity"]
                except (
                    OSError,
                    TypeError,
                    KeyError,
                    json.JSONDecodeError,
                ):
                    continue
                if isinstance(identity, dict):
                    identities.append(dict(identity))
        _IDENTITY_INDEX[key] = identities
    return _IDENTITY_INDEX[key]


def _register_identity(cache: CacheStore, kind: str, identity: Mapping) -> None:
    """Keep the process-local identity index coherent after a cache write."""
    if not cache.enabled:
        return
    identities = _cached_identities(cache, kind)
    serialized = canonical_json(identity)
    if all(canonical_json(item) != serialized for item in identities):
        identities.append(dict(identity))

# These settings affect one mass/seed numerical table.  The configured maximum
# and chunk size are orchestration details: the actual event-count array is
# stored separately, and the stable pseudoexperiment stream is invariant under
# chunking.
_SUPERSET_COMPATIBLE_SETTING_FIELDS = (
    "pseudoexperiments_per_truth_and_seed",
    "base_seed",
    "number_of_seeds",
    "seed_step",
    "target_accuracy",
    "tie_tolerance",
    "persistent_criterion",
    "truth_lifetime_grid",
    "profile_lifetime_grid",
    "rebin_factor",
    "shape_only",
    "independent_lifetime_profiling",
)


# Only settings that change one truth template's numerical pseudoexperiment
# table belong in the truth-level identity. Seed-plan, threshold, persistent
# reduction and chunk-size settings are orchestration details.
_TRUTH_CORE_SETTING_FIELDS = (
    "tie_tolerance",
    "profile_lifetime_grid",
    "rebin_factor",
    "shape_only",
    "independent_lifetime_profiling",
)


def validated_truth_indices(
    bank: LifetimeTemplateBank,
    config: AnalysisConfig,
    truth_indices: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return validated global bank indices for each truth model.

    Explicit truth subsets affect only which truth hypotheses are generated.
    The profile lifetime grids remain controlled independently by
    ``profile_lifetime_grid`` and are never reduced here.
    """
    settings = config.profiled_likelihood
    default = {
        "photon": lifetime_grid_indices(
            len(bank.photon_ctau_m),
            settings.truth_lifetime_grid,
        ),
        "su2": lifetime_grid_indices(
            len(bank.su2_ctau_m),
            settings.truth_lifetime_grid,
        ),
    }
    if truth_indices is None:
        return default
    if set(truth_indices) != set(TRUTH_MODELS):
        raise ValueError("Truth indices must be provided for photon and su2.")

    resolved: dict[str, np.ndarray] = {}
    lengths = {
        "photon": len(bank.photon_ctau_m),
        "su2": len(bank.su2_ctau_m),
    }
    for model in TRUTH_MODELS:
        raw = np.asarray(truth_indices[model])
        if raw.ndim != 1 or len(raw) == 0:
            raise ValueError(f"{model} truth indices must be a non-empty 1D array.")
        integer = raw.astype(int)
        if not np.array_equal(raw, integer):
            raise ValueError(f"{model} truth indices must be integers.")
        if np.any(integer < 0) or np.any(integer >= lengths[model]):
            raise ValueError(f"{model} truth index lies outside the saved bank.")
        if len(np.unique(integer)) != len(integer):
            raise ValueError(f"{model} truth indices contain duplicates.")
        resolved[model] = np.sort(integer)
    return resolved


def _truth_core_settings_from_mapping(settings: Mapping) -> dict | None:
    try:
        return {name: settings[name] for name in _TRUTH_CORE_SETTING_FIELDS}
    except (KeyError, TypeError):
        return None


def _truth_numerical_settings(
    config: AnalysisConfig,
    *,
    number_of_pseudoexperiments: int | None = None,
) -> dict:
    settings = asdict(config.profiled_likelihood)
    result = {name: settings[name] for name in _TRUTH_CORE_SETTING_FIELDS}
    result["pseudoexperiments_per_truth_and_seed"] = int(
        settings["pseudoexperiments_per_truth_and_seed"]
        if number_of_pseudoexperiments is None
        else number_of_pseudoexperiments
    )
    return result


def input_fingerprint(bank_path: Path) -> dict:
    """Fingerprint a bank and its producing manifest when one is present."""
    fingerprint = {"bank": file_fingerprint(bank_path)}
    manifest_path = bank_path.parent.parent / "manifest.json"
    if manifest_path.is_file():
        fingerprint["manifest"] = file_fingerprint(manifest_path)
    return fingerprint


def _truth_count(
    bank: LifetimeTemplateBank,
    config: AnalysisConfig,
    truth_indices: Mapping[str, np.ndarray] | None = None,
) -> int:
    selected = validated_truth_indices(bank, config, truth_indices)
    return sum(len(selected[model]) for model in TRUTH_MODELS)


def _validator(
    bank: LifetimeTemplateBank,
    config: AnalysisConfig,
    seed: int,
    event_counts: np.ndarray,
    truth_indices: Mapping[str, np.ndarray] | None = None,
):
    selected = validated_truth_indices(bank, config, truth_indices)
    expected_rows = _truth_count(bank, config, selected) * len(event_counts)
    expected_counts = np.asarray(event_counts, dtype=int)

    def validate(arrays: dict[str, np.ndarray], metadata: dict) -> None:
        if tuple(metadata.get("columns", ())) != PROFILED_ACCURACY_COLUMNS:
            raise ValueError("Cached profiled table has different columns.")
        for column in PROFILED_ACCURACY_COLUMNS:
            if column not in arrays or arrays[column].shape != (expected_rows,):
                raise ValueError(
                    f"Cached profiled column {column!r} has the wrong shape."
                )
        if not np.all(arrays["seed"] == seed):
            raise ValueError("Cached profiled table has a different seed.")
        if not np.allclose(
            arrays["mass_GeV"], bank.mass_gev, rtol=0.0, atol=0.0,
        ):
            raise ValueError("Cached profiled table has a different mass.")
        if set(np.unique(arrays["truth_model"])) != set(TRUTH_MODELS):
            raise ValueError("Cached profiled table must contain both truth models.")
        for model in TRUTH_MODELS:
            mask = arrays["truth_model"] == model
            cached_indices = np.unique(
                arrays["truth_lifetime_index"][mask].astype(int, copy=False)
            )
            if not np.array_equal(cached_indices, selected[model]):
                raise ValueError(
                    f"Cached profiled table has different {model} truth indices."
                )
        cached_counts = np.unique(
            arrays["number_of_events"].astype(int, copy=False)
        )
        if not np.array_equal(cached_counts, expected_counts):
            raise ValueError("Cached profiled table has different event counts.")

    return validate


def _truth_validator(
    bank: LifetimeTemplateBank,
    seed: int,
    event_counts: np.ndarray,
    truth_model: str,
    truth_index: int,
    number_of_pseudoexperiments: int | None = None,
):
    expected_rows = len(event_counts)
    expected_counts = np.asarray(event_counts, dtype=int)
    lifetimes = (
        bank.photon_ctau_m
        if truth_model == "photon"
        else bank.su2_ctau_m
    )
    expected_ctau = float(lifetimes[truth_index])

    def validate(arrays: dict[str, np.ndarray], metadata: dict) -> None:
        if tuple(metadata.get("columns", ())) != PROFILED_ACCURACY_COLUMNS:
            raise ValueError("Cached truth table has different columns.")
        for column in PROFILED_ACCURACY_COLUMNS:
            if column not in arrays or arrays[column].shape != (expected_rows,):
                raise ValueError(
                    f"Cached truth column {column!r} has the wrong shape."
                )
        if not np.all(arrays["seed"] == seed):
            raise ValueError("Cached truth table has a different seed.")
        if not np.allclose(
            arrays["mass_GeV"], bank.mass_gev, rtol=0.0, atol=0.0,
        ):
            raise ValueError("Cached truth table has a different mass.")
        if not np.all(arrays["truth_model"] == truth_model):
            raise ValueError("Cached truth table has a different truth model.")
        if not np.all(arrays["truth_lifetime_index"] == truth_index):
            raise ValueError("Cached truth table has a different truth index.")
        if not np.allclose(
            arrays["truth_ctau_m"], expected_ctau, rtol=0.0, atol=0.0,
        ):
            raise ValueError("Cached truth table has a different truth lifetime.")
        if number_of_pseudoexperiments is not None and not np.all(
            arrays["number_of_pseudoexperiments"]
            == int(number_of_pseudoexperiments)
        ):
            raise ValueError(
                "Cached truth table has a different pseudoexperiment count."
            )
        cached_counts = np.unique(
            arrays["number_of_events"].astype(int, copy=False)
        )
        if not np.array_equal(cached_counts, expected_counts):
            raise ValueError("Cached truth table has different event counts.")

    return validate

def _table_from_arrays(arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: arrays[column] for column in PROFILED_ACCURACY_COLUMNS},
        columns=PROFILED_ACCURACY_COLUMNS,
    )


def _compatible_settings(settings: Mapping) -> dict | None:
    try:
        return {
            name: settings[name]
            for name in _SUPERSET_COMPATIBLE_SETTING_FIELDS
        }
    except (KeyError, TypeError):
        return None


def _find_cached_superset_identity(
    cache: CacheStore,
    requested_identity: Mapping,
) -> tuple[dict, np.ndarray] | None:
    """Find the smallest compatible cached event-count superset, if any."""
    directory = cache.root / CACHE_KIND
    if not directory.is_dir():
        return None

    requested_counts = {
        int(value) for value in requested_identity["event_counts"]
    }
    requested_settings = _compatible_settings(requested_identity["settings"])
    candidates: list[tuple[int, str, dict, np.ndarray]] = []
    for metadata_path in directory.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
            identity = metadata["identity"]
            candidate_counts = np.asarray(identity["event_counts"], dtype=int)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if identity.get("workflow_format_version") != WORKFLOW_FORMAT_VERSION:
            continue
        if identity.get("mass_gev") != requested_identity["mass_gev"]:
            continue
        if identity.get("seed") != requested_identity["seed"]:
            continue
        if canonical_json(identity.get("input")) != canonical_json(
            requested_identity["input"]
        ):
            continue
        if canonical_json(_compatible_settings(identity.get("settings"))) != canonical_json(
            requested_settings
        ):
            continue
        if candidate_counts.ndim != 1 or len(candidate_counts) == 0:
            continue
        candidate_set = set(int(value) for value in candidate_counts)
        if not requested_counts.issubset(candidate_set):
            continue
        candidates.append(
            (
                len(candidate_set),
                canonical_json(identity),
                dict(identity),
                np.unique(candidate_counts),
            )
        )

    if not candidates:
        return None
    _, _, identity, counts = min(candidates)
    return identity, counts



def _find_cached_truth_superset_identity(
    cache: CacheStore,
    requested_identity: Mapping,
) -> tuple[dict, np.ndarray] | None:
    """Find the smallest compatible truth-cache event-count superset."""
    requested_counts = {
        int(value) for value in requested_identity["event_counts"]
    }
    candidates: list[tuple[int, str, dict, np.ndarray]] = []
    for identity in _cached_identities(cache, TRUTH_CACHE_KIND):
        try:
            candidate_counts = np.asarray(identity["event_counts"], dtype=int)
        except (ValueError, TypeError, KeyError):
            continue
        if (
            identity.get("truth_cache_format_version")
            != TRUTH_CACHE_FORMAT_VERSION
        ):
            continue
        for name in (
            "mass_gev",
            "seed",
            "truth_model",
            "truth_lifetime_index",
        ):
            if identity.get(name) != requested_identity[name]:
                break
        else:
            if canonical_json(identity.get("input")) != canonical_json(
                requested_identity["input"]
            ):
                continue
            if canonical_json(identity.get("settings")) != canonical_json(
                requested_identity["settings"]
            ):
                continue
            if candidate_counts.ndim != 1 or len(candidate_counts) == 0:
                continue
            candidate_set = set(int(value) for value in candidate_counts)
            if not requested_counts.issubset(candidate_set):
                continue
            candidates.append(
                (
                    len(candidate_set),
                    canonical_json(identity),
                    dict(identity),
                    np.unique(candidate_counts),
                )
            )

    if not candidates:
        return None
    _, _, identity, counts = min(candidates)
    return identity, counts


def _truth_identity(
    *,
    bank: LifetimeTemplateBank,
    bank_path: Path,
    config: AnalysisConfig,
    seed: int,
    truth_model: str,
    truth_index: int,
    event_counts: np.ndarray,
    number_of_pseudoexperiments: int,
) -> dict:
    return {
        "truth_cache_format_version": TRUTH_CACHE_FORMAT_VERSION,
        "input": input_fingerprint(bank_path),
        "mass_gev": bank.mass_gev,
        "seed": int(seed),
        "truth_model": truth_model,
        "truth_lifetime_index": int(truth_index),
        "settings": _truth_numerical_settings(
            config,
            number_of_pseudoexperiments=number_of_pseudoexperiments,
        ),
        "event_counts": np.asarray(event_counts, dtype=int).tolist(),
    }


def _identity_pseudoexperiment_count(identity: Mapping) -> int | None:
    try:
        value = int(
            identity["settings"]["pseudoexperiments_per_truth_and_seed"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _find_cached_truth_prefix_identity(
    cache: CacheStore,
    requested_identity: Mapping,
) -> tuple[dict, np.ndarray, int] | None:
    """Find the largest compatible lower-statistics truth checkpoint."""
    requested_total = _identity_pseudoexperiment_count(requested_identity)
    if requested_total is None:
        return None
    requested_counts = {
        int(value) for value in requested_identity["event_counts"]
    }
    requested_core = _truth_core_settings_from_mapping(
        requested_identity.get("settings")
    )
    candidates: list[tuple[int, int, str, dict, np.ndarray]] = []
    for identity in _cached_identities(cache, TRUTH_CACHE_KIND):
        try:
            candidate_counts = np.asarray(identity["event_counts"], dtype=int)
        except (ValueError, TypeError, KeyError):
            continue
        if (
            identity.get("truth_cache_format_version")
            != TRUTH_CACHE_FORMAT_VERSION
        ):
            continue
        if any(
            identity.get(name) != requested_identity[name]
            for name in (
                "mass_gev",
                "seed",
                "truth_model",
                "truth_lifetime_index",
            )
        ):
            continue
        if canonical_json(identity.get("input")) != canonical_json(
            requested_identity["input"]
        ):
            continue
        if canonical_json(
            _truth_core_settings_from_mapping(identity.get("settings"))
        ) != canonical_json(requested_core):
            continue
        candidate_total = _identity_pseudoexperiment_count(identity)
        if candidate_total is None or candidate_total >= requested_total:
            continue
        if candidate_counts.ndim != 1 or len(candidate_counts) == 0:
            continue
        candidate_set = set(int(value) for value in candidate_counts)
        if not requested_counts.issubset(candidate_set):
            continue
        candidates.append(
            (
                -candidate_total,
                len(candidate_set),
                canonical_json(identity),
                dict(identity),
                np.unique(candidate_counts),
            )
        )

    if not candidates:
        return None
    negative_total, _, _, identity, counts = min(candidates)
    return identity, counts, -negative_total


def _find_cached_legacy_prefix_identity(
    cache: CacheStore,
    requested_identity: Mapping,
) -> tuple[dict, np.ndarray, int] | None:
    """Find the largest compatible lower-statistics legacy seed cache."""
    requested_total = _identity_pseudoexperiment_count(requested_identity)
    if requested_total is None:
        return None
    requested_counts = {
        int(value) for value in requested_identity["event_counts"]
    }
    requested_core = _truth_core_settings_from_mapping(
        requested_identity.get("settings")
    )
    candidates: list[tuple[int, int, str, dict, np.ndarray]] = []
    for identity in _cached_identities(cache, CACHE_KIND):
        try:
            candidate_counts = np.asarray(identity["event_counts"], dtype=int)
        except (ValueError, TypeError, KeyError):
            continue
        if identity.get("workflow_format_version") != WORKFLOW_FORMAT_VERSION:
            continue
        if identity.get("mass_gev") != requested_identity["mass_gev"]:
            continue
        if identity.get("seed") != requested_identity["seed"]:
            continue
        if canonical_json(identity.get("input")) != canonical_json(
            requested_identity["input"]
        ):
            continue
        if canonical_json(
            _truth_core_settings_from_mapping(identity.get("settings"))
        ) != canonical_json(requested_core):
            continue
        candidate_total = _identity_pseudoexperiment_count(identity)
        if candidate_total is None or candidate_total >= requested_total:
            continue
        if candidate_counts.ndim != 1 or len(candidate_counts) == 0:
            continue
        candidate_set = set(int(value) for value in candidate_counts)
        if not requested_counts.issubset(candidate_set):
            continue
        candidates.append(
            (
                -candidate_total,
                len(candidate_set),
                canonical_json(identity),
                dict(identity),
                np.unique(candidate_counts),
            )
        )

    if not candidates:
        return None
    negative_total, _, _, identity, counts = min(candidates)
    return identity, counts, -negative_total


def _pseudoexperiment_ranges_from_metadata(
    metadata: Mapping,
    number_of_pseudoexperiments: int,
) -> list[list[int]]:
    raw = metadata.get("pseudoexperiment_ranges")
    if raw is None:
        return [[0, int(number_of_pseudoexperiments)]]
    ranges = [[int(item[0]), int(item[1])] for item in raw]
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != int(
        number_of_pseudoexperiments
    ):
        raise ValueError("Cached pseudoexperiment ranges are not a full prefix.")
    if any(
        lower < 0 or upper <= lower or (
            index > 0 and lower != ranges[index - 1][1]
        )
        for index, (lower, upper) in enumerate(ranges)
    ):
        raise ValueError("Cached pseudoexperiment ranges are not contiguous.")
    return ranges


def _attach_pseudoexperiment_ranges(
    detailed: pd.DataFrame,
    ranges: list[list[int]],
) -> pd.DataFrame:
    detailed.attrs["pseudoexperiment_ranges"] = [
        [int(lower), int(upper)] for lower, upper in ranges
    ]
    return detailed


def _slice_truth_table(
    detailed: pd.DataFrame,
    *,
    truth_model: str,
    truth_index: int,
    event_counts: np.ndarray,
) -> pd.DataFrame:
    requested = set(int(value) for value in event_counts)
    return detailed.loc[
        (detailed["truth_model"] == truth_model)
        & (detailed["truth_lifetime_index"].astype(int) == int(truth_index))
        & detailed["number_of_events"].astype(int).isin(requested),
        PROFILED_ACCURACY_COLUMNS,
    ].reset_index(drop=True)

def _arrays_from_table(detailed: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: detailed[column].to_numpy(
            dtype=str if column == "truth_model" else None
        )
        for column in PROFILED_ACCURACY_COLUMNS
    }


def cached_profiled_truth(
    bank: LifetimeTemplateBank,
    bank_path: Path,
    config: AnalysisConfig,
    seed: int,
    truth_model: str,
    truth_index: int,
    event_counts: np.ndarray,
    cache: CacheStore,
    *,
    force: bool,
) -> pd.DataFrame:
    """Load, extend, or calculate one checkpointed truth hypothesis.

    When a compatible lower-statistics prefix exists, only the missing
    contiguous pseudoexperiment range is profiled.  The stable truth RNG is
    advanced to the cached prefix length, and exact classification numerators
    are combined with statistically equivalent first and second T moments.
    """
    if truth_model not in TRUTH_MODELS:
        raise ValueError(f"Unknown truth model: {truth_model}")
    lengths = {
        "photon": len(bank.photon_ctau_m),
        "su2": len(bank.su2_ctau_m),
    }
    truth_index = int(truth_index)
    if truth_index < 0 or truth_index >= lengths[truth_model]:
        raise ValueError(f"{truth_model} truth index lies outside the saved bank.")

    event_counts = np.asarray(event_counts, dtype=int)
    settings = config.profiled_likelihood
    target_total = int(settings.pseudoexperiments_per_truth_and_seed)
    identity = _truth_identity(
        bank=bank,
        bank_path=bank_path,
        config=config,
        seed=seed,
        truth_model=truth_model,
        truth_index=truth_index,
        event_counts=event_counts,
        number_of_pseudoexperiments=target_total,
    )
    validator = _truth_validator(
        bank,
        seed,
        event_counts,
        truth_model,
        truth_index,
        target_total,
    )

    if not force:
        loaded = cache.load(TRUTH_CACHE_KIND, identity, validator)
        if loaded is not None:
            arrays, metadata = loaded
            detailed = _table_from_arrays(arrays)
            ranges = _pseudoexperiment_ranges_from_metadata(
                metadata,
                target_total,
            )
            return _attach_pseudoexperiment_ranges(detailed, ranges)

        compatible = _find_cached_truth_superset_identity(cache, identity)
        if compatible is not None:
            superset_identity, superset_counts = compatible
            loaded = cache.load(
                TRUTH_CACHE_KIND,
                superset_identity,
                _truth_validator(
                    bank,
                    seed,
                    superset_counts,
                    truth_model,
                    truth_index,
                    target_total,
                ),
            )
            if loaded is not None:
                arrays, metadata = loaded
                detailed = _table_from_arrays(arrays)
                requested = set(int(value) for value in event_counts)
                detailed = detailed.loc[
                    detailed["number_of_events"].astype(int).isin(requested),
                    PROFILED_ACCURACY_COLUMNS,
                ].reset_index(drop=True)
                ranges = _pseudoexperiment_ranges_from_metadata(
                    metadata,
                    target_total,
                )
                simulation_maximum_events = int(
                    metadata.get(
                        "simulation_maximum_events",
                        int(np.max(superset_counts)),
                    )
                )
                cache.save(
                    TRUTH_CACHE_KIND,
                    identity,
                    _arrays_from_table(detailed),
                    {
                        "columns": PROFILED_ACCURACY_COLUMNS,
                        "rows": len(detailed),
                        "derived_from_cached_event_count_superset": True,
                        "pseudoexperiment_ranges": ranges,
                        "simulation_maximum_events": simulation_maximum_events,
                        "rng_state_after": metadata.get("rng_state_after"),
                        "rng_state_resume_available": (
                            metadata.get("rng_state_after") is not None
                        ),
                    },
                )
                _register_identity(cache, TRUTH_CACHE_KIND, identity)
                return _attach_pseudoexperiment_ranges(detailed, ranges)

    photon_profile = bank.photon_probabilities[
        lifetime_grid_indices(
            len(bank.photon_ctau_m),
            settings.profile_lifetime_grid,
        )
    ]
    su2_profile = bank.su2_probabilities[
        lifetime_grid_indices(
            len(bank.su2_ctau_m),
            settings.profile_lifetime_grid,
        )
    ]
    if truth_model == "photon":
        truth_ctau_m = float(bank.photon_ctau_m[truth_index])
        truth_probabilities = bank.photon_probabilities[truth_index]
    else:
        truth_ctau_m = float(bank.su2_ctau_m[truth_index])
        truth_probabilities = bank.su2_probabilities[truth_index]

    prefix_table: pd.DataFrame | None = None
    prefix_ranges: list[list[int]] = []
    prefix_total = 0
    prefix_rng_state: Mapping | None = None
    simulation_maximum_events = int(event_counts[-1])
    if not force:
        prefix = _find_cached_truth_prefix_identity(cache, identity)
        if prefix is not None:
            prefix_identity, prefix_counts, prefix_total = prefix
            loaded = cache.load(
                TRUTH_CACHE_KIND,
                prefix_identity,
                _truth_validator(
                    bank,
                    seed,
                    prefix_counts,
                    truth_model,
                    truth_index,
                    prefix_total,
                ),
            )
            if loaded is not None:
                arrays, metadata = loaded
                prefix_table = _table_from_arrays(arrays)
                requested = set(int(value) for value in event_counts)
                prefix_table = prefix_table.loc[
                    prefix_table["number_of_events"].astype(int).isin(requested),
                    PROFILED_ACCURACY_COLUMNS,
                ].reset_index(drop=True)
                prefix_ranges = _pseudoexperiment_ranges_from_metadata(
                    metadata,
                    prefix_total,
                )
                simulation_maximum_events = int(
                    metadata.get(
                        "simulation_maximum_events",
                        int(np.max(prefix_counts)),
                    )
                )
                prefix_rng_state = metadata.get("rng_state_after")

    if prefix_table is None:
        detailed = simulate_truth_template(
            mass_gev=bank.mass_gev,
            truth_model=truth_model,
            truth_index=truth_index,
            truth_ctau_m=truth_ctau_m,
            truth_probabilities=truth_probabilities,
            photon_probabilities=photon_profile,
            su2_probabilities=su2_profile,
            event_counts=event_counts,
            number_of_pseudoexperiments=target_total,
            seed=seed,
            chunk_size=settings.chunk_size,
            tie_tolerance=settings.tie_tolerance,
        )
        ranges = [[0, target_total]]
        simulation_maximum_events = int(event_counts[-1])
        progressive = False
        rng_state_after = detailed.attrs.get("rng_state_after")
        rng_state_resume_used = False
    else:
        extension = simulate_truth_template(
            mass_gev=bank.mass_gev,
            truth_model=truth_model,
            truth_index=truth_index,
            truth_ctau_m=truth_ctau_m,
            truth_probabilities=truth_probabilities,
            photon_probabilities=photon_profile,
            su2_probabilities=su2_profile,
            event_counts=event_counts,
            number_of_pseudoexperiments=target_total - prefix_total,
            seed=seed,
            chunk_size=settings.chunk_size,
            tie_tolerance=settings.tie_tolerance,
            pseudoexperiment_start=(
                prefix_total if prefix_rng_state is None else 0
            ),
            maximum_sampled_events=simulation_maximum_events,
            initial_rng_state=prefix_rng_state,
        )
        detailed = combine_profiled_truth_tables([prefix_table, extension])
        ranges = [*prefix_ranges, [prefix_total, target_total]]
        progressive = True
        rng_state_after = extension.attrs.get("rng_state_after")
        rng_state_resume_used = prefix_rng_state is not None

    cache.save(
        TRUTH_CACHE_KIND,
        identity,
        _arrays_from_table(detailed),
        {
            "columns": PROFILED_ACCURACY_COLUMNS,
            "rows": len(detailed),
            "truth_level_checkpoint": True,
            "progressively_extended": progressive,
            "pseudoexperiment_ranges": ranges,
            "simulation_maximum_events": simulation_maximum_events,
            "rng_state_after": rng_state_after,
            "rng_state_resume_used": rng_state_resume_used,
            "classification_fraction_combination": (
                "exact_half_integer_numerators"
            ),
            "profile_statistic_moment_combination": (
                "population_first_and_second_moments"
            ),
        },
    )
    _register_identity(cache, TRUTH_CACHE_KIND, identity)
    return _attach_pseudoexperiment_ranges(detailed, ranges)


def _materialize_legacy_seed_prefix_truth_caches(
    *,
    bank: LifetimeTemplateBank,
    bank_path: Path,
    config: AnalysisConfig,
    seed: int,
    event_counts: np.ndarray,
    selected: Mapping[str, np.ndarray],
    cache: CacheStore,
) -> None:
    """Split one lower-statistics legacy seed cache into truth checkpoints."""
    requested = {
        "input": input_fingerprint(bank_path),
        "mass_gev": bank.mass_gev,
        "seed": int(seed),
        "settings": _truth_numerical_settings(config),
        "event_counts": np.asarray(event_counts, dtype=int).tolist(),
    }
    candidate = _find_cached_legacy_prefix_identity(cache, requested)
    if candidate is None:
        return
    legacy_identity, legacy_counts, prefix_total = candidate
    loaded = cache.load(
        CACHE_KIND,
        legacy_identity,
        _validator(bank, config, seed, legacy_counts),
    )
    if loaded is None:
        return
    arrays, metadata = loaded
    detailed = _table_from_arrays(arrays)
    if not np.all(
        detailed["number_of_pseudoexperiments"].to_numpy(dtype=int)
        == prefix_total
    ):
        raise ValueError(
            "Legacy profiled cache has an inconsistent pseudoexperiment count."
        )
    ranges = _pseudoexperiment_ranges_from_metadata(metadata, prefix_total)
    simulation_maximum_events = int(
        metadata.get("simulation_maximum_events", int(np.max(legacy_counts)))
    )

    for truth_model in TRUTH_MODELS:
        for truth_index in selected[truth_model]:
            truth_identity = _truth_identity(
                bank=bank,
                bank_path=bank_path,
                config=config,
                seed=seed,
                truth_model=truth_model,
                truth_index=int(truth_index),
                event_counts=event_counts,
                number_of_pseudoexperiments=prefix_total,
            )
            array_path, metadata_path, _ = cache.paths(
                TRUTH_CACHE_KIND,
                truth_identity,
            )
            if array_path.is_file() and metadata_path.is_file():
                continue
            truth_table = _slice_truth_table(
                detailed,
                truth_model=truth_model,
                truth_index=int(truth_index),
                event_counts=event_counts,
            )
            if len(truth_table) != len(event_counts):
                raise ValueError(
                    "Legacy seed cache is missing a requested truth checkpoint."
                )
            cache.save(
                TRUTH_CACHE_KIND,
                truth_identity,
                _arrays_from_table(truth_table),
                {
                    "columns": PROFILED_ACCURACY_COLUMNS,
                    "rows": len(truth_table),
                    "truth_level_checkpoint": True,
                    "migrated_from_legacy_seed_cache": True,
                    "pseudoexperiment_ranges": ranges,
                    "simulation_maximum_events": simulation_maximum_events,
                },
            )
            _register_identity(cache, TRUTH_CACHE_KIND, truth_identity)


def cached_profiled_seed(
    bank: LifetimeTemplateBank,
    bank_path: Path,
    config: AnalysisConfig,
    seed: int,
    event_counts: np.ndarray,
    cache: CacheStore,
    *,
    force: bool,
    truth_indices: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Load or calculate one mass/seed detailed pseudoexperiment table.

    Existing complete mass-seed caches remain readable for the default truth
    grid.  Otherwise each truth lifetime is loaded or calculated independently,
    so completed truths survive interruption and explicit truth subsets can be
    evaluated without reducing either model's profile lifetime grid.
    """
    selected = validated_truth_indices(bank, config, truth_indices)
    default = validated_truth_indices(bank, config)
    is_default_selection = all(
        np.array_equal(selected[model], default[model])
        for model in TRUTH_MODELS
    )
    settings = config.profiled_likelihood
    legacy_identity = {
        "workflow_format_version": WORKFLOW_FORMAT_VERSION,
        "input": input_fingerprint(bank_path),
        "mass_gev": bank.mass_gev,
        "seed": int(seed),
        "settings": asdict(settings),
        "event_counts": event_counts.tolist(),
    }

    # Preserve all expensive pre-patch mass-seed caches.  Disabled caches skip
    # this legacy probe so their miss count reflects only actual truth tasks.
    if cache.enabled and not force and is_default_selection:
        legacy_array_path, legacy_metadata_path, _ = cache.paths(
            CACHE_KIND,
            legacy_identity,
        )
        if legacy_array_path.is_file() and legacy_metadata_path.is_file():
            loaded = cache.load(
                CACHE_KIND,
                legacy_identity,
                _validator(bank, config, seed, event_counts, selected),
            )
            if loaded is not None:
                return _table_from_arrays(loaded[0])

        compatible = _find_cached_superset_identity(cache, legacy_identity)
        if compatible is not None:
            superset_identity, superset_counts = compatible
            loaded = cache.load(
                CACHE_KIND,
                superset_identity,
                _validator(bank, config, seed, superset_counts, selected),
            )
            if loaded is not None:
                requested = set(int(value) for value in event_counts)
                detailed = _table_from_arrays(loaded[0])
                detailed = detailed.loc[
                    detailed["number_of_events"].astype(int).isin(requested),
                    PROFILED_ACCURACY_COLUMNS,
                ].reset_index(drop=True)
                cache.save(
                    CACHE_KIND,
                    legacy_identity,
                    _arrays_from_table(detailed),
                    {
                        "columns": PROFILED_ACCURACY_COLUMNS,
                        "rows": len(detailed),
                        "derived_from_cached_event_count_superset": True,
                    },
                )
                _register_identity(cache, CACHE_KIND, legacy_identity)
                return detailed

    if cache.enabled and not force:
        first_model = TRUTH_MODELS[0]
        first_index = int(selected[first_model][0])
        first_identity = _truth_identity(
            bank=bank,
            bank_path=bank_path,
            config=config,
            seed=seed,
            truth_model=first_model,
            truth_index=first_index,
            event_counts=event_counts,
            number_of_pseudoexperiments=(
                settings.pseudoexperiments_per_truth_and_seed
            ),
        )
        target_array, target_metadata, _ = cache.paths(
            TRUTH_CACHE_KIND,
            first_identity,
        )
        has_truth_checkpoint = (
            target_array.is_file() and target_metadata.is_file()
        ) or _find_cached_truth_prefix_identity(cache, first_identity) is not None
        if not has_truth_checkpoint:
            _materialize_legacy_seed_prefix_truth_caches(
                bank=bank,
                bank_path=bank_path,
                config=config,
                seed=seed,
                event_counts=event_counts,
                selected=selected,
                cache=cache,
            )

    frames = []
    for truth_model in TRUTH_MODELS:
        for truth_index in selected[truth_model]:
            frames.append(
                cached_profiled_truth(
                    bank,
                    bank_path,
                    config,
                    seed,
                    truth_model,
                    int(truth_index),
                    event_counts,
                    cache,
                    force=force,
                )
            )
    detailed = pd.concat(frames, ignore_index=True)
    unique_ranges = sorted(
        {
            (int(lower), int(upper))
            for frame in frames
            for lower, upper in frame.attrs.get(
                "pseudoexperiment_ranges",
                [[0, int(settings.pseudoexperiments_per_truth_and_seed)]],
            )
        }
    )
    detailed.attrs["pseudoexperiment_ranges"] = [
        [lower, upper] for lower, upper in unique_ranges
    ]
    return detailed
