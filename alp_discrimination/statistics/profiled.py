"""Lifetime-profiled frozen-reference shape-only pseudoexperiments.

For an observed bin sequence, the lifetime is maximized independently on the
two model grids and the test statistic is

``T = 2 * (max_log_L_SU2 - max_log_L_photon)``.

Expected event rates do not enter this module: every pseudoexperiment is
conditioned on its observed event count.  Random streams and accumulation
order reproduce the legacy frozen-reference implementation.
"""

from __future__ import annotations

import zlib
from typing import Mapping

import numpy as np
import pandas as pd


TRUTH_MODELS = ("photon", "su2")
PROFILED_ACCURACY_COLUMNS = (
    "mass_GeV",
    "seed",
    "truth_model",
    "truth_lifetime_index",
    "truth_ctau_m",
    "number_of_events",
    "number_of_pseudoexperiments",
    "correct_fraction",
    "selected_photon_fraction",
    "selected_su2_fraction",
    "tie_fraction",
    "mean_profile_statistic_T",
    "std_profile_statistic_T",
)


def lifetime_grid_indices(length: int, mode: str) -> np.ndarray:
    """Select all, even, or odd zero-based indices from a lifetime grid."""
    if length < 1:
        raise ValueError("A lifetime grid must contain at least one template.")
    if mode == "all":
        indices = np.arange(length, dtype=int)
    elif mode == "even":
        indices = np.arange(0, length, 2, dtype=int)
    elif mode == "odd":
        indices = np.arange(1, length, 2, dtype=int)
    else:
        raise ValueError(f"Unknown lifetime-grid mode: {mode}")
    if len(indices) == 0:
        raise ValueError(
            f"Lifetime-grid mode {mode!r} selects no templates from length {length}."
        )
    return indices


def _probability_matrix(probabilities: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{label} probabilities must be a non-empty matrix.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{label} probabilities must be finite and positive.")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Every {label} probability template must sum to one.")
    return values


_PROFILE_TEMPORARY_TARGET_BYTES = 32 * 1024**2


def _profile_log_likelihoods(
    sampled_bins: np.ndarray,
    log_templates: np.ndarray,
    event_indices: np.ndarray,
) -> np.ndarray:
    """Maximize prefix log likelihoods over one model's lifetime grid.

    Block over pseudoexperiments, not lifetime templates.  For each block,
    ``log_templates[:, sampled_bins_block]`` therefore has the same
    lifetime-fast memory layout and uses the same rank-three cumulative-sum
    path as the legacy implementation.  This preserves the frozen numerical
    result exactly while bounding the dominant temporary allocation.
    """
    number_of_pseudoexperiments = sampled_bins.shape[0]
    maximum_events = sampled_bins.shape[1]
    bytes_per_pseudoexperiment = (
        log_templates.shape[0]
        * maximum_events
        * np.dtype(float).itemsize
    )
    pseudoexperiment_block_size = max(
        1,
        min(
            number_of_pseudoexperiments,
            _PROFILE_TEMPORARY_TARGET_BYTES
            // bytes_per_pseudoexperiment,
        ),
    )
    best = np.empty(
        (number_of_pseudoexperiments, len(event_indices)),
        dtype=float,
        order="F",
    )
    for block_start in range(
        0,
        number_of_pseudoexperiments,
        pseudoexperiment_block_size,
    ):
        block_stop = min(
            number_of_pseudoexperiments,
            block_start + pseudoexperiment_block_size,
        )
        contributions = log_templates[:, sampled_bins[block_start:block_stop]]
        np.cumsum(contributions, axis=2, out=contributions)
        best[block_start:block_stop] = np.max(
            contributions[:, :, event_indices],
            axis=0,
        )
    return best


def profile_log_likelihoods(
    sampled_bins: np.ndarray,
    template_probabilities: np.ndarray,
    event_counts: np.ndarray,
) -> np.ndarray:
    """Return independently usable profile maxima for fixed bin sequences."""
    templates = _probability_matrix(template_probabilities, label="template")
    samples = np.asarray(sampled_bins)
    counts = np.asarray(event_counts, dtype=int)
    if samples.ndim != 2 or not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("sampled_bins must be a two-dimensional integer array.")
    if np.any(samples < 0) or np.any(samples >= templates.shape[1]):
        raise ValueError("A sampled bin lies outside the template binning.")
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("At least one event count is required.")
    if np.any(counts <= 0) or np.any(np.diff(counts) <= 0):
        raise ValueError("event_counts must be positive, unique, and increasing.")
    if counts[-1] > samples.shape[1]:
        raise ValueError("An event count exceeds the sampled prefix length.")
    return _profile_log_likelihoods(samples, np.log(templates), counts - 1)


def stable_truth_rng(
    *,
    seed: int,
    mass_gev: float,
    truth_model: str,
    truth_index: int,
) -> np.random.Generator:
    """Build the legacy order-independent stream for one truth template."""
    if truth_model not in TRUTH_MODELS:
        raise ValueError(f"Unknown truth model: {truth_model}")
    mass_hash = zlib.crc32(f"{mass_gev:.16g}".encode("ascii"))
    model_code = 0 if truth_model == "photon" else 1
    sequence = np.random.SeedSequence(
        [int(seed), int(mass_hash), model_code, int(truth_index)]
    )
    return np.random.default_rng(sequence)


def _discard_truth_stream_prefix(
    *,
    rng: np.random.Generator,
    truth: np.ndarray,
    maximum_events: int,
    number_of_pseudoexperiments: int,
    chunk_size: int,
) -> None:
    """Advance one stable truth stream without profiling discarded draws."""
    discarded = 0
    while discarded < number_of_pseudoexperiments:
        current_chunk = min(
            chunk_size,
            number_of_pseudoexperiments - discarded,
        )
        rng.choice(
            len(truth),
            size=(current_chunk, maximum_events),
            replace=True,
            p=truth,
        )
        discarded += current_chunk


def simulate_truth_template(
    *,
    mass_gev: float,
    truth_model: str,
    truth_index: int,
    truth_ctau_m: float,
    truth_probabilities: np.ndarray,
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
    event_counts: np.ndarray,
    number_of_pseudoexperiments: int,
    seed: int,
    chunk_size: int,
    tie_tolerance: float,
    pseudoexperiment_start: int = 0,
    maximum_sampled_events: int | None = None,
    initial_rng_state: Mapping | None = None,
) -> pd.DataFrame:
    """Run one deterministic contiguous range of correlated-prefix PEs.

    ``pseudoexperiment_start`` advances the same stable truth stream used by a
    direct run and then evaluates only the requested number of subsequent
    pseudoexperiments.  This permits progressive extensions without repeating
    the expensive likelihood profiling of an already cached prefix.
    """
    event_counts = np.asarray(event_counts, dtype=int)
    truth = np.asarray(truth_probabilities, dtype=float)
    photon = _probability_matrix(photon_probabilities, label="photon")
    su2 = _probability_matrix(su2_probabilities, label="SU(2)_L")

    if event_counts.ndim != 1 or len(event_counts) == 0:
        raise ValueError("At least one event count is required.")
    if np.any(event_counts <= 0) or np.any(np.diff(event_counts) <= 0):
        raise ValueError("event_counts must be positive, unique, and increasing.")
    if number_of_pseudoexperiments <= 0 or chunk_size <= 0:
        raise ValueError("Pseudoexperiment count and chunk size must be positive.")
    if pseudoexperiment_start < 0:
        raise ValueError("pseudoexperiment_start cannot be negative.")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance cannot be negative.")
    if truth.ndim != 1 or len(truth) == 0:
        raise ValueError("Truth probabilities must be a non-empty vector.")
    if np.any(~np.isfinite(truth)) or np.any(truth < 0.0):
        raise ValueError("Truth probabilities must be finite and non-negative.")
    if not np.isclose(truth.sum(), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("Truth probabilities must sum to one.")
    if photon.shape[1] != len(truth):
        raise ValueError("Photon templates and truth use different energy bins.")
    if su2.shape[1] != len(truth):
        raise ValueError("SU(2)_L templates and truth use different energy bins.")

    maximum_events = (
        int(event_counts[-1])
        if maximum_sampled_events is None
        else int(maximum_sampled_events)
    )
    if maximum_events < int(event_counts[-1]):
        raise ValueError(
            "maximum_sampled_events cannot be below the largest event count."
        )
    event_indices = event_counts - 1
    number_of_counts = len(event_counts)

    log_photon = np.log(photon)
    log_su2 = np.log(su2)
    correct_sum = np.zeros(number_of_counts, dtype=float)
    photon_selected_sum = np.zeros(number_of_counts, dtype=float)
    su2_selected_sum = np.zeros(number_of_counts, dtype=float)
    tie_sum = np.zeros(number_of_counts, dtype=np.int64)
    statistic_sum = np.zeros(number_of_counts, dtype=float)
    statistic_squared_sum = np.zeros(number_of_counts, dtype=float)
    rng = stable_truth_rng(
        seed=seed,
        mass_gev=mass_gev,
        truth_model=truth_model,
        truth_index=truth_index,
    )
    if initial_rng_state is None:
        _discard_truth_stream_prefix(
            rng=rng,
            truth=truth,
            maximum_events=maximum_events,
            number_of_pseudoexperiments=int(pseudoexperiment_start),
            chunk_size=chunk_size,
        )
    else:
        try:
            rng.bit_generator.state = dict(initial_rng_state)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("initial_rng_state is invalid") from error

    processed = 0
    while processed < number_of_pseudoexperiments:
        current_chunk = min(chunk_size, number_of_pseudoexperiments - processed)
        sampled_bins = rng.choice(
            len(truth),
            size=(current_chunk, maximum_events),
            replace=True,
            p=truth,
        )
        photon_best = _profile_log_likelihoods(
            sampled_bins,
            log_photon,
            event_indices,
        )
        su2_best = _profile_log_likelihoods(sampled_bins, log_su2, event_indices)
        statistic = 2.0 * (su2_best - photon_best)

        ties = np.abs(statistic) <= tie_tolerance
        photon_selected = statistic < -tie_tolerance
        su2_selected = statistic > tie_tolerance
        if truth_model == "photon":
            correct_sum += photon_selected.sum(axis=0) + 0.5 * ties.sum(axis=0)
        elif truth_model == "su2":
            correct_sum += su2_selected.sum(axis=0) + 0.5 * ties.sum(axis=0)
        else:
            raise ValueError(f"Unknown truth model: {truth_model}")

        photon_selected_sum += photon_selected.sum(axis=0) + 0.5 * ties.sum(axis=0)
        su2_selected_sum += su2_selected.sum(axis=0) + 0.5 * ties.sum(axis=0)
        tie_sum += ties.sum(axis=0)
        statistic_sum += statistic.sum(axis=0)
        statistic_squared_sum += np.square(statistic).sum(axis=0)
        processed += current_chunk

    normalization = float(number_of_pseudoexperiments)
    mean_statistic = statistic_sum / normalization
    variance = statistic_squared_sum / normalization - np.square(mean_statistic)
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    result = pd.DataFrame(
        {
            "mass_GeV": float(mass_gev),
            "seed": int(seed),
            "truth_model": truth_model,
            "truth_lifetime_index": int(truth_index),
            "truth_ctau_m": float(truth_ctau_m),
            "number_of_events": event_counts,
            "number_of_pseudoexperiments": int(number_of_pseudoexperiments),
            "correct_fraction": correct_sum / normalization,
            "selected_photon_fraction": photon_selected_sum / normalization,
            "selected_su2_fraction": su2_selected_sum / normalization,
            "tie_fraction": tie_sum / normalization,
            "mean_profile_statistic_T": mean_statistic,
            "std_profile_statistic_T": standard_deviation,
        },
        columns=PROFILED_ACCURACY_COLUMNS,
    )
    result.attrs["rng_state_after"] = rng.bit_generator.state
    result.attrs["rng_state_resume_used"] = initial_rng_state is not None
    return result


def _half_integer_numerators(
    values: np.ndarray,
    number_of_pseudoexperiments: int,
    *,
    label: str,
) -> np.ndarray:
    raw = 2.0 * float(number_of_pseudoexperiments) * np.asarray(values, dtype=float)
    rounded = np.rint(raw).astype(np.int64)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1.0e-7):
        raise ValueError(f"Cannot reconstruct exact half-integer {label} counts.")
    return rounded


def combine_profiled_truth_tables(
    tables: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
) -> pd.DataFrame:
    """Combine disjoint deterministic PE ranges for one truth hypothesis.

    Classification numerators are reconstructed as exact half-integers before
    summation.  The profiled-statistic moments are combined from their first
    and second population moments and are therefore statistically equivalent,
    although their final floating-point roundoff need not be bitwise identical
    to a single uninterrupted reduction.
    """
    if not tables:
        raise ValueError("At least one profiled truth table is required.")

    ordered = [
        table.loc[:, PROFILED_ACCURACY_COLUMNS].reset_index(drop=True)
        for table in tables
    ]
    reference = ordered[0]
    identity_columns = (
        "mass_GeV",
        "seed",
        "truth_model",
        "truth_lifetime_index",
        "truth_ctau_m",
        "number_of_events",
    )
    for table in ordered[1:]:
        if len(table) != len(reference):
            raise ValueError("Profiled truth tables have different row counts.")
        for column in identity_columns:
            left = reference[column].to_numpy()
            right = table[column].to_numpy()
            if np.issubdtype(left.dtype, np.number):
                if not np.array_equal(left, right):
                    raise ValueError(
                        f"Profiled truth tables disagree in {column}."
                    )
            elif not np.array_equal(left.astype(str), right.astype(str)):
                raise ValueError(f"Profiled truth tables disagree in {column}.")

    counts = [
        int(table["number_of_pseudoexperiments"].iloc[0])
        for table in ordered
    ]
    for table, count in zip(ordered, counts):
        if count <= 0 or not np.all(
            table["number_of_pseudoexperiments"] == count
        ):
            raise ValueError("Each profiled truth table must have one positive PE count.")
    total = int(sum(counts))

    combined = reference.copy()
    for column in (
        "correct_fraction",
        "selected_photon_fraction",
        "selected_su2_fraction",
    ):
        numerator = sum(
            (
                _half_integer_numerators(
                    table[column].to_numpy(float),
                    count,
                    label=column,
                )
                for table, count in zip(ordered, counts)
            ),
            start=np.zeros(len(reference), dtype=np.int64),
        )
        combined[column] = numerator / (2.0 * float(total))

    tie_numerator = sum(
        (
            np.rint(
                table["tie_fraction"].to_numpy(float) * float(count)
            ).astype(np.int64)
            for table, count in zip(ordered, counts)
        ),
        start=np.zeros(len(reference), dtype=np.int64),
    )
    combined["tie_fraction"] = tie_numerator / float(total)

    statistic_sum = np.zeros(len(reference), dtype=float)
    statistic_squared_sum = np.zeros(len(reference), dtype=float)
    for table, count in zip(ordered, counts):
        mean = table["mean_profile_statistic_T"].to_numpy(float)
        standard_deviation = table["std_profile_statistic_T"].to_numpy(float)
        statistic_sum += mean * float(count)
        statistic_squared_sum += (
            np.square(standard_deviation) + np.square(mean)
        ) * float(count)
    mean = statistic_sum / float(total)
    variance = statistic_squared_sum / float(total) - np.square(mean)
    combined["mean_profile_statistic_T"] = mean
    combined["std_profile_statistic_T"] = np.sqrt(np.maximum(variance, 0.0))
    combined["number_of_pseudoexperiments"] = total
    return combined.loc[:, PROFILED_ACCURACY_COLUMNS]

def run_profiled_seed(
    *,
    mass_gev: float,
    photon_ctau_m: np.ndarray,
    photon_probabilities: np.ndarray,
    su2_ctau_m: np.ndarray,
    su2_probabilities: np.ndarray,
    event_counts: np.ndarray,
    number_of_pseudoexperiments: int,
    seed: int,
    chunk_size: int,
    tie_tolerance: float,
    truth_grid: str = "all",
    profile_grid: str = "all",
) -> pd.DataFrame:
    """Run one seed over both truth models and selected lifetime grids."""
    photon_ctau = np.asarray(photon_ctau_m, dtype=float)
    su2_ctau = np.asarray(su2_ctau_m, dtype=float)
    photon_all = _probability_matrix(photon_probabilities, label="photon")
    su2_all = _probability_matrix(su2_probabilities, label="SU(2)_L")
    if photon_ctau.ndim != 1 or len(photon_ctau) != len(photon_all):
        raise ValueError("Photon lifetime and probability grids do not match.")
    if su2_ctau.ndim != 1 or len(su2_ctau) != len(su2_all):
        raise ValueError("SU(2)_L lifetime and probability grids do not match.")
    if photon_all.shape[1] != su2_all.shape[1]:
        raise ValueError("Photon and SU(2)_L templates use different energy bins.")

    photon_truth_indices = lifetime_grid_indices(len(photon_ctau), truth_grid)
    su2_truth_indices = lifetime_grid_indices(len(su2_ctau), truth_grid)
    photon_profile = photon_all[
        lifetime_grid_indices(len(photon_ctau), profile_grid)
    ]
    su2_profile = su2_all[lifetime_grid_indices(len(su2_ctau), profile_grid)]

    frames: list[pd.DataFrame] = []
    for truth_model, lifetimes, templates, truth_indices in (
        ("photon", photon_ctau, photon_all, photon_truth_indices),
        ("su2", su2_ctau, su2_all, su2_truth_indices),
    ):
        for truth_index in truth_indices:
            frames.append(
                simulate_truth_template(
                    mass_gev=mass_gev,
                    truth_model=truth_model,
                    truth_index=int(truth_index),
                    truth_ctau_m=float(lifetimes[truth_index]),
                    truth_probabilities=templates[truth_index],
                    photon_probabilities=photon_profile,
                    su2_probabilities=su2_profile,
                    event_counts=event_counts,
                    number_of_pseudoexperiments=number_of_pseudoexperiments,
                    seed=seed,
                    chunk_size=chunk_size,
                    tie_tolerance=tie_tolerance,
                )
            )
    return pd.concat(frames, ignore_index=True)
