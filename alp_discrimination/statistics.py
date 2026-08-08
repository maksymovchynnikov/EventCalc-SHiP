"""EventCalc-independent distances and direct shape-only pseudoexperiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_probabilities(probabilities: np.ndarray, *, strictly_positive: bool = False) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite non-empty one-dimensional array")
    if np.any(values <= 0.0) if strictly_positive else np.any(values < 0.0):
        raise ValueError("probabilities have invalid values")
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    return values


def total_variation_distance(first: np.ndarray, second: np.ndarray) -> float:
    first, second = validate_probabilities(first), validate_probabilities(second)
    if first.shape != second.shape:
        raise ValueError("probability arrays must have identical shapes")
    return 0.5 * float(np.abs(first - second).sum())


def maximum_cdf_distance(first: np.ndarray, second: np.ndarray) -> float:
    first, second = validate_probabilities(first), validate_probabilities(second)
    if first.shape != second.shape:
        raise ValueError("probability arrays must have identical shapes")
    return float(np.max(np.abs(np.cumsum(first) - np.cumsum(second))))


def kl_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = validate_probabilities(first)
    second = validate_probabilities(second, strictly_positive=True)
    if first.shape != second.shape:
        raise ValueError("probability arrays must have identical shapes")
    positive = first > 0.0
    return float(np.sum(first[positive] * np.log(first[positive] / second[positive])))


def same_lifetime_log_likelihood_ratio(
    observed_bins: np.ndarray, photon_probabilities: np.ndarray, su2_probabilities: np.ndarray,
) -> float:
    """Return sum log(p_SU2/p_photon); positive selects SU(2)L."""
    photon = validate_probabilities(photon_probabilities, strictly_positive=True)
    su2 = validate_probabilities(su2_probabilities, strictly_positive=True)
    if photon.shape != su2.shape:
        raise ValueError("templates must have identical shapes")
    indices = np.asarray(observed_bins)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("observed bins must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= len(photon)):
        raise ValueError("observed bin index outside template")
    return float(np.log(su2 / photon)[indices].sum())


def conditional_classification_accuracy(llr: np.ndarray, true_hypothesis: str) -> float:
    values = np.asarray(llr, dtype=float)
    if not np.all(np.isfinite(values)) or true_hypothesis not in {"photon", "su2"}:
        raise ValueError("invalid likelihood ratios or hypothesis")
    correct = values < 0.0 if true_hypothesis == "photon" else values > 0.0
    return float(np.mean(correct) + 0.5 * np.mean(values == 0.0))


@dataclass(frozen=True)
class ClassificationTable:
    number_of_events: np.ndarray
    photon_correct_fraction: np.ndarray
    su2_correct_fraction: np.ndarray
    balanced_accuracy: np.ndarray
    worst_case_correct_fraction: np.ndarray
    photon_llr_median: np.ndarray
    su2_llr_median: np.ndarray

    def records(self) -> list[dict]:
        names = self.__dataclass_fields__
        return [{name: getattr(self, name)[index].item() for name in names} for index in range(len(self.number_of_events))]


def simulate_shape_discrimination(
    photon_probabilities: np.ndarray, su2_probabilities: np.ndarray, maximum_events: int,
    number_of_pseudoexperiments: int, seed: int,
) -> ClassificationTable:
    photon = validate_probabilities(photon_probabilities, strictly_positive=True)
    su2 = validate_probabilities(su2_probabilities, strictly_positive=True)
    if photon.shape != su2.shape or maximum_events < 1 or number_of_pseudoexperiments < 1:
        raise ValueError("templates must match and simulation sizes must be positive")
    log_ratio = np.log(su2 / photon)
    photon_bins = np.random.default_rng(seed).choice(
        len(photon), size=(number_of_pseudoexperiments, maximum_events), p=photon
    )
    su2_bins = np.random.default_rng(seed + 1).choice(
        len(su2), size=(number_of_pseudoexperiments, maximum_events), p=su2
    )
    photon_llr = np.cumsum(log_ratio[photon_bins], axis=1)
    su2_llr = np.cumsum(log_ratio[su2_bins], axis=1)
    photon_correct = np.asarray([
        conditional_classification_accuracy(photon_llr[:, i], "photon") for i in range(maximum_events)
    ])
    su2_correct = np.asarray([
        conditional_classification_accuracy(su2_llr[:, i], "su2") for i in range(maximum_events)
    ])
    return ClassificationTable(
        number_of_events=np.arange(1, maximum_events + 1),
        photon_correct_fraction=photon_correct, su2_correct_fraction=su2_correct,
        balanced_accuracy=0.5 * (photon_correct + su2_correct),
        worst_case_correct_fraction=np.minimum(photon_correct, su2_correct),
        photon_llr_median=np.median(photon_llr, axis=0), su2_llr_median=np.median(su2_llr, axis=0),
    )


def minimum_events_for_accuracy(
    event_counts: np.ndarray, accuracies: np.ndarray, target_accuracy: float,
) -> int | None:
    events, values = np.asarray(event_counts, int), np.asarray(accuracies, float)
    if events.ndim != 1 or values.shape != events.shape or np.any(np.diff(events) <= 0.0):
        raise ValueError("event counts and accuracies must be ordered one-dimensional arrays")
    passing = np.flatnonzero(values >= target_accuracy)
    return None if not len(passing) else int(events[passing[0]])


def minimum_persistent_events(
    event_counts: np.ndarray, accuracies: np.ndarray, target_accuracy: float,
) -> int | None:
    events, values = np.asarray(event_counts, int), np.asarray(accuracies, float)
    if events.ndim != 1 or values.shape != events.shape or not np.all(np.isfinite(values)):
        raise ValueError("event counts and finite accuracies must have matching shapes")
    passing_from_here = np.logical_and.accumulate((values >= target_accuracy)[::-1])[::-1]
    indices = np.flatnonzero(passing_from_here)
    return None if not len(indices) else int(events[indices[0]])


def finite_threshold_summary(values: np.ndarray) -> dict[str, float | bool]:
    """Summarize possibly missing validation thresholds without hiding failures."""
    numeric = np.asarray(values, float)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return {"all_reached": False, "minimum": np.nan, "median": np.nan,
                "maximum": np.nan, "spread": np.nan}
    minimum, maximum = float(finite.min()), float(finite.max())
    return {
        "all_reached": len(finite) == len(numeric), "minimum": minimum,
        "median": float(np.median(finite)), "maximum": maximum, "spread": maximum - minimum,
    }
