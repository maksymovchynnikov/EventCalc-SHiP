"""Pure utilities for observable lifetime intervals and logarithmic grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class LifetimeInterval:
    lower_m: float
    upper_m: float | None

    def __post_init__(self) -> None:
        if self.lower_m <= 0.0 or (self.upper_m is not None and self.upper_m <= self.lower_m):
            raise ValueError("a lifetime interval must be positive and ordered")

    @property
    def extends_beyond_scan(self) -> bool:
        return self.upper_m is None


def geometric_coarse_grid(lower_m: float, upper_m: float, factor: float) -> np.ndarray:
    if lower_m <= 0.0 or upper_m <= lower_m or factor <= 1.0:
        raise ValueError("require 0 < lower_m < upper_m and factor > 1")
    values = [float(lower_m)]
    while values[-1] < upper_m:
        candidate = min(values[-1] * factor, upper_m)
        if candidate <= values[-1]:
            raise RuntimeError("coarse lifetime grid stopped increasing")
        values.append(candidate)
    return np.asarray(values)


def threshold_brackets(ctau_m: np.ndarray, rates: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    ctau_m, rates = np.asarray(ctau_m, float), np.asarray(rates, float)
    if ctau_m.ndim != 1 or rates.shape != ctau_m.shape or len(ctau_m) < 2:
        raise ValueError("lifetimes and rates must be equally sized one-dimensional arrays")
    if np.any(ctau_m <= 0.0) or np.any(np.diff(ctau_m) <= 0.0) or not np.all(np.isfinite(rates)):
        raise ValueError("lifetimes must increase and rates must be finite")
    passing = rates >= threshold
    return [
        (float(left), float(right)) for left, right, a, b in
        zip(ctau_m[:-1], ctau_m[1:], passing[:-1], passing[1:]) if bool(a) != bool(b)
    ]


def logarithmic_bisection(
    evaluate_rate: Callable[[float], float], left_m: float, right_m: float,
    threshold: float, steps: int,
) -> float:
    if left_m <= 0.0 or right_m <= left_m or steps < 1:
        raise ValueError("invalid logarithmic-bisection arguments")
    state_left, state_right = evaluate_rate(left_m) >= threshold, evaluate_rate(right_m) >= threshold
    if state_left == state_right:
        raise ValueError("the supplied interval does not bracket the threshold")
    left, right = float(left_m), float(right_m)
    for _ in range(steps):
        middle = float(np.sqrt(left * right))
        if (evaluate_rate(middle) >= threshold) == state_left:
            left = middle
        else:
            right = middle
    return float(np.sqrt(left * right))


def intervals_from_crossings(
    lower_scan_m: float, crossings_m: Iterable[float], starts_above_threshold: bool,
) -> list[LifetimeInterval]:
    crossings = sorted(float(value) for value in crossings_m)
    if lower_scan_m <= 0.0 or any(value <= lower_scan_m for value in crossings):
        raise ValueError("crossings must lie above the positive scan lower bound")
    intervals: list[LifetimeInterval] = []
    state, start = starts_above_threshold, lower_scan_m if starts_above_threshold else None
    for crossing in crossings:
        if state:
            intervals.append(LifetimeInterval(float(start), crossing))
            start = None
        else:
            start = crossing
        state = not state
    if state:
        intervals.append(LifetimeInterval(float(start), None))
    return intervals


def intersect_intervals(
    first: Iterable[LifetimeInterval], second: Iterable[LifetimeInterval]
) -> list[LifetimeInterval]:
    intersections = []
    for left in first:
        for right in second:
            lower = max(left.lower_m, right.lower_m)
            upper_value = min(left.upper_m or np.inf, right.upper_m or np.inf)
            if lower < upper_value:
                intersections.append(LifetimeInterval(lower, None if np.isinf(upper_value) else upper_value))
    return intersections


def logarithmic_fraction(interval: LifetimeInterval, fraction: float) -> float:
    if interval.upper_m is None:
        raise ValueError("logarithmic interpolation requires a finite interval")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    log_lower, log_upper = np.log(interval.lower_m), np.log(interval.upper_m)
    return float(np.exp(log_lower + fraction * (log_upper - log_lower)))


def interior_lifetime_points(
    interval: LifetimeInterval, labelled_fractions: Iterable[tuple[str, float]]
) -> list[tuple[str, float, float]]:
    points = [(label, fraction, logarithmic_fraction(interval, fraction)) for label, fraction in labelled_fractions]
    values = np.asarray([point[2] for point in points])
    if interval.upper_m is None or np.any(values <= interval.lower_m) or np.any(values >= interval.upper_m):
        raise ValueError("all requested points must be strictly inside a finite interval")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("lifetime fractions must be strictly ordered")
    return points


def lifetime_point_records(
    intervals: Iterable[Mapping[str, float]], labelled_fractions: Iterable[tuple[str, float]],
) -> list[dict[str, float | str]]:
    """Tabulate labelled log-interior points from mass/interval records."""
    rows = []
    for record in sorted(intervals, key=lambda item: float(item["mass_GeV"])):
        interval = LifetimeInterval(float(record["ctau_lower_m"]), float(record["ctau_upper_m"]))
        for label, fraction, ctau_m in interior_lifetime_points(interval, labelled_fractions):
            rows.append({
                "mass_GeV": float(record["mass_GeV"]), "lifetime_label": label,
                "log_interval_fraction": fraction, "ctau_m": ctau_m,
                "ctau_lower_m": interval.lower_m, "ctau_upper_m": interval.upper_m,
            })
    return rows


def dense_log_grid(interval: LifetimeInterval, number_of_points: int) -> np.ndarray:
    if interval.upper_m is None or number_of_points < 2:
        raise ValueError("a finite interval and at least two points are required")
    return np.geomspace(interval.lower_m, interval.upper_m, number_of_points)
