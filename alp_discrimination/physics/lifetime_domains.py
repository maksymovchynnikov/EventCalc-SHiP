"""Pure geometry and interpolation utilities for the allowed lifetime domains.

The allowed domain at fixed ALP mass is the part of the geom-only
``N_events >= event_level`` sensitivity interval that is not covered by any
existing exclusion polygon.  All polygon slicing is performed in the same
log-mass/log-coupling coordinates used by the sensitivity plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib.path import Path as MatplotlibPath


@dataclass(frozen=True)
class Interval:
    """A finite ordered interval with explicit endpoint inclusivity."""

    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.lower)
            or not np.isfinite(self.upper)
            or self.lower <= 0.0
            or self.upper <= self.lower
        ):
            raise ValueError("interval endpoints must be finite, positive and ordered")


def _log_log_interpolate(
    mass_gev: float,
    masses_gev: np.ndarray,
    values: np.ndarray,
    *,
    quantity: str,
) -> float:
    """Interpolate a positive quantity linearly in log(mass)-log(value)."""
    mass_gev = float(mass_gev)
    masses = np.asarray(masses_gev, dtype=float)
    values = np.asarray(values, dtype=float)

    if mass_gev <= 0.0:
        raise ValueError("mass must be positive")
    if masses.ndim != 1 or values.shape != masses.shape or len(masses) < 2:
        raise ValueError(f"{quantity}: masses and values must be equally sized 1D arrays")
    if (
        np.any(~np.isfinite(masses))
        or np.any(~np.isfinite(values))
        or np.any(masses <= 0.0)
        or np.any(values <= 0.0)
    ):
        raise ValueError(f"{quantity}: interpolation inputs must be finite and positive")

    order = np.argsort(masses)
    masses = masses[order]
    values = values[order]
    if np.any(np.diff(masses) <= 0.0):
        raise ValueError(f"{quantity}: interpolation masses must be unique")

    exact = np.flatnonzero(np.isclose(masses, mass_gev, rtol=0.0, atol=1.0e-12))
    if len(exact):
        return float(values[int(exact[0])])

    if mass_gev < masses[0] or mass_gev > masses[-1]:
        raise ValueError(
            f"{quantity}: mass {mass_gev:g} GeV lies outside the resolved range "
            f"[{masses[0]:g}, {masses[-1]:g}] GeV"
        )

    result = np.interp(np.log(mass_gev), np.log(masses), np.log(values))
    return float(np.exp(result))


def sensitivity_coupling_interval(
    boundaries: pd.DataFrame,
    model: str,
    mass_gev: float,
    event_level: float = 2.3,
) -> Interval:
    """Interpolate the two resolved coupling branches at one fixed mass."""
    required = {
        "model",
        "mass_GeV",
        "event_level",
        "status",
        "lower_coupling_GeV_inv",
        "upper_coupling_GeV_inv",
    }
    missing = required - set(boundaries.columns)
    if missing:
        raise ValueError(f"boundary table is missing columns: {sorted(missing)}")

    selected = boundaries[
        (boundaries["model"] == model)
        & np.isclose(boundaries["event_level"].to_numpy(float), event_level)
        & (boundaries["status"] == "resolved")
    ].copy()
    if selected.empty:
        raise ValueError(f"no resolved N={event_level:g} boundary rows for {model}")

    selected = selected.sort_values("mass_GeV")
    masses = selected["mass_GeV"].to_numpy(float)
    lower = _log_log_interpolate(
        mass_gev,
        masses,
        selected["lower_coupling_GeV_inv"].to_numpy(float),
        quantity=f"{model} lower sensitivity branch",
    )
    upper = _log_log_interpolate(
        mass_gev,
        masses,
        selected["upper_coupling_GeV_inv"].to_numpy(float),
        quantity=f"{model} upper sensitivity branch",
    )
    return Interval(lower, upper, True, True)


def unit_coupling_ctau_at_mass(
    scan_data: pd.DataFrame,
    model: str,
    mass_gev: float,
) -> float:
    """Interpolate ``c tau`` at unit coupling from the saved event-density scan."""
    required = {"model", "mass_GeV", "unit_coupling_ctau_m"}
    missing = required - set(scan_data.columns)
    if missing:
        raise ValueError(f"event-density scan is missing columns: {sorted(missing)}")

    selected = scan_data[scan_data["model"] == model]
    if selected.empty:
        raise ValueError(f"event-density scan has no rows for {model}")

    rows: list[tuple[float, float]] = []
    for scanned_mass, group in selected.groupby("mass_GeV", sort=True):
        values = group["unit_coupling_ctau_m"].to_numpy(float)
        if (
            len(values) == 0
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or not np.allclose(values, values[0], rtol=1.0e-10, atol=0.0)
        ):
            raise ValueError(
                f"unit-coupling lifetime is not unique at {model}, m={scanned_mass:g} GeV"
            )
        rows.append((float(scanned_mass), float(values[0])))

    masses, lifetimes = np.asarray(rows, dtype=float).T
    return _log_log_interpolate(
        mass_gev,
        masses,
        lifetimes,
        quantity=f"{model} unit-coupling lifetime",
    )


def _log_polygon(polygon: np.ndarray) -> np.ndarray:
    """Return a closed polygon in log10(mass)-log10(coupling) coordinates."""
    data = np.asarray(polygon, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or len(data) < 3:
        raise ValueError("an exclusion polygon must have shape (N, >=2), N >= 3")
    data = data[:, :2]
    if (
        np.any(~np.isfinite(data))
        or np.any(data[:, 0] < 0.0)
        or np.any(data[:, 1] <= 0.0)
    ):
        raise ValueError("polygon masses must be non-negative and couplings positive")

    positive_masses = data[data[:, 0] > 0.0, 0]
    if len(positive_masses) == 0:
        raise ValueError("polygon must contain at least one positive mass")
    mass_floor = float(np.min(positive_masses) * 1.0e-12)
    safe_mass = np.where(data[:, 0] > 0.0, data[:, 0], mass_floor)
    result = np.column_stack((np.log10(safe_mass), np.log10(data[:, 1])))
    if not np.allclose(result[0], result[-1], rtol=0.0, atol=1.0e-14):
        result = np.vstack((result, result[0]))
    return result


def polygon_vertical_slice_intervals(
    polygon: np.ndarray,
    mass_gev: float,
) -> list[Interval]:
    """Intersect one plotted exclusion polygon with a fixed-mass vertical line.

    Edges are interpreted in log-log coordinates, matching how Matplotlib draws
    the polygon on logarithmic axes.  Returned endpoints are exclusion
    boundaries and are therefore inclusive.
    """
    if mass_gev <= 0.0:
        raise ValueError("mass must be positive")

    vertices = _log_polygon(polygon)
    x_query = float(np.log10(mass_gev))
    if x_query < np.min(vertices[:, 0]) or x_query > np.max(vertices[:, 0]):
        return []

    intersections: list[float] = []
    for (x1, y1), (x2, y2) in zip(vertices[:-1], vertices[1:]):
        tolerance = 1.0e-13 * max(1.0, abs(x_query), abs(x1), abs(x2))
        if abs(x2 - x1) <= tolerance:
            if abs(x_query - x1) <= tolerance:
                intersections.extend((float(y1), float(y2)))
            continue
        if min(x1, x2) - tolerance <= x_query <= max(x1, x2) + tolerance:
            fraction = (x_query - x1) / (x2 - x1)
            if -tolerance <= fraction <= 1.0 + tolerance:
                intersections.append(float(y1 + fraction * (y2 - y1)))

    if len(intersections) < 2:
        return []

    intersections.sort()
    unique: list[float] = []
    for value in intersections:
        if not unique or abs(value - unique[-1]) > 1.0e-10 * max(1.0, abs(value)):
            unique.append(value)
    if len(unique) < 2:
        return []

    path = MatplotlibPath(vertices, closed=True)
    intervals: list[Interval] = []
    for lower_log, upper_log in zip(unique[:-1], unique[1:]):
        if upper_log <= lower_log:
            continue
        midpoint = 0.5 * (lower_log + upper_log)
        if path.contains_point((x_query, midpoint)):
            intervals.append(
                Interval(float(10.0**lower_log), float(10.0**upper_log), True, True)
            )
    return merge_intervals(intervals)


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Return the ordered union of overlapping or touching intervals."""
    ordered = sorted(intervals, key=lambda item: (item.lower, item.upper))
    if not ordered:
        return []

    merged = [ordered[0]]
    for interval in ordered[1:]:
        current = merged[-1]
        tolerance = 1.0e-12 * max(1.0, current.upper, interval.lower)
        if interval.lower <= current.upper + tolerance:
            if interval.upper > current.upper:
                merged[-1] = Interval(
                    current.lower,
                    interval.upper,
                    current.lower_inclusive,
                    interval.upper_inclusive,
                )
            elif np.isclose(interval.upper, current.upper, rtol=1.0e-12, atol=0.0):
                merged[-1] = Interval(
                    current.lower,
                    current.upper,
                    current.lower_inclusive,
                    current.upper_inclusive or interval.upper_inclusive,
                )
        else:
            merged.append(interval)
    return merged


def clip_intervals(intervals: Iterable[Interval], domain: Interval) -> list[Interval]:
    """Clip intervals to a finite domain."""
    clipped: list[Interval] = []
    for interval in intervals:
        lower = max(interval.lower, domain.lower)
        upper = min(interval.upper, domain.upper)
        if lower >= upper:
            continue
        lower_inclusive = (
            interval.lower_inclusive if lower == interval.lower else domain.lower_inclusive
        )
        upper_inclusive = (
            interval.upper_inclusive if upper == interval.upper else domain.upper_inclusive
        )
        clipped.append(Interval(lower, upper, lower_inclusive, upper_inclusive))
    return merge_intervals(clipped)


def subtract_intervals(domain: Interval, excluded: Iterable[Interval]) -> list[Interval]:
    """Subtract closed exclusion intervals while retaining open boundary flags."""
    excluded_clipped = clip_intervals(excluded, domain)
    allowed: list[Interval] = []
    current_lower = domain.lower
    current_lower_inclusive = domain.lower_inclusive

    for interval in excluded_clipped:
        if interval.lower > current_lower:
            allowed.append(
                Interval(
                    current_lower,
                    interval.lower,
                    current_lower_inclusive,
                    not interval.lower_inclusive,
                )
            )
        if interval.upper > current_lower:
            current_lower = interval.upper
            current_lower_inclusive = not interval.upper_inclusive

    if current_lower < domain.upper:
        allowed.append(
            Interval(
                current_lower,
                domain.upper,
                current_lower_inclusive,
                domain.upper_inclusive,
            )
        )
    return allowed


def allowed_coupling_intervals(
    sensitivity: Interval,
    polygons: Iterable[np.ndarray],
    mass_gev: float,
) -> tuple[list[Interval], list[Interval]]:
    """Return allowed intervals and the clipped union of excluded intervals."""
    excluded = merge_intervals(
        interval
        for polygon in polygons
        for interval in polygon_vertical_slice_intervals(polygon, mass_gev)
    )
    excluded = clip_intervals(excluded, sensitivity)
    return subtract_intervals(sensitivity, excluded), excluded


def coupling_interval_to_ctau(interval: Interval, unit_coupling_ctau_m: float) -> Interval:
    """Map ``g`` to ``c tau = c tau(g=1) / g^2``, reversing endpoint order."""
    unit_ctau = float(unit_coupling_ctau_m)
    if not np.isfinite(unit_ctau) or unit_ctau <= 0.0:
        raise ValueError("unit-coupling lifetime must be finite and positive")
    return Interval(
        unit_ctau / interval.upper**2,
        unit_ctau / interval.lower**2,
        interval.upper_inclusive,
        interval.lower_inclusive,
    )

# -----------------------------------------------------------------------------
# Loading and sampling saved allowed lifetime domains
# -----------------------------------------------------------------------------

from pathlib import Path


WEEK8_MODEL_NAMES = (
    "ALP-photon-combined",
    "ALP-SU2L",
)

_ALLOWED_DOMAIN_REQUIRED_COLUMNS = {
    "model",
    "mass_GeV",
    "event_level",
    "interval_index",
    "coupling_min_GeV_inv",
    "coupling_max_GeV_inv",
    "unit_coupling_ctau_m",
    "ctau_min_m",
    "ctau_max_m",
}


def load_allowed_ctau_domains(
    path: str | Path,
    *,
    expected_event_level: float = 2.3,
) -> pd.DataFrame:
    """Load and validate ``allowed_ctau_domains.csv``.

    Each row represents one connected lifetime interval.  Disconnected
    intervals are deliberately retained as separate rows and must never be
    replaced by their convex hull.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Allowed lifetime-domain table not found: {path}\n"
            "Build the allowed lifetime-domain table before loading it."
        )

    data = pd.read_csv(path)

    missing = _ALLOWED_DOMAIN_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError(f"{path} contains no allowed lifetime intervals.")

    data = data.copy()

    numeric_columns = (
        "mass_GeV",
        "event_level",
        "interval_index",
        "coupling_min_GeV_inv",
        "coupling_max_GeV_inv",
        "unit_coupling_ctau_m",
        "ctau_min_m",
        "ctau_max_m",
    )

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    physical_columns = (
        "mass_GeV",
        "event_level",
        "coupling_min_GeV_inv",
        "coupling_max_GeV_inv",
        "unit_coupling_ctau_m",
        "ctau_min_m",
        "ctau_max_m",
    )

    physical_values = data[list(physical_columns)].to_numpy(dtype=float)

    if np.any(~np.isfinite(physical_values)):
        raise ValueError("The allowed lifetime-domain table contains non-finite values.")

    if np.any(physical_values <= 0.0):
        raise ValueError("All physical lifetime-domain values must be positive.")

    interval_indices = data["interval_index"].to_numpy(dtype=float)

    if np.any(~np.isfinite(interval_indices)):
        raise ValueError("Interval indices must be finite.")

    rounded_indices = np.rint(interval_indices)

    if not np.allclose(
        interval_indices,
        rounded_indices,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Interval indices must be integers.")

    data["interval_index"] = rounded_indices.astype(int)

    if np.any(data["interval_index"] < 0):
        raise ValueError("Interval indices cannot be negative.")

    unknown_models = sorted(
        set(data["model"].astype(str)) - set(WEEK8_MODEL_NAMES)
    )
    if unknown_models:
        raise ValueError(
            "Unknown model identifiers in the allowed lifetime domains: "
            + ", ".join(unknown_models)
        )

    if not np.allclose(
        data["event_level"].to_numpy(dtype=float),
        float(expected_event_level),
        rtol=0.0,
        atol=1.0e-12,
    ):
        found = sorted(
            set(data["event_level"].to_numpy(dtype=float))
        )
        raise ValueError(
            f"Expected event_level={expected_event_level:g}, "
            f"but found {found}."
        )

    if np.any(
        data["coupling_min_GeV_inv"].to_numpy(dtype=float)
        >= data["coupling_max_GeV_inv"].to_numpy(dtype=float)
    ):
        raise ValueError("At least one coupling interval is not ordered.")

    if np.any(
        data["ctau_min_m"].to_numpy(dtype=float)
        >= data["ctau_max_m"].to_numpy(dtype=float)
    ):
        raise ValueError("At least one lifetime interval is not ordered.")

    expected_ctau_min = (
        data["unit_coupling_ctau_m"].to_numpy(dtype=float)
        / data["coupling_max_GeV_inv"].to_numpy(dtype=float) ** 2
    )
    expected_ctau_max = (
        data["unit_coupling_ctau_m"].to_numpy(dtype=float)
        / data["coupling_min_GeV_inv"].to_numpy(dtype=float) ** 2
    )

    np.testing.assert_allclose(
        data["ctau_min_m"].to_numpy(dtype=float),
        expected_ctau_min,
        rtol=2.0e-12,
        atol=0.0,
        err_msg=(
            "A lower lifetime endpoint is inconsistent with "
            "c*tau = c*tau(g=1)/g^2."
        ),
    )

    np.testing.assert_allclose(
        data["ctau_max_m"].to_numpy(dtype=float),
        expected_ctau_max,
        rtol=2.0e-12,
        atol=0.0,
        err_msg=(
            "An upper lifetime endpoint is inconsistent with "
            "c*tau = c*tau(g=1)/g^2."
        ),
    )

    duplicated = data.duplicated(
        ["model", "mass_GeV", "interval_index"],
        keep=False,
    )
    if duplicated.any():
        duplicates = data.loc[
            duplicated,
            ["model", "mass_GeV", "interval_index"],
        ]
        raise ValueError(
            "Duplicate model-mass-interval rows:\n"
            + duplicates.to_string(index=False)
        )

    for (model, mass_gev), group in data.groupby(
        ["model", "mass_GeV"],
        sort=True,
    ):
        indices = sorted(group["interval_index"].astype(int).tolist())
        expected_indices = list(range(len(indices)))

        if indices != expected_indices:
            raise ValueError(
                f"{model}, m_a={mass_gev:g} GeV has interval indices "
                f"{indices}; expected {expected_indices}."
            )

        ordered = group.sort_values("ctau_min_m")
        lower = ordered["ctau_min_m"].to_numpy(dtype=float)
        upper = ordered["ctau_max_m"].to_numpy(dtype=float)

        if len(ordered) > 1 and np.any(lower[1:] < upper[:-1]):
            raise ValueError(
                f"{model}, m_a={mass_gev:g} GeV contains overlapping "
                "allowed lifetime intervals."
            )

    return data.sort_values(
        ["mass_GeV", "model", "ctau_min_m", "interval_index"],
        ignore_index=True,
    )


def available_lifetime_domain_masses(domains: pd.DataFrame) -> list[float]:
    """Return masses with at least one allowed interval for both hypotheses."""
    model_masses = {
        model: {
            float(value)
            for value in domains.loc[
                domains["model"] == model,
                "mass_GeV",
            ]
        }
        for model in WEEK8_MODEL_NAMES
    }

    common = set.intersection(
        *(model_masses[model] for model in WEEK8_MODEL_NAMES)
    )

    return sorted(common)


def build_lifetime_grid(
    domains: pd.DataFrame,
    *,
    model: str,
    mass_gev: float,
    points_per_interval: int,
) -> pd.DataFrame:
    """Sample every connected allowed interval independently in log lifetime.

    The returned table retains the original ``interval_index`` for every
    template.  No points are ever inserted into excluded gaps.
    """
    if model not in WEEK8_MODEL_NAMES:
        raise ValueError(f"Unknown lifetime-domain model: {model}")

    if points_per_interval < 2:
        raise ValueError("At least two lifetime points per interval are required.")

    mass_values = domains["mass_GeV"].to_numpy(dtype=float)
    selected = domains[
        (domains["model"] == model)
        & np.isclose(
            mass_values,
            float(mass_gev),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No allowed lifetime interval for {model}, "
            f"m_a={mass_gev:g} GeV."
        )

    selected = selected.sort_values(
        ["ctau_min_m", "interval_index"],
        ignore_index=True,
    )

    rows: list[dict[str, float | int | str | bool]] = []

    for domain_row in selected.itertuples(index=False):
        interval_index = int(domain_row.interval_index)
        lower = float(domain_row.ctau_min_m)
        upper = float(domain_row.ctau_max_m)

        lifetimes = np.geomspace(
            lower,
            upper,
            points_per_interval,
        )

        for local_index, ctau_m in enumerate(lifetimes):
            rows.append(
                {
                    "model": model,
                    "mass_GeV": float(mass_gev),
                    "interval_index": interval_index,
                    "lifetime_index_within_interval": local_index,
                    "ctau_m": float(ctau_m),
                    "is_interval_endpoint": (
                        local_index == 0
                        or local_index == points_per_interval - 1
                    ),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["ctau_m", "interval_index"],
        ignore_index=True,
    )

    lifetimes = result["ctau_m"].to_numpy(dtype=float)

    if np.any(~np.isfinite(lifetimes)) or np.any(lifetimes <= 0.0):
        raise RuntimeError("Generated lifetimes must be finite and positive.")

    if np.any(np.diff(lifetimes) <= 0.0):
        raise RuntimeError(
            "Generated lifetime grid is not strictly increasing. "
            "Inspect overlapping or duplicate allowed intervals."
        )

    result.insert(
        3,
        "global_lifetime_index",
        np.arange(len(result), dtype=int),
    )

    return result