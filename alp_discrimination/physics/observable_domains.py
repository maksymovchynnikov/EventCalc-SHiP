"""Observable lifetime domains with both approved endpoint conventions.

The production template domain is reconstructed from saved scan rows by
log--log interpolation at the event-rate threshold.  The fixed-step scan
bisection midpoint is retained separately as a diagnostic; it is never used
to build the frozen frozen-reference template grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_SCAN_COLUMNS = {
    "model",
    "mass_GeV",
    "ctau_m",
    "N_events",
    "passes_event_cut",
}


@dataclass(frozen=True)
class ObservableLifetimeDomain:
    """One finite observable interval and its diagnostic scan endpoints."""

    lower_m: float
    upper_m: float
    lower_is_scan_boundary: bool
    upper_is_scan_boundary: bool
    bisection_lower_m: float
    bisection_upper_m: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.lower_m, self.upper_m, self.bisection_lower_m, self.bisection_upper_m],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Observable lifetime endpoints must be finite and positive.")
        if self.upper_m <= self.lower_m:
            raise ValueError("Observable lifetime endpoints must be ordered.")
        if self.bisection_upper_m <= self.bisection_lower_m:
            raise ValueError("Diagnostic bisection endpoints must be ordered.")


def _boolean_values(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    text = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    if invalid := sorted(set(text) - allowed):
        raise ValueError(f"Column {column!r} contains invalid Boolean values: {invalid}")
    return text.isin({"true", "1"})


def load_lifetime_scan(path: Path) -> pd.DataFrame:
    """Load and strictly validate an ECAL-aware lifetime scan."""
    data = pd.read_csv(path)
    if missing := REQUIRED_SCAN_COLUMNS - set(data.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    data = data.copy()
    for column in ("mass_GeV", "ctau_m", "N_events"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    numeric = data[["mass_GeV", "ctau_m", "N_events"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("Lifetime scan contains non-finite numerical values.")
    if np.any(data["mass_GeV"] <= 0.0) or np.any(data["ctau_m"] <= 0.0):
        raise ValueError("Masses and lifetimes must be positive.")
    if np.any(data["N_events"] < 0.0):
        raise ValueError("Expected event rates cannot be negative.")
    data["passes_event_cut"] = _boolean_values(data["passes_event_cut"], "passes_event_cut")
    return data.sort_values(["mass_GeV", "model", "ctau_m"], ignore_index=True)


def logarithmic_threshold_crossing(
    ctau_left_m: float,
    rate_left: float,
    ctau_right_m: float,
    rate_right: float,
    threshold: float,
) -> float:
    """Reproduce the frozen bank builder's local log--log interpolation."""
    values = np.asarray(
        [ctau_left_m, rate_left, ctau_right_m, rate_right, threshold],
        dtype=float,
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Logarithmic threshold interpolation requires positive values.")
    log_left = np.log(rate_left)
    log_right = np.log(rate_right)
    if np.isclose(log_left, log_right, rtol=0.0, atol=1.0e-15):
        return float(np.sqrt(ctau_left_m * ctau_right_m))
    fraction = (np.log(threshold) - log_left) / (log_right - log_left)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(
        np.exp(
            np.log(ctau_left_m)
            + fraction * (np.log(ctau_right_m) - np.log(ctau_left_m))
        )
    )


def domains_from_scan_group(
    group: pd.DataFrame,
    *,
    threshold: float,
    allow_truncated: bool = False,
) -> list[ObservableLifetimeDomain]:
    """Find observable intervals while retaining both endpoint definitions."""
    if threshold <= 0.0:
        raise ValueError("Event threshold must be positive.")
    if group.empty:
        return []
    ordered = (
        group.sort_values("ctau_m")
        .drop_duplicates(subset="ctau_m", keep="last")
        .reset_index(drop=True)
    )
    ctaus = ordered["ctau_m"].to_numpy(dtype=float)
    rates = ordered["N_events"].to_numpy(dtype=float)
    if len(ctaus) < 2 or np.any(np.diff(ctaus) <= 0.0):
        raise ValueError("Each model/mass scan needs at least two increasing lifetimes.")
    passing = rates >= threshold

    intervals: list[ObservableLifetimeDomain] = []
    start = float(ctaus[0]) if passing[0] else None
    diagnostic_start = start
    lower_is_boundary = bool(passing[0])
    for index in range(len(ctaus) - 1):
        if bool(passing[index]) == bool(passing[index + 1]):
            continue
        interpolated = logarithmic_threshold_crossing(
            ctaus[index], rates[index], ctaus[index + 1], rates[index + 1], threshold
        )
        diagnostic = float(np.sqrt(ctaus[index] * ctaus[index + 1]))
        if not passing[index] and passing[index + 1]:
            start = interpolated
            diagnostic_start = diagnostic
            lower_is_boundary = False
            continue
        if start is None or diagnostic_start is None:
            raise RuntimeError("Internal error while closing an observable interval.")
        intervals.append(
            ObservableLifetimeDomain(
                lower_m=float(start),
                upper_m=interpolated,
                lower_is_scan_boundary=lower_is_boundary,
                upper_is_scan_boundary=False,
                bisection_lower_m=float(diagnostic_start),
                bisection_upper_m=diagnostic,
            )
        )
        start = None
        diagnostic_start = None

    if passing[-1]:
        if not allow_truncated:
            raise RuntimeError(
                "Observable interval reaches the largest scanned lifetime; "
                "production requires a finite upper crossing."
            )
        if start is None or diagnostic_start is None:
            raise RuntimeError("Internal error while extending an observable interval.")
        intervals.append(
            ObservableLifetimeDomain(
                lower_m=float(start),
                upper_m=float(ctaus[-1]),
                lower_is_scan_boundary=lower_is_boundary,
                upper_is_scan_boundary=True,
                bisection_lower_m=float(diagnostic_start),
                bisection_upper_m=float(ctaus[-1]),
            )
        )
    return intervals


def collect_observable_domains(
    scan: pd.DataFrame,
    *,
    threshold: float,
    allow_truncated: bool = False,
) -> dict[tuple[str, float], ObservableLifetimeDomain]:
    """Return exactly one model-specific domain per observable model/mass."""
    domains: dict[tuple[str, float], ObservableLifetimeDomain] = {}
    for (model, mass_gev), group in scan.groupby(["model", "mass_GeV"], sort=True):
        intervals = domains_from_scan_group(
            group,
            threshold=threshold,
            allow_truncated=allow_truncated,
        )
        if len(intervals) > 1:
            raise RuntimeError(
                f"{model}, m_a={mass_gev:g} GeV has multiple observable intervals."
            )
        if intervals:
            domains[(str(model), float(mass_gev))] = intervals[0]
    return domains


def padded_lifetime_grid(
    domain: ObservableLifetimeDomain,
    number_of_points: int,
    log_padding_fraction: float,
) -> np.ndarray:
    """Build the approved grid, shifting only interpolated endpoints inward."""
    if number_of_points < 2:
        raise ValueError("At least two lifetime templates are required.")
    if not 0.0 <= log_padding_fraction < 0.5:
        raise ValueError("Logarithmic endpoint padding must lie in [0, 0.5).")
    log_lower = np.log(domain.lower_m)
    log_upper = np.log(domain.upper_m)
    span = log_upper - log_lower
    if not domain.lower_is_scan_boundary:
        log_lower += log_padding_fraction * span
    if not domain.upper_is_scan_boundary:
        log_upper -= log_padding_fraction * span
    if log_upper <= log_lower:
        raise RuntimeError("Lifetime domain vanishes after endpoint padding.")
    return np.geomspace(np.exp(log_lower), np.exp(log_upper), number_of_points)


def domain_table(
    domains: Mapping[tuple[str, float], ObservableLifetimeDomain],
    *,
    log_padding_fraction: float,
) -> pd.DataFrame:
    """Tabulate production and bisection-diagnostic endpoints explicitly."""
    rows = []
    for (model, mass_gev), domain in sorted(
        domains.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        rows.append(
            {
                "model": model,
                "mass_GeV": mass_gev,
                "template_domain_lower_m": domain.lower_m,
                "template_domain_upper_m": domain.upper_m,
                "bisection_diagnostic_lower_m": domain.bisection_lower_m,
                "bisection_diagnostic_upper_m": domain.bisection_upper_m,
                "lower_is_scan_boundary": domain.lower_is_scan_boundary,
                "upper_is_scan_boundary": domain.upper_is_scan_boundary,
                "template_log_endpoint_padding_fraction": log_padding_fraction,
            }
        )
    return pd.DataFrame(rows)
