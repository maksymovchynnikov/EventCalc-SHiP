"""Pairwise distances for lifetime-template banks.

This module contains no EventCalc calls.  It deliberately preserves the
frozen-reference total-variation definition and the legacy long-form table layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DISTANCE_TABLE_COLUMNS = (
    "mass_GeV",
    "photon_lifetime_index",
    "photon_ctau_m",
    "photon_N_events",
    "su2_lifetime_index",
    "su2_ctau_m",
    "su2_N_events",
    "D_TV",
)


def _probability_matrix(probabilities: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{label} probabilities must be a non-empty matrix.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{label} probabilities must be finite and non-negative.")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Every {label} probability template must sum to one.")
    return values


def total_variation_matrix(
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
) -> np.ndarray:
    """Return ``D_TV`` for every photon/SU(2)_L lifetime pair.

    The first result axis follows the photon lifetime grid and the second
    follows the SU(2)_L grid.  The two grids need not have equal lengths.
    """
    photon = _probability_matrix(photon_probabilities, label="photon")
    su2 = _probability_matrix(su2_probabilities, label="SU(2)_L")
    if photon.shape[1] != su2.shape[1]:
        raise ValueError("Photon and SU(2)_L templates use different energy bins.")

    distances = 0.5 * np.sum(
        np.abs(photon[:, np.newaxis, :] - su2[np.newaxis, :, :]),
        axis=2,
    )
    if np.any(~np.isfinite(distances)):
        raise RuntimeError("The total-variation matrix contains non-finite values.")
    if np.any(distances < -1.0e-14) or np.any(distances > 1.0 + 1.0e-14):
        raise RuntimeError("A total-variation distance lies outside [0, 1].")
    return np.clip(distances, 0.0, 1.0)


def build_distance_table(
    *,
    mass_gev: float,
    photon_ctau_m: np.ndarray,
    photon_expected_events: np.ndarray,
    su2_ctau_m: np.ndarray,
    su2_expected_events: np.ndarray,
    distances: np.ndarray,
) -> pd.DataFrame:
    """Create the legacy long-form row for every lifetime pair."""
    photon_ctau = np.asarray(photon_ctau_m, dtype=float)
    photon_events = np.asarray(photon_expected_events, dtype=float)
    su2_ctau = np.asarray(su2_ctau_m, dtype=float)
    su2_events = np.asarray(su2_expected_events, dtype=float)
    values = np.asarray(distances, dtype=float)

    if photon_ctau.ndim != 1 or photon_events.shape != photon_ctau.shape:
        raise ValueError("Photon lifetimes and expected events must have matching shapes.")
    if su2_ctau.ndim != 1 or su2_events.shape != su2_ctau.shape:
        raise ValueError("SU(2)_L lifetimes and expected events must have matching shapes.")
    if values.shape != (len(photon_ctau), len(su2_ctau)):
        raise ValueError("Distance matrix shape does not match the lifetime grids.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Distance matrix must be finite.")

    photon_indices, su2_indices = np.indices(values.shape)
    table = pd.DataFrame(
        {
            "mass_GeV": float(mass_gev),
            "photon_lifetime_index": photon_indices.ravel(),
            "photon_ctau_m": photon_ctau[photon_indices.ravel()],
            "photon_N_events": photon_events[photon_indices.ravel()],
            "su2_lifetime_index": su2_indices.ravel(),
            "su2_ctau_m": su2_ctau[su2_indices.ravel()],
            "su2_N_events": su2_events[su2_indices.ravel()],
            "D_TV": values.ravel(),
        },
        columns=DISTANCE_TABLE_COLUMNS,
    )
    return table.sort_values(
        ["photon_lifetime_index", "su2_lifetime_index"],
        ignore_index=True,
    )


def summarize_distance_matrix(
    *,
    mass_gev: float,
    energy_edges_gev: np.ndarray,
    photon_ctau_m: np.ndarray,
    photon_expected_events: np.ndarray,
    su2_ctau_m: np.ndarray,
    su2_expected_events: np.ndarray,
    distances: np.ndarray,
) -> dict:
    """Return the legacy summary of the global minimum and maximum pairs."""
    photon_ctau = np.asarray(photon_ctau_m, dtype=float)
    photon_events = np.asarray(photon_expected_events, dtype=float)
    su2_ctau = np.asarray(su2_ctau_m, dtype=float)
    su2_events = np.asarray(su2_expected_events, dtype=float)
    values = np.asarray(distances, dtype=float)
    if values.shape != (len(photon_ctau), len(su2_ctau)):
        raise ValueError("Distance matrix shape does not match lifetime grids.")
    minimum = tuple(map(int, np.unravel_index(np.argmin(values), values.shape)))
    maximum = tuple(map(int, np.unravel_index(np.argmax(values), values.shape)))
    photon_min, su2_min = minimum
    photon_max, su2_max = maximum
    return {
        "mass_GeV": float(mass_gev),
        "number_of_energy_bins": len(np.asarray(energy_edges_gev)) - 1,
        "number_of_photon_lifetimes": len(photon_ctau),
        "number_of_su2_lifetimes": len(su2_ctau),
        "minimum_D_TV": float(values[photon_min, su2_min]),
        "minimum_photon_lifetime_index": photon_min,
        "minimum_photon_ctau_m": float(photon_ctau[photon_min]),
        "minimum_photon_N_events": float(photon_events[photon_min]),
        "minimum_su2_lifetime_index": su2_min,
        "minimum_su2_ctau_m": float(su2_ctau[su2_min]),
        "minimum_su2_N_events": float(su2_events[su2_min]),
        "maximum_D_TV": float(values[photon_max, su2_max]),
        "maximum_photon_lifetime_index": photon_max,
        "maximum_photon_ctau_m": float(photon_ctau[photon_max]),
        "maximum_su2_lifetime_index": su2_max,
        "maximum_su2_ctau_m": float(su2_ctau[su2_max]),
    }


def minimum_pair_bin_table(
    *,
    mass_gev: float,
    energy_edges_gev: np.ndarray,
    photon_ctau_m: float,
    photon_probabilities: np.ndarray,
    su2_ctau_m: float,
    su2_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Tabulate bin contributions for the least-distinguishable pair."""
    edges = np.asarray(energy_edges_gev, dtype=float)
    photon = np.asarray(photon_probabilities, dtype=float)
    su2 = np.asarray(su2_probabilities, dtype=float)
    if photon.ndim != 1 or su2.shape != photon.shape or len(edges) != len(photon) + 1:
        raise ValueError("Minimum-pair probabilities and energy edges do not match.")
    absolute_difference = np.abs(photon - su2)
    return pd.DataFrame(
        {
            "mass_GeV": float(mass_gev),
            "photon_ctau_m": float(photon_ctau_m),
            "su2_ctau_m": float(su2_ctau_m),
            "bin_index": np.arange(len(photon), dtype=int),
            "energy_low_GeV": edges[:-1],
            "energy_high_GeV": edges[1:],
            "photon_probability": photon,
            "su2_probability": su2,
            "absolute_probability_difference": absolute_difference,
            "D_TV_bin_contribution": 0.5 * absolute_difference,
        }
    )
