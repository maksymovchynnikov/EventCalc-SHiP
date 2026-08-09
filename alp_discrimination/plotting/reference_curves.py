"""Pure loading and comparison of bundled SHiP photon sensitivity curves."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

MODEL_NAME = "ALP-photon-combined"
EVENT_LEVEL = 2.3
N_COMPARISON_POINTS = 600
ENDPOINT_FRACTION = 0.98
CONTAINMENT_TOLERANCE_DEX = 1.0e-3
REFERENCE_FILENAMES = {
    "epsilon_dec_1": (
        "Sensitivity_ALP-photon_at_SHiP-ECN3-"
        "epsilon-dec-1_Nev=2.3_Npot=6.e20.json"
    ),
    "geom_only": (
        "Sensitivity_ALP-photon_at_SHiP-ECN3-"
        "geom-only_Nev=2.3_Npot=6.e20.json"
    ),
}


@dataclass(frozen=True)
class SensitivityReference:
    name: str
    path: Path
    production_modes: tuple[str, ...]
    decay_description: str
    points: np.ndarray


def load_reference(path: Path, name: str) -> SensitivityReference:
    """Load and validate one closed sensitivity domain."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError(f"expected one top-level reference entry in {path}")
    entry = payload[0]
    missing = {"Production modes", "Decay description", "Sensitivity domains"} - set(entry)
    if missing:
        raise ValueError(f"missing reference keys {sorted(missing)} in {path}")
    domains = entry["Sensitivity domains"]
    if not isinstance(domains, list) or len(domains) != 1:
        raise ValueError(f"expected exactly one sensitivity domain in {path}")
    points = np.asarray(domains[0], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError(f"expected at least four (mass, coupling) points in {path}")
    if not np.all(np.isfinite(points)) or np.any(points <= 0.0):
        raise ValueError(f"reference coordinates must be finite and positive in {path}")
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    return SensitivityReference(
        name=name, path=path, production_modes=tuple(entry["Production modes"]),
        decay_description=str(entry["Decay description"]), points=points,
    )


def _reduce_branch(points: np.ndarray, aggregation: str) -> pd.DataFrame:
    branch = pd.DataFrame({"mass_GeV": points[:, 0], "coupling_GeV_inv": points[:, 1]})
    branch = (branch.groupby("mass_GeV", as_index=False)
              .agg(coupling_GeV_inv=("coupling_GeV_inv", aggregation))
              .sort_values("mass_GeV").reset_index(drop=True))
    if len(branch) < 2:
        raise ValueError("reference branch contains fewer than two masses")
    return branch


def interpolate_log_coupling(branch: pd.DataFrame, masses_gev: np.ndarray) -> np.ndarray:
    """Interpolate linearly in log10(mass)-log10(coupling) space."""
    return 10.0 ** np.interp(
        np.log10(masses_gev), np.log10(branch["mass_GeV"].to_numpy(float)),
        np.log10(branch["coupling_GeV_inv"].to_numpy(float)),
    )


def split_reference_branches(points: np.ndarray) -> dict[str, pd.DataFrame]:
    """Split an ordered closed polygon at its maximum-mass turning point."""
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    minimum_mass = points[:, 0].min()
    candidates = np.flatnonzero(np.isclose(points[:, 0], minimum_mass, rtol=1e-12, atol=0.0))
    if not len(candidates):
        raise RuntimeError("could not identify the minimum-mass edge")
    start = candidates[np.argmin(points[candidates, 1])]
    ordered = np.vstack((points[start:], points[:start]))
    turn = int(np.argmax(ordered[:, 0]))
    if turn == 0 or turn == len(ordered) - 1:
        raise ValueError("maximum-mass point does not divide the reference polygon")
    candidate_a, candidate_b = ordered[:turn + 1], ordered[turn:]
    probe_a, probe_b = _reduce_branch(candidate_a, "median"), _reduce_branch(candidate_b, "median")
    probe_min = max(probe_a["mass_GeV"].min(), probe_b["mass_GeV"].min())
    probe_max = min(probe_a["mass_GeV"].max(), probe_b["mass_GeV"].max())
    if probe_min >= probe_max:
        raise ValueError("reference branches have no overlapping mass interval")
    probe_mass = np.asarray([np.sqrt(probe_min * probe_max)])
    a_above = interpolate_log_coupling(probe_a, probe_mass)[0] > interpolate_log_coupling(probe_b, probe_mass)[0]
    upper_points, lower_points = ((candidate_a, candidate_b) if a_above else (candidate_b, candidate_a))
    return {"lower": _reduce_branch(lower_points, "min"),
            "upper": _reduce_branch(upper_points, "max")}


def load_eventcalc_branches(boundaries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Select the combined photon N=2.3 lower and upper EventCalc branches."""
    required = {"model", "mass_GeV", "event_level", "lower_coupling_GeV_inv", "upper_coupling_GeV_inv"}
    missing = required - set(boundaries)
    if missing:
        raise ValueError(f"EventCalc contour table is missing columns: {sorted(missing)}")
    selected = boundaries.loc[
        (boundaries["model"] == MODEL_NAME)
        & np.isclose(boundaries["event_level"], EVENT_LEVEL, rtol=0.0, atol=1e-12)
    ]
    if selected.empty:
        raise ValueError(f"no {MODEL_NAME!r} contour found at N_events={EVENT_LEVEL:g}")
    result = {}
    for name, column in (("lower", "lower_coupling_GeV_inv"), ("upper", "upper_coupling_GeV_inv")):
        branch = selected[["mass_GeV", column]].rename(columns={column: "coupling_GeV_inv"})
        branch = branch.loc[np.isfinite(branch).all(axis=1) & (branch > 0.0).all(axis=1)]
        branch = (branch.groupby("mass_GeV", as_index=False)
                  .agg(coupling_GeV_inv=("coupling_GeV_inv", "median"))
                  .sort_values("mass_GeV").reset_index(drop=True))
        if len(branch) < 2:
            raise ValueError(f"EventCalc {name} branch contains fewer than two points")
        result[name] = branch
    return result


def make_pointwise_comparison(
    eventcalc: Mapping[str, pd.DataFrame], references: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Evaluate signed log10 coupling differences on common mass grids."""
    rows = []
    for reference_name, branches in references.items():
        minimum = max(eventcalc[name]["mass_GeV"].min() for name in ("lower", "upper"))
        minimum = max(minimum, *(branches[name]["mass_GeV"].min() for name in ("lower", "upper")))
        maximum = min(eventcalc[name]["mass_GeV"].max() for name in ("lower", "upper"))
        maximum = min(maximum, *(branches[name]["mass_GeV"].max() for name in ("lower", "upper")))
        if minimum >= maximum:
            raise ValueError(f"no common mass interval for reference {reference_name!r}")
        masses = np.geomspace(minimum, ENDPOINT_FRACTION * maximum, N_COMPARISON_POINTS)
        for branch_name in ("lower", "upper"):
            current = interpolate_log_coupling(eventcalc[branch_name], masses)
            reference = interpolate_log_coupling(branches[branch_name], masses)
            distance = np.log10(reference / current)
            rows.append(pd.DataFrame({
                "reference": reference_name, "branch": branch_name, "mass_GeV": masses,
                "eventcalc_coupling_GeV_inv": current,
                "reference_coupling_GeV_inv": reference,
                "log10_reference_over_eventcalc_dex": distance,
                "absolute_log_distance_dex": np.abs(distance),
                "coupling_ratio_reference_over_eventcalc": reference / current,
            }))
    return pd.concat(rows, ignore_index=True)


def make_distance_summary(pointwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize maximum deviations and closed-region containment."""
    rows = []
    for reference_name in REFERENCE_FILENAMES:
        selected = pointwise.loc[pointwise["reference"] == reference_name]
        lower = selected.loc[selected["branch"] == "lower"].reset_index(drop=True)
        upper = selected.loc[selected["branch"] == "upper"].reset_index(drop=True)
        if len(lower) != len(upper) or not np.allclose(lower["mass_GeV"], upper["mass_GeV"], rtol=1e-12, atol=0.0):
            raise RuntimeError(f"lower and upper mass grids differ for {reference_name!r}")
        masses = lower["mass_GeV"].to_numpy(float)
        lower_distance = lower["log10_reference_over_eventcalc_dex"].to_numpy(float)
        upper_distance = upper["log10_reference_over_eventcalc_dex"].to_numpy(float)
        lower_max, upper_max = int(np.argmax(np.abs(lower_distance))), int(np.argmax(np.abs(upper_distance)))
        lower_inside = lower_distance >= -CONTAINMENT_TOLERANCE_DEX
        upper_inside = upper_distance <= CONTAINMENT_TOLERANCE_DEX
        both_inside = lower_inside & upper_inside
        rows.append({
            "reference": reference_name, "number_of_mass_points": len(masses),
            "minimum_mass_GeV": float(masses[0]), "maximum_mass_GeV": float(masses[-1]),
            "maximum_absolute_lower_distance_dex": float(abs(lower_distance[lower_max])),
            "mass_at_maximum_lower_distance_GeV": float(masses[lower_max]),
            "maximum_absolute_upper_distance_dex": float(abs(upper_distance[upper_max])),
            "mass_at_maximum_upper_distance_GeV": float(masses[upper_max]),
            "largest_lower_coupling_factor_difference": float(10.0 ** abs(lower_distance[lower_max])),
            "largest_upper_coupling_factor_difference": float(10.0 ** abs(upper_distance[upper_max])),
            "lower_branch_inside_fraction": float(np.mean(lower_inside)),
            "upper_branch_inside_fraction": float(np.mean(upper_inside)),
            "both_branches_inside_fraction": float(np.mean(both_inside)),
            "worst_lower_outward_violation_dex": max(0.0, -float(np.min(lower_distance))),
            "worst_upper_outward_violation_dex": max(0.0, float(np.max(upper_distance))),
            "strictly_inside_with_tolerance": bool(np.all(both_inside)),
            "containment_tolerance_dex": CONTAINMENT_TOLERANCE_DEX,
        })
    return pd.DataFrame(rows)


def make_reference_summary(references: list[SensitivityReference]) -> pd.DataFrame:
    return pd.DataFrame({
        "reference": reference.name, "path": str(reference.path), "event_level": EVENT_LEVEL,
        "number_of_points": len(reference.points), "minimum_mass_GeV": float(reference.points[:, 0].min()),
        "maximum_mass_GeV": float(reference.points[:, 0].max()),
        "minimum_coupling_GeV_inv": float(reference.points[:, 1].min()),
        "maximum_coupling_GeV_inv": float(reference.points[:, 1].max()),
        "decay_description": reference.decay_description,
        "production_modes": "; ".join(reference.production_modes),
    } for reference in references)
