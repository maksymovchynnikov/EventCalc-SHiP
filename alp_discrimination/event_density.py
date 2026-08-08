"""Pure composition and contour utilities for the event-density workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceScanDefinition:
    identifier: str
    model_id: str
    source_id: str
    mass_min_gev: float
    mass_max_gev: float
    coupling_min_gev_inv: float
    coupling_max_gev_inv: float
    seed_offset: int


SOURCE_SCANS = (
    SourceScanDefinition("ALP-photon-primary", "alp_photon_combined", "primary", 0.02, 4.0, 1e-10, 1e-2, 0),
    SourceScanDefinition("ALP-photon-cascades", "alp_photon_combined", "cascade", 0.02, 4.0, 1e-10, 1e-2, 2_000),
    SourceScanDefinition("ALP-SU2L", "alp_su2l", "inclusive", 0.01, 5.1, 1e-8, 3.0, 1_000),
)


def stable_float_key(value: float) -> str:
    return f"{float(value):.12e}"


def combine_photon_sources(source_data: pd.DataFrame, require_complete: bool = True) -> pd.DataFrame:
    """Add primary and cascade event rates at identical mass-coupling points."""
    su2 = source_data[source_data["model"] == "ALP-SU2L"].copy()
    primary = source_data[source_data["model"] == "ALP-photon-primary"].copy()
    cascade = source_data[source_data["model"] == "ALP-photon-cascades"].copy()
    if primary.empty or cascade.empty:
        if require_complete:
            raise ValueError("primary and cascade source rows are both required")
        return su2
    for frame in (primary, cascade):
        frame["_mass"] = frame["mass_GeV"].map(stable_float_key)
        frame["_coupling"] = frame["coupling_GeV_inv"].map(stable_float_key)
    merged = primary.merge(
        cascade, on=["_mass", "_coupling"], suffixes=("_primary", "_cascade"),
        how="inner", validate="one_to_one",
    )
    if require_complete and (len(merged) != len(primary) or len(merged) != len(cascade)):
        raise ValueError("primary and cascade mass-coupling grids differ")
    common = ("mass_GeV", "coupling_GeV_inv", "coupling_squared_GeV_inv2", "ctau_m",
              "unit_coupling_ctau_m", "visible_Br")
    for column in common:
        if not np.allclose(merged[f"{column}_primary"], merged[f"{column}_cascade"], rtol=1e-10, atol=1e-14):
            raise ValueError(f"photon source column {column} differs")
    primary_events = merged["N_events_primary"].to_numpy(float)
    cascade_events = merged["N_events_cascade"].to_numpy(float)
    total = primary_events + cascade_events
    combined = pd.DataFrame({
        "model": "ALP-photon-combined", "mass_GeV": merged["mass_GeV_primary"],
        "coupling_GeV_inv": merged["coupling_GeV_inv_primary"],
        "coupling_squared_GeV_inv2": merged["coupling_squared_GeV_inv2_primary"],
        "ctau_m": merged["ctau_m_primary"],
        "unit_coupling_ctau_m": merged["unit_coupling_ctau_m_primary"],
        "yield_per_PoT_per_coupling_squared": (
            merged["yield_per_PoT_per_coupling_squared_primary"]
            + merged["yield_per_PoT_per_coupling_squared_cascade"]
        ),
        "N_LLP_total": merged["N_LLP_total_primary"] + merged["N_LLP_total_cascade"],
        "epsilon_polar": np.nan, "epsilon_azimuthal": np.nan, "mean_P_decay": np.nan,
        "sum_P_decay": np.nan, "visible_Br": merged["visible_Br_primary"],
        "sampled_inside_volume": (
            merged["sampled_inside_volume_primary"] + merged["sampled_inside_volume_cascade"]
        ),
        "N_events": total, "N_events_primary": primary_events,
        "N_events_cascades": cascade_events,
        "cascade_event_fraction": np.divide(cascade_events, total, out=np.zeros_like(total), where=total > 0.0),
        "cascade_to_primary_event_ratio": np.divide(
            cascade_events, primary_events, out=np.full_like(total, np.nan), where=primary_events > 0.0
        ),
    })
    return pd.concat([combined, su2], ignore_index=True, sort=False).sort_values(
        ["model", "mass_GeV", "coupling_GeV_inv"]
    ).reset_index(drop=True)


def find_level_crossings(mass_data: pd.DataFrame, event_level: float) -> list[float]:
    data = mass_data.sort_values("coupling_GeV_inv")
    couplings = data["coupling_GeV_inv"].to_numpy(float)
    log_difference = np.log10(np.maximum(data["N_events"].to_numpy(float), 1e-300)) - np.log10(event_level)
    log_couplings, crossings = np.log10(couplings), []
    for index in range(len(couplings) - 1):
        left, right = log_difference[index:index + 2]
        if left == 0.0:
            crossings.append(float(couplings[index]))
        if left * right < 0.0:
            fraction = -left / (right - left)
            crossings.append(float(10.0 ** (
                log_couplings[index] + fraction * (log_couplings[index + 1] - log_couplings[index])
            )))
    return crossings


def build_boundary_table(scan_data: pd.DataFrame, event_levels: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for (model, mass_gev), data in scan_data.groupby(["model", "mass_GeV"], sort=False):
        data = data.sort_values("coupling_GeV_inv")
        rates, couplings = data["N_events"].to_numpy(float), data["coupling_GeV_inv"].to_numpy(float)
        peak = int(np.argmax(rates))
        for level in event_levels:
            crossings = find_level_crossings(data, level)
            if rates[peak] < level:
                status = "outside_mass_reach"
            elif len(crossings) == 2:
                status = "resolved"
            elif len(crossings) > 2:
                status = "multiple_crossings"
            elif len(crossings) == 1:
                if rates[0] < level <= rates[-1]:
                    status = "upper_boundary_above_scan"
                elif rates[0] >= level > rates[-1]:
                    status = "lower_boundary_below_scan"
                else:
                    status = "one_crossing_unclassified"
            elif rates[0] >= level and rates[-1] >= level:
                status = "both_boundaries_outside_scan"
            else:
                status = "unresolved_numerically"
            rows.append({
                "model": model, "mass_GeV": mass_gev, "event_level": level,
                "status": status, "number_of_crossings": len(crossings),
                "maximum_N_events": float(rates[peak]),
                "peak_coupling_GeV_inv": float(couplings[peak]),
                "lower_coupling_GeV_inv": crossings[0] if crossings else np.nan,
                "upper_coupling_GeV_inv": crossings[-1] if len(crossings) >= 2 else np.nan,
            })
    return pd.DataFrame(rows)


def endpoint_refinement_masses(
    boundaries: pd.DataFrame, points_per_bracket: int, relative_width_tolerance: float,
) -> dict[str, np.ndarray]:
    result = {model: [] for model in boundaries["model"].unique()}
    for (model, _), data in boundaries.groupby(["model", "event_level"], sort=False):
        resolved = data[data["status"] == "resolved"]
        if resolved.empty:
            continue
        left = float(resolved["mass_GeV"].max())
        outside = data[(data["status"] == "outside_mass_reach") & (data["mass_GeV"] > left)]
        if outside.empty:
            continue
        right = float(outside["mass_GeV"].min())
        if (right - left) / (0.5 * (right + left)) > relative_width_tolerance:
            result[model].extend(np.linspace(left, right, points_per_bracket + 2)[1:-1])
    return {model: np.unique(values) for model, values in result.items()}


def add_interpolated_closing_points(boundaries: pd.DataFrame) -> pd.DataFrame:
    output, rows = boundaries.copy(), []
    output["is_interpolated"] = False
    for (model, level), data in boundaries.groupby(["model", "event_level"], sort=False):
        resolved = data[data["status"] == "resolved"]
        if resolved.empty:
            continue
        left = resolved.loc[resolved["mass_GeV"].idxmax()]
        outside = data[(data["status"] == "outside_mass_reach") & (data["mass_GeV"] > left.mass_GeV)]
        if outside.empty:
            continue
        right = outside.loc[outside["mass_GeV"].idxmin()]
        fraction = (np.log(level) - np.log(left.maximum_N_events)) / (
            np.log(right.maximum_N_events) - np.log(left.maximum_N_events)
        )
        mass = left.mass_GeV + fraction * (right.mass_GeV - left.mass_GeV)
        coupling = float(np.exp(
            np.log(left.peak_coupling_GeV_inv)
            + fraction * (np.log(right.peak_coupling_GeV_inv) - np.log(left.peak_coupling_GeV_inv))
        ))
        rows.append({
            "model": model, "mass_GeV": mass, "event_level": level,
            "status": "interpolated_closing_point", "number_of_crossings": np.nan,
            "maximum_N_events": level, "peak_coupling_GeV_inv": coupling,
            "lower_coupling_GeV_inv": coupling, "upper_coupling_GeV_inv": coupling,
            "is_interpolated": True,
        })
    return pd.concat([output, pd.DataFrame(rows)], ignore_index=True).sort_values(
        ["model", "event_level", "mass_GeV"]
    ).reset_index(drop=True)
