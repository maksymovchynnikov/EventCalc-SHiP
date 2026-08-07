"""Adaptive planning and convergence utilities for the Week-8 SHiP scan.

The expensive EventCalc and profiled-likelihood kernels remain in their
validated workflow modules.  This module contains only deterministic planning,
selection, confidence-bound and result-reduction logic, so it is inexpensive to
test and safe to reuse from a resumable orchestration layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, log10, sqrt
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis2.lifetime_template_banks import LifetimeTemplateBank
from analysis2.profiled_reduction import minimum_persistent_events


TRUTH_MODELS = ("photon", "su2")
DOMAIN_MODEL_LABELS = {
    "photon": "ALP-photon-combined",
    "su2": "ALP-SU2L",
}
SELECTIONS = ("diphoton_ecal", "diphoton_ecal_e1gev")


@dataclass(frozen=True)
class AdaptiveLifetimeSettings:
    """Controls an interval-aware adaptive profile-lifetime grid.

    The initial grid scales with the logarithmic width of each connected
    sensitivity component.  Subsequent rounds add geometric midpoints only
    where adjacent detector-level templates or distance-map rows/columns vary
    too rapidly.  This automatically allocates more points to whichever model
    controls the distance surface; there is no hard-coded preference for one
    model.
    """

    initial_points_per_decade: float = 4.0
    minimum_points_per_interval: int = 5
    maximum_log_gap_decades: float = 0.25
    minimum_region_log_gap_decades: float = 0.045
    maximum_adjacent_template_tv: float = 0.018
    maximum_log_interpolation_tv: float = 0.004
    maximum_adjacent_distance_change: float = 0.035
    distance_relevance_margin: float = 0.15
    maximum_rounds: int = 8
    maximum_total_lifetimes_per_model: int = 120
    maximum_new_points_per_model_per_round: int = 16
    minimum_distance_relative_tolerance: float = 0.02
    maximum_soft_priority_at_convergence: float = 6.0

    def __post_init__(self) -> None:
        positive = (
            self.initial_points_per_decade,
            self.maximum_log_gap_decades,
            self.minimum_region_log_gap_decades,
            self.maximum_adjacent_template_tv,
            self.maximum_log_interpolation_tv,
            self.maximum_adjacent_distance_change,
            self.distance_relevance_margin,
            self.minimum_distance_relative_tolerance,
            self.maximum_soft_priority_at_convergence,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Adaptive lifetime tolerances must be finite and positive.")
        if self.minimum_points_per_interval < 2:
            raise ValueError("Every connected interval needs at least two points.")
        if self.maximum_rounds < 1:
            raise ValueError("At least one lifetime-grid round is required.")
        if self.maximum_total_lifetimes_per_model < 2:
            raise ValueError("Maximum lifetime-grid size must be at least two.")
        if self.maximum_new_points_per_model_per_round < 1:
            raise ValueError("At least one point must be addable per refinement round.")


@dataclass(frozen=True)
class AdaptivePseudoexperimentSettings:
    """Controls range finding, truth promotion and PE convergence."""

    target_accuracy: float = 0.90
    rangefinder_pseudoexperiments: int = 1000
    rangefinder_seeds: int = 2
    full_domain_pilot_pseudoexperiments: int = 2000
    minimum_final_pseudoexperiments: int = 10000
    final_seeds: int = 5
    pseudoexperiment_ladder: tuple[int, ...] = (5000, 10000, 20000)
    screening_truths_per_model: int = 8
    screening_neighbourhood: int = 2
    hard_truth_accuracy_gap: float = 0.030
    minimum_hard_truths_per_model: int = 8
    maximum_hard_truth_fraction_per_model: float = 0.75
    threshold_stability_events: int = 1
    required_stable_transitions: int = 2
    audit_global_alpha: float = 0.01
    rangefinder_scale_constant: float = 0.30
    rangefinder_minimum_events: int = 2
    rangefinder_maximum_events: int = 20000
    unit_window_minimum_half_width: int = 30
    unit_window_bracket_fraction: float = 0.40
    persistence_tail_factor: float = 1.8
    persistence_tail_minimum_extra: int = 75
    maximum_unit_window_points: int = 241

    def __post_init__(self) -> None:
        if not 0.0 < self.target_accuracy < 1.0:
            raise ValueError("Target accuracy must lie strictly between zero and one.")
        if self.rangefinder_pseudoexperiments < 1:
            raise ValueError("Range-finder PE count must be positive.")
        if self.full_domain_pilot_pseudoexperiments < 1:
            raise ValueError("Pilot PE count must be positive.")
        if self.minimum_final_pseudoexperiments < self.full_domain_pilot_pseudoexperiments:
            raise ValueError("Minimum final PE count cannot be below the pilot.")
        if self.rangefinder_seeds < 1 or self.final_seeds < 1:
            raise ValueError("Seed counts must be positive.")
        if not self.pseudoexperiment_ladder:
            raise ValueError("The PE ladder cannot be empty.")
        previous = self.full_domain_pilot_pseudoexperiments
        for level in self.pseudoexperiment_ladder:
            if level <= previous:
                raise ValueError("PE ladder levels must increase above the pilot.")
            previous = level
        if self.minimum_final_pseudoexperiments > self.pseudoexperiment_ladder[-1]:
            raise ValueError("Minimum final PE count exceeds the PE ladder maximum.")
        if self.screening_truths_per_model < 1:
            raise ValueError("At least one screening truth per model is required.")
        if self.screening_neighbourhood < 0:
            raise ValueError("Screening neighbourhood cannot be negative.")
        if self.hard_truth_accuracy_gap <= 0.0:
            raise ValueError("Hard-truth accuracy gap must be positive.")
        if not 0.0 < self.maximum_hard_truth_fraction_per_model <= 1.0:
            raise ValueError("Maximum hard-truth fraction must lie in (0,1].")
        if self.minimum_hard_truths_per_model < 1:
            raise ValueError("Minimum hard-truth count must be positive.")
        if self.threshold_stability_events < 0:
            raise ValueError("Threshold stability tolerance cannot be negative.")
        if self.required_stable_transitions < 1:
            raise ValueError("At least one stable PE transition is required.")
        if not 0.0 < self.audit_global_alpha < 1.0:
            raise ValueError("Audit alpha must lie strictly between zero and one.")
        if self.rangefinder_scale_constant <= 0.0:
            raise ValueError("Range-finder scale constant must be positive.")
        if self.rangefinder_minimum_events < 1:
            raise ValueError("Minimum range-finder event count must be positive.")
        if self.rangefinder_maximum_events < self.rangefinder_minimum_events:
            raise ValueError("Range-finder maximum is below its minimum.")
        if self.maximum_unit_window_points < 3:
            raise ValueError("Unit window must allow at least three points.")


@dataclass(frozen=True)
class AdaptiveWeek8Settings:
    """Complete settings serialized into every adaptive scan state."""

    lifetime: AdaptiveLifetimeSettings = AdaptiveLifetimeSettings()
    pseudoexperiments: AdaptivePseudoexperimentSettings = (
        AdaptivePseudoexperimentSettings()
    )
    initial_energy_bins: int = 200
    minimum_bin_n_eff: float = 100.0
    conditional_fine_binning_bins: int = 400
    maximum_binning_refinement_rounds: int = 2
    binning_refinement_factor: int = 2
    fine_binning_minimum_final_bins: int = 6
    fine_binning_distance_threshold: float = 0.08
    fine_binning_relative_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.initial_energy_bins < 1:
            raise ValueError("Initial energy-bin count must be positive.")
        if self.minimum_bin_n_eff <= 0.0:
            raise ValueError("Minimum bin effective sample size must be positive.")
        if self.conditional_fine_binning_bins <= self.initial_energy_bins:
            raise ValueError("Fine binning must start from more bins than the default.")
        if self.maximum_binning_refinement_rounds < 1:
            raise ValueError("At least one conditional binning round is required.")
        if self.binning_refinement_factor < 2:
            raise ValueError("Binning refinement factor must be at least two.")
        if self.fine_binning_minimum_final_bins < 2:
            raise ValueError("At least two final energy bins are required.")
        if self.fine_binning_distance_threshold <= 0.0:
            raise ValueError("Fine-binning distance trigger must be positive.")
        if self.fine_binning_relative_tolerance <= 0.0:
            raise ValueError("Fine-binning tolerance must be positive.")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LifetimeRefinementDecision:
    grid: pd.DataFrame
    additions: pd.DataFrame
    diagnostics: pd.DataFrame
    minimum_distance: float
    previous_minimum_distance: float | None
    relative_minimum_distance_change: float | None
    converged: bool
    reached_round_limit: bool
    reached_size_limit: bool


@dataclass(frozen=True)
class RangefinderBracket:
    lower_failing_events: int
    upper_passing_events: int
    estimated_crossing_events: float
    threshold_reached: bool


@dataclass(frozen=True)
class OmittedTruthAudit:
    point_table: pd.DataFrame
    truth_summary: pd.DataFrame
    simultaneous_bounds: int
    adjusted_z: float
    overlapping_truths: pd.DataFrame


@dataclass(frozen=True)
class MonteCarloThresholdDiagnostics:
    point_estimate: int | None
    previous_tested_events: int | None
    previous_accuracy: float | None
    accuracy_at_point: float | None
    local_sigma_events: float | None
    simultaneous_lower_events: int | None
    simultaneous_upper_events: int | None
    simultaneous_interval_closed: bool


_REQUIRED_DOMAIN_COLUMNS = {
    "model",
    "mass_GeV",
    "interval_index",
    "ctau_min_m",
    "ctau_max_m",
}


def _mass_rows(domains: pd.DataFrame, mass_gev: float) -> pd.DataFrame:
    missing = _REQUIRED_DOMAIN_COLUMNS - set(domains.columns)
    if missing:
        raise ValueError(f"Domain table is missing columns: {sorted(missing)}")
    rows = domains.loc[
        np.isclose(
            domains["mass_GeV"].to_numpy(dtype=float),
            float(mass_gev),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()
    if rows.empty:
        raise ValueError(f"No Week-8 lifetime domains for m_a={mass_gev:g} GeV.")
    return rows


def initial_adaptive_lifetime_grid(
    domains: pd.DataFrame,
    mass_gev: float,
    settings: AdaptiveLifetimeSettings,
) -> pd.DataFrame:
    """Build a width-scaled endpoint-preserving initial lifetime grid."""
    rows = _mass_rows(domains, mass_gev)
    frames: list[pd.DataFrame] = []
    for truth_model in TRUTH_MODELS:
        label = DOMAIN_MODEL_LABELS[truth_model]
        selected = rows.loc[rows["model"] == label].sort_values(
            "interval_index", ignore_index=True
        )
        if selected.empty:
            raise ValueError(
                f"No {label} lifetime domain for m_a={mass_gev:g} GeV."
            )
        for interval in selected.itertuples(index=False):
            lower = float(interval.ctau_min_m)
            upper = float(interval.ctau_max_m)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0:
                raise ValueError("Lifetime-domain endpoints must be finite and positive.")
            if upper <= lower:
                raise ValueError("Lifetime-domain upper endpoint must exceed lower.")
            width_decades = log10(upper / lower)
            points = max(
                settings.minimum_points_per_interval,
                int(ceil(width_decades * settings.initial_points_per_decade)) + 1,
            )
            values = np.geomspace(lower, upper, points)
            frames.append(
                pd.DataFrame(
                    {
                        "model": label,
                        "mass_GeV": float(mass_gev),
                        "interval_index": int(interval.interval_index),
                        "ctau_m": values,
                        "is_interval_endpoint": [
                            index in {0, points - 1} for index in range(points)
                        ],
                        "adaptive_round_added": 0,
                        "adaptive_reason": "initial_log_width_grid",
                    }
                )
            )
    return canonicalize_lifetime_grid(pd.concat(frames, ignore_index=True))


def canonicalize_lifetime_grid(grid: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "mass_GeV", "interval_index", "ctau_m"}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Lifetime grid is missing columns: {sorted(missing)}")
    result = grid.copy()
    result["mass_GeV"] = pd.to_numeric(result["mass_GeV"], errors="raise")
    result["interval_index"] = pd.to_numeric(
        result["interval_index"], errors="raise"
    ).astype(int)
    result["ctau_m"] = pd.to_numeric(result["ctau_m"], errors="raise")
    if np.any(~np.isfinite(result["ctau_m"])) or np.any(result["ctau_m"] <= 0.0):
        raise ValueError("Lifetime-grid values must be finite and positive.")
    if "is_interval_endpoint" not in result:
        result["is_interval_endpoint"] = False
    if "adaptive_round_added" not in result:
        result["adaptive_round_added"] = 0
    if "adaptive_reason" not in result:
        result["adaptive_reason"] = "unspecified"
    result = result.sort_values(
        ["mass_GeV", "model", "interval_index", "ctau_m"],
        kind="mergesort",
        ignore_index=True,
    )
    duplicate = result.duplicated(
        ["mass_GeV", "model", "interval_index", "ctau_m"]
    )
    if duplicate.any():
        result = result.loc[~duplicate].reset_index(drop=True)
    return result


def lifetime_grid_from_bank(
    bank: LifetimeTemplateBank,
    *,
    adaptive_round_added: int = 0,
    reason: str = "loaded_bank",
) -> pd.DataFrame:
    frames = []
    for truth_model, prefix in (("photon", "photon"), ("su2", "su2")):
        label = DOMAIN_MODEL_LABELS[truth_model]
        ctaus = np.asarray(getattr(bank, f"{prefix}_ctau_m"), dtype=float)
        intervals = np.asarray(
            getattr(bank, f"{prefix}_interval_index"), dtype=int
        )
        frame = pd.DataFrame(
            {
                "model": label,
                "mass_GeV": float(bank.mass_gev),
                "interval_index": intervals,
                "ctau_m": ctaus,
                "is_interval_endpoint": False,
                "adaptive_round_added": int(adaptive_round_added),
                "adaptive_reason": reason,
            }
        )
        for interval_index, group in frame.groupby("interval_index"):
            ordered = group.sort_values("ctau_m")
            frame.loc[ordered.index[[0, -1]], "is_interval_endpoint"] = True
        frames.append(frame)
    return canonicalize_lifetime_grid(pd.concat(frames, ignore_index=True))


def total_variation_matrix(bank: LifetimeTemplateBank) -> np.ndarray:
    return 0.5 * np.abs(
        bank.photon_probabilities[:, None, :]
        - bank.su2_probabilities[None, :, :]
    ).sum(axis=2)


def _adjacent_interval_pairs(interval_indices: np.ndarray) -> list[tuple[int, int]]:
    return [
        (index, index + 1)
        for index in range(len(interval_indices) - 1)
        if int(interval_indices[index]) == int(interval_indices[index + 1])
    ]


def _axis_distance_change(
    distances: np.ndarray,
    truth_model: str,
    left: int,
    right: int,
    *,
    minimum_distance: float,
    relevance_margin: float,
) -> float:
    """Return the largest row/column change in the competitive corridor.

    Large changes in regions where the hypotheses are already almost perfectly
    separated do not control the profiled threshold and should not force dense
    lifetime sampling.  The comparison therefore retains entries within a
    fixed total-variation margin of the global minimum.  The independent
    template-TV and maximum-log-gap criteria still protect the rest of each
    connected lifetime component.
    """
    if truth_model == "photon":
        first = np.asarray(distances[left, :], dtype=float)
        second = np.asarray(distances[right, :], dtype=float)
    else:
        first = np.asarray(distances[:, left], dtype=float)
        second = np.asarray(distances[:, right], dtype=float)
    competitive = np.minimum(first, second) <= min(
        1.0, float(minimum_distance) + float(relevance_margin)
    )
    difference = np.abs(second - first)
    if np.any(competitive):
        return float(np.max(difference[competitive]))
    return float(np.max(difference))


def _minimum_neighbour_pairs(
    minimum_index: int,
    interval_indices: np.ndarray,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for left, right in ((minimum_index - 1, minimum_index), (minimum_index, minimum_index + 1)):
        if left < 0 or right >= len(interval_indices):
            continue
        if int(interval_indices[left]) == int(interval_indices[right]):
            pairs.add((left, right))
    return pairs


def _log_interpolation_residual_by_pair(
    ctaus: np.ndarray,
    probabilities: np.ndarray,
    interval_indices: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Assign leave-one-out log-lifetime interpolation residuals to pairs."""
    residuals: dict[tuple[int, int], float] = {}
    log_ctau = np.log(np.asarray(ctaus, dtype=float))
    for index in range(1, len(ctaus) - 1):
        if not (
            int(interval_indices[index - 1])
            == int(interval_indices[index])
            == int(interval_indices[index + 1])
        ):
            continue
        denominator = log_ctau[index + 1] - log_ctau[index - 1]
        if denominator <= 0.0:
            continue
        weight = (log_ctau[index] - log_ctau[index - 1]) / denominator
        interpolated = (
            (1.0 - weight) * probabilities[index - 1]
            + weight * probabilities[index + 1]
        )
        residual = float(
            0.5 * np.abs(probabilities[index] - interpolated).sum()
        )
        for pair in ((index - 1, index), (index, index + 1)):
            residuals[pair] = max(residuals.get(pair, 0.0), residual)
    return residuals


def propose_lifetime_refinement(
    bank: LifetimeTemplateBank,
    distances: np.ndarray,
    current_grid: pd.DataFrame,
    settings: AdaptiveLifetimeSettings,
    *,
    round_index: int,
    previous_minimum_distance: float | None = None,
) -> LifetimeRefinementDecision:
    """Insert the highest-priority unresolved lifetime midpoints.

    The nominal template/interpolation/distance tolerances are deliberately
    sensitive diagnostics.  Requiring every pair in the full allowed domain to
    satisfy them caused dense refinement in regions that cannot control the
    profiled discrimination threshold.  Convergence therefore distinguishes
    two classes:

    * hard requirements: global log spacing, minimum-basin spacing/stability,
      shape variation adjacent to the current minimum, and exceptionally large
      soft-priority violations anywhere in the domain;
    * soft diagnostics: moderate detector-template curvature away from the
      current minimum.  These guide refinement while a hard requirement is
      unresolved, but do not by themselves prevent convergence.

    This calibrated rule retains disconnected-domain and profile-grid coverage,
    protects against false early plateaus, and avoids refining every harmless
    high-curvature tail to the nominal diagnostic tolerance.
    """
    expected_shape = (len(bank.photon_ctau_m), len(bank.su2_ctau_m))
    matrix = np.asarray(distances, dtype=float)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Distance matrix has shape {matrix.shape}, expected {expected_shape}."
        )
    if np.any(~np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("Distance matrix must be finite and lie in [0,1].")

    minimum_flat = int(np.argmin(matrix))
    minimum_photon, minimum_su2 = np.unravel_index(minimum_flat, matrix.shape)
    minimum_distance = float(matrix[minimum_photon, minimum_su2])
    relative_change = None
    if previous_minimum_distance is not None:
        scale = max(abs(float(previous_minimum_distance)), np.finfo(float).eps)
        relative_change = abs(
            minimum_distance - float(previous_minimum_distance)
        ) / scale

    grid = canonicalize_lifetime_grid(current_grid)
    diagnostics: list[dict] = []
    candidates_by_model: dict[str, list[dict]] = {}
    required_candidate_count = 0

    model_data = {
        "photon": (
            bank.photon_ctau_m,
            bank.photon_probabilities,
            bank.photon_interval_index,
            minimum_photon,
        ),
        "su2": (
            bank.su2_ctau_m,
            bank.su2_probabilities,
            bank.su2_interval_index,
            minimum_su2,
        ),
    }

    for truth_model in TRUTH_MODELS:
        ctaus, probabilities, interval_indices, minimum_index = model_data[
            truth_model
        ]
        ctaus = np.asarray(ctaus, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        interval_indices = np.asarray(interval_indices, dtype=int)
        pairs = _adjacent_interval_pairs(interval_indices)
        minimum_pairs = _minimum_neighbour_pairs(
            minimum_index, interval_indices
        )
        interpolation_residuals = _log_interpolation_residual_by_pair(
            ctaus, probabilities, interval_indices
        )
        candidates: list[dict] = []

        for left, right in pairs:
            gap = float(log10(ctaus[right] / ctaus[left]))
            adjacent_tv = float(
                0.5 * np.abs(probabilities[right] - probabilities[left]).sum()
            )
            distance_change = _axis_distance_change(
                matrix,
                truth_model,
                left,
                right,
                minimum_distance=minimum_distance,
                relevance_margin=settings.distance_relevance_margin,
            )
            interpolation_residual = float(
                interpolation_residuals.get((left, right), 0.0)
            )
            near_minimum = (left, right) in minimum_pairs
            ratios = {
                "log_gap": gap / settings.maximum_log_gap_decades,
                "template_tv": (
                    adjacent_tv / settings.maximum_adjacent_template_tv
                ),
                "interpolation_tv": (
                    interpolation_residual
                    / settings.maximum_log_interpolation_tv
                ),
                "distance_change": (
                    distance_change
                    / settings.maximum_adjacent_distance_change
                ),
                "minimum_region": (
                    gap / settings.minimum_region_log_gap_decades
                    if near_minimum
                    else 0.0
                ),
                "minimum_stability": (
                    relative_change
                    / settings.minimum_distance_relative_tolerance
                    if near_minimum and relative_change is not None
                    else 0.0
                ),
            }
            reason = max(ratios, key=ratios.get)
            priority = float(max(ratios.values()))
            soft_reason = max(
                ("template_tv", "interpolation_tv", "distance_change"),
                key=lambda key: ratios[key],
            )
            soft_priority = float(ratios[soft_reason])
            exceeds_nominal_tolerance = priority > 1.0

            required_reasons: list[tuple[str, float]] = []
            if (
                previous_minimum_distance is None
                and exceeds_nominal_tolerance
            ):
                required_reasons.append(("initial_resolution", priority))
            for hard_reason in (
                "log_gap",
                "minimum_region",
                "minimum_stability",
            ):
                if ratios[hard_reason] > 1.0:
                    required_reasons.append(
                        (hard_reason, float(ratios[hard_reason]))
                    )
            if near_minimum and soft_priority > 1.0:
                required_reasons.append(
                    (f"near_minimum_{soft_reason}", soft_priority)
                )
            if (
                soft_priority
                > settings.maximum_soft_priority_at_convergence
            ):
                required_reasons.append(
                    (f"extreme_{soft_reason}", soft_priority)
                )

            required_for_convergence = bool(required_reasons)
            required_reason = (
                max(required_reasons, key=lambda item: item[1])[0]
                if required_reasons
                else "soft_only"
            )
            diagnostic_index = len(diagnostics)
            diagnostics.append(
                {
                    "truth_model": truth_model,
                    "left_index": left,
                    "right_index": right,
                    "interval_index": int(interval_indices[left]),
                    "ctau_left_m": float(ctaus[left]),
                    "ctau_right_m": float(ctaus[right]),
                    "log_gap_decades": gap,
                    "adjacent_template_D_TV": adjacent_tv,
                    "log_interpolation_residual_D_TV": interpolation_residual,
                    "maximum_distance_map_change": distance_change,
                    "adjacent_to_global_minimum": near_minimum,
                    "priority": priority,
                    "soft_priority": soft_priority,
                    "dominant_reason": reason,
                    "required_reason": required_reason,
                    "exceeds_nominal_tolerance": exceeds_nominal_tolerance,
                    "required_for_convergence": required_for_convergence,
                    "selected_for_refinement": False,
                }
            )
            if exceeds_nominal_tolerance:
                candidates.append(
                    {
                        "model": DOMAIN_MODEL_LABELS[truth_model],
                        "mass_GeV": float(bank.mass_gev),
                        "interval_index": int(interval_indices[left]),
                        "ctau_m": float(sqrt(ctaus[left] * ctaus[right])),
                        "is_interval_endpoint": False,
                        "adaptive_round_added": int(round_index + 1),
                        "adaptive_reason": reason,
                        "priority": priority,
                        "required_for_convergence": required_for_convergence,
                        "diagnostic_index": diagnostic_index,
                    }
                )
                required_candidate_count += int(required_for_convergence)
        candidates_by_model[truth_model] = candidates

    distance_stable = (
        relative_change is None
        or relative_change <= settings.minimum_distance_relative_tolerance
    )
    reached_round_limit = round_index + 1 >= settings.maximum_rounds
    converged = required_candidate_count == 0 and distance_stable

    additions: list[dict] = []
    reached_size_limit = False
    if not converged:
        for truth_model in TRUTH_MODELS:
            candidates = candidates_by_model[truth_model]
            current_count = len(model_data[truth_model][0])
            capacity = (
                settings.maximum_total_lifetimes_per_model - current_count
            )
            required_count = sum(
                int(item["required_for_convergence"])
                for item in candidates
            )
            if capacity <= 0:
                if required_count:
                    reached_size_limit = True
                continue
            if required_count > capacity:
                reached_size_limit = True

            limit = min(
                capacity,
                settings.maximum_new_points_per_model_per_round,
            )
            chosen = sorted(
                candidates,
                key=lambda item: (
                    not bool(item["required_for_convergence"]),
                    -float(item["priority"]),
                    float(item["ctau_m"]),
                ),
            )[:limit]
            for item in chosen:
                diagnostics[int(item["diagnostic_index"])][
                    "selected_for_refinement"
                ] = True
                output_item = dict(item)
                for key in (
                    "priority",
                    "required_for_convergence",
                    "diagnostic_index",
                ):
                    output_item.pop(key, None)
                additions.append(output_item)

    additions_table = pd.DataFrame(additions)
    if additions_table.empty:
        combined = grid
    else:
        combined = canonicalize_lifetime_grid(
            pd.concat([grid, additions_table], ignore_index=True)
        )
    diagnostics_table = pd.DataFrame(diagnostics)

    return LifetimeRefinementDecision(
        grid=combined,
        additions=additions_table,
        diagnostics=diagnostics_table,
        minimum_distance=minimum_distance,
        previous_minimum_distance=previous_minimum_distance,
        relative_minimum_distance_change=relative_change,
        converged=converged and not reached_size_limit,
        reached_round_limit=reached_round_limit,
        reached_size_limit=reached_size_limit,
    )

def should_run_fine_binning_check(
    bank: LifetimeTemplateBank,
    minimum_distance: float,
    settings: AdaptiveWeek8Settings,
) -> bool:
    """Trigger the cached 400-bin rehistogram only for fragile banks."""
    return (
        bank.number_of_energy_bins < settings.fine_binning_minimum_final_bins
        or float(minimum_distance) < settings.fine_binning_distance_threshold
    )


def binning_is_stable(
    baseline_minimum_distance: float,
    fine_minimum_distance: float,
    baseline_minimum_intervals: tuple[int, int],
    fine_minimum_intervals: tuple[int, int],
    settings: AdaptiveWeek8Settings,
) -> bool:
    scale = max(abs(float(baseline_minimum_distance)), np.finfo(float).eps)
    relative = abs(float(fine_minimum_distance) - float(baseline_minimum_distance)) / scale
    return (
        relative <= settings.fine_binning_relative_tolerance
        and baseline_minimum_intervals == fine_minimum_intervals
    )


def _add_neighbourhood(
    selected: set[int],
    index: int,
    length: int,
    radius: int,
) -> None:
    for value in range(max(0, index - radius), min(length, index + radius + 1)):
        selected.add(value)


def distance_screening_truth_indices(
    bank: LifetimeTemplateBank,
    distances: np.ndarray,
    settings: AdaptivePseudoexperimentSettings,
) -> dict[str, np.ndarray]:
    """Choose a compact range-finding set from the distance surface.

    The set contains low-distance rows/columns, neighbours of the global
    minimum and every connected-domain endpoint.  It is deliberately used only
    for bracketing the event-count scale; final thresholds always receive a
    full-domain pilot and omitted-truth audit.
    """
    matrix = np.asarray(distances, dtype=float)
    if matrix.shape != (len(bank.photon_ctau_m), len(bank.su2_ctau_m)):
        raise ValueError("Distance matrix and bank lifetime grids disagree.")
    photon_scores = np.min(matrix, axis=1)
    su2_scores = np.min(matrix, axis=0)
    minimum_photon, minimum_su2 = np.unravel_index(
        int(np.argmin(matrix)), matrix.shape
    )
    result: dict[str, np.ndarray] = {}
    for truth_model, scores, minimum_index, intervals in (
        ("photon", photon_scores, minimum_photon, bank.photon_interval_index),
        ("su2", su2_scores, minimum_su2, bank.su2_interval_index),
    ):
        selected: set[int] = set()
        count = min(settings.screening_truths_per_model, len(scores))
        for index in np.argsort(scores, kind="mergesort")[:count]:
            _add_neighbourhood(
                selected,
                int(index),
                len(scores),
                settings.screening_neighbourhood,
            )
        _add_neighbourhood(
            selected,
            int(minimum_index),
            len(scores),
            settings.screening_neighbourhood,
        )
        intervals = np.asarray(intervals, dtype=int)
        for interval_index in np.unique(intervals):
            indices = np.flatnonzero(intervals == interval_index)
            selected.update({int(indices[0]), int(indices[-1])})
        result[truth_model] = np.asarray(sorted(selected), dtype=int)
    return result


def truth_subset_table(
    bank: LifetimeTemplateBank,
    truth_indices: Mapping[str, Sequence[int]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for truth_model in TRUTH_MODELS:
        indices = np.asarray(truth_indices[truth_model], dtype=int)
        ctaus = bank.photon_ctau_m if truth_model == "photon" else bank.su2_ctau_m
        intervals = (
            bank.photon_interval_index
            if truth_model == "photon"
            else bank.su2_interval_index
        )
        for index in np.unique(indices):
            if index < 0 or index >= len(ctaus):
                raise ValueError(f"{truth_model} truth index {index} is out of range.")
            rows.append(
                {
                    "mass_GeV": float(bank.mass_gev),
                    "truth_model": truth_model,
                    "truth_lifetime_index": int(index),
                    "truth_interval_index": int(intervals[index]),
                    "truth_ctau_m": float(ctaus[index]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["truth_model", "truth_lifetime_index"], ignore_index=True
    )


def estimate_event_scale_from_distance(
    minimum_distance: float,
    settings: AdaptivePseudoexperimentSettings,
) -> int:
    """Return a conservative order-of-magnitude seed for range finding.

    The inverse-square scaling is only a planner.  The actual threshold is
    always obtained from profiled pseudoexperiments.
    """
    distance = float(minimum_distance)
    if not np.isfinite(distance) or distance <= 0.0:
        return settings.rangefinder_maximum_events
    estimate = int(ceil(settings.rangefinder_scale_constant / distance**2))
    return int(
        np.clip(
            estimate,
            settings.rangefinder_minimum_events,
            settings.rangefinder_maximum_events,
        )
    )


def rangefinder_event_grid(
    estimated_events: int,
    settings: AdaptivePseudoexperimentSettings,
) -> np.ndarray:
    estimate = max(settings.rangefinder_minimum_events, int(estimated_events))
    factors = np.asarray([0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0])
    values = np.rint(estimate * factors).astype(int)
    values = np.clip(
        values,
        settings.rangefinder_minimum_events,
        settings.rangefinder_maximum_events,
    )
    values = np.unique(values)
    if values[0] > settings.rangefinder_minimum_events:
        values = np.insert(values, 0, settings.rangefinder_minimum_events)
    return values


def rangefinder_bracket(
    conservative_curve: pd.DataFrame,
    settings: AdaptivePseudoexperimentSettings,
) -> RangefinderBracket:
    required = {"number_of_events", "worst_case_correct_fraction"}
    missing = required - set(conservative_curve.columns)
    if missing:
        raise ValueError(f"Range-finder curve is missing columns: {sorted(missing)}")
    curve = conservative_curve.sort_values("number_of_events", ignore_index=True)
    threshold = minimum_persistent_events(
        curve,
        accuracy_column="worst_case_correct_fraction",
        target_accuracy=settings.target_accuracy,
    )
    events = curve["number_of_events"].to_numpy(dtype=int)
    accuracy = curve["worst_case_correct_fraction"].to_numpy(dtype=float)
    if threshold is None:
        last = int(events[-1])
        return RangefinderBracket(
            lower_failing_events=last,
            upper_passing_events=min(
                settings.rangefinder_maximum_events,
                max(last + 1, int(ceil(last * 2.0))),
            ),
            estimated_crossing_events=float(last * 1.5),
            threshold_reached=False,
        )
    passing_index = int(np.flatnonzero(events == threshold)[0])
    if passing_index == 0:
        lower = max(settings.rangefinder_minimum_events, int(floor(threshold / 2)))
        lower_accuracy = 0.0
    else:
        lower = int(events[passing_index - 1])
        lower_accuracy = float(accuracy[passing_index - 1])
    upper = int(threshold)
    upper_accuracy = float(accuracy[passing_index])
    denominator = upper_accuracy - lower_accuracy
    if denominator <= 0.0:
        estimate = 0.5 * (lower + upper)
    else:
        fraction = (settings.target_accuracy - lower_accuracy) / denominator
        estimate = lower + np.clip(fraction, 0.0, 1.0) * (upper - lower)
    return RangefinderBracket(
        lower_failing_events=lower,
        upper_passing_events=upper,
        estimated_crossing_events=float(estimate),
        threshold_reached=True,
    )


def final_event_grid_from_bracket(
    bracket: RangefinderBracket,
    settings: AdaptivePseudoexperimentSettings,
) -> np.ndarray:
    """Create one cache-stable final grid before the full-domain pilot.

    A unit-spaced window surrounds the interpolated crossing while a sparse
    persistence tail extends well beyond the passing range.  Every later PE
    level reuses this exact event grid, avoiding the expensive second unit-grid
    run required during the manual m_a=0.3 validation.
    """
    lower_fail = int(bracket.lower_failing_events)
    upper_pass = int(bracket.upper_passing_events)
    width = max(1, upper_pass - lower_fail)
    half_width = max(
        settings.unit_window_minimum_half_width,
        int(ceil(settings.unit_window_bracket_fraction * width)),
    )
    centre = int(round(bracket.estimated_crossing_events))
    unit_lower = max(1, centre - half_width)
    unit_upper = centre + half_width

    if unit_upper - unit_lower + 1 > settings.maximum_unit_window_points:
        half = (settings.maximum_unit_window_points - 1) // 2
        unit_lower = max(1, centre - half)
        unit_upper = unit_lower + settings.maximum_unit_window_points - 1

    unit_lower = min(unit_lower, lower_fail)
    unit_upper = max(unit_upper, upper_pass)
    tail_stop = max(
        unit_upper + settings.persistence_tail_minimum_extra,
        int(ceil(upper_pass * settings.persistence_tail_factor)),
    )
    tail_stop = min(tail_stop, settings.rangefinder_maximum_events)
    tail_step = max(5, int(round(max(upper_pass, 50) * 0.05)))
    tail_start = unit_upper + tail_step
    unit = np.arange(unit_lower, unit_upper + 1, dtype=int)
    tail = (
        np.arange(tail_start, tail_stop + 1, tail_step, dtype=int)
        if tail_start <= tail_stop
        else np.asarray([], dtype=int)
    )
    values = np.unique(np.concatenate([unit, tail, [tail_stop]]))
    return values[values >= 1]


def event_grid_specification(event_counts: Sequence[int]) -> str:
    """Serialize sorted integers into compact inclusive CLI ranges."""
    values = np.unique(np.asarray(event_counts, dtype=int))
    if values.ndim != 1 or len(values) == 0 or np.any(values < 1):
        raise ValueError("Event counts must be a non-empty positive integer vector.")
    pieces: list[str] = []
    start = int(values[0])
    previous = int(values[0])
    step: int | None = None
    for value_raw in values[1:]:
        value = int(value_raw)
        delta = value - previous
        if step is None:
            step = delta
        elif delta != step:
            if start == previous:
                pieces.append(str(start))
            elif step == 1:
                pieces.append(f"{start}:{previous}")
            else:
                pieces.append(f"{start}:{previous}:{step}")
            start = previous = value
            step = None
            continue
        previous = value
    if start == previous:
        pieces.append(str(start))
    elif step == 1:
        pieces.append(f"{start}:{previous}")
    else:
        pieces.append(f"{start}:{previous}:{step}")
    return ",".join(pieces)


def _truth_ranking_against_envelope(
    detailed: pd.DataFrame,
    conservative_curve: pd.DataFrame,
    event_counts: Iterable[int],
) -> pd.DataFrame:
    counts = set(int(value) for value in event_counts)
    selected = detailed.loc[
        detailed["number_of_events"].astype(int).isin(counts)
    ].copy()
    envelope = conservative_curve.loc[
        conservative_curve["number_of_events"].astype(int).isin(counts),
        ["number_of_events", "worst_case_correct_fraction"],
    ].rename(
        columns={"worst_case_correct_fraction": "conservative_accuracy"}
    )
    merged = selected.merge(envelope, on="number_of_events", how="inner")
    merged["accuracy_gap"] = (
        merged["correct_fraction"] - merged["conservative_accuracy"]
    )
    ranking = (
        merged.groupby(
            [
                "truth_model",
                "truth_lifetime_index",
                "truth_interval_index",
                "truth_ctau_m",
            ],
            as_index=False,
        )
        .agg(
            minimum_accuracy_gap=("accuracy_gap", "min"),
            minimum_accuracy_over_seeds_and_N=("correct_fraction", "min"),
            mean_accuracy_over_seeds_and_N=("correct_fraction", "mean"),
        )
        .sort_values(
            ["minimum_accuracy_gap", "truth_model", "truth_lifetime_index"],
            ignore_index=True,
        )
    )
    return ranking


def select_hard_truth_indices(
    bank: LifetimeTemplateBank,
    detailed: pd.DataFrame,
    conservative_curve: pd.DataFrame,
    distances: np.ndarray,
    event_counts: Sequence[int],
    settings: AdaptivePseudoexperimentSettings,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Select high-statistics truths from the full-domain pilot.

    Selection is driven by closeness to the conservative envelope throughout
    the critical persistence window.  Distance-map minima, their neighbours and
    connected-interval endpoints are mandatory anchors.  Per-model caps guard
    runtime, while the omitted-truth audit remains the final correctness check.
    """
    ranking = _truth_ranking_against_envelope(
        detailed, conservative_curve, event_counts
    )
    anchors = distance_screening_truth_indices(bank, distances, settings)
    result: dict[str, np.ndarray] = {}
    ranking = ranking.copy()
    ranking["selected_for_high_statistics"] = False
    ranking["selection_reason"] = "omitted_pending_audit"

    lengths = {"photon": len(bank.photon_ctau_m), "su2": len(bank.su2_ctau_m)}
    for truth_model in TRUTH_MODELS:
        model_ranking = ranking.loc[ranking["truth_model"] == truth_model].copy()
        selected = set(
            int(index)
            for index in model_ranking.loc[
                model_ranking["minimum_accuracy_gap"]
                <= settings.hard_truth_accuracy_gap,
                "truth_lifetime_index",
            ]
        )
        selected.update(int(index) for index in anchors[truth_model])

        minimum_count = min(
            lengths[truth_model], settings.minimum_hard_truths_per_model
        )
        if len(selected) < minimum_count:
            selected.update(
                int(index)
                for index in model_ranking.head(minimum_count)[
                    "truth_lifetime_index"
                ]
            )

        maximum_count = max(
            minimum_count,
            int(ceil(
                settings.maximum_hard_truth_fraction_per_model
                * lengths[truth_model]
            )),
        )
        if len(selected) > maximum_count:
            mandatory = set(int(index) for index in anchors[truth_model])
            ordered = [
                int(index)
                for index in model_ranking["truth_lifetime_index"]
                if int(index) not in mandatory
            ]
            keep = set(mandatory)
            for index in ordered:
                if len(keep) >= maximum_count:
                    break
                keep.add(index)
            selected = keep

        selected_array = np.asarray(sorted(selected), dtype=int)
        result[truth_model] = selected_array
        mask = (
            (ranking["truth_model"] == truth_model)
            & ranking["truth_lifetime_index"].astype(int).isin(selected)
        )
        ranking.loc[mask, "selected_for_high_statistics"] = True
        anchor_mask = mask & ranking["truth_lifetime_index"].astype(int).isin(
            set(int(index) for index in anchors[truth_model])
        )
        ranking.loc[mask, "selection_reason"] = "competitive_accuracy_gap"
        ranking.loc[anchor_mask, "selection_reason"] = "distance_or_endpoint_anchor"
    return result, ranking


def _wald_one_sided_lower(
    probabilities: np.ndarray,
    pseudoexperiments: np.ndarray,
    z_value: float,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    n = np.asarray(pseudoexperiments, dtype=float)
    standard_error = np.sqrt(np.clip(p * (1.0 - p) / n, 0.0, None))
    return np.clip(p - z_value * standard_error, 0.0, 1.0)


def _wald_one_sided_upper(
    probabilities: np.ndarray,
    pseudoexperiments: np.ndarray,
    z_value: float,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    n = np.asarray(pseudoexperiments, dtype=float)
    standard_error = np.sqrt(np.clip(p * (1.0 - p) / n, 0.0, None))
    return np.clip(p + z_value * standard_error, 0.0, 1.0)


def audit_omitted_truths(
    omitted_detailed: pd.DataFrame,
    selected_conservative_curve: pd.DataFrame,
    *,
    total_truth_count: int,
    number_of_seeds: int,
    global_alpha: float,
) -> OmittedTruthAudit:
    """Certify omitted truths against the selected high-statistics envelope.

    The family size deliberately uses every truth, seed and tested event count,
    matching the conservative convention validated at m_a=0.3 GeV.
    """
    required_detailed = {
        "truth_model",
        "truth_lifetime_index",
        "truth_interval_index",
        "truth_ctau_m",
        "seed",
        "number_of_events",
        "number_of_pseudoexperiments",
        "correct_fraction",
    }
    missing = required_detailed - set(omitted_detailed.columns)
    if missing:
        raise ValueError(f"Omitted-truth table is missing: {sorted(missing)}")
    required_curve = {"number_of_events", "worst_case_correct_fraction"}
    missing_curve = required_curve - set(selected_conservative_curve.columns)
    if missing_curve:
        raise ValueError(f"Selected curve is missing: {sorted(missing_curve)}")
    n_event_points = selected_conservative_curve["number_of_events"].nunique()
    n_bounds = int(total_truth_count) * int(number_of_seeds) * int(n_event_points)
    if n_bounds < 1:
        raise ValueError("Simultaneous-bound family must be non-empty.")
    z_value = float(norm.ppf(1.0 - float(global_alpha) / n_bounds))

    curve = selected_conservative_curve[
        ["number_of_events", "worst_case_correct_fraction"]
    ].rename(
        columns={
            "worst_case_correct_fraction": "selected_conservative_accuracy"
        }
    )
    points = omitted_detailed.merge(
        curve, on="number_of_events", how="inner", validate="many_to_one"
    )
    points["global_one_sided_lower_bound"] = _wald_one_sided_lower(
        points["correct_fraction"].to_numpy(dtype=float),
        points["number_of_pseudoexperiments"].to_numpy(dtype=float),
        z_value,
    )
    points["global_lower_margin"] = (
        points["global_one_sided_lower_bound"]
        - points["selected_conservative_accuracy"]
    )
    points["overlaps_selected_envelope"] = points["global_lower_margin"] <= 0.0

    rows: list[dict] = []
    group_columns = [
        "truth_model",
        "truth_lifetime_index",
        "truth_interval_index",
        "truth_ctau_m",
    ]
    for keys, group in points.groupby(group_columns, sort=False):
        ordered = group.sort_values(
            ["global_lower_margin", "number_of_events", "seed"],
            kind="mergesort",
            ignore_index=True,
        )
        worst = ordered.iloc[0]
        rows.append(
            {
                "truth_model": keys[0],
                "truth_lifetime_index": int(keys[1]),
                "truth_interval_index": int(keys[2]),
                "truth_ctau_m": float(keys[3]),
                "available_pseudoexperiments": int(
                    group["number_of_pseudoexperiments"].min()
                ),
                "minimum_global_lower_margin": float(
                    worst["global_lower_margin"]
                ),
                "number_of_overlapping_points": int(
                    group["overlaps_selected_envelope"].sum()
                ),
                "worst_seed": int(worst["seed"]),
                "worst_number_of_events": int(worst["number_of_events"]),
                "requires_promotion": bool(
                    group["overlaps_selected_envelope"].any()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        [
            "requires_promotion",
            "minimum_global_lower_margin",
            "truth_model",
            "truth_lifetime_index",
        ],
        ascending=[False, True, True, True],
        ignore_index=True,
    )
    overlapping = summary.loc[summary["requires_promotion"]].copy()
    return OmittedTruthAudit(
        point_table=points,
        truth_summary=summary,
        simultaneous_bounds=n_bounds,
        adjusted_z=z_value,
        overlapping_truths=overlapping,
    )


def merge_truth_indices(
    first: Mapping[str, Sequence[int]],
    second: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    return {
        model: np.unique(
            np.concatenate(
                [
                    np.asarray(first[model], dtype=int),
                    np.asarray(second[model], dtype=int),
                ]
            )
        )
        for model in TRUTH_MODELS
    }


def omitted_truth_indices(
    bank: LifetimeTemplateBank,
    selected: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    lengths = {"photon": len(bank.photon_ctau_m), "su2": len(bank.su2_ctau_m)}
    result = {}
    for model in TRUTH_MODELS:
        all_indices = np.arange(lengths[model], dtype=int)
        result[model] = np.setdiff1d(
            all_indices, np.asarray(selected[model], dtype=int), assume_unique=False
        )
    return result


def threshold_history_is_stable(
    thresholds: Sequence[int | None],
    settings: AdaptivePseudoexperimentSettings,
) -> bool:
    finite = [int(value) for value in thresholds if value is not None]
    needed = settings.required_stable_transitions + 1
    if len(finite) < needed:
        return False
    recent = finite[-needed:]
    return all(
        abs(right - left) <= settings.threshold_stability_events
        for left, right in zip(recent, recent[1:])
    )


def _persistent_from_boolean(
    event_counts: np.ndarray,
    passing: np.ndarray,
) -> int | None:
    persistent = np.logical_and.accumulate(passing[::-1])[::-1]
    indices = np.flatnonzero(persistent)
    return None if len(indices) == 0 else int(event_counts[indices[0]])


def monte_carlo_threshold_diagnostics(
    detailed: pd.DataFrame,
    conservative_curve: pd.DataFrame,
    *,
    target_accuracy: float,
    global_alpha: float,
    total_truth_count: int,
    number_of_seeds: int,
) -> MonteCarloThresholdDiagnostics:
    """Return point, local-slope and simultaneous threshold diagnostics."""
    curve = conservative_curve.sort_values("number_of_events", ignore_index=True)
    point = minimum_persistent_events(
        curve,
        accuracy_column="worst_case_correct_fraction",
        target_accuracy=target_accuracy,
    )
    previous_events = previous_accuracy = accuracy_at_point = None
    local_sigma = None
    if point is not None:
        location = int(np.flatnonzero(
            curve["number_of_events"].to_numpy(dtype=int) == point
        )[0])
        accuracy_at_point = float(
            curve.iloc[location]["worst_case_correct_fraction"]
        )
        if location > 0:
            previous_events = int(curve.iloc[location - 1]["number_of_events"])
            previous_accuracy = float(
                curve.iloc[location - 1]["worst_case_correct_fraction"]
            )
            delta_n = point - previous_events
            slope = (accuracy_at_point - previous_accuracy) / delta_n
            limiter = curve.iloc[location]
            matching = detailed.loc[
                (detailed["number_of_events"].astype(int) == point)
                & (detailed["seed"].astype(int) == int(limiter["limiting_seed"]))
                & (detailed["truth_model"] == limiter["limiting_truth_model"])
                & (
                    detailed["truth_lifetime_index"].astype(int)
                    == int(limiter["limiting_truth_lifetime_index"])
                )
            ]
            if slope > 0.0 and not matching.empty:
                n_pe = float(matching.iloc[0]["number_of_pseudoexperiments"])
                sigma_p = sqrt(
                    accuracy_at_point * (1.0 - accuracy_at_point) / n_pe
                )
                local_sigma = sigma_p / slope

    events = curve["number_of_events"].to_numpy(dtype=int)
    n_event_points = len(events)
    n_bounds = total_truth_count * number_of_seeds * n_event_points
    z = float(norm.ppf(1.0 - global_alpha / n_bounds))
    grouped = detailed.groupby("number_of_events", sort=True)
    lower_values = []
    upper_values = []
    for event in events:
        group = grouped.get_group(int(event))
        p = group["correct_fraction"].to_numpy(dtype=float)
        n = group["number_of_pseudoexperiments"].to_numpy(dtype=float)
        lower_values.append(float(np.min(_wald_one_sided_lower(p, n, z))))
        upper_values.append(float(np.min(_wald_one_sided_upper(p, n, z))))
    lower_values_array = np.asarray(lower_values)
    upper_values_array = np.asarray(upper_values)
    sufficient = _persistent_from_boolean(
        events, lower_values_array >= target_accuracy
    )
    insufficient_points = events[upper_values_array < target_accuracy]
    lower_event = (
        None if len(insufficient_points) == 0 else int(insufficient_points.max() + 1)
    )
    return MonteCarloThresholdDiagnostics(
        point_estimate=point,
        previous_tested_events=previous_events,
        previous_accuracy=previous_accuracy,
        accuracy_at_point=accuracy_at_point,
        local_sigma_events=(None if local_sigma is None else float(local_sigma)),
        simultaneous_lower_events=lower_event,
        simultaneous_upper_events=sufficient,
        simultaneous_interval_closed=(
            lower_event is not None and sufficient is not None
        ),
    )


def result_row(
    *,
    mass_gev: float,
    selection_name: str,
    status: str,
    threshold: int | None,
    diagnostics: MonteCarloThresholdDiagnostics | None,
    final_pseudoexperiments: int,
    selected_truths: Mapping[str, Sequence[int]],
    omitted_truths: Mapping[str, Sequence[int]],
    number_of_energy_bins: int,
    minimum_distance: float,
    limiting_row: Mapping | None,
    runtime_seconds: float,
    lifetime_rounds: int,
    profile_lifetime_counts: Mapping[str, int],
    audit: OmittedTruthAudit | None,
) -> dict:
    limiting_row = {} if limiting_row is None else dict(limiting_row)
    return {
        "mass_GeV": float(mass_gev),
        "selection_name": selection_name,
        "N90": -1 if threshold is None else int(threshold),
        "N90_mc_lower": (
            -1
            if diagnostics is None or diagnostics.simultaneous_lower_events is None
            else int(diagnostics.simultaneous_lower_events)
        ),
        "N90_mc_upper": (
            -1
            if diagnostics is None or diagnostics.simultaneous_upper_events is None
            else int(diagnostics.simultaneous_upper_events)
        ),
        "local_mc_sigma_events": (
            np.nan
            if diagnostics is None or diagnostics.local_sigma_events is None
            else float(diagnostics.local_sigma_events)
        ),
        "convergence_status": status,
        "final_PE_count": int(final_pseudoexperiments),
        "number_of_selected_photon_truths": len(selected_truths["photon"]),
        "number_of_selected_su2_truths": len(selected_truths["su2"]),
        "number_of_omitted_photon_truths": len(omitted_truths["photon"]),
        "number_of_omitted_su2_truths": len(omitted_truths["su2"]),
        "number_of_photon_profile_lifetimes": int(
            profile_lifetime_counts["photon"]
        ),
        "number_of_su2_profile_lifetimes": int(profile_lifetime_counts["su2"]),
        "number_of_energy_bins": int(number_of_energy_bins),
        "minimum_D_TV": float(minimum_distance),
        "limiting_truth_model": limiting_row.get("limiting_truth_model", ""),
        "limiting_truth_lifetime_index": int(
            limiting_row.get("limiting_truth_lifetime_index", -1)
        ),
        "limiting_truth_ctau_m": float(
            limiting_row.get("limiting_truth_ctau_m", np.nan)
        ),
        "limiting_seed": int(limiting_row.get("limiting_seed", -1)),
        "accuracy_at_threshold": float(
            limiting_row.get("worst_case_correct_fraction", np.nan)
        ),
        "audit_simultaneous_bounds": (
            0 if audit is None else int(audit.simultaneous_bounds)
        ),
        "minimum_omitted_lower_margin": (
            np.nan
            if audit is None or audit.truth_summary.empty
            else float(audit.truth_summary.iloc[0]["minimum_global_lower_margin"])
        ),
        "lifetime_refinement_rounds": int(lifetime_rounds),
        "runtime_seconds": float(runtime_seconds),
    }


def write_settings_json(settings: AdaptiveWeek8Settings, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.as_dict(), indent=2, sort_keys=True) + "\n")
