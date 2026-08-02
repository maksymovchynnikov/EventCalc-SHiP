"""Run lifetime-profiled shape-only pseudoexperiments for the Week-7 ALP study.

For every saved detector-level template bank and every lifetime template taken
as the truth, this script generates fixed-size pseudo-datasets and evaluates

    T = 2 * [max_ctau_W log L_W - max_ctau_gamma log L_gamma],

with

    log L_H = sum_i n_i log p_H,i(ctau_H).

The lifetime is profiled independently under the photophilic and SU(2)_L
hypotheses on their saved discrete grids.  The classification convention is

    T < 0  -> ALP-photon,
    T > 0  -> ALP-SU(2)_L,

with numerical ties split equally.  The analysis remains shape-only: the
observed event count N is conditioned on, so expected rates are not included
in the likelihood.

The script does not launch EventCalc.  It reads the ``.npz`` banks produced by
``analysis.lifetime_blind_discrimination`` and validated by
``analysis.lifetime_blind_distance_maps``.

Run from the repository root with

    python -m analysis.lifetime_blind_profiled_likelihood

Useful staged commands
----------------------

    python -m analysis.lifetime_blind_profiled_likelihood --self-test

    python -m analysis.lifetime_blind_profiled_likelihood \
        --masses 0.3 --max-events 8 --pseudoexperiments 2000 --overwrite

    python -m analysis.lifetime_blind_profiled_likelihood \
        --max-events 20 --pseudoexperiments 20000 --overwrite

    python -m analysis.lifetime_blind_profiled_likelihood \
        --max-events 20 --pseudoexperiments 100000 \
        --number-of-seeds 5 --overwrite

    python -m analysis.lifetime_blind_profiled_likelihood \
        --masses 0.75 1.0 --truth-grid odd --profile-grid even \
        --output-dir analysis/lifetime_blind_discrimination/profiled_off_grid \
        --overwrite

    python -m analysis.lifetime_blind_profiled_likelihood \
        --rebin-factor 2 --output-dir \
        analysis/lifetime_blind_discrimination/profiled_likelihood_rebin2 \
        --overwrite

    python -m analysis.lifetime_blind_profiled_likelihood \
        --masses 0.75 1.0 --jeffreys-alpha 0.25 \
        --output-dir analysis/lifetime_blind_discrimination/profiled_alpha_0p25 \
        --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent

if (REPO_ROOT / "analysis").is_dir():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

INPUT_DIR = ANALYSIS_DIR / "lifetime_blind_discrimination" / "template_banks"
OUTPUT_DIR = ANALYSIS_DIR / "lifetime_blind_discrimination" / "profiled_likelihood"

DEFAULT_MAXIMUM_EVENTS = 20
DEFAULT_PSEUDOEXPERIMENTS = 20_000
DEFAULT_BASE_SEED = 73_241
DEFAULT_NUMBER_OF_SEEDS = 1
DEFAULT_SEED_STEP = 10_003
DEFAULT_CHUNK_SIZE = 5_000
DEFAULT_TARGET_ACCURACY = 0.90
DEFAULT_TIE_TOLERANCE = 1.0e-12
DEFAULT_REBIN_FACTOR = 1
DEFAULT_TRUTH_GRID = "all"
DEFAULT_PROFILE_GRID = "all"

TRUTH_MODELS = ("photon", "su2")


def _report_style_functions() -> tuple[Callable[[], None], Callable[[plt.Axes], None]]:
    """Load the shared report style, with a fallback for standalone self-tests."""
    try:
        from analysis.plot_style import style_axis, use_report_style
    except ModuleNotFoundError:
        return (lambda: None), (lambda axis: None)
    return use_report_style, style_axis


def _bank_helpers() -> tuple[Callable, Callable]:
    """Import the existing strict bank loader and path selector."""
    try:
        from analysis.lifetime_blind_distance_maps import (
            load_template_bank,
            select_bank_paths,
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Could not import analysis.lifetime_blind_distance_maps. "
            "Place this file in the repository's analysis/ directory."
        ) from error
    return load_template_bank, select_bank_paths


def mass_token(mass_gev: float) -> str:
    """Return the filename token used by the template-bank builder."""
    return f"{mass_gev:g}".replace("-", "m").replace(".", "p")


# -----------------------------------------------------------------------------
# Jeffreys-smoothing reconstruction
# -----------------------------------------------------------------------------


def add_smoothing_metadata(bank: dict, path: Path) -> dict:
    """Attach the metadata needed to vary the saved Jeffreys pseudocount.

    The strict loader in ``lifetime_blind_distance_maps`` intentionally keeps
    only the arrays needed for distance calculations.  The compact bank also
    stores the original Jeffreys alpha and one total effective sample size per
    lifetime template.  Together with the saved smoothed probabilities, these
    quantities are sufficient to reconstruct the pre-smoothing probabilities
    exactly for the smoothing prescription used by the template builder.
    """
    with np.load(path, allow_pickle=False) as raw:
        required = {
            "jeffreys_alpha",
            "photon_total_n_eff",
            "su2_total_n_eff",
        }
        missing = required - set(raw.files)
        if missing:
            raise ValueError(
                f"Missing smoothing metadata in {path}: {sorted(missing)}"
            )

        enriched = dict(bank)
        enriched["stored_jeffreys_alpha"] = float(
            np.asarray(raw["jeffreys_alpha"]).item()
        )
        enriched["jeffreys_alpha"] = enriched["stored_jeffreys_alpha"]
        enriched["photon_total_n_eff"] = np.asarray(
            raw["photon_total_n_eff"],
            dtype=float,
        )
        enriched["su2_total_n_eff"] = np.asarray(
            raw["su2_total_n_eff"],
            dtype=float,
        )

    stored_alpha = float(enriched["stored_jeffreys_alpha"])
    if not np.isfinite(stored_alpha) or stored_alpha <= 0.0:
        raise ValueError(f"Invalid stored Jeffreys alpha in {path}: {stored_alpha}")

    for prefix in ("photon", "su2"):
        probabilities = np.asarray(
            enriched[f"{prefix}_probabilities"],
            dtype=float,
        )
        total_n_eff = np.asarray(
            enriched[f"{prefix}_total_n_eff"],
            dtype=float,
        )
        if total_n_eff.shape != (probabilities.shape[0],):
            raise ValueError(
                f"{prefix} total N_eff has the wrong shape in {path}: "
                f"{total_n_eff.shape}, expected {(probabilities.shape[0],)}"
            )
        if np.any(~np.isfinite(total_n_eff)) or np.any(total_n_eff <= 0.0):
            raise ValueError(f"Invalid {prefix} total N_eff values in {path}.")

    return enriched


def _resmooth_probability_matrix(
    *,
    smoothed_probabilities: np.ndarray,
    total_n_eff: np.ndarray,
    stored_alpha: float,
    target_alpha: float,
    label: str,
) -> np.ndarray:
    """Undo the stored pseudocount and apply a requested new value.

    The template builder used

        p_smooth = (N_eff * p_raw + alpha) /
                   (N_eff + alpha * K),

    where K is the number of energy bins.  Hence p_raw is recoverable from the
    saved p_smooth, N_eff, alpha, and K; raw histogram arrays are not required.
    """
    smoothed = np.asarray(smoothed_probabilities, dtype=float)
    n_eff = np.asarray(total_n_eff, dtype=float)

    if smoothed.ndim != 2:
        raise ValueError(f"{label} probabilities must be two-dimensional.")
    if n_eff.shape != (smoothed.shape[0],):
        raise ValueError(f"{label} total N_eff has an inconsistent shape.")
    if stored_alpha <= 0.0 or target_alpha <= 0.0:
        raise ValueError("Jeffreys alpha values must be positive.")

    # Preserve the central bank bit-for-bit when no variation is requested.
    if target_alpha == stored_alpha:
        return smoothed.copy()

    number_of_bins = smoothed.shape[1]
    n_eff_column = n_eff[:, np.newaxis]

    raw = (
        smoothed * (n_eff_column + stored_alpha * number_of_bins)
        - stored_alpha
    ) / n_eff_column

    # Only round-off-sized excursions are permitted.  The adaptive binning in
    # the builder makes every raw model probability strictly positive.
    minimum_raw = float(np.min(raw))
    if minimum_raw < -1.0e-12:
        raise RuntimeError(
            f"Could not reconstruct valid raw {label} probabilities; "
            f"minimum value is {minimum_raw:.6g}."
        )
    raw = np.clip(raw, 0.0, None)
    raw /= raw.sum(axis=1, keepdims=True)

    resmoothed = (
        n_eff_column * raw + target_alpha
    ) / (
        n_eff_column + target_alpha * number_of_bins
    )
    resmoothed /= resmoothed.sum(axis=1, keepdims=True)

    if np.any(~np.isfinite(resmoothed)) or np.any(resmoothed <= 0.0):
        raise RuntimeError(f"Invalid resmoothed {label} probabilities.")
    if not np.allclose(
        resmoothed.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(f"Resmoothed {label} probabilities are not normalized.")

    return resmoothed


def resmooth_template_bank(bank: dict, target_alpha: float | None) -> dict:
    """Return a bank using either its stored or a requested Jeffreys alpha."""
    resmoothed = dict(bank)
    stored_alpha = float(bank["stored_jeffreys_alpha"])
    alpha = stored_alpha if target_alpha is None else float(target_alpha)

    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("The requested Jeffreys alpha must be positive.")

    for prefix, label in (("photon", "photon"), ("su2", "SU(2)_L")):
        resmoothed[f"{prefix}_probabilities"] = _resmooth_probability_matrix(
            smoothed_probabilities=np.asarray(
                bank[f"{prefix}_probabilities"],
                dtype=float,
            ),
            total_n_eff=np.asarray(
                bank[f"{prefix}_total_n_eff"],
                dtype=float,
            ),
            stored_alpha=stored_alpha,
            target_alpha=alpha,
            label=label,
        )

    resmoothed["jeffreys_alpha"] = alpha
    return resmoothed


# -----------------------------------------------------------------------------
# Energy-bin coarse graining
# -----------------------------------------------------------------------------


def rebin_template_bank(bank: dict, factor: int) -> dict:
    """Merge consecutive energy bins while preserving each template's norm.

    A factor of one returns a copy of the original bank.  If the number of
    original bins is not divisible by ``factor``, the final coarse bin contains
    the remaining bins.  This is an exact coarse graining of the saved
    probabilities; it does not rerun EventCalc or reapply smoothing.
    """
    if factor < 1:
        raise ValueError("Rebin factor must be a positive integer.")

    rebinned = dict(bank)
    edges = np.asarray(bank["energy_edges_GeV"], dtype=float)
    photon = np.asarray(bank["photon_probabilities"], dtype=float)
    su2 = np.asarray(bank["su2_probabilities"], dtype=float)

    number_of_bins = len(edges) - 1
    if photon.ndim != 2 or su2.ndim != 2:
        raise ValueError("Template probabilities must be two-dimensional.")
    if photon.shape[1] != number_of_bins or su2.shape[1] != number_of_bins:
        raise ValueError("Energy edges and template arrays are inconsistent.")

    if factor == 1:
        rebinned["energy_edges_GeV"] = edges.copy()
        rebinned["photon_probabilities"] = photon.copy()
        rebinned["su2_probabilities"] = su2.copy()
        return rebinned

    starts = np.arange(0, number_of_bins, factor, dtype=int)
    coarse_edges = np.concatenate((edges[starts], edges[-1:]))
    coarse_photon = np.add.reduceat(photon, starts, axis=1)
    coarse_su2 = np.add.reduceat(su2, starts, axis=1)

    # Protect against tiny floating-point normalization drift.
    coarse_photon /= coarse_photon.sum(axis=1, keepdims=True)
    coarse_su2 /= coarse_su2.sum(axis=1, keepdims=True)

    if np.any(coarse_photon <= 0.0) or np.any(coarse_su2 <= 0.0):
        raise ValueError(
            "Rebinned templates contain non-positive probabilities; "
            "the saved bank is not suitable for log-likelihood evaluation."
        )

    rebinned["energy_edges_GeV"] = coarse_edges
    rebinned["photon_probabilities"] = coarse_photon
    rebinned["su2_probabilities"] = coarse_su2
    return rebinned


# -----------------------------------------------------------------------------
# Lifetime-grid selection
# -----------------------------------------------------------------------------


def lifetime_grid_indices(length: int, mode: str) -> np.ndarray:
    """Return original template indices selected by a lifetime-grid mode.

    ``even`` and ``odd`` refer to zero-based indices in the saved bank.  For a
    41-point dense grid, ``even`` therefore contains 21 profiling templates
    including both endpoints, while ``odd`` contains the 20 interleaved truth
    templates.
    """
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


# -----------------------------------------------------------------------------
# Profile-likelihood engine
# -----------------------------------------------------------------------------


def _profile_log_likelihoods(
    sampled_bins: np.ndarray,
    log_templates: np.ndarray,
    event_indices: np.ndarray,
) -> np.ndarray:
    """Return the maximum log likelihood for every pseudoexperiment and N."""
    contributions = log_templates[:, sampled_bins]
    np.cumsum(contributions, axis=2, out=contributions)
    selected = contributions[:, :, event_indices]
    return np.max(selected, axis=0)


def _stable_truth_rng(
    *,
    seed: int,
    mass_gev: float,
    truth_model: str,
    truth_index: int,
) -> np.random.Generator:
    """Build an order-independent random generator for one truth template."""
    if truth_model not in TRUTH_MODELS:
        raise ValueError(f"Unknown truth model: {truth_model}")
    mass_hash = zlib.crc32(f"{mass_gev:.16g}".encode("ascii"))
    model_code = 0 if truth_model == "photon" else 1
    sequence = np.random.SeedSequence(
        [int(seed), int(mass_hash), model_code, int(truth_index)]
    )
    return np.random.default_rng(sequence)


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
) -> pd.DataFrame:
    """Run all event counts for one true lifetime template."""
    event_counts = np.asarray(event_counts, dtype=int)
    truth_probabilities = np.asarray(truth_probabilities, dtype=float)
    photon_probabilities = np.asarray(photon_probabilities, dtype=float)
    su2_probabilities = np.asarray(su2_probabilities, dtype=float)

    if event_counts.ndim != 1 or len(event_counts) == 0:
        raise ValueError("At least one event count is required.")
    if np.any(event_counts <= 0) or np.any(np.diff(event_counts) <= 0):
        raise ValueError("event_counts must be positive, unique, and increasing.")
    if number_of_pseudoexperiments <= 0 or chunk_size <= 0:
        raise ValueError("Pseudoexperiment count and chunk size must be positive.")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance cannot be negative.")

    number_of_bins = len(truth_probabilities)
    if photon_probabilities.shape[1] != number_of_bins:
        raise ValueError("Photon templates and truth use different energy bins.")
    if su2_probabilities.shape[1] != number_of_bins:
        raise ValueError("SU(2)_L templates and truth use different energy bins.")

    log_photon = np.log(photon_probabilities)
    log_su2 = np.log(su2_probabilities)
    maximum_events = int(event_counts[-1])
    event_indices = event_counts - 1
    number_of_counts = len(event_counts)

    correct_sum = np.zeros(number_of_counts, dtype=float)
    photon_selected_sum = np.zeros(number_of_counts, dtype=float)
    su2_selected_sum = np.zeros(number_of_counts, dtype=float)
    tie_sum = np.zeros(number_of_counts, dtype=np.int64)
    statistic_sum = np.zeros(number_of_counts, dtype=float)
    statistic_squared_sum = np.zeros(number_of_counts, dtype=float)

    rng = _stable_truth_rng(
        seed=seed,
        mass_gev=mass_gev,
        truth_model=truth_model,
        truth_index=truth_index,
    )

    processed = 0
    while processed < number_of_pseudoexperiments:
        current_chunk = min(chunk_size, number_of_pseudoexperiments - processed)
        sampled_bins = rng.choice(
            number_of_bins,
            size=(current_chunk, maximum_events),
            replace=True,
            p=truth_probabilities,
        )

        photon_best = _profile_log_likelihoods(
            sampled_bins,
            log_photon,
            event_indices,
        )
        su2_best = _profile_log_likelihoods(
            sampled_bins,
            log_su2,
            event_indices,
        )
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

    return pd.DataFrame(
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
        }
    )


def run_seed(
    bank: dict,
    *,
    event_counts: np.ndarray,
    number_of_pseudoexperiments: int,
    seed: int,
    chunk_size: int,
    tie_tolerance: float,
    truth_grid: str,
    profile_grid: str,
) -> pd.DataFrame:
    """Run selected truth lifetimes against independently selected fit grids."""
    mass_gev = float(bank["mass_GeV"])
    photon_ctau = np.asarray(bank["photon_ctau_m"], dtype=float)
    photon_all = np.asarray(bank["photon_probabilities"], dtype=float)
    su2_ctau = np.asarray(bank["su2_ctau_m"], dtype=float)
    su2_all = np.asarray(bank["su2_probabilities"], dtype=float)

    photon_truth_indices = lifetime_grid_indices(len(photon_ctau), truth_grid)
    su2_truth_indices = lifetime_grid_indices(len(su2_ctau), truth_grid)
    photon_profile_indices = lifetime_grid_indices(len(photon_ctau), profile_grid)
    su2_profile_indices = lifetime_grid_indices(len(su2_ctau), profile_grid)

    photon_profile = photon_all[photon_profile_indices]
    su2_profile = su2_all[su2_profile_indices]

    frames: list[pd.DataFrame] = []
    for truth_model, truth_ctaus, truth_templates, truth_indices in (
        ("photon", photon_ctau, photon_all, photon_truth_indices),
        ("su2", su2_ctau, su2_all, su2_truth_indices),
    ):
        for subset_position, truth_index in enumerate(truth_indices):
            truth_ctau_m = truth_ctaus[truth_index]
            print(
                f"    seed={seed}, truth={truth_model:6s} "
                f"{subset_position + 1:2d}/{len(truth_indices):2d}, "
                f"bank index={truth_index:2d}, c*tau={truth_ctau_m:.6g} m"
            )
            frames.append(
                simulate_truth_template(
                    mass_gev=mass_gev,
                    truth_model=truth_model,
                    truth_index=int(truth_index),
                    truth_ctau_m=float(truth_ctau_m),
                    truth_probabilities=truth_templates[truth_index],
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


# -----------------------------------------------------------------------------
# Worst-case reduction and persistent thresholds
# -----------------------------------------------------------------------------


def _limiting_row(group: pd.DataFrame) -> pd.Series:
    """Return the deterministic minimum-accuracy row in a truth subset."""
    return group.sort_values(
        ["correct_fraction", "truth_lifetime_index"],
        kind="mergesort",
    ).iloc[0]


def build_seed_worst_case_table(detailed: pd.DataFrame) -> pd.DataFrame:
    """Reduce all true lifetimes to conservative accuracies for each seed and N."""
    rows: list[dict] = []
    for (mass_gev, seed, number_of_events), group in detailed.groupby(
        ["mass_GeV", "seed", "number_of_events"],
        sort=True,
    ):
        photon_group = group.loc[group["truth_model"] == "photon"]
        su2_group = group.loc[group["truth_model"] == "su2"]
        if photon_group.empty or su2_group.empty:
            raise RuntimeError("Both truth models are required for aggregation.")

        photon_row = _limiting_row(photon_group)
        su2_row = _limiting_row(su2_group)
        global_row = _limiting_row(group)
        rows.append(
            {
                "mass_GeV": float(mass_gev),
                "seed": int(seed),
                "number_of_events": int(number_of_events),
                "photon_truth_worst_accuracy": float(photon_row["correct_fraction"]),
                "photon_limiting_lifetime_index": int(
                    photon_row["truth_lifetime_index"]
                ),
                "photon_limiting_ctau_m": float(photon_row["truth_ctau_m"]),
                "su2_truth_worst_accuracy": float(su2_row["correct_fraction"]),
                "su2_limiting_lifetime_index": int(su2_row["truth_lifetime_index"]),
                "su2_limiting_ctau_m": float(su2_row["truth_ctau_m"]),
                "worst_case_correct_fraction": float(global_row["correct_fraction"]),
                "limiting_truth_model": str(global_row["truth_model"]),
                "limiting_truth_lifetime_index": int(
                    global_row["truth_lifetime_index"]
                ),
                "limiting_truth_ctau_m": float(global_row["truth_ctau_m"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mass_GeV", "seed", "number_of_events"],
        ignore_index=True,
    )


def build_conservative_seed_envelope(seed_table: pd.DataFrame) -> pd.DataFrame:
    """Take the minimum accuracy over all validation seeds for every N."""
    rows: list[dict] = []
    for (mass_gev, number_of_events), group in seed_table.groupby(
        ["mass_GeV", "number_of_events"],
        sort=True,
    ):
        limiting = group.sort_values(
            ["worst_case_correct_fraction", "seed"],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "mass_GeV": float(mass_gev),
                "number_of_events": int(number_of_events),
                "photon_truth_worst_accuracy": float(
                    group["photon_truth_worst_accuracy"].min()
                ),
                "su2_truth_worst_accuracy": float(
                    group["su2_truth_worst_accuracy"].min()
                ),
                "worst_case_correct_fraction": float(
                    group["worst_case_correct_fraction"].min()
                ),
                "limiting_seed": int(limiting["seed"]),
                "limiting_truth_model": str(limiting["limiting_truth_model"]),
                "limiting_truth_lifetime_index": int(
                    limiting["limiting_truth_lifetime_index"]
                ),
                "limiting_truth_ctau_m": float(limiting["limiting_truth_ctau_m"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mass_GeV", "number_of_events"],
        ignore_index=True,
    )


def minimum_persistent_events(
    curve: pd.DataFrame,
    *,
    accuracy_column: str,
    target_accuracy: float,
) -> int | None:
    """First tested N from which every larger tested N passes the target."""
    ordered = curve.sort_values("number_of_events")
    values = ordered[accuracy_column].to_numpy(dtype=float)
    event_counts = ordered["number_of_events"].to_numpy(dtype=int)
    passing = values >= target_accuracy
    persistent = np.logical_and.accumulate(passing[::-1])[::-1]
    indices = np.flatnonzero(persistent)
    return None if len(indices) == 0 else int(event_counts[indices[0]])


def summarize_mass_threshold(
    *,
    bank: dict,
    conservative_curve: pd.DataFrame,
    target_accuracy: float,
    number_of_pseudoexperiments: int,
    number_of_seeds: int,
    rebin_factor: int,
    truth_grid: str,
    profile_grid: str,
    jeffreys_alpha: float,
    stored_jeffreys_alpha: float,
) -> dict:
    """Summarize the conservative persistent threshold for one mass."""
    threshold = minimum_persistent_events(
        conservative_curve,
        accuracy_column="worst_case_correct_fraction",
        target_accuracy=target_accuracy,
    )
    maximum_events = int(conservative_curve["number_of_events"].max())
    maximum_row = conservative_curve.loc[
        conservative_curve["number_of_events"] == maximum_events
    ].iloc[0]

    summary = {
        "mass_GeV": float(bank["mass_GeV"]),
        "rebin_factor": int(rebin_factor),
        "number_of_energy_bins": len(np.asarray(bank["energy_edges_GeV"])) - 1,
        "jeffreys_alpha": float(jeffreys_alpha),
        "stored_jeffreys_alpha": float(stored_jeffreys_alpha),
        "truth_grid": truth_grid,
        "profile_grid": profile_grid,
        "number_of_photon_truth_lifetimes": len(
            lifetime_grid_indices(len(np.asarray(bank["photon_ctau_m"])), truth_grid)
        ),
        "number_of_su2_truth_lifetimes": len(
            lifetime_grid_indices(len(np.asarray(bank["su2_ctau_m"])), truth_grid)
        ),
        "number_of_photon_profile_lifetimes": len(
            lifetime_grid_indices(len(np.asarray(bank["photon_ctau_m"])), profile_grid)
        ),
        "number_of_su2_profile_lifetimes": len(
            lifetime_grid_indices(len(np.asarray(bank["su2_ctau_m"])), profile_grid)
        ),
        "pseudoexperiments_per_truth_and_seed": int(number_of_pseudoexperiments),
        "number_of_seeds": int(number_of_seeds),
        "target_accuracy": float(target_accuracy),
        "threshold_reached": threshold is not None,
        "minimum_persistent_events": int(threshold) if threshold is not None else -1,
        "maximum_tested_events": maximum_events,
        "worst_case_accuracy_at_maximum_events": float(
            maximum_row["worst_case_correct_fraction"]
        ),
    }

    if threshold is None:
        summary.update(
            {
                "accuracy_at_threshold": np.nan,
                "limiting_seed_at_threshold": -1,
                "limiting_truth_model_at_threshold": "not_reached",
                "limiting_truth_lifetime_index_at_threshold": -1,
                "limiting_truth_ctau_m_at_threshold": np.nan,
            }
        )
    else:
        row = conservative_curve.loc[
            conservative_curve["number_of_events"] == threshold
        ].iloc[0]
        summary.update(
            {
                "accuracy_at_threshold": float(row["worst_case_correct_fraction"]),
                "limiting_seed_at_threshold": int(row["limiting_seed"]),
                "limiting_truth_model_at_threshold": str(
                    row["limiting_truth_model"]
                ),
                "limiting_truth_lifetime_index_at_threshold": int(
                    row["limiting_truth_lifetime_index"]
                ),
                "limiting_truth_ctau_m_at_threshold": float(
                    row["limiting_truth_ctau_m"]
                ),
            }
        )
    return summary


# -----------------------------------------------------------------------------
# Plots and output protection
# -----------------------------------------------------------------------------


def plot_accuracy_curve(
    curve: pd.DataFrame,
    *,
    mass_gev: float,
    target_accuracy: float,
    threshold: int | None,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Plot conservative accuracy after profiling over lifetime and seeds."""
    use_report_style, style_axis = _report_style_functions()
    use_report_style()
    figure, axis = plt.subplots(figsize=(8.2, 5.8))
    event_counts = curve["number_of_events"].to_numpy(dtype=int)

    axis.plot(
        event_counts,
        curve["photon_truth_worst_accuracy"],
        marker="o",
        label="Worst photophilic truth lifetime",
    )
    axis.plot(
        event_counts,
        curve["su2_truth_worst_accuracy"],
        marker="s",
        label=r"Worst $SU(2)_L$ truth lifetime",
    )
    axis.plot(
        event_counts,
        curve["worst_case_correct_fraction"],
        marker="^",
        linewidth=2.2,
        label="Overall worst case",
    )
    axis.axhline(
        target_accuracy,
        linestyle="--",
        linewidth=1.4,
        label=f"Target = {100.0 * target_accuracy:.0f}%",
    )
    if threshold is not None:
        axis.axvline(
            threshold,
            linestyle=":",
            linewidth=1.4,
            label=f"Persistent threshold: N = {threshold}",
        )

    axis.set_xlabel("Observed ALP decays, $N$")
    axis.set_ylabel("Correct-classification probability")
    axis.set_title(
        rf"Lifetime-profiled shape discrimination, $m_a={mass_gev:g}\,$GeV"
    )
    axis.set_xticks(event_counts)
    axis.set_ylim(0.45, 1.01)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    style_axis(axis)
    figure.tight_layout()

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def plot_threshold_summary(
    summary_table: pd.DataFrame,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Plot the conservative event requirement versus ALP mass."""
    use_report_style, style_axis = _report_style_functions()
    use_report_style()
    figure, axis = plt.subplots(figsize=(8.2, 5.8))

    masses = summary_table["mass_GeV"].to_numpy(dtype=float)
    reached = summary_table["threshold_reached"].to_numpy(dtype=bool)
    maximum_tested = int(summary_table["maximum_tested_events"].max())
    event_requirement = np.where(
        reached,
        summary_table["minimum_persistent_events"].to_numpy(dtype=float),
        maximum_tested + 1.0,
    )

    axis.plot(masses, event_requirement, marker="o", linewidth=1.8)
    for mass, reached_here, value in zip(masses, reached, event_requirement):
        if not reached_here:
            axis.annotate(
                f">{maximum_tested}",
                (mass, value),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
            )

    target = float(summary_table["target_accuracy"].iloc[0])
    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel("Minimum persistent observed events")
    axis.set_title(
        "Observed events required for "
        f"{100.0 * target:.0f}% worst-case model classification"
    )
    axis.set_xscale("log")

    finite_requirements = event_requirement[reached]

    if len(finite_requirements) > 0:
        upper_limit = max(
            5.5,
            float(np.max(finite_requirements)) + 1.0,
        )
    else:
        upper_limit = maximum_tested + 1.8

    axis.set_ylim(0.5, upper_limit)
    axis.set_yticks(
        np.arange(
            1,
            int(np.ceil(upper_limit)),
            dtype=int,
        )
    )
    axis.grid(True, alpha=0.25)
    style_axis(axis)
    figure.tight_layout()

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def ensure_no_existing_outputs(paths: list[Path], *, overwrite: bool) -> None:
    """Protect existing outputs unless replacement was requested."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Profiled-likelihood output already exists. "
            "Use --overwrite to replace it:\n" + formatted
        )


# -----------------------------------------------------------------------------
# Built-in statistical self-test
# -----------------------------------------------------------------------------


def run_self_test() -> None:
    """Test sign, profiling, ties, and reproducibility without repository data."""
    photon = np.array(
        [
            [0.90, 0.09, 0.01],
            [0.80, 0.18, 0.02],
        ]
    )
    su2 = np.array(
        [
            [0.01, 0.09, 0.90],
            [0.02, 0.18, 0.80],
        ]
    )
    kwargs = {
        "mass_gev": 0.3,
        "truth_model": "photon",
        "truth_index": 0,
        "truth_ctau_m": 1.0,
        "truth_probabilities": photon[0],
        "photon_probabilities": photon,
        "su2_probabilities": su2,
        "event_counts": np.array([1, 4, 8], dtype=int),
        "number_of_pseudoexperiments": 4_000,
        "seed": 12345,
        "chunk_size": 700,
        "tie_tolerance": DEFAULT_TIE_TOLERANCE,
    }
    first = simulate_truth_template(**kwargs)
    repeated = simulate_truth_template(**kwargs)
    pd.testing.assert_frame_equal(first, repeated, check_exact=True)

    high_n_accuracy = float(
        first.loc[first["number_of_events"] == 8, "correct_fraction"].iloc[0]
    )
    if high_n_accuracy < 0.98:
        raise AssertionError(
            f"Separated-template accuracy is unexpectedly low: {high_n_accuracy}"
        )

    identical = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.55, 0.35, 0.10],
        ]
    )
    tie_result = simulate_truth_template(
        mass_gev=0.5,
        truth_model="photon",
        truth_index=0,
        truth_ctau_m=1.0,
        truth_probabilities=identical[0],
        photon_probabilities=identical,
        su2_probabilities=identical,
        event_counts=np.array([1, 5], dtype=int),
        number_of_pseudoexperiments=1_000,
        seed=54321,
        chunk_size=400,
        tie_tolerance=DEFAULT_TIE_TOLERANCE,
    )
    if not np.allclose(tie_result["correct_fraction"], 0.5, atol=0.0, rtol=0.0):
        raise AssertionError("Identical hypotheses must give 50% tie-split accuracy.")
    if not np.allclose(tie_result["tie_fraction"], 1.0, atol=0.0, rtol=0.0):
        raise AssertionError("Identical hypotheses must give T=0 for every dataset.")

    raw = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.35, 0.50],
        ],
        dtype=float,
    )
    n_eff = np.array([250.0, 400.0], dtype=float)
    stored_alpha = 0.5
    stored = (
        n_eff[:, np.newaxis] * raw + stored_alpha
    ) / (
        n_eff[:, np.newaxis] + stored_alpha * raw.shape[1]
    )
    unchanged = _resmooth_probability_matrix(
        smoothed_probabilities=stored,
        total_n_eff=n_eff,
        stored_alpha=stored_alpha,
        target_alpha=stored_alpha,
        label="self-test",
    )
    if not np.array_equal(unchanged, stored):
        raise AssertionError("Stored-alpha smoothing must be preserved exactly.")

    target_alpha = 1.0
    expected = (
        n_eff[:, np.newaxis] * raw + target_alpha
    ) / (
        n_eff[:, np.newaxis] + target_alpha * raw.shape[1]
    )
    reconstructed = _resmooth_probability_matrix(
        smoothed_probabilities=stored,
        total_n_eff=n_eff,
        stored_alpha=stored_alpha,
        target_alpha=target_alpha,
        label="self-test",
    )
    if not np.allclose(reconstructed, expected, rtol=0.0, atol=1.0e-15):
        raise AssertionError("Jeffreys-alpha reconstruction failed.")

    print("Self-test passed:")
    print("  reproducibility: exact")
    print(f"  separated templates at N=8: accuracy={high_n_accuracy:.4f}")
    print("  identical hypotheses: T=0 and tie-split accuracy=0.5")
    print("  Jeffreys-alpha reconstruction: exact within floating precision")


# -----------------------------------------------------------------------------
# CLI and main program
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independently lifetime-profiled, shape-only pseudoexperiments "
            "from the saved detector-level ALP template banks."
        )
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        default=None,
        help="Subset of masses in GeV. By default, process every bank.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Template-bank directory (default: {INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAXIMUM_EVENTS,
        help=f"Largest tested N (default: {DEFAULT_MAXIMUM_EVENTS}).",
    )
    parser.add_argument(
        "--pseudoexperiments",
        type=int,
        default=DEFAULT_PSEUDOEXPERIMENTS,
        help=(
            "Pseudoexperiments per true lifetime and seed "
            f"(default: {DEFAULT_PSEUDOEXPERIMENTS})."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"First seed (default: {DEFAULT_BASE_SEED}).",
    )
    parser.add_argument(
        "--number-of-seeds",
        type=int,
        default=DEFAULT_NUMBER_OF_SEEDS,
        help=(
            "Independent validation seeds; the final curve uses their minimum "
            f"(default: {DEFAULT_NUMBER_OF_SEEDS})."
        ),
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=DEFAULT_SEED_STEP,
        help=f"Spacing between seeds (default: {DEFAULT_SEED_STEP}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Pseudoexperiments per memory chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=DEFAULT_TARGET_ACCURACY,
        help=f"Persistent worst-case target (default: {DEFAULT_TARGET_ACCURACY}).",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=DEFAULT_TIE_TOLERANCE,
        help=f"Absolute |T| tie tolerance (default: {DEFAULT_TIE_TOLERANCE:g}).",
    )
    parser.add_argument(
        "--rebin-factor",
        type=int,
        default=DEFAULT_REBIN_FACTOR,
        help=(
            "Merge this many consecutive saved energy bins before the "
            f"likelihood calculation (default: {DEFAULT_REBIN_FACTOR})."
        ),
    )
    parser.add_argument(
        "--jeffreys-alpha",
        type=float,
        default=None,
        help=(
            "Jeffreys pseudocount applied to all templates. By default, use "
            "the value stored in each bank."
        ),
    )
    parser.add_argument(
        "--truth-grid",
        choices=("all", "even", "odd"),
        default=DEFAULT_TRUTH_GRID,
        help=(
            "Saved lifetime indices used as pseudoexperiment truth: all, "
            "even, or odd (default: all)."
        ),
    )
    parser.add_argument(
        "--profile-grid",
        choices=("all", "even", "odd"),
        default=DEFAULT_PROFILE_GRID,
        help=(
            "Saved lifetime indices used when profiling each hypothesis: all, "
            "even, or odd (default: all)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing profiled-likelihood outputs.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in statistical tests and exit without reading banks.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_events < 1:
        raise ValueError("--max-events must be at least one.")
    if args.pseudoexperiments < 1:
        raise ValueError("--pseudoexperiments must be positive.")
    if args.base_seed < 0:
        raise ValueError("--base-seed cannot be negative.")
    if args.number_of_seeds < 1 or args.seed_step < 1:
        raise ValueError("Seed count and seed step must be positive.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")
    if not 0.0 < args.target_accuracy < 1.0:
        raise ValueError("--target-accuracy must lie strictly between zero and one.")
    if args.tie_tolerance < 0.0:
        raise ValueError("--tie-tolerance cannot be negative.")
    if args.rebin_factor < 1:
        raise ValueError("--rebin-factor must be a positive integer.")
    if args.jeffreys_alpha is not None:
        if not np.isfinite(args.jeffreys_alpha) or args.jeffreys_alpha <= 0.0:
            raise ValueError("--jeffreys-alpha must be positive.")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    if args.self_test:
        run_self_test()
        return

    load_template_bank, select_bank_paths = _bank_helpers()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    bank_paths = select_bank_paths(input_dir, args.masses)
    event_counts = np.arange(1, args.max_events + 1, dtype=int)
    seeds = [
        int(args.base_seed + seed_index * args.seed_step)
        for seed_index in range(args.number_of_seeds)
    ]

    combined_summary_path = output_dir / "profiled_threshold_summary.csv"
    combined_plot_stem = plot_dir / "profiled_minimum_events_vs_mass"
    ensure_no_existing_outputs(
        [
            combined_summary_path,
            combined_plot_stem.with_suffix(".pdf"),
            combined_plot_stem.with_suffix(".png"),
        ],
        overwrite=args.overwrite,
    )

    print()
    print("=" * 84)
    print("Lifetime-profiled ALP shape pseudoexperiments")
    print("EventCalc is not launched; saved detector-level template banks are used.")
    print(f"Banks: {len(bank_paths)}")
    print(f"Observed event counts: 1--{args.max_events}")
    print(f"Pseudoexperiments per truth and seed: {args.pseudoexperiments:,}")
    print(f"Seeds: {seeds}")
    print(f"Target worst-case accuracy: {100.0 * args.target_accuracy:.1f}%")
    print(f"Energy-bin rebin factor: {args.rebin_factor}")
    print(
        "Jeffreys alpha: "
        + (
            "stored bank value"
            if args.jeffreys_alpha is None
            else f"{args.jeffreys_alpha:g}"
        )
    )
    print(f"Truth lifetime grid: {args.truth_grid}")
    print(f"Profile lifetime grid: {args.profile_grid}")
    print("=" * 84)

    summary_rows: list[dict] = []
    for bank_path in bank_paths:
        bank = load_template_bank(bank_path)
        bank = add_smoothing_metadata(bank, bank_path)
        bank = resmooth_template_bank(bank, args.jeffreys_alpha)
        bank = rebin_template_bank(bank, args.rebin_factor)
        mass_gev = float(bank["mass_GeV"])
        token = mass_token(mass_gev)

        detailed_path = table_dir / f"profiled_accuracy_ma_{token}.csv"
        seed_worst_path = table_dir / f"profiled_worst_case_by_seed_ma_{token}.csv"
        conservative_path = table_dir / f"profiled_conservative_curve_ma_{token}.csv"
        threshold_path = table_dir / f"profiled_threshold_ma_{token}.csv"
        accuracy_stem = plot_dir / f"profiled_accuracy_ma_{token}"
        ensure_no_existing_outputs(
            [
                detailed_path,
                seed_worst_path,
                conservative_path,
                threshold_path,
                accuracy_stem.with_suffix(".pdf"),
                accuracy_stem.with_suffix(".png"),
            ],
            overwrite=args.overwrite,
        )

        photon_truth_count = len(
            lifetime_grid_indices(
                len(np.asarray(bank["photon_ctau_m"])), args.truth_grid
            )
        )
        su2_truth_count = len(
            lifetime_grid_indices(
                len(np.asarray(bank["su2_ctau_m"])), args.truth_grid
            )
        )
        photon_profile_count = len(
            lifetime_grid_indices(
                len(np.asarray(bank["photon_ctau_m"])), args.profile_grid
            )
        )
        su2_profile_count = len(
            lifetime_grid_indices(
                len(np.asarray(bank["su2_ctau_m"])), args.profile_grid
            )
        )
        number_of_truths = photon_truth_count + su2_truth_count
        print()
        print("-" * 84)
        print(f"m_a = {mass_gev:g} GeV")
        print(f"Energy bins: {len(np.asarray(bank['energy_edges_GeV'])) - 1}")
        print(
            "Jeffreys alpha: "
            f"{float(bank['jeffreys_alpha']):g} "
            f"(stored: {float(bank['stored_jeffreys_alpha']):g})"
        )
        print(
            "Truth lifetime templates: "
            f"{number_of_truths} "
            f"(photon={photon_truth_count}, su2={su2_truth_count})"
        )
        print(
            "Profile lifetime templates: "
            f"photon={photon_profile_count}, su2={su2_profile_count}"
        )
        print("-" * 84)

        detailed = pd.concat(
            [
                run_seed(
                    bank,
                    event_counts=event_counts,
                    number_of_pseudoexperiments=args.pseudoexperiments,
                    seed=seed,
                    chunk_size=args.chunk_size,
                    tie_tolerance=args.tie_tolerance,
                    truth_grid=args.truth_grid,
                    profile_grid=args.profile_grid,
                )
                for seed in seeds
            ],
            ignore_index=True,
        )
        seed_worst = build_seed_worst_case_table(detailed)
        conservative = build_conservative_seed_envelope(seed_worst)
        threshold_summary = summarize_mass_threshold(
            bank=bank,
            conservative_curve=conservative,
            target_accuracy=args.target_accuracy,
            number_of_pseudoexperiments=args.pseudoexperiments,
            number_of_seeds=args.number_of_seeds,
            rebin_factor=args.rebin_factor,
            truth_grid=args.truth_grid,
            profile_grid=args.profile_grid,
            jeffreys_alpha=float(bank["jeffreys_alpha"]),
            stored_jeffreys_alpha=float(bank["stored_jeffreys_alpha"]),
        )
        threshold = (
            int(threshold_summary["minimum_persistent_events"])
            if bool(threshold_summary["threshold_reached"])
            else None
        )

        detailed.to_csv(detailed_path, index=False)
        seed_worst.to_csv(seed_worst_path, index=False)
        conservative.to_csv(conservative_path, index=False)
        pd.DataFrame([threshold_summary]).to_csv(threshold_path, index=False)
        accuracy_pdf, _ = plot_accuracy_curve(
            conservative,
            mass_gev=mass_gev,
            target_accuracy=args.target_accuracy,
            threshold=threshold,
            output_stem=accuracy_stem,
        )
        summary_rows.append(threshold_summary)

        if threshold is None:
            print(
                f"  target not reached by N={args.max_events}; "
                f"worst-case accuracy="
                f"{threshold_summary['worst_case_accuracy_at_maximum_events']:.4f}"
            )
        else:
            print(
                f"  persistent {100.0 * args.target_accuracy:.0f}% threshold: "
                f"N={threshold}"
            )
            print(
                "  limiting truth at threshold: "
                f"{threshold_summary['limiting_truth_model_at_threshold']}, "
                f"c*tau="
                f"{threshold_summary['limiting_truth_ctau_m_at_threshold']:.6g} m"
            )
        print(f"  accuracy plot: {accuracy_pdf}")
        print(f"  detailed table: {detailed_path}")

    summary_table = pd.DataFrame(summary_rows).sort_values(
        "mass_GeV", ignore_index=True
    )
    summary_table.to_csv(combined_summary_path, index=False)
    summary_pdf, _ = plot_threshold_summary(
        summary_table,
        output_stem=combined_plot_stem,
    )

    display = summary_table[
        [
            "mass_GeV",
            "minimum_persistent_events",
            "threshold_reached",
            "worst_case_accuracy_at_maximum_events",
            "limiting_truth_model_at_threshold",
            "limiting_truth_ctau_m_at_threshold",
        ]
    ]
    print()
    print("=" * 84)
    print("Profiled-likelihood pseudoexperiments finished")
    print("=" * 84)
    print(display.to_string(index=False))
    print()
    print(f"Threshold summary: {combined_summary_path}")
    print(f"Threshold plot:    {summary_pdf}")


if __name__ == "__main__":
    main()