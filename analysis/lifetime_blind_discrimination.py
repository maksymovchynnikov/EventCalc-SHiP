"""Build detector-level lifetime template banks for lifetime-blind ALP discrimination.
1. reads the model-specific observable lifetime domains from
   ``analysis/ctau_scan/ctau_scan.csv``;
2. samples each domain on an independent logarithmic grid for
   ALP-photon and ALP-SU(2)L;
3. generates primary+cascade photophilic spectra and inclusive SU(2)L spectra;
4. applies the two-photon ECAL acceptance already implemented in ``analysis.ECAL``;
5. constructs one common adaptive energy binning for every lifetime and both
   hypotheses at the given mass;
6. stores smoothed, normalized probability templates for the later 2D-distance
   map and profiled-likelihood pseudoexperiments.

Run from the repository root with
    python -m analysis.lifetime_blind_discrimination

Useful examples:
    python -m analysis.lifetime_blind_discrimination --masses 0.3
    python -m analysis.lifetime_blind_discrimination --lifetime-points 25
    python -m analysis.lifetime_blind_discrimination --overwrite
    python -m analysis.lifetime_blind_discrimination \
        --masses 0.75 1.0 --lifetime-points 41 --seed-offset 1000000 \
        --output-dir analysis/lifetime_blind_discrimination_seed_validation
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from analysis.ECAL import DEFAULT_ECAL, diphoton_ecal_acceptance
from analysis.compare_energy_spectra import (
    BASE_SEED,
    ECAL_SEED_OFFSET,
    MODEL_CONFIGS,
    N_INTERPOLATION_POINTS,
    N_POT,
    NUMBER_OF_ENERGY_BINS,
    RESAMPLE_SIZE,
    normalized_weighted_energy_spectrum,
)
from analysis.energy_spectrum_discrimination import (
    JEFFREYS_ALPHA,
    MINIMUM_BIN_N_EFF,
    float_token,
    make_common_adaptive_energy_edges,
    smoothed_bin_probabilities,
)
from funcs.initLLP import LLP
from funcs.kinematics import Grids
from funcs.ship_setup import theta_max_dec_vol

# CONFIGURATION
ANALYSIS_DIR = Path(__file__).resolve().parent
CTAU_SCAN_PATH = ANALYSIS_DIR / "ctau_scan" / "ctau_scan.csv"
OUTPUT_DIR = ANALYSIS_DIR / "lifetime_blind_discrimination"
TEMPLATE_DIR = OUTPUT_DIR / "template_banks"
TABLE_DIR = OUTPUT_DIR / "tables"

MODEL_NAMES = ("ALP-photon-combined", "ALP-SU2L")

EVENT_THRESHOLD = 10.0
ENERGY_MAX_GEV = 400.0
DEFAULT_LIFETIME_POINTS = 20

EVENT_RATE_RELATIVE_TOLERANCE = 0.05
LOG_ENDPOINT_PADDING_FRACTION = 2.0e-3


@dataclass(frozen=True)
class ObservableInterval:
    """One finite c*tau interval satisfying the event-rate requirement."""

    lower_m: float
    upper_m: float
    lower_is_scan_boundary: bool
    upper_is_scan_boundary: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower_m) or not np.isfinite(self.upper_m):
            raise ValueError("Observable lifetime boundaries must be finite.")
        if self.lower_m <= 0.0 or self.upper_m <= self.lower_m:
            raise ValueError("Invalid observable lifetime interval.")


@dataclass
class PreparedSource:
    """A production source whose theta-energy sample is reused in c*tau."""

    source_label: str
    llp: LLP
    kinematics: Grids
    visible_branching_ratio: float
    true_sample_seed: int
    ecal_seed: int


def _as_bool(series: pd.Series, column_name: str) -> pd.Series:
    """Parse a CSV Boolean column without relying on Python truthiness."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    text = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    invalid = ~text.isin(allowed)
    if invalid.any():
        bad_values = sorted(text.loc[invalid].unique().tolist())
        raise ValueError(
            f"Column {column_name!r} contains invalid Boolean values: {bad_values}"
        )
    return text.isin({"true", "1"})


def load_ctau_scan(path: Path) -> pd.DataFrame:
    """Load and validate the event-rate scan used to define T_gamma and T_W."""
    if not path.exists():
        raise FileNotFoundError(
            f"Lifetime scan not found: {path}\n"
            "Run `python -m analysis.scan_ctau_ranges` first."
        )

    data = pd.read_csv(path)
    required_columns = {
        "model",
        "mass_GeV",
        "ctau_m",
        "N_events",
        "passes_event_cut",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    data = data.copy()
    for column in ("mass_GeV", "ctau_m", "N_events"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    numeric = data[["mass_GeV", "ctau_m", "N_events"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("The lifetime scan contains non-finite numerical values.")
    if np.any(data["mass_GeV"] <= 0.0) or np.any(data["ctau_m"] <= 0.0):
        raise ValueError("Masses and lifetimes in the scan must be positive.")
    if np.any(data["N_events"] < 0.0):
        raise ValueError("Expected event rates cannot be negative.")

    data["passes_event_cut"] = _as_bool(
        data["passes_event_cut"],
        "passes_event_cut",
    )

    unknown_models = sorted(set(data["model"]) - set(MODEL_NAMES))
    if unknown_models:
        print(
            "Ignoring models not used in the lifetime-blind analysis: "
            + ", ".join(unknown_models)
        )

    return data.loc[data["model"].isin(MODEL_NAMES)].sort_values(
        ["mass_GeV", "model", "ctau_m"],
        ignore_index=True,
    )


def _logarithmic_threshold_crossing(
    ctau_left: float,
    rate_left: float,
    ctau_right: float,
    rate_right: float,
    threshold: float,
) -> float:
    """Interpolate a local N_events=threshold crossing in log-log space."""
    values = np.asarray(
        [ctau_left, rate_left, ctau_right, rate_right, threshold],
        dtype=float,
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Logarithmic threshold interpolation requires positive values.")

    log_rate_left = np.log(rate_left)
    log_rate_right = np.log(rate_right)
    if np.isclose(log_rate_left, log_rate_right, rtol=0.0, atol=1.0e-15):
        return float(np.sqrt(ctau_left * ctau_right))

    fraction = (
        np.log(threshold) - log_rate_left
    ) / (
        log_rate_right - log_rate_left
    )
    fraction = float(np.clip(fraction, 0.0, 1.0))

    return float(
        np.exp(
            np.log(ctau_left)
            + fraction * (np.log(ctau_right) - np.log(ctau_left))
        )
    )


def find_observable_intervals(
    model_mass_scan: pd.DataFrame,
    *,
    threshold: float,
    allow_truncated: bool,
) -> list[ObservableInterval]:
    """Find all contiguous c*tau regions with N_events >= threshold."""
    if threshold <= 0.0:
        raise ValueError("The event threshold must be positive.")
    if model_mass_scan.empty:
        return []

    ordered = (
        model_mass_scan.sort_values("ctau_m")
        .drop_duplicates(subset="ctau_m", keep="last")
        .reset_index(drop=True)
    )
    ctaus = ordered["ctau_m"].to_numpy(dtype=float)
    rates = ordered["N_events"].to_numpy(dtype=float)

    # Recompute the state from N_events. This makes the template builder robust
    # if a stale passes_event_cut column used a different threshold.
    passes = rates >= threshold
    if len(ctaus) < 2:
        raise RuntimeError("At least two lifetime-scan points are required.")

    intervals: list[ObservableInterval] = []
    current_start: float | None = float(ctaus[0]) if passes[0] else None
    current_lower_is_boundary = bool(passes[0])

    for index in range(len(ctaus) - 1):
        if passes[index] == passes[index + 1]:
            continue

        crossing = _logarithmic_threshold_crossing(
            ctaus[index],
            rates[index],
            ctaus[index + 1],
            rates[index + 1],
            threshold,
        )

        if not passes[index] and passes[index + 1]:
            current_start = crossing
            current_lower_is_boundary = False
        else:
            if current_start is None:
                raise RuntimeError("Internal error while closing an observable interval.")
            intervals.append(
                ObservableInterval(
                    lower_m=float(current_start),
                    upper_m=float(crossing),
                    lower_is_scan_boundary=current_lower_is_boundary,
                    upper_is_scan_boundary=False,
                )
            )
            current_start = None

    if passes[-1]:
        if current_start is None:
            raise RuntimeError("Internal error while extending an observable interval.")
        if not allow_truncated:
            raise RuntimeError(
                "The N_events >= threshold interval reaches the largest scanned "
                f"lifetime ({ctaus[-1]:g} m). Rerun the lifetime scan with a larger "
                "MAX_CTAU_M, or use --allow-truncated for an explicitly preliminary bank."
            )
        intervals.append(
            ObservableInterval(
                lower_m=float(current_start),
                upper_m=float(ctaus[-1]),
                lower_is_scan_boundary=current_lower_is_boundary,
                upper_is_scan_boundary=True,
            )
        )

    return intervals


def collect_model_domains(
    scan: pd.DataFrame,
    *,
    threshold: float,
    allow_truncated: bool,
) -> dict[tuple[str, float], ObservableInterval]:
    """Return one validated observable interval for each model and mass."""
    domains: dict[tuple[str, float], ObservableInterval] = {}

    for (model_name, mass_gev), group in scan.groupby(["model", "mass_GeV"], sort=True):
        intervals = find_observable_intervals(
            group,
            threshold=threshold,
            allow_truncated=allow_truncated,
        )
        if not intervals:
            continue
        if len(intervals) != 1:
            formatted = ", ".join(
                f"[{interval.lower_m:.6g}, {interval.upper_m:.6g}]"
                for interval in intervals
            )
            raise RuntimeError(
                f"{model_name}, m_a={mass_gev:g} GeV has multiple observable "
                f"lifetime intervals: {formatted}. Inspect ctau_scan.csv before profiling."
            )
        domains[(str(model_name), float(mass_gev))] = intervals[0]

    return domains


def available_masses(
    domains: dict[tuple[str, float], ObservableInterval],
) -> list[float]:
    """Masses for which both independently profiled hypotheses are observable."""
    photon_masses = {mass for model, mass in domains if model == "ALP-photon-combined"}
    su2_masses = {mass for model, mass in domains if model == "ALP-SU2L"}
    return sorted(photon_masses & su2_masses)


def resolve_requested_masses(requested: list[float] | None, candidates: list[float]) -> list[float]:
    """Resolve command-line masses against the values present in the scan."""
    if not candidates:
        raise RuntimeError(
            "No mass has an observable lifetime interval for both hypotheses."
        )
    if requested is None:
        return candidates

    selected: list[float] = []
    for requested_mass in requested:
        matches = [
            candidate
            for candidate in candidates
            if np.isclose(candidate, requested_mass, rtol=0.0, atol=1.0e-12)
        ]
        if not matches:
            raise ValueError(
                f"Requested mass {requested_mass:g} GeV is unavailable. "
                f"Available masses: {candidates}"
            )
        selected.append(matches[0])

    return sorted(set(selected))


def build_lifetime_grid(interval: ObservableInterval, number_of_points: int) -> np.ndarray:
    """Construct a dense logarithmic grid slightly inside threshold crossings."""
    if number_of_points < 2:
        raise ValueError("At least two lifetime points are required per model.")

    log_lower = np.log(interval.lower_m)
    log_upper = np.log(interval.upper_m)
    log_span = log_upper - log_lower

    if not interval.lower_is_scan_boundary:
        log_lower += LOG_ENDPOINT_PADDING_FRACTION * log_span
    if not interval.upper_is_scan_boundary:
        log_upper -= LOG_ENDPOINT_PADDING_FRACTION * log_span

    if log_upper <= log_lower:
        raise RuntimeError("Lifetime interval is too narrow after endpoint padding.")

    return np.geomspace(
        np.exp(log_lower),
        np.exp(log_upper),
        number_of_points,
    )


def prepare_model_sources(
    *,
    model_name: str,
    mass_gev: float,
    preparation_ctau_m: float,
    seed: int,
) -> dict[str, PreparedSource]:
    """Interpolate and resample each production source once for a mass."""
    config = MODEL_CONFIGS[model_name]
    prepared: dict[str, PreparedSource] = {}

    for source_index, production_mode in enumerate(config["alp_production_modes"]):
        source_label = "inclusive" if production_mode is None else str(production_mode)
        source_seed = seed + 1_000 * source_index

        llp = LLP(
            mass=None,
            particle_selection=config["particle_selection"],
            mixing_pattern=None,
            uncertainty=None,
            alp_production_mode=production_mode,
        )
        llp.set_mass(mass_gev)
        llp.compute_mass_dependent_properties()
        llp.set_c_tau(preparation_ctau_m)

        visible_branching_ratio = float(np.sum(llp.BrRatios_distr))
        if not np.isfinite(visible_branching_ratio) or visible_branching_ratio <= 0.0:
            raise RuntimeError(
                f"Invalid visible branching ratio for {model_name}, "
                f"m_a={mass_gev:g} GeV."
            )

        np.random.seed(source_seed)
        kinematics = Grids(
            llp.Distr,
            llp.Energy_distr,
            N_INTERPOLATION_POINTS,
            llp.mass,
            preparation_ctau_m,
            theta_max_sim=theta_max_dec_vol,
        )
        kinematics.interpolate(False)

        # Reuse across lifetimes only when EventCalc's proposal distribution
        # has the same lower energy bound at every requested c*tau.
        if not np.allclose(
            kinematics.e_min_sampling,
            mass_gev,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            minimum = float(np.min(kinematics.e_min_sampling))
            maximum = float(np.max(kinematics.e_min_sampling))
            raise RuntimeError(
                "The theta-energy sample cannot be reused across lifetimes.\n"
                f"Model: {model_name}\n"
                f"Source: {source_label}\n"
                f"m_a: {mass_gev:g} GeV\n"
                f"e_min_sampling range: [{minimum:.6g}, {maximum:.6g}] GeV"
            )

        kinematics.resample(RESAMPLE_SIZE, False)
        prepared[source_label] = PreparedSource(
            source_label=source_label,
            llp=llp,
            kinematics=kinematics,
            visible_branching_ratio=visible_branching_ratio,
            true_sample_seed=source_seed + 1,
            ecal_seed=source_seed + ECAL_SEED_OFFSET,
        )

    return prepared


def evaluate_prepared_source(
    prepared: PreparedSource,
    *,
    model_name: str,
    mass_gev: float,
    ctau_m: float,
) -> dict:
    """Evaluate one prepared source at a new lifetime and apply ECAL acceptance."""
    if ctau_m <= 0.0:
        raise ValueError("c*tau must be positive.")

    llp = prepared.llp
    kinematics = prepared.kinematics

    llp.set_c_tau(float(ctau_m))
    coupling_squared = float(llp.c_tau_int / ctau_m)
    n_llp_total = N_POT * float(llp.Yield) * coupling_squared

    kinematics.c_tau = float(ctau_m)
    np.random.seed(prepared.true_sample_seed)
    kinematics.true_samples(False)
    results = np.asarray(kinematics.get_kinematics(), dtype=float)

    if results.ndim != 2 or results.shape[1] <= 6:
        raise RuntimeError(
            f"Invalid EventCalc output for {model_name}, {prepared.source_label}, "
            f"m_a={mass_gev:g} GeV, c*tau={ctau_m:g} m."
        )

    valid = (
        np.isfinite(results[:, 3])
        & np.isfinite(results[:, 6])
        & (results[:, 6] >= 0.0)
    )
    results = results[valid]
    if len(results) == 0:
        raise RuntimeError(
            f"No valid mother-level events for {model_name}, "
            f"m_a={mass_gev:g} GeV, c*tau={ctau_m:g} m."
        )

    epsilon_polar = float(kinematics.epsilon_polar)
    decay_probabilities = np.asarray(results[:, 6], dtype=float)
    event_weight_scale = (
        n_llp_total
        * epsilon_polar
        * prepared.visible_branching_ratio
        / RESAMPLE_SIZE
    )
    event_weights_before_ecal = event_weight_scale * decay_probabilities
    n_events_before_ecal = float(np.sum(event_weights_before_ecal))

    ecal_mask = diphoton_ecal_acceptance(
        results,
        geometry=DEFAULT_ECAL,
        seed=prepared.ecal_seed,
    )
    if not np.any(ecal_mask):
        raise RuntimeError(
            f"No event passes the diphoton ECAL acceptance for {model_name}, "
            f"m_a={mass_gev:g} GeV, c*tau={ctau_m:g} m."
        )

    energies = np.asarray(results[ecal_mask, 3], dtype=float)
    event_weights = np.asarray(event_weights_before_ecal[ecal_mask], dtype=float)
    n_events = float(np.sum(event_weights))

    return {
        "energies": energies,
        "weights": event_weights,
        "n_events": n_events,
        "n_events_before_ecal": n_events_before_ecal,
        "epsilon_ecal_weighted": (
            n_events / n_events_before_ecal
            if n_events_before_ecal > 0.0
            else 0.0
        ),
        "source_label": prepared.source_label,
        "coupling_squared": coupling_squared,
    }


def evaluate_prepared_model(
    prepared_sources: dict[str, PreparedSource],
    *,
    model_name: str,
    mass_gev: float,
    ctau_m: float,
    initial_energy_edges: np.ndarray,
) -> dict:
    """Combine all production sources and return one normalized spectrum."""
    source_results = {
        source_label: evaluate_prepared_source(
            prepared,
            model_name=model_name,
            mass_gev=mass_gev,
            ctau_m=ctau_m,
        )
        for source_label, prepared in prepared_sources.items()
    }

    energies = np.concatenate([source["energies"] for source in source_results.values()])
    weights = np.concatenate([source["weights"] for source in source_results.values()])

    spectrum = normalized_weighted_energy_spectrum(
        energies=energies,
        weights=weights,
        energy_edges=initial_energy_edges,
    )
    source_n_events = {label: float(source["n_events"]) for label, source in source_results.items()}
    source_n_events_before_ecal = {
        label: float(source["n_events_before_ecal"]) for label, source in source_results.items()
    }
    n_events = float(sum(source_n_events.values()))
    n_events_before_ecal = float(sum(source_n_events_before_ecal.values()))

    if not np.isclose(
        spectrum["total_weight"],
        n_events,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise RuntimeError("Combined spectrum weight and event rate disagree.")

    spectrum.update(
        {
            "model": model_name,
            "mass_GeV": float(mass_gev),
            "ctau_m": float(ctau_m),
            "n_events": n_events,
            "n_events_before_ecal": n_events_before_ecal,
            "epsilon_ecal_weighted": (
                n_events / n_events_before_ecal
                if n_events_before_ecal > 0.0
                else 0.0
            ),
            "source_n_events": source_n_events,
            "source_n_events_before_ecal": source_n_events_before_ecal,
        }
    )
    return spectrum


# -----------------------------------------------------------------------------
# Common binning and persistence
# -----------------------------------------------------------------------------


def _validate_template_event_rate(
    *,
    model_name: str,
    mass_gev: float,
    ctau_m: float,
    n_events: float,
) -> None:
    minimum_allowed = EVENT_THRESHOLD * (1.0 - EVENT_RATE_RELATIVE_TOLERANCE)
    if n_events < minimum_allowed:
        raise RuntimeError(
            f"Template lies outside the intended observable domain: {model_name}, "
            f"m_a={mass_gev:g} GeV, c*tau={ctau_m:.6g} m gives "
            f"N_events={n_events:.6g}. Rerun scan_ctau_ranges or inspect the "
            "threshold interpolation."
        )
    if n_events < EVENT_THRESHOLD:
        print(
            "WARNING: endpoint-level Monte Carlo difference: "
            f"{model_name}, m_a={mass_gev:g} GeV, c*tau={ctau_m:.6g} m, "
            f"N_events={n_events:.6g}."
        )


def generate_raw_spectra(
    *,
    mass_gev: float,
    mass_index: int,
    lifetime_grids: dict[str, np.ndarray],
    seed_offset: int = 0,
) -> dict[str, dict[float, dict]]:
    """Generate all detector-level spectra before selecting common bins."""
    initial_energy_edges = np.geomspace(
        mass_gev,
        ENERGY_MAX_GEV,
        NUMBER_OF_ENERGY_BINS + 1,
    )
    all_spectra: dict[str, dict[float, dict]] = {}

    for model_index, model_name in enumerate(MODEL_NAMES):
        lifetimes = lifetime_grids[model_name]
        model_seed = (
            BASE_SEED
            + seed_offset
            + 10_000 * mass_index
            + 100 * model_index
        )

        print()
        print("=" * 76)
        print(f"Preparing {model_name}, m_a={mass_gev:g} GeV")
        print(
            f"Independent lifetime domain: "
            f"[{lifetimes[0]:.6g}, {lifetimes[-1]:.6g}] m"
        )
        print(f"Grid points: {len(lifetimes)}")
        print("=" * 76)

        prepared_sources = prepare_model_sources(
            model_name=model_name,
            mass_gev=mass_gev,
            preparation_ctau_m=float(lifetimes[0]),
            seed=model_seed,
        )

        spectra_by_ctau: dict[float, dict] = {}
        for lifetime_index, ctau_m in enumerate(lifetimes, start=1):
            print(
                f"[{model_name}] lifetime {lifetime_index}/{len(lifetimes)}: "
                f"c*tau={ctau_m:.6g} m"
            )
            spectrum = evaluate_prepared_model(
                prepared_sources,
                model_name=model_name,
                mass_gev=mass_gev,
                ctau_m=float(ctau_m),
                initial_energy_edges=initial_energy_edges,
            )
            _validate_template_event_rate(
                model_name=model_name,
                mass_gev=mass_gev,
                ctau_m=float(ctau_m),
                n_events=float(spectrum["n_events"]),
            )
            spectra_by_ctau[float(ctau_m)] = spectrum

        all_spectra[model_name] = spectra_by_ctau

    return all_spectra


def make_bank(
    *,
    mass_gev: float,
    all_spectra: dict[str, dict[float, dict]],
    intervals: dict[str, ObservableInterval],
    seed_offset: int = 0,
) -> dict:
    """Construct one common-binning probability bank for a mass."""
    initial_energy_edges = np.geomspace(
        mass_gev,
        ENERGY_MAX_GEV,
        NUMBER_OF_ENERGY_BINS + 1,
    )

    flat_spectra = {
        f"{model_name}::{ctau_m:.16g}": spectrum
        for model_name, spectra_by_ctau in all_spectra.items()
        for ctau_m, spectrum in spectra_by_ctau.items()
    }
    energy_edges = make_common_adaptive_energy_edges(
        spectra=flat_spectra,
        initial_energy_edges=initial_energy_edges,
        minimum_n_eff=MINIMUM_BIN_N_EFF,
    )

    bank: dict[str, object] = {
        "mass_GeV": float(mass_gev),
        "energy_edges_GeV": energy_edges,
        "minimum_bin_N_eff": float(MINIMUM_BIN_N_EFF),
        "jeffreys_alpha": float(JEFFREYS_ALPHA),
        "event_threshold": float(EVENT_THRESHOLD),
        "template_seed_offset": int(seed_offset),
        "template_base_seed": int(BASE_SEED + seed_offset),
    }

    for model_name, prefix in (
        ("ALP-photon-combined", "photon"),
        ("ALP-SU2L", "su2"),
    ):
        spectra_by_ctau = all_spectra[model_name]
        lifetimes = np.asarray(sorted(spectra_by_ctau), dtype=float)
        probabilities = []
        total_n_eff = []
        n_events = []
        n_events_before_ecal = []
        epsilon_ecal = []

        for ctau_m in lifetimes:
            spectrum = spectra_by_ctau[float(ctau_m)]
            template, n_eff = smoothed_bin_probabilities(
                spectrum=spectrum,
                energy_edges=energy_edges,
                alpha=JEFFREYS_ALPHA,
            )
            probabilities.append(template)
            total_n_eff.append(n_eff)
            n_events.append(float(spectrum["n_events"]))
            n_events_before_ecal.append(float(spectrum["n_events_before_ecal"]))
            epsilon_ecal.append(float(spectrum["epsilon_ecal_weighted"]))

        probability_matrix = np.vstack(probabilities)
        if np.any(~np.isfinite(probability_matrix)) or np.any(probability_matrix <= 0.0):
            raise RuntimeError(f"Invalid smoothed probabilities for {model_name}.")
        if not np.allclose(
            probability_matrix.sum(axis=1),
            1.0,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"Probability normalization failed for {model_name}.")

        interval = intervals[model_name]
        bank[f"{prefix}_ctau_m"] = lifetimes
        bank[f"{prefix}_probabilities"] = probability_matrix
        bank[f"{prefix}_n_events"] = np.asarray(n_events, dtype=float)
        bank[f"{prefix}_n_events_before_ecal"] = np.asarray(
            n_events_before_ecal,
            dtype=float,
        )
        bank[f"{prefix}_epsilon_ecal_weighted"] = np.asarray(
            epsilon_ecal,
            dtype=float,
        )
        bank[f"{prefix}_total_n_eff"] = np.asarray(total_n_eff, dtype=float)
        bank[f"{prefix}_interval_m"] = np.asarray(
            [interval.lower_m, interval.upper_m],
            dtype=float,
        )

    return bank


def bank_summary_table(bank: dict) -> pd.DataFrame:
    """One diagnostic row per model and lifetime."""
    rows: list[dict] = []
    mass_gev = float(bank["mass_GeV"])
    number_of_bins = len(np.asarray(bank["energy_edges_GeV"])) - 1

    for model_name, prefix in (
        ("ALP-photon-combined", "photon"),
        ("ALP-SU2L", "su2"),
    ):
        lifetimes = np.asarray(bank[f"{prefix}_ctau_m"], dtype=float)
        n_events = np.asarray(bank[f"{prefix}_n_events"], dtype=float)
        before_ecal = np.asarray(
            bank[f"{prefix}_n_events_before_ecal"],
            dtype=float,
        )
        epsilon_ecal = np.asarray(
            bank[f"{prefix}_epsilon_ecal_weighted"],
            dtype=float,
        )
        total_n_eff = np.asarray(bank[f"{prefix}_total_n_eff"], dtype=float)

        for index, ctau_m in enumerate(lifetimes):
            rows.append(
                {
                    "mass_GeV": mass_gev,
                    "model": model_name,
                    "lifetime_index": index,
                    "ctau_m": ctau_m,
                    "N_events": n_events[index],
                    "N_events_before_ECAL": before_ecal[index],
                    "epsilon_ECAL_weighted": epsilon_ecal[index],
                    "template_total_N_eff": total_n_eff[index],
                    "number_of_common_energy_bins": number_of_bins,
                    "passes_N_events_threshold": n_events[index] >= EVENT_THRESHOLD,
                }
            )

    return pd.DataFrame(rows)


def probability_table(bank: dict) -> pd.DataFrame:
    """Long-form table containing every stored bin probability."""
    rows: list[dict] = []
    mass_gev = float(bank["mass_GeV"])
    energy_edges = np.asarray(bank["energy_edges_GeV"], dtype=float)

    for model_name, prefix in (
        ("ALP-photon-combined", "photon"),
        ("ALP-SU2L", "su2"),
    ):
        lifetimes = np.asarray(bank[f"{prefix}_ctau_m"], dtype=float)
        probabilities = np.asarray(bank[f"{prefix}_probabilities"], dtype=float)

        for lifetime_index, ctau_m in enumerate(lifetimes):
            for bin_index, probability in enumerate(probabilities[lifetime_index]):
                rows.append(
                    {
                        "mass_GeV": mass_gev,
                        "model": model_name,
                        "lifetime_index": lifetime_index,
                        "ctau_m": ctau_m,
                        "bin_index": bin_index,
                        "energy_low_GeV": energy_edges[bin_index],
                        "energy_high_GeV": energy_edges[bin_index + 1],
                        "probability": probability,
                    }
                )

    return pd.DataFrame(rows)


def save_bank(bank: dict, *, overwrite: bool) -> tuple[Path, Path, Path]:
    """Persist the compact bank and two human-readable diagnostic tables."""
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    mass_gev = float(bank["mass_GeV"])
    token = float_token(mass_gev)
    bank_path = TEMPLATE_DIR / f"template_bank_ma_{token}.npz"
    summary_path = TABLE_DIR / f"template_summary_ma_{token}.csv"
    probability_path = TABLE_DIR / f"probability_templates_ma_{token}.csv"

    existing = [path for path in (bank_path, summary_path, probability_path) if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output already exists. Use --overwrite to replace it:\n" + formatted
        )

    np.savez_compressed(
        bank_path,
        **{
            key: value
            for key, value in bank.items()
            if isinstance(value, (float, int, np.ndarray))
        },
    )
    bank_summary_table(bank).to_csv(summary_path, index=False)
    probability_table(bank).to_csv(probability_path, index=False)

    return bank_path, summary_path, probability_path


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate common-binning detector-level lifetime template banks "
            "for independently profiled ALP-photon and ALP-SU(2)L hypotheses."
        )
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        default=None,
        help="Subset of masses in GeV. By default, use every available mass.",
    )
    parser.add_argument(
        "--lifetime-points",
        type=int,
        default=DEFAULT_LIFETIME_POINTS,
        help=(
            "Number of logarithmically spaced c*tau templates per model and mass "
            f"(default: {DEFAULT_LIFETIME_POINTS})."
        ),
    )
    parser.add_argument(
        "--scan-path",
        type=Path,
        default=CTAU_SCAN_PATH,
        help=f"Path to ctau_scan.csv (default: {CTAU_SCAN_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Integer added to every EventCalc/resampling/ECAL seed. "
            "Use a large non-zero value for an independent template-bank "
            "regeneration (default: 0)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing template-bank outputs.",
    )
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help=(
            "Allow an observable interval to end at the largest scanned lifetime. "
            "Use only for a clearly preliminary result."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global OUTPUT_DIR, TEMPLATE_DIR, TABLE_DIR

    args = parse_arguments()
    OUTPUT_DIR = args.output_dir.resolve()
    TEMPLATE_DIR = OUTPUT_DIR / "template_banks"
    TABLE_DIR = OUTPUT_DIR / "tables"
    if args.lifetime_points < 2:
        raise ValueError("--lifetime-points must be at least two.")
    if args.seed_offset < 0:
        raise ValueError("--seed-offset cannot be negative.")

    scan = load_ctau_scan(args.scan_path)
    domains = collect_model_domains(
        scan,
        threshold=EVENT_THRESHOLD,
        allow_truncated=args.allow_truncated,
    )
    candidate_masses = available_masses(domains)
    masses = resolve_requested_masses(
        args.masses,
        candidate_masses,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "template_bank_manifest.csv"

    if not args.overwrite:
        existing_outputs = []
        for mass_gev in masses:
            token = float_token(mass_gev)
            existing_outputs.extend(
                path
                for path in (
                    TEMPLATE_DIR / f"template_bank_ma_{token}.npz",
                    TABLE_DIR / f"template_summary_ma_{token}.csv",
                    TABLE_DIR / f"probability_templates_ma_{token}.csv",
                )
                if path.exists()
            )
        if manifest_path.exists():
            existing_outputs.append(manifest_path)
        if existing_outputs:
            formatted = "\n".join(f"  {path}" for path in existing_outputs)
            raise FileExistsError(
                "Output already exists. Use --overwrite to replace it:\n"
                + formatted
            )

    manifest_rows: list[pd.DataFrame] = []

    print()
    print("#" * 76)
    print("Lifetime-blind discrimination: detector-level template-bank generation")
    print(f"Masses: {', '.join(f'{mass:g}' for mass in masses)} GeV")
    print(f"Lifetime points per model: {args.lifetime_points}")
    print(f"Template seed offset: {args.seed_offset}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Lifetime domains are model-specific, not common intersections.")
    print("#" * 76)

    for mass_gev in masses:
        # Keep seeds identical whether one mass or the full set is requested.
        mass_index = candidate_masses.index(mass_gev)
        intervals = {
            model_name: domains[(model_name, mass_gev)]
            for model_name in MODEL_NAMES
        }
        lifetime_grids = {
            model_name: build_lifetime_grid(
                intervals[model_name],
                args.lifetime_points,
            )
            for model_name in MODEL_NAMES
        }

        print()
        print("#" * 76)
        print(f"m_a = {mass_gev:g} GeV")
        for model_name in MODEL_NAMES:
            interval = intervals[model_name]
            print(
                f"  {model_name}: T_H = "
                f"[{interval.lower_m:.6g}, {interval.upper_m:.6g}] m"
            )
        print("#" * 76)

        all_spectra = generate_raw_spectra(
            mass_gev=mass_gev,
            mass_index=mass_index,
            lifetime_grids=lifetime_grids,
            seed_offset=args.seed_offset,
        )
        bank = make_bank(
            mass_gev=mass_gev,
            all_spectra=all_spectra,
            intervals=intervals,
            seed_offset=args.seed_offset,
        )
        bank_path, summary_path, probability_path = save_bank(
            bank,
            overwrite=args.overwrite,
        )

        summary = bank_summary_table(bank)
        summary.insert(0, "template_bank_path", str(bank_path))
        manifest_rows.append(summary)

        print()
        print(f"Common adaptive energy bins: {len(bank['energy_edges_GeV']) - 1}")
        print(f"Template bank: {bank_path}")
        print(f"Template summary: {summary_path}")
        print(f"Long probability table: {probability_path}")

    manifest = pd.concat(manifest_rows, ignore_index=True)
    manifest.to_csv(manifest_path, index=False)

    print()
    print("=" * 76)
    print("Template-bank generation finished.")
    print(f"Manifest: {manifest_path}")
    print("The saved .npz files are the inputs for the 2D distance map and the")
    print("profiled-likelihood pseudoexperiments.")
    print("=" * 76)


if __name__ == "__main__":
    main()