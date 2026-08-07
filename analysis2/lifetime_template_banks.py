"""Mass-wide detector-level lifetime template banks.

Source generation and detector selection produce :class:`WeightedSpectrum`
objects. This module performs only the next stage: common mass-wide binning,
Jeffreys smoothing, bank validation, caching, and portable tabulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .cache import atomic_output_path
from .spectra import WeightedSpectrum, validate_energy_edges
from .statistics import validate_probabilities
from .templates import (
    common_adaptive_energy_edges,
    first_problem_bin,
    jeffreys_regularized_probabilities,
)


BANK_FORMAT_VERSION = 3
SUPPORTED_BANK_FORMAT_VERSIONS = {1, 2, 3}
MODEL_PREFIXES = {
    "alp_photon_combined": "photon",
    "alp_su2l": "su2",
}


@dataclass(frozen=True)
class LifetimeTemplateBank:
    """Two independently sampled lifetime grids sharing one energy binning.

    ``*_interval_index`` identifies which connected allowed Week-8 lifetime
    interval each template belongs to. ``*_allowed_intervals_m`` stores the
    corresponding interval bounds. ``*_interval_m`` is retained as the full
    envelope for backwards-compatible plotting only; it must not be interpreted
    as a continuously allowed domain when more than one interval is present.
    """

    mass_gev: float
    energy_edges_gev: np.ndarray
    minimum_bin_n_eff: float
    jeffreys_alpha: float
    event_threshold: float
    template_seed_offset: int
    template_base_seed: int
    photon_ctau_m: np.ndarray
    photon_probabilities: np.ndarray
    photon_n_events: np.ndarray
    photon_n_events_before_ecal: np.ndarray
    photon_epsilon_ecal_weighted: np.ndarray
    photon_total_n_eff: np.ndarray
    photon_interval_m: np.ndarray
    su2_ctau_m: np.ndarray
    su2_probabilities: np.ndarray
    su2_n_events: np.ndarray
    su2_n_events_before_ecal: np.ndarray
    su2_epsilon_ecal_weighted: np.ndarray
    su2_total_n_eff: np.ndarray
    su2_interval_m: np.ndarray
    photon_interval_index: np.ndarray | None = None
    photon_allowed_intervals_m: np.ndarray | None = None
    su2_interval_index: np.ndarray | None = None
    su2_allowed_intervals_m: np.ndarray | None = None
    profile: str = "legacy"
    selection_name: str = "diphoton_ecal"
    minimum_photon_energy_gev: float | None = None
    cache_key: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass_gev) or self.mass_gev <= 0.0:
            raise ValueError("Template-bank mass must be finite and positive.")
        edges = np.asarray(self.energy_edges_gev, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)):
            raise ValueError("Template-bank energy edges are invalid.")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("Template-bank energy edges must increase.")
        object.__setattr__(self, "energy_edges_gev", edges)

        threshold = self.minimum_photon_energy_gev
        if threshold is not None:
            threshold = float(threshold)
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    "Minimum photon energy must be finite and non-negative."
                )
            object.__setattr__(self, "minimum_photon_energy_gev", threshold)
        if self.selection_name == "diphoton_ecal_e1gev":
            if threshold is None or not np.isclose(
                threshold,
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "diphoton_ecal_e1gev banks require a 1 GeV photon threshold."
                )
        elif self.selection_name == "diphoton_ecal" and threshold is not None:
            raise ValueError(
                "Geometry-only diphoton_ecal banks cannot record an energy threshold."
            )

        for prefix in ("photon", "su2"):
            self._validate_model_arrays(prefix, len(edges) - 1)

    def _validate_model_arrays(self, prefix: str, number_of_bins: int) -> None:
        lifetimes = np.asarray(getattr(self, f"{prefix}_ctau_m"), dtype=float)
        probabilities = np.asarray(
            getattr(self, f"{prefix}_probabilities"), dtype=float
        )
        if lifetimes.ndim != 1 or len(lifetimes) < 2:
            raise ValueError(f"{prefix} lifetime grid is invalid.")
        if np.any(~np.isfinite(lifetimes)) or np.any(lifetimes <= 0.0):
            raise ValueError(f"{prefix} lifetime grid is invalid.")
        if np.any(np.diff(lifetimes) <= 0.0):
            raise ValueError(f"{prefix} lifetime grid must increase.")
        if probabilities.shape != (len(lifetimes), number_of_bins):
            raise ValueError(f"{prefix} probability matrix has the wrong shape.")
        for template in probabilities:
            validate_probabilities(template, strictly_positive=True)
        object.__setattr__(self, f"{prefix}_ctau_m", lifetimes)
        object.__setattr__(self, f"{prefix}_probabilities", probabilities)

        raw_interval_index = getattr(self, f"{prefix}_interval_index")
        if raw_interval_index is None:
            raw_interval_index = np.zeros(len(lifetimes), dtype=int)
        interval_index_raw = np.asarray(raw_interval_index, dtype=float)
        if interval_index_raw.shape != lifetimes.shape:
            raise ValueError(f"{prefix}_interval_index has the wrong shape.")
        if np.any(~np.isfinite(interval_index_raw)):
            raise ValueError(f"{prefix}_interval_index contains non-finite values.")
        interval_index = np.rint(interval_index_raw).astype(int)
        if not np.allclose(
            interval_index_raw,
            interval_index,
            rtol=0.0,
            atol=1.0e-12,
        ) or np.any(interval_index < 0):
            raise ValueError(f"{prefix}_interval_index must contain non-negative integers.")

        raw_allowed = getattr(self, f"{prefix}_allowed_intervals_m")
        if raw_allowed is None:
            raw_allowed = np.asarray(getattr(self, f"{prefix}_interval_m"), dtype=float).reshape(1, 2)
        allowed = np.asarray(raw_allowed, dtype=float)
        if allowed.ndim != 2 or allowed.shape[1] != 2 or len(allowed) < 1:
            raise ValueError(f"{prefix}_allowed_intervals_m has the wrong shape.")
        if np.any(~np.isfinite(allowed)) or np.any(allowed <= 0.0):
            raise ValueError(f"{prefix}_allowed_intervals_m contains invalid values.")
        if np.any(allowed[:, 1] <= allowed[:, 0]):
            raise ValueError(f"{prefix} allowed intervals are not ordered.")
        ordered_allowed = allowed[np.argsort(allowed[:, 0])]
        if len(ordered_allowed) > 1 and np.any(
            ordered_allowed[1:, 0] < ordered_allowed[:-1, 1]
        ):
            raise ValueError(f"{prefix} allowed intervals overlap.")
        if np.max(interval_index) >= len(allowed):
            raise ValueError(f"{prefix}_interval_index refers to a missing interval.")
        lower = allowed[interval_index, 0]
        upper = allowed[interval_index, 1]
        tolerance = 1.0e-12
        if np.any(lifetimes < lower * (1.0 - tolerance)) or np.any(
            lifetimes > upper * (1.0 + tolerance)
        ):
            raise ValueError(f"{prefix} lifetime lies outside its allowed interval.")
        object.__setattr__(self, f"{prefix}_interval_index", interval_index)
        object.__setattr__(self, f"{prefix}_allowed_intervals_m", allowed)

        for suffix in (
            "n_events",
            "n_events_before_ecal",
            "epsilon_ecal_weighted",
            "total_n_eff",
        ):
            values = np.asarray(getattr(self, f"{prefix}_{suffix}"), dtype=float)
            if values.shape != lifetimes.shape or np.any(~np.isfinite(values)):
                raise ValueError(f"{prefix}_{suffix} has the wrong shape or values.")
            if np.any(values < 0.0):
                raise ValueError(f"{prefix}_{suffix} cannot be negative.")
            object.__setattr__(self, f"{prefix}_{suffix}", values)

        interval = np.asarray(getattr(self, f"{prefix}_interval_m"), dtype=float)
        expected_envelope = np.asarray(
            [np.min(allowed[:, 0]), np.max(allowed[:, 1])],
            dtype=float,
        )
        if interval.shape != (2,) or not np.allclose(
            interval,
            expected_envelope,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(f"{prefix} lifetime envelope is invalid.")
        object.__setattr__(self, f"{prefix}_interval_m", interval)

    @property
    def number_of_energy_bins(self) -> int:
        return len(self.energy_edges_gev) - 1

    def arrays(self) -> dict[str, np.ndarray]:
        """Return numerical arrays for the compressed bank artifact."""
        arrays: dict[str, np.ndarray] = {
            "bank_format_version": np.asarray(BANK_FORMAT_VERSION),
            "profile": np.asarray(self.profile),
            "selection_name": np.asarray(self.selection_name),
            "minimum_photon_energy_GeV": np.asarray(
                np.nan
                if self.minimum_photon_energy_gev is None
                else self.minimum_photon_energy_gev
            ),
            "mass_GeV": np.asarray(self.mass_gev),
            "energy_edges_GeV": self.energy_edges_gev,
            "minimum_bin_N_eff": np.asarray(self.minimum_bin_n_eff),
            "jeffreys_alpha": np.asarray(self.jeffreys_alpha),
            "event_threshold": np.asarray(self.event_threshold),
            "template_seed_offset": np.asarray(self.template_seed_offset),
            "template_base_seed": np.asarray(self.template_base_seed),
        }
        for prefix in ("photon", "su2"):
            for suffix in (
                "ctau_m",
                "interval_index",
                "allowed_intervals_m",
                "probabilities",
                "n_events",
                "n_events_before_ecal",
                "epsilon_ecal_weighted",
                "total_n_eff",
                "interval_m",
            ):
                arrays[f"{prefix}_{suffix}"] = np.asarray(
                    getattr(self, f"{prefix}_{suffix}")
                )
        return arrays

    def metadata(self) -> dict:
        return {
            "bank_format_version": BANK_FORMAT_VERSION,
            "mass_gev": self.mass_gev,
            "profile": self.profile,
            "selection_name": self.selection_name,
            "minimum_photon_energy_gev": self.minimum_photon_energy_gev,
        }

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping | None = None,
    ) -> "LifetimeTemplateBank":
        metadata = dict(metadata or {})
        if "profile" in arrays:
            metadata["profile"] = str(np.asarray(arrays["profile"]).item())
        if "selection_name" in arrays:
            metadata["selection_name"] = str(
                np.asarray(arrays["selection_name"]).item()
            )
        raw_threshold = arrays.get("minimum_photon_energy_GeV")
        minimum_photon_energy_gev = None
        if raw_threshold is not None:
            threshold_value = float(np.asarray(raw_threshold).item())
            if np.isfinite(threshold_value):
                minimum_photon_energy_gev = threshold_value

        values = {
            "mass_gev": float(np.asarray(arrays["mass_GeV"]).item()),
            "energy_edges_gev": arrays["energy_edges_GeV"],
            "minimum_bin_n_eff": float(np.asarray(arrays["minimum_bin_N_eff"]).item()),
            "jeffreys_alpha": float(np.asarray(arrays["jeffreys_alpha"]).item()),
            "event_threshold": float(np.asarray(arrays["event_threshold"]).item()),
            "template_seed_offset": int(np.asarray(arrays["template_seed_offset"]).item()),
            "template_base_seed": int(np.asarray(arrays["template_base_seed"]).item()),
            "profile": str(metadata.get("profile", "legacy")),
            "selection_name": str(metadata.get("selection_name", "diphoton_ecal")),
            "minimum_photon_energy_gev": minimum_photon_energy_gev,
            "cache_key": metadata.get("cache_key"),
        }
        for prefix in ("photon", "su2"):
            interval_m = np.asarray(arrays[f"{prefix}_interval_m"], dtype=float)
            values[f"{prefix}_ctau_m"] = arrays[f"{prefix}_ctau_m"]
            values[f"{prefix}_probabilities"] = arrays[f"{prefix}_probabilities"]
            values[f"{prefix}_n_events"] = arrays[f"{prefix}_n_events"]
            values[f"{prefix}_n_events_before_ecal"] = arrays[
                f"{prefix}_n_events_before_ecal"
            ]
            values[f"{prefix}_epsilon_ecal_weighted"] = arrays[
                f"{prefix}_epsilon_ecal_weighted"
            ]
            values[f"{prefix}_total_n_eff"] = arrays[f"{prefix}_total_n_eff"]
            values[f"{prefix}_interval_m"] = interval_m
            values[f"{prefix}_interval_index"] = arrays.get(
                f"{prefix}_interval_index",
                np.zeros(len(arrays[f"{prefix}_ctau_m"]), dtype=int),
            )
            values[f"{prefix}_allowed_intervals_m"] = arrays.get(
                f"{prefix}_allowed_intervals_m",
                interval_m.reshape(1, 2),
            )
        return cls(**values)


def load_template_bank(path: Path) -> LifetimeTemplateBank:
    """Load a current or frozen legacy bank with strict validation."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    version = int(np.asarray(arrays.get("bank_format_version", 1)).item())
    if version not in SUPPORTED_BANK_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported template-bank format {version} in {path}; "
            f"supported versions are {sorted(SUPPORTED_BANK_FORMAT_VERSIONS)}."
        )
    required_names = {
        "mass_GeV",
        "energy_edges_GeV",
        "minimum_bin_N_eff",
        "jeffreys_alpha",
        "event_threshold",
        "template_seed_offset",
        "template_base_seed",
    }
    for prefix in ("photon", "su2"):
        required_names.update(
            {
                f"{prefix}_ctau_m",
                f"{prefix}_probabilities",
                f"{prefix}_n_events",
                f"{prefix}_n_events_before_ecal",
                f"{prefix}_epsilon_ecal_weighted",
                f"{prefix}_total_n_eff",
                f"{prefix}_interval_m",
            }
        )
    if missing := required_names - set(arrays):
        raise ValueError(f"Missing template-bank arrays in {path}: {sorted(missing)}")
    return LifetimeTemplateBank.from_arrays(arrays)


def _spectrum_preselection_events(spectrum: WeightedSpectrum) -> float:
    value = getattr(spectrum, "preselection_expected_events", None)
    return spectrum.expected_events if value is None else float(value)


def build_lifetime_template_bank(
    *,
    mass_gev: float,
    spectra: Mapping[str, Mapping[float, WeightedSpectrum]],
    lifetime_grids: Mapping[str, pd.DataFrame],
    allowed_intervals_m: Mapping[str, np.ndarray],
    initial_energy_edges_gev: np.ndarray,
    minimum_bin_n_eff: float,
    fixed_energy_edges_gev: np.ndarray | None = None,
    jeffreys_alpha: float,
    event_threshold: float,
    template_base_seed: int,
    template_seed_offset: int,
    profile: str,
    selection_name: str,
    minimum_photon_energy_gev: float | None = None,
) -> LifetimeTemplateBank:
    """Construct one common-binning bank from all lifetimes and both models."""
    required_models = set(MODEL_PREFIXES)
    if (
        set(spectra) != required_models
        or set(lifetime_grids) != required_models
        or set(allowed_intervals_m) != required_models
    ):
        raise ValueError("A bank requires photon and SU(2)_L spectra and domains.")
    flat = {
        f"{model_id}::{ctau_m:.16g}": spectrum
        for model_id, by_lifetime in spectra.items()
        for ctau_m, spectrum in by_lifetime.items()
    }
    if fixed_energy_edges_gev is None:
        edges = common_adaptive_energy_edges(
            flat,
            initial_energy_edges_gev,
            minimum_bin_n_eff,
        )
    else:
        edges = validate_energy_edges(fixed_energy_edges_gev).copy()
        problem_bin = first_problem_bin(flat, edges, minimum_bin_n_eff)
        if problem_bin is not None:
            raise ValueError(
                "Fixed energy edges fail the common minimum-N_eff requirement "
                f"in bin {problem_bin}; refusing to merge or alter them."
            )
    values: dict[str, object] = {}
    for model_id, prefix in MODEL_PREFIXES.items():
        grid = lifetime_grids[model_id].sort_values("ctau_m", ignore_index=True)
        lifetimes = grid["ctau_m"].to_numpy(dtype=float)
        interval_index = grid["interval_index"].to_numpy(dtype=int)
        by_lifetime = spectra[model_id]
        available_lifetimes = np.asarray(sorted(by_lifetime), dtype=float)
        np.testing.assert_allclose(
            available_lifetimes,
            lifetimes,
            rtol=1.0e-12,
            atol=0.0,
            err_msg=f"{model_id} spectra do not match the requested lifetime grid.",
        )
        probabilities = []
        total_n_eff = []
        events = []
        before = []
        efficiency = []
        for ctau_m in lifetimes:
            spectrum = by_lifetime[float(ctau_m)]
            template, n_eff = jeffreys_regularized_probabilities(
                spectrum, edges, jeffreys_alpha
            )
            preselection = _spectrum_preselection_events(spectrum)
            probabilities.append(template)
            total_n_eff.append(n_eff)
            events.append(spectrum.expected_events)
            before.append(preselection)
            efficiency.append(
                spectrum.expected_events / preselection if preselection > 0.0 else 0.0
            )
        allowed = np.asarray(allowed_intervals_m[model_id], dtype=float)
        values.update(
            {
                f"{prefix}_ctau_m": lifetimes,
                f"{prefix}_interval_index": interval_index,
                f"{prefix}_allowed_intervals_m": allowed,
                f"{prefix}_probabilities": np.vstack(probabilities),
                f"{prefix}_n_events": np.asarray(events),
                f"{prefix}_n_events_before_ecal": np.asarray(before),
                f"{prefix}_epsilon_ecal_weighted": np.asarray(efficiency),
                f"{prefix}_total_n_eff": np.asarray(total_n_eff),
                f"{prefix}_interval_m": np.asarray(
                    [np.min(allowed[:, 0]), np.max(allowed[:, 1])]
                ),
            }
        )
    return LifetimeTemplateBank(
        mass_gev=mass_gev,
        energy_edges_gev=edges,
        minimum_bin_n_eff=minimum_bin_n_eff,
        jeffreys_alpha=jeffreys_alpha,
        event_threshold=event_threshold,
        template_seed_offset=template_seed_offset,
        template_base_seed=template_base_seed,
        profile=profile,
        selection_name=selection_name,
        minimum_photon_energy_gev=minimum_photon_energy_gev,
        **values,
    )


def bank_summary_table(bank: LifetimeTemplateBank) -> pd.DataFrame:
    """Return one diagnostic row per model and lifetime template."""
    rows = []
    for model, prefix in (
        ("ALP-photon-combined", "photon"),
        ("ALP-SU2L", "su2"),
    ):
        lifetimes = getattr(bank, f"{prefix}_ctau_m")
        interval_indices = getattr(bank, f"{prefix}_interval_index")
        for index, ctau_m in enumerate(lifetimes):
            n_events = getattr(bank, f"{prefix}_n_events")[index]
            rows.append(
                {
                    "mass_GeV": bank.mass_gev,
                    "model": model,
                    "lifetime_index": index,
                    "interval_index": int(interval_indices[index]),
                    "ctau_m": ctau_m,
                    "N_events": n_events,
                    "N_events_before_ECAL": getattr(
                        bank, f"{prefix}_n_events_before_ecal"
                    )[index],
                    "epsilon_ECAL_weighted": getattr(
                        bank, f"{prefix}_epsilon_ecal_weighted"
                    )[index],
                    "template_total_N_eff": getattr(
                        bank, f"{prefix}_total_n_eff"
                    )[index],
                    "number_of_common_energy_bins": bank.number_of_energy_bins,
                    "domain_event_level_geom_only": bank.event_threshold,
                    "passes_domain_event_level_after_ECAL": (
                        n_events >= bank.event_threshold
                    ),
                }
            )
    return pd.DataFrame(rows)


def probability_table(bank: LifetimeTemplateBank) -> pd.DataFrame:
    """Return the portable long-form probability table."""
    rows = []
    for model, prefix in (
        ("ALP-photon-combined", "photon"),
        ("ALP-SU2L", "su2"),
    ):
        lifetimes = getattr(bank, f"{prefix}_ctau_m")
        interval_indices = getattr(bank, f"{prefix}_interval_index")
        probabilities = getattr(bank, f"{prefix}_probabilities")
        for lifetime_index, ctau_m in enumerate(lifetimes):
            for bin_index, probability in enumerate(probabilities[lifetime_index]):
                rows.append(
                    {
                        "mass_GeV": bank.mass_gev,
                        "model": model,
                        "lifetime_index": lifetime_index,
                        "interval_index": int(interval_indices[lifetime_index]),
                        "ctau_m": ctau_m,
                        "bin_index": bin_index,
                        "energy_low_GeV": bank.energy_edges_gev[bin_index],
                        "energy_high_GeV": bank.energy_edges_gev[bin_index + 1],
                        "probability": probability,
                    }
                )
    return pd.DataFrame(rows)


def save_bank_artifacts(
    bank: LifetimeTemplateBank,
    *,
    bank_path: Path,
    summary_path: Path,
    probability_path: Path,
) -> None:
    """Atomically write one compact bank and its two diagnostic tables."""
    with atomic_output_path(bank_path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **bank.arrays())
    for table, path in (
        (bank_summary_table(bank), summary_path),
        (probability_table(bank), probability_path),
    ):
        with atomic_output_path(path) as temporary:
            table.to_csv(temporary, index=False)
