"""Mass-wide detector-level lifetime template banks.

Source generation and detector selection produce :class:`WeightedSpectrum`
objects.  This module performs only the next stage: common mass-wide binning,
Jeffreys smoothing, bank validation, caching, and portable tabulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .cache import CacheStore, atomic_output_path
from .observable_domains import ObservableLifetimeDomain
from .spectra import WeightedSpectrum
from .statistics import validate_probabilities
from .templates import common_adaptive_energy_edges, jeffreys_regularized_probabilities


BANK_FORMAT_VERSION = 1
MODEL_PREFIXES = {
    "alp_photon_combined": "photon",
    "alp_su2l": "su2",
}


@dataclass(frozen=True)
class LifetimeTemplateBank:
    """Two independently sampled lifetime grids sharing one energy binning."""

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
    profile: str = "legacy"
    selection_name: str = "diphoton_ecal"
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
        if interval.shape != (2,) or interval[0] <= 0.0 or interval[1] <= interval[0]:
            raise ValueError(f"{prefix} observable interval is invalid.")
        object.__setattr__(self, f"{prefix}_interval_m", interval)

    @property
    def number_of_energy_bins(self) -> int:
        return len(self.energy_edges_gev) - 1

    def arrays(self) -> dict[str, np.ndarray]:
        """Return legacy-compatible numerical array names."""
        arrays: dict[str, np.ndarray] = {
            "bank_format_version": np.asarray(BANK_FORMAT_VERSION),
            "profile": np.asarray(self.profile),
            "selection_name": np.asarray(self.selection_name),
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
            "cache_key": metadata.get("cache_key"),
        }
        for prefix in ("photon", "su2"):
            for suffix in (
                "ctau_m",
                "probabilities",
                "n_events",
                "n_events_before_ecal",
                "epsilon_ecal_weighted",
                "total_n_eff",
                "interval_m",
            ):
                values[f"{prefix}_{suffix}"] = arrays[f"{prefix}_{suffix}"]
        return cls(**values)


def load_template_bank(path: Path) -> LifetimeTemplateBank:
    """Load a refactored or tracked frozen legacy bank with strict validation."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if "bank_format_version" in arrays:
        version = int(np.asarray(arrays["bank_format_version"]).item())
        if version != BANK_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported template-bank format {version} in {path}; "
                f"expected {BANK_FORMAT_VERSION}."
            )
    required = set(LifetimeTemplateBank.__dataclass_fields__) - {
        "mass_gev",
        "energy_edges_gev",
        "minimum_bin_n_eff",
        "profile",
        "selection_name",
        "cache_key",
    }
    required_names = {
        "mass_GeV",
        "energy_edges_GeV",
        "minimum_bin_N_eff",
        *required,
    }
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
    domains: Mapping[str, ObservableLifetimeDomain],
    initial_energy_edges_gev: np.ndarray,
    minimum_bin_n_eff: float,
    jeffreys_alpha: float,
    event_threshold: float,
    template_base_seed: int,
    template_seed_offset: int,
    profile: str,
    selection_name: str,
) -> LifetimeTemplateBank:
    """Construct one common-binning bank from all lifetimes and both models."""
    if set(spectra) != set(MODEL_PREFIXES) or set(domains) != set(MODEL_PREFIXES):
        raise ValueError("A bank requires photon and SU(2)_L spectra and domains.")
    flat = {
        f"{model_id}::{ctau_m:.16g}": spectrum
        for model_id, by_lifetime in spectra.items()
        for ctau_m, spectrum in by_lifetime.items()
    }
    edges = common_adaptive_energy_edges(flat, initial_energy_edges_gev, minimum_bin_n_eff)
    values: dict[str, object] = {}
    for model_id, prefix in MODEL_PREFIXES.items():
        by_lifetime = spectra[model_id]
        lifetimes = np.asarray(sorted(by_lifetime), dtype=float)
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
        domain = domains[model_id]
        values.update(
            {
                f"{prefix}_ctau_m": lifetimes,
                f"{prefix}_probabilities": np.vstack(probabilities),
                f"{prefix}_n_events": np.asarray(events),
                f"{prefix}_n_events_before_ecal": np.asarray(before),
                f"{prefix}_epsilon_ecal_weighted": np.asarray(efficiency),
                f"{prefix}_total_n_eff": np.asarray(total_n_eff),
                f"{prefix}_interval_m": np.asarray([domain.lower_m, domain.upper_m]),
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
        for index, ctau_m in enumerate(lifetimes):
            rows.append(
                {
                    "mass_GeV": bank.mass_gev,
                    "model": model,
                    "lifetime_index": index,
                    "ctau_m": ctau_m,
                    "N_events": getattr(bank, f"{prefix}_n_events")[index],
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
                    "passes_N_events_threshold": (
                        getattr(bank, f"{prefix}_n_events")[index]
                        >= bank.event_threshold
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
        probabilities = getattr(bank, f"{prefix}_probabilities")
        for lifetime_index, ctau_m in enumerate(lifetimes):
            for bin_index, probability in enumerate(probabilities[lifetime_index]):
                rows.append(
                    {
                        "mass_GeV": bank.mass_gev,
                        "model": model,
                        "lifetime_index": lifetime_index,
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
