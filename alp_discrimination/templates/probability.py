"""Common adaptive binning and Jeffreys-regularized probability templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from alp_discrimination.cache import CacheStore
from alp_discrimination.physics.spectra import WeightedSpectrum, histogram_moments, validate_energy_edges
from alp_discrimination.statistics.basic import validate_probabilities


def first_problem_bin(
    spectra: Mapping[str, WeightedSpectrum], energy_edges_gev: np.ndarray, minimum_n_eff: float,
) -> int | None:
    if not spectra or minimum_n_eff <= 0.0:
        raise ValueError("spectra are required and minimum_n_eff must be positive")
    weights_by_spectrum, low_statistics = [], np.zeros(len(energy_edges_gev) - 1, dtype=bool)
    for spectrum in spectra.values():
        sum_weights, sum_squared = histogram_moments(spectrum, energy_edges_gev)
        weights_by_spectrum.append(sum_weights)
        n_eff = np.divide(sum_weights**2, sum_squared, out=np.zeros_like(sum_weights), where=sum_squared > 0.0)
        low_statistics |= (sum_weights > 0.0) & (n_eff < minimum_n_eff)
    empty = np.any(np.vstack(weights_by_spectrum) == 0.0, axis=0)
    problems = np.flatnonzero(low_statistics | empty)
    return None if not len(problems) else int(problems[0])


def common_adaptive_energy_edges(
    spectra: Mapping[str, WeightedSpectrum], initial_energy_edges_gev: np.ndarray,
    minimum_n_eff: float,
) -> np.ndarray:
    """Apply the committed deterministic merge rule to any template collection."""
    edges = validate_energy_edges(initial_energy_edges_gev).copy()
    while len(edges) > 2:
        problem = first_problem_bin(spectra, edges, minimum_n_eff)
        if problem is None:
            break
        number_of_bins = len(edges) - 1
        if problem == 0:
            edge_to_remove = 1
        elif problem == number_of_bins - 1:
            edge_to_remove = len(edges) - 2
        else:
            left_width = np.log(edges[problem + 1] / edges[problem - 1])
            right_width = np.log(edges[problem + 2] / edges[problem])
            edge_to_remove = problem if left_width <= right_width else problem + 1
        edges = np.delete(edges, edge_to_remove)
    if first_problem_bin(spectra, edges, minimum_n_eff) is not None:
        raise RuntimeError("could not construct a statistically reliable common binning")
    return edges


def jeffreys_regularized_probabilities(
    spectrum: WeightedSpectrum, energy_edges_gev: np.ndarray, alpha: float,
) -> tuple[np.ndarray, float]:
    """Convert weighted MC probabilities to Neff*p effective counts, then add alpha."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    sum_weights, _ = histogram_moments(spectrum, energy_edges_gev)
    if not np.isclose(sum_weights.sum(), spectrum.expected_events, rtol=0.0, atol=1e-10 * spectrum.expected_events):
        raise ValueError("template energy edges do not cover the full weighted spectrum")
    raw = sum_weights / sum_weights.sum()
    total_n_eff = spectrum.total_n_eff
    probabilities = (total_n_eff * raw + alpha) / (total_n_eff + alpha * len(raw))
    probabilities /= probabilities.sum()
    return probabilities, total_n_eff


@dataclass(frozen=True)
class ProbabilityTemplate:
    model_id: str
    mass_gev: float
    ctau_m: float
    selection_name: str
    energy_edges_gev: np.ndarray
    probabilities: np.ndarray
    total_n_eff: float
    spectrum_cache_key: str | None = None

    def __post_init__(self) -> None:
        edges = validate_energy_edges(self.energy_edges_gev)
        probabilities = validate_probabilities(self.probabilities, strictly_positive=True)
        if len(edges) != len(probabilities) + 1 or self.mass_gev <= 0.0 or self.ctau_m <= 0.0:
            raise ValueError("template dimensions, mass or lifetime are invalid")
        object.__setattr__(self, "energy_edges_gev", edges)
        object.__setattr__(self, "probabilities", probabilities)


def validate_spectrum_collection(
    spectra: Mapping[str, WeightedSpectrum], *, same_lifetime: bool = False,
    keys_are_model_ids: bool = False,
) -> None:
    if not spectra:
        raise ValueError("at least one spectrum is required")
    reference = next(iter(spectra.values()))
    for model_id, spectrum in spectra.items():
        if keys_are_model_ids and spectrum.model_id != model_id:
            raise ValueError("spectrum mapping key and model identifier disagree")
        if (spectrum.mass_gev, spectrum.selection_name) != (
            reference.mass_gev, reference.selection_name
        ):
            raise ValueError("all templates must share mass and selection")
        if same_lifetime and spectrum.ctau_m != reference.ctau_m:
            raise ValueError("this template cache requires a common lifetime")


def build_probability_templates(
    spectra: Mapping[str, WeightedSpectrum], initial_energy_edges_gev: np.ndarray,
    minimum_n_eff: float, alpha: float,
) -> dict[str, ProbabilityTemplate]:
    validate_spectrum_collection(spectra)
    edges = common_adaptive_energy_edges(spectra, initial_energy_edges_gev, minimum_n_eff)
    templates = {}
    for model_id, spectrum in spectra.items():
        probabilities, total_n_eff = jeffreys_regularized_probabilities(spectrum, edges, alpha)
        templates[model_id] = ProbabilityTemplate(
            model_id=spectrum.model_id, mass_gev=spectrum.mass_gev, ctau_m=spectrum.ctau_m,
            selection_name=spectrum.selection_name, energy_edges_gev=edges,
            probabilities=probabilities, total_n_eff=total_n_eff,
            spectrum_cache_key=spectrum.cache_key,
        )
    return templates


def cached_probability_templates(
    cache: CacheStore, spectra: Mapping[str, WeightedSpectrum],
    initial_energy_edges_gev: np.ndarray, minimum_n_eff: float, alpha: float,
    *, force: bool = False,
) -> dict[str, ProbabilityTemplate]:
    """Load or atomically save Level-B templates keyed by Level-A spectra and final edges."""
    validate_spectrum_collection(spectra, same_lifetime=True, keys_are_model_ids=True)
    if cache.enabled and any(spectrum.cache_key is None for spectrum in spectra.values()):
        raise ValueError("enabled Level-B caching requires Level-A spectrum cache keys")
    edges = common_adaptive_energy_edges(spectra, initial_energy_edges_gev, minimum_n_eff)
    model_ids = tuple(spectra)
    reference = next(iter(spectra.values()))
    identity = {
        "template_format_version": 2, "profile": cache.profile,
        "spectrum_cache_keys": {model: spectra[model].cache_key for model in model_ids},
        "energy_edges_gev": edges.tolist(), "minimum_n_eff": minimum_n_eff,
        "jeffreys_alpha": alpha, "selection_name": reference.selection_name,
    }

    def validate(arrays: dict[str, np.ndarray], metadata: dict) -> None:
        if tuple(metadata["model_ids"]) != model_ids:
            raise ValueError("cached template model order differs")
        probabilities = arrays["probabilities"]
        if probabilities.shape != (len(model_ids), len(edges) - 1):
            raise ValueError("cached probability array has the wrong shape")
        for row in probabilities:
            validate_probabilities(row, strictly_positive=True)
        shape = probabilities.shape
        for name in ("sum_weights_per_bin", "sum_squared_weights_per_bin", "effective_samples_per_bin"):
            values = arrays[name]
            if values.shape != shape or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"cached {name} is invalid")
        total_n_eff = arrays["total_n_eff"]
        if total_n_eff.shape != (len(model_ids),) or not np.all(np.isfinite(total_n_eff)) or np.any(total_n_eff <= 0.0):
            raise ValueError("cached total effective sample sizes are invalid")
        if not np.array_equal(arrays["energy_edges_gev"], edges):
            raise ValueError("cached template edges differ")

    if not force:
        loaded = cache.load("probability_template", identity, validate)
        if loaded:
            arrays, metadata = loaded
            return {
                model_id: ProbabilityTemplate(
                    model_id=model_id, mass_gev=float(metadata["mass_gev"]),
                    ctau_m=float(metadata["ctau_m"]), selection_name=metadata["selection_name"],
                    energy_edges_gev=arrays["energy_edges_gev"], probabilities=arrays["probabilities"][index],
                    total_n_eff=float(arrays["total_n_eff"][index]),
                    spectrum_cache_key=spectra[model_id].cache_key,
                )
                for index, model_id in enumerate(model_ids)
            }
    else:
        _, _, key = cache.paths("probability_template", identity)
        print(f"CACHE FORCED   [probability_template] {key[:12]}")
    templates = build_probability_templates(spectra, edges, minimum_n_eff, alpha)
    arrays = {
        "energy_edges_gev": edges,
        "probabilities": np.vstack([templates[model].probabilities for model in model_ids]),
        "total_n_eff": np.asarray([templates[model].total_n_eff for model in model_ids]),
    }
    moments = [histogram_moments(spectra[model], edges) for model in model_ids]
    arrays["sum_weights_per_bin"] = np.vstack([item[0] for item in moments])
    arrays["sum_squared_weights_per_bin"] = np.vstack([item[1] for item in moments])
    arrays["effective_samples_per_bin"] = np.divide(
        arrays["sum_weights_per_bin"] ** 2, arrays["sum_squared_weights_per_bin"],
        out=np.zeros_like(arrays["sum_weights_per_bin"]),
        where=arrays["sum_squared_weights_per_bin"] > 0.0,
    )
    cache.save("probability_template", identity, arrays, {
        "model_ids": model_ids, "mass_gev": reference.mass_gev, "ctau_m": reference.ctau_m,
        "selection_name": reference.selection_name,
    })
    return templates


@dataclass(frozen=True)
class TemplateBank:
    """Future-ready p[model, lifetime, energy bin] on one fixed-mass binning."""

    model_ids: tuple[str, ...]
    lifetimes_m: np.ndarray
    energy_edges_gev: np.ndarray
    probabilities: np.ndarray
    mass_gev: float
    selection_name: str

    def __post_init__(self) -> None:
        lifetimes = np.asarray(self.lifetimes_m, float)
        edges, probabilities = validate_energy_edges(self.energy_edges_gev), np.asarray(self.probabilities, float)
        if lifetimes.ndim == 1:
            lifetimes = np.broadcast_to(lifetimes, (len(self.model_ids), len(lifetimes))).copy()
        expected_shape = (*lifetimes.shape, len(edges) - 1)
        if lifetimes.ndim != 2 or lifetimes.shape[0] != len(self.model_ids) or (
            np.any(~np.isfinite(lifetimes)) or np.any(lifetimes <= 0.0)
            or np.any(np.diff(lifetimes, axis=1) <= 0.0) or probabilities.shape != expected_shape
        ):
            raise ValueError(f"template bank must have shape {expected_shape}")
        for model_templates in probabilities:
            for template in model_templates:
                validate_probabilities(template, strictly_positive=True)
        object.__setattr__(self, "lifetimes_m", lifetimes)
        object.__setattr__(self, "energy_edges_gev", edges)
        object.__setattr__(self, "probabilities", probabilities)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "lifetimes_m": self.lifetimes_m, "energy_edges_gev": self.energy_edges_gev,
            "probabilities": self.probabilities,
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], metadata: Mapping) -> "TemplateBank":
        return cls(
            model_ids=tuple(metadata["model_ids"]), mass_gev=float(metadata["mass_gev"]),
            selection_name=str(metadata["selection_name"]), lifetimes_m=arrays["lifetimes_m"],
            energy_edges_gev=arrays["energy_edges_gev"], probabilities=arrays["probabilities"],
        )


@dataclass(frozen=True)
class SavedTemplatePair:
    mass_gev: float
    ctau_m: float
    lifetime_label: str
    number_of_bins: int
    photon: np.ndarray
    su2: np.ndarray


def load_saved_template_pair(path: Path) -> SavedTemplatePair:
    """Load and strictly validate a same-lifetime probability-template CSV."""
    data = pd.read_csv(path).sort_values("bin_index")
    required = {
        "mass_GeV", "ctau_m", "lifetime_label", "bin_index", "energy_low_GeV",
        "energy_high_GeV", "photon_probability", "su2_probability",
    }
    if missing := required - set(data.columns):
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    if not np.array_equal(data["bin_index"].to_numpy(int), np.arange(len(data))):
        raise ValueError(f"non-consecutive bins in {path}")
    low, high = data["energy_low_GeV"].to_numpy(float), data["energy_high_GeV"].to_numpy(float)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low <= 0.0):
        raise ValueError(f"invalid energy edges in {path}")
    if np.any(high <= low) or not np.allclose(low[1:], high[:-1], rtol=0.0, atol=1e-12):
        raise ValueError(f"template bins are not positive and contiguous in {path}")
    photon = validate_probabilities(data["photon_probability"].to_numpy(float), strictly_positive=True)
    su2 = validate_probabilities(data["su2_probability"].to_numpy(float), strictly_positive=True)
    if "log_su2_over_photon" in data and not np.allclose(
        data["log_su2_over_photon"], np.log(su2 / photon), rtol=1e-12, atol=1e-12
    ):
        raise ValueError(f"stored log likelihood ratio is inconsistent in {path}")

    def unique(column: str):
        values = data[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"non-unique {column} in {path}")
        return values.iloc[0]

    return SavedTemplatePair(
        mass_gev=float(unique("mass_GeV")), ctau_m=float(unique("ctau_m")),
        lifetime_label=str(unique("lifetime_label")), number_of_bins=len(data),
        photon=photon, su2=su2,
    )
