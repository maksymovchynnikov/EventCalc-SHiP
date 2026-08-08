"""Absolute weighted samples and their normalized mother-ALP energy spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """Return (sum w)^2/sum(w^2) for finite non-negative weights."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be a finite, non-negative one-dimensional array")
    total_weight = float(np.sum(weights))
    total_squared_weight = float(np.sum(weights**2))
    # Preserve the frozen builder's direct summation and operation order.
    direct = total_weight**2 / total_squared_weight if total_squared_weight else np.nan
    if np.isfinite(direct):
        return float(direct)
    scale = float(weights.max(initial=0.0))
    if scale == 0.0:
        return 0.0
    # Pathological under/overflow fallback; production weights use the direct path.
    scaled = weights / scale
    return float(scaled.sum() ** 2 / np.dot(scaled, scaled))


@dataclass
class WeightedSpectrum:
    """Accepted energies [GeV] and absolute expected-event weights per sample."""

    model_id: str
    source: str
    mass_gev: float
    ctau_m: float
    selection_name: str
    energies_gev: np.ndarray
    absolute_event_weights: np.ndarray
    expected_events: float
    seed: int
    generated_samples: int
    accepted_samples: int
    exposure_pot: float
    visible_br: float
    yield_per_pot_per_coupling_squared: float
    unit_coupling_ctau_m: float
    coupling_squared_gev_inv2: float
    n_llp_total: float
    epsilon_polar: float
    epsilon_azimuthal: float
    mean_decay_probability: float
    preselection_expected_events: float | None = None
    preselection_samples: int | None = None
    selection_efficiency_weighted: float | None = None
    source_expected_events: dict[str, float] = field(default_factory=dict)
    cache_key: str | None = None

    def __post_init__(self) -> None:
        self.energies_gev = np.asarray(self.energies_gev, dtype=float)
        self.absolute_event_weights = np.asarray(self.absolute_event_weights, dtype=float)
        if self.mass_gev <= 0.0 or self.ctau_m <= 0.0:
            raise ValueError("mass_gev and ctau_m must be positive")
        if self.energies_gev.ndim != 1 or self.absolute_event_weights.ndim != 1:
            raise ValueError("energies and weights must be one-dimensional")
        if self.energies_gev.shape != self.absolute_event_weights.shape or not len(self.energies_gev):
            raise ValueError("energies and weights must be non-empty and equally sized")
        if not np.all(np.isfinite(self.energies_gev)) or np.any(self.energies_gev < self.mass_gev):
            raise ValueError("energies must be finite and at least the ALP mass")
        if not np.all(np.isfinite(self.absolute_event_weights)) or np.any(
            self.absolute_event_weights < 0.0
        ):
            raise ValueError("absolute event weights must be finite and non-negative")
        total = float(self.absolute_event_weights.sum())
        if not np.isfinite(self.expected_events) or self.expected_events < 0.0:
            raise ValueError("expected_events must be finite and non-negative")
        if not np.isclose(total, self.expected_events, rtol=1e-12, atol=0.0):
            raise ValueError("expected_events must equal the sum of absolute weights")
        if self.accepted_samples != len(self.energies_gev):
            raise ValueError("accepted_samples must equal the stored array length")
        if self.preselection_expected_events is None:
            self.preselection_expected_events = self.expected_events
        if self.preselection_samples is None:
            self.preselection_samples = self.accepted_samples
        if self.selection_efficiency_weighted is None:
            self.selection_efficiency_weighted = (
                self.expected_events / self.preselection_expected_events
                if self.preselection_expected_events > 0.0
                else 0.0
            )
        if (
            not np.isfinite(self.preselection_expected_events)
            or self.preselection_expected_events < 0.0
        ):
            raise ValueError("preselection_expected_events must be finite and non-negative")
        if self.preselection_samples < self.accepted_samples or self.preselection_samples < 0:
            raise ValueError("preselection_samples cannot be smaller than accepted_samples")
        if self.expected_events > self.preselection_expected_events and not np.isclose(
            self.expected_events,
            self.preselection_expected_events,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("selection cannot increase expected events")
        expected_efficiency = (
            self.expected_events / self.preselection_expected_events
            if self.preselection_expected_events > 0.0
            else 0.0
        )
        if not np.isclose(
            self.selection_efficiency_weighted,
            expected_efficiency,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "selection_efficiency_weighted must equal after/before expected events"
            )

    @property
    def total_n_eff(self) -> float:
        return effective_sample_size(self.absolute_event_weights)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"energies_gev": self.energies_gev, "absolute_event_weights": self.absolute_event_weights}

    def metadata(self) -> dict:
        scalars = {
            key: value for key, value in vars(self).items()
            if key not in {"energies_gev", "absolute_event_weights", "cache_key"}
        }
        scalars["total_n_eff"] = self.total_n_eff
        return scalars

    @classmethod
    def from_cache(cls, arrays: Mapping[str, np.ndarray], metadata: Mapping) -> "WeightedSpectrum":
        fields = cls.__dataclass_fields__
        values = {key: metadata[key] for key in fields if key in metadata and key != "cache_key"}
        for key in ("epsilon_polar", "epsilon_azimuthal", "mean_decay_probability"):
            if values.get(key) is None:
                values[key] = np.nan
        values.update(energies_gev=arrays["energies_gev"], absolute_event_weights=arrays["absolute_event_weights"])
        values["cache_key"] = metadata.get("cache_key")
        return cls(**values)


@dataclass(frozen=True)
class HistogramSpectrum:
    energy_edges_gev: np.ndarray
    bin_probabilities: np.ndarray
    density_per_gev: np.ndarray
    density_error_per_gev: np.ndarray
    sum_weights_per_bin: np.ndarray
    sum_squared_weights_per_bin: np.ndarray
    effective_samples_per_bin: np.ndarray
    range_coverage: float


def validate_energy_edges(energy_edges_gev: np.ndarray) -> np.ndarray:
    edges = np.asarray(energy_edges_gev, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("energy edges must be a finite one-dimensional array")
    if edges[0] <= 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy edges must be positive and strictly increasing")
    return edges


def histogram_moments(spectrum: WeightedSpectrum, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = validate_energy_edges(edges)
    weights = spectrum.absolute_event_weights
    sum_weights = np.histogram(spectrum.energies_gev, bins=edges, weights=weights)[0]
    sum_squared = np.histogram(spectrum.energies_gev, bins=edges, weights=weights**2)[0]
    return sum_weights, sum_squared


def normalized_weighted_spectrum(
    spectrum: WeightedSpectrum, energy_edges_gev: np.ndarray, coverage_tolerance: float = 1e-10,
) -> HistogramSpectrum:
    """Histogram dN/dE and normalize it by the separately retained event rate."""
    edges = validate_energy_edges(energy_edges_gev)
    sum_weights, sum_squared = histogram_moments(spectrum, edges)
    histogram_weight = float(sum_weights.sum())
    if spectrum.expected_events <= 0.0 or histogram_weight <= 0.0:
        raise ValueError("a normalized spectrum requires positive event weight")
    coverage = histogram_weight / spectrum.expected_events
    if not np.isclose(coverage, 1.0, rtol=0.0, atol=coverage_tolerance):
        raise ValueError(f"energy range does not cover the weighted sample (coverage={coverage:.12g})")
    probabilities = sum_weights / histogram_weight
    widths = np.diff(edges)
    n_eff = np.divide(sum_weights**2, sum_squared, out=np.zeros_like(sum_weights), where=sum_squared > 0.0)
    return HistogramSpectrum(
        energy_edges_gev=edges, bin_probabilities=probabilities,
        density_per_gev=probabilities / widths,
        density_error_per_gev=np.sqrt(sum_squared) / (histogram_weight * widths),
        sum_weights_per_bin=sum_weights, sum_squared_weights_per_bin=sum_squared,
        effective_samples_per_bin=n_eff, range_coverage=coverage,
    )


def combine_absolute_source_spectra(
    model_id: str, source_spectra: Mapping[str, WeightedSpectrum]
) -> WeightedSpectrum:
    """Add source event weights before any shape normalization."""
    if not source_spectra:
        raise ValueError("at least one source spectrum is required")
    spectra = list(source_spectra.values())
    reference = spectra[0]
    for spectrum in spectra:
        if spectrum.model_id != model_id:
            raise ValueError("all source spectra must belong to model_id")
        if (spectrum.mass_gev, spectrum.ctau_m, spectrum.selection_name) != (
            reference.mass_gev, reference.ctau_m, reference.selection_name
        ):
            raise ValueError("source mass, lifetime and selection must agree")
        if not np.isclose(spectrum.visible_br, reference.visible_br, rtol=1e-12, atol=0.0):
            raise ValueError("source visible branching ratios disagree")
    source_events = {label: value.expected_events for label, value in source_spectra.items()}
    preselection_events = float(
        sum(item.preselection_expected_events for item in spectra)
    )
    expected_events = float(sum(source_events.values()))
    return WeightedSpectrum(
        model_id=model_id, source="combined", mass_gev=reference.mass_gev,
        ctau_m=reference.ctau_m, selection_name=reference.selection_name,
        energies_gev=np.concatenate([item.energies_gev for item in spectra]),
        absolute_event_weights=np.concatenate([item.absolute_event_weights for item in spectra]),
        expected_events=expected_events, seed=reference.seed,
        generated_samples=sum(item.generated_samples for item in spectra),
        accepted_samples=sum(item.accepted_samples for item in spectra),
        exposure_pot=reference.exposure_pot, visible_br=reference.visible_br,
        yield_per_pot_per_coupling_squared=sum(
            item.yield_per_pot_per_coupling_squared for item in spectra
        ),
        unit_coupling_ctau_m=reference.unit_coupling_ctau_m,
        coupling_squared_gev_inv2=reference.coupling_squared_gev_inv2,
        n_llp_total=sum(item.n_llp_total for item in spectra), epsilon_polar=np.nan,
        epsilon_azimuthal=np.nan, mean_decay_probability=np.nan,
        preselection_expected_events=preselection_events,
        preselection_samples=sum(item.preselection_samples for item in spectra),
        selection_efficiency_weighted=(
            expected_events / preselection_events if preselection_events > 0.0 else 0.0
        ),
        source_expected_events=source_events,
    )


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    values, weights, quantiles = map(lambda item: np.asarray(item, dtype=float), (values, weights, quantiles))
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be equally sized one-dimensional arrays")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("values and weights must be finite and weights non-negative")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    positive = weights > 0.0
    if not np.any(positive):
        raise ValueError("at least one weight must be positive")
    order = np.argsort(values[positive])
    sorted_values, sorted_weights = values[positive][order], weights[positive][order]
    cdf = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(quantiles, cdf, sorted_values, left=sorted_values[0], right=sorted_values[-1])
