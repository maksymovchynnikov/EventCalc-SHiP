"""Immutable run profiles and the frozen frozen-reference random-stream policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose
from types import MappingProxyType
from typing import Mapping


PRODUCTION_MASSES_GEV = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.05)
PRODUCTION_MODEL_ORDER = ("alp_photon_combined", "alp_su2l")


@dataclass(frozen=True)
class SamplingSettings:
    interpolation_points: int
    resample_size: int

    def __post_init__(self) -> None:
        if self.interpolation_points < 1 or self.resample_size < 1:
            raise ValueError("sampling sizes must be positive")


@dataclass(frozen=True)
class ECALGeometrySettings:
    """The rectangular ECAL plane used by ``DiphotonECALSelection``."""

    z_m: float = 95.0
    width_x_m: float = 4.0
    height_y_m: float = 6.0
    centre_x_m: float = 0.0
    centre_y_m: float = 0.0


@dataclass(frozen=True)
class LifetimeSettings:
    event_threshold: float = 10.0
    maximum_ctau_m: float = 1.0e3
    coarse_factor: float = 1.7
    bisection_steps: int = 14
    scan_points: int = 12
    reference_ctau_m: float = 1.0e3
    diagnostic_endpoint_convention: str = "fixed_step_log_bisection_midpoint"


@dataclass(frozen=True)
class TemplateSettings:
    """Settings for the frozen detector-level lifetime template bank.

    Production threshold crossings come from log-log interpolation of the scan
    rows.  Each interpolated (non-scan-boundary) endpoint is moved inward by
    ``log_endpoint_padding_fraction`` of the log interval; scan-boundary
    endpoints are unchanged.  The bisection midpoint is only a scan diagnostic.
    """

    lifetime_points_per_model: int = 20
    observable_endpoint_convention: str = "log_log_rate_interpolation"
    log_endpoint_padding_fraction: float = 2.0e-3
    energy_lower_bound_convention: str = "mass_gev"
    energy_max_gev: float = 400.0
    initial_energy_bins: int = 50
    minimum_bin_n_eff: float = 100.0
    jeffreys_alpha: float = 0.5
    seed_offset: int = 0


@dataclass(frozen=True)
class ProfileLikelihoodSettings:
    """Shape-only, independently lifetime-profiled pseudoexperiment settings."""

    pseudoexperiments_per_truth_and_seed: int = 100_000
    base_seed: int = 73_241
    number_of_seeds: int = 5
    seed_step: int = 10_003
    maximum_observed_events: int = 12
    chunk_size: int = 5_000
    target_accuracy: float = 0.90
    tie_tolerance: float = 1.0e-12
    persistent_criterion: str = "all_larger_tested_event_counts"
    truth_lifetime_grid: str = "all"
    profile_lifetime_grid: str = "all"
    rebin_factor: int = 1
    shape_only: bool = True
    independent_lifetime_profiling: bool = True

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(
            self.base_seed + self.seed_step * index
            for index in range(self.number_of_seeds)
        )


@dataclass(frozen=True)
class SeedPolicy:
    """Exact frozen-reference EventCalc, resampling and ECAL random-stream mapping."""

    base_seed: int = 54_321
    mass_stride: int = 10_000
    model_stride: int = 100
    source_stride: int = 1_000
    true_sample_seed_offset: int = 1
    ecal_seed_offset: int = 2
    mass_order_gev: tuple[float, ...] = PRODUCTION_MASSES_GEV
    model_order: tuple[str, ...] = PRODUCTION_MODEL_ORDER

    def mass_index(self, mass_gev: float) -> int:
        for index, configured_mass in enumerate(self.mass_order_gev):
            if isclose(mass_gev, configured_mass, rel_tol=0.0, abs_tol=1.0e-12):
                return index
        raise ValueError(
            f"mass {mass_gev!r} is not in the stable mass order {self.mass_order_gev}"
        )

    def model_index(self, model_id: str) -> int:
        try:
            return self.model_order.index(model_id)
        except ValueError as error:
            raise ValueError(
                f"model {model_id!r} is not in the stable model order {self.model_order}"
            ) from error

    def model_seed_from_indices(
        self, mass_index: int, model_index: int, *, seed_offset: int = 0,
    ) -> int:
        if mass_index < 0 or model_index < 0 or seed_offset < 0:
            raise ValueError("seed indices and seed offset must be non-negative")
        return (
            self.base_seed
            + self.mass_stride * mass_index
            + self.model_stride * model_index
            + seed_offset
        )

    def model_seed(
        self, mass_gev: float, model_id: str, *, seed_offset: int = 0,
    ) -> int:
        return self.model_seed_from_indices(
            self.mass_index(mass_gev),
            self.model_index(model_id),
            seed_offset=seed_offset,
        )

    def mass_index_from_model_seed(
        self, model_seed: int, model_id: str, *, seed_offset: int = 0,
    ) -> int:
        """Recover and validate the deterministic mass index in a model seed.

        This supports Week-8 masses appended after the frozen Week-7 mass order
        without changing the seeds of the original masses.
        """
        if model_seed < 0 or seed_offset < 0:
            raise ValueError("model seed and seed offset must be non-negative")
        model_base = (
            self.base_seed
            + self.model_stride * self.model_index(model_id)
            + seed_offset
        )
        delta = int(model_seed) - model_base
        if delta < 0 or delta % self.mass_stride != 0:
            raise ValueError(
                "model_seed disagrees with the deterministic profile seed policy"
            )
        return delta // self.mass_stride

    def source_proposal_seed_from_model_seed(
        self, model_seed: int, source_index: int,
    ) -> int:
        if model_seed < 0 or source_index < 0:
            raise ValueError("model_seed and source_index must be non-negative")
        return int(model_seed) + self.source_stride * source_index

    def true_sample_seed_from_model_seed(
        self, model_seed: int, source_index: int,
    ) -> int:
        return (
            self.source_proposal_seed_from_model_seed(model_seed, source_index)
            + self.true_sample_seed_offset
        )

    def ecal_seed_from_model_seed(
        self, model_seed: int, source_index: int,
    ) -> int:
        return (
            self.source_proposal_seed_from_model_seed(model_seed, source_index)
            + self.ecal_seed_offset
        )

    def source_proposal_seed(
        self, mass_gev: float, model_id: str, source_index: int,
        *, seed_offset: int = 0,
    ) -> int:
        return self.source_proposal_seed_from_model_seed(
            self.model_seed(mass_gev, model_id, seed_offset=seed_offset),
            source_index,
        )

    def true_sample_seed(
        self, mass_gev: float, model_id: str, source_index: int,
        *, seed_offset: int = 0,
    ) -> int:
        return self.true_sample_seed_from_model_seed(
            self.model_seed(mass_gev, model_id, seed_offset=seed_offset),
            source_index,
        )

    def ecal_seed(
        self, mass_gev: float, model_id: str, source_index: int,
        *, seed_offset: int = 0,
    ) -> int:
        return self.ecal_seed_from_model_seed(
            self.model_seed(mass_gev, model_id, seed_offset=seed_offset),
            source_index,
        )


@dataclass(frozen=True)
class DiscriminationSettings:
    """Compatibility settings for the earlier same-lifetime workflows."""

    minimum_bin_n_eff: float
    jeffreys_alpha: float = 0.5
    pseudoexperiments: int = 20_000
    maximum_observed_events: int = 100
    validation_pseudoexperiments: int = 100_000
    validation_maximum_events: int = 15
    validation_seeds: int = 5
    maximum_threshold_spread: int = 1
    target_accuracies: tuple[float, ...] = (0.90, 0.95, 0.99)
    lifetime_points: tuple[tuple[str, float], ...] = (
        ("low", 0.10), ("mid", 0.50), ("high", 0.90)
    )


@dataclass(frozen=True)
class EventDensitySettings:
    photon_masses: int = 50
    su2_masses: int = 50
    coupling_points: int = 111
    sampling_ctau_m: float = 1.0e99
    event_levels: tuple[float, ...] = (2.3, 3.0, 10.0, 30.0, 100.0)
    endpoint_refinement_points: int = 3
    endpoint_relative_width: float = 5.0e-3


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete immutable settings for one cache/output namespace."""

    name: str
    masses_gev: tuple[float, ...]
    exposure_pot: float
    n_eff_warning: float
    selection_name: str
    ecal_geometry: ECALGeometrySettings
    seed_policy: SeedPolicy
    ctau_sampling: SamplingSettings
    template_sampling: SamplingSettings
    event_density_sampling: SamplingSettings
    lifetimes: LifetimeSettings
    templates: TemplateSettings
    profiled_likelihood: ProfileLikelihoodSettings
    discrimination: DiscriminationSettings
    event_density: EventDensitySettings

    @property
    def spectrum_sampling(self) -> SamplingSettings:
        """Compatibility name for pre-refactor spectrum workflows."""
        return self.template_sampling

    @property
    def energy_max_gev(self) -> float:
        return self.templates.energy_max_gev

    @property
    def initial_energy_bins(self) -> int:
        return self.templates.initial_energy_bins


_PRODUCTION_SAMPLING = SamplingSettings(10_000_000, 1_000_000)
_PRODUCTION_SEED_POLICY = SeedPolicy()
_PRODUCTION_ECAL = ECALGeometrySettings()

PRODUCTION = AnalysisConfig(
    name="production",
    masses_gev=PRODUCTION_MASSES_GEV,
    exposure_pot=6.0e20,
    n_eff_warning=20.0,
    selection_name="diphoton_ecal",
    ecal_geometry=_PRODUCTION_ECAL,
    seed_policy=_PRODUCTION_SEED_POLICY,
    ctau_sampling=_PRODUCTION_SAMPLING,
    template_sampling=_PRODUCTION_SAMPLING,
    event_density_sampling=SamplingSettings(1_000_000, 100_000),
    lifetimes=LifetimeSettings(),
    templates=TemplateSettings(),
    profiled_likelihood=ProfileLikelihoodSettings(),
    discrimination=DiscriminationSettings(minimum_bin_n_eff=100.0),
    event_density=EventDensitySettings(),
)

QUICK = replace(
    PRODUCTION,
    name="quick",
    masses_gev=(0.3,),
    n_eff_warning=5.0,
    ctau_sampling=SamplingSettings(20_000, 4_000),
    template_sampling=SamplingSettings(20_000, 4_000),
    event_density_sampling=SamplingSettings(10_000, 2_000),
    lifetimes=replace(PRODUCTION.lifetimes, coarse_factor=5.0, bisection_steps=4, scan_points=3),
    templates=replace(PRODUCTION.templates, lifetime_points_per_model=5, initial_energy_bins=20, minimum_bin_n_eff=5.0),
    profiled_likelihood=replace(PRODUCTION.profiled_likelihood, pseudoexperiments_per_truth_and_seed=500, number_of_seeds=1, maximum_observed_events=8),
    discrimination=DiscriminationSettings(
        minimum_bin_n_eff=5.0, pseudoexperiments=500, maximum_observed_events=8,
        validation_pseudoexperiments=1_000, validation_maximum_events=8,
        validation_seeds=2,
    ),
    event_density=EventDensitySettings(
        photon_masses=2, su2_masses=2, coupling_points=7,
        endpoint_relative_width=10.0,
    ),
)

VALIDATION = replace(
    PRODUCTION,
    name="validation",
    ctau_sampling=SamplingSettings(1_000_000, 100_000),
    template_sampling=SamplingSettings(1_000_000, 100_000),
    profiled_likelihood=replace(
        PRODUCTION.profiled_likelihood,
        pseudoexperiments_per_truth_and_seed=20_000,
    ),
)

# Kept as a separate namespace for callers of the original analysis2 smoke profile.
SMOKE = replace(QUICK, name="smoke")

PROFILES: Mapping[str, AnalysisConfig] = MappingProxyType(
    {profile.name: profile for profile in (PRODUCTION, QUICK, VALIDATION, SMOKE)}
)


def get_config(profile: str) -> AnalysisConfig:
    try:
        return PROFILES[profile]
    except KeyError as error:
        raise ValueError(
            f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}"
        ) from error


def lower_ctau_m(mass_gev: float) -> float:
    """Frozen scan lower bound: 3 m * mass/(0.3 GeV)."""
    if mass_gev <= 0.0:
        raise ValueError("mass_gev must be positive")
    # Preserve the legacy operation order.  The unparenthesized equivalent is
    # one ULP higher at 0.4 and 0.9 GeV.
    return 3.0 * (mass_gev / 0.3)


def template_model_seed(mass_gev: float, model_id: str, seed_offset: int = 0) -> int:
    return PRODUCTION.seed_policy.model_seed(
        mass_gev, model_id, seed_offset=seed_offset,
    )


def template_source_seed(
    mass_gev: float, model_id: str, source_index: int, seed_offset: int = 0,
) -> int:
    return PRODUCTION.seed_policy.source_proposal_seed(
        mass_gev, model_id, source_index, seed_offset=seed_offset,
    )


def template_true_sample_seed(
    mass_gev: float, model_id: str, source_index: int, seed_offset: int = 0,
) -> int:
    return PRODUCTION.seed_policy.true_sample_seed(
        mass_gev, model_id, source_index, seed_offset=seed_offset,
    )


def template_ecal_seed(
    mass_gev: float, model_id: str, source_index: int, seed_offset: int = 0,
) -> int:
    return PRODUCTION.seed_policy.ecal_seed(
        mass_gev, model_id, source_index, seed_offset=seed_offset,
    )


def profiled_likelihood_seeds(
    config: AnalysisConfig = PRODUCTION,
) -> tuple[int, ...]:
    return config.profiled_likelihood.seeds


# Compatibility helpers for the original analysis2 workflows.
def ctau_model_seed(model_index: int, mass_index: int) -> int:
    return PRODUCTION.seed_policy.model_seed_from_indices(mass_index, model_index)


def ctau_source_seed(model_seed: int, source_index: int) -> int:
    return model_seed + PRODUCTION.seed_policy.source_stride * source_index


def spectrum_model_seed(mass_index: int, model_index: int) -> int:
    return PRODUCTION.seed_policy.model_seed_from_indices(mass_index, model_index)


def spectrum_source_seed(model_seed: int, source_index: int) -> int:
    return model_seed + PRODUCTION.seed_policy.source_stride * source_index


def pseudoexperiment_seed(point_index: int) -> int:
    return 20_260_723 + 10_000 * point_index


def validation_seed(point_index: int, seed_index: int) -> int:
    return 20_260_724 + 100_000 * point_index + 1_000 * seed_index


def event_density_seed(source_offset: int, mass_index: int) -> int:
    return 24_680 + source_offset + 100 * mass_index
