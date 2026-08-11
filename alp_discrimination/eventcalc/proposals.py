"""Lifetime-independent EventCalc proposals and exact mother realization."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import platform
from typing import Mapping

import numpy as np
import scipy

from funcs.initLLP import LLP
from funcs.kinematics import Grids
from funcs.ship_setup import theta_max_dec_vol, x_max, y_max, z_max, z_min

from alp_discrimination.cache import CacheStore, file_fingerprint
from alp_discrimination.config import AnalysisConfig, SamplingSettings, lower_ctau_m
from alp_discrimination.physics.models import ModelDefinition, ProductionSource


ADAPTER_VERSION = 4

# EventCalc uses e_min = max(m, min(2.133*m/c_tau, 0.5*E_max)).
# At and above this lifetime the kinematic proposal has full E_a >= m_a support.
EVENTCALC_FULL_SUPPORT_CTAU_M = float(np.nextafter(2.133, np.inf))


@contextmanager
def legacy_numpy_seed(seed: int):
    """Isolate EventCalc's global RandomState while preserving its exact stream."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass(frozen=True)
class MotherSample:
    """Mother-ALP kinematics after decay-volume geometry; GeV, rad and metres."""

    px_gev: np.ndarray
    py_gev: np.ndarray
    pz_gev: np.ndarray
    energy_gev: np.ndarray
    decay_probability: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    mass_gev: float

    def __len__(self) -> int:
        return len(self.energy_gev)


@dataclass
class KinematicProposal:
    """Lifetime-independent resampled (theta,E) proposal for a guarded domain."""

    model_id: str
    source: str
    mass_gev: float
    proposal_ctau_m: float
    proposal_seed: int
    interpolation_points: int
    resample_size: int
    r_theta_rad: np.ndarray
    r_energy_gev: np.ndarray
    epsilon_polar: float
    visible_br: float
    yield_per_pot_per_coupling_squared: float
    unit_coupling_ctau_m: float
    theta_min_rad: float
    theta_max_rad: float
    sanitation_policy: str
    input_fingerprints: tuple[dict, ...]
    energy_support_mode: str = "full_Ea_ge_mass"
    e_min_sampling_min_gev: float | None = None
    e_min_sampling_max_gev: float | None = None
    cache_key: str | None = None

    def __post_init__(self) -> None:
        self.r_theta_rad = np.asarray(self.r_theta_rad, float)
        self.r_energy_gev = np.asarray(self.r_energy_gev, float)
        if self.r_theta_rad.ndim != 1 or self.r_theta_rad.shape != self.r_energy_gev.shape:
            raise ValueError("proposal theta and energy arrays must be one-dimensional and equal")
        if len(self.r_theta_rad) != self.resample_size or not np.all(np.isfinite(self.r_theta_rad)):
            raise ValueError("proposal arrays do not match the configured resample size")
        if not np.all(np.isfinite(self.r_energy_gev)) or np.any(self.r_energy_gev < self.mass_gev):
            raise ValueError("proposal energies must be finite and at least the mass")
        if self.mass_gev <= 0.0 or self.proposal_ctau_m <= 0.0 or self.epsilon_polar <= 0.0:
            raise ValueError("proposal mass, lifetime and polar efficiency must be positive")
        if self.energy_support_mode not in {
            "full_Ea_ge_mass",
            "lifetime_specific_truncated_Ea",
        }:
            raise ValueError(f"unknown proposal energy support {self.energy_support_mode!r}")
        for value in (self.e_min_sampling_min_gev, self.e_min_sampling_max_gev):
            if value is not None and (not np.isfinite(value) or value < self.mass_gev):
                raise ValueError("proposal minimum-energy metadata must be finite and >= mass")

    def arrays(self) -> dict[str, np.ndarray]:
        return {"r_theta_rad": self.r_theta_rad, "r_energy_gev": self.r_energy_gev}

    def metadata(self) -> dict:
        return {
            key: value for key, value in vars(self).items()
            if key not in {"r_theta_rad", "r_energy_gev", "cache_key"}
        }

    @classmethod
    def from_cache(cls, arrays: Mapping[str, np.ndarray], metadata: Mapping) -> "KinematicProposal":
        fields = cls.__dataclass_fields__
        values = {key: metadata[key] for key in fields if key in metadata and key != "cache_key"}
        values.update(r_theta_rad=arrays["r_theta_rad"], r_energy_gev=arrays["r_energy_gev"])
        values["input_fingerprints"] = tuple(values["input_fingerprints"])
        values["cache_key"] = metadata.get("cache_key")
        return cls(**values)


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _validate_proposal_arrays(arrays: dict[str, np.ndarray], metadata: dict) -> None:
    proposal = KinematicProposal.from_cache(arrays, metadata)
    identity = metadata["identity"]
    expected = {
        "model_id": identity["model"], "source": identity["source"],
        "mass_gev": identity["mass_gev"], "proposal_ctau_m": identity["proposal_ctau_m"],
        "proposal_seed": identity["seed"], "sanitation_policy": identity["sanitation_policy"],
    }
    if any(getattr(proposal, name) != value for name, value in expected.items()):
        raise ValueError("proposal metadata disagrees with its cache identity")
    sampling = identity["sampling"]
    if (proposal.interpolation_points, proposal.resample_size) != (
        sampling["interpolation_points"], sampling["resample_size"]
    ):
        raise ValueError("proposal sampling metadata disagrees with its cache identity")


def _input_fingerprints(
    model: ModelDefinition,
    source: ProductionSource,
) -> tuple[dict, ...]:
    return tuple(file_fingerprint(path) for path in model.input_paths(source))


def _proposal_identity(
    config: AnalysisConfig, model: ModelDefinition, source: ProductionSource,
    mass_gev: float, proposal_ctau_m: float, seed: int, sampling: SamplingSettings,
    sanitation_policy: str, fingerprints: tuple[dict, ...],
) -> dict:
    return {
        "adapter_version": ADAPTER_VERSION, "profile": config.name,
        "model": model.identifier, "eventcalc_model": model.eventcalc_name,
        "source": source.identifier, "eventcalc_mode": source.eventcalc_mode,
        "mass_gev": mass_gev, "proposal_ctau_m": proposal_ctau_m, "seed": seed,
        "sampling": asdict(sampling), "theta_max_sim_rad": theta_max_dec_vol,
        "emin_policy": "adaptive_exact_lifetime_or_full_support",
        "sanitation_policy": sanitation_policy,
        "inputs": fingerprints, "runtime": _runtime_versions(),
    }


def _sanitize_interpolation(kin: Grids, policy: str) -> None:
    values = np.asarray(kin.interpolated_values, float)
    widths = np.asarray(kin.max_energy - kin.e_min_sampling, float)
    if policy == "event_density_legacy":
        outside = (kin.energy < np.min(kin.grid_z) - 1e-12 * max(1.0, np.max(np.abs(kin.grid_z)))) | (
            kin.energy > np.max(kin.grid_z) + 1e-12 * max(1.0, np.max(np.abs(kin.grid_z)))
        )
        values = values.copy()
        values[outside] = 0.0
        energy_scale = max(
            1.0, float(np.max(np.abs(kin.max_energy))),
            float(np.max(np.abs(kin.e_min_sampling))),
        )
        if float(np.min(widths)) < -1e-12 * energy_scale:
            raise RuntimeError("substantially negative proposal energy interval")
        kin.max_energy = np.maximum(kin.max_energy, kin.e_min_sampling)
        widths = kin.max_energy - kin.e_min_sampling
        negative = values < 0.0
        if np.any(negative):
            positive_weight = float(np.sum(np.clip(values, 0.0, None) * widths))
            negative_weight = float(np.sum(np.clip(-values, 0.0, None) * widths))
            if positive_weight <= 0.0 or negative_weight / positive_weight > 1e-3:
                raise RuntimeError("negative interpolation contribution exceeds the legacy 1e-3 limit")
            values = np.clip(values, 0.0, None)
        kin.interpolated_values = values
    elif policy != "strict_core":
        raise ValueError(f"unknown interpolation sanitation policy {policy!r}")
    weights = np.asarray(kin.interpolated_values, float) * widths
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise RuntimeError("EventCalc interpolation produced invalid proposal weights")


def generate_mother_sample(
    proposal: KinematicProposal,
    ctau_m: float,
    seed: int,
) -> MotherSample:
    """Reproduce Grids.true_samples exactly from cached plain proposal arrays."""
    if ctau_m <= 0.0:
        raise ValueError("ctau_m must be positive")
    theta, energy, mass = proposal.r_theta_rad, proposal.r_energy_gev, proposal.mass_gev
    rng = np.random.RandomState(seed)
    phi = rng.uniform(-np.pi, np.pi, proposal.resample_size)
    momentum_abs = np.sqrt(energy**2 - mass**2)
    px = momentum_abs * np.cos(phi) * np.sin(theta)
    py = momentum_abs * np.sin(phi) * np.sin(theta)
    pz = momentum_abs * np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cmin = 1.0 - np.exp(-z_min * mass / (np.cos(theta) * ctau_m * momentum_abs))
        cmax = 1.0 - np.exp(-z_max * mass / (np.cos(theta) * ctau_m * momentum_abs))
        c = rng.uniform(cmin, cmax)
        safe_c = np.minimum(c, 0.9999999995)
        z = np.where(
            c > 0.9999999995, z_min,
            np.cos(theta) * ctau_m * (momentum_abs / mass) * np.log(1.0 / (1.0 - safe_c)),
        )
        decay_probability = (
            np.exp(-z_min * mass / (np.cos(theta) * ctau_m * momentum_abs))
            - np.exp(-z_max * mass / (np.cos(theta) * ctau_m * momentum_abs))
        )
    x = z * np.cos(phi) * np.tan(theta)
    y = z * np.sin(phi) * np.tan(theta)
    geometry_accepted = (
        (-x_max(z) < x) & (x < x_max(z)) & (-y_max(z) < y) & (y < y_max(z))
        & (z_min <= z) & (z <= z_max)
    )
    # The frozen reference implementation applied this validity filter to
    # Grids.get_kinematics() before ECAL selection.  Valid EventCalc proposals
    # make it a no-op, but retaining it here preserves the legacy boundary.
    accepted = (
        geometry_accepted
        & np.isfinite(energy)
        & np.isfinite(decay_probability)
        & (decay_probability >= 0.0)
    )
    return MotherSample(
        px_gev=px[accepted],
        py_gev=py[accepted],
        pz_gev=pz[accepted],
        energy_gev=energy[accepted],
        decay_probability=decay_probability[accepted],
        x_m=x[accepted],
        y_m=y[accepted],
        z_m=z[accepted],
        mass_gev=mass,
    )


class ProposalGenerator:
    """Prepare and retain one lifetime-independent proposal per cache identity."""

    def __init__(self, config: AnalysisConfig, cache: CacheStore):
        self.config = config
        self.cache = cache
        self.proposals: dict[str, KinematicProposal] = {}

    def sampling(self, stage: str) -> SamplingSettings:
        try:
            return {
                "ctau": self.config.ctau_sampling,
                "spectrum": self.config.spectrum_sampling,
                "event_density": self.config.event_density_sampling,
            }[stage]
        except KeyError as error:
            raise ValueError("stage must be ctau, spectrum or event_density") from error

    def prepare(
        self,
        model: ModelDefinition,
        source: ProductionSource,
        mass_gev: float,
        seed: int,
        stage: str,
        *,
        force: bool,
        proposal_ctau_m: float | None = None,
    ) -> KinematicProposal:
        sampling = self.sampling(stage)
        if proposal_ctau_m is None:
            proposal_ctau_m = (
                self.config.event_density.sampling_ctau_m
                if stage == "event_density"
                else lower_ctau_m(mass_gev)
            )
        proposal_ctau_m = float(proposal_ctau_m)
        if not np.isfinite(proposal_ctau_m) or proposal_ctau_m <= 0.0:
            raise ValueError("proposal_ctau_m must be finite and positive")
        policy = "event_density_legacy" if stage == "event_density" else "strict_core"
        fingerprints = _input_fingerprints(model, source)
        identity = _proposal_identity(
            self.config, model, source, mass_gev, proposal_ctau_m, seed,
            sampling, policy, fingerprints,
        )
        _, _, key = self.cache.paths("proposal", identity)
        if key in self.proposals:
            return self.proposals[key]
        if not force:
            loaded = self.cache.load("proposal", identity, _validate_proposal_arrays)
            if loaded:
                proposal = KinematicProposal.from_cache(*loaded)
                self.proposals[key] = proposal
                return proposal
        else:
            print(f"CACHE FORCED   [proposal] {key[:12]}")

        llp = LLP(
            mass=None, particle_selection=model.particle_selection, mixing_pattern=None,
            uncertainty=None, alp_production_mode=source.eventcalc_mode,
        )
        if not llp.mass_range[0] <= mass_gev <= llp.mass_range[1]:
            raise ValueError(
                f"mass {mass_gev:g} GeV lies outside "
                f"{model.identifier} tables {llp.mass_range}"
            )
        llp.set_mass(mass_gev)
        llp.compute_mass_dependent_properties()
        llp.set_c_tau(proposal_ctau_m)
        with legacy_numpy_seed(seed):
            kin = Grids(
                llp.Distr, llp.Energy_distr, sampling.interpolation_points,
                mass_gev, proposal_ctau_m, theta_max_sim=theta_max_dec_vol,
            )
            kin.interpolate(False)
            full_energy_support = bool(
                np.allclose(
                    kin.e_min_sampling,
                    mass_gev,
                    rtol=1.0e-12,
                    atol=1.0e-14,
                )
            )
            energy_support_mode = (
                "full_Ea_ge_mass"
                if full_energy_support
                else "lifetime_specific_truncated_Ea"
            )
            _sanitize_interpolation(kin, policy)
            kin.resample(sampling.resample_size, False)
        proposal = KinematicProposal(
            model_id=model.identifier, source=source.identifier, mass_gev=mass_gev,
            proposal_ctau_m=proposal_ctau_m, proposal_seed=seed,
            interpolation_points=sampling.interpolation_points,
            resample_size=sampling.resample_size,
            r_theta_rad=kin.r_theta, r_energy_gev=kin.r_energy,
            epsilon_polar=float(kin.epsilon_polar),
            visible_br=float(np.sum(llp.BrRatios_distr)),
            yield_per_pot_per_coupling_squared=float(llp.Yield),
            unit_coupling_ctau_m=float(llp.c_tau_int),
            theta_min_rad=float(kin.thetamin), theta_max_rad=float(kin.theta_max),
            sanitation_policy=policy, input_fingerprints=fingerprints,
            energy_support_mode=energy_support_mode,
            e_min_sampling_min_gev=float(np.min(kin.e_min_sampling)),
            e_min_sampling_max_gev=float(np.max(kin.e_min_sampling)),
        )
        metadata = self.cache.save(
            "proposal", identity, proposal.arrays(), proposal.metadata()
        )
        proposal.cache_key = metadata.get("cache_key", key)
        self.proposals[key] = proposal
        return proposal


__all__ = [
    "ADAPTER_VERSION",
    "EVENTCALC_FULL_SUPPORT_CTAU_M",
    "KinematicProposal",
    "MotherSample",
    "ProposalGenerator",
    "generate_mother_sample",
    "legacy_numpy_seed",
]
