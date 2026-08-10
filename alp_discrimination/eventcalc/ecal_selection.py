"""Daughter-level diphoton ECAL geometry and Monte Carlo selection.

The random generator, draw order, boost, straight-line projection and inclusive
rectangle are kept fixed for numerical reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .selections import MotherSampleLike, SelectionContext
else:
    MotherSampleLike = Any
    SelectionContext = Any


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DIPHOTON_ECAL_ALGORITHM: Final[str] = (
    "isotropic_scalar_to_two_photons;lorentz_boost;straight_line_projection;"
    "inclusive_rectangle;require_both_photons"
)
DIPHOTON_ECAL_ALGORITHM_VERSION: Final[int] = 1
DEFAULT_ECAL_SEED_OFFSET: Final[int] = 2


@dataclass(frozen=True)
class ECALGeometry:
    """Rectangular ECAL plane perpendicular to the beam axis."""

    z_m: float = 95.0
    width_x_m: float = 4.0
    height_y_m: float = 6.0
    centre_x_m: float = 0.0
    centre_y_m: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.z_m,
                self.width_x_m,
                self.height_y_m,
                self.centre_x_m,
                self.centre_y_m,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("all ECAL geometry parameters must be finite")
        if self.width_x_m <= 0.0 or self.height_y_m <= 0.0:
            raise ValueError("ECAL width and height must be positive")

    @property
    def half_width_x_m(self) -> float:
        return 0.5 * self.width_x_m

    @property
    def half_height_y_m(self) -> float:
        return 0.5 * self.height_y_m


DEFAULT_ECAL: Final[ECALGeometry] = ECALGeometry()


@dataclass(frozen=True)
class DiphotonECALResult:
    """Per-event photon momenta, ECAL intercepts and acceptance masks."""

    event_mask: BoolArray
    photon_1_hit_mask: BoolArray
    photon_2_hit_mask: BoolArray
    photon_1_four_momentum: FloatArray
    photon_2_four_momentum: FloatArray
    photon_1_x_ecal_m: FloatArray
    photon_1_y_ecal_m: FloatArray
    photon_2_x_ecal_m: FloatArray
    photon_2_y_ecal_m: FloatArray


def _validated_mother_arrays(
    sample: MotherSampleLike,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    momentum = np.column_stack((sample.px_gev, sample.py_gev, sample.pz_gev)).astype(
        float,
        copy=False,
    )
    energy = np.asarray(sample.energy_gev, dtype=float)
    vertices = np.column_stack((sample.x_m, sample.y_m, sample.z_m)).astype(
        float,
        copy=False,
    )
    probabilities = np.asarray(sample.decay_probability, dtype=float)
    mass = float(sample.mass_gev)

    number_of_events = len(sample)
    if momentum.shape != (number_of_events, 3):
        raise ValueError("mother momentum arrays must be one-dimensional and equal")
    if vertices.shape != (number_of_events, 3):
        raise ValueError("mother vertex arrays must be one-dimensional and equal")
    if energy.shape != (number_of_events,) or probabilities.shape != (number_of_events,):
        raise ValueError("mother energy and probability arrays have the wrong shape")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mother mass must be finite and positive")
    if not (
        np.all(np.isfinite(momentum))
        and np.all(np.isfinite(energy))
        and np.all(np.isfinite(vertices))
        and np.all(np.isfinite(probabilities))
    ):
        raise ValueError("mother kinematics must be finite")
    if np.any(energy <= 0.0):
        raise ValueError("mother energies must be positive")

    invariant_mass_squared = energy**2 - np.sum(momentum**2, axis=1)
    if not np.allclose(
        invariant_mass_squared,
        mass**2,
        rtol=2.0e-8,
        atol=1.0e-10,
    ):
        maximum_residual = float(np.max(np.abs(invariant_mass_squared - mass**2)))
        raise ValueError(
            "mother four-momenta are inconsistent with mass_gev; "
            f"maximum |E^2-p^2-m^2| = {maximum_residual:.6g} GeV^2"
        )
    return momentum, energy, vertices, mass


def sample_diphoton_lab_four_momenta(
    sample: MotherSampleLike,
    *,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Sample isotropic ``a -> gamma gamma`` decays and boost to the lab."""

    mother_momentum, mother_energy, _, mass = _validated_mother_arrays(sample)
    number_of_events = len(sample)
    if number_of_events == 0:
        empty = np.empty((0, 4), dtype=float)
        return empty.copy(), empty.copy()

    random_generator = np.random.default_rng(seed)
    cos_theta_star = random_generator.uniform(-1.0, 1.0, number_of_events)
    sin_theta_star = np.sqrt(np.maximum(0.0, 1.0 - cos_theta_star**2))
    phi_star = random_generator.uniform(0.0, 2.0 * np.pi, number_of_events)
    unit_vectors = np.column_stack(
        (
            sin_theta_star * np.cos(phi_star),
            sin_theta_star * np.sin(phi_star),
            cos_theta_star,
        )
    )

    photon_energy_star = np.full(number_of_events, 0.5 * mass, dtype=float)
    photon_1_momentum_star = photon_energy_star[:, None] * unit_vectors
    photon_2_momentum_star = -photon_1_momentum_star

    beta = mother_momentum / mother_energy[:, None]
    beta_squared = np.sum(beta**2, axis=1)
    gamma = mother_energy / mass
    if np.any(beta_squared >= 1.0):
        raise ValueError("at least one mother particle has |beta| >= 1")

    gamma_minus_one_over_beta_squared = np.divide(
        gamma - 1.0,
        beta_squared,
        out=np.zeros_like(gamma),
        where=beta_squared > 1.0e-30,
    )

    def boost_one(photon_momentum_star: FloatArray) -> FloatArray:
        beta_dot_p_star = np.sum(beta * photon_momentum_star, axis=1)
        boost_coefficient = (
            gamma_minus_one_over_beta_squared * beta_dot_p_star
            + gamma * photon_energy_star
        )
        momentum_lab = photon_momentum_star + boost_coefficient[:, None] * beta
        energy_lab = gamma * (photon_energy_star + beta_dot_p_star)
        return np.column_stack((momentum_lab, energy_lab))

    return boost_one(photon_1_momentum_star), boost_one(photon_2_momentum_star)


def project_particles_to_ecal(
    decay_vertices_m: np.ndarray,
    particle_four_momenta: np.ndarray,
    *,
    geometry: ECALGeometry = DEFAULT_ECAL,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Project downstream straight trajectories onto the ECAL z-plane."""

    vertices = np.asarray(decay_vertices_m, dtype=float)
    momenta = np.asarray(particle_four_momenta, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("decay_vertices_m must have shape (N, 3)")
    if momenta.ndim != 2 or momenta.shape[1] != 4:
        raise ValueError("particle_four_momenta must have shape (N, 4)")
    if vertices.shape[0] != momenta.shape[0]:
        raise ValueError("vertex and momentum arrays must contain the same events")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(momenta)):
        raise ValueError("vertices and four-momenta must be finite")

    pz = momenta[:, 2]
    delta_z = geometry.z_m - vertices[:, 2]
    reaches_plane = (pz > 0.0) & (delta_z >= 0.0)
    propagation_parameter = np.divide(
        delta_z,
        pz,
        out=np.full_like(delta_z, np.nan),
        where=reaches_plane,
    )
    x_ecal = vertices[:, 0] + momenta[:, 0] * propagation_parameter
    y_ecal = vertices[:, 1] + momenta[:, 1] * propagation_parameter
    return x_ecal, y_ecal, reaches_plane


def rectangular_ecal_hit_mask(
    x_ecal_m: np.ndarray,
    y_ecal_m: np.ndarray,
    reaches_plane: np.ndarray,
    *,
    geometry: ECALGeometry = DEFAULT_ECAL,
) -> BoolArray:
    """Apply the legacy inclusive rectangular ECAL boundary."""

    x_ecal = np.asarray(x_ecal_m, dtype=float)
    y_ecal = np.asarray(y_ecal_m, dtype=float)
    reaches = np.asarray(reaches_plane, dtype=bool)
    if x_ecal.ndim != 1 or y_ecal.ndim != 1 or reaches.ndim != 1:
        raise ValueError("projected coordinates and mask must be one-dimensional")
    if not (len(x_ecal) == len(y_ecal) == len(reaches)):
        raise ValueError("projected-coordinate arrays must have identical lengths")
    return (
        reaches
        & (np.abs(x_ecal - geometry.centre_x_m) <= geometry.half_width_x_m)
        & (np.abs(y_ecal - geometry.centre_y_m) <= geometry.half_height_y_m)
    )


@dataclass(frozen=True)
class DiphotonECALSelection:
    """Require both sampled photons to intersect the configured ECAL plane."""

    geometry: ECALGeometry = DEFAULT_ECAL
    seed_offset: int = DEFAULT_ECAL_SEED_OFFSET
    name: ClassVar[str] = "diphoton_ecal"

    def __post_init__(self) -> None:
        if not isinstance(self.seed_offset, (int, np.integer)):
            raise ValueError("ECAL seed offset must be an integer")

    def selection_seed(self, context: SelectionContext) -> int:
        seed = int(context.source_seed) + int(self.seed_offset)
        if seed < 0:
            raise ValueError("resolved ECAL seed must be non-negative")
        return seed

    def cache_identity(self, context: SelectionContext) -> dict:
        return {
            "name": self.name,
            "algorithm": DIPHOTON_ECAL_ALGORITHM,
            "algorithm_version": DIPHOTON_ECAL_ALGORITHM_VERSION,
            "rng": "numpy.random.default_rng",
            "draw_order": ["cos_theta_star", "phi_star"],
            "rectangle_boundary": "inclusive",
            "both_photons_required": True,
            "geometry": asdict(self.geometry),
            "source_seed": int(context.source_seed),
            "seed_offset": int(self.seed_offset),
            "selection_seed": self.selection_seed(context),
        }

    def details(
        self,
        sample: MotherSampleLike,
        context: SelectionContext,
    ) -> DiphotonECALResult:
        photon_1, photon_2 = sample_diphoton_lab_four_momenta(
            sample,
            seed=self.selection_seed(context),
        )
        vertices = np.column_stack((sample.x_m, sample.y_m, sample.z_m))
        photon_1_x, photon_1_y, photon_1_reaches = project_particles_to_ecal(
            vertices,
            photon_1,
            geometry=self.geometry,
        )
        photon_2_x, photon_2_y, photon_2_reaches = project_particles_to_ecal(
            vertices,
            photon_2,
            geometry=self.geometry,
        )
        photon_1_hits = rectangular_ecal_hit_mask(
            photon_1_x,
            photon_1_y,
            photon_1_reaches,
            geometry=self.geometry,
        )
        photon_2_hits = rectangular_ecal_hit_mask(
            photon_2_x,
            photon_2_y,
            photon_2_reaches,
            geometry=self.geometry,
        )
        return DiphotonECALResult(
            event_mask=photon_1_hits & photon_2_hits,
            photon_1_hit_mask=photon_1_hits,
            photon_2_hit_mask=photon_2_hits,
            photon_1_four_momentum=photon_1,
            photon_2_four_momentum=photon_2,
            photon_1_x_ecal_m=photon_1_x,
            photon_1_y_ecal_m=photon_1_y,
            photon_2_x_ecal_m=photon_2_x,
            photon_2_y_ecal_m=photon_2_y,
        )

    def mask(
        self,
        sample: MotherSampleLike,
        context: SelectionContext | None = None,
    ) -> BoolArray:
        if context is None:
            raise ValueError("DiphotonECALSelection requires an explicit SelectionContext")
        return self.details(sample, context).event_mask


__all__ = [
    "DEFAULT_ECAL",
    "DEFAULT_ECAL_SEED_OFFSET",
    "DIPHOTON_ECAL_ALGORITHM",
    "DIPHOTON_ECAL_ALGORITHM_VERSION",
    "DiphotonECALResult",
    "DiphotonECALSelection",
    "ECALGeometry",
    "project_particles_to_ecal",
    "rectangular_ecal_hit_mask",
    "sample_diphoton_lab_four_momenta",
]


@dataclass(frozen=True)
class DiphotonECALEnergySelection(DiphotonECALSelection):
    minimum_photon_energy_gev: float = 1.0
    name: ClassVar[str] = "diphoton_ecal_e1gev"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not np.isfinite(self.minimum_photon_energy_gev)
            or self.minimum_photon_energy_gev < 0.0
        ):
            raise ValueError(
                "minimum photon energy must be finite and non-negative"
            )

    def cache_identity(self, context: SelectionContext) -> dict:
        geometry_identity = super().cache_identity(context)
        return {
            **geometry_identity,
            "name": self.name,
            "algorithm": (
                "diphoton_ecal_geometry;"
                "inclusive_minimum_lab_photon_energy;"
                "require_both_photons"
            ),
            "algorithm_version": 1,
            "geometry_algorithm": geometry_identity["algorithm"],
            "geometry_algorithm_version": (
                geometry_identity["algorithm_version"]
            ),
            "minimum_photon_energy_gev": (
                self.minimum_photon_energy_gev
            ),
            "photon_energy_boundary": "inclusive",
        }

    def details(
        self,
        sample: MotherSampleLike,
        context: SelectionContext,
    ) -> DiphotonECALResult:
        geometry_result = super().details(sample, context)

        photon_energy_mask = (
            (
                geometry_result.photon_1_four_momentum[:, 3]
                >= self.minimum_photon_energy_gev
            )
            & (
                geometry_result.photon_2_four_momentum[:, 3]
                >= self.minimum_photon_energy_gev
            )
        )

        return replace(
            geometry_result,
            event_mask=geometry_result.event_mask & photon_energy_mask,
        )