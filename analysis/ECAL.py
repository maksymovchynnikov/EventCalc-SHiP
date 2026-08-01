from __future__ import annotations

"""
1. samples an isotropic a -> gamma gamma decay in the ALP rest frame,
2. boosts both photons to the laboratory frame,
3. propagates each photon in a straight line to a rectangular ECAL plane,
4. accepts the event only when both photons intersect that rectangle.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# EventCalc mother-table column indices.
PX_A: Final[int] = 0
PY_A: Final[int] = 1
PZ_A: Final[int] = 2
E_A: Final[int] = 3
M_A: Final[int] = 4
P_DECAY: Final[int] = 6
X_DECAY: Final[int] = 7
Y_DECAY: Final[int] = 8
Z_DECAY: Final[int] = 9

MIN_MOTHER_COLUMNS: Final[int] = 10
DEFAULT_SEED: Final[int] = 97531


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
            [
                self.z_m,
                self.width_x_m,
                self.height_y_m,
                self.centre_x_m,
                self.centre_y_m,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("All ECAL geometry parameters must be finite.")
        if self.width_x_m <= 0.0 or self.height_y_m <= 0.0:
            raise ValueError("ECAL width and height must be positive.")

    @property
    def half_width_x_m(self) -> float:
        return 0.5 * self.width_x_m

    @property
    def half_height_y_m(self) -> float:
        return 0.5 * self.height_y_m


DEFAULT_ECAL: Final[ECALGeometry] = ECALGeometry()


@dataclass(frozen=True)
class DiphotonECALResult:
    """
    Detailed result of one vectorised ECAL-acceptance calculation.

    Arrays have one entry per input mother particle.  The four-momenta use the
    column order (px, py, pz, E).
    """

    event_mask: BoolArray
    photon_1_hit_mask: BoolArray
    photon_2_hit_mask: BoolArray
    photon_1_four_momentum: FloatArray
    photon_2_four_momentum: FloatArray
    photon_1_x_ecal_m: FloatArray
    photon_1_y_ecal_m: FloatArray
    photon_2_x_ecal_m: FloatArray
    photon_2_y_ecal_m: FloatArray


def _as_mother_results(mother_particle_results: ArrayLike) -> FloatArray:
    """Validate and return the EventCalc mother table as a float array."""

    results = np.asarray(mother_particle_results, dtype=float)

    if results.ndim != 2:
        raise ValueError("mother_particle_results must be a two-dimensional array.")
    if results.shape[1] < MIN_MOTHER_COLUMNS:
        raise ValueError(
            "mother_particle_results must contain at least ten columns: "
            "(px, py, pz, E, m, PDG, P_decay, x, y, z)."
        )
    if not np.all(np.isfinite(results[:, :MIN_MOTHER_COLUMNS])):
        raise ValueError("The first ten mother-particle columns must be finite.")
    if np.any(results[:, E_A] <= 0.0):
        raise ValueError("All mother-particle energies must be positive.")
    if np.any(results[:, M_A] <= 0.0):
        raise ValueError("All mother-particle masses must be positive.")

    momentum_squared = np.sum(results[:, :3] ** 2, axis=1)
    invariant_mass_squared = results[:, E_A] ** 2 - momentum_squared
    expected_mass_squared = results[:, M_A] ** 2

    # Allow small interpolation / floating-point residuals, but catch wrong
    # column ordering or genuinely off-shell input.
    if not np.allclose(
        invariant_mass_squared,
        expected_mass_squared,
        rtol=2.0e-8,
        atol=1.0e-10,
    ):
        max_residual = float(
            np.max(np.abs(invariant_mass_squared - expected_mass_squared))
        )
        raise ValueError(
            "Mother four-momenta are inconsistent with the mass column. "
            f"Maximum |E^2-p^2-m^2| = {max_residual:.6g} GeV^2."
        )

    return results


def _resolve_rng(*, seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def sample_diphoton_lab_four_momenta(
    mother_particle_results: ArrayLike,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Sample and boost a -> gamma gamma for every EventCalc mother particle.

    The spin-zero two-photon decay is sampled isotropically in the ALP rest
    frame.  Each returned array has shape (N, 4) and column order
    (px, py, pz, E).
    """

    results = _as_mother_results(mother_particle_results)
    random_generator = _resolve_rng(seed=seed, rng=rng)

    number_of_events = results.shape[0]
    if number_of_events == 0:
        empty = np.empty((0, 4), dtype=float)
        return empty.copy(), empty.copy()

    # Uniform solid-angle sampling: cos(theta*) uniform in [-1, 1] and
    # phi* uniform in [0, 2 pi).
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

    masses = results[:, M_A]
    photon_energy_star = 0.5 * masses
    photon_1_momentum_star = photon_energy_star[:, None] * unit_vectors
    photon_2_momentum_star = -photon_1_momentum_star

    mother_energy = results[:, E_A]
    mother_momentum = results[:, :3]
    beta = mother_momentum / mother_energy[:, None]
    beta_squared = np.sum(beta**2, axis=1)
    gamma = mother_energy / masses

    if np.any(beta_squared >= 1.0):
        raise ValueError("At least one mother particle has |beta| >= 1.")

    # Lorentz boost of a rest-frame daughter:
    #
    # p_lab = p_* + [((gamma-1)/beta^2)(beta.p_*) + gamma E_*] beta
    # E_lab = gamma(E_* + beta.p_*)
    #
    # The beta -> 0 limit is finite.  Setting the coefficient to zero there
    # gives the correct no-boost result.
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

    photon_1 = boost_one(photon_1_momentum_star)
    photon_2 = boost_one(photon_2_momentum_star)

    return photon_1, photon_2


def project_particles_to_ecal(
    decay_vertices_m: ArrayLike,
    particle_four_momenta: ArrayLike,
    *,
    geometry: ECALGeometry = DEFAULT_ECAL,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """
    Project straight particle trajectories to the ECAL z-plane.

    Returns
    -------
    x_ecal_m, y_ecal_m, reaches_plane
        `reaches_plane` is true only for particles travelling downstream
        (p_z > 0) from a vertex upstream of the ECAL plane.
    """

    vertices = np.asarray(decay_vertices_m, dtype=float)
    momenta = np.asarray(particle_four_momenta, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("decay_vertices_m must have shape (N, 3).")
    if momenta.ndim != 2 or momenta.shape[1] != 4:
        raise ValueError("particle_four_momenta must have shape (N, 4).")
    if vertices.shape[0] != momenta.shape[0]:
        raise ValueError("Vertex and momentum arrays must contain the same events.")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(momenta)):
        raise ValueError("Vertices and four-momenta must be finite.")

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
    x_ecal_m: ArrayLike,
    y_ecal_m: ArrayLike,
    reaches_plane: ArrayLike,
    *,
    geometry: ECALGeometry = DEFAULT_ECAL,
) -> BoolArray:
    """Return whether each projected trajectory intersects the ECAL rectangle."""

    x_ecal = np.asarray(x_ecal_m, dtype=float)
    y_ecal = np.asarray(y_ecal_m, dtype=float)
    reaches = np.asarray(reaches_plane, dtype=bool)

    if x_ecal.ndim != 1 or y_ecal.ndim != 1 or reaches.ndim != 1:
        raise ValueError("Projected coordinates and mask must be one-dimensional.")
    if not (len(x_ecal) == len(y_ecal) == len(reaches)):
        raise ValueError("Projected-coordinate arrays must have identical lengths.")

    return (
        reaches
        & (np.abs(x_ecal - geometry.centre_x_m) <= geometry.half_width_x_m)
        & (np.abs(y_ecal - geometry.centre_y_m) <= geometry.half_height_y_m)
    )


def diphoton_ecal_acceptance(
    mother_particle_results: ArrayLike,
    *,
    geometry: ECALGeometry = DEFAULT_ECAL,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_details: bool = False,
) -> BoolArray | DiphotonECALResult:
    """
    Evaluate the daughter-level ECAL acceptance for a -> gamma gamma.

    An event passes precisely when both photons travel downstream and intersect
    the rectangular ECAL plane.

    Parameters
    ----------
    mother_particle_results:
        EventCalc mother table with at least the first ten standard columns.
    geometry:
        ECAL plane and rectangle definition.
    seed, rng:
        Reproducible random-number control.  Pass at most one.
    return_details:
        If false, return only the event mask.  If true, also return photon
        four-momenta, projected coordinates and individual hit masks.
    """

    results = _as_mother_results(mother_particle_results)
    photon_1, photon_2 = sample_diphoton_lab_four_momenta(
        results,
        seed=seed,
        rng=rng,
    )

    vertices = results[:, [X_DECAY, Y_DECAY, Z_DECAY]]

    photon_1_x, photon_1_y, photon_1_reaches = project_particles_to_ecal(
        vertices,
        photon_1,
        geometry=geometry,
    )
    photon_2_x, photon_2_y, photon_2_reaches = project_particles_to_ecal(
        vertices,
        photon_2,
        geometry=geometry,
    )

    photon_1_hits = rectangular_ecal_hit_mask(
        photon_1_x,
        photon_1_y,
        photon_1_reaches,
        geometry=geometry,
    )
    photon_2_hits = rectangular_ecal_hit_mask(
        photon_2_x,
        photon_2_y,
        photon_2_reaches,
        geometry=geometry,
    )
    event_mask = photon_1_hits & photon_2_hits

    if not return_details:
        return event_mask

    return DiphotonECALResult(
        event_mask=event_mask,
        photon_1_hit_mask=photon_1_hits,
        photon_2_hit_mask=photon_2_hits,
        photon_1_four_momentum=photon_1,
        photon_2_four_momentum=photon_2,
        photon_1_x_ecal_m=photon_1_x,
        photon_1_y_ecal_m=photon_1_y,
        photon_2_x_ecal_m=photon_2_x,
        photon_2_y_ecal_m=photon_2_y,
    )


def filter_diphoton_ecal_events(
    mother_particle_results: ArrayLike,
    *,
    geometry: ECALGeometry = DEFAULT_ECAL,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_mask: bool = False,
) -> FloatArray | tuple[FloatArray, BoolArray]:
    """Filter an EventCalc mother table to events whose two photons hit ECAL."""

    results = _as_mother_results(mother_particle_results)
    mask = diphoton_ecal_acceptance(
        results,
        geometry=geometry,
        seed=seed,
        rng=rng,
        return_details=False,
    )
    accepted = results[mask]

    if return_mask:
        return accepted, mask
    return accepted


def weighted_ecal_acceptance(
    event_mask: ArrayLike,
    event_weights: ArrayLike,
) -> float:
    """
    Calculate the weighted fraction of events passing the ECAL requirement.

    For EventCalc mother samples, `event_weights` will usually be P_decay or an
    absolute event contribution proportional to P_decay.
    """

    mask = np.asarray(event_mask, dtype=bool)
    weights = np.asarray(event_weights, dtype=float)

    if mask.ndim != 1 or weights.ndim != 1:
        raise ValueError("event_mask and event_weights must be one-dimensional.")
    if len(mask) != len(weights):
        raise ValueError("event_mask and event_weights must have identical lengths.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("event_weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("event_weights must be non-negative.")

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("The total event weight must be positive.")

    return float(np.sum(weights[mask]) / total_weight)


__all__ = [
    "DEFAULT_ECAL",
    "DEFAULT_SEED",
    "DiphotonECALResult",
    "ECALGeometry",
    "P_DECAY",
    "diphoton_ecal_acceptance",
    "filter_diphoton_ecal_events",
    "project_particles_to_ecal",
    "rectangular_ecal_hit_mask",
    "sample_diphoton_lab_four_momenta",
    "weighted_ecal_acceptance",
]