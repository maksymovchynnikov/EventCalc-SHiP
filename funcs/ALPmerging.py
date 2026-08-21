"""Build a photon--SU(2)_L ALP mixture from the installed public tables.

The mixed model uses the paper convention

    C_W = Lambda * xi,
    C_gamma_direct = sign * Lambda * (1 - xi),

where ``sign`` is constructive (+1) or destructive (-1).  The coupling
reported by EventCalc is the *total* diphoton coupling

    g_agammagamma_total = 4 Lambda [sign (1-xi) + sin(theta_W)^2 xi].

This module does not generate new source spectra.  It reconstructs the
flavour-changing B+K component from the installed inclusive SU(2)_L table and
the two installed photon-induced tables, then combines those components for
the requested xi and interference sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ALPHA_EM = 1.0 / 137.035999177
SIN2_THETA_W = 0.23122
ALPHA_SU2 = ALPHA_EM / SIN2_THETA_W
HBARC_GEV_M = 1.973269804e-16
INDUCED_PHOTON_RATIO = ALPHA_EM / np.pi
DISTRIBUTION_FLOOR = 1.0e-90
CANCELLATION_TOLERANCE = 1.0e-12


def normalize_interference(value: str) -> str:
    """Return the canonical interference label."""
    label = str(value).strip().casefold()
    aliases = {
        "+": "constructive",
        "plus": "constructive",
        "constructive": "constructive",
        "-": "destructive",
        "minus": "destructive",
        "destructive": "destructive",
    }
    try:
        return aliases[label]
    except KeyError as exc:
        raise ValueError("interference must be 'constructive' or 'destructive'") from exc


def amplitude_coefficient(xi: float, interference: str) -> float:
    """Return C_gamma_total / Lambda in the paper's xi convention."""
    xi = float(xi)
    if not 0.0 <= xi <= 1.0:
        raise ValueError("xi must lie in the closed interval [0, 1]")
    sign = 1.0 if normalize_interference(interference) == "constructive" else -1.0
    return sign * (1.0 - xi) + SIN2_THETA_W * xi


def flavor_to_photon_ratio(xi: float, interference: str) -> float:
    """Return g_W^2 / |g_agammagamma_total|^2 for a fixed xi and sign."""
    coefficient = amplitude_coefficient(xi, interference)
    if abs(coefficient) <= CANCELLATION_TOLERANCE:
        raise ValueError(
            "This destructive xi cancels the diphoton amplitude; the "
            "diphoton-only mixed model has no finite signal."
        )
    g_w_per_lambda = 4.0 * np.pi / ALPHA_SU2
    return float((g_w_per_lambda * float(xi)) ** 2 / (16.0 * coefficient**2))


def diphoton_ctau_coefficient(mass: float) -> float:
    """Return c*tau at |g_agammagamma_total| = 1 GeV^-1, in metres."""
    mass = float(mass)
    if mass <= 0.0:
        raise ValueError("ALP mass must be positive")
    return float(64.0 * np.pi * HBARC_GEV_M / mass**3)


def _load_numeric_table(path: Path, columns: int) -> np.ndarray:
    table = np.loadtxt(path, dtype=float)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.ndim != 2 or table.shape[1] != columns:
        raise ValueError(f"{path} must contain exactly {columns} columns")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path} contains a non-finite value")
    return table


def _log_linear_positive(table: np.ndarray, target: float | np.ndarray) -> np.ndarray:
    """Log-log interpolation, returning zero outside the tabulated support."""
    x = np.asarray(table[:, 0], dtype=float)
    y = np.asarray(table[:, 1], dtype=float)
    target_array = np.atleast_1d(np.asarray(target, dtype=float))
    if np.any(x <= 0.0) or np.any(y < 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("yield tables need increasing positive masses and non-negative yields")

    result = np.zeros_like(target_array)
    inside = (target_array >= x[0]) & (target_array <= x[-1])
    active = target_array[inside]
    if len(active):
        upper = np.searchsorted(x, active, side="right")
        upper = np.clip(upper, 1, len(x) - 1)
        lower = upper - 1
        fraction = (np.log(active) - np.log(x[lower])) / (
            np.log(x[upper]) - np.log(x[lower])
        )
        values = np.empty_like(active)
        positive = (y[lower] > 0.0) & (y[upper] > 0.0)
        values[positive] = np.exp(
            (1.0 - fraction[positive]) * np.log(y[lower[positive]])
            + fraction[positive] * np.log(y[upper[positive]])
        )
        fraction_linear = (active - x[lower]) / (x[upper] - x[lower])
        values[~positive] = (
            (1.0 - fraction_linear[~positive]) * y[lower[~positive]]
            + fraction_linear[~positive] * y[upper[~positive]]
        )
        result[inside] = values

    for index, node in enumerate(x):
        exact = np.isclose(target_array, node, rtol=1.0e-12, atol=1.0e-15)
        result[exact] = y[index]
    return result


def _snap_to_bounds(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[np.isclose(result, lower, rtol=1.0e-12, atol=1.0e-15)] = lower
    result[np.isclose(result, upper, rtol=1.0e-12, atol=1.0e-15)] = upper
    return result


@dataclass
class _PhotonSource:
    yield_table: np.ndarray
    mass: np.ndarray
    theta: np.ndarray
    energy: np.ndarray
    density_interpolator: RegularGridInterpolator
    emax_mass: np.ndarray
    emax_theta: np.ndarray
    emax_interpolator: RegularGridInterpolator

    @classmethod
    def load(cls, directory: Path, mode: str) -> "_PhotonSource":
        distribution = _load_numeric_table(
            directory / f"DoubleDistr-ALP-photon_{mode}.txt", 4
        )
        emax = _load_numeric_table(directory / f"Emax-ALP-photon_{mode}.txt", 3)
        yield_table = _load_numeric_table(
            directory / f"Total-yield-ALP-photon_{mode}.txt", 2
        )

        mass = np.unique(distribution[:, 0])
        theta = np.unique(distribution[:, 1])
        energy = np.unique(distribution[:, 2])
        if len(distribution) != len(mass) * len(theta) * len(energy):
            raise ValueError(f"{mode} photon distribution is not a rectangular grid")
        values = distribution[:, 3].reshape(len(mass), len(theta), len(energy))
        density_interpolator = RegularGridInterpolator(
            (np.log(mass), np.log(theta), np.log(energy)),
            np.log(np.maximum(values, DISTRIBUTION_FLOOR)),
            bounds_error=False,
            fill_value=-np.inf,
        )

        emax_mass = np.unique(emax[:, 0])
        emax_theta = np.unique(emax[:, 1])
        if len(emax) != len(emax_mass) * len(emax_theta):
            raise ValueError(f"{mode} photon Emax table is not a rectangular grid")
        emax_values = emax[:, 2].reshape(len(emax_mass), len(emax_theta))
        emax_interpolator = RegularGridInterpolator(
            (np.log(emax_mass), np.log(emax_theta)),
            np.log(emax_values),
            bounds_error=False,
            fill_value=-np.inf,
        )
        return cls(
            yield_table,
            mass,
            theta,
            energy,
            density_interpolator,
            emax_mass,
            emax_theta,
            emax_interpolator,
        )

    def yield_at(self, mass: float | np.ndarray) -> np.ndarray:
        return _log_linear_positive(self.yield_table, mass)

    def density_slice(
        self,
        mass: float,
        target_theta: np.ndarray,
        target_energy: np.ndarray,
    ) -> np.ndarray:
        """Evaluate this source on one rectangular theta-energy target grid."""
        mass_eval = _snap_to_bounds(
            np.asarray([mass]), self.mass[0], self.mass[-1]
        )[0]
        if mass_eval < self.mass[0] or mass_eval > self.mass[-1]:
            return np.zeros((len(target_theta), len(target_energy)))

        theta_eval = _snap_to_bounds(target_theta, self.theta[0], self.theta[-1])
        energy_eval = _snap_to_bounds(target_energy, self.energy[0], self.energy[-1])
        theta_mesh, energy_mesh = np.meshgrid(theta_eval, energy_eval, indexing="ij")
        points = np.column_stack(
            (
                np.full(theta_mesh.size, np.log(mass_eval)),
                np.log(theta_mesh.ravel()),
                np.log(energy_mesh.ravel()),
            )
        )
        density = np.exp(self.density_interpolator(points)).reshape(theta_mesh.shape)

        emax_mass = _snap_to_bounds(
            np.asarray([mass]), self.emax_mass[0], self.emax_mass[-1]
        )[0]
        if emax_mass < self.emax_mass[0] or emax_mass > self.emax_mass[-1]:
            return np.zeros_like(density)
        emax_theta = np.clip(target_theta, self.emax_theta[0], self.emax_theta[-1])
        emax_points = np.column_stack(
            (np.full(len(target_theta), np.log(emax_mass)), np.log(emax_theta))
        )
        emax = np.exp(self.emax_interpolator(emax_points))

        supported_theta = (
            (target_theta >= self.theta[0] - 1.0e-15)
            & (target_theta <= self.theta[-1] + 1.0e-15)
        )
        supported_energy = (
            (target_energy >= self.energy[0] - 1.0e-15)
            & (target_energy <= self.energy[-1] + 1.0e-15)
        )
        support = (
            supported_theta[:, None]
            & supported_energy[None, :]
            & (target_energy[None, :] >= mass)
            & (target_energy[None, :] <= emax[:, None])
        )
        density[~support] = 0.0
        return density


@dataclass(frozen=True)
class MixedTables:
    distribution: pd.DataFrame
    emax: pd.DataFrame
    yield_table: pd.DataFrame
    ctau_table: pd.DataFrame
    mass_min: float
    mass_max: float
    source_ratio: float


def build_mixed_tables(
    distributions_root: str | Path,
    xi: float,
    interference: str,
) -> MixedTables:
    """Construct the normalized mixed production table from installed inputs."""
    root = Path(distributions_root)
    su2_directory = root / "ALP-SU2L"
    photon_directory = root / "ALP-photon"
    source_ratio = flavor_to_photon_ratio(xi, interference)

    distribution = pd.read_csv(
        su2_directory / "DoubleDistr-ALP-SU2L.txt", header=None, sep="\t"
    )
    emax = pd.read_csv(
        su2_directory / "Emax-ALP-SU2L.txt", header=None, sep="\t"
    )
    su2_yield = _load_numeric_table(
        su2_directory / "Total-yield-ALP-SU2L.txt", 2
    )
    sources = tuple(
        _PhotonSource.load(photon_directory, mode)
        for mode in ("primary", "cascades")
    )

    masses = np.unique(distribution.iloc[:, 0].to_numpy())
    theta = np.unique(distribution.iloc[:, 1].to_numpy())
    energy = np.unique(distribution.iloc[:, 2].to_numpy())
    block_size = len(theta) * len(energy)
    if len(distribution) != len(masses) * block_size:
        raise ValueError("inclusive SU(2)_L distribution is not a rectangular grid")
    mass_blocks = distribution.iloc[:, 0].to_numpy().reshape(len(masses), block_size)
    if not np.all(mass_blocks == masses[:, None]):
        raise ValueError("inclusive SU(2)_L distribution has unexpected row ordering")

    output_density = distribution.iloc[:, 3].to_numpy(copy=True)
    k_squared = INDUCED_PHOTON_RATIO**2
    for mass_index, mass in enumerate(masses):
        start = mass_index * block_size
        stop = start + block_size
        su2_density = output_density[start:stop].copy()
        su2_coefficient = _log_linear_positive(su2_yield, mass)[0]

        photon_numerator = np.zeros((len(theta), len(energy)))
        photon_coefficient = 0.0
        for source in sources:
            source_yield = source.yield_at(mass)[0]
            photon_coefficient += source_yield
            photon_numerator += source_yield * source.density_slice(mass, theta, energy)

        flavor_numerator = (
            su2_coefficient * su2_density.reshape(len(theta), len(energy))
            - k_squared * photon_numerator
        )
        numerical_scale = max(
            float(np.max(np.abs(su2_coefficient * su2_density))),
            float(np.max(np.abs(k_squared * photon_numerator))),
            np.finfo(float).tiny,
        )
        if float(np.min(flavor_numerator)) < -1.0e-9 * numerical_scale:
            raise ValueError(
                f"installed source tables give a negative B+K density at m={mass:g} GeV"
            )
        flavor_numerator = np.maximum(flavor_numerator, 0.0)
        flavor_coefficient = su2_coefficient - k_squared * photon_coefficient
        if flavor_coefficient < -1.0e-12 * max(su2_coefficient, np.finfo(float).tiny):
            raise ValueError(
                f"installed source tables give a negative B+K yield at m={mass:g} GeV"
            )
        flavor_coefficient = max(float(flavor_coefficient), 0.0)
        total_coefficient = photon_coefficient + source_ratio * flavor_coefficient
        if total_coefficient <= 0.0:
            mixed_density = np.full_like(flavor_numerator, DISTRIBUTION_FLOOR)
        else:
            mixed_density = (
                photon_numerator + source_ratio * flavor_numerator
            ) / total_coefficient
            mixed_density = np.maximum(mixed_density, DISTRIBUTION_FLOOR)
        output_density[start:stop] = mixed_density.ravel()

    distribution.iloc[:, 3] = output_density

    yield_masses = su2_yield[:, 0]
    photon_yield = sum(source.yield_at(yield_masses) for source in sources)
    flavor_yield = su2_yield[:, 1] - k_squared * photon_yield
    tolerance = 1.0e-12 * np.maximum(su2_yield[:, 1], np.finfo(float).tiny)
    if np.any(flavor_yield < -tolerance):
        bad_mass = yield_masses[np.flatnonzero(flavor_yield < -tolerance)[0]]
        raise ValueError(
            f"installed source tables give a negative B+K yield at m={bad_mass:g} GeV"
        )
    flavor_yield = np.maximum(flavor_yield, 0.0)
    mixed_yield = photon_yield + source_ratio * flavor_yield
    yield_table = pd.DataFrame({0: yield_masses, 1: mixed_yield})

    # Both operators are supported without extrapolation on this common range.
    mass_min = float(
        format(max(float(masses[0]), *(float(source.mass[0]) for source in sources)), ".12g")
    )
    mass_max = float(
        format(min(float(masses[-1]), *(float(source.mass[-1]) for source in sources)), ".12g")
    )
    ctau_table = pd.DataFrame(
        {0: masses, 1: [diphoton_ctau_coefficient(mass) for mass in masses]}
    )
    return MixedTables(
        distribution=distribution,
        emax=emax,
        yield_table=yield_table,
        ctau_table=ctau_table,
        mass_min=mass_min,
        mass_max=mass_max,
        source_ratio=source_ratio,
    )


__all__ = [
    "CANCELLATION_TOLERANCE",
    "SIN2_THETA_W",
    "MixedTables",
    "amplitude_coefficient",
    "build_mixed_tables",
    "diphoton_ctau_coefficient",
    "flavor_to_photon_ratio",
    "normalize_interference",
]
