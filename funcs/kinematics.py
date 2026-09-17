# funcs/kinematics.py
"""
Sampling, interpolation and re-weighting of LLP lab-frame kinematics.

`Grids` draws production angles and energies of a long-lived particle from its
tabulated double-differential yield, places the decay vertex inside the decay
volume according to the exponential decay law, and reports the polar
acceptance of the drawn sample.

The angular interval that is sampled is `[theta_min_tab, theta_max_sim]`, where
`theta_min_tab` is the smallest tabulated production angle and `theta_max_sim`
defaults to the angular half-opening of the decay volume.  A script that wants
the full forward hemisphere passes `theta_max_sim=math.pi / 2`, which requires
production tables reaching that far.
"""

from collections import OrderedDict

import numpy as np
import time
import pandas as pd
import numba as nb

from .interpolation_functions import (
    _searchsorted_opt,
    _bilinear_interpolation,
    _trilinear_interpolation,
    _fill_distr_2D,
    _fill_distr_3D,
)
from .ship_setup import (
    z_min,
    z_max,
    x_max,
    y_max,
    theta_max_dec_vol,   # default angular upper limit
)

_GRID_CACHE_MAXSIZE = 8
_GRID_CACHE = OrderedDict()


class ProductionMixture:
    """A set of independently tabulated production sources to be added.

    Each component is ``(label, distribution, maximum_energy, yield_function)``.
    The component distributions remain on their native grids.  ``Grids`` uses
    the absolute source yields to construct the correct conditional mixture
    after polar acceptance, avoiding a lossy interpolation onto one giant
    common grid.
    """

    def __init__(self, components):
        normalized = []
        labels = set()
        for component in components:
            if len(component) != 4:
                raise ValueError(
                    "production-mixture components must have four fields")
            label, distribution, maximum_energy, yield_function = component
            label = str(label)
            if not label or label in labels:
                raise ValueError(
                    "production-mixture labels must be non-empty and unique")
            if not callable(yield_function):
                raise TypeError(
                    "production-mixture yield functions must be callable")
            labels.add(label)
            normalized.append(
                (label, distribution, maximum_energy, yield_function))
        if len(normalized) < 2:
            raise ValueError(
                "a production mixture requires at least two components")
        self.components = tuple(normalized)

    def component_yields(self, mass):
        values = []
        for label, _distribution, _maximum_energy, function in self.components:
            value = float(function(float(mass)))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"production source {label!r} has invalid yield {value} "
                    f"at mass {float(mass):.9g} GeV")
            values.append(value)
        if not any(value > 0.0 for value in values):
            raise ValueError(
                f"all production-source yields vanish at "
                f"mass {float(mass):.9g} GeV")
        return np.asarray(values, dtype=float)

    def total_yield(self, mass):
        return float(np.sum(self.component_yields(mass)))


class SourceGrids:
    """The dense interpolation grids of one pair of tabulated source tables.

    Nothing here depends on the mass or on the lifetime.  The axis nodes, the
    filled production density and the filled maximum energy are functions of
    the two tables alone, and filling them is the expensive part of preparing
    a scan point, so it belongs to the source and not to the point.  One
    instance serves every mass and every lifetime a scan visits on the same
    tables.

    Parameters
    ----------
    Distr, Energy_distr : pandas.DataFrame
        Original tabulated 3-D and 2-D distributions.
    """

    def __init__(self, Distr, Energy_distr):
        # ----- build 2-D grid for max-energy interpolation -------------
        self.grid_m = np.unique(Energy_distr.iloc[:, 0])
        self.grid_a = np.unique(Energy_distr.iloc[:, 1])
        self.energy_distr = _fill_distr_2D(
            self.grid_m, self.grid_a, Energy_distr
        )

        # ----- build 3-D grid for full distribution -------------------
        self.grid_x = np.unique(Distr.iloc[:, 0])
        self.grid_y = np.unique(Distr.iloc[:, 1])
        self.grid_z = np.unique(Distr.iloc[:, 2])
        self.distr = _fill_distr_3D(
            self.grid_x, self.grid_y, self.grid_z, Distr
        )

        # the limits the tables themselves impose: the angular interval the
        # production table covers, and the energy support outside which the
        # distribution is undefined
        self.theta_min = Distr[1].min()
        self.theta_max = Distr[1].max()
        self.energy_min = self.grid_z[0]
        self.energy_max = self.grid_z[-1]


def source_grids(Distr, Energy_distr, cache_token=None):
    """Return the grids of one source pair, built once for a named source.

    A caller that holds the returned `SourceGrids` and passes it to `Grids`
    in place of the tables pays for the fill once and needs nothing else.

    `cache_token` serves a caller that keeps only the tables: it is that
    caller's own identity and revision for the pair, and equal tokens read
    the same grids.  There is no implicit reuse, because a pandas object's
    identity and shape do not identify its contents and an in-place,
    same-shape edit of a table would otherwise return stale physics.  The
    registry is a small LRU, so walking many revisions does not retain their
    grids indefinitely.
    """
    if cache_token is None:
        return SourceGrids(Distr, Energy_distr)

    try:
        hash(cache_token)
    except TypeError as exc:
        raise TypeError("kinematics cache_token must be hashable") from exc

    grids = _GRID_CACHE.get(cache_token)
    if grids is None:
        grids = SourceGrids(Distr, Energy_distr)
        _GRID_CACHE[cache_token] = grids
    _GRID_CACHE.move_to_end(cache_token)
    while len(_GRID_CACHE) > _GRID_CACHE_MAXSIZE:
        _GRID_CACHE.popitem(last=False)
    return grids


class Grids:
    """
    Handle interpolation of tabulated (θ, E) distributions and resampling
    with geometry / weighting corrections.

    Parameters
    ----------
    Distr, Energy_distr : pandas.DataFrame
        Original tabulated 3-D and 2-D distributions.  A `SourceGrids` built
        from that pair may be passed in place of `Distr`, with `Energy_distr`
        left as None, and its filled grids are then used as they are.
    nPoints : int
        Raw Monte-Carlo points for the internal interpolation grid.
    mass : float
        Mass of the LLP in GeV.
    c_tau : float
        Proper decay length (m).
    theta_max_sim : float, optional
        Largest polar angle (rad) at which production is sampled.  It defaults
        to the angular half-opening of the decay volume.  A wider value, up to
        `np.pi / 2` for the full forward hemisphere, is honoured as far as the
        production and maximum-energy tables reach.
    survival_energy_floor : float, optional
        Coefficient of ``mass / c_tau`` in the sampling-energy cutoff.  Its
        default reproduces the SHiP value; runtime experiment cards may set it
        from their own upstream distance and exponential cutoff.
    cache_token : hashable, optional
        Explicit source identity and revision for a caller that passes the
        tables rather than a `SourceGrids`.  Equal tokens read the same
        filled grids; a caller must change the token whenever either input
        table changes.  If omitted, the grids are built for this point and
        nothing is kept.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        Distr,
        Energy_distr,
        nPoints,
        mass,
        c_tau,
        theta_max_sim=theta_max_dec_vol,
        *,
        survival_energy_floor=2.133,
        cache_token=None,
    ):
        self.Distr = Distr
        self.Energy_distr = Energy_distr
        self.nPoints = nPoints
        self.m = mass
        self.c_tau = c_tau
        self.survival_energy_floor = survival_energy_floor
        self.source_grids = None
        self.production_mixture = (
            Distr if isinstance(Distr, ProductionMixture) else None)

        if self.production_mixture is not None:
            if Energy_distr is not None:
                raise ValueError(
                    "a ProductionMixture supplies its own maximum-energy "
                    "tables; Energy_distr must be None")
            component_yields = self.production_mixture.component_yields(mass)
            active = []
            for component_index, (
                    label, distribution, maximum_energy, _yield_function
            ) in enumerate(self.production_mixture.components):
                component_yield = float(component_yields[component_index])
                if component_yield == 0.0:
                    continue
                component_token = (
                    None if cache_token is None
                    else (cache_token, "production-component",
                          component_index, label))
                active.append((
                    label,
                    component_yield,
                    Grids(
                        distribution,
                        maximum_energy,
                        nPoints,
                        mass,
                        c_tau,
                        theta_max_sim=theta_max_sim,
                        survival_energy_floor=survival_energy_floor,
                        cache_token=component_token,
                    ),
                ))
            self._mixture_components = tuple(active)
            self._mixture_total_yield = float(np.sum(component_yields))
            self.thetamin = min(
                grid.thetamin for _label, _yield, grid in active)
            self.theta_max = max(
                grid.theta_max for _label, _yield, grid in active)
            return

        # ----- the grids of the source tables --------------------------
        if isinstance(Distr, SourceGrids):
            if Energy_distr is not None:
                raise ValueError(
                    "grids built from a source carry their own "
                    "maximum-energy table; Energy_distr must be None")
            self.source_grids = Distr
        else:
            self.source_grids = source_grids(
                Distr, Energy_distr, cache_token=cache_token)

        # ----- θ range -------------------------------------------------
        self.thetamin = self.source_grids.theta_min
        # honour either table limit or user limit, whichever is smaller
        self.theta_max = min(self.source_grids.theta_max, theta_max_sim)

        self._require_emax_covers_theta()

    # The filled grids and the tabulated limits belong to the source tables,
    # not to this mass and lifetime.  They are read from the source rather
    # than copied onto the point, so that nothing here can fall out of step
    # with the tables it was filled from.
    @property
    def grid_m(self):
        return self.source_grids.grid_m

    @property
    def grid_a(self):
        return self.source_grids.grid_a

    @property
    def energy_distr(self):
        return self.source_grids.energy_distr

    @property
    def grid_x(self):
        return self.source_grids.grid_x

    @property
    def grid_y(self):
        return self.source_grids.grid_y

    @property
    def grid_z(self):
        return self.source_grids.grid_z

    @property
    def distr(self):
        return self.source_grids.distr

    @property
    def energy_min_tab(self):
        """Lowest tabulated energy; below it the distribution is undefined."""
        return self.source_grids.energy_min

    @property
    def energy_max_tab(self):
        """Highest tabulated energy; above it the distribution is undefined."""
        return self.source_grids.energy_max

    @property
    def energy_floor(self):
        """Lowest energy an LLP of this mass can carry and still be tabulated.

        A particle cannot have less energy than its rest mass, and below the
        first tabulated energy node the production density is undefined.  The
        value follows the mass and the table, so a caller that sets a new mass
        on an existing grid reads the floor that belongs to it.
        """
        return max(self.m, self.energy_min_tab)

    def _require_emax_covers_theta(self):
        """Refuse a sampling interval that leaves the maximum-energy table.

        Outside its nodes the bilinear lookup continues the outermost cell in a
        straight line, and that continuation carries no information about the
        beam dump: it can fall below the true kinematic ceiling and cut away
        real production, or rise above it and open an interval the particle can
        never occupy.  The two table families are written on angular grids that
        agree to a fraction of their first node, so the comparison carries a
        tolerance of that size.
        """
        theta_tolerance = 1.0e-6
        if (
            self.thetamin < self.grid_a[0] - theta_tolerance
            or self.theta_max > self.grid_a[-1] + theta_tolerance
        ):
            raise ValueError(
                f"The maximum-energy table spans polar angles "
                f"[{self.grid_a[0]}, {self.grid_a[-1]}] rad while sampling was "
                f"requested over [{self.thetamin}, {self.theta_max}] rad. "
                f"E_max(m, theta) is undefined outside the tabulated interval.")

    # ==================================================================
    # Interpolation of θ, E and weight grid
    # ==================================================================
    def interpolate(self, timing=False):
        if timing:
            t0 = time.time()

        if self.production_mixture is not None:
            for _label, _yield, component in self._mixture_components:
                component.interpolate(False)
            if timing:
                print(
                    f"\nMixture interpolation time: "
                    f"{time.time() - t0:.2f} s")
            return

        # the angular interval may have been widened since construction
        self._require_emax_covers_theta()
        energy_floor = self.energy_floor

        # random θ in [θ_min, θ_max_sim]
        self.theta = np.random.uniform(self.thetamin, self.theta_max, self.nPoints)
        self.mass = self.m * np.ones(self.nPoints)

        # max energy E_max(m, θ) by bilinear interpolation
        points_2d = np.column_stack((self.mass, self.theta))
        self.max_energy = _bilinear_interpolation(
            points_2d, self.grid_m, self.grid_a, self.energy_distr
        )

        # Lowest sampled energy.  It sits at `energy_floor` or above, so that
        # the drawn energy is both a physical state of the particle and inside
        # the tabulated support; above that floor it is raised further to skip
        # the energies whose survival to the decay volume is exponentially
        # suppressed.
        self.e_min_sampling = np.maximum(
            energy_floor,
            np.minimum(
                self.survival_energy_floor * self.m / self.c_tau,
                0.5 * self.max_energy,
            ),
        )

        # An angle whose tabulated maximum energy does not reach that floor is
        # closed: the most energetic particle of this mass produced there still
        # falls below the lowest energy the production table describes.  A
        # closed angle gets a sampling interval of zero width, which gives it
        # exactly zero weight in `resample`, so no event is drawn from it.
        self.production_open = self.max_energy > energy_floor
        self.e_min_sampling = np.where(
            self.production_open, self.e_min_sampling, self.max_energy
        )

        # draw energies uniformly in [E_min, E_max]
        self.energy = np.random.uniform(self.e_min_sampling, self.max_energy)
        points_3d = np.column_stack((self.mass, self.theta, self.energy))

        # distribution weight f(m, θ, E)
        self.interpolated_values = _trilinear_interpolation(
            points_3d,
            self.grid_x,
            self.grid_y,
            self.grid_z,
            self.distr,
            self.max_energy,
        )

        if timing:
            print(f"\nInterpolation time: {time.time() - t0:.2f} s")

    # ==================================================================
    # Resampling with acceptance weights
    # ==================================================================
    def resample(self, rsample_size, timing=False):
        if timing:
            t0 = time.time()

        self.rsample_size = rsample_size
        if self.production_mixture is not None:
            weighted_acceptances = []
            for _label, source_yield, component in self._mixture_components:
                component.resample(rsample_size, False)
                weighted_acceptances.append(
                    source_yield * component.epsilon_polar)
            weighted_acceptances = np.asarray(
                weighted_acceptances, dtype=float)
            accepted_weight = float(np.sum(weighted_acceptances))
            if not np.isfinite(accepted_weight) or accepted_weight <= 0.0:
                raise ValueError(
                    "combined production sources have zero polar acceptance")

            probabilities = weighted_acceptances / accepted_weight
            source_counts = np.random.multinomial(
                int(rsample_size), probabilities)
            theta_parts = []
            energy_parts = []
            component_summary = {}
            for (label, source_yield, component), count in zip(
                    self._mixture_components, source_counts):
                count = int(count)
                component_summary[label] = {
                    "yield": source_yield,
                    "epsilon_polar": component.epsilon_polar,
                    "sample_points": count,
                }
                if count:
                    theta_parts.append(component.r_theta[:count])
                    energy_parts.append(component.r_energy[:count])

            self.r_theta = np.concatenate(theta_parts)
            self.r_energy = np.concatenate(energy_parts)
            permutation = np.random.permutation(len(self.r_theta))
            self.r_theta = self.r_theta[permutation]
            self.r_energy = self.r_energy[permutation]
            self.true_points_indices = np.arange(len(self.r_theta))
            self.epsilon_polar = (
                accepted_weight / self._mixture_total_yield)
            self.production_component_summary = component_summary
            if timing:
                print(
                    f"Mixture resample time: "
                    f"{time.time() - t0:.2f} s")
            return

        weights = self.interpolated_values * (self.max_energy - self.e_min_sampling)
        self.weights = weights

        # The three ways a weight vector can be unusable are separated here,
        # because the message numpy raises on a bad probability vector names
        # neither the scan point nor the reason.  The two reductions are single
        # passes over the weights and allocate nothing; the counts that appear
        # in the messages are computed only on the failing path.
        total_weight = float(weights.sum())
        smallest_weight = float(weights.min()) if weights.size else 0.0
        if not (np.isfinite(total_weight) and np.isfinite(smallest_weight)):
            raise ValueError(
                f"Non-finite sampling weights for m={self.m} GeV, "
                f"c_tau={self.c_tau} m "
                f"({np.count_nonzero(~np.isfinite(weights))} of {len(weights)} points).")
        if smallest_weight < 0:
            raise ValueError(
                f"Negative sampling weights for m={self.m} GeV, "
                f"c_tau={self.c_tau} m: {np.count_nonzero(weights < 0)} of "
                f"{len(weights)} points, most negative {smallest_weight:.3e}. "
                "Energies were sampled outside the tabulated support "
                f"[{self.energy_min_tab}, {self.energy_max_tab}] GeV.")
        if total_weight <= 0:
            if not np.any(self.production_open):
                raise ValueError(
                    f"No LLP of mass {self.m} GeV is produced at any sampled "
                    f"angle: over theta=[{self.thetamin}, {self.theta_max}] rad "
                    f"the tabulated maximum energy reaches only "
                    f"{self.max_energy.max()} GeV, below the lowest energy "
                    f"{self.energy_floor} GeV this mass can carry inside the "
                    f"tabulated support.")
            raise ValueError(
                f"All sampling weights vanish for m={self.m} GeV, "
                f"c_tau={self.c_tau} m: the tabulated distribution has no support "
                f"in theta=[{self.thetamin}, {self.theta_max}] rad.")

        prob = weights / total_weight
        self.true_points_indices = np.random.choice(
            self.nPoints, size=self.rsample_size, p=prob
        )

        self.r_theta = self.theta[self.true_points_indices]
        self.r_energy = self.energy[self.true_points_indices]

        # polar acceptance: the mean weight times the width of the sampled
        # angular interval, i.e. the Monte-Carlo estimate of the fraction of
        # produced LLPs emitted into that interval
        self.epsilon_polar = (
            total_weight * (self.theta_max - self.thetamin) / len(weights)
        )

        if timing:
            print(f"Resample time: {time.time() - t0:.2f} s")

    # ==================================================================
    # Generate true decay vertices and momenta with SHiP cuts
    # ==================================================================
    def true_samples(self, timing=False):
        if timing:
            t0 = time.time()

        self.phi = np.random.uniform(-np.pi, np.pi, len(self.true_points_indices))

        # An energy below the rest mass belongs to no physical particle: its
        # momentum is imaginary.  Closed angles carry zero weight and cannot be
        # drawn, so a tabulated model never reaches this branch; it keeps a
        # four-vector with a negative invariant mass squared out of the event
        # file should a table ever place one there.
        energy_excess = self.r_energy**2 - self.m**2
        if energy_excess.size and energy_excess.min() < 0.0:
            raise ValueError(
                f"{int(np.count_nonzero(energy_excess < 0.0))} of "
                f"{len(energy_excess)} sampled LLPs of mass {self.m} GeV carry "
                f"an energy below their rest mass, the smallest being "
                f"{self.r_energy.min()} GeV.")
        momentum_abs = np.sqrt(energy_excess)

        px = momentum_abs * np.cos(self.phi) * np.sin(self.r_theta)
        py = momentum_abs * np.sin(self.phi) * np.sin(self.r_theta)
        pz = momentum_abs * np.cos(self.r_theta)

        # Longitudinal decay position.  Along its straight trajectory the LLP
        # decays with the exponential law, whose length projected on the beam
        # axis is cos(theta) * c_tau * p / m.  The vertex is drawn from that law
        # conditioned on the decay falling between the two faces of the decay
        # volume: with u uniform on [0, 1),
        #     z = z_min - lambda * ln(1 - u * [1 - exp(-(z_max - z_min)/lambda)]).
        # Written through log1p and expm1 this stays exact for every lifetime,
        # from lambda far above the volume length, where the vertex is uniform
        # in z, down to lambda far below it, where the vertices pile up within
        # one decay length of the entrance face.
        decay_length_z = (
            np.cos(self.r_theta) * self.c_tau * momentum_abs / self.m
        )
        decay_inside = -np.expm1(-(z_max - z_min) / decay_length_z)
        z = z_min - decay_length_z * np.log1p(
            -np.random.uniform(0.0, 1.0, len(decay_length_z)) * decay_inside
        )

        # transverse decay coordinates
        x = z * np.cos(self.phi) * np.tan(self.r_theta)
        y = z * np.sin(self.phi) * np.tan(self.r_theta)

        # decay-inside-volume flag
        geom_acceptance = (
            (-x_max(z) < x)
            & (x < x_max(z))
            & (-y_max(z) < y)
            & (y < y_max(z))
            & (z_min <= z)
            & (z <= z_max)
        )

        # Probability that the LLP survives to the entrance face and then decays
        # before the exit face.  The factored form keeps the small difference of
        # two survival probabilities accurate when both are close to one.
        P_decay = np.exp(-z_min / decay_length_z) * decay_inside

        self.kinematics_dic = {
            "px": px[geom_acceptance],
            "py": py[geom_acceptance],
            "pz": pz[geom_acceptance],
            "energy": self.r_energy[geom_acceptance],
            "m": self.m * np.ones_like(px[geom_acceptance]),
            "PDG": 12345678 * np.ones_like(px[geom_acceptance]),
            "P_decay": P_decay[geom_acceptance],
            "x": x[geom_acceptance],
            "y": y[geom_acceptance],
            "z": z[geom_acceptance],
        }

        self.momentum = np.column_stack(
            (
                px[geom_acceptance],
                py[geom_acceptance],
                pz[geom_acceptance],
                self.r_energy[geom_acceptance],
            )
        )

        if timing:
            print(f"Vertex sampling time: {time.time() - t0:.2f} s")

    # ==================================================================
    # Convenience getters
    # ==================================================================
    def get_kinematics(self):
        return np.column_stack(list(self.kinematics_dic.values()))

    def save_kinematics(self, path, name):
        pd.DataFrame(self.kinematics_dic).to_csv(
            f"{path}/{name}_kinematics_sampling.dat", sep="\t", index=False
        )

    def get_energy(self):
        return self.r_energy

    def get_theta(self):
        return self.r_theta

    def get_momentum(self):
        return self.momentum
