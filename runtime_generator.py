#!/usr/bin/env python3
"""Python runtime interface over the installed EventCalc physics tables.

This module keeps EventCalc's tabulated production and decay inputs while
providing an amortized ``init()``/``next()`` interface, direct inverse-CDF
sampling, fiducial and attempted event modes, explicit decay weights, and
running yield estimates.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, replace
import io
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator, Literal, Mapping, Sequence

import numba as nb
import numpy as np

from funcs.interpolation_functions import (
    _bilinear_interpolation,
    _trilinear_interpolation,
)
from funcs.simulation_config import (
    PROJECT_ROOT,
    SimulationConfig,
    config_from_mapping,
    load_card,
)
from simulate import _load_runtime, _make_llp, _mass_is_tabulated


RuntimeMode = Literal["fiducial", "attempted"]


@nb.njit
def _seed_numba_random(seed: int) -> None:
    """Seed the legacy random streams used inside EventCalc's Numba decays."""
    np.random.seed(seed)
    random.seed(seed)


@contextmanager
def _legacy_random_seed(seed: int, *, seed_numba: bool = False) -> Iterator[None]:
    """Use a deterministic legacy RNG state without changing the caller's state."""
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    if seed_numba:
        _seed_numba_random(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class ExperimentCard:
    """Geometry needed by the truth-level runtime sampler."""

    name: str
    z_min_m: float
    z_max_m: float
    x_width_in_m: float
    x_width_out_m: float
    y_width_in_m: float
    y_width_out_m: float
    survival_cutoff: float = 15.0

    @classmethod
    def ship(cls) -> "ExperimentCard":
        return cls(
            name="SHiP",
            z_min_m=32.0,
            z_max_m=82.0,
            x_width_in_m=1.0,
            x_width_out_m=4.0,
            y_width_in_m=2.7,
            y_width_out_m=6.2,
            survival_cutoff=15.0,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentCard":
        required = (
            "z_min_m",
            "z_max_m",
            "x_width_in_m",
            "x_width_out_m",
            "y_width_in_m",
            "y_width_out_m",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(
                "experiment card is missing: " + ", ".join(sorted(missing))
            )
        card = cls(
            name=str(values.get("name", "experiment")),
            z_min_m=_positive_finite(values["z_min_m"], "z_min_m"),
            z_max_m=_positive_finite(values["z_max_m"], "z_max_m"),
            x_width_in_m=_positive_finite(
                values["x_width_in_m"], "x_width_in_m"
            ),
            x_width_out_m=_positive_finite(
                values["x_width_out_m"], "x_width_out_m"
            ),
            y_width_in_m=_positive_finite(
                values["y_width_in_m"], "y_width_in_m"
            ),
            y_width_out_m=_positive_finite(
                values["y_width_out_m"], "y_width_out_m"
            ),
            survival_cutoff=_positive_finite(
                values.get("survival_cutoff", 15.0), "survival_cutoff"
            ),
        )
        if card.z_max_m <= card.z_min_m:
            raise ValueError("z_max_m must be greater than z_min_m")
        return card

    @classmethod
    def load(cls, path: Path) -> "ExperimentCard":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("an experiment card must contain one JSON object")
        return cls.from_mapping(values)

    @property
    def theta_max_rad(self) -> float:
        inner = math.hypot(self.x_width_in_m / 2, self.y_width_in_m / 2)
        outer = math.hypot(self.x_width_out_m / 2, self.y_width_out_m / 2)
        return math.atan(max(inner / self.z_min_m, outer / self.z_max_m))

    def _half_width(
        self, z: np.ndarray, width_in: float, width_out: float
    ) -> np.ndarray:
        fraction = (z - self.z_min_m) / (self.z_max_m - self.z_min_m)
        return 0.5 * (width_in + fraction * (width_out - width_in))

    def contains(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        x_max = self._half_width(z, self.x_width_in_m, self.x_width_out_m)
        y_max = self._half_width(z, self.y_width_in_m, self.y_width_out_m)
        return (
            (self.z_min_m <= z)
            & (z <= self.z_max_m)
            & (-x_max < x)
            & (x < x_max)
            & (-y_max < y)
            & (y < y_max)
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventRecord:
    """One generated event returned by :meth:`RuntimeEventGenerator.next_event`."""

    index: int
    mother: np.ndarray
    daughters: np.ndarray
    channel: str
    inside_volume: bool
    decay_weight: float
    status: str = "ok"

    @property
    def flat_record(self) -> np.ndarray:
        return np.concatenate((self.mother, self.daughters.reshape(-1)))


@dataclass(frozen=True)
class EventBatch:
    """A bounded-memory batch in the standard EventCalc row layout."""

    index: int
    start: int
    stop: int
    records: np.ndarray
    channels: np.ndarray
    inside_volume: np.ndarray
    decay_weights: np.ndarray

    def __len__(self) -> int:
        return self.stop - self.start

    def event(self, position: int) -> EventRecord:
        if position < 0 or position >= len(self):
            raise IndexError(position)
        row = self.records[position]
        daughters = row[10:].reshape(-1, 6)
        daughters = daughters[daughters[:, 5] != -999].copy()
        return EventRecord(
            index=self.start + position,
            mother=row[:10].copy(),
            daughters=daughters,
            channel=str(self.channels[position]),
            inside_volume=bool(self.inside_volume[position]),
            decay_weight=float(self.decay_weights[position]),
            status=(
                "ok" if self.inside_volume[position] else "outside_fiducial"
            ),
        )


@dataclass(frozen=True)
class YieldEstimate:
    """Exact inputs and running Monte Carlo estimates for the physical yield."""

    coupling_squared: float
    produced_llps: float
    visible_branching_ratio: float
    epsilon_polar: float | None
    epsilon_polar_error: float | None
    epsilon_transverse: float | None
    epsilon_transverse_error: float | None
    mean_decay_probability: float | None
    mean_decay_probability_error: float | None
    expected_events: float | None
    expected_events_error: float | None
    attempted_events: int
    fiducial_events: int
    cdf_cells: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ParentSample:
    mother: np.ndarray
    momentum: np.ndarray
    inside_volume: np.ndarray


def _mean_and_error(
    count: int, total: float, total_squared: float
) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    mean = total / count
    if count == 1:
        return mean, None
    variance = max((total_squared - count * mean * mean) / (count - 1), 0.0)
    return mean, math.sqrt(variance / count)


def _sample_linear_density(
    left: np.ndarray, right: np.ndarray, unit: np.ndarray
) -> np.ndarray:
    """Sample x in [0, 1] from a non-negative linear density."""
    slope = right - left
    integral = 0.5 * (left + right)
    target = unit * integral
    scale = np.maximum(
        np.maximum(np.abs(left), np.abs(right)), np.finfo(float).tiny
    )
    flat = np.abs(slope) <= 1.0e-14 * scale
    result = np.empty_like(unit)
    result[flat] = unit[flat]

    curved = ~flat
    discriminant = np.maximum(
        left[curved] ** 2 + 2.0 * slope[curved] * target[curved], 0.0
    )
    denominator = left[curved] + np.sqrt(discriminant)
    result[curved] = np.divide(
        2.0 * target[curved],
        denominator,
        out=unit[curved].copy(),
        where=denominator > 0.0,
    )
    return np.clip(result, 0.0, 1.0)


class _ProductionInverseCDF:
    """Direct sampler for the tabulated density at one mass and lifetime.

    Energy is represented by u in [0, 1] between the lifetime-dependent lower
    sampling bound and E_max(theta).  The tabulated density, including the
    dE/du Jacobian, is evaluated once on a (theta, u) grid.  Sampling then uses
    a flattened cell CDF followed by analytic inversion of the bilinear density
    inside the selected cell.
    """

    def __init__(
        self,
        grid: Any,
        *,
        energy_cells: int = 512,
        theta_subdivisions: int = 1,
    ) -> None:
        if energy_cells <= 0 or theta_subdivisions <= 0:
            raise ValueError("CDF grid sizes must be positive")
        self.grid = grid
        self.energy_cells = int(energy_cells)

        mass_tolerance = 1.0e-12 * max(abs(grid.m), 1.0)
        if (
            grid.m < grid.grid_m[0] - mass_tolerance
            or grid.m > grid.grid_m[-1] + mass_tolerance
        ):
            raise ValueError("Emax table does not cover the requested mass")
        theta_tolerance = 1.0e-6
        if (
            grid.thetamin < grid.grid_a[0] - theta_tolerance
            or grid.theta_max > grid.grid_a[-1] + theta_tolerance
        ):
            raise ValueError(
                "Emax table does not cover the requested angular interval"
            )

        interior = grid.grid_y[
            (grid.grid_y > grid.thetamin) & (grid.grid_y < grid.theta_max)
        ]
        base_theta = np.unique(
            np.concatenate(([grid.thetamin], interior, [grid.theta_max]))
        )
        if len(base_theta) < 2:
            raise ValueError("production table has no angular interval to sample")
        if theta_subdivisions == 1:
            self.theta_nodes = base_theta
        else:
            pieces = [
                np.linspace(left, right, theta_subdivisions + 1)[:-1]
                for left, right in zip(base_theta[:-1], base_theta[1:])
            ]
            self.theta_nodes = np.concatenate((*pieces, base_theta[-1:]))
        linear_u = np.linspace(0.0, 1.0, self.energy_cells + 1)
        self.u_nodes = np.unique(np.concatenate((linear_u, linear_u**2)))

        minimum, maximum = self._energy_bounds(self.theta_nodes)
        width = maximum - minimum
        energy = minimum[:, None] + width[:, None] * self.u_nodes[None, :]
        theta = np.broadcast_to(self.theta_nodes[:, None], energy.shape)
        points = np.column_stack(
            (
                np.full(energy.size, grid.m),
                theta.reshape(-1),
                energy.reshape(-1),
            )
        )
        values = _trilinear_interpolation(
            points,
            grid.grid_x,
            grid.grid_y,
            grid.grid_z,
            grid.distr,
            np.repeat(maximum, len(self.u_nodes)),
        ).reshape(energy.shape)
        density = np.maximum(values, 0.0) * width[:, None]

        dtheta = np.diff(self.theta_nodes)
        du = np.diff(self.u_nodes)
        cell_mass = (
            0.25
            * (
                density[:-1, :-1]
                + density[1:, :-1]
                + density[:-1, 1:]
                + density[1:, 1:]
            )
            * dtheta[:, None]
            * du[None, :]
        )
        cell_mass = np.maximum(cell_mass, 0.0)
        flat_mass = cell_mass.reshape(-1)
        total = float(flat_mass.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("inverse-CDF production density has zero integral")

        self.density = density
        self.cell_shape = cell_mass.shape
        self.cdf = np.cumsum(flat_mass)
        self.integral = total
        self.cell_count = int(np.count_nonzero(flat_mass))

    def _energy_bounds(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid
        points = np.column_stack((np.full(len(theta), grid.m), theta))
        maximum = _bilinear_interpolation(
            points, grid.grid_m, grid.grid_a, grid.energy_distr
        )
        minimum = np.maximum(
            max(grid.m, grid.energy_min_tab),
            np.minimum(
                grid.survival_energy_floor * grid.m / grid.c_tau,
                0.5 * maximum,
            ),
        )
        minimum = np.minimum(minimum, maximum)
        return minimum, maximum

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        unit = rng.random((count, 3))
        targets = unit[:, 0] * self.integral
        flat_index = np.searchsorted(self.cdf, targets, side="right")
        flat_index = np.minimum(flat_index, len(self.cdf) - 1)
        theta_index, energy_index = np.unravel_index(
            flat_index, self.cell_shape
        )

        g00 = self.density[theta_index, energy_index]
        g10 = self.density[theta_index + 1, energy_index]
        g01 = self.density[theta_index, energy_index + 1]
        g11 = self.density[theta_index + 1, energy_index + 1]

        theta_fraction = _sample_linear_density(
            0.5 * (g00 + g01),
            0.5 * (g10 + g11),
            unit[:, 1],
        )
        lower_density = g00 + theta_fraction * (g10 - g00)
        upper_density = g01 + theta_fraction * (g11 - g01)
        energy_fraction = _sample_linear_density(
            lower_density, upper_density, unit[:, 2]
        )

        theta = self.theta_nodes[theta_index] + theta_fraction * (
            self.theta_nodes[theta_index + 1] - self.theta_nodes[theta_index]
        )
        unit_energy = self.u_nodes[energy_index] + energy_fraction * (
            self.u_nodes[energy_index + 1] - self.u_nodes[energy_index]
        )
        minimum, maximum = self._energy_bounds(theta)
        energy = minimum + unit_energy * (maximum - minimum)
        return theta, energy


class RuntimeModelScan:
    """Shared model state for many mass--lifetime runtime generators.

    Loading and merging the model tables and filling EventCalc's dense
    interpolation grids are scan-level work.  The lifetime-dependent CDF is
    cached separately for every requested point.
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        experiment: ExperimentCard | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.experiment = experiment or ExperimentCard.ship()
        self.verbose = bool(verbose)
        self.runtime: Any = None
        self._llp_template: Any = None
        self._mass_templates: dict[float, Any] = {}
        self._grid_templates: dict[tuple[int, int], Any] = {}
        self._cdf_cache: dict[tuple[float, float, int, int], _ProductionInverseCDF] = {}
        self._initialized = False

    def init(self) -> bool:
        if self._initialized:
            return True
        self.runtime = _load_runtime(headless=True)
        self._llp_template = _make_llp(self.runtime, self.config)
        self._initialized = True
        return True

    def _mass_template(self, mass: float) -> Any:
        self.init()
        mass = float(mass)
        cached = self._mass_templates.get(mass)
        if cached is not None:
            return cached
        if not _mass_is_tabulated(mass, self._llp_template):
            raise ValueError(
                f"mass {mass:g} GeV lies outside the installed range "
                f"[{self._llp_template.m_min_tabulated:g}, "
                f"{self._llp_template.m_max_tabulated:g}] GeV"
            )
        llp = copy.copy(self._llp_template)
        llp.set_mass(mass)
        llp.compute_mass_dependent_properties()
        self._mass_templates[mass] = llp
        return llp

    def _compiled_point(
        self, mass: float, c_tau: float
    ) -> tuple[Any, Any, _ProductionInverseCDF]:
        mass_template = self._mass_template(mass)
        llp = copy.copy(mass_template)
        llp.set_c_tau(float(c_tau))

        grid_key = (id(llp.Distr), id(llp.Energy_distr))
        grid_template = self._grid_templates.get(grid_key)
        if grid_template is None:
            grid_template = self.runtime.kinematics.Grids(
                llp.Distr,
                llp.Energy_distr,
                1,
                llp.mass,
                llp.c_tau_input,
                theta_max_sim=self.experiment.theta_max_rad,
                survival_energy_floor=(
                    self.experiment.z_min_m / self.experiment.survival_cutoff
                ),
            )
            self._grid_templates[grid_key] = grid_template

        grid = copy.copy(grid_template)
        grid.m = float(mass)
        grid.c_tau = float(c_tau)
        grid.theta_max = min(
            float(grid.Distr[1].max()), self.experiment.theta_max_rad
        )
        grid.survival_energy_floor = (
            self.experiment.z_min_m / self.experiment.survival_cutoff
        )

        cdf_key = (float(mass), float(c_tau), *grid_key)
        cdf = self._cdf_cache.get(cdf_key)
        if cdf is None:
            cdf = _ProductionInverseCDF(grid)
            self._cdf_cache[cdf_key] = cdf
        return llp, grid, cdf

    def compile_points(self, points: Sequence[tuple[float, float]]) -> None:
        """Compile and cache all ``(mass, c_tau)`` points in a scan."""
        for mass, c_tau in points:
            self._compiled_point(mass, c_tau)

    def generator(
        self,
        *,
        mass: float,
        c_tau: float,
        mode: RuntimeMode = "fiducial",
        seed: int | None = None,
        prefetch: int = 256,
        verbose: bool | None = None,
    ) -> "RuntimeEventGenerator":
        config = self.config if seed is None else replace(self.config, seed=seed)
        return RuntimeEventGenerator(
            config,
            mass=mass,
            c_tau=c_tau,
            experiment=self.experiment,
            mode=mode,
            prefetch=prefetch,
            verbose=self.verbose if verbose is None else verbose,
            scan=self,
        )

    def cache_info(self) -> dict[str, int]:
        return {
            "masses": len(self._mass_templates),
            "dense_grids": len(self._grid_templates),
            "lifetime_cdfs": len(self._cdf_cache),
        }


class RuntimeEventGenerator:
    """Amortized Python runtime interface backed by EventCalc tables."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        mass: float,
        c_tau: float,
        experiment: ExperimentCard | None = None,
        mode: RuntimeMode = "fiducial",
        prefetch: int = 256,
        verbose: bool = False,
        scan: RuntimeModelScan | None = None,
    ) -> None:
        if mass <= 0.0 or c_tau <= 0.0:
            raise ValueError("mass and c_tau must be positive")
        if prefetch <= 0:
            raise ValueError("prefetch must be positive")
        if mode not in {"fiducial", "attempted"}:
            raise ValueError("mode must be 'fiducial' or 'attempted'")

        self.config = config
        self.mass = float(mass)
        self.c_tau = float(c_tau)
        self.experiment = experiment or ExperimentCard.ship()
        self.mode: RuntimeMode = mode
        self.prefetch = int(prefetch)
        self.verbose = bool(verbose)
        self._scan = scan
        self.resolved_seed = int(
            config.seed
            if config.seed is not None
            else np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0]
        )

        self.runtime: Any = None
        self.llp: Any = None
        self._grid: Any = None
        self._production_cdf: _ProductionInverseCDF | None = None
        self._parent_rng: np.random.Generator | None = None
        self._vertex_rng: np.random.Generator | None = None
        self._initialized = False
        self._decay_block = 0
        self._batch_index = 0
        self._generated_events = 0
        self._delivered_events = 0
        self._next_batch: EventBatch | None = None
        self._next_position = 0
        self.current_event: EventRecord | None = None

        self._attempted_count = 0
        self._fiducial_count = 0
        self._accepted_decay_sum = 0.0
        self._accepted_decay_squared_sum = 0.0
        self._attempt_decay_weight_sum = 0.0
        self._attempt_decay_weight_squared_sum = 0.0
        self._coupling_squared = 0.0
        self._produced_llps = 0.0
        self.visible_branching_ratio = 0.0
        self.selected_decay_indices: list[int] = []

    def init(self) -> bool:
        """Load the selected model and precompute mass/lifetime state once."""
        if self._initialized:
            return True

        if self._scan is not None:
            self._scan.init()
            self.runtime = self._scan.runtime
            self.llp, self._grid, self._production_cdf = (
                self._scan._compiled_point(self.mass, self.c_tau)
            )
        else:
            self.runtime = _load_runtime(headless=True)
            self.llp = _make_llp(self.runtime, self.config)
            if not _mass_is_tabulated(self.mass, self.llp):
                raise ValueError(
                    f"mass {self.mass:g} GeV lies outside the installed range "
                    f"[{self.llp.m_min_tabulated:g}, "
                    f"{self.llp.m_max_tabulated:g}] GeV"
                )

            self.llp.set_mass(self.mass)
            self.llp.compute_mass_dependent_properties()
            self.llp.set_c_tau(self.c_tau)
        self.selected_decay_indices = self.config.selected_decay_indices(
            tuple(str(item) for item in self.llp.decayChannels)
        )
        self.visible_branching_ratio = float(
            sum(
                self.llp.BrRatios_distr[index]
                for index in self.selected_decay_indices
            )
        )
        if self.visible_branching_ratio <= 0.0:
            raise ValueError("the selected decay channels have zero branching ratio")

        self._coupling_squared = float(
            self.llp.c_tau_int / self.c_tau
            if self.llp.LLP_name != "Scalar-quartic"
            else 0.01
        )
        self._produced_llps = float(
            self.config.n_pot * self.llp.Yield * self._coupling_squared
        )
        if self._production_cdf is None:
            self._grid = self.runtime.kinematics.Grids(
                self.llp.Distr,
                self.llp.Energy_distr,
                1,
                self.llp.mass,
                self.llp.c_tau_input,
                theta_max_sim=self.experiment.theta_max_rad,
                survival_energy_floor=(
                    self.experiment.z_min_m / self.experiment.survival_cutoff
                ),
            )
            self._production_cdf = _ProductionInverseCDF(self._grid)
        parent_seed, vertex_seed = np.random.SeedSequence(
            self.resolved_seed
        ).spawn(2)
        self._parent_rng = np.random.Generator(np.random.Philox(parent_seed))
        self._vertex_rng = np.random.Generator(np.random.Philox(vertex_seed))
        self._initialized = True
        return True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.init()

    def _block_seeds(self, kind: int, block: int, count: int = 2) -> list[int]:
        sequence = np.random.SeedSequence([self.resolved_seed, kind, block])
        return [int(item) for item in sequence.generate_state(count, dtype=np.uint32)]

    def _sample_attempts(self, count: int) -> _ParentSample:
        self._ensure_initialized()
        if (
            self._production_cdf is None
            or self._parent_rng is None
            or self._vertex_rng is None
        ):
            raise RuntimeError("runtime random streams were not initialized")
        theta, energy = self._production_cdf.sample(count, self._parent_rng)
        rng = self._vertex_rng
        vertex_unit = rng.random((count, 2))
        phi = -math.pi + 2.0 * math.pi * vertex_unit[:, 0]
        momentum_abs = np.sqrt(
            np.maximum(energy * energy - self.mass * self.mass, 0.0)
        )
        cos_theta = np.cos(theta)
        denominator = self.c_tau * momentum_abs * cos_theta
        rate = np.divide(
            self.mass,
            denominator,
            out=np.full_like(denominator, np.inf),
            where=denominator > 0.0,
        )

        delta_z = self.experiment.z_max_m - self.experiment.z_min_m
        interval_probability = -np.expm1(-rate * delta_z)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.experiment.z_min_m - np.log1p(
                -vertex_unit[:, 1] * interval_probability
            ) / rate
        z = np.where(np.isfinite(z), z, self.experiment.z_min_m)

        sin_theta = np.sin(theta)
        px = momentum_abs * np.cos(phi) * sin_theta
        py = momentum_abs * np.sin(phi) * sin_theta
        pz = momentum_abs * cos_theta
        x = z * np.cos(phi) * np.tan(theta)
        y = z * np.sin(phi) * np.tan(theta)
        inside = self.experiment.contains(x, y, z)
        decay_probability = (
            np.exp(-rate * self.experiment.z_min_m) * interval_probability
        )

        mother = np.column_stack(
            (
                px,
                py,
                pz,
                energy,
                np.full(count, self.mass),
                np.full(count, 12345678.0),
                decay_probability,
                x,
                y,
                z,
            )
        )
        momentum = np.column_stack((px, py, pz, energy))

        decay_weight = np.where(inside, decay_probability, 0.0)
        accepted_probability = decay_probability[inside]
        self._attempted_count += count
        self._fiducial_count += int(inside.sum())
        self._attempt_decay_weight_sum += float(decay_weight.sum())
        self._attempt_decay_weight_squared_sum += float(
            np.square(decay_weight).sum()
        )
        self._accepted_decay_sum += float(accepted_probability.sum())
        self._accepted_decay_squared_sum += float(
            np.square(accepted_probability).sum()
        )
        return _ParentSample(mother, momentum, inside)

    def _parents_for_output(self, count: int) -> _ParentSample:
        if self.mode == "attempted":
            return self._sample_attempts(count)

        mother_chunks: list[np.ndarray] = []
        momentum_chunks: list[np.ndarray] = []
        accepted = 0
        empty_attempts = 0
        while accepted < count:
            sample = self._sample_attempts(count - accepted)
            mask = sample.inside_volume
            if not np.any(mask):
                empty_attempts += 1
                if empty_attempts >= 20:
                    raise RuntimeError(
                        "no parent trajectories entered the fiducial volume in 20 attempts"
                    )
                continue
            empty_attempts = 0
            mother_chunks.append(sample.mother[mask])
            momentum_chunks.append(sample.momentum[mask])
            accepted += int(mask.sum())

        return _ParentSample(
            mother=np.concatenate(mother_chunks),
            momentum=np.concatenate(momentum_chunks),
            inside_volume=np.ones(count, dtype=bool),
        )

    def _decay(
        self, mother: np.ndarray, momentum: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        decay_seed = self._block_seeds(2, self._decay_block, count=1)[0]
        self._decay_block += 1
        output_context = nullcontext() if self.verbose else redirect_stdout(io.StringIO())
        with _legacy_random_seed(decay_seed, seed_numba=True), output_context:
            rest_frame, sizes = self.runtime.decayProducts.simulateDecays_rest_frame(
                self.llp.mass,
                self.llp.PDGs,
                self.llp.BrRatios_distr,
                len(mother),
                self.llp.Matrix_elements,
                list(self.selected_decay_indices),
                self.visible_branching_ratio,
            )
        rest_frame = np.asarray(rest_frame, dtype=float)
        boosted = self.runtime.boost.tab_boosted_decay_products(
            self.llp.mass, momentum, rest_frame
        )
        channel_names = np.asarray(
            [
                str(self.llp.decayChannels[index])
                for index, size in zip(self.selected_decay_indices, sizes, strict=True)
                for _ in range(int(size))
            ],
            dtype=str,
        )
        if len(mother) != len(channel_names):
            raise RuntimeError("decay-channel labels do not match generated events")
        return boosted, channel_names

    def _make_batch(self, count: int) -> EventBatch:
        if count <= 0:
            raise ValueError("batch event count must be positive")
        sample = self._parents_for_output(count)
        boosted, channels = self._decay(sample.mother, sample.momentum)
        records = np.concatenate((sample.mother, boosted), axis=1)
        decay_weights = np.where(
            sample.inside_volume, sample.mother[:, 6], 0.0
        )
        start = self._generated_events
        stop = start + count
        batch = EventBatch(
            index=self._batch_index,
            start=start,
            stop=stop,
            records=records,
            channels=channels,
            inside_volume=sample.inside_volume.copy(),
            decay_weights=decay_weights,
        )
        self._generated_events = stop
        self._batch_index += 1
        return batch

    def next_event(self) -> EventRecord:
        """Return one event, refilling an internal in-memory buffer as needed."""
        if self._next_batch is None or self._next_position >= len(self._next_batch):
            self._next_batch = self._make_batch(self.prefetch)
            self._next_position = 0
        event = self._next_batch.event(self._next_position)
        self._next_position += 1
        self._delivered_events += 1
        self.current_event = event
        return event

    def next(self) -> bool:
        """Pythia-shaped interface: fill ``current_event`` and return success."""
        self.next_event()
        return True

    def generate(self, events: int, *, batch_size: int = 10_000) -> Iterator[EventBatch]:
        """Yield a finite sample in bounded-memory batches."""
        if events <= 0 or batch_size <= 0:
            raise ValueError("events and batch_size must be positive integers")
        if self._next_batch is not None and self._next_position < len(self._next_batch):
            raise RuntimeError("finish the pending next() buffer before calling generate()")
        remaining = events
        while remaining:
            batch = self._make_batch(min(batch_size, remaining))
            self._delivered_events += len(batch)
            remaining -= len(batch)
            yield batch

    def yields(self) -> YieldEstimate:
        """Return exact normalization inputs and current Monte Carlo estimates."""
        self._ensure_initialized()
        if self._production_cdf is None:
            raise RuntimeError("production CDF was not initialized")
        epsilon_polar = self._production_cdf.integral
        epsilon_polar_error = None

        if self._attempted_count:
            epsilon_transverse = self._fiducial_count / self._attempted_count
            epsilon_transverse_error = math.sqrt(
                epsilon_transverse
                * (1.0 - epsilon_transverse)
                / self._attempted_count
            )
        else:
            epsilon_transverse = None
            epsilon_transverse_error = None

        mean_decay, mean_decay_error = _mean_and_error(
            self._fiducial_count,
            self._accepted_decay_sum,
            self._accepted_decay_squared_sum,
        )
        attempt_weight_mean, attempt_weight_error = _mean_and_error(
            self._attempted_count,
            self._attempt_decay_weight_sum,
            self._attempt_decay_weight_squared_sum,
        )

        normalization = self._produced_llps * self.visible_branching_ratio
        expected_events = None
        expected_events_error = None
        if epsilon_polar is not None and attempt_weight_mean is not None:
            expected_events = normalization * epsilon_polar * attempt_weight_mean
            if attempt_weight_error is not None:
                expected_events_error = (
                    normalization * epsilon_polar * attempt_weight_error
                )

        return YieldEstimate(
            coupling_squared=self._coupling_squared,
            produced_llps=self._produced_llps,
            visible_branching_ratio=self.visible_branching_ratio,
            epsilon_polar=epsilon_polar,
            epsilon_polar_error=epsilon_polar_error,
            epsilon_transverse=epsilon_transverse,
            epsilon_transverse_error=epsilon_transverse_error,
            mean_decay_probability=mean_decay,
            mean_decay_probability_error=mean_decay_error,
            expected_events=expected_events,
            expected_events_error=expected_events_error,
            attempted_events=self._attempted_count,
            fiducial_events=self._fiducial_count,
            cdf_cells=self._production_cdf.cell_count,
        )

    def stat(self) -> dict[str, Any]:
        """Return generation counters together with :meth:`yields`."""
        return {
            "mode": self.mode,
            "resolved_seed": self.resolved_seed,
            "generated_events": self._generated_events,
            "delivered_events": self._delivered_events,
            "yield": self.yields().as_dict(),
        }


def generator_from_card(
    card: Path,
    *,
    mass: float,
    c_tau: float,
    experiment_card: Path | None = None,
    mode: RuntimeMode = "fiducial",
    seed: int | None = None,
    prefetch: int = 256,
    verbose: bool = False,
) -> RuntimeEventGenerator:
    """Construct a runtime generator from existing model and experiment cards."""
    values: Mapping[str, object] = load_card(card)
    resolved = dict(values)
    resolved.update(
        {
            "masses": [mass],
            "c_taus": [c_tau],
            "plots": False,
            "export_events": False,
            "min_events_threshold": 0.0,
        }
    )
    if seed is not None:
        resolved["seed"] = seed
    config = config_from_mapping(resolved, project_root=PROJECT_ROOT)
    experiment = (
        ExperimentCard.load(experiment_card)
        if experiment_card is not None
        else ExperimentCard.ship()
    )
    return RuntimeEventGenerator(
        config,
        mass=mass,
        c_tau=c_tau,
        experiment=experiment,
        mode=mode,
        prefetch=prefetch,
        verbose=verbose,
    )


def scan_from_card(
    card: Path,
    *,
    experiment_card: Path | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> RuntimeModelScan:
    """Load one model context that can serve every point in a scan."""
    resolved = dict(load_card(card))
    resolved.update(
        {
            "plots": False,
            "export_events": False,
            "min_events_threshold": 0.0,
        }
    )
    if seed is not None:
        resolved["seed"] = seed
    config = config_from_mapping(resolved, project_root=PROJECT_ROOT)
    experiment = (
        ExperimentCard.load(experiment_card)
        if experiment_card is not None
        else ExperimentCard.ship()
    )
    return RuntimeModelScan(
        config, experiment=experiment, verbose=verbose
    )


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--experiment-card", type=Path)
    parser.add_argument("--mass", type=float, required=True, help="LLP mass in GeV")
    parser.add_argument("--ctau", type=float, required=True, help="proper decay length in m")
    parser.add_argument("--events", type=_positive_int, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=10_000)
    parser.add_argument("--prefetch", type=_positive_int, default=256)
    parser.add_argument("--mode", choices=("fiducial", "attempted"), default="fiducial")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output-format",
        choices=("npz", "hepmc3", "both"),
        default="npz",
        help="write compressed NumPy batches, one HepMC3 ASCII stream, or both",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generator = generator_from_card(
        args.card,
        mass=args.mass,
        c_tau=args.ctau,
        experiment_card=args.experiment_card,
        mode=args.mode,
        seed=args.seed,
        prefetch=args.prefetch,
        verbose=args.verbose,
    )
    generator.init()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    write_npz = args.output_format in {"npz", "both"}
    write_hepmc = args.output_format in {"hepmc3", "both"}
    files: list[str] = []
    if write_hepmc:
        from hepmc_export import HepMC3Writer

        hepmc_filename = "events.hepmc3"
        hepmc_context: Any = HepMC3Writer(output_dir / hepmc_filename)
    else:
        hepmc_filename = None
        hepmc_context = nullcontext(None)

    with hepmc_context as hepmc_writer:
        for batch in generator.generate(args.events, batch_size=args.batch_size):
            if write_npz:
                filename = f"events_{batch.start:012d}_{batch.stop:012d}.npz"
                np.savez_compressed(
                    output_dir / filename,
                    records=batch.records,
                    channels=batch.channels,
                    inside_volume=batch.inside_volume,
                    decay_weights=batch.decay_weights,
                    start=np.asarray(batch.start),
                    stop=np.asarray(batch.stop),
                )
                files.append(filename)
            if hepmc_writer is not None:
                hepmc_writer.write_batch(batch)
            print(f"generated {batch.stop}/{args.events} events")
    if hepmc_filename is not None:
        files.append(hepmc_filename)

    metadata = {
        "format": "EventCalc Python runtime batches v3",
        "model": generator.config.model,
        "mass_GeV": generator.mass,
        "c_tau_m": generator.c_tau,
        "mode": generator.mode,
        "events": args.events,
        "batch_size": args.batch_size,
        "output_format": args.output_format,
        "seed": generator.resolved_seed,
        "experiment": generator.experiment.as_dict(),
        "selected_decay_channels": [
            str(generator.llp.decayChannels[index])
            for index in generator.selected_decay_indices
        ],
        "record_layout": {
            "mother": ["px", "py", "pz", "E", "mass", "PDG", "P_decay", "x", "y", "z"],
            "each_final_state_particle": ["px", "py", "pz", "E", "mass", "PDG"],
            "padding_PDG": -999,
        },
        "decay_weight": "P_decay for fiducial events and zero otherwise",
        "production_sampler": "precomputed inverse CDF in (theta, u_energy)",
        "production_cdf": {
            "theta_nodes": len(generator._production_cdf.theta_nodes),
            "unit_energy_nodes": len(generator._production_cdf.u_nodes),
            "nonzero_cells": generator._production_cdf.cell_count,
        },
        "random_streams": {
            "parent": "NumPy Philox, persistent stream",
            "vertex": "NumPy Philox, persistent stream",
            "decay": "seeded per prefetched/batched decay block",
        },
        "yield": generator.yields().as_dict(),
        "files": files,
        "limitations": (
            "Python runtime over EventCalc tables; no binary physics cards or "
            "event-index random access, and no native FairShip adapter"
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
