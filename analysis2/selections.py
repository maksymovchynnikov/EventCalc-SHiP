"""Selection abstraction, context and configured strategy factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

from .ecal_selection import (
    DEFAULT_ECAL,
    DEFAULT_ECAL_SEED_OFFSET,
    DIPHOTON_ECAL_ALGORITHM,
    DIPHOTON_ECAL_ALGORITHM_VERSION,
    BoolArray,
    DiphotonECALResult,
    DiphotonECALEnergySelection,
    DiphotonECALSelection,
    ECALGeometry,
    project_particles_to_ecal,
    rectangular_ecal_hit_mask,
    sample_diphoton_lab_four_momenta,
)


class MotherSampleLike(Protocol):
    """The kinematic fields required by selections, in GeV and metres."""

    px_gev: np.ndarray
    py_gev: np.ndarray
    pz_gev: np.ndarray
    energy_gev: np.ndarray
    decay_probability: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    mass_gev: float

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class SelectionContext:
    """Seeds belonging to one source proposal and its mother realization."""

    source_seed: int
    true_sample_seed: int

    def __post_init__(self) -> None:
        for name in ("source_seed", "true_sample_seed"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


class Selection(Protocol):
    """Strategy applied after mother-particle generation."""

    name: str

    def cache_identity(self, context: SelectionContext) -> dict: ...

    def mask(
        self,
        sample: MotherSampleLike,
        context: SelectionContext | None = None,
    ) -> BoolArray: ...


@dataclass(frozen=True)
class MotherLevelSelection:
    """No daughter-level cut; retained for non-ECAL compatibility studies."""

    name: ClassVar[str] = "mother_level"

    def cache_identity(self, context: SelectionContext) -> dict:
        return {
            "name": self.name,
            "algorithm": "accept_all_mothers_inside_decay_volume",
            "algorithm_version": 1,
        }

    def mask(
        self,
        sample: MotherSampleLike,
        context: SelectionContext | None = None,
    ) -> BoolArray:
        return np.ones(len(sample), dtype=bool)


def selection_for_name(
    name: str,
    *,
    geometry: ECALGeometry = DEFAULT_ECAL,
    ecal_seed_offset: int = DEFAULT_ECAL_SEED_OFFSET,
) -> Selection:
    """Construct a supported selection without hiding its numerical settings."""

    if name == "mother_level":
        return MotherLevelSelection()
    if name == "diphoton_ecal":
        return DiphotonECALSelection(geometry=geometry, seed_offset=ecal_seed_offset)
    if name == "diphoton_ecal_e1gev":
        return DiphotonECALEnergySelection(
            geometry=geometry,
            seed_offset=ecal_seed_offset,
            minimum_photon_energy_gev=1.0,
        )
    raise ValueError(f"unknown selection {name!r}")


__all__ = [
    "DEFAULT_ECAL",
    "DEFAULT_ECAL_SEED_OFFSET",
    "DIPHOTON_ECAL_ALGORITHM",
    "DIPHOTON_ECAL_ALGORITHM_VERSION",
    "DiphotonECALResult",
    "DiphotonECALEnergySelection",
    "DiphotonECALSelection",
    "ECALGeometry",
    "MotherLevelSelection",
    "Selection",
    "SelectionContext",
    "project_particles_to_ecal",
    "rectangular_ecal_hit_mask",
    "sample_diphoton_lab_four_momenta",
    "selection_for_name",
]
