"""Deterministic seed resolution for frozen and appended analysis masses."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analysis2.paths import OUTPUT_ROOT


DEFAULT_WEEK8_DOMAIN_PATH = (
    OUTPUT_ROOT
    / "production"
    / "week8_domains"
    / "allowed_ctau_domains.csv"
)


def _unique_sorted(values: Iterable[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Available masses must be finite and positive.")
        if not any(
            np.isclose(value, old, rtol=0.0, atol=1.0e-12)
            for old in result
        ):
            result.append(value)
    return tuple(result)


def available_masses_from_domain(domain_path: Path) -> tuple[float, ...]:
    path = Path(domain_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "The allowed lifetime-domain table is required to resolve appended-mass "
            f"seeds: {path}"
        )
    frame = pd.read_csv(path)
    if "mass_GeV" not in frame.columns:
        raise ValueError(
            f"Allowed lifetime-domain table lacks the mass_GeV column: {path}"
        )
    masses = _unique_sorted(frame["mass_GeV"].to_numpy(dtype=float))
    if not masses:
        raise ValueError(f"Allowed lifetime-domain table contains no masses: {path}")
    return masses


def stable_mass_seed_indices(
    seed_policy,
    available_masses: Iterable[float],
) -> dict[float, int]:
    frozen = tuple(float(value) for value in seed_policy.mass_order_gev)
    available = _unique_sorted(available_masses)
    extras = tuple(
        mass
        for mass in available
        if not any(
            np.isclose(mass, old, rtol=0.0, atol=1.0e-12)
            for old in frozen
        )
    )
    result = {mass: index for index, mass in enumerate(frozen)}
    result.update(
        {
            mass: len(frozen) + index
            for index, mass in enumerate(extras)
        }
    )
    return result


def mass_seed_index(
    *,
    seed_policy,
    mass_gev: float,
    available_masses: Iterable[float],
) -> int:
    mapping = stable_mass_seed_indices(seed_policy, available_masses)
    matches = [
        index
        for mass, index in mapping.items()
        if np.isclose(
            float(mass),
            float(mass_gev),
            rtol=0.0,
            atol=1.0e-12,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Mass {mass_gev:g} GeV is not uniquely represented in the "
            "frozen-plus-appended seed ordering."
        )
    return int(matches[0])


def model_seed_for_bank(
    *,
    config,
    bank,
    model_id: str,
    domain_path: Path = DEFAULT_WEEK8_DOMAIN_PATH,
) -> int:
    expected_base = (
        int(config.seed_policy.base_seed)
        + int(bank.template_seed_offset)
    )
    if int(bank.template_base_seed) != expected_base:
        raise ValueError(
            "Template-bank base seed disagrees with the active profile: "
            f"bank={bank.template_base_seed}, expected={expected_base}."
        )

    index = mass_seed_index(
        seed_policy=config.seed_policy,
        mass_gev=float(bank.mass_gev),
        available_masses=available_masses_from_domain(domain_path),
    )
    return config.seed_policy.model_seed_from_indices(
        index,
        config.seed_policy.model_index(model_id),
        seed_offset=int(bank.template_seed_offset),
    )


__all__ = [
    "DEFAULT_WEEK8_DOMAIN_PATH",
    "available_masses_from_domain",
    "mass_seed_index",
    "model_seed_for_bank",
    "stable_mass_seed_indices",
]
