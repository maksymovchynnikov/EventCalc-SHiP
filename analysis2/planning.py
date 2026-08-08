"""Deterministic planning utilities for the unified ALP-SU2L analysis.

This module contains no EventCalc calls and no pseudoexperiments.  It converts
a reproducible analysis configuration plus the local bank registry into an
explicit per-mass/per-selection execution plan.

Expensive workflow modules consume this plan later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis2.workflows import float_token


SELECTIONS = (
    "diphoton_ecal",
    "diphoton_ecal_e1gev",
)

OBSERVABLES = (
    "energy",
    "energy_mean_z",
    "energy_mean_r_perp",
    "energy_mean_z_r_perp",
)

PROFILES = (
    "quick",
    "validation",
    "production",
)

RUN_MODES = (
    "automatic",
    "custom",
    "reuse_only",
)


class BankState(str, Enum):
    """Scientific readiness of a lifetime-template bank."""

    VALIDATED = "validated"
    PRODUCTION = "production"
    PRODUCTION_NOISE_FLOOR = "production_noise_floor_limited"
    INCOMPLETE = "incomplete"
    SMOKE = "smoke"
    MISSING = "missing"


_STATUS_ALIASES = {
    "validated": BankState.VALIDATED,
    "production": BankState.PRODUCTION,
    "production_bank": BankState.PRODUCTION,
    "production_noise_floor_limited": (
        BankState.PRODUCTION_NOISE_FLOOR
    ),
    "production_bank_noise_floor_limited": (
        BankState.PRODUCTION_NOISE_FLOOR
    ),
    "incomplete": BankState.INCOMPLETE,
    "incomplete_bank": BankState.INCOMPLETE,
    "smoke": BankState.SMOKE,
    "smoke_bank": BankState.SMOKE,
    "missing": BankState.MISSING,
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Single reproducible user-level configuration."""

    masses: tuple[float, ...]
    selections: tuple[str, ...]
    observables: tuple[str, ...]
    profile: str
    workers: int
    run_mode: str
    output_dir: Path
    domain_path: Path
    bank_manifest: Path
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.masses:
            raise ValueError("At least one mass is required.")
        if any(
            not np.isfinite(mass) or mass <= 0.0
            for mass in self.masses
        ):
            raise ValueError(
                "All requested masses must be finite and positive."
            )

        unknown_selections = sorted(
            set(self.selections) - set(SELECTIONS)
        )
        if unknown_selections:
            raise ValueError(
                f"Unknown selections: {unknown_selections}"
            )

        unknown_observables = sorted(
            set(self.observables) - set(OBSERVABLES)
        )
        if unknown_observables:
            raise ValueError(
                f"Unknown observables: {unknown_observables}"
            )

        if self.profile not in PROFILES:
            raise ValueError(
                f"profile must be one of {PROFILES}."
            )

        if self.workers not in (1, 2):
            raise ValueError("workers must be 1 or 2.")

        if self.run_mode not in RUN_MODES:
            raise ValueError(
                f"run_mode must be one of {RUN_MODES}."
            )

    def as_dict(self) -> dict:
        return {
            "masses": list(self.masses),
            "selections": list(self.selections),
            "observables": list(self.observables),
            "profile": self.profile,
            "workers": self.workers,
            "run_mode": self.run_mode,
            "output_dir": str(self.output_dir),
            "domain_path": str(self.domain_path),
            "bank_manifest": str(self.bank_manifest),
            "resume": self.resume,
        }


def selection_token(selection_name: str) -> str:
    if selection_name == "diphoton_ecal":
        return "geom"
    if selection_name == "diphoton_ecal_e1gev":
        return "e1gev"
    raise ValueError(f"Unknown selection: {selection_name}")


def normalise_bank_state(value: str) -> BankState:
    token = str(value).strip().lower()
    try:
        return _STATUS_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            f"Unknown bank status in registry: {value!r}"
        ) from exc


def resolve_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def bank_workspace_dir(
    output_dir: Path,
    mass_gev: float,
    selection_name: str,
) -> Path:
    """Return a provenance-isolated workspace for one physics point."""
    return (
        Path(output_dir)
        / "bank_workspaces"
        / "per_mass"
        / f"ma_{float_token(mass_gev)}"
        / selection_token(selection_name)
    )


def result_dir(
    output_dir: Path,
    mass_gev: float,
    selection_name: str,
) -> Path:
    return (
        Path(output_dir)
        / "per_mass"
        / f"ma_{float_token(mass_gev)}"
        / selection_token(selection_name)
        / "conditional_features"
    )


def _manifest_record(
    manifest: pd.DataFrame,
    mass_gev: float,
    selection_name: str,
) -> dict | None:
    mask = (
        np.isclose(
            manifest["mass_GeV"].to_numpy(dtype=float),
            float(mass_gev),
            rtol=0.0,
            atol=1.0e-12,
        )
        & (
            manifest["selection_name"].astype(str)
            == selection_name
        )
    )

    matches = manifest.loc[mask]

    if len(matches) == 0:
        return None
    if len(matches) != 1:
        raise ValueError(
            "Bank registry contains multiple entries for "
            f"m_a={mass_gev:g} GeV, selection={selection_name}."
        )

    return matches.iloc[0].to_dict()


def build_analysis_plan(
    *,
    config: AnalysisConfig,
    manifest: pd.DataFrame,
    repo: Path,
) -> pd.DataFrame:
    """Build the deterministic per-point execution plan."""

    required = {
        "mass_GeV",
        "selection_name",
        "status",
        "bank_path",
    }
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            "Bank registry is missing columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    safe_reuse_states = {
        BankState.VALIDATED,
        BankState.PRODUCTION,
        BankState.PRODUCTION_NOISE_FLOOR,
    }

    for mass_gev in config.masses:
        for selection_name in config.selections:
            record = _manifest_record(
                manifest,
                mass_gev,
                selection_name,
            )

            bank_path: Path | None = None

            if record is None:
                state = BankState.MISSING
            else:
                state = normalise_bank_state(record["status"])

                raw_path = record.get("bank_path")
                if (
                    isinstance(raw_path, str)
                    and raw_path.strip()
                ):
                    bank_path = resolve_path(repo, raw_path)

            bank_exists = bool(
                bank_path is not None and bank_path.is_file()
            )

            if state in safe_reuse_states and bank_exists:
                action = "reuse"
            elif config.run_mode == "reuse_only":
                action = "skip_unavailable"
            elif config.run_mode == "automatic":
                if state in (
                    BankState.INCOMPLETE,
                    BankState.SMOKE,
                ):
                    action = "build_or_resume"
                else:
                    action = "build"
            else:
                action = "requires_bank"

            rows.append(
                {
                    "mass_GeV": float(mass_gev),
                    "selection_name": selection_name,
                    "bank_state": state.value,
                    "bank_exists": bank_exists,
                    "bank_action": action,
                    "bank_path": (
                        str(bank_path)
                        if bank_path is not None
                        else ""
                    ),
                    "bank_workspace": str(
                        bank_workspace_dir(
                            config.output_dir,
                            mass_gev,
                            selection_name,
                        )
                    ),
                    "result_dir": str(
                        result_dir(
                            config.output_dir,
                            mass_gev,
                            selection_name,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def write_run_configuration(
    config: AnalysisConfig,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "run_config.json"
    path.write_text(
        json.dumps(config.as_dict(), indent=2) + "\n"
    )
    return path
