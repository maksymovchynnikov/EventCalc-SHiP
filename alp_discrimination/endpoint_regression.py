"""Independent endpoint checks for the frozen frozen-reference lifetime domains."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .numerical_comparison import mismatch_record, numeric_record, text_record
from .observable_domains import (
    collect_observable_domains,
    load_lifetime_scan,
    padded_lifetime_grid,
)


MODEL_LABELS = (
    ("alp_photon_combined", "ALP-photon-combined"),
    ("alp_su2l", "ALP-SU2L"),
)


def _reference_values(
    path: Path,
    masses: tuple[float, ...],
    *,
    event_threshold: float,
    lifetime_points: int,
    padding_fraction: float,
) -> dict[tuple[str, float], dict]:
    domains = collect_observable_domains(
        load_lifetime_scan(path),
        threshold=event_threshold,
        allow_truncated=False,
    )
    values = {}
    for model_id, legacy_name in MODEL_LABELS:
        for mass in masses:
            matches = [
                domain
                for (label, available_mass), domain in domains.items()
                if label in {model_id, legacy_name}
                and np.isclose(available_mass, mass, rtol=0.0, atol=1.0e-12)
            ]
            if len(matches) != 1:
                raise ValueError("scan does not contain exactly one requested domain")
            domain = matches[0]
            grid = padded_lifetime_grid(domain, lifetime_points, padding_fraction)
            values[(model_id, mass)] = {
                "raw_log_log_lower_m": domain.lower_m,
                "raw_log_log_upper_m": domain.upper_m,
                "production_padded_grid_lower_m": grid[0],
                "production_padded_grid_upper_m": grid[-1],
                "fixed_step_bisection_diagnostic_lower_m": domain.bisection_lower_m,
                "fixed_step_bisection_diagnostic_upper_m": domain.bisection_upper_m,
                "lower_is_scan_boundary": domain.lower_is_scan_boundary,
                "upper_is_scan_boundary": domain.upper_is_scan_boundary,
            }
    return values


def _matching_row(
    table: pd.DataFrame,
    model_id: str,
    legacy_name: str,
    mass_gev: float,
) -> pd.Series:
    mass_matches = np.isclose(
        pd.to_numeric(table["mass_GeV"], errors="raise"),
        mass_gev,
        rtol=0.0,
        atol=1.0e-12,
    )
    model_matches = np.zeros(len(table), dtype=bool)
    if "model_id" in table:
        model_matches |= table["model_id"].astype(str).eq(model_id).to_numpy()
    if "model" in table:
        model_matches |= table["model"].astype(str).isin(
            (model_id, legacy_name)
        ).to_numpy()
    matches = table.loc[mass_matches & model_matches]
    if len(matches) != 1:
        raise ValueError("endpoint table does not contain exactly one requested row")
    return matches.iloc[0]


def _validate_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(table)
    if not ({"model", "model_id"} & set(table)):
        missing.add("model or model_id")
    if missing:
        raise ValueError(f"{label} has missing columns: {sorted(missing)}")


def _saved_domain_values(
    path: Path,
    masses: tuple[float, ...],
) -> dict[tuple[str, float], dict]:
    table = pd.read_csv(path)
    _validate_columns(
        table,
        {
            "mass_GeV",
            "template_domain_lower_m",
            "template_domain_upper_m",
            "template_grid_lower_m",
            "template_grid_upper_m",
            "lower_is_scan_boundary",
            "upper_is_scan_boundary",
            "number_of_lifetime_templates",
            "template_endpoint_convention",
            "diagnostic_endpoint_convention",
            "template_log_endpoint_padding_fraction",
        },
        "saved domain table",
    )
    values = {}
    for model_id, legacy_name in MODEL_LABELS:
        for mass in masses:
            row = _matching_row(table, model_id, legacy_name, mass)
            values[(model_id, mass)] = {
                "raw_log_log_lower_m": row["template_domain_lower_m"],
                "raw_log_log_upper_m": row["template_domain_upper_m"],
                "production_padded_grid_lower_m": row["template_grid_lower_m"],
                "production_padded_grid_upper_m": row["template_grid_upper_m"],
                "lower_is_scan_boundary": row["lower_is_scan_boundary"],
                "upper_is_scan_boundary": row["upper_is_scan_boundary"],
                "number_of_lifetime_templates": row["number_of_lifetime_templates"],
                "template_endpoint_convention": row["template_endpoint_convention"],
                "diagnostic_endpoint_convention": row["diagnostic_endpoint_convention"],
                "template_log_endpoint_padding_fraction": row[
                    "template_log_endpoint_padding_fraction"
                ],
            }
    return values


def _saved_bisection_values(
    path: Path,
    masses: tuple[float, ...],
) -> dict[tuple[str, float], dict]:
    table = pd.read_csv(path)
    _validate_columns(
        table,
        {
            "mass_GeV",
            "bisection_diagnostic_lower_m",
            "bisection_diagnostic_upper_m",
        },
        "bisection table",
    )
    values = {}
    for model_id, legacy_name in MODEL_LABELS:
        for mass in masses:
            row = _matching_row(table, model_id, legacy_name, mass)
            values[(model_id, mass)] = {
                "fixed_step_bisection_diagnostic_lower_m": row[
                    "bisection_diagnostic_lower_m"
                ],
                "fixed_step_bisection_diagnostic_upper_m": row[
                    "bisection_diagnostic_upper_m"
                ],
            }
    return values


def _comparison_rows(
    reference: dict,
    current: dict,
    *,
    category: str,
    artifact: str,
    quantities: tuple[str, ...],
) -> list[dict]:
    rows = []
    for key in sorted(reference, key=lambda item: (item[1], item[0])):
        for quantity in quantities:
            label = f"{key[0]}:ma={key[1]:g}:{quantity}"
            arguments = (
                category,
                artifact,
                label,
                [reference[key][quantity]],
                [current[key][quantity]],
            )
            if isinstance(reference[key][quantity], (bool, np.bool_, str)):
                rows.append(text_record(*arguments, "frozen_scan_to_saved_csv"))
            else:
                rows.append(
                    numeric_record(
                        *arguments,
                        mode="frozen_scan_to_saved_csv",
                        allow_csv_tolerance=True,
                    )
                )
    return rows


def compare_endpoint_artifacts(
    reference_scan_path: Path,
    current_domain_path: Path,
    current_bisection_path: Path,
    masses: tuple[float, ...],
    *,
    event_threshold: float,
    lifetime_points: int,
    padding_fraction: float,
    endpoint_convention: str,
    diagnostic_convention: str,
) -> list[dict]:
    """Compare raw, padded, and fixed-step diagnostic endpoints separately."""
    try:
        reference = _reference_values(
            reference_scan_path,
            masses,
            event_threshold=event_threshold,
            lifetime_points=lifetime_points,
            padding_fraction=padding_fraction,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        return [
            mismatch_record(
                "lifetime_endpoint_reference",
                reference_scan_path.name,
                type(error).__name__,
            )
        ]
    expected_domain = {
        key: {
            **value,
            "number_of_lifetime_templates": lifetime_points,
            "template_endpoint_convention": endpoint_convention,
            "diagnostic_endpoint_convention": diagnostic_convention,
            "template_log_endpoint_padding_fraction": padding_fraction,
        }
        for key, value in reference.items()
    }
    rows = []
    domain_specs = (
        (
            "lifetime_raw_log_log_endpoints",
            (
                "raw_log_log_lower_m",
                "raw_log_log_upper_m",
                "lower_is_scan_boundary",
                "upper_is_scan_boundary",
            ),
        ),
        (
            "lifetime_production_padded_endpoints",
            ("production_padded_grid_lower_m", "production_padded_grid_upper_m"),
        ),
        (
            "lifetime_endpoint_metadata",
            (
                "number_of_lifetime_templates",
                "template_endpoint_convention",
                "diagnostic_endpoint_convention",
                "template_log_endpoint_padding_fraction",
            ),
        ),
    )
    try:
        current_domain = _saved_domain_values(current_domain_path, masses)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        rows.extend(
            mismatch_record(category, current_domain_path.name, type(error).__name__)
            for category, _ in domain_specs
        )
    else:
        for category, quantities in domain_specs:
            rows.extend(
                _comparison_rows(
                    expected_domain,
                    current_domain,
                    category=category,
                    artifact=current_domain_path.name,
                    quantities=quantities,
                )
            )

    diagnostic_category = "lifetime_fixed_step_bisection_diagnostics"
    diagnostic_quantities = (
        "fixed_step_bisection_diagnostic_lower_m",
        "fixed_step_bisection_diagnostic_upper_m",
    )
    try:
        current_diagnostic = _saved_bisection_values(current_bisection_path, masses)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        rows.append(
            mismatch_record(
                diagnostic_category,
                current_bisection_path.name,
                type(error).__name__,
            )
        )
    else:
        rows.extend(
            _comparison_rows(
                reference,
                current_diagnostic,
                category=diagnostic_category,
                artifact=current_bisection_path.name,
                quantities=diagnostic_quantities,
            )
        )
    return rows
