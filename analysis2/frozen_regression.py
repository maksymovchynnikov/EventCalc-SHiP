"""Pure numerical regression checks for the frozen frozen-reference calculation.

Compact NPZ banks are compared exactly.  CSV-derived quantities alone receive
the documented round-trip allowance ``rtol=1e-14, atol=5e-15``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .numerical_comparison import (
    CSV_ATOL,
    CSV_RTOL,
    REPORT_COLUMNS,
    compare_csv_files,
    compare_npz_files,
)
from .endpoint_regression import compare_endpoint_artifacts


BANK_FORMAT_VERSION = 1


class RegressionMismatchError(RuntimeError):
    """Raised after the complete report identifies a genuine mismatch."""


@dataclass(frozen=True)
class RegressionPaths:
    reference_scan_path: Path
    frozen_root: Path
    current_scan_path: Path
    current_bank_root: Path
    current_distance_root: Path
    current_profiled_root: Path
    current_domain_path: Path | None = None
    current_bisection_diagnostic_path: Path | None = None

    @property
    def resolved_current_domain_path(self) -> Path:
        return self.current_domain_path or self.current_scan_path.with_name(
            "observable_lifetime_domains.csv"
        )

    @property
    def resolved_current_bisection_diagnostic_path(self) -> Path:
        return self.current_bisection_diagnostic_path or self.current_scan_path.with_name(
            "bisection_diagnostic_ranges.csv"
        )


def _mass_token(mass: float) -> str:
    return f"{mass:.12g}".replace("-", "m").replace(".", "p")


def compare_frozen_outputs(
    paths: RegressionPaths,
    masses: Iterable[float],
    *,
    event_threshold: float = 10.0,
    lifetime_points: int = 20,
    padding_fraction: float = 0.002,
    expected_profile: str = "production",
    expected_selection_name: str = "diphoton_ecal",
    endpoint_convention: str = "log_log_rate_interpolation",
    diagnostic_convention: str = "fixed_step_log_bisection_midpoint",
) -> pd.DataFrame:
    """Return all endpoint, bank, distance, and profiled comparisons."""
    masses = tuple(float(mass) for mass in masses)
    rows = compare_endpoint_artifacts(
        paths.reference_scan_path,
        paths.resolved_current_domain_path,
        paths.resolved_current_bisection_diagnostic_path,
        masses,
        event_threshold=event_threshold,
        lifetime_points=lifetime_points,
        padding_fraction=padding_fraction,
        endpoint_convention=endpoint_convention,
        diagnostic_convention=diagnostic_convention,
    )
    for mass in masses:
        name = f"template_bank_ma_{_mass_token(mass)}.npz"
        rows.extend(
            compare_npz_files(
                paths.frozen_root / "template_banks" / name,
                paths.current_bank_root / "template_banks" / name,
                category="template_bank",
                artifact=name,
                expected_current_metadata={
                    "bank_format_version": np.asarray(BANK_FORMAT_VERSION),
                    "profile": np.asarray(expected_profile),
                    "selection_name": np.asarray(expected_selection_name),
                },
            )
        )
    csv_specs = [
        (
            "distance_summary",
            paths.frozen_root / "distance_maps" / "distance_map_summary.csv",
            paths.current_distance_root / "distance_map_summary.csv",
            "distance_map_summary.csv",
            ("mass_GeV",),
        ),
        (
            "profiled_threshold_summary",
            paths.frozen_root / "profiled_likelihood" / "profiled_threshold_summary.csv",
            paths.current_profiled_root / "profiled_threshold_summary.csv",
            "profiled_threshold_summary.csv",
            ("mass_GeV",),
        ),
    ]
    for mass in masses:
        token = _mass_token(mass)
        csv_specs.extend(
            [
                (
                    "distance_table",
                    paths.frozen_root
                    / "distance_maps"
                    / "tables"
                    / f"distance_map_ma_{token}.csv",
                    paths.current_distance_root / "tables" / f"distance_map_ma_{token}.csv",
                    f"distance_map_ma_{token}.csv",
                    ("mass_GeV", "photon_lifetime_index", "su2_lifetime_index"),
                ),
                (
                    "profiled_worst_case_by_seed",
                    paths.frozen_root
                    / "profiled_likelihood"
                    / "tables"
                    / f"profiled_worst_case_by_seed_ma_{token}.csv",
                    paths.current_profiled_root
                    / "tables"
                    / f"profiled_worst_case_by_seed_ma_{token}.csv",
                    f"profiled_worst_case_by_seed_ma_{token}.csv",
                    ("mass_GeV", "seed", "number_of_events"),
                ),
            ]
        )
    for category, reference, current, artifact, keys in csv_specs:
        rows.extend(
            compare_csv_files(
                reference,
                current,
                category=category,
                artifact=artifact,
                keys=keys,
            )
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def assert_regression_matches(report: pd.DataFrame) -> None:
    mismatches = report.loc[report["status"] == "genuine_mismatch"]
    if len(mismatches):
        raise RegressionMismatchError(
            f"frozen-reference regression found {len(mismatches)} genuine mismatch(es)."
        )
