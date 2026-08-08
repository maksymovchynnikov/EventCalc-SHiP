"""Compare refactored production artifacts with the frozen frozen-reference result."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from alp_discrimination.cache import atomic_output_path
from alp_discrimination.config import PROFILES, get_config
from alp_discrimination.paths import LEGACY_ANALYSIS_ROOT, portable_path, profile_output_dir
from alp_discrimination.frozen_regression import (
    CSV_ATOL,
    CSV_RTOL,
    RegressionPaths,
    assert_regression_matches,
    compare_frozen_outputs,
)
from alp_discrimination.workflows import write_dataframe


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument("--masses", nargs="+", type=float)
    parser.add_argument("--current-root", type=Path)
    parser.add_argument("--current-scan-path", type=Path)
    parser.add_argument("--current-domain-path", type=Path)
    parser.add_argument("--current-bisection-diagnostic-path", type=Path)
    parser.add_argument("--current-bank-root", type=Path)
    parser.add_argument("--current-distance-root", type=Path)
    parser.add_argument("--current-profiled-root", type=Path)
    parser.add_argument("--reference-scan-path", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def resolve_paths(args, profile: str) -> RegressionPaths:
    current = args.current_root or profile_output_dir(profile)
    current_scan = (
        args.current_scan_path
        or current / "scan_ctau_ranges" / "ctau_scan.csv"
    )
    return RegressionPaths(
        reference_scan_path=(
            args.reference_scan_path
            or LEGACY_ANALYSIS_ROOT / "ctau_scan" / "ctau_scan.csv"
        ),
        frozen_root=(
            args.frozen_root
            or LEGACY_ANALYSIS_ROOT / "lifetime_blind_discrimination_final"
        ),
        current_scan_path=current_scan,
        current_bank_root=(
            args.current_bank_root or current / "lifetime_blind_discrimination"
        ),
        current_distance_root=(
            args.current_distance_root or current / "lifetime_blind_distance_maps"
        ),
        current_profiled_root=(
            args.current_profiled_root
            or current / "lifetime_blind_profiled_likelihood"
        ),
        current_domain_path=(
            args.current_domain_path
            or current_scan.with_name("observable_lifetime_domains.csv")
        ),
        current_bisection_diagnostic_path=(
            args.current_bisection_diagnostic_path
            or current_scan.with_name("bisection_diagnostic_ranges.csv")
        ),
    )


def write_regression_outputs(
    report: pd.DataFrame,
    output_dir: Path,
    *,
    profile: str,
    paths: RegressionPaths,
) -> tuple[Path, Path]:
    """Write a portable detailed CSV and machine-readable JSON summary."""
    csv_path = output_dir / "frozen_numerical_comparison.csv"
    json_path = output_dir / "frozen_numerical_comparison.json"
    write_dataframe(report, csv_path)
    counts = report["status"].value_counts().to_dict()
    payload = {
        "profile": profile,
        "status": (
            "genuine_mismatch"
            if counts.get("genuine_mismatch", 0)
            else "agreement"
        ),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "csv_roundtrip_tolerance": {"rtol": CSV_RTOL, "atol": CSV_ATOL},
        "inputs": {
            "reference_scan": portable_path(paths.reference_scan_path),
            "frozen_root": portable_path(paths.frozen_root),
            "current_scan": portable_path(paths.current_scan_path),
            "current_observable_domains": portable_path(
                paths.resolved_current_domain_path
            ),
            "current_bisection_diagnostics": portable_path(
                paths.resolved_current_bisection_diagnostic_path
            ),
            "current_bank_root": portable_path(paths.current_bank_root),
            "current_distance_root": portable_path(paths.current_distance_root),
            "current_profiled_root": portable_path(paths.current_profiled_root),
        },
        "report_csv": portable_path(csv_path),
        "comparisons": json.loads(report.to_json(orient="records")),
    }
    with atomic_output_path(json_path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


def run_workflow(
    *,
    profile: str,
    paths: RegressionPaths,
    output_dir: Path,
    masses: Sequence[float] | None = None,
) -> pd.DataFrame:
    config = get_config(profile)
    selected = config.masses_gev if masses is None else tuple(masses)
    report = compare_frozen_outputs(
        paths,
        selected,
        event_threshold=config.lifetimes.event_threshold,
        lifetime_points=config.templates.lifetime_points_per_model,
        padding_fraction=config.templates.log_endpoint_padding_fraction,
        expected_profile=config.name,
        expected_selection_name=config.selection_name,
        endpoint_convention=config.templates.observable_endpoint_convention,
        diagnostic_convention=config.lifetimes.diagnostic_endpoint_convention,
    )
    write_regression_outputs(
        report,
        output_dir,
        profile=profile,
        paths=paths,
    )
    assert_regression_matches(report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    paths = resolve_paths(args, args.profile)
    output_dir = args.output_dir or profile_output_dir(args.profile, "regression")
    report = run_workflow(
        profile=args.profile,
        paths=paths,
        output_dir=output_dir,
        masses=args.masses,
    )
    columns = [
        "category",
        "artifact",
        "quantity",
        "max_abs_difference",
        "status",
    ]
    print(report[columns].to_string(index=False))


if __name__ == "__main__":
    main()
