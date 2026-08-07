"""Build Week-8 allowed coupling and lifetime domains from saved scan outputs."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis2.config import PRODUCTION_MASSES_GEV, PROFILES, get_config
from analysis2.constraints.bc9 import load_bc9_constraints
from analysis2.constraints.plotting import SU2_SPECS, load_constraint
from analysis2.paths import profile_output_dir
from analysis2.plot_style import style_axis, use_report_style
from analysis2.week8_domains import (
    Interval,
    allowed_coupling_intervals,
    coupling_interval_to_ctau,
    sensitivity_coupling_interval,
    unit_coupling_ctau_at_mass,
)
from analysis2.workflows import float_token, require_columns, write_dataframe, write_manifest

WEEK8_MASSES_GEV = (
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    1.00,
    1.05,
    1.20,
    1.40,
    1.60,
    1.80,
    2.00,
    2.20,
    2.40,
    2.50,
)


MODEL_ORDER = ("ALP-photon-combined", "ALP-SU2L")
MODEL_LABELS = {
    "ALP-photon-combined": r"Photophilic ALP",
    "ALP-SU2L": r"$SU(2)_L$ ALP",
}


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument(
        "--mass",
        dest="masses",
        action="append",
        type=float,
        help="Mass in GeV. Repeat to request several masses. Default: 0.3 GeV.",
    )
    parser.add_argument(
        "--all-production-masses",
        action="store_true",
        help="Use every mass in analysis2.config.PRODUCTION_MASSES_GEV.",
    )
    parser.add_argument(
        "--all-common-masses",
        action="store_true",
        help=(
            "Use every exact mass point for which both models have "
            "a resolved sensitivity boundary and a saved unit-coupling lifetime."
        ),
    )
    parser.add_argument(
        "--mass-min",
        type=float,
        default=None,
        help="Optional lower mass limit in GeV.",
    )
    parser.add_argument(
        "--mass-max",
        type=float,
        default=None,
        help="Optional upper mass limit in GeV.",
    )
    parser.add_argument("--event-level", type=float, default=2.3)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()



def _common_supported_mass_range(
    boundaries: pd.DataFrame,
    scan_data: pd.DataFrame,
    event_level: float,
) -> tuple[float, float]:
    """Return the continuous interpolation range supported by both models."""

    model_ranges: list[tuple[float, float]] = []

    for model in MODEL_ORDER:
        boundary_rows = boundaries[
            (boundaries["model"] == model)
            & np.isclose(
                boundaries["event_level"].to_numpy(float),
                event_level,
            )
            & (boundaries["status"] == "resolved")
        ]

        scan_rows = scan_data[
            scan_data["model"] == model
        ]

        if boundary_rows.empty:
            raise ValueError(
                f"{model} has no resolved N_events={event_level:g} "
                "sensitivity boundaries"
            )

        if scan_rows.empty:
            raise ValueError(
                f"{model} has no saved unit-coupling lifetime scan"
            )

        boundary_masses = boundary_rows[
            "mass_GeV"
        ].to_numpy(float)

        scan_masses = scan_rows[
            "mass_GeV"
        ].to_numpy(float)

        model_min = max(
            float(np.min(boundary_masses)),
            float(np.min(scan_masses)),
        )
        model_max = min(
            float(np.max(boundary_masses)),
            float(np.max(scan_masses)),
        )

        if model_min > model_max:
            raise ValueError(
                f"{model} has no overlap between its sensitivity "
                "and lifetime mass ranges"
            )

        model_ranges.append((model_min, model_max))

    common_min = max(
        model_min
        for model_min, _ in model_ranges
    )
    common_max = min(
        model_max
        for _, model_max in model_ranges
    )

    if common_min > common_max:
        raise ValueError(
            "the two models have no common supported mass range"
        )

    return common_min, common_max


def _requested_masses(
    args,
    boundaries: pd.DataFrame,
    scan_data: pd.DataFrame,
) -> tuple[float, ...]:
    selectors = sum((
        bool(args.masses),
        bool(args.all_production_masses),
        bool(args.all_common_masses),
    ))

    if selectors > 1:
        raise ValueError(
            "use only one of --mass, --all-production-masses, "
            "or --all-common-masses"
        )

    common_min, common_max = _common_supported_mass_range(
        boundaries,
        scan_data,
        args.event_level,
    )

    if args.all_common_masses:
        masses = WEEK8_MASSES_GEV
    elif args.all_production_masses:
        masses = tuple(PRODUCTION_MASSES_GEV)
    else:
        masses = tuple(args.masses or (0.3,))

    masses = tuple(
        float(value)
        for value in masses
        if (
            (args.mass_min is None or value >= args.mass_min)
            and
            (args.mass_max is None or value <= args.mass_max)
        )
    )

    if not masses:
        raise ValueError(
            "no requested masses remain after applying the mass limits"
        )

    invalid = [
        value
        for value in masses
        if (
            not np.isfinite(value)
            or value <= 0.0
            or value < common_min
            or value > common_max
        )
    ]

    if invalid:
        raise ValueError(
            "requested masses outside the common supported range "
            f"[{common_min:.12g}, {common_max:.12g}] GeV: "
            f"{invalid}"
        )

    print(
        "Common supported mass range: "
        f"[{common_min:.12g}, {common_max:.12g}] GeV"
    )

    return tuple(sorted(set(masses)))


def _load_polygons(profile: str) -> dict[str, list[np.ndarray]]:
    bc9 = load_bc9_constraints()
    photon_polygons = [np.asarray(value, dtype=float) for value in bc9.values()]

    su2_directory = profile_output_dir(profile, "constraints") / "converted/alp_su2l"
    if not su2_directory.exists():
        raise FileNotFoundError(
            f"converted SU(2)_L constraint directory not found: {su2_directory}"
        )
    su2_polygons = [load_constraint(su2_directory / filename) for filename, _ in SU2_SPECS]
    return {
        "ALP-photon-combined": photon_polygons,
        "ALP-SU2L": su2_polygons,
    }


def _interval_record(
    interval: Interval,
    quantity: str,
    unit: str,
) -> dict[str, float | bool]:
    return {
        f"{quantity}_min_{unit}": interval.lower,
        f"{quantity}_max_{unit}": interval.upper,
        f"{quantity}_min_inclusive": interval.lower_inclusive,
        f"{quantity}_max_inclusive": interval.upper_inclusive,
    }


def _draw_mass_diagnostic(
    mass_gev: float,
    model_results: dict[str, dict],
    output: Path,
) -> None:
    use_report_style()
    figure, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(8.5, 5.8))
    axes = np.atleast_1d(axes)

    for axis, model in zip(axes, MODEL_ORDER):
        result = model_results[model]
        sensitivity: Interval = result["sensitivity"]
        excluded: list[Interval] = result["excluded"]
        allowed: list[Interval] = result["allowed"]

        axis.hlines(0.0, sensitivity.lower, sensitivity.upper, color="0.75", linewidth=14.0,
                    label=r"Geom-only $N_{\rm events}\geq2.3$")
        for index, interval in enumerate(excluded):
            axis.hlines(0.0, interval.lower, interval.upper, color="C3", linewidth=9.0,
                        label="Excluded" if index == 0 else "_nolegend_")
        for index, interval in enumerate(allowed):
            axis.hlines(0.0, interval.lower, interval.upper, color="C2", linewidth=4.0,
                        label="Week-8 allowed" if index == 0 else "_nolegend_")

        axis.set_xscale("log")
        axis.set_yticks([])
        axis.set_ylim(-0.8, 0.8)
        axis.set_xlabel(r"Coupling [GeV$^{-1}$]")
        axis.text(0.02, 0.82, MODEL_LABELS[model], transform=axis.transAxes, ha="left", va="top")
        axis.grid(True, which="both", axis="x", alpha=0.2)
        axis.legend(loc="upper right", frameon=True)
        style_axis(axis)

    figure.suptitle(rf"Week-8 domain slices at $m_a={mass_gev:g}$ GeV", fontsize=15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    config = get_config(args.profile)

    event_dir = profile_output_dir(args.profile, "event_density")
    boundary_path = event_dir / "event_contour_boundaries.csv"
    scan_path = event_dir / "event_density_scan_coarse.csv"
    if not boundary_path.exists():
        raise FileNotFoundError(f"missing saved contour table: {boundary_path}")
    if not scan_path.exists():
        raise FileNotFoundError(f"missing saved event-density scan: {scan_path}")

    boundaries = pd.read_csv(boundary_path)
    scan_data = pd.read_csv(scan_path)
    require_columns(
        boundaries,
        {
            "model", "mass_GeV", "event_level", "status",
            "lower_coupling_GeV_inv", "upper_coupling_GeV_inv",
        },
        boundary_path,
    )
    require_columns(
        scan_data,
        {"model", "mass_GeV", "unit_coupling_ctau_m"},
        scan_path,
    )
    masses = _requested_masses(
        args,
        boundaries,
        scan_data,
    )

    polygons = _load_polygons(args.profile)
    output_dir = profile_output_dir(args.profile, "week8_domains")
    domain_rows: list[dict] = []
    excluded_rows: list[dict] = []
    summary_rows: list[dict] = []
    plot_paths: list[Path] = []

    for mass_gev in masses:
        model_results: dict[str, dict] = {}
        for model in MODEL_ORDER:
            sensitivity = sensitivity_coupling_interval(
                boundaries, model, mass_gev, args.event_level,
            )
            unit_ctau_m = unit_coupling_ctau_at_mass(scan_data, model, mass_gev)
            allowed, excluded = allowed_coupling_intervals(
                sensitivity, polygons[model], mass_gev,
            )
            model_results[model] = {
                "sensitivity": sensitivity,
                "allowed": allowed,
                "excluded": excluded,
            }

            sensitivity_ctau = coupling_interval_to_ctau(sensitivity, unit_ctau_m)
            summary_rows.append({
                "model": model,
                "mass_GeV": mass_gev,
                "event_level": args.event_level,
                "unit_coupling_ctau_m": unit_ctau_m,
                "sensitivity_coupling_min_GeV_inv": sensitivity.lower,
                "sensitivity_coupling_max_GeV_inv": sensitivity.upper,
                "sensitivity_ctau_min_m": sensitivity_ctau.lower,
                "sensitivity_ctau_max_m": sensitivity_ctau.upper,
                "number_of_excluded_intervals": len(excluded),
                "number_of_allowed_intervals": len(allowed),
            })

            for index, interval in enumerate(excluded):
                excluded_rows.append({
                    "model": model,
                    "mass_GeV": mass_gev,
                    "event_level": args.event_level,
                    "excluded_interval_index": index,
                    **_interval_record(interval, "coupling", "GeV_inv"),
                })

            for index, coupling_interval in enumerate(allowed):
                ctau_interval = coupling_interval_to_ctau(coupling_interval, unit_ctau_m)
                domain_rows.append({
                    "model": model,
                    "mass_GeV": mass_gev,
                    "event_level": args.event_level,
                    "interval_index": index,
                    "unit_coupling_ctau_m": unit_ctau_m,
                    **_interval_record(coupling_interval, "coupling", "GeV_inv"),
                    **_interval_record(ctau_interval, "ctau", "m"),
                })

            print(f"\n{model}, m_a={mass_gev:g} GeV")
            print(
                "  sensitivity: "
                f"g=[{sensitivity.lower:.6e}, {sensitivity.upper:.6e}], "
                f"ctau=[{sensitivity_ctau.lower:.6e}, {sensitivity_ctau.upper:.6e}] m"
            )
            if not allowed:
                print("  allowed: none after exclusions")
            for index, coupling_interval in enumerate(allowed):
                ctau_interval = coupling_interval_to_ctau(coupling_interval, unit_ctau_m)
                print(
                    f"  allowed {index}: "
                    f"g=[{coupling_interval.lower:.6e}, {coupling_interval.upper:.6e}] GeV^-1, "
                    f"ctau=[{ctau_interval.lower:.6e}, {ctau_interval.upper:.6e}] m"
                )

        if not args.no_plots:
            plot_path = output_dir / "plots" / f"week8_domain_slice_ma_{float_token(mass_gev)}.pdf"
            _draw_mass_diagnostic(mass_gev, model_results, plot_path)
            plot_paths.append(plot_path)
            print(f"Saved {plot_path}")

    domain_path = write_dataframe(
        pd.DataFrame(domain_rows), output_dir / "allowed_ctau_domains.csv"
    )
    excluded_path = write_dataframe(
        pd.DataFrame(excluded_rows), output_dir / "excluded_coupling_slices.csv"
    )
    summary_path = write_dataframe(
        pd.DataFrame(summary_rows), output_dir / "sensitivity_slice_summary.csv"
    )
    artifacts = [domain_path, excluded_path, summary_path, *plot_paths]
    write_manifest(
        config,
        "build_week8_ctau_domains",
        output_dir,
        artifacts=artifacts,
        extra={
            "event_level": args.event_level,
            "requested_masses_GeV": list(masses),
            "photon_exclusions": "BC9 laboratory + astrophysical polygons",
            "su2_exclusions": "converted FORESEE polygons",
            "boundary_policy": {
                "sensitivity_boundary_included": True,
                "exclusion_boundary_included_in_exclusion": True,
            },
        },
    )
    print(f"\nSaved Week-8 domains to {output_dir}")


if __name__ == "__main__":
    main()
