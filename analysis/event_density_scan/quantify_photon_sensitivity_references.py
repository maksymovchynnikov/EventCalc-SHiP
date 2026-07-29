from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_event_density_with_constraints import (
    BOUNDARY_PATH,
    PLOT_DIR,
)


ANALYSIS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = ANALYSIS_DIR.parents[1]
REFERENCE_DIR = ANALYSIS_DIR / "reference_curves"

MODEL_NAME = "ALP-photon-combined"
EVENT_LEVEL = 2.3
N_COMPARISON_POINTS = 600
CONTAINMENT_TOLERANCE_DEX = 1.0e-3

ENDPOINT_FRACTION = 0.98


REFERENCE_PATTERNS = {
    "geom_only": ("Sensitivity_ALP-photon_at_SHiP-ECN3-" "geom-only_Nev=2.3_Npot=6.e20*.json"),
    "baseline": ("Sensitivity_ALP-photon_at_SHiP-ECN3-" "baseline_Nev=2.3_Npot=6.e20*.json"),
}

REFERENCE_STYLES = {
    "geom_only": {
        "label": "Reference: geom only",
        "color": "C3",
        "linestyle": "--",
        "linewidth": 2.2,
    },
    "baseline": {
        "label": "Reference: baseline",
        "color": "C2",
        "linestyle": "-.",
        "linewidth": 2.2,
    },
}

POINTWISE_OUTPUT_PATH = ANALYSIS_DIR / "photon_sensitivity_reference_log_distances.csv"
SUMMARY_OUTPUT_PATH = ANALYSIS_DIR / "photon_sensitivity_reference_distance_summary.csv"
LOWER_PLOT_STEM = PLOT_DIR / "photon_sensitivity_reference_log_distance_lower"
UPPER_PLOT_STEM = PLOT_DIR / "photon_sensitivity_reference_log_distance_upper"


def resolve_reference_path(reference_name: str) -> Path:
    """Locate exactly one JSON file for a reference contour."""
    pattern = REFERENCE_PATTERNS[reference_name]
    search_directories = (
        REFERENCE_DIR,
        ANALYSIS_DIR,
        REPOSITORY_ROOT,
    )

    matches: list[Path] = []

    for directory in search_directories:
        if directory.exists():
            matches.extend(path.resolve() for path in directory.glob(pattern))

    matches = sorted(set(matches), key=str)

    if not matches:
        raise FileNotFoundError(
            "Could not find a reference JSON matching:\n"
            f"  {pattern}\n\n"
            "Place one copy in:\n"
            f"  {REFERENCE_DIR}"
        )

    if len(matches) > 1:
        match_text = "\n".join(f"  {path}" for path in matches)
        raise RuntimeError(
            "Found multiple matching reference JSON files. "
            "Keep only one copy of each curve:\n"
            f"{match_text}"
        )

    return matches[0]


def load_reference_points(path: Path) -> np.ndarray:
    """Load and validate one closed sensitivity polygon."""
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Expected a top-level list with exactly one entry in:\n  {path}")

    domains = payload[0].get("Sensitivity domains")

    if not isinstance(domains, list) or len(domains) != 1:
        raise ValueError(f"Expected exactly one sensitivity domain in:\n  {path}")

    points = np.asarray(domains[0], dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected sensitivity points with shape (N, 2) in:\n  {path}")

    if len(points) < 4:
        raise ValueError(f"The reference contour contains too few points in:\n  {path}")

    if not np.all(np.isfinite(points)) or np.any(points <= 0.0):
        raise ValueError(
            f"Reference masses and couplings must be finite and positive in:\n  {path}"
        )

    # Remove a repeated closing point. The branch splitter closes the
    # polygon conceptually and does not need the duplicate row.
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]

    return points


def rotate_polygon_to_lower_left(points: np.ndarray) -> np.ndarray:
    """Rotate the ordered polygon so it starts at its lower-left point."""
    minimum_mass = np.min(points[:, 0])
    minimum_mass_indices = np.flatnonzero(
        np.isclose(
            points[:, 0],
            minimum_mass,
            rtol=1.0e-12,
            atol=0.0,
        )
    )

    if len(minimum_mass_indices) == 0:
        raise RuntimeError("Could not identify the minimum-mass edge.")

    start_index = minimum_mass_indices[np.argmin(points[minimum_mass_indices, 1])]

    return np.vstack(
        [
            points[start_index:],
            points[:start_index],
        ]
    )


def reduce_branch(
    points: np.ndarray,
    aggregation: str,
) -> pd.DataFrame:
    """Convert one branch into a single-valued coupling function of mass."""
    branch = pd.DataFrame(
        {
            "mass_GeV": points[:, 0],
            "coupling_GeV_inv": points[:, 1],
        }
    )

    branch = (
        branch.groupby("mass_GeV", as_index=False)
        .agg(coupling_GeV_inv=("coupling_GeV_inv", aggregation))
        .sort_values("mass_GeV")
        .reset_index(drop=True)
    )

    if len(branch) < 2:
        raise ValueError("A contour branch contains fewer than two masses.")

    return branch


def interpolate_log_coupling(
    branch: pd.DataFrame,
    masses_GeV: np.ndarray,
) -> np.ndarray:
    """Interpolate g(m) linearly in log10(m)-log10(g) space."""
    log_mass = np.log10(branch["mass_GeV"].to_numpy(dtype=float))
    log_coupling = np.log10(branch["coupling_GeV_inv"].to_numpy(dtype=float))

    interpolated_log_coupling = np.interp(
        np.log10(masses_GeV),
        log_mass,
        log_coupling,
    )

    return 10.0**interpolated_log_coupling


def split_reference_branches(
    points: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Split an ordered closed contour at its maximum-mass turning point."""
    ordered = rotate_polygon_to_lower_left(points)
    maximum_mass_index = int(np.argmax(ordered[:, 0]))

    if maximum_mass_index == 0 or maximum_mass_index == len(ordered) - 1:
        raise ValueError("The maximum-mass point does not divide the polygon into two branches.")

    candidate_a = ordered[: maximum_mass_index + 1]
    candidate_b = ordered[maximum_mass_index:]

    # Use median aggregation only to determine which candidate lies above
    # the other at an interior probe mass.
    candidate_a_probe = reduce_branch(candidate_a, "median")
    candidate_b_probe = reduce_branch(candidate_b, "median")

    probe_minimum = max(
        candidate_a_probe["mass_GeV"].min(),
        candidate_b_probe["mass_GeV"].min(),
    )
    probe_maximum = min(
        candidate_a_probe["mass_GeV"].max(),
        candidate_b_probe["mass_GeV"].max(),
    )

    if probe_minimum >= probe_maximum:
        raise ValueError("The two reference branches have no mass overlap.")

    probe_mass = np.sqrt(probe_minimum * probe_maximum)
    coupling_a = interpolate_log_coupling(
        candidate_a_probe,
        np.asarray([probe_mass]),
    )[0]
    coupling_b = interpolate_log_coupling(
        candidate_b_probe,
        np.asarray([probe_mass]),
    )[0]

    if coupling_a > coupling_b:
        upper_points = candidate_a
        lower_points = candidate_b
    else:
        upper_points = candidate_b
        lower_points = candidate_a

    # At repeated masses, the upper edge is the maximum coupling and the
    # lower edge is the minimum coupling. This also handles the vertical
    # closure at the minimum mass.
    upper = reduce_branch(upper_points, "max")
    lower = reduce_branch(lower_points, "min")

    return {
        "lower": lower,
        "upper": upper,
    }


def load_eventcalc_branches() -> dict[str, pd.DataFrame]:
    """Load the EventCalc lower and upper branches at the chosen event level."""
    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            "Could not find the EventCalc contour table:\n"
            f"  {BOUNDARY_PATH}\n"
            "Run scan_event_density.py first."
        )

    boundary_data = pd.read_csv(BOUNDARY_PATH)
    required_columns = {
        "model",
        "mass_GeV",
        "event_level",
        "lower_coupling_GeV_inv",
        "upper_coupling_GeV_inv",
    }
    missing_columns = required_columns - set(boundary_data.columns)

    if missing_columns:
        raise ValueError(
            f"The EventCalc contour table is missing columns:\n  {sorted(missing_columns)}"
        )

    selected = boundary_data.loc[
        (boundary_data["model"] == MODEL_NAME)
        & np.isclose(
            boundary_data["event_level"],
            EVENT_LEVEL,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No EventCalc contour found for model {MODEL_NAME!r} "
            f"and N_events = {EVENT_LEVEL:g} in:\n"
            f"  {BOUNDARY_PATH}"
        )

    branches: dict[str, pd.DataFrame] = {}

    for branch_name, coupling_column in (
        ("lower", "lower_coupling_GeV_inv"),
        ("upper", "upper_coupling_GeV_inv"),
    ):
        branch = selected[["mass_GeV", coupling_column]].rename(
            columns={coupling_column: "coupling_GeV_inv"}
        )
        branch = branch.loc[
            np.isfinite(branch["mass_GeV"])
            & np.isfinite(branch["coupling_GeV_inv"])
            & (branch["mass_GeV"] > 0.0)
            & (branch["coupling_GeV_inv"] > 0.0)
        ]
        branch = (
            branch.groupby("mass_GeV", as_index=False)
            .agg(coupling_GeV_inv=("coupling_GeV_inv", "median"))
            .sort_values("mass_GeV")
            .reset_index(drop=True)
        )

        if len(branch) < 2:
            raise ValueError(f"The EventCalc {branch_name} branch contains fewer than two points.")

        branches[branch_name] = branch

    return branches


def make_pointwise_comparison(
    eventcalc_branches: dict[str, pd.DataFrame],
    reference_branches: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Evaluate signed log10 coupling differences on common mass grids."""
    rows: list[pd.DataFrame] = []

    for reference_name, branches in reference_branches.items():
        minimum_mass = max(
            eventcalc_branches["lower"]["mass_GeV"].min(),
            eventcalc_branches["upper"]["mass_GeV"].min(),
            branches["lower"]["mass_GeV"].min(),
            branches["upper"]["mass_GeV"].min(),
        )
        maximum_mass = min(
            eventcalc_branches["lower"]["mass_GeV"].max(),
            eventcalc_branches["upper"]["mass_GeV"].max(),
            branches["lower"]["mass_GeV"].max(),
            branches["upper"]["mass_GeV"].max(),
        )

        if minimum_mass >= maximum_mass:
            raise ValueError(f"No common mass interval for reference {reference_name!r}.")

        comparison_maximum_mass = ENDPOINT_FRACTION * maximum_mass

        masses = np.geomspace(
            minimum_mass,
            comparison_maximum_mass,
            N_COMPARISON_POINTS,
        )

        for branch_name in ("lower", "upper"):
            eventcalc_coupling = interpolate_log_coupling(
                eventcalc_branches[branch_name],
                masses,
            )
            reference_coupling = interpolate_log_coupling(
                branches[branch_name],
                masses,
            )

            log_distance_dex = np.log10(reference_coupling / eventcalc_coupling)

            rows.append(
                pd.DataFrame(
                    {
                        "reference": reference_name,
                        "branch": branch_name,
                        "mass_GeV": masses,
                        "eventcalc_coupling_GeV_inv": eventcalc_coupling,
                        "reference_coupling_GeV_inv": reference_coupling,
                        "log10_reference_over_eventcalc_dex": log_distance_dex,
                        "absolute_log_distance_dex": np.abs(log_distance_dex),
                        "coupling_ratio_reference_over_eventcalc": (
                            reference_coupling / eventcalc_coupling
                        ),
                    }
                )
            )

    return pd.concat(rows, ignore_index=True)


def make_summary(pointwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize maximum deviations and EventCalc containment."""
    rows: list[dict[str, float | str | bool | int]] = []

    for reference_name in REFERENCE_PATTERNS:
        selected = pointwise.loc[pointwise["reference"] == reference_name]
        lower = selected.loc[selected["branch"] == "lower"].reset_index(drop=True)
        upper = selected.loc[selected["branch"] == "upper"].reset_index(drop=True)

        if len(lower) != len(upper) or not np.allclose(
            lower["mass_GeV"],
            upper["mass_GeV"],
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RuntimeError(f"Lower and upper mass grids differ for {reference_name!r}.")

        lower_distance = lower["log10_reference_over_eventcalc_dex"].to_numpy(dtype=float)
        upper_distance = upper["log10_reference_over_eventcalc_dex"].to_numpy(dtype=float)
        masses = lower["mass_GeV"].to_numpy(dtype=float)

        lower_maximum_index = int(np.argmax(np.abs(lower_distance)))
        upper_maximum_index = int(np.argmax(np.abs(upper_distance)))

        # A closed reference sensitivity region is inside EventCalc when
        # its lower boundary is above EventCalc and its upper boundary is
        # below EventCalc.
        lower_inside = lower_distance >= -CONTAINMENT_TOLERANCE_DEX
        upper_inside = upper_distance <= CONTAINMENT_TOLERANCE_DEX
        both_inside = lower_inside & upper_inside

        worst_lower_outward = max(0.0, -float(np.min(lower_distance)))
        worst_upper_outward = max(0.0, float(np.max(upper_distance)))

        rows.append(
            {
                "reference": reference_name,
                "number_of_mass_points": len(masses),
                "minimum_mass_GeV": float(masses[0]),
                "maximum_mass_GeV": float(masses[-1]),
                "maximum_absolute_lower_distance_dex": float(
                    np.abs(lower_distance[lower_maximum_index])
                ),
                "mass_at_maximum_lower_distance_GeV": float(masses[lower_maximum_index]),
                "maximum_absolute_upper_distance_dex": float(
                    np.abs(upper_distance[upper_maximum_index])
                ),
                "mass_at_maximum_upper_distance_GeV": float(masses[upper_maximum_index]),
                "largest_lower_coupling_factor_difference": float(
                    10.0 ** np.abs(lower_distance[lower_maximum_index])
                ),
                "largest_upper_coupling_factor_difference": float(
                    10.0 ** np.abs(upper_distance[upper_maximum_index])
                ),
                "lower_branch_inside_fraction": float(np.mean(lower_inside)),
                "upper_branch_inside_fraction": float(np.mean(upper_inside)),
                "both_branches_inside_fraction": float(np.mean(both_inside)),
                "worst_lower_outward_violation_dex": worst_lower_outward,
                "worst_upper_outward_violation_dex": worst_upper_outward,
                "strictly_inside_with_tolerance": bool(np.all(both_inside)),
                "containment_tolerance_dex": CONTAINMENT_TOLERANCE_DEX,
            }
        )

    return pd.DataFrame(rows)


def plot_log_distance(
    pointwise: pd.DataFrame,
    branch_name: str,
    output_stem: Path,
) -> None:
    """Plot the signed vertical log-distance from EventCalc."""
    figure, axis = plt.subplots(figsize=(8.2, 5.4))

    selected_branch = pointwise.loc[pointwise["branch"] == branch_name]

    for reference_name in ("geom_only", "baseline"):
        selected = selected_branch.loc[selected_branch["reference"] == reference_name]
        style = REFERENCE_STYLES[reference_name]

        axis.plot(
            selected["mass_GeV"],
            selected["log10_reference_over_eventcalc_dex"],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    maximum_absolute_distance = float(selected_branch["absolute_log_distance_dex"].max())
    y_limit = max(0.02, 1.10 * maximum_absolute_distance)

    axis.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="Exact agreement with EventCalc",
    )

    axis.set_xscale("log")
    axis.set_xlim(
        selected_branch["mass_GeV"].min(),
        selected_branch["mass_GeV"].max(),
    )
    axis.set_ylim(-y_limit, y_limit)
    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel(r"$\Delta\log_{10}g=" r"\log_{10}(g_{\rm ref}/g_{\rm EventCalc})$ [dex]")
    axis.set_title(
        "ALP-photon reference distance from EventCalc\n"
        f"{branch_name.capitalize()} sensitivity branch, "
        rf"$N_{{\rm events}}={EVENT_LEVEL:g}$"
    )

    interpretation = (
        r"Inside EventCalc: $\Delta\log_{10}g\geq 0$"
        if branch_name == "lower"
        else r"Inside EventCalc: $\Delta\log_{10}g\leq 0$"
    )
    axis.text(
        0.02,
        0.04,
        interpretation,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
            "pad": 4.0,
        },
    )

    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    axis.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="whitesmoke",
        edgecolor="gray",
        fontsize=9.5,
    )
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def print_summary(summary: pd.DataFrame) -> None:
    print()
    print("=" * 88)
    print("Quantitative ALP-photon sensitivity-reference comparison")
    print("=" * 88)
    print()
    print(
        "Signed distance definition:\n"
        "  Delta log10(g) = log10(g_reference / g_EventCalc)\n"
        "  0 dex: exact agreement\n"
        "  +0.1 dex: reference coupling is a factor 1.26 higher\n"
        "  -0.1 dex: reference coupling is a factor 1.26 lower"
    )

    for _, row in summary.iterrows():
        print()
        print(f"Reference: {row['reference']}")
        print(
            "  Common mass interval: "
            f"{row['minimum_mass_GeV']:.6g}--"
            f"{row['maximum_mass_GeV']:.6g} GeV"
        )
        print(
            "  Maximum |lower distance|: "
            f"{row['maximum_absolute_lower_distance_dex']:.6g} dex "
            "(factor "
            f"{row['largest_lower_coupling_factor_difference']:.6g}) "
            "at m = "
            f"{row['mass_at_maximum_lower_distance_GeV']:.6g} GeV"
        )
        print(
            "  Maximum |upper distance|: "
            f"{row['maximum_absolute_upper_distance_dex']:.6g} dex "
            "(factor "
            f"{row['largest_upper_coupling_factor_difference']:.6g}) "
            "at m = "
            f"{row['mass_at_maximum_upper_distance_GeV']:.6g} GeV"
        )
        print(
            "  Lower branch inside EventCalc: "
            f"{100.0 * row['lower_branch_inside_fraction']:.2f}%"
        )
        print(
            "  Upper branch inside EventCalc: "
            f"{100.0 * row['upper_branch_inside_fraction']:.2f}%"
        )
        print(
            "  Both branches inside EventCalc: "
            f"{100.0 * row['both_branches_inside_fraction']:.2f}%"
        )
        print(f"  Strictly inside within tolerance: {row['strictly_inside_with_tolerance']}")
        print(
            "  Worst outward violation: lower = "
            f"{row['worst_lower_outward_violation_dex']:.6g} dex, "
            "upper = "
            f"{row['worst_upper_outward_violation_dex']:.6g} dex"
        )


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    eventcalc_branches = load_eventcalc_branches()

    reference_paths = {
        reference_name: resolve_reference_path(reference_name)
        for reference_name in REFERENCE_PATTERNS
    }
    reference_branches = {
        reference_name: split_reference_branches(load_reference_points(path))
        for reference_name, path in reference_paths.items()
    }

    pointwise = make_pointwise_comparison(
        eventcalc_branches,
        reference_branches,
    )
    summary = make_summary(pointwise)

    pointwise.to_csv(POINTWISE_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    plot_log_distance(
        pointwise,
        branch_name="lower",
        output_stem=LOWER_PLOT_STEM,
    )
    plot_log_distance(
        pointwise,
        branch_name="upper",
        output_stem=UPPER_PLOT_STEM,
    )

    print_summary(summary)

    print()
    print("Saved pointwise comparison to:")
    print(f"  {POINTWISE_OUTPUT_PATH}")
    print("Saved summary to:")
    print(f"  {SUMMARY_OUTPUT_PATH}")
    print("Saved lower-branch plots to:")
    print(f"  {LOWER_PLOT_STEM.with_suffix('.pdf')}")
    print("Saved upper-branch plots to:")
    print(f"  {UPPER_PLOT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
