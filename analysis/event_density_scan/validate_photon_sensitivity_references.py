from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_event_density_with_constraints import (
    BOUNDARY_PATH,
    COMBINED_LIMITS,
    PLOT_DIR,
    TABLE_LIMITS_GEV,
)

from analysis.plot_style import (
    style_axis,
    use_report_style,
)

from analysis.constraints.plotting_helpers import (
    PHOTON_SPECS,
    draw_constraints,
    load_label_config,
)

ANALYSIS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = ANALYSIS_DIR.parents[1]
REFERENCE_DIR = ANALYSIS_DIR / "reference_curves"
CONSTRAINTS_DIR = ANALYSIS_DIR.parent / "constraints"
PHOTON_CONSTRAINT_DIR = CONSTRAINTS_DIR / "raw" / "alp_photon"

MODEL_NAME = "ALP-photon-combined"
REFERENCE_EVENT_LEVEL = 2.3
EVENTCALC_EVENT_LEVEL = 2.3

REFERENCE_PATTERNS = {
    "epsilon_dec_1": (
        "Sensitivity_ALP-photon_at_SHiP-ECN3-"
        "epsilon-dec-1_Nev=2.3_Npot=6.e20.json"
    ),
    "geom_only": (
        "Sensitivity_ALP-photon_at_SHiP-ECN3-"
        "geom-only_Nev=2.3_Npot=6.e20.json"
    ),
}

REFERENCE_STYLES = {
    "epsilon_dec_1": {
        "label": (
            r"Reference: $\epsilon_{\rm dec}=1$"
        ),
        "color": "C1",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    "geom_only": {
        "label": "Reference: Geom only",
        "color": "C2",
        "linestyle": "--",
        "linewidth": 2.0,
    },
}

OUTPUT_STEM = PLOT_DIR / "photon_sensitivity_reference_comparison"
SUMMARY_PATH = ANALYSIS_DIR / "photon_sensitivity_reference_summary.csv"


def resolve_reference_path(reference_name: str) -> Path:
    """Locate one of the reference JSON files."""
    pattern = REFERENCE_PATTERNS[reference_name]
    search_directories = (REFERENCE_DIR,)
    matches: list[Path] = []

    for directory in search_directories:
        if not directory.exists():
            continue
        matches.extend(path.resolve() for path in directory.glob(pattern))

    # Remove duplicates while preserving deterministic ordering.
    matches = sorted(set(matches), key=lambda path: (len(path.name), str(path)))

    if not matches:
        raise FileNotFoundError(
            "Could not locate the reference JSON matching:\n"
            f"  {pattern}\n\n"
            "Place the file in:\n"
            f"  {REFERENCE_DIR}\n"
            "The script accepts both the original filename and "
            "filenames with suffixes such as '(2)'."
        )

    if len(matches) > 1:
        match_text = "\n".join(f"  {path}" for path in matches)

        raise RuntimeError(
            "Found multiple matching reference JSON files. "
            "Keep only one copy of each reference curve:\n"
            f"{match_text}"
        )

    return matches[0]


def load_reference_curve(path: Path, reference_name: str) -> dict:
    """Load and validate one closed sensitivity domain from JSON."""
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(
            "Expected the top-level JSON object to be a list "
            f"containing exactly one entry:\n  {path}"
        )

    entry = payload[0]
    required_keys = {"Production modes", "Decay description", "Sensitivity domains"}
    missing_keys = required_keys - set(entry)

    if missing_keys:
        raise ValueError(f"Missing JSON keys in {path}:\n  {sorted(missing_keys)}")

    domains = entry["Sensitivity domains"]

    if not isinstance(domains, list) or len(domains) != 1:
        raise ValueError(f"Expected exactly one sensitivity domain in:\n  {path}")

    points = np.asarray(domains[0], dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"The sensitivity domain must have shape (N, 2) in:\n  {path}")
    if len(points) < 3:
        raise ValueError(f"The sensitivity domain contains too few points in:\n  {path}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"The sensitivity domain contains non-finite values in:\n  {path}")
    if np.any(points <= 0.0):
        raise ValueError(f"Masses and couplings must be strictly positive in:\n  {path}")

    if not np.allclose(points[0], points[-1], rtol=0.0, atol=0.0):
        points = np.vstack([points, points[0]])

    return {
        "name": reference_name,
        "path": path,
        "production_modes": list(entry["Production modes"]),
        "decay_description": str(entry["Decay description"]),
        "points": points,
    }


def load_eventcalc_contour() -> pd.DataFrame:
    """Load the current primary-only EventCalc N=3 boundary."""
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
        raise ValueError(f"The contour table is missing columns:\n  {sorted(missing_columns)}")

    selected = (
        boundary_data.loc[
            (boundary_data["model"] == MODEL_NAME)
            & np.isclose(
                boundary_data["event_level"],
                EVENTCALC_EVENT_LEVEL,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]
        .sort_values("mass_GeV")
        .reset_index(drop=True)
    )

    if selected.empty:
        raise ValueError(
            "No primary-only ALP-photon contour was found for "
            f"N_events = {EVENTCALC_EVENT_LEVEL:g} in:\n"
            f"  {BOUNDARY_PATH}"
        )

    return selected


def draw_eventcalc_contour(axis: plt.Axes, contour_data: pd.DataFrame) -> None:
    """Draw both branches of the current primary-only contour."""
    masses = contour_data["mass_GeV"].to_numpy(dtype=float)
    lower = contour_data["lower_coupling_GeV_inv"].to_numpy(dtype=float)
    upper = contour_data["upper_coupling_GeV_inv"].to_numpy(dtype=float)

    valid_lower = np.isfinite(masses) & np.isfinite(lower) & (masses > 0.0) & (lower > 0.0)
    valid_upper = np.isfinite(masses) & np.isfinite(upper) & (masses > 0.0) & (upper > 0.0)
    label = "EventCalc: Geom only"

    axis.plot(
        masses[valid_lower],
        lower[valid_lower],
        color="C0",
        linestyle="-",
        linewidth=2.0,
        label=label,
        zorder=20,
    )

    axis.plot(
        masses[valid_upper],
        upper[valid_upper],
        color="C0",
        linestyle=":",
        linewidth=2.6,
        label="_nolegend_",
        zorder=20,
    )


def draw_reference_curve(axis: plt.Axes, reference: dict) -> None:
    """Draw one closed JSON sensitivity domain."""
    points = reference["points"]
    style = REFERENCE_STYLES[reference["name"]]
    axis.plot(
        points[:, 0],
        points[:, 1],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        label=style["label"],
        zorder=21,
    )


def configure_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")

    axis.set_xlim(*COMBINED_LIMITS[MODEL_NAME]["x"])
    axis.set_ylim(*COMBINED_LIMITS[MODEL_NAME]["y"])

    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel(r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]")

    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.grid(False)

    legend = axis.legend(
        loc="center right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="whitesmoke",
        edgecolor="gray",
    )

    legend.get_frame().set_linewidth(0.8)
    style_axis(axis)



def make_summary(references: list[dict]) -> pd.DataFrame:
    rows = []
    for reference in references:
        points = reference["points"]
        rows.append(
            {
                "reference": reference["name"],
                "path": str(reference["path"]),
                "event_level": REFERENCE_EVENT_LEVEL,
                "number_of_points": len(points),
                "minimum_mass_GeV": float(np.min(points[:, 0])),
                "maximum_mass_GeV": float(np.max(points[:, 0])),
                "minimum_coupling_GeV_inv": float(np.min(points[:, 1])),
                "maximum_coupling_GeV_inv": float(np.max(points[:, 1])),
                "decay_description": reference["decay_description"],
                "production_modes": "; ".join(reference["production_modes"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True,)
    reference_paths = {name: resolve_reference_path(name) for name in REFERENCE_PATTERNS}
    references = [
        load_reference_curve(reference_paths[name], name) for name in ("epsilon_dec_1","geom_only")
    ]

    if references[0]["production_modes"] != references[1]["production_modes"]:
        raise ValueError(
            "The epsilon_dec=1 and geom-only JSON files "
            "list different production modes."
        )

    eventcalc_contour = load_eventcalc_contour()
    use_report_style()
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    draw_constraints(
        axis,
        PHOTON_CONSTRAINT_DIR,
        PHOTON_SPECS,
        model="alp_photon",
        context="event_density_overlay",
        config=load_label_config(),
    )

    draw_eventcalc_contour(axis, eventcalc_contour,)
    for reference in references:
        draw_reference_curve(axis, reference)

    axis.axvline(
        TABLE_LIMITS_GEV[MODEL_NAME],
        color="black",
        linewidth=1.5,
        linestyle=":",
        label="EventCalc table limit",
        zorder=22,
    )

    configure_axis(axis)
    figure.tight_layout()

    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight")

    plt.close(figure)

    summary = make_summary(references)
    summary.to_csv(SUMMARY_PATH, index=False)

    print()
    print("=" * 80)
    print("ALP-photon reference-curve comparison")
    print("=" * 80)
    print()
    print(
        summary[
            [
                "reference",
                "number_of_points",
                "minimum_mass_GeV",
                "maximum_mass_GeV",
                "minimum_coupling_GeV_inv",
                "maximum_coupling_GeV_inv",
            ]
        ].to_string(index=False)
    )

    print()
    print("Production modes in both reference files:")

    for mode in references[0]["production_modes"]:
        print(f"  - {mode}")

    print()
    print("Saved comparison plot to:")
    print(f"  {pdf_path}")
    print()
    print("Saved JSON metadata summary to:")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
