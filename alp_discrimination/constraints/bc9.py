"""Load and draw the combined BC9 photophilic-ALP exclusions.

The supplied files are closed exclusion polygons in the plane

    mass [GeV], coupling [GeV^-1].

They combine the relevant constraints into two polygons:

* laboratory exclusions;
* astrophysical exclusions.

This module is deliberately separate from the older per-experiment
FORESEE polygons.  The same loader will later be reused when constructing
the Week-8 unexcluded lifetime domains.
"""

from __future__ import annotations

from csv import DictReader
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BC9_DIRECTORY = Path(__file__).with_name("bc9")

BC9_FILENAMES = {
    "astrophysical": "Constraints_BC9_astro.csv",
    "laboratory": "Constraints_BC9_lab.csv",
}


def _parse_positive_number(value: str, *, location: str) -> float:
    """Parse either a decimal/scientific number or a rational such as 1/5000."""
    text = value.strip()

    try:
        number = (
            float(Fraction(text))
            if "/" in text
            else float(text)
        )
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"Could not parse {location}: {value!r}"
        ) from error

    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(
            f"{location} must be finite and positive, found {number!r}"
        )

    return number


def load_bc9_polygon(path: Path) -> np.ndarray:
    """Load one BC9 exclusion polygon as an array of shape (N, 2)."""
    if not path.exists():
        raise FileNotFoundError(
            f"BC9 constraint file not found: {path}"
        )

    rows: list[tuple[float, float]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = DictReader(stream)

        if (
            reader.fieldnames is None
            or set(reader.fieldnames) != {"mass", "coupling"}
        ):
            raise ValueError(
                f"{path} must contain exactly the columns "
                "'mass' and 'coupling'; found {reader.fieldnames!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            mass = _parse_positive_number(
                row["mass"],
                location=f"{path.name}:{line_number}:mass",
            )
            coupling = _parse_positive_number(
                row["coupling"],
                location=f"{path.name}:{line_number}:coupling",
            )
            rows.append((mass, coupling))

    data = np.asarray(rows, dtype=float)

    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 3:
        raise ValueError(
            f"{path} does not contain a valid polygon"
        )

    return data


def load_bc9_constraints(
    directory: Path = BC9_DIRECTORY,
) -> dict[str, np.ndarray]:
    """Load both combined BC9 exclusion polygons."""
    return {
        name: load_bc9_polygon(directory / filename)
        for name, filename in BC9_FILENAMES.items()
    }


def draw_bc9_constraints(
    axis: plt.Axes,
    directory: Path = BC9_DIRECTORY,
) -> None:
    """Draw the union of the BC9 laboratory and astrophysical exclusions."""
    polygons = load_bc9_constraints(directory)

    styles = {
        "astrophysical": {
            "facecolor": "0.88",
            "zorder": -210,
        },
        "laboratory": {
            "facecolor": "0.76",
            "zorder": -200,
        },
    }

    for index, name in enumerate(
        ("astrophysical", "laboratory")
    ):
        data = polygons[name]
        style = styles[name]

        axis.fill(
            data[:, 0],
            data[:, 1],
            facecolor=style["facecolor"],
            edgecolor="0.35",
            linewidth=0.8,
            label=(
                "Existing exclusions"
                if index == 0
                else "_nolegend_"
            ),
            zorder=style["zorder"],
        )