"""Diphoton width and lifetime of the pure SU(2)_L ALP benchmark.

The model uses c_W != 0 with c_B = c_{aPhi} = 0 and
cW_over_fa = c_W / f_a in GeV^-1. The generated EventCalc decay table retains
only a -> gamma gamma with branching ratio one. The electroweak expressions are
converted from arXiv:1901.02031 to the coupling convention used here.
"""

import numpy as np
import os
import json
from pathlib import Path

from .config import (
    MASSES_GEV,
    COUPLING_NORMALIZATION_GEV_INV,
    OUTPUT_ROOT,
)

from .constants import (
    HBARC_GEV_M,
    PHOTON_OPERATOR_FACTOR,
)


# Helpers
def _validate_mass(mass: float, name: str) -> float:
    """Return a finite positive mass as a Python float."""
    value = float(mass)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number in GeV.")
    return value


def _validate_width(width: float, name: str = "width") -> float:
    """Return a finite non-negative decay width in GeV."""
    value = float(width)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number in GeV.")
    return value


# a -> gamma gamma
def gamma_a_to_gamma_gamma(
    m_a: float,
    c_w_over_f_a: float,
) -> float:
    """Gamma(a -> gamma gamma) for the coupling convention used here."""

    m_a = _validate_mass(m_a, "m_a")
    c_w_over_f_a = float(c_w_over_f_a)

    if not np.isfinite(c_w_over_f_a):
        raise ValueError("c_w_over_f_a must be finite and given in GeV^-1.")

    effective_photon_coupling = PHOTON_OPERATOR_FACTOR * c_w_over_f_a

    return float(effective_photon_coupling**2 * m_a**3 / (4.0 * np.pi))


# Lifetime conversion
def proper_decay_length_m(total_width: float) -> float:
    """Convert a total width in GeV to the proper decay length c*tau in metres."""
    total_width = _validate_width(total_width, "total_width")
    if total_width == 0.0:
        return np.inf
    return HBARC_GEV_M / total_width


def make_lifetime_table(
    masses=MASSES_GEV,
    c_w_over_f_a=COUPLING_NORMALIZATION_GEV_INV,
):
    rows = []

    for m_a in masses:
        width = gamma_a_to_gamma_gamma(
            m_a=m_a,
            c_w_over_f_a=c_w_over_f_a,
        )

        ctau = proper_decay_length_m(width)
        rows.append([m_a, ctau])

    return np.asarray(rows)


def write_lifetime_table(
    output_path=None,
    masses=MASSES_GEV,
    c_w_over_f_a=COUPLING_NORMALIZATION_GEV_INV,
):
    if output_path is None:
        output_path = os.path.join(
            OUTPUT_ROOT,
            "ctau-ALP-SU2L.txt",
        )

    table = make_lifetime_table(
        masses=masses,
        c_w_over_f_a=c_w_over_f_a,
    )

    folder = os.path.dirname(output_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    np.savetxt(
        output_path,
        table,
        fmt="%.8e",
        delimiter="\t",
    )

    print(f"Lifetime table written to {output_path}")

    return output_path


def make_decay_json_data():
    return [
        [
            "2gamma",
            [22, 22, -999, -999],
            1.0,
            "1.",
        ]
    ]


def write_decay_json(
    output_path=None,
):
    """Write the EventCalc-compatible ALP decay JSON."""

    if output_path is None:
        output_path = Path(OUTPUT_ROOT) / "ALP-SU2L-decay.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decay_data = make_decay_json_data()

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            decay_data,
            file,
            indent="\t",
        )
        file.write("\n")

    print(f"Decay JSON written to {output_path}")

    return output_path
