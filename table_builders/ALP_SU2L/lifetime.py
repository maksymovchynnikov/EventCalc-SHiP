"""Photon and charged-lepton widths of a pure SU(2)_L ALP.
Model:
    c_W != 0,    c_B = 0,    c_{aPhi} = 0.

The input coupling is
    cW_over_fa = c_W / f_a    [GeV^-1].

Implemented sources:
M. B. Gavela et al., "Flavor constraints on electroweak ALP couplings",
arXiv:1901.02031v2:

* Gamma(a -> l+ l-): Eq. (9)
* c_ll at one loop: Eq. (10)
* Gamma(a -> gamma gamma): Eq. (11)
* c_{a gamma gamma}: Eqs. (12)-(13)
* B_2 and f loop functions: Eqs. (A2)-(A3)
"""

import numpy as np
import os
import json
from pathlib import Path

from .config import (
    MASSES_GEV,
    COUPLING_NORMALIZATION_GEV_INV,
    F_A_MATCHING_GEV,
    OUTPUT_ROOT,
)

from .constants import (
    ALPHA_EM,
    SIN2_THETA_W,
    M_W,
    LEPTON_MASSES,
    HBARC_GEV_M,
)


# Helpers
def _validate_mass(mass: float, name: str) -> float:
    """Return a finite positive mass as a Python float."""
    value = float(mass)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number in GeV.")
    return value

def _validate_fa(f_a: float) -> float:
    """Return a finite positive ALP matching scale in GeV."""
    value = float(f_a)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("f_a must be a finite positive number in GeV.")
    return value

def _validate_width(width: float, name: str = "width") -> float:
    """Return a finite non-negative decay width in GeV."""
    value = float(width)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number in GeV.")
    return value



def loop_f(tau: float) -> complex:
    """ Eq. (A3) """
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be a finite positive number.")

    if tau >= 1.0:
        return complex(np.arcsin(1.0 / np.sqrt(tau)))

    root = np.sqrt(1.0 - tau)
    return complex(
        np.pi / 2.0 + 0.5j * np.log((1.0 + root) / (1.0 - root))
    )


def B2(tau: float) -> complex:
    """Bosonic loop function B_2(tau), Eq. (A2)."""
    f_tau = loop_f(tau)
    return 1.0 - (float(tau) - 1.0) * f_tau**2



# a -> gamma gamma

def effective_photon_coupling(
    m_a: float,
    cW_over_fa: float,
) -> complex:
    """
    From Eqs. (12)-(13), after setting c_B = c_{aPhi} = 0:
        c_{a gamma gamma}/f_a
          = (c_W/f_a) [s_w^2 + (2 alpha_em/pi) B_2(tau_W)],
    where tau_W = 4 m_W^2/m_a^2.
    """
    m_a = _validate_mass(m_a, "m_a")
    coupling = float(cW_over_fa)
    if not np.isfinite(coupling):
        raise ValueError("cW_over_fa must be finite and given in GeV^-1.")

    tau_w = 4.0 * M_W**2 / m_a**2
    return coupling * (
        SIN2_THETA_W
        + (2.0 * ALPHA_EM / np.pi) * B2(tau_w)
    )


def gamma_a_to_gamma_gamma(
    m_a: float,
    cW_over_fa: float,
) -> float:
    """
    Partial width Gamma(a -> gamma gamma) in GeV.
    Eq. (11)
    """
    m_a = _validate_mass(m_a, "m_a")
    g_a_gamma_gamma = effective_photon_coupling(m_a, cW_over_fa)
    return float(m_a**3 * abs(g_a_gamma_gamma) ** 2 / (4.0 * np.pi))


# a -> l+ l-
def effective_lepton_coupling(
    lepton: str,
    cW_over_fa: float,
    f_a: float = F_A_MATCHING_GEV,
) -> float:
    """
    Return c_ll/f_a in GeV^-1 for l = e, mu, tau.
    Eq. (10), specialized to c_B = c_{aPhi} = 0:
    """
    if lepton not in LEPTON_MASSES:
        allowed = ", ".join(LEPTON_MASSES)
        raise ValueError(f"Unknown lepton {lepton!r}; choose one of: {allowed}.")

    f_a = _validate_fa(f_a)
    coupling = float(cW_over_fa)
    if not np.isfinite(coupling):
        raise ValueError("cW_over_fa must be finite and given in GeV^-1.")

    m_l = LEPTON_MASSES[lepton]

    uv_log_term = (
        9.0 * ALPHA_EM
        / (4.0 * np.pi * SIN2_THETA_W)
        * np.log(f_a / M_W)
    )
    infrared_log_term = (
        6.0 * ALPHA_EM
        / np.pi
        * SIN2_THETA_W
        * np.log(M_W / m_l)
    )

    return float(coupling * (uv_log_term + infrared_log_term))


def gamma_a_to_lepton_pair(
    m_a: float,
    lepton: str,
    cW_over_fa: float,
    f_a: float = F_A_MATCHING_GEV,
) -> float:
    """
    Partial width Gamma(a -> l+ l-) in GeV.
    Implements Eq. (9). The width is exactly zero at and below threshold m_a <= 2 m_l.
    """
    m_a = _validate_mass(m_a, "m_a")
    if lepton not in LEPTON_MASSES:
        allowed = ", ".join(LEPTON_MASSES)
        raise ValueError(f"Unknown lepton {lepton!r}; choose one of: {allowed}.")

    m_l = LEPTON_MASSES[lepton]
    if m_a <= 2.0 * m_l:
        return 0.0

    g_ll = effective_lepton_coupling(lepton, cW_over_fa, f_a)
    beta = np.sqrt(1.0 - 4.0 * m_l**2 / m_a**2)

    return float(abs(g_ll) ** 2 * m_a * m_l**2 / (8.0 * np.pi) * beta)


def photon_and_lepton_widths(
    m_a: float,
    cW_over_fa: float,
    f_a: float = F_A_MATCHING_GEV,
) -> dict[str, float]:
    """Return all implemented partial widths in GeV."""
    return {
        "gamma_gamma": gamma_a_to_gamma_gamma(m_a, cW_over_fa),
        "e_e": gamma_a_to_lepton_pair(m_a, "e", cW_over_fa, f_a),
        "mu_mu": gamma_a_to_lepton_pair(m_a, "mu", cW_over_fa, f_a),
        "tau_tau": gamma_a_to_lepton_pair(m_a, "tau", cW_over_fa, f_a),
    }


def gamma_without_hadrons(
    m_a: float,
    cW_over_fa: float,
    f_a: float = F_A_MATCHING_GEV,
) -> float:
    """Sum of the photon and charged-lepton widths in GeV."""
    return float(sum(photon_and_lepton_widths(m_a, cW_over_fa, f_a).values()))


def gamma_total(
    m_a: float,
    cW_over_fa: float,
    gamma_hadronic: float,
    f_a: float = F_A_MATCHING_GEV,
) -> float:
    """Complete width after an externally computed hadronic width is supplied.

    Parameters
    ----------
    gamma_hadronic:
        Hadronic partial width Gamma(a -> hadrons) in GeV. Until the hadronic
        model is implemented, use ``gamma_without_hadrons`` rather than passing
        zero and calling the result the physical total width.
    """
    gamma_hadronic = _validate_width(gamma_hadronic, "gamma_hadronic")
    return gamma_without_hadrons(m_a, cW_over_fa, f_a) + gamma_hadronic



# Lifetime conversion
def proper_decay_length_m(total_width: float) -> float:
    """Convert a total width in GeV to the proper decay length c*tau in metres."""
    total_width = _validate_width(total_width, "total_width")
    if total_width == 0.0:
        return np.inf
    return HBARC_GEV_M / total_width


def lifetime_seconds(total_width: float) -> float:
    """Convert a total width in GeV to the proper lifetime tau in seconds."""
    total_width = _validate_width(total_width, "total_width")
    if total_width == 0.0:
        return np.inf

    hbar_gev_s = 6.582119569e-25
    return hbar_gev_s / total_width


def make_lifetime_table(
    masses=MASSES_GEV,
    c_w_over_f_a=COUPLING_NORMALIZATION_GEV_INV,
    f_a_matching=F_A_MATCHING_GEV,
):
    rows = []

    for m_a in masses:
        width = gamma_without_hadrons(
            m_a=m_a,
            cW_over_fa=c_w_over_f_a,
            f_a=f_a_matching,
        )

        if width > 0.0:
            ctau = HBARC_GEV_M / width
        else:
            ctau = np.inf

        rows.append([
            m_a,
            ctau,
        ])

    return np.asarray(rows)

def write_lifetime_table(
    output_path=None,
    masses=MASSES_GEV,
    c_w_over_f_a=COUPLING_NORMALIZATION_GEV_INV,
    f_a_matching=F_A_MATCHING_GEV,
):
    if output_path is None:
        output_path = os.path.join(
            OUTPUT_ROOT,
            "ctau-ALP-SU2L.txt",
        )

    table = make_lifetime_table(
        masses=masses,
        c_w_over_f_a=c_w_over_f_a,
        f_a_matching=f_a_matching,
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

def make_decay_json_data(
    masses=MASSES_GEV,
    c_w_over_f_a=COUPLING_NORMALIZATION_GEV_INV,
    f_a_matching=F_A_MATCHING_GEV,
):
    """Build decay-channel data in the exact format expected by EventCalc."""

    channel_specs = [
        {
            "name": "2gamma",
            "pdgs": [22, 22, -999, -999],
            "width_key": "gamma_gamma",
        },
        {
            "name": "ep_em",
            "pdgs": [-11, 11, -999, -999],
            "width_key": "e_e",
        },
        {
            "name": "mup_mum",
            "pdgs": [-13, 13, -999, -999],
            "width_key": "mu_mu",
        },
        {
            "name": "taup_taum",
            "pdgs": [-15, 15, -999, -999],
            "width_key": "tau_tau",
        },
    ]

    br_tables = {
        specification["width_key"]: []
        for specification in channel_specs
    }

    for mass in masses:
        mass = float(mass)

        widths = photon_and_lepton_widths(
            m_a=mass,
            cW_over_fa=c_w_over_f_a,
            f_a=f_a_matching,
        )

        total_width = float(sum(widths.values()))

        if total_width <= 0.0:
            raise RuntimeError(
                f"The total implemented width is zero at m_a = {mass} GeV."
            )

        branching_ratios = {
            key: float(width / total_width)
            for key, width in widths.items()
        }

        br_sum = sum(branching_ratios.values())

        if not np.isclose(br_sum, 1.0, rtol=1e-12, atol=1e-14):
            raise RuntimeError(
                f"Branching ratios sum to {br_sum} at m_a = {mass} GeV."
            )

        for specification in channel_specs:
            key = specification["width_key"]

            br_tables[key].append([
                mass,
                branching_ratios[key],
            ])

    decay_data = []

    for specification in channel_specs:
        decay_data.append([
            specification["name"],
            specification["pdgs"],
            br_tables[specification["width_key"]],
            "1.",
        ])

    return decay_data


def write_decay_json(
    output_path=None,
    masses=MASSES_GEV,
    c_w_over_f_a=COUPLING_NORMALIZATION_GEV_INV,
    f_a_matching=F_A_MATCHING_GEV,
):
    """Write the EventCalc-compatible ALP decay JSON."""

    if output_path is None:
        output_path = (
            Path(OUTPUT_ROOT)
            / "ALP-SU2L-decay.json"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decay_data = make_decay_json_data(
        masses=masses,
        c_w_over_f_a=c_w_over_f_a,
        f_a_matching=f_a_matching,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            decay_data,
            file,
            indent="\t",
        )
        file.write("\n")

    print(f"Decay JSON written to {output_path}")

    return output_path
