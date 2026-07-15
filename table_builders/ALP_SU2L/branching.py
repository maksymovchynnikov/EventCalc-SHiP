# Br(B+ -> K+ a), Br(B+ -> X_s a), Pprod

import numpy as np
from .constants import (
    M_B_PLUS,
    M_K_PLUS,
    M_PI_PLUS,
    TAU_B_PLUS_GEV_INV,
    M_W,
    G2_EW_SQUARED,
    M_U,
    M_C,
    M_T,
    LAMBDA_CKM,
    A_CKM,
    RHOBAR,
    ETABAR,
)

def make_ckm_matrix():
    """
    Construct CKM matrix from Wolfenstein-like input.

    Source:
        PDG Review of Particle Physics, CKM standard parameterization.
        URL: https://pdg.lbl.gov/
    """
    lam = LAMBDA_CKM
    A = A_CKM

    rho = RHOBAR / (1.0 - lam**2 / 2.0) # ONLY leading correction here
    eta = ETABAR / (1.0 - lam**2 / 2.0) # ONLY leading correction here

    s12 = lam
    s23 = A * lam**2
    s13 = A * lam**3 * np.sqrt(rho**2 + eta**2)
    delta = np.arctan2(eta, rho)

    c12 = np.sqrt(1.0 - s12**2)
    c23 = np.sqrt(1.0 - s23**2)
    c13 = np.sqrt(1.0 - s13**2)

    return {
        "ud": c12 * c13,
        "us": s12 * c13,
        "ub": s13 * np.exp(-1j * delta),

        "cd": -s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta),
        "cs": c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta),
        "cb": s23 * c13,

        "td": s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta),
        "ts": -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta),
        "tb": c23 * c13,
    }


CKM = make_ckm_matrix()


def g_function(x):
    """
    Loop function:
        g(x) = x [1 + x (log x - 1)] / (1 - x)^2
        arXiv:1901.02031v2, Eq. (7).
        Top-quark must dominate here
    """
    x = np.asarray(x, dtype=float)

    return np.where(
        np.isclose(x, 1.0), # limit = 0.5 to avoid 0/0 at x=1
        0.5,
        x * (1.0 + x * (np.log(x) - 1.0)) / (1.0 - x)**2
    )


def lambda_two_body_sqrt(m_parent, m1, m2):
    """
        Same structure appears in arXiv:1901.02031v2 Eq. (8)
    """
    if m2 >= m_parent - m1:
        return 0.0

    term_plus = 1.0 - ((m1 + m2) / m_parent)**2
    term_minus = 1.0 - ((m1 - m2) / m_parent)**2

    return np.sqrt(max(0.0, term_plus * term_minus))


def f0_B_to_K(q2):
    """
    Scalar form factor f_0^{B -> K}(q^2).

    Parametrization:
        f0(q2) = F0 / (1 - q2 / mfit^2)

    Numerical values:
        F0 = 0.33
        mfit = 6.16 GeV

    Source:
        arXiv:1904.10447v4, Appendix F.1.1 / Table of form-factor inputs
        for B -> K scalar/pseudoscalar transitions.
    """
    F0_BK = 0.33
    MFIT_BK = 6.16

    return F0_BK / (1.0 - q2 / MFIT_BK**2)


def f0_B_to_pi(q2):
    """
    Scalar form factor f_0^{B -> pi}(q^2).
    Sources: arXiv:1904.10447v4, Appendix F.1.1, Eq. (F.11), Table 8.
        Original form-factor calculation:
        P. Ball and R. Zwicky,
        Phys. Rev. D 71, 014015 (2005),
        hep-ph/0406232.
    """
    F0_BPI = 0.258 #+- 0.031
    MFIT_BPI = 6.16

    q2 = float(q2)

    return F0_BPI / (1.0 - q2 / MFIT_BPI**2)

def ckm_loop_sum_b_to_q(final_quark):
    if final_quark not in {"s", "d"}:
        raise ValueError("final_quark must be 's' or 'd'.")

    quark_masses = {
        "u": M_U,
        "c": M_C,
        "t": M_T,
    }

    total = 0.0 + 0.0j

    for up_quark in ["u", "c", "t"]:
        v_qb = CKM[f"{up_quark}b"]
        v_qq = CKM[f"{up_quark}{final_quark}"]
        x_q = (quark_masses[up_quark] / M_W) ** 2

        total += v_qb * np.conjugate(v_qq) * g_function(x_q)

    return total

def br_Bplus_to_Pplus_a(
    alp_mass,
    cW_over_fa,
    meson_mass,
    form_factor,
    final_quark,
):
    ma = float(alp_mass)

    if ma <= 0.0 or ma >= M_B_PLUS - meson_mass:
        return 0.0

    loop_sum = ckm_loop_sum_b_to_q(final_quark)

    effective_prefactor = (
        3.0 * G2_EW_SQUARED
        / (16.0 * np.pi**2)
        * loop_sum
        * cW_over_fa
    )

    lambda_sqrt = lambda_two_body_sqrt(
        M_B_PLUS,
        meson_mass,
        ma,
    )

    width = (
        M_B_PLUS**3
        / (64.0 * np.pi)
        * abs(effective_prefactor)**2
        * form_factor(ma**2)**2
        * lambda_sqrt
        * (1.0 - meson_mass**2 / M_B_PLUS**2)**2
    )

    return TAU_B_PLUS_GEV_INV * width


def load_scalar_br_table(path):
    """
    Load scalar-reference branching ratios Br(B+ -> X S) / theta^2.

    Source:
        Br-ratios-scalar.csv, based on scalar B -> X_s S results from
        arXiv:1904.10447v4.
    """
    return np.genfromtxt(path, delimiter=",", names=True)


def scalar_br_over_theta2(alp_mass, column, scalar_table):
    """
    Interpolate scalar Br(B+ -> X S) / theta^2 at m_S = m_a.

    No extrapolation is allowed.
    """
    masses = scalar_table["m_S_GeV"]

    if alp_mass < masses[0] or alp_mass > masses[-1]:
        raise ValueError(
            f"m_a = {alp_mass} GeV is outside scalar table range "
            f"[{masses[0]}, {masses[-1]}] GeV."
        )

    return np.interp(alp_mass, masses, scalar_table[column])


def get_Bplus_to_Xa_branching_ratios(
    alp_mass,
    cW_over_fa,
    scalar_table_path,
    channels,
    scalar_table=None,
):
    if scalar_table is None:
        scalar_table = load_scalar_br_table(scalar_table_path)

    available_columns = scalar_table.dtype.names

    # Directly calculated reference channels.
    br_Ka = br_Bplus_to_Pplus_a(
        alp_mass=alp_mass,
        cW_over_fa=cW_over_fa,
        meson_mass=M_K_PLUS,
        form_factor=f0_B_to_K,
        final_quark="s",
    )

    br_pia = br_Bplus_to_Pplus_a(
        alp_mass=alp_mass,
        cW_over_fa=cW_over_fa,
        meson_mass=M_PI_PLUS,
        form_factor=f0_B_to_pi,
        final_quark="d",
    )

    # The scalar K reference is only needed while B -> K a is open.
    br_KS = None

    if br_Ka > 0.0:
        br_KS = scalar_br_over_theta2(
            alp_mass,
            "K",
            scalar_table,
        )

    channel_brs = {}

    for channel in channels:
        name = channel["name"]
        recoil_mass = channel["mass"]

        # Kinematically closed channel.
        if alp_mass >= M_B_PLUS - recoil_mass:
            br_i = 0.0

        # The pion channel is calculated directly using b -> d.
        elif name == "pi+":
            br_i = br_pia

        # All strange resonance channels are normalized from B -> K a.
        elif br_Ka > 0.0 and br_KS is not None and br_KS > 0.0:
            column = channel["scalar_csv_column"]

            if column not in available_columns:
                raise KeyError(
                    f"CSV column {column!r} for channel {name!r} "
                    f"not found. Available columns: {available_columns}"
                )

            br_XS = scalar_br_over_theta2(
                alp_mass,
                column,
                scalar_table,
            )

            br_i = br_Ka * br_XS / br_KS

        else:
            br_i = 0.0

        channel_brs[name] = float(br_i)

    total_br = float(sum(channel_brs.values()))

    if total_br > 0.0:
        probabilities = {
            name: br_i / total_br
            for name, br_i in channel_brs.items()
        }
    else:
        probabilities = {
            name: 0.0
            for name in channel_brs
        }

    return br_Ka, channel_brs, probabilities, total_br