"""Regression tests for ALP production and decay branching ratios."""

import numpy as np
import pytest

from table_builders.ALP_SU2L.branching import (
    br_Bplus_to_Pplus_a,
    f0_B_to_K,
    f0_B_to_pi,
    get_Bplus_to_Xa_branching_ratios,
    lambda_two_body_sqrt,
    load_scalar_br_table,
)
from table_builders.ALP_SU2L.config import (
    MASSES_GEV,
    SCALAR_TABLE_PATH,
)
from table_builders.ALP_SU2L.constants import (
    BPLUS_TO_XA_CHANNELS,
    F_BPLUS,
    F_BZERO,
    M_B_PLUS,
    M_K_PLUS,
    M_PI_PLUS,
    N_BB_PER_POT,
)
from table_builders.ALP_SU2L.lifetime import make_decay_json_data
from table_builders.ALP_SU2L.production import (
    production_probability_Bplus_reference,
)


REFERENCE_COUPLING_GEV_INV = 1.0e-4
COUPLING_SCALE_FACTOR = -3.7


@pytest.fixture(scope="module")
def scalar_table():
    return load_scalar_br_table(SCALAR_TABLE_PATH)


def get_branching_ratios(mass, coupling, scalar_table):
    return get_Bplus_to_Xa_branching_ratios(
        alp_mass=mass,
        cW_over_fa=coupling,
        scalar_table_path=SCALAR_TABLE_PATH,
        channels=BPLUS_TO_XA_CHANNELS,
        scalar_table=scalar_table,
    )


def test_scalar_reference_table_is_finite_non_negative_and_complete(scalar_table):
    expected_columns = {"m_S_GeV"}
    expected_columns.update(
        channel["scalar_csv_column"] for channel in BPLUS_TO_XA_CHANNELS
    )

    assert expected_columns <= set(scalar_table.dtype.names)
    assert np.all(np.diff(scalar_table["m_S_GeV"]) > 0.0)

    for column in scalar_table.dtype.names:
        values = scalar_table[column]
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)


@pytest.mark.parametrize("mass", MASSES_GEV, ids=lambda mass: f"m_a={mass:g}")
def test_production_branching_ratios_are_non_negative_and_normalized(
    mass,
    scalar_table,
):
    br_Ka, channel_brs, probabilities, total_br = get_branching_ratios(
        mass,
        REFERENCE_COUPLING_GEV_INV,
        scalar_table,
    )
    br_values = np.asarray(list(channel_brs.values()))
    probability_values = np.asarray(list(probabilities.values()))

    assert np.isfinite(br_Ka)
    assert br_Ka >= 0.0
    assert np.all(np.isfinite(br_values))
    assert np.all(br_values >= 0.0)
    assert np.all(np.isfinite(probability_values))
    assert np.all(probability_values >= 0.0)
    np.testing.assert_allclose(total_br, np.sum(br_values), rtol=1.0e-14, atol=0.0)
    np.testing.assert_allclose(channel_brs["K+"], br_Ka, rtol=1.0e-12, atol=0.0)

    for channel in BPLUS_TO_XA_CHANNELS:
        name = channel["name"]
        if mass >= M_B_PLUS - channel["mass"]:
            assert channel_brs[name] == 0.0
            assert probabilities[name] == 0.0

    if total_br > 0.0:
        np.testing.assert_allclose(
            np.sum(probability_values),
            1.0,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        for name, branching_ratio in channel_brs.items():
            np.testing.assert_allclose(
                probabilities[name],
                branching_ratio / total_br,
                rtol=1.0e-12,
                atol=0.0,
            )
    else:
        assert np.all(br_values == 0.0)
        assert np.all(probability_values == 0.0)


@pytest.mark.parametrize(
    "meson_mass, form_factor, final_quark",
    [
        pytest.param(M_K_PLUS, f0_B_to_K, "s", id="K+"),
        pytest.param(M_PI_PLUS, f0_B_to_pi, "d", id="pi+"),
    ],
)
def test_direct_branching_ratio_obeys_two_body_threshold(
    meson_mass,
    form_factor,
    final_quark,
):
    threshold = M_B_PLUS - meson_mass
    open_mass = threshold * (1.0 - 1.0e-8)

    assert lambda_two_body_sqrt(M_B_PLUS, meson_mass, open_mass) > 0.0
    open_br = br_Bplus_to_Pplus_a(
        alp_mass=open_mass,
        cW_over_fa=REFERENCE_COUPLING_GEV_INV,
        meson_mass=meson_mass,
        form_factor=form_factor,
        final_quark=final_quark,
    )
    assert np.isfinite(open_br)
    assert open_br > 0.0

    for closed_mass in (threshold, threshold * (1.0 + 1.0e-8)):
        assert lambda_two_body_sqrt(M_B_PLUS, meson_mass, closed_mass) == 0.0
        assert (
            br_Bplus_to_Pplus_a(
                alp_mass=closed_mass,
                cW_over_fa=REFERENCE_COUPLING_GEV_INV,
                meson_mass=meson_mass,
                form_factor=form_factor,
                final_quark=final_quark,
            )
            == 0.0
        )


@pytest.mark.parametrize(
    "channel",
    BPLUS_TO_XA_CHANNELS,
    ids=lambda channel: channel["name"],
)
def test_each_configured_channel_is_zero_at_threshold(channel, scalar_table):
    threshold = M_B_PLUS - channel["mass"]
    _, channel_brs, probabilities, _ = get_branching_ratios(
        threshold,
        REFERENCE_COUPLING_GEV_INV,
        scalar_table,
    )

    assert channel_brs[channel["name"]] == 0.0
    assert probabilities[channel["name"]] == 0.0


def test_branching_ratios_and_production_yield_scale_with_coupling_squared(
    scalar_table,
):
    mass = 1.0  # GeV; safely inside every configured channel threshold.
    scaled_coupling = COUPLING_SCALE_FACTOR * REFERENCE_COUPLING_GEV_INV
    scale_squared = COUPLING_SCALE_FACTOR**2

    reference_br_Ka, reference_brs, reference_probs, reference_total = (
        get_branching_ratios(mass, REFERENCE_COUPLING_GEV_INV, scalar_table)
    )
    scaled_br_Ka, scaled_brs, scaled_probs, scaled_total = get_branching_ratios(
        mass,
        scaled_coupling,
        scalar_table,
    )

    np.testing.assert_allclose(
        scaled_br_Ka,
        scale_squared * reference_br_Ka,
        rtol=1.0e-12,
        atol=0.0,
    )
    for name, reference_br in reference_brs.items():
        np.testing.assert_allclose(
            scaled_brs[name],
            scale_squared * reference_br,
            rtol=1.0e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            scaled_probs[name],
            reference_probs[name],
            rtol=1.0e-12,
            atol=0.0,
        )
    np.testing.assert_allclose(
        scaled_total,
        scale_squared * reference_total,
        rtol=1.0e-12,
        atol=0.0,
    )

    reference_yield = production_probability_Bplus_reference(
        N_bb_per_POT=N_BB_PER_POT,
        f_b_to_Bplus=F_BPLUS,
        f_b_to_B0=F_BZERO,
        BR_Bplus_to_Xa_total=reference_total,
    )
    scaled_yield = production_probability_Bplus_reference(
        N_bb_per_POT=N_BB_PER_POT,
        f_b_to_Bplus=F_BPLUS,
        f_b_to_B0=F_BZERO,
        BR_Bplus_to_Xa_total=scaled_total,
    )
    np.testing.assert_allclose(
        scaled_yield,
        scale_squared * reference_yield,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_zero_coupling_returns_zero_production_probabilities(scalar_table):
    br_Ka, channel_brs, probabilities, total_br = get_branching_ratios(
        1.0,
        0.0,
        scalar_table,
    )

    assert br_Ka == 0.0
    assert total_br == 0.0
    assert all(branching_ratio == 0.0 for branching_ratio in channel_brs.values())
    assert all(probability == 0.0 for probability in probabilities.values())
