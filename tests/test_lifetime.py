"""Regression tests for the pure SU(2)_L ALP lifetime implementation."""

import numpy as np
import pytest

from table_builders.ALP_SU2L.config import F_A_MATCHING_GEV, MASSES_GEV
from table_builders.ALP_SU2L.constants import LEPTON_MASSES
from table_builders.ALP_SU2L.lifetime import (
    gamma_a_to_lepton_pair,
    gamma_without_hadrons,
    photon_and_lepton_widths,
    proper_decay_length_m,
)


REFERENCE_COUPLING_GEV_INV = 1.0e-4
COUPLING_SCALE_FACTOR = 3.7


@pytest.mark.parametrize(
    "coupling",
    [-REFERENCE_COUPLING_GEV_INV, 0.0, REFERENCE_COUPLING_GEV_INV],
)
def test_all_implemented_partial_widths_are_finite_and_non_negative(coupling):
    for mass in MASSES_GEV:
        widths = photon_and_lepton_widths(
            m_a=mass,
            cW_over_fa=coupling,
            f_a=F_A_MATCHING_GEV,
        )
        values = np.asarray(list(widths.values()))

        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)


@pytest.mark.parametrize("lepton, lepton_mass", LEPTON_MASSES.items())
def test_lepton_width_obeys_pair_production_threshold(lepton, lepton_mass):
    threshold = 2.0 * lepton_mass

    for closed_mass in (0.99 * threshold, threshold):
        width = gamma_a_to_lepton_pair(
            m_a=closed_mass,
            lepton=lepton,
            cW_over_fa=REFERENCE_COUPLING_GEV_INV,
            f_a=F_A_MATCHING_GEV,
        )
        assert width == 0.0

    open_width = gamma_a_to_lepton_pair(
        m_a=threshold * (1.0 + 1.0e-8),
        lepton=lepton,
        cW_over_fa=REFERENCE_COUPLING_GEV_INV,
        f_a=F_A_MATCHING_GEV,
    )
    assert np.isfinite(open_width)
    assert open_width > 0.0


def test_widths_scale_with_coupling_squared():
    mass = 4.0  # GeV; above the e, mu, and tau pair thresholds.
    scaled_coupling = COUPLING_SCALE_FACTOR * REFERENCE_COUPLING_GEV_INV

    reference_widths = photon_and_lepton_widths(
        mass,
        REFERENCE_COUPLING_GEV_INV,
        F_A_MATCHING_GEV,
    )
    scaled_widths = photon_and_lepton_widths(
        mass,
        scaled_coupling,
        F_A_MATCHING_GEV,
    )

    for channel, reference_width in reference_widths.items():
        np.testing.assert_allclose(
            scaled_widths[channel],
            COUPLING_SCALE_FACTOR**2 * reference_width,
            rtol=1.0e-12,
            atol=0.0,
        )

    np.testing.assert_allclose(
        gamma_without_hadrons(mass, scaled_coupling, F_A_MATCHING_GEV),
        COUPLING_SCALE_FACTOR**2
        * gamma_without_hadrons(
            mass,
            REFERENCE_COUPLING_GEV_INV,
            F_A_MATCHING_GEV,
        ),
        rtol=1.0e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("mass", [0.1, 0.5, 4.0])
def test_proper_decay_length_scales_with_inverse_coupling_squared(mass):
    reference_width = gamma_without_hadrons(
        mass,
        REFERENCE_COUPLING_GEV_INV,
        F_A_MATCHING_GEV,
    )
    scaled_width = gamma_without_hadrons(
        mass,
        COUPLING_SCALE_FACTOR * REFERENCE_COUPLING_GEV_INV,
        F_A_MATCHING_GEV,
    )

    reference_ctau = proper_decay_length_m(reference_width)
    scaled_ctau = proper_decay_length_m(scaled_width)

    np.testing.assert_allclose(
        scaled_ctau,
        reference_ctau / COUPLING_SCALE_FACTOR**2,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_zero_width_has_infinite_proper_decay_length():
    assert np.isinf(proper_decay_length_m(0.0))
