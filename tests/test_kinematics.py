"""Regression tests for ALP two-body kinematics and Lorentz boosts."""

import numpy as np
import pytest

from table_builders.ALP_SU2L import kinematics as alp_kinematics
from table_builders.ALP_SU2L.channels_file import PDG_ALP, PDG_DUMMY_RECOIL
from table_builders.ALP_SU2L.constants import (
    BPLUS_TO_XA_CHANNELS,
    M_B_PLUS,
)
from table_builders.ALP_SU2L.distribution import theta_energy_from_momenta


TEST_CHANNEL = BPLUS_TO_XA_CHANNELS[0]
TEST_SAMPLE_SIZE = 32
RNG_SEED = 12345


def invariant_mass_squared(four_momenta):
    return four_momenta[:, 3] ** 2 - np.sum(four_momenta[:, :3] ** 2, axis=1)


def simulate_single_channel(monkeypatch, alp_mass, size=TEST_SAMPLE_SIZE):
    monkeypatch.setattr(
        alp_kinematics,
        "rng",
        np.random.default_rng(RNG_SEED),
    )
    return alp_kinematics.simulate_B_to_Xa_rest_frame_fast(
        size=size,
        alp_mass=alp_mass,
        probabilities_by_name={TEST_CHANNEL["name"]: 1.0},
        channels=[TEST_CHANNEL],
    )


def make_on_shell_mother_momenta(size):
    parameter = np.linspace(-1.0, 1.0, size)
    spatial_momenta = np.column_stack(
        (
            3.0 * parameter,
            2.0 * np.sin(np.pi * parameter),
            10.0 * parameter,
        )
    )
    spatial_momenta[0] = 0.0
    energies = np.sqrt(
        np.sum(spatial_momenta**2, axis=1) + M_B_PLUS**2
    )
    return np.column_stack((spatial_momenta, energies))


@pytest.mark.parametrize("alp_mass", [0.05, 1.0, 4.5])
def test_two_body_generation_and_boost_preserve_four_momentum(
    alp_mass,
    monkeypatch,
):
    alp_rest = simulate_single_channel(monkeypatch, alp_mass)
    recoil_mass = TEST_CHANNEL["mass"]

    assert alp_rest.shape == (TEST_SAMPLE_SIZE, 6)
    assert alp_rest.dtype == np.float64
    assert np.all(np.isfinite(alp_rest))
    assert np.all(alp_rest[:, 4] == alp_mass)
    assert np.all(alp_rest[:, 5] == PDG_ALP)

    expected_alp_energy = (
        M_B_PLUS**2 + alp_mass**2 - recoil_mass**2
    ) / (2.0 * M_B_PLUS)
    np.testing.assert_allclose(
        alp_rest[:, 3],
        expected_alp_energy,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        invariant_mass_squared(alp_rest),
        alp_mass**2,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    recoil_rest = np.empty_like(alp_rest)
    recoil_rest[:, :3] = -alp_rest[:, :3]
    recoil_rest[:, 3] = M_B_PLUS - alp_rest[:, 3]
    recoil_rest[:, 4] = recoil_mass
    recoil_rest[:, 5] = PDG_DUMMY_RECOIL

    np.testing.assert_allclose(
        invariant_mass_squared(recoil_rest),
        recoil_mass**2,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    recoil_energy_from_mass_shell = np.sqrt(
        np.sum(recoil_rest[:, :3] ** 2, axis=1) + recoil_mass**2
    )
    np.testing.assert_allclose(
        alp_rest[:, 3] + recoil_energy_from_mass_shell,
        M_B_PLUS,
        rtol=0.0,
        atol=1.0e-13,
    )

    mother_lab = make_on_shell_mother_momenta(TEST_SAMPLE_SIZE)
    alp_lab = alp_kinematics.boost_alp_rest_to_lab_fast(
        M_B_PLUS,
        mother_lab,
        alp_rest,
    )
    recoil_lab = alp_kinematics.boost_alp_rest_to_lab_fast(
        M_B_PLUS,
        mother_lab,
        recoil_rest,
    )

    assert np.all(np.isfinite(alp_lab))
    assert np.all(np.isfinite(recoil_lab))
    np.testing.assert_allclose(
        invariant_mass_squared(alp_lab),
        alp_mass**2,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        invariant_mass_squared(recoil_lab),
        recoil_mass**2,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        alp_lab[:, :4] + recoil_lab[:, :4],
        mother_lab,
        rtol=1.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_array_equal(alp_lab[:, 4:], alp_rest[:, 4:])
    np.testing.assert_array_equal(recoil_lab[:, 4:], recoil_rest[:, 4:])


def test_stationary_mother_boost_is_identity():
    alp_mass = 0.7
    spatial_momenta = np.asarray(
        [
            [0.3, -0.4, 0.5],
            [-1.0, 0.2, 0.8],
        ]
    )
    energies = np.sqrt(np.sum(spatial_momenta**2, axis=1) + alp_mass**2)
    alp_rest = np.column_stack(
        (
            spatial_momenta,
            energies,
            np.full(len(energies), alp_mass),
            np.full(len(energies), PDG_ALP),
        )
    )
    mother_at_rest = np.tile(
        np.asarray([0.0, 0.0, 0.0, M_B_PLUS]),
        (len(alp_rest), 1),
    )

    alp_lab = alp_kinematics.boost_alp_rest_to_lab_fast(
        M_B_PLUS,
        mother_at_rest,
        alp_rest,
    )

    np.testing.assert_array_equal(alp_lab, alp_rest)


def test_z_directed_boost_matches_analytic_lorentz_transform():
    beta = 0.6
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    alp_mass = 0.7
    spatial_momentum = np.asarray([0.3, -0.4, 1.2])
    energy = np.sqrt(np.sum(spatial_momentum**2) + alp_mass**2)
    alp_rest = np.asarray(
        [[*spatial_momentum, energy, alp_mass, PDG_ALP]],
        dtype=np.float64,
    )
    mother_lab = np.asarray(
        [[0.0, 0.0, gamma * beta * M_B_PLUS, gamma * M_B_PLUS]]
    )

    alp_lab = alp_kinematics.boost_alp_rest_to_lab_fast(
        M_B_PLUS,
        mother_lab,
        alp_rest,
    )

    expected = alp_rest.copy()
    expected[0, 2] = gamma * (spatial_momentum[2] + beta * energy)
    expected[0, 3] = gamma * (energy + beta * spatial_momentum[2])
    np.testing.assert_allclose(alp_lab, expected, rtol=1.0e-14, atol=1.0e-14)


def test_two_body_channel_is_closed_at_and_above_threshold(monkeypatch):
    threshold = M_B_PLUS - TEST_CHANNEL["mass"]

    for closed_mass in (threshold, threshold * (1.0 + 1.0e-8)):
        with pytest.raises(ValueError, match="No kinematically allowed channels"):
            simulate_single_channel(monkeypatch, closed_mass, size=4)

    open_mass = threshold * (1.0 - 1.0e-8)
    alp_rest = simulate_single_channel(monkeypatch, open_mass, size=4)
    momentum_squared = np.sum(alp_rest[:, :3] ** 2, axis=1)
    assert np.all(np.isfinite(momentum_squared))
    assert np.all(momentum_squared > 0.0)


def test_simulation_rejects_zero_total_channel_probability(monkeypatch):
    monkeypatch.setattr(
        alp_kinematics,
        "rng",
        np.random.default_rng(RNG_SEED),
    )

    with pytest.raises(ValueError, match="Total channel probability is zero"):
        alp_kinematics.simulate_B_to_Xa_rest_frame_fast(
            size=4,
            alp_mass=1.0,
            probabilities_by_name={TEST_CHANNEL["name"]: 0.0},
            channels=[TEST_CHANNEL],
        )


@pytest.mark.parametrize(
    "invalid_input",
    ["mother_shape", "alp_shape", "event_count"],
)
def test_boost_rejects_incompatible_array_shapes(invalid_input):
    mother_lab = np.tile(
        np.asarray([0.0, 0.0, 0.0, M_B_PLUS]),
        (2, 1),
    )
    alp_rest = np.zeros((2, 6))

    if invalid_input == "mother_shape":
        mother_lab = mother_lab[:, :3]
        message = "B_momenta must have shape"
    elif invalid_input == "alp_shape":
        alp_rest = alp_rest[:, :5]
        message = "alp_rest must have shape"
    else:
        mother_lab = mother_lab[:1]
        message = "must contain the same number of events"

    with pytest.raises(ValueError, match=message):
        alp_kinematics.boost_alp_rest_to_lab_fast(
            M_B_PLUS,
            mother_lab,
            alp_rest,
        )


def test_theta_energy_conversion_uses_px_py_pz_E_convention():
    momenta = np.asarray(
        [
            [0.0, 0.0, 2.0, 3.0, 0.0, PDG_ALP],
            [2.0, 0.0, 0.0, 4.0, 0.0, PDG_ALP],
            [0.0, 1.0, -1.0, 5.0, 0.0, PDG_ALP],
            [0.0, 0.0, -2.0, 6.0, 0.0, PDG_ALP],
        ]
    )

    theta, energy = theta_energy_from_momenta(momenta)

    np.testing.assert_allclose(
        theta,
        [0.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(energy, momenta[:, 3])
