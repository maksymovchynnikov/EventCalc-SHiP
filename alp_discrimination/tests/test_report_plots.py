import numpy as np

from alp_discrimination.plotting.report import (
    observable_token,
    pairwise_total_variation,
    selection_token,
)


def test_public_plot_tokens_are_physics_facing():
    assert selection_token("diphoton_ecal") == "ecal"
    assert selection_token("diphoton_ecal_e1gev") == "ecal_e1gev"
    assert observable_token("energy_mean_z_r_perp") == "energy_z_r_perp"


def test_pairwise_total_variation():
    photon = np.asarray([[0.75, 0.25], [0.50, 0.50]])
    su2 = np.asarray([[0.25, 0.75]])
    distance = pairwise_total_variation(photon, su2)
    assert distance.shape == (2, 1)
    assert np.allclose(distance[:, 0], [0.50, 0.25])
