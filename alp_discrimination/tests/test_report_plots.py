import numpy as np
import pandas as pd

from alp_discrimination.plotting.report import (
    observable_token,
    pairwise_total_variation,
    plot_headline_observable_comparison,
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



def test_headline_observable_comparison_is_written(tmp_path):
    curves = {
        name: pd.DataFrame(
            {
                "number_of_events": [2, 4, 10],
                "accuracy": [0.75, 0.91, 0.98],
            }
        )
        for name in (
            "energy",
            "energy_mean_z",
            "energy_mean_r_perp",
            "energy_mean_z_r_perp",
        )
    }
    n90 = {
        "energy": 142,
        "energy_mean_z": 36,
        "energy_mean_r_perp": 5,
        "energy_mean_z_r_perp": 4,
    }

    plot_headline_observable_comparison(
        curves=curves,
        n90=n90,
        output_dir=tmp_path,
    )

    assert (
        tmp_path
        / "classification_observable_comparison_ma0p3_report.pdf"
    ).is_file()
    assert (
        tmp_path
        / "classification_observable_comparison_ma0p3_report.png"
    ).is_file()
