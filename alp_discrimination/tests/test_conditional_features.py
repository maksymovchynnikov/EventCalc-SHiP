"""Focused tests for richer conditional event features."""

from __future__ import annotations

import numpy as np

from alp_discrimination.conditional_features import (
    FEATURE_LABELS,
    FEATURE_SUBSETS,
    R_SCALE_M,
    SelectedFeatureSample,
    Z_LENGTH_M,
    Z_MIN_M,
    gaussian_bhattacharyya_coefficient,
    joint_energy_feature_hellinger_squared,
    pairwise_joint_energy_feature_hellinger_squared,
    profiled_feature_scores,
    regularize_covariance,
    stable_feature_rng,
    weighted_feature_moments_by_energy_bin,
)


def test_master_feature_definition_includes_r_perp():
    sample = SelectedFeatureSample(
        energy_gev=np.asarray([1.0, 2.0]),
        z_m=np.asarray([Z_MIN_M, Z_MIN_M + Z_LENGTH_M]),
        r_perp_m=np.asarray([0.0, R_SCALE_M]),
        weights=np.asarray([1.0, 1.0]),
    )
    np.testing.assert_allclose(
        sample.master_features,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )


def test_weighted_feature_moments_match_direct_calculation():
    sample = SelectedFeatureSample(
        energy_gev=np.asarray([1.2, 1.4, 2.2, 2.4]),
        z_m=Z_MIN_M + Z_LENGTH_M * np.asarray([0.1, 0.3, 0.5, 0.9]),
        r_perp_m=R_SCALE_M * np.asarray([0.2, 0.4, 0.1, 0.7]),
        weights=np.asarray([1.0, 3.0, 2.0, 2.0]),
    )
    result = weighted_feature_moments_by_energy_bin(
        sample=sample,
        energy_edges_gev=np.asarray([1.0, 2.0, 3.0]),
    )
    first_features = sample.master_features[:2]
    first_weights = sample.weights[:2]
    expected_mean = np.average(
        first_features,
        axis=0,
        weights=first_weights,
    )
    np.testing.assert_allclose(result["mean"][0], expected_mean)
    assert np.all(np.linalg.eigvalsh(result["covariance"]) > 0.0)


def test_covariance_regularization_is_positive_definite():
    singular = np.zeros((2, 3, 3), dtype=float)
    regularized = regularize_covariance(singular)
    assert np.all(np.linalg.eigvalsh(regularized) > 0.0)


def test_bhattacharyya_and_hellinger_identity():
    mean = np.asarray([0.2, 0.4])
    covariance = np.asarray([[0.1, 0.02], [0.02, 0.2]])
    assert np.isclose(
        gaussian_bhattacharyya_coefficient(
            mean,
            covariance,
            mean,
            covariance,
        ),
        1.0,
    )
    probabilities = np.asarray([0.4, 0.6])
    means = np.asarray([[0.2, 0.4, 0.1], [0.5, 0.2, 0.3]])
    covariances = np.repeat(np.eye(3)[None, :, :] * 0.1, 2, axis=0)
    distance = joint_energy_feature_hellinger_squared(
        first_probabilities=probabilities,
        first_means=means,
        first_covariances=covariances,
        second_probabilities=probabilities,
        second_means=means,
        second_covariances=covariances,
        feature_indices=FEATURE_SUBSETS["energy_mean_z_r_perp"],
    )
    assert np.isclose(distance, 0.0, atol=1.0e-12)


def test_profiled_energy_only_ignores_empty_feature_vector():
    sampled_bins = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=int)
    probabilities = np.asarray([[0.7, 0.3], [0.3, 0.7]], dtype=float)
    means = np.zeros((2, 2, 3), dtype=float)
    covariances = np.repeat(
        np.eye(3)[None, None, :, :] * 0.1,
        4,
        axis=0,
    ).reshape(2, 2, 3, 3)
    energy, combined = profiled_feature_scores(
        sampled_bins=sampled_bins,
        observed_feature_means=np.empty((2, 2, 0)),
        probabilities=probabilities,
        conditional_feature_mean=means,
        conditional_feature_covariance=covariances,
        event_counts=np.asarray([1, 3]),
        feature_indices=(),
    )
    np.testing.assert_allclose(energy, combined)


def test_stable_feature_rng_is_reproducible():
    first = stable_feature_rng(
        seed=73241,
        mass_gev=0.5,
        truth_model="su2",
        truth_index=12,
    ).normal(size=10)
    second = stable_feature_rng(
        seed=73241,
        mass_gev=0.5,
        truth_model="su2",
        truth_index=12,
    ).normal(size=10)
    np.testing.assert_array_equal(first, second)



def test_pairwise_hellinger_matches_scalar_implementation():
    photon_probabilities = np.asarray([[0.4, 0.6], [0.7, 0.3]])
    su2_probabilities = np.asarray([[0.5, 0.5], [0.2, 0.8]])
    photon_means = np.asarray(
        [
            [[0.2, 0.1, 0.3], [0.4, 0.2, 0.5]],
            [[0.3, 0.1, 0.2], [0.6, 0.4, 0.7]],
        ]
    )
    su2_means = photon_means + 0.05
    photon_covariances = np.repeat(
        np.eye(3)[None, None, :, :] * 0.1,
        4,
        axis=0,
    ).reshape(2, 2, 3, 3)
    su2_covariances = photon_covariances * 1.2
    indices = FEATURE_SUBSETS["energy_mean_z_r_perp"]
    pairwise = pairwise_joint_energy_feature_hellinger_squared(
        photon_probabilities=photon_probabilities,
        photon_means=photon_means,
        photon_covariances=photon_covariances,
        su2_probabilities=su2_probabilities,
        su2_means=su2_means,
        su2_covariances=su2_covariances,
        feature_indices=indices,
    )
    scalar = np.empty((2, 2))
    for photon_index in range(2):
        for su2_index in range(2):
            scalar[photon_index, su2_index] = (
                joint_energy_feature_hellinger_squared(
                    first_probabilities=photon_probabilities[photon_index],
                    first_means=photon_means[photon_index],
                    first_covariances=photon_covariances[photon_index],
                    second_probabilities=su2_probabilities[su2_index],
                    second_means=su2_means[su2_index],
                    second_covariances=su2_covariances[su2_index],
                    feature_indices=indices,
                )
            )
    np.testing.assert_allclose(pairwise, scalar, rtol=1.0e-12, atol=1.0e-12)


def test_normalized_mean_z_reproduces_scalar_mean_z_profile():
    from alp_discrimination.conditional_mean_z import profiled_scores as scalar_scores

    sampled_bins = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=int)
    event_counts = np.asarray([1, 3], dtype=int)
    probabilities = np.asarray([[0.7, 0.3], [0.3, 0.7]], dtype=float)
    mean_z = np.asarray([[40.0, 60.0], [45.0, 65.0]], dtype=float)
    variance_z = np.asarray([[9.0, 16.0], [4.0, 25.0]], dtype=float)
    observed_z = np.asarray([[41.0, 50.0], [62.0, 54.0]], dtype=float)

    _, scalar_combined = scalar_scores(
        sampled_bins=sampled_bins,
        observed_mean_z=observed_z,
        probabilities=probabilities,
        conditional_mean_z=mean_z,
        conditional_variance_z=variance_z,
        event_counts=event_counts,
    )

    means = np.zeros((2, 2, 3), dtype=float)
    covariances = np.repeat(
        np.eye(3)[None, None, :, :] * 0.1,
        4,
        axis=0,
    ).reshape(2, 2, 3, 3)
    means[..., 0] = (mean_z - Z_MIN_M) / Z_LENGTH_M
    covariances[..., 0, 0] = variance_z / (Z_LENGTH_M**2)
    observed_u = ((observed_z - Z_MIN_M) / Z_LENGTH_M)[..., None]
    _, vector_combined = profiled_feature_scores(
        sampled_bins=sampled_bins,
        observed_feature_means=observed_u,
        probabilities=probabilities,
        conditional_feature_mean=means,
        conditional_feature_covariance=covariances,
        event_counts=event_counts,
        feature_indices=(0,),
    )
    difference = vector_combined - scalar_combined
    np.testing.assert_allclose(
        difference,
        np.full_like(difference, np.log(Z_LENGTH_M)),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_transverse_geometry_scale_is_finite():
    assert np.isfinite(R_SCALE_M)
    assert R_SCALE_M > 0.0



def test_all_feature_labels_render_with_matplotlib():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    for row, label in enumerate(FEATURE_LABELS.values()):
        axis.text(0.05, float(row), label)
    figure.canvas.draw()
    plt.close(figure)



def test_all_truths_from_bank_covers_both_models():
    from alp_discrimination.workflows.conditional_feature_scan import (
        all_truths_from_bank,
    )

    class Bank:
        photon_ctau_m = np.asarray([1.0, 2.0, 3.0])
        su2_ctau_m = np.asarray([4.0, 5.0])
        photon_interval_index = np.asarray([0, 0, 1])
        su2_interval_index = np.asarray([0, 1])

    table, selected = all_truths_from_bank(Bank())
    np.testing.assert_array_equal(
        selected["photon"],
        np.asarray([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        selected["su2"],
        np.asarray([0, 1]),
    )
    assert len(table) == 5
    assert set(table["selection_reasons"]) == {
        "full_domain_truth_grid"
    }



def test_callable_conditional_feature_runner_forwards_settings(
    monkeypatch,
    tmp_path,
):
    from alp_discrimination.workflows import conditional_feature_scan as pilot

    captured = {}

    def fake_runner(args):
        captured.update(vars(args))
        return {"status": "ok"}

    monkeypatch.setattr(
        pilot,
        "_run_conditional_feature_args",
        fake_runner,
    )

    result = pilot.run_conditional_feature_point(
        bank_path=tmp_path / "bank.npz",
        output_dir=tmp_path / "output",
        pseudoexperiments=123,
        seeds=(11, 22),
        workers=1,
        chunk_size=7,
        event_counts=(3, 4, 5),
        observables=("energy", "energy_mean_z_r_perp"),
        pairs_per_interval=2,
        truth_grid="all",
        neighbour_radius=2,
        reuse_moments=True,
    )

    assert result == {"status": "ok"}
    assert captured["bank_path"] == tmp_path / "bank.npz"
    assert captured["output_dir"] == tmp_path / "output"
    assert captured["pseudoexperiments"] == 123
    assert captured["seeds"] == [11, 22]
    assert captured["workers"] == 1
    assert captured["chunk_size"] == 7
    assert captured["event_counts"] == [3, 4, 5]
    assert captured["observables"] == [
        "energy",
        "energy_mean_z_r_perp",
    ]
    assert captured["pairs_per_interval"] == 2
    assert captured["truth_grid"] == "all"
    assert captured["neighbour_radius"] == 2
    assert captured["reuse_moments"] is True
