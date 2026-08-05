"""Tests for empirical conditional-feature resampling."""

import numpy as np

from analysis2.workflows.conditional_feature_empirical_validation import (
    draw_empirical_feature_rows,
)


def test_empirical_draw_preserves_complete_feature_rows():
    sampled_bins = np.asarray([[0, 0, 1, 1]], dtype=int)
    uniforms = np.asarray([[0.1, 0.9, 0.2, 0.8]], dtype=float)
    rows = [
        np.asarray([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]]),
        np.asarray([[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]]),
    ]
    cdf = [
        np.asarray([0.5, 1.0]),
        np.asarray([0.5, 1.0]),
    ]
    sampled = draw_empirical_feature_rows(
        sampled_bins=sampled_bins,
        uniforms=uniforms,
        feature_rows=rows,
        cumulative_weights=cdf,
    )
    expected = np.asarray(
        [
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
            ]
        ]
    )
    np.testing.assert_array_equal(sampled, expected)
