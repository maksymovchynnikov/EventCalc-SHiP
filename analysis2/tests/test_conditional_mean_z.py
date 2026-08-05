"""Focused tests for the conditional-mean-z backend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis2 import conditional_mean_z as cmz
from analysis2.workflows.conditional_mean_z_decision_audit import (
    decision_relevant_audit,
)
from analysis2.workflows.conditional_mean_z_selected import (
    persistent_threshold,
)


def test_profiled_scores_shapes_and_finiteness():
    sampled_bins = np.asarray([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=int)
    event_counts = np.asarray([2, 4], dtype=int)
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    means = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    variances = np.ones_like(means)
    observed = np.asarray([[10.0, 15.0], [40.0, 35.0]], dtype=float)

    energy, combined = cmz.profiled_scores(
        sampled_bins=sampled_bins,
        observed_mean_z=observed,
        probabilities=probabilities,
        conditional_mean_z=means,
        conditional_variance_z=variances,
        event_counts=event_counts,
    )
    assert energy.shape == (2, 2)
    assert combined.shape == (2, 2)
    assert np.all(np.isfinite(combined))


def test_stable_z_rng_is_reproducible():
    first = cmz.stable_z_rng(
        seed=73241,
        mass_gev=0.5,
        truth_model="su2",
        truth_index=7,
    ).normal(size=8)
    second = cmz.stable_z_rng(
        seed=73241,
        mass_gev=0.5,
        truth_model="su2",
        truth_index=7,
    ).normal(size=8)
    np.testing.assert_array_equal(first, second)


def test_persistent_threshold_handles_local_dip():
    curve = pd.DataFrame(
        {
            "number_of_events": [38, 39, 40, 41],
            "worst_case_accuracy": [0.91, 0.89, 0.901, 0.907],
        }
    )
    assert persistent_threshold(curve) == 40


def test_decision_audit_checks_target_not_selected_envelope():
    rows = []
    for seed in (1, 2):
        for n, accuracy in ((40, 0.99), (100, 0.995)):
            rows.append(
                {
                    "truth_model": "photon",
                    "truth_lifetime_index": 0,
                    "truth_ctau_m": 1.0,
                    "seed": seed,
                    "number_of_events": n,
                    "number_of_pseudoexperiments": 2000,
                    "correct_fraction": accuracy,
                    "observable": "conditional_combined",
                }
            )
    detailed = pd.DataFrame(rows)
    audit, promotions, summary = decision_relevant_audit(
        detailed_2k=detailed,
        selected_keys=set(),
        candidate_threshold=40,
        tested_event_counts=np.asarray([40, 100]),
        target_accuracy=0.9,
        global_alpha=0.01,
    )
    assert not audit.empty
    assert promotions.empty
    assert summary["number_of_failing_rows"] == 0
