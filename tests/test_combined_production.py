#!/usr/bin/env python3
"""Contracts for yield-weighted primary-plus-cascade production."""

import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import initLLP, kinematics  # noqa: E402


def _toy_distribution(scale):
    rows = []
    for mass in (1.0, 2.0):
        for theta in (0.001, 0.01):
            for energy in (1.5, 3.5):
                rows.append((mass, theta, energy, float(scale)))
    return pd.DataFrame(rows)


def _toy_maximum_energy():
    return pd.DataFrame([
        (mass, theta, 3.5)
        for mass in (1.0, 2.0)
        for theta in (0.001, 0.01)
    ])


class CombinedProductionTest(unittest.TestCase):
    def test_mixture_uses_absolute_yield_and_polar_acceptance_weights(self):
        mixture = kinematics.ProductionMixture([
            ("primary", _toy_distribution(1.0), _toy_maximum_energy(),
             lambda _mass: 1.0),
            ("cascade", _toy_distribution(2.0), _toy_maximum_energy(),
             lambda _mass: 3.0),
        ])
        np.random.seed(12345)
        grids = kinematics.Grids(
            mixture, None, 2000, 1.25, 100.0, cache_token=("toy", 1))
        grids.interpolate()
        grids.resample(400)

        summary = grids.production_component_summary
        self.assertEqual(
            sum(item["sample_points"] for item in summary.values()), 400)
        expected = (
            summary["primary"]["yield"]
            * summary["primary"]["epsilon_polar"]
            + summary["cascade"]["yield"]
            * summary["cascade"]["epsilon_polar"]
        ) / 4.0
        self.assertAlmostEqual(grids.epsilon_polar, expected, places=14)
        self.assertGreater(
            summary["cascade"]["sample_points"],
            summary["primary"]["sample_points"])

        grids.true_samples()
        self.assertEqual(grids.get_kinematics().shape[1], 10)
        self.assertEqual(grids.get_momentum().shape[1], 4)

    def test_production_aliases_have_one_canonical_identity(self):
        for alias in ("combined", "primary+cascade", "cascade+primary"):
            self.assertEqual(
                initLLP._normalize_alp_production_mode(alias), "combined")
            self.assertEqual(
                initLLP._normalize_dp_production_mode(alias), "combined")
        self.assertEqual(
            initLLP._normalize_alp_production_mode("cascade"), "cascades")

    def test_alp_combined_yield_is_primary_plus_cascade(self):
        selection = {
            "particle_path": os.path.join(ROOT, "Distributions", "ALP-photon"),
            "LLP_name": "ALP-photon",
        }
        combined = initLLP.LLP(
            None, selection, alp_production_mode="combined")
        mass = 0.1
        component_yields = combined.Distr.component_yields(mass)
        self.assertEqual(
            combined.production_component_labels, ("primary", "cascades"))
        self.assertAlmostEqual(
            combined.get_total_yield(mass),
            float(np.sum(component_yields)),
            places=15)

    def test_dark_photon_combined_yield_is_primary_plus_full_cascade(self):
        selection = {
            "particle_path": os.path.join(
                ROOT, "Distributions", "Dark-photons"),
            "LLP_name": "Dark-photons",
        }
        combined = initLLP.LLP(
            None, selection, uncertainty="central",
            dp_production_mode="combined")
        mass = 1.8
        component_yields = combined.Distr.component_yields(mass)
        self.assertEqual(
            combined.production_component_labels, ("primary", "cascade"))
        self.assertEqual(combined.uncertainty_label, "combined-central")
        self.assertAlmostEqual(
            combined.get_total_yield(mass),
            float(np.sum(component_yields)),
            places=15)


if __name__ == "__main__":
    unittest.main()
