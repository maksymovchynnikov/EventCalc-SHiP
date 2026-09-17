#!/usr/bin/env python3
"""Regression tests for three-body matrix-element weighting."""

import os
import sys
import unittest
import importlib.util

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

try:
    spec = importlib.util.spec_from_file_location(
        "funcs._three_body_decay_real",
        os.path.join(ROOT, "funcs", "ThreeBodyDecay.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    block_random_energies = module.block_random_energies
    matrix_element_upper_bound = module.matrix_element_upper_bound
except ModuleNotFoundError:
    block_random_energies = None

from funcs import sampling_bounds  # noqa: E402


def constant(mass, energy1, energy3, mass1=0.0, mass2=0.0, mass3=0.0):
    return 1.0


def narrow_peak(mass, energy1, energy3, mass1=0.0, mass2=0.0, mass3=0.0):
    """A resonance too narrow for a pilot sample of a few thousand points."""
    return 1.0 + 2.0e4 * 1.0e-12 / ((energy1 - 0.55) ** 2 + 1.0e-12)


@unittest.skipIf(block_random_energies is None,
                 "EventCalc's numba runtime is not installed")
class ThreeBodyDecayTest(unittest.TestCase):
    def test_constant_matrix_element_is_broadcast_over_phase_space(self):
        energies = block_random_energies(
            1.91, 0.49761, 0.49761, 0.13498, 40, constant, 130, 310, 111,
        )
        self.assertEqual(energies.shape, (40, 2))
        self.assertTrue(np.isfinite(energies).all())

    def test_every_returned_event_is_a_distinct_draw(self):
        energies = block_random_energies(
            1.91, 0.49761, 0.49761, 0.13498, 5000, constant, 130, 310, 111,
        )
        self.assertEqual(len(np.unique(energies, axis=0)), 5000)

    def test_sampled_energies_stay_inside_the_allowed_region(self):
        energies = block_random_energies(
            1.5, 0.13957, 0.13957, 0.13498, 5000, narrow_peak, 211, -211, 111,
        )
        inside = sampling_bounds.dalitz_region_mask(
            1.5, 0.13957, 0.13957, 0.13498, energies[:, 0], energies[:, 1])
        self.assertTrue(bool(inside.all()))

    def test_no_weight_can_exceed_the_upper_bound(self):
        """The bound is worked out before sampling and covers a narrow peak
        that a trial sample would miss."""
        upper_bound = matrix_element_upper_bound(
            1.5, 0.13957, 0.13957, 0.13498, narrow_peak, 211, -211, 111)
        self.assertGreaterEqual(upper_bound.global_ceiling,
                                narrow_peak(1.5, np.array([0.55]), None)[0])
        energy1, energy3, ceiling = upper_bound.draw_candidates(200000)
        inside = sampling_bounds.dalitz_region_mask(
            1.5, 0.13957, 0.13957, 0.13498, energy1, energy3)
        weight = np.zeros(len(energy1))
        weight[inside] = narrow_peak(1.5, energy1[inside], energy3[inside])
        self.assertTrue(bool(np.all(weight <= ceiling)))

    def test_four_body_bound_is_never_reached_by_a_sampled_weight(self):
        parent, masses = 1.5, (0.13957, 0.13957, 0.13957, 0.13957)
        bound = sampling_bounds.four_body_weight_bound(parent, *masses)
        rng = np.random.default_rng(5)
        mass234 = rng.uniform(sum(masses[1:]), parent - masses[0], 400000)
        mass34 = rng.uniform(masses[2] + masses[3], mass234 - masses[1])

        def momentum(a, b, c):
            radicand = (a ** 2 - (b + c) ** 2) * (a ** 2 - (b - c) ** 2)
            return np.sqrt(np.maximum(radicand, 0.0)) / (2.0 * a)

        weight = ((mass234 - masses[1] - masses[2] - masses[3]) *
                  momentum(parent, masses[0], mass234) *
                  momentum(mass234, masses[1], mass34) *
                  momentum(mass34, masses[2], masses[3]))
        self.assertTrue(bool(np.all(weight <= bound)))

    def test_chain_bound_is_never_reached_by_a_sampled_weight(self):
        from funcs import NBodyDecay
        parent = 1.69
        masses = np.array([0.13957] * 4 + [0.13498] * 2)
        bound = sampling_bounds.chain_weight_bound(parent, masses)
        _intermediate, weight = NBodyDecay._intermediate_masses(
            parent, masses, 400000, np.random.default_rng(6))
        self.assertTrue(bool(np.all(weight <= bound)))


if __name__ == "__main__":
    unittest.main()
