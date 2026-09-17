#!/usr/bin/env python3
"""Contracts for EventCalc's native five/six-body phase space."""

import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import NBodyDecay  # noqa: E402


class NBodyDecayTest(unittest.TestCase):
    def test_six_pion_records_are_on_shell_and_conserve_four_momentum(self):
        pdgs = [211, 211, -211, -211, 111, 111]
        masses = [0.13957, 0.13957, 0.13957, 0.13957, 0.13498, 0.13498]
        events = NBodyDecay.decay_products(
            1.69,
            pdgs,
            masses,
            [1, 1, -1, -1, 0, 0],
            [1] * 6,
            500,
            rng=np.random.default_rng(1729),
        )
        self.assertEqual(events.shape, (500, 48))
        particles = events.reshape(500, 6, 8)
        total = particles[:, :, :4].sum(axis=1)
        np.testing.assert_allclose(total[:, :3], 0.0, atol=2.0e-12)
        np.testing.assert_allclose(total[:, 3], 1.69, atol=2.0e-12)
        shell = (
            np.square(particles[:, :, 3])
            - np.sum(np.square(particles[:, :, :3]), axis=2)
        )
        np.testing.assert_allclose(
            shell,
            np.tile(np.square(np.asarray(masses)), (500, 1)),
            rtol=2.0e-11,
            atol=2.0e-12,
        )
        np.testing.assert_array_equal(
            particles[0, :, 5].astype(int), np.asarray(pdgs)
        )

    def test_isolated_seed_is_reproducible(self):
        args = (
            1.5,
            [211, -211, 111, 111, 111],
            [0.13957, 0.13957, 0.13498, 0.13498, 0.13498],
            [1, -1, 0, 0, 0],
            [1] * 5,
            12,
        )
        first = NBodyDecay.decay_products(
            *args, rng=np.random.default_rng(91)
        )
        np.random.seed(123456)
        np.random.random(100)
        second = NBodyDecay.decay_products(
            *args, rng=np.random.default_rng(91)
        )
        third = NBodyDecay.decay_products(
            *args, rng=np.random.default_rng(92)
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_invalid_inputs_fail_before_sampling(self):
        common = ([211] * 5, [0.13957] * 5, [1] * 5, [1] * 5)
        with self.assertRaisesRegex(ValueError, "below threshold"):
            NBodyDecay.decay_products(
                0.5, *common, 1, rng=np.random.default_rng(1)
            )
        with self.assertRaisesRegex(ValueError, "zero volume"):
            NBodyDecay.decay_products(
                5 * 0.13957, *common, 1, rng=np.random.default_rng(1)
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            NBodyDecay.decay_products(
                1.0, *common, -1, rng=np.random.default_rng(1)
            )
        with self.assertRaisesRegex(ValueError, "different lengths"):
            NBodyDecay.decay_products(
                1.0,
                common[0],
                common[1][:-1],
                common[2],
                common[3],
                1,
                rng=np.random.default_rng(1),
            )

    def test_zero_events_retains_particle_width(self):
        events = NBodyDecay.decay_products(
            1.0,
            [111] * 5,
            [0.13498] * 5,
            [0] * 5,
            [1] * 5,
            0,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(events.shape, (0, 40))


if __name__ == "__main__":
    unittest.main()
