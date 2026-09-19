#!/usr/bin/env python3
"""Production contracts for native explicit >4-body rows."""

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import decayProducts, exhadDecays  # noqa: E402


class ExhadExplicitMultibodyTest(unittest.TestCase):
    @staticmethod
    def _without_downstream_pythia(events, _mass, seed=None):
        del seed
        return decayProducts.convert_events_without_pythia(events)

    def tearDown(self):
        exhadDecays.set_selection(None, None)
        exhadDecays.set_force(None)

    def test_dark_photon_six_pion_rows_are_not_skipped(self):
        # DeLiVeR fractions at the 1.70-GeV boundary.  The 1.69-GeV parent
        # keeps this test below the partonic handoff while remaining above
        # both six-pion thresholds.
        six_charged = 0.009939603238068704
        six_neutral = 0.050806989623807884
        branching = np.asarray([
            1.0 - six_charged - six_neutral,
            six_charged,
            six_neutral,
        ])
        pdgs = np.asarray([
            [211, -211, -999, -999, -999, -999],
            [211, 211, 211, -211, -211, -211],
            [211, 211, -211, -211, 111, 111],
        ], dtype=int)
        n_events = 20000

        # The channel allocation is a multinomial draw from NumPy's global
        # stream, which a driver seeds once per scan point.  Do the same here
        # so the two backend choices are compared on one sample.
        allocation_seed = decayProducts.derive_exhad_seed(314159, 0, 0)

        def generate():
            with mock.patch.object(
                    decayProducts, "process_events_with_pythia",
                    side_effect=self._without_downstream_pythia
            ) as downstream:
                np.random.seed(allocation_seed)
                events, sizes = decayProducts.simulateDecays_rest_frame(
                    1.69, pdgs, branching, n_events,
                    [None, None, None], [0, 1, 2], 1.0,
                    llp_name="Dark-photons", seed=314159)
            return events, sizes, downstream.call_count

        exhadDecays.set_selection(
            "Dark-photons", os.path.join(ROOT, "Distributions", "Dark-photons"))
        exhadDecays.set_force(False)
        events, sizes, downstream_calls = generate()

        exhadDecays.set_force(True)
        matched_available = exhadDecays.exclusive_enabled(1.69)
        matched_events, matched_sizes, matched_downstream = generate()

        np.random.seed(allocation_seed)
        expected = decayProducts.distribute_events(n_events, branching)
        np.testing.assert_array_equal(sizes, expected)
        self.assertEqual(len(events), n_events)
        self.assertEqual(int(np.sum(sizes)), n_events)
        # Each count is binomial about its branching fraction; five standard
        # deviations bound how far a sample of this size can stray.
        for position, fraction in ((1, six_charged), (2, six_neutral)):
            spread = math.sqrt(fraction * (1.0 - fraction) / n_events)
            self.assertAlmostEqual(sizes[position] / n_events, fraction,
                                   delta=5.0 * spread)
        # exHad off sends the pi0-containing row downstream for the neutral
        # pion decays.
        self.assertEqual(downstream_calls, 1)

        particles = events.reshape(n_events, -1, 6)
        start_charged = int(sizes[0])
        stop_charged = start_charged + int(sizes[1])
        charged_ids = particles[start_charged:stop_charged, :, 5].astype(int)
        neutral_ids = particles[stop_charged:, :, 5].astype(int)
        self.assertTrue(np.all(np.sort(charged_ids, axis=1) ==
                               np.asarray([-211, -211, -211, 211, 211, 211])))
        self.assertTrue(np.all(np.sort(neutral_ids, axis=1) ==
                               np.asarray([-211, -211, 111, 111, 211, 211])))

        # Below the matched window the release's exclusive rows own the rows
        # it names, so the same allocation produces the measured exclusive
        # state instead of flat phase space: the charged-pion content of the
        # row survives, the neutral pions arrive already decayed, and no row
        # is skipped.
        if not matched_available:
            self.skipTest("no exHad release is configured, so the exclusive "
                          "window is closed and both modes sample flat phase "
                          "space")
        np.testing.assert_array_equal(matched_sizes, sizes)
        self.assertEqual(len(matched_events), n_events)
        self.assertEqual(matched_downstream, 0)
        matched_ids = matched_events.reshape(n_events, -1, 6)[:, :, 5].astype(int)
        for block, expected in ((matched_ids[start_charged:stop_charged], 3),
                                (matched_ids[stop_charged:], 2)):
            self.assertTrue(np.all((block == 211).sum(axis=1) == expected))
            self.assertTrue(np.all((block == -211).sum(axis=1) == expected))
            self.assertFalse(np.any(block == 111))

    def test_helper_is_seeded_and_independent_of_raw_matched_choice(self):
        args = (1.7, [211] * 3 + [-211] * 3,
                [0.13957] * 6, [1] * 3 + [-1] * 3, [1] * 6, 25)
        exhadDecays.set_force(False)
        raw = exhadDecays.generate_explicit_multibody_rest_frame(
            *args, seed=1729)
        exhadDecays.set_force(True)
        matched = exhadDecays.generate_explicit_multibody_rest_frame(
            *args, seed=1729)
        np.testing.assert_array_equal(raw, matched)

    def test_explicit_phase_space_does_not_load_exhad(self):
        with mock.patch.object(
                exhadDecays, "_release",
                side_effect=AssertionError("release must not be loaded")):
            events = exhadDecays.generate_explicit_multibody_rest_frame(
                1.7, [211] * 3 + [-211] * 3,
                [0.13957] * 6, [1] * 3 + [-1] * 3, [1] * 6, 3,
                seed=7)
        self.assertEqual(events.shape, (3, 48))


if __name__ == "__main__":
    unittest.main()
