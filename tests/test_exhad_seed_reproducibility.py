#!/usr/bin/env python3
"""Focused regression tests for EventCalc/exhad RNG stream propagation."""

import contextlib
import io
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The test replaces the decay kernel before using it; lightweight stubs keep
# this RNG regression independent of EventCalc's optional numba installation.
for module_name in ("TwoBodyDecay", "ThreeBodyDecay", "FourBodyDecay"):
    module = types.ModuleType("funcs." + module_name)
    module.decay_products = lambda *_args, **_kwargs: None
    sys.modules.setdefault("funcs." + module_name, module)

from funcs import decayProducts, exhadDecays  # noqa: E402

try:
    import exhad_bridge  # noqa: E402
except ImportError:
    exhad_bridge = None

NO_BRIDGE = (
    "exhad_bridge belongs to the exHad installation and is not part of this "
    "repository; put the directory that contains it on PYTHONPATH to run "
    "this test"
)


class ExhadSeedPropagationTest(unittest.TestCase):
    def test_seed_coordinates_are_repeatable_and_distinct(self):
        coords = [(23, batch, channel)
                  for batch in range(3) for channel in range(4)]
        first = [decayProducts.derive_exhad_seed(*coord) for coord in coords]
        second = [decayProducts.derive_exhad_seed(*coord) for coord in coords]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(1 <= seed <= decayProducts._EXHAD_SEED_LIMIT
                            for seed in first))

    def test_simulator_passes_one_deterministic_seed_per_aggregated_batch(self):
        seen = []

        def fake_two_body(_mass, size, *_args):
            return np.zeros((size, 16), dtype=np.float64)

        def fake_exhad(n_events, _mass, seed=1):
            seen.append(seed)
            out = np.zeros((n_events, 6), dtype=np.float64)
            out[:, 5] = 22.0
            return out

        pdg_rows = np.array([[1, -1], [2, -2]], dtype=np.int64)
        # The routing predicates the merged entry point consults are pinned so
        # the fixed-mass pool is the route under test: a partonic row inside
        # the matched support, above the exclusive window, on a backend that
        # does not own named EventCalc rows.
        patches = (
            mock.patch.object(decayProducts.TwoBodyDecay, "decay_products",
                              side_effect=fake_two_body),
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench", return_value=False),
            mock.patch.object(exhadDecays, "exclusive_enabled",
                              return_value=False),
            mock.patch.object(exhadDecays, "uses_portable_model1_backend",
                              return_value=False),
            mock.patch.object(exhadDecays, "fixed_partonic_row_supported",
                              return_value=True),
            mock.patch.object(exhadDecays, "enabled", return_value=True),
            mock.patch.object(exhadDecays, "process_events_with_exhad",
                              side_effect=fake_exhad),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            with contextlib.redirect_stdout(io.StringIO()):
                for batch in (7, 8):
                    decayProducts.simulateDecays_rest_frame(
                        2.0, pdg_rows, [0.5, 0.5], 4, None, [0, 1], 1.0,
                        seed=23, batch_index=batch)

        # Both partonic rows are aggregated, keyed by the first original row.
        expected = [decayProducts.derive_exhad_seed(23, batch, 0)
                    for batch in (7, 8)]
        self.assertEqual(seen, expected)
        self.assertEqual(len(seen), len(set(seen)))


class EventCalcDecayRngTest(unittest.TestCase):
    @unittest.skipIf(exhad_bridge is None, NO_BRIDGE)
    def test_pi0_decay_rng_is_per_call_and_seeded(self):
        pi0 = np.array([0.0, 0.0, 0.0, exhad_bridge._MPI0,
                        exhad_bridge._MPI0, 111.0, 0.0, 1.0])

        def fixed_pi0_events(_bench, _mass, n, seed=1, residual=False):
            del seed, residual
            return [pi0.copy() for _ in range(n)]

        with mock.patch.object(exhad_bridge, "sample_hadronic_decays",
                               side_effect=fixed_pi0_events):
            first = np.asarray(exhad_bridge.sample_for_eventcalc(
                "alp", 2.0, 1, seed=31415))
            exhad_bridge.sample_for_eventcalc("alp", 2.0, 1, seed=999)
            repeated = np.asarray(exhad_bridge.sample_for_eventcalc(
                "alp", 2.0, 1, seed=31415))
            different = np.asarray(exhad_bridge.sample_for_eventcalc(
                "alp", 2.0, 1, seed=31416))

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))
        particles = first.reshape(-1, 6)
        np.testing.assert_allclose(particles[:, :3].sum(axis=0), 0.0,
                                   atol=1e-15)
        self.assertAlmostEqual(float(particles[:, 3].sum()),
                               exhad_bridge._MPI0, places=14)
        np.testing.assert_array_equal(particles[:, 5], [22.0, 22.0])


if __name__ == "__main__":
    unittest.main()
