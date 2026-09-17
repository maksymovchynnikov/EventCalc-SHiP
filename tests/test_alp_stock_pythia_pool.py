#!/usr/bin/env python3
"""Regression tests for the complete ALP stock-Pythia comparator."""

import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import decayProducts  # noqa: E402


def _two_body_rows(size, _mass1, _mass2, pdg1, pdg2):
    rows = np.zeros((int(size), 16), dtype=np.float64)
    rows[:, 5] = int(pdg1)
    rows[:, 13] = int(pdg2)
    rows[:, 7] = 1
    rows[:, 15] = 1
    return rows


class AlpStockPythiaPoolTest(unittest.TestCase):
    def setUp(self):
        self.pdgs = np.asarray(
            [
                [-13, 13],
                [21, 21],
                [3, -3],
                [4, -4],
                [2212, -2212],
                [2112, -2112],
            ],
            dtype=float,
        )
        self.ratios = np.asarray([0.10, 0.50, 0.20, 0.0, 0.10, 0.10])
        self.selected = list(range(len(self.pdgs)))
        # The channel allocation is a multinomial draw from NumPy's global
        # stream.  Fixing the stream makes the counts these tests build their
        # pools from the counts the simulator draws for the same sample.
        self.allocation_seed = 4242
        np.random.seed(self.allocation_seed)
        self.counts = decayProducts.distribute_events(100, self.ratios)

    def test_complete_pool_uses_only_partonic_pythia_inputs(self):
        captured = {}

        def fake_two_body(
            mass, size, mass1, mass2, pdg1, pdg2,
            charge1, charge2, stability1, stability2,
        ):
            del mass, charge1, charge2, stability1, stability2
            return _two_body_rows(size, mass1, mass2, pdg1, pdg2)

        def fake_pythia(events, mass, seed):
            del mass, seed
            captured["signatures"] = {
                tuple(sorted((int(row[5]), int(row[13]))))
                for row in np.asarray(events)
            }
            return np.zeros((len(events), 12), dtype=np.float64)

        with (
            mock.patch.object(
                decayProducts.TwoBodyDecay,
                "decay_products",
                side_effect=fake_two_body,
            ),
            mock.patch.object(
                decayProducts,
                "process_events_with_pythia",
                side_effect=fake_pythia,
            ) as pythia,
            mock.patch.object(decayProducts, "seed_decay_generators"),
        ):
            events, plan = decayProducts._generate_alp_stock_pythia_pool(
                2.0,
                self.pdgs,
                self.ratios,
                self.selected,
                self.counts,
                "ALP-fermion",
                seed=250725,
                batch_index=8,
            )

        expected_pool = int(np.sum(self.counts[1:]))
        self.assertEqual(len(events), expected_pool)
        self.assertEqual(plan["pool_size"], expected_pool)
        self.assertEqual(
            captured["signatures"], {(-3, 3), (21, 21)}
        )
        self.assertNotIn((-2212, 2212), captured["signatures"])
        self.assertNotIn((-2112, 2112), captured["signatures"])
        pythia.assert_called_once()

    def test_process_labels_replace_every_physical_hadronic_row(self):
        pooled_positions = tuple(range(1, 6))
        pool_size = int(np.sum(self.counts[1:]))
        stock_events = np.zeros((pool_size, 12), dtype=np.float64)
        plan = {
            "pooled_positions": pooled_positions,
            "pool_size": pool_size,
            "sources": (),
        }

        def fake_two_body(
            mass, size, mass1, mass2, pdg1, pdg2,
            charge1, charge2, stability1, stability2,
        ):
            del mass, charge1, charge2, stability1, stability2
            return _two_body_rows(size, mass1, mass2, pdg1, pdg2)

        with (
            mock.patch.object(
                decayProducts,
                "_fixed_exhad_context",
                return_value=(None, []),
            ),
            mock.patch.object(
                decayProducts,
                "_generate_alp_stock_pythia_pool",
                return_value=(stock_events, plan),
            ),
            mock.patch.object(
                decayProducts.TwoBodyDecay,
                "decay_products",
                side_effect=fake_two_body,
            ),
            mock.patch.object(decayProducts, "seed_decay_generators"),
        ):
            np.random.seed(self.allocation_seed)
            events, sizes, labels = (
                decayProducts.simulateDecays_rest_frame(
                    2.0,
                    self.pdgs,
                    self.ratios,
                    100,
                    None,
                    self.selected,
                    1.0,
                    llp_name="ALP-fermion",
                    stock_pythia_pool=True,
                    return_process_labels=True,
                )
            )

        self.assertEqual(len(events), 100)
        np.testing.assert_array_equal(sizes, self.counts)
        self.assertIsNone(labels[0])
        self.assertEqual(
            labels[1:],
            [decayProducts.ALP_STOCK_PYTHIA_PROCESS_LABEL] * 5,
        )

    def test_wrong_portal_and_unknown_positive_row_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "only for ALP-fermion"):
            decayProducts._alp_stock_pythia_plan(
                2.0,
                self.pdgs,
                self.ratios,
                self.selected,
                self.counts,
                "Scalar-mixing",
            )

        pdgs = np.vstack((self.pdgs, np.asarray([[211, -211]])))
        ratios = np.append(self.ratios, 0.1)
        counts = decayProducts.distribute_events(100, ratios)
        with self.assertRaisesRegex(ValueError, "outside the complete"):
            decayProducts._alp_stock_pythia_plan(
                2.0,
                pdgs,
                ratios,
                list(range(len(pdgs))),
                counts,
                "ALP-fermion",
            )


if __name__ == "__main__":
    unittest.main()
