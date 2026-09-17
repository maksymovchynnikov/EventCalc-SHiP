#!/usr/bin/env python3
"""Regression tests for deterministic split/resumed EventCalc scans."""

import os
import sys
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_batch  # noqa: E402
from funcs import output_provenance  # noqa: E402
from funcs import decayProducts  # noqa: E402


class BatchIndexOffsetTest(unittest.TestCase):
    def test_one_mass_split_reproduces_full_scan_parent_and_exhad_seeds(self):
        base_seed = 424242
        full_mass_index = 7
        n_ctaus = 3

        for ctau_index in range(n_ctaus):
            with self.subTest(ctau_index=ctau_index):
                full_fixed, full_point = run_batch.scan_batch_indices(
                    full_mass_index, ctau_index, n_ctaus)
                split_fixed, split_point = run_batch.scan_batch_indices(
                    0, ctau_index, n_ctaus,
                    batch_index_offset=full_mass_index)

                self.assertEqual((split_fixed, split_point),
                                 (full_fixed, full_point))
                self.assertEqual(
                    decayProducts.derive_exhad_seed(
                        base_seed, split_point, 0x45564341),
                    decayProducts.derive_exhad_seed(
                        base_seed, full_point, 0x45564341))
                self.assertEqual(
                    decayProducts.derive_exhad_seed(base_seed, split_fixed, 4),
                    decayProducts.derive_exhad_seed(base_seed, full_fixed, 4))
                self.assertEqual(
                    decayProducts.derive_exhad_seed(base_seed, split_point, 4),
                    decayProducts.derive_exhad_seed(base_seed, full_point, 4))

    def test_offset_does_not_enter_output_generator_tag(self):
        _, full_point = run_batch.scan_batch_indices(5, 0, 1)
        _, split_point = run_batch.scan_batch_indices(
            0, 0, 1, batch_index_offset=5)
        self.assertEqual(split_point, full_point)
        self.assertEqual(output_provenance.build_generator_tag("dv", 424242),
                         "exhad-dv-seed424242")

    def test_cli_accepts_nonnegative_mass_grid_offset(self):
        argv = [
            "run_batch.py", "--llp", "Dark-photons", "--nevents", "10",
            "--masses", "2.4", "--ctaus", "10",
            "--batch-index-offset", "7",
        ]
        with mock.patch.object(sys, "argv", argv):
            _, args = run_batch.parse_args()
        self.assertEqual(args.batch_index_offset, 7)

    def test_cli_accepts_stock_pythia_pool_flag(self):
        argv = [
            "run_batch.py", "--llp", "ALP-fermion", "--nevents", "10",
            "--masses", "2.0", "--ctaus", "10", "--channels", "all",
            "--exhad", "off", "--stock-pythia-pool",
        ]
        with mock.patch.object(sys, "argv", argv):
            _, args = run_batch.parse_args()
        self.assertTrue(args.stock_pythia_pool)

    def test_invalid_coordinates_are_rejected(self):
        invalid = (
            (-1, 0, 1, 0),
            (0, -1, 1, 0),
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 1, -1),
        )
        for coordinates in invalid:
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    run_batch.scan_batch_indices(*coordinates)


if __name__ == "__main__":
    unittest.main()
