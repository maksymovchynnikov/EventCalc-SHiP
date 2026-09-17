#!/usr/bin/env python3
"""Regression tests for fixed-mass exhad pool reuse in batch scans."""

import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_batch  # noqa: E402


class FixedExhadPoolReuseTest(unittest.TestCase):
    def test_same_mass_lifetimes_build_pool_once(self):
        pool = np.zeros((10, 6), dtype=np.float64)
        decay_products = mock.Mock()
        decay_products.prepare_fixed_exhad_pool.return_value = pool
        cache = {}

        first_lifetime = run_batch.get_or_create_fixed_exhad_pool(
            cache, 2.0, decay_products, size=10, seed=23)
        second_lifetime = run_batch.get_or_create_fixed_exhad_pool(
            cache, 2.0, decay_products, size=10, seed=23)

        self.assertIs(first_lifetime, pool)
        self.assertIs(second_lifetime, pool)
        decay_products.prepare_fixed_exhad_pool.assert_called_once_with(
            mass=2.0, size=10, seed=23)

    def test_inapplicable_none_is_also_cached(self):
        decay_products = mock.Mock()
        decay_products.prepare_fixed_exhad_pool.return_value = None
        cache = {}

        for _lifetime in (0.1, 1.0, 10.0):
            self.assertIsNone(run_batch.get_or_create_fixed_exhad_pool(
                cache, 2.0, decay_products, size=10))

        decay_products.prepare_fixed_exhad_pool.assert_called_once()


if __name__ == "__main__":
    unittest.main()
