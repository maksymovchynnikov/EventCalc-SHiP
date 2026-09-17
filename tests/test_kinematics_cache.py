#!/usr/bin/env python3
"""Contracts for explicit, revisioned kinematics-grid reuse."""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import initLLP, kinematics  # noqa: E402


def _source_frames(value=1.0):
    distr = pd.DataFrame([
        [1.0, 0.01, 2.0, value],
        [1.0, 0.02, 3.0, value + 1.0],
    ])
    energy = pd.DataFrame([
        [1.0, 0.01, 10.0],
        [1.0, 0.02, 11.0],
    ])
    return distr, energy


class KinematicsCacheTest(unittest.TestCase):
    def setUp(self):
        kinematics._GRID_CACHE.clear()

    def tearDown(self):
        kinematics._GRID_CACHE.clear()

    @staticmethod
    def _grid_patches():
        fill_2d = mock.patch.object(
            kinematics, "_fill_distr_2D",
            side_effect=lambda _m, _a, frame:
                np.asarray([[float(frame.iloc[0, 2])]]))
        fill_3d = mock.patch.object(
            kinematics, "_fill_distr_3D",
            side_effect=lambda _x, _y, _z, frame:
                np.asarray([[[float(frame.iloc[0, 3])]]]))
        return fill_2d, fill_3d

    def test_no_token_rebuilds_and_observes_same_shape_mutation(self):
        distr, energy = _source_frames()
        fill_2d, fill_3d = self._grid_patches()
        with fill_2d as energy_fill, fill_3d as distr_fill:
            first = kinematics.Grids(distr, energy, 1, 1.0, 1.0)
            distr.iloc[0, 3] = 7.0
            second = kinematics.Grids(distr, energy, 1, 1.0, 1.0)

        self.assertEqual(float(first.distr[0, 0, 0]), 1.0)
        self.assertEqual(float(second.distr[0, 0, 0]), 7.0)
        self.assertEqual(energy_fill.call_count, 2)
        self.assertEqual(distr_fill.call_count, 2)
        self.assertEqual(len(kinematics._GRID_CACHE), 0)

    def test_explicit_revision_reuses_unchanged_source_then_refreshes(self):
        distr, energy = _source_frames()
        source = object()
        fill_2d, fill_3d = self._grid_patches()
        with fill_2d as energy_fill, fill_3d as distr_fill:
            first = kinematics.Grids(
                distr, energy, 1, 1.0, 1.0,
                cache_token=(source, 0))
            reused = kinematics.Grids(
                distr, energy, 1, 1.0, 2.0,
                cache_token=(source, 0))
            distr.iloc[0, 3] = 9.0
            refreshed = kinematics.Grids(
                distr, energy, 1, 1.0, 2.0,
                cache_token=(source, 1))

        self.assertIs(first.distr, reused.distr)
        self.assertEqual(float(refreshed.distr[0, 0, 0]), 9.0)
        self.assertEqual(energy_fill.call_count, 2)
        self.assertEqual(distr_fill.call_count, 2)

    def test_explicit_cache_is_bounded(self):
        distr, energy = _source_frames()
        source = object()
        fill_2d, fill_3d = self._grid_patches()
        with fill_2d, fill_3d:
            for revision in range(kinematics._GRID_CACHE_MAXSIZE + 3):
                kinematics.Grids(
                    distr, energy, 1, 1.0, 1.0,
                    cache_token=(source, revision))

        self.assertEqual(len(kinematics._GRID_CACHE),
                         kinematics._GRID_CACHE_MAXSIZE)
        self.assertNotIn((source, 0), kinematics._GRID_CACHE)
        self.assertIn(
            (source, kinematics._GRID_CACHE_MAXSIZE + 2),
            kinematics._GRID_CACHE)

    def test_unhashable_explicit_token_is_rejected(self):
        distr, energy = _source_frames()
        with self.assertRaisesRegex(TypeError, "must be hashable"):
            kinematics.Grids(
                distr, energy, 1, 1.0, 1.0, cache_token=[])

    def test_llp_token_is_stable_across_lifetimes_and_revised_on_replace(self):
        llp = initLLP.LLP.__new__(initLLP.LLP)
        llp._kinematics_cache_namespace = object()
        llp._kinematics_source_revision = 1
        initial = llp.kinematics_cache_token()

        llp.set_c_tau(1.0)
        self.assertEqual(llp.kinematics_cache_token(), initial)
        llp.set_c_tau(10.0)
        self.assertEqual(llp.kinematics_cache_token(), initial)

        replacement, _energy = _source_frames(4.0)
        llp._replace_kinematics_distribution(replacement)
        self.assertNotEqual(llp.kinematics_cache_token(), initial)
        self.assertIs(llp.Distr, replacement)


if __name__ == "__main__":
    unittest.main()
