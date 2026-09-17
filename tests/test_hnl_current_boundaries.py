#!/usr/bin/env python3
"""Source-grid and runtime-boundary contracts for matched HNL currents."""

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs.hnl_source_validation import load_canonical_hnl_source  # noqa: E402


MIXINGS = (("e", 0), ("mu", 1), ("tau", 2))
POLE_PREFIXES = ("Pi", "K", "D", "Ds")

NC_FIRST_POSITIVE = {
    "NC_ud": (("Jets-uuv", "Jets-ddv"), 1.38),
    "NC_s": (("Jets-ssv",), 2.59),
    "NC_c": (("Jets-ccv",), 5.80),
}

CC_FIRST_POSITIVE = {
    "ud": {"e": 1.85, "mu": 1.865, "tau": 3.64},
    "us": {"e": 1.38, "mu": 1.40, "tau": 2.92},
    "cd": {"e": 3.16, "mu": 3.19, "tau": 4.57},
    "cs": {"e": 3.55, "mu": 3.58, "tau": 4.90},
}


def _load_boundary_runtime():
    """Load HNLmerging without optional pandas/sympy test dependencies.

    The tested interpolation and threshold functions use NumPy/SciPy only.
    Loading the module under a private name keeps these lightweight import
    placeholders isolated from the rest of an eventual discovery run.
    """

    module_path = Path(ROOT) / "funcs" / "HNLmerging.py"
    module_name = "funcs._hnl_boundary_runtime"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load HNL boundary runtime")
    module = importlib.util.module_from_spec(spec)
    placeholders = {
        "pandas": types.ModuleType("pandas"),
        "sympy": types.ModuleType("sympy"),
    }
    with mock.patch.dict(sys.modules, placeholders):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def _first_positive(table):
    for index, (mass, value) in enumerate(table):
        if value > 0.0:
            return index, mass
    raise AssertionError("branching table never becomes positive")


def _supported_jet_rows():
    rows = {"Jets-uuv", "Jets-ddv", "Jets-ssv"}
    for current in ("ud", "us"):
        for suffix, _mixing_index in MIXINGS:
            rows.add("Jets-" + current + suffix)
            rows.add("Jets-" + current + suffix + "bar")
    return frozenset(rows)


def _supported_current(channel):
    if channel in ("Jets-uuv", "Jets-ddv"):
        return "NC_ud"
    if channel == "Jets-ssv":
        return "NC_s"
    if channel.startswith("Jets-ud"):
        return "CC_ud"
    if channel.startswith("Jets-us"):
        return "CC_us"
    return None


class HNLCurrentBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_canonical_hnl_source()
        cls.rows = cls.source.decay_rows
        cls.by_name = {row[0]: row for row in cls.rows}
        cls.row_index = {row[0]: index
                         for index, row in enumerate(cls.rows)}

    def assert_first_positive_node(self, channel, mixing_index, expected):
        table = self.by_name[channel][2 + mixing_index]
        index, mass = _first_positive(table)
        self.assertEqual(mass, self.source.mass_grid[index])
        self.assertTrue(all(value == 0.0 for _, value in table[:index]))
        self.assertGreater(table[index][1], 0.0)
        self.assertAlmostEqual(mass, expected, places=12)

    def test_nc_first_positive_nodes_on_canonical_grid(self):
        for current, (channels, expected) in NC_FIRST_POSITIVE.items():
            for channel in channels:
                for mixing, mixing_index in MIXINGS:
                    with self.subTest(current=current, channel=channel,
                                      mixing=mixing):
                        self.assert_first_positive_node(
                            channel, mixing_index, expected)

    def test_cc_first_positive_nodes_on_canonical_grid(self):
        for current, expected_by_mixing in CC_FIRST_POSITIVE.items():
            for mixing, mixing_index in MIXINGS:
                for conjugate in ("", "bar"):
                    channel = ("Jets-" + current + mixing + conjugate)
                    with self.subTest(current=current, channel=channel,
                                      mixing=mixing):
                        self.assert_first_positive_node(
                            channel, mixing_index,
                            expected_by_mixing[mixing])

    def test_explicit_pi_k_d_ds_poles_remain_positive_through_40(self):
        self.assertAlmostEqual(self.source.mass_grid[-1], 40.0, places=12)
        for mixing, mixing_index in MIXINGS:
            for prefix in POLE_PREFIXES:
                for conjugate in ("", "bar"):
                    channel = prefix + mixing + conjugate
                    table = self.by_name[channel][2 + mixing_index]
                    first, _mass = _first_positive(table)
                    with self.subTest(channel=channel, mixing=mixing):
                        self.assertEqual(
                            table[-1][0], self.source.mass_grid[-1])
                        self.assertGreater(table[-1][1], 0.0)
                        self.assertTrue(all(
                            value > 0.0 for _, value in table[first:]))

    def test_unsupported_currents_are_zero_through_3gev(self):
        supported = _supported_jet_rows()
        jets = {name for name in self.by_name
                if name.startswith("Jets-")}
        unsupported = jets.difference(supported)
        low_indices = tuple(index for index, mass in
                            enumerate(self.source.mass_grid)
                            if mass <= 3.0)
        self.assertTrue(low_indices)
        self.assertTrue(unsupported)

        for channel in sorted(unsupported):
            row = self.by_name[channel]
            for mixing, mixing_index in MIXINGS:
                with self.subTest(channel=channel, mixing=mixing):
                    self.assertTrue(all(
                        row[2 + mixing_index][index][1] == 0.0
                        for index in low_indices))

        positive_currents = set()
        for channel in sorted(jets):
            row = self.by_name[channel]
            for _mixing, mixing_index in MIXINGS:
                if any(row[2 + mixing_index][index][1] > 0.0
                       for index in low_indices):
                    self.assertIn(channel, supported)
                    current = _supported_current(channel)
                    self.assertIsNotNone(current)
                    positive_currents.add(current)
        self.assertEqual(
            positive_currents, {"CC_ud", "CC_us", "NC_ud", "NC_s"})


class HNLExclusiveRuntimeBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_canonical_hnl_source()
        cls.rows = cls.source.decay_rows
        cls.by_name = {row[0]: row for row in cls.rows}
        cls.row_index = {row[0]: index
                         for index, row in enumerate(cls.rows)}
        cls.runtime = _load_boundary_runtime()

        channel_count = len(cls.rows)
        cls.br_ratios = np.empty((3, channel_count), dtype=object)
        cls.pdgs = np.empty(channel_count, dtype=object)
        for row_index, row in enumerate(cls.rows):
            cls.pdgs[row_index] = np.asarray(row[1], dtype=int)
            for mixing_index in range(3):
                cls.br_ratios[mixing_index, row_index] = \
                    row[2 + mixing_index]
        cls.decay_width_data = np.asarray(
            cls.source.width_rows, dtype=float).T
        cls.merged_by_mixing = tuple(
            cls.runtime.get_BrMerged_func(
                cls.br_ratios,
                cls.decay_width_data,
                tuple(1.0 if index == mixing_index else 0.0
                      for index in range(3)),
                cls.pdgs,
            )
            for mixing_index in range(3)
        )

    def test_runtime_exclusive_interpolation_clamps_physical_threshold(self):
        epsilon = 1.0e-9
        for mixing, mixing_index in MIXINGS:
            get_br = self.merged_by_mixing[mixing_index]
            for prefix in POLE_PREFIXES:
                for conjugate in ("", "bar"):
                    channel = prefix + mixing + conjugate
                    index = self.row_index[channel]
                    threshold = self.runtime.get_exclusive_thresholds(
                        [self.pdgs[index]])[0]
                    self.assertIsNotNone(threshold)
                    below = threshold - epsilon
                    above = threshold + epsilon
                    table = np.asarray(
                        self.by_name[channel][2 + mixing_index],
                        dtype=float)

                    # This is deliberately inside the linear seam: the raw
                    # interpolation is already positive, so a zero result can
                    # only come from the run-time physical-threshold clamp.
                    raw_below = np.interp(
                        below, table[:, 0], table[:, 1])
                    with self.subTest(channel=channel, mixing=mixing):
                        self.assertGreater(raw_below, 0.0)
                        self.assertEqual(get_br(below)[index], 0.0)
                        self.assertEqual(get_br(threshold)[index], 0.0)
                        self.assertGreater(get_br(above)[index], 0.0)


if __name__ == "__main__":
    unittest.main()
