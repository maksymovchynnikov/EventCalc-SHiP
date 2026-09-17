#!/usr/bin/env python3
"""Regressions for the native fermion-ALP coupling normalization."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from funcs import coupling_conventions, mergeResults  # noqa: E402
from funcs.selecting_processing import read_file  # noqa: E402


class ALPFermionCouplingConventionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / "Distributions" / "ALP-fermion"
        cls.metadata = coupling_conventions.load_coupling_metadata(cls.folder)

    @staticmethod
    def table_value(path, target_mass):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                mass, value = map(float, line.split())
                if abs(mass - target_mass) < 1.0e-12:
                    return value
        raise AssertionError("mass %.12g is absent from %s" %
                             (target_mass, path))

    def test_canonical_definition_and_non_coupling_scale_are_explicit(self):
        coupling = self.metadata["coupling"]
        self.assertEqual(coupling["symbol"], "g_Y")
        self.assertEqual(coupling["alias"], "y")
        self.assertEqual(coupling["definition"], "2*v_h/f_a")
        self.assertEqual(coupling["squared_output_quantity"], "g_Y^2")

        benchmark = self.metadata["benchmark"]
        self.assertEqual(benchmark["name"], "BC10")
        self.assertEqual(benchmark["eft_rg_matching_scale_symbol"], "Lambda")
        self.assertEqual(benchmark["eft_rg_matching_scale_GeV"], 1000.0)
        self.assertIs(benchmark["matching_scale_is_f_a"], False)
        self.assertIn("stale and non-normative",
                      self.metadata["provenance"]["non_normative_label"])

    def test_validator_rejects_factor_four_and_scale_ambiguities(self):
        wrong_definition = copy.deepcopy(self.metadata)
        wrong_definition["coupling"]["definition"] = "v_h/f_a"
        with self.assertRaisesRegex(ValueError, "definition"):
            coupling_conventions.validate_coupling_metadata(wrong_definition)

        wrong_scale = copy.deepcopy(self.metadata)
        wrong_scale["benchmark"]["matching_scale_is_f_a"] = True
        with self.assertRaisesRegex(ValueError, "matching_scale_is_f_a"):
            coupling_conventions.validate_coupling_metadata(wrong_scale)

        missing_stale_label = copy.deepcopy(self.metadata)
        missing_stale_label["provenance"]["non_normative_label"] = ""
        with self.assertRaisesRegex(ValueError, "non_normative_label"):
            coupling_conventions.validate_coupling_metadata(missing_stale_label)

    def test_native_scaling_has_no_extra_factor_of_four(self):
        coefficient = self.table_value(
            self.folder / "ctau-ALP-fermion.txt", 2.0)
        production_per_g_y_squared = self.table_value(
            self.folder / "Total-yield-ALP-fermion.txt", 2.0)
        self.assertAlmostEqual(coefficient, 5.3033717550756286e-10)
        self.assertAlmostEqual(
            production_per_g_y_squared, 3.13914471831389e-6)

        # g_Y=1 returns the tabulated lifetime; g_Y=2 gives c_tau=C/4
        # and four times the tabulated production probability.
        self.assertAlmostEqual(
            coupling_conventions.native_coupling_squared(
                "ALP-fermion", coefficient, coefficient,
                metadata=self.metadata),
            1.0)
        requested_ctau = coefficient / 4.0
        self.assertAlmostEqual(
            coupling_conventions.native_coupling_squared(
                "ALP-fermion", coefficient, requested_ctau,
                metadata=self.metadata),
            4.0)
        self.assertAlmostEqual(
            production_per_g_y_squared * 4.0,
            1.255657887325556e-5)

    def test_alp_outputs_are_annotated_and_have_sidecars(self):
        mothers = np.zeros((1, 10), dtype=float)
        decays = np.zeros((1, 6), dtype=float)
        channels = np.array(["muPmuM"], dtype=object)

        with tempfile.TemporaryDirectory() as directory:
            # the writer takes its output root; no caller changes directory
            mergeResults.save(
                mothers, decays, "ALP-fermion", 2.0, None, 1.0,
                channels, [1], 1, 1.0, 1.0, 8.0, 4.0, 0.5, 2.0,
                1.0, [0], None, True,
                coupling_metadata=self.metadata,
                output_root=directory)

            output_root = Path(directory) / "outputs" / "ALP-fermion"
            event_path = next((output_root / "eventData").glob("*_data.dat"))
            total_path = output_root / "total" / "ALP-fermion_total.txt"
            with event_path.open(encoding="utf-8") as stream:
                first_line = stream.readline()
            self.assertIn("coupling_squared = g_Y^2", first_line)
            self.assertIn("g_Y = y = 2 v_h/f_a", first_line)

            # The historical reader's regex deliberately ignores the suffix.
            parsed = read_file(event_path)
            self.assertEqual(parsed[4], 4.0)

            with total_path.open(encoding="utf-8") as stream:
                self.assertEqual(
                    stream.readline(),
                    "mass coupling_squared c_tau N_LLP_tot epsilon_polar "
                    "epsilon_azimuthal P_decay_averaged Br_visible "
                    "N_ev_tot\n")

            for output_path in (event_path, total_path):
                sidecar_path = Path(str(output_path) + ".coupling.json")
                self.assertTrue(sidecar_path.is_file())
                with sidecar_path.open(encoding="utf-8") as stream:
                    sidecar = json.load(stream)
                self.assertEqual(sidecar["applies_to"], output_path.name)
                self.assertEqual(
                    sidecar["coupling_metadata"]["coupling"]["definition"],
                    "2*v_h/f_a")
                self.assertIs(
                    sidecar["coupling_metadata"]["benchmark"]
                    ["matching_scale_is_f_a"],
                    False)


if __name__ == "__main__":
    unittest.main()
