#!/usr/bin/env python3
"""Regressions for the Mathematica-to-EventCalc ALP card postprocessor."""

import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPT = os.path.join(
    ROOT, "Distributions", "ALP-fermion", "postprocess_decay_card.py")
SPEC = importlib.util.spec_from_file_location("alp_decay_postprocess", SCRIPT)
POSTPROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POSTPROCESS)


class ALPDecayPostprocessTest(unittest.TestCase):
    def test_exact_nucleon_pairs_have_canonical_names(self):
        self.assertEqual(
            POSTPROCESS.NAME_BY_PDGS[POSTPROCESS.key([2212, -2212])],
            "ppbar")
        self.assertEqual(
            POSTPROCESS.NAME_BY_PDGS[POSTPROCESS.key([2112, -2112])],
            "nnbar")

    def test_python_piecewise_is_accepted_unchanged(self):
        source = "Piecewise((x**2, x > 0), (c, True))"
        self.assertEqual(
            POSTPROCESS.require_python_matrix_syntax(source, "mode"), source)

    def test_mathematica_piecewise_fails_closed(self):
        source = "Piecewise({{x**2, x > 0}}, c)"
        with self.assertRaisesRegex(ValueError, "source exporter"):
            POSTPROCESS.require_python_matrix_syntax(source, "mode")

    def test_mathematica_scientific_notation_fails_closed(self):
        for source in ("1.0*^6*x", "1.0***6*x"):
            with self.assertRaisesRegex(ValueError, "source exporter"):
                POSTPROCESS.require_python_matrix_syntax(source, "mode")

    def test_numeric_unit_is_accepted_only_outside_three_body_dynamics(self):
        for daughters in (2, 4):
            self.assertEqual(
                POSTPROCESS.canonical_matrix_element_type(
                    1.0, "phase-space mode", daughters),
                "1.")
        with self.assertRaisesRegex(ValueError, "source expression"):
            POSTPROCESS.canonical_matrix_element_type(
                1.0, "three-body mode", 3)
        with self.assertRaisesRegex(ValueError, "source expression"):
            POSTPROCESS.canonical_matrix_element_type(
                2.0, "two-body mode", 2)

    def test_lifetime_tracks_the_surviving_partial_width(self):
        self.assertAlmostEqual(
            POSTPROCESS.corrected_ctau(2.0, 1.0, 0.25), 8.0)
        self.assertAlmostEqual(
            POSTPROCESS.corrected_ctau(2.0, 0.8, 0.6), 8.0 / 3.0)
        with self.assertRaisesRegex(ValueError, "allowed-width"):
            POSTPROCESS.corrected_ctau(2.0, 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
