import unittest

import numpy as np

from alp_discrimination.templates.probability import TemplateBank, build_probability_templates, common_adaptive_energy_edges
from alp_discrimination.tests.helpers import spectrum


class BinningTests(unittest.TestCase):
    def test_template_bank_accepts_model_specific_lifetimes(self):
        bank = TemplateBank(
            model_ids=("a", "b"), lifetimes_m=np.array([[1.0, 2.0], [3.0, 4.0]]),
            energy_edges_gev=np.array([1.0, 2.0, 3.0]),
            probabilities=np.full((2, 2, 2), 0.5), mass_gev=1.0,
            selection_name="mother_level",
        )
        self.assertEqual(bank.probabilities.shape, (2, 2, 2))

    def test_problematic_first_bin_merges_deterministically(self):
        spectra = {
            "a": spectrum([1.5, 2.5, 4.5, 5.5], np.ones(4), model="a"),
            "b": spectrum([1.5, 2.5, 4.5, 5.5], np.ones(4), model="b"),
        }
        edges = common_adaptive_energy_edges(spectra, [1.0, 2.0, 4.0, 8.0], 2.0)
        np.testing.assert_array_equal(edges, [1.0, 4.0, 8.0])
        self.assertTrue(np.all(np.diff(edges) > 0.0))
        templates = build_probability_templates(spectra, [1.0, 2.0, 4.0, 8.0], 2.0, 0.5)
        np.testing.assert_array_equal(templates["a"].energy_edges_gev, templates["b"].energy_edges_gev)

    def test_common_binning_can_span_different_lifetimes(self):
        first = spectrum([1.5, 2.5], [1.0, 1.0], model="a")
        second = spectrum([1.5, 2.5], [1.0, 1.0], model="a")
        second.ctau_m = 20.0
        templates = build_probability_templates(
            {"a_short": first, "a_long": second}, [1.0, 2.0, 3.0], 1.0, 0.5
        )
        self.assertEqual(templates["a_long"].ctau_m, 20.0)

    def test_impossible_case_terminates_with_error(self):
        spectra = {
            "a": spectrum([1.5], [1.0], model="a"),
            "b": spectrum([2.5], [1.0], model="b"),
        }
        with self.assertRaisesRegex(RuntimeError, "reliable common binning"):
            common_adaptive_energy_edges(spectra, [1.0, 2.0, 3.0], 2.0)


if __name__ == "__main__":
    unittest.main()
