import unittest
from dataclasses import replace

import numpy as np

from alp_discrimination.physics.spectra import (
    WeightedSpectrum, combine_absolute_source_spectra, effective_sample_size,
    normalized_weighted_spectrum,
)
from alp_discrimination.tests.helpers import spectrum


class SpectrumTests(unittest.TestCase):
    def test_probabilities_and_weights_are_consistent(self):
        item = spectrum([1.5, 2.5, 3.5], [1.0, 2.0, 3.0])
        result = normalized_weighted_spectrum(item, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result.bin_probabilities, [1 / 6, 2 / 6, 3 / 6])
        self.assertAlmostEqual(result.bin_probabilities.sum(), 1.0)
        self.assertAlmostEqual(result.sum_weights_per_bin.sum(), item.expected_events)

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size(np.array([1.0, 2.0, 3.0])), 36 / 14)
        self.assertAlmostEqual(effective_sample_size(np.array([1e-250, 2e-250])), 9 / 5)

    def test_zero_rate_is_valid_but_cannot_be_normalized(self):
        item = spectrum([1.5, 2.5], [0.0, 0.0])
        self.assertEqual(item.expected_events, 0.0)
        self.assertEqual(item.total_n_eff, 0.0)
        with self.assertRaises(ValueError):
            normalized_weighted_spectrum(item, [1.0, 2.0, 3.0])

    def test_negative_and_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            spectrum([1.5, 2.5], [1.0, -1.0])
        with self.assertRaises(ValueError):
            spectrum([1.5, 2.5], [1.0, np.nan])

    def test_sources_are_combined_absolutely_before_normalizing(self):
        primary = replace(
            spectrum([1.5], [9.0], model="photon", source="primary"),
            preselection_expected_events=12.0,
            preselection_samples=2,
            selection_efficiency_weighted=0.75,
        )
        cascade = replace(
            spectrum([2.5], [1.0], model="photon", source="cascade"),
            preselection_expected_events=2.0,
            preselection_samples=2,
            selection_efficiency_weighted=0.5,
        )
        combined = combine_absolute_source_spectra("photon", {"primary": primary, "cascade": cascade})
        result = normalized_weighted_spectrum(combined, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result.bin_probabilities, [0.9, 0.1])
        self.assertEqual(combined.source_expected_events, {"primary": 9.0, "cascade": 1.0})
        self.assertEqual(combined.preselection_expected_events, 14.0)
        self.assertEqual(combined.preselection_samples, 4)
        self.assertEqual(combined.selection_efficiency_weighted, 10.0 / 14.0)


if __name__ == "__main__":
    unittest.main()
