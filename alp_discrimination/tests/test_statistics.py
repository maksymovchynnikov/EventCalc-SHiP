import unittest

import numpy as np

from alp_discrimination.statistics import (
    conditional_classification_accuracy, minimum_events_for_accuracy,
    same_lifetime_log_likelihood_ratio, simulate_shape_discrimination,
    total_variation_distance,
)


class StatisticsTests(unittest.TestCase):
    def test_identical_templates_are_ties(self):
        result = simulate_shape_discrimination([0.5, 0.5], [0.5, 0.5], 3, 100, 7)
        np.testing.assert_array_equal(result.photon_correct_fraction, 0.5)
        np.testing.assert_array_equal(result.su2_correct_fraction, 0.5)

    def test_separated_templates_classify_well(self):
        result = simulate_shape_discrimination([0.99, 0.01], [0.01, 0.99], 2, 10_000, 8)
        self.assertGreater(result.worst_case_correct_fraction[-1], 0.97)

    def test_likelihood_ratio_sign_and_ties(self):
        photon, su2 = np.array([0.9, 0.1]), np.array([0.1, 0.9])
        self.assertLess(same_lifetime_log_likelihood_ratio(np.array([0]), photon, su2), 0.0)
        self.assertGreater(same_lifetime_log_likelihood_ratio(np.array([1]), photon, su2), 0.0)
        self.assertEqual(conditional_classification_accuracy(np.zeros(4), "photon"), 0.5)

    def test_worst_case_and_first_threshold(self):
        result = simulate_shape_discrimination([0.8, 0.2], [0.2, 0.8], 4, 5_000, 9)
        np.testing.assert_array_equal(
            result.worst_case_correct_fraction,
            np.minimum(result.photon_correct_fraction, result.su2_correct_fraction),
        )
        self.assertEqual(minimum_events_for_accuracy([1, 2, 3], [0.8, 0.91, 0.95], 0.9), 2)

    def test_total_variation(self):
        self.assertEqual(total_variation_distance([0.5, 0.5], [0.5, 0.5]), 0.0)
        self.assertEqual(total_variation_distance([1.0, 0.0], [0.0, 1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
