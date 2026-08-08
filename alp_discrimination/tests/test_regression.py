import unittest

import pandas as pd

from alp_discrimination.workflows.compare_legacy_results import compare_frames, stochastic_frame_comparison


class RegressionTests(unittest.TestCase):
    def test_exact_float_and_mismatch_categories(self):
        legacy = pd.DataFrame({"key": [1], "label": ["x"], "value": [1.0]})
        exact = compare_frames("toy", legacy, legacy.copy(), ["key"], exact_columns=["label"], float_columns=["value"])
        self.assertEqual(exact.status, "exact_agreement")
        close = legacy.copy()
        close["value"] += 1e-13
        floating = compare_frames("toy", legacy, close, ["key"], float_columns=["value"])
        self.assertEqual(floating.status, "floating_point_agreement")
        far = legacy.copy()
        far["value"] = 2.0
        mismatch = compare_frames("toy", legacy, far, ["key"], float_columns=["value"])
        self.assertEqual(mismatch.status, "genuine_mismatch")

    def test_duplicate_and_unmatched_stochastic_keys_fail(self):
        legacy = pd.DataFrame({"key": [1], "accuracy": [0.8]})
        duplicate = pd.concat([legacy, legacy], ignore_index=True)
        self.assertEqual(compare_frames("toy", duplicate, legacy, ["key"]).status, "genuine_mismatch")
        current = pd.DataFrame({"key": [2], "accuracy": [0.8]})
        result = stochastic_frame_comparison("toy", legacy, current, ["key"], ["accuracy"], 100)
        self.assertEqual(result.status, "genuine_mismatch")


if __name__ == "__main__":
    unittest.main()
