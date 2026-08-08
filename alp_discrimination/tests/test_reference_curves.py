from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from alp_discrimination.paths import LEGACY_ANALYSIS_ROOT, PACKAGE_ROOT
from alp_discrimination.reference_curves import (
    REFERENCE_FILENAMES, load_eventcalc_branches, load_reference,
    make_distance_summary, make_pointwise_comparison, split_reference_branches,
)


class ReferenceCurveTests(unittest.TestCase):
    def test_closed_polygon_splits_into_log_interpolatable_branches(self):
        points = np.asarray([[1, 1], [2, 2], [4, 4], [2, 8], [1, 10], [1, 1]], float)
        branches = split_reference_branches(points)
        np.testing.assert_allclose(branches["lower"]["coupling_GeV_inv"], [1, 2, 4])
        np.testing.assert_allclose(branches["upper"]["coupling_GeV_inv"], [10, 8, 4])

    def test_bundled_curves_reproduce_legacy_distance_summary(self):
        reference_dir = PACKAGE_ROOT / "reference_curves"
        references = [load_reference(reference_dir / filename, name)
                      for name, filename in REFERENCE_FILENAMES.items()]
        self.assertEqual([len(item.points) for item in references], [2641, 2628])
        branches = {item.name: split_reference_branches(item.points) for item in references}
        boundary_path = LEGACY_ANALYSIS_ROOT / "event_density_scan/event_contour_boundaries.csv"
        pointwise = make_pointwise_comparison(
            load_eventcalc_branches(pd.read_csv(boundary_path)), branches,
        )
        summary = make_distance_summary(pointwise).set_index("reference")
        expected = {
            "epsilon_dec_1": (0.0689190692889179, 0.03461029778530555, 0.7933333333333333),
            "geom_only": (0.24463558216477216, 0.045974457745375154, 0.8766666666666667),
        }
        for name, values in expected.items():
            actual = summary.loc[name, [
                "maximum_absolute_lower_distance_dex",
                "maximum_absolute_upper_distance_dex", "both_branches_inside_fraction",
            ]].to_numpy(float)
            np.testing.assert_allclose(actual, values, rtol=1e-12, atol=1e-15)
        self.assertEqual(len(pointwise), 2400)


if __name__ == "__main__":
    unittest.main()
