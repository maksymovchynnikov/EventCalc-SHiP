from pathlib import Path
import unittest

import numpy as np

from alp_discrimination.paths import PACKAGE_ROOT
from alp_discrimination.plotting.reference_curves import (
    REFERENCE_FILENAMES, load_reference, split_reference_branches,
)


class ReferenceCurveTests(unittest.TestCase):
    def test_closed_polygon_splits_into_log_interpolatable_branches(self):
        points = np.asarray([[1, 1], [2, 2], [4, 4], [2, 8], [1, 10], [1, 1]], float)
        branches = split_reference_branches(points)
        np.testing.assert_allclose(branches["lower"]["coupling_GeV_inv"], [1, 2, 4])
        np.testing.assert_allclose(branches["upper"]["coupling_GeV_inv"], [10, 8, 4])

    def test_bundled_reference_curves_load_and_split(self):
        reference_dir = PACKAGE_ROOT / "reference_data" / "photon_sensitivity"
        references = [
            load_reference(reference_dir / filename, name)
            for name, filename in REFERENCE_FILENAMES.items()
        ]
        self.assertEqual([len(item.points) for item in references], [2641, 2628])
        for reference in references:
            branches = split_reference_branches(reference.points)
            self.assertEqual(set(branches), {"lower", "upper"})
            self.assertGreaterEqual(len(branches["lower"]), 2)
            self.assertGreaterEqual(len(branches["upper"]), 2)



if __name__ == "__main__":
    unittest.main()
