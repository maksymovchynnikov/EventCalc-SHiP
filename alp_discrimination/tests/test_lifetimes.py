import unittest

import numpy as np

from alp_discrimination.physics.lifetimes import (
    LifetimeInterval, dense_log_grid, interior_lifetime_points, intersect_intervals,
    lifetime_point_records, logarithmic_fraction,
)


class LifetimeTests(unittest.TestCase):
    def test_logarithmic_midpoint_and_fraction(self):
        interval = LifetimeInterval(1.0, 100.0)
        self.assertAlmostEqual(logarithmic_fraction(interval, 0.5), 10.0)
        self.assertAlmostEqual(logarithmic_fraction(interval, 0.25), np.sqrt(10.0))

    def test_intersections(self):
        result = intersect_intervals([LifetimeInterval(1.0, 10.0)], [LifetimeInterval(5.0, 20.0)])
        self.assertEqual(result, [LifetimeInterval(5.0, 10.0)])

    def test_invalid_intervals_fail(self):
        with self.assertRaises(ValueError):
            LifetimeInterval(10.0, 1.0)
        with self.assertRaises(ValueError):
            dense_log_grid(LifetimeInterval(1.0, None), 4)

    def test_interior_points_are_ordered(self):
        interval = LifetimeInterval(1.0, 100.0)
        points = interior_lifetime_points(interval, (("low", 0.1), ("mid", 0.5), ("high", 0.9)))
        values = np.asarray([point[2] for point in points])
        self.assertTrue(np.all((values > 1.0) & (values < 100.0)))
        self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_lifetime_point_records(self):
        rows = lifetime_point_records(
            [{"mass_GeV": 0.3, "ctau_lower_m": 1.0, "ctau_upper_m": 100.0}],
            (("mid", 0.5),),
        )
        self.assertAlmostEqual(rows[0]["ctau_m"], 10.0)


if __name__ == "__main__":
    unittest.main()
