import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from alp_discrimination.lifetime_domains import (
    Interval,
    allowed_coupling_intervals,
    coupling_interval_to_ctau,
    polygon_vertical_slice_intervals,
    sensitivity_coupling_interval,
    unit_coupling_ctau_at_mass,
)

# -----------------------------------------------------------------------------
# Saved-domain loading and disconnected lifetime-grid tests
# -----------------------------------------------------------------------------

from alp_discrimination.lifetime_domains import (
    available_lifetime_domain_masses,
    build_lifetime_grid,
    load_allowed_ctau_domains,
)





class Week8DomainTests(unittest.TestCase):
    def test_log_log_sensitivity_branch_interpolation(self):
        boundaries = pd.DataFrame({
            "model": ["toy", "toy"],
            "mass_GeV": [0.1, 1.0],
            "event_level": [2.3, 2.3],
            "status": ["resolved", "resolved"],
            "lower_coupling_GeV_inv": [1.0e-6, 1.0e-5],
            "upper_coupling_GeV_inv": [1.0e-3, 1.0e-2],
        })
        mass = np.sqrt(0.1)
        interval = sensitivity_coupling_interval(boundaries, "toy", mass)
        self.assertAlmostEqual(interval.lower, np.sqrt(1.0e-11))
        self.assertAlmostEqual(interval.upper, np.sqrt(1.0e-5))

    def test_unit_coupling_lifetime_is_interpolated_per_mass(self):
        scan = pd.DataFrame({
            "model": ["toy"] * 4,
            "mass_GeV": [0.1, 0.1, 1.0, 1.0],
            "unit_coupling_ctau_m": [100.0, 100.0, 1.0, 1.0],
        })
        result = unit_coupling_ctau_at_mass(scan, "toy", np.sqrt(0.1))
        self.assertAlmostEqual(result, 10.0)

    def test_polygon_slice_and_exclusion_subtraction_preserve_gap(self):
        polygon = np.asarray([
            [0.1, 1.0e-4],
            [1.0, 1.0e-4],
            [1.0, 1.0e-3],
            [0.1, 1.0e-3],
        ])
        excluded = polygon_vertical_slice_intervals(polygon, 0.3)
        self.assertEqual(len(excluded), 1)
        np.testing.assert_allclose(
            [excluded[0].lower, excluded[0].upper],
            [1.0e-4, 1.0e-3],
            rtol=1.0e-12,
        )

        allowed, clipped_excluded = allowed_coupling_intervals(
            Interval(1.0e-5, 1.0e-2), [polygon], 0.3,
        )
        self.assertEqual(len(clipped_excluded), 1)
        self.assertEqual(len(allowed), 2)
        np.testing.assert_allclose(
            [[item.lower, item.upper] for item in allowed],
            [[1.0e-5, 1.0e-4], [1.0e-3, 1.0e-2]],
            rtol=1.0e-12,
        )
        self.assertFalse(allowed[0].upper_inclusive)
        self.assertFalse(allowed[1].lower_inclusive)

    def test_coupling_to_ctau_reverses_order_and_inclusivity(self):
        coupling = Interval(1.0e-4, 1.0e-3, True, False)
        ctau = coupling_interval_to_ctau(coupling, 2.0e-8)
        np.testing.assert_allclose([ctau.lower, ctau.upper], [2.0e-2, 2.0])
        self.assertFalse(ctau.lower_inclusive)
        self.assertTrue(ctau.upper_inclusive)

    @staticmethod
    def _saved_domain_fixture() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model": "ALP-photon-combined",
                    "mass_GeV": 0.3,
                    "event_level": 2.3,
                    "interval_index": 0,
                    "coupling_min_GeV_inv": 0.5,
                    "coupling_max_GeV_inv": 1.0,
                    "unit_coupling_ctau_m": 1.0,
                    "ctau_min_m": 1.0,
                    "ctau_max_m": 4.0,
                },
                {
                    "model": "ALP-photon-combined",
                    "mass_GeV": 0.3,
                    "event_level": 2.3,
                    "interval_index": 1,
                    "coupling_min_GeV_inv": 0.1,
                    "coupling_max_GeV_inv": 0.2,
                    "unit_coupling_ctau_m": 1.0,
                    "ctau_min_m": 25.0,
                    "ctau_max_m": 100.0,
                },
                {
                    "model": "ALP-SU2L",
                    "mass_GeV": 0.3,
                    "event_level": 2.3,
                    "interval_index": 0,
                    "coupling_min_GeV_inv": 0.2,
                    "coupling_max_GeV_inv": 2.0,
                    "unit_coupling_ctau_m": 1.0,
                    "ctau_min_m": 0.25,
                    "ctau_max_m": 25.0,
                },
            ]
        )

    def test_load_allowed_ctau_domains_preserves_disconnected_intervals(
        self,
    ):
        with TemporaryDirectory() as temporary_directory:
            path = (
                Path(temporary_directory)
                / "allowed_ctau_domains.csv"
            )
            self._saved_domain_fixture().to_csv(path, index=False)

            domains = load_allowed_ctau_domains(path)

        photon = domains[
            domains["model"] == "ALP-photon-combined"
        ]

        self.assertEqual(
            photon["interval_index"].tolist(),
            [0, 1],
        )
        self.assertEqual(
            photon["ctau_min_m"].tolist(),
            [1.0, 25.0],
        )
        self.assertEqual(
            photon["ctau_max_m"].tolist(),
            [4.0, 100.0],
        )
        self.assertEqual(
            available_lifetime_domain_masses(domains),
            [0.3],
        )

    def test_build_lifetime_grid_does_not_bridge_excluded_gap(
        self,
    ):
        with TemporaryDirectory() as temporary_directory:
            path = (
                Path(temporary_directory)
                / "allowed_ctau_domains.csv"
            )
            self._saved_domain_fixture().to_csv(path, index=False)

            domains = load_allowed_ctau_domains(path)

        grid = build_lifetime_grid(
            domains,
            model="ALP-photon-combined",
            mass_gev=0.3,
            points_per_interval=3,
        )

        self.assertEqual(len(grid), 6)
        self.assertEqual(
            grid["interval_index"].tolist(),
            [0, 0, 0, 1, 1, 1],
        )
        self.assertEqual(
            grid["global_lifetime_index"].tolist(),
            list(range(6)),
        )

        lifetimes = grid["ctau_m"].to_numpy(dtype=float)

        self.assertTrue(np.isclose(lifetimes[0], 1.0))
        self.assertTrue(np.isclose(lifetimes[2], 4.0))
        self.assertTrue(np.isclose(lifetimes[3], 25.0))
        self.assertTrue(np.isclose(lifetimes[-1], 100.0))

        self.assertFalse(
            np.any(
                (lifetimes > 4.0)
                & (lifetimes < 25.0)
            )
        )

    def test_load_allowed_ctau_domains_rejects_wrong_event_level(
        self,
    ):
        data = self._saved_domain_fixture()
        data.loc[0, "event_level"] = 10.0

        with TemporaryDirectory() as temporary_directory:
            path = (
                Path(temporary_directory)
                / "allowed_ctau_domains.csv"
            )
            data.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "event_level"):
                load_allowed_ctau_domains(path)


if __name__ == "__main__":
    unittest.main()