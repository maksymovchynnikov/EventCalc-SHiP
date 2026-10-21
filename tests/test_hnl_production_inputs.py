"""SPS HNL production inputs use the same tungsten target as SensCalc."""
from pathlib import Path
import unittest

import numpy as np


INPUTS = Path(__file__).resolve().parents[1] / "Distributions" / "HNL"


class HNLProductionInputsTest(unittest.TestCase):
    def test_tungsten_yields_include_the_unscaled_kaon_component(self):
        # Independent SensCalc ProductionWeightsHNL evaluation at 0.02,
        # 0.1 and 1 GeV. Charm and bottom use A=184; kaons retain Y_K=0.36.
        expected = {
            "e": [0.005122276418947407, 0.1033671630174029, 0.0002839313579791945],
            "mu": [0.11732445734775267, 0.2142100278795929, 0.0002836125731458833],
            "tau": [7.956085868271027e-5, 7.23877206121253e-5, 2.90861721858536e-6],
        }
        for flavour, values in expected.items():
            with self.subTest(flavour=flavour):
                table = np.loadtxt(INPUTS / f"Total-yield-HNL-{flavour}.txt")
                np.testing.assert_allclose(
                    np.interp([0.02, 0.1, 1.], table[:, 0], table[:, 1]),
                    values, rtol=2e-12, atol=0.)

    def test_shapes_and_energy_bounds_share_the_regenerated_mass_grid(self):
        reference = None
        for flavour in ("e", "mu", "tau"):
            with np.load(INPUTS / f"DoubleDistr-HNL-mixing-{flavour}.npz") as table:
                self.assertTrue(np.isfinite(table["density"]).all())
                self.assertTrue((table["density"] >= 0.).all())
                if reference is None:
                    reference = table["mass"].copy()
                np.testing.assert_array_equal(table["mass"], reference)
                self.assertTrue(np.any(np.isclose(reference, 0.48825711, atol=1e-14, rtol=0.)))
        endpoint = np.loadtxt(INPUTS / "Emax-HNL.txt")
        np.testing.assert_allclose(np.unique(endpoint[:, 0]), reference, rtol=1e-14)
        self.assertTrue(np.isfinite(endpoint).all())
        self.assertTrue((endpoint[:, 2] >= endpoint[:, 0]).all())
        self.assertLessEqual(endpoint[:, 2].max(), 400.)


if __name__ == "__main__":
    unittest.main()
