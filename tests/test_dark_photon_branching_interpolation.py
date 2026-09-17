#!/usr/bin/env python3
"""EventCalc-facing contract for dark-photon probability interpolation."""

from pathlib import Path
import sys
import unittest

import numpy as np


try:
    import exhad_matched  # noqa: E402
    from exhad_matched.inputs.dark_photon_bundle import (  # noqa: E402
        build_dark_photon_production_inputs,
    )
    from exhad_matched.inputs.dark_photon_exclusive import (  # noqa: E402
        DELIVER_CHANNEL_IDS,
        dark_photon_probability_vector_at,
    )
except ImportError:
    raise unittest.SkipTest(
        "exhad_matched belongs to the exHad installation and is not part of "
        "this repository; put the directory that contains it on PYTHONPATH "
        "to run these tests"
    )

# the root of the exHad installation the imported package came from
REPOSITORY_ROOT = Path(exhad_matched.__file__).resolve().parents[3]


class DarkPhotonEventCalcBranchingInterpolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = build_dark_photon_production_inputs(
            project_root=REPOSITORY_ROOT)

    def test_prefragmentation_probabilities_are_the_direct_table_pchip(
            self) -> None:
        """No separately optimized curve may alter the tabulated vector."""

        bundle = self.inputs.exclusive_edge.bundle
        channels = {
            channel.channel_id: channel
            for channel in bundle.rates.channels
        }
        independently_evaluated_nucleons = {"p_pbar", "n_nbar"}
        for mass in np.linspace(1.70, 2.00, 121):
            expected = dark_photon_probability_vector_at(
                project_root=REPOSITORY_ROOT,
                mass_gev=float(mass),
            )
            inclusive = bundle.inclusive.width_at(float(mass))
            for family, channel_id in DELIVER_CHANNEL_IDS.items():
                if family in independently_evaluated_nucleons:
                    continue
                actual = channels[channel_id].width_at(
                    float(mass), bundle.rates.support_gev,
                ) / inclusive
                with self.subTest(mass=float(mass), family=family):
                    self.assertAlmostEqual(
                        actual, expected[family], places=13)

            closure = self.inputs.exclusive_edge.closure_at(float(mass))
            self.assertGreaterEqual(closure.residual_width_gev, 0.0)
            self.assertLessEqual(
                abs(closure.closure_error_gev),
                8.0 * np.spacing(closure.inclusive_width_gev),
            )

    def test_nucleon_rows_have_their_physical_thresholds(self) -> None:
        bundle = self.inputs.exclusive_edge.bundle
        channels = {
            channel.channel_id: channel
            for channel in bundle.rates.channels
        }
        for family in ("p_pbar", "n_nbar"):
            channel = channels[DELIVER_CHANNEL_IDS[family]]
            threshold = channel.threshold_gev
            with self.subTest(family=family):
                self.assertEqual(
                    channel.width_at(
                        threshold, bundle.rates.support_gev),
                    0.0,
                )
                self.assertGreater(
                    channel.width_at(
                        threshold + 5.0e-5,
                        bundle.rates.support_gev),
                    0.0,
                )


if __name__ == "__main__":
    unittest.main()
