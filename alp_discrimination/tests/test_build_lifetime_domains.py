import unittest

import pandas as pd

from alp_discrimination.workflows.build_lifetime_domains import (
    _common_supported_mass_range,
)


class BuildWeek8DomainWorkflowTests(unittest.TestCase):
    def test_common_supported_mass_range_uses_continuous_overlap(self):
        boundaries = pd.DataFrame({
            "model": [
                "ALP-photon-combined",
                "ALP-photon-combined",
                "ALP-photon-combined",
                "ALP-SU2L",
                "ALP-SU2L",
            ],
            "mass_GeV": [
                0.20,
                0.50,
                1.00,
                0.30,
                0.80,
            ],
            "event_level": [2.3] * 5,
            "status": ["resolved"] * 5,
        })

        scan = pd.DataFrame({
            "model": [
                "ALP-photon-combined",
                "ALP-photon-combined",
                "ALP-SU2L",
                "ALP-SU2L",
            ],
            "mass_GeV": [
                0.10,
                1.20,
                0.20,
                1.00,
            ],
        })

        result = _common_supported_mass_range(
            boundaries,
            scan,
            2.3,
        )

        self.assertEqual(result, (0.30, 0.80))


if __name__ == "__main__":
    unittest.main()