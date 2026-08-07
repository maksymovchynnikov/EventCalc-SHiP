import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis2.workflows.compare_photon_sensitivity_references import (
    _validate_analysis2_provenance,
    load_saved_eventcalc_boundaries,
)


def _boundary_table(lower: float, upper: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["ALP-photon"],
            "mass_GeV": [0.3],
            "event_level": [2.3],
            "lower_coupling_GeV_inv": [lower],
            "upper_coupling_GeV_inv": [upper],
        }
    )


class PhotonSensitivityComparisonTests(unittest.TestCase):
    def test_maps_saved_contours_to_matching_selections(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            epsilon_path = root / "epsilon.csv"
            geom_path = root / "geom.csv"
            _boundary_table(1.0e-6, 2.0e-4).to_csv(epsilon_path, index=False)
            _boundary_table(2.0e-6, 1.0e-4).to_csv(geom_path, index=False)

            loaded = load_saved_eventcalc_boundaries(
                epsilon_path,
                geom_path,
            )

            self.assertEqual(set(loaded), {"epsilon_dec_1", "geom_only"})
            self.assertEqual(
                float(loaded["epsilon_dec_1"].iloc[0]["lower_coupling_GeV_inv"]),
                1.0e-6,
            )
            self.assertEqual(
                float(loaded["geom_only"].iloc[0]["lower_coupling_GeV_inv"]),
                2.0e-6,
            )

    def test_requires_mother_level_analysis2_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            event_dir = Path(temporary_directory)
            (event_dir / "manifest.json").write_text(
                json.dumps({"selection_name": "mother_level"}),
                encoding="utf-8",
            )
            _validate_analysis2_provenance(event_dir)

            (event_dir / "manifest.json").write_text(
                json.dumps({"selection_name": "diphoton_ecal"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mother_level"):
                _validate_analysis2_provenance(event_dir)


if __name__ == "__main__":
    unittest.main()