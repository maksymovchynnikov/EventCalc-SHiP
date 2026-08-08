import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from alp_discrimination.event_density import (
    build_boundary_table, endpoint_refinement_masses, find_level_crossings,
)
from alp_discrimination.config import SMOKE
from alp_discrimination.workflows.scan_event_density import (
    SOURCE_SCANS, load_resumable_source_rows, scan_configuration_key,
)


class EventDensityTests(unittest.TestCase):
    def test_log_log_crossings_and_boundary_status(self):
        data = pd.DataFrame({
            "model": "toy", "mass_GeV": 1.0,
            "coupling_GeV_inv": [1.0, 10.0, 100.0], "N_events": [1.0, 100.0, 1.0],
        })
        np.testing.assert_allclose(find_level_crossings(data, 10.0), [np.sqrt(10.0), 10 * np.sqrt(10.0)])
        boundary = build_boundary_table(data, (10.0,)).iloc[0]
        self.assertEqual(boundary.status, "resolved")
        self.assertEqual(boundary.number_of_crossings, 2)

    def test_endpoint_refinement_adds_interior_masses(self):
        boundaries = pd.DataFrame({
            "model": ["toy", "toy"], "event_level": [10.0, 10.0],
            "mass_GeV": [1.0, 2.0], "status": ["resolved", "outside_mass_reach"],
        })
        result = endpoint_refinement_masses(boundaries, 3, 0.1)
        np.testing.assert_allclose(result["toy"], [1.25, 1.5, 1.75])

    def test_resume_accepts_complete_groups_and_rejects_partial_groups(self):
        definition = SOURCE_SCANS[0]
        couplings = np.geomspace(
            definition.coupling_min_gev_inv, definition.coupling_max_gev_inv,
            SMOKE.event_density.coupling_points,
        )
        rows = []
        for mass, values in ((0.1, couplings), (0.2, couplings[:-1])):
            for coupling in values:
                rows.append({
                    "profile": SMOKE.name, "selection_name": SMOKE.selection_name,
                    "scan_configuration_key": scan_configuration_key(SMOKE),
                    "model": definition.identifier, "mass_GeV": mass,
                    "coupling_GeV_inv": coupling, "N_events": 1.0,
                })
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            pd.DataFrame(rows).to_csv(
                output_dir / "event_density_scan_sources_checkpoint.csv", index=False,
            )
            resumed = load_resumable_source_rows(output_dir, SMOKE)
        self.assertEqual(len(resumed), len(couplings))
        self.assertEqual(set(resumed["mass_GeV"]), {0.1})


if __name__ == "__main__":
    unittest.main()
