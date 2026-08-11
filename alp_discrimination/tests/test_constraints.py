import json
import tempfile
from pathlib import Path
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from alp_discrimination.constraints.convert import COUPLING_CONVERSION_FACTOR, convert_constraint
from alp_discrimination.constraints.plotting import (
    LABEL_CONFIG_PATH, LABEL_CONTEXTS, MODEL_SPECS, PHOTON_SPECS,
    draw_constraints, load_constraint, load_label_config,
)
from alp_discrimination.constraints.bc9 import load_bc9_polygon

import matplotlib.pyplot as plt  # noqa: E402


class ConstraintTests(unittest.TestCase):
    def test_zero_mass_polygon_edge_is_allowed_but_negative_mass_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constraint.txt"
            np.savetxt(path, [[0.0, 1e-4], [0.2, 2e-4]])
            np.testing.assert_allclose(load_constraint(path)[:, 0], [0.0, 0.2])
            np.savetxt(path, [[-0.1, 1e-4], [0.2, 2e-4]])
            with self.assertRaisesRegex(ValueError, "invalid constraint polygon"):
                load_constraint(path)

    def test_su2_conversion_changes_only_coupling(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "in.txt", Path(directory) / "out.txt"
            np.savetxt(source, [[0.3, 1e-6], [0.4, 2e-6]])
            convert_constraint(source, target)
            converted = np.loadtxt(target)
            np.testing.assert_allclose(converted[:, 0], [0.3, 0.4])
            np.testing.assert_allclose(converted[:, 1], [1e-6, 2e-6] * np.asarray(COUPLING_CONVERSION_FACTOR))

    def test_default_label_config_is_complete_and_uses_axes_fractions(self):
        config = load_label_config()
        self.assertEqual(set(config), set(MODEL_SPECS))
        for model, specs in MODEL_SPECS.items():
            self.assertEqual(set(config[model]), {filename for filename, _ in specs})
            for filename, label in specs:
                self.assertEqual(set(config[model][filename]), set(LABEL_CONTEXTS))
                for placement in config[model][filename].values():
                    self.assertEqual(placement.text, label)
                    self.assertTrue(0.0 <= placement.x <= 1.0)
                    self.assertTrue(0.0 <= placement.y <= 1.0)
                    self.assertGreaterEqual(placement.fontsize, 12.0)

    def test_label_config_rejects_missing_context_and_non_axes_coordinates(self):
        original = json.loads(LABEL_CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            del original["models"]["alp_photon"]["bounds_E141.txt"]["positions"]["constraint_only"]
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must define"):
                load_label_config(path)

            original = json.loads(LABEL_CONFIG_PATH.read_text(encoding="utf-8"))
            original["models"]["alp_su2l"]["bounds_CDF.txt"]["positions"]["constraint_only"]["coordinate_system"] = "data"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be 'axes'"):
                load_label_config(path)

    def test_draw_constraints_uses_configured_axes_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / PHOTON_SPECS[0][0]
            np.savetxt(path, [[0.1, 1e-5], [0.2, 2e-5], [0.1, 2e-5]])
            figure, axis = plt.subplots()
            try:
                draw_constraints(
                    axis, path.parent, PHOTON_SPECS[:1], model="alp_photon",
                    context="event_density_overlay", config=load_label_config(),
                )
                self.assertEqual(len(axis.texts), 1)
                placement = load_label_config()["alp_photon"][PHOTON_SPECS[0][0]]["event_density_overlay"]
                self.assertEqual(axis.texts[0].get_position(), (placement.x, placement.y))
                self.assertIs(axis.texts[0].get_transform(), axis.transAxes)
                self.assertEqual(axis.texts[0].get_text(), "BESIII")
            finally:
                plt.close(figure)
    def test_bc9_loader_accepts_decimal_and_rational_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Constraints_BC9_test.csv"

            path.write_text(
                '"mass","coupling"\n'
                '0.1,1.0e-6\n'
                '0.2,1/5000\n'
                '0.3,2.0e-6\n',
                encoding="utf-8",
            )

            data = load_bc9_polygon(path)

            np.testing.assert_allclose(
                data[:, 0],
                [0.1, 0.2, 0.3],
            )
            np.testing.assert_allclose(
                data[:, 1],
                [1.0e-6, 2.0e-4, 2.0e-6],
            )


if __name__ == "__main__":
    unittest.main()
