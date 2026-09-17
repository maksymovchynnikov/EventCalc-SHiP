#!/usr/bin/env python3
"""Real six-current EventCalc/exhad checks through the production endpoint."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for module_name in ("TwoBodyDecay", "ThreeBodyDecay", "FourBodyDecay"):
    module = types.ModuleType("funcs." + module_name)
    module.decay_products = lambda *_args, **_kwargs: None
    sys.modules.setdefault("funcs." + module_name, module)

try:
    import exhad_bridge  # noqa: E402
except ImportError:
    raise unittest.SkipTest(
        "these tests generate real events with the exHad worker: exhad_bridge "
        "and the compiled worker belong to the exHad installation and are not "
        "part of this repository; put the directory that contains "
        "exhad_bridge on PYTHONPATH to run them"
    )
from funcs import exhadDecays  # noqa: E402

EXHAD_ROOT = Path(exhad_bridge.__file__).resolve().parent.parent


def _particle(px, py, pz, energy, mass, pdg):
    return [px, py, pz, energy, mass, float(pdg), 0.0, 1.0]


def _event(pdg_row, exact_w, spectator_energy=0.25):
    active = [int(pdg) for pdg in pdg_row if int(pdg) != -999]
    payload = []
    payload += _particle(0.0, 0.0, 0.0, exact_w / 2.0, 0.0, active[0])
    payload += _particle(0.0, 0.0, 0.0, exact_w / 2.0, 0.0, active[1])
    payload += _particle(
        0.0, 0.0, 0.0, spectator_energy, 0.0, active[2])
    return np.asarray([payload], dtype=np.float64)


def _active_products(row):
    return row.reshape(-1, 6)[row.reshape(-1, 6)[:, 5] != -999.0]


class HNLFullRangeRealEventTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = EXHAD_ROOT / "cpp" / "exhad"
        config = EXHAD_ROOT / "cpp" / ".pythia8-build"
        if (not binary.is_file() or not os.access(binary, os.X_OK)
                or not config.is_file()):
            raise unittest.SkipTest(
                "configured Pythia 8.317 exhad worker is unavailable")

    @classmethod
    def tearDownClass(cls):
        exhad_bridge._close_hnl_exact_w_generator()

    def _generate(self, current, pdgs, exact_w, seed):
        source = _event(pdgs, exact_w)
        with contextlib.redirect_stdout(io.StringIO()):
            output = (
                exhadDecays.process_hnl_cc_with_exhad(
                    source, pdgs, mass=4.5, seed=seed)
                if current.startswith("CC")
                else exhadDecays.process_hnl_nc_with_exhad(
                    source, pdgs, mass=4.5, seed=seed)
            )
        particles = _active_products(output[0])
        spectator = particles[0]
        np.testing.assert_allclose(
            spectator[:4], source.reshape(-1, 8)[2, :4],
            rtol=0.0, atol=0.0)
        self.assertEqual(int(spectator[5]), int(pdgs[2]))
        hadrons = particles[1:]
        np.testing.assert_allclose(
            hadrons[:, :4].sum(axis=0),
            [0.0, 0.0, 0.0, exact_w],
            rtol=0.0,
            atol=2.0e-9,
        )
        observed_w = np.sqrt(max(
            hadrons[:, 3].sum() ** 2
            - np.sum(hadrons[:, :3].sum(axis=0) ** 2),
            0.0,
        ))
        self.assertAlmostEqual(observed_w, exact_w, places=9)
        return particles

    def test_all_six_currents_work_for_a_parent_above_3_GeV(self):
        rows = (
            ("CC_ud", [2, -1, 11, -999], 2.20),
            ("CC_us", [2, -3, 11, -999], 2.20),
            ("CC_cd", [4, -1, 11, -999], 2.50),
            ("CC_cs", [4, -3, 11, -999], 2.70),
            ("NC_ud", [2, -2, 12, -999], 2.20),
            ("NC_s", [3, -3, 12, -999], 2.20),
        )
        for index, (current, pdgs, exact_w) in enumerate(rows):
            with self.subTest(current=current):
                self._generate(current, pdgs, exact_w, 700 + index)

    def test_charm_current_and_conjugate_rows_are_charge_complete(self):
        for index, (current, particle, antiparticle, exact_w) in enumerate((
                ("CC_cd", [4, -1, 11, -999],
                 [-4, 1, -11, -999], 2.50),
                ("CC_cs", [4, -3, 11, -999],
                 [-4, 3, -11, -999], 2.70),
                )):
            with self.subTest(current=current):
                positive = self._generate(
                    current, particle, exact_w, 900 + 2 * index)
                negative = self._generate(
                    current, antiparticle, exact_w, 901 + 2 * index)
                positive_hadronic_charge = sum(
                    exhad_bridge.CHARGE.get(int(pdg), 0)
                    for pdg in positive[1:, 5])
                negative_hadronic_charge = sum(
                    exhad_bridge.CHARGE.get(int(pdg), 0)
                    for pdg in negative[1:, 5])
                self.assertEqual(positive_hadronic_charge, 1)
                self.assertEqual(negative_hadronic_charge, -1)
                self.assertEqual(
                    exhad_bridge.CHARGE[int(positive[0, 5])], -1)
                self.assertEqual(
                    exhad_bridge.CHARGE[int(negative[0, 5])], 1)


if __name__ == "__main__":
    unittest.main()
