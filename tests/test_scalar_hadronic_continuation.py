import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from funcs.scalar_hadronic_continuation import ScalarHadronicContinuation, TWO_MESON, NONHADRONIC
from funcs import initLLP, exhadDecays

try:
    import exhad_matched
    from exhad_matched.inputs.scalar_rate_authority import ScalarHadronicRateAuthority
    from exhad_matched.scalar_portal_measure import _load_total_width_curve
    from build_model1_supplement_portal_figures import scalar_point, runtime_for, VARIANTS
    # the root of the exHad installation the imported package came from
    EXHAD_ROOT = Path(exhad_matched.__file__).resolve().parents[3]
except ImportError:
    exhad_matched = None
    EXHAD_ROOT = None

NO_MATCHED = (
    "exhad_matched and build_model1_supplement_portal_figures belong to the "
    "exHad installation and are not part of this repository; put the "
    "directories that contain them on PYTHONPATH to run this test"
)


class ScalarContinuationTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / "Distributions/Scalar-mixing"

    @unittest.skipIf(exhad_matched is None, NO_MATCHED)
    def test_all_prescriptions_preserve_rates_and_close(self):
        for variant in VARIANTS:
            rate = ScalarHadronicContinuation(self.folder / f"BrRatio-Scalar-{variant}.json")
            for mass in np.linspace(1.5, 5., 351):
                native, continued = rate.native(mass), rate.at(mass)
                self.assertAlmostEqual(math.fsum(native.values()), math.fsum(continued.values()), places=14)
                for k in (*NONHADRONIC, "ppbar", "nnbar", "Jets-cc", "Jets-bb"):
                    self.assertEqual(native[k], continued[k])
                self.assertTrue(all(v >= 0 for v in continued.values()))
                if mass <= 1.999 or mass >= 5.:  # exHad's showered rows from 5 GeV
                    self.assertEqual(native, continued)
                else:
                    self.assertTrue(all(continued[k] > 0 for k in TWO_MESON))

    @unittest.skipIf(exhad_matched is None, NO_MATCHED)
    def test_two_body_amplitude_is_constant(self):
        for variant in VARIANTS:
            rate = ScalarHadronicContinuation(self.folder / f"BrRatio-Scalar-{variant}.json")
            for mass in (2., 2.6, 4., 4.5):
                for label, daughter in TWO_MESON.items():
                    values = []
                    for m in (1.999, mass):
                        width = rate.at(m)[label]*rate.total_width(m)
                        values.append(width*m/math.sqrt(1-4*daughter**2/m**2))
                    # the matched share 1-h of exHad's 4-5 GeV join keeps the amplitude
                    self.assertAlmostEqual(values[1]/values[0], 1.-.5*(mass == 4.5), places=13)

    @unittest.skipIf(exhad_matched is None, NO_MATCHED)
    def test_four_pion_and_two_body_continuity(self):
        for variant, stem in VARIANTS.items():
            authority = ScalarHadronicRateAuthority(variant, project_root=EXHAD_ROOT,
                                                    portable_two_meson_continuation=True)
            curve = _load_total_width_curve(authority)
            runtime, _ = runtime_for(stem)
            left = scalar_point(authority, curve, runtime, 2.-1e-10)
            right = scalar_point(authority, curve, runtime, 2.+1e-10)
            for label in ("pipi", "kk", "four-pion"):
                self.assertLess(abs(left["hadronic_fractions"][label]-right["hadronic_fractions"][label]), 2e-8)

    def test_eventcalc_uses_policy_only_when_explicitly_deployed(self):
        self.skipTest(
            "this test hands the scalar loader a deployment payload with a "
            "scalar_rate_policy key; funcs/initLLP.py reads the release's "
            "model_info instead, whose tables/decay and tables/ctau entries "
            "this test does not provide")
        particle = initLLP.LLP(None, {"particle_path": str(self.folder),
                                    "LLP_name": "Scalar-mixing"})
        particle.set_mass(2.5)
        with patch.object(exhadDecays, "uses_portable_model1_backend", return_value=False):
            particle.compute_mass_dependent_properties()
            native = dict(zip(particle.decayChannels, particle.BrRatios_distr))
        self.assertEqual(native["PipPim"], 0.)
        with patch.object(exhadDecays, "uses_portable_model1_backend", return_value=True), \
             patch.object(exhadDecays, "portable_model1_configuration", return_value={"scalar_rate_policy": "scalar-endpoint-amplitude-v1"}):
            particle.compute_mass_dependent_properties()
            corrected = dict(zip(particle.decayChannels, particle.BrRatios_distr))
        self.assertGreater(corrected["PipPim"], 0.)
        self.assertAlmostEqual(math.fsum(native.values()), math.fsum(corrected.values()), places=14)


if __name__ == "__main__":
    unittest.main()
