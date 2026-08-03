import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis2.cache import CacheStore
from analysis2.config import SMOKE, spectrum_model_seed
from analysis2.eventcalc_adapter import EventCalcAdapter
from analysis2.models import MODELS
from analysis2.statistics import simulate_shape_discrimination
from analysis2.templates import cached_probability_templates


@unittest.skipUnless(os.environ.get("EVENTCALC_RUN_SMOKE") == "1", "set EVENTCALC_RUN_SMOKE=1")
class SmokeIntegrationTests(unittest.TestCase):
    def test_real_eventcalc_to_templates_and_pseudoexperiments(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheStore("smoke", Path(directory) / "smoke")
            adapter = EventCalcAdapter(SMOKE, cache=cache)
            spectra = {
                model.identifier: adapter.evaluate_model(
                    model.identifier, 0.3, 20.0, spectrum_model_seed(0, model_index), "spectrum"
                )
                for model_index, model in enumerate(MODELS)
            }
            self.assertTrue(all(spectrum.expected_events > 0.0 for spectrum in spectra.values()))
            edges = np.geomspace(0.3, SMOKE.energy_max_gev, SMOKE.initial_energy_bins + 1)
            templates = cached_probability_templates(
                cache, spectra, edges, SMOKE.discrimination.minimum_bin_n_eff,
                SMOKE.discrimination.jeffreys_alpha,
            )
            photon, su2 = (templates[model.identifier].probabilities for model in MODELS)
            result = simulate_shape_discrimination(photon, su2, 3, 200, 123)
            self.assertEqual(len(result.number_of_events), 3)
            self.assertAlmostEqual(photon.sum(), 1.0)
            self.assertAlmostEqual(su2.sum(), 1.0)
            # The second request must be safely reusable from Level A.
            repeated = adapter.evaluate_model(MODELS[0].identifier, 0.3, 20.0, spectrum_model_seed(0, 0), "spectrum")
            self.assertEqual(repeated.cache_key, spectra[MODELS[0].identifier].cache_key)


if __name__ == "__main__":
    unittest.main()
