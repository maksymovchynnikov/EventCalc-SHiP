import json
import os
import tempfile
import unittest
from unittest import mock

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import exhadDecays  # noqa: E402

# This module was written against an exHad adapter that carried its own lazily
# imported bridge object.  The adapter of this repository, funcs/exhadDecays.py,
# talks to a configured exHad release instead and defines none of the names
# below, so the module cannot run as written.  The check is by name, so the
# module runs again by itself if the adapter ever defines them.
_ADAPTER_NAMES = ("STABLE_TERMINAL_COUNTING_MODEL", "stable_terminal_counting_config")
_ABSENT = [name for name in _ADAPTER_NAMES if not hasattr(exhadDecays, name)]
if _ABSENT:
    raise unittest.SkipTest(
        "funcs/exhadDecays.py does not define " + ", ".join(_ABSENT)
        + "; these tests were written against an adapter that did"
    )



def _alp_card(**updates):
    card = {
        "bench": "alp",
        "backend": "matched-spinzero-v1",
        "model_portal": "ALP-fermion",
        "model_variant": "bc10-fermion-universal-alp",
        "m_start": 1.911,
        "m_edge": 1.911,
        "m_H": 4.0,
        "m_high": 5.0,
    }
    card.update(updates)
    return card


class StableTerminalCountingConfigTests(unittest.TestCase):
    def tearDown(self):
        exhadDecays.set_selection(None, None, None)
        exhadDecays.set_force(None)
        exhadDecays._card_config.cache_clear()

    def _select(self, card, llp="ALP-fermion"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        with open(os.path.join(temporary.name, "exhad.json"), "w") as stream:
            json.dump(card, stream)
        exhadDecays._card_config.cache_clear()
        exhadDecays.set_selection(llp, temporary.name, None)

    def test_option_is_off_when_absent(self):
        self._select(_alp_card())
        self.assertIsNone(exhadDecays.stable_terminal_counting_config("alp"))

    def test_versioned_opt_in_requires_the_declared_edge(self):
        self._select(_alp_card(
            stable_terminal_counting=(
                exhadDecays.STABLE_TERMINAL_COUNTING_MODEL),
            stable_terminal_counting_edge_gev=1.911,
        ))
        self.assertEqual(
            exhadDecays.stable_terminal_counting_config("alp"),
            {
                "model": exhadDecays.STABLE_TERMINAL_COUNTING_MODEL,
                "matching_mass_gev": 1.911,
            },
        )

    def test_enabled_card_reaches_spinzero_bridge_with_runtime_options(self):
        self._select(_alp_card(
            stable_terminal_counting=(
                exhadDecays.STABLE_TERMINAL_COUNTING_MODEL),
            stable_terminal_counting_edge_gev=1.911,
        ))
        bridge = mock.Mock()
        bridge.sample_spinzero_for_eventcalc.return_value = [
            [0.0, 0.0, 0.0, 3.0, 3.0, 22.0],
        ]
        with mock.patch.object(
                exhadDecays, "_load_bridge", return_value=bridge), (
                mock.patch.object(exhadDecays, "_bridge", bridge)):
            result = exhadDecays.process_events_with_exhad(
                1, 3.0, seed=29)

        self.assertEqual(result.shape[0], 1)
        bridge.sample_spinzero_for_eventcalc.assert_called_once_with(
            "ALP-fermion",
            "bc10-fermion-universal-alp",
            3.0,
            1,
            seed=29,
            stable_terminal_counting_model=(
                exhadDecays.STABLE_TERMINAL_COUNTING_MODEL),
            stable_terminal_counting_edge_gev=1.911,
        )

    def test_arbitrary_matching_mass_is_rejected(self):
        self._select(_alp_card(
            stable_terminal_counting=(
                exhadDecays.STABLE_TERMINAL_COUNTING_MODEL),
            stable_terminal_counting_edge_gev=2.32,
        ))
        with self.assertRaisesRegex(RuntimeError, "portal matching mass"):
            exhadDecays.stable_terminal_counting_config("alp")

    def test_historical_threshold_model_is_not_an_alias(self):
        self._select(_alp_card(final_state_brodsky_model=True))
        with self.assertRaisesRegex(RuntimeError, "historical dark-photon"):
            exhadDecays.stable_terminal_counting_config("alp")

    def test_fitted_full_range_em_authority_cannot_be_reweighted(self):
        card = {
            "bench": "dv",
            "backend": "matched-dark-photon-em-v1",
            "m_inclusive": 1.7,
            "m_frag": 2.0,
            "m_H": 4.0,
            "m_free": 5.0,
            "frozen_em_authority": "authority.json",
            "stable_terminal_counting": (
                exhadDecays.STABLE_TERMINAL_COUNTING_MODEL),
            "stable_terminal_counting_edge_gev": 2.0,
        }
        self._select(card, llp="Dark-photons")
        with self.assertRaisesRegex(RuntimeError, "fitted full-range"):
            exhadDecays.stable_terminal_counting_config("dv")


if __name__ == "__main__":
    unittest.main()
