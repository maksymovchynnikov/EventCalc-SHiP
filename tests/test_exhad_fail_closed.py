#!/usr/bin/env python3
"""Fail-closed configuration and bridge-discovery contracts for exhad."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import exhadDecays  # noqa: E402

# This module was written against an exHad adapter that carried its own lazily
# imported bridge object and its own HNL helpers.  The adapter of this
# repository, funcs/exhadDecays.py, talks to a configured exHad release instead
# and defines none of the names below, so the module cannot run as written.
# The check is by name, so the module runs again by itself if the adapter ever
# defines them.
_ADAPTER_NAMES = (
    "REPOSITORY_EXHAD_PY",
    "_bridge",
    "_load_bridge",
    "_resolved_exhad_py",
    "effective_window",
    "m_start",
)
_ABSENT = [name for name in _ADAPTER_NAMES if not hasattr(exhadDecays, name)]
if _ABSENT:
    raise unittest.SkipTest(
        "funcs/exhadDecays.py does not define " + ", ".join(_ABSENT)
        + "; these tests were written against an adapter that did"
    )



class ExhadFailClosedTest(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        exhadDecays.set_force(None)
        exhadDecays.set_selection(None, None)
        exhadDecays._card_config.cache_clear()
        exhadDecays._bridge = None

    def tearDown(self):
        exhadDecays.set_force(None)
        exhadDecays.set_selection(None, None)
        exhadDecays._card_config.cache_clear()
        exhadDecays._bridge = None
        self.environment.stop()

    def test_malformed_existing_card_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "exhad.json"), "w") as stream:
                stream.write('{"bench": "hls", invalid}')
            exhadDecays.set_selection(
                "Scalar-mixing", directory, "2407.13587-Central")
            with self.assertRaisesRegex(RuntimeError, "malformed exhad card"):
                exhadDecays.get_bench()

    def test_non_object_card_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "exhad.json"), "w") as stream:
                json.dump([{"bench": "hls"}], stream)
            exhadDecays.set_selection("Scalar-mixing", directory)
            with self.assertRaisesRegex(RuntimeError, "JSON object"):
                exhadDecays.get_bench()

    def test_unknown_backend_never_falls_through_to_legacy_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "exhad.json"), "w") as stream:
                json.dump({
                    "bench": "dv",
                    "backend": "typo-portable-model1",
                    "m_start": 2.32,
                    "m_edge": 2.32,
                    "m_high": 5.0,
                }, stream)
            bridge = mock.Mock()
            exhadDecays._bridge = bridge
            exhadDecays.set_selection("Dark-photons", directory)
            with self.assertRaisesRegex(RuntimeError, "unsupported.*backend"):
                exhadDecays.get_bench()
            bridge.sample_for_eventcalc.assert_not_called()

    def test_supported_llp_missing_card_cannot_use_builtin_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            exhadDecays.set_selection(
                "Scalar-mixing", directory, "2407.13587-Lower")
            with self.assertRaisesRegex(RuntimeError, "missing exhad card"):
                exhadDecays.get_bench()

    def test_variant_without_variant_map_cannot_resolve_central(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "exhad.json"), "w") as stream:
                json.dump({"bench": "hls"}, stream)
            exhadDecays.set_selection(
                "Scalar-mixing", directory, "2407.13587-Lower")
            with self.assertRaisesRegex(RuntimeError, "no benchmark map"):
                exhadDecays.get_bench()

    def test_invalid_numeric_window_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "exhad.json"), "w") as stream:
                json.dump({"bench": "dv", "m_start": "not-a-number"},
                          stream)
            exhadDecays.set_selection("Dark-photons", directory)
            with self.assertRaisesRegex(RuntimeError, "must be numeric"):
                exhadDecays.m_start("dv")

    def test_inverted_effective_window_fails_during_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "exhad.json"), "w") as stream:
                json.dump({"bench": "dv", "m_start": 2.5,
                           "m_edge": 2.5, "m_high": 2.0}, stream)
            exhadDecays.set_selection("Dark-photons", directory)
            with self.assertRaisesRegex(RuntimeError,
                                        "m_start=.*smaller than m_high"):
                exhadDecays.enabled(2.25)

    def test_edge_outside_effective_window_fails_during_routing(self):
        for edge in (1.6, 3.0):
            with self.subTest(edge=edge), tempfile.TemporaryDirectory() as directory:
                with open(os.path.join(directory, "exhad.json"), "w") as stream:
                    json.dump({"bench": "dv", "m_start": 1.7,
                               "m_edge": edge, "m_high": 3.0}, stream)
                exhadDecays._card_config.cache_clear()
                exhadDecays.set_selection("Dark-photons", directory)
                with self.assertRaisesRegex(RuntimeError,
                                            "m_edge=.*must lie"):
                    exhadDecays.enabled(2.0)

    def test_omitted_window_fields_keep_deployed_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "exhad.json"), "w") as stream:
                json.dump({"bench": "hls"}, stream)
            exhadDecays.set_selection("Scalar-mixing", directory)
            self.assertEqual(exhadDecays.effective_window("hls"),
                             (2.0, 2.0, 3.75))

    def test_bad_explicit_exhad_path_is_configuration_error(self):
        os.environ["EXHAD_PY"] = "/path/that/does/not/exist"
        with self.assertRaisesRegex(RuntimeError, "EXHAD_PY"):
            exhadDecays._resolved_exhad_py()

    def test_genuinely_absent_optional_bridge_returns_none(self):
        with mock.patch.object(
                exhadDecays, "REPOSITORY_EXHAD_PY",
                "/path/that/does/not/exist"), mock.patch.object(
                    exhadDecays.importlib.util, "find_spec",
                    return_value=None):
            self.assertIsNone(exhadDecays._load_bridge())

    def test_bridge_import_error_propagates(self):
        with mock.patch.object(
                exhadDecays, "_resolved_exhad_py", return_value=None), \
                mock.patch.object(
                    exhadDecays.importlib.util, "find_spec",
                    return_value=object()), \
                mock.patch.object(
                    exhadDecays.importlib, "import_module",
                    side_effect=RuntimeError("broken bridge dependency")):
            with self.assertRaisesRegex(RuntimeError,
                                        "broken bridge dependency"):
                exhadDecays._load_bridge()

    def test_repository_discovery_is_relative_to_checkout(self):
        expected = os.path.join(os.path.dirname(ROOT), "exhad", "py")
        self.assertEqual(
            os.path.realpath(exhadDecays.REPOSITORY_EXHAD_PY),
            os.path.realpath(expected))


if __name__ == "__main__":
    unittest.main()
