#!/usr/bin/env python3
"""Fail-closed contracts for EventCalc's single Pythia runtime."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import decayProducts  # noqa: E402


def installed_binding_version():
    """The version of the importable Pythia binding, or None if there is none."""
    try:
        module = decayProducts._import_pythia8_binding()
        probe = module.Pythia("", False)
        return "%.3f" % probe.settings.parm("Pythia:versionNumber")
    except Exception:
        return None


BINDING_VERSION = installed_binding_version()

class PythiaRuntimeTest(unittest.TestCase):
    def setUp(self):
        decayProducts._PYTHIA8 = None
        decayProducts._PYTHIA8_XML = None

    def tearDown(self):
        decayProducts._PYTHIA8 = None
        decayProducts._PYTHIA8_XML = None

    @unittest.skipUnless(BINDING_VERSION, "Pythia Python binding is not installed")
    def test_installed_binding_and_xml_match_without_exporting_host_paths(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHIA8DATA", None)
            os.environ.pop("PYTHIA8_DIR", None)
            os.environ.pop("PYTHIA8_LIB", None)
            module = decayProducts.load_pythia8()
            self.assertNotIn("PYTHIA8DATA", os.environ)
        probe = module.Pythia(decayProducts._PYTHIA8_XML or "", False)
        self.assertEqual(
            "%.3f" % probe.settings.parm("Pythia:versionNumber"), BINDING_VERSION)
        if decayProducts._PYTHIA8_XML is not None:
            self.assertEqual(decayProducts._pythia8_xml_version(
                decayProducts._PYTHIA8_XML), BINDING_VERSION)

    def test_mismatched_explicit_xml_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmldoc = Path(tmp)
            (xmldoc / "Version.xml").write_text(
                '<parmfix name="Pythia:versionNumber" default="8.316">\n',
                encoding="utf-8")
            binding = SimpleNamespace(__file__="/test/pythia8.so")
            with mock.patch.object(decayProducts, "_import_pythia8_binding",
                                   return_value=binding), \
                    mock.patch.object(decayProducts, "_candidate_pythia8_xml_dirs",
                                      return_value=[xmldoc]), \
                    mock.patch.object(decayProducts, "_pythia8_accepts_xml",
                                      return_value=(False, "8.317")):
                # The payload and the binding must be the same version; the
                # refusal names both and the file that declares the mismatch.
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"XML payload of another version.*8\.316.*8\.317"):
                    decayProducts.load_pythia8()

    def test_cern_module_name_is_supported_after_wheel_name_is_absent(self):
        cern_binding = SimpleNamespace(__file__="/cvmfs/lcg/lib/pythia8.so")

        def import_module(name):
            if name == "pythia8mc":
                raise ModuleNotFoundError(name)
            if name == "pythia8":
                return cern_binding
            self.fail("unexpected module import: %s" % name)

        with mock.patch.object(
                decayProducts.importlib, "import_module",
                side_effect=import_module) as imported:
            self.assertIs(decayProducts._import_pythia8_binding(), cern_binding)
        self.assertEqual(
            [call.args[0] for call in imported.call_args_list],
            ["pythia8mc", "pythia8"])


if __name__ == "__main__":
    unittest.main()
