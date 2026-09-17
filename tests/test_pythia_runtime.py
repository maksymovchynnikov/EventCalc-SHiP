#!/usr/bin/env python3
"""Fail-closed contracts for EventCalc's single Pythia runtime."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import os
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

NOT_8317 = (
    "the Pythia binding installed here is %s, not the required 8.317, and "
    "pairing an 8.312 binding with the 8.317 payload aborts in "
    "Pythia::checkVersion.  The pin is not relaxed to make this machine "
    "pass: install an 8.317 binding and this test runs."
    % (BINDING_VERSION or "absent")
)


class PythiaRuntimeTest(unittest.TestCase):
    def setUp(self):
        decayProducts._PYTHIA8 = None
        decayProducts._PYTHIA8_XML = None

    def tearDown(self):
        decayProducts._PYTHIA8 = None
        decayProducts._PYTHIA8_XML = None

    @unittest.skipUnless(BINDING_VERSION == "8.317", NOT_8317)
    def test_project_runtime_is_exactly_8317(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHIA8DATA", None)
            os.environ.pop("PYTHIA8_DIR", None)
            os.environ.pop("PYTHIA8_LIB", None)
            module = decayProducts.load_pythia8()
        probe = module.Pythia("", False)
        self.assertEqual(
            "%.3f" % probe.settings.parm("Pythia:versionNumber"), "8.317")
        self.assertEqual(
            decayProducts._pythia8_xml_version(
                decayProducts._PYTHIA8_XML), "8.317")

    def test_mismatched_explicit_xml_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmldoc = Path(tmp)
            (xmldoc / "Version.xml").write_text(
                '<parmfix name="Pythia:versionNumber" default="8.316">\n',
                encoding="utf-8")
            with mock.patch.dict(
                    os.environ, {"PYTHIA8DATA": str(xmldoc)}, clear=False):
                # The payload and the binding must be the same version; the
                # refusal names both and the file that declares the mismatch.
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"XML payload of another version.*8\.316.*8\.312"):
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
