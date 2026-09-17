#!/usr/bin/env python3
"""Regression tests for exhad bridge temporary-file handling."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import exhad_bridge  # noqa: E402
except ImportError:
    raise unittest.SkipTest(
        "exhad_bridge belongs to the exHad installation and is not part of "
        "this repository; put the directory that contains it on PYTHONPATH "
        "to run these tests"
    )


def _output_path(command):
    return command[command.index("--out") + 1]


class ExhadBridgeResourceTest(unittest.TestCase):
    def test_child_runtime_replaces_parent_pythia_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp, "pythia8")
            xmldoc = prefix / "share" / "Pythia8" / "xmldoc"
            libdir = prefix / "lib"
            xmldoc.mkdir(parents=True)
            libdir.mkdir()
            (xmldoc / "Version.xml").write_text(
                '<parmfix name="Pythia:versionNumber" default="8.317">\n',
                encoding="utf-8")
            (libdir / "libpythia8.dylib").write_bytes(b"fixture")
            config = Path(tmp, "build")
            config.write_text(
                "prefix=%s\nxmldoc=%s\nlibdir=%s\nversion=8.317\n" %
                (prefix, xmldoc, libdir), encoding="utf-8")
            parent = {
                "PATH": "/usr/bin",
                "PYTHIA8DATA": "/parent/stale-pythia/xml",
                "EXHAD_PYTHIA8DATA": "/parent/override/xml",
                "PYTHIA8_DIR": "/parent/stale-pythia",
                "DYLD_LIBRARY_PATH": "/parent/stale-pythia/lib:/other",
                "DYLD_FALLBACK_LIBRARY_PATH": "/parent/fallback",
                "LD_LIBRARY_PATH": "/parent/linux-pythia/lib",
                "LD_PRELOAD": "/parent/libpythia8.so",
            }
            child = exhad_bridge._exhad_child_environment(
                parent, build_config=str(config), platform="darwin")

        self.assertEqual(child["PYTHIA8DATA"], str(xmldoc.resolve()))
        self.assertEqual(child["EXHAD_PYTHIA8DATA"], str(xmldoc.resolve()))
        self.assertEqual(child["PYTHIA8_DIR"], str(prefix.resolve()))
        self.assertEqual(child["DYLD_LIBRARY_PATH"], str(libdir.resolve()))
        self.assertNotIn("DYLD_FALLBACK_LIBRARY_PATH", child)
        self.assertNotIn("LD_LIBRARY_PATH", child)
        self.assertNotIn("LD_PRELOAD", child)
        self.assertEqual(parent["PYTHIA8DATA"],
                         "/parent/stale-pythia/xml")

    def test_child_runtime_fails_closed_on_build_xml_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmldoc = Path(tmp, "xml")
            libdir = Path(tmp, "lib")
            xmldoc.mkdir()
            libdir.mkdir()
            (xmldoc / "Version.xml").write_text(
                '<parmfix name="Pythia:versionNumber" default="8.316">\n',
                encoding="utf-8")
            (libdir / "libpythia8.dylib").write_bytes(b"fixture")
            config = Path(tmp, "build")
            config.write_text(
                "xmldoc=%s\nlibdir=%s\nversion=8.317\n" %
                (xmldoc, libdir), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,
                                        "build/XML mismatch"):
                exhad_bridge._exhad_child_environment(
                    {}, build_config=str(config), platform="darwin")

    def test_success_removes_reserved_output_file(self):
        seen = []

        def fake_run(command, **_kwargs):
            path = _output_path(command)
            seen.append(path)
            with open(path, "w") as stream:
                stream.write("# event 0 mass 2 channel test\n")
                stream.write("211 0 0 0 0.13957\n")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(exhad_bridge.subprocess, "run",
                               side_effect=fake_run):
            events = exhad_bridge._one_batch("alp", 2.0, 1, 7)

        self.assertEqual(len(events), 1)
        self.assertEqual(len(seen), 1)
        self.assertFalse(os.path.exists(seen[0]))

    def test_failure_removes_reserved_output_file(self):
        seen = []

        def fake_run(command, **_kwargs):
            seen.append(_output_path(command))
            return types.SimpleNamespace(
                returncode=1, stdout="failed",
                stderr="ALP morph provenance mismatch")

        with mock.patch.object(exhad_bridge.subprocess, "run",
                               side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError,
                                        "morph provenance mismatch"):
                exhad_bridge._one_batch("alp", 2.0, 1, 7)

        self.assertEqual(len(seen), 1)
        self.assertFalse(os.path.exists(seen[0]))

    def test_short_batch_fails_instead_of_silently_continuing(self):
        def fake_run(command, **_kwargs):
            with open(_output_path(command), "w") as stream:
                stream.write("# event 0 mass 2 channel test\n")
                stream.write("211 0 0 0 0.13957\n")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(exhad_bridge.subprocess, "run",
                               side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError,
                                        "returned 1 events, but 2 were requested"):
                exhad_bridge._one_batch("alp", 2.0, 2, 7)


if __name__ == "__main__":
    unittest.main()
