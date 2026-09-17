#!/usr/bin/env python3
"""Regression tests for legacy/tagged EventCalc analysis filenames."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from funcs.selecting_processing import (  # noqa: E402
    OutputFileMetadata,
    discover_output_files,
    filter_output_files,
    parse_event_filename,
    parse_filenames,
    parse_total_filename,
)
from funcs import mergeResults  # noqa: E402


def _load_script(module_name, filename):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(ROOT, filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OutputFilenameParserTest(unittest.TestCase):
    def test_legacy_hnl_shape_and_key_are_preserved(self):
        name = (
            "HNL_2.000e+00_1.000e+01_"
            "1.000e+00_0.000e+00_0.000e+00_data.dat"
        )
        record = parse_event_filename(name)
        self.assertTrue(record.is_legacy)
        self.assertEqual(record.mixing_pattern, (1.0, 0.0, 0.0))
        self.assertEqual(record.selector, (1.0, 0.0, 0.0))

        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory, "HNL", "eventData")
            event_dir.mkdir(parents=True)
            Path(event_dir, name).touch()
            index = parse_filenames(directory)
            variants = index["HNL"][(2.0, 10.0)]
            self.assertIn((1.0, 0.0, 0.0), variants)
            self.assertTrue(variants[(1.0, 0.0, 0.0)].endswith(name))

    def test_generator_and_channel_tags_parse_in_either_order(self):
        suffixes = (
            "exhad-hnl-seed17-srcdeadbeef_channels-i5.6.7",
            "channels-i5.6.7_exhad-hnl-seed17-srcdeadbeef",
        )
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                record = parse_event_filename(
                    "HNL_2.000e+00_1.000e+01_"
                    "1.000e+00_0.000e+00_0.000e+00_"
                    f"{suffix}_data.dat"
                )
                self.assertEqual(record.generator, "exhad")
                self.assertEqual(record.backend, "hnl")
                self.assertEqual(record.seed, 17)
                self.assertEqual(record.source_digest, "deadbeef")
                self.assertEqual(record.channel_tag, "channels-i5.6.7")
                self.assertEqual(record.channel_indices, (5, 6, 7))

    def test_long_channel_identity_is_preserved(self):
        record = parse_total_filename(
            "ALP-fermion_raw-pythia-seed9_"
            "channels-n12-0123456789abcdef_total.txt"
        )
        self.assertEqual(record.generator, "raw-pythia")
        self.assertEqual(record.backend, "pythia")
        self.assertEqual(record.channel_count, 12)
        self.assertEqual(record.channel_digest, "0123456789abcdef")

    def test_direct_eventcalc_tag_is_distinct_from_legacy_raw_pythia(self):
        direct = parse_event_filename(
            "ALP-photon_1.000e-01_1.000e+01_primary_"
            "eventcalc-direct-seed7_channels-all_data.dat"
        )
        raw = parse_event_filename(
            "ALP-photon_1.000e-01_1.000e+01_primary_"
            "raw-pythia-seed7_channels-all_data.dat"
        )
        self.assertEqual(direct.generator, "eventcalc-direct")
        self.assertEqual(direct.backend, "direct")
        self.assertEqual(raw.generator, "raw-pythia")
        self.assertEqual(raw.backend, "pythia")

    def test_dark_photon_current_and_future_modes(self):
        labels = (
            ("central", "primary"),
            ("cascade-central", "cascade"),
            ("brem-cascade-central", "brem-cascade"),
            ("combined-central", "combined"),
        )
        for label, production in labels:
            with self.subTest(label=label):
                record = parse_event_filename(
                    "Dark-photons_2.000e+00_1.000e+01_"
                    f"{label}_raw-pythia-seed4_channels-all_data.dat"
                )
                self.assertEqual(record.production_mode, production)
                self.assertEqual(record.uncertainty, "central")
                self.assertEqual(record.channel_tag, "channels-all")

    def test_alp_combined_alias_canonicalizes_without_losing_label(self):
        for alias in ("primary+cascade", "cascade+primary"):
            with self.subTest(alias=alias):
                record = parse_event_filename(
                    "ALP-photon_1.000e+00_1.000e+03_"
                    f"{alias}_raw-pythia-seed7_channels-all_data.dat"
                )
                self.assertEqual(record.physics_label, alias)
                self.assertEqual(record.production_mode, "combined")
                self.assertEqual(
                    filter_output_files([record], physics_label="combined"),
                    [record],
                )

    def test_tagged_variants_do_not_collapse_at_one_coordinate(self):
        names = (
            "ALP-fermion_2.000e+00_1.000e+01_"
            "raw-pythia-seed1_channels-all_data.dat",
            "ALP-fermion_2.000e+00_1.000e+01_"
            "exhad-alp-seed1_channels-all_data.dat",
            "ALP-fermion_2.000e+00_1.000e+01_"
            "exhad-alp-seed1_channels-i5.6.7_data.dat",
        )
        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory, "ALP-fermion", "eventData")
            event_dir.mkdir(parents=True)
            for name in names:
                Path(event_dir, name).touch()

            variants = parse_filenames(directory)["ALP-fermion"][(2.0, 10.0)]
            self.assertEqual(len(variants), 3)
            metadata = list(variants)
            self.assertTrue(
                all(isinstance(item, OutputFileMetadata) for item in metadata)
            )
            self.assertEqual(
                {item.generator for item in metadata},
                {"raw-pythia", "exhad"},
            )
            self.assertEqual(
                {item.channel_tag for item in metadata},
                {"channels-all", "channels-i5.6.7"},
            )

    def test_parser_round_trips_writer_event_and_total_names(self):
        event_path = mergeResults._event_output_path(
            "/tmp/events",
            "Dark-photons",
            2.0,
            10.0,
            None,
            "combined-central",
            None,
            generator_tag="exhad-dv-seed1",
            channel_tag="channels-i5.6.7",
        )
        total_name = mergeResults._total_filename(
            "Dark-photons",
            None,
            "combined-central",
            None,
            generator_tag="exhad-dv-seed1",
            channel_tag="channels-i5.6.7",
        )
        for record in (
            parse_event_filename(event_path),
            parse_total_filename(total_name),
        ):
            self.assertEqual(record.production_mode, "combined")
            self.assertEqual(record.uncertainty, "central")
            self.assertEqual(record.generator_tag, "exhad-dv-seed1")
            self.assertEqual(record.channel_indices, (5, 6, 7))


class AnalysisConsumerMigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_plots = _load_script("eventcalc_total_plots", "total-plots.py")
        cls.cascade = _load_script(
            "eventcalc_cascade_analysis", "analyze_cascade_vs_primary.py"
        )
        cls.distributions = _load_script(
            "eventcalc_event_distributions", "plot_event_distributions.py"
        )

    def test_total_plots_extracts_legacy_and_tagged_files(self):
        names = (
            "Dark-photons_central_total.txt",
            "Dark-photons_cascade-central_"
            "exhad-dv-seed4_channels-all_total.txt",
            "Dark-photons_combined-central_"
            "channels-i5.6.7_raw-pythia-seed4_total.txt",
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                Path(directory, name).touch()
            extracted = self.total_plots.extract_files(
                directory, "Dark-photons"
            )
            self.assertEqual(len(extracted), 3)
            identifiers = {identifier for _, identifier in extracted}
            self.assertIn("uncertainty=central", identifiers)
            self.assertTrue(
                any(
                    "production=cascade" in value
                    and "generator=exhad-dv-seed4" in value
                    and "channels=all" in value
                    for value in identifiers
                )
            )
            self.assertTrue(
                any("production=combined" in value for value in identifiers)
            )

    def test_cascade_total_resolver_requires_or_uses_metadata(self):
        names = (
            "Dark-photons_combined-central_"
            "raw-pythia-seed4_channels-all_total.txt",
            "Dark-photons_combined-central_"
            "exhad-dv-seed4_channels-all_total.txt",
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                Path(directory, name).touch()
            with self.assertRaisesRegex(RuntimeError, "multiple"):
                self.cascade.resolve_total_path(
                    "combined-central",
                    total_dir=directory,
                    generator_tag=None,
                    channel_tag="all",
                )
            path = self.cascade.resolve_total_path(
                "combined-central",
                total_dir=directory,
                generator_tag="exhad-dv-seed4",
                channel_tag="all",
            )
            self.assertTrue(path.endswith(names[1]))

    def test_distribution_resolver_selects_exact_tagged_variant(self):
        names = (
            "Dark-photons_5.000e-01_1.000e+03_"
            "brem-cascade-central_raw-pythia-seed4_channels-all_data.dat",
            "Dark-photons_5.000e-01_1.000e+03_"
            "brem-cascade-central_exhad-dv-seed4_channels-all_data.dat",
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                Path(directory, name).touch()
            path = self.distributions.resolve_event_path(
                0.5,
                "brem-cascade-central",
                c_tau=1000.0,
                event_data_dir=directory,
                generator_tag="raw-pythia-seed4",
                channel_tag="channels-all",
            )
            self.assertTrue(path.endswith(names[0]))

    def test_discovery_ignores_sidecars_and_retains_all_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(
                directory,
                "ALP-fermion_exhad-alp-seed1_channels-all_total.txt",
            ).touch()
            Path(
                directory,
                "ALP-fermion_exhad-alp-seed1_channels-all_total.txt"
                ".coupling.json",
            ).touch()
            records = discover_output_files(
                directory, "total", llp_name="ALP-fermion"
            )
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
