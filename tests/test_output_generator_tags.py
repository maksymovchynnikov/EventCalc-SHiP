#!/usr/bin/env python3
"""Focused regression tests for batch output generator provenance."""

import os
import multiprocessing
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_batch  # noqa: E402
from funcs import mergeResults, output_provenance, selecting_processing  # noqa: E402


def _portable_provenance(**overrides):
    result = {
        "schema": output_provenance.GENERATOR_PROVENANCE_SCHEMA,
        "backend": "portable-model1-v1",
        "model_id": "portable-model1-dark-photon-remainder-v1",
        "variant_id": "data-primary-2320mev-remainder",
        "variation_id": "central",
        "contract_path": "/contracts/dark-photon-remainder.json",
        "contract_sha256": "a" * 64,
        "runtime_card_path": "/runtime/dark-photon-central.json",
        "runtime_card_sha256": "b" * 64,
        "route_ownership": {
            "outer_owner": "total-em-rate-authority",
            "model1_owner": "unmeasured-active-remainder-only",
        },
    }
    result.update(overrides)
    return result


def _append_concurrent_total(args):
    directory, filename, values = args
    return mergeResults._append_total_row(
        directory, filename, values, replace_coordinate=True)


class GeneratorTagTest(unittest.TestCase):
    def test_resolved_generator_tags(self):
        self.assertEqual(output_provenance.build_generator_tag(None, 424242),
                         "raw-pythia-seed424242")
        self.assertEqual(
            output_provenance.build_generator_tag(
                None, 424242, eventcalc_direct=True
            ),
            "eventcalc-direct-seed424242",
        )
        self.assertEqual(output_provenance.build_generator_tag("alp", 424242),
                         "exhad-alp-seed424242")
        self.assertEqual(
            output_provenance.build_generator_tag(
                None, 424242, stock_pythia_pool=True
            ),
            "stock-pythia-alp-seed424242",
        )
        with self.assertRaisesRegex(ValueError, "cannot be tagged as exhad"):
            output_provenance.build_generator_tag(
                "alp", 424242, stock_pythia_pool=True
            )

    def test_portable_model_tag_is_complete_and_collision_safe(self):
        central = _portable_provenance()
        tag = output_provenance.build_generator_tag(
            "portable1", 17, generator_provenance=central)
        self.assertEqual(
            tag,
            "exhad-portable1-"
            "m.portable-model1-dark-photon-remainder-v1-"
            "v.data-primary-2320mev-remainder-u.central-"
            "c.aaaaaaaaaaaa-seed17",
        )
        variants = (
            central,
            _portable_provenance(variation_id="saturation-off"),
            _portable_provenance(variation_id="p-low"),
            _portable_provenance(contract_sha256="c" * 64),
        )
        tags = {
            output_provenance.build_generator_tag(
                "portable1", 17, generator_provenance=value)
            for value in variants
        }
        self.assertEqual(len(tags), len(variants))

    def test_portable_model_tag_round_trips_through_analysis_parser(self):
        tag = output_provenance.build_generator_tag(
            "portable1", 73,
            generator_provenance=_portable_provenance(
                variation_id="p-high", contract_sha256="c" * 64))
        parsed = selecting_processing._parse_generator_tag(tag)
        self.assertEqual(parsed["generator"], "exhad")
        self.assertEqual(parsed["backend"], "portable1")
        self.assertEqual(
            parsed["hadronization_model"],
            "portable-model1-dark-photon-remainder-v1")
        self.assertEqual(
            parsed["hadronization_variant"],
            "data-primary-2320mev-remainder")
        self.assertEqual(parsed["hadronization_variation"], "p-high")
        self.assertEqual(parsed["contract_digest"], "c" * 12)
        self.assertEqual(parsed["seed"], 73)

    def test_generator_sidecar_is_atomic_and_legacy_none_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "events.dat")
            self.assertIsNone(
                output_provenance.write_generator_sidecar(output, None))
            self.assertFalse(os.path.exists(output + ".generator.json"))

            provenance = _portable_provenance()
            sidecar = output_provenance.write_generator_sidecar(
                output, provenance)
            self.assertEqual(sidecar, output + ".generator.json")
            with open(sidecar, encoding="utf-8") as stream:
                payload = __import__("json").load(stream)
            self.assertEqual(
                payload["schema"],
                output_provenance.GENERATOR_PROVENANCE_SCHEMA)
            self.assertEqual(payload["output_file"], "events.dat")
            self.assertEqual(payload["generator"], provenance)

    def test_portable_provenance_rejects_bad_digest_and_unsafe_id(self):
        with self.assertRaisesRegex(ValueError, "contract_sha256"):
            output_provenance.build_generator_tag(
                "portable1", 1,
                generator_provenance=_portable_provenance(
                    contract_sha256="abc"))
        with self.assertRaisesRegex(ValueError, "variation_id"):
            output_provenance.build_generator_tag(
                "portable1", 1,
                generator_provenance=_portable_provenance(
                    variation_id="../../upper"))

    def test_flux_and_hadronization_variations_are_orthogonal(self):
        paths = set()
        tags = {}
        for flux in ("lower", "upper"):
            for variation in ("central", "p-high"):
                tag = output_provenance.build_generator_tag(
                    "portable1", 17,
                    generator_provenance=_portable_provenance(
                        variation_id=variation))
                tags[(flux, variation)] = tag
                paths.add(mergeResults._event_output_path(
                    "/tmp/events", "Dark-photons", 2.5, 10.0,
                    None, "primary-" + flux, None,
                    generator_tag=tag, channel_tag="channels-all"))
        self.assertEqual(len(paths), 4)
        self.assertEqual(tags[("lower", "central")],
                         tags[("upper", "central")])
        self.assertNotEqual(tags[("lower", "central")],
                            tags[("lower", "p-high")])

    def test_stock_pool_request_is_strictly_scoped(self):
        run_batch.validate_stock_pythia_pool_request(
            True, "ALP-fermion", "off", ["all"], [1.911, 2.0]
        )
        invalid = (
            ("Dark-photons", "off", ["all"], [2.0]),
            ("ALP-fermion", "on", ["all"], [2.0]),
            ("ALP-fermion", "off", ["jets"], [2.0]),
            ("ALP-fermion", "off", ["all"], [1.9]),
        )
        for llp, exhad, channels, masses in invalid:
            with self.subTest(
                llp=llp, exhad=exhad, channels=channels, masses=masses
            ):
                with self.assertRaises(ValueError):
                    run_batch.validate_stock_pythia_pool_request(
                        True, llp, exhad, channels, masses
                    )

    def test_event_paths_separate_raw_and_exhad(self):
        common = ("/tmp/events", "ALP-fermion", 2.0, 1.0,
                  None, None, None)
        raw = mergeResults._event_output_path(
            *common, generator_tag="raw-pythia-seed424242")
        exhad = mergeResults._event_output_path(
            *common, generator_tag="exhad-alp-seed424242")
        self.assertEqual(os.path.basename(raw),
                         "ALP-fermion_2.000e+00_1.000e+00_"
                         "raw-pythia-seed424242_data.dat")
        self.assertEqual(
            os.path.basename(exhad),
            "ALP-fermion_2.000e+00_1.000e+00_exhad-alp-seed424242_data.dat")
        self.assertNotEqual(raw, exhad)

    def test_nearby_output_coordinates_do_not_collide(self):
        first = mergeResults._event_output_path(
            "/tmp/events", "ALP-fermion", 2.0001, 10.0,
            None, None, None, generator_tag="raw-pythia-seed1")
        second = mergeResults._event_output_path(
            "/tmp/events", "ALP-fermion", 2.0002, 10.0,
            None, None, None, generator_tag="raw-pythia-seed1")
        shifted_ctau = mergeResults._event_output_path(
            "/tmp/events", "ALP-fermion", 2.0001, 10.001,
            None, None, None, generator_tag="raw-pythia-seed1")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, shifted_ctau)

    def test_total_paths_separate_mode_and_seed(self):
        raw = mergeResults._total_filename(
            "ALP-fermion", None, None, None, "raw-pythia-seed1")
        seed_one = mergeResults._total_filename(
            "ALP-fermion", None, None, None, "exhad-alp-seed1")
        seed_two = mergeResults._total_filename(
            "ALP-fermion", None, None, None, "exhad-alp-seed2")
        self.assertEqual(raw, "ALP-fermion_raw-pythia-seed1_total.txt")
        self.assertEqual(seed_one, "ALP-fermion_exhad-alp-seed1_total.txt")
        self.assertNotEqual(seed_one, seed_two)

    def test_channel_selection_tags_are_canonical_and_collision_safe(self):
        channels = np.array(["a", "b", "c", "d"], dtype=object)
        self.assertEqual(
            mergeResults.channel_selection_tag(channels, [0, 1, 2, 3]),
            "channels-all")
        self.assertEqual(
            mergeResults.channel_selection_tag(channels, [2, 0]),
            "channels-i1.3")
        self.assertEqual(
            mergeResults.channel_selection_tag(channels, [0, 2]),
            mergeResults.channel_selection_tag(channels, [2, 0]))
        self.assertNotEqual(
            mergeResults.channel_selection_tag(channels, [0, 2]),
            mergeResults.channel_selection_tag(channels, [0, 3]))

    def test_event_and_total_paths_separate_channel_selections(self):
        common = (
            "/tmp/events", "Dark-photons", 2.0, 10.0, None,
            "combined-central", None)
        all_path = mergeResults._event_output_path(
            *common, generator_tag="exhad-dv-seed1",
            channel_tag="channels-all")
        subset_path = mergeResults._event_output_path(
            *common, generator_tag="exhad-dv-seed1",
            channel_tag="channels-i5.6.7")
        self.assertNotEqual(all_path, subset_path)
        self.assertTrue(
            all_path.endswith(
                "_combined-central_exhad-dv-seed1_channels-all_data.dat"))

        total = mergeResults._total_filename(
            "Dark-photons", None, "combined-central", None,
            "exhad-dv-seed1", "channels-i5.6.7")
        self.assertEqual(
            total,
            "Dark-photons_combined-central_exhad-dv-seed1_"
            "channels-i5.6.7_total.txt")

    def test_interactive_callers_keep_historical_names(self):
        mixing = np.array([1.0, 0.0, 0.0])
        event_path = mergeResults._event_output_path(
            "/tmp/events", "HNL", 2.0, 1.0, mixing, None, None)
        total_name = mergeResults._total_filename(
            "HNL", mixing, None, None)
        self.assertEqual(
            os.path.basename(event_path),
            "HNL_2.000e+00_1.000e+00_1.000e+00_0.000e+00_0.000e+00_data.dat")
        self.assertEqual(
            total_name,
            "HNL_1.000e+00_0.000e+00_0.000e+00_total.txt")

    def test_unsafe_tag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "generator_tag"):
            mergeResults._total_filename(
                "ALP-fermion", None, None, None, "../../overwrite")

    def test_explicit_exhad_request_is_fail_closed(self):
        bridge = mock.Mock()
        bridge.can_use_exhad.return_value = False
        with self.assertRaisesRegex(RuntimeError, "--exhad on"):
            run_batch.resolve_exhad_selection(bridge, "on", "ALP-fermion")
        bridge.set_force.assert_called_once_with(True)

    def test_auto_missing_bridge_is_explicit_raw_mode(self):
        bridge = mock.Mock()
        bridge.can_use_exhad.return_value = False
        bench, status = run_batch.resolve_exhad_selection(
            bridge, "auto", "ALP-fermion")
        self.assertIsNone(bench)
        self.assertIn("EventCalc baseline", status)
        bridge.set_force.assert_called_once_with(None)

    def test_explicit_on_is_a_noop_when_exhad_is_not_applicable(self):
        bridge = mock.Mock()
        bench, status = run_batch.resolve_exhad_selection(
            bridge, "on", "ALP-photon", applicable=False
        )
        self.assertIsNone(bench)
        self.assertIn("not applicable", status)
        bridge.set_force.assert_called_once_with(False)
        bridge.can_use_exhad.assert_not_called()

    def test_resolved_scalar_variant_is_preserved_in_tag(self):
        bridge = mock.Mock()
        bridge.can_use_exhad.return_value = True
        bridge.get_bench.return_value = "hls-lower"
        bench, _ = run_batch.resolve_exhad_selection(
            bridge, "on", "Scalar-mixing")
        self.assertEqual(
            output_provenance.build_generator_tag(bench, 17),
            "exhad-hls-lower-seed17")

    def test_run_batch_gates_both_sides_of_hnl_matched_scope(self):
        bridge = mock.Mock()
        bridge.require_hnl_parent_masses_supported.side_effect = RuntimeError(
            "exhad generates HNL decays for HNL mass 0.02 to 40 GeV")
        for mass in (0.019, 40.1):
            with self.subTest(mass=mass):
                with self.assertRaisesRegex(RuntimeError, "HNL mass"):
                    run_batch.enforce_hnl_matched_mass_scope(
                        bridge, "HNL", "hnl", [mass])
        self.assertEqual(
            bridge.require_hnl_parent_masses_supported.call_args_list,
            [mock.call([0.019]), mock.call([40.1])])

    def test_raw_hnl_scan_does_not_apply_matched_scope_gate(self):
        bridge = mock.Mock()
        run_batch.enforce_hnl_matched_mass_scope(
            bridge, "HNL", None, [0.019, 40.1])
        bridge.require_hnl_parent_masses_supported.assert_not_called()

    def test_untagged_total_rows_retain_legacy_append_semantics(self):
        values = [2.0, 1e-6, 10.0, 4.0, 0.7, 0.8, 0.2, 0.4, 3.0]
        with tempfile.TemporaryDirectory() as directory:
            filename = "deterministic_total.txt"
            self.assertTrue(mergeResults._append_total_row(
                directory, filename, values))
            path = os.path.join(directory, filename)
            with open(path, "rb") as stream:
                first = stream.read()

            self.assertFalse(mergeResults._append_total_row(
                directory, filename, values))
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), first)

            changed = list(values)
            changed[-1] = 4.0
            self.assertTrue(mergeResults._append_total_row(
                directory, filename, changed))
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(len(stream.readlines()), 3)

    def test_tagged_total_rows_replace_and_compact_same_coordinate(self):
        values = [2.0, 1e-6, 10.0, 4.0, 0.7, 0.8, 0.2, 0.4, 3.0]
        changed = list(values)
        changed[1] = 2e-6
        changed[-1] = 4.0
        other = list(values)
        other[0] = 3.0

        with tempfile.TemporaryDirectory() as directory:
            filename = "tagged_total.txt"
            self.assertTrue(mergeResults._append_total_row(
                directory, filename, values, replace_coordinate=True))
            self.assertTrue(mergeResults._append_total_row(
                directory, filename, changed, replace_coordinate=True))
            self.assertTrue(mergeResults._append_total_row(
                directory, filename, other, replace_coordinate=True))

            path = os.path.join(directory, filename)
            with open(path, encoding="utf-8") as stream:
                lines = stream.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(
                lines[1],
                ' '.join("{:.17e}".format(x) for x in changed) + "\n")

            # Simulate a pre-fix file with two historical rows at one key.
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(
                    ' '.join("{:.9e}".format(x) for x in values) + "\n")
            self.assertTrue(mergeResults._append_total_row(
                directory, filename, changed, replace_coordinate=True))
            with open(path, encoding="utf-8") as stream:
                compacted = stream.readlines()
            self.assertEqual(len(compacted), 3)
            self.assertFalse(mergeResults._append_total_row(
                directory, filename, changed, replace_coordinate=True))

            nearby = list(changed)
            nearby[0] = 2.0001
            self.assertTrue(mergeResults._append_total_row(
                directory, filename, nearby, replace_coordinate=True))
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(len(stream.readlines()), 4)

    @unittest.skipIf(mergeResults.fcntl is None, "requires POSIX file locks")
    def test_parallel_total_updates_do_not_lose_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = "parallel_total.txt"
            requests = []
            for index in range(12):
                values = [
                    1.0 + index / 10.0, 1e-6, 10.0, 4.0, 0.7,
                    0.8, 0.2, 0.4, 3.0,
                ]
                requests.append((directory, filename, values))
            context = multiprocessing.get_context("fork")
            with context.Pool(processes=4) as pool:
                self.assertTrue(all(pool.map(
                    _append_concurrent_total, requests)))

            path = os.path.join(directory, filename)
            with open(path, encoding="utf-8") as stream:
                lines = stream.readlines()
            self.assertEqual(len(lines), len(requests) + 1)
            self.assertEqual(
                sorted(float(line.split()[0]) for line in lines[1:]),
                sorted(request[2][0] for request in requests))

    def test_failed_event_write_preserves_previous_complete_file(self):
        mothers = np.zeros((1, 10), dtype=float)
        decays = np.zeros((1, 6), dtype=float)
        channels = np.array(["visible"], dtype=object)
        with tempfile.TemporaryDirectory() as directory:
            event_dir = os.path.join(
                directory, "outputs", "ALP-photon", "eventData")
            os.makedirs(event_dir)
            target = mergeResults._event_output_path(
                event_dir, "ALP-photon", 1.0, 10.0, None, None,
                "combined", "raw-pythia-seed1", "channels-all")
            with open(target, "w", encoding="utf-8") as stream:
                stream.write("previous-complete-output\n")

            with mock.patch.object(
                    mergeResults, "_write_event_rows",
                    side_effect=RuntimeError("injected write failure")):
                with self.assertRaisesRegex(
                        RuntimeError, "injected write failure"):
                    mergeResults.save(
                        mothers, decays, "ALP-photon", 1.0, None, 10.0,
                        channels, [1], 1, 1.0, 1.0, 8.0, 4.0, 0.5,
                        2.0, 1.0, [0], None, True, "combined",
                        generator_tag="raw-pythia-seed1",
                        channel_tag="channels-all",
                        output_root=directory)
            with open(target, encoding="utf-8") as stream:
                self.assertEqual(
                    stream.read(), "previous-complete-output\n")


if __name__ == "__main__":
    unittest.main()
