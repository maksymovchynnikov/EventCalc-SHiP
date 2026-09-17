#!/usr/bin/env python3
"""Tests for model-independent aggregation of Jets-* selections."""

import contextlib
import io
import json
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs.channel_selection import (  # noqa: E402
    ALL_JETS_LABEL,
    channel_choices,
    prompt_decay_channels,
    resolve_channel_tokens,
    selected_channel_labels,
)
import run_batch  # noqa: E402


CHANNELS = ["2e", "Jets-GG", "2mu", "Jets-ss", "Jets-cc", "2gamma"]


class ChannelSelectionTest(unittest.TestCase):
    def test_dark_photon_table_switches_owners_at_1p70(self):
        path = os.path.join(
            ROOT, "Distributions", "Dark-photons", "DP-decay.json")
        with open(path) as stream:
            table = json.load(stream)
        rows = {
            str(row[0]): {
                round(float(mass), 12): float(weight)
                for mass, weight in row[2]
            }
            for row in table
        }
        jets = [name for name in rows if name.startswith("Jets-")]
        exclusive = [name for name in rows
                     if not name.startswith("Jets-")
                     and not name.endswith("-pair")]

        def total(names, mass):
            key = round(float(mass), 12)
            return sum(rows[name][key] for name in names)

        self.assertGreater(total(exclusive, 1.699), 0.0)
        self.assertAlmostEqual(total(jets, 1.699), 0.0, places=14)
        self.assertAlmostEqual(total(exclusive, 1.70), 0.0, places=14)
        self.assertGreater(total(jets, 1.70), 0.0)

    def test_choices_collapse_every_jet_row_at_first_position(self):
        self.assertEqual(
            channel_choices(CHANNELS),
            [("2e", (0,)),
             (ALL_JETS_LABEL, (1, 3, 4)),
             ("2mu", (2,)),
             ("2gamma", (5,))])

    def test_batch_aggregate_and_alias_expand_all_jet_rows(self):
        self.assertEqual(resolve_channel_tokens(["jets"], CHANNELS),
                         [1, 3, 4])
        self.assertEqual(resolve_channel_tokens(["all-jets"], CHANNELS),
                         [1, 3, 4])
        self.assertEqual(resolve_channel_tokens(["jets", "2mu"], CHANNELS),
                         [1, 2, 3, 4])

    def test_numeric_indices_use_collapsed_display(self):
        self.assertEqual(resolve_channel_tokens(["2"], CHANNELS), [1, 3, 4])
        self.assertEqual(resolve_channel_tokens(["3"], CHANNELS), [2])

    def test_individual_jet_name_is_rejected_with_batch_guidance(self):
        with self.assertRaisesRegex(
                ValueError, r"individual jet channel.*--channels jets"):
            resolve_channel_tokens(["Jets-GG"], CHANNELS)
        with self.assertRaisesRegex(SystemExit, r"--channels jets"):
            run_batch.resolve_channels(["Jets-ss"], CHANNELS)

    def test_all_and_non_jet_choices_are_preserved(self):
        self.assertEqual(resolve_channel_tokens(["all"], CHANNELS),
                         list(range(len(CHANNELS))))
        self.assertEqual(resolve_channel_tokens(["2e", "2gamma"], CHANNELS),
                         [0, 5])
        self.assertEqual(
            selected_channel_labels(list(range(len(CHANNELS))), CHANNELS),
            ["2e", ALL_JETS_LABEL, "2mu", "2gamma"])

    def test_interactive_menu_hides_rows_and_expands_group(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            selected = prompt_decay_channels(CHANNELS, input_fn=lambda _: "2")
        shown = output.getvalue()
        self.assertEqual(selected, [1, 3, 4])
        self.assertIn("2. " + ALL_JETS_LABEL, shown)
        self.assertNotIn("Jets-GG", shown)
        self.assertNotIn("Jets-ss", shown)
        self.assertNotIn("Jets-cc", shown)

    def test_jets_token_requires_a_jet_group(self):
        with self.assertRaisesRegex(ValueError, "no Jets-\\* channels"):
            resolve_channel_tokens(["jets"], ["2e", "2mu"])

    def test_all_shipped_models_resolve_the_complete_jet_group(self):
        fixtures = {
            "ALP-fermion/ALP-fermion-decay.json":
                ["Jets-GG", "Jets-cc", "Jets-ss"],
            "Dark-photons/DP-decay.json":
                ["Jets-cc", "Jets-dd", "Jets-ss", "Jets-uu"],
            "HNL/HNL-decay.json": [
                "Jets-bbv", "Jets-cbe", "Jets-cbebar", "Jets-cbmu",
                "Jets-cbmubar", "Jets-cbtau", "Jets-cbtaubar",
                "Jets-ccv", "Jets-cde", "Jets-cdebar", "Jets-cdmu",
                "Jets-cdmubar", "Jets-cdtau", "Jets-cdtaubar",
                "Jets-cse", "Jets-csebar", "Jets-csmu",
                "Jets-csmubar", "Jets-cstau", "Jets-cstaubar",
                "Jets-ddv", "Jets-ssv", "Jets-ube", "Jets-ubebar",
                "Jets-ubmu", "Jets-ubmubar", "Jets-ubtau",
                "Jets-ubtaubar", "Jets-ude", "Jets-udebar",
                "Jets-udmu", "Jets-udmubar", "Jets-udtau",
                "Jets-udtaubar", "Jets-use", "Jets-usebar",
                "Jets-usmu", "Jets-usmubar", "Jets-ustau",
                "Jets-ustaubar", "Jets-uuv",
            ],
            "Scalar-mixing/BrRatio-Scalar-1809.01876.json":
                ["Jets-cc", "Jets-GG", "Jets-bb", "Jets-ss"],
            "Scalar-mixing/BrRatio-Scalar-2407.13587-Central.json":
                ["Jets-cc", "Jets-GG", "Jets-bb", "Jets-ss"],
            "Scalar-mixing/BrRatio-Scalar-2407.13587-Lower.json":
                ["Jets-cc", "Jets-GG", "Jets-bb", "Jets-ss"],
            "Scalar-mixing/BrRatio-Scalar-2407.13587-Upper.json":
                ["Jets-cc", "Jets-GG", "Jets-bb", "Jets-ss"],
        }
        distributions = os.path.join(ROOT, "Distributions")
        for relative_path, expected in fixtures.items():
            with self.subTest(table=relative_path):
                with open(os.path.join(distributions, relative_path)) as src:
                    names = [str(row[0]) for row in json.load(src)]
                selected = resolve_channel_tokens(["jets"], names)
                self.assertEqual([names[index] for index in selected],
                                 expected)

        with open(os.path.join(
                distributions, "ALP-photon/ALP-photon-decay.json")) as src:
            alp_photon_names = [str(row[0]) for row in json.load(src)]
        with self.assertRaisesRegex(ValueError, "no Jets-\\* channels"):
            resolve_channel_tokens(["jets"], alp_photon_names)


if __name__ == "__main__":
    unittest.main()
