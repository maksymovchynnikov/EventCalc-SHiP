#!/usr/bin/env python3
"""Regression tests for truthful matched-event process headers."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import mergeResults  # noqa: E402


class MatchedProcessLabelTest(unittest.TestCase):
    def test_adjacent_matched_rows_are_written_as_one_aggregate_block(self):
        # The numerical marker in column zero makes row-order preservation
        # observable independently of the process headers.
        mothers = np.zeros((4, 8), dtype=np.float64)
        mothers[:, 0] = [10.0, 20.0, 30.0, 40.0]
        decays = np.zeros((4, 6), dtype=np.float64)
        decay_channels = np.array(
            ["Jets-dd", "Jets-ss", "gamma-gamma"], dtype=object)

        with tempfile.TemporaryDirectory() as directory, (
            mock.patch.object(mergeResults, "_append_total_row")
        ):
            # the writer takes its output root; no caller changes directory
            mergeResults.save(
                mothers, decays, "Dark-photons", 2.0, None, 1.0,
                decay_channels, [1, 2, 1], 4, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, [0, 1, 2], None, True,
                process_labels=["Jets-matched", "Jets-matched", None],
                output_root=directory)

            event_dir = os.path.join(
                directory, "outputs", "Dark-photons", "eventData")
            output_path = os.path.join(event_dir, os.listdir(event_dir)[0])
            with open(output_path, encoding="utf-8") as stream:
                contents = stream.read()

        self.assertEqual(contents.count("process=Jets-matched"), 1)
        self.assertIn("process=Jets-matched; sample_points=3", contents)
        self.assertIn("process=gamma-gamma; sample_points=1", contents)
        self.assertNotIn("process=Jets-dd", contents)
        self.assertNotIn("process=Jets-ss", contents)

        rows = [
            line for line in contents.splitlines()
            if line and not line.startswith(("Sampled ", "#"))
        ]
        self.assertEqual([float(line.split()[0]) for line in rows],
                         [10.0, 20.0, 30.0, 40.0])

    def test_missing_overrides_preserve_raw_process_names(self):
        blocks = mergeResults._output_process_blocks(
            ["Jets-dd", "Jets-ss"], [0, 1], [2, 3])
        self.assertEqual(
            blocks,
            [("Jets-dd", 0, 2), ("Jets-ss", 2, 3)])


if __name__ == "__main__":
    unittest.main()
