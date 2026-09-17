#!/usr/bin/env python3
"""Focused regression tests for the fail-closed HNL source contract."""

import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs.hnl_source_validation import (  # noqa: E402
    HNLSourceValidationError,
    load_canonical_hnl_source,
    load_validated_hnl_source,
    validate_decay_rows,
    validate_width_rows,
)


# The installed HNL tables are the exHad release's own
# data/hnl/eventcalc_branching_ratios.json and data/hnl/eventcalc_total_widths.dat,
# byte for byte, so the rates EventCalc samples and the final states the release
# produces come from one table set.  These are the digests of those two files.
EXPECTED_JSON_SHA256 = \
    "7a206bdf31eb107e771e76c50f58df0cc756e7bb4613323ebe2526036a215a9a"
EXPECTED_WIDTH_SHA256 = \
    "77cf93693b478cdc30a730ffcd64a07e5d55c5d240d43636034c363e2266b096"


def _replace_branching(rows, channel, mixing_index, mass, value):
    """Return a shallow copy with one branching point replaced."""

    changed = list(rows)
    row_index = next(index for index, row in enumerate(changed)
                     if row[0] == channel)
    row = list(changed[row_index])
    table = list(row[2 + mixing_index])
    point_index = next(index for index, point in enumerate(table)
                       if point[0] == mass)
    table[point_index] = [mass, value]
    row[2 + mixing_index] = table
    changed[row_index] = row
    return changed


def _replace_matrix_element(rows, channel, mixing_index, expression):
    changed = list(rows)
    row_index = next(index for index, row in enumerate(changed)
                     if row[0] == channel)
    row = list(changed[row_index])
    row[5 + mixing_index] = expression
    changed[row_index] = row
    return changed


class HNLSourceValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_canonical_hnl_source()

    def test_canonical_source_and_metadata_pass(self):
        self.assertEqual(len(self.source.decay_rows), 88)
        self.assertEqual(len(self.source.mass_grid), 1724)
        self.assertEqual(self.source.decay_json_sha256,
                         EXPECTED_JSON_SHA256)
        self.assertEqual(self.source.width_table_sha256,
                         EXPECTED_WIDTH_SHA256)
        self.assertIsNotNone(self.source.metadata_path)
        self.assertEqual(self.source.metadata_path.name, "HNL-source.json")

    def test_published_source_identifiers_are_portable(self):
        metadata_path = os.path.join(
            ROOT, "Distributions", "HNL", "HNL-source.json")
        with open(metadata_path, encoding="utf-8") as stream:
            source = json.load(stream)["source"]
        for key in ("decay_json", "width_table", "notebook"):
            with self.subTest(key=key):
                self.assertFalse(os.path.isabs(source[key]))
                self.assertEqual(source[key], os.path.basename(source[key]))

    def test_metadata_hash_mismatch_fails_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = os.path.join(directory, "HNL-source.json")
            with open(metadata_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "source": {
                        "decay_json_sha256": "0" * 64,
                        "width_table_sha256": EXPECTED_WIDTH_SHA256,
                    }
                }, stream)
            with self.assertRaisesRegex(
                    HNLSourceValidationError, "SHA-256 mismatch"):
                load_validated_hnl_source(
                    os.path.join(ROOT, "Distributions", "HNL",
                                 "HNL-decay.json"),
                    os.path.join(ROOT, "Distributions", "HNL",
                                 "HNLdecayWidth.dat"),
                    metadata_path=metadata_path,
                )

    def test_subthreshold_d_plus_is_rejected(self):
        # Preserve conjugate equality so this specifically exercises the
        # m(D+) + m(e) support check that caught the stale D0 mass.
        bad_rows = _replace_branching(
            self.source.decay_rows, "De", 0, 1.87, 1.0e-12)
        bad_rows = _replace_branching(
            bad_rows, "Debar", 0, 1.87, 1.0e-12)
        with self.assertRaisesRegex(
                HNLSourceValidationError, "below physical threshold"):
            validate_decay_rows(bad_rows)

    def test_matrix_element_identifier_whitelist_is_enforced(self):
        bad_rows = _replace_matrix_element(
            self.source.decay_rows, "2ev", 0, "abs(E1)")
        with self.assertRaisesRegex(
                HNLSourceValidationError, "disallowed matrix-element syntax"):
            validate_decay_rows(bad_rows)

    def test_conjugate_branching_difference_is_rejected(self):
        original = next(row for row in self.source.decay_rows
                        if row[0] == "Piebar")
        point = next(point for point in original[2] if point[0] == 1.0)
        bad_rows = _replace_branching(
            self.source.decay_rows, "Piebar", 0, 1.0,
            point[1] + 1.0e-12)
        with self.assertRaisesRegex(
                HNLSourceValidationError, "branching table differs"):
            validate_decay_rows(bad_rows)

    def test_width_grid_must_match_exactly(self):
        bad_widths = list(self.source.width_rows)
        first = list(bad_widths[0])
        first[0] += 1.0e-12
        bad_widths[0] = tuple(first)
        with self.assertRaisesRegex(
                HNLSourceValidationError, "width grid differs"):
            validate_width_rows(bad_widths, self.source.mass_grid)


if __name__ == "__main__":
    unittest.main()
