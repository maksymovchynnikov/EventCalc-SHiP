#!/usr/bin/env python3
"""Focused tests for the interim SHiP spectrometer-only post-processor."""

import importlib.util
import copy
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


try:
    import ship_spectrometer_only_interim as ship
except ImportError:
    raise unittest.SkipTest(
        "ship_spectrometer_only_interim belongs to the exHad installation and "
        "is not part of this repository; put the directory that contains it "
        "on PYTHONPATH to run these tests"
    )


def event_row(weight, products, padding="raw", vertex=(0.0, 0.0, 50.0)):
    p4 = np.sum(np.asarray(products, dtype=float)[:, :4], axis=0)
    mass2 = p4[3] ** 2 - np.dot(p4[:3], p4[:3])
    mother = [*p4[:4], math.sqrt(max(mass2, 0.0)), 12345678, weight, *vertex]
    records = [list(product) for product in products]
    if padding == "raw":
        records.append([0.0, 0.0, 0.0, 0.0, 0.0, -999.0])
    elif padding == "exhad":
        records.append([0.0] * 6)
    return " ".join(str(value) for value in mother + sum(records, []))


CH_PLUS = [0.0, 0.0, 1.0, 1.01, 0.139, 211]
CH_MINUS = [0.0, 0.0, 1.0, 1.01, 0.139, -211]
PHOTON = [0.0, 0.0, 1.0, 1.0, 0.0, 22]
PHOTON_SAME_P4 = [0.0, 0.0, 1.0, 1.01, 0.0, 22]
MISS = [1.0, 0.0, 1.0, 1.5, 0.139, 211]
ANTIPROTON = [0.0, 0.0, 1.0, 1.5, 0.938, -2212]
ANTINEUTRON = [0.0, 0.0, 1.0, 1.5, 0.940, -2112]
NEUTRINO = [0.0, 0.0, 1.0, 1.0, 0.0, 12]


class InterimShipAnalysisTests(unittest.TestCase):
    def write_dp_primary_inputs(self, directory, emax_nodes=None):
        base = Path(directory) / "Distributions" / "Dark-photons"
        base.mkdir(parents=True)
        nodes = (1.65, 1.699, 1.75)
        for stem, values in (
            ("DoubleDistr", nodes),
            ("Emax", nodes if emax_nodes is None else emax_nodes),
            ("Total-yield", (1.698, 1.700, 1.702)),
        ):
            path = base / (stem + "-DP-central.txt")
            with open(path, "w") as stream:
                for value in values:
                    stream.write("%.6f 1.0\n" % value)
                if stem == "DoubleDistr":
                    # Repeated first-column values are normal in this table.
                    stream.write("1.699000 2.0\n")
        return base

    def write_sample(self, directory):
        path = Path(directory) / "events.dat"
        with open(path, "w") as stream:
            stream.write("Sampled 5 events inside SHiP volume.\n\n")
            stream.write("#<process=Jets-uu; sample_points=4>\n\n")
            stream.write(event_row(1.0, [CH_PLUS, CH_MINUS], "raw") + "\n")
            stream.write(
                event_row(2.0, [CH_PLUS, CH_MINUS, CH_PLUS, CH_MINUS], "exhad")
                + "\n"
            )
            stream.write(event_row(3.0, [CH_PLUS, PHOTON], "raw") + "\n")
            stream.write(event_row(4.0, [CH_PLUS, MISS], "exhad") + "\n\n")
            # This unchanged leptonic row must not enter the hadronic denominator.
            stream.write("#<process=e-pair; sample_points=1>\n\n")
            stream.write(event_row(100.0, [CH_PLUS, CH_MINUS], "raw") + "\n")
        return path

    def test_exact_definitions_padding_and_weighted_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ship.analyze_file(
                self.write_sample(tmp), nonhadronic_channels=("e-pair",))

        self.assertEqual(result["n_events"], 4)
        self.assertEqual(result["skipped_nonhadronic_events"], 1)
        self.assertAlmostEqual(result["sum_weight"], 10.0)
        self.assertAlmostEqual(result["two_charged"]["fraction"], 0.1)
        self.assertAlmostEqual(result["spectrometer_only"]["fraction"], 0.3)
        self.assertAlmostEqual(
            ship.publication_track_fraction(result, "exactly_two_tracks"),
            0.1,
        )
        self.assertAlmostEqual(
            ship.publication_track_fraction(result, "four_or_more_tracks"),
            0.2,
        )
        two_factor = result["factorization"]["two_charged"]
        all_factor = result["factorization"]["spectrometer_only"]
        self.assertAlmostEqual(two_factor["topology_fraction"]["fraction"], 0.5)
        self.assertAlmostEqual(
            two_factor["conditional_geometry_acceptance"]["fraction"], 0.2,
        )
        self.assertAlmostEqual(
            two_factor["topology_fraction"]["fraction"]
            * two_factor["conditional_geometry_acceptance"]["fraction"],
            two_factor["accepted_fraction"]["fraction"],
        )
        self.assertAlmostEqual(all_factor["topology_fraction"]["fraction"], 0.7)
        self.assertAlmostEqual(
            all_factor["conditional_geometry_acceptance"]["fraction"], 3.0 / 7.0,
        )
        self.assertAlmostEqual(
            all_factor["topology_fraction"]["fraction"]
            * all_factor["conditional_geometry_acceptance"]["fraction"],
            all_factor["accepted_fraction"]["fraction"],
        )
        species = result["selected_stable_species_fractions"]
        self.assertAlmostEqual(species["pi+pi-"]["fraction"], 0.1)
        self.assertAlmostEqual(species["4pi_charged"]["fraction"], 0.2)

        expected_two_variance = (
            1.0**2 * 0.9**2
            + 2.0**2 * 0.1**2
            + 3.0**2 * 0.1**2
            + 4.0**2 * 0.1**2
        ) / 10.0**2
        expected_spectrometer_variance = (
            1.0**2 * 0.7**2
            + 2.0**2 * 0.7**2
            + 3.0**2 * 0.3**2
            + 4.0**2 * 0.3**2
        ) / 10.0**2
        self.assertAlmostEqual(
            result["two_charged"]["error"], math.sqrt(expected_two_variance)
        )
        self.assertAlmostEqual(
            result["spectrometer_only"]["error"],
            math.sqrt(expected_spectrometer_variance),
        )

    def test_plane_is_centered_four_by_six_at_z96_with_no_threshold(self):
        mother = np.array([0, 0, 0, 0, 2, 0, 1, 0, 0, 50], dtype=float)
        # A very soft forward charged particle is accepted: epsilon_reco=1
        # and there is deliberately no momentum threshold.
        soft = np.array([0, 0, 1e-6, 1e-6, 0, 211], dtype=float)
        self.assertTrue(ship.trajectory_hits_plane(mother, soft))

        on_x_edge = np.array([2.0 / 46.0, 0, 1, 1.1, 0.139, 211])
        on_y_edge = np.array([0, 3.0 / 46.0, 1, 1.1, 0.139, 211])
        outside_x = np.array([2.001 / 46.0, 0, 1, 1.1, 0.139, 211])
        backward = np.array([0, 0, -1, 1.1, 0.139, 211])
        self.assertTrue(ship.trajectory_hits_plane(mother, on_x_edge))
        self.assertTrue(ship.trajectory_hits_plane(mother, on_y_edge))
        self.assertFalse(ship.trajectory_hits_plane(mother, outside_x))
        self.assertFalse(ship.trajectory_hits_plane(mother, backward))

    def test_any_neutral_or_missed_charged_product_fails(self):
        mother = np.array([0, 0, 0, 0, 2, 0, 1, 0, 0, 50], dtype=float)
        self.assertEqual(
            ship.classify_event(mother, np.array([CH_PLUS, PHOTON], dtype=float)),
            (False, False),
        )
        self.assertEqual(
            ship.classify_event(mother, np.array([CH_PLUS, MISS], dtype=float)),
            (False, False),
        )
        self.assertEqual(
            ship.classify_event(
                mother, np.array([CH_PLUS, CH_MINUS, NEUTRINO], dtype=float)
            ),
            (False, False),
        )
        self.assertEqual(
            ship.classify_event(
                mother,
                np.array([CH_PLUS, CH_MINUS, CH_PLUS, CH_MINUS], dtype=float),
            ),
            (False, True),
        )

    def test_negative_baryon_ids_are_not_mistaken_for_raw_padding(self):
        flattened = np.array(
            ANTIPROTON
            + ANTINEUTRON
            + [0.0, 0.0, 0.0, 0.0, 0.0, -999.0]
            + [0.0] * 6,
            dtype=float,
        )
        products = ship._active_products(flattened)
        self.assertEqual(products[:, 5].tolist(), [-2212.0, -2112.0])

    def test_filename_discovery_requires_provenance_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "outputs" / "ALP-fermion" / "eventData"
            event_dir.mkdir(parents=True)
            expected = event_dir / (
                "ALP-fermion_2.000e+00_1.000e+01_exhad-alp-seed424242_data.dat"
            )
            expected.touch()
            resolved = ship.find_event_file(
                tmp, "ALP-fermion", 2.0, 10.0, "exhad-alp-seed424242"
            )
            self.assertEqual(resolved, expected)
            with self.assertRaises(FileNotFoundError):
                ship.find_event_file(tmp, "ALP-fermion", 2.0, 10.0, "raw-pythia")

    def test_filename_discovery_can_use_explicit_output_archive(self):
        with tempfile.TemporaryDirectory() as primary, tempfile.TemporaryDirectory() as archive:
            event_dir = (
                Path(archive) / "outputs" / "ALP-fermion" / "eventData"
            )
            event_dir.mkdir(parents=True)
            expected = event_dir / (
                "ALP-fermion_2.000e+00_1.000e+01_"
                "stock-pythia-alp-seed250725_data.dat"
            )
            expected.touch()
            with mock.patch.dict(
                os.environ,
                {"EVENTCALC_OUTPUT_ARCHIVE": archive},
                clear=False,
            ):
                resolved = ship.find_event_file(
                    primary,
                    "ALP-fermion",
                    2.0,
                    10.0,
                    "stock-pythia-alp-seed250725",
                )
            self.assertEqual(resolved, expected)

    def test_alp_stock_tag_replaces_raw_only_above_transition(self):
        calls = []

        def fake_find(
            eventcalc_dir, llp, mass, ctau, generator_tag,
            production_label, channel_tag=None,
        ):
            del eventcalc_dir, ctau, production_label, channel_tag
            calls.append((llp, float(mass), generator_tag))
            return Path(f"/{llp}-{mass}-{generator_tag}.dat")

        with mock.patch.object(
            ship, "find_event_file", side_effect=fake_find
        ):
            inputs = ship.resolve_inputs(
                Path("/events"),
                {"dv": (), "alp": (1.9, 1.911, 2.0)},
                10.0,
                250725,
                "central",
                alp_default_tag="stock-pythia-alp-seed250725",
            )

        raw_tags = [
            tag for llp, mass, tag in calls
            if llp == "ALP-fermion" and (
                mass < 1.911 - 1.0e-12 or "stock-pythia" in tag
            )
        ]
        self.assertEqual(
            raw_tags,
            [
                "raw-pythia-seed250725",
                "stock-pythia-alp-seed250725",
                "stock-pythia-alp-seed250725",
            ],
        )
        self.assertIn(
            "raw-pythia-seed250725",
            str(inputs["alp"]["raw"][0]),
        )
        self.assertIn(
            "stock-pythia-alp-seed250725",
            str(inputs["alp"]["raw"][1]),
        )

    def test_incomplete_stable_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "incomplete.dat"
            fields = event_row(1.0, [CH_PLUS, CH_MINUS], "raw").split()
            fields[3] = str(float(fields[3]) + 1.0)  # mother energy only
            with open(path, "w") as stream:
                stream.write("Sampled 1 events inside SHiP volume.\n\n")
                stream.write("#<process=Jets-uu; sample_points=1>\n\n")
                stream.write(" ".join(fields) + "\n")
            with self.assertRaisesRegex(ValueError, "four-momentum residual"):
                ship.analyze_file(path)

    def test_exclusive_hadronic_labels_enter_below_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exclusive.dat"
            with open(path, "w") as stream:
                stream.write("Sampled 2 events inside SHiP volume.\n\n")
                stream.write("#<process=Pip_Pim; sample_points=1>\n\n")
                stream.write(event_row(1.0, [CH_PLUS, CH_MINUS]) + "\n")
                stream.write("#<process=e-pair; sample_points=1>\n\n")
                stream.write(event_row(100.0, [CH_PLUS, CH_MINUS]) + "\n")
            result = ship.analyze_file(
                path, nonhadronic_channels=("e-pair",))
        self.assertEqual(result["n_events"], 1)
        self.assertEqual(result["skipped_nonhadronic_events"], 1)
        self.assertAlmostEqual(result["two_charged"]["fraction"], 1.0)

    def test_default_dark_photon_grid_resolves_both_sides_of_transition(self):
        masses = ship.BENCHMARKS["dv"]["masses_GeV"]
        dense = (
            1.695, 1.699, 1.70, 1.705, 1.71, 1.72, 1.75, 1.78, 1.80,
        )
        for mass in dense:
            self.assertIn(mass, masses)
        self.assertLess(masses.index(1.695), masses.index(1.70))
        self.assertLess(masses.index(1.699), masses.index(1.70))
        self.assertEqual(tuple(sorted(masses)), masses)

        publication = ship.BENCHMARKS["dv"]["publication_masses_GeV"]
        self.assertTrue(set(publication).issubset(masses))
        self.assertIn(1.6, publication)
        self.assertIn(1.8, publication)
        # The dense form-factor/seam nodes remain signed diagnostics and are
        # intentionally not rendered as a precision mass scan in the Letter.
        for mass in (1.69, 1.695, 1.699, 1.70, 1.705, 1.71, 1.72, 1.75, 1.78):
            self.assertNotIn(mass, publication)

    def test_plot_masks_carry_exclusive_line_to_exact_transition_without_points(self):
        masses = ship.BENCHMARKS["dv"]["masses_GeV"]
        publication = ship.BENCHMARKS["dv"]["publication_masses_GeV"]
        exclusive, production = ship.publication_plot_masks(
            masses, publication, 1.70,
        )
        selected_exclusive = np.asarray(masses)[exclusive].tolist()
        selected_production = np.asarray(masses)[production].tolist()
        self.assertEqual(selected_exclusive[-4:], [1.69, 1.695, 1.699, 1.70])
        self.assertEqual(selected_production[0], 1.70)
        self.assertEqual(selected_production[1], 1.80)
        self.assertNotIn(1.705, selected_production)

    def test_alp_plot_masks_share_only_the_exact_transition_point(self):
        masses = ship.BENCHMARKS["alp"]["masses_GeV"]
        publication = ship.BENCHMARKS["alp"]["publication_masses_GeV"]
        exclusive, production = ship.publication_plot_masks(
            masses, publication, 1.911,
        )
        selected_exclusive = np.asarray(masses)[exclusive].tolist()
        selected_production = np.asarray(masses)[production].tolist()
        self.assertEqual(selected_exclusive[-3:], [1.9, 1.910, 1.911])
        self.assertEqual(selected_production[:2], [1.911, 2.0])
        self.assertEqual(
            set(selected_exclusive).intersection(selected_production),
            {1.911},
        )

    def test_alp_above_transition_requires_all_channel_scope(self):
        complete_raw = {
            "process_labels": [
                "muPmuM", "Jets-GG", "Jets-ss", "ppbar", "nnbar"
            ]
        }
        complete_matched = {
            "process_labels": ["muPmuM", "Jets-matched"]
        }
        complete_stock = {
            "process_labels": ["muPmuM", "Jets-stock"]
        }
        ship.validate_alp_all_channel_scope(
            complete_raw, 1.911, "raw.dat", "raw")
        ship.validate_alp_all_channel_scope(
            complete_matched, 1.911, "matched.dat", "exhad")
        ship.validate_alp_all_channel_scope(
            complete_stock, 1.911, "stock.dat", "stock")
        ship.validate_alp_all_channel_scope(
            {"process_labels": ["Jets-GG"]}, 1.910, "exclusive.dat", "raw",
        )
        with self.assertRaisesRegex(ValueError, r"--channels all"):
            ship.validate_alp_all_channel_scope(
                {"process_labels": ["Jets-GG", "Jets-ss"]},
                2.0, "jets-only.dat", "raw",
            )
        with self.assertRaisesRegex(ValueError, r"missing exact nucleon.*ppbar"):
            ship.validate_alp_all_channel_scope(
                {"process_labels": ["muPmuM", "Jets-GG", "Jets-ss"]},
                2.0, "raw-no-nucleons.dat", "raw",
            )
        with self.assertRaisesRegex(ValueError, r"no aggregate Jets-matched"):
            ship.validate_alp_all_channel_scope(
                {"process_labels": ["muPmuM", "Jets-GG", "ppbar", "nnbar"]},
                2.0, "unpooled.dat", "exhad",
            )
        with self.assertRaisesRegex(ValueError, r"constituent ALP"):
            ship.validate_alp_all_channel_scope(
                {
                    "process_labels": [
                        "muPmuM", "Jets-matched", "ppbar", "nnbar"
                    ]
                },
                2.0, "leaked.dat", "exhad",
            )
        with self.assertRaisesRegex(ValueError, r"constituent ALP"):
            ship.validate_alp_all_channel_scope(
                {
                    "process_labels": [
                        "muPmuM", "Jets-stock", "ppbar", "nnbar"
                    ]
                },
                2.0, "stock-leaked.dat", "stock",
            )

    def test_alp_lifetime_table_is_a_signed_generation_input(self):
        ctau = ship.BENCHMARKS["alp"]["ctau_table"]
        self.assertEqual(ctau.name, "ctau-ALP-fermion.txt")
        self.assertTrue(ctau.is_file())

    def test_primary_dp_provenance_signs_flux_and_records_1699_knot(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_dp_primary_inputs(tmp)
            provenance = ship.dp_primary_production_provenance(
                tmp, "central", 1.70,
            )

        self.assertEqual(provenance["uncertainty"], "central")
        self.assertEqual(
            set(provenance["files"]),
            {"double_distribution", "maximum_energy", "total_yield"},
        )
        self.assertEqual(
            provenance["mass_knots_near_transition_GeV"],
            [1.65, 1.699, 1.75],
        )
        self.assertEqual(provenance["interpolation_bracket_GeV"], [1.699, 1.75])
        self.assertEqual(
            provenance["mass_knot_immediately_below_transition_GeV"], 1.699,
        )
        for signature in provenance["files"].values():
            self.assertEqual(len(signature["sha256"]), 64)
            self.assertGreater(signature["size"], 0)

    def test_primary_dp_provenance_rejects_mismatched_mass_grids(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_dp_primary_inputs(tmp, emax_nodes=(1.65, 1.70, 1.75))
            with self.assertRaisesRegex(ValueError, "mass grids disagree"):
                ship.dp_primary_production_provenance(tmp, "central", 1.70)

    def test_cascade_dp_provenance_signs_its_independent_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Distributions" / "Dark-photons"
            base.mkdir(parents=True)
            for stem in ("DoubleDistr", "Emax"):
                with open(base / (stem + "-DP-cascade-central.txt"), "w") as stream:
                    stream.write("1.6 1.0\n1.75 1.0\n1.8 1.0\n")
            with open(base / "Total-yield-DP-cascade-central.txt", "w") as stream:
                stream.write("1.6 1.0\n1.7 1.0\n1.8 1.0\n")
            provenance = ship.dp_source_production_provenance(
                tmp, "cascade", "central", 1.70,
            )

        self.assertEqual(provenance["source"], "cascade")
        self.assertEqual(provenance["file_label"], "cascade-central")
        self.assertEqual(provenance["interpolation_bracket_GeV"], [1.6, 1.75])
        self.assertEqual(len(provenance["files"]), 3)
        for signature in provenance["files"].values():
            self.assertEqual(len(signature["sha256"]), 64)

    def test_topology_conditionals_separate_mixture_from_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ship.analyze_topology_conditionals(
                self.write_sample(tmp), nonhadronic_channels=("e-pair",),
            )

        classes = result["classes"]
        self.assertAlmostEqual(
            classes["exactly_two_charged"]["weight_fraction"], 0.5,
        )
        self.assertAlmostEqual(
            classes["other_all_charged"]["weight_fraction"], 0.2,
        )
        self.assertAlmostEqual(
            classes["neutral_containing"]["weight_fraction"], 0.3,
        )
        self.assertAlmostEqual(
            classes["exactly_two_charged"]["conditional_acceptance"][
                "two_charged"
            ],
            0.2,
        )
        self.assertAlmostEqual(
            classes["other_all_charged"]["conditional_acceptance"][
                "spectrometer_only"
            ],
            1.0,
        )

    def test_primary_cascade_combination_uses_absolute_and_topology_weights(self):
        """Source fractions are never averaged, including conditionals."""
        with tempfile.TemporaryDirectory() as tmp:
            primary_path = Path(tmp) / "primary.dat"
            cascade_path = Path(tmp) / "cascade.dat"
            with open(primary_path, "w") as stream:
                stream.write("#<process=Jets-uu; sample_points=2>\n")
                # One accepted exactly-two topology and nine units of a
                # neutral-containing topology.
                stream.write(event_row(1.0, [CH_PLUS, CH_MINUS]) + "\n")
                stream.write(event_row(9.0, [CH_PLUS, PHOTON]) + "\n")
            with open(cascade_path, "w") as stream:
                stream.write("#<process=Jets-uu; sample_points=2>\n")
                # Nine topology units fail the plane requirement; the
                # remaining unit is neutral-containing.
                stream.write(event_row(9.0, [CH_PLUS, MISS]) + "\n")
                stream.write(event_row(1.0, [CH_PLUS, PHOTON]) + "\n")

            points = {
                "primary": ship.analyze_file(primary_path),
                "cascade": ship.analyze_file(cascade_path),
            }
            for source, point in points.items():
                point["source"] = str(
                    primary_path if source == "primary" else cascade_path
                )
                point["source_normalization"] = {
                    "source": source,
                    "common_flux_factor": 1.0,
                    "scale_per_event_P_decay_weight": 1.0,
                    "absolute_hadronic_weight": point["sum_weight"],
                }
            combined = ship.combine_dp_source_points(points)

        # Both sources carry total absolute weight ten, but their
        # exactly-two topology weights are one and nine.  The correct
        # conditional plane acceptance is therefore 1/(1+9), not the
        # arithmetic average (1+0)/2.
        factor = combined["factorization"]["two_charged"]
        self.assertAlmostEqual(factor["topology_fraction"]["fraction"], 0.5)
        self.assertAlmostEqual(
            factor["conditional_geometry_acceptance"]["fraction"], 0.1,
        )
        self.assertAlmostEqual(factor["accepted_fraction"]["fraction"], 0.05)
        self.assertAlmostEqual(
            factor["topology_fraction"]["fraction"]
            * factor["conditional_geometry_acceptance"]["fraction"],
            factor["accepted_fraction"]["fraction"],
        )
        self.assertAlmostEqual(
            combined["source_components"]["primary"][
                "relative_hadronic_weight"
            ],
            0.5,
        )

    def test_source_normalization_reconstructs_eventcalc_absolute_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_dir = root / "outputs" / "Dark-photons" / "eventData"
            total_dir = root / "outputs" / "Dark-photons" / "total"
            input_dir = root / "Distributions" / "Dark-photons"
            event_dir.mkdir(parents=True)
            total_dir.mkdir(parents=True)
            input_dir.mkdir(parents=True)
            event_path = event_dir / (
                "Dark-photons_1.700e+00_1.000e+01_central_"
                "raw-pythia-seed1_data.dat"
            )
            with open(event_path, "w") as stream:
                stream.write("#<process=Jets-uu; sample_points=2>\n")
                stream.write(event_row(0.2, [CH_PLUS, CH_MINUS]) + "\n")
                stream.write(event_row(0.4, [CH_PLUS, CH_MINUS]) + "\n")
            with open(input_dir / "Total-yield-DP-central.txt", "w") as stream:
                stream.write("1.6 2.0\n1.8 4.0\n")
            total_path = total_dir / (
                "Dark-photons_central_raw-pythia-seed1_total.txt"
            )
            with open(total_path, "w") as stream:
                stream.write(
                    "mass coupling_squared c_tau N_LLP_tot epsilon_polar "
                    "epsilon_azimuthal P_decay_averaged Br_visible N_ev_tot\n"
                )
                stream.write("1.7 1.0 10.0 300.0 0.5 0.25 0.3 0.6 6.75\n")
            point = ship.analyze_file(event_path)
            normalization = ship.dp_source_normalization(
                point, event_path, root, 1.7, 10.0, "primary", "central",
            )

        # resample_size = 2 / 0.25 = 8 and Y(1.7)=3, hence the multiplier is
        # 3 * 0.5 * 0.6 / 8.  The event-level P_decay sum is 0.6.
        self.assertEqual(normalization["resample_size"], 8)
        self.assertAlmostEqual(
            normalization["scale_per_event_P_decay_weight"], 0.1125,
        )
        self.assertAlmostEqual(
            normalization["absolute_hadronic_weight"], 0.0675,
        )

    def test_paired_path_diagnostic_requires_and_records_same_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.dat"
            matched = Path(tmp) / "matched.dat"
            for path, products in (
                (raw, [CH_PLUS, CH_MINUS]),
                (matched, [PHOTON_SAME_P4, PHOTON_SAME_P4]),
            ):
                with open(path, "w") as stream:
                    stream.write("#<process=Jets-uu; sample_points=1>\n")
                    stream.write(event_row(1.0, products) + "\n")
            result = ship.paired_path_diagnostic(raw, matched)

        self.assertTrue(result["mother_records_bitwise_identical"])
        self.assertEqual(result["n_identical_parent_events"], 1)
        self.assertEqual(result["paths"]["stock"]["two_charged"], 1.0)
        self.assertEqual(result["paths"]["matched"]["two_charged"], 0.0)
        self.assertEqual(len(result["mother_records_sha256"]), 64)

    def test_paired_path_diagnostic_rejects_different_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.dat"
            matched = Path(tmp) / "matched.dat"
            with open(raw, "w") as stream:
                stream.write("#<process=Jets-uu; sample_points=1>\n")
                stream.write(event_row(1.0, [CH_PLUS, CH_MINUS]) + "\n")
            with open(matched, "w") as stream:
                stream.write("#<process=Jets-uu; sample_points=1>\n")
                stream.write(
                    event_row(
                        1.0, [CH_PLUS, CH_MINUS], vertex=(0.0, 0.0, 51.0)
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "mother records first differ"):
                ship.paired_path_diagnostic(raw, matched)

    def test_certified_dv_runtime_provenance_is_explicit(self):
        for label in ("routing_card", "morph_card", "generator"):
            signature = ship._file_signature(
                Path(ship.DV_MATCHED_RUNTIME_INPUTS[label])
            )
            self.assertEqual(len(signature["sha256"]), 64)
            self.assertGreater(signature["size"], 0)

        runtime = ship.dp_matched_runtime_signature()
        self.assertEqual(
            set(runtime),
            {"routing_card", "morph_card", "generator", "classifier",
             "injection_table", "injection_modes", "matched_architecture",
             "production_authorities", "anchor_cards",
             "generator_runtime_identity"},
        )
        self.assertEqual(len(runtime["classifier"]["sha256"]), 64)
        self.assertEqual(len(runtime["injection_table"]["sha256"]), 64)
        self.assertEqual(len(runtime["injection_modes"]["sha256"]), 64)
        architecture_paths = {
            Path(item["path"]).resolve()
            for item in runtime["matched_architecture"]
        }
        self.assertEqual(
            architecture_paths,
            {
                Path(ship.ROOT / "exhad" / "py" / "exhad_bridge.py").resolve(),
                *(
                    Path(path).resolve()
                    for path in ship._MATCHED_PACKAGE_RUNTIME_INPUTS
                ),
            },
        )
        self.assertIn(
            (
                ship.ROOT / "exhad" / "py" / "exhad_matched" /
                "inputs" / "dark_photon_widths.py"
            ).resolve(),
            architecture_paths,
        )
        for item in runtime["matched_architecture"]:
            self.assertEqual(len(item["sha256"]), 64)
        authorities = runtime["production_authorities"]
        self.assertEqual(
            set(authorities),
            {
                "frozen_em_response",
                "exclusive_probability_table",
                "exclusive_to_perturbative_table",
            },
        )
        self.assertEqual(
            {
                label: Path(record["path"]).name
                for label, record in authorities.items()
            },
            {
                "frozen_em_response": "em_response_authority.json",
                "exclusive_probability_table": "dv_channels.csv",
                "exclusive_to_perturbative_table":
                    "dv_exclusive_to_perturbative.csv",
            },
        )
        for item in authorities.values():
            self.assertEqual(len(item["sha256"]), 64)
        self.assertEqual(
            len(runtime["generator_runtime_identity"]["semantic_sha256"]),
            64,
        )

        anchors = ship.DV_MATCHED_RUNTIME_INPUTS["anchor_cards"]
        self.assertEqual(len(anchors), 6)
        self.assertEqual(
            [Path(path).name for path in anchors],
            [
                "dv_anchor_1.20.json", "dv_anchor_1.30.json",
                "dv_anchor_1.40.json", "dv_anchor_1.50.json",
                "dv_anchor_1.60.json", "dv_anchor_1.70.json",
            ],
        )
        for path in anchors:
            self.assertEqual(len(ship._file_signature(Path(path))["sha256"]), 64)

        signed = {"dp_matched_runtime_inputs": runtime}
        ship.validate_dp_render_cache_compatibility(signed)
        stale = copy.deepcopy(signed)
        stale["dp_matched_runtime_inputs"]["classifier"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "incompatible matched-DP"):
            ship.validate_dp_render_cache_compatibility(stale)

        stale = copy.deepcopy(signed)
        stale["dp_matched_runtime_inputs"]["production_authorities"][
            "exclusive_probability_table"
        ]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "incompatible matched-DP"):
            ship.validate_dp_render_cache_compatibility(stale)

    def test_certified_alp_runtime_provenance_is_complete_and_render_gating(self):
        runtime = ship.alp_matched_runtime_signature()
        self.assertEqual(
            set(runtime),
            {"routing_card", "morph_card", "generator", "classifier",
             "matched_architecture", "anchor_cards",
             "generator_runtime_identity"},
        )
        self.assertEqual(
            [Path(path).name
             for path in ship.ALP_MATCHED_RUNTIME_INPUTS["anchor_cards"]],
            [
                "alp_anchor_1.20.json", "alp_anchor_1.25.json",
                "alp_anchor_1.30.json", "alp_anchor_1.40.json",
                "alp_anchor_1.50.json", "alp_anchor_1.60.json",
                "alp_anchor_1.70.json", "alp_anchor_1.80.json",
                "alp_anchor_1.90.json", "alp_anchor_1.91.json",
            ],
        )
        self.assertEqual(len(runtime["anchor_cards"]), 10)
        for label in ("routing_card", "morph_card", "generator", "classifier"):
            self.assertEqual(len(runtime[label]["sha256"]), 64)
        self.assertGreater(len(runtime["matched_architecture"]), 10)
        architecture_names = {
            Path(item["path"]).name
            for item in runtime["matched_architecture"]
        }
        self.assertTrue({
            "exhad_bridge.py",
            "alp_fermion_portal_measure.py",
            "alp_fermion_authority.py",
            "spinzero_eventcalc_adapter.py",
            "contract.py",
            "charge_completion.py",
            "model.py",
        }.issubset(architecture_names))
        for item in runtime["matched_architecture"]:
            self.assertEqual(len(item["sha256"]), 64)
        identity = runtime["generator_runtime_identity"]
        self.assertEqual(len(identity["semantic_sha256"]), 64)
        self.assertEqual(len(identity["audit"]["exhad_binary_sha256"]), 64)
        signature = {"alp_matched_runtime_inputs": runtime}
        ship.validate_alp_render_cache_compatibility(signature)

        stale = copy.deepcopy(signature)
        stale["alp_matched_runtime_inputs"]["morph_card"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "incompatible matched-ALP"):
            ship.validate_alp_render_cache_compatibility(stale)

    def test_combined_slope_diagnostic_keeps_sources_separate(self):
        masses = [1.695, 1.699, 1.700, 1.705]
        values = [0.100, 0.104, 0.105, 0.110]

        def stat(value):
            return {"fraction": value, "error": 0.001}

        points = []
        for value in values:
            points.append({
                "production_sources": ["primary", "cascade"],
                "two_charged": stat(value),
                "spectrometer_only": stat(value),
                "source_components": {
                    "primary": {
                        "two_charged": stat(value),
                        "spectrometer_only": stat(value),
                        "relative_hadronic_weight": 0.75,
                    },
                    "cascade": {
                        "two_charged": stat(value),
                        "spectrometer_only": stat(value),
                        "relative_hadronic_weight": 0.25,
                    },
                },
            })
        result = ship.compute_dp_combined_slope_diagnostics({
            "dv": {
                "masses_GeV": masses,
                "exhad": points,
                "production_sources": ["primary", "cascade"],
            }
        })

        row = result["metrics"]["spectrometer_only"]["combined"]
        self.assertAlmostEqual(row["left_slope_per_GeV"], 1.0)
        self.assertAlmostEqual(row["right_slope_per_GeV"], 1.0)
        self.assertAlmostEqual(row["right_minus_left_slope_z"], 0.0)
        self.assertEqual(
            result["relative_hadronic_source_weights"]["cascade"],
            [0.25, 0.25, 0.25, 0.25],
        )

    def test_alp_boundary_closure_records_topology_and_accepted_z_values(self):
        def stat(fraction, error):
            return {"fraction": fraction, "error": error}

        points = []
        for shift in (0.0, 0.001):
            points.append({
                "two_charged": stat(0.004 + shift, 0.0005),
                "spectrometer_only": stat(0.009 + shift, 0.0007),
                "factorization": {
                    "two_charged": {
                        "topology_fraction": stat(0.006 + shift, 0.0006),
                    },
                    "spectrometer_only": {
                        "topology_fraction": stat(0.024 + shift, 0.0012),
                    },
                },
            })
        diagnostic = ship.compute_alp_boundary_closure({
            "alp": {
                "masses_GeV": [1.910, 1.911],
                "exhad": points,
            },
        })
        self.assertEqual(len(diagnostic["rows"]), 4)
        for row in diagnostic["rows"]:
            self.assertAlmostEqual(row["above_minus_below"], 0.001)
            self.assertIsNotNone(row["difference_z"])


if __name__ == "__main__":
    unittest.main()
