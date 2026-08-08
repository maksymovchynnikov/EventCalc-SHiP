import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from funcs.kinematics import Grids
from funcs.initLLP import LLP
from funcs.ship_setup import theta_max_dec_vol
from analysis.ECAL import diphoton_ecal_acceptance
from alp_discrimination.cache import CacheStore, cache_key
from alp_discrimination.config import PRODUCTION, SMOKE, lower_ctau_m
from alp_discrimination.eventcalc_adapter import (
    EventCalcAdapter,
    KinematicProposal,
    MotherSample,
    generate_mother_sample,
    legacy_numpy_seed,
)
from alp_discrimination.eventcalc_proposals import _proposal_identity
from alp_discrimination.models import MODELS
from alp_discrimination.selections import (
    DiphotonECALEnergySelection,
    DiphotonECALSelection,
    SelectionContext,
)


class EventCalcAdapterTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("EVENTCALC_RUN_PROPOSAL_DIFFERENTIAL") == "1",
        "set EVENTCALC_RUN_PROPOSAL_DIFFERENTIAL=1",
    )
    def test_real_proposal_mothers_and_ecal_match_legacy_exactly(self):
        model = MODELS[0]
        source = model.sources[0]
        mass_gev = 0.5
        grid_lower_m = float(np.exp(np.log(lower_ctau_m(mass_gev))))
        source_seed = SMOKE.seed_policy.source_proposal_seed(
            mass_gev,
            model.identifier,
            0,
            seed_offset=SMOKE.templates.seed_offset,
        )
        true_sample_seed = SMOKE.seed_policy.true_sample_seed(
            mass_gev,
            model.identifier,
            0,
            seed_offset=SMOKE.templates.seed_offset,
        )

        with tempfile.TemporaryDirectory() as directory:
            adapter = EventCalcAdapter(
                SMOKE,
                cache=CacheStore("smoke", Path(directory)),
            )
            proposal = adapter.prepare_kinematic_proposal(
                model,
                source,
                mass_gev,
                source_seed,
                "spectrum",
                proposal_ctau_m=grid_lower_m,
            )
            repeats = [
                adapter.prepare_kinematic_proposal(
                    model,
                    source,
                    mass_gev,
                    source_seed,
                    "spectrum",
                    proposal_ctau_m=grid_lower_m,
                )
                for _ in range(PRODUCTION.templates.lifetime_points_per_model)
            ]
            self.assertTrue(all(item is proposal for item in repeats))

            llp = LLP(
                mass=None,
                particle_selection=model.particle_selection,
                mixing_pattern=None,
                uncertainty=None,
                alp_production_mode=source.eventcalc_mode,
            )
            llp.set_mass(mass_gev)
            llp.compute_mass_dependent_properties()
            llp.set_c_tau(grid_lower_m)
            with legacy_numpy_seed(source_seed):
                legacy = Grids(
                    llp.Distr,
                    llp.Energy_distr,
                    SMOKE.spectrum_sampling.interpolation_points,
                    mass_gev,
                    grid_lower_m,
                    theta_max_sim=theta_max_dec_vol,
                )
                legacy.interpolate(False)
                legacy.resample(SMOKE.spectrum_sampling.resample_size, False)
            np.testing.assert_array_equal(proposal.r_theta_rad, legacy.r_theta)
            np.testing.assert_array_equal(proposal.r_energy_gev, legacy.r_energy)
            self.assertEqual(proposal.epsilon_polar, legacy.epsilon_polar)

            mothers = generate_mother_sample(
                proposal,
                grid_lower_m,
                true_sample_seed,
            )
            legacy.c_tau = grid_lower_m
            with legacy_numpy_seed(true_sample_seed):
                legacy.true_samples(False)
            results = np.asarray(legacy.get_kinematics(), dtype=float)
            valid = (
                np.isfinite(results[:, 3])
                & np.isfinite(results[:, 6])
                & (results[:, 6] >= 0.0)
            )
            results = results[valid]
            np.testing.assert_array_equal(mothers.energy_gev, results[:, 3])
            np.testing.assert_array_equal(mothers.decay_probability, results[:, 6])
            np.testing.assert_array_equal(mothers.x_m, results[:, 7])
            np.testing.assert_array_equal(mothers.y_m, results[:, 8])
            np.testing.assert_array_equal(mothers.z_m, results[:, 9])

            context = SelectionContext(source_seed, true_sample_seed)
            current_ecal = adapter.selection.details(mothers, context)
            legacy_ecal = diphoton_ecal_acceptance(
                results,
                seed=source_seed + SMOKE.seed_policy.ecal_seed_offset,
                return_details=True,
            )
            np.testing.assert_array_equal(
                current_ecal.event_mask,
                legacy_ecal.event_mask,
            )
            spectrum = adapter.evaluate_spectrum(
                proposal,
                grid_lower_m,
                true_sample_seed,
                cache_result=False,
            )
            coupling_squared = proposal.unit_coupling_ctau_m / grid_lower_m
            event_weight_scale = (
                SMOKE.exposure_pot
                * proposal.yield_per_pot_per_coupling_squared
                * coupling_squared
                * proposal.epsilon_polar
                * proposal.visible_br
                / proposal.resample_size
            )
            legacy_preselection_weights = event_weight_scale * results[:, 6]
            legacy_weights = np.asarray(
                legacy_preselection_weights[legacy_ecal.event_mask],
                dtype=float,
            )
            np.testing.assert_array_equal(
                spectrum.absolute_event_weights,
                legacy_weights,
            )
            self.assertEqual(
                spectrum.preselection_expected_events,
                float(np.sum(legacy_preselection_weights)),
            )
            self.assertEqual(
                spectrum.expected_events,
                float(np.sum(legacy_weights)),
            )

    def test_cache_profile_must_match_configuration(self):
        with self.assertRaises(ValueError):
            EventCalcAdapter(SMOKE, cache=CacheStore("production"))

    def test_cached_plain_arrays_reproduce_true_samples(self):
        theta = np.array([0.001, 0.005, 0.01, 0.02])
        energy = np.array([2.0, 5.0, 10.0, 20.0])
        proposal = KinematicProposal(
            model_id="toy", source="inclusive", mass_gev=1.0, proposal_ctau_m=10.0,
            proposal_seed=1, interpolation_points=4, resample_size=4,
            r_theta_rad=theta, r_energy_gev=energy, epsilon_polar=0.5, visible_br=1.0,
            yield_per_pot_per_coupling_squared=1.0, unit_coupling_ctau_m=1.0,
            theta_min_rad=0.0, theta_max_rad=0.1, sanitation_policy="strict_core",
            input_fingerprints=(),
        )
        legacy = object.__new__(Grids)
        legacy.r_theta, legacy.r_energy, legacy.m, legacy.c_tau = theta, energy, 1.0, 10.0
        legacy.true_points_indices = np.arange(4)
        state = np.random.get_state()
        np.random.seed(123)
        legacy.true_samples(False)
        np.random.set_state(state)
        current = generate_mother_sample(proposal, 10.0, 123)
        expected = legacy.kinematics_dic
        np.testing.assert_array_equal(current.px_gev, expected["px"])
        np.testing.assert_array_equal(current.py_gev, expected["py"])
        np.testing.assert_array_equal(current.pz_gev, expected["pz"])
        np.testing.assert_array_equal(current.energy_gev, expected["energy"])
        np.testing.assert_array_equal(current.decay_probability, expected["P_decay"])
        np.testing.assert_array_equal(current.x_m, expected["x"])
        np.testing.assert_array_equal(current.y_m, expected["y"])
        np.testing.assert_array_equal(current.z_m, expected["z"])
        self.assertEqual(current.mass_gev, 1.0)

    def test_default_selection_uses_profile_geometry_and_seed_offset(self):
        adapter = EventCalcAdapter(SMOKE, cache=Mock(profile="smoke"))
        self.assertIsInstance(adapter.selection, DiphotonECALSelection)
        self.assertEqual(adapter.selection.geometry.z_m, SMOKE.ecal_geometry.z_m)
        self.assertEqual(
            adapter.selection.geometry.width_x_m,
            SMOKE.ecal_geometry.width_x_m,
        )
        context = SelectionContext(source_seed=54_321, true_sample_seed=54_322)
        identity = adapter.selection.cache_identity(context)
        self.assertEqual(identity["selection_seed"], 54_323)

    def test_selection_change_reuses_proposal_identity_but_splits_weighted_cache(self):
        energy_config = replace(
            SMOKE,
            selection_name="diphoton_ecal_e1gev",
        )
        model = MODELS[0]
        source = model.sources[0]
        proposal_identity_geometry = _proposal_identity(
            SMOKE,
            model,
            source,
            1.0,
            10.0,
            123,
            SMOKE.spectrum_sampling,
            "strict_core",
            (),
        )
        proposal_identity_energy = _proposal_identity(
            energy_config,
            model,
            source,
            1.0,
            10.0,
            123,
            energy_config.spectrum_sampling,
            "strict_core",
            (),
        )
        self.assertEqual(proposal_identity_geometry, proposal_identity_energy)

        proposal = KinematicProposal(
            model_id="toy",
            source="inclusive",
            mass_gev=1.0,
            proposal_ctau_m=10.0,
            proposal_seed=123,
            interpolation_points=4,
            resample_size=4,
            r_theta_rad=np.zeros(4),
            r_energy_gev=np.array([2.0, 10.0, 100.0, 3.0]),
            epsilon_polar=0.5,
            visible_br=1.0,
            yield_per_pot_per_coupling_squared=1.0,
            unit_coupling_ctau_m=10.0,
            theta_min_rad=0.0,
            theta_max_rad=0.1,
            sanitation_policy="strict_core",
            input_fingerprints=(),
            cache_key="shared-proposal-key",
        )
        geometry_cache = Mock(profile="smoke")
        geometry_cache.paths.return_value = (Mock(), Mock(), "geometry-key")
        geometry_cache.load.return_value = None
        geometry_cache.save.return_value = {"cache_key": "geometry-key"}
        energy_cache = Mock(profile="smoke")
        energy_cache.paths.return_value = (Mock(), Mock(), "energy-key")
        energy_cache.load.return_value = None
        energy_cache.save.return_value = {"cache_key": "energy-key"}

        geometry_adapter = EventCalcAdapter(
            SMOKE,
            cache=geometry_cache,
            selection=DiphotonECALSelection(),
        )
        energy_adapter = EventCalcAdapter(
            energy_config,
            cache=energy_cache,
            selection=DiphotonECALEnergySelection(),
        )
        mothers = MotherSample(
            px_gev=np.array([0.0, 0.1, 0.2, -0.1]),
            py_gev=np.array([0.0, -0.05, 0.1, 0.2]),
            pz_gev=np.sqrt(
                np.array([2.0, 10.0, 100.0, 3.0]) ** 2
                - 1.0
                - np.array([0.0, 0.1, 0.2, -0.1]) ** 2
                - np.array([0.0, -0.05, 0.1, 0.2]) ** 2
            ),
            energy_gev=np.array([2.0, 10.0, 100.0, 3.0]),
            decay_probability=np.array([0.1, 0.2, 0.3, 0.4]),
            x_m=np.array([0.0, 0.5, -0.2, 1.0]),
            y_m=np.array([0.0, -0.2, 0.3, 0.1]),
            z_m=np.array([50.0, 70.0, 80.0, 94.0]),
            mass_gev=1.0,
        )

        with patch(
            "alp_discrimination.eventcalc_adapter.generate_mother_sample",
            return_value=mothers,
        ):
            geometry_adapter.evaluate_spectrum(proposal, 10.0, 124)
            energy_adapter.evaluate_spectrum(proposal, 10.0, 124)

        geometry_identity = geometry_cache.paths.call_args.args[1]
        energy_identity = energy_cache.paths.call_args.args[1]
        self.assertEqual(
            geometry_identity["proposal_cache_key"],
            energy_identity["proposal_cache_key"],
        )
        self.assertEqual(
            geometry_identity["proposal_cache_key"],
            "shared-proposal-key",
        )
        common_geometry = {
            key: value
            for key, value in geometry_identity.items()
            if key not in {"selection_name", "selection"}
        }
        common_energy = {
            key: value
            for key, value in energy_identity.items()
            if key not in {"selection_name", "selection"}
        }
        self.assertEqual(common_geometry, common_energy)
        self.assertEqual(geometry_identity["selection_name"], "diphoton_ecal")
        self.assertEqual(
            energy_identity["selection_name"],
            "diphoton_ecal_e1gev",
        )
        self.assertEqual(
            energy_identity["selection"]["minimum_photon_energy_gev"],
            1.0,
        )
        self.assertNotEqual(
            cache_key(geometry_identity),
            cache_key(energy_identity),
        )

    def test_spectrum_cache_can_be_bypassed_for_scalar_only_workflows(self):
        cache = Mock(profile="smoke")
        adapter = EventCalcAdapter(SMOKE, cache=cache)
        proposal = KinematicProposal(
            model_id="toy", source="inclusive", mass_gev=0.3, proposal_ctau_m=3.0,
            proposal_seed=1, interpolation_points=4, resample_size=4,
            r_theta_rad=np.zeros(4), r_energy_gev=np.full(4, 10.0),
            epsilon_polar=0.5, visible_br=1.0,
            yield_per_pot_per_coupling_squared=1.0, unit_coupling_ctau_m=3.0,
            theta_min_rad=0.0, theta_max_rad=0.1, sanitation_policy="strict_core",
            input_fingerprints=(),
        )
        spectrum = adapter.evaluate_spectrum(
            proposal, 3.0, 123, cache_result=False,
        )
        self.assertGreater(spectrum.expected_events, 0.0)
        mothers = generate_mother_sample(proposal, 3.0, 123)
        mask = adapter.selection.mask(
            mothers,
            SelectionContext(source_seed=proposal.proposal_seed, true_sample_seed=123),
        )
        scale = SMOKE.exposure_pot * proposal.epsilon_polar / proposal.resample_size
        preselection_weights = scale * mothers.decay_probability
        expected_before = float(np.sum(preselection_weights))
        expected_after = float(np.sum(preselection_weights[mask]))
        self.assertEqual(spectrum.preselection_samples, len(mothers))
        self.assertEqual(spectrum.accepted_samples, np.count_nonzero(mask))
        self.assertEqual(spectrum.preselection_expected_events, expected_before)
        self.assertEqual(spectrum.expected_events, expected_after)
        self.assertEqual(
            spectrum.selection_efficiency_weighted,
            expected_after / expected_before,
        )
        self.assertEqual(spectrum.epsilon_azimuthal, len(mothers) / proposal.resample_size)
        self.assertIsNone(spectrum.cache_key)
        cache.paths.assert_not_called()
        cache.load.assert_not_called()
        cache.save.assert_not_called()

    def test_strict_proposal_guard_uses_exact_preparation_lifetime(self):
        cache = Mock(profile="smoke")
        adapter = EventCalcAdapter(SMOKE, cache=cache)
        for mass_gev in (0.5, 0.75):
            raw_lower_m = 3.0 * (mass_gev / 0.3)
            grid_lower_m = float(np.exp(np.log(raw_lower_m)))
            self.assertLess(grid_lower_m, raw_lower_m)
            proposal = KinematicProposal(
                model_id="toy",
                source="inclusive",
                mass_gev=mass_gev,
                proposal_ctau_m=grid_lower_m,
                proposal_seed=1,
                interpolation_points=4,
                resample_size=4,
                r_theta_rad=np.zeros(4),
                r_energy_gev=np.full(4, 10.0),
                epsilon_polar=0.5,
                visible_br=1.0,
                yield_per_pot_per_coupling_squared=1.0,
                unit_coupling_ctau_m=grid_lower_m,
                theta_min_rad=0.0,
                theta_max_rad=0.1,
                sanitation_policy="strict_core",
                input_fingerprints=(),
            )
            spectrum = adapter.evaluate_spectrum(
                proposal,
                grid_lower_m,
                123,
                cache_result=False,
            )
            self.assertGreater(spectrum.expected_events, 0.0)
            with self.assertRaisesRegex(ValueError, "preparation lifetime"):
                adapter.evaluate_spectrum(
                    proposal,
                    np.nextafter(grid_lower_m, 0.0),
                    123,
                    cache_result=False,
                )

    def test_lifetime_specific_proposal_is_exact_lifetime_only(self):
        cache = Mock(profile="smoke")
        adapter = EventCalcAdapter(SMOKE, cache=cache)
        proposal = KinematicProposal(
            model_id="toy",
            source="inclusive",
            mass_gev=0.3,
            proposal_ctau_m=0.5,
            proposal_seed=1,
            interpolation_points=4,
            resample_size=4,
            r_theta_rad=np.zeros(4),
            r_energy_gev=np.full(4, 10.0),
            epsilon_polar=0.5,
            visible_br=1.0,
            yield_per_pot_per_coupling_squared=1.0,
            unit_coupling_ctau_m=0.5,
            theta_min_rad=0.0,
            theta_max_rad=0.1,
            sanitation_policy="strict_core",
            input_fingerprints=(),
            energy_support_mode="lifetime_specific_truncated_Ea",
            e_min_sampling_min_gev=1.0,
            e_min_sampling_max_gev=2.0,
        )
        spectrum = adapter.evaluate_spectrum(
            proposal,
            0.5,
            123,
            cache_result=False,
        )
        self.assertGreater(spectrum.expected_events, 0.0)
        with self.assertRaisesRegex(ValueError, "preparation lifetime"):
            adapter.evaluate_spectrum(
                proposal,
                0.6,
                123,
                cache_result=False,
            )

    def test_model_evaluation_uses_profile_source_stride_and_template_offset(self):
        config = replace(
            SMOKE,
            templates=replace(SMOKE.templates, seed_offset=7),
        )
        cache = Mock(profile="smoke")
        cache.paths.return_value = (Mock(), Mock(), "combined-key")
        cache.load.return_value = None
        cache.save.return_value = {}
        adapter = EventCalcAdapter(config, cache=cache)
        adapter.prepare_kinematic_proposal = Mock(side_effect=(Mock(), Mock()))
        adapter.evaluate_spectrum = Mock(
            side_effect=(Mock(cache_key="primary"), Mock(cache_key="cascade"))
        )
        combined = Mock(cache_key=None)
        combined.arrays.return_value = {}
        combined.metadata.return_value = {}

        model_id = "alp_photon_combined"
        model_seed = config.seed_policy.model_seed(0.3, model_id, seed_offset=7)
        with patch(
            "alp_discrimination.eventcalc_adapter.combine_absolute_source_spectra",
            return_value=combined,
        ):
            adapter.evaluate_model(
                model_id,
                0.3,
                10.0,
                model_seed,
                "spectrum",
                proposal_ctau_m=np.exp(np.log(3.0)),
            )

        proposal_seeds = [
            call.args[3]
            for call in adapter.prepare_kinematic_proposal.call_args_list
        ]
        true_sample_seeds = [
            call.args[2]
            for call in adapter.evaluate_spectrum.call_args_list
        ]
        self.assertEqual(proposal_seeds, [54_328, 55_328])
        self.assertEqual(true_sample_seeds, [54_329, 55_329])
        self.assertEqual(
            {
                call.kwargs["proposal_ctau_m"]
                for call in adapter.prepare_kinematic_proposal.call_args_list
            },
            {np.exp(np.log(3.0))},
        )


    def test_model_evaluation_accepts_appended_week8_mass_seed_index(self):
        config = replace(
            SMOKE,
            templates=replace(SMOKE.templates, seed_offset=7),
        )
        cache = Mock(profile="smoke")
        cache.paths.return_value = (Mock(), Mock(), "combined-key")
        cache.load.return_value = None
        cache.save.return_value = {}
        adapter = EventCalcAdapter(config, cache=cache)
        adapter.prepare_kinematic_proposal = Mock(side_effect=(Mock(), Mock()))
        adapter.evaluate_spectrum = Mock(
            side_effect=(Mock(cache_key="primary"), Mock(cache_key="cascade"))
        )
        combined = Mock(cache_key=None)
        combined.arrays.return_value = {}
        combined.metadata.return_value = {}

        model_id = "alp_photon_combined"
        appended_mass_index = len(config.seed_policy.mass_order_gev) + 3
        model_seed = config.seed_policy.model_seed_from_indices(
            appended_mass_index,
            config.seed_policy.model_index(model_id),
            seed_offset=7,
        )

        with patch(
            "alp_discrimination.eventcalc_adapter.combine_absolute_source_spectra",
            return_value=combined,
        ):
            adapter.evaluate_model(
                model_id,
                2.5,
                10.0,
                model_seed,
                "spectrum",
                proposal_ctau_m=3.0,
            )

        proposal_seeds = [
            call.args[3]
            for call in adapter.prepare_kinematic_proposal.call_args_list
        ]
        true_sample_seeds = [
            call.args[2]
            for call in adapter.evaluate_spectrum.call_args_list
        ]
        self.assertEqual(
            proposal_seeds,
            [model_seed, model_seed + config.seed_policy.source_stride],
        )
        self.assertEqual(
            true_sample_seeds,
            [
                model_seed + config.seed_policy.true_sample_seed_offset,
                model_seed
                + config.seed_policy.source_stride
                + config.seed_policy.true_sample_seed_offset,
            ],
        )


if __name__ == "__main__":
    unittest.main()
