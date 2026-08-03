from dataclasses import FrozenInstanceError, asdict
from types import MappingProxyType
import unittest

from analysis2.config import (
    PRODUCTION,
    PROFILES,
    QUICK,
    SMOKE,
    VALIDATION,
    PRODUCTION_MASSES_GEV,
    get_config,
    profiled_likelihood_seeds,
    lower_ctau_m,
    spectrum_model_seed,
    spectrum_source_seed,
    template_ecal_seed,
    template_model_seed,
    template_source_seed,
    template_true_sample_seed,
)
from analysis2.paths import profile_cache_dir, profile_output_dir


class ProfileConfigurationTests(unittest.TestCase):
    def test_production_is_the_exact_frozen_profile(self):
        self.assertEqual(PRODUCTION.masses_gev, PRODUCTION_MASSES_GEV)
        self.assertEqual(PRODUCTION.selection_name, "diphoton_ecal")
        self.assertEqual(
            (PRODUCTION.ctau_sampling.interpolation_points, PRODUCTION.ctau_sampling.resample_size),
            (10_000_000, 1_000_000),
        )
        self.assertEqual(PRODUCTION.template_sampling, PRODUCTION.ctau_sampling)

        lifetime = PRODUCTION.lifetimes
        self.assertEqual(
            (
                lifetime.event_threshold,
                lifetime.maximum_ctau_m,
                lifetime.coarse_factor,
                lifetime.bisection_steps,
            ),
            (10.0, 1_000.0, 1.7, 14),
        )
        self.assertEqual(
            lifetime.diagnostic_endpoint_convention,
            "fixed_step_log_bisection_midpoint",
        )

        templates = PRODUCTION.templates
        self.assertEqual(templates.lifetime_points_per_model, 20)
        self.assertEqual(templates.observable_endpoint_convention, "log_log_rate_interpolation")
        self.assertEqual(templates.log_endpoint_padding_fraction, 0.002)
        self.assertEqual(templates.energy_lower_bound_convention, "mass_gev")
        self.assertEqual(
            (
                templates.initial_energy_bins,
                templates.energy_max_gev,
                templates.minimum_bin_n_eff,
                templates.jeffreys_alpha,
            ),
            (50, 400.0, 100.0, 0.5),
        )

        profiling = PRODUCTION.profiled_likelihood
        self.assertEqual(profiling.pseudoexperiments_per_truth_and_seed, 100_000)
        self.assertEqual(profiling.seeds, (73_241, 83_244, 93_247, 103_250, 113_253))
        self.assertEqual(profiling.maximum_observed_events, 12)
        self.assertEqual(profiling.chunk_size, 5_000)
        self.assertEqual(profiling.target_accuracy, 0.90)
        self.assertEqual(profiling.tie_tolerance, 1.0e-12)
        self.assertEqual(profiling.persistent_criterion, "all_larger_tested_event_counts")
        self.assertTrue(profiling.shape_only)
        self.assertTrue(profiling.independent_lifetime_profiling)

    def test_asdict_fingerprints_endpoint_and_profile_conventions(self):
        payload = asdict(PRODUCTION)
        self.assertEqual(
            payload["templates"]["observable_endpoint_convention"],
            "log_log_rate_interpolation",
        )
        self.assertEqual(payload["templates"]["log_endpoint_padding_fraction"], 0.002)
        self.assertEqual(
            payload["lifetimes"]["diagnostic_endpoint_convention"],
            "fixed_step_log_bisection_midpoint",
        )
        self.assertEqual(payload["profiled_likelihood"]["number_of_seeds"], 5)
        self.assertEqual(payload["profiled_likelihood"]["seed_step"], 10_003)

    def test_profiles_and_registry_are_immutable(self):
        self.assertIsInstance(PROFILES, MappingProxyType)
        with self.assertRaises(TypeError):
            PROFILES["new"] = PRODUCTION
        with self.assertRaises(FrozenInstanceError):
            PRODUCTION.name = "changed"
        with self.assertRaises(FrozenInstanceError):
            PRODUCTION.templates.jeffreys_alpha = 1.0

    def test_quick_validation_and_smoke_have_distinct_namespaces(self):
        self.assertIs(get_config("quick"), QUICK)
        self.assertIs(get_config("validation"), VALIDATION)
        self.assertIs(get_config("smoke"), SMOKE)
        self.assertEqual({"production", "quick", "validation", "smoke"}, set(PROFILES))
        for left, right in ((PRODUCTION, QUICK), (QUICK, VALIDATION), (QUICK, SMOKE)):
            self.assertNotEqual(profile_cache_dir(left.name), profile_cache_dir(right.name))
            self.assertNotEqual(profile_output_dir(left.name), profile_output_dir(right.name))

    def test_frozen_seed_policy(self):
        policy = PRODUCTION.seed_policy
        self.assertEqual(
            (
                policy.base_seed,
                policy.mass_stride,
                policy.model_stride,
                policy.source_stride,
                policy.true_sample_seed_offset,
                policy.ecal_seed_offset,
            ),
            (54_321, 10_000, 100, 1_000, 1, 2),
        )
        self.assertEqual(policy.mass_order_gev, PRODUCTION_MASSES_GEV)
        self.assertEqual(template_model_seed(0.3, "alp_photon_combined"), 54_321)
        self.assertEqual(template_model_seed(0.3, "alp_su2l"), 54_421)
        self.assertEqual(template_model_seed(0.4, "alp_photon_combined"), 64_321)
        self.assertEqual(template_model_seed(1.05, "alp_photon_combined"), 124_321)
        self.assertEqual(template_source_seed(0.3, "alp_photon_combined", 1), 55_321)
        self.assertEqual(template_true_sample_seed(0.3, "alp_photon_combined", 1), 55_322)
        self.assertEqual(template_ecal_seed(0.3, "alp_photon_combined", 1), 55_323)
        self.assertEqual(spectrum_model_seed(1, 0), 64_321)
        self.assertEqual(spectrum_source_seed(54_321, 1), 55_321)
        self.assertEqual(profiled_likelihood_seeds(), PRODUCTION.profiled_likelihood.seeds)

    def test_lower_lifetime_bound_preserves_frozen_operation_order(self):
        for mass_gev in PRODUCTION_MASSES_GEV:
            self.assertEqual(lower_ctau_m(mass_gev), 3.0 * (mass_gev / 0.3))
        self.assertNotEqual(lower_ctau_m(0.4), 3.0 * 0.4 / 0.3)
        self.assertNotEqual(lower_ctau_m(0.9), 3.0 * 0.9 / 0.3)


if __name__ == "__main__":
    unittest.main()
