from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from analysis2.cache import cache_key
from analysis2.config import PRODUCTION
from analysis2.distance_statistics import total_variation_matrix
from analysis2.eventcalc_adapter import _proposal_identity
from analysis2.frozen_reference import (
    ENDPOINTS,
    FIVE_SEED_ACCURACY_AT_THRESHOLD,
    MASSES_GEV,
    MINIMUM_TOTAL_VARIATION,
    NUMBER_OF_ENERGY_BINS,
    PERSISTENT_EVENT_THRESHOLDS,
    PROFILE_SEEDS,
    SELECTED_PROBABILITIES,
)
from analysis2.lifetime_template_banks import load_template_bank
from analysis2.models import get_model
from analysis2.observable_domains import (
    collect_observable_domains,
    load_lifetime_scan,
    padded_lifetime_grid,
)
from analysis2.paths import LEGACY_ANALYSIS_ROOT
from analysis2.profiled_reduction import (
    build_conservative_seed_envelope,
    minimum_persistent_events,
)
from analysis2.templates import (
    common_adaptive_energy_edges,
    jeffreys_regularized_probabilities,
)
from analysis2.tests.helpers import spectrum


FROZEN_ROOT = LEGACY_ANALYSIS_ROOT / "lifetime_blind_discrimination_final"
SCAN_PATH = LEGACY_ANALYSIS_ROOT / "ctau_scan" / "ctau_scan.csv"
LEGACY_MODEL_NAME = {
    "alp_photon_combined": "ALP-photon-combined",
    "alp_su2l": "ALP-SU2L",
}


def mass_token(mass_gev: float) -> str:
    return f"{mass_gev:g}".replace(".", "p")


def bank_path(mass_gev: float) -> Path:
    return FROZEN_ROOT / "template_banks" / (
        f"template_bank_ma_{mass_token(mass_gev)}.npz"
    )


class FrozenReferenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = load_lifetime_scan(SCAN_PATH)
        cls.domains = collect_observable_domains(
            cls.scan,
            threshold=PRODUCTION.lifetimes.event_threshold,
        )
        cls.banks = {
            mass_gev: load_template_bank(bank_path(mass_gev))
            for mass_gev in MASSES_GEV
        }

    def test_log_log_and_bisection_endpoints_are_independent_targets(self):
        for expected in ENDPOINTS:
            key = (LEGACY_MODEL_NAME[expected.model_id], expected.mass_gev)
            domain = self.domains[key]
            actual_raw = (domain.lower_m, domain.upper_m)
            actual_bisection = (
                domain.bisection_lower_m,
                domain.bisection_upper_m,
            )
            expected_raw = (
                expected.raw_log_log_lower_m,
                expected.raw_log_log_upper_m,
            )
            expected_bisection = (
                expected.bisection_lower_m,
                expected.bisection_upper_m,
            )
            np.testing.assert_allclose(
                actual_raw, expected_raw, rtol=0.0, atol=5.0e-13
            )
            np.testing.assert_allclose(
                actual_bisection,
                expected_bisection,
                rtol=0.0,
                atol=5.0e-13,
            )
            self.assertNotEqual(domain.upper_m, domain.bisection_upper_m)

    def test_inward_shift_and_bank_domains_match_frozen_values(self):
        for expected in ENDPOINTS:
            domain = self.domains[
                (LEGACY_MODEL_NAME[expected.model_id], expected.mass_gev)
            ]
            grid = padded_lifetime_grid(
                domain,
                PRODUCTION.templates.lifetime_points_per_model,
                PRODUCTION.templates.log_endpoint_padding_fraction,
            )
            np.testing.assert_allclose(
                [grid[0], grid[-1]],
                [expected.padded_grid_lower_m, expected.padded_grid_upper_m],
                rtol=0.0,
                atol=5.0e-13,
            )
            prefix = (
                "photon" if expected.model_id == "alp_photon_combined" else "su2"
            )
            bank = self.banks[expected.mass_gev]
            np.testing.assert_allclose(
                getattr(bank, f"{prefix}_interval_m"),
                [expected.raw_log_log_lower_m, expected.raw_log_log_upper_m],
                rtol=0.0,
                atol=5.0e-13,
            )
            np.testing.assert_allclose(
                getattr(bank, f"{prefix}_ctau_m")[[0, -1]],
                [expected.padded_grid_lower_m, expected.padded_grid_upper_m],
                rtol=0.0,
                atol=5.0e-13,
            )

    def test_template_counts_bins_and_selected_probabilities(self):
        for mass_gev, number_of_bins in zip(MASSES_GEV, NUMBER_OF_ENERGY_BINS):
            bank = self.banks[mass_gev]
            self.assertEqual(len(bank.photon_ctau_m), 20)
            self.assertEqual(len(bank.su2_ctau_m), 20)
            self.assertEqual(bank.number_of_energy_bins, number_of_bins)

        for expected in SELECTED_PROBABILITIES:
            probabilities = getattr(
                self.banks[expected.mass_gev],
                f"{expected.model_prefix}_probabilities",
            )
            self.assertEqual(
                probabilities[expected.lifetime_index, expected.bin_index],
                expected.probability,
            )

    def test_minimum_total_variation_distances(self):
        actual = []
        for mass_gev in MASSES_GEV:
            bank = self.banks[mass_gev]
            distances = total_variation_matrix(
                bank.photon_probabilities,
                bank.su2_probabilities,
            )
            actual.append(float(np.min(distances)))
        np.testing.assert_allclose(
            actual,
            MINIMUM_TOTAL_VARIATION,
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_persistent_thresholds_and_five_seed_summary(self):
        actual_thresholds = []
        for mass_index, mass_gev in enumerate(MASSES_GEV):
            token = mass_token(mass_gev)
            seed_table = pd.read_csv(
                FROZEN_ROOT
                / "profiled_likelihood"
                / "tables"
                / f"profiled_worst_case_by_seed_ma_{token}.csv"
            )
            envelope = build_conservative_seed_envelope(seed_table)
            actual_thresholds.append(
                minimum_persistent_events(
                    envelope,
                    accuracy_column="worst_case_correct_fraction",
                    target_accuracy=PRODUCTION.profiled_likelihood.target_accuracy,
                )
            )

            threshold = PERSISTENT_EVENT_THRESHOLDS[mass_index]
            rows = seed_table.loc[
                seed_table["number_of_events"] == threshold
            ].sort_values("seed")
            self.assertEqual(tuple(rows["seed"]), PROFILE_SEEDS)
            np.testing.assert_allclose(
                rows["worst_case_correct_fraction"],
                FIVE_SEED_ACCURACY_AT_THRESHOLD[mass_index],
                rtol=0.0,
                atol=5.0e-15,
            )
        self.assertEqual(tuple(actual_thresholds), PERSISTENT_EVENT_THRESHOLDS)

    def test_common_adaptive_binning_uses_all_models_and_lifetimes(self):
        spectra = {
            "photon_short": spectrum([1.2, 2.2, 4.2], [3.0, 3.0, 3.0]),
            "photon_long": spectrum([1.2, 2.2, 4.2], [3.0, 3.0, 3.0]),
            "su2_short": spectrum([1.2, 2.2, 4.2], [3.0, 3.0, 3.0]),
            "su2_long": spectrum([1.2, 2.2, 4.2], [3.0, 3.0, 3.0]),
        }
        spectra["photon_long"].ctau_m = 20.0
        spectra["su2_short"].model_id = "su2"
        spectra["su2_long"].model_id = "su2"
        spectra["su2_long"].ctau_m = 20.0
        edges = common_adaptive_energy_edges(
            spectra,
            np.array([1.0, 2.0, 4.0, 8.0]),
            minimum_n_eff=2.0,
        )
        # Every template participates in one deterministic merge sequence.
        np.testing.assert_array_equal(edges, [1.0, 8.0])

    def test_jeffreys_smoothing_uses_total_effective_sample_size(self):
        weighted = spectrum([1.2, 1.8, 2.2], [1.0, 3.0, 2.0])
        probabilities, total_n_eff = jeffreys_regularized_probabilities(
            weighted,
            np.array([1.0, 2.0, 3.0]),
            alpha=0.5,
        )
        expected_n_eff = 36.0 / 14.0
        raw = np.array([4.0 / 6.0, 2.0 / 6.0])
        expected = (expected_n_eff * raw + 0.5) / (expected_n_eff + 1.0)
        self.assertEqual(total_n_eff, expected_n_eff)
        np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1.0e-16)

    def test_proposal_cache_changes_with_source_and_template_seeds(self):
        model = get_model("alp_photon_combined")
        source = model.sources[0]
        common = dict(
            config=PRODUCTION,
            model=model,
            source=source,
            mass_gev=0.3,
            proposal_ctau_m=3.0,
            sampling=PRODUCTION.template_sampling,
            sanitation_policy="strict_core",
            fingerprints=(),
        )
        base_seed = PRODUCTION.seed_policy.source_proposal_seed(
            0.3, model.identifier, 0
        )
        source_seed = PRODUCTION.seed_policy.source_proposal_seed(
            0.3, model.identifier, 1
        )
        changed_template_config = replace(
            PRODUCTION,
            templates=replace(PRODUCTION.templates, seed_offset=1),
        )
        template_seed = changed_template_config.seed_policy.source_proposal_seed(
            0.3,
            model.identifier,
            0,
            seed_offset=changed_template_config.templates.seed_offset,
        )
        keys = {
            cache_key(_proposal_identity(seed=seed, **common))
            for seed in (base_seed, source_seed, template_seed)
        }
        self.assertEqual(len(keys), 3)


if __name__ == "__main__":
    unittest.main()
