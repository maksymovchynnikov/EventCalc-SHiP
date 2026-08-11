import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import alp_discrimination.statistics.profiled as profiled_statistics

from alp_discrimination.statistics.distances import (
    DISTANCE_TABLE_COLUMNS,
    build_distance_table,
    total_variation_matrix,
)
from alp_discrimination.statistics.reduction import (
    CONSERVATIVE_ENVELOPE_COLUMNS,
    SEED_WORST_CASE_COLUMNS,
    build_conservative_seed_envelope,
    build_seed_worst_case_table,
    minimum_persistent_events,
)
from alp_discrimination.statistics.profiled import (
    PROFILED_ACCURACY_COLUMNS,
    combine_profiled_truth_tables,
    profile_log_likelihoods,
    simulate_truth_template,
    stable_truth_rng,
)


class DistanceStatisticsTests(unittest.TestCase):
    def test_pairwise_total_variation_and_legacy_table_order(self):
        photon = np.array([[0.5, 0.5], [1.0, 0.0]])
        su2 = np.array([[0.5, 0.5], [0.0, 1.0], [0.25, 0.75]])
        distances = total_variation_matrix(photon, su2)
        np.testing.assert_array_equal(
            distances,
            np.array([[0.0, 0.5, 0.25], [0.5, 1.0, 0.75]]),
        )

        table = build_distance_table(
            mass_gev=0.75,
            photon_ctau_m=np.array([3.0, 30.0]),
            photon_expected_events=np.array([100.0, 10.0]),
            su2_ctau_m=np.array([4.0, 40.0, 400.0]),
            su2_expected_events=np.array([120.0, 20.0, 2.0]),
            distances=distances,
        )
        self.assertEqual(tuple(table.columns), DISTANCE_TABLE_COLUMNS)
        np.testing.assert_array_equal(table["D_TV"], distances.ravel())
        np.testing.assert_array_equal(
            table[["photon_lifetime_index", "su2_lifetime_index"]],
            np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
        )


class ProfiledLikelihoodTests(unittest.TestCase):
    def test_blocked_profiler_is_bitwise_identical_to_legacy_vectorization(self):
        rng = np.random.default_rng(20260803)
        templates = rng.dirichlet(np.ones(6), size=40)
        sampled_bins = rng.choice(
            6,
            size=(17, 53),
            replace=True,
            p=templates[7],
        )
        event_counts = np.array([1, 2, 7, 19, 53])

        log_templates = np.log(templates)
        legacy_contributions = log_templates[:, sampled_bins]
        np.cumsum(legacy_contributions, axis=2, out=legacy_contributions)
        legacy = np.max(
            legacy_contributions[:, :, event_counts - 1],
            axis=0,
        )

        # Force several pseudoexperiment blocks while retaining the legacy
        # lifetime-fast, rank-three cumulative-sum path within each block.
        with patch.object(
            profiled_statistics,
            "_PROFILE_TEMPORARY_TARGET_BYTES",
            1,
        ):
            blocked = profile_log_likelihoods(
                sampled_bins,
                templates,
                event_counts,
            )
        np.testing.assert_array_equal(blocked, legacy)

    def test_lifetimes_are_profiled_independently_between_models(self):
        samples = np.array([[0, 0], [1, 1]])
        photon = np.array([[0.9, 0.1], [0.6, 0.4]])
        # A deliberately different number of SU(2)_L lifetimes makes a
        # same-index pairing ill-defined.  The maxima below come from photon
        # indices (0, 1) and SU(2)_L indices (1, 2), respectively.
        su2 = np.array([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8]])

        photon_best = profile_log_likelihoods(samples, photon, [1, 2])
        su2_best = profile_log_likelihoods(samples, su2, [1, 2])
        np.testing.assert_allclose(
            photon_best,
            [[np.log(0.9), 2.0 * np.log(0.9)],
             [np.log(0.4), 2.0 * np.log(0.4)]],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            su2_best,
            [[np.log(0.8), 2.0 * np.log(0.8)],
             [np.log(0.8), 2.0 * np.log(0.8)]],
            rtol=0.0,
            atol=1.0e-15,
        )
        statistic = 2.0 * (su2_best - photon_best)
        self.assertLess(statistic[0, 0], 0.0)
        self.assertGreater(statistic[1, 0], 0.0)

    def test_correlated_prefixes_and_chunked_stable_stream(self):
        arguments = {
            "mass_gev": 0.75,
            "truth_model": "photon",
            "truth_index": 1,
            "truth_ctau_m": 4.0,
            "truth_probabilities": np.array([0.6, 0.3, 0.1]),
            "photon_probabilities": np.array(
                [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]
            ),
            "su2_probabilities": np.array(
                [[0.3, 0.3, 0.4], [0.1, 0.2, 0.7]]
            ),
            "event_counts": np.array([1, 2, 4]),
            "number_of_pseudoexperiments": 11,
            "seed": 17,
            "tie_tolerance": 1.0e-12,
        }
        chunked = simulate_truth_template(**arguments, chunk_size=4)
        unchunked = simulate_truth_template(**arguments, chunk_size=11)
        exact_columns = [
            "correct_fraction",
            "selected_photon_fraction",
            "selected_su2_fraction",
            "tie_fraction",
        ]
        pd.testing.assert_frame_equal(
            chunked[exact_columns],
            unchunked[exact_columns],
            check_exact=True,
        )
        np.testing.assert_allclose(
            chunked[["mean_profile_statistic_T", "std_profile_statistic_T"]],
            unchunked[["mean_profile_statistic_T", "std_profile_statistic_T"]],
            rtol=0.0,
            atol=5.0e-15,
        )

    def test_progressive_ranges_reproduce_direct_classification_exactly(self):
        arguments = {
            "mass_gev": 0.75,
            "truth_model": "su2",
            "truth_index": 1,
            "truth_ctau_m": 9.0,
            "truth_probabilities": np.array([0.2, 0.5, 0.3]),
            "photon_probabilities": np.array(
                [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]
            ),
            "su2_probabilities": np.array(
                [[0.3, 0.3, 0.4], [0.1, 0.2, 0.7]]
            ),
            "event_counts": np.array([2, 4, 7]),
            "seed": 71,
            "chunk_size": 13,
            "tie_tolerance": 1.0e-12,
        }
        direct = simulate_truth_template(
            **arguments,
            number_of_pseudoexperiments=100,
        )
        first = simulate_truth_template(
            **arguments,
            number_of_pseudoexperiments=40,
        )
        second = simulate_truth_template(
            **arguments,
            number_of_pseudoexperiments=60,
            pseudoexperiment_start=40,
        )
        staged = combine_profiled_truth_tables([first, second])

        exact_columns = [
            "correct_fraction",
            "selected_photon_fraction",
            "selected_su2_fraction",
            "tie_fraction",
        ]
        pd.testing.assert_frame_equal(
            staged[exact_columns],
            direct[exact_columns],
            check_exact=True,
        )
        np.testing.assert_allclose(
            staged[["mean_profile_statistic_T", "std_profile_statistic_T"]],
            direct[["mean_profile_statistic_T", "std_profile_statistic_T"]],
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertEqual(
            set(staged["number_of_pseudoexperiments"]),
            {100},
        )

    def test_ties_are_split_equally_with_the_legacy_schema(self):
        identical = np.array([[0.4, 0.6], [0.7, 0.3]])
        result = simulate_truth_template(
            mass_gev=1.0,
            truth_model="su2",
            truth_index=0,
            truth_ctau_m=5.0,
            truth_probabilities=identical[0],
            photon_probabilities=identical,
            su2_probabilities=identical,
            event_counts=np.array([1, 3]),
            number_of_pseudoexperiments=17,
            seed=9,
            chunk_size=5,
            tie_tolerance=1.0e-12,
        )
        self.assertEqual(tuple(result.columns), PROFILED_ACCURACY_COLUMNS)
        np.testing.assert_array_equal(result["correct_fraction"], 0.5)
        np.testing.assert_array_equal(result["selected_photon_fraction"], 0.5)
        np.testing.assert_array_equal(result["selected_su2_fraction"], 0.5)
        np.testing.assert_array_equal(result["tie_fraction"], 1.0)
        np.testing.assert_array_equal(result["mean_profile_statistic_T"], 0.0)
        np.testing.assert_array_equal(result["std_profile_statistic_T"], 0.0)

    def test_stable_truth_rng_preserves_seed_sequence_mapping(self):
        draws = stable_truth_rng(
            seed=73_241,
            mass_gev=0.3,
            truth_model="su2",
            truth_index=7,
        ).integers(0, 2**31, size=8)
        np.testing.assert_array_equal(
            draws,
            [
                1_717_178_185,
                842_891_636,
                649_484_923,
                469_373_596,
                2_094_842_941,
                1_383_880_904,
                1_453_864_082,
                2_114_846_439,
            ],
        )


class ProfiledReductionTests(unittest.TestCase):
    @staticmethod
    def _detailed_table() -> pd.DataFrame:
        rows = []
        accuracies = {
            11: {"photon": [0.91, 0.91], "su2": [0.92, 0.93]},
            22: {"photon": [0.90, 0.94], "su2": [0.89, 0.96]},
        }
        for seed, by_model in accuracies.items():
            for model, values in by_model.items():
                for lifetime_index, accuracy in enumerate(values):
                    rows.append(
                        {
                            "mass_GeV": 0.5,
                            "seed": seed,
                            "truth_model": model,
                            "truth_lifetime_index": lifetime_index,
                            "truth_ctau_m": 10.0 ** lifetime_index,
                            "number_of_events": 3,
                            "correct_fraction": accuracy,
                        }
                    )
        return pd.DataFrame(rows)

    def test_truth_lifetime_model_and_seed_worst_cases(self):
        seed_table = build_seed_worst_case_table(self._detailed_table())
        self.assertEqual(tuple(seed_table.columns), SEED_WORST_CASE_COLUMNS)
        self.assertEqual(seed_table.loc[0, "photon_limiting_lifetime_index"], 0)
        self.assertEqual(seed_table.loc[0, "worst_case_correct_fraction"], 0.91)
        self.assertEqual(seed_table.loc[1, "worst_case_correct_fraction"], 0.89)

        envelope = build_conservative_seed_envelope(seed_table)
        self.assertEqual(tuple(envelope.columns), CONSERVATIVE_ENVELOPE_COLUMNS)
        self.assertEqual(envelope.loc[0, "photon_truth_worst_accuracy"], 0.90)
        self.assertEqual(envelope.loc[0, "su2_truth_worst_accuracy"], 0.89)
        self.assertEqual(envelope.loc[0, "worst_case_correct_fraction"], 0.89)
        self.assertEqual(envelope.loc[0, "limiting_seed"], 22)
        self.assertEqual(envelope.loc[0, "limiting_truth_model"], "su2")

    def test_persistent_threshold_rejects_an_earlier_nonmonotonic_crossing(self):
        curve = pd.DataFrame(
            {
                "number_of_events": [5, 2, 4, 1, 3],
                "worst_case_correct_fraction": [0.92, 0.91, 0.91, 0.89, 0.89],
            }
        )
        threshold = minimum_persistent_events(
            curve,
            accuracy_column="worst_case_correct_fraction",
            target_accuracy=0.90,
        )
        self.assertEqual(threshold, 4)


if __name__ == "__main__":
    unittest.main()
