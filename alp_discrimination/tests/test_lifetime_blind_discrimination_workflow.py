from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from alp_discrimination.config import QUICK
from alp_discrimination.templates.lifetime_banks import load_template_bank
from alp_discrimination.physics.spectra import WeightedSpectrum
from alp_discrimination.eventcalc.proposals import EVENTCALC_FULL_SUPPORT_CTAU_M
from alp_discrimination.paths import profile_output_dir
from alp_discrimination.workflows.lifetime_blind_discrimination import (
    _mass_seed_indices,
    apply_cli_overrides,
    build_mass_bank,
    build_template_lifetime_grid_table,
    load_custom_lifetime_grid,
    parse_arguments,
    resolve_requested_masses,
    proposal_lifetime_for_target,
    resolve_template_output_dir,
    run_template_bank_workflow,
)

class _CounterCache:
    def counter_snapshot(self):
        return {"hits": 2, "misses": 1, "writes": 3, "rejected": 0}


class _FakeAdapter:
    def __init__(self, config):
        self.config = config
        self.cache = _CounterCache()
        self.calls = []

    def evaluate_model(
        self,
        model_id,
        mass_gev,
        ctau_m,
        model_seed,
        stage,
        *,
        proposal_ctau_m=None,
    ):
        self.calls.append(
            (model_id, mass_gev, ctau_m, model_seed, stage, proposal_ctau_m)
        )
        energies = np.repeat([0.5, 2.0, 20.0, 200.0], 4)
        model_scale = 1.0 if model_id == "alp_photon_combined" else 1.2
        lifetime_scale = 1.0 + 0.01 * np.log(ctau_m)
        pattern = np.repeat([1.0, 1.4, 1.8, 2.2], 4)
        weights = pattern * model_scale * lifetime_scale
        # Deliberately below both the old N_events >= 10 cut and the new
        # geom-only domain event level. The post-ECAL rate is diagnostic only.
        weights *= 1.0 / weights.sum()
        return WeightedSpectrum(
            model_id=model_id,
            source="combined",
            mass_gev=mass_gev,
            ctau_m=ctau_m,
            selection_name=self.config.selection_name,
            energies_gev=energies,
            absolute_event_weights=weights,
            expected_events=float(weights.sum()),
            seed=model_seed,
            generated_samples=20,
            accepted_samples=len(energies),
            exposure_pot=self.config.exposure_pot,
            visible_br=1.0,
            yield_per_pot_per_coupling_squared=1.0,
            unit_coupling_ctau_m=ctau_m,
            coupling_squared_gev_inv2=1.0,
            n_llp_total=12.0,
            epsilon_polar=1.0,
            epsilon_azimuthal=1.0,
            mean_decay_probability=1.0,
            preselection_expected_events=12.0,
            preselection_samples=20,
            source_expected_events={"combined": float(weights.sum())},
            cache_key=f"{model_id}-{ctau_m}",
        )


def _domains():
    return pd.DataFrame(
        [
            {
                "model": "ALP-photon-combined",
                "mass_GeV": 0.3,
                "event_level": 2.3,
                "interval_index": 1,
                "coupling_min_GeV_inv": 0.5,
                "coupling_max_GeV_inv": 1.0,
                "unit_coupling_ctau_m": 1.0,
                "ctau_min_m": 1.0,
                "ctau_max_m": 4.0,
            },
            {
                "model": "ALP-photon-combined",
                "mass_GeV": 0.3,
                "event_level": 2.3,
                "interval_index": 0,
                "coupling_min_GeV_inv": 0.1,
                "coupling_max_GeV_inv": 0.2,
                "unit_coupling_ctau_m": 1.0,
                "ctau_min_m": 25.0,
                "ctau_max_m": 100.0,
            },
            {
                "model": "ALP-SU2L",
                "mass_GeV": 0.3,
                "event_level": 2.3,
                "interval_index": 0,
                "coupling_min_GeV_inv": 0.2,
                "coupling_max_GeV_inv": 2.0,
                "unit_coupling_ctau_m": 1.0,
                "ctau_min_m": 0.25,
                "ctau_max_m": 25.0,
            },
        ]
    )


class LifetimeBlindDiscriminationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            QUICK,
            templates=replace(
                QUICK.templates,
                lifetime_points_per_model=3,
                initial_energy_bins=4,
                minimum_bin_n_eff=2.0,
            ),
        )

    def test_template_cli_overrides_are_validated_and_require_separate_output(self):
        args = parse_arguments(
            [
                "--lifetime-points-per-interval",
                "39",
                "--initial-energy-bins",
                "100",
                "--minimum-bin-n-eff",
                "125.5",
                "--output-dir",
                "convergence-output",
            ]
        )
        config = apply_cli_overrides(self.config, args)
        self.assertEqual(config.templates.lifetime_points_per_model, 39)
        self.assertEqual(config.templates.initial_energy_bins, 100)
        self.assertEqual(config.templates.minimum_bin_n_eff, 125.5)
        self.assertEqual(
            resolve_template_output_dir(config, args),
            Path("convergence-output"),
        )

        defaults = parse_arguments([])
        self.assertEqual(
            apply_cli_overrides(self.config, defaults).templates,
            self.config.templates,
        )
        self.assertEqual(
            resolve_template_output_dir(self.config, defaults),
            profile_output_dir(
                self.config.name,
                "lifetime_blind_discrimination_week8",
            ),
        )

        with self.assertRaisesRegex(ValueError, "explicit --output-dir"):
            resolve_template_output_dir(
                self.config,
                parse_arguments(["--initial-energy-bins", "100"]),
            )
        with self.assertRaisesRegex(ValueError, "at least 2"):
            apply_cli_overrides(
                self.config,
                parse_arguments(["--lifetime-points-per-interval", "1"]),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_cli_overrides(
                self.config,
                parse_arguments(["--initial-energy-bins", "0"]),
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            apply_cli_overrides(
                self.config,
                parse_arguments(["--minimum-bin-n-eff", "nan"]),
            )

    def test_disconnected_domains_and_common_bank_construction(self):
        domains = _domains()
        grid_table = build_template_lifetime_grid_table(domains, (0.3,), 3)
        photon_grid = grid_table.loc[
            grid_table["model"] == "ALP-photon-combined"
        ]
        self.assertEqual(photon_grid["interval_index"].tolist(), [1, 1, 1, 0, 0, 0])
        self.assertFalse(
            np.any(
                (photon_grid["ctau_m"].to_numpy(float) > 4.0)
                & (photon_grid["ctau_m"].to_numpy(float) < 25.0)
            )
        )

        adapter = _FakeAdapter(self.config)
        bank = build_mass_bank(
            config=self.config,
            adapter=adapter,
            mass_gev=0.3,
            domains=domains,
            mass_seed_index=0,
        )
        self.assertEqual(bank.photon_probabilities.shape, (6, 4))
        self.assertEqual(bank.su2_probabilities.shape, (3, 4))
        np.testing.assert_array_equal(bank.energy_edges_gev.shape, (5,))
        np.testing.assert_allclose(bank.photon_probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all(bank.photon_probabilities > 0.0))
        np.testing.assert_array_equal(
            bank.photon_interval_index,
            [1, 1, 1, 0, 0, 0],
        )
        np.testing.assert_allclose(
            bank.photon_allowed_intervals_m,
            [[25.0, 100.0], [1.0, 4.0]],
        )
        # Old frozen seeds are retained for old masses.
        self.assertEqual({call[3] for call in adapter.calls}, {54_321, 54_421})
        self.assertEqual({call[4] for call in adapter.calls}, {"spectrum"})
        proposal_by_model = {
            model_id: {call[5] for call in adapter.calls if call[0] == model_id}
            for model_id in ("alp_photon_combined", "alp_su2l")
        }
        self.assertEqual(
            proposal_by_model["alp_photon_combined"],
            {1.0, 2.0, EVENTCALC_FULL_SUPPORT_CTAU_M},
        )
        self.assertEqual(
            proposal_by_model["alp_su2l"],
            {0.25, EVENTCALC_FULL_SUPPORT_CTAU_M},
        )
        self.assertTrue(np.all(bank.photon_n_events < 2.3))

    def test_workflow_writes_week8_provenance_and_portable_manifest(self):
        adapter = _FakeAdapter(self.config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain_path = root / "allowed_ctau_domains.csv"
            _domains().to_csv(domain_path, index=False)
            output_dir = root / "outputs" / self.config.name
            summary = run_template_bank_workflow(
                config=self.config,
                adapter=adapter,
                domain_path=domain_path,
                output_dir=output_dir,
            )
            self.assertEqual(summary["number_of_photon_lifetimes"].tolist(), [6])
            self.assertEqual(summary["number_of_photon_intervals"].tolist(), [2])
            self.assertTrue(
                (output_dir / "template_banks" / "template_bank_ma_0p3.npz").is_file()
            )
            self.assertTrue(
                (output_dir / "tables" / "week8_template_lifetime_grid.csv").is_file()
            )
            payload = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(payload["profile"], "quick")
            self.assertEqual(payload["cache_stats"]["hits"], 2)
            self.assertEqual(payload["domain_event_level"], 2.3)
            self.assertIn("exact-lifetime adaptive-Emin", payload["proposal_strategy"])
            self.assertEqual(
                payload["eventcalc_full_support_ctau_m"],
                EVENTCALC_FULL_SUPPORT_CTAU_M,
            )
            self.assertEqual(
                payload["lifetime_points_per_connected_interval"],
                3,
            )
            self.assertEqual(payload["initial_energy_bins"], 4)
            self.assertEqual(payload["minimum_bin_N_eff"], 2.0)
            self.assertFalse(payload["old_N_events_ge_10_cut_applied"])
            self.assertFalse(
                payload["old_mass_scaled_ctau_lower_cut_applied"]
            )
            serialized = json.dumps(payload)
            self.assertNotIn("/Users/", serialized)
            self.assertNotIn(directory, serialized)
            with self.assertRaises(FileExistsError):
                run_template_bank_workflow(
                    config=self.config,
                    adapter=adapter,
                    domain_path=domain_path,
                    output_dir=output_dir,
                )

    def test_custom_grid_and_fixed_edges_are_preserved_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain_path = root / "allowed_ctau_domains.csv"
            domains = _domains()
            domains.to_csv(domain_path, index=False)

            source_output = root / "source"
            run_template_bank_workflow(
                config=self.config,
                adapter=_FakeAdapter(self.config),
                domain_path=domain_path,
                output_dir=source_output,
            )
            source_bank_path = (
                source_output / "template_banks/template_bank_ma_0p3.npz"
            )

            custom_path = root / "custom_lifetimes.csv"
            custom = build_template_lifetime_grid_table(
                domains,
                (0.3,),
                3,
            )[["model", "mass_GeV", "interval_index", "ctau_m"]]
            custom.to_csv(custom_path, index=False, float_format="%.17g")

            loaded_grid = load_custom_lifetime_grid(
                custom_path,
                domains=domains,
                masses=(0.3,),
            )
            self.assertEqual(len(loaded_grid), 9)
            self.assertEqual(set(loaded_grid["model_id"]), {
                "alp_photon_combined",
                "alp_su2l",
            })

            destination = root / "derived"
            run_template_bank_workflow(
                config=self.config,
                adapter=_FakeAdapter(self.config),
                domain_path=domain_path,
                output_dir=destination,
                energy_edges_from_bank=source_bank_path,
                lifetime_grid_path=custom_path,
            )

            source = load_template_bank(source_bank_path)
            derived = load_template_bank(
                destination / "template_banks/template_bank_ma_0p3.npz"
            )
            np.testing.assert_array_equal(
                derived.energy_edges_gev,
                source.energy_edges_gev,
            )
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["energy_binning_mode"], "fixed_from_bank")
            self.assertEqual(manifest["lifetime_grid_mode"], "custom_csv")
            self.assertEqual(manifest["number_of_fixed_energy_bins"], 4)
            self.assertIsNone(
                manifest["lifetime_points_per_connected_interval"]
            )

    def test_custom_grid_rejects_missing_endpoints_and_fixed_edges_never_merge(self):
        domains = _domains()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_grid.csv"
            bad = build_template_lifetime_grid_table(
                domains,
                (0.3,),
                3,
            )[["model", "mass_GeV", "interval_index", "ctau_m"]]
            bad = bad.drop(bad.index[0])
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "retain both endpoints"):
                load_custom_lifetime_grid(
                    path,
                    domains=domains,
                    masses=(0.3,),
                )

        strict = replace(
            self.config,
            templates=replace(
                self.config.templates,
                minimum_bin_n_eff=5.0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "Fixed energy edges fail"):
            build_mass_bank(
                config=strict,
                adapter=_FakeAdapter(strict),
                mass_gev=0.3,
                domains=domains,
                mass_seed_index=0,
                fixed_energy_edges_gev=np.array(
                    [0.3, 0.6, 1.0, 4.0, 400.0]
                ),
            )

    def test_proposal_lifetime_strategy_preserves_short_lifetime_sampling(self):
        self.assertEqual(proposal_lifetime_for_target(0.25), 0.25)
        self.assertEqual(proposal_lifetime_for_target(2.0), 2.0)
        self.assertEqual(
            proposal_lifetime_for_target(3.0),
            EVENTCALC_FULL_SUPPORT_CTAU_M,
        )

    def test_requested_masses_and_seed_indices_are_stable(self):
        available = (0.3, 0.4, 0.5, 1.2, 2.5)
        self.assertEqual(
            resolve_requested_masses([2.5, 0.3], available),
            (0.3, 2.5),
        )
        with self.assertRaises(ValueError):
            resolve_requested_masses([0.6], available)

        indices = _mass_seed_indices(self.config, available)
        self.assertEqual(indices[0.3], 0)
        self.assertEqual(indices[0.4], 1)
        self.assertEqual(indices[0.5], 2)
        self.assertGreaterEqual(indices[1.2], len(self.config.seed_policy.mass_order_gev))
        self.assertGreater(indices[2.5], indices[1.2])


if __name__ == "__main__":
    unittest.main()
