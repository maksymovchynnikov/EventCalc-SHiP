from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis2.config import QUICK
from analysis2.spectra import WeightedSpectrum
from analysis2.workflows.lifetime_blind_discrimination import (
    build_mass_bank,
    collect_profile_domains,
    profile_domain_table,
    resolve_requested_masses,
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
        weights *= 11.0 / weights.sum()
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


def _scan(config):
    rows = []
    rates = {
        "ALP-photon-combined": (20.0, 15.0, 5.0),
        "ALP-SU2L": (20.0, 13.0, 4.0),
    }
    for model, values in rates.items():
        for ctau_m, rate in zip((3.0, 10.0, 30.0), values):
            rows.append(
                {
                    "profile": config.name,
                    "selection_name": config.selection_name,
                    "model": model,
                    "mass_GeV": 0.3,
                    "ctau_m": ctau_m,
                    "N_events": rate,
                    "passes_event_cut": rate >= 10.0,
                }
            )
    return pd.DataFrame(rows)


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

    def test_domain_conventions_and_common_bank_construction(self):
        domains = collect_profile_domains(_scan(self.config), self.config, (0.3,))
        table = profile_domain_table(domains, self.config)
        photon = table.loc[table["model"] == "ALP-photon-combined"].iloc[0]
        crossing_fraction = (np.log(10.0) - np.log(15.0)) / (
            np.log(5.0) - np.log(15.0)
        )
        expected_crossing = np.exp(
            np.log(10.0)
            + crossing_fraction * (np.log(30.0) - np.log(10.0))
        )
        expected_grid_upper = np.exp(
            np.log(expected_crossing)
            - 0.002 * (np.log(expected_crossing) - np.log(3.0))
        )
        self.assertEqual(photon["template_domain_lower_m"], 3.0)
        self.assertAlmostEqual(photon["template_grid_lower_m"], 3.0)
        self.assertAlmostEqual(
            photon["template_domain_upper_m"], expected_crossing
        )
        self.assertAlmostEqual(
            photon["template_grid_upper_m"], expected_grid_upper
        )
        self.assertEqual(
            photon["bisection_diagnostic_upper_m"],
            np.sqrt(10.0 * 30.0),
        )

        adapter = _FakeAdapter(self.config)
        bank = build_mass_bank(
            config=self.config,
            adapter=adapter,
            mass_gev=0.3,
            domains=domains,
        )
        self.assertEqual(bank.photon_probabilities.shape, (3, 4))
        self.assertEqual(bank.su2_probabilities.shape, (3, 4))
        np.testing.assert_array_equal(bank.energy_edges_gev.shape, (5,))
        np.testing.assert_allclose(bank.photon_probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all(bank.photon_probabilities > 0.0))
        self.assertEqual({call[3] for call in adapter.calls}, {54_321, 54_421})
        self.assertEqual({call[4] for call in adapter.calls}, {"spectrum"})
        expected_proposal_ctau_m = float(np.exp(np.log(3.0)))
        self.assertNotEqual(expected_proposal_ctau_m, 3.0)
        self.assertEqual(
            {call[5] for call in adapter.calls},
            {expected_proposal_ctau_m},
        )

    def test_workflow_writes_portable_manifest_and_profile_outputs(self):
        adapter = _FakeAdapter(self.config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan_path = root / "scan.csv"
            _scan(self.config).to_csv(scan_path, index=False)
            output_dir = root / "outputs" / self.config.name
            summary = run_template_bank_workflow(
                config=self.config,
                adapter=adapter,
                scan_path=scan_path,
                output_dir=output_dir,
            )
            self.assertEqual(summary["number_of_photon_lifetimes"].tolist(), [3])
            self.assertTrue(
                (output_dir / "template_banks" / "template_bank_ma_0p3.npz").is_file()
            )
            payload = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(payload["profile"], "quick")
            self.assertEqual(payload["cache_stats"]["hits"], 2)
            serialized = json.dumps(payload)
            self.assertNotIn("/Users/", serialized)
            self.assertNotIn(directory, serialized)
            with self.assertRaises(FileExistsError):
                run_template_bank_workflow(
                    config=self.config,
                    adapter=adapter,
                    scan_path=scan_path,
                    output_dir=output_dir,
                )

    def test_requested_masses_retain_stable_order(self):
        configured = (0.3, 0.4, 0.5)
        self.assertEqual(resolve_requested_masses([0.5, 0.3], configured), (0.3, 0.5))
        with self.assertRaises(ValueError):
            resolve_requested_masses([0.6], configured)


if __name__ == "__main__":
    unittest.main()
