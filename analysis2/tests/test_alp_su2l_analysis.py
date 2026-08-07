from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from analysis2.workflows import alp_su2l_analysis as workflow


def write_manifest(path: Path, bank_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "mass_GeV": 0.3,
                "selection_name": "diphoton_ecal",
                "status": "validated",
                "bank_path": str(bank_path),
                "note": "test",
            }
        ]
    ).to_csv(path, index=False)


def test_resolve_bank_record():
    table = pd.DataFrame(
        [
            {
                "mass_GeV": 0.3,
                "selection_name": "diphoton_ecal",
                "status": "validated",
                "bank_path": "bank.npz",
            }
        ]
    )

    row = workflow.resolve_bank_record(
        table,
        0.3,
        "diphoton_ecal",
    )

    assert row["status"] == "validated"
    assert row["bank_path"] == "bank.npz"


def test_dry_run_uses_registered_bank(tmp_path):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test")

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, bank_path)

    domain_path = tmp_path / "domains.csv"
    domain_path.write_text("mass_GeV\n0.3\n")

    output_dir = tmp_path / "output"

    workflow.main(
        [
            "--masses",
            "0.3",
            "--selections",
            "diphoton_ecal",
            "--bank-manifest",
            str(manifest_path),
            "--domain-path",
            str(domain_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    plan = pd.read_csv(output_dir / "latest_run_plan.csv")
    assert len(plan) == 1
    assert plan.iloc[0]["mass_GeV"] == 0.3
    assert plan.iloc[0]["bank_status"] == "validated"


def test_controller_calls_conditional_feature_runner(
    tmp_path,
    monkeypatch,
):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test")

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, bank_path)

    domain_path = tmp_path / "domains.csv"
    domain_path.write_text("mass_GeV\n0.3\n")

    output_dir = tmp_path / "output"

    fake_bank = SimpleNamespace(
        mass_gev=0.3,
        selection_name="diphoton_ecal",
    )
    monkeypatch.setattr(
        workflow,
        "load_template_bank",
        lambda path: fake_bank,
    )

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "mass_GeV": 0.3,
            "selection_name": "diphoton_ecal",
            "observables": [
                "energy",
                "energy_mean_z_r_perp",
            ],
            "truth_grid": "screening",
            "pseudoexperiments_per_truth_and_seed": 10,
            "provisional_thresholds": {
                "energy": 20,
                "energy_mean_z_r_perp": 4,
            },
            "distance_minima": {
                "energy": {
                    "minimum_H2": 0.01,
                    "photon_ctau_m": 1.0,
                    "su2_ctau_m": 2.0,
                },
                "energy_mean_z_r_perp": {
                    "minimum_H2": 0.3,
                    "photon_ctau_m": 1.5,
                    "su2_ctau_m": 3.0,
                },
            },
        }

    monkeypatch.setattr(
        workflow,
        "run_conditional_feature_point",
        fake_run,
    )

    workflow.main(
        [
            "--masses",
            "0.3",
            "--selections",
            "diphoton_ecal",
            "--observables",
            "energy",
            "energy_mean_z_r_perp",
            "--pseudoexperiments",
            "10",
            "--bank-manifest",
            str(manifest_path),
            "--domain-path",
            str(domain_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert captured["bank_path"] == bank_path
    assert captured["pseudoexperiments"] == 10
    assert captured["observables"] == (
        "energy",
        "energy_mean_z_r_perp",
    )

    summary = pd.read_csv(
        output_dir / "analysis_summary.csv"
    )
    assert list(summary["observable"]) == [
        "energy",
        "energy_mean_z_r_perp",
    ]
    assert list(summary["provisional_N90"]) == [20, 4]
