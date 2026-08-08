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


def test_generated_bank_is_resolved_from_adaptive_state(
    tmp_path,
    monkeypatch,
):
    bank_dir = tmp_path / "banks" / "round_07"
    template_dir = bank_dir / "template_banks"
    template_dir.mkdir(parents=True)

    bank_path = template_dir / "template_bank_ma_0p5.npz"
    bank_path.write_bytes(b"test")

    state_path = tmp_path / "state.json"
    state_path.write_text(
        __import__("json").dumps(
            {
                "status": "bank_complete",
                "bank_dir": str(bank_dir),
                "bank_status": "converged",
            }
        )
    )

    fake_bank = SimpleNamespace(
        mass_gev=0.5,
        selection_name="diphoton_ecal_e1gev",
    )
    monkeypatch.setattr(
        workflow,
        "load_template_bank",
        lambda path: fake_bank,
    )

    resolved, status = workflow.generated_bank_from_state(
        repo=tmp_path,
        state_path=state_path,
        mass_gev=0.5,
        selection_name="diphoton_ecal_e1gev",
    )

    assert resolved == bank_path
    assert status == "converged"


def test_build_or_resume_bank_calls_adaptive_api(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "analysis"
    domain_path = tmp_path / "domains.csv"
    manifest_path = tmp_path / "manifest.csv"

    domain_path.write_text("mass_GeV\n0.5\n")
    manifest_path.write_text(
        "mass_GeV,selection_name,status,bank_path\n"
    )

    config = workflow.AnalysisConfig(
        masses=(0.5,),
        selections=("diphoton_ecal_e1gev",),
        observables=("energy",),
        profile="production",
        workers=2,
        run_mode="automatic",
        output_dir=output_dir,
        domain_path=domain_path,
        bank_manifest=manifest_path,
        resume=True,
    )

    captured = {}

    def fake_adaptive_run(**kwargs):
        captured.update(kwargs)

        point = (
            kwargs["output_dir"]
            / "per_mass"
            / "ma_0p5"
            / "e1gev"
        )
        bank_dir = point / "banks" / "round_03"
        template_dir = bank_dir / "template_banks"
        template_dir.mkdir(parents=True)

        (
            template_dir / "template_bank_ma_0p5.npz"
        ).write_bytes(b"test")

        point.mkdir(parents=True, exist_ok=True)
        (
            point / "state.json"
        ).write_text(
            __import__("json").dumps(
                {
                    "status": "bank_complete",
                    "bank_dir": str(bank_dir),
                    "bank_status": "converged",
                }
            )
        )

    monkeypatch.setattr(
        workflow,
        "run_adaptive_bank_point",
        fake_adaptive_run,
    )

    fake_bank = SimpleNamespace(
        mass_gev=0.5,
        selection_name="diphoton_ecal_e1gev",
    )
    monkeypatch.setattr(
        workflow,
        "load_template_bank",
        lambda path: fake_bank,
    )

    bank_path, status = workflow.build_or_resume_bank(
        config=config,
        repo=tmp_path,
        domains=pd.DataFrame({"mass_GeV": [0.5]}),
        mass_gev=0.5,
        selection_name="diphoton_ecal_e1gev",
    )

    assert bank_path.is_file()
    assert status == "converged"
    assert captured["mass_gev"] == 0.5
    assert captured["selection_name"] == "diphoton_ecal_e1gev"
    assert captured["profile"] == "production"
    assert captured["workers"] == 2
    assert captured["stop_after"] == "bank"


def test_adaptive_bank_status_mapping():
    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_converged"
        )
        == "production"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "fine_binning_converged"
        )
        == "production"
    )

    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_size_limit"
        )
        == "incomplete"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_round_limit"
        )
        == "incomplete"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_distance_unstable"
        )
        == "incomplete"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "binning_refinement_limit"
        )
        == "incomplete"
    )


def test_persist_generated_bank_record(tmp_path):
    manifest = tmp_path / "manifest.csv"

    pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
            "note",
        ]
    ).to_csv(manifest, index=False)

    bank = (
        tmp_path
        / "banks"
        / "template_bank_ma_0p5.npz"
    )
    bank.parent.mkdir()
    bank.write_bytes(b"test")

    status = workflow.persist_generated_bank_record(
        manifest_path=manifest,
        repo=tmp_path,
        mass_gev=0.5,
        selection_name="diphoton_ecal_e1gev",
        bank_path=bank,
        adaptive_status="fine_binning_converged",
    )

    assert status == "production"

    table = pd.read_csv(manifest)
    assert len(table) == 1
    assert table.iloc[0]["mass_GeV"] == 0.5
    assert (
        table.iloc[0]["selection_name"]
        == "diphoton_ecal_e1gev"
    )
    assert table.iloc[0]["status"] == "production"
    assert (
        table.iloc[0]["bank_path"]
        == "banks/template_bank_ma_0p5.npz"
    )
    assert "fine_binning_converged" in table.iloc[0]["note"]


def test_nonconverged_generated_bank_is_persisted_incomplete(
    tmp_path,
):
    manifest = tmp_path / "manifest.csv"

    pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
            "note",
        ]
    ).to_csv(manifest, index=False)

    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"test")

    status = workflow.persist_generated_bank_record(
        manifest_path=manifest,
        repo=tmp_path,
        mass_gev=2.5,
        selection_name="diphoton_ecal_e1gev",
        bank_path=bank,
        adaptive_status="lifetime_grid_size_limit",
    )

    assert status == "incomplete"

    table = pd.read_csv(manifest)
    assert table.iloc[0]["status"] == "incomplete"


def test_stop_after_bank_does_not_run_conditional_features(
    tmp_path,
    monkeypatch,
):
    bank_path = tmp_path / "generated_bank.npz"
    bank_path.write_bytes(b"test")

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
            "note",
        ]
    ).to_csv(manifest_path, index=False)

    domain_path = tmp_path / "domains.csv"
    domain_path.write_text(
        "model,mass_GeV,interval_index,ctau_min_m,ctau_max_m\n"
        "ALP-photon-combined,0.5,0,1,2\n"
        "ALP-SU2L,0.5,0,1,2\n"
    )

    def fake_build(**kwargs):
        return bank_path, "fine_binning_converged"

    monkeypatch.setattr(
        workflow,
        "build_or_resume_bank",
        fake_build,
    )

    def forbidden_conditional_run(**kwargs):
        raise AssertionError(
            "Conditional-feature runner must not be called."
        )

    monkeypatch.setattr(
        workflow,
        "run_conditional_feature_point",
        forbidden_conditional_run,
    )

    output_dir = tmp_path / "output"

    workflow.main(
        [
            "--masses",
            "0.5",
            "--selections",
            "diphoton_ecal_e1gev",
            "--bank-manifest",
            str(manifest_path),
            "--domain-path",
            str(domain_path),
            "--output-dir",
            str(output_dir),
            "--run-mode",
            "automatic",
            "--profile",
            "production",
            "--stop-after",
            "bank",
        ]
    )

    persisted = pd.read_csv(manifest_path)

    assert len(persisted) == 1
    assert persisted.iloc[0]["status"] == "production"

    plan = pd.read_csv(
        output_dir / "latest_run_plan.csv"
    )
    assert plan.iloc[0]["bank_action"] == "reuse"
    assert plan.iloc[0]["bank_state"] == "production"
    assert (
        plan.iloc[0]["adaptive_bank_status"]
        == "fine_binning_converged"
    )


def test_adaptive_bank_status_mapping():
    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_converged"
        )
        == "production"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "fine_binning_converged"
        )
        == "production"
    )

    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_size_limit"
        )
        == "incomplete"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_round_limit"
        )
        == "incomplete"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "lifetime_grid_distance_unstable"
        )
        == "incomplete"
    )
    assert (
        workflow.registry_status_from_adaptive_status(
            "binning_refinement_limit"
        )
        == "incomplete"
    )


def test_persist_generated_bank_record(tmp_path):
    manifest = tmp_path / "manifest.csv"

    pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
            "note",
        ]
    ).to_csv(manifest, index=False)

    bank = (
        tmp_path
        / "banks"
        / "template_bank_ma_0p5.npz"
    )
    bank.parent.mkdir()
    bank.write_bytes(b"test")

    status = workflow.persist_generated_bank_record(
        manifest_path=manifest,
        repo=tmp_path,
        mass_gev=0.5,
        selection_name="diphoton_ecal_e1gev",
        bank_path=bank,
        adaptive_status="fine_binning_converged",
    )

    assert status == "production"

    table = pd.read_csv(manifest)
    assert len(table) == 1
    assert table.iloc[0]["mass_GeV"] == 0.5
    assert (
        table.iloc[0]["selection_name"]
        == "diphoton_ecal_e1gev"
    )
    assert table.iloc[0]["status"] == "production"
    assert (
        table.iloc[0]["bank_path"]
        == "banks/template_bank_ma_0p5.npz"
    )
    assert "fine_binning_converged" in table.iloc[0]["note"]


def test_nonconverged_generated_bank_is_persisted_incomplete(
    tmp_path,
):
    manifest = tmp_path / "manifest.csv"

    pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
            "note",
        ]
    ).to_csv(manifest, index=False)

    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"test")

    status = workflow.persist_generated_bank_record(
        manifest_path=manifest,
        repo=tmp_path,
        mass_gev=2.5,
        selection_name="diphoton_ecal_e1gev",
        bank_path=bank,
        adaptive_status="lifetime_grid_size_limit",
    )

    assert status == "incomplete"

    table = pd.read_csv(manifest)
    assert table.iloc[0]["status"] == "incomplete"


def test_stop_after_bank_does_not_run_conditional_features(
    tmp_path,
    monkeypatch,
):
    bank_path = tmp_path / "generated_bank.npz"
    bank_path.write_bytes(b"test")

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
            "note",
        ]
    ).to_csv(manifest_path, index=False)

    domain_path = tmp_path / "domains.csv"
    domain_path.write_text(
        "model,mass_GeV,interval_index,ctau_min_m,ctau_max_m\n"
        "ALP-photon-combined,0.5,0,1,2\n"
        "ALP-SU2L,0.5,0,1,2\n"
    )

    def fake_build(**kwargs):
        return bank_path, "fine_binning_converged"

    monkeypatch.setattr(
        workflow,
        "build_or_resume_bank",
        fake_build,
    )

    def forbidden_conditional_run(**kwargs):
        raise AssertionError(
            "Conditional-feature runner must not be called."
        )

    monkeypatch.setattr(
        workflow,
        "run_conditional_feature_point",
        forbidden_conditional_run,
    )

    output_dir = tmp_path / "output"

    workflow.main(
        [
            "--masses",
            "0.5",
            "--selections",
            "diphoton_ecal_e1gev",
            "--bank-manifest",
            str(manifest_path),
            "--domain-path",
            str(domain_path),
            "--output-dir",
            str(output_dir),
            "--run-mode",
            "automatic",
            "--profile",
            "production",
            "--stop-after",
            "bank",
        ]
    )

    persisted = pd.read_csv(manifest_path)

    assert len(persisted) == 1
    assert persisted.iloc[0]["status"] == "production"

    plan = pd.read_csv(
        output_dir / "latest_run_plan.csv"
    )
    assert plan.iloc[0]["bank_action"] == "reuse"
    assert plan.iloc[0]["bank_state"] == "production"
    assert (
        plan.iloc[0]["adaptive_bank_status"]
        == "fine_binning_converged"
    )
