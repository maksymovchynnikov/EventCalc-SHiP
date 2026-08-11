from pathlib import Path

import pandas as pd
import pytest

from alp_discrimination.workflows import analysis as workflow


def write_manifest(path: Path, bank_path: Path) -> None:
    pd.DataFrame(
        [{
            "mass_GeV": 0.3,
            "selection_name": "diphoton_ecal",
            "status": "validated",
            "bank_path": str(bank_path),
            "note": "test",
        }]
    ).to_csv(path, index=False)


def test_resolve_bank_record():
    table = pd.DataFrame([{
        "mass_GeV": 0.3,
        "selection_name": "diphoton_ecal",
        "status": "validated",
        "bank_path": "bank.npz",
    }])
    assert workflow.resolve_bank_record(
        table, 0.3, "diphoton_ecal"
    )["status"] == "validated"


def test_load_frozen_manifest_schema(tmp_path):
    path = tmp_path / "frozen.csv"
    pd.DataFrame([{
        "mass_GeV": 0.3,
        "selection_name": "diphoton_ecal",
        "status": "validated",
        "frozen_bank_path": "frozen/bank.npz",
    }]).to_csv(path, index=False)
    table = workflow.load_bank_manifest(path)
    assert table.iloc[0]["bank_path"] == "frozen/bank.npz"


def test_dry_run_reuse_only_marks_missing(tmp_path):
    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, bank)
    domain = tmp_path / "domains.csv"
    domain.write_text("mass_GeV\n0.3\n")
    out = tmp_path / "out"

    workflow.main([
        "--masses", "0.3", "0.5",
        "--selections", "diphoton_ecal",
        "--bank-manifest", str(manifest),
        "--domain-path", str(domain),
        "--output-dir", str(out),
        "--run-mode", "reuse_only",
        "--dry-run",
    ])
    plan = pd.read_csv(out / "latest_run_plan.csv")
    assert list(plan["bank_action"]) == ["reuse", "skip_unavailable"]


def test_disk_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "free_gib", lambda path: 4.0)
    with pytest.raises(RuntimeError, match="Disk guard"):
        workflow.disk_guard(
            tmp_path,
            minimum_gib=8.0,
            warning_gib=10.0,
            operation="test",
        )


def test_adaptive_bank_status_mapping():
    assert workflow.registry_status_from_adaptive_status(
        "fine_binning_converged"
    ) == "production"
    assert workflow.registry_status_from_adaptive_status(
        "lifetime_grid_size_limit"
    ) == "incomplete"


def test_persist_generated_bank_record(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(columns=[
        "mass_GeV", "selection_name", "status", "bank_path", "note"
    ]).to_csv(manifest, index=False)
    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"x")
    status = workflow.persist_generated_bank_record(
        manifest_path=manifest,
        repo=tmp_path,
        mass_gev=0.5,
        selection_name="diphoton_ecal_e1gev",
        bank_path=bank,
        adaptive_status="fine_binning_converged",
    )
    assert status == "production"
    assert pd.read_csv(manifest).iloc[0]["status"] == "production"



def test_progress_meter_reports_eta(monkeypatch):
    from alp_discrimination import progress

    values = iter([100.0, 110.0])
    monkeypatch.setattr(progress.time, "perf_counter", lambda: next(values))
    meter = progress.ProgressMeter(total=10, label="test")
    message = meter.message(2)
    assert "ETA=" in message
    assert "finish~" in message


def test_bank_quality_noise_floor_is_stable_and_usable():
    from alp_discrimination.workflows.analysis import bank_quality_metadata

    quality = bank_quality_metadata("production_noise_floor_limited")
    assert quality["physics_usable"] is True
    assert quality["global_minimum_status"] == "stable"
    assert quality["refinement_status"] == "numerical_template_statistical_noise_floor"


def test_public_stop_stage_aliases():
    assert workflow.parse_stop_after("threshold_scan") == "rangefinder"
    assert workflow.parse_stop_after("lifetime-scan") == "full_domain"
    assert workflow.parse_stop_after("validation") == "selected"
    assert workflow.parse_stop_after("final") == "final"


def test_legacy_stop_stage_aliases_remain_compatible():
    assert workflow.parse_stop_after("rangefinder") == "rangefinder"
    assert workflow.parse_stop_after("full_domain") == "full_domain"
    assert workflow.parse_stop_after("selected") == "selected"


def test_single_event_validation_grids_are_supported():
    available = [1, 2, 3, 4]
    assert workflow.selection_counts(1, available) == [1, 2]
    assert workflow.empirical_counts(1, available) == [1, 2, 3]


def test_merge_observable_records_preserves_subset_results():
    existing = [{"observable": "energy_mean_z_r_perp", "N90": 4}]
    subset = [{"observable": "energy_mean_r_perp", "N90": 5}]

    merged = workflow.merge_observable_records(existing, subset)

    assert [(row["observable"], row["N90"]) for row in merged] == [
        ("energy_mean_r_perp", 5),
        ("energy_mean_z_r_perp", 4),
    ]


def test_merge_observable_records_updates_matching_observable():
    existing = [{"observable": "energy_mean_z_r_perp", "N90": 6}]
    updated = [{"observable": "energy_mean_z_r_perp", "N90": 4}]

    merged = workflow.merge_observable_records(existing, updated)

    assert merged == [
        {"observable": "energy_mean_z_r_perp", "N90": 4}
    ]



def test_source_only_reuse_dry_run_without_bank_registry(
    tmp_path,
    monkeypatch,
):
    domain = tmp_path / "domains.csv"
    domain.write_text("mass_GeV\n0.3\n")
    output = tmp_path / "out"
    local_manifest = tmp_path / "missing_local.csv"

    monkeypatch.setattr(
        workflow,
        "DEFAULT_FROZEN_BANK_MANIFEST",
        tmp_path / "missing_frozen.csv",
    )
    monkeypatch.setattr(
        workflow,
        "DEFAULT_EXISTING_BANK_MANIFEST",
        local_manifest,
    )

    workflow.main([
        "--masses", "0.3",
        "--selections", "diphoton_ecal",
        "--domain-path", str(domain),
        "--output-dir", str(output),
        "--run-mode", "reuse_only",
        "--dry-run",
    ])

    plan = pd.read_csv(output / "latest_run_plan.csv")
    assert plan.iloc[0]["bank_state"] == "missing"
    assert plan.iloc[0]["bank_action"] == "skip_unavailable"
    assert not local_manifest.exists()


def test_source_only_automatic_dry_run_without_bank_registry(
    tmp_path,
    monkeypatch,
):
    domain = tmp_path / "domains.csv"
    domain.write_text("mass_GeV\n0.3\n")
    output = tmp_path / "out"
    local_manifest = tmp_path / "missing_local.csv"

    monkeypatch.setattr(
        workflow,
        "DEFAULT_FROZEN_BANK_MANIFEST",
        tmp_path / "missing_frozen.csv",
    )
    monkeypatch.setattr(
        workflow,
        "DEFAULT_EXISTING_BANK_MANIFEST",
        local_manifest,
    )

    workflow.main([
        "--masses", "0.3",
        "--selections", "diphoton_ecal",
        "--domain-path", str(domain),
        "--output-dir", str(output),
        "--run-mode", "automatic",
        "--dry-run",
    ])

    plan = pd.read_csv(output / "latest_run_plan.csv")
    assert plan.iloc[0]["bank_state"] == "missing"
    assert plan.iloc[0]["bank_action"] == "build"
    assert not local_manifest.exists()


def test_initialise_bank_manifest_has_expected_schema(tmp_path):
    path = workflow.initialise_bank_manifest(tmp_path / "registry.csv")
    assert path.is_file()
    assert list(pd.read_csv(path).columns) == list(
        workflow.BANK_MANIFEST_COLUMNS
    )
