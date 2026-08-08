from pathlib import Path

import pandas as pd
import pytest

from analysis2.alp_su2l_planning import (
    AnalysisConfig,
    build_analysis_plan,
)


def config(tmp_path, *, run_mode="automatic"):
    return AnalysisConfig(
        masses=(0.3, 1.0),
        selections=(
            "diphoton_ecal",
            "diphoton_ecal_e1gev",
        ),
        observables=(
            "energy",
            "energy_mean_z_r_perp",
        ),
        profile="production",
        workers=2,
        run_mode=run_mode,
        output_dir=tmp_path / "analysis",
        domain_path=tmp_path / "domains.csv",
        bank_manifest=tmp_path / "manifest.csv",
        resume=True,
    )


def test_analysis_config_rejects_more_than_two_workers(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        AnalysisConfig(
            masses=(0.3,),
            selections=("diphoton_ecal",),
            observables=("energy",),
            profile="production",
            workers=3,
            run_mode="automatic",
            output_dir=tmp_path,
            domain_path=tmp_path / "domains.csv",
            bank_manifest=tmp_path / "manifest.csv",
        )


def test_validated_bank_is_reused(tmp_path):
    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"test")

    manifest = pd.DataFrame(
        [
            {
                "mass_GeV": 0.3,
                "selection_name": "diphoton_ecal",
                "status": "validated",
                "bank_path": str(bank),
            }
        ]
    )

    plan = build_analysis_plan(
        config=config(tmp_path),
        manifest=manifest,
        repo=tmp_path,
    )

    row = plan.loc[
        (plan["mass_GeV"] == 0.3)
        & (plan["selection_name"] == "diphoton_ecal")
    ].iloc[0]

    assert row["bank_state"] == "validated"
    assert bool(row["bank_exists"])
    assert row["bank_action"] == "reuse"


def test_missing_bank_is_planned_for_build_in_automatic_mode(
    tmp_path,
):
    manifest = pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
        ]
    )

    plan = build_analysis_plan(
        config=config(tmp_path, run_mode="automatic"),
        manifest=manifest,
        repo=tmp_path,
    )

    assert set(plan["bank_action"]) == {"build"}
    assert set(plan["bank_state"]) == {"missing"}


def test_missing_bank_requires_user_bank_in_custom_mode(tmp_path):
    manifest = pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
        ]
    )

    plan = build_analysis_plan(
        config=config(tmp_path, run_mode="custom"),
        manifest=manifest,
        repo=tmp_path,
    )

    assert set(plan["bank_action"]) == {"requires_bank"}


def test_every_mass_selection_has_isolated_bank_workspace(
    tmp_path,
):
    manifest = pd.DataFrame(
        columns=[
            "mass_GeV",
            "selection_name",
            "status",
            "bank_path",
        ]
    )

    plan = build_analysis_plan(
        config=config(tmp_path),
        manifest=manifest,
        repo=tmp_path,
    )

    assert len(plan) == 4
    assert plan["bank_workspace"].nunique() == 4
    assert plan["result_dir"].nunique() == 4


def test_reuse_only_skips_missing_banks(tmp_path):
    manifest = pd.DataFrame(
        columns=["mass_GeV", "selection_name", "status", "bank_path"]
    )
    plan = build_analysis_plan(
        config=config(tmp_path, run_mode="reuse_only"),
        manifest=manifest,
        repo=tmp_path,
    )
    assert set(plan["bank_action"]) == {"skip_unavailable"}
