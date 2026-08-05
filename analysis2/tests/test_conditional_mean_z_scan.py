"""Tests for the generic conditional-mean-z scan controller."""

from pathlib import Path

from analysis2.workflows.conditional_mean_z_scan import (
    ScanPoint,
    command_for_stage,
    float_token,
    parse_points,
    point_root,
    stage_paths,
)


def test_parse_points():
    points = parse_points(
        [
            ["1.0", "diphoton_ecal", "bank1.npz"],
            ["2.5", "diphoton_ecal", "bank2.npz"],
        ]
    )
    assert points == [
        ScanPoint(1.0, "diphoton_ecal", Path("bank1.npz")),
        ScanPoint(2.5, "diphoton_ecal", Path("bank2.npz")),
    ]


def test_float_token():
    assert float_token(0.5) == "0p5"
    assert float_token(2.5) == "2p5"


def test_stage_commands_use_package_modules(tmp_path):
    root = point_root(tmp_path, 1.0, "diphoton_ecal")
    paths = stage_paths(root, 1.0)
    command = command_for_stage(
        stage="full_domain",
        bank_path=tmp_path / "bank.npz",
        paths=paths,
        workers=2,
        domain_path=tmp_path / "domains.csv",
    )
    assert "-m" in command
    assert (
        "analysis2.workflows.conditional_mean_z_full_domain"
        in command
    )
    assert "--rangefinder-summary" in command
