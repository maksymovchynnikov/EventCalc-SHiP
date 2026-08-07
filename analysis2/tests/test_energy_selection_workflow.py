from pathlib import Path

import pytest

from analysis2.config import VALIDATION
from analysis2.workflows.lifetime_blind_discrimination import (
    apply_cli_overrides,
    parse_arguments,
    resolve_template_output_dir,
)


def test_energy_selection_cli_override_requires_distinct_output_directory():
    args = parse_arguments(
        [
            "--profile",
            "validation",
            "--selection-name",
            "diphoton_ecal_e1gev",
        ]
    )
    config = apply_cli_overrides(VALIDATION, args)

    assert config.name == "validation"
    assert config.selection_name == "diphoton_ecal_e1gev"
    with pytest.raises(ValueError, match="explicit --output-dir"):
        resolve_template_output_dir(config, args)


def test_energy_selection_cli_override_accepts_explicit_output_directory():
    args = parse_arguments(
        [
            "--profile",
            "validation",
            "--selection-name",
            "diphoton_ecal_e1gev",
            "--output-dir",
            "analysis2/outputs/validation/week8_ma0p3_e1gev",
        ]
    )
    config = apply_cli_overrides(VALIDATION, args)

    assert config.selection_name == "diphoton_ecal_e1gev"
    assert resolve_template_output_dir(config, args) == Path(
        "analysis2/outputs/validation/week8_ma0p3_e1gev"
    )


def test_geometry_default_remains_unchanged():
    args = parse_arguments(["--profile", "validation"])
    config = apply_cli_overrides(VALIDATION, args)

    assert config.selection_name == "diphoton_ecal"
