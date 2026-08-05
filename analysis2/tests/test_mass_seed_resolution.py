"""Regression tests for Week-8 appended-mass seed resolution."""

from pathlib import Path

import pandas as pd

from analysis2.config import get_config
from analysis2.mass_seed_resolution import (
    available_masses_from_domain,
    mass_seed_index,
    model_seed_for_bank,
    stable_mass_seed_indices,
)


class _Bank:
    mass_gev = 2.5
    template_seed_offset = 0
    template_base_seed = 54_321


def test_frozen_mass_indices_are_unchanged():
    policy = get_config("production").seed_policy
    mapping = stable_mass_seed_indices(
        policy,
        [0.3, 0.5, 1.0, 1.2, 2.5],
    )
    assert mapping[0.3] == 0
    assert mapping[1.0] == 6


def test_extra_masses_are_appended_in_sorted_order():
    policy = get_config("production").seed_policy
    index_12 = mass_seed_index(
        seed_policy=policy,
        mass_gev=1.2,
        available_masses=[2.5, 0.3, 1.2, 1.0],
    )
    index_25 = mass_seed_index(
        seed_policy=policy,
        mass_gev=2.5,
        available_masses=[2.5, 0.3, 1.2, 1.0],
    )
    assert index_12 == len(policy.mass_order_gev)
    assert index_25 == len(policy.mass_order_gev) + 1


def test_model_seed_for_bank_uses_complete_domain_mass_set(tmp_path: Path):
    domain = tmp_path / "allowed_ctau_domains.csv"
    pd.DataFrame(
        {
            "mass_GeV": [0.3, 1.0, 1.2, 2.5, 2.5],
            "model": ["a", "a", "a", "a", "b"],
        }
    ).to_csv(domain, index=False)

    config = get_config("production")
    seed = model_seed_for_bank(
        config=config,
        bank=_Bank(),
        model_id="alp_photon_combined",
        domain_path=domain,
    )
    expected_index = len(config.seed_policy.mass_order_gev) + 1
    expected = config.seed_policy.model_seed_from_indices(
        expected_index,
        config.seed_policy.model_index("alp_photon_combined"),
        seed_offset=0,
    )
    assert seed == expected
    assert available_masses_from_domain(domain) == (0.3, 1.0, 1.2, 2.5)
