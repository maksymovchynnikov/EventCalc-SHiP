import pandas as pd

from alp_discrimination.workflows.conditional_feature_production import (
    bracket,
    full_grid,
    refinement_grid,
    selection_grid,
)


def test_bracket():
    curve = pd.DataFrame({
        "number_of_events": [2, 3, 4, 5],
        "worst_case_accuracy": [0.7, 0.87, 0.92, 0.95],
    })
    assert bracket(curve) == (4, 3, 4)


def test_bracket_can_resolve_single_event_threshold():
    curve = pd.DataFrame({
        "number_of_events": [1, 2, 3],
        "worst_case_accuracy": [0.91, 0.94, 0.96],
    })
    assert bracket(curve) == (1, 0, 1)


def test_no_bracket():
    curve = pd.DataFrame({
        "number_of_events": [20, 30],
        "worst_case_accuracy": [0.7, 0.8],
    })
    assert bracket(curve) == (None, 30, None)


def test_refinement_unit_grid():
    grid = refinement_grid(120, 130)
    assert all(value in grid for value in range(120, 131))
    assert max(grid) > 130


def test_full_grid():
    grid = full_grid(142)
    assert all(value in grid for value in range(134, 158))
    assert max(grid) >= 213


def test_validated_low_grid_is_unchanged_away_from_edge():
    grid = full_grid(4)
    assert min(grid) == 2
    assert 4 in grid


def test_low_edge_grid_includes_single_event():
    grid = full_grid(3)
    assert min(grid) == 1
    assert 3 in grid


def test_selection_grid():
    assert selection_grid(4, full_grid(4)) == (3, 4, 5)
    assert selection_grid(2, full_grid(2)) == (1, 2, 3)
    assert selection_grid(1, full_grid(1)) == (1, 2)
