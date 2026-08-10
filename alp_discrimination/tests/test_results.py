import json

from alp_discrimination.workflows.results import write_project_outputs


def test_project_outputs_from_point_summary(tmp_path):
    point = tmp_path / "per_point" / "ma_0p3" / "geom"
    point.mkdir(parents=True)
    summary = {
        "mass_GeV": 0.3,
        "selection_name": "diphoton_ecal",
        "status": "final_for_project",
        "results": [{
            "mass_GeV": 0.3,
            "selection_name": "diphoton_ecal",
            "observable": "energy_mean_z_r_perp",
            "N90": 4,
            "project_final": True,
        }],
    }
    (point / "point_summary.json").write_text(json.dumps(summary))
    result = write_project_outputs(tmp_path)
    assert result["number_of_completed_points"] == 1
    assert (tmp_path / "tables" / "n90_summary.csv").is_file()
    assert (tmp_path / "plots" / "n90_vs_mass_headline.pdf").is_file()


def test_headline_legacy_curves_are_bundled_and_loadable(tmp_path):
    import pandas as pd

    from alp_discrimination.workflows import results as workflow_results

    tables = tmp_path / "report" / "tables"
    tables.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "number_of_events": [2, 4, 10],
            "high_statistics_accuracy": [0.75, 0.91, 0.98],
        }
    )

    frame.to_csv(
        tables / "classification_accuracy_ma_0p3_ecal_energy_r_perp.csv",
        index=False,
    )
    frame.to_csv(
        tables / "classification_accuracy_ma_0p3_ecal_energy_z_r_perp.csv",
        index=False,
    )

    curves, missing = workflow_results._headline_observable_comparison_curves(
        tmp_path
    )

    assert missing == []
    assert set(curves) == {
        "energy",
        "energy_mean_z",
        "energy_mean_r_perp",
        "energy_mean_z_r_perp",
    }
    assert not curves["energy"].empty
    assert not curves["energy_mean_z"].empty
