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
