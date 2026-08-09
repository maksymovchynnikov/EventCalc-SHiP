"""Aggregate completed ALP-SU2L point outputs into final project products."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from alp_discrimination.templates.conditional_features import FEATURE_LABELS
from alp_discrimination.plotting.report import (
    plot_classification_accuracy,
    plot_distance_diagnostics,
    plot_n90_vs_mass as plot_report_n90_vs_mass,
    plot_observable_comparison as plot_report_observable_comparison,
)
from alp_discrimination.workflows import float_token


SELECTION_LABELS = {
    "diphoton_ecal": "ECAL only",
    "diphoton_ecal_e1gev": r"ECAL + $E_\gamma\geq1$ GeV",
}


def collect_tables(output_root: Path) -> dict[str, pd.DataFrame]:
    n90_rows = []
    limiting_rows = []
    distance_rows = []
    empirical_rows = []
    point_rows = []
    bank_quality_rows = []

    for summary_path in sorted(
        Path(output_root).glob("per_point/ma_*/*/point_summary.json")
    ):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if not summary.get("results"):
            continue

        point_rows.append(
            {
                "mass_GeV": float(summary["mass_GeV"]),
                "selection_name": str(summary["selection_name"]),
                "status": summary.get("status"),
                "point_summary_path": str(summary_path),
            }
        )
        quality = summary.get("bank_quality")
        if isinstance(quality, dict):
            bank_quality_rows.append(
                {
                    "mass_GeV": float(summary["mass_GeV"]),
                    "selection_name": str(summary["selection_name"]),
                    **quality,
                }
            )
        n90_rows.extend(summary["results"])

        table_root = summary_path.parent / "tables"
        for filename, destination in (
            ("limiting_truths.csv", limiting_rows),
            ("distance_summary.csv", distance_rows),
            ("empirical_summary.csv", empirical_rows),
        ):
            path = table_root / filename
            if not path.is_file() or path.stat().st_size == 0:
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            destination.extend(frame.to_dict(orient="records"))

    return {
        "n90": pd.DataFrame(n90_rows),
        "limiting": pd.DataFrame(limiting_rows),
        "distance": pd.DataFrame(distance_rows),
        "empirical": pd.DataFrame(empirical_rows),
        "points": pd.DataFrame(point_rows),
        "bank_quality": pd.DataFrame(bank_quality_rows),
    }


def save_figure(fig, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_n90_vs_mass(table: pd.DataFrame, plots: Path) -> None:
    if table.empty:
        return
    for observable, subset in table.groupby("observable"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for selection, group in subset.groupby("selection_name"):
            group = group.sort_values("mass_GeV")
            ax.plot(
                group["mass_GeV"],
                group["N90"],
                marker="o",
                label=SELECTION_LABELS.get(selection, selection),
            )
        ax.set_xlabel(r"ALP mass, $m_a$ [GeV]")
        ax.set_ylabel(r"Minimum observed events, $N_{90}$")
        ax.set_title(FEATURE_LABELS.get(observable, observable))
        ax.grid(alpha=0.25)
        ax.legend()
        base = plots / f"n90_vs_mass_{observable}"
        save_figure(fig, base)

        if observable == "energy_mean_z_r_perp":
            for suffix in (".pdf", ".png"):
                source = base.with_suffix(suffix)
                target = plots / f"n90_vs_mass_headline{suffix}"
                target.write_bytes(source.read_bytes())


def plot_ablation(table: pd.DataFrame, plots: Path) -> None:
    if table.empty or table["observable"].nunique() < 2:
        return
    for selection, subset in table.groupby("selection_name"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for observable, group in subset.groupby("observable"):
            group = group.sort_values("mass_GeV")
            ax.plot(
                group["mass_GeV"],
                group["N90"],
                marker="o",
                label=FEATURE_LABELS.get(observable, observable),
            )
        ax.set_xlabel(r"ALP mass, $m_a$ [GeV]")
        ax.set_ylabel(r"Minimum observed events, $N_{90}$")
        ax.grid(alpha=0.25)
        ax.legend()
        token = "geom" if selection == "diphoton_ecal" else "e1gev"
        save_figure(fig, plots / f"observable_ablation_{token}")



def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_report_outputs(output_root: Path, thresholds: pd.DataFrame) -> dict:
    """Create compact report-ready products from completed analysis outputs."""

    output_root = Path(output_root)
    report_root = output_root / "report"
    plot_dir = report_root / "plots"
    table_dir = report_root / "tables"
    data_dir = report_root / "data"
    for path in (plot_dir, table_dir, data_dir):
        path.mkdir(parents=True, exist_ok=True)

    thresholds.to_csv(table_dir / "discrimination_thresholds.csv", index=False)
    validation_columns = [
        name
        for name in (
            "mass_GeV",
            "selection_name",
            "observable",
            "N90",
            "validation_level",
            "omitted_truth_audit_passed",
            "empirical_threshold_confirmed",
            "project_final",
        )
        if name in thresholds.columns
    ]
    thresholds[validation_columns].to_csv(
        table_dir / "validation_summary.csv", index=False
    )

    distance_minima = []
    completed_points = 0
    for summary_path in sorted(
        output_root.glob("per_point/ma_*/*/point_summary.json")
    ):
        summary = json.loads(summary_path.read_text())
        if not summary.get("results"):
            continue
        completed_points += 1
        point_root = summary_path.parent
        mass = float(summary["mass_GeV"])
        selection = str(summary["selection_name"])
        token = float_token(mass)

        for result in summary["results"]:
            selected_value = result.get("selected_summary_path")
            if not selected_value:
                continue
            selected_summary_path = Path(str(selected_value))
            selected_dir = selected_summary_path.parent
            if not selected_summary_path.is_file():
                continue
            selected_summary = json.loads(selected_summary_path.read_text())
            high_curve = _read_csv_if_present(
                selected_dir / f"selected_5k_curve_ma_{token}.csv"
            )
            validation = _read_csv_if_present(
                selected_dir / f"full_domain_2k_vs_selected_5k_ma_{token}.csv"
            )
            empirical = _read_csv_if_present(
                point_root
                / "empirical"
                / str(result["observable"])
                / f"conditional_feature_empirical_comparison_ma_{token}.csv"
            )
            plot_classification_accuracy(
                mass=mass,
                selection=selection,
                observable=str(result["observable"]),
                n90=int(result["N90"]),
                high_statistics_curve=high_curve,
                validation_comparison=validation,
                empirical_comparison=empirical,
                pseudoexperiments=int(
                    selected_summary.get(
                        "pseudoexperiments_per_selected_truth_and_seed", 0
                    )
                ),
                output_dir=plot_dir,
            )
            if not high_curve.empty:
                export = high_curve[
                    ["number_of_events", "worst_case_accuracy"]
                ].rename(
                    columns={
                        "worst_case_accuracy": "high_statistics_accuracy"
                    }
                )
                if not validation.empty:
                    export = export.merge(
                        validation[
                            ["number_of_events", "full_domain_2k_accuracy"]
                        ],
                        on="number_of_events",
                        how="left",
                    )
                if not empirical.empty:
                    export = export.merge(
                        empirical[
                            [
                                "number_of_events",
                                "gaussian_truth",
                                "empirical_truth",
                            ]
                        ],
                        on="number_of_events",
                        how="left",
                    )
                stem = (
                    f"classification_accuracy_ma_{token}_"
                    + (
                        "ecal"
                        if selection == "diphoton_ecal"
                        else "ecal_e1gev"
                    )
                    + "_"
                    + str(result["observable"])
                    .replace("energy_mean_z_r_perp", "energy_z_r_perp")
                    .replace("energy_mean_r_perp", "energy_r_perp")
                    .replace("energy_mean_z", "energy_z")
                )
                export.to_csv(table_dir / f"{stem}.csv", index=False)

        moment_path = (
            point_root
            / "moments"
            / f"conditional_feature_moments_ma_{token}.npz"
        )
        bank_value = summary.get("bank_path")
        bank_path = Path(str(bank_value)) if bank_value else Path()
        if bank_value and bank_path.is_file() and moment_path.is_file():
            try:
                distance_minima.append(
                    plot_distance_diagnostics(
                        bank_path=bank_path,
                        moments_path=moment_path,
                        selection=selection,
                        output_dir=plot_dir,
                        data_dir=data_dir,
                    )
                )
            except ValueError:
                pass

        point_results = pd.DataFrame(summary["results"])
        plot_report_observable_comparison(
            results=point_results,
            mass=mass,
            selection=selection,
            output_dir=plot_dir,
        )

    plot_report_n90_vs_mass(thresholds, plot_dir)
    if distance_minima:
        pd.concat(distance_minima, ignore_index=True).to_csv(
            table_dir / "distance_minima.csv", index=False
        )

    report_summary = {
        "status": "report_products",
        "number_of_completed_points": completed_points,
        "plots_dir": str(plot_dir),
        "tables_dir": str(table_dir),
        "data_dir": str(data_dir),
    }
    (report_root / "summary.json").write_text(
        json.dumps(report_summary, indent=2) + "\n"
    )
    return report_summary


def write_project_outputs(
    output_root: Path,
    *,
    requested_plan: pd.DataFrame | None = None,
) -> dict:
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    tables = collect_tables(output_root)
    n90 = tables["n90"]
    if not n90.empty:
        n90 = n90.sort_values(
            ["observable", "selection_name", "mass_GeV"],
            ignore_index=True,
        )

    n90.to_csv(tables_dir / "n90_summary.csv", index=False)
    (tables_dir / "n90_summary.json").write_text(
        json.dumps(n90.to_dict(orient="records"), indent=2) + "\n"
    )
    tables["limiting"].to_csv(tables_dir / "limiting_truths.csv", index=False)
    tables["distance"].to_csv(tables_dir / "distance_summary.csv", index=False)
    tables["empirical"].to_csv(tables_dir / "empirical_summary.csv", index=False)
    tables["points"].to_csv(tables_dir / "completed_points.csv", index=False)
    tables["bank_quality"].to_csv(tables_dir / "bank_quality_summary.csv", index=False)
    n90.to_csv(tables_dir / "validation_summary.csv", index=False)

    if requested_plan is not None:
        requested_plan.to_csv(
            tables_dir / "requested_point_status.csv", index=False
        )

    plot_n90_vs_mass(n90, plots_dir)
    plot_ablation(n90, plots_dir)
    report = write_report_outputs(output_root, n90)

    project_final = 0
    if not n90.empty and "project_final" in n90.columns:
        project_final = int(
            n90["project_final"].fillna(False).astype(bool).sum()
        )

    summary = {
        "status": "project_results_index",
        "number_of_requested_points": (
            0 if requested_plan is None else int(len(requested_plan))
        ),
        "number_of_unavailable_requested_points": (
            0
            if requested_plan is None
            else int(
                (
                    requested_plan["bank_action"].astype(str)
                    == "skip_unavailable"
                ).sum()
            )
        ),
        "number_of_completed_points": int(len(tables["points"])),
        "number_of_completed_observable_results": int(len(n90)),
        "number_of_project_final_results": project_final,
        "tables": {
            "n90": str(tables_dir / "n90_summary.csv"),
            "validation": str(tables_dir / "validation_summary.csv"),
            "limiting_truths": str(tables_dir / "limiting_truths.csv"),
            "distance": str(tables_dir / "distance_summary.csv"),
            "empirical": str(tables_dir / "empirical_summary.csv"),
            "bank_quality": str(tables_dir / "bank_quality_summary.csv"),
        },
        "plots_dir": str(plots_dir),
        "report_dir": str(output_root / "report"),
        "report": report,
    }
    (output_root / "project_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
