"""Generic multi-point controller for the conditional-mean-z analysis.

The controller runs the validated package-native stages sequentially for any
explicit list of mass/selection/template-bank points:

  range finder -> full-domain 2k -> selected 5k
  -> decision-relevant audit -> optional uniform 10k

It preserves one output tree per point, resumes completed stages, records
runtime and provenance, and writes a partial N90-versus-mass table/plot.

Important: this controller consumes existing template banks. It does not yet
construct or refine lifetime banks. Therefore a point is final only when its
input bank has separately passed the required lifetime-grid/binning checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGES = ("rangefinder", "full_domain", "selected", "audit", "uniform")
MODULES = {
    "rangefinder": "alp_discrimination.workflows.conditional_mean_z_rangefinder",
    "full_domain": "alp_discrimination.workflows.conditional_mean_z_full_domain",
    "selected": "alp_discrimination.workflows.conditional_mean_z_selected",
    "audit": "alp_discrimination.workflows.conditional_mean_z_decision_audit",
    "uniform": "alp_discrimination.workflows.conditional_mean_z_uniform",
}


@dataclass(frozen=True)
class ScanPoint:
    requested_mass: float
    requested_selection: str
    bank_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point",
        nargs=3,
        action="append",
        metavar=("MASS_GEV", "SELECTION", "BANK_PATH"),
        help=(
            "Repeat once per point, for example: "
            "--point 1.0 diphoton_ecal path/to/template_bank_ma_1.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument(
        "--domain-path",
        type=Path,
        default=Path(
            "analysis2/outputs/production/week8_domains/"
            "allowed_ctau_domains.csv"
        ),
    )
    parser.add_argument(
        "--allow-low-neff-bank",
        action="store_true",
        help=(
            "Allow a bank whose configured minimum bin N_eff is "
            "below 100. Such points are diagnostics only."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        default="uniform",
    )
    parser.add_argument(
        "--uniform-policy",
        choices=("always", "if_marginal", "never"),
        default="if_marginal",
    )
    parser.add_argument(
        "--uniform-margin",
        type=float,
        default=0.005,
        help=(
            "For --uniform-policy if_marginal, run 10k when the 5k "
            "accuracy at N90 is within this amount of the 90%% target."
        ),
    )
    parser.add_argument(
        "--bank-status",
        choices=("validated", "provisional"),
        default="provisional",
        help="Recorded provenance label; does not change the calculation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("analysis2/outputs"),
    )
    parser.add_argument("--masses", nargs="+", type=float)
    parser.add_argument("--selection", type=str)
    parser.add_argument("--inventory-csv", type=Path)
    return parser.parse_args()


def float_token(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def selection_token(selection: str) -> str:
    aliases = {
        "diphoton_ecal": "geom",
        "diphoton_ecal_e1gev": "e1gev",
    }
    return aliases.get(selection, selection.replace("/", "_"))


def parse_points(raw_points: list[list[str]] | None) -> list[ScanPoint]:
    if not raw_points:
        return []
    points: list[ScanPoint] = []
    seen: set[tuple[float, str]] = set()
    for mass_raw, selection, bank_raw in raw_points:
        mass = float(mass_raw)
        if mass <= 0:
            raise ValueError("Masses must be positive.")
        key = (mass, selection)
        if key in seen:
            raise ValueError(f"Duplicate scan point: {key}")
        seen.add(key)
        points.append(
            ScanPoint(
                requested_mass=mass,
                requested_selection=str(selection),
                bank_path=Path(bank_raw).expanduser(),
            )
        )
    return points


def resolve(repo: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_bank_metadata(bank_path: Path) -> dict:
    from alp_discrimination import conditional_mean_z

    bank = conditional_mean_z.load_template_bank(bank_path)
    return {
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "profile": str(bank.profile),
        "minimum_bin_n_eff": float(bank.minimum_bin_n_eff),
        "template_seed_offset": int(bank.template_seed_offset),
        "template_base_seed": int(bank.template_base_seed),
        "number_of_energy_bins": int(
            np.asarray(bank.energy_edges_gev).size - 1
        ),
        "number_of_photon_lifetimes": int(
            np.asarray(bank.photon_ctau_m).size
        ),
        "number_of_su2_lifetimes": int(
            np.asarray(bank.su2_ctau_m).size
        ),
    }


def validate_point(
    repo: Path,
    point: ScanPoint,
    *,
    allow_low_neff_bank: bool,
) -> tuple[Path, dict]:
    bank_path = resolve(repo, point.bank_path)
    if not bank_path.is_file():
        raise FileNotFoundError(f"Missing template bank: {bank_path}")
    metadata = load_bank_metadata(bank_path)
    if not np.isclose(
        metadata["mass_GeV"],
        point.requested_mass,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"Requested mass {point.requested_mass:g} does not match "
            f"bank mass {metadata['mass_GeV']:g}: {bank_path}"
        )
    if metadata["selection_name"] != point.requested_selection:
        raise ValueError(
            f"Requested selection {point.requested_selection!r} does not "
            f"match bank selection {metadata['selection_name']!r}: "
            f"{bank_path}"
        )
    if (
        metadata["minimum_bin_n_eff"] < 100.0
        and not allow_low_neff_bank
    ):
        raise ValueError(
            "Template bank is not eligible for conditional-feature "
            "production: configured minimum bin N_eff="
            f"{metadata['minimum_bin_n_eff']:g} < 100. "
            "Build a validation/production bank instead of lowering "
            "the quality threshold."
        )
    return bank_path, metadata


def point_root(output_root: Path, mass: float, selection: str) -> Path:
    return (
        output_root
        / "per_point"
        / f"ma_{float_token(mass)}"
        / selection_token(selection)
    )


def stage_paths(root: Path, mass: float) -> dict[str, tuple[Path, Path]]:
    token = float_token(mass)
    return {
        "rangefinder": (
            root / "rangefinder",
            root / "rangefinder" / f"rangefinder_summary_ma_{token}.json",
        ),
        "full_domain": (
            root / "full_domain_2k",
            root
            / "full_domain_2k"
            / f"full_domain_2k_summary_ma_{token}.json",
        ),
        "selected": (
            root / "selected_5k",
            root / "selected_5k" / f"selected_5k_summary_ma_{token}.json",
        ),
        "audit": (
            root / "decision_audit_5k",
            root
            / "decision_audit_5k"
            / f"decision_audit_summary_ma_{token}.json",
        ),
        "uniform": (
            root / "uniform_10k",
            root / "uniform_10k" / f"uniform_10k_summary_ma_{token}.json",
        ),
    }


def command_for_stage(
    *,
    stage: str,
    bank_path: Path,
    paths: dict[str, tuple[Path, Path]],
    workers: int,
    domain_path: Path,
) -> list[str]:
    output_dir = paths[stage][0]
    command = [
        sys.executable,
        "-m",
        MODULES[stage],
    ]
    if stage == "rangefinder":
        command += [
            "--bank-path",
            str(bank_path),
            "--domain-path",
            str(domain_path),
            "--output-dir",
            str(output_dir),
            "--workers",
            str(workers),
        ]
    elif stage == "full_domain":
        command += [
            "--rangefinder-summary",
            str(paths["rangefinder"][1]),
            "--output-dir",
            str(output_dir),
            "--workers",
            str(workers),
        ]
    elif stage == "selected":
        command += [
            "--full-domain-summary",
            str(paths["full_domain"][1]),
            "--output-dir",
            str(output_dir),
            "--workers",
            str(workers),
        ]
    elif stage == "audit":
        command += [
            "--full-domain-summary",
            str(paths["full_domain"][1]),
            "--selected-5k-dir",
            str(paths["selected"][0]),
            "--output-dir",
            str(output_dir),
            "--workers",
            str(workers),
        ]
    elif stage == "uniform":
        command += [
            "--full-domain-summary",
            str(paths["full_domain"][1]),
            "--decision-audit-dir",
            str(paths["audit"][0]),
            "--output-dir",
            str(output_dir),
            "--workers",
            str(workers),
        ]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return command


def stream_command(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\nCOMMAND: " + " ".join(command) + "\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    elapsed = perf_counter() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def accuracy_at_threshold(curve_path: Path, threshold: int) -> float:
    curve = pd.read_csv(curve_path)
    row = curve[
        curve["number_of_events"].astype(int) == int(threshold)
    ]
    if row.empty:
        raise ValueError(
            f"N90={threshold} is absent from curve: {curve_path}"
        )
    return float(row.iloc[0]["worst_case_accuracy"])


def should_run_uniform(
    *,
    policy: str,
    margin: float,
    target: float,
    audit_summary: dict,
    paths: dict[str, tuple[Path, Path]],
    mass: float,
) -> tuple[bool, str]:
    if policy == "always":
        return True, "uniform_policy_always"
    if policy == "never":
        return False, "uniform_policy_never"

    token = float_token(mass)
    threshold_5k = int(
        audit_summary["persistent_thresholds"][
            "final_decision_audited_5k"
        ]
    )
    curve_path = (
        paths["audit"][0]
        / f"decision_audit_final_conservative_curve_ma_{token}.csv"
    )
    accuracy = accuracy_at_threshold(curve_path, threshold_5k)
    full_summary = json.loads(paths["full_domain"][1].read_text())
    threshold_2k = int(
        full_summary["persistent_threshold_all_truths_and_seeds"]
    )
    reasons = []
    if threshold_2k != threshold_5k:
        reasons.append("2k_5k_threshold_changed")
    if accuracy - target <= float(margin):
        reasons.append("5k_crossing_marginal")
    return bool(reasons), ",".join(reasons) or "5k_crossing_comfortable"


def read_final_result(
    *,
    mass: float,
    selection: str,
    bank_path: Path,
    bank_metadata: dict,
    bank_status: str,
    paths: dict[str, tuple[Path, Path]],
    uniform_ran: bool,
    uniform_reason: str,
    stage_runtime: dict[str, float],
) -> dict:
    token = float_token(mass)
    if uniform_ran and paths["uniform"][1].is_file():
        summary_path = paths["uniform"][1]
        summary = json.loads(summary_path.read_text())
        threshold = int(
            summary["persistent_thresholds"]["uniform_selected_10k"]
        )
        validation_level = "10k"
        curve_path = (
            paths["uniform"][0]
            / f"uniform_10k_conservative_curve_ma_{token}.csv"
        )
        limiting_path = (
            paths["uniform"][0]
            / f"uniform_10k_limiting_points_ma_{token}.csv"
        )
    elif paths["audit"][1].is_file():
        summary_path = paths["audit"][1]
        summary = json.loads(summary_path.read_text())
        threshold = int(
            summary["persistent_thresholds"][
                "final_decision_audited_5k"
            ]
        )
        validation_level = "5k_audited"
        curve_path = (
            paths["audit"][0]
            / f"decision_audit_final_conservative_curve_ma_{token}.csv"
        )
        limiting_path = (
            paths["audit"][0]
            / f"decision_audit_final_limiting_points_ma_{token}.csv"
        )
    elif paths["full_domain"][1].is_file():
        summary_path = paths["full_domain"][1]
        summary = json.loads(summary_path.read_text())
        threshold = int(
            summary["persistent_threshold_all_truths_and_seeds"]
        )
        validation_level = "2k_screen"
        curve_path = (
            paths["full_domain"][0]
            / f"full_domain_2k_conservative_curve_ma_{token}.csv"
        )
        limiting_path = (
            paths["full_domain"][0]
            / f"full_domain_2k_limiting_points_ma_{token}.csv"
        )
    else:
        raise RuntimeError("No completed result stage exists.")

    return {
        "mass_GeV": float(mass),
        "selection_name": selection,
        "N90": int(threshold),
        "validation_level": validation_level,
        "bank_status": bank_status,
        "bank_path": str(bank_path),
        **bank_metadata,
        "result_summary_path": str(summary_path),
        "conservative_curve_path": str(curve_path),
        "limiting_points_path": str(limiting_path),
        "distance_map_energy_tv_path": str(
            paths["full_domain"][0]
            / f"distance_map_energy_tv_ma_{token}.pdf"
        ),
        "distance_map_joint_hellinger_path": str(
            paths["full_domain"][0]
            / f"distance_map_joint_hellinger_ma_{token}.pdf"
        ),
        "uniform_ran": bool(uniform_ran),
        "uniform_reason": uniform_reason,
        "runtime_seconds_by_stage": stage_runtime,
        "runtime_seconds_total": float(sum(stage_runtime.values())),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_master_outputs(output_root: Path, results: list[dict]) -> None:
    if not results:
        return
    table = pd.DataFrame(results).sort_values(
        ["selection_name", "mass_GeV"],
        ignore_index=True,
    )
    table.to_csv(
        output_root / "conditional_mean_z_scan_results.csv",
        index=False,
    )
    (output_root / "conditional_mean_z_scan_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    for selection, subset in table.groupby("selection_name"):
        subset = subset.sort_values("mass_GeV")
        axis.plot(
            subset["mass_GeV"],
            subset["N90"],
            marker="o",
            label=selection,
        )
        provisional = subset[subset["bank_status"] == "provisional"]
        if not provisional.empty:
            axis.scatter(
                provisional["mass_GeV"],
                provisional["N90"],
                marker="o",
                facecolors="none",
                edgecolors="black",
                label=f"{selection}: provisional bank",
            )
    axis.set_xlabel(r"ALP mass, $m_a$ [GeV]")
    axis.set_ylabel(r"Minimum observed events, $N_{90}$")
    axis.set_title(
        "Conditional energy + mean-$z$ discrimination"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_root / "conditional_mean_z_n90_vs_mass_partial.pdf"
    )
    plt.close(figure)


def inventory(
    *,
    repo: Path,
    search_root: Path,
    masses: list[float] | None,
    selection: str | None,
) -> pd.DataFrame:
    root = resolve(repo, search_root)
    candidates: list[Path] = []
    if masses:
        for mass in masses:
            candidates.extend(
                root.rglob(
                    f"template_bank_ma_{float_token(mass)}.npz"
                )
            )
    else:
        candidates = list(root.rglob("template_bank_ma_*.npz"))

    rows = []
    for path in sorted(set(candidate.resolve() for candidate in candidates)):
        try:
            metadata = load_bank_metadata(path)
        except Exception as error:
            rows.append(
                {
                    "path": str(path),
                    "load_status": f"ERROR: {type(error).__name__}: {error}",
                }
            )
            continue
        if masses and not any(
            np.isclose(metadata["mass_GeV"], mass, atol=1e-9, rtol=0)
            for mass in masses
        ):
            continue
        if selection and metadata["selection_name"] != selection:
            continue
        rows.append(
            {
                **metadata,
                "path": str(path),
                "load_status": "OK",
                "modified_unix": float(path.stat().st_mtime),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo = Path.cwd().resolve()
    if not (repo / "alp_discrimination").is_dir():
        raise SystemExit(
            "Run this command from the EventCalc-SHiP repository root."
        )

    if args.inventory_only:
        frame = inventory(
            repo=repo,
            search_root=args.search_root,
            masses=args.masses,
            selection=args.selection,
        )
        if frame.empty:
            print("No matching template banks found.")
        else:
            columns = [
                column
                for column in (
                    "mass_GeV",
                    "selection_name",
                    "profile",
                    "minimum_bin_n_eff",
                    "number_of_energy_bins",
                    "number_of_photon_lifetimes",
                    "number_of_su2_lifetimes",
                    "path",
                    "load_status",
                )
                if column in frame.columns
            ]
            print(frame[columns].to_string(index=False))
        if args.inventory_csv:
            destination = resolve(repo, args.inventory_csv)
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination, index=False)
            print(f"Inventory written: {destination}")
        return

    points = parse_points(args.point)
    if not points:
        raise SystemExit(
            "Supply at least one --point, or use --inventory-only."
        )
    if args.output_dir is None:
        raise SystemExit("--output-dir is required for scan execution.")
    if args.uniform_margin < 0:
        raise ValueError("--uniform-margin must be non-negative.")

    output_root = resolve(repo, args.output_dir)
    domain_path = resolve(repo, args.domain_path)
    if not domain_path.is_file():
        raise FileNotFoundError(
            f"Week-8 domain table not found: {domain_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    point_records = []
    for point in points:
        bank_path, metadata = validate_point(
            repo,
            point,
            allow_low_neff_bank=args.allow_low_neff_bank,
        )
        root = point_root(
            output_root,
            point.requested_mass,
            point.requested_selection,
        )
        root.mkdir(parents=True, exist_ok=True)
        paths = stage_paths(root, point.requested_mass)

        point_manifest_path = root / "controller_point_manifest.json"
        point_manifest = {
            "mass_GeV": point.requested_mass,
            "selection_name": point.requested_selection,
            "bank_path": str(bank_path),
            "bank_status": args.bank_status,
            "bank_metadata": metadata,
            "status": "running",
            "stage_runtime_seconds": {},
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        point_manifest_path.write_text(
            json.dumps(point_manifest, indent=2) + "\n"
        )

        stage_runtime: dict[str, float] = {}
        uniform_ran = False
        uniform_reason = "not_evaluated"

        try:
            for stage in STAGES:
                if stage == "uniform":
                    if args.dry_run:
                        uniform_ran = args.uniform_policy != "never"
                        uniform_reason = (
                            "dry_run_uniform_would_run"
                            if uniform_ran
                            else "dry_run_uniform_would_skip"
                        )
                    else:
                        audit_summary = json.loads(
                            paths["audit"][1].read_text()
                        )
                        uniform_ran, uniform_reason = should_run_uniform(
                            policy=args.uniform_policy,
                            margin=args.uniform_margin,
                            target=0.90,
                            audit_summary=audit_summary,
                            paths=paths,
                            mass=point.requested_mass,
                        )
                    if not uniform_ran:
                        print(
                            f"SKIP uniform 10k for "
                            f"m={point.requested_mass:g}: "
                            f"{uniform_reason}",
                            flush=True,
                        )
                        break

                output_dir, summary_path = paths[stage]
                if summary_path.is_file():
                    print(
                        f"SKIP completed {stage}: {summary_path}",
                        flush=True,
                    )
                    stage_runtime[stage] = 0.0
                else:
                    command = command_for_stage(
                        stage=stage,
                        bank_path=bank_path,
                        paths=paths,
                        workers=args.workers,
                        domain_path=domain_path,
                    )
                    print(
                        "\nRUN "
                        f"m={point.requested_mass:g}, "
                        f"selection={point.requested_selection}, "
                        f"stage={stage}\n"
                        + " ".join(command),
                        flush=True,
                    )
                    if args.dry_run:
                        stage_runtime[stage] = 0.0
                    else:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        stage_runtime[stage] = stream_command(
                            command,
                            output_dir / "controller_stage.log",
                        )

                if STAGES.index(stage) >= STAGES.index(args.stop_after):
                    break

            if args.dry_run:
                point_manifest["status"] = "dry_run"
                point_manifest["stage_runtime_seconds"] = stage_runtime
                point_manifest_path.write_text(
                    json.dumps(point_manifest, indent=2) + "\n"
                )
                continue

            result = read_final_result(
                mass=point.requested_mass,
                selection=point.requested_selection,
                bank_path=bank_path,
                bank_metadata=metadata,
                bank_status=args.bank_status,
                paths=paths,
                uniform_ran=uniform_ran,
                uniform_reason=uniform_reason,
                stage_runtime=stage_runtime,
            )
            (root / "point_result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            point_records.append(result)
            point_manifest["status"] = "complete"
            point_manifest["stage_runtime_seconds"] = stage_runtime
            point_manifest["result"] = result
            point_manifest["completed_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            point_manifest_path.write_text(
                json.dumps(point_manifest, indent=2) + "\n"
            )
            write_master_outputs(output_root, point_records)

        except Exception as error:
            point_manifest["status"] = "failed"
            point_manifest["error"] = (
                f"{type(error).__name__}: {error}"
            )
            point_manifest["stage_runtime_seconds"] = stage_runtime
            point_manifest_path.write_text(
                json.dumps(point_manifest, indent=2) + "\n"
            )
            if not args.continue_on_error:
                raise
            print(
                f"FAILED point m={point.requested_mass:g}: {error}",
                file=sys.stderr,
                flush=True,
            )

    if point_records:
        write_master_outputs(output_root, point_records)
        print(
            f"\nScan outputs: {output_root}\n"
            f"Completed points: {len(point_records)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
