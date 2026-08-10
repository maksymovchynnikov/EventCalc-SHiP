"""Unified user-facing ALP-photon versus ALP-SU(2)L analysis.

This is the normal entry point for the final project workflow. It resolves
registered lifetime-template banks, optionally builds missing banks, executes
the staged conditional-feature production analysis, performs optional 10k and
empirical-resampling checks, and regenerates compact project tables/plots after
every completed point.

The safe default is ``reuse_only``: use existing validated/production banks and
skip unavailable points rather than unexpectedly starting an expensive bank
construction.
"""

from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alp_discrimination.statistics.adaptive_grid import AdaptiveScanSettings
from alp_discrimination.statistics.basic import MINIMUM_OBSERVED_EVENTS
from alp_discrimination.planning import (
    AnalysisConfig,
    build_analysis_plan,
    write_run_configuration,
)
from alp_discrimination.templates.conditional_features import FEATURE_LABELS, FEATURE_SUBSETS
from alp_discrimination.templates.lifetime_banks import load_template_bank
from alp_discrimination.paths import OUTPUT_ROOT
from alp_discrimination.workflows import float_token
from alp_discrimination.workflows.lifetime_bank_builder import (
    run_point as run_adaptive_bank_point,
)
from alp_discrimination.workflows.results import write_project_outputs


SELECTIONS = ("diphoton_ecal", "diphoton_ecal_e1gev")
ALL_OBSERVABLES = (
    "energy",
    "energy_mean_z",
    "energy_mean_r_perp",
    "energy_mean_z_r_perp",
)
DEFAULT_OBSERVABLES = ("energy_mean_z_r_perp",)
PRODUCTION_SEEDS = (73241, 83244, 93247, 103250, 113253)

# Public stage names are physics-facing. Legacy tokens remain accepted so
# existing checkpoint/resume commands continue to work unchanged.
STOP_AFTER_ALIASES = {
    "bank": "bank",
    "moments": "moments",
    "threshold_scan": "rangefinder",
    "rangefinder": "rangefinder",
    "lifetime_scan": "full_domain",
    "full_domain": "full_domain",
    "validation": "selected",
    "selected": "selected",
    "empirical": "empirical",
    "final": "final",
}


def parse_stop_after(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    try:
        return STOP_AFTER_ALIASES[token]
    except KeyError as exc:
        public = (
            "bank, moments, threshold_scan, lifetime_scan, "
            "validation, empirical, final"
        )
        raise ArgumentTypeError(
            f"Unknown stop stage {value!r}; choose one of: {public}."
        ) from exc


DEFAULT_DOMAIN_PATH = (
    OUTPUT_ROOT
    / "production"
    / "alp_su2l_analysis"
    / "final_results"
    / "provenance"
    / "allowed_lifetime_domains.csv"
)
DEFAULT_EXISTING_BANK_MANIFEST = (
    OUTPUT_ROOT / "production" / "alp_su2l_analysis" / "existing_bank_manifest.csv"
)
DEFAULT_FROZEN_BANK_MANIFEST = (
    OUTPUT_ROOT
    / "production"
    / "alp_su2l_analysis"
    / "final_results"
    / "provenance"
    / "frozen_bank_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    OUTPUT_ROOT / "production" / "alp_su2l_analysis" / "final_results"
)


def selection_token(selection_name: str) -> str:
    if selection_name == "diphoton_ecal":
        return "geom"
    if selection_name == "diphoton_ecal_e1gev":
        return "e1gev"
    raise ValueError(f"Unknown selection: {selection_name}")


def bank_quality_metadata(bank_state: str) -> dict:
    state = str(bank_state)
    if state == "production_noise_floor_limited":
        return {
            "bank_state": state,
            "physics_usable": True,
            "global_minimum_status": "stable",
            "refinement_status": "numerical_template_statistical_noise_floor",
            "interpretation": (
                "The global distance minimum remained stable under lifetime-grid "
                "refinement. Only the formal local refinement/interpolation "
                "criterion reached the numerical/template-statistical noise floor; "
                "this is not an unstable physical minimum."
            ),
        }
    if state == "validated":
        return {
            "bank_state": state,
            "physics_usable": True,
            "global_minimum_status": "validated",
            "refinement_status": "validated",
            "interpretation": "Validated registered lifetime-template bank.",
        }
    if state == "production":
        return {
            "bank_state": state,
            "physics_usable": True,
            "global_minimum_status": "stable",
            "refinement_status": "converged",
            "interpretation": "Converged production lifetime-template bank.",
        }
    return {
        "bank_state": state,
        "physics_usable": False,
        "global_minimum_status": "not_certified",
        "refinement_status": "not_certified",
        "interpretation": "This bank state is not certified for final physics use.",
    }


def _prompt_text(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def configure_interactively(args):
    print("\n===== INTERACTIVE ALP-SU2L ANALYSIS =====")
    print("Press Enter to accept the safe/default choice.\n")

    masses = _prompt_text("Masses in GeV (space separated)", "0.3 0.5 1.0 2.5")
    args.masses = [float(x) for x in masses.replace(",", " ").split()]

    print("Selections: 1=both, 2=ECAL only, 3=ECAL + E_gamma>=1 GeV")
    selection_choice = _prompt_text("Selection", "1")
    selection_map = {
        "1": list(SELECTIONS),
        "2": ["diphoton_ecal"],
        "3": ["diphoton_ecal_e1gev"],
    }
    if selection_choice not in selection_map:
        raise ValueError("Selection must be 1, 2, or 3.")
    args.selections = selection_map[selection_choice]

    print("Observables: 1=headline E+<z>+<r_perp>, 2=all four production observables")
    observable_choice = _prompt_text("Observable set", "1")
    if observable_choice == "1":
        args.all_observables = False
        args.observables = ["energy_mean_z_r_perp"]
    elif observable_choice == "2":
        args.all_observables = True
        args.observables = list(ALL_OBSERVABLES)
    else:
        raise ValueError("Observable set must be 1 or 2.")

    print("Bank mode: 1=reuse existing only (safe), 2=automatic build/resume missing")
    mode_choice = _prompt_text("Bank mode", "1")
    if mode_choice not in {"1", "2"}:
        raise ValueError("Bank mode must be 1 or 2.")
    args.run_mode = "reuse_only" if mode_choice == "1" else "automatic"

    args.profile = _prompt_text("Profile (production/validation/quick)", "production")
    if args.profile not in {"production", "validation", "quick"}:
        raise ValueError("Unknown profile.")
    args.workers = int(_prompt_text("Workers (1 or 2)", "2"))
    if args.workers not in (1, 2):
        raise ValueError("Workers must be 1 or 2.")
    args.stop_after = parse_stop_after(
        _prompt_text(
            "Stop after "
            "(moments/threshold_scan/lifetime_scan/validation/empirical/final)",
            "final",
        )
    )
    return args


def historical_runtime_minutes(output_dir: Path, mass: float, selection: str, stop_after: str):
    path = Path(output_dir) / "runtime_history.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    required = {"mass_GeV", "selection_name", "stop_after", "status", "runtime_seconds"}
    if not required.issubset(frame.columns):
        return None
    frame = frame[
        np.isclose(frame["mass_GeV"].to_numpy(float), float(mass), atol=1e-12, rtol=0)
        & (frame["selection_name"].astype(str) == str(selection))
        & (frame["stop_after"].astype(str) == str(stop_after))
    ].copy()
    frame = frame[
        (~frame["status"].astype(str).isin(["failed", "running"]))
        & (pd.to_numeric(frame["runtime_seconds"], errors="coerce") > 0)
    ]
    if frame.empty:
        return None
    return float(pd.to_numeric(frame["runtime_seconds"]).median()) / 60.0


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--masses", nargs="+", type=float)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Configure the normal analysis interactively.",
    )
    parser.add_argument(
        "--selections",
        nargs="+",
        choices=SELECTIONS,
        default=list(SELECTIONS),
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        choices=ALL_OBSERVABLES,
        default=list(DEFAULT_OBSERVABLES),
    )
    parser.add_argument(
        "--all-observables",
        action="store_true",
        help="Run all four production observables.",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "validation", "production"),
        default="production",
    )
    parser.add_argument(
        "--run-mode",
        choices=("reuse_only", "automatic", "custom"),
        default="reuse_only",
    )
    parser.add_argument(
        "--stop-after",
        type=parse_stop_after,
        default="final",
        metavar="STAGE",
        help=(
            "Stop after bank, moments, threshold_scan, lifetime_scan, "
            "validation, empirical, or final. Legacy stage tokens remain "
            "accepted for checkpoint compatibility."
        ),
    )
    parser.add_argument("--bank-manifest", type=Path)
    parser.add_argument("--domain-path", type=Path, default=DEFAULT_DOMAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument(
        "--threshold-scan-pseudoexperiments",
        "--screen-pseudoexperiments",
        dest="screen_pseudoexperiments",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lifetime-scan-pseudoexperiments",
        "--full-domain-pseudoexperiments",
        dest="full_domain_pseudoexperiments",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--validation-pseudoexperiments",
        "--selected-pseudoexperiments",
        dest="selected_pseudoexperiments",
        type=int,
        default=5000,
    )
    parser.add_argument("--empirical-pseudoexperiments", type=int, default=2000)
    parser.add_argument(
        "--validation-10k-policy",
        "--selected-10k-policy",
        dest="selected_10k_policy",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument("--warning-free-gib", type=float, default=10.0)
    return parser.parse_args(argv)


def resolve_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def choose_manifest(repo: Path, requested: Path | None) -> Path:
    if requested is not None:
        return resolve_path(repo, requested)
    frozen = resolve_path(repo, DEFAULT_FROZEN_BANK_MANIFEST)
    if frozen.is_file():
        return frozen
    return resolve_path(repo, DEFAULT_EXISTING_BANK_MANIFEST)


def load_bank_manifest(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "bank_path" not in table.columns and "frozen_bank_path" in table.columns:
        table = table.copy()
        table["bank_path"] = table["frozen_bank_path"]
    required = {"mass_GeV", "selection_name", "status", "bank_path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Bank manifest is missing columns: {sorted(missing)}")
    return table


def resolve_bank_record(
    table: pd.DataFrame,
    mass_gev: float,
    selection_name: str,
) -> dict:
    mask = (
        np.isclose(
            table["mass_GeV"].to_numpy(dtype=float),
            float(mass_gev),
            rtol=0.0,
            atol=1.0e-12,
        )
        & (table["selection_name"].astype(str) == str(selection_name))
    )
    matches = table.loc[mask]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No bank registered for m_a={mass_gev:g}, {selection_name}."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Multiple bank records for m_a={mass_gev:g}, {selection_name}."
        )
    return matches.iloc[0].to_dict()


def validate_bank(bank_path: Path, mass_gev: float, selection_name: str):
    bank = load_template_bank(bank_path)
    if not np.isclose(float(bank.mass_gev), float(mass_gev), atol=1e-12, rtol=0):
        raise ValueError("Bank mass does not match the requested mass.")
    if str(bank.selection_name) != str(selection_name):
        raise ValueError("Bank selection does not match the requested selection.")
    return bank


def adaptive_bank_output_root(config: AnalysisConfig) -> Path:
    return Path(config.output_dir) / "bank_workspaces"


def generated_bank_from_state(
    *,
    repo: Path,
    state_path: Path,
    mass_gev: float,
    selection_name: str,
) -> tuple[Path, str]:
    if not state_path.is_file():
        raise RuntimeError(f"Adaptive bank stage produced no state file: {state_path}")
    state = json.loads(state_path.read_text())
    if state.get("status") != "bank_complete":
        raise RuntimeError(
            f"Adaptive bank stage did not reach bank_complete: {state.get('status')!r}"
        )
    bank_dir = state.get("bank_dir")
    if not bank_dir:
        raise RuntimeError("Completed adaptive state does not record bank_dir.")
    bank_dir = resolve_path(repo, bank_dir)
    bank_path = (
        bank_dir
        / "template_banks"
        / f"template_bank_ma_{float_token(mass_gev)}.npz"
    )
    if not bank_path.is_file():
        raise FileNotFoundError(bank_path)
    validate_bank(bank_path, mass_gev, selection_name)
    return bank_path, str(state.get("bank_status", "generated"))


CLEAN_ADAPTIVE_BANK_STATUSES = {
    "lifetime_grid_converged",
    "fine_binning_converged",
}


def registry_status_from_adaptive_status(adaptive_status: str) -> str:
    return (
        "production"
        if str(adaptive_status) in CLEAN_ADAPTIVE_BANK_STATUSES
        else "incomplete"
    )


def portable_registry_path(repo: Path, path: Path) -> str:
    repo = Path(repo).resolve()
    path = Path(path).resolve()
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def persist_generated_bank_record(
    *,
    manifest_path: Path,
    repo: Path,
    mass_gev: float,
    selection_name: str,
    bank_path: Path,
    adaptive_status: str,
) -> str:
    raw_columns = pd.read_csv(manifest_path, nrows=1).columns
    if "frozen_bank_path" in raw_columns:
        raise RuntimeError(
            "Automatic construction must not rewrite the frozen provenance "
            "manifest. Pass existing_bank_manifest.csv."
        )
    status = registry_status_from_adaptive_status(adaptive_status)
    table = load_bank_manifest(manifest_path)
    record = {
        "mass_GeV": float(mass_gev),
        "selection_name": str(selection_name),
        "status": status,
        "bank_path": portable_registry_path(repo, bank_path),
        "note": (
            "Automatically generated by the unified ALP-SU2L controller. "
            f"Adaptive bank status: {adaptive_status}."
        ),
    }
    mask = (
        np.isclose(
            table["mass_GeV"].to_numpy(dtype=float),
            float(mass_gev),
            atol=1e-12,
            rtol=0,
        )
        & (table["selection_name"].astype(str) == str(selection_name))
    )
    if mask.any():
        if int(mask.sum()) != 1:
            raise ValueError("Duplicate bank registry records.")
        for key, value in record.items():
            table.loc[mask, key] = value
    else:
        table = pd.concat([table, pd.DataFrame([record])], ignore_index=True)
    table.sort_values(["mass_GeV", "selection_name"], inplace=True, ignore_index=True)
    table.to_csv(manifest_path, index=False)
    return status


def build_or_resume_bank(
    *,
    config: AnalysisConfig,
    repo: Path,
    domains: pd.DataFrame,
    mass_gev: float,
    selection_name: str,
) -> tuple[Path, str]:
    output_root = adaptive_bank_output_root(config)
    run_adaptive_bank_point(
        mass_gev=float(mass_gev),
        selection_name=str(selection_name),
        profile=str(config.profile),
        domain_path=Path(config.domain_path),
        domains=domains,
        output_dir=output_root,
        settings=AdaptiveScanSettings(),
        workers=int(config.workers),
        stop_after="bank",
        skip_conditional_binning_check=False,
        diagnostic_plots=False,
    )
    state_path = (
        output_root
        / "per_mass"
        / f"ma_{float_token(mass_gev)}"
        / selection_token(selection_name)
        / "state.json"
    )
    return generated_bank_from_state(
        repo=repo,
        state_path=state_path,
        mass_gev=mass_gev,
        selection_name=selection_name,
    )


def point_root(output_dir: Path, mass_gev: float, selection_name: str) -> Path:
    return (
        Path(output_dir)
        / "per_point"
        / f"ma_{float_token(mass_gev)}"
        / selection_token(selection_name)
    )


def free_gib(path: Path) -> float:
    return float(shutil.disk_usage(path).free) / (1024.0**3)


def disk_guard(
    path: Path,
    *,
    minimum_gib: float,
    warning_gib: float,
    operation: str,
) -> None:
    available = free_gib(path)
    if available < float(minimum_gib):
        raise RuntimeError(
            f"Disk guard: only {available:.2f} GiB free before {operation}; "
            f"minimum is {minimum_gib:.2f} GiB."
        )
    if available < float(warning_gib):
        print(f"WARNING: {available:.2f} GiB free before {operation}.", flush=True)


def selected_summary_path(directory: Path, mass: float) -> Path:
    return directory / f"selected_5k_audit_summary_ma_{float_token(mass)}.json"


def range_result(point: Path, observable: str) -> dict:
    path = point / "rangefinder" / observable / "rangefinder_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def selection_counts(threshold: int, event_counts: Sequence[int]) -> list[int]:
    available = set(int(x) for x in event_counts)
    values = [
        x for x in (int(threshold) - 1, int(threshold), int(threshold) + 1)
        if x >= MINIMUM_OBSERVED_EVENTS and x in available
    ]
    if int(threshold) not in values:
        raise ValueError("Selected threshold is absent from the event grid.")
    return values


def empirical_counts(threshold: int, event_counts: Sequence[int]) -> list[int]:
    available = set(int(x) for x in event_counts)
    values = sorted(
        {
            x
            for x in (
                max(MINIMUM_OBSERVED_EVENTS, int(threshold) - 1),
                int(threshold),
                int(threshold) + 1,
                int(threshold) + 2,
            )
            if x in available
        }
    )
    if int(threshold) not in values:
        raise ValueError("Empirical grid lacks selected threshold.")
    return values


def run_selected_10k(
    *,
    point: Path,
    bank_path: Path,
    observable: str,
    workers: int,
    chunk_size: int,
    resume: bool,
) -> tuple[Path, dict]:
    bank = load_template_bank(bank_path)
    out = point / "selected_10k" / observable
    summary_path = selected_summary_path(out, float(bank.mass_gev))
    if summary_path.is_file() and resume:
        return out, json.loads(summary_path.read_text())

    selected_5k_summary = json.loads(
        selected_summary_path(point / "selected" / observable, float(bank.mass_gev))
        .read_text()
    )
    threshold = int(selected_5k_summary["persistent_thresholds"]["selected_5k"])
    event_counts = [int(x) for x in selected_5k_summary["event_counts"]]

    full_domain_dir = Path(
        selected_5k_summary.get(
            "full_domain_dir", point / "full_domain" / observable
        )
    )
    command = [
        sys.executable,
        "-m",
        "alp_discrimination.workflows.conditional_feature_selected",
        "--full-domain-dir",
        str(full_domain_dir),
        "--bank-path",
        str(bank_path),
        "--moments-path",
        str(
            point
            / "moments"
            / f"conditional_feature_moments_ma_{float_token(bank.mass_gev)}.npz"
        ),
        "--output-dir",
        str(out),
        "--observable",
        observable,
        "--pseudoexperiments",
        "10000",
        "--seeds",
        *[str(x) for x in PRODUCTION_SEEDS],
        "--event-counts",
        *[str(x) for x in event_counts],
        "--selection-event-counts",
        *[str(x) for x in selection_counts(threshold, event_counts)],
        "--workers",
        str(int(workers)),
        "--chunk-size",
        str(int(chunk_size)),
    ]
    subprocess.run(command, check=True)
    return out, json.loads(summary_path.read_text())


def run_empirical_validation(
    *,
    point: Path,
    bank_path: Path,
    domain_path: Path,
    selected_dir: Path,
    observable: str,
    threshold: int,
    pseudoexperiments: int,
    workers: int,
    chunk_size: int,
    resume: bool,
) -> dict:
    bank = load_template_bank(bank_path)
    token = float_token(float(bank.mass_gev))
    out = point / "empirical" / observable
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"conditional_feature_empirical_summary_ma_{token}.json"
    if summary_path.is_file() and resume:
        return json.loads(summary_path.read_text())

    truth_file = out / f"empirical_truths_ma_{token}.csv"
    if not truth_file.is_file():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "alp_discrimination.workflows.conditional_feature_empirical_truths",
                "--selected-dir",
                str(selected_dir),
                "--bank-path",
                str(bank_path),
                "--output",
                str(truth_file),
            ],
            check=True,
        )

    selected_summary = json.loads(
        selected_summary_path(selected_dir, float(bank.mass_gev)).read_text()
    )
    event_counts = [int(x) for x in selected_summary["event_counts"]]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "alp_discrimination.workflows.conditional_feature_empirical_validation",
            "--bank-path",
            str(bank_path),
            "--moments-path",
            str(
                point
                / "moments"
                / f"conditional_feature_moments_ma_{token}.npz"
            ),
            "--selected-truths",
            str(truth_file),
            "--observable",
            observable,
            "--output-dir",
            str(out),
            "--domain-path",
            str(domain_path),
            "--pseudoexperiments",
            str(int(pseudoexperiments)),
            "--seeds",
            *[str(x) for x in PRODUCTION_SEEDS],
            "--event-counts",
            *[str(x) for x in empirical_counts(threshold, event_counts)],
            "--workers",
            str(int(workers)),
            "--chunk-size",
            str(int(chunk_size)),
        ],
        check=True,
    )
    return json.loads(summary_path.read_text())


def empirical_n90(summary: dict) -> tuple[int | None, int | None]:
    values = summary.get("persistent_thresholds", {})
    gaussian = values.get("gaussian_truth")
    empirical = values.get("empirical_truth")
    return (
        None if gaussian is None else int(gaussian),
        None if empirical is None else int(empirical),
    )


def _observable_sort_key(row: dict) -> tuple[int, str]:
    observable = str(row.get("observable", ""))
    try:
        index = ALL_OBSERVABLES.index(observable)
    except ValueError:
        index = len(ALL_OBSERVABLES)
    return index, observable


def merge_observable_records(
    existing_rows: Sequence[dict],
    new_rows: Sequence[dict],
) -> list[dict]:
    """Upsert per-observable records without dropping results from subset runs."""

    merged: dict[str, dict] = {}
    for row in [*existing_rows, *new_rows]:
        observable = str(row.get("observable", ""))
        if not observable:
            raise ValueError("Observable result row is missing 'observable'.")
        merged[observable] = dict(row)
    return sorted(merged.values(), key=_observable_sort_key)


def _read_csv_records(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def finalize_point(
    *,
    point: Path,
    bank_path: Path,
    domain_path: Path,
    observables: Sequence[str],
    selected_10k_policy: str,
    empirical_pseudoexperiments: int,
    workers: int,
    chunk_size: int,
    resume: bool,
    bank_state: str,
) -> dict:
    bank = load_template_bank(bank_path)
    quality = bank_quality_metadata(bank_state)
    rows = []
    limiting_rows = []
    distance_rows = []
    empirical_rows = []

    for observable in observables:
        selected_dir = point / "selected" / observable
        summary = json.loads(
            selected_summary_path(selected_dir, float(bank.mass_gev)).read_text()
        )

        run_10k = (
            selected_10k_policy == "always"
            or (
                selected_10k_policy == "auto"
                and bool(summary.get("recommend_selected_10k", False))
            )
        )
        validation_level = "selected_5k_decision_audited"
        if run_10k:
            selected_dir, summary = run_selected_10k(
                point=point,
                bank_path=bank_path,
                observable=observable,
                workers=workers,
                chunk_size=chunk_size,
                resume=resume,
            )
            validation_level = "selected_10k_decision_audited"

        threshold = int(summary["persistent_thresholds"]["selected_5k"])
        audit_passed = bool(summary.get("final_omitted_truth_audit_passed", False))
        gaussian_empirical = empirical = None
        empirical_confirmed = True

        if observable != "energy":
            empirical_summary = run_empirical_validation(
                point=point,
                bank_path=bank_path,
                domain_path=domain_path,
                selected_dir=selected_dir,
                observable=observable,
                threshold=threshold,
                pseudoexperiments=empirical_pseudoexperiments,
                workers=workers,
                chunk_size=chunk_size,
                resume=resume,
            )
            gaussian_empirical, empirical = empirical_n90(empirical_summary)
            empirical_confirmed = empirical == threshold
            validation_level += (
                "_empirical_confirmed"
                if empirical_confirmed
                else "_empirical_mismatch"
            )
            diff = empirical_summary.get("curve_difference", {})
            empirical_rows.append(
                {
                    "mass_GeV": float(bank.mass_gev),
                    "selection_name": str(bank.selection_name),
                    "observable": observable,
                    "selected_N90": threshold,
                    "gaussian_empirical_N90": gaussian_empirical,
                    "empirical_N90": empirical,
                    "threshold_confirmed": empirical_confirmed,
                    "maximum_absolute_curve_difference": diff.get(
                        "maximum_absolute_difference"
                    ),
                    "mean_absolute_curve_difference": diff.get(
                        "mean_absolute_difference"
                    ),
                }
            )

        active_full_domain = Path(
            summary.get(
                "full_domain_dir", point / "full_domain" / observable
            )
        )
        full_curve = pd.read_csv(
            active_full_domain
            / f"conditional_feature_pilot_thresholds_ma_{float_token(bank.mass_gev)}.csv"
        )
        full_row = full_curve[
            full_curve["observable"].astype(str) == observable
        ]
        full_n90 = (
            None
            if full_row.empty
            else int(full_row.iloc[0]["provisional_persistent_threshold"])
        )

        rows.append(
            {
                "mass_GeV": float(bank.mass_gev),
                "selection_name": str(bank.selection_name),
                "observable": observable,
                "N90": threshold,
                "bank_state": quality["bank_state"],
                "bank_physics_usable": quality["physics_usable"],
                "bank_global_minimum_status": quality["global_minimum_status"],
                "bank_refinement_status": quality["refinement_status"],
                "full_domain_2k_N90": full_n90,
                "validation_level": validation_level,
                "omitted_truth_audit_passed": audit_passed,
                "empirical_validation_required": observable != "energy",
                "empirical_gaussian_N90": gaussian_empirical,
                "empirical_N90": empirical,
                "empirical_threshold_confirmed": empirical_confirmed,
                "project_final": bool(audit_passed and empirical_confirmed),
                "selected_summary_path": str(
                    selected_summary_path(selected_dir, float(bank.mass_gev))
                ),
            }
        )

        limiter = summary.get("limiting_truth_at_selected_5k_threshold", {})
        if limiter:
            limiting_rows.append(
                {
                    "mass_GeV": float(bank.mass_gev),
                    "selection_name": str(bank.selection_name),
                    "observable": observable,
                    "N90": threshold,
                    **limiter,
                }
            )

        distance_file = (
            active_full_domain
            / f"conditional_feature_distance_minima_ma_{float_token(bank.mass_gev)}.csv"
        )
        if distance_file.is_file():
            table = pd.read_csv(distance_file)
            row = table[table["observable"].astype(str) == observable]
            if not row.empty:
                distance_rows.append(
                    {
                        "mass_GeV": float(bank.mass_gev),
                        "selection_name": str(bank.selection_name),
                        **row.iloc[0].to_dict(),
                    }
                )

    tables = point / "tables"
    plots = point / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    previous_summary_path = point / "point_summary.json"
    previous_rows: list[dict] = []
    if previous_summary_path.is_file():
        try:
            previous_summary = json.loads(previous_summary_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed existing point summary: {previous_summary_path}"
            ) from exc
        if (
            float(previous_summary.get("mass_GeV")) != float(bank.mass_gev)
            or str(previous_summary.get("selection_name"))
            != str(bank.selection_name)
        ):
            raise ValueError(
                f"Existing point summary does not match {bank.mass_gev:g}, "
                f"{bank.selection_name}: {previous_summary_path}"
            )
        previous_rows = list(previous_summary.get("results", []))

    rows = merge_observable_records(previous_rows, rows)
    limiting_rows = merge_observable_records(
        _read_csv_records(tables / "limiting_truths.csv"),
        limiting_rows,
    )
    distance_rows = merge_observable_records(
        _read_csv_records(tables / "distance_summary.csv"),
        distance_rows,
    )
    empirical_rows = merge_observable_records(
        _read_csv_records(tables / "empirical_summary.csv"),
        empirical_rows,
    )

    result_table = pd.DataFrame(rows)
    result_table.to_csv(tables / "n90_by_observable.csv", index=False)
    pd.DataFrame(limiting_rows).to_csv(tables / "limiting_truths.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(tables / "distance_summary.csv", index=False)
    pd.DataFrame(empirical_rows).to_csv(tables / "empirical_summary.csv", index=False)

    if not result_table.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(result_table))
        ax.bar(x, result_table["N90"].to_numpy(dtype=float))
        ax.set_xticks(
            x,
            [FEATURE_LABELS.get(o, o) for o in result_table["observable"]],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(r"Minimum observed events, $N_{90}$")
        # Observable names are already on the x axis; keep the title to mass only.
        ax.set_title(rf"$m_a={float(bank.mass_gev):g}$ GeV")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "n90_observable_ablation.pdf")
        fig.savefig(plots / "n90_observable_ablation.png", dpi=200)
        plt.close(fig)

    point_summary = {
        "status": (
            "final_for_project"
            if rows and all(row["project_final"] for row in rows)
            else "validation_attention_required"
        ),
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "bank_path": str(bank_path),
        "bank_quality": quality,
        "observables": [str(row["observable"]) for row in rows],
        "results": rows,
        "interpretation": (
            "Project-final means the lifetime-domain audit passed and every "
            "spatial observable reproduced its selected N90 under direct "
            "empirical EventCalc-row resampling. Detector reconstruction and "
            "systematic effects remain outside this truth-level project."
        ),
    }
    (point / "point_summary.json").write_text(
        json.dumps(point_summary, indent=2) + "\n"
    )
    return point_summary


def run_production_subprocess(
    *,
    bank_path: Path,
    domain_path: Path,
    point: Path,
    observables: Sequence[str],
    stop_after: str,
    workers: int,
    chunk_size: int,
    screen_pes: int,
    full_pes: int,
    selected_pes: int,
    resume: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "alp_discrimination.workflows.conditional_feature_production",
        "--bank-path",
        str(bank_path),
        "--domain-path",
        str(domain_path),
        "--output-dir",
        str(point),
        "--observables",
        *[str(x) for x in observables],
        "--workers",
        str(int(workers)),
        "--chunk-size",
        str(int(chunk_size)),
        "--screen-pseudoexperiments",
        str(int(screen_pes)),
        "--full-domain-pseudoexperiments",
        str(int(full_pes)),
        "--selected-pseudoexperiments",
        str(int(selected_pes)),
        "--stop-after",
        str(stop_after),
    ]
    if resume:
        command.append("--resume")
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def append_runtime_history(output_dir: Path, row: dict) -> None:
    path = Path(output_dir) / "runtime_history.csv"
    frame = pd.DataFrame([row])
    if path.is_file():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.interactive:
        args = configure_interactively(args)
    elif not args.masses:
        raise SystemExit("Provide --masses ... or use --interactive.")

    repo = Path.cwd().resolve()
    if not (repo / "alp_discrimination").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")

    if args.bank_manifest is None and str(args.run_mode) == "automatic":
        manifest_path = resolve_path(repo, DEFAULT_EXISTING_BANK_MANIFEST)
    else:
        manifest_path = choose_manifest(repo, args.bank_manifest)
    domain_path = resolve_path(repo, args.domain_path)
    output_dir = resolve_path(repo, args.output_dir)

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not domain_path.is_file():
        raise FileNotFoundError(domain_path)

    manifest = load_bank_manifest(manifest_path)
    observables = (
        ALL_OBSERVABLES
        if args.all_observables
        else tuple(dict.fromkeys(str(x) for x in args.observables))
    )
    config = AnalysisConfig(
        masses=tuple(dict.fromkeys(float(x) for x in args.masses)),
        selections=tuple(dict.fromkeys(str(x) for x in args.selections)),
        observables=observables,
        profile=str(args.profile),
        workers=int(args.workers),
        run_mode=str(args.run_mode),
        output_dir=output_dir,
        domain_path=domain_path,
        bank_manifest=manifest_path,
        resume=bool(args.resume),
    )

    plan = build_analysis_plan(config=config, manifest=manifest, repo=repo)
    plan["bank_status"] = plan["bank_state"]
    plan["adaptive_bank_status"] = ""
    qualities = [bank_quality_metadata(state) for state in plan["bank_state"]]
    plan["physics_usable_bank"] = [q["physics_usable"] for q in qualities]
    plan["global_minimum_status"] = [q["global_minimum_status"] for q in qualities]
    plan["refinement_status"] = [q["refinement_status"] for q in qualities]
    plan["point_root"] = [
        str(point_root(output_dir, float(r.mass_GeV), str(r.selection_name)))
        for r in plan.itertuples()
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_dir / "latest_run_plan.csv", index=False)
    write_run_configuration(config, output_dir)

    print("===== ALP-SU2L ANALYSIS PLAN =====")
    display_plan = plan[
        [
            "mass_GeV", "selection_name", "bank_state", "bank_action",
            "global_minimum_status", "refinement_status",
        ]
    ].copy()
    estimates = []
    for planned in plan.itertuples(index=False):
        if str(planned.bank_action) == "skip_unavailable":
            estimates.append("unavailable")
            continue
        minutes = historical_runtime_minutes(
            output_dir, float(planned.mass_GeV), str(planned.selection_name), str(args.stop_after)
        )
        estimates.append(
            "learning" if minutes is None else f"~{minutes:.0f} min"
        )
    display_plan["historical_runtime_hint"] = estimates
    print(display_plan.to_string(index=False))
    print(f"Free disk space: {free_gib(repo):.2f} GiB")
    print(
        "Runtime hints are based only on completed comparable runs; "
        "live ETA is updated from measured checkpoint throughput."
    )

    if args.interactive and not args.dry_run:
        answer = input("Proceed with this plan? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled before any EventCalc or pseudoexperiments were launched.")
            return

    if args.dry_run:
        write_project_outputs(output_dir, requested_plan=plan)
        print("\nDRY RUN: no EventCalc or pseudoexperiments launched.")
        return

    domains = pd.read_csv(domain_path)

    # Bank preparation only when explicitly requested through automatic mode.
    for index, row in plan.iterrows():
        action = str(row["bank_action"])
        if action in ("reuse", "skip_unavailable"):
            continue
        if action == "requires_bank":
            raise RuntimeError(
                f"Custom mode requires a reusable bank for "
                f"m={float(row['mass_GeV']):g}, {row['selection_name']}."
            )
        if action not in ("build", "build_or_resume"):
            raise RuntimeError(f"Unknown bank action: {action}")
        if "frozen_bank_path" in pd.read_csv(manifest_path, nrows=1).columns:
            raise RuntimeError(
                "Automatic mode cannot update the frozen manifest. "
                "Pass existing_bank_manifest.csv."
            )
        disk_guard(
            repo,
            minimum_gib=max(10.0, float(args.minimum_free_gib)),
            warning_gib=max(15.0, float(args.warning_free_gib)),
            operation="adaptive bank construction",
        )
        mass = float(row["mass_GeV"])
        selection = str(row["selection_name"])
        bank_path, adaptive_status = build_or_resume_bank(
            config=config,
            repo=repo,
            domains=domains,
            mass_gev=mass,
            selection_name=selection,
        )
        registry_status = persist_generated_bank_record(
            manifest_path=manifest_path,
            repo=repo,
            mass_gev=mass,
            selection_name=selection,
            bank_path=bank_path,
            adaptive_status=adaptive_status,
        )
        plan.at[index, "bank_path"] = str(bank_path)
        plan.at[index, "bank_exists"] = True
        plan.at[index, "adaptive_bank_status"] = adaptive_status
        plan.at[index, "bank_state"] = registry_status
        plan.at[index, "bank_status"] = registry_status
        plan.at[index, "bank_action"] = (
            "reuse" if registry_status == "production" else "skip_unavailable"
        )

    plan.to_csv(output_dir / "latest_run_plan.csv", index=False)
    if args.stop_after == "bank":
        write_project_outputs(output_dir, requested_plan=plan)
        return

    statuses = []
    for row in plan.itertuples(index=False):
        mass = float(row.mass_GeV)
        selection = str(row.selection_name)
        if str(row.bank_action) == "skip_unavailable":
            statuses.append(
                {
                    "mass_GeV": mass,
                    "selection_name": selection,
                    "status": "skipped_unavailable_bank",
                }
            )
            continue

        bank_path = Path(str(row.bank_path))
        if not bank_path.is_absolute():
            bank_path = resolve_path(repo, bank_path)
        validate_bank(bank_path, mass, selection)
        point = point_root(output_dir, mass, selection)

        moment = (
            point
            / "moments"
            / f"conditional_feature_moments_ma_{float_token(mass)}.npz"
        )
        if not moment.is_file():
            disk_guard(
                repo,
                minimum_gib=float(args.minimum_free_gib),
                warning_gib=float(args.warning_free_gib),
                operation=f"feature moments for m={mass:g}, {selection}",
            )

        started = perf_counter()
        status = "running"
        error_text = ""
        try:
            production_stop = (
                args.stop_after
                if args.stop_after in ("moments", "rangefinder", "full_domain", "selected")
                else "selected"
            )
            run_production_subprocess(
                bank_path=bank_path,
                domain_path=domain_path,
                point=point,
                observables=observables,
                stop_after=production_stop,
                workers=int(args.workers),
                chunk_size=int(args.chunk_size),
                screen_pes=int(args.screen_pseudoexperiments),
                full_pes=int(args.full_domain_pseudoexperiments),
                selected_pes=int(args.selected_pseudoexperiments),
                resume=bool(args.resume),
            )

            if args.stop_after in ("empirical", "final"):
                if any(observable != "energy" for observable in observables):
                    disk_guard(
                        repo,
                        minimum_gib=float(args.minimum_free_gib),
                        warning_gib=float(args.warning_free_gib),
                        operation=f"empirical EventCalc resampling for m={mass:g}, {selection}",
                    )
                summary = finalize_point(
                    point=point,
                    bank_path=bank_path,
                    domain_path=domain_path,
                    observables=observables,
                    selected_10k_policy=str(args.selected_10k_policy),
                    empirical_pseudoexperiments=int(
                        args.empirical_pseudoexperiments
                    ),
                    workers=int(args.workers),
                    chunk_size=int(args.chunk_size),
                    resume=bool(args.resume),
                    bank_state=str(row.bank_state),
                )
                status = str(summary["status"])
            else:
                status = f"{args.stop_after}_complete"
        except Exception as error:
            status = "failed"
            error_text = f"{type(error).__name__}: {error}"
            print(f"FAILED m={mass:g}, {selection}: {error_text}", flush=True)
            if args.fail_fast:
                raise
        finally:
            elapsed = perf_counter() - started
            statuses.append(
                {
                    "mass_GeV": mass,
                    "selection_name": selection,
                    "status": status,
                    "error": error_text,
                    "runtime_seconds": float(elapsed),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            (output_dir / "latest_run_status.json").write_text(
                json.dumps(statuses, indent=2) + "\n"
            )
            append_runtime_history(
                output_dir,
                {
                    "mass_GeV": mass,
                    "selection_name": selection,
                    "stop_after": str(args.stop_after),
                    "status": status,
                    "runtime_seconds": float(elapsed),
                },
            )
            write_project_outputs(output_dir, requested_plan=plan)

    summary = write_project_outputs(output_dir, requested_plan=plan)
    print("\n===== PROJECT SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
