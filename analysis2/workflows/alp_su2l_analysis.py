"""Unified ALP-photon versus ALP-SU(2)L analysis workflow.

The controller resolves an existing lifetime-template bank for each requested
mass and event selection, then runs the common conditional-feature analysis.
It is intentionally reuse-only in this first version: missing banks are
reported rather than generated automatically.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis2.alp_su2l_planning import (
    AnalysisConfig,
    build_analysis_plan,
    write_run_configuration,
)
from analysis2.adaptive_week8 import AdaptiveWeek8Settings
from analysis2.conditional_features import FEATURE_SUBSETS
from analysis2.lifetime_template_banks import load_template_bank
from analysis2.paths import OUTPUT_ROOT
from analysis2.workflows import float_token
from analysis2.workflows.adaptive_week8_scan import (
    run_point as run_adaptive_bank_point,
)
from analysis2.workflows.conditional_feature_pilot import (
    run_conditional_feature_point,
)


SELECTIONS = (
    "diphoton_ecal",
    "diphoton_ecal_e1gev",
)

DEFAULT_OBSERVABLES = (
    "energy",
    "energy_mean_z",
    "energy_mean_r_perp",
    "energy_mean_z_r_perp",
)

DEFAULT_EVENT_COUNTS = (
    2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 15, 18, 20, 25, 30,
)

DEFAULT_DOMAIN_PATH = (
    OUTPUT_ROOT
    / "production"
    / "week8_domains"
    / "allowed_ctau_domains.csv"
)

DEFAULT_BANK_MANIFEST = (
    OUTPUT_ROOT
    / "production"
    / "alp_su2l_analysis"
    / "existing_bank_manifest.csv"
)

DEFAULT_OUTPUT_DIR = (
    OUTPUT_ROOT
    / "production"
    / "alp_su2l_analysis"
    / "results"
)


def selection_token(selection_name: str) -> str:
    if selection_name == "diphoton_ecal":
        return "geom"
    if selection_name == "diphoton_ecal_e1gev":
        return "e1gev"
    raise ValueError(f"Unknown selection: {selection_name}")


def parse_arguments(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--masses", nargs="+", type=float, required=True)
    parser.add_argument(
        "--selections",
        nargs="+",
        choices=SELECTIONS,
        default=["diphoton_ecal"],
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        choices=tuple(FEATURE_SUBSETS),
        default=list(DEFAULT_OBSERVABLES),
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "validation", "production"),
        default="validation",
    )
    parser.add_argument(
        "--run-mode",
        choices=("automatic", "custom"),
        default="automatic",
        help=(
            "automatic plans construction/resumption of missing banks; "
            "custom requires a reusable registered bank."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=("bank", "analysis"),
        default="analysis",
        help=(
            "Stop after all requested lifetime-template banks are ready, "
            "or continue into the conditional-feature analysis."
        ),
    )
    parser.add_argument(
        "--bank-manifest",
        type=Path,
        default=DEFAULT_BANK_MANIFEST,
    )
    parser.add_argument(
        "--domain-path",
        type=Path,
        default=DEFAULT_DOMAIN_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--pseudoexperiments", type=int, default=500)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[73241, 83244],
    )
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument(
        "--event-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVENT_COUNTS),
    )
    parser.add_argument(
        "--truth-grid",
        choices=("screening", "all"),
        default="screening",
    )
    parser.add_argument("--pairs-per-interval", type=int, default=4)
    parser.add_argument("--neighbour-radius", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed points and existing feature moments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the analysis plan without running calculations.",
    )
    return parser.parse_args(argv)


def resolve_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def load_bank_manifest(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "mass_GeV",
        "selection_name",
        "status",
        "bank_path",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Bank manifest is missing columns: {sorted(missing)}"
        )
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
        & (table["selection_name"].astype(str) == selection_name)
    )
    matches = table.loc[mask]

    if len(matches) == 0:
        raise FileNotFoundError(
            "No canonical bank is registered for "
            f"m_a={mass_gev:g} GeV, selection={selection_name}. "
            "Automatic construction of missing banks is not enabled yet."
        )

    if len(matches) != 1:
        raise ValueError(
            "Bank manifest contains multiple entries for "
            f"m_a={mass_gev:g} GeV, selection={selection_name}."
        )

    return matches.iloc[0].to_dict()


def point_output_dir(
    output_dir: Path,
    mass_gev: float,
    selection_name: str,
) -> Path:
    return (
        output_dir
        / "per_mass"
        / f"ma_{float_token(mass_gev)}"
        / selection_token(selection_name)
        / "conditional_features"
    )


def validate_bank(
    bank_path: Path,
    mass_gev: float,
    selection_name: str,
):
    bank = load_template_bank(bank_path)

    if not np.isclose(
        float(bank.mass_gev),
        float(mass_gev),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"Bank mass {bank.mass_gev:g} does not match "
            f"requested mass {mass_gev:g}."
        )

    if str(bank.selection_name) != selection_name:
        raise ValueError(
            f"Bank selection {bank.selection_name} does not match "
            f"requested selection {selection_name}."
        )

    return bank



def adaptive_bank_output_root(config: AnalysisConfig) -> Path:
    """Root consumed by adaptive_week8_scan.run_point()."""
    return Path(config.output_dir) / "bank_workspaces"


def generated_bank_from_state(
    *,
    repo: Path,
    state_path: Path,
    mass_gev: float,
    selection_name: str,
) -> tuple[Path, str]:
    """Resolve the canonical generated bank recorded by adaptive state."""

    if not state_path.is_file():
        raise RuntimeError(
            f"Adaptive bank stage produced no state file: {state_path}"
        )

    state = json.loads(state_path.read_text())

    if state.get("status") != "bank_complete":
        raise RuntimeError(
            "Adaptive bank stage did not reach bank_complete: "
            f"{state.get('status')!r}"
        )

    raw_bank_dir = state.get("bank_dir")
    if not raw_bank_dir:
        raise RuntimeError(
            "Completed adaptive state does not record bank_dir."
        )

    bank_dir = resolve_path(repo, raw_bank_dir)
    bank_path = (
        bank_dir
        / "template_banks"
        / f"template_bank_ma_{float_token(mass_gev)}.npz"
    )

    if not bank_path.is_file():
        raise FileNotFoundError(
            "Adaptive state records a completed bank directory, "
            f"but the bank file is absent: {bank_path}"
        )

    validate_bank(
        bank_path,
        mass_gev,
        selection_name,
    )

    return bank_path, str(
        state.get("bank_status", "generated")
    )



CLEAN_ADAPTIVE_BANK_STATUSES = {
    "lifetime_grid_converged",
    "fine_binning_converged",
}


def registry_status_from_adaptive_status(
    adaptive_status: str,
) -> str:
    """Map adaptive bank completion to scientific registry readiness."""

    if adaptive_status in CLEAN_ADAPTIVE_BANK_STATUSES:
        return "production"

    # A bank may exist physically while having stopped at a lifetime-grid,
    # distance-stability or binning-refinement limit.  Preserve it for safe
    # resumption, but never silently promote it to physics-production use.
    return "incomplete"


def portable_registry_path(repo: Path, path: Path) -> str:
    """Prefer repository-relative paths in the persistent bank registry."""

    resolved_repo = Path(repo).resolve()
    resolved_path = Path(path).resolve()

    try:
        return str(resolved_path.relative_to(resolved_repo))
    except ValueError:
        return str(resolved_path)


def persist_generated_bank_record(
    *,
    manifest_path: Path,
    repo: Path,
    mass_gev: float,
    selection_name: str,
    bank_path: Path,
    adaptive_status: str,
) -> str:
    """Persist a controller-built bank for discovery on later runs."""

    registry_status = registry_status_from_adaptive_status(
        adaptive_status
    )

    table = load_bank_manifest(manifest_path)

    record = {
        "mass_GeV": float(mass_gev),
        "selection_name": str(selection_name),
        "status": registry_status,
        "bank_path": portable_registry_path(
            repo,
            bank_path,
        ),
        "note": (
            "Automatically generated by the unified ALP-SU2L controller. "
            f"Adaptive bank status: {adaptive_status}."
        ),
    }

    mask = (
        np.isclose(
            table["mass_GeV"].to_numpy(dtype=float),
            float(mass_gev),
            rtol=0.0,
            atol=1.0e-12,
        )
        & (
            table["selection_name"].astype(str)
            == str(selection_name)
        )
    )

    if mask.any():
        if int(mask.sum()) != 1:
            raise ValueError(
                "Cannot persist generated bank because the registry "
                "contains duplicate mass/selection records."
            )

        for key, value in record.items():
            table.loc[mask, key] = value
    else:
        record_table = pd.DataFrame([record])
        if table.empty:
            table = record_table
        else:
            table = pd.concat(
                [table, record_table],
                ignore_index=True,
            )

    table = table.sort_values(
        ["mass_GeV", "selection_name"],
        ignore_index=True,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(manifest_path, index=False)

    return registry_status



def build_or_resume_bank(
    *,
    config: AnalysisConfig,
    repo: Path,
    domains: pd.DataFrame,
    mass_gev: float,
    selection_name: str,
) -> tuple[Path, str]:
    """Build or safely resume one adaptive lifetime-template bank."""

    output_root = adaptive_bank_output_root(config)

    # AdaptiveWeek8Settings is the canonical settings dataclass used by
    # adaptive_week8_scan itself.  Do not reproduce these numerical
    # defaults inside the unified controller.
    settings = AdaptiveWeek8Settings()

    run_adaptive_bank_point(
        mass_gev=float(mass_gev),
        selection_name=str(selection_name),
        profile=str(config.profile),
        domain_path=Path(config.domain_path),
        domains=domains,
        output_dir=output_root,
        settings=settings,
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


def summary_rows(summary: dict, bank_status: str) -> list[dict]:
    rows = []
    thresholds = summary.get("provisional_thresholds", {})
    minima = summary.get("distance_minima", {})

    for observable in summary.get("observables", thresholds.keys()):
        distance = minima.get(observable, {})
        rows.append(
            {
                "mass_GeV": float(summary["mass_GeV"]),
                "selection_name": str(summary["selection_name"]),
                "bank_status": bank_status,
                "observable": observable,
                "provisional_N90": thresholds.get(observable),
                "minimum_H2": distance.get("minimum_H2"),
                "photon_ctau_at_minimum_m": distance.get(
                    "photon_ctau_m"
                ),
                "su2_ctau_at_minimum_m": distance.get(
                    "su2_ctau_m"
                ),
                "pseudoexperiments_per_truth_and_seed": summary.get(
                    "pseudoexperiments_per_truth_and_seed"
                ),
                "truth_grid": summary.get("truth_grid"),
            }
        )

    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    repo = Path.cwd().resolve()
    if not (repo / "analysis2").is_dir():
        raise SystemExit(
            "Run this workflow from the EventCalc-SHiP repository root."
        )

    manifest_path = resolve_path(repo, args.bank_manifest)
    domain_path = resolve_path(repo, args.domain_path)
    output_dir = resolve_path(repo, args.output_dir)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Bank manifest not found: {manifest_path}"
        )
    if not domain_path.is_file():
        raise FileNotFoundError(
            f"Week-8 domain table not found: {domain_path}"
        )

    manifest = load_bank_manifest(manifest_path)

    masses = tuple(dict.fromkeys(float(x) for x in args.masses))
    selections = tuple(dict.fromkeys(args.selections))
    observables = tuple(dict.fromkeys(args.observables))

    config = AnalysisConfig(
        masses=masses,
        selections=selections,
        observables=observables,
        profile=str(args.profile),
        workers=int(args.workers),
        run_mode=str(args.run_mode),
        output_dir=output_dir,
        domain_path=domain_path,
        bank_manifest=manifest_path,
        resume=bool(args.resume),
    )

    plan = build_analysis_plan(
        config=config,
        manifest=manifest,
        repo=repo,
    )

    # Compatibility aliases for the existing execution layer.
    # These disappear once the executor consumes the planner schema directly.
    plan["bank_status"] = plan["bank_state"]
    plan["output_dir"] = plan["result_dir"]

    print("===== ALP-SU2L ANALYSIS PLAN =====")
    print(plan.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_dir / "latest_run_plan.csv", index=False)
    write_run_configuration(config, output_dir)

    if args.dry_run:
        print()
        print("DRY RUN: no EventCalc or pseudoexperiments were launched.")
        return

    domains = pd.read_csv(domain_path)

    if "adaptive_bank_status" not in plan.columns:
        plan["adaptive_bank_status"] = ""

    unusable_generated_banks = []

    for index, row in plan.iterrows():
        action = str(row["bank_action"])

        if action == "reuse":
            continue

        if action == "requires_bank":
            raise RuntimeError(
                "Custom mode requires a reusable registered bank for "
                f"m_a={float(row['mass_GeV']):g} GeV, "
                f"selection={row['selection_name']}."
            )

        if action not in ("build", "build_or_resume"):
            raise RuntimeError(
                f"Unknown bank action: {action}"
            )

        mass_gev = float(row["mass_GeV"])
        selection_name = str(row["selection_name"])

        print()
        print("=" * 78)
        print(
            f"BANK {action.upper()}: "
            f"m_a={mass_gev:g} GeV, "
            f"selection={selection_name}"
        )
        print("=" * 78)

        bank_path, generated_status = build_or_resume_bank(
            config=config,
            repo=repo,
            domains=domains,
            mass_gev=mass_gev,
            selection_name=selection_name,
        )

        registry_status = persist_generated_bank_record(
            manifest_path=manifest_path,
            repo=repo,
            mass_gev=mass_gev,
            selection_name=selection_name,
            bank_path=bank_path,
            adaptive_status=generated_status,
        )

        plan.at[index, "bank_path"] = str(bank_path)
        plan.at[index, "bank_exists"] = True
        plan.at[index, "adaptive_bank_status"] = generated_status
        plan.at[index, "bank_state"] = registry_status
        plan.at[index, "bank_status"] = registry_status

        if registry_status == "production":
            plan.at[index, "bank_action"] = "reuse"
        else:
            plan.at[index, "bank_action"] = "build_or_resume"
            unusable_generated_banks.append(
                {
                    "mass_GeV": mass_gev,
                    "selection_name": selection_name,
                    "adaptive_bank_status": generated_status,
                    "bank_path": str(bank_path),
                }
            )

    # Re-save the plan after preparation so the file records the exact
    # artifacts actually consumed by the subsequent analysis.
    plan.to_csv(
        output_dir / "latest_run_plan.csv",
        index=False,
    )

    if args.stop_after == "bank":
        print()
        print("===== BANK PREPARATION COMPLETE =====")
        print(
            plan[
                [
                    "mass_GeV",
                    "selection_name",
                    "bank_state",
                    "adaptive_bank_status",
                    "bank_action",
                    "bank_path",
                ]
            ].to_string(index=False)
        )
        print()
        print(
            "Stopped after requested stage: bank. "
            "No conditional-feature pseudoexperiments were launched."
        )
        return

    if unusable_generated_banks:
        failed = pd.DataFrame(unusable_generated_banks)
        print()
        print("===== NON-PRODUCTION GENERATED BANKS =====")
        print(failed.to_string(index=False))

        raise RuntimeError(
            "One or more generated banks did not satisfy the adaptive "
            "production convergence criteria. They were persisted as "
            "'incomplete' and will not be used for conditional-feature "
            "physics results."
        )

    all_rows = []

    for row in plan.itertuples(index=False):
        mass_gev = float(row.mass_GeV)
        selection_name = str(row.selection_name)
        bank_status = str(row.bank_status)
        bank_path = Path(row.bank_path)
        result_dir = Path(row.output_dir)

        bank = validate_bank(
            bank_path,
            mass_gev,
            selection_name,
        )
        del bank

        token = float_token(mass_gev)
        summary_path = (
            result_dir
            / f"conditional_feature_pilot_summary_ma_{token}.json"
        )

        if summary_path.is_file():
            if not args.resume:
                raise FileExistsError(
                    "Completed result already exists. "
                    "Use --resume to reuse it: "
                    f"{summary_path}"
                )
            print(
                f"REUSE COMPLETE: m_a={mass_gev:g} GeV, "
                f"selection={selection_name}"
            )
            summary = json.loads(summary_path.read_text())
        else:
            moment_path = (
                result_dir
                / f"conditional_feature_moments_ma_{token}.npz"
            )
            reuse_moments = bool(
                args.resume and moment_path.is_file()
            )

            print()
            print("=" * 78)
            print(
                f"RUN: m_a={mass_gev:g} GeV, "
                f"selection={selection_name}"
            )
            print("=" * 78)

            summary = run_conditional_feature_point(
                bank_path=bank_path,
                output_dir=result_dir,
                domain_path=domain_path,
                pseudoexperiments=int(args.pseudoexperiments),
                seeds=tuple(int(x) for x in args.seeds),
                workers=int(args.workers),
                chunk_size=int(args.chunk_size),
                event_counts=tuple(
                    int(x) for x in args.event_counts
                ),
                observables=observables,
                pairs_per_interval=int(args.pairs_per_interval),
                truth_grid=str(args.truth_grid),
                neighbour_radius=int(args.neighbour_radius),
                reuse_moments=reuse_moments,
            )

            if summary is None:
                raise RuntimeError(
                    "Conditional-feature workflow returned no summary."
                )

        all_rows.extend(
            summary_rows(summary, bank_status)
        )

    summary_table = pd.DataFrame(all_rows)
    summary_table.to_csv(
        output_dir / "analysis_summary.csv",
        index=False,
    )

    print()
    print("===== ANALYSIS SUMMARY =====")
    print(summary_table.to_string(index=False))
    print()
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
