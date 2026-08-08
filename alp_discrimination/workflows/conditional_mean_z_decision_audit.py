"""Decision-relevant omitted-truth audit with automatic 5k promotion.

This stage consumes:
  1. a completed generic full-domain 2k screen; and
  2. a completed generic selected-truth 5k run.

Why this exists
---------------
The original generic selected-5k audit compared every omitted 2k lower
confidence bound with the complete selected-5k conservative curve. At large
N, where both accuracies are already far above 90%, that can promote harmless
truths simply because a 2k lower bound cannot match a near-unity 5k point
estimate.

For N90, the necessary omitted-truth statement is instead:

  For every tested N at or above the candidate persistent threshold,
  every omitted truth/seed has true correct-classification probability
  at least the target (normally 90%).

This script applies simultaneous one-sided Clopper-Pearson bounds to exactly
that decision-relevant statement. It then evaluates only genuinely failing
truths at 5k, merges them with the selected set, recomputes N90, and repeats
until the audit passes.

It reuses the existing template bank and conditional-z moments. It does not
regenerate EventCalc proposals or lifetime banks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-domain-summary", type=Path, required=True)
    parser.add_argument("--selected-5k-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-script-dir",
        type=Path,
        default=Path.home() / "Downloads",
    )
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--audit-global-alpha", type=float, default=0.01)
    parser.add_argument("--maximum-promotion-rounds", type=int, default=5)
    parser.add_argument("--restart-checkpoint", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()



def load_download_module(script_dir: Path, module_name: str, filename: str):
    """Resolve validated helpers from analysis2, not Downloads."""
    del script_dir, filename
    if module_name == "run_week8_mean_z_selected_5k_generic":
        from alp_discrimination.workflows import conditional_mean_z_selected
        return conditional_mean_z_selected
    if module_name == "run_week8_energy_plus_mean_z_conditional_pilot":
        from alp_discrimination import conditional_mean_z
        return conditional_mean_z
    raise ValueError(f"Unknown conditional-mean-z helper: {module_name}")

def conservative_curve(detailed: pd.DataFrame) -> pd.DataFrame:
    return (
        detailed.groupby("number_of_events", as_index=False)["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values("number_of_events", ignore_index=True)
    )


def persistent_threshold(
    curve: pd.DataFrame,
    target: float,
) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix_minimum = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix_minimum >= float(target))
    return None if len(passing) == 0 else int(counts[passing[0]])


def clopper_pearson_lower(
    successes: np.ndarray,
    trials: np.ndarray,
    alpha_each: float,
) -> np.ndarray:
    successes = np.asarray(successes, dtype=int)
    trials = np.asarray(trials, dtype=int)
    lower = np.zeros(successes.shape, dtype=float)
    positive = successes > 0
    lower[positive] = beta.ppf(
        alpha_each,
        successes[positive],
        trials[positive] - successes[positive] + 1,
    )
    return lower


def decision_relevant_audit(
    *,
    detailed_2k: pd.DataFrame,
    selected_keys: set[tuple[str, int]],
    candidate_threshold: int,
    tested_event_counts: np.ndarray,
    target_accuracy: float,
    global_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    relevant_counts = {
        int(value)
        for value in tested_event_counts
        if int(value) >= int(candidate_threshold)
    }
    if not relevant_counts:
        raise ValueError(
            "No tested event counts exist at or above the candidate threshold."
        )

    frame = detailed_2k[
        detailed_2k["number_of_events"].astype(int).isin(relevant_counts)
    ].copy()
    keys = list(
        zip(
            frame["truth_model"].astype(str),
            frame["truth_lifetime_index"].astype(int),
        )
    )
    omitted = frame[
        np.asarray([key not in selected_keys for key in keys], dtype=bool)
    ].copy()

    promotion_columns = [
        "truth_model",
        "truth_lifetime_index",
        "truth_ctau_m",
        "minimum_target_margin",
        "number_of_failing_rows",
        "minimum_correct_fraction",
        "worst_seed",
        "worst_number_of_events",
    ]

    if omitted.empty:
        return omitted, pd.DataFrame(columns=promotion_columns), {
            "candidate_threshold": int(candidate_threshold),
            "target_accuracy": float(target_accuracy),
            "global_alpha": float(global_alpha),
            "number_of_relevant_event_counts": int(len(relevant_counts)),
            "number_of_comparisons": 0,
            "alpha_per_comparison": None,
            "minimum_target_margin": None,
            "number_of_failing_rows": 0,
            "number_of_truths_requiring_promotion": 0,
        }

    comparisons = int(len(omitted))
    alpha_each = float(global_alpha) / comparisons

    omitted["successes"] = np.rint(
        omitted["correct_fraction"]
        * omitted["number_of_pseudoexperiments"]
    ).astype(int)
    omitted["audit_lower_bound"] = clopper_pearson_lower(
        omitted["successes"].to_numpy(dtype=int),
        omitted["number_of_pseudoexperiments"].to_numpy(dtype=int),
        alpha_each,
    )
    omitted["target_accuracy"] = float(target_accuracy)
    omitted["target_margin"] = (
        omitted["audit_lower_bound"] - float(target_accuracy)
    )
    omitted["requires_promotion"] = omitted["target_margin"] < 0.0

    failing = omitted[omitted["requires_promotion"]].copy()
    if failing.empty:
        promotions = pd.DataFrame(columns=promotion_columns)
    else:
        worst_rows = (
            failing.sort_values("target_margin")
            .drop_duplicates(
                ["truth_model", "truth_lifetime_index"],
                keep="first",
            )
            .set_index(["truth_model", "truth_lifetime_index"])
        )
        promotions = (
            failing.groupby(
                ["truth_model", "truth_lifetime_index", "truth_ctau_m"],
                as_index=False,
            )
            .agg(
                minimum_target_margin=("target_margin", "min"),
                number_of_failing_rows=("requires_promotion", "sum"),
                minimum_correct_fraction=("correct_fraction", "min"),
            )
        )
        promotions["worst_seed"] = [
            int(
                worst_rows.loc[
                    (row.truth_model, row.truth_lifetime_index),
                    "seed",
                ]
            )
            for row in promotions.itertuples()
        ]
        promotions["worst_number_of_events"] = [
            int(
                worst_rows.loc[
                    (row.truth_model, row.truth_lifetime_index),
                    "number_of_events",
                ]
            )
            for row in promotions.itertuples()
        ]
        promotions.sort_values(
            [
                "minimum_target_margin",
                "truth_model",
                "truth_lifetime_index",
            ],
            inplace=True,
            ignore_index=True,
        )

    summary = {
        "candidate_threshold": int(candidate_threshold),
        "target_accuracy": float(target_accuracy),
        "global_alpha": float(global_alpha),
        "number_of_relevant_event_counts": int(len(relevant_counts)),
        "minimum_relevant_event_count": int(min(relevant_counts)),
        "maximum_relevant_event_count": int(max(relevant_counts)),
        "number_of_comparisons": int(comparisons),
        "alpha_per_comparison": float(alpha_each),
        "minimum_target_margin": float(omitted["target_margin"].min()),
        "number_of_failing_rows": int(
            omitted["requires_promotion"].sum()
        ),
        "number_of_truths_requiring_promotion": int(len(promotions)),
    }
    return omitted, promotions, summary


def load_parts(paths: list[Path]) -> pd.DataFrame:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} promoted-truth checkpoint files."
        )
    if not paths:
        return pd.DataFrame()
    return pd.concat(
        (pd.read_csv(path) for path in paths),
        ignore_index=True,
    )


def main() -> None:
    args = parse_args()
    started = perf_counter()

    repo = Path.cwd().resolve()
    if not (repo / "alp_discrimination").is_dir():
        raise SystemExit(
            "Run this script from the EventCalc-SHiP repository root."
        )
    if not (0.0 < args.target_accuracy < 1.0):
        raise ValueError("--target-accuracy must lie between zero and one.")
    if not (0.0 < args.audit_global_alpha < 1.0):
        raise ValueError(
            "--audit-global-alpha must lie between zero and one."
        )

    helper_dir = args.pilot_script_dir.expanduser().resolve()
    selected_helper = load_download_module(
        helper_dir,
        "run_week8_mean_z_selected_5k_generic",
        "run_week8_mean_z_selected_5k_generic.py",
    )
    pilot = load_download_module(
        helper_dir,
        "run_week8_energy_plus_mean_z_conditional_pilot",
        "run_week8_energy_plus_mean_z_conditional_pilot.py",
    )

    full_summary_path = resolve(repo, args.full_domain_summary)
    full_dir = full_summary_path.parent
    selected_dir = resolve(repo, args.selected_5k_dir)
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_summary = json.loads(full_summary_path.read_text())
    bank_path = resolve(repo, Path(full_summary["bank_path"]))
    moments_path = resolve(
        repo,
        Path(full_summary["conditional_moments_path"]),
    )

    bank = pilot.load_template_bank(bank_path)
    moments = pilot.load_conditional_moments(moments_path)
    pilot.validate_conditional_moments(moments, bank)

    from alp_discrimination.workflows import float_token

    token = float_token(float(bank.mass_gev))
    summary_output = (
        output_dir
        / f"decision_audit_summary_ma_{token}.json"
    )
    if summary_output.exists() and not args.restart_checkpoint:
        raise FileExistsError(
            "This decision-audit result is already complete. "
            f"Preserve it and use a new output directory: {summary_output}"
        )

    detailed_2k_path = (
        full_dir
        / f"full_domain_2k_detailed_accuracy_ma_{token}.csv"
    )
    initial_selected_path = (
        selected_dir
        / f"selected_5k_truths_ma_{token}.csv"
    )
    initial_detailed_path = (
        selected_dir
        / f"selected_5k_detailed_accuracy_ma_{token}.csv"
    )
    for path in (
        detailed_2k_path,
        initial_selected_path,
        initial_detailed_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    detailed_2k = pd.read_csv(detailed_2k_path)
    initial_selected = pd.read_csv(initial_selected_path)
    initial_detailed = pd.read_csv(initial_detailed_path)

    event_counts = np.asarray(
        sorted(
            initial_detailed["number_of_events"]
            .astype(int)
            .unique()
        ),
        dtype=int,
    )
    seeds = np.asarray(
        sorted(initial_detailed["seed"].astype(int).unique()),
        dtype=int,
    )

    initial_keys = {
        (str(row.truth_model), int(row.truth_lifetime_index))
        for row in initial_selected.itertuples()
    }
    selected_keys = set(initial_keys)
    current_detailed = initial_detailed.copy()

    checkpoint_dir = output_dir / "promoted_5k_truth_parts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.restart_checkpoint:
        for path in checkpoint_dir.glob("*.csv"):
            path.unlink()

    common = {
        "mass_gev": float(bank.mass_gev),
        "photon_probabilities": np.asarray(
            bank.photon_probabilities,
            dtype=float,
        ),
        "su2_probabilities": np.asarray(
            bank.su2_probabilities,
            dtype=float,
        ),
        "photon_conditional_mean_z": moments[
            "photon_mean_z_by_energy_bin_m"
        ],
        "photon_conditional_variance_z": moments[
            "photon_variance_z_by_energy_bin_m2"
        ],
        "su2_conditional_mean_z": moments[
            "su2_mean_z_by_energy_bin_m"
        ],
        "su2_conditional_variance_z": moments[
            "su2_variance_z_by_energy_bin_m2"
        ],
        "event_counts": event_counts,
        "number_of_pseudoexperiments": 5000,
        "chunk_size": int(args.chunk_size),
    }

    audit_history: list[dict] = []
    newly_promoted_keys: set[tuple[str, int]] = set()

    for round_index in range(
        int(args.maximum_promotion_rounds) + 1
    ):
        curve = conservative_curve(current_detailed)
        threshold = persistent_threshold(
            curve,
            args.target_accuracy,
        )
        if threshold is None:
            raise RuntimeError(
                "The merged 5k curve does not reach the target "
                "persistently."
            )

        audit_rows, promotions, audit_summary = (
            decision_relevant_audit(
                detailed_2k=detailed_2k,
                selected_keys=selected_keys,
                candidate_threshold=threshold,
                tested_event_counts=event_counts,
                target_accuracy=float(args.target_accuracy),
                global_alpha=float(args.audit_global_alpha),
            )
        )
        audit_rows.to_csv(
            output_dir
            / (
                f"decision_audit_round_{round_index:02d}"
                f"_rows_ma_{token}.csv"
            ),
            index=False,
        )
        promotions.to_csv(
            output_dir
            / (
                f"decision_audit_round_{round_index:02d}"
                f"_promotions_ma_{token}.csv"
            ),
            index=False,
        )

        audit_history.append(
            {
                "round": int(round_index),
                "candidate_threshold": int(threshold),
                "number_of_current_selected_truths": int(
                    len(selected_keys)
                ),
                **audit_summary,
            }
        )

        print(
            f"AUDIT ROUND {round_index}: "
            f"N90={threshold}, "
            f"selected={len(selected_keys)}, "
            f"promotions={len(promotions)}",
            flush=True,
        )

        if promotions.empty:
            final_audit_rows = audit_rows
            break

        if round_index >= int(args.maximum_promotion_rounds):
            raise RuntimeError(
                "Maximum promotion rounds reached before the "
                "audit passed."
            )

        jobs = []
        part_paths: list[Path] = []

        for row in promotions.itertuples():
            model = str(row.truth_model)
            index = int(row.truth_lifetime_index)
            key = (model, index)
            if key in selected_keys:
                continue

            part_path = (
                checkpoint_dir / f"{model}_{index:04d}.csv"
            )
            part_paths.append(part_path)

            if part_path.is_file():
                continue

            probabilities = np.asarray(
                getattr(bank, f"{model}_probabilities"),
                dtype=float,
            )
            lifetimes = np.asarray(
                getattr(bank, f"{model}_ctau_m"),
                dtype=float,
            )
            means = moments[
                f"{model}_mean_z_by_energy_bin_m"
            ]
            variances = moments[
                f"{model}_variance_z_by_energy_bin_m2"
            ]

            jobs.append(
                (
                    model,
                    index,
                    float(lifetimes[index]),
                    probabilities[index],
                    means[index],
                    variances[index],
                    part_path,
                )
            )

        if jobs:
            print(
                f"RUNNING {len(jobs)} promoted truths at 5k "
                f"with {args.workers} workers.",
                flush=True,
            )

            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=selected_helper.initialize_worker,
                initargs=(str(helper_dir), common),
            ) as executor:
                futures = {}
                for job in jobs:
                    (
                        model,
                        index,
                        ctau,
                        probabilities,
                        mean_z,
                        variance_z,
                        part_path,
                    ) = job

                    future = executor.submit(
                        selected_helper.simulate_truth_all_seeds,
                        model,
                        index,
                        ctau,
                        probabilities,
                        mean_z,
                        variance_z,
                        seeds.tolist(),
                    )
                    futures[future] = (
                        model,
                        index,
                        part_path,
                    )

                for completed, future in enumerate(
                    as_completed(futures),
                    start=1,
                ):
                    model, index, part_path = futures[future]
                    frame = future.result()

                    temporary = part_path.with_suffix(
                        part_path.suffix + ".tmp"
                    )
                    frame.to_csv(temporary, index=False)
                    temporary.replace(part_path)

                    print(
                        f"COMPLETED PROMOTION "
                        f"{model:6s} index={index:3d} "
                        f"({completed}/{len(jobs)})",
                        flush=True,
                    )

        promoted_frame = load_parts(part_paths)
        if not promoted_frame.empty:
            current_detailed = pd.concat(
                [current_detailed, promoted_frame],
                ignore_index=True,
            )
            current_detailed.drop_duplicates(
                subset=[
                    "truth_model",
                    "truth_lifetime_index",
                    "seed",
                    "number_of_events",
                    "observable",
                ],
                keep="last",
                inplace=True,
            )

        for row in promotions.itertuples():
            key = (
                str(row.truth_model),
                int(row.truth_lifetime_index),
            )
            selected_keys.add(key)
            newly_promoted_keys.add(key)

    else:
        raise RuntimeError("Internal promotion-loop failure.")

    final_curve = conservative_curve(current_detailed)
    final_threshold = persistent_threshold(
        final_curve,
        args.target_accuracy,
    )

    current_detailed.sort_values(
        [
            "seed",
            "number_of_events",
            "correct_fraction",
            "truth_model",
            "truth_lifetime_index",
        ],
        inplace=True,
        ignore_index=True,
    )
    current_detailed.to_csv(
        output_dir
        / f"decision_audit_final_detailed_5k_ma_{token}.csv",
        index=False,
    )
    final_curve.to_csv(
        output_dir
        / (
            f"decision_audit_final_conservative_curve"
            f"_ma_{token}.csv"
        ),
        index=False,
    )
    final_audit_rows.to_csv(
        output_dir
        / f"decision_audit_final_rows_ma_{token}.csv",
        index=False,
    )

    selected_rows = []
    for model, index in sorted(selected_keys):
        lifetimes = np.asarray(
            getattr(bank, f"{model}_ctau_m"),
            dtype=float,
        )
        intervals = np.asarray(
            getattr(bank, f"{model}_interval_index"),
            dtype=int,
        )
        selected_rows.append(
            {
                "truth_model": model,
                "truth_lifetime_index": int(index),
                "truth_interval_index": int(intervals[index]),
                "truth_ctau_m": float(lifetimes[index]),
                "selection_reason": (
                    "initial_full_domain_selection"
                    if (model, index) in initial_keys
                    else "decision_relevant_audit_promotion"
                ),
            }
        )
    final_selected = pd.DataFrame(selected_rows)
    final_selected.to_csv(
        output_dir
        / f"decision_audit_final_selected_truths_ma_{token}.csv",
        index=False,
    )

    threshold_rows = []
    for seed, subset in current_detailed.groupby("seed"):
        threshold_rows.append(
            {
                "seed": int(seed),
                "persistent_threshold": persistent_threshold(
                    conservative_curve(subset),
                    args.target_accuracy,
                ),
            }
        )
    threshold_by_seed = pd.DataFrame(threshold_rows)
    threshold_by_seed.to_csv(
        output_dir
        / f"decision_audit_final_threshold_by_seed_ma_{token}.csv",
        index=False,
    )

    limiting = pd.DataFrame()
    if final_threshold is not None:
        limiting = (
            current_detailed[
                current_detailed["number_of_events"]
                == int(final_threshold)
            ]
            .nsmallest(20, "correct_fraction")
            .copy()
        )
    limiting.to_csv(
        output_dir
        / f"decision_audit_final_limiting_points_ma_{token}.csv",
        index=False,
    )

    initial_curve = conservative_curve(initial_detailed)
    comparison = initial_curve.rename(
        columns={
            "worst_case_accuracy": "initial_selected_5k"
        }
    ).merge(
        final_curve.rename(
            columns={
                "worst_case_accuracy": "final_decision_audited_5k"
            }
        ),
        on="number_of_events",
        how="inner",
    )
    comparison["signed_difference_final_minus_initial"] = (
        comparison["final_decision_audited_5k"]
        - comparison["initial_selected_5k"]
    )
    comparison["absolute_difference"] = np.abs(
        comparison["signed_difference_final_minus_initial"]
    )
    comparison.to_csv(
        output_dir
        / f"decision_audit_initial_vs_final_curve_ma_{token}.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.plot(
        comparison["number_of_events"],
        comparison["initial_selected_5k"],
        marker="o",
        markersize=3,
        label="Initial selected set, 5k",
    )
    axis.plot(
        comparison["number_of_events"],
        comparison["final_decision_audited_5k"],
        marker="s",
        markersize=3,
        label="Decision-audited set, 5k",
    )
    axis.axhline(
        args.target_accuracy,
        linestyle="--",
        linewidth=1.2,
        label="90% target",
    )
    if final_threshold is not None:
        axis.axvline(
            final_threshold,
            linestyle=":",
            linewidth=1.2,
        )
    axis.set_xlabel("Observed ALP decays, $N$")
    axis.set_ylabel(
        "Worst-case correct-classification probability"
    )
    axis.set_title(
        "Decision-relevant omitted-truth audit, "
        f"$m_a={bank.mass_gev:g}$ GeV"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir
        / (
            f"decision_audit_initial_vs_final_accuracy"
            f"_ma_{token}.pdf"
        )
    )
    plt.close(figure)

    elapsed = perf_counter() - started
    result = {
        "status": (
            "decision_relevant_omitted_truth_audit_"
            "with_automatic_5k_promotion"
        ),
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "target_accuracy": float(args.target_accuracy),
        "full_domain_summary_path": str(full_summary_path),
        "selected_5k_dir": str(selected_dir),
        "bank_path": str(bank_path),
        "conditional_moments_path": str(moments_path),
        "event_counts": [
            int(value) for value in event_counts
        ],
        "seeds": [int(value) for value in seeds],
        "persistent_thresholds": {
            "initial_selected_5k": persistent_threshold(
                initial_curve,
                args.target_accuracy,
            ),
            "final_decision_audited_5k": final_threshold,
        },
        "number_of_truths": {
            "initial_selected": int(len(initial_keys)),
            "newly_promoted": int(len(newly_promoted_keys)),
            "final_selected": int(len(selected_keys)),
            "total_full_domain": int(
                detailed_2k[
                    ["truth_model", "truth_lifetime_index"]
                ]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "audit_rounds": audit_history,
        "final_audit": audit_history[-1],
        "curve_change": {
            "mean_absolute_difference": float(
                comparison["absolute_difference"].mean()
            ),
            "maximum_absolute_difference": float(
                comparison["absolute_difference"].max()
            ),
            "event_at_maximum_difference": int(
                comparison.loc[
                    comparison["absolute_difference"].idxmax(),
                    "number_of_events",
                ]
            ),
        },
        "runtime": {
            "elapsed_seconds_this_invocation": float(elapsed),
        },
        "next_action": (
            "If the audit passes, use the final selected set for a "
            "uniform 10k crossing validation when the 5k threshold "
            "is numerically marginal. Do not run all originally "
            "listed 176 promotions."
        ),
    }

    summary_output.write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
