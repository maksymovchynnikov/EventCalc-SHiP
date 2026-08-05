"""Generic uniform-selected 10k validation with decision-relevant audit.

This stage starts from a completed decision-audited 5k selected set. It:

1. reevaluates every final selected truth with 10,000 pseudoexperiments
   per truth and seed;
2. uses the exact same tested N grid as the audited 5k stage;
3. compares the 5k and 10k conservative curves;
4. applies the decision-relevant omitted-truth audit:
      for every tested N >= candidate N90, every omitted 2k truth/seed
      must have a simultaneous one-sided lower bound >= the target;
5. automatically evaluates only genuinely failing omitted truths at 10k
   and repeats the audit until it passes;
6. writes the final N90, seed thresholds, limiting points, curves, and
   runtime/provenance summary.

The script reuses the existing template bank and conditional-z moments.
It does not regenerate EventCalc proposals, energy templates, or lifetime
banks. Use a fresh output directory. Re-running the same command resumes
truth-level checkpoints.
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
    parser.add_argument("--decision-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-script-dir",
        type=Path,
        default=Path.home() / "Downloads",
    )
    parser.add_argument("--pseudoexperiments", type=int, default=10000)
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
        from analysis2.workflows import conditional_mean_z_selected
        return conditional_mean_z_selected
    if module_name == "run_week8_energy_plus_mean_z_conditional_pilot":
        from analysis2 import conditional_mean_z
        return conditional_mean_z
    raise ValueError(f"Unknown conditional-mean-z helper: {module_name}")

def conservative_curve(detailed: pd.DataFrame) -> pd.DataFrame:
    return (
        detailed.groupby("number_of_events", as_index=False)["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values("number_of_events", ignore_index=True)
    )


def persistent_threshold(curve: pd.DataFrame, target: float) -> int | None:
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
        raise ValueError("No tested N exists at or above candidate N90.")

    frame = detailed_2k[
        detailed_2k["number_of_events"].astype(int).isin(relevant_counts)
    ].copy()
    if "observable" in frame.columns:
        frame = frame[
            frame["observable"].astype(str) == "conditional_combined"
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
            ["minimum_target_margin", "truth_model", "truth_lifetime_index"],
            inplace=True,
            ignore_index=True,
        )

    return omitted, promotions, {
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


def load_checkpoint_parts(paths: list[Path]) -> pd.DataFrame:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} truth checkpoint files; rerun."
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
    if not (repo / "analysis2").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")
    if args.pseudoexperiments != 10000:
        raise ValueError(
            "This validation stage is intentionally fixed at 10,000 PEs."
        )
    if not (0.0 < args.target_accuracy < 1.0):
        raise ValueError("--target-accuracy must lie between 0 and 1.")
    if not (0.0 < args.audit_global_alpha < 1.0):
        raise ValueError("--audit-global-alpha must lie between 0 and 1.")

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
    audit_dir = resolve(repo, args.decision_audit_dir)
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_summary = json.loads(full_summary_path.read_text())
    bank_path = resolve(repo, Path(full_summary["bank_path"]))
    moments_path = resolve(
        repo, Path(full_summary["conditional_moments_path"])
    )
    bank = pilot.load_template_bank(bank_path)
    moments = pilot.load_conditional_moments(moments_path)
    pilot.validate_conditional_moments(moments, bank)

    from analysis2.workflows import float_token

    token = float_token(float(bank.mass_gev))
    final_summary_path = (
        output_dir / f"uniform_10k_summary_ma_{token}.json"
    )
    if final_summary_path.exists() and not args.restart_checkpoint:
        raise FileExistsError(
            "This uniform-10k result is complete. Preserve it and use a "
            f"new output directory: {final_summary_path}"
        )

    selected_path = (
        audit_dir
        / f"decision_audit_final_selected_truths_ma_{token}.csv"
    )
    detailed_5k_path = (
        audit_dir
        / f"decision_audit_final_detailed_5k_ma_{token}.csv"
    )
    detailed_2k_path = (
        full_dir
        / f"full_domain_2k_detailed_accuracy_ma_{token}.csv"
    )
    for path in (selected_path, detailed_5k_path, detailed_2k_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    selected_table = pd.read_csv(selected_path)
    detailed_5k = pd.read_csv(detailed_5k_path)
    detailed_2k = pd.read_csv(detailed_2k_path)

    selected = selected_helper.validate_selected_table(
        selected_table, bank
    )
    initial_keys = {
        (model, int(index))
        for model in ("photon", "su2")
        for index in selected[model]
    }
    selected_keys = set(initial_keys)

    event_counts = np.asarray(
        sorted(detailed_5k["number_of_events"].astype(int).unique()),
        dtype=int,
    )
    seeds = np.asarray(
        sorted(detailed_5k["seed"].astype(int).unique()),
        dtype=int,
    )

    checkpoint_dir = output_dir / "uniform_10k_truth_parts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.restart_checkpoint:
        for path in checkpoint_dir.glob("*.csv"):
            path.unlink()

    common = {
        "mass_gev": float(bank.mass_gev),
        "photon_probabilities": np.asarray(
            bank.photon_probabilities, dtype=float
        ),
        "su2_probabilities": np.asarray(
            bank.su2_probabilities, dtype=float
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
        "number_of_pseudoexperiments": 10000,
        "chunk_size": int(args.chunk_size),
    }

    all_part_paths: dict[tuple[str, int], Path] = {}
    jobs = []
    for model in ("photon", "su2"):
        probabilities = np.asarray(
            getattr(bank, f"{model}_probabilities"), dtype=float
        )
        lifetimes = np.asarray(
            getattr(bank, f"{model}_ctau_m"), dtype=float
        )
        means = moments[f"{model}_mean_z_by_energy_bin_m"]
        variances = moments[
            f"{model}_variance_z_by_energy_bin_m2"
        ]
        for index in selected[model]:
            part = checkpoint_dir / f"{model}_{index:04d}.csv"
            all_part_paths[(model, int(index))] = part
            if not part.is_file():
                jobs.append(
                    (
                        model,
                        int(index),
                        float(lifetimes[index]),
                        probabilities[index],
                        means[index],
                        variances[index],
                        part,
                    )
                )

    total = len(initial_keys)
    completed_initially = total - len(jobs)
    print(
        f"UNIFORM 10K: mass={bank.mass_gev:g} GeV, "
        f"selection={bank.selection_name}, truths={total}, "
        f"completed={completed_initially}, remaining={len(jobs)}, "
        f"N-grid={event_counts.tolist()}",
        flush=True,
    )

    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=selected_helper.initialize_worker,
            initargs=(str(helper_dir), common),
        ) as executor:
            futures = {}
            submitted = {}
            for job in jobs:
                model, index, ctau, probs, means, variances, part = job
                future = executor.submit(
                    selected_helper.simulate_truth_all_seeds,
                    model,
                    index,
                    ctau,
                    probs,
                    means,
                    variances,
                    seeds.tolist(),
                )
                futures[future] = (model, index, part)
                submitted[future] = perf_counter()

            for completed, future in enumerate(
                as_completed(futures), start=1
            ):
                model, index, part = futures[future]
                frame = future.result()
                temporary = part.with_suffix(part.suffix + ".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(part)
                elapsed = perf_counter() - started
                remaining = len(jobs) - completed
                projected = remaining * elapsed / max(completed, 1)
                print(
                    f"COMPLETED {model:6s} index={index:3d} "
                    f"({completed_initially + completed}/{total}) | "
                    f"elapsed={elapsed/60:.1f} min, "
                    f"rough remaining={projected/60:.1f} min",
                    flush=True,
                )

    current_detailed = load_checkpoint_parts(
        [all_part_paths[key] for key in sorted(initial_keys)]
    )
    newly_promoted_keys: set[tuple[str, int]] = set()
    audit_history: list[dict] = []

    for round_index in range(args.maximum_promotion_rounds + 1):
        curve_10k = conservative_curve(current_detailed)
        threshold_10k = persistent_threshold(
            curve_10k, args.target_accuracy
        )
        if threshold_10k is None:
            raise RuntimeError(
                "The uniform-10k curve does not reach the target "
                "persistently on the tested grid."
            )

        audit_rows, promotions, audit_summary = (
            decision_relevant_audit(
                detailed_2k=detailed_2k,
                selected_keys=selected_keys,
                candidate_threshold=threshold_10k,
                tested_event_counts=event_counts,
                target_accuracy=args.target_accuracy,
                global_alpha=args.audit_global_alpha,
            )
        )
        audit_rows.to_csv(
            output_dir
            / f"uniform_10k_audit_round_{round_index:02d}_rows_ma_{token}.csv",
            index=False,
        )
        promotions.to_csv(
            output_dir
            / f"uniform_10k_audit_round_{round_index:02d}_promotions_ma_{token}.csv",
            index=False,
        )
        audit_history.append(
            {
                "round": int(round_index),
                "candidate_threshold": int(threshold_10k),
                "number_of_current_selected_truths": int(
                    len(selected_keys)
                ),
                **audit_summary,
            }
        )
        print(
            f"AUDIT ROUND {round_index}: N90={threshold_10k}, "
            f"selected={len(selected_keys)}, "
            f"promotions={len(promotions)}",
            flush=True,
        )

        if promotions.empty:
            final_audit_rows = audit_rows
            break

        if round_index >= args.maximum_promotion_rounds:
            raise RuntimeError(
                "Maximum promotion rounds reached before audit passed."
            )

        promotion_jobs = []
        promotion_paths: list[Path] = []
        for row in promotions.itertuples():
            model = str(row.truth_model)
            index = int(row.truth_lifetime_index)
            key = (model, index)
            if key in selected_keys:
                continue
            part = checkpoint_dir / f"{model}_{index:04d}.csv"
            all_part_paths[key] = part
            promotion_paths.append(part)
            if part.is_file():
                continue

            probabilities = np.asarray(
                getattr(bank, f"{model}_probabilities"), dtype=float
            )
            lifetimes = np.asarray(
                getattr(bank, f"{model}_ctau_m"), dtype=float
            )
            means = moments[f"{model}_mean_z_by_energy_bin_m"]
            variances = moments[
                f"{model}_variance_z_by_energy_bin_m2"
            ]
            promotion_jobs.append(
                (
                    model,
                    index,
                    float(lifetimes[index]),
                    probabilities[index],
                    means[index],
                    variances[index],
                    part,
                )
            )

        if promotion_jobs:
            print(
                f"RUNNING {len(promotion_jobs)} promoted truths at 10k.",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=selected_helper.initialize_worker,
                initargs=(str(helper_dir), common),
            ) as executor:
                futures = {}
                for job in promotion_jobs:
                    model, index, ctau, probs, means, variances, part = job
                    future = executor.submit(
                        selected_helper.simulate_truth_all_seeds,
                        model,
                        index,
                        ctau,
                        probs,
                        means,
                        variances,
                        seeds.tolist(),
                    )
                    futures[future] = (model, index, part)

                for completed, future in enumerate(
                    as_completed(futures), start=1
                ):
                    model, index, part = futures[future]
                    frame = future.result()
                    temporary = part.with_suffix(part.suffix + ".tmp")
                    frame.to_csv(temporary, index=False)
                    temporary.replace(part)
                    print(
                        f"COMPLETED PROMOTION {model:6s} "
                        f"index={index:3d} "
                        f"({completed}/{len(promotion_jobs)})",
                        flush=True,
                    )

        promoted = load_checkpoint_parts(promotion_paths)
        if not promoted.empty:
            current_detailed = pd.concat(
                [current_detailed, promoted], ignore_index=True
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
        raise RuntimeError("Internal audit-loop failure.")

    final_curve = conservative_curve(current_detailed)
    final_threshold = persistent_threshold(
        final_curve, args.target_accuracy
    )
    curve_5k = conservative_curve(detailed_5k)
    threshold_5k = persistent_threshold(
        curve_5k, args.target_accuracy
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
        output_dir / f"uniform_10k_detailed_accuracy_ma_{token}.csv",
        index=False,
    )
    final_curve.to_csv(
        output_dir / f"uniform_10k_conservative_curve_ma_{token}.csv",
        index=False,
    )
    final_audit_rows.to_csv(
        output_dir / f"uniform_10k_final_audit_rows_ma_{token}.csv",
        index=False,
    )

    selected_rows = []
    for model, index in sorted(selected_keys):
        lifetimes = np.asarray(
            getattr(bank, f"{model}_ctau_m"), dtype=float
        )
        intervals = np.asarray(
            getattr(bank, f"{model}_interval_index"), dtype=int
        )
        selected_rows.append(
            {
                "truth_model": model,
                "truth_lifetime_index": int(index),
                "truth_interval_index": int(intervals[index]),
                "truth_ctau_m": float(lifetimes[index]),
                "selection_reason": (
                    "decision_audited_5k_selected"
                    if (model, index) in initial_keys
                    else "uniform_10k_decision_audit_promotion"
                ),
            }
        )
    pd.DataFrame(selected_rows).to_csv(
        output_dir / f"uniform_10k_final_selected_truths_ma_{token}.csv",
        index=False,
    )

    threshold_rows = []
    for seed, subset in current_detailed.groupby("seed"):
        threshold_rows.append(
            {
                "seed": int(seed),
                "persistent_threshold": persistent_threshold(
                    conservative_curve(subset), args.target_accuracy
                ),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / f"uniform_10k_threshold_by_seed_ma_{token}.csv",
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
        output_dir / f"uniform_10k_limiting_points_ma_{token}.csv",
        index=False,
    )

    comparison = curve_5k.rename(
        columns={"worst_case_accuracy": "selected_5k"}
    ).merge(
        final_curve.rename(
            columns={"worst_case_accuracy": "uniform_10k"}
        ),
        on="number_of_events",
        how="inner",
    )
    comparison["signed_difference_10k_minus_5k"] = (
        comparison["uniform_10k"] - comparison["selected_5k"]
    )
    comparison["absolute_difference"] = np.abs(
        comparison["signed_difference_10k_minus_5k"]
    )
    comparison.to_csv(
        output_dir / f"uniform_10k_curve_5k_vs_10k_ma_{token}.csv",
        index=False,
    )

    crossing_lower = min(
        value for value in (threshold_5k, final_threshold)
        if value is not None
    ) - 5
    crossing_upper = max(
        value for value in (threshold_5k, final_threshold)
        if value is not None
    ) + 5
    crossing = comparison[
        comparison["number_of_events"].between(
            crossing_lower, crossing_upper
        )
    ].copy()

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.plot(
        comparison["number_of_events"],
        comparison["selected_5k"],
        marker="o",
        markersize=3,
        label=f"Decision-audited selected set, 5k ($N_{{90}}={threshold_5k}$)",
    )
    axis.plot(
        comparison["number_of_events"],
        comparison["uniform_10k"],
        marker="s",
        markersize=3,
        label=f"Uniform selected set, 10k ($N_{{90}}={final_threshold}$)",
    )
    axis.axhline(
        args.target_accuracy,
        linestyle="--",
        linewidth=1.2,
        label="90% target",
    )
    if final_threshold is not None:
        axis.axvline(final_threshold, linestyle=":", linewidth=1.2)
    axis.set_xlabel("Observed ALP decays, $N$")
    axis.set_ylabel(
        "Worst-case correct-classification probability"
    )
    axis.set_title(
        f"5k versus 10k validation, $m_a={bank.mass_gev:g}$ GeV"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / f"uniform_10k_vs_5k_accuracy_ma_{token}.pdf"
    )
    plt.close(figure)

    result = {
        "status": (
            "generic_uniform_selected_10k_with_"
            "decision_relevant_omitted_truth_audit"
        ),
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "target_accuracy": float(args.target_accuracy),
        "pseudoexperiments_per_truth_and_seed": 10000,
        "seeds": [int(seed) for seed in seeds],
        "event_counts": [int(value) for value in event_counts],
        "persistent_thresholds": {
            "decision_audited_selected_5k": threshold_5k,
            "uniform_selected_10k": final_threshold,
        },
        "number_of_truths": {
            "initial_selected": int(len(initial_keys)),
            "newly_promoted_at_10k": int(len(newly_promoted_keys)),
            "final_selected": int(len(selected_keys)),
            "total_full_domain": int(
                detailed_2k[
                    ["truth_model", "truth_lifetime_index"]
                ].drop_duplicates().shape[0]
            ),
        },
        "curve_stability_5k_vs_10k": {
            "number_of_overlapping_event_counts": int(len(comparison)),
            "mean_absolute_difference_all_counts": float(
                comparison["absolute_difference"].mean()
            ),
            "maximum_absolute_difference_all_counts": float(
                comparison["absolute_difference"].max()
            ),
            "event_at_maximum_difference_all_counts": int(
                comparison.loc[
                    comparison["absolute_difference"].idxmax(),
                    "number_of_events",
                ]
            ),
            "mean_absolute_difference_crossing_window": float(
                crossing["absolute_difference"].mean()
            ),
            "maximum_absolute_difference_crossing_window": float(
                crossing["absolute_difference"].max()
            ),
        },
        "audit_rounds": audit_history,
        "final_audit": audit_history[-1],
        "runtime": {
            "elapsed_seconds_this_invocation": float(
                perf_counter() - started
            )
        },
        "next_action": (
            "If N90 is stable, the audit passes, and the crossing "
            "curve is stable, freeze this mass-selection point. "
            "Then integrate the generic workflow into analysis2."
        ),
    }
    final_summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
