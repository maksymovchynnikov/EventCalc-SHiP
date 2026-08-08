"""Generic selected-truth 5k validation and omitted-truth audit.

This stage consumes one completed generic full-domain 2k result for the
energy + conditional-mean-z observable. It reevaluates only the difficult
truth lifetimes selected by the full-domain screen, using 5,000
pseudoexperiments per truth and seed on a unit-spaced crossing window plus a
small persistence tail. Every omitted 2k truth is then audited against the
selected-5k conservative envelope with simultaneous one-sided Clopper-Pearson
bounds.

The script is generic in mass, detector selection, lifetime-bank size and
connected lifetime intervals. It reuses the existing template bank and
conditional-z moments and never regenerates EventCalc proposals.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


DEFAULT_SEEDS = (73241, 83244, 93247, 103250, 113253)
_WORKER_PILOT = None
_WORKER_COMMON: dict | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-domain-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-script-dir", type=Path, default=Path.home() / "Downloads"
    )
    parser.add_argument("--pseudoexperiments", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--unit-below-threshold", type=int, default=10)
    parser.add_argument("--unit-above-threshold", type=int, default=20)
    parser.add_argument("--lower-anchor-count", type=int, default=3)
    parser.add_argument("--audit-global-alpha", type=float, default=0.01)
    parser.add_argument(
        "--event-count-grid",
        type=str,
        help="Optional explicit N grid, e.g. '25,28,30:61,67,85,115,145'.",
    )
    parser.add_argument("--restart-checkpoint", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()



def load_pilot(script_dir: Path):
    """Return the package-native validated implementation."""
    del script_dir
    from alp_discrimination import conditional_mean_z
    return conditional_mean_z

def parse_event_counts(specification: str) -> np.ndarray:
    counts: set[int] = set()
    for raw in specification.split(","):
        token = raw.strip()
        if not token:
            continue
        pieces = token.split(":")
        try:
            values = [int(piece) for piece in pieces]
        except ValueError as error:
            raise ValueError(f"Invalid event-count token: {token}") from error
        if len(values) == 1:
            counts.add(values[0])
        elif len(values) in (2, 3):
            start, stop = values[:2]
            step = values[2] if len(values) == 3 else 1
            if start < 1 or stop < start or step < 1:
                raise ValueError(f"Invalid event-count token: {token}")
            counts.update(range(start, stop + 1, step))
        else:
            raise ValueError(f"Invalid event-count token: {token}")
    result = np.asarray(sorted(counts), dtype=int)
    if len(result) == 0 or np.any(result < 1):
        raise ValueError("The event-count grid must contain positive integers.")
    return result


def selected_grid_from_full_domain(
    summary: dict,
    *,
    below: int,
    above: int,
    lower_anchor_count: int,
) -> np.ndarray:
    source = np.asarray(summary["event_counts"], dtype=int)
    threshold = summary.get("persistent_threshold_all_truths_and_seeds")
    if threshold is None:
        raise ValueError("The full-domain screen did not reach a persistent threshold.")
    threshold = int(threshold)
    lower = max(1, threshold - int(below))
    upper = threshold + int(above)
    unit = np.arange(lower, upper + 1, dtype=int)

    lower_candidates = source[source < lower]
    anchors = np.asarray([], dtype=int)
    if len(lower_candidates) and lower_anchor_count > 0:
        positions = np.linspace(
            0, len(lower_candidates) - 1, min(int(lower_anchor_count), len(lower_candidates))
        )
        anchors = lower_candidates[np.unique(np.rint(positions).astype(int))]

    tail = source[source > upper]
    return np.unique(np.concatenate([anchors, unit, tail])).astype(int)


def persistent_threshold(curve: pd.DataFrame, target: float = 0.90) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix >= float(target))
    return None if len(passing) == 0 else int(counts[passing[0]])


def initialize_worker(pilot_script_dir: str, common: dict) -> None:
    global _WORKER_PILOT, _WORKER_COMMON
    _WORKER_PILOT = load_pilot(Path(pilot_script_dir))
    _WORKER_COMMON = common


def simulate_truth_all_seeds(
    truth_model: str,
    truth_index: int,
    truth_ctau_m: float,
    truth_probabilities: np.ndarray,
    truth_mean: np.ndarray,
    truth_variance: np.ndarray,
    seeds: Iterable[int],
) -> pd.DataFrame:
    if _WORKER_PILOT is None or _WORKER_COMMON is None:
        raise RuntimeError("Worker was not initialized.")
    frames = []
    for seed in seeds:
        frame = _WORKER_PILOT.simulate_truth(
            {
                **_WORKER_COMMON,
                "truth_model": str(truth_model),
                "truth_index": int(truth_index),
                "truth_ctau_m": float(truth_ctau_m),
                "truth_probabilities": np.asarray(truth_probabilities, dtype=float),
                "truth_conditional_mean_z": np.asarray(truth_mean, dtype=float),
                "truth_conditional_variance_z": np.asarray(truth_variance, dtype=float),
                "seed": int(seed),
            }
        )
        frames.append(frame[frame["observable"] == "conditional_combined"].copy())
    return pd.concat(frames, ignore_index=True)


def validate_selected_table(selected_table: pd.DataFrame, bank) -> dict[str, list[int]]:
    required = {"truth_model", "truth_lifetime_index"}
    missing = required - set(selected_table.columns)
    if missing:
        raise ValueError(f"Selected-truth table is missing columns: {sorted(missing)}")
    selected: dict[str, list[int]] = {}
    for model in ("photon", "su2"):
        indices = sorted(
            selected_table.loc[
                selected_table["truth_model"].astype(str) == model,
                "truth_lifetime_index",
            ].astype(int).unique()
        )
        limit = len(getattr(bank, f"{model}_ctau_m"))
        invalid = [index for index in indices if not 0 <= index < limit]
        if invalid:
            raise ValueError(f"Invalid {model} lifetime indices: {invalid}")
        if not indices:
            raise ValueError(f"No selected truths found for model {model}.")
        selected[model] = indices
    return selected


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


def audit_omitted_truths(
    *,
    detailed_2k: pd.DataFrame,
    selected_5k_curve: pd.DataFrame,
    selected_keys: set[tuple[str, int]],
    global_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    shared_counts = set(selected_5k_curve["number_of_events"].astype(int))
    combined = detailed_2k[
        detailed_2k["number_of_events"].astype(int).isin(shared_counts)
    ].copy()
    keys = list(
        zip(
            combined["truth_model"].astype(str),
            combined["truth_lifetime_index"].astype(int),
        )
    )
    omitted = combined[
        np.asarray([key not in selected_keys for key in keys], dtype=bool)
    ].copy()
    if omitted.empty:
        empty_promotions = pd.DataFrame(
            columns=[
                "truth_model",
                "truth_lifetime_index",
                "truth_ctau_m",
                "minimum_lower_margin",
                "number_of_failing_rows",
                "worst_seed",
                "worst_number_of_events",
                "minimum_correct_fraction",
            ]
        )
        return omitted, empty_promotions, {
            "global_alpha": float(global_alpha),
            "number_of_comparisons": 0,
            "alpha_per_comparison": None,
            "minimum_lower_margin": None,
            "number_of_failing_rows": 0,
            "number_of_truths_requiring_promotion": 0,
        }

    comparisons = int(len(omitted))
    alpha_each = float(global_alpha) / comparisons
    omitted["successes"] = np.rint(
        omitted["correct_fraction"] * omitted["number_of_pseudoexperiments"]
    ).astype(int)
    omitted["audit_lower_bound"] = clopper_pearson_lower(
        omitted["successes"].to_numpy(),
        omitted["number_of_pseudoexperiments"].to_numpy(dtype=int),
        alpha_each,
    )
    reference = selected_5k_curve.set_index("number_of_events")[
        "worst_case_accuracy"
    ]
    omitted["selected_5k_envelope"] = omitted["number_of_events"].map(reference)
    omitted["lower_margin"] = (
        omitted["audit_lower_bound"] - omitted["selected_5k_envelope"]
    )
    omitted["requires_promotion"] = omitted["lower_margin"] < 0.0

    failing = omitted[omitted["requires_promotion"]].copy()
    if failing.empty:
        promotions = pd.DataFrame(
            columns=[
                "truth_model",
                "truth_lifetime_index",
                "truth_ctau_m",
                "minimum_lower_margin",
                "number_of_failing_rows",
                "worst_seed",
                "worst_number_of_events",
                "minimum_correct_fraction",
            ]
        )
    else:
        worst_rows = (
            failing.sort_values("lower_margin")
            .drop_duplicates(["truth_model", "truth_lifetime_index"], keep="first")
            .set_index(["truth_model", "truth_lifetime_index"])
        )
        promotions = (
            failing.groupby(
                ["truth_model", "truth_lifetime_index", "truth_ctau_m"],
                as_index=False,
            )
            .agg(
                minimum_lower_margin=("lower_margin", "min"),
                number_of_failing_rows=("requires_promotion", "sum"),
                minimum_correct_fraction=("correct_fraction", "min"),
            )
        )
        promotions["worst_seed"] = [
            int(worst_rows.loc[(row.truth_model, row.truth_lifetime_index), "seed"])
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
            ["minimum_lower_margin", "truth_model", "truth_lifetime_index"],
            inplace=True,
            ignore_index=True,
        )

    summary = {
        "global_alpha": float(global_alpha),
        "number_of_comparisons": comparisons,
        "alpha_per_comparison": float(alpha_each),
        "minimum_lower_margin": float(omitted["lower_margin"].min()),
        "number_of_failing_rows": int(omitted["requires_promotion"].sum()),
        "number_of_truths_requiring_promotion": int(len(promotions)),
    }
    return omitted, promotions, summary


def main() -> None:
    args = parse_args()
    start_total = perf_counter()
    repo = Path.cwd().resolve()
    if not (repo / "alp_discrimination").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")
    if args.pseudoexperiments != 5000:
        raise ValueError("This stage is intentionally fixed at 5,000 pseudoexperiments.")
    if not (0.0 < args.audit_global_alpha < 1.0):
        raise ValueError("--audit-global-alpha must lie between zero and one.")

    pilot = load_pilot(args.pilot_script_dir)
    summary_path = resolve(repo, args.full_domain_summary)
    full_dir = summary_path.parent
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text())

    bank_path = resolve(repo, Path(summary["bank_path"]))
    moments_path = resolve(repo, Path(summary["conditional_moments_path"]))
    bank = pilot.load_template_bank(bank_path)
    arrays = pilot.load_conditional_moments(moments_path)
    pilot.validate_conditional_moments(arrays, bank)

    from alp_discrimination.workflows import float_token

    token = float_token(float(bank.mass_gev))
    selected_path = full_dir / f"full_domain_2k_selected_truths_ma_{token}.csv"
    detailed_2k_path = full_dir / f"full_domain_2k_detailed_accuracy_ma_{token}.csv"
    curve_2k_path = full_dir / f"full_domain_2k_conservative_curve_ma_{token}.csv"
    if not selected_path.is_file() or not detailed_2k_path.is_file():
        raise FileNotFoundError("The full-domain directory is missing required detailed outputs.")

    final_summary = output_dir / f"selected_5k_summary_ma_{token}.json"
    if final_summary.exists() and not args.restart_checkpoint:
        raise FileExistsError(
            "This selected-5k point is already complete. Preserve it and use a new "
            f"output directory: {final_summary}"
        )

    selected_table = pd.read_csv(selected_path)
    selected = validate_selected_table(selected_table, bank)
    selected_keys = {
        (model, int(index))
        for model in ("photon", "su2")
        for index in selected[model]
    }

    if args.event_count_grid:
        event_counts = parse_event_counts(args.event_count_grid)
        grid_source = "explicit_cli"
    else:
        event_counts = selected_grid_from_full_domain(
            summary,
            below=args.unit_below_threshold,
            above=args.unit_above_threshold,
            lower_anchor_count=args.lower_anchor_count,
        )
        grid_source = "derived_from_full_domain_threshold"
    (output_dir / f"selected_5k_event_grid_ma_{token}.txt").write_text(
        ",".join(str(int(value)) for value in event_counts) + "\n"
    )

    checkpoint_dir = output_dir / "selected_5k_truth_parts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.restart_checkpoint:
        for path in checkpoint_dir.glob("*.csv"):
            path.unlink()

    common = {
        "mass_gev": float(bank.mass_gev),
        "photon_probabilities": np.asarray(bank.photon_probabilities, dtype=float),
        "su2_probabilities": np.asarray(bank.su2_probabilities, dtype=float),
        "photon_conditional_mean_z": arrays["photon_mean_z_by_energy_bin_m"],
        "photon_conditional_variance_z": arrays[
            "photon_variance_z_by_energy_bin_m2"
        ],
        "su2_conditional_mean_z": arrays["su2_mean_z_by_energy_bin_m"],
        "su2_conditional_variance_z": arrays["su2_variance_z_by_energy_bin_m2"],
        "event_counts": event_counts,
        "number_of_pseudoexperiments": int(args.pseudoexperiments),
        "chunk_size": int(args.chunk_size),
    }

    jobs = []
    all_parts: list[Path] = []
    for model in ("photon", "su2"):
        probabilities = np.asarray(getattr(bank, f"{model}_probabilities"), dtype=float)
        lifetimes = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
        means = arrays[f"{model}_mean_z_by_energy_bin_m"]
        variances = arrays[f"{model}_variance_z_by_energy_bin_m2"]
        for index in selected[model]:
            part = checkpoint_dir / f"{model}_{index:04d}.csv"
            all_parts.append(part)
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

    total = sum(len(indices) for indices in selected.values())
    already_done = total - len(jobs)
    print(
        f"SELECTED 5K: mass={bank.mass_gev:g} GeV, selection={bank.selection_name}, "
        f"truths={total}, completed={already_done}, remaining={len(jobs)}, "
        f"N-grid={event_counts.tolist()}",
        flush=True,
    )

    durations: list[float] = []
    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(str(args.pilot_script_dir.expanduser().resolve()), common),
        ) as executor:
            futures = {}
            submitted = {}
            for job in jobs:
                model, index, ctau, probabilities, mean, variance, part = job
                future = executor.submit(
                    simulate_truth_all_seeds,
                    model,
                    index,
                    ctau,
                    probabilities,
                    mean,
                    variance,
                    [int(seed) for seed in args.seeds],
                )
                futures[future] = (model, index, part)
                submitted[future] = perf_counter()

            for completed, future in enumerate(as_completed(futures), start=1):
                model, index, part = futures[future]
                frame = future.result()
                temporary = part.with_suffix(part.suffix + ".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(part)
                durations.append(perf_counter() - submitted[future])
                elapsed = perf_counter() - start_total
                remaining = len(jobs) - completed
                projected = remaining * elapsed / max(completed, 1)
                print(
                    f"COMPLETED {model:6s} index={index:3d} "
                    f"({already_done + completed}/{total}) | "
                    f"elapsed={elapsed/60:.1f} min, rough remaining={projected/60:.1f} min",
                    flush=True,
                )

    missing = [path for path in all_parts if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected-truth checkpoints; rerun.")

    detailed_5k = pd.concat((pd.read_csv(path) for path in all_parts), ignore_index=True)
    detailed_5k.sort_values(
        ["seed", "number_of_events", "correct_fraction", "truth_model", "truth_lifetime_index"],
        inplace=True,
        ignore_index=True,
    )
    detailed_5k.to_csv(
        output_dir / f"selected_5k_detailed_accuracy_ma_{token}.csv", index=False
    )

    curve_5k = (
        detailed_5k.groupby("number_of_events", as_index=False)["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values("number_of_events", ignore_index=True)
    )
    curve_5k.to_csv(
        output_dir / f"selected_5k_conservative_curve_ma_{token}.csv", index=False
    )
    threshold_5k = persistent_threshold(curve_5k)

    by_seed_rows = []
    for seed, subset in detailed_5k.groupby("seed"):
        seed_curve = (
            subset.groupby("number_of_events", as_index=False)["correct_fraction"]
            .min()
            .rename(columns={"correct_fraction": "worst_case_accuracy"})
        )
        by_seed_rows.append(
            {"seed": int(seed), "persistent_threshold": persistent_threshold(seed_curve)}
        )
    by_seed = pd.DataFrame(by_seed_rows)
    by_seed.to_csv(
        output_dir / f"selected_5k_threshold_by_seed_ma_{token}.csv", index=False
    )

    limiting = pd.DataFrame()
    if threshold_5k is not None:
        limiting = (
            detailed_5k[detailed_5k["number_of_events"] == int(threshold_5k)]
            .nsmallest(20, "correct_fraction")
            .copy()
        )
    limiting.to_csv(
        output_dir / f"selected_5k_limiting_points_ma_{token}.csv", index=False
    )

    detailed_2k = pd.read_csv(detailed_2k_path)
    audit_rows, promotions, audit_summary = audit_omitted_truths(
        detailed_2k=detailed_2k,
        selected_5k_curve=curve_5k,
        selected_keys=selected_keys,
        global_alpha=float(args.audit_global_alpha),
    )
    audit_rows.to_csv(
        output_dir / f"selected_5k_omitted_audit_rows_ma_{token}.csv", index=False
    )
    promotions.to_csv(
        output_dir / f"selected_5k_promotions_ma_{token}.csv", index=False
    )
    selected_table.to_csv(
        output_dir / f"selected_5k_truths_ma_{token}.csv", index=False
    )

    curve_2k = pd.read_csv(curve_2k_path)
    comparison = curve_2k.rename(
        columns={"worst_case_accuracy": "accuracy_2k"}
    )[["number_of_events", "accuracy_2k"]].merge(
        curve_5k.rename(columns={"worst_case_accuracy": "accuracy_5k"})[
            ["number_of_events", "accuracy_5k"]
        ],
        on="number_of_events",
        how="inner",
    )
    comparison["signed_difference_5k_minus_2k"] = (
        comparison["accuracy_5k"] - comparison["accuracy_2k"]
    )
    comparison["absolute_difference"] = np.abs(
        comparison["signed_difference_5k_minus_2k"]
    )
    comparison.to_csv(
        output_dir / f"conservative_curve_2k_vs_5k_ma_{token}.csv", index=False
    )

    threshold_2k = summary.get("persistent_threshold_all_truths_and_seeds")
    crossing = comparison[
        comparison["number_of_events"].between(
            max(1, int(threshold_2k) - 5), int(threshold_2k) + 5
        )
    ]
    elapsed = perf_counter() - start_total
    result = {
        "status": "generic_selected_5k_with_omitted_2k_audit",
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "full_domain_summary_path": str(summary_path),
        "bank_path": str(bank_path),
        "conditional_moments_path": str(moments_path),
        "pseudoexperiments_per_selected_truth_and_seed": int(args.pseudoexperiments),
        "seeds": [int(seed) for seed in args.seeds],
        "event_counts": [int(value) for value in event_counts],
        "event_grid_source": grid_source,
        "number_of_selected_truths": {
            "photon": int(len(selected["photon"])),
            "su2": int(len(selected["su2"])),
        },
        "persistent_thresholds": {
            "full_domain_2k": None if threshold_2k is None else int(threshold_2k),
            "selected_5k": threshold_5k,
        },
        "curve_stability_2k_vs_5k": {
            "number_of_overlapping_event_counts": int(len(comparison)),
            "mean_absolute_difference_all_counts": float(comparison["absolute_difference"].mean()),
            "maximum_absolute_difference_all_counts": float(comparison["absolute_difference"].max()),
            "event_at_maximum_difference_all_counts": int(
                comparison.loc[comparison["absolute_difference"].idxmax(), "number_of_events"]
            ),
            "mean_absolute_difference_crossing_window": float(crossing["absolute_difference"].mean()),
            "maximum_absolute_difference_crossing_window": float(crossing["absolute_difference"].max()),
        },
        "limiting_points_at_5k_threshold": limiting[
            [
                column
                for column in (
                    "seed",
                    "truth_model",
                    "truth_lifetime_index",
                    "truth_ctau_m",
                    "number_of_events",
                    "correct_fraction",
                )
                if column in limiting.columns
            ]
        ].to_dict(orient="records"),
        "omitted_truth_audit": audit_summary,
        "runtime": {
            "elapsed_seconds_this_invocation": float(elapsed),
            "new_truths_completed_this_invocation": int(len(durations)),
        },
        "next_action": (
            "If promotions are listed, run only those truths at 5k and re-audit. "
            "If the audit passes, decide from the 2k-versus-5k crossing stability "
            "whether this point needs a uniform selected 10k stage."
        ),
    }
    final_summary.write_text(json.dumps(result, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8.3, 5.3))
    ax.plot(
        comparison["number_of_events"],
        comparison["accuracy_2k"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label=f"Full domain, 2k (N90={threshold_2k})",
    )
    ax.plot(
        comparison["number_of_events"],
        comparison["accuracy_5k"],
        marker="o",
        markersize=3,
        linewidth=1.7,
        label=f"Selected truths, 5k (N90={threshold_5k})",
    )
    ax.axhline(0.9, linestyle="--", linewidth=1.0, label="90% target")
    ax.set_xlabel("Observed ALP decays, N")
    ax.set_ylabel("Worst-case correct-classification probability")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"selected_5k_vs_full_2k_accuracy_ma_{token}.pdf")
    plt.close(fig)

    print("\n" + json.dumps(result, indent=2), flush=True)
    print("\n5k audit promotions:", flush=True)
    print("none" if promotions.empty else promotions.to_string(index=False), flush=True)
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
