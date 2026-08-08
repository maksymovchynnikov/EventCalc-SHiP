#!/usr/bin/env python3
"""Selected 5k validation and decision-relevant omitted-truth audit.

This workflow starts from a completed full-domain conditional-feature screen
(2k pseudoexperiments per truth and seed).  It then:

1. selects difficult truths around the candidate N90 crossing;
2. includes direct per-seed limiters, disconnected-domain endpoints,
   the minimum-distance pair, and same-interval neighbours;
3. reevaluates the selected set with 5k pseudoexperiments and the same seeds;
4. performs a simultaneous one-sided omitted-truth audit only at event counts
   relevant to the persistent N90 decision;
5. automatically promotes genuinely uncertified omitted truths (plus their
   same-interval neighbours) to 5k, up to a small number of rounds.

The likelihood and truth generator are the same conditional-Gaussian feature
implementation used by analysis2.workflows.conditional_feature_pilot.  The
result remains subject to empirical conditional-resampling and an independent
EventCalc/template-stream check before publication use.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

from analysis2.progress import ProgressMeter


OBSERVABLE_CHOICES = (
    "energy",
    "energy_mean_z",
    "energy_mean_z_spread",
    "energy_mean_r_perp",
    "energy_mean_z_r_perp",
    "energy_mean_z_spread_r_perp",
)


def float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def load_bank_light(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = [
            "mass_GeV",
            "selection_name",
            "energy_edges_GeV",
            "photon_ctau_m",
            "photon_interval_index",
            "photon_probabilities",
            "su2_ctau_m",
            "su2_interval_index",
            "su2_probabilities",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Template bank is missing keys: {missing}")
        energy_edges = np.asarray(data["energy_edges_GeV"], dtype=float)
        return SimpleNamespace(
            mass_gev=float(np.asarray(data["mass_GeV"]).item()),
            selection_name=_scalar_text(data["selection_name"]),
            energy_edges_gev=energy_edges,
            number_of_energy_bins=int(len(energy_edges) - 1),
            photon_ctau_m=np.asarray(data["photon_ctau_m"], dtype=float),
            photon_interval_index=np.asarray(
                data["photon_interval_index"], dtype=int
            ),
            photon_probabilities=np.asarray(
                data["photon_probabilities"], dtype=float
            ),
            su2_ctau_m=np.asarray(data["su2_ctau_m"], dtype=float),
            su2_interval_index=np.asarray(data["su2_interval_index"], dtype=int),
            su2_probabilities=np.asarray(data["su2_probabilities"], dtype=float),
        )


def load_moments_light(path: Path, bank) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    required = [
        "energy_edges_GeV",
        "photon_feature_mean",
        "photon_feature_covariance",
        "su2_feature_mean",
        "su2_feature_covariance",
    ]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Conditional-feature moments are missing keys: {missing}")
    if not np.allclose(
        np.asarray(arrays["energy_edges_GeV"], dtype=float),
        bank.energy_edges_gev,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Moment file and bank use different energy edges.")
    for model in ("photon", "su2"):
        means = np.asarray(arrays[f"{model}_feature_mean"], dtype=float)
        covariance = np.asarray(
            arrays[f"{model}_feature_covariance"], dtype=float
        )
        expected_n = len(getattr(bank, f"{model}_ctau_m"))
        if means.shape != (expected_n, bank.number_of_energy_bins, 3):
            raise ValueError(f"{model} feature means have the wrong shape.")
        if covariance.shape != (
            expected_n,
            bank.number_of_energy_bins,
            3,
            3,
        ):
            raise ValueError(f"{model} feature covariance has the wrong shape.")
        if np.any(~np.isfinite(means)) or np.any(~np.isfinite(covariance)):
            raise ValueError(f"{model} feature moments are non-finite.")
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise ValueError(f"{model} covariance is not positive definite.")
    return arrays


DEFAULT_SEEDS = [73241, 83244, 93247, 103250, 113253]
DEFAULT_EVENT_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-domain-dir", type=Path, required=True)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--moments-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--observable",
        default="energy_mean_z_r_perp",
        choices=OBSERVABLE_CHOICES,
    )
    parser.add_argument("--pseudoexperiments", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--event-counts",
        nargs="+",
        type=int,
        default=DEFAULT_EVENT_COUNTS,
    )
    parser.add_argument(
        "--selection-event-counts",
        nargs="+",
        type=int,
        default=[3, 4, 5],
    )
    parser.add_argument(
        "--hard-truth-gap",
        type=float,
        default=0.02,
        help=(
            "Select a truth if any seed/event-count accuracy is within this "
            "amount of the full-domain 2k conservative envelope."
        ),
    )
    parser.add_argument("--top-per-model-and-count", type=int, default=3)
    parser.add_argument("--neighbour-radius", type=int, default=1)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--audit-global-alpha", type=float, default=0.01)
    parser.add_argument("--maximum-promotion-rounds", type=int, default=3)
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the initial selected-truth table and stop before 5k.",
    )
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def persistent_threshold(curve: pd.DataFrame, target: float) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix >= float(target))
    return None if len(passing) == 0 else int(counts[passing[0]])


def conservative_curve(detailed: pd.DataFrame) -> pd.DataFrame:
    return (
        detailed.groupby("number_of_events", as_index=False)["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values("number_of_events", ignore_index=True)
    )


def same_interval_neighbours(
    *,
    indices: Iterable[int],
    interval_index: np.ndarray,
    radius: int,
) -> set[int]:
    interval_index = np.asarray(interval_index, dtype=int)
    output: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if not 0 <= index < len(interval_index):
            raise IndexError(f"Lifetime index out of range: {index}")
        interval = int(interval_index[index])
        for candidate in range(index - int(radius), index + int(radius) + 1):
            if (
                0 <= candidate < len(interval_index)
                and int(interval_index[candidate]) == interval
            ):
                output.add(int(candidate))
    return output


def add_reason(
    reasons: dict[tuple[str, int], set[str]],
    model: str,
    index: int,
    reason: str,
) -> None:
    reasons.setdefault((str(model), int(index)), set()).add(str(reason))


def build_initial_selection(
    *,
    detailed_2k: pd.DataFrame,
    screening_truths: pd.DataFrame,
    distance_minima: pd.DataFrame,
    bank,
    observable: str,
    selection_event_counts: np.ndarray,
    hard_truth_gap: float,
    top_per_model_and_count: int,
    neighbour_radius: int,
) -> pd.DataFrame:
    detailed = detailed_2k[
        (detailed_2k["observable"] == observable)
        & detailed_2k["number_of_events"].astype(int).isin(
            selection_event_counts.astype(int).tolist()
        )
    ].copy()
    if detailed.empty:
        raise ValueError("No full-domain rows match the selection event counts.")

    envelope = (
        detailed.groupby("number_of_events")["correct_fraction"]
        .min()
        .rename("full_domain_envelope")
    )
    detailed = detailed.join(envelope, on="number_of_events")
    detailed["gap_to_envelope"] = (
        detailed["correct_fraction"] - detailed["full_domain_envelope"]
    )

    reasons: dict[tuple[str, int], set[str]] = {}
    base: dict[str, set[int]] = {"photon": set(), "su2": set()}

    close = detailed[detailed["gap_to_envelope"] <= float(hard_truth_gap)]
    for row in close.itertuples(index=False):
        model = str(row.truth_model)
        index = int(row.truth_lifetime_index)
        base[model].add(index)
        add_reason(
            reasons,
            model,
            index,
            f"within_{hard_truth_gap:g}_of_2k_envelope_at_N{int(row.number_of_events)}",
        )

    # Every direct model-specific limiter for every seed and crossing count.
    for (model, seed, number_of_events), subset in detailed.groupby(
        ["truth_model", "seed", "number_of_events"]
    ):
        row = subset.loc[subset["correct_fraction"].idxmin()]
        index = int(row["truth_lifetime_index"])
        model = str(model)
        base[model].add(index)
        add_reason(
            reasons,
            model,
            index,
            f"direct_model_limiter_seed_{int(seed)}_N{int(number_of_events)}",
        )

    # A few globally most competitive truths per model and crossing count.
    top_n = int(top_per_model_and_count)
    for model in ("photon", "su2"):
        for number_of_events in selection_event_counts:
            subset = detailed[
                (detailed["truth_model"] == model)
                & (detailed["number_of_events"] == int(number_of_events))
            ].nsmallest(top_n, "correct_fraction")
            for row in subset.itertuples(index=False):
                index = int(row.truth_lifetime_index)
                base[model].add(index)
                add_reason(
                    reasons,
                    model,
                    index,
                    f"top_{top_n}_{model}_at_N{int(number_of_events)}",
                )

    # Every endpoint of every disconnected lifetime component.
    for model in ("photon", "su2"):
        model_table = screening_truths[
            screening_truths["truth_model"] == model
        ]
        for interval, subset in model_table.groupby("truth_interval_index"):
            lower = int(subset["truth_lifetime_index"].min())
            upper = int(subset["truth_lifetime_index"].max())
            for index, side in ((lower, "lower"), (upper, "upper")):
                base[model].add(index)
                add_reason(
                    reasons,
                    model,
                    index,
                    f"interval_{int(interval)}_{side}_endpoint",
                )

    # Minimum diagnostic distance pair.
    row = distance_minima[
        distance_minima["observable"] == observable
    ].iloc[0]
    for model, column in (
        ("photon", "photon_lifetime_index"),
        ("su2", "su2_lifetime_index"),
    ):
        index = int(row[column])
        base[model].add(index)
        add_reason(reasons, model, index, "minimum_H2_pair")

    selected: dict[str, set[int]] = {}
    for model in ("photon", "su2"):
        intervals = np.asarray(
            getattr(bank, f"{model}_interval_index"), dtype=int
        )
        selected[model] = same_interval_neighbours(
            indices=base[model],
            interval_index=intervals,
            radius=int(neighbour_radius),
        )
        for index in selected[model]:
            if index not in base[model]:
                add_reason(
                    reasons,
                    model,
                    index,
                    f"same_interval_neighbour_radius_{int(neighbour_radius)}",
                )

    rows: list[dict] = []
    for model in ("photon", "su2"):
        lifetimes = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
        intervals = np.asarray(
            getattr(bank, f"{model}_interval_index"), dtype=int
        )
        for index in sorted(selected[model]):
            subset = detailed[
                (detailed["truth_model"] == model)
                & (detailed["truth_lifetime_index"].astype(int) == int(index))
            ]
            rows.append(
                {
                    "truth_model": model,
                    "truth_lifetime_index": int(index),
                    "truth_interval_index": int(intervals[index]),
                    "truth_ctau_m": float(lifetimes[index]),
                    "minimum_gap_to_2k_envelope": (
                        None
                        if subset.empty
                        else float(subset["gap_to_envelope"].min())
                    ),
                    "selection_reasons": ";".join(
                        sorted(reasons.get((model, int(index)), {"selected"}))
                    ),
                    "selection_round": 0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["truth_model", "truth_lifetime_index"], ignore_index=True
    )


def checkpoint_path(
    checkpoint_dir: Path,
    model: str,
    index: int,
    seed: int,
) -> Path:
    return checkpoint_dir / f"{model}_{int(index):04d}_seed_{int(seed)}.csv"


def run_missing_5k_jobs(
    *,
    selected_table: pd.DataFrame,
    checkpoint_dir: Path,
    common: dict,
    seeds: list[int],
    workers: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, int, int, Path]] = []
    for row in selected_table.itertuples(index=False):
        model = str(row.truth_model)
        index = int(row.truth_lifetime_index)
        for seed in seeds:
            part = checkpoint_path(checkpoint_dir, model, index, int(seed))
            if not part.is_file():
                jobs.append((model, index, int(seed), part))

    if not jobs:
        print("All selected 5k truth/seed checkpoints already exist.", flush=True)
        return

    print(
        f"Running {len(jobs)} missing selected-5k truth/seed jobs ",
        f"with {workers} worker(s).",
        flush=True,
    )
    progress = ProgressMeter(total=len(jobs), label="selected")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=pilot.initialize_worker,
        initargs=(common,),
    ) as executor:
        futures = {
            executor.submit(
                pilot.simulate_truth_all_observables,
                model,
                index,
                seed,
            ): (model, index, seed, part)
            for model, index, seed, part in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            model, index, seed, part = futures[future]
            frame = future.result()
            temporary = part.with_suffix(part.suffix + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(part)
            print(
                f"COMPLETED {model:6s} index={index:3d} seed={seed} "
                f"({completed}/{len(jobs)}) | {progress.message(completed)}",
                flush=True,
            )


def load_selected_parts(
    *,
    selected_table: pd.DataFrame,
    checkpoint_dir: Path,
    seeds: list[int],
) -> pd.DataFrame:
    paths: list[Path] = []
    for row in selected_table.itertuples(index=False):
        for seed in seeds:
            paths.append(
                checkpoint_path(
                    checkpoint_dir,
                    str(row.truth_model),
                    int(row.truth_lifetime_index),
                    int(seed),
                )
            )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected-5k checkpoints.")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


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
        float(alpha_each),
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
    alpha_each = float(global_alpha) / float(comparisons)
    # Half-credit ties are possible. Flooring is conservative for this audit.
    successes = np.floor(
        omitted["correct_fraction"].to_numpy(dtype=float)
        * omitted["number_of_pseudoexperiments"].to_numpy(dtype=int)
        + 1.0e-12
    ).astype(int)
    omitted["successes_for_conservative_binomial_bound"] = successes
    omitted["audit_lower_bound"] = clopper_pearson_lower(
        successes,
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
                ["truth_model", "truth_lifetime_index"], keep="first"
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
            int(worst_rows.loc[(r.truth_model, r.truth_lifetime_index), "seed"])
            for r in promotions.itertuples()
        ]
        promotions["worst_number_of_events"] = [
            int(
                worst_rows.loc[
                    (r.truth_model, r.truth_lifetime_index),
                    "number_of_events",
                ]
            )
            for r in promotions.itertuples()
        ]
        promotions.sort_values(
            ["minimum_target_margin", "truth_model", "truth_lifetime_index"],
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
        "number_of_comparisons": comparisons,
        "alpha_per_comparison": float(alpha_each),
        "minimum_target_margin": float(omitted["target_margin"].min()),
        "number_of_failing_rows": int(omitted["requires_promotion"].sum()),
        "number_of_truths_requiring_promotion": int(len(promotions)),
    }
    return omitted, promotions, summary


def add_promotions_with_neighbours(
    *,
    selected_table: pd.DataFrame,
    promotions: pd.DataFrame,
    bank,
    round_index: int,
    neighbour_radius: int,
) -> pd.DataFrame:
    existing = {
        (str(row.truth_model), int(row.truth_lifetime_index))
        for row in selected_table.itertuples(index=False)
    }
    rows = selected_table.to_dict("records")
    for model in ("photon", "su2"):
        model_promotions = promotions[
            promotions["truth_model"] == model
        ]["truth_lifetime_index"].astype(int).tolist()
        if not model_promotions:
            continue
        intervals = np.asarray(
            getattr(bank, f"{model}_interval_index"), dtype=int
        )
        lifetimes = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
        expanded = same_interval_neighbours(
            indices=model_promotions,
            interval_index=intervals,
            radius=int(neighbour_radius),
        )
        for index in sorted(expanded):
            key = (model, int(index))
            if key in existing:
                continue
            rows.append(
                {
                    "truth_model": model,
                    "truth_lifetime_index": int(index),
                    "truth_interval_index": int(intervals[index]),
                    "truth_ctau_m": float(lifetimes[index]),
                    "minimum_gap_to_2k_envelope": None,
                    "selection_reasons": (
                        f"decision_audit_promotion_round_{int(round_index)}"
                    ),
                    "selection_round": int(round_index),
                }
            )
            existing.add(key)
    return pd.DataFrame(rows).sort_values(
        ["truth_model", "truth_lifetime_index"], ignore_index=True
    )


def nonmonotonic_steps(curve: pd.DataFrame, tolerance: float = 0.0) -> list[dict]:
    ordered = curve.sort_values("number_of_events")
    values = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    output: list[dict] = []
    for index in range(1, len(values)):
        change = float(values[index] - values[index - 1])
        if change < -float(tolerance):
            output.append(
                {
                    "from_N": int(counts[index - 1]),
                    "to_N": int(counts[index]),
                    "accuracy_change": change,
                }
            )
    return output


def main() -> None:
    args = parse_args()
    started = perf_counter()
    repo = Path.cwd().resolve()
    if not (repo / "analysis2").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")
    if args.pseudoexperiments <= 0 or args.chunk_size <= 0:
        raise ValueError("Pseudoexperiment and chunk sizes must be positive.")
    if not (0.0 < args.target_accuracy < 1.0):
        raise ValueError("--target-accuracy must lie between zero and one.")
    if not (0.0 < args.audit_global_alpha < 1.0):
        raise ValueError("--audit-global-alpha must lie between zero and one.")
    if args.hard_truth_gap < 0.0:
        raise ValueError("--hard-truth-gap must be non-negative.")

    full_dir = resolve(repo, args.full_domain_dir)
    bank_path = resolve(repo, args.bank_path)
    moments_path = resolve(repo, args.moments_path)
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (full_dir, bank_path, moments_path):
        if not path.exists():
            raise FileNotFoundError(path)

    bank = load_bank_light(bank_path)
    moments = load_moments_light(moments_path, bank)
    token = float_token(float(bank.mass_gev))

    summary_path = output_dir / f"selected_5k_audit_summary_ma_{token}.json"
    if summary_path.exists():
        raise FileExistsError(
            "This selected-5k audit is already complete. Preserve it and "
            f"use a new output directory: {summary_path}"
        )

    detailed_2k_path = (
        full_dir / f"conditional_feature_pilot_detailed_accuracy_ma_{token}.csv"
    )
    screening_path = (
        full_dir / f"conditional_feature_screening_truths_ma_{token}.csv"
    )
    distance_path = (
        full_dir / f"conditional_feature_distance_minima_ma_{token}.csv"
    )
    full_summary_path = (
        full_dir / f"conditional_feature_pilot_summary_ma_{token}.json"
    )
    for path in (
        detailed_2k_path,
        screening_path,
        distance_path,
        full_summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    detailed_2k = pd.read_csv(detailed_2k_path)
    detailed_2k = detailed_2k[
        detailed_2k["observable"] == str(args.observable)
    ].copy()
    screening_truths = pd.read_csv(screening_path)
    distance_minima = pd.read_csv(distance_path)
    full_summary = json.loads(full_summary_path.read_text())

    if not np.isclose(float(full_summary["mass_GeV"]), float(bank.mass_gev)):
        raise ValueError("Full-domain summary and bank mass do not agree.")
    if str(full_summary["selection_name"]) != str(bank.selection_name):
        raise ValueError("Full-domain summary and bank selection do not agree.")

    event_counts = np.asarray(sorted(set(args.event_counts)), dtype=int)
    selection_counts = np.asarray(
        sorted(set(args.selection_event_counts)), dtype=int
    )
    if np.any(event_counts <= 0) or np.any(selection_counts <= 0):
        raise ValueError("Event counts must be positive.")
    if not set(selection_counts.tolist()).issubset(
        set(detailed_2k["number_of_events"].astype(int).unique())
    ):
        raise ValueError("Selection event counts are missing from the 2k result.")
    if not set(event_counts.tolist()).issubset(
        set(detailed_2k["number_of_events"].astype(int).unique())
    ):
        raise ValueError("Requested 5k event counts must exist in the 2k result.")

    selected_table = build_initial_selection(
        detailed_2k=detailed_2k,
        screening_truths=screening_truths,
        distance_minima=distance_minima,
        bank=bank,
        observable=str(args.observable),
        selection_event_counts=selection_counts,
        hard_truth_gap=float(args.hard_truth_gap),
        top_per_model_and_count=int(args.top_per_model_and_count),
        neighbour_radius=int(args.neighbour_radius),
    )
    selected_path = output_dir / f"selected_5k_truths_ma_{token}.csv"
    if selected_path.exists():
        existing = pd.read_csv(selected_path)
        existing_keys = set(
            zip(
                existing["truth_model"].astype(str),
                existing["truth_lifetime_index"].astype(int),
            )
        )
        initial_keys = set(
            zip(
                selected_table["truth_model"].astype(str),
                selected_table["truth_lifetime_index"].astype(int),
            )
        )
        # A resumed run may already contain audit-promoted truths.  Accept a
        # strict superset, but never silently accept a table missing any of
        # the reproducibly selected initial truths.
        if not initial_keys.issubset(existing_keys):
            raise FileExistsError(
                "An incompatible selected-truth table already exists in the "
                "output directory. Use a new output directory."
            )
        selected_table = existing
    else:
        selected_table.to_csv(selected_path, index=False)

    counts_by_model = (
        selected_table.groupby("truth_model").size().to_dict()
    )
    print(
        "INITIAL SELECTED TRUTHS: "
        f"photon={int(counts_by_model.get('photon', 0))}, "
        f"su2={int(counts_by_model.get('su2', 0))}, "
        f"total={len(selected_table)}",
        flush=True,
    )
    if args.prepare_only:
        print(f"Selected-truth table: {selected_path}", flush=True)
        return

    # Delay the full EventCalc/analysis2 import until simulation is required.
    # This also keeps --prepare-only useful for lightweight audits.
    global pilot
    from analysis2.workflows import conditional_feature_pilot as pilot

    common = {
        "mass_gev": float(bank.mass_gev),
        "event_counts": event_counts,
        "number_of_pseudoexperiments": int(args.pseudoexperiments),
        "chunk_size": int(args.chunk_size),
        "observables": (str(args.observable),),
        "photon_probabilities": np.asarray(bank.photon_probabilities, dtype=float),
        "su2_probabilities": np.asarray(bank.su2_probabilities, dtype=float),
        "photon_ctau_m": np.asarray(bank.photon_ctau_m, dtype=float),
        "su2_ctau_m": np.asarray(bank.su2_ctau_m, dtype=float),
        "photon_feature_mean": moments["photon_feature_mean"],
        "photon_feature_covariance": moments[
            "photon_feature_covariance"
        ],
        "su2_feature_mean": moments["su2_feature_mean"],
        "su2_feature_covariance": moments["su2_feature_covariance"],
    }

    seeds = [int(seed) for seed in args.seeds]
    checkpoint_dir = output_dir / "truth_parts"
    audit_history: list[dict] = []
    final_audit_rows = pd.DataFrame()
    final_promotions = pd.DataFrame()

    for round_index in range(int(args.maximum_promotion_rounds) + 1):
        run_missing_5k_jobs(
            selected_table=selected_table,
            checkpoint_dir=checkpoint_dir,
            common=common,
            seeds=seeds,
            workers=int(args.workers),
        )
        detailed_5k = load_selected_parts(
            selected_table=selected_table,
            checkpoint_dir=checkpoint_dir,
            seeds=seeds,
        )
        detailed_5k = detailed_5k[
            detailed_5k["observable"] == str(args.observable)
        ].copy()
        curve_5k = conservative_curve(detailed_5k)
        threshold_5k = persistent_threshold(curve_5k, float(args.target_accuracy))
        if threshold_5k is None:
            raise RuntimeError(
                "Selected 5k curve does not reach the target on the event grid."
            )

        selected_keys = {
            (str(row.truth_model), int(row.truth_lifetime_index))
            for row in selected_table.itertuples(index=False)
        }
        audit_rows, promotions, audit_summary = decision_relevant_audit(
            detailed_2k=detailed_2k,
            selected_keys=selected_keys,
            candidate_threshold=int(threshold_5k),
            tested_event_counts=event_counts,
            target_accuracy=float(args.target_accuracy),
            global_alpha=float(args.audit_global_alpha),
        )
        audit_summary["round"] = int(round_index)
        audit_summary["number_of_selected_truths"] = int(len(selected_table))
        audit_history.append(audit_summary)

        round_audit_path = (
            output_dir
            / f"omitted_2k_decision_audit_round_{round_index}_ma_{token}.csv"
        )
        round_promotions_path = (
            output_dir
            / f"omitted_truth_promotions_round_{round_index}_ma_{token}.csv"
        )
        audit_rows.to_csv(round_audit_path, index=False)
        promotions.to_csv(round_promotions_path, index=False)
        final_audit_rows = audit_rows
        final_promotions = promotions

        print(
            f"AUDIT ROUND {round_index}: threshold={threshold_5k}, "
            f"promotions={len(promotions)}",
            flush=True,
        )
        if promotions.empty:
            break
        if round_index >= int(args.maximum_promotion_rounds):
            raise RuntimeError(
                "Decision audit still requires promotions after the maximum "
                "number of rounds. Preserve outputs and inspect manually."
            )

        selected_table = add_promotions_with_neighbours(
            selected_table=selected_table,
            promotions=promotions,
            bank=bank,
            round_index=round_index + 1,
            neighbour_radius=int(args.neighbour_radius),
        )
        selected_table.to_csv(selected_path, index=False)

    detailed_5k = load_selected_parts(
        selected_table=selected_table,
        checkpoint_dir=checkpoint_dir,
        seeds=seeds,
    )
    detailed_5k = detailed_5k[
        detailed_5k["observable"] == str(args.observable)
    ].copy()
    detailed_5k.sort_values(
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
    detailed_5k_path = (
        output_dir / f"selected_5k_detailed_accuracy_ma_{token}.csv"
    )
    detailed_5k.to_csv(detailed_5k_path, index=False)

    curve_5k = conservative_curve(detailed_5k)
    curve_5k_path = output_dir / f"selected_5k_curve_ma_{token}.csv"
    curve_5k.to_csv(curve_5k_path, index=False)
    threshold_5k = persistent_threshold(curve_5k, float(args.target_accuracy))

    curve_2k = conservative_curve(detailed_2k)
    threshold_2k = persistent_threshold(curve_2k, float(args.target_accuracy))
    comparison = curve_2k.rename(
        columns={"worst_case_accuracy": "full_domain_2k_accuracy"}
    ).merge(
        curve_5k.rename(
            columns={"worst_case_accuracy": "selected_5k_accuracy"}
        ),
        on="number_of_events",
        how="inner",
    )
    comparison["absolute_difference"] = np.abs(
        comparison["selected_5k_accuracy"]
        - comparison["full_domain_2k_accuracy"]
    )
    comparison.to_csv(
        output_dir / f"full_domain_2k_vs_selected_5k_ma_{token}.csv",
        index=False,
    )

    by_seed_rows: list[dict] = []
    for seed, subset in detailed_5k.groupby("seed"):
        seed_curve = conservative_curve(subset)
        by_seed_rows.append(
            {
                "seed": int(seed),
                "persistent_threshold": persistent_threshold(
                    seed_curve, float(args.target_accuracy)
                ),
            }
        )
    by_seed = pd.DataFrame(by_seed_rows).sort_values("seed")
    by_seed.to_csv(
        output_dir / f"selected_5k_threshold_by_seed_ma_{token}.csv",
        index=False,
    )

    limiting_rows: list[pd.DataFrame] = []
    for number_of_events in sorted(
        set([int(threshold_5k) - 1, int(threshold_5k), int(threshold_5k) + 1])
        & set(event_counts.tolist())
    ):
        subset = detailed_5k[
            detailed_5k["number_of_events"] == int(number_of_events)
        ].nsmallest(25, "correct_fraction")
        limiting_rows.append(subset)
    limiting = pd.concat(limiting_rows, ignore_index=True)
    limiting.to_csv(
        output_dir / f"selected_5k_limiting_rows_ma_{token}.csv",
        index=False,
    )

    final_audit_rows.to_csv(
        output_dir / f"omitted_2k_decision_audit_final_ma_{token}.csv",
        index=False,
    )
    final_promotions.to_csv(
        output_dir / f"omitted_truth_promotions_final_ma_{token}.csv",
        index=False,
    )
    selected_table.to_csv(selected_path, index=False)

    # Flag whether the selected 5k crossing itself is statistically marginal.
    at_threshold = detailed_5k[
        detailed_5k["number_of_events"] == int(threshold_5k)
    ]
    limiting_row = at_threshold.loc[at_threshold["correct_fraction"].idxmin()]
    successes = int(
        np.floor(
            float(limiting_row.correct_fraction)
            * int(limiting_row.number_of_pseudoexperiments)
            + 1.0e-12
        )
    )
    single_row_lower_99 = float(
        beta.ppf(
            0.01,
            successes,
            int(limiting_row.number_of_pseudoexperiments) - successes + 1,
        )
        if successes > 0
        else 0.0
    )
    recommend_10k = bool(
        float(limiting_row.correct_fraction) < float(args.target_accuracy) + 0.015
        or single_row_lower_99 < float(args.target_accuracy)
        or len(nonmonotonic_steps(curve_5k, tolerance=0.002)) > 0
    )

    summary = {
        "status": "selected_5k_with_decision_relevant_omitted_truth_audit",
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "observable": str(args.observable),
        "full_domain_dir": str(full_dir),
        "bank_path": str(bank_path),
        "moments_path": str(moments_path),
        "pseudoexperiments_per_selected_truth_and_seed": int(
            args.pseudoexperiments
        ),
        "seeds": seeds,
        "event_counts": [int(value) for value in event_counts],
        "selection_event_counts": [int(value) for value in selection_counts],
        "hard_truth_gap": float(args.hard_truth_gap),
        "neighbour_radius": int(args.neighbour_radius),
        "number_of_final_selected_truths": {
            model: int(
                (selected_table["truth_model"] == model).sum()
            )
            for model in ("photon", "su2")
        },
        "persistent_thresholds": {
            "full_domain_2k": threshold_2k,
            "selected_5k": threshold_5k,
        },
        "selected_5k_accuracy_near_threshold": {
            str(int(row.number_of_events)): float(row.worst_case_accuracy)
            for row in curve_5k[
                curve_5k["number_of_events"].isin(
                    [int(threshold_5k) - 1, int(threshold_5k), int(threshold_5k) + 1]
                )
            ].itertuples()
        },
        "limiting_truth_at_selected_5k_threshold": {
            "truth_model": str(limiting_row.truth_model),
            "truth_lifetime_index": int(limiting_row.truth_lifetime_index),
            "truth_ctau_m": float(limiting_row.truth_ctau_m),
            "seed": int(limiting_row.seed),
            "correct_fraction": float(limiting_row.correct_fraction),
            "single_row_one_sided_99pct_lower_bound": single_row_lower_99,
        },
        "threshold_by_seed": {
            str(int(row.seed)): (
                None
                if pd.isna(row.persistent_threshold)
                else int(row.persistent_threshold)
            )
            for row in by_seed.itertuples()
        },
        "curve_stability_2k_vs_5k": {
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
        "nonmonotonic_steps_larger_than_0p002": nonmonotonic_steps(
            curve_5k, tolerance=0.002
        ),
        "audit_history": audit_history,
        "final_omitted_truth_audit_passed": bool(final_promotions.empty),
        "recommend_selected_10k": recommend_10k,
        "remaining_publication_checks": [
            "Empirical conditional-resampling on the final difficult truth set.",
            "Selected 10k only if the 5k crossing is statistically marginal.",
            "Independent EventCalc/template proposal stream.",
            "Detector-resolution and reconstruction study beyond this project scope.",
        ],
        "runtime_seconds": float(perf_counter() - started),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.plot(
        comparison["number_of_events"],
        comparison["full_domain_2k_accuracy"],
        marker="o",
        markersize=3,
        label="Full domain, 2k",
    )
    axis.plot(
        comparison["number_of_events"],
        comparison["selected_5k_accuracy"],
        marker="o",
        markersize=3,
        label="Selected/audited truths, 5k",
    )
    axis.axhline(float(args.target_accuracy), linestyle="--", linewidth=1.1)
    axis.set_xlabel("Observed ALP decays, $N$")
    axis.set_ylabel("Worst-case correct-classification probability")
    axis.set_title(
        f"Selected 5k validation, $m_a={float(bank.mass_gev):g}$ GeV"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / f"selected_5k_vs_full_domain_2k_ma_{token}.pdf"
    )
    plt.close(figure)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
