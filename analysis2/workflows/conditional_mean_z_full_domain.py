"""Generic full-domain 2k / five-seed scan for energy + conditional mean-z.

This reusable stage consumes one completed generic range-finder point. It
profiles every allowed photon and SU(2)_L truth lifetime against both complete
lifetime banks, using the validated conditional-Gaussian mean-z likelihood.

The run is a full-domain screening layer, not normally the final high-statistic
N90 result. It produces:

* the conservative discrimination curve and seed-specific thresholds;
* a compact selected-truth table for the subsequent 5k/10k ladder;
* connected-component-safe energy-TV and joint-Hellinger diagnostic maps;
* per-truth checkpoints and online runtime projections.

The joint Hellinger map is a diagnostic proxy for p(E) Normal(z|E). It is not
the project test statistic and must not be reported as D_TV.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (73241, 83244, 93247, 103250, 113253)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rangefinder-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-script-dir", type=Path, default=Path.home() / "Downloads"
    )
    parser.add_argument("--pseudoexperiments", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument(
        "--event-count-grid",
        type=str,
        help=(
            "Optional explicit N grid, e.g. '25:60,70,85,110,145'. By "
            "default a compact full-domain grid is derived from the range finder."
        ),
    )
    parser.add_argument("--unit-below-crossing", type=int, default=15)
    parser.add_argument("--unit-above-crossing", type=int, default=20)
    parser.add_argument("--tail-points", type=int, default=5)
    parser.add_argument("--hard-truth-gap", type=float, default=0.03)
    parser.add_argument("--hard-window-half-width", type=int, default=10)
    parser.add_argument("--restart-checkpoint", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()



def load_pilot(script_dir: Path):
    """Return the package-native validated implementation."""
    del script_dir
    from analysis2 import conditional_mean_z
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
            counts.add(stop)
        else:
            raise ValueError(f"Invalid event-count token: {token}")
    result = np.asarray(sorted(counts), dtype=int)
    if len(result) == 0 or np.any(result < 1):
        raise ValueError("The event-count grid must contain positive integers.")
    return result


def compact_grid_from_rangefinder(
    summary: dict,
    *,
    below: int,
    above: int,
    tail_points: int,
) -> np.ndarray:
    range_info = summary["rangefinder"]
    centre = int(range_info.get("estimated_crossing_events") or 1)
    suggested = np.asarray(range_info["suggested_final_event_grid"], dtype=int)
    unit_lower = max(1, centre - int(below))
    unit_upper = centre + int(above)
    unit = np.arange(unit_lower, unit_upper + 1, dtype=int)

    candidates = suggested[suggested > unit_upper]
    if len(candidates) == 0:
        candidates = np.asarray([max(unit_upper + 10, int(ceil(1.8 * centre)))])
    count = max(1, int(tail_points))
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)))
    tail = candidates[np.unique(np.rint(positions).astype(int))]
    return np.unique(np.concatenate([unit, tail])).astype(int)


def persistent_threshold(curve: pd.DataFrame, target: float = 0.90) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix >= float(target))
    return None if len(passing) == 0 else int(counts[passing[0]])


def add_neighbours(indices: set[int], size: int) -> list[int]:
    expanded = set(int(index) for index in indices)
    for index in tuple(expanded):
        if index > 0:
            expanded.add(index - 1)
        if index + 1 < size:
            expanded.add(index + 1)
    return sorted(expanded)


def select_hard_truths(
    detailed: pd.DataFrame,
    bank,
    *,
    threshold: int | None,
    gap: float,
    half_width: int,
) -> pd.DataFrame:
    combined = detailed.copy()
    if threshold is None:
        maximum = int(combined["number_of_events"].max())
        minimum = max(1, maximum - 2 * int(half_width))
    else:
        minimum = max(1, int(threshold) - int(half_width))
        maximum = int(threshold) + int(half_width)
    combined = combined[combined["number_of_events"].between(minimum, maximum)].copy()
    envelope = (
        combined.groupby("number_of_events")["correct_fraction"]
        .min()
        .rename("envelope_accuracy")
    )
    combined = combined.join(envelope, on="number_of_events")
    combined["gap_to_envelope"] = (
        combined["correct_fraction"] - combined["envelope_accuracy"]
    )
    base_rows = combined[combined["gap_to_envelope"] <= float(gap)]
    base = {
        model: set(
            base_rows.loc[
                base_rows["truth_model"] == model, "truth_lifetime_index"
            ].astype(int)
        )
        for model in ("photon", "su2")
    }
    selected = {
        "photon": add_neighbours(base["photon"], len(bank.photon_ctau_m)),
        "su2": add_neighbours(base["su2"], len(bank.su2_ctau_m)),
    }
    rows: list[dict] = []
    for model in ("photon", "su2"):
        lifetimes = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
        intervals = np.asarray(getattr(bank, f"{model}_interval_index"), dtype=int)
        for index in selected[model]:
            subset = combined[
                (combined["truth_model"] == model)
                & (combined["truth_lifetime_index"].astype(int) == int(index))
            ]
            rows.append(
                {
                    "mass_GeV": float(bank.mass_gev),
                    "truth_model": model,
                    "truth_lifetime_index": int(index),
                    "truth_interval_index": int(intervals[index]),
                    "truth_ctau_m": float(lifetimes[index]),
                    "was_within_gap_before_neighbours": bool(index in base[model]),
                    "minimum_gap_to_2k_envelope": (
                        None if subset.empty else float(subset["gap_to_envelope"].min())
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["truth_model", "truth_lifetime_index"], ignore_index=True
    )


def total_variation_matrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(
        np.abs(np.asarray(p, dtype=float)[:, None, :] - np.asarray(q, dtype=float)[None, :, :]),
        axis=2,
    )


def joint_hellinger_squared(
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
    photon_mean: np.ndarray,
    photon_variance: np.ndarray,
    su2_mean: np.ndarray,
    su2_variance: np.ndarray,
) -> np.ndarray:
    p = np.asarray(photon_probabilities, dtype=float)
    q = np.asarray(su2_probabilities, dtype=float)
    mp = np.asarray(photon_mean, dtype=float)
    vp = np.asarray(photon_variance, dtype=float)
    mq = np.asarray(su2_mean, dtype=float)
    vq = np.asarray(su2_variance, dtype=float)
    result = np.empty((len(p), len(q)), dtype=float)
    for index in range(len(p)):
        variance_sum = vp[index][None, :] + vq
        gaussian_bc = np.sqrt(
            2.0 * np.sqrt(vp[index][None, :] * vq) / variance_sum
        ) * np.exp(-np.square(mp[index][None, :] - mq) / (4.0 * variance_sum))
        coefficient = np.sum(np.sqrt(p[index][None, :] * q) * gaussian_bc, axis=1)
        result[index] = np.clip(1.0 - coefficient, 0.0, 1.0)
    return result


def interval_slices(interval_index: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(interval_index, dtype=int)
    return [np.flatnonzero(values == interval) for interval in np.unique(values)]


def plot_component_distance_map(
    matrix: np.ndarray,
    bank,
    *,
    output_path: Path,
    title: str,
    label: str,
) -> dict:
    photon_parts = interval_slices(bank.photon_interval_index)
    su2_parts = interval_slices(bank.su2_interval_index)
    figure, axes = plt.subplots(
        len(su2_parts),
        len(photon_parts),
        figsize=(5.2 * len(photon_parts), 4.3 * len(su2_parts)),
        squeeze=False,
        constrained_layout=True,
    )
    global_index = np.unravel_index(int(np.argmin(matrix)), matrix.shape)
    image = None
    for row, su2_indices in enumerate(su2_parts):
        for column, photon_indices in enumerate(photon_parts):
            ax = axes[row, column]
            block = matrix[np.ix_(photon_indices, su2_indices)].T
            x = np.log10(np.asarray(bank.photon_ctau_m)[photon_indices])
            y = np.log10(np.asarray(bank.su2_ctau_m)[su2_indices])
            image = ax.imshow(
                block,
                origin="lower",
                aspect="auto",
                extent=[x.min(), x.max(), y.min(), y.max()],
                interpolation="nearest",
            )
            if global_index[0] in photon_indices and global_index[1] in su2_indices:
                ax.scatter(
                    [np.log10(bank.photon_ctau_m[global_index[0]])],
                    [np.log10(bank.su2_ctau_m[global_index[1]])],
                    marker="x",
                    s=60,
                    linewidths=1.5,
                )
            ax.set_xlabel(r"$\log_{10}(c\tau_\gamma/\mathrm{m})$")
            ax.set_ylabel(r"$\log_{10}(c\tau_{SU(2)_L}/\mathrm{m})$")
            ax.set_title(f"Photon interval {column}; SU(2) interval {row}")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label=label)
    figure.suptitle(title)
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "minimum": float(matrix[global_index]),
        "photon_lifetime_index": int(global_index[0]),
        "photon_ctau_m": float(bank.photon_ctau_m[global_index[0]]),
        "su2_lifetime_index": int(global_index[1]),
        "su2_ctau_m": float(bank.su2_ctau_m[global_index[1]]),
    }


_WORKER_PILOT = None
_WORKER_COMMON = None


def initialise_worker(pilot_script_dir: str, common: dict) -> None:
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
                "truth_model": truth_model,
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


def main() -> None:
    args = parse_args()
    start_total = perf_counter()
    repo = Path.cwd().resolve()
    if not (repo / "analysis2").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")
    if args.pseudoexperiments < 1 or args.chunk_size < 1:
        raise ValueError("Pseudoexperiment count and chunk size must be positive.")
    if args.hard_truth_gap < 0.0:
        raise ValueError("--hard-truth-gap must be non-negative.")

    pilot = load_pilot(args.pilot_script_dir)
    summary_path = resolve(repo, args.rangefinder_summary)
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text())
    bank_path = resolve(repo, Path(summary["bank_path"]))
    moments_path = resolve(repo, Path(summary["conditional_moments_path"]))
    bank = pilot.load_template_bank(bank_path)
    arrays = pilot.load_conditional_moments(moments_path)
    pilot.validate_conditional_moments(arrays, bank)

    from analysis2.workflows import float_token

    token = float_token(float(bank.mass_gev))
    final_summary_path = output_dir / f"full_domain_2k_summary_ma_{token}.json"
    if final_summary_path.is_file() and not args.restart_checkpoint:
        raise FileExistsError(
            "This point is already complete. Preserve it and use a new output "
            "directory: " + str(final_summary_path)
        )

    if args.event_count_grid:
        event_counts = parse_event_counts(args.event_count_grid)
        grid_source = "explicit_cli"
    else:
        event_counts = compact_grid_from_rangefinder(
            summary,
            below=args.unit_below_crossing,
            above=args.unit_above_crossing,
            tail_points=args.tail_points,
        )
        grid_source = "compact_from_rangefinder"
    (output_dir / f"full_domain_event_grid_ma_{token}.txt").write_text(
        ",".join(str(int(value)) for value in event_counts) + "\n"
    )

    checkpoint_dir = output_dir / "full_domain_truth_parts"
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
        for index in range(len(lifetimes)):
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

    total_truths = len(bank.photon_ctau_m) + len(bank.su2_ctau_m)
    already_done = total_truths - len(jobs)
    print(
        f"FULL DOMAIN: mass={bank.mass_gev:g} GeV, selection={bank.selection_name}, "
        f"truths={total_truths}, completed={already_done}, remaining={len(jobs)}, "
        f"N-grid={event_counts.tolist()}",
        flush=True,
    )

    truth_durations: list[float] = []
    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialise_worker,
            initargs=(str(args.pilot_script_dir.expanduser().resolve()), common),
        ) as executor:
            futures = {}
            submitted_at = {}
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
                submitted_at[future] = perf_counter()

            for completed, future in enumerate(as_completed(futures), start=1):
                model, index, part = futures[future]
                frame = future.result()
                temporary = part.with_suffix(part.suffix + ".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(part)
                duration = perf_counter() - submitted_at[future]
                truth_durations.append(duration)
                overall_done = already_done + completed
                message = (
                    f"COMPLETED {model:6s} index={index:3d} "
                    f"({overall_done}/{total_truths})"
                )
                if truth_durations:
                    elapsed = perf_counter() - start_total
                    rate = elapsed / max(completed, 1)
                    remaining = len(jobs) - completed
                    projected = remaining * rate
                    message += f" | elapsed={elapsed/60:.1f} min, rough remaining={projected/60:.1f} min"
                print(message, flush=True)

    missing = [path for path in all_parts if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} truth checkpoints; rerun the same command.")

    detailed = pd.concat((pd.read_csv(path) for path in all_parts), ignore_index=True)
    detailed.sort_values(
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
    detailed_path = output_dir / f"full_domain_2k_detailed_accuracy_ma_{token}.csv"
    detailed.to_csv(detailed_path, index=False)

    curve = (
        detailed.groupby("number_of_events", as_index=False)["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values("number_of_events", ignore_index=True)
    )
    curve.to_csv(
        output_dir / f"full_domain_2k_conservative_curve_ma_{token}.csv", index=False
    )
    model_curves = (
        detailed.groupby(["truth_model", "number_of_events"], as_index=False)[
            "correct_fraction"
        ]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
    )
    model_curves.to_csv(
        output_dir / f"full_domain_2k_model_curves_ma_{token}.csv", index=False
    )

    threshold = persistent_threshold(curve)
    by_seed_rows = []
    for seed, subset in detailed.groupby("seed"):
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
        output_dir / f"full_domain_2k_threshold_by_seed_ma_{token}.csv", index=False
    )

    hard = select_hard_truths(
        detailed,
        bank,
        threshold=threshold,
        gap=float(args.hard_truth_gap),
        half_width=int(args.hard_window_half_width),
    )
    hard.to_csv(
        output_dir / f"full_domain_2k_selected_truths_ma_{token}.csv", index=False
    )

    limiting = pd.DataFrame()
    if threshold is not None:
        limiting = (
            detailed[detailed["number_of_events"] == int(threshold)]
            .nsmallest(20, "correct_fraction")
            .copy()
        )
    limiting.to_csv(
        output_dir / f"full_domain_2k_limiting_points_ma_{token}.csv", index=False
    )

    energy_tv = total_variation_matrix(
        bank.photon_probabilities, bank.su2_probabilities
    )
    joint_h2 = joint_hellinger_squared(
        bank.photon_probabilities,
        bank.su2_probabilities,
        arrays["photon_mean_z_by_energy_bin_m"],
        arrays["photon_variance_z_by_energy_bin_m2"],
        arrays["su2_mean_z_by_energy_bin_m"],
        arrays["su2_variance_z_by_energy_bin_m2"],
    )
    energy_map_summary = plot_component_distance_map(
        energy_tv,
        bank,
        output_path=output_dir / f"distance_map_energy_tv_ma_{token}.pdf",
        title=f"Energy-only total-variation map, m={bank.mass_gev:g} GeV",
        label=r"$D_{TV}$",
    )
    joint_map_summary = plot_component_distance_map(
        joint_h2,
        bank,
        output_path=output_dir / f"distance_map_joint_hellinger_ma_{token}.pdf",
        title=f"Energy + conditional-z Hellinger proxy, m={bank.mass_gev:g} GeV",
        label=r"$H^2$ diagnostic proxy",
    )

    fig, ax = plt.subplots(figsize=(8.3, 5.3))
    for model, subset in model_curves.groupby("truth_model"):
        ax.plot(
            subset["number_of_events"],
            subset["worst_case_accuracy"],
            linewidth=1.4,
            marker="o",
            markersize=3,
            label=f"Worst {model} truth",
        )
    ax.plot(
        curve["number_of_events"],
        curve["worst_case_accuracy"],
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="Overall worst case",
    )
    ax.axhline(0.9, linestyle="--", linewidth=1.0, label="90% target")
    if threshold is not None:
        ax.axvline(int(threshold), linestyle=":", linewidth=1.0, label=f"2k N90={threshold}")
    ax.set_xlabel("Observed ALP decays, N")
    ax.set_ylabel("Worst-case correct-classification probability")
    ax.set_ylim(max(0.0, float(curve["worst_case_accuracy"].min()) - 0.04), 1.01)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"full_domain_2k_accuracy_ma_{token}.pdf")
    plt.close(fig)

    elapsed = perf_counter() - start_total
    durations = np.asarray(truth_durations, dtype=float)
    result = {
        "status": "generic_full_domain_2k_5seed_screen",
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "bank_path": str(bank_path),
        "conditional_moments_path": str(moments_path),
        "rangefinder_summary_path": str(summary_path),
        "pseudoexperiments_per_truth_and_seed": int(args.pseudoexperiments),
        "seeds": [int(seed) for seed in args.seeds],
        "event_counts": [int(value) for value in event_counts],
        "event_grid_source": grid_source,
        "number_of_truth_lifetimes": {
            "photon": int(len(bank.photon_ctau_m)),
            "su2": int(len(bank.su2_ctau_m)),
        },
        "number_of_energy_bins": int(bank.number_of_energy_bins),
        "persistent_threshold_all_truths_and_seeds": threshold,
        "number_of_selected_truths_for_next_stage": {
            "photon": int((hard["truth_model"] == "photon").sum()),
            "su2": int((hard["truth_model"] == "su2").sum()),
        },
        "distance_map_diagnostics": {
            "energy_total_variation": energy_map_summary,
            "joint_hellinger_proxy": joint_map_summary,
        },
        "runtime": {
            "elapsed_seconds_this_invocation": float(elapsed),
            "new_truths_completed_this_invocation": int(len(truth_durations)),
            "median_observed_completion_seconds": (
                None if len(durations) == 0 else float(np.median(durations))
            ),
            "note": (
                "Per-future completion times overlap under two workers and are only "
                "a rough throughput diagnostic. Use total elapsed time per completed "
                "mass-selection point for multi-mass projections."
            ),
        },
        "next_action": (
            "Run the selected truths at 5k on a unit-spaced crossing window, then "
            "audit every omitted 2k truth. The 2k threshold is a screening result."
        ),
    }
    final_summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print("\n" + json.dumps(result, indent=2), flush=True)
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
