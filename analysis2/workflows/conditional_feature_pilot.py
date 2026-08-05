"""Focused comparison of richer z information and transverse displacement.

This workflow is deliberately a screening pilot, not a final N90 calculation.
For one existing lifetime template bank it:

1. builds conditional moments of (u_z, u_z^2, rho_perp) in every energy bin;
2. creates Gaussian-Hellinger distance maps for several feature subsets;
3. selects difficult truths from every disconnected interval combination;
4. runs paired low-statistics pseudoexperiments for all candidate observables;
5. writes provisional discrimination curves and a recommendation table.

The same sampled energy bins and full feature-vector draws are used for every
observable, so differences between curves are paired.  A winning observable
must subsequently pass empirical-resampling, seed, binning and full-domain
validations before entering the final mass scan.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis2.conditional_features import (
    FEATURE_LABELS,
    FEATURE_SUBSETS,
    build_conditional_feature_moments,
    pairwise_joint_energy_feature_hellinger_squared,
    load_conditional_feature_moments,
    profiled_feature_scores,
    sample_master_features,
    stable_feature_rng,
    validate_conditional_feature_moments,
)
from analysis2.lifetime_template_banks import load_template_bank
from analysis2.profiled_statistics import stable_truth_rng
from analysis2.workflows import float_token


DEFAULT_EVENT_COUNTS = np.asarray(
    [10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 130, 170, 220, 300],
    dtype=int,
)
DEFAULT_OBSERVABLES = (
    "energy",
    "energy_mean_z",
    "energy_mean_z_spread",
    "energy_mean_r_perp",
    "energy_mean_z_r_perp",
    "energy_mean_z_spread_r_perp",
)

_WORKER_COMMON: dict | None = None


@dataclass(frozen=True)
class TruthKey:
    model: str
    index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument(
        "--domain-path",
        type=Path,
        default=Path(
            "analysis2/outputs/production/week8_domains/"
            "allowed_ctau_domains.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
        default=DEFAULT_EVENT_COUNTS.tolist(),
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        choices=tuple(FEATURE_SUBSETS),
        default=list(DEFAULT_OBSERVABLES),
    )
    parser.add_argument("--pairs-per-interval", type=int, default=4)
    parser.add_argument("--neighbour-radius", type=int, default=1)
    parser.add_argument("--reuse-moments", action="store_true")
    parser.add_argument(
        "--moments-only",
        action="store_true",
        help="Build and validate feature moments, then stop.",
    )
    parser.add_argument("--restart-checkpoint", action="store_true")
    return parser.parse_args()


def persistent_threshold(curve: pd.DataFrame, target: float = 0.9) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix >= float(target))
    return None if len(passing) == 0 else int(counts[passing[0]])


def conservative_curves(detailed: pd.DataFrame) -> pd.DataFrame:
    return (
        detailed.groupby(
            ["observable", "number_of_events"],
            as_index=False,
        )["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values(["observable", "number_of_events"], ignore_index=True)
    )


def initialize_worker(common: dict) -> None:
    global _WORKER_COMMON
    _WORKER_COMMON = common


def simulate_truth_all_observables(
    truth_model: str,
    truth_index: int,
    seed: int,
) -> pd.DataFrame:
    if _WORKER_COMMON is None:
        raise RuntimeError("Feature-pilot worker was not initialized.")
    common = _WORKER_COMMON
    event_counts = np.asarray(common["event_counts"], dtype=int)
    maximum_events = int(event_counts[-1])
    number_of_pes = int(common["number_of_pseudoexperiments"])
    chunk_size = int(common["chunk_size"])
    observables = tuple(common["observables"])

    truth_probability = np.asarray(
        common[f"{truth_model}_probabilities"][truth_index],
        dtype=float,
    )
    truth_mean = np.asarray(
        common[f"{truth_model}_feature_mean"][truth_index],
        dtype=float,
    )
    truth_covariance = np.asarray(
        common[f"{truth_model}_feature_covariance"][truth_index],
        dtype=float,
    )
    truth_ctau = float(common[f"{truth_model}_ctau_m"][truth_index])

    energy_rng = stable_truth_rng(
        seed=int(seed),
        mass_gev=float(common["mass_gev"]),
        truth_model=truth_model,
        truth_index=int(truth_index),
    )
    feature_rng = stable_feature_rng(
        seed=int(seed),
        mass_gev=float(common["mass_gev"]),
        truth_model=truth_model,
        truth_index=int(truth_index),
    )

    correct = {
        observable: np.zeros(len(event_counts), dtype=float)
        for observable in observables
    }
    processed = 0
    while processed < number_of_pes:
        current = min(chunk_size, number_of_pes - processed)
        sampled_bins = energy_rng.choice(
            len(truth_probability),
            size=(current, maximum_events),
            replace=True,
            p=truth_probability,
        )
        sampled_features = sample_master_features(
            sampled_bins=sampled_bins,
            truth_mean=truth_mean,
            truth_covariance=truth_covariance,
            rng=feature_rng,
        )
        cumulative_features = np.cumsum(sampled_features, axis=1)

        for observable in observables:
            feature_indices = FEATURE_SUBSETS[observable]
            if feature_indices:
                observed = (
                    cumulative_features[:, event_counts - 1, :]
                    / event_counts[None, :, None]
                )[:, :, feature_indices]
            else:
                observed = np.empty((current, len(event_counts), 0), dtype=float)

            photon_energy, photon_combined = profiled_feature_scores(
                sampled_bins=sampled_bins,
                observed_feature_means=observed,
                probabilities=common["photon_probabilities"],
                conditional_feature_mean=common["photon_feature_mean"],
                conditional_feature_covariance=common[
                    "photon_feature_covariance"
                ],
                event_counts=event_counts,
                feature_indices=feature_indices,
            )
            su2_energy, su2_combined = profiled_feature_scores(
                sampled_bins=sampled_bins,
                observed_feature_means=observed,
                probabilities=common["su2_probabilities"],
                conditional_feature_mean=common["su2_feature_mean"],
                conditional_feature_covariance=common[
                    "su2_feature_covariance"
                ],
                event_counts=event_counts,
                feature_indices=feature_indices,
            )
            photon_best = photon_energy if not feature_indices else photon_combined
            su2_best = su2_energy if not feature_indices else su2_combined
            statistic = 2.0 * (su2_best - photon_best)
            ties = np.abs(statistic) <= 1.0e-12
            su2_selected = statistic > 1.0e-12
            if truth_model == "photon":
                correct[observable] += (
                    (~su2_selected & ~ties).sum(axis=0)
                    + 0.5 * ties.sum(axis=0)
                )
            else:
                correct[observable] += (
                    su2_selected.sum(axis=0)
                    + 0.5 * ties.sum(axis=0)
                )
        processed += current

    frames = []
    for observable, values in correct.items():
        frames.append(
            pd.DataFrame(
                {
                    "mass_GeV": float(common["mass_gev"]),
                    "seed": int(seed),
                    "truth_model": truth_model,
                    "truth_lifetime_index": int(truth_index),
                    "truth_ctau_m": truth_ctau,
                    "observable": observable,
                    "number_of_events": event_counts,
                    "number_of_pseudoexperiments": number_of_pes,
                    "correct_fraction": values / float(number_of_pes),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_distance_maps(
    *,
    bank,
    moments: Mapping[str, np.ndarray],
    observables: tuple[str, ...],
) -> dict[str, np.ndarray]:
    maps = {}
    for observable in observables:
        maps[observable] = pairwise_joint_energy_feature_hellinger_squared(
            photon_probabilities=bank.photon_probabilities,
            photon_means=moments["photon_feature_mean"],
            photon_covariances=moments["photon_feature_covariance"],
            su2_probabilities=bank.su2_probabilities,
            su2_means=moments["su2_feature_mean"],
            su2_covariances=moments["su2_feature_covariance"],
            feature_indices=FEATURE_SUBSETS[observable],
        )
    return maps


def screening_truths_from_maps(
    *,
    bank,
    maps: Mapping[str, np.ndarray],
    pairs_per_interval: int,
    neighbour_radius: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    selected = {"photon": set(), "su2": set()}
    reasons: dict[tuple[str, int], set[str]] = {}

    def add(model: str, index: int, reason: str) -> None:
        lifetimes = getattr(bank, f"{model}_ctau_m")
        clipped = int(np.clip(index, 0, len(lifetimes) - 1))
        selected[model].add(clipped)
        reasons.setdefault((model, clipped), set()).add(reason)

    for model in ("photon", "su2"):
        intervals = np.asarray(getattr(bank, f"{model}_interval_index"), dtype=int)
        for interval in np.unique(intervals):
            indices = np.flatnonzero(intervals == interval)
            add(model, int(indices[0]), f"interval_{interval}_lower_endpoint")
            add(model, int(indices[-1]), f"interval_{interval}_upper_endpoint")

    photon_intervals = np.asarray(bank.photon_interval_index, dtype=int)
    su2_intervals = np.asarray(bank.su2_interval_index, dtype=int)
    for observable, values in maps.items():
        for photon_interval in np.unique(photon_intervals):
            photon_indices = np.flatnonzero(
                photon_intervals == photon_interval
            )
            for su2_interval in np.unique(su2_intervals):
                su2_indices = np.flatnonzero(su2_intervals == su2_interval)
                submap = values[np.ix_(photon_indices, su2_indices)]
                order = np.argsort(submap, axis=None)
                for flat in order[: int(pairs_per_interval)]:
                    local_photon, local_su2 = np.unravel_index(
                        int(flat), submap.shape
                    )
                    photon_index = int(photon_indices[local_photon])
                    su2_index = int(su2_indices[local_su2])
                    for offset in range(-neighbour_radius, neighbour_radius + 1):
                        add(
                            "photon",
                            photon_index + offset,
                            f"{observable}_difficult_pair",
                        )
                        add(
                            "su2",
                            su2_index + offset,
                            f"{observable}_difficult_pair",
                        )

    rows = []
    arrays = {}
    for model in ("photon", "su2"):
        indices = np.asarray(sorted(selected[model]), dtype=int)
        arrays[model] = indices
        lifetimes = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
        intervals = np.asarray(
            getattr(bank, f"{model}_interval_index"), dtype=int
        )
        for index in indices:
            rows.append(
                {
                    "truth_model": model,
                    "truth_lifetime_index": int(index),
                    "truth_interval_index": int(intervals[index]),
                    "truth_ctau_m": float(lifetimes[index]),
                    "selection_reasons": ";".join(
                        sorted(reasons[(model, int(index))])
                    ),
                }
            )
    return pd.DataFrame(rows), arrays


def plot_distance_map(
    *,
    bank,
    values: np.ndarray,
    observable: str,
    output_path: Path,
) -> None:
    photon_intervals = np.asarray(bank.photon_interval_index, dtype=int)
    su2_intervals = np.asarray(bank.su2_interval_index, dtype=int)
    interval_pairs = [
        (photon_interval, su2_interval)
        for photon_interval in np.unique(photon_intervals)
        for su2_interval in np.unique(su2_intervals)
    ]
    figure, axes = plt.subplots(
        len(interval_pairs),
        1,
        figsize=(7.2, 4.2 * len(interval_pairs)),
        squeeze=False,
    )
    for axis, (photon_interval, su2_interval) in zip(
        axes[:, 0], interval_pairs
    ):
        photon_indices = np.flatnonzero(
            photon_intervals == photon_interval
        )
        su2_indices = np.flatnonzero(su2_intervals == su2_interval)
        image = axis.pcolormesh(
            np.log10(bank.photon_ctau_m[photon_indices]),
            np.log10(bank.su2_ctau_m[su2_indices]),
            values[np.ix_(photon_indices, su2_indices)].T,
            shading="auto",
        )
        figure.colorbar(image, ax=axis, label=r"$H^2$ diagnostic proxy")
        axis.set_xlabel(r"$\log_{10}(c\tau_\gamma/\mathrm{m})$")
        axis.set_ylabel(r"$\log_{10}(c\tau_{\mathrm{SU(2)}_L}/\mathrm{m})$")
        axis.set_title(
            f"Photon interval {photon_interval}; "
            f"SU(2) interval {su2_interval}"
        )
    figure.suptitle(
        FEATURE_LABELS[observable]
        + f", $m_a={bank.mass_gev:g}$ GeV",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = perf_counter()
    repo = Path.cwd().resolve()
    if not (repo / "analysis2").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")
    if args.pseudoexperiments <= 0 or args.chunk_size <= 0:
        raise ValueError("Pseudoexperiment and chunk sizes must be positive.")
    event_counts = np.asarray(sorted(set(args.event_counts)), dtype=int)
    if np.any(event_counts <= 0):
        raise ValueError("Event counts must be positive.")
    observables = tuple(dict.fromkeys(args.observables))

    bank_path = args.bank_path.expanduser().resolve()
    domain_path = args.domain_path.expanduser().resolve()
    bank = load_template_bank(bank_path)
    if not domain_path.is_file():
        raise FileNotFoundError(
            f"Week-8 domain table not found: {domain_path}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    token = float_token(float(bank.mass_gev))
    summary_path = output_dir / f"conditional_feature_pilot_summary_ma_{token}.json"
    if summary_path.exists() and not args.restart_checkpoint:
        raise FileExistsError(
            "This feature-pilot result is complete. Preserve it and use a "
            f"new output directory: {summary_path}"
        )

    moment_filename = f"conditional_feature_moments_ma_{token}.npz"
    moment_path = output_dir / moment_filename
    quality_path = output_dir / moment_filename.replace(
        ".npz", "_quality.csv"
    )
    if args.reuse_moments:
        if not moment_path.is_file() or not quality_path.is_file():
            raise FileNotFoundError("--reuse-moments requested but files are missing.")
        moments = load_conditional_feature_moments(moment_path)
        quality = pd.read_csv(quality_path)
    else:
        if moment_path.exists():
            raise FileExistsError(
                f"Moment file already exists. Use --reuse-moments: {moment_path}"
            )
        moments, quality = build_conditional_feature_moments(
            bank=bank,
            output_dir=output_dir,
            filename=moment_filename,
            domain_path=domain_path,
        )
    validate_conditional_feature_moments(moments, bank)
    if args.moments_only:
        print(
            "FEATURE MOMENTS VALIDATED: "
            f"minimum bin N_eff="
            f"{quality['minimum_bin_feature_N_eff'].min():.3f}, "
            f"maximum |p_raw-p_bank|="
            f"{quality['maximum_absolute_raw_vs_bank_probability_difference'].max():.6g}",
            flush=True,
        )
        return

    maps = build_distance_maps(
        bank=bank,
        moments=moments,
        observables=observables,
    )
    minima_rows = []
    for observable, values in maps.items():
        photon_index, su2_index = np.unravel_index(
            int(np.argmin(values)), values.shape
        )
        minima_rows.append(
            {
                "observable": observable,
                "minimum_H2": float(values[photon_index, su2_index]),
                "photon_lifetime_index": int(photon_index),
                "photon_ctau_m": float(bank.photon_ctau_m[photon_index]),
                "su2_lifetime_index": int(su2_index),
                "su2_ctau_m": float(bank.su2_ctau_m[su2_index]),
            }
        )
        plot_distance_map(
            bank=bank,
            values=values,
            observable=observable,
            output_path=(
                output_dir
                / f"distance_map_{observable}_ma_{token}.pdf"
            ),
        )
    minima = pd.DataFrame(minima_rows).sort_values("minimum_H2")
    minima.to_csv(
        output_dir / f"conditional_feature_distance_minima_ma_{token}.csv",
        index=False,
    )

    screening_table, screening = screening_truths_from_maps(
        bank=bank,
        maps=maps,
        pairs_per_interval=int(args.pairs_per_interval),
        neighbour_radius=int(args.neighbour_radius),
    )
    screening_table.to_csv(
        output_dir / f"conditional_feature_screening_truths_ma_{token}.csv",
        index=False,
    )

    common = {
        "mass_gev": float(bank.mass_gev),
        "event_counts": event_counts,
        "number_of_pseudoexperiments": int(args.pseudoexperiments),
        "chunk_size": int(args.chunk_size),
        "observables": observables,
        "photon_probabilities": np.asarray(
            bank.photon_probabilities, dtype=float
        ),
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

    checkpoint_dir = output_dir / "truth_parts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.restart_checkpoint:
        for path in checkpoint_dir.glob("*.csv"):
            path.unlink()

    jobs = []
    part_paths = []
    for model in ("photon", "su2"):
        for index in screening[model]:
            for seed in args.seeds:
                part_path = checkpoint_dir / (
                    f"{model}_{int(index):04d}_seed_{int(seed)}.csv"
                )
                part_paths.append(part_path)
                if not part_path.is_file():
                    jobs.append((model, int(index), int(seed), part_path))

    print(
        f"FEATURE PILOT: mass={bank.mass_gev:g} GeV, "
        f"selection={bank.selection_name}, "
        f"screening truths={len(screening['photon'])}+{len(screening['su2'])}, "
        f"jobs={len(jobs)}, observables={list(observables)}",
        flush=True,
    )
    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(common,),
        ) as executor:
            futures = {
                executor.submit(
                    simulate_truth_all_observables,
                    model,
                    index,
                    seed,
                ): (model, index, seed, part_path)
                for model, index, seed, part_path in jobs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                model, index, seed, part_path = futures[future]
                frame = future.result()
                temporary = part_path.with_suffix(part_path.suffix + ".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(part_path)
                print(
                    f"COMPLETED {model:6s} index={index:3d} "
                    f"seed={seed} ({completed}/{len(jobs)})",
                    flush=True,
                )

    missing = [path for path in part_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} truth checkpoint files.")
    detailed = pd.concat(
        (pd.read_csv(path) for path in part_paths),
        ignore_index=True,
    )
    detailed.to_csv(
        output_dir / f"conditional_feature_pilot_detailed_accuracy_ma_{token}.csv",
        index=False,
    )
    curves = conservative_curves(detailed)
    curves.to_csv(
        output_dir / f"conditional_feature_pilot_curves_ma_{token}.csv",
        index=False,
    )

    threshold_rows = []
    for observable, subset in curves.groupby("observable"):
        threshold_rows.append(
            {
                "observable": observable,
                "provisional_persistent_threshold": persistent_threshold(subset),
                "minimum_H2": float(
                    minima.loc[
                        minima["observable"] == observable,
                        "minimum_H2",
                    ].iloc[0]
                ),
            }
        )
    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["provisional_persistent_threshold", "minimum_H2"],
        na_position="last",
    )
    thresholds.to_csv(
        output_dir / f"conditional_feature_pilot_thresholds_ma_{token}.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for observable, subset in curves.groupby("observable"):
        axis.plot(
            subset["number_of_events"],
            subset["worst_case_accuracy"],
            marker="o",
            markersize=3,
            label=FEATURE_LABELS[observable],
        )
    axis.axhline(0.9, linestyle="--", linewidth=1.1, label="90% target")
    axis.set_xlabel("Observed ALP decays, $N$")
    axis.set_ylabel("Worst-case correct-classification probability")
    axis.set_title(
        f"Conditional-feature screening, $m_a={bank.mass_gev:g}$ GeV"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(
        output_dir / f"conditional_feature_pilot_accuracy_ma_{token}.pdf"
    )
    plt.close(figure)

    summary = {
        "status": "focused_conditional_feature_screening_pilot",
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "bank_path": str(bank_path),
        "domain_path": str(domain_path),
        "observables": list(observables),
        "pseudoexperiments_per_truth_and_seed": int(args.pseudoexperiments),
        "seeds": [int(seed) for seed in args.seeds],
        "event_counts": [int(value) for value in event_counts],
        "number_of_screening_truths": {
            "photon": int(len(screening["photon"])),
            "su2": int(len(screening["su2"])),
        },
        "conditional_feature_quality": {
            "minimum_bin_N_eff": float(
                quality["minimum_bin_feature_N_eff"].min()
            ),
            "minimum_covariance_eigenvalue": float(
                quality["minimum_covariance_eigenvalue"].min()
            ),
            "maximum_absolute_raw_vs_bank_probability_difference": float(
                quality[
                    "maximum_absolute_raw_vs_bank_probability_difference"
                ].max()
            ),
        },
        "provisional_thresholds": {
            row.observable: (
                None
                if pd.isna(row.provisional_persistent_threshold)
                else int(row.provisional_persistent_threshold)
            )
            for row in thresholds.itertuples()
        },
        "distance_minima": {
            row.observable: {
                "minimum_H2": float(row.minimum_H2),
                "photon_ctau_m": float(row.photon_ctau_m),
                "su2_ctau_m": float(row.su2_ctau_m),
            }
            for row in minima.itertuples()
        },
        "runtime_seconds": float(perf_counter() - started),
        "interpretation_guardrails": [
            "All thresholds are screening values based on selected difficult truths.",
            "The Gaussian feature-vector truth generator is an approximation.",
            "The winning observable must pass empirical conditional resampling.",
            "A full-domain 2k plus selected 5k/10k audit is still required.",
            "Distance maps are diagnostic proxies, not the project test statistic.",
        ],
        "next_action": (
            "Compare feature gains at the validated 0.3 and 0.5 GeV anchors. "
            "Choose at most two candidate observables for empirical-resampling "
            "validation before any new full mass scan."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
