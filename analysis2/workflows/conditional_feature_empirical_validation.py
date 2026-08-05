"""Empirical-resampling validation for conditional feature observables.

This workflow compares two paired truth generators while keeping the candidate
likelihood unchanged:

1. conditional multivariate Gaussian truth, used by the fast feature pilot;
2. direct weighted empirical resampling of the selected EventCalc events
   within exactly the same sampled energy bins.

The empirical draw selects a complete event-feature row, so correlations among
z, z^2 and r_perp are preserved. The comparison isolates the Gaussian
truth-generator approximation. It does not replace full-domain lifetime
auditing or validate the candidate Gaussian sample-mean likelihood by itself.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from analysis2.cache import CacheStore
from analysis2.conditional_features import (
    FEATURE_LABELS,
    FEATURE_SUBSETS,
    combine_source_feature_samples,
    load_conditional_feature_moments,
    profiled_feature_scores,
    selected_source_feature_sample,
    stable_feature_rng,
    validate_conditional_feature_moments,
)
from analysis2.config import get_config
from analysis2.eventcalc_adapter import EventCalcAdapter
from analysis2.lifetime_template_banks import load_template_bank
from analysis2.mass_seed_resolution import model_seed_for_bank
from analysis2.models import get_model
from analysis2.profiled_statistics import stable_truth_rng
from analysis2.workflows import float_token


MODEL_IDS = {
    "photon": "alp_photon_combined",
    "su2": "alp_su2l",
}
TRUTH_GENERATORS = ("gaussian_truth", "empirical_truth")
_WORKER: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--moments-path", type=Path, required=True)
    parser.add_argument("--selected-truths", type=Path, required=True)
    parser.add_argument("--observable", choices=tuple(FEATURE_SUBSETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--domain-path",
        type=Path,
        default=Path(
            "analysis2/outputs/production/week8_domains/"
            "allowed_ctau_domains.csv"
        ),
    )
    parser.add_argument("--pseudoexperiments", type=int, default=2000)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[73241, 83244, 93247, 103250, 113253],
    )
    parser.add_argument("--event-counts", nargs="+", type=int, required=True)
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--restart-checkpoint", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def persistent_threshold(
    curve: pd.DataFrame,
    target: float = 0.90,
) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix >= float(target))
    return None if len(passing) == 0 else int(counts[passing[0]])


def classify(statistic: np.ndarray, truth_model: str) -> np.ndarray:
    ties = np.abs(statistic) <= 1.0e-12
    su2_selected = statistic > 1.0e-12
    if truth_model == "photon":
        return (
            (~su2_selected & ~ties).sum(axis=0)
            + 0.5 * ties.sum(axis=0)
        )
    if truth_model == "su2":
        return su2_selected.sum(axis=0) + 0.5 * ties.sum(axis=0)
    raise ValueError(f"Unknown truth model: {truth_model}")


def _draw_gaussian_features(
    *,
    sampled_bins: np.ndarray,
    standard_normals: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    cholesky = np.linalg.cholesky(covariances)
    sampled = np.empty_like(standard_normals, dtype=float)
    for energy_bin in np.unique(sampled_bins):
        mask = sampled_bins == int(energy_bin)
        sampled[mask] = (
            means[int(energy_bin)]
            + standard_normals[mask] @ cholesky[int(energy_bin)].T
        )
    return sampled


def prepare_empirical_conditionals(
    *,
    adapter: EventCalcAdapter,
    bank,
    domain_path: Path,
    truth_model: str,
    truth_index: int,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float]:
    model_id = MODEL_IDS[truth_model]
    model = get_model(model_id)
    ctau_m = float(getattr(bank, f"{truth_model}_ctau_m")[truth_index])
    model_seed = model_seed_for_bank(
        config=adapter.config,
        bank=bank,
        model_id=model_id,
        domain_path=domain_path,
    )
    combined = combine_source_feature_samples(
        [
            selected_source_feature_sample(
                adapter=adapter,
                model_id=model_id,
                source_index=source_index,
                mass_gev=float(bank.mass_gev),
                ctau_m=ctau_m,
                model_seed=model_seed,
            )
            for source_index in range(len(model.sources))
        ]
    )

    features = combined.master_features
    weights = combined.weights
    edges = np.asarray(bank.energy_edges_gev, dtype=float)
    number_of_bins = len(edges) - 1
    bins = np.searchsorted(edges, combined.energy_gev, side="right") - 1
    bins = np.where(
        np.isclose(
            combined.energy_gev,
            edges[-1],
            rtol=0.0,
            atol=1.0e-12,
        ),
        number_of_bins - 1,
        bins,
    )
    valid = (
        (bins >= 0)
        & (bins < number_of_bins)
        & np.all(np.isfinite(features), axis=1)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    bins = bins[valid]
    features = features[valid]
    weights = weights[valid]

    sum_w = np.bincount(bins, weights=weights, minlength=number_of_bins)
    sum_w2 = np.bincount(
        bins,
        weights=np.square(weights),
        minlength=number_of_bins,
    )
    if np.any(sum_w <= 0.0):
        raise RuntimeError(
            "Empirical conditional sample has empty energy bins: "
            f"{np.flatnonzero(sum_w <= 0.0).tolist()}"
        )

    feature_rows: list[np.ndarray] = []
    cumulative_weights: list[np.ndarray] = []
    for energy_bin in range(number_of_bins):
        mask = bins == energy_bin
        rows = np.asarray(features[mask], dtype=float)
        bin_weights = np.asarray(weights[mask], dtype=float)
        order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
        rows = rows[order]
        bin_weights = bin_weights[order]
        cdf = np.cumsum(bin_weights / float(bin_weights.sum()))
        cdf[-1] = 1.0
        feature_rows.append(rows)
        cumulative_weights.append(cdf)

    raw_probability = sum_w / float(sum_w.sum())
    bank_probability = np.asarray(
        getattr(bank, f"{truth_model}_probabilities")[truth_index],
        dtype=float,
    )
    minimum_n_eff = float(np.min(np.square(sum_w) / sum_w2))
    maximum_probability_difference = float(
        np.max(np.abs(raw_probability - bank_probability))
    )
    return (
        feature_rows,
        cumulative_weights,
        maximum_probability_difference,
        minimum_n_eff,
    )


def draw_empirical_feature_rows(
    *,
    sampled_bins: np.ndarray,
    uniforms: np.ndarray,
    feature_rows: list[np.ndarray],
    cumulative_weights: list[np.ndarray],
) -> np.ndarray:
    sampled = np.empty(sampled_bins.shape + (3,), dtype=float)
    for energy_bin, (rows, cdf) in enumerate(
        zip(feature_rows, cumulative_weights)
    ):
        mask = sampled_bins == int(energy_bin)
        if not np.any(mask):
            continue
        indices = np.searchsorted(cdf, uniforms[mask], side="left")
        indices = np.minimum(indices, len(rows) - 1)
        sampled[mask] = rows[indices]
    return sampled


def initialize_worker(
    bank_path: str,
    moments_path: str,
    domain_path: str,
    common: dict,
) -> None:
    global _WORKER
    bank = load_template_bank(Path(bank_path))
    moments = load_conditional_feature_moments(Path(moments_path))
    validate_conditional_feature_moments(moments, bank)
    config = replace(
        get_config(bank.profile),
        selection_name=bank.selection_name,
    )
    adapter = EventCalcAdapter(
        config,
        cache=CacheStore(config.name),
        force=False,
    )
    _WORKER = {
        "bank": bank,
        "moments": moments,
        "domain_path": Path(domain_path),
        "adapter": adapter,
        "common": common,
    }


def simulate_truth(
    truth_model: str,
    truth_index: int,
) -> pd.DataFrame:
    if _WORKER is None:
        raise RuntimeError("Worker was not initialized.")

    bank = _WORKER["bank"]
    moments = _WORKER["moments"]
    adapter = _WORKER["adapter"]
    domain_path = _WORKER["domain_path"]
    common = _WORKER["common"]

    observable = str(common["observable"])
    feature_indices = FEATURE_SUBSETS[observable]
    if not feature_indices:
        raise ValueError("Empirical feature validation requires a feature observable.")

    event_counts = np.asarray(common["event_counts"], dtype=int)
    number_of_pes = int(common["number_of_pseudoexperiments"])
    chunk_size = int(common["chunk_size"])
    maximum_events = int(event_counts[-1])
    seeds = [int(seed) for seed in common["seeds"]]

    truth_probability = np.asarray(
        getattr(bank, f"{truth_model}_probabilities")[truth_index],
        dtype=float,
    )
    truth_mean = np.asarray(
        moments[f"{truth_model}_feature_mean"][truth_index],
        dtype=float,
    )
    truth_covariance = np.asarray(
        moments[f"{truth_model}_feature_covariance"][truth_index],
        dtype=float,
    )
    truth_ctau_m = float(
        getattr(bank, f"{truth_model}_ctau_m")[truth_index]
    )

    (
        empirical_rows,
        empirical_cdf,
        maximum_probability_difference,
        minimum_n_eff,
    ) = prepare_empirical_conditionals(
        adapter=adapter,
        bank=bank,
        domain_path=domain_path,
        truth_model=truth_model,
        truth_index=truth_index,
    )

    output_frames = []
    for seed in seeds:
        energy_rng = stable_truth_rng(
            seed=seed,
            mass_gev=float(bank.mass_gev),
            truth_model=truth_model,
            truth_index=truth_index,
        )
        feature_rng = stable_feature_rng(
            seed=seed,
            mass_gev=float(bank.mass_gev),
            truth_model=truth_model,
            truth_index=truth_index,
        )
        correct = {
            generator: np.zeros(len(event_counts), dtype=float)
            for generator in TRUTH_GENERATORS
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
            standard_normals = feature_rng.standard_normal(
                sampled_bins.shape + (3,)
            )
            gaussian_features = _draw_gaussian_features(
                sampled_bins=sampled_bins,
                standard_normals=standard_normals,
                means=truth_mean,
                covariances=truth_covariance,
            )
            empirical_features = draw_empirical_feature_rows(
                sampled_bins=sampled_bins,
                uniforms=ndtr(standard_normals[..., 0]),
                feature_rows=empirical_rows,
                cumulative_weights=empirical_cdf,
            )

            for generator, sampled_features in (
                ("gaussian_truth", gaussian_features),
                ("empirical_truth", empirical_features),
            ):
                observed = (
                    np.cumsum(sampled_features, axis=1)[:, event_counts - 1, :]
                    / event_counts[None, :, None]
                )[:, :, feature_indices]

                _, photon_best = profiled_feature_scores(
                    sampled_bins=sampled_bins,
                    observed_feature_means=observed,
                    probabilities=bank.photon_probabilities,
                    conditional_feature_mean=moments["photon_feature_mean"],
                    conditional_feature_covariance=moments[
                        "photon_feature_covariance"
                    ],
                    event_counts=event_counts,
                    feature_indices=feature_indices,
                )
                _, su2_best = profiled_feature_scores(
                    sampled_bins=sampled_bins,
                    observed_feature_means=observed,
                    probabilities=bank.su2_probabilities,
                    conditional_feature_mean=moments["su2_feature_mean"],
                    conditional_feature_covariance=moments[
                        "su2_feature_covariance"
                    ],
                    event_counts=event_counts,
                    feature_indices=feature_indices,
                )
                statistic = 2.0 * (su2_best - photon_best)
                correct[generator] += classify(statistic, truth_model)

            processed += current

        for generator in TRUTH_GENERATORS:
            output_frames.append(
                pd.DataFrame(
                    {
                        "mass_GeV": float(bank.mass_gev),
                        "selection_name": str(bank.selection_name),
                        "observable": observable,
                        "truth_generator": generator,
                        "seed": int(seed),
                        "truth_model": truth_model,
                        "truth_lifetime_index": int(truth_index),
                        "truth_ctau_m": truth_ctau_m,
                        "number_of_events": event_counts,
                        "number_of_pseudoexperiments": number_of_pes,
                        "correct_fraction": (
                            correct[generator] / float(number_of_pes)
                        ),
                        "minimum_raw_bin_feature_N_eff": minimum_n_eff,
                        (
                            "maximum_absolute_raw_vs_bank_"
                            "probability_difference"
                        ): maximum_probability_difference,
                    }
                )
            )
    return pd.concat(output_frames, ignore_index=True)


def conservative_curves(detailed: pd.DataFrame) -> pd.DataFrame:
    return (
        detailed.groupby(
            ["truth_generator", "number_of_events"],
            as_index=False,
        )["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values(
            ["truth_generator", "number_of_events"],
            ignore_index=True,
        )
    )


def main() -> None:
    args = parse_args()
    started = perf_counter()
    repo = Path.cwd().resolve()
    if not (repo / "analysis2").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")
    if args.pseudoexperiments <= 0 or args.chunk_size <= 0:
        raise ValueError("Pseudoexperiment and chunk sizes must be positive.")

    bank_path = resolve(repo, args.bank_path)
    moments_path = resolve(repo, args.moments_path)
    selected_path = resolve(repo, args.selected_truths)
    domain_path = resolve(repo, args.domain_path)
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (bank_path, moments_path, selected_path, domain_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    bank = load_template_bank(bank_path)
    token = float_token(float(bank.mass_gev))
    summary_path = (
        output_dir
        / f"conditional_feature_empirical_summary_ma_{token}.json"
    )
    if summary_path.exists() and not args.restart_checkpoint:
        raise FileExistsError(
            f"Completed result exists: {summary_path}. "
            "Use a new output directory."
        )

    selected = pd.read_csv(selected_path)
    required = {"truth_model", "truth_lifetime_index"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(
            f"Selected-truth table is missing: {sorted(missing)}"
        )
    selected = (
        selected.loc[:, ["truth_model", "truth_lifetime_index"]]
        .drop_duplicates()
        .sort_values(["truth_model", "truth_lifetime_index"])
        .reset_index(drop=True)
    )
    selected["truth_model"] = selected["truth_model"].astype(str)
    selected["truth_lifetime_index"] = selected[
        "truth_lifetime_index"
    ].astype(int)

    event_counts = np.asarray(
        sorted(set(int(value) for value in args.event_counts)),
        dtype=int,
    )
    if np.any(event_counts <= 0):
        raise ValueError("Event counts must be positive.")

    checkpoint_dir = output_dir / "truth_parts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.restart_checkpoint:
        for path in checkpoint_dir.glob("*.csv"):
            path.unlink()

    part_paths: list[Path] = []
    tasks = []
    for row in selected.itertuples(index=False):
        model = str(row.truth_model)
        index = int(row.truth_lifetime_index)
        if model not in MODEL_IDS:
            raise ValueError(f"Unknown truth model: {model}")
        part = checkpoint_dir / f"{model}_{index:04d}.csv"
        part_paths.append(part)
        if not part.is_file():
            tasks.append((model, index, part))

    common = {
        "observable": str(args.observable),
        "event_counts": event_counts,
        "number_of_pseudoexperiments": int(args.pseudoexperiments),
        "chunk_size": int(args.chunk_size),
        "seeds": [int(seed) for seed in args.seeds],
    }

    print(
        f"EMPIRICAL FEATURE VALIDATION: mass={bank.mass_gev:g} GeV, "
        f"selection={bank.selection_name}, observable={args.observable}, "
        f"truths={len(selected)}, remaining={len(tasks)}",
        flush=True,
    )

    if tasks:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(
                str(bank_path),
                str(moments_path),
                str(domain_path),
                common,
            ),
        ) as executor:
            futures = {
                executor.submit(simulate_truth, model, index): (
                    model,
                    index,
                    part,
                )
                for model, index, part in tasks
            }
            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                model, index, part = futures[future]
                frame = future.result()
                temporary = part.with_suffix(part.suffix + ".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(part)
                print(
                    f"COMPLETED {model:6s} index={index:3d} "
                    f"({completed}/{len(tasks)})",
                    flush=True,
                )

    missing_parts = [path for path in part_paths if not path.is_file()]
    if missing_parts:
        raise RuntimeError(
            f"Missing {len(missing_parts)} truth checkpoint files."
        )
    detailed = pd.concat(
        (pd.read_csv(path) for path in part_paths),
        ignore_index=True,
    )
    detailed.to_csv(
        output_dir
        / f"conditional_feature_empirical_detailed_ma_{token}.csv",
        index=False,
    )

    curves = conservative_curves(detailed)
    curves.to_csv(
        output_dir
        / f"conditional_feature_empirical_curves_ma_{token}.csv",
        index=False,
    )

    thresholds = {
        generator: persistent_threshold(
            curves[curves["truth_generator"] == generator]
        )
        for generator in TRUTH_GENERATORS
    }
    comparison = (
        curves[curves["truth_generator"] == "gaussian_truth"]
        .drop(columns="truth_generator")
        .rename(columns={"worst_case_accuracy": "gaussian_truth"})
        .merge(
            curves[curves["truth_generator"] == "empirical_truth"]
            .drop(columns="truth_generator")
            .rename(columns={"worst_case_accuracy": "empirical_truth"}),
            on="number_of_events",
            how="inner",
        )
    )
    comparison["signed_empirical_minus_gaussian"] = (
        comparison["empirical_truth"] - comparison["gaussian_truth"]
    )
    comparison["absolute_difference"] = np.abs(
        comparison["signed_empirical_minus_gaussian"]
    )
    comparison.to_csv(
        output_dir
        / f"conditional_feature_empirical_comparison_ma_{token}.csv",
        index=False,
    )

    limiting_frames = []
    for generator, threshold in thresholds.items():
        if threshold is None:
            continue
        subset = detailed[
            (detailed["truth_generator"] == generator)
            & (detailed["number_of_events"] == int(threshold))
        ]
        limiting_frames.append(subset.nsmallest(20, "correct_fraction"))
    limiting = (
        pd.concat(limiting_frames, ignore_index=True)
        if limiting_frames
        else pd.DataFrame()
    )
    limiting.to_csv(
        output_dir
        / f"conditional_feature_empirical_limiting_ma_{token}.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(7.8, 5.0))
    labels = {
        "gaussian_truth": "Conditional Gaussian truth",
        "empirical_truth": "Empirical conditional truth",
    }
    for generator in TRUTH_GENERATORS:
        subset = curves[
            curves["truth_generator"] == generator
        ].sort_values("number_of_events")
        label = labels[generator]
        threshold = thresholds[generator]
        if threshold is not None:
            label += rf" ($N_{{90}}={threshold}$)"
        axis.plot(
            subset["number_of_events"],
            subset["worst_case_accuracy"],
            marker="o",
            markersize=3,
            label=label,
        )
    axis.axhline(0.9, linestyle="--", linewidth=1.1, label="90% target")
    axis.set_xlabel("Observed ALP decays, $N$")
    axis.set_ylabel("Worst-case correct-classification probability")
    axis.set_title(
        FEATURE_LABELS[args.observable]
        + f", $m_a={bank.mass_gev:g}$ GeV"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir
        / f"conditional_feature_empirical_accuracy_ma_{token}.pdf"
    )
    plt.close(figure)

    summary = {
        "status": "conditional_feature_empirical_truth_validation",
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "observable": str(args.observable),
        "bank_path": str(bank_path),
        "moments_path": str(moments_path),
        "selected_truths_path": str(selected_path),
        "domain_path": str(domain_path),
        "pseudoexperiments_per_truth_and_seed": int(
            args.pseudoexperiments
        ),
        "seeds": [int(seed) for seed in args.seeds],
        "event_counts": [int(value) for value in event_counts],
        "number_of_truths": int(len(selected)),
        "persistent_thresholds": thresholds,
        "curve_difference": {
            "mean_absolute_difference": float(
                comparison["absolute_difference"].mean()
            ),
            "maximum_absolute_difference": float(
                comparison["absolute_difference"].max()
            ),
            "minimum_signed_empirical_minus_gaussian": float(
                comparison["signed_empirical_minus_gaussian"].min()
            ),
        },
        "raw_sample_quality": {
            "minimum_bin_feature_N_eff": float(
                detailed["minimum_raw_bin_feature_N_eff"].min()
            ),
            "maximum_absolute_raw_vs_bank_probability_difference": float(
                detailed[
                    "maximum_absolute_raw_vs_bank_probability_difference"
                ].max()
            ),
        },
        "runtime_seconds": float(perf_counter() - started),
        "interpretation": (
            "This validates the Gaussian truth generator for the selected "
            "truth set. A full-domain audit and independent template seed "
            "remain required before publication use."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
