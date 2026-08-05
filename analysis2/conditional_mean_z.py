"""Reusable conditional-mean-z likelihood and template utilities.

The script first builds conditional z moments for every model/lifetime/energy-bin:
    E[z | energy bin], Var(z | energy bin).

It then runs focused pseudoexperiments in which energy bins and decay positions
are generated coherently. For each candidate lifetime, the expected sample mean
and its variance are conditioned on the actually observed energy-bin sequence.

This retains the main E-z correlation while still using only the sample mean
<z>, as requested. It is a proof of principle, not yet the final full-domain
adaptive scan.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Iterable
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis2.cache import CacheStore
from analysis2.config import get_config
from analysis2.eventcalc_adapter import EventCalcAdapter
from analysis2.eventcalc_proposals import generate_mother_sample
from analysis2.lifetime_template_banks import load_template_bank
from analysis2.mass_seed_resolution import (
    DEFAULT_WEEK8_DOMAIN_PATH,
    model_seed_for_bank,
)
from analysis2.models import get_model
from analysis2.profiled_statistics import stable_truth_rng
from analysis2.selections import SelectionContext
from analysis2.spectra import effective_sample_size
from analysis2.workflows.lifetime_blind_discrimination import (
    proposal_lifetime_for_target,
)


MODEL_SPECS = (
    ("alp_photon_combined", "photon"),
    ("alp_su2l", "su2"),
)

PHOTON_TARGETS_M = (
    0.01,
    0.0976138631732929,
    0.5348894979243304,
)
SU2_TARGETS_M = (
    0.1011540227544267,
    0.17423143995019894,
    76.40704096934792,
    89.65883827377026,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "analysis2/outputs/validation/week8_mean_z_pilot_ma0p3_geom/"
            "mean_z_summary_ma_0p3.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis2/outputs/validation/"
            "week8_energy_plus_mean_z_conditional_pilot_ma0p3_geom"
        ),
    )
    parser.add_argument("--pseudoexperiments", type=int, default=500)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[73241, 113253],
    )
    parser.add_argument("--minimum-events", type=int, default=2)
    parser.add_argument("--maximum-events", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument(
        "--reuse-moments",
        action="store_true",
        help="Reuse an existing conditional_z_moments_ma_0p3.npz.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def nearest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(np.log(values) - np.log(float(target)))))


def resolve_truths(bank) -> dict[str, np.ndarray]:
    return {
        "photon": np.asarray(
            sorted(
                {
                    nearest_index(bank.photon_ctau_m, target)
                    for target in PHOTON_TARGETS_M
                }
            ),
            dtype=int,
        ),
        "su2": np.asarray(
            sorted(
                {
                    nearest_index(bank.su2_ctau_m, target)
                    for target in SU2_TARGETS_M
                }
            ),
            dtype=int,
        ),
    }


def selected_source_sample(
    *,
    adapter: EventCalcAdapter,
    model_id: str,
    source_index: int,
    mass_gev: float,
    ctau_m: float,
    model_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = get_model(model_id)
    source = model.sources[source_index]
    seed_policy = adapter.config.seed_policy
    proposal_seed = seed_policy.source_proposal_seed_from_model_seed(
        model_seed,
        source_index,
    )
    true_sample_seed = seed_policy.true_sample_seed_from_model_seed(
        model_seed,
        source_index,
    )
    proposal = adapter.prepare_kinematic_proposal(
        model,
        source,
        mass_gev,
        proposal_seed,
        "spectrum",
        proposal_ctau_m=proposal_lifetime_for_target(ctau_m),
    )
    mothers = generate_mother_sample(proposal, ctau_m, true_sample_seed)
    context = SelectionContext(
        source_seed=proposal_seed,
        true_sample_seed=true_sample_seed,
    )
    mask = np.asarray(adapter.selection.mask(mothers, context), dtype=bool)
    if mask.shape != (len(mothers),):
        raise ValueError("Selection mask has the wrong shape.")

    coupling_squared = proposal.unit_coupling_ctau_m / ctau_m
    n_llp_total = (
        adapter.config.exposure_pot
        * proposal.yield_per_pot_per_coupling_squared
        * coupling_squared
    )
    scale = (
        n_llp_total
        * proposal.epsilon_polar
        * proposal.visible_br
        / proposal.resample_size
    )
    energies = np.asarray(mothers.energy_gev[mask], dtype=float)
    z_m = np.asarray(mothers.z_m[mask], dtype=float)
    weights = np.asarray(
        scale * mothers.decay_probability[mask],
        dtype=float,
    )
    if (
        len(energies) == 0
        or energies.shape != z_m.shape
        or z_m.shape != weights.shape
        or float(weights.sum()) <= 0.0
    ):
        raise RuntimeError(
            f"Invalid selected sample for {model_id}, c*tau={ctau_m:g} m."
        )
    return energies, z_m, weights


def conditional_moments_for_lifetime(
    *,
    adapter: EventCalcAdapter,
    bank,
    model_id: str,
    prefix: str,
    lifetime_index: int,
    domain_path: Path = DEFAULT_WEEK8_DOMAIN_PATH,
) -> dict[str, np.ndarray | float | int | str]:
    ctau_m = float(getattr(bank, f"{prefix}_ctau_m")[lifetime_index])
    model_seed = model_seed_for_bank(
        config=adapter.config,
        bank=bank,
        model_id=model_id,
        domain_path=domain_path,
    )
    model = get_model(model_id)

    energy_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for source_index in range(len(model.sources)):
        energies, z_m, weights = selected_source_sample(
            adapter=adapter,
            model_id=model_id,
            source_index=source_index,
            mass_gev=bank.mass_gev,
            ctau_m=ctau_m,
            model_seed=model_seed,
        )
        energy_parts.append(energies)
        z_parts.append(z_m)
        weight_parts.append(weights)

    energies = np.concatenate(energy_parts)
    z_m = np.concatenate(z_parts)
    weights = np.concatenate(weight_parts)
    edges = np.asarray(bank.energy_edges_gev, dtype=float)
    number_of_bins = len(edges) - 1

    indices = np.searchsorted(edges, energies, side="right") - 1
    indices = np.where(
        np.isclose(energies, edges[-1], rtol=0.0, atol=1.0e-12),
        number_of_bins - 1,
        indices,
    )
    valid = (
        (indices >= 0)
        & (indices < number_of_bins)
        & np.isfinite(z_m)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    indices = indices[valid]
    z_m = z_m[valid]
    weights = weights[valid]

    sum_w = np.bincount(indices, weights=weights, minlength=number_of_bins)
    sum_w2 = np.bincount(
        indices,
        weights=np.square(weights),
        minlength=number_of_bins,
    )
    sum_wz = np.bincount(
        indices,
        weights=weights * z_m,
        minlength=number_of_bins,
    )
    sum_wz2 = np.bincount(
        indices,
        weights=weights * np.square(z_m),
        minlength=number_of_bins,
    )
    if np.any(sum_w <= 0.0):
        missing = np.flatnonzero(sum_w <= 0.0).tolist()
        raise RuntimeError(
            f"Raw z sample has empty adaptive bins for {prefix} "
            f"lifetime index {lifetime_index}: {missing}"
        )

    mean = sum_wz / sum_w
    variance = np.maximum(sum_wz2 / sum_w - np.square(mean), 1.0e-10)
    n_eff = np.divide(
        np.square(sum_w),
        sum_w2,
        out=np.zeros_like(sum_w),
        where=sum_w2 > 0.0,
    )
    return {
        "model_prefix": prefix,
        "model_id": model_id,
        "lifetime_index": int(lifetime_index),
        "ctau_m": ctau_m,
        "mean_z_by_energy_bin_m": mean,
        "variance_z_by_energy_bin_m2": variance,
        "n_eff_by_energy_bin": n_eff,
        "raw_probability_by_energy_bin": sum_w / float(sum_w.sum()),
    }


def build_conditional_moments(
    *,
    bank,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    config = replace(
        get_config(bank.profile),
        selection_name=bank.selection_name,
    )
    adapter = EventCalcAdapter(
        config,
        cache=CacheStore(config.name),
        force=False,
    )

    arrays: dict[str, np.ndarray] = {
        "energy_edges_GeV": np.asarray(bank.energy_edges_gev, dtype=float),
    }
    rows: list[dict] = []
    for model_id, prefix in MODEL_SPECS:
        lifetimes = np.asarray(getattr(bank, f"{prefix}_ctau_m"), dtype=float)
        means = []
        variances = []
        n_effs = []
        raw_probabilities = []
        for index in range(len(lifetimes)):
            result = conditional_moments_for_lifetime(
                adapter=adapter,
                bank=bank,
                model_id=model_id,
                prefix=prefix,
                lifetime_index=index,
            )
            means.append(result["mean_z_by_energy_bin_m"])
            variances.append(result["variance_z_by_energy_bin_m2"])
            n_effs.append(result["n_eff_by_energy_bin"])
            raw_probabilities.append(result["raw_probability_by_energy_bin"])
            rows.append(
                {
                    "model_prefix": prefix,
                    "lifetime_index": index,
                    "ctau_m": float(lifetimes[index]),
                    "minimum_bin_z_N_eff": float(
                        np.min(result["n_eff_by_energy_bin"])
                    ),
                    "maximum_absolute_raw_vs_bank_probability_difference": float(
                        np.max(
                            np.abs(
                                result["raw_probability_by_energy_bin"]
                                - getattr(bank, f"{prefix}_probabilities")[index]
                            )
                        )
                    ),
                }
            )
            print(
                f"{prefix:6s} {index + 1:3d}/{len(lifetimes):3d} "
                f"ctau={lifetimes[index]:.6g} m"
            )

        arrays[f"{prefix}_mean_z_by_energy_bin_m"] = np.asarray(means)
        arrays[f"{prefix}_variance_z_by_energy_bin_m2"] = np.asarray(variances)
        arrays[f"{prefix}_z_n_eff_by_energy_bin"] = np.asarray(n_effs)
        arrays[f"{prefix}_raw_probability_by_energy_bin"] = np.asarray(
            raw_probabilities
        )

    np.savez_compressed(
        output_dir / "conditional_z_moments_ma_0p3.npz",
        **arrays,
    )
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "conditional_z_moment_quality_ma_0p3.csv",
        index=False,
    )
    return arrays, table


def load_conditional_moments(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def validate_conditional_moments(arrays: dict[str, np.ndarray], bank) -> None:
    edges = arrays["energy_edges_GeV"]
    if not np.allclose(
        edges,
        bank.energy_edges_gev,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Conditional moments use different energy edges.")
    for prefix in ("photon", "su2"):
        expected = (
            len(getattr(bank, f"{prefix}_ctau_m")),
            bank.number_of_energy_bins,
        )
        mean = arrays[f"{prefix}_mean_z_by_energy_bin_m"]
        variance = arrays[f"{prefix}_variance_z_by_energy_bin_m2"]
        if mean.shape != expected or variance.shape != expected:
            raise ValueError(f"{prefix} conditional moments have wrong shape.")
        if (
            np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(variance))
            or np.any(variance <= 0.0)
        ):
            raise ValueError(f"{prefix} conditional moments are invalid.")


def stable_z_rng(
    *,
    seed: int,
    mass_gev: float,
    truth_model: str,
    truth_index: int,
) -> np.random.Generator:
    mass_hash = zlib.crc32(f"{mass_gev:.16g}".encode("ascii"))
    model_code = 0 if truth_model == "photon" else 1
    sequence = np.random.SeedSequence(
        [int(seed), int(mass_hash), model_code, int(truth_index), 0xE2A7]
    )
    return np.random.default_rng(sequence)


def profiled_scores(
    *,
    sampled_bins: np.ndarray,
    observed_mean_z: np.ndarray,
    probabilities: np.ndarray,
    conditional_mean_z: np.ndarray,
    conditional_variance_z: np.ndarray,
    event_counts: np.ndarray,
    target_bytes: int = 18 * 1024**2,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    means = np.asarray(conditional_mean_z, dtype=float)
    variances = np.asarray(conditional_variance_z, dtype=float)
    counts = np.asarray(event_counts, dtype=int)
    event_indices = counts - 1

    number_of_pes, maximum_events = sampled_bins.shape
    bytes_per_pe = (
        probabilities.shape[0]
        * maximum_events
        * np.dtype(float).itemsize
    )
    block_size = max(
        1,
        min(number_of_pes, target_bytes // max(bytes_per_pe * 3, 1)),
    )
    energy_best = np.empty((number_of_pes, len(counts)), dtype=float)
    combined_best = np.empty_like(energy_best)
    log_probabilities = np.log(probabilities)

    for start in range(0, number_of_pes, block_size):
        stop = min(number_of_pes, start + block_size)
        bins = sampled_bins[start:stop]

        energy_contributions = log_probabilities[:, bins]
        mean_contributions = means[:, bins]
        variance_contributions = variances[:, bins]

        np.cumsum(energy_contributions, axis=2, out=energy_contributions)
        np.cumsum(mean_contributions, axis=2, out=mean_contributions)
        np.cumsum(variance_contributions, axis=2, out=variance_contributions)

        for column, (count, event_index) in enumerate(
            zip(counts, event_indices)
        ):
            energy_at_count = energy_contributions[:, :, event_index]
            energy_best[start:stop, column] = np.max(
                energy_at_count,
                axis=0,
            )

            predicted_mean = (
                mean_contributions[:, :, event_index] / float(count)
            )
            variance_of_mean = (
                variance_contributions[:, :, event_index]
                / float(count * count)
            )
            residual = (
                observed_mean_z[start:stop, column][None, :]
                - predicted_mean
            )
            log_mean_z = -0.5 * (
                np.square(residual) / variance_of_mean
                + np.log(2.0 * np.pi * variance_of_mean)
            )
            combined_best[start:stop, column] = np.max(
                energy_at_count + log_mean_z,
                axis=0,
            )

    return energy_best, combined_best


def simulate_truth(task: dict) -> pd.DataFrame:
    truth_model = task["truth_model"]
    truth_index = int(task["truth_index"])
    truth_probabilities = np.asarray(
        task["truth_probabilities"],
        dtype=float,
    )
    truth_conditional_mean = np.asarray(
        task["truth_conditional_mean_z"],
        dtype=float,
    )
    truth_conditional_variance = np.asarray(
        task["truth_conditional_variance_z"],
        dtype=float,
    )
    event_counts = np.asarray(task["event_counts"], dtype=int)
    maximum_events = int(event_counts[-1])
    number_of_pes = int(task["number_of_pseudoexperiments"])
    chunk_size = int(task["chunk_size"])
    seed = int(task["seed"])

    correct = {
        "energy": np.zeros(len(event_counts), dtype=float),
        "conditional_combined": np.zeros(len(event_counts), dtype=float),
    }

    energy_rng = stable_truth_rng(
        seed=seed,
        mass_gev=float(task["mass_gev"]),
        truth_model=truth_model,
        truth_index=truth_index,
    )
    z_rng = stable_z_rng(
        seed=seed,
        mass_gev=float(task["mass_gev"]),
        truth_model=truth_model,
        truth_index=truth_index,
    )

    processed = 0
    while processed < number_of_pes:
        current = min(chunk_size, number_of_pes - processed)
        sampled_bins = energy_rng.choice(
            len(truth_probabilities),
            size=(current, maximum_events),
            replace=True,
            p=truth_probabilities,
        )
        per_event_mean = truth_conditional_mean[sampled_bins]
        per_event_std = np.sqrt(
            truth_conditional_variance[sampled_bins]
        )
        sampled_z = (
            per_event_mean
            + per_event_std
            * z_rng.standard_normal((current, maximum_events))
        )
        observed_mean_z = (
            np.cumsum(sampled_z, axis=1)[:, event_counts - 1]
            / event_counts[None, :]
        )

        photon_energy, photon_combined = profiled_scores(
            sampled_bins=sampled_bins,
            observed_mean_z=observed_mean_z,
            probabilities=task["photon_probabilities"],
            conditional_mean_z=task["photon_conditional_mean_z"],
            conditional_variance_z=task["photon_conditional_variance_z"],
            event_counts=event_counts,
        )
        su2_energy, su2_combined = profiled_scores(
            sampled_bins=sampled_bins,
            observed_mean_z=observed_mean_z,
            probabilities=task["su2_probabilities"],
            conditional_mean_z=task["su2_conditional_mean_z"],
            conditional_variance_z=task["su2_conditional_variance_z"],
            event_counts=event_counts,
        )

        for observable, photon_best, su2_best in (
            ("energy", photon_energy, su2_energy),
            (
                "conditional_combined",
                photon_combined,
                su2_combined,
            ),
        ):
            statistic = 2.0 * (su2_best - photon_best)
            ties = statistic == 0.0
            su2_selected = statistic > 0.0
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
    for observable in correct:
        frames.append(
            pd.DataFrame(
                {
                    "mass_GeV": float(task["mass_gev"]),
                    "seed": seed,
                    "truth_model": truth_model,
                    "truth_lifetime_index": truth_index,
                    "truth_ctau_m": float(task["truth_ctau_m"]),
                    "observable": observable,
                    "number_of_events": event_counts,
                    "number_of_pseudoexperiments": number_of_pes,
                    "correct_fraction": correct[observable] / number_of_pes,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def conservative_curve(detailed: pd.DataFrame) -> pd.DataFrame:
    return (
        detailed.groupby(
            ["observable", "number_of_events"],
            as_index=False,
        )["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
    )


def persistent_threshold(curve: pd.DataFrame) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix_minimum = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix_minimum >= 0.9)
    return None if len(passing) == 0 else int(counts[passing[0]])


def plot_curves(curve: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "energy": "Energy only",
        "conditional_combined": (
            r"Energy + conditional mean $\langle z\rangle$"
        ),
    }
    fig, ax = plt.subplots(figsize=(8.3, 5.3))
    for observable, subset in curve.groupby("observable"):
        ax.plot(
            subset["number_of_events"],
            subset["worst_case_accuracy"],
            linewidth=1.6,
            label=labels[observable],
        )
    ax.axhline(0.9, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Observed ALP decays, N")
    ax.set_ylabel("Worst-case correct-classification probability")
    ax.set_ylim(0.45, 1.01)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "focused_accuracy_energy_vs_conditional_mean_z_ma_0p3.pdf"
    )
    fig.savefig(
        output_dir
        / "focused_accuracy_energy_vs_conditional_mean_z_ma_0p3.png",
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = Path.cwd().resolve()
    if not (repo / "analysis2").is_dir():
        raise SystemExit(
            "Run this script from the EventCalc-SHiP repository root."
        )
    if args.workers < 1 or args.workers > 2:
        raise ValueError("Use one or two workers on this laptop.")

    summary = json.loads(args.summary_json.resolve().read_text())
    bank_path = Path(summary["bank_path"])
    bank = load_template_bank(bank_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    moment_path = output_dir / "conditional_z_moments_ma_0p3.npz"
    if args.reuse_moments:
        if not moment_path.exists():
            raise FileNotFoundError(
                f"--reuse-moments requested but missing: {moment_path}"
            )
        moment_arrays = load_conditional_moments(moment_path)
        quality_table = pd.read_csv(
            output_dir / "conditional_z_moment_quality_ma_0p3.csv"
        )
    else:
        if moment_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{moment_path} already exists. Use --reuse-moments."
            )
        moment_arrays, quality_table = build_conditional_moments(
            bank=bank,
            output_dir=output_dir,
        )
    validate_conditional_moments(moment_arrays, bank)

    event_counts = np.arange(
        args.minimum_events,
        args.maximum_events + 1,
        dtype=int,
    )
    truths = resolve_truths(bank)
    common = {
        "mass_gev": float(bank.mass_gev),
        "photon_probabilities": bank.photon_probabilities,
        "su2_probabilities": bank.su2_probabilities,
        "photon_conditional_mean_z": moment_arrays[
            "photon_mean_z_by_energy_bin_m"
        ],
        "photon_conditional_variance_z": moment_arrays[
            "photon_variance_z_by_energy_bin_m2"
        ],
        "su2_conditional_mean_z": moment_arrays[
            "su2_mean_z_by_energy_bin_m"
        ],
        "su2_conditional_variance_z": moment_arrays[
            "su2_variance_z_by_energy_bin_m2"
        ],
        "event_counts": event_counts,
        "number_of_pseudoexperiments": int(args.pseudoexperiments),
        "chunk_size": int(args.chunk_size),
    }

    tasks = []
    for truth_model, indices in truths.items():
        probabilities = (
            bank.photon_probabilities
            if truth_model == "photon"
            else bank.su2_probabilities
        )
        lifetimes = (
            bank.photon_ctau_m
            if truth_model == "photon"
            else bank.su2_ctau_m
        )
        means = moment_arrays[
            f"{truth_model}_mean_z_by_energy_bin_m"
        ]
        variances = moment_arrays[
            f"{truth_model}_variance_z_by_energy_bin_m2"
        ]
        for truth_index in indices:
            for seed in args.seeds:
                tasks.append(
                    {
                        **common,
                        "truth_model": truth_model,
                        "truth_index": int(truth_index),
                        "truth_ctau_m": float(lifetimes[truth_index]),
                        "truth_probabilities": probabilities[truth_index],
                        "truth_conditional_mean_z": means[truth_index],
                        "truth_conditional_variance_z": variances[
                            truth_index
                        ],
                        "seed": int(seed),
                    }
                )

    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(simulate_truth, task) for task in tasks]
        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            frame = future.result()
            frames.append(frame)
            row = frame.iloc[0]
            print(
                f"[{completed:02d}/{len(futures):02d}] "
                f"{row.truth_model} "
                f"ctau={row.truth_ctau_m:.6g} m "
                f"seed={int(row.seed)}"
            )

    detailed = pd.concat(frames, ignore_index=True)
    detailed.sort_values(
        [
            "observable",
            "truth_model",
            "truth_lifetime_index",
            "seed",
            "number_of_events",
        ],
        inplace=True,
        ignore_index=True,
    )
    detailed_path = (
        output_dir
        / "focused_conditional_detailed_accuracy_ma_0p3.csv"
    )
    detailed.to_csv(detailed_path, index=False)

    curve = conservative_curve(detailed)
    curve_path = (
        output_dir
        / "focused_conditional_conservative_curve_ma_0p3.csv"
    )
    curve.to_csv(curve_path, index=False)
    plot_curves(curve, output_dir)

    thresholds = {
        observable: persistent_threshold(subset)
        for observable, subset in curve.groupby("observable")
    }
    truth_rows = []
    for model, indices in truths.items():
        lifetimes = (
            bank.photon_ctau_m
            if model == "photon"
            else bank.su2_ctau_m
        )
        for index in indices:
            truth_rows.append(
                {
                    "truth_model": model,
                    "truth_lifetime_index": int(index),
                    "truth_ctau_m": float(lifetimes[index]),
                }
            )

    result = {
        "status": "focused_correlation_aware_proof_of_principle",
        "mass_GeV": float(bank.mass_gev),
        "selection_name": bank.selection_name,
        "bank_path": str(bank_path),
        "pseudoexperiments_per_truth_and_seed": int(
            args.pseudoexperiments
        ),
        "seeds": [int(seed) for seed in args.seeds],
        "event_count_range": [
            int(args.minimum_events),
            int(args.maximum_events),
        ],
        "truths": truth_rows,
        "persistent_threshold_selected_truths": thresholds,
        "minimum_conditional_z_bin_N_eff": {
            prefix: float(
                quality_table.loc[
                    quality_table["model_prefix"] == prefix,
                    "minimum_bin_z_N_eff",
                ].min()
            )
            for prefix in ("photon", "su2")
        },
        "assumption": (
            "Within each adaptive energy bin, z is approximated by a "
            "Gaussian with the simulated weighted conditional mean and "
            "variance. The candidate mean-z likelihood is conditioned on "
            "the observed energy-bin sequence. This retains the leading "
            "E-z correlation but remains a focused proof of principle."
        ),
    }
    (output_dir / "focused_conditional_summary_ma_0p3.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("\n" + json.dumps(result, indent=2))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
