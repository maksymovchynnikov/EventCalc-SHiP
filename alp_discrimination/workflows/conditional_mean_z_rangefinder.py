"""Generic, resumable range finder for energy + conditional mean-z.

This is the first reusable multi-mass stage.  For one existing Week-8 template
bank it

1. builds (or reuses) conditional z moments for every model/lifetime/energy bin;
2. constructs a cheap joint energy-z Hellinger proxy used only to select a
   compact range-finding truth set;
3. runs low-statistics profiled pseudoexperiments on a broad event-count grid;
4. writes a suggested unit-spaced final grid for the full-domain 2k stage.

The physical likelihood remains the validated one:

    log L = log L_energy + log L_<z>|observed energy bins.

The Hellinger proxy never determines the final N90.  It is only a planner.  The
next stage must still evaluate every allowed truth lifetime at 2k and five
seeds, followed by the selected-truth ladder and omitted-truth audit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (73241, 83244)
MODEL_SPECS = (
    ("alp_photon_combined", "photon"),
    ("alp_su2l", "su2"),
)


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
    parser.add_argument(
        "--pilot-script-dir",
        type=Path,
        default=Path.home() / "Downloads",
    )
    parser.add_argument(
        "--event-count-grid",
        default="10,15,20,25,30,35,40,50,60,75,100,130,170,220,300",
    )
    parser.add_argument("--pseudoexperiments", type=int, default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--screening-truths-per-model", type=int, default=8)
    parser.add_argument("--screening-neighbourhood", type=int, default=2)
    parser.add_argument("--unit-window-half-width", type=int, default=30)
    parser.add_argument("--maximum-unit-window-points", type=int, default=241)
    parser.add_argument("--persistence-tail-factor", type=float, default=1.8)
    parser.add_argument("--restart-rangefinder", action="store_true")
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
        values = [int(piece) for piece in pieces]
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


def persistent_threshold(curve: pd.DataFrame, target: float = 0.90) -> int | None:
    ordered = curve.sort_values("number_of_events")
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)
    suffix = np.minimum.accumulate(accuracy[::-1])[::-1]
    passing = np.flatnonzero(suffix >= float(target))
    return None if len(passing) == 0 else int(counts[passing[0]])


def suggested_final_grid(
    curve: pd.DataFrame,
    *,
    unit_half_width: int,
    maximum_unit_points: int,
    persistence_tail_factor: float,
) -> tuple[np.ndarray, dict]:
    ordered = curve.sort_values("number_of_events").reset_index(drop=True)
    threshold = persistent_threshold(ordered)
    events = ordered["number_of_events"].to_numpy(dtype=int)
    accuracy = ordered["worst_case_accuracy"].to_numpy(dtype=float)

    if threshold is None:
        lower = int(events[-1])
        upper = max(lower + 1, int(np.ceil(2.0 * lower)))
        centre = int(np.ceil(1.5 * lower))
        reached = False
    else:
        index = int(np.flatnonzero(events == threshold)[0])
        upper = int(threshold)
        lower = int(events[index - 1]) if index > 0 else max(1, upper // 2)
        lower_accuracy = float(accuracy[index - 1]) if index > 0 else 0.0
        upper_accuracy = float(accuracy[index])
        if upper_accuracy > lower_accuracy:
            fraction = (0.90 - lower_accuracy) / (upper_accuracy - lower_accuracy)
            centre = int(round(lower + np.clip(fraction, 0.0, 1.0) * (upper - lower)))
        else:
            centre = int(round(0.5 * (lower + upper)))
        reached = True

    half = max(int(unit_half_width), int(np.ceil(0.4 * max(1, upper - lower))))
    if 2 * half + 1 > int(maximum_unit_points):
        half = (int(maximum_unit_points) - 1) // 2
    unit_lower = max(1, min(lower, centre - half))
    unit_upper = max(upper, centre + half)
    if unit_upper - unit_lower + 1 > int(maximum_unit_points):
        unit_upper = unit_lower + int(maximum_unit_points) - 1

    tail_stop = max(unit_upper + 75, int(np.ceil(upper * persistence_tail_factor)))
    tail_step = max(5, int(round(max(upper, 50) * 0.05)))
    unit = np.arange(unit_lower, unit_upper + 1, dtype=int)
    tail_start = unit_upper + tail_step
    tail = (
        np.arange(tail_start, tail_stop + 1, tail_step, dtype=int)
        if tail_start <= tail_stop
        else np.asarray([], dtype=int)
    )
    grid = np.unique(np.concatenate([unit, tail, [tail_stop]])).astype(int)
    return grid, {
        "threshold_reached": reached,
        "rangefinder_persistent_threshold": threshold,
        "lower_failing_events": lower,
        "upper_passing_events": upper,
        "estimated_crossing_events": centre,
        "unit_window": [int(unit_lower), int(unit_upper)],
        "persistence_tail_stop": int(tail_stop),
    }


# ---------------------------------------------------------------------------
# Conditional-moment workers
# ---------------------------------------------------------------------------

_MOMENT_PILOT = None
_MOMENT_ADAPTER = None
_MOMENT_BANK = None
_MOMENT_DOMAIN_PATH = None


def initialise_moment_worker(
    pilot_script_dir: str,
    repo_root: str,
    bank_path: str,
    domain_path: str,
) -> None:
    global _MOMENT_PILOT, _MOMENT_ADAPTER, _MOMENT_BANK
    global _MOMENT_DOMAIN_PATH
    repo = Path(repo_root)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _MOMENT_PILOT = load_pilot(Path(pilot_script_dir))
    from alp_discrimination.cache import CacheStore
    from alp_discrimination.config import get_config
    from alp_discrimination.eventcalc_adapter import EventCalcAdapter

    _MOMENT_BANK = _MOMENT_PILOT.load_template_bank(Path(bank_path))
    _MOMENT_DOMAIN_PATH = Path(domain_path)
    config = replace(
        get_config(_MOMENT_BANK.profile),
        selection_name=_MOMENT_BANK.selection_name,
    )
    _MOMENT_ADAPTER = EventCalcAdapter(
        config,
        cache=CacheStore(config.name),
        force=False,
    )


def build_one_moment(model_id: str, prefix: str, lifetime_index: int) -> dict:
    if (
        _MOMENT_PILOT is None
        or _MOMENT_ADAPTER is None
        or _MOMENT_BANK is None
        or _MOMENT_DOMAIN_PATH is None
    ):
        raise RuntimeError("Moment worker was not initialized.")
    return _MOMENT_PILOT.conditional_moments_for_lifetime(
        adapter=_MOMENT_ADAPTER,
        bank=_MOMENT_BANK,
        model_id=model_id,
        prefix=prefix,
        lifetime_index=int(lifetime_index),
        domain_path=_MOMENT_DOMAIN_PATH,
    )


def save_moment_part(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            model_prefix=np.asarray(str(result["model_prefix"])),
            model_id=np.asarray(str(result["model_id"])),
            lifetime_index=np.asarray(int(result["lifetime_index"])),
            ctau_m=np.asarray(float(result["ctau_m"])),
            mean_z_by_energy_bin_m=np.asarray(
                result["mean_z_by_energy_bin_m"], dtype=float
            ),
            variance_z_by_energy_bin_m2=np.asarray(
                result["variance_z_by_energy_bin_m2"], dtype=float
            ),
            n_eff_by_energy_bin=np.asarray(
                result["n_eff_by_energy_bin"], dtype=float
            ),
            raw_probability_by_energy_bin=np.asarray(
                result["raw_probability_by_energy_bin"], dtype=float
            ),
        )
    temporary.replace(path)


def load_moment_part(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def build_or_load_moments(
    *,
    pilot,
    bank,
    bank_path: Path,
    output_dir: Path,
    pilot_script_dir: Path,
    repo: Path,
    domain_path: Path,
    workers: int,
    token: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, Path]:
    master_path = output_dir / f"conditional_z_moments_ma_{token}.npz"
    quality_path = output_dir / f"conditional_z_moment_quality_ma_{token}.csv"
    if master_path.is_file() and quality_path.is_file():
        arrays = pilot.load_conditional_moments(master_path)
        pilot.validate_conditional_moments(arrays, bank)
        return arrays, pd.read_csv(quality_path), master_path

    parts_dir = output_dir / "conditional_z_moment_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, int, Path]] = []
    for model_id, prefix in MODEL_SPECS:
        lifetimes = np.asarray(getattr(bank, f"{prefix}_ctau_m"), dtype=float)
        for index in range(len(lifetimes)):
            path = parts_dir / f"{prefix}_{index:04d}.npz"
            if not path.is_file():
                jobs.append((model_id, prefix, index, path))

    if jobs:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialise_moment_worker,
            initargs=(
                str(pilot_script_dir),
                str(repo),
                str(bank_path),
                str(domain_path),
            ),
        ) as executor:
            futures = {
                executor.submit(build_one_moment, model_id, prefix, index): (
                    prefix,
                    index,
                    path,
                )
                for model_id, prefix, index, path in jobs
            }
            completed = 0
            for future in as_completed(futures):
                prefix, index, path = futures[future]
                result = future.result()
                save_moment_part(path, result)
                completed += 1
                print(
                    f"MOMENT {prefix:6s} index={index:3d} "
                    f"({completed}/{len(jobs)})",
                    flush=True,
                )

    arrays: dict[str, np.ndarray] = {
        "energy_edges_GeV": np.asarray(bank.energy_edges_gev, dtype=float),
    }
    quality_rows: list[dict] = []
    for _, prefix in MODEL_SPECS:
        lifetimes = np.asarray(getattr(bank, f"{prefix}_ctau_m"), dtype=float)
        means = []
        variances = []
        n_effs = []
        raw_probabilities = []
        for index, ctau in enumerate(lifetimes):
            path = parts_dir / f"{prefix}_{index:04d}.npz"
            if not path.is_file():
                raise RuntimeError(f"Missing conditional-moment part: {path}")
            part = load_moment_part(path)
            if int(part["lifetime_index"]) != index:
                raise ValueError(f"Moment part has wrong index: {path}")
            means.append(part["mean_z_by_energy_bin_m"])
            variances.append(part["variance_z_by_energy_bin_m2"])
            n_effs.append(part["n_eff_by_energy_bin"])
            raw_probabilities.append(part["raw_probability_by_energy_bin"])
            bank_probability = np.asarray(
                getattr(bank, f"{prefix}_probabilities")[index], dtype=float
            )
            quality_rows.append(
                {
                    "model_prefix": prefix,
                    "lifetime_index": index,
                    "ctau_m": float(ctau),
                    "minimum_bin_z_N_eff": float(
                        np.min(part["n_eff_by_energy_bin"])
                    ),
                    "maximum_absolute_raw_vs_bank_probability_difference": float(
                        np.max(
                            np.abs(
                                part["raw_probability_by_energy_bin"]
                                - bank_probability
                            )
                        )
                    ),
                }
            )
        arrays[f"{prefix}_mean_z_by_energy_bin_m"] = np.asarray(means)
        arrays[f"{prefix}_variance_z_by_energy_bin_m2"] = np.asarray(variances)
        arrays[f"{prefix}_z_n_eff_by_energy_bin"] = np.asarray(n_effs)
        arrays[f"{prefix}_raw_probability_by_energy_bin"] = np.asarray(
            raw_probabilities
        )

    with master_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(quality_path, index=False)
    pilot.validate_conditional_moments(arrays, bank)
    return arrays, quality, master_path


# ---------------------------------------------------------------------------
# Screening proxy
# ---------------------------------------------------------------------------


def joint_hellinger_squared(
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
    photon_mean: np.ndarray,
    photon_variance: np.ndarray,
    su2_mean: np.ndarray,
    su2_variance: np.ndarray,
) -> np.ndarray:
    """Pairwise Hellinger^2 for p(E) Normal(z|E), used only for planning."""
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
        ) * np.exp(
            -np.square(mp[index][None, :] - mq) / (4.0 * variance_sum)
        )
        coefficient = np.sum(
            np.sqrt(p[index][None, :] * q) * gaussian_bc,
            axis=1,
        )
        result[index] = np.clip(1.0 - coefficient, 0.0, 1.0)
    return result


def add_neighbourhood(selected: set[int], index: int, length: int, radius: int) -> None:
    for value in range(max(0, index - radius), min(length, index + radius + 1)):
        selected.add(int(value))


def interval_endpoints(intervals: np.ndarray) -> set[int]:
    result: set[int] = set()
    values = np.asarray(intervals, dtype=int)
    for interval in np.unique(values):
        indices = np.flatnonzero(values == interval)
        result.update((int(indices[0]), int(indices[-1])))
    return result


def screening_truths(
    bank,
    proxy: np.ndarray,
    *,
    count_per_model: int,
    neighbourhood: int,
) -> dict[str, np.ndarray]:
    global_photon, global_su2 = np.unravel_index(int(np.argmin(proxy)), proxy.shape)
    scores = {
        "photon": np.min(proxy, axis=1),
        "su2": np.min(proxy, axis=0),
    }
    minima = {"photon": global_photon, "su2": global_su2}
    intervals = {
        "photon": bank.photon_interval_index,
        "su2": bank.su2_interval_index,
    }
    result = {}
    for model in ("photon", "su2"):
        selected = interval_endpoints(intervals[model])
        for index in np.argsort(scores[model], kind="mergesort")[:count_per_model]:
            add_neighbourhood(selected, int(index), len(scores[model]), neighbourhood)
        add_neighbourhood(
            selected,
            int(minima[model]),
            len(scores[model]),
            neighbourhood,
        )
        result[model] = np.asarray(sorted(selected), dtype=int)
    return result


# ---------------------------------------------------------------------------
# Range-finder workers
# ---------------------------------------------------------------------------

_RANGE_PILOT = None
_RANGE_COMMON = None


def initialise_range_worker(pilot_script_dir: str, common: dict) -> None:
    global _RANGE_PILOT, _RANGE_COMMON
    _RANGE_PILOT = load_pilot(Path(pilot_script_dir))
    _RANGE_COMMON = common


def simulate_range_truth(
    truth_model: str,
    truth_index: int,
    truth_ctau_m: float,
    truth_probabilities: np.ndarray,
    truth_mean: np.ndarray,
    truth_variance: np.ndarray,
    seeds: Iterable[int],
) -> pd.DataFrame:
    if _RANGE_PILOT is None or _RANGE_COMMON is None:
        raise RuntimeError("Range-finder worker was not initialized.")
    frames = []
    for seed in seeds:
        frames.append(
            _RANGE_PILOT.simulate_truth(
                {
                    **_RANGE_COMMON,
                    "truth_model": truth_model,
                    "truth_index": int(truth_index),
                    "truth_ctau_m": float(truth_ctau_m),
                    "truth_probabilities": np.asarray(
                        truth_probabilities, dtype=float
                    ),
                    "truth_conditional_mean_z": np.asarray(truth_mean, dtype=float),
                    "truth_conditional_variance_z": np.asarray(
                        truth_variance, dtype=float
                    ),
                    "seed": int(seed),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    repo = Path.cwd().resolve()
    if not (repo / "alp_discrimination").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root.")
    if args.pseudoexperiments < 1 or args.chunk_size < 1:
        raise ValueError("Pseudoexperiment count and chunk size must be positive.")

    pilot = load_pilot(args.pilot_script_dir)
    bank_path = resolve(repo, args.bank_path)
    domain_path = resolve(repo, args.domain_path)
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bank_path.is_file():
        raise FileNotFoundError(f"Template bank not found: {bank_path}")
    if not domain_path.is_file():
        raise FileNotFoundError(
            f"Week-8 domain table not found: {domain_path}"
        )

    from alp_discrimination.workflows import float_token

    bank = pilot.load_template_bank(bank_path)
    token = float_token(float(bank.mass_gev))
    final_summary_path = output_dir / f"rangefinder_summary_ma_{token}.json"
    if final_summary_path.is_file() and not args.restart_rangefinder:
        raise FileExistsError(
            "This range-finder point is already complete. Preserve it and use "
            "a fresh --output-dir for a new run: " + str(final_summary_path)
        )
    event_counts = parse_event_counts(args.event_count_grid)

    arrays, quality, moments_path = build_or_load_moments(
        pilot=pilot,
        bank=bank,
        bank_path=bank_path,
        output_dir=output_dir,
        pilot_script_dir=args.pilot_script_dir.expanduser().resolve(),
        repo=repo,
        domain_path=domain_path,
        workers=args.workers,
        token=token,
    )

    minimum_neff = float(quality["minimum_bin_z_N_eff"].min())
    maximum_probability_difference = float(
        quality["maximum_absolute_raw_vs_bank_probability_difference"].max()
    )
    if minimum_neff < 100.0:
        raise RuntimeError(
            f"Conditional-z moment quality failed: minimum N_eff={minimum_neff:.3f}"
        )

    proxy = joint_hellinger_squared(
        bank.photon_probabilities,
        bank.su2_probabilities,
        arrays["photon_mean_z_by_energy_bin_m"],
        arrays["photon_variance_z_by_energy_bin_m2"],
        arrays["su2_mean_z_by_energy_bin_m"],
        arrays["su2_variance_z_by_energy_bin_m2"],
    )
    selected = screening_truths(
        bank,
        proxy,
        count_per_model=args.screening_truths_per_model,
        neighbourhood=args.screening_neighbourhood,
    )

    rows = []
    for model in ("photon", "su2"):
        lifetimes = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
        intervals = np.asarray(getattr(bank, f"{model}_interval_index"), dtype=int)
        score = np.min(proxy, axis=1 if model == "photon" else 0)
        for index in selected[model]:
            rows.append(
                {
                    "mass_GeV": float(bank.mass_gev),
                    "truth_model": model,
                    "truth_lifetime_index": int(index),
                    "truth_interval_index": int(intervals[index]),
                    "truth_ctau_m": float(lifetimes[index]),
                    "minimum_joint_proxy_H2": float(score[index]),
                }
            )
    screening_table = pd.DataFrame(rows).sort_values(
        ["truth_model", "truth_lifetime_index"], ignore_index=True
    )
    screening_path = output_dir / f"rangefinder_screening_truths_ma_{token}.csv"
    screening_table.to_csv(screening_path, index=False)

    proxy_minimum = np.unravel_index(int(np.argmin(proxy)), proxy.shape)
    proxy_summary = {
        "minimum_joint_proxy_H2": float(proxy[proxy_minimum]),
        "minimum_photon_lifetime_index": int(proxy_minimum[0]),
        "minimum_photon_ctau_m": float(bank.photon_ctau_m[proxy_minimum[0]]),
        "minimum_su2_lifetime_index": int(proxy_minimum[1]),
        "minimum_su2_ctau_m": float(bank.su2_ctau_m[proxy_minimum[1]]),
        "note": "The joint Hellinger proxy selects range-finding truths only; it is not the project test statistic and does not determine N90.",
    }
    (output_dir / f"joint_proxy_summary_ma_{token}.json").write_text(
        json.dumps(proxy_summary, indent=2) + "\n"
    )

    checkpoint_dir = output_dir / "rangefinder_truth_parts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.restart_rangefinder:
        for path in checkpoint_dir.glob("*.csv"):
            path.unlink()

    common = {
        "mass_gev": float(bank.mass_gev),
        "photon_probabilities": np.asarray(bank.photon_probabilities, dtype=float),
        "su2_probabilities": np.asarray(bank.su2_probabilities, dtype=float),
        "photon_conditional_mean_z": arrays[
            "photon_mean_z_by_energy_bin_m"
        ],
        "photon_conditional_variance_z": arrays[
            "photon_variance_z_by_energy_bin_m2"
        ],
        "su2_conditional_mean_z": arrays["su2_mean_z_by_energy_bin_m"],
        "su2_conditional_variance_z": arrays[
            "su2_variance_z_by_energy_bin_m2"
        ],
        "event_counts": event_counts,
        "number_of_pseudoexperiments": int(args.pseudoexperiments),
        "chunk_size": int(args.chunk_size),
    }

    jobs = []
    for model in ("photon", "su2"):
        probabilities = np.asarray(getattr(bank, f"{model}_probabilities"), dtype=float)
        lifetimes = np.asarray(getattr(bank, f"{model}_ctau_m"), dtype=float)
        means = arrays[f"{model}_mean_z_by_energy_bin_m"]
        variances = arrays[f"{model}_variance_z_by_energy_bin_m2"]
        for index in selected[model]:
            part = checkpoint_dir / f"{model}_{int(index):04d}.csv"
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

    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialise_range_worker,
            initargs=(str(args.pilot_script_dir.expanduser().resolve()), common),
        ) as executor:
            futures = {
                executor.submit(
                    simulate_range_truth,
                    model,
                    index,
                    ctau,
                    probabilities,
                    mean,
                    variance,
                    [int(seed) for seed in args.seeds],
                ): (model, index, part)
                for model, index, ctau, probabilities, mean, variance, part in jobs
            }
            completed = 0
            for future in as_completed(futures):
                model, index, part = futures[future]
                frame = future.result()
                frame.to_csv(part, index=False)
                completed += 1
                print(
                    f"RANGE {model:6s} index={index:3d} "
                    f"({completed}/{len(jobs)})",
                    flush=True,
                )

    part_paths = []
    for model in ("photon", "su2"):
        for index in selected[model]:
            path = checkpoint_dir / f"{model}_{int(index):04d}.csv"
            if not path.is_file():
                raise RuntimeError(f"Missing range-finder checkpoint: {path}")
            part_paths.append(path)
    detailed = pd.concat((pd.read_csv(path) for path in part_paths), ignore_index=True)
    detailed = detailed[detailed["observable"] == "conditional_combined"].copy()
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
    detailed_path = output_dir / f"rangefinder_detailed_accuracy_ma_{token}.csv"
    detailed.to_csv(detailed_path, index=False)

    curve = (
        detailed.groupby("number_of_events", as_index=False)["correct_fraction"]
        .min()
        .rename(columns={"correct_fraction": "worst_case_accuracy"})
        .sort_values("number_of_events", ignore_index=True)
    )
    curve_path = output_dir / f"rangefinder_conservative_curve_ma_{token}.csv"
    curve.to_csv(curve_path, index=False)

    by_seed_rows = []
    for seed, subset in detailed.groupby("seed"):
        seed_curve = (
            subset.groupby("number_of_events", as_index=False)["correct_fraction"]
            .min()
            .rename(columns={"correct_fraction": "worst_case_accuracy"})
        )
        by_seed_rows.append(
            {
                "seed": int(seed),
                "persistent_threshold": persistent_threshold(seed_curve),
            }
        )
    by_seed = pd.DataFrame(by_seed_rows)
    by_seed.to_csv(
        output_dir / f"rangefinder_threshold_by_seed_ma_{token}.csv",
        index=False,
    )

    final_grid, bracket = suggested_final_grid(
        curve,
        unit_half_width=args.unit_window_half_width,
        maximum_unit_points=args.maximum_unit_window_points,
        persistence_tail_factor=args.persistence_tail_factor,
    )
    final_grid_path = output_dir / f"suggested_final_event_grid_ma_{token}.txt"
    final_grid_path.write_text(",".join(str(int(value)) for value in final_grid) + "\n")

    threshold = persistent_threshold(curve)
    limiting_rows = []
    if threshold is not None:
        limiting_rows = (
            detailed.loc[detailed["number_of_events"] == threshold]
            .nsmallest(10, "correct_fraction")
            .to_dict(orient="records")
        )

    summary = {
        "status": "generic_conditional_mean_z_rangefinder",
        "mass_GeV": float(bank.mass_gev),
        "selection_name": str(bank.selection_name),
        "bank_path": str(bank_path),
        "domain_path": str(domain_path),
        "conditional_moments_path": str(moments_path),
        "number_of_energy_bins": int(bank.number_of_energy_bins),
        "number_of_profile_lifetimes": {
            "photon": int(len(bank.photon_ctau_m)),
            "su2": int(len(bank.su2_ctau_m)),
        },
        "conditional_z_quality": {
            "minimum_bin_z_N_eff": minimum_neff,
            "maximum_absolute_raw_vs_bank_probability_difference": maximum_probability_difference,
        },
        "rangefinder": {
            "pseudoexperiments_per_truth_and_seed": int(args.pseudoexperiments),
            "seeds": [int(seed) for seed in args.seeds],
            "event_counts": event_counts.tolist(),
            "number_of_screening_truths": {
                "photon": int(len(selected["photon"])),
                "su2": int(len(selected["su2"])),
            },
            **bracket,
            "suggested_final_event_grid": final_grid.tolist(),
            "limiting_rows_at_rangefinder_threshold": limiting_rows,
        },
        "joint_proxy": proxy_summary,
        "next_action": (
            "Use the suggested event grid for the full-domain 2k / five-seed stage. "
            "Do not quote the range-finder threshold as a result."
        ),
    }
    summary_path = final_summary_path
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        curve["number_of_events"],
        curve["worst_case_accuracy"],
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="Screening-truth range finder",
    )
    ax.axhline(0.90, linestyle="--", linewidth=1.0, label="90% target")
    if threshold is not None:
        ax.axvline(threshold, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Observed ALP decays, N")
    ax.set_ylabel("Worst-case correct-classification probability")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"rangefinder_accuracy_ma_{token}.pdf")
    fig.savefig(output_dir / f"rangefinder_accuracy_ma_{token}.png", dpi=180)
    plt.close(fig)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
