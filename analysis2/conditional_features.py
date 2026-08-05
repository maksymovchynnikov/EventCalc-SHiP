"""Generic conditional event-feature moments for lifetime-profiled discrimination.

The existing conditional-mean-z method uses the energy-bin sequence together
with the sample mean of z.  This module generalizes the same statistically
controlled construction to a small vector of per-event features:

    u_z       = (z - z_min) / (z_max - z_min),
    u_z^2     = u_z**2,
    rho_perp  = sqrt(x**2 + y**2) / sqrt(x_max**2 + y_max**2).

The available observable subsets therefore test, without changing the
lifetime-profiling framework:

* mean z only (the validated baseline),
* mean z plus a z-spread proxy,
* mean transverse displacement,
* mean z and mean transverse displacement,
* mean z, z-spread and transverse displacement together.

For each model/lifetime/energy bin the module stores the weighted feature mean
and covariance.  Pseudoexperiments sample the full feature vector coherently,
and candidate likelihoods use the multivariate Gaussian distribution of the
sample-mean vector conditioned on the observed energy-bin sequence.

The multivariate Gaussian is an approximation and must be checked against
empirical conditional resampling before a new observable is used in the final
mass scan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence
import zlib

import numpy as np
import pandas as pd

from funcs.ship_setup import x_max, y_max, z_max, z_min

from analysis2.cache import CacheStore
from analysis2.config import get_config
from analysis2.eventcalc_adapter import EventCalcAdapter
from analysis2.eventcalc_proposals import generate_mother_sample
from analysis2.models import get_model
from analysis2.mass_seed_resolution import (
    DEFAULT_WEEK8_DOMAIN_PATH,
    model_seed_for_bank,
)
from analysis2.profiled_statistics import stable_truth_rng
from analysis2.selections import SelectionContext
from analysis2.workflows.lifetime_blind_discrimination import (
    proposal_lifetime_for_target,
)


MODEL_SPECS = (
    ("alp_photon_combined", "photon"),
    ("alp_su2l", "su2"),
)

MASTER_FEATURE_LABELS = (
    "normalized_z",
    "normalized_z_squared",
    "normalized_r_perp",
)

FEATURE_SUBSETS: Mapping[str, tuple[int, ...]] = {
    "energy": (),
    "energy_mean_z": (0,),
    "energy_mean_z_spread": (0, 1),
    "energy_mean_r_perp": (2,),
    "energy_mean_z_r_perp": (0, 2),
    "energy_mean_z_spread_r_perp": (0, 1, 2),
}

FEATURE_LABELS: Mapping[str, str] = {
    "energy": "Energy only",
    "energy_mean_z": r"Energy + mean $z$",
    "energy_mean_z_spread": r"Energy + mean and spread of $z$",
    "energy_mean_r_perp": r"Energy + mean $r_\perp$",
    "energy_mean_z_r_perp": r"Energy + mean $z$ and $r_\perp$",
    "energy_mean_z_spread_r_perp": (
        r"Energy + mean/spread of $z$ + mean $r_\perp$"
    ),
}

Z_MIN_M = float(z_min)
Z_MAX_M = float(z_max)
Z_LENGTH_M = float(Z_MAX_M - Z_MIN_M)
def _maximum_decay_volume_radius_m() -> float:
    """Evaluate the z-dependent SHiP transverse envelope."""

    z_grid = np.linspace(Z_MIN_M, Z_MAX_M, 1025)
    x_limit = np.asarray(x_max(z_grid), dtype=float)
    y_limit = np.asarray(y_max(z_grid), dtype=float)
    x_limit = np.broadcast_to(x_limit, z_grid.shape)
    y_limit = np.broadcast_to(y_limit, z_grid.shape)
    if (
        np.any(~np.isfinite(x_limit))
        or np.any(~np.isfinite(y_limit))
        or np.any(x_limit <= 0.0)
        or np.any(y_limit <= 0.0)
    ):
        raise RuntimeError(
            "Invalid z-dependent SHiP transverse envelope."
        )
    return float(np.max(np.hypot(x_limit, y_limit)))


R_SCALE_M = _maximum_decay_volume_radius_m()

if not np.isfinite(Z_LENGTH_M) or Z_LENGTH_M <= 0.0:
    raise RuntimeError("Invalid SHiP decay-volume z range.")
if not np.isfinite(R_SCALE_M) or R_SCALE_M <= 0.0:
    raise RuntimeError("Invalid SHiP transverse geometry scale.")


@dataclass(frozen=True)
class SelectedFeatureSample:
    """Selected EventCalc mother events with shape-only feature information."""

    energy_gev: np.ndarray
    z_m: np.ndarray
    r_perp_m: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.energy_gev,
                self.z_m,
                self.r_perp_m,
                self.weights,
            )
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Selected feature arrays must be one-dimensional.")
        if len({len(array) for array in arrays}) != 1 or len(arrays[0]) == 0:
            raise ValueError("Selected feature arrays must be non-empty and aligned.")
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("Selected feature arrays contain non-finite values.")
        if np.any(arrays[-1] < 0.0) or float(arrays[-1].sum()) <= 0.0:
            raise ValueError("Selected feature weights must have positive total weight.")
        object.__setattr__(self, "energy_gev", arrays[0])
        object.__setattr__(self, "z_m", arrays[1])
        object.__setattr__(self, "r_perp_m", arrays[2])
        object.__setattr__(self, "weights", arrays[3])

    @property
    def master_features(self) -> np.ndarray:
        normalized_z = (self.z_m - Z_MIN_M) / Z_LENGTH_M
        normalized_r = self.r_perp_m / R_SCALE_M
        tolerance = 5.0e-10
        if np.any(normalized_z < -tolerance) or np.any(
            normalized_z > 1.0 + tolerance
        ):
            raise ValueError("A selected decay position lies outside the z volume.")
        normalized_z = np.clip(normalized_z, 0.0, 1.0)
        if np.any(normalized_r < -tolerance) or np.any(
            normalized_r > 1.0 + tolerance
        ):
            raise ValueError(
                "A selected transverse displacement lies outside "
                "the decay-volume normalization envelope."
            )
        normalized_r = np.clip(normalized_r, 0.0, 1.0)
        return np.column_stack(
            (
                normalized_z,
                np.square(normalized_z),
                normalized_r,
            )
        )


def selected_source_feature_sample(
    *,
    adapter: EventCalcAdapter,
    model_id: str,
    source_index: int,
    mass_gev: float,
    ctau_m: float,
    model_seed: int,
) -> SelectedFeatureSample:
    """Regenerate one cached proposal realization and retain E, z and r_perp."""

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
    weights = np.asarray(
        scale * mothers.decay_probability[mask],
        dtype=float,
    )
    return SelectedFeatureSample(
        energy_gev=np.asarray(mothers.energy_gev[mask], dtype=float),
        z_m=np.asarray(mothers.z_m[mask], dtype=float),
        r_perp_m=np.hypot(
            np.asarray(mothers.x_m[mask], dtype=float),
            np.asarray(mothers.y_m[mask], dtype=float),
        ),
        weights=weights,
    )


def combine_source_feature_samples(
    samples: Sequence[SelectedFeatureSample],
) -> SelectedFeatureSample:
    if not samples:
        raise ValueError("At least one source feature sample is required.")
    return SelectedFeatureSample(
        energy_gev=np.concatenate([sample.energy_gev for sample in samples]),
        z_m=np.concatenate([sample.z_m for sample in samples]),
        r_perp_m=np.concatenate([sample.r_perp_m for sample in samples]),
        weights=np.concatenate([sample.weights for sample in samples]),
    )


def regularize_covariance(
    covariance: np.ndarray,
    *,
    relative_floor: float = 1.0e-9,
    absolute_floor: float = 1.0e-12,
) -> np.ndarray:
    """Return symmetric positive-definite covariance matrices."""

    matrices = np.asarray(covariance, dtype=float)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("Master-feature covariance matrices must be 3x3.")
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=-1), absolute_floor)
    floor = np.maximum(relative_floor * scale, absolute_floor)
    clipped = np.maximum(eigenvalues, floor[..., None])
    return np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        clipped,
        eigenvectors,
    )


def weighted_feature_moments_by_energy_bin(
    *,
    sample: SelectedFeatureSample,
    energy_edges_gev: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute weighted means/covariances of the master features in E bins."""

    edges = np.asarray(energy_edges_gev, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Energy edges must be strictly increasing.")
    features = sample.master_features
    weights = sample.weights
    number_of_bins = len(edges) - 1
    indices = np.searchsorted(edges, sample.energy_gev, side="right") - 1
    indices = np.where(
        np.isclose(sample.energy_gev, edges[-1], rtol=0.0, atol=1.0e-12),
        number_of_bins - 1,
        indices,
    )
    valid = (indices >= 0) & (indices < number_of_bins) & (weights > 0.0)
    indices = indices[valid]
    features = features[valid]
    weights = weights[valid]

    sum_w = np.bincount(indices, weights=weights, minlength=number_of_bins)
    sum_w2 = np.bincount(
        indices,
        weights=np.square(weights),
        minlength=number_of_bins,
    )
    if np.any(sum_w <= 0.0):
        raise RuntimeError(
            "The raw selected sample has an empty adaptive energy bin: "
            f"{np.flatnonzero(sum_w <= 0.0).tolist()}"
        )

    sum_wf = np.empty((number_of_bins, 3), dtype=float)
    for feature_index in range(3):
        sum_wf[:, feature_index] = np.bincount(
            indices,
            weights=weights * features[:, feature_index],
            minlength=number_of_bins,
        )
    means = sum_wf / sum_w[:, None]

    second = np.empty((number_of_bins, 3, 3), dtype=float)
    for first in range(3):
        for second_index in range(3):
            second[:, first, second_index] = np.bincount(
                indices,
                weights=(
                    weights
                    * features[:, first]
                    * features[:, second_index]
                ),
                minlength=number_of_bins,
            ) / sum_w
    covariance = second - np.einsum("bi,bj->bij", means, means)
    covariance = regularize_covariance(covariance)
    n_eff = np.divide(
        np.square(sum_w),
        sum_w2,
        out=np.zeros_like(sum_w),
        where=sum_w2 > 0.0,
    )
    return {
        "mean": means,
        "covariance": covariance,
        "n_eff": n_eff,
        "raw_probability": sum_w / float(sum_w.sum()),
    }


def feature_moments_for_lifetime(
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
    samples = [
        selected_source_feature_sample(
            adapter=adapter,
            model_id=model_id,
            source_index=source_index,
            mass_gev=bank.mass_gev,
            ctau_m=ctau_m,
            model_seed=model_seed,
        )
        for source_index in range(len(model.sources))
    ]
    combined = combine_source_feature_samples(samples)
    moments = weighted_feature_moments_by_energy_bin(
        sample=combined,
        energy_edges_gev=bank.energy_edges_gev,
    )
    return {
        "model_prefix": prefix,
        "model_id": model_id,
        "lifetime_index": int(lifetime_index),
        "ctau_m": ctau_m,
        **moments,
    }


def build_conditional_feature_moments(
    *,
    bank,
    output_dir: Path,
    filename: str,
    domain_path: Path = DEFAULT_WEEK8_DOMAIN_PATH,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Build and persist the master-feature moment bank for one template bank."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "feature_format_version": np.asarray(1),
        "feature_labels": np.asarray(MASTER_FEATURE_LABELS),
        "energy_edges_GeV": np.asarray(bank.energy_edges_gev, dtype=float),
        "z_min_m": np.asarray(Z_MIN_M),
        "z_max_m": np.asarray(Z_MAX_M),
        "r_scale_m": np.asarray(R_SCALE_M),
    }
    rows: list[dict] = []
    for model_id, prefix in MODEL_SPECS:
        lifetimes = np.asarray(getattr(bank, f"{prefix}_ctau_m"), dtype=float)
        model_means = []
        model_covariances = []
        model_n_effs = []
        model_probabilities = []
        for index in range(len(lifetimes)):
            result = feature_moments_for_lifetime(
                adapter=adapter,
                bank=bank,
                model_id=model_id,
                prefix=prefix,
                lifetime_index=index,
                domain_path=domain_path,
            )
            model_means.append(result["mean"])
            model_covariances.append(result["covariance"])
            model_n_effs.append(result["n_eff"])
            model_probabilities.append(result["raw_probability"])
            eigenvalues = np.linalg.eigvalsh(result["covariance"])
            rows.append(
                {
                    "model_prefix": prefix,
                    "lifetime_index": int(index),
                    "ctau_m": float(lifetimes[index]),
                    "minimum_bin_feature_N_eff": float(
                        np.min(result["n_eff"])
                    ),
                    "minimum_covariance_eigenvalue": float(
                        np.min(eigenvalues)
                    ),
                    "maximum_absolute_raw_vs_bank_probability_difference": float(
                        np.max(
                            np.abs(
                                result["raw_probability"]
                                - getattr(bank, f"{prefix}_probabilities")[index]
                            )
                        )
                    ),
                }
            )
            print(
                f"FEATURE MOMENTS {prefix:6s} "
                f"{index + 1:3d}/{len(lifetimes):3d} "
                f"ctau={lifetimes[index]:.6g} m",
                flush=True,
            )

        arrays[f"{prefix}_feature_mean"] = np.asarray(model_means)
        arrays[f"{prefix}_feature_covariance"] = np.asarray(model_covariances)
        arrays[f"{prefix}_feature_n_eff"] = np.asarray(model_n_effs)
        arrays[f"{prefix}_raw_probability"] = np.asarray(model_probabilities)

    np.savez_compressed(output_dir / filename, **arrays)
    quality = pd.DataFrame(rows)
    quality.to_csv(
        output_dir / filename.replace(".npz", "_quality.csv"),
        index=False,
    )
    return arrays, quality


def load_conditional_feature_moments(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def validate_conditional_feature_moments(
    arrays: Mapping[str, np.ndarray],
    bank,
) -> None:
    if int(np.asarray(arrays["feature_format_version"]).item()) != 1:
        raise ValueError("Unsupported conditional-feature moment format.")
    labels = tuple(str(value) for value in arrays["feature_labels"].tolist())
    if labels != MASTER_FEATURE_LABELS:
        raise ValueError("Conditional-feature labels are incompatible.")
    if not np.allclose(
        arrays["energy_edges_GeV"],
        bank.energy_edges_gev,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Conditional features use different energy edges.")
    for prefix in ("photon", "su2"):
        number_of_lifetimes = len(getattr(bank, f"{prefix}_ctau_m"))
        expected_mean = (number_of_lifetimes, bank.number_of_energy_bins, 3)
        expected_cov = (
            number_of_lifetimes,
            bank.number_of_energy_bins,
            3,
            3,
        )
        means = np.asarray(arrays[f"{prefix}_feature_mean"], dtype=float)
        covariances = np.asarray(
            arrays[f"{prefix}_feature_covariance"],
            dtype=float,
        )
        if means.shape != expected_mean or covariances.shape != expected_cov:
            raise ValueError(f"{prefix} conditional features have wrong shape.")
        if np.any(~np.isfinite(means)) or np.any(~np.isfinite(covariances)):
            raise ValueError(f"{prefix} conditional features are non-finite.")
        if np.any(np.linalg.eigvalsh(covariances) <= 0.0):
            raise ValueError(f"{prefix} conditional covariance is not positive definite.")


def stable_feature_rng(
    *,
    seed: int,
    mass_gev: float,
    truth_model: str,
    truth_index: int,
) -> np.random.Generator:
    mass_hash = zlib.crc32(f"{mass_gev:.16g}".encode("ascii"))
    model_code = 0 if truth_model == "photon" else 1
    sequence = np.random.SeedSequence(
        [int(seed), int(mass_hash), model_code, int(truth_index), 0xF347]
    )
    return np.random.default_rng(sequence)


def _subset_arrays(
    means: np.ndarray,
    covariances: np.ndarray,
    feature_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(feature_indices, dtype=int)
    return (
        np.take(means, indices, axis=-1),
        np.take(np.take(covariances, indices, axis=-2), indices, axis=-1),
    )


def profiled_feature_scores(
    *,
    sampled_bins: np.ndarray,
    observed_feature_means: np.ndarray,
    probabilities: np.ndarray,
    conditional_feature_mean: np.ndarray,
    conditional_feature_covariance: np.ndarray,
    event_counts: np.ndarray,
    feature_indices: Sequence[int],
    target_bytes: int = 48 * 1024**2,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile energy and energy+feature sample-mean likelihoods."""

    probabilities = np.asarray(probabilities, dtype=float)
    sampled_bins = np.asarray(sampled_bins, dtype=int)
    observed = np.asarray(observed_feature_means, dtype=float)
    counts = np.asarray(event_counts, dtype=int)
    if probabilities.ndim != 2 or np.any(probabilities <= 0.0):
        raise ValueError("Candidate probabilities must be a positive matrix.")
    if sampled_bins.ndim != 2:
        raise ValueError("sampled_bins must be a matrix.")
    if counts.ndim != 1 or np.any(np.diff(counts) <= 0):
        raise ValueError("event_counts must be increasing.")
    if not feature_indices:
        if observed.shape[-1] != 0:
            raise ValueError("Energy-only observed features must have dimension zero.")
    means, covariances = _subset_arrays(
        np.asarray(conditional_feature_mean, dtype=float),
        np.asarray(conditional_feature_covariance, dtype=float),
        feature_indices,
    )
    dimension = len(feature_indices)
    if observed.shape != (sampled_bins.shape[0], len(counts), dimension):
        raise ValueError("Observed feature-mean array has the wrong shape.")

    number_of_pes, maximum_events = sampled_bins.shape
    number_of_candidates = probabilities.shape[0]
    bytes_per_pe = max(
        1,
        number_of_candidates
        * maximum_events
        * max(1, dimension * dimension + dimension + 1)
        * np.dtype(float).itemsize,
    )
    block_size = max(1, min(number_of_pes, target_bytes // bytes_per_pe))
    energy_best = np.empty((number_of_pes, len(counts)), dtype=float)
    combined_best = np.empty_like(energy_best)
    log_probabilities = np.log(probabilities)

    for start in range(0, number_of_pes, block_size):
        stop = min(number_of_pes, start + block_size)
        bins = sampled_bins[start:stop]
        energy_contributions = log_probabilities[:, bins]
        np.cumsum(energy_contributions, axis=2, out=energy_contributions)

        if dimension:
            mean_contributions = means[:, bins, :]
            covariance_contributions = covariances[:, bins, :, :]
            np.cumsum(mean_contributions, axis=2, out=mean_contributions)
            np.cumsum(
                covariance_contributions,
                axis=2,
                out=covariance_contributions,
            )

        for column, count in enumerate(counts):
            event_index = int(count - 1)
            energy_at_count = energy_contributions[:, :, event_index]
            energy_best[start:stop, column] = np.max(energy_at_count, axis=0)
            if not dimension:
                combined_best[start:stop, column] = energy_best[start:stop, column]
                continue

            predicted_mean = (
                mean_contributions[:, :, event_index, :] / float(count)
            )
            covariance_of_mean = (
                covariance_contributions[:, :, event_index, :, :]
                / float(count * count)
            )
            trace = np.trace(covariance_of_mean, axis1=-2, axis2=-1)
            ridge = np.maximum(
                1.0e-10 * trace / float(dimension),
                1.0e-14,
            )
            diagonal = np.arange(dimension)
            covariance_of_mean[..., diagonal, diagonal] += ridge[..., None]

            residual = (
                observed[start:stop, column, :][None, :, :]
                - predicted_mean
            )
            sign, log_determinant = np.linalg.slogdet(covariance_of_mean)
            if np.any(sign <= 0.0):
                raise RuntimeError("A profiled feature covariance is not positive definite.")
            solved = np.linalg.solve(
                covariance_of_mean,
                residual[..., None],
            )[..., 0]
            quadratic = np.sum(residual * solved, axis=-1)
            log_feature = -0.5 * (
                quadratic
                + log_determinant
                + dimension * np.log(2.0 * np.pi)
            )
            combined_best[start:stop, column] = np.max(
                energy_at_count + log_feature,
                axis=0,
            )

    return energy_best, combined_best


def sample_master_features(
    *,
    sampled_bins: np.ndarray,
    truth_mean: np.ndarray,
    truth_covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample paired three-dimensional per-event Gaussian feature vectors."""

    bins = np.asarray(sampled_bins, dtype=int)
    means = np.asarray(truth_mean, dtype=float)
    covariances = np.asarray(truth_covariance, dtype=float)
    cholesky = np.linalg.cholesky(covariances)
    standard = rng.standard_normal(bins.shape + (3,))
    sampled = np.empty_like(standard)
    for energy_bin in np.unique(bins):
        mask = bins == int(energy_bin)
        sampled[mask] = (
            means[int(energy_bin)]
            + standard[mask] @ cholesky[int(energy_bin)].T
        )
    return sampled


def gaussian_bhattacharyya_coefficient(
    mean_first: np.ndarray,
    covariance_first: np.ndarray,
    mean_second: np.ndarray,
    covariance_second: np.ndarray,
) -> float:
    first_mean = np.asarray(mean_first, dtype=float)
    second_mean = np.asarray(mean_second, dtype=float)
    first_cov = np.asarray(covariance_first, dtype=float)
    second_cov = np.asarray(covariance_second, dtype=float)
    average = 0.5 * (first_cov + second_cov)
    sign_first, log_first = np.linalg.slogdet(first_cov)
    sign_second, log_second = np.linalg.slogdet(second_cov)
    sign_average, log_average = np.linalg.slogdet(average)
    if min(sign_first, sign_second, sign_average) <= 0.0:
        raise ValueError("Bhattacharyya covariance is not positive definite.")
    difference = first_mean - second_mean
    quadratic = float(difference @ np.linalg.solve(average, difference))
    log_coefficient = (
        0.25 * log_first
        + 0.25 * log_second
        - 0.5 * log_average
        - 0.125 * quadratic
    )
    return float(np.clip(np.exp(log_coefficient), 0.0, 1.0))


def pairwise_joint_energy_feature_hellinger_squared(
    *,
    photon_probabilities: np.ndarray,
    photon_means: np.ndarray,
    photon_covariances: np.ndarray,
    su2_probabilities: np.ndarray,
    su2_means: np.ndarray,
    su2_covariances: np.ndarray,
    feature_indices: Sequence[int],
) -> np.ndarray:
    """Vectorized Hellinger proxy for every photon/SU(2) lifetime pair."""

    photon_probability = np.asarray(photon_probabilities, dtype=float)
    su2_probability = np.asarray(su2_probabilities, dtype=float)
    if photon_probability.ndim != 2 or su2_probability.ndim != 2:
        raise ValueError("Probability banks must be matrices.")
    if photon_probability.shape[1] != su2_probability.shape[1]:
        raise ValueError("Photon and SU(2) banks use different energy bins.")
    if not feature_indices:
        coefficient = np.sqrt(photon_probability) @ np.sqrt(
            su2_probability
        ).T
        return np.clip(1.0 - coefficient, 0.0, 1.0)

    photon_mean, photon_covariance = _subset_arrays(
        np.asarray(photon_means, dtype=float),
        np.asarray(photon_covariances, dtype=float),
        feature_indices,
    )
    su2_mean, su2_covariance = _subset_arrays(
        np.asarray(su2_means, dtype=float),
        np.asarray(su2_covariances, dtype=float),
        feature_indices,
    )
    number_of_photon = photon_probability.shape[0]
    number_of_su2 = su2_probability.shape[0]
    coefficient = np.zeros((number_of_photon, number_of_su2), dtype=float)

    for energy_bin in range(photon_probability.shape[1]):
        first_cov = photon_covariance[:, energy_bin, :, :]
        second_cov = su2_covariance[:, energy_bin, :, :]
        average = 0.5 * (
            first_cov[:, None, :, :] + second_cov[None, :, :, :]
        )
        sign_first, log_first = np.linalg.slogdet(first_cov)
        sign_second, log_second = np.linalg.slogdet(second_cov)
        sign_average, log_average = np.linalg.slogdet(average)
        if (
            np.any(sign_first <= 0.0)
            or np.any(sign_second <= 0.0)
            or np.any(sign_average <= 0.0)
        ):
            raise ValueError(
                "Pairwise Bhattacharyya covariance is not positive definite."
            )
        difference = (
            photon_mean[:, energy_bin, :][:, None, :]
            - su2_mean[:, energy_bin, :][None, :, :]
        )
        solved = np.linalg.solve(average, difference[..., None])[..., 0]
        quadratic = np.sum(difference * solved, axis=-1)
        log_coefficient = (
            0.25 * log_first[:, None]
            + 0.25 * log_second[None, :]
            - 0.5 * log_average
            - 0.125 * quadratic
        )
        conditional_coefficient = np.clip(
            np.exp(log_coefficient),
            0.0,
            1.0,
        )
        coefficient += (
            np.sqrt(
                photon_probability[:, energy_bin][:, None]
                * su2_probability[:, energy_bin][None, :]
            )
            * conditional_coefficient
        )
    return np.clip(1.0 - coefficient, 0.0, 1.0)


def joint_energy_feature_hellinger_squared(
    *,
    first_probabilities: np.ndarray,
    first_means: np.ndarray,
    first_covariances: np.ndarray,
    second_probabilities: np.ndarray,
    second_means: np.ndarray,
    second_covariances: np.ndarray,
    feature_indices: Sequence[int],
) -> float:
    first_probability = np.asarray(first_probabilities, dtype=float)
    second_probability = np.asarray(second_probabilities, dtype=float)
    if not feature_indices:
        coefficient = float(
            np.sum(np.sqrt(first_probability * second_probability))
        )
        return float(np.clip(1.0 - coefficient, 0.0, 1.0))

    first_mean, first_cov = _subset_arrays(
        np.asarray(first_means, dtype=float),
        np.asarray(first_covariances, dtype=float),
        feature_indices,
    )
    second_mean, second_cov = _subset_arrays(
        np.asarray(second_means, dtype=float),
        np.asarray(second_covariances, dtype=float),
        feature_indices,
    )
    coefficient = 0.0
    for energy_bin in range(len(first_probability)):
        coefficient += (
            np.sqrt(
                first_probability[energy_bin]
                * second_probability[energy_bin]
            )
            * gaussian_bhattacharyya_coefficient(
                first_mean[energy_bin],
                first_cov[energy_bin],
                second_mean[energy_bin],
                second_cov[energy_bin],
            )
        )
    return float(np.clip(1.0 - coefficient, 0.0, 1.0))


__all__ = [
    "FEATURE_LABELS",
    "FEATURE_SUBSETS",
    "MASTER_FEATURE_LABELS",
    "MODEL_SPECS",
    "R_SCALE_M",
    "SelectedFeatureSample",
    "Z_LENGTH_M",
    "Z_MAX_M",
    "Z_MIN_M",
    "build_conditional_feature_moments",
    "combine_source_feature_samples",
    "feature_moments_for_lifetime",
    "gaussian_bhattacharyya_coefficient",
    "joint_energy_feature_hellinger_squared",
    "load_conditional_feature_moments",
    "pairwise_joint_energy_feature_hellinger_squared",
    "profiled_feature_scores",
    "regularize_covariance",
    "sample_master_features",
    "selected_source_feature_sample",
    "stable_feature_rng",
    "validate_conditional_feature_moments",
    "weighted_feature_moments_by_energy_bin",
]
