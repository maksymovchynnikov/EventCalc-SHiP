#!/usr/bin/env python3
"""
ALP-photon event-count grid runner.

This is a non-interactive, dependency-light version of the ALP-photon event
count part of simulate.py. It samples the ALP kinematics, applies the SHiP
azimuthal/decay-volume selection, and reports the total expected event count
for primary and cascade photon sources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_MASSES_GEV = (0.1, 0.2, 0.5, 1.0)
DEFAULT_C_TAU_M = 3000.0
DEFAULT_RESAMPLE_SIZE = 200_000
DEFAULT_SEED = 12345
N_RAW_MULTIPLIER = 10
N_POT = 6.0e20
MODES = ("primary", "cascades")

Z_MIN = 32.0
Z_MAX = 82.0
DELTA_X_IN = 1.0
DELTA_X_OUT = 4.0
DELTA_Y_IN = 2.7
DELTA_Y_OUT = 6.2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ALP_DIR = REPO_ROOT / "Distributions" / "ALP-photon"


def x_max(z):
    return (
        DELTA_X_IN / 2.0 * (z - Z_MAX) / (Z_MIN - Z_MAX)
        + DELTA_X_OUT / 2.0 * (z - Z_MIN) / (Z_MAX - Z_MIN)
    )


def y_max(z):
    return (
        DELTA_Y_IN / 2.0 * (z - Z_MAX) / (Z_MIN - Z_MAX)
        + DELTA_Y_OUT / 2.0 * (z - Z_MIN) / (Z_MAX - Z_MIN)
    )


def theta_max_dec_vol():
    theta_in = math.sqrt((DELTA_Y_IN / 2.0) ** 2 + (DELTA_X_IN / 2.0) ** 2) / Z_MIN
    theta_out = math.sqrt((DELTA_Y_OUT / 2.0) ** 2 + (DELTA_X_OUT / 2.0) ** 2) / Z_MAX
    return math.atan(max(theta_in, theta_out))


def interpolate_table(path: Path, mass: float) -> float:
    data = np.loadtxt(path)
    return float(np.interp(mass, data[:, 0], data[:, 1]))


def visible_branching_ratio(mass: float) -> float:
    with (ALP_DIR / "ALP-photon-decay.json").open() as handle:
        channels = json.load(handle)

    branching_ratio = 0.0
    for channel in channels:
        br_data = channel[2]
        if isinstance(br_data, (float, int)):
            branching_ratio += float(br_data)
        else:
            br_table = np.asarray(br_data, dtype=float)
            branching_ratio += float(np.interp(mass, br_table[:, 0], br_table[:, 1]))
    return branching_ratio


def lower_indices(grid: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx1 = np.searchsorted(grid, values, side="left") - 1
    idx1 = np.clip(idx1, 0, len(grid) - 2)
    return idx1, idx1 + 1


def fill_2d(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_x = np.unique(data[:, 0])
    grid_y = np.unique(data[:, 1])
    values = np.zeros((len(grid_x), len(grid_y)))
    ix = np.searchsorted(grid_x, data[:, 0], side="left")
    iy = np.searchsorted(grid_y, data[:, 1], side="left")
    values[ix, iy] = data[:, 2]
    return grid_x, grid_y, values


def fill_3d(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_x = np.unique(data[:, 0])
    grid_y = np.unique(data[:, 1])
    grid_z = np.unique(data[:, 2])
    values = np.zeros((len(grid_x), len(grid_y), len(grid_z)))
    ix = np.searchsorted(grid_x, data[:, 0], side="left")
    iy = np.searchsorted(grid_y, data[:, 1], side="left")
    iz = np.searchsorted(grid_z, data[:, 2], side="left")
    values[ix, iy, iz] = data[:, 3]
    return grid_x, grid_y, grid_z, values


def interpolate_2d(
    x: np.ndarray,
    y: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    ix1, ix2 = lower_indices(grid_x, x)
    iy1, iy2 = lower_indices(grid_y, y)

    x1, x2 = grid_x[ix1], grid_x[ix2]
    y1, y2 = grid_y[iy1], grid_y[iy2]
    xd = (x - x1) / (x2 - x1)
    yd = (y - y1) / (y2 - y1)

    z11 = values[ix1, iy1]
    z21 = values[ix2, iy1]
    z12 = values[ix1, iy2]
    z22 = values[ix2, iy2]

    c0 = z11 * (1.0 - xd) + z21 * xd
    c1 = z12 * (1.0 - xd) + z22 * xd
    return c0 * (1.0 - yd) + c1 * yd


def interpolate_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    values: np.ndarray,
    max_energy: np.ndarray,
) -> np.ndarray:
    ix1, ix2 = lower_indices(grid_x, x)
    iy1, iy2 = lower_indices(grid_y, y)
    iz1, iz2 = lower_indices(grid_z, z)

    x1, x2 = grid_x[ix1], grid_x[ix2]
    y1, y2 = grid_y[iy1], grid_y[iy2]
    z1, z2 = grid_z[iz1], grid_z[iz2]
    xd = (x - x1) / (x2 - x1)
    yd = (y - y1) / (y2 - y1)
    zd = (z - z1) / (z2 - z1)

    z111 = values[ix1, iy1, iz1]
    z211 = values[ix2, iy1, iz1]
    z121 = values[ix1, iy2, iz1]
    z221 = values[ix2, iy2, iz1]
    z112 = values[ix1, iy1, iz2]
    z212 = values[ix2, iy1, iz2]
    z122 = values[ix1, iy2, iz2]
    z222 = values[ix2, iy2, iz2]

    c00 = z111 * (1.0 - xd) + z211 * xd
    c01 = z112 * (1.0 - xd) + z212 * xd
    c10 = z121 * (1.0 - xd) + z221 * xd
    c11 = z122 * (1.0 - xd) + z222 * xd

    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    interpolated = c0 * (1.0 - zd) + c1 * zd
    return np.where(z > max_energy, 0.0, interpolated)


def load_mode_tables(mode: str):
    distr_data = np.loadtxt(ALP_DIR / f"DoubleDistr-ALP-photon_{mode}.txt")
    emax_data = np.loadtxt(ALP_DIR / f"Emax-ALP-photon_{mode}.txt")
    mass_grid, theta_grid, energy_grid, distr_values = fill_3d(distr_data)
    emax_mass_grid, emax_theta_grid, emax_values = fill_2d(emax_data)
    return {
        "mass_grid": mass_grid,
        "theta_grid": theta_grid,
        "energy_grid": energy_grid,
        "distr_values": distr_values,
        "emax_mass_grid": emax_mass_grid,
        "emax_theta_grid": emax_theta_grid,
        "emax_values": emax_values,
    }


def simulate_mode(
    mass: float,
    c_tau: float,
    mode: str,
    resample_size: int,
    seed: int,
    visible_br: float,
    coupling_squared: float,
):
    tables = load_mode_tables(mode)
    raw_size = resample_size * N_RAW_MULTIPLIER
    rng = np.random.default_rng(seed)

    theta_min = float(tables["theta_grid"].min())
    theta_max = min(float(tables["theta_grid"].max()), theta_max_dec_vol())
    theta = rng.uniform(theta_min, theta_max, raw_size)
    mass_values = np.full(raw_size, mass)

    max_energy = interpolate_2d(
        mass_values,
        theta,
        tables["emax_mass_grid"],
        tables["emax_theta_grid"],
        tables["emax_values"],
    )
    e_min_sampling = np.maximum(
        mass,
        np.minimum(2.133 * mass / c_tau, 0.5 * max_energy),
    )
    energy = rng.uniform(e_min_sampling, max_energy)
    density = interpolate_3d(
        mass_values,
        theta,
        energy,
        tables["mass_grid"],
        tables["theta_grid"],
        tables["energy_grid"],
        tables["distr_values"],
        max_energy,
    )

    polar_weights = density * (max_energy - e_min_sampling)
    if polar_weights.sum() <= 0.0:
        raise ValueError(f"All polar weights are zero for mass={mass:g} GeV, mode={mode}.")
    epsilon_polar = float(polar_weights.sum() * (theta_max - theta_min) / raw_size)

    probabilities = polar_weights / polar_weights.sum()
    sampled_indices = rng.choice(raw_size, size=resample_size, replace=True, p=probabilities)
    sampled_theta = theta[sampled_indices]
    sampled_energy = energy[sampled_indices]

    phi = rng.uniform(-math.pi, math.pi, resample_size)
    momentum_abs = np.sqrt(sampled_energy * sampled_energy - mass * mass)
    cos_theta = np.cos(sampled_theta)

    exponent_min = -Z_MIN * mass / (cos_theta * c_tau * momentum_abs)
    exponent_max = -Z_MAX * mass / (cos_theta * c_tau * momentum_abs)
    cmin = 1.0 - np.exp(exponent_min)
    cmax = 1.0 - np.exp(exponent_max)
    c = rng.uniform(cmin, cmax)
    safe_c = np.minimum(c, 0.9999999995)
    z = np.where(
        c > 0.9999999995,
        Z_MIN,
        cos_theta * c_tau * (momentum_abs / mass) * np.log(1.0 / (1.0 - safe_c)),
    )

    x = z * np.cos(phi) * np.tan(sampled_theta)
    y = z * np.sin(phi) * np.tan(sampled_theta)
    accepted = (
        (-x_max(z) < x)
        & (x < x_max(z))
        & (-y_max(z) < y)
        & (y < y_max(z))
        & (Z_MIN <= z)
        & (z <= Z_MAX)
    )

    p_decay = np.exp(exponent_min) - np.exp(exponent_max)
    accepted_p_decay = p_decay[accepted]
    epsilon_azimuthal = float(accepted.sum() / resample_size)
    p_decay_averaged = float(accepted_p_decay.mean()) if accepted_p_decay.size else 0.0

    yield_per_pot = interpolate_table(ALP_DIR / f"Total-yield-ALP-photon_{mode}.txt", mass)
    produced_llps = N_POT * yield_per_pot * coupling_squared
    total_events = (
        produced_llps
        * visible_br
        * epsilon_polar
        * epsilon_azimuthal
        * p_decay_averaged
    )

    return {
        "mass_GeV": mass,
        "mass_MeV": mass * 1000.0,
        "c_tau_m": c_tau,
        "mode": mode,
        "yield_per_pot": yield_per_pot,
        "produced_llps": produced_llps,
        "epsilon_polar": epsilon_polar,
        "epsilon_azimuthal": epsilon_azimuthal,
        "P_decay_averaged": p_decay_averaged,
        "visible_br": visible_br,
        "accepted_sample_events": int(accepted.sum()),
        "sampled_events": resample_size,
        "N_events": total_events,
    }


def run_grid(masses: list[float], c_tau: float, resample_size: int, seed: int):
    rows = []
    for mass_index, mass in enumerate(masses):
        intrinsic_ctau = interpolate_table(ALP_DIR / "ctau-ALP-photon.txt", mass)
        coupling_squared = intrinsic_ctau / c_tau
        visible_br = visible_branching_ratio(mass)
        for mode_index, mode in enumerate(MODES):
            row = simulate_mode(
                mass,
                c_tau,
                mode,
                resample_size,
                seed + 1000 * mass_index + mode_index,
                visible_br,
                coupling_squared,
            )
            row["intrinsic_ctau_m"] = intrinsic_ctau
            row["coupling_squared"] = coupling_squared
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mass_MeV",
        "mass_GeV",
        "c_tau_m",
        "mode",
        "intrinsic_ctau_m",
        "coupling_squared",
        "yield_per_pot",
        "produced_llps",
        "epsilon_polar",
        "epsilon_azimuthal",
        "P_decay_averaged",
        "visible_br",
        "sampled_events",
        "accepted_sample_events",
        "N_events",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(rows: list[dict[str, float]]) -> None:
    print("mass_MeV primary_events cascades_events cascades_over_primary difference")
    masses = sorted({row["mass_MeV"] for row in rows})
    for mass in masses:
        by_mode = {row["mode"]: row for row in rows if row["mass_MeV"] == mass}
        primary = by_mode["primary"]["N_events"]
        cascades = by_mode["cascades"]["N_events"]
        ratio = cascades / primary if primary else math.inf
        diff = cascades - primary
        print(f"{mass:.0f} {primary:.9e} {cascades:.9e} {ratio:.9e} {diff:.9e}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masses", type=float, nargs="+", default=list(DEFAULT_MASSES_GEV))
    parser.add_argument("--c-tau", type=float, default=DEFAULT_C_TAU_M)
    parser.add_argument("--resample-size", type=int, default=DEFAULT_RESAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "alp_photon_3000m_event_counts.csv",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = run_grid(args.masses, args.c_tau, args.resample_size, args.seed)
    write_csv(rows, args.output)
    print(f"resample_size {args.resample_size}")
    print(f"seed {args.seed}")
    print(f"output {args.output}")
    print_comparison(rows)


if __name__ == "__main__":
    main()
