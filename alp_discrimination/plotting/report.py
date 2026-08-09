"""Report-facing plots for the ALP model-discrimination analysis.

Only completed, saved analysis products are read here.  No EventCalc sampling or
pseudoexperiments are launched by this module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alp_discrimination.templates.conditional_features import (
    FEATURE_LABELS,
    FEATURE_SUBSETS,
    load_conditional_feature_moments,
    pairwise_joint_energy_feature_hellinger_squared,
    validate_conditional_feature_moments,
)
from alp_discrimination.templates.lifetime_banks import load_template_bank
from alp_discrimination.workflows import float_token


SELECTION_LABELS = {
    "diphoton_ecal": "ECAL only",
    "diphoton_ecal_e1gev": r"ECAL + $E_\gamma\geq1$ GeV",
}
SELECTION_TOKENS = {
    "diphoton_ecal": "ecal",
    "diphoton_ecal_e1gev": "ecal_e1gev",
}
OBSERVABLE_TOKENS = {
    "energy": "energy",
    "energy_mean_z": "energy_z",
    "energy_mean_r_perp": "energy_r_perp",
    "energy_mean_z_r_perp": "energy_z_r_perp",
}


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def selection_token(name: str) -> str:
    return SELECTION_TOKENS.get(str(name), str(name))


def observable_token(name: str) -> str:
    return OBSERVABLE_TOKENS.get(str(name), str(name))


OBSERVABLE_TITLES = {
    "energy": r"$E_a$",
    "energy_mean_z": r"$E_a,\langle z_d\rangle$",
    "energy_mean_r_perp": r"$E_a,\langle r_\perp\rangle$",
    "energy_mean_z_r_perp": (
        r"$E_a,\langle z_d\rangle,\langle r_\perp\rangle$"
    ),
}


def observable_title(name: str) -> str:
    return OBSERVABLE_TITLES.get(str(name), FEATURE_LABELS.get(name, str(name)))


def mass_observable_title(mass: float, observable: str) -> str:
    return rf"$m_a={float(mass):g}$ GeV, " + observable_title(observable)


def plot_classification_accuracy(
    *,
    mass: float,
    selection: str,
    observable: str,
    n90: int,
    high_statistics_curve: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    empirical_comparison: pd.DataFrame,
    pseudoexperiments: int,
    output_dir: Path,
) -> None:
    """Plot the main P(correct model | N) result and its validation comparison."""

    if high_statistics_curve.empty:
        return
    token = (
        f"ma_{float_token(mass)}_{selection_token(selection)}_"
        f"{observable_token(observable)}"
    )
    label = (
        f"High-statistics validation ({pseudoexperiments // 1000}k)"
        if pseudoexperiments >= 1000 and pseudoexperiments % 1000 == 0
        else "High-statistics validation"
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(
        high_statistics_curve["number_of_events"],
        high_statistics_curve["worst_case_accuracy"],
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=label,
    )
    if not empirical_comparison.empty:
        ax.plot(
            empirical_comparison["number_of_events"],
            empirical_comparison["empirical_truth"],
            linestyle="none",
            marker="s",
            markersize=5,
            label="Direct EventCalc resampling",
        )
    ax.axhline(0.90, linestyle="--", linewidth=1.0, label="90% target")
    ax.axvline(n90, linestyle=":", linewidth=1.0)
    ax.annotate(
        rf"$N_{{90}}={n90}$",
        xy=(n90, 0.90),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
    )
    ymin = max(
        0.45,
        float(high_statistics_curve["worst_case_accuracy"].min()) - 0.035,
    )
    ax.set_ylim(ymin, 1.005)
    ax.set_xlabel("Observed ALP decays, $N$")
    ax.set_ylabel("Worst-case correct-classification probability")
    ax.set_title(mass_observable_title(mass, observable))
    xmax = float(high_statistics_curve["number_of_events"].max())
    note_x = min(float(n90) + 0.8, xmax - 0.5)
    note_y = ymin + 0.035 * (1.005 - ymin)
    ax.text(
        note_x,
        note_y,
        SELECTION_LABELS.get(selection, selection),
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, output_dir / f"classification_accuracy_{token}")

    if validation_comparison.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(
        validation_comparison["number_of_events"],
        validation_comparison["full_domain_2k_accuracy"],
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="All allowed lifetimes, 2k",
    )
    ax.plot(
        validation_comparison["number_of_events"],
        validation_comparison["selected_5k_accuracy"],
        marker="o",
        markersize=3,
        linewidth=1.7,
        label=label,
    )
    ax.axhline(0.90, linestyle="--", linewidth=1.0, label="90% target")
    ax.axvline(n90, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Observed ALP decays, $N$")
    ax.set_ylabel("Worst-case correct-classification probability")
    validation_ymin = max(
        0.45,
        float(
            min(
                validation_comparison["full_domain_2k_accuracy"].min(),
                validation_comparison["selected_5k_accuracy"].min(),
            )
        )
        - 0.035,
    )
    ax.set_ylim(validation_ymin, 1.005)
    ax.set_title(mass_observable_title(mass, observable))
    xmax = float(validation_comparison["number_of_events"].max())
    note_x = min(float(n90) + 0.8, xmax - 0.5)
    note_y = validation_ymin + 0.035 * (1.005 - validation_ymin)
    ax.text(
        note_x,
        note_y,
        SELECTION_LABELS.get(selection, selection),
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, output_dir / f"classification_validation_{token}")


def pairwise_total_variation(
    photon_probabilities: np.ndarray,
    su2_probabilities: np.ndarray,
) -> np.ndarray:
    photon = np.asarray(photon_probabilities, dtype=float)
    su2 = np.asarray(su2_probabilities, dtype=float)
    return 0.5 * np.sum(
        np.abs(photon[:, None, :] - su2[None, :, :]),
        axis=2,
    )


def _draw_allowed_blocks(ax, bank, values: np.ndarray):
    p_ctau = np.asarray(bank.photon_ctau_m, dtype=float)
    s_ctau = np.asarray(bank.su2_ctau_m, dtype=float)
    p_interval = np.asarray(bank.photon_interval_index, dtype=int)
    s_interval = np.asarray(bank.su2_interval_index, dtype=int)
    ax.set_facecolor("lightgray")
    image = None
    for p_value in np.unique(p_interval):
        p_idx = np.flatnonzero(p_interval == p_value)
        for s_value in np.unique(s_interval):
            s_idx = np.flatnonzero(s_interval == s_value)
            image = ax.pcolormesh(
                p_ctau[p_idx],
                s_ctau[s_idx],
                values[np.ix_(p_idx, s_idx)].T,
                shading="nearest",
                vmin=0.0,
                vmax=1.0,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(p_ctau.min(), p_ctau.max())
    ax.set_ylim(s_ctau.min(), s_ctau.max())
    ax.set_xlabel(r"Photophilic $c\tau$ [m]")
    return image


def plot_distance_diagnostics(
    *,
    bank_path: Path,
    moments_path: Path,
    selection: str,
    output_dir: Path,
    data_dir: Path,
) -> pd.DataFrame:
    """Create the compact report-style DTV/H2 lifetime-domain figure."""

    bank = load_template_bank(bank_path)
    moments = load_conditional_feature_moments(moments_path)
    validate_conditional_feature_moments(moments, bank)
    mass = float(bank.mass_gev)

    dtv = pairwise_total_variation(
        bank.photon_probabilities,
        bank.su2_probabilities,
    )
    h2 = pairwise_joint_energy_feature_hellinger_squared(
        photon_probabilities=bank.photon_probabilities,
        photon_means=moments["photon_feature_mean"],
        photon_covariances=moments["photon_feature_covariance"],
        su2_probabilities=bank.su2_probabilities,
        su2_means=moments["su2_feature_mean"],
        su2_covariances=moments["su2_feature_covariance"],
        feature_indices=FEATURE_SUBSETS["energy_mean_z_r_perp"],
    )
    p_dtv, s_dtv = np.unravel_index(int(np.argmin(dtv)), dtv.shape)
    p_h2, s_h2 = np.unravel_index(int(np.argmin(h2)), h2.shape)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    image = _draw_allowed_blocks(axes[0], bank, dtv)
    _draw_allowed_blocks(axes[1], bank, h2)
    axes[0].set_ylabel(r"$SU(2)_L$ $c\tau$ [m]")
    axes[0].set_title(r"(a) $E_a$: $D_{\rm TV}$", fontsize=10)
    axes[1].set_title(
        r"(b) $E_a,\langle z_d\rangle,\langle r_\perp\rangle$: $H^2$",
        fontsize=10,
    )
    for ax, p_index, s_index in (
        (axes[0], p_dtv, s_dtv),
        (axes[1], p_h2, s_h2),
    ):
        ax.plot(
            bank.photon_ctau_m[p_index],
            bank.su2_ctau_m[s_index],
            marker="*",
            markersize=10,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="none",
        )
    fig.suptitle(rf"$m_a={mass:g}$ GeV", fontsize=10)
    fig.subplots_adjust(
        left=0.10,
        right=0.86,
        bottom=0.17,
        top=0.82,
        wspace=0.12,
    )
    colorbar_axis = fig.add_axes([0.875, 0.17, 0.020, 0.65])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Distance diagnostic")
    stem = f"distance_diagnostics_ma_{float_token(mass)}_{selection_token(selection)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / f"{stem}.npz",
        photon_ctau_m=np.asarray(bank.photon_ctau_m, dtype=float),
        su2_ctau_m=np.asarray(bank.su2_ctau_m, dtype=float),
        photon_interval_index=np.asarray(bank.photon_interval_index, dtype=int),
        su2_interval_index=np.asarray(bank.su2_interval_index, dtype=int),
        energy_total_variation=dtv,
        joint_hellinger_squared=h2,
    )
    return pd.DataFrame(
        [
            {
                "mass_GeV": mass,
                "selection_name": selection,
                "distance": "energy_total_variation",
                "minimum": float(dtv[p_dtv, s_dtv]),
                "photon_ctau_m": float(bank.photon_ctau_m[p_dtv]),
                "su2_ctau_m": float(bank.su2_ctau_m[s_dtv]),
            },
            {
                "mass_GeV": mass,
                "selection_name": selection,
                "distance": "joint_hellinger_squared",
                "minimum": float(h2[p_h2, s_h2]),
                "photon_ctau_m": float(bank.photon_ctau_m[p_h2]),
                "su2_ctau_m": float(bank.su2_ctau_m[s_h2]),
            },
        ]
    )


def plot_observable_comparison(
    *,
    results: pd.DataFrame,
    mass: float,
    selection: str,
    output_dir: Path,
) -> None:
    if results.empty or results["observable"].nunique() < 2:
        return
    results = results.sort_values("N90", ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(results))
    ax.bar(x, results["N90"].to_numpy(dtype=float))
    ax.set_xticks(
        x,
        [FEATURE_LABELS.get(name, name) for name in results["observable"]],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel(r"Minimum observed events, $N_{90}$")
    ax.set_title(rf"$m_a={mass:g}$ GeV")
    ax.grid(axis="y", alpha=0.22)
    _save(
        fig,
        output_dir
        / f"observable_comparison_ma_{float_token(mass)}_{selection_token(selection)}",
    )


def plot_n90_vs_mass(thresholds: pd.DataFrame, output_dir: Path) -> None:
    if thresholds.empty:
        return
    table = thresholds[
        thresholds["observable"].astype(str) == "energy_mean_z_r_perp"
    ].copy()
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for selection, group in table.groupby("selection_name"):
        group = group.sort_values("mass_GeV")
        ax.plot(
            group["mass_GeV"],
            group["N90"],
            marker="o",
            linewidth=1.7,
            label=SELECTION_LABELS.get(selection, selection),
        )
    ax.set_xlabel(r"ALP mass, $m_a$ [GeV]")
    ax.set_ylabel(r"Minimum observed events, $N_{90}$")
    ax.set_title(observable_title("energy_mean_z_r_perp"))
    ax.grid(alpha=0.22)
    ax.legend()
    _save(fig, output_dir / "n90_vs_mass")
