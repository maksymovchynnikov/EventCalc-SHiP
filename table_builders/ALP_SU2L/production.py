#  Generate DoubleDistr/Emax/Validation

import json
import os
from pathlib import Path
import numpy as np

#from funcs import boost
#from funcs import TwoBodyDecay
from .kinematics import (
    simulate_B_to_Xa_rest_frame_fast,
    boost_alp_rest_to_lab_fast,
)

from .constants import (
    N_BB_PER_POT,
    F_BPLUS,
    F_BZERO,
    THETA_MAX_TABLE,
    THETA_MAX_SHIP,
    SU2_OPERATOR_FACTOR,
)

from .branching import (
    get_Bplus_to_Xa_branching_ratios,
    load_scalar_br_table,
)

from .channels_file import (
    BPLUS_TO_XA_CHANNELS,
    M_B_PLUS,
    get_allowed_channels,
    effective_B_fragmentation_factor,
)

from .config import (
    MASSES_GEV,
    DEBUG_MASSES_GEV,
    COUPLING_NORMALIZATION_GEV_INV,
    RUN_MODE,
    OUTPUT_ROOT,
    OUTPUT_ROOT_PLOTS,
    B_MOMENTA_PATH,
    SCALAR_TABLE_PATH,
    rng,
    N_THETA_FORWARD,
    N_THETA_TAIL,
    DISTRIBUTION_FLOOR,
)

from .distribution import (
    polar_angle_from_momenta,
    theta_energy_from_momenta,
    make_distribution_table,
    make_emax_table,
    make_constant_emax_table,
    make_theta_edges,
    make_energy_edges,
    truncate_table_in_theta,
    integrate_density,
)

from .validation import (
    validate_with_eventcalc_interpolation,
    plot_debug_distributions,
    plot_B_theta_energy_distribution,
    plot_energy_spectrum_from_density,
)


# Input and test B-meson momenta
def generate_test_B_momenta(size, energy_B):
    """
    Generate a simple test sample of B mesons moving along the z-axis.

    Output convention: px, py, pz, E.
    """
    pz_B = np.sqrt(energy_B**2 - M_B_PLUS**2)

    return np.column_stack(
        (
            np.zeros(size),
            np.zeros(size),
            np.full(size, pz_B),
            np.full(size, energy_B),
        )
    )


def load_B_momenta(path, m_B=M_B_PLUS, check=True):
    """
    Load B-meson momenta.

    Input convention:
        px, py, pz, E_raw

    The raw E column is used only as a diagnostic. Because the file
    was truncated, we put each B meson back on shell using

        E = sqrt(px^2 + py^2 + pz^2 + m_B^2).

    Returned convention:
        px, py, pz, E_on_shell
    """

    def mathematica_float(x):
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        return float(x.replace("*^", "e"))

    data_raw = np.loadtxt(
        path,
        converters={
            0: mathematica_float,
            1: mathematica_float,
            2: mathematica_float,
            3: mathematica_float,
        },
    )

    if data_raw.ndim == 1:
        data_raw = data_raw.reshape(1, -1)

    if data_raw.shape[1] < 4:
        raise ValueError("Expected at least four columns: px, py, pz, E.")

    px = data_raw[:, 0]
    py = data_raw[:, 1]
    pz = data_raw[:, 2]
    E_raw = data_raw[:, 3]

    p2 = px**2 + py**2 + pz**2
    E_on_shell = np.sqrt(p2 + m_B**2)

    if check:
        m2_raw = E_raw**2 - p2
        rel_E_shift = (E_on_shell - E_raw) / E_on_shell

        print(f"Raw m^2: mean = {np.mean(m2_raw):.6g}, std = {np.std(m2_raw):.6g}")
        print(f"Expected m_B^2 = {m_B**2:.6g}")
        print(f"Max relative E correction = {np.max(np.abs(rel_E_shift)):.3e}")

    return np.column_stack((px, py, pz, E_on_shell))


def _B_momenta_cache_signature(path, m_B):
    source = Path(path)
    stat = source.stat()
    return {
        "version": 1,
        "source_path": str(source.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "m_B_GeV": float(m_B),
    }


def load_B_momenta_cached(path, cache_path=None, m_B=M_B_PLUS, check=True):
    """
    Load B momenta from a cached .npy file when its input metadata still matches.

    The cache depends on the source file and on the B-meson mass used to put the
    momenta on shell.
    """
    if cache_path is None:
        cache_path = str(path) + ".on_shell.npy"

    cache_path = Path(cache_path)
    metadata_path = Path(str(cache_path) + ".meta.json")
    signature = _B_momenta_cache_signature(path, m_B)

    cache_is_current = False
    if cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            cache_is_current = metadata == signature
        except (OSError, json.JSONDecodeError):
            cache_is_current = False

    if cache_is_current:
        B_momenta = np.load(cache_path)
        if check:
            p2 = np.sum(B_momenta[:, :3] ** 2, axis=1)
            m2 = B_momenta[:, 3] ** 2 - p2
            print(f"Loaded cached B momenta from {cache_path}")
            print(f"Cached m^2: mean = {np.mean(m2):.6g}, std = {np.std(m2):.6g}")
            print(f"Expected m_B^2 = {m_B**2:.6g}")
        return B_momenta

    B_momenta = load_B_momenta(path=path, m_B=m_B, check=check)
    np.save(cache_path, B_momenta)
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n")
    print(f"Saved cached B momenta to {cache_path}")

    return B_momenta


def production_probability_Bplus_reference(
    N_bb_per_POT,
    f_b_to_Bplus,
    f_b_to_B0,
    BR_Bplus_to_Xa_total,
    include_B0=True,
):
    """
    Production probability per proton on target using B+ as the reference BR.

    BR_Bplus_to_Xa_total is sum_X BR(B+ -> X + a).
    """
    f_eff = effective_B_fragmentation_factor(
        f_b_to_Bplus=f_b_to_Bplus,
        f_b_to_B0=f_b_to_B0,
        include_B0=include_B0,
    )

    return 2 * N_bb_per_POT * f_eff * BR_Bplus_to_Xa_total


def write_table(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        path,
        array,
        fmt="%.12g",
        delimiter="\t",
    )


def write_tsv_with_header(path, header, rows):
    """
    Write a tab-separated audit/validation file with a header.
    This is intended for diagnostics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("\t".join(header) + "\n")

        for row in rows:
            file.write("\t".join(str(x) for x in row) + "\n")


def build_B_to_Xa_tables():
    if RUN_MODE == "debug":
        masses = DEBUG_MASSES_GEV
        N_B_USED = None  #100000
        ifMakePlots = True
        ifValidateInterpolation = True
        ifShowPlots = False
        ifValidateConstantEmax = True
        ifSeeWeigths = True
        output_tag = "ALP-SU2L-debug"

    elif RUN_MODE == "final":
        masses = MASSES_GEV
        N_B_USED = None
        ifMakePlots = False
        ifValidateInterpolation = False
        ifShowPlots = False
        ifValidateConstantEmax = False
        ifSeeWeigths = False
        output_tag = "ALP-SU2L"

    #B_momenta = generate_test_B_momenta(size=100000, energy_B=50.0)
    B_momenta = load_B_momenta_cached(path=B_MOMENTA_PATH)

    if N_B_USED is not None:
        idx = rng.choice(len(B_momenta), size=N_B_USED, replace=False)
        B_momenta = B_momenta[idx]

    theta_B = polar_angle_from_momenta(B_momenta)
    energy_B = B_momenta[:, 3]

    fraction_B_ship = float(np.mean(theta_B < THETA_MAX_SHIP))
    fraction_B_table = float(np.mean(theta_B < THETA_MAX_TABLE))

    fraction_B_error = float(np.sqrt(fraction_B_table * (1.0 - fraction_B_table) / len(theta_B)))
    print(
        f"f_B(theta_B < {THETA_MAX_TABLE:.7f} rad) = "
        f"{fraction_B_table:.6e} ± {fraction_B_error:.2e}"
    )

    theta_edges_full = make_theta_edges(THETA_MAX_TABLE, N_THETA_FORWARD, N_THETA_TAIL)
    energy_edges = make_energy_edges(MASSES_GEV, B_momenta)
    channels = BPLUS_TO_XA_CHANNELS
    scalar_table = load_scalar_br_table(SCALAR_TABLE_PATH)

    houtz_cW_over_fa = SU2_OPERATOR_FACTOR * COUPLING_NORMALIZATION_GEV_INV

    if ifMakePlots:
        plot_B_theta_energy_distribution(
            theta_B,
            energy_B,
            fraction_B_ship,
            OUTPUT_ROOT_PLOTS,
            ifShowPlots,
        )

    total_yield_rows = []

    all_distribution_rows = []
    all_emax_rows = []
    all_emax_validation_rows = []
    fractions_out = []

    normalization_rows = []
    channel_rows = []
    validation_by_mass = {}

    for alp_mass in masses:
        print(f"\nProcessing m_a = {alp_mass:.6g} GeV")

        size = len(B_momenta)

        try:
            br_Ka, channel_brs_by_name, probs_by_name, br_Bplus_total = (
                get_Bplus_to_Xa_branching_ratios(
                    alp_mass=alp_mass,
                    cW_over_fa=houtz_cW_over_fa,
                    scalar_table_path=SCALAR_TABLE_PATH,
                    channels=channels,
                    scalar_table=scalar_table,
                )
            )
            br_pia = channel_brs_by_name.get("pi+", 0.0)

            if ifSeeWeigths:
                print("Channel weights:")
                for channel in get_allowed_channels(alp_mass, channels):
                    name = channel["name"]
                    print(
                        f"  {name:12s}  "
                        f"prob = {probs_by_name[name]:.4f}, "
                        f"BR = {channel_brs_by_name[name]:.4e}"
                    )

            alp_rest = simulate_B_to_Xa_rest_frame_fast(
                size=len(B_momenta),
                alp_mass=alp_mass,
                probabilities_by_name=probs_by_name,
                channels=channels,
            )
        except ValueError as error:
            print(f"Skipping m_a = {alp_mass:.6g} GeV: {error}")
            continue

        alp_lab = boost_alp_rest_to_lab_fast(
            M_B_PLUS,
            B_momenta,
            alp_rest,
        )

        theta, energy = theta_energy_from_momenta(alp_lab)
        distribution_table, _, density_full = make_distribution_table(
            alp_mass,
            theta,
            energy,
            theta_edges_full,
            energy_edges,
        )

        fraction_a_ship = float(np.mean(theta < THETA_MAX_SHIP))
        fraction_a_table = float(np.mean(theta < THETA_MAX_TABLE))
        fractions_out.append([alp_mass, fraction_a_table])
        print(f"Fraction with theta < {THETA_MAX_TABLE:.6g}: {fraction_a_table:.6e}")

        direct_integral = integrate_density(density_full, theta_edges_full, energy_edges)
        print(f"Direct histogram full integral: {direct_integral:.6f}")
        validation_by_mass[float(alp_mass)] = {
            "m_a": float(alp_mass),
            "n_B_used": int(size),
            "theta_cut_rad": float(THETA_MAX_TABLE),
            "direct_full_integral": float(direct_integral),
            "direct_theta_fraction": float(fraction_a_table),
            "eventcalc_full_integral": np.nan,
            "eventcalc_full_error": np.nan,
            "eventcalc_theta_integral": np.nan,
            "eventcalc_theta_error": np.nan,
        }

        p_prod = production_probability_Bplus_reference(
            N_bb_per_POT=N_BB_PER_POT,
            f_b_to_Bplus=F_BPLUS,
            f_b_to_B0=F_BZERO,
            BR_Bplus_to_Xa_total=br_Bplus_total,
            include_B0=True,
        )

        yield_coefficient = p_prod / COUPLING_NORMALIZATION_GEV_INV**2

        total_yield_rows.append(
            [
                alp_mass,
                yield_coefficient,
            ]
        )

        normalization_rows.append(
            [
                alp_mass,
                br_Ka,
                br_pia,
                br_Bplus_total,
                p_prod,
            ]
        )

        for channel in get_allowed_channels(alp_mass, channels):
            name = channel["name"]
            br_channel = channel_brs_by_name[name]
            prob_channel = probs_by_name[name]

            channel_rows.append(
                [
                    float(alp_mass),
                    name,
                    float(channel["mass"]),
                    float(br_channel),
                    float(prob_channel),
                    float(br_channel / br_Bplus_total) if br_Bplus_total > 0 else np.nan,
                ]
            )

        print(
            f"Br(B+ -> K+ a) at cW/fa = "
            f"{COUPLING_NORMALIZATION_GEV_INV:.1e} GeV^-1: "
            f"{br_Ka:.6e}"
        )

        print(
            f"Br(B+ -> pi+ a) at cW/fa = "
            f"{COUPLING_NORMALIZATION_GEV_INV:.1e} GeV^-1: "
            f"{br_pia:.6e}"
        )

        print(f"Sum Br(B+ -> X_s/d + a): {br_Bplus_total:.6e}")
        print(f"P(pN -> a) from B+,B0 approximation: {p_prod:.6e}")
        print(f"Sum of BR probabilities: {sum(probs_by_name.values()):.6f}")

        emax_table = make_emax_table(alp_mass, theta_edges_full, energy_edges, density_full)
        emax_table_validation = make_constant_emax_table(alp_mass, theta_edges_full, energy_edges)
        all_emax_validation_rows.append(emax_table_validation)

        if ifMakePlots:
            plot_debug_distributions(
                alp_mass,
                theta,
                energy,
                fraction_a_ship,
                fraction_B_ship,
                OUTPUT_ROOT_PLOTS,
                ifShowPlots,
            )
            plot_energy_spectrum_from_density(
                alp_mass,
                theta_edges_full,
                energy_edges,
                density_full,
                OUTPUT_ROOT_PLOTS,
                ifShowPlots,
            )

        all_distribution_rows.append(distribution_table)
        all_emax_rows.append(emax_table)

    if not all_distribution_rows:
        raise RuntimeError("No valid ALP masses were processed.")

    distribution_table = np.vstack(all_distribution_rows)
    emax_table = np.vstack(all_emax_rows)
    emax_table_validation = np.vstack(all_emax_validation_rows)
    fractions_out = dict(fractions_out)

    if ifValidateInterpolation:
        for alp_mass in masses:
            integral, mc_error = validate_with_eventcalc_interpolation(
                distribution_table,
                emax_table,
                alp_mass,
                theta_max_sim=np.pi,
                n_points=100000,
            )
            full_tolerance = max(
                5.0 * mc_error,
                0.03,
            )

            if abs(integral - 1.0) > full_tolerance:
                raise RuntimeError(
                    "EventCalc interpolation failed for "
                    f"m_a={alp_mass:g} GeV: "
                    f"integral={integral:.6g}, "
                    f"MC error={mc_error:.3g}"
                )

            print(
                f"\nEventCalc-interpolated full integral for m_a = {alp_mass:g} GeV: "
                f"{integral:.6f} ± {mc_error:.2e}"
            )

            integral_theta, mc_error_theta = validate_with_eventcalc_interpolation(
                distribution_table,
                emax_table,
                alp_mass,
                theta_max_sim=THETA_MAX_TABLE,
                n_points=100000,
            )
            direct_fraction = fractions_out[float(alp_mass)]

            theta_tolerance = max(
                5.0 * mc_error_theta,
                0.05 * direct_fraction,
                1.0e-4,
            )

            if abs(integral_theta - direct_fraction) > theta_tolerance:
                raise RuntimeError(
                    "EventCalc interpolation disagrees with the "
                    "direct forward fraction for "
                    f"m_a={alp_mass:g} GeV: "
                    f"interpolated={integral_theta:.6g}, "
                    f"direct={direct_fraction:.6g}"
                )

            print(
                f"EventCalc-interpolated SHiP-angle integral for m_a = {alp_mass:g} GeV: "
                f"{integral_theta:.6f} ± {mc_error_theta:.2e}; "
                f"direct fraction = {fractions_out[alp_mass]:.6e}"
            )

            if float(alp_mass) in validation_by_mass:
                validation_by_mass[float(alp_mass)]["eventcalc_full_integral"] = float(integral)
                validation_by_mass[float(alp_mass)]["eventcalc_full_error"] = float(mc_error)
                validation_by_mass[float(alp_mass)]["eventcalc_theta_integral"] = float(
                    integral_theta
                )
                validation_by_mass[float(alp_mass)]["eventcalc_theta_error"] = float(mc_error_theta)

            if ifValidateConstantEmax:
                integral_validation, mc_error_validation = validate_with_eventcalc_interpolation(
                    distribution_table,
                    emax_table_validation,
                    alp_mass,
                    theta_max_sim=np.pi,
                    n_points=100000,
                )
                print(
                    f"EventCalc interpolation with constant Emax for m_a = {alp_mass:g} GeV: "
                    f"{integral_validation:.6f} ± {mc_error_validation:.2e}"
                )

    if RUN_MODE == "final":
        distribution_table_to_write = truncate_table_in_theta(
            distribution_table,
            theta_cut=THETA_MAX_TABLE,
        )

        emax_table_to_write = truncate_table_in_theta(
            emax_table,
            theta_cut=THETA_MAX_TABLE,
        )

        distribution_table_to_write = distribution_table_to_write.copy()
        zero_mask = distribution_table_to_write[:, 3] <= 0.0
        number_of_floored_bins = np.count_nonzero(zero_mask)
        distribution_table_to_write[zero_mask, 3] = DISTRIBUTION_FLOOR

        print(
            f"Replaced {number_of_floored_bins:,} empty distribution bins "
            f"with {DISTRIBUTION_FLOOR:.0e}."
        )

        print(f"\nWriting final truncated tables with theta < {THETA_MAX_TABLE:.6g} rad.")

    else:
        distribution_table_to_write = distribution_table
        emax_table_to_write = emax_table
        print("\nWriting debug tables over the full theta range.")

    if RUN_MODE == "debug":
        write_table(
            OUTPUT_ROOT / f"DoubleDistr-{output_tag}.txt",
            distribution_table_to_write,
        )

        write_table(
            OUTPUT_ROOT / f"Emax-{output_tag}.txt",
            emax_table_to_write,
        )

        write_tsv_with_header(
            OUTPUT_ROOT / f"B-angle-{output_tag}.txt",
            header=[
                "n_B_used",
                "theta_B_cut_rad",
                "fraction_B",
                "binomial_MC_error",
            ],
            rows=[
                [
                    len(B_momenta),
                    THETA_MAX_TABLE,
                    fraction_B_table,
                    fraction_B_error,
                ]
            ],
        )

        if normalization_rows:
            write_table(
                OUTPUT_ROOT / f"Pprod-{output_tag}.txt",
                np.asarray(normalization_rows),
            )

        if channel_rows:
            write_tsv_with_header(
                OUTPUT_ROOT / f"BRchannels-{output_tag}.txt",
                header=[
                    "m_a_GeV",
                    "channel_name",
                    "m_X_GeV",
                    "BR_Bplus_to_Xa",
                    "probability_used_in_MC",
                    "BR_channel_over_sum_BR",
                ],
                rows=channel_rows,
            )

        if validation_by_mass:
            validation_rows = []

            for _, values in sorted(validation_by_mass.items()):
                validation_rows.append(
                    [
                        values["m_a"],
                        values["n_B_used"],
                        values["theta_cut_rad"],
                        values["direct_full_integral"],
                        values["direct_theta_fraction"],
                        values["eventcalc_full_integral"],
                        values["eventcalc_full_error"],
                        values["eventcalc_theta_integral"],
                        values["eventcalc_theta_error"],
                    ]
                )

            write_tsv_with_header(
                os.path.join(OUTPUT_ROOT, f"Validation-{output_tag}.txt"),
                header=[
                    "m_a_GeV",
                    "n_B_used",
                    "theta_cut_rad",
                    "direct_full_integral",
                    "direct_theta_fraction",
                    "eventcalc_full_integral",
                    "eventcalc_full_error",
                    "eventcalc_theta_integral",
                    "eventcalc_theta_error",
                ],
                rows=validation_rows,
            )

    if RUN_MODE == "final":
        write_table(
            OUTPUT_ROOT / "DoubleDistr-ALP-SU2L.txt",
            distribution_table_to_write,
        )

        write_table(
            OUTPUT_ROOT / "Emax-ALP-SU2L.txt",
            emax_table_to_write,
        )

        write_table(
            OUTPUT_ROOT / "Total-yield-ALP-SU2L.txt",
            np.asarray(total_yield_rows),
        )

    print("\nDone.")

    return OUTPUT_ROOT
