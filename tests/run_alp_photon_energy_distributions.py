#!/usr/bin/env python3
"""
Event-level ALP-photon check for m_a = 100 MeV and c*tau = 300 m.

This is a non-interactive version of the ALP part of simulate.py. It samples
ALP lab-frame events, applies the SHiP azimuthal/decay-volume selection, and
histograms E_ALP with per-event decay-probability weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


MASS_GEV = 0.1
C_TAU_M = 300.0
N_POT = 6.0e20
MODES = ("primary", "cascades")
ALPETITE_ONE_INTERACTION_LENGTH_FACTOR = 1.0 / (1.0 - math.exp(-1.0))
ALPETITE_RENORMALIZATION = ALPETITE_ONE_INTERACTION_LENGTH_FACTOR
COMPARISON_SPECTRA = (
    ("SensCalc", "primary", "dNevdEa-100-MeV-primary.txt", "dash", 1.0),
    ("SensCalc", "cascades", "dNevdEa-100-MeV-cascades.txt", "dash", 1.0),
    ("ALPETITE", "primary", "dNevdEa-100-MeV-primary-ALPETITE.txt", "dot", ALPETITE_RENORMALIZATION),
    ("ALPETITE", "cascades", "dNevdEa-100-MeV-cascades-ALPETITE.txt", "dot", ALPETITE_RENORMALIZATION),
)

DEFAULT_RESAMPLE_SIZE = 500_000
DEFAULT_SEED = 12345
N_RAW_MULTIPLIER = 10
N_BINS = 70

Z_MIN = 32.0
Z_MAX = 82.0
DELTA_X_IN = 1.0
DELTA_X_OUT = 4.0
DELTA_Y_IN = 2.7
DELTA_Y_OUT = 6.2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ALP_DIR = REPO_ROOT / "Distributions" / "ALP-photon"
PALATINO_TTC = Path("/System/Library/Fonts/Palatino.ttc")


def register_plot_fonts():
    if PALATINO_TTC.exists():
        try:
            pdfmetrics.registerFont(TTFont("Palatino-Roman", str(PALATINO_TTC), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("Palatino-Italic", str(PALATINO_TTC), subfontIndex=1))
            pdfmetrics.registerFont(TTFont("Palatino-Bold", str(PALATINO_TTC), subfontIndex=2))
            pdfmetrics.registerFont(TTFont("Palatino-BoldItalic", str(PALATINO_TTC), subfontIndex=3))
            return "Palatino-Roman", "Palatino-Italic", "Palatino-Bold", "Palatino-BoldItalic"
        except Exception:
            pass
    return "Times-Roman", "Times-Italic", "Times-Bold", "Times-BoldItalic"


FONT_REGULAR, FONT_ITALIC, FONT_BOLD, FONT_BOLD_ITALIC = register_plot_fonts()


def output_stem(output_suffix=""):
    mass_mev = int(round(MASS_GEV * 1000.0))
    ctau_m = int(round(C_TAU_M))
    return f"alp_photon_{mass_mev}MeV_{ctau_m}m{output_suffix}"


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


def linear_interp_extrap(x, xp, fp):
    values = np.interp(x, xp, fp)
    left = x < xp[0]
    if np.any(left):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        values[left] = fp[0] + slope * (x[left] - xp[0])
    right = x > xp[-1]
    if np.any(right):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        values[right] = fp[-1] + slope * (x[right] - xp[-1])
    return values


def load_mode_tables(mode: str):
    distribution = np.loadtxt(ALP_DIR / f"DoubleDistr-ALP-photon_{mode}.txt")
    energy_max = np.loadtxt(ALP_DIR / f"Emax-ALP-photon_{mode}.txt")

    distribution = distribution[np.isclose(distribution[:, 0], MASS_GEV, rtol=0.0, atol=1.0e-12)]
    energy_max = energy_max[np.isclose(energy_max[:, 0], MASS_GEV, rtol=0.0, atol=1.0e-12)]
    if distribution.size == 0 or energy_max.size == 0:
        raise ValueError(f"No ALP-photon table entries found for mass {MASS_GEV:g} GeV in {mode}.")

    theta_nodes = np.unique(distribution[:, 1])
    energy_nodes = np.unique(distribution[:, 2])
    values = np.zeros((len(theta_nodes), len(energy_nodes)))
    theta_index = np.searchsorted(theta_nodes, distribution[:, 1])
    energy_index = np.searchsorted(energy_nodes, distribution[:, 2])
    values[theta_index, energy_index] = distribution[:, 3]

    emax_theta = energy_max[:, 1]
    emax_values = energy_max[:, 2]
    order = np.argsort(emax_theta)
    return theta_nodes, energy_nodes, values, emax_theta[order], emax_values[order]


def bilinear_distribution(theta, energy, theta_nodes, energy_nodes, values, max_energy):
    theta_pos = np.searchsorted(theta_nodes, theta, side="left") - 1
    energy_pos = np.searchsorted(energy_nodes, energy, side="left") - 1
    theta_pos = np.clip(theta_pos, 0, len(theta_nodes) - 2)
    energy_pos = np.clip(energy_pos, 0, len(energy_nodes) - 2)

    theta_1 = theta_nodes[theta_pos]
    theta_2 = theta_nodes[theta_pos + 1]
    energy_1 = energy_nodes[energy_pos]
    energy_2 = energy_nodes[energy_pos + 1]

    theta_weight = (theta - theta_1) / (theta_2 - theta_1)
    energy_weight = (energy - energy_1) / (energy_2 - energy_1)

    v11 = values[theta_pos, energy_pos]
    v21 = values[theta_pos + 1, energy_pos]
    v12 = values[theta_pos, energy_pos + 1]
    v22 = values[theta_pos + 1, energy_pos + 1]

    v1 = v11 * (1.0 - theta_weight) + v21 * theta_weight
    v2 = v12 * (1.0 - theta_weight) + v22 * theta_weight
    interpolated = v1 * (1.0 - energy_weight) + v2 * energy_weight
    interpolated = np.where(energy > max_energy, 0.0, interpolated)
    return np.maximum(interpolated, 0.0)


def simulate_mode(
    mode: str,
    resample_size: int,
    rng: np.random.Generator,
    visible_br: float,
    coupling_squared: float,
    theta_preselection_max: float | None = None,
):
    theta_nodes, energy_nodes, values, emax_theta, emax_values = load_mode_tables(mode)
    theta_min = float(theta_nodes.min())
    theta_max = min(float(theta_nodes.max()), theta_max_dec_vol())
    raw_size = resample_size * N_RAW_MULTIPLIER

    theta = rng.uniform(theta_min, theta_max, raw_size)
    max_energy = linear_interp_extrap(theta, emax_theta, emax_values)
    e_min_sampling = np.maximum(
        MASS_GEV,
        np.minimum(2.133 * MASS_GEV / C_TAU_M, 0.5 * max_energy),
    )
    energy = rng.uniform(e_min_sampling, max_energy)
    density = bilinear_distribution(theta, energy, theta_nodes, energy_nodes, values, max_energy)

    polar_weights = density * (max_energy - e_min_sampling)
    positive = polar_weights > 0.0
    if not np.any(positive):
        raise ValueError(f"All polar weights are zero for mode {mode}.")

    epsilon_polar = float(polar_weights.sum() * (theta_max - theta_min) / raw_size)
    probabilities = polar_weights / polar_weights.sum()
    sampled_indices = rng.choice(raw_size, size=resample_size, replace=True, p=probabilities)
    sampled_theta = theta[sampled_indices]
    sampled_energy = energy[sampled_indices]

    phi = rng.uniform(-math.pi, math.pi, resample_size)
    momentum_abs = np.sqrt(sampled_energy * sampled_energy - MASS_GEV * MASS_GEV)
    cos_theta = np.cos(sampled_theta)

    exponent_min = -Z_MIN * MASS_GEV / (cos_theta * C_TAU_M * momentum_abs)
    exponent_max = -Z_MAX * MASS_GEV / (cos_theta * C_TAU_M * momentum_abs)
    cmin = 1.0 - np.exp(exponent_min)
    cmax = 1.0 - np.exp(exponent_max)
    c = rng.uniform(cmin, cmax)
    safe_c = np.minimum(c, 0.9999999995)
    z = cos_theta * C_TAU_M * (momentum_abs / MASS_GEV) * np.log(1.0 / (1.0 - safe_c))

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
    accepted_energy = sampled_energy[accepted]
    accepted_theta = sampled_theta[accepted]
    accepted_p_decay = p_decay[accepted]
    if theta_preselection_max is None:
        theta_preselection = np.ones_like(accepted_p_decay, dtype=bool)
    else:
        theta_preselection = accepted_theta < theta_preselection_max

    yield_per_pot = interpolate_table(ALP_DIR / f"Total-yield-ALP-photon_{mode}.txt", MASS_GEV)
    produced_llps = N_POT * yield_per_pot * coupling_squared
    event_weight_prefactor = produced_llps * visible_br * epsilon_polar / resample_size
    event_weights = event_weight_prefactor * accepted_p_decay
    event_weights = np.where(theta_preselection, event_weights, 0.0)

    epsilon_azimuthal = float(accepted.sum() / resample_size)
    theta_preselected_sample_events = int(theta_preselection.sum())
    epsilon_theta_preselected = float(theta_preselected_sample_events / resample_size)
    selected_p_decay = accepted_p_decay[theta_preselection]
    p_decay_averaged = float(selected_p_decay.mean()) if selected_p_decay.size else 0.0
    total_events = float(event_weights.sum())
    total_events_formula = produced_llps * visible_br * epsilon_polar * epsilon_theta_preselected * p_decay_averaged

    return {
        "mode": mode,
        "yield_per_pot": yield_per_pot,
        "produced_llps": produced_llps,
        "epsilon_polar": epsilon_polar,
        "epsilon_azimuthal": epsilon_azimuthal,
        "epsilon_theta_preselected": epsilon_theta_preselected,
        "p_decay_averaged": p_decay_averaged,
        "sampled_events": resample_size,
        "accepted_sample_events": int(accepted.sum()),
        "theta_preselected_sample_events": theta_preselected_sample_events,
        "theta_preselection_max": theta_preselection_max,
        "accepted_energy": accepted_energy,
        "event_weights": event_weights,
        "total_events": total_events,
        "total_events_formula": total_events_formula,
    }


def make_histograms(results):
    all_energies = np.concatenate([result["accepted_energy"] for result in results])
    min_energy = max(MASS_GEV, float(all_energies.min()))
    max_energy = float(all_energies.max())
    bins = np.geomspace(min_energy, max_energy, N_BINS + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    widths = np.diff(bins)

    for result in results:
        counts, _ = np.histogram(result["accepted_energy"], bins=bins, weights=result["event_weights"])
        result["bin_centers"] = centers
        result["dnde"] = counts / widths
        result["hist_total_events"] = float(counts.sum())

    return bins


def load_numeric_table(path: Path):
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.asarray(rows, dtype=float)


def load_comparison_curves():
    curves = []
    for source, mode, filename, style, scale in COMPARISON_SPECTRA:
        path = SCRIPT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing comparison spectrum: {path}")
        data = load_numeric_table(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in comparison spectrum: {path}")
        energy = data[:, 0]
        dnde = data[:, 1]
        positive = (energy > 0.0) & (dnde > 0.0)
        order = np.argsort(energy[positive])
        energy = energy[positive][order]
        dnde = dnde[positive][order] * scale
        curves.append({
            "source": source,
            "mode": mode,
            "style": style,
            "label": f"{source} {mode}",
            "energy": energy,
            "dnde": dnde,
            "scale": scale,
            "path": path,
            "total_trapezoid": float(np.trapezoid(dnde, energy)),
        })
    return curves


def format_sci(value: float) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10 ** exponent
    return f"{mantissa:.3g}e{exponent:d}"


def log_ticks(vmin: float, vmax: float):
    start = math.floor(math.log10(vmin))
    stop = math.ceil(math.log10(vmax))
    bases = (1.0,) if stop - start > 8 else (1.0, 2.0, 5.0)
    ticks = []
    for exponent in range(start, stop + 1):
        for base in bases:
            value = base * 10 ** exponent
            if vmin <= value <= vmax:
                ticks.append(value)
    return ticks


def set_curve_dash(pdf, style):
    if style == "dash":
        pdf.setDash(7, 4)
    elif style == "dot":
        pdf.setDash(1.5, 4)
    else:
        pdf.setDash()


def eventcalc_label(result):
    return f"EVENTCALC {display_mode(result['mode'])}"


def display_mode(mode: str) -> str:
    return "secondary" if mode == "cascades" else mode


def source_label(source: str) -> str:
    return {
        "EventCalc": "EVENTCALC",
        "SensCalc": "SENSCALC",
        "ALPETITE": "ALPETITE",
    }.get(source, source.upper())


def draw_rich_text(pdf, x: float, y: float, chunks, center: bool = False, right: bool = False):
    total_width = sum(pdf.stringWidth(text, font, size) for text, font, size, _ in chunks)
    if center:
        cursor = x - total_width / 2.0
    elif right:
        cursor = x - total_width
    else:
        cursor = x
    for text, font, size, dy in chunks:
        pdf.setFont(font, size)
        pdf.drawString(cursor, y + dy, text)
        cursor += pdf.stringWidth(text, font, size)


def draw_x_axis_label(pdf, x: float, y: float, size: float):
    draw_rich_text(
        pdf,
        x,
        y,
        [
            ("E", FONT_ITALIC, size, 0),
            ("a", FONT_ITALIC, size * 0.66, -0.28 * size),
            (" [GeV]", FONT_REGULAR, size, 0),
        ],
        center=True,
    )


def draw_y_axis_label(pdf, x: float, y: float, size: float):
    pdf.saveState()
    pdf.translate(x, y)
    pdf.rotate(90)
    draw_rich_text(
        pdf,
        0,
        0,
        [
            ("dN", FONT_ITALIC, size, 0),
            ("events", FONT_REGULAR, size * 0.58, -0.26 * size),
            (" / dE", FONT_ITALIC, size, 0),
            ("a", FONT_ITALIC, size * 0.66, -0.28 * size),
            (" [GeV", FONT_REGULAR, size, 0),
            ("-1", FONT_REGULAR, size * 0.62, 0.34 * size),
            ("]", FONT_REGULAR, size, 0),
        ],
        center=True,
    )
    pdf.restoreState()


def decade_ticks(vmin: float, vmax: float):
    start = math.floor(math.log10(vmin))
    stop = math.ceil(math.log10(vmax))
    return [10.0 ** exponent for exponent in range(start, stop + 1) if vmin <= 10.0 ** exponent <= vmax]


def draw_decade_label(pdf, x: float, y: float, exponent: int, size: float, center: bool = False, right: bool = False):
    if exponent == 0:
        draw_rich_text(pdf, x, y, [("1", FONT_REGULAR, size, 0)], center=center, right=right)
    else:
        draw_rich_text(
            pdf,
            x,
            y,
            [
                ("10", FONT_REGULAR, size, 0),
                (str(exponent), FONT_REGULAR, size * 0.62, 0.34 * size),
            ],
            center=center,
            right=right,
        )


def draw_focused_plot_label(pdf, x: float, y: float, theta_preselection_max: float):
    size = 21
    draw_rich_text(
        pdf,
        x,
        y,
        [
            ("m", FONT_ITALIC, size, 0),
            ("a", FONT_ITALIC, size * 0.66, -0.28 * size),
            (" = 100 MeV,  c", FONT_REGULAR, size, 0),
            ("τ", FONT_REGULAR, size, 0),
            ("a", FONT_ITALIC, size * 0.66, -0.28 * size),
            (f" = {int(round(C_TAU_M))} m,  ", FONT_REGULAR, size, 0),
            ("θ", FONT_REGULAR, size, 0),
            ("a", FONT_ITALIC, size * 0.66, -0.28 * size),
            (f" < {theta_preselection_max:.3f} rad", FONT_REGULAR, size, 0),
        ],
        center=True,
    )


def draw_pdf(
    results,
    comparison_curves,
    output_path: Path,
    theta_preselection_max: float | None = None,
    focused_eventcalc_alpetite: bool = False,
):
    plot_series = [result["dnde"][result["dnde"] > 0.0] for result in results]
    plot_series.extend(curve["dnde"] for curve in comparison_curves)
    positive_y = np.concatenate(
        plot_series
    )
    positive_x = np.concatenate(
        [result["bin_centers"][result["dnde"] > 0.0] for result in results]
        + [curve["energy"] for curve in comparison_curves]
    )
    x_min = float(positive_x.min())
    x_max_plot = float(positive_x.max())
    y_max_raw = float(positive_y.max())
    y_min = min(float(series.max()) * 1.0e-8 for series in plot_series)
    y_max_plot = y_max_raw * 2.0

    width, height = landscape(letter)
    if focused_eventcalc_alpetite:
        margin_left, margin_right = 112, 38
        margin_bottom = 92
        plot_w = width - margin_left - margin_right
        plot_h = 0.6 * plot_w
    else:
        margin_left, margin_right = 112, 38
        margin_bottom, margin_top = 104, 184
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_bottom - margin_top

    def map_x(value):
        return margin_left + (math.log10(value) - math.log10(x_min)) / (
            math.log10(x_max_plot) - math.log10(x_min)
        ) * plot_w

    def map_y(value):
        return margin_bottom + (math.log10(value) - math.log10(y_min)) / (
            math.log10(y_max_plot) - math.log10(y_min)
        ) * plot_h

    pdf = canvas.Canvas(str(output_path), pagesize=landscape(letter))
    pdf.setTitle("ALP-photon event-weighted dN_events/dE_ALP")
    pdf.setFillColor(colors.white)
    pdf.rect(0, 0, width, height, stroke=0, fill=1)

    if focused_eventcalc_alpetite:
        pdf.setFillColor(colors.black)
        draw_focused_plot_label(
            pdf,
            margin_left + plot_w / 2.0,
            margin_bottom + plot_h + 14,
            theta_preselection_max or 0.0,
        )
    else:
        title = "ALP-photon event-weighted energy spectra at SHiP"
        pdf.setFillColor(colors.black)
        pdf.setFont(FONT_BOLD, 23)
        pdf.drawString(margin_left, height - 36, title)
        pdf.setFont(FONT_REGULAR, 18)
        subtitle = f"m_a = {MASS_GEV:.3f} GeV, c*tau = {C_TAU_M:.1f} m, N_POT = {N_POT:.2e}"
        pdf.drawString(margin_left, height - 58, subtitle)
        if theta_preselection_max is not None:
            pdf.drawString(margin_left, height - 79, f"EVENTCALC theta_a < {theta_preselection_max:.3f} rad")

    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(2.4)
    pdf.rect(margin_left, margin_bottom, plot_w, plot_h, stroke=1, fill=0)

    tick_size = 20
    tick_length = 10
    pdf.setFont(FONT_REGULAR, tick_size)
    x_ticks = decade_ticks(x_min, x_max_plot) if focused_eventcalc_alpetite else log_ticks(x_min, x_max_plot)
    y_ticks = decade_ticks(y_min, y_max_plot) if focused_eventcalc_alpetite else log_ticks(y_min, y_max_plot)
    for tick in x_ticks:
        x = map_x(tick)
        pdf.setStrokeColor(colors.black)
        pdf.setLineWidth(1.8)
        pdf.line(x, margin_bottom, x, margin_bottom + tick_length)
        pdf.setFillColor(colors.black)
        if focused_eventcalc_alpetite:
            exponent = int(round(math.log10(tick)))
            if exponent % 2 == 0:
                draw_decade_label(pdf, x, margin_bottom - 30, exponent, tick_size, center=True)
        else:
            pdf.drawCentredString(x, margin_bottom - 27, format_sci(tick))

    for tick in y_ticks:
        y = map_y(tick)
        pdf.setStrokeColor(colors.black)
        pdf.setLineWidth(1.8)
        pdf.line(margin_left, y, margin_left + tick_length, y)
        pdf.setFillColor(colors.black)
        if focused_eventcalc_alpetite:
            exponent = int(round(math.log10(tick)))
            if exponent % 2 == 0:
                draw_decade_label(pdf, margin_left - 16, y - 7, exponent, tick_size, right=True)
        else:
            pdf.drawRightString(margin_left - 14, y - 7, format_sci(tick))

    pdf.setFillColor(colors.black)
    draw_x_axis_label(pdf, margin_left + plot_w / 2.0, 42, 24)
    draw_y_axis_label(pdf, 38, margin_bottom + plot_h / 2.0, 24)

    palette = {
        "primary": colors.Color(0.05, 0.25, 0.72),
        "cascades": colors.Color(0.82, 0.22, 0.10),
    }

    for result in results:
        mask = result["dnde"] >= y_min
        xs = result["bin_centers"][mask]
        ys = result["dnde"][mask]
        if len(xs) == 0:
            continue
        path = pdf.beginPath()
        path.moveTo(map_x(float(xs[0])), map_y(float(ys[0])))
        for x_value, y_value in zip(xs[1:], ys[1:]):
            path.lineTo(map_x(float(x_value)), map_y(float(y_value)))
        pdf.setStrokeColor(palette[result["mode"]])
        pdf.setLineWidth(4.0 if focused_eventcalc_alpetite else 2.9)
        pdf.setDash()
        pdf.drawPath(path, stroke=1, fill=0)

    for curve in comparison_curves:
        mask = curve["dnde"] >= y_min
        xs = curve["energy"][mask]
        ys = curve["dnde"][mask]
        if len(xs) == 0:
            continue
        path = pdf.beginPath()
        path.moveTo(map_x(float(xs[0])), map_y(float(ys[0])))
        for x_value, y_value in zip(xs[1:], ys[1:]):
            path.lineTo(map_x(float(x_value)), map_y(float(y_value)))
        pdf.setStrokeColor(palette[curve["mode"]])
        line_width = 4.1 if focused_eventcalc_alpetite and curve["source"] == "ALPETITE" else 2.7
        pdf.setLineWidth(line_width)
        set_curve_dash(pdf, curve["style"])
        pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()
        if focused_eventcalc_alpetite and curve["source"] == "ALPETITE":
            pdf.setFillColor(palette[curve["mode"]])
            stride = max(1, len(xs) // 45)
            for x_value, y_value in zip(xs[::stride], ys[::stride]):
                pdf.circle(map_x(float(x_value)), map_y(float(y_value)), 2.7, stroke=0, fill=1)

    if focused_eventcalc_alpetite:
        legend_y = margin_bottom + 82
        legend_font_size = 18
        legend_step = 34
        legend_columns = (margin_left + 22, margin_left + 320)
    else:
        legend_y = height - 101
        legend_font_size = 20
        legend_step = 30
        legend_columns = (margin_left + 5, margin_left + 330)
    pdf.setFont(FONT_BOLD, legend_font_size)
    legend_entries = []
    for result in results:
        legend_entries.append({
            "mode": result["mode"],
            "style": "solid",
            "label": eventcalc_label(result),
            "source": "EventCalc",
        })
    for curve in comparison_curves:
        legend_entries.append({
            "mode": curve["mode"],
            "style": curve["style"],
            "label": f"{source_label(curve['source'])} {display_mode(curve['mode'])}",
            "source": curve["source"],
        })

    for index, entry in enumerate(legend_entries):
        row = index // 2
        column = index % 2
        x = legend_columns[column]
        y = legend_y - legend_step * row
        pdf.setStrokeColor(palette[entry["mode"]])
        line_width = 4.2 if entry["source"] == "EventCalc" and focused_eventcalc_alpetite else 3.2
        if entry["source"] == "ALPETITE" and focused_eventcalc_alpetite:
            line_width = 4.1
        pdf.setLineWidth(line_width)
        set_curve_dash(pdf, entry["style"])
        pdf.line(x, y, x + 42, y)
        pdf.setDash()
        if focused_eventcalc_alpetite and entry["source"] == "ALPETITE":
            pdf.setFillColor(palette[entry["mode"]])
            pdf.circle(x + 21, y, 3.2, stroke=0, fill=1)
        pdf.setFillColor(colors.black)
        pdf.setFont(FONT_BOLD, legend_font_size)
        pdf.drawString(x + 54, y - 7, entry["label"])

    pdf.showPage()
    pdf.save()


def write_outputs(
    results,
    comparison_curves,
    bins,
    intrinsic_ctau: float,
    coupling_squared: float,
    visible_br: float,
    seed: int,
    output_suffix: str = "",
    theta_preselection_max: float | None = None,
):
    stem = output_stem(output_suffix)
    summary_path = SCRIPT_DIR / f"{stem}_summary.txt"
    csv_path = SCRIPT_DIR / f"{stem}_dnde.csv"
    pdf_path = SCRIPT_DIR / f"{stem}_dnde.pdf"

    with summary_path.open("w") as handle:
        handle.write("# ALP-photon event-level non-interactive run\n")
        handle.write(f"mass_GeV {MASS_GEV:.9e}\n")
        handle.write(f"c_tau_m {C_TAU_M:.9e}\n")
        handle.write(f"N_pot {N_POT:.9e}\n")
        handle.write(f"seed {seed}\n")
        if theta_preselection_max is not None:
            handle.write(f"eventcalc_theta_preselection_max_rad {theta_preselection_max:.9e}\n")
        handle.write(f"intrinsic_ctau_m {intrinsic_ctau:.9e}\n")
        handle.write(f"coupling_squared {coupling_squared:.9e}\n")
        handle.write(f"visible_branching_ratio {visible_br:.9e}\n")
        handle.write(
            "mode yield_per_pot produced_llps epsilon_polar epsilon_azimuthal "
            "epsilon_theta_preselected P_decay_averaged sampled_events accepted_sample_events "
            "theta_preselected_sample_events total_events\n"
        )
        for result in results:
            handle.write(
                f"{result['mode']} {result['yield_per_pot']:.9e} "
                f"{result['produced_llps']:.9e} {result['epsilon_polar']:.9e} "
                f"{result['epsilon_azimuthal']:.9e} {result['epsilon_theta_preselected']:.9e} "
                f"{result['p_decay_averaged']:.9e} {result['sampled_events']} "
                f"{result['accepted_sample_events']} {result['theta_preselected_sample_events']} "
                f"{result['total_events']:.9e}\n"
            )
        handle.write("\n# Per accepted event weight used in each histogram bin:\n")
        handle.write(
            "# w_i = N_pot * Yield(mode,m) * coupling_squared * Br_visible "
            "* epsilon_polar(mode) / N_resampled * P_decay_i\n"
        )
        handle.write("# The azimuthal acceptance is represented by accepting/rejecting simulated phi,z points.\n")
        if theta_preselection_max is not None:
            handle.write("# EventCalc events with theta_ALP above the preselection have zero histogram weight.\n")
        handle.write("\n# Comparison curves overlaid on the PDF:\n")
        for curve in comparison_curves:
            handle.write(
                f"# {curve['source']} {curve['mode']} {curve['path'].name} "
                f"scale {curve['scale']:.9e} trapz_total {curve['total_trapezoid']:.9e}\n"
            )

    columns = [bins[:-1], bins[1:]]
    header = ["E_low_GeV", "E_high_GeV"]
    for result in results:
        columns.append(result["dnde"])
        header.append(f"{result['mode']}_dN_events_dE_ALP_per_GeV")
    np.savetxt(csv_path, np.column_stack(columns), header=",".join(header), delimiter=",", comments="")

    draw_pdf(results, comparison_curves, pdf_path, theta_preselection_max)
    focused_pdf_path = None
    if theta_preselection_max is not None:
        focused_pdf_path = SCRIPT_DIR / f"{stem}_eventcalc_alpetite_dnde.pdf"
        alpetite_curves = [curve for curve in comparison_curves if curve["source"] == "ALPETITE"]
        draw_pdf(
            results,
            alpetite_curves,
            focused_pdf_path,
            theta_preselection_max,
            focused_eventcalc_alpetite=True,
        )
    return summary_path, csv_path, pdf_path, focused_pdf_path


def default_output_suffix(theta_preselection_max):
    if theta_preselection_max is None:
        return ""
    theta_mrad = int(round(theta_preselection_max * 1000.0))
    return f"_theta_lt_{theta_mrad}mrad"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resample-size", type=int, default=DEFAULT_RESAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eventcalc-theta-max", type=float, default=None)
    parser.add_argument("--output-suffix", default=None)
    return parser.parse_args(argv)


def run(resample_size=DEFAULT_RESAMPLE_SIZE, seed=DEFAULT_SEED, eventcalc_theta_max=None, output_suffix=None):
    intrinsic_ctau = interpolate_table(ALP_DIR / "ctau-ALP-photon.txt", MASS_GEV)
    coupling_squared = intrinsic_ctau / C_TAU_M
    visible_br = visible_branching_ratio(MASS_GEV)

    results = []
    for mode_index, mode in enumerate(MODES):
        rng = np.random.default_rng(seed + mode_index)
        results.append(
            simulate_mode(
                mode,
                resample_size,
                rng,
                visible_br,
                coupling_squared,
                theta_preselection_max=eventcalc_theta_max,
            )
        )

    bins = make_histograms(results)
    comparison_curves = load_comparison_curves()
    output_suffix = default_output_suffix(eventcalc_theta_max) if output_suffix is None else output_suffix
    summary_path, csv_path, pdf_path, focused_pdf_path = write_outputs(
        results,
        comparison_curves,
        bins,
        intrinsic_ctau,
        coupling_squared,
        visible_br,
        seed,
        output_suffix=output_suffix,
        theta_preselection_max=eventcalc_theta_max,
    )

    print(f"mass_GeV = {MASS_GEV:.9e}")
    print(f"c_tau_m = {C_TAU_M:.9e}")
    print(f"intrinsic_ctau_m = {intrinsic_ctau:.9e}")
    print(f"coupling_squared = {coupling_squared:.9e}")
    print(f"visible_branching_ratio = {visible_br:.9e}")
    print(f"resample_size = {resample_size}")
    print(f"seed = {seed}")
    if eventcalc_theta_max is not None:
        print(f"eventcalc_theta_max = {eventcalc_theta_max:.9e}")
    for result in results:
        print(f"{result['mode']}_accepted_sample_events = {result['accepted_sample_events']}")
        print(f"{result['mode']}_theta_preselected_sample_events = {result['theta_preselected_sample_events']}")
        print(f"{result['mode']}_epsilon_polar = {result['epsilon_polar']:.9e}")
        print(f"{result['mode']}_epsilon_azimuthal = {result['epsilon_azimuthal']:.9e}")
        print(f"{result['mode']}_epsilon_theta_preselected = {result['epsilon_theta_preselected']:.9e}")
        print(f"{result['mode']}_P_decay_averaged = {result['p_decay_averaged']:.9e}")
        print(f"{result['mode']}_events = {result['total_events']:.9e}")
    for curve in comparison_curves:
        print(f"{curve['source']}_{curve['mode']}_comparison_trapz_events = {curve['total_trapezoid']:.9e}")
    print(f"summary = {summary_path}")
    print(f"csv = {csv_path}")
    print(f"pdf = {pdf_path}")
    if focused_pdf_path is not None:
        print(f"focused_pdf = {focused_pdf_path}")
    return summary_path, csv_path, pdf_path, focused_pdf_path


def main(argv=None):
    args = parse_args(argv)
    run(
        resample_size=args.resample_size,
        seed=args.seed,
        eventcalc_theta_max=args.eventcalc_theta_max,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
