"""Shared report-ready Matplotlib typography."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import matplotlib as mpl
from matplotlib.axes import Axes

REPORT_STYLE = {
    "font.size": 14.0,
    "axes.labelsize": 14.0,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
    "legend.fontsize": 14.0,
    "legend.title_fontsize": 14.0,
}


@dataclass(frozen=True)
class FrozenPlotConfiguration:
    """Centralized presentation values copied from the frozen plot scripts."""

    lifetime_scan_figsize: tuple[float, float] = (11.0, 7.2)
    distance_map_figsize: tuple[float, float] = (8.2, 6.4)
    spectrum_figsize: tuple[float, float] = (8.2, 5.8)
    profiled_figsize: tuple[float, float] = (8.2, 5.8)
    event_density_figsize: tuple[float, float] = (8.5, 6.5)
    event_rate_figsize: tuple[float, float] = (8.5, 6.0)
    png_dpi: int = 200
    grid_alpha: float = 0.25
    distance_cmap: str = "viridis"
    distance_vmin: float = 0.0
    distance_vmax: float = 1.0
    distance_minimum_marker_size: float = 180.0
    spectrum_line_width: float = 2.0
    event_density_levels: tuple[float, ...] = (2.3, 3.0, 10.0, 30.0, 100.0)
    event_density_line_styles: tuple[object, ...] = (
        ":", "--", "-.", "-", (0, (3, 1, 1, 1)),
    )
    event_density_line_width: float = 2.0
    event_density_grid_alpha: float = 0.25
    event_density_table_line_width: float = 1.5
    event_density_legend_frame_width: float = 0.8
    photon_table_mass_limit_gev: float = 4.0
    su2_table_mass_limit_gev: float = 5.1


PLOT_CONFIG = FrozenPlotConfiguration()


@dataclass(frozen=True)
class EventDensityOverlayLayout:
    """Frozen axes and legend placement for one constraint overlay."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    table_mass_limit_gev: float
    legend_anchor: tuple[float, float]
    corner_polygon_axes: tuple[tuple[float, float], ...] = ()


EVENT_DENSITY_OVERLAY_LAYOUT: Mapping[str, EventDensityOverlayLayout] = (
    MappingProxyType(
        {
            "ALP-photon-combined": EventDensityOverlayLayout(
                xlim=(1.5e-2, 5.0),
                ylim=(1.0e-8, 1.2e-2),
                table_mass_limit_gev=4.0,
                legend_anchor=(0.95, 0.7),
            ),
            "ALP-SU2L": EventDensityOverlayLayout(
                xlim=(5.0e-2, 7.0),
                ylim=(8.0e-7, 6.0e-1),
                table_mass_limit_gev=5.1,
                legend_anchor=(0.9, 0.83),
                corner_polygon_axes=((0.0, 1.0), (0.0, 0.935), (0.135, 1.0)),
            ),
        }
    )
)


def use_report_style() -> None:
    """Apply the common typography before constructing a figure."""
    mpl.rcParams.update(REPORT_STYLE)


def style_axis(axis: Axes) -> None:
    """Remove titles and enforce report typography on an existing axis."""
    axis.set_title("")
    axis.xaxis.label.set_fontsize(REPORT_STYLE["axes.labelsize"])
    axis.yaxis.label.set_fontsize(REPORT_STYLE["axes.labelsize"])
    axis.tick_params(which="both", labelsize=REPORT_STYLE["xtick.labelsize"])
    axis.xaxis.get_offset_text().set_fontsize(REPORT_STYLE["xtick.labelsize"])
    axis.yaxis.get_offset_text().set_fontsize(REPORT_STYLE["ytick.labelsize"])
    for annotation in axis.texts:
        annotation.set_fontsize(max(annotation.get_fontsize(), REPORT_STYLE["font.size"]))
    legend = axis.get_legend()
    if legend is not None:
        for label in legend.get_texts():
            label.set_fontsize(REPORT_STYLE["legend.fontsize"])
        legend.get_title().set_fontsize(REPORT_STYLE["legend.title_fontsize"])
