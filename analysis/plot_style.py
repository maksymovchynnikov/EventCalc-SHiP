"""Shared report-ready Matplotlib typography."""

from __future__ import annotations

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
