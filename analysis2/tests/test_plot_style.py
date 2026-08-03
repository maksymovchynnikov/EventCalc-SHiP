from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis2.plot_style import REPORT_STYLE, style_axis, use_report_style


class PlotStyleTests(unittest.TestCase):
    def test_report_style_removes_title_and_enlarges_text(self):
        use_report_style()
        figure, axis = plt.subplots()
        axis.set(title="remove me", xlabel="Mass", ylabel="Coupling")
        annotation = axis.text(0.5, 0.5, "experiment", fontsize=8)
        axis.plot([0, 1], [0, 1], label="signal")
        legend = axis.legend(title="Curves")

        style_axis(axis)

        self.assertEqual(axis.get_title(), "")
        self.assertEqual(axis.xaxis.label.get_fontsize(), REPORT_STYLE["axes.labelsize"])
        self.assertGreaterEqual(annotation.get_fontsize(), REPORT_STYLE["font.size"])
        self.assertTrue(all(
            text.get_fontsize() == REPORT_STYLE["legend.fontsize"]
            for text in legend.get_texts()
        ))
        self.assertEqual(legend.get_title().get_fontsize(), REPORT_STYLE["legend.title_fontsize"])
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
