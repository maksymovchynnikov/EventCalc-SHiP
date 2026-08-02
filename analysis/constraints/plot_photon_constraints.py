from pathlib import Path

import matplotlib.pyplot as plt

from analysis.plot_style import (
    style_axis,
    use_report_style,
)

from .plotting_helpers import (
    PHOTON_SPECS,
    draw_constraints,
    load_label_config,
)


CONSTRAINTS_DIR = Path(__file__).resolve().parent
RAW_DIR = CONSTRAINTS_DIR / "raw" / "alp_photon"
PLOT_DIR = CONSTRAINTS_DIR / "plots"

X_LIMITS = (1.0e-2, 1.0)
Y_LIMITS = (3.0e-7, 2.0e-2)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    use_report_style()

    figure, axis = plt.subplots(
        figsize=(8.0, 6.2),
    )

    draw_constraints(
        axis,
        RAW_DIR,
        PHOTON_SPECS,
        model="alp_photon",
        context="constraint_only",
        config=load_label_config(),
    )

    axis.set(
        xscale="log",
        yscale="log",
        xlim=X_LIMITS,
        ylim=Y_LIMITS,
        xlabel=r"$m_a$ [GeV]",
        ylabel=r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]",
    )

    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    axis.grid(False)

    style_axis(axis)
    figure.tight_layout()

    output_path = PLOT_DIR / "photon_constraints.pdf"

    figure.savefig(output_path)
    plt.close(figure)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()