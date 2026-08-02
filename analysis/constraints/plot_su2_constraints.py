from pathlib import Path

import matplotlib.pyplot as plt

from analysis.plot_style import (
    style_axis,
    use_report_style,
)

from .convert_su_2_constraints import (
    COUPLING_CONVERSION_FACTOR,
)

from .plotting_helpers import (
    SU2_SPECS,
    draw_constraints,
    load_label_config,
)


CONSTRAINTS_DIR = Path(__file__).resolve().parent
CONVERTED_DIR = CONSTRAINTS_DIR / "converted" / "alp_su2l"
PLOT_DIR = CONSTRAINTS_DIR / "plots"

X_LIMITS = (5.0e-2, 1.0)

FORESEE_Y_LIMITS = (7.0e-7, 1.0e-3)
Y_LIMITS = tuple(
    value * COUPLING_CONVERSION_FACTOR
    for value in FORESEE_Y_LIMITS
)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    use_report_style()

    figure, axis = plt.subplots(
        figsize=(8.0, 6.2),
    )

    draw_constraints(
        axis,
        CONVERTED_DIR,
        SU2_SPECS,
        model="alp_su2l",
        context="constraint_only",
        config=load_label_config(),
    )

    axis.set(
        xscale="log",
        yscale="log",
        xlim=X_LIMITS,
        ylim=Y_LIMITS,
        xlabel=r"$m_a$ [GeV]",
        ylabel=r"$c_W/f_a$ [GeV$^{-1}$]",
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

    output_path = PLOT_DIR / "su2_constraints.pdf"

    figure.savefig(output_path)
    plt.close(figure)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()