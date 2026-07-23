from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .convert_su_2_constraints import (
    COUPLING_CONVERSION_FACTOR,
)


CONSTRAINTS_DIR = Path(__file__).resolve().parent
PLOT_DIR = CONSTRAINTS_DIR / "plots"
CONVERTED_DIR = CONSTRAINTS_DIR / "converted" / "alp_su2l"


X_LIMITS = (5.0e-2, 1.0)
FORESEE_Y_LIMITS = (7.0e-7, 1.0e-3)
Y_LIMITS = tuple(
    value * COUPLING_CONVERSION_FACTOR
    for value in FORESEE_Y_LIMITS
)


# Entries copied from the FORESEE ALP-W notebook:
#
# filename, label, label_x, label_y, rotation
#
# The label_y values are originally given in the FORESEE
# coupling convention and must therefore also be converted.
BOUND_SPECS = [
    ("bounds_BaBar.txt", "BaBar", 0.30, 9.0e-5, 0),
    ("bounds_SN1987.txt", "SN1987", 0.065, 9.0e-7, 0),
    ("bounds_E137.txt", "E137", 0.100, 1.2e-6, -8),
    ("bounds_LEP.txt", "LEP", 0.650, 6.7e-4, 0),
    (
        "bounds_E949_displ.txt",
        "E949",
        0.065,
        9.0e-5,
        -9,
    ),
    (
        "bounds_NA62_1.txt",
        "NA62",
        0.245,
        4.5e-4,
        90,
    ),
    (
        "bounds_NA62_2.txt",
        "NA62",
        0.065,
        9.2e-6,
        2,
    ),
    (
        "bounds_KOTO.txt",
        "KOTO",
        0.090,
        3.4e-5,
        2,
    ),
    (
        "bounds_KTEV.txt",
        "KTEV",
        0.200,
        4.5e-4,
        90,
    ),
    (
        "bounds_NA6264.txt",
        "+ NA48/2",
        0.270,
        4.5e-4, # Changed
        90,
    ),
    (
        "bounds_E949_prompt.txt",
        "E949",
        0.065,
        3.0e-6,
        -5,
    ),
    (
        "bounds_CDF.txt",
        "CDF",
        0.065,
        7.5e-4, # Changed
        -12,
    ),
]

SU2_STANDALONE_LABEL_POSITIONS_AXES = {
    "bounds_CDF.txt": (
        0.065,
        0.965,
        -12,
        "left",
        "center",
    ),
    "bounds_E949_displ.txt": (
        0.085,
        0.675,
        -9,
        "left",
        "center",
    ),
    "bounds_KOTO.txt": (
        0.195,
        0.535,
        0,
        "center",
        "center",
    ),
    "bounds_NA62_2.txt": (
        0.085,
        0.355,
        0,
        "left",
        "center",
    ),
    "bounds_E949_prompt.txt": (
        0.085,
        0.200,
        -5,
        "left",
        "center",
    ),
    "bounds_E137.txt": (
        0.235,
        0.075,
        -8,
        "center",
        "center",
    ),
    "bounds_SN1987.txt": (
        0.050,
        0.035,
        0,
        "left",
        "center",
    ),
    "bounds_KTEV.txt": (
        0.465,
        0.890,
        90,
        "center",
        "center",
    ),
    "bounds_NA62_1.txt": (
        0.530,
        0.890,
        90,
        "center",
        "center",
    ),
    "bounds_NA6264.txt": (
        0.565,
        0.890,
        90,
        "center",
        "center",
    ),
    "bounds_BaBar.txt": (
        0.600,
        0.670,
        0,
        "center",
        "center",
    ),
    "bounds_LEP.txt": (
        0.855,
        0.945,
        0,
        "center",
        "center",
    ),
}


def load_constraint(path: Path) -> np.ndarray:
    """Load and validate one converted constraint polygon."""
    data = np.loadtxt(
        path,
        comments="#",
        ndmin=2,
    )

    if data.shape[1] < 2:
        raise ValueError(
            f"{path.name} contains fewer than two columns."
        )

    if not np.all(np.isfinite(data[:, :2])):
        raise ValueError(
            f"{path.name} contains non-finite values."
        )

    return data

def draw_su2_constraints(
    axis: plt.Axes,
    *,
    draw_labels: bool = True,
    label_positions_axes: dict[
        str,
        tuple[
            float,
            float,
            float,
            str,
            str,
        ],
    ] | None = None,
    label_fontsize: float = 10.0,
) -> None:
    for index, (
        filename,
        label,
        label_x,
        label_y_foresee,
        rotation,
    ) in enumerate(BOUND_SPECS):
        constraint_path = CONVERTED_DIR / filename

        if not constraint_path.exists():
            raise FileNotFoundError(
                f"Missing constraint file: {constraint_path}"
            )

        data = load_constraint(constraint_path)

        masses = data[:, 0]
        couplings = data[:, 1]

        z_order = -100 + index

        axis.fill(
            masses,
            couplings,
            color="gainsboro",
            zorder=z_order,
        )

        axis.plot(
            masses,
            couplings,
            color="dimgray",
            linewidth=1.0,
            zorder=z_order,
        )

        if draw_labels:
            if (
                label_positions_axes is not None
                and filename in label_positions_axes
            ):
                (
                    plot_x,
                    plot_y,
                    plot_rotation,
                    horizontal_alignment,
                    vertical_alignment,
                ) = label_positions_axes[filename]

                coordinate_transform = axis.transAxes

            else:
                plot_x = label_x
                plot_y = (
                    label_y_foresee
                    * COUPLING_CONVERSION_FACTOR
                )
                plot_rotation = rotation
                horizontal_alignment = "center"
                vertical_alignment = "center"

                coordinate_transform = axis.transData

            axis.text(
                plot_x,
                plot_y,
                label,
                transform=coordinate_transform,
                fontsize=label_fontsize,
                color="dimgray",
                rotation=plot_rotation,
                ha=horizontal_alignment,
                va=vertical_alignment,
                clip_on=True,
                zorder=100,
            )


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(
        figsize=(8.0, 6.2),
    )

    draw_su2_constraints(
        axis,
        draw_labels=True,
        label_positions_axes=(
            SU2_STANDALONE_LABEL_POSITIONS_AXES
        ),
    )

    axis.set_xscale("log")
    axis.set_yscale("log")

    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)

    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel(
        r"$c_W/f_a$ [GeV$^{-1}$]"
    )

    axis.set_title(
        r"Existing constraints on ALP-$SU(2)_L$"
    )

    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    # FORESEE-style exclusion plots are cleaner without
    # a large grid or legend.
    axis.grid(False)

    figure.tight_layout()

    output_path = (
        PLOT_DIR
        / "su2_constraints_foresee_style.pdf"
    )

    figure.savefig(output_path)
    plt.close(figure)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()