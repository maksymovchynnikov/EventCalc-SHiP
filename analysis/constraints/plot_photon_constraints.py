from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


CONSTRAINTS_DIR = Path(__file__).resolve().parent

RAW_DIR = CONSTRAINTS_DIR / "raw" / "alp_photon"

PLOT_DIR = CONSTRAINTS_DIR / "plots"


# These limits are only presentation choices for the
# standalone constraint plot.
X_LIMITS = (1.0e-2, 1.0)
Y_LIMITS = (3.0e-7, 2.0e-2)


# Entries copied from the FORESEE ALP-photon notebook:
#
# filename, label, label_x, label_y, rotation
#
# The coupling convention already agrees with EventCalc,
# so neither the files nor the label positions are converted.
BOUND_SPECS = [
    (
        "bounds_BESIII_2024.txt",
        "BESIII",
        0.145,
        6.5e-4,
        0,
    ),
    (
        "bounds_BESIII_2022.txt",
        "BESIII",
        0.170,
        1.7e-4,
        0,
    ),
    (
        "bounds_Belle2.txt",
        "Belle II",
        0.215,
        1.45e-3,
        0,
    ),
    (
        "bounds_PrimEx.txt",
        "PrimEx",
        0.125,
        1.25e-3,
        90,
    ),
    (
        "bounds_LEP.txt",
        "LEP",
        0.080,
        6.0e-3,
        0,
    ),
    (
        "bounds_SN1987.txt",
        "SN1987",
        0.0135,
        6.0e-7,
        0,
    ),
    (
        "bounds_E137.txt",
        "E137",
        2.05e-2,
        6.0e-6,
        0,
    ),
    (
        "bounds_NuCal.txt",
        "NuCal",
        0.080,
        5.0e-6,
        -15,
    ),
    (
        "bounds_CHARM.txt",
        "CHARM",
        0.043,
        3.2e-5,
        -45,
    ),
    (
        "bounds_NA64.txt",
        "NA64",
        0.035,
        2.6e-4,
        0,
    ),
    (
        "bounds_E141.txt",
        "E141",
        0.0175,
        1.7e-3,
        0,
    ),
]

PHOTON_STANDALONE_LABEL_POSITIONS_AXES = {
    "bounds_E141.txt": (
        0.12,
        0.78,
        0,
        "center",
        "center",
    ),
    "bounds_LEP.txt": (
        0.45,
        0.895,
        0,
        "center",
        "center",
    ),
    "bounds_NA64.txt": (
        0.28,
        0.61,
        0,
        "center",
        "center",
    ),
    "bounds_CHARM.txt": (
        0.32,
        0.43,
        -45,
        "center",
        "center",
    ),
    "bounds_E137.txt": (
        0.15,
        0.28,
        0,
        "center",
        "center",
    ),
    "bounds_NuCal.txt": (
        0.49,
        0.29,
        -15,
        "center",
        "center",
    ),
    "bounds_SN1987.txt": (
        0.025,
        0.065,
        0,
        "left",
        "center",
    ),
    "bounds_PrimEx.txt": (
        0.565,
        0.805,
        90,
        "center",
        "center",
    ),
    "bounds_BESIII_2024.txt": (
        0.68,
        0.695,
        0,
        "center",
        "center",
    ),
    "bounds_Belle2.txt": (
        0.71,
        0.780,
        0,
        "center",
        "center",
    ),
    "bounds_BESIII_2022.txt": (
        0.6575,
        0.635,
        0,
        "center",
        "center",
    ),
}


def load_constraint(
    path: Path,
) -> np.ndarray:
    """Load and validate one FORESEE constraint polygon."""
    data = np.loadtxt(
        path,
        comments="#",
        ndmin=2,
    )

    if data.shape[1] < 2:
        raise ValueError(f"{path.name} contains fewer than two columns.")

    if not np.all(np.isfinite(data[:, :2])):
        raise ValueError(f"{path.name} contains non-finite values.")

    masses = data[:, 0]
    couplings = data[:, 1]

    if np.any(masses <= 0.0):
        raise ValueError(f"{path.name} contains non-positive masses.")

    if np.any(couplings <= 0.0):
        raise ValueError(f"{path.name} contains non-positive couplings.")

    return data


def draw_photon_constraints(
    axis: plt.Axes,
    *,
    draw_labels: bool = True,
    label_positions_axes: dict[str, tuple[float, float, float, str, str]] | None = None,
    label_fontsize: float = 10.0,
) -> None:
    """Draw ALP-photon exclusions on an existing axis."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            "Could not find the ALP-photon raw "
            f"constraint directory:\n  {RAW_DIR}\n"
            "Run download_foresee_constraints.py first."
        )

    for index, (
        filename,
        label,
        label_x,
        label_y,
        rotation,
    ) in enumerate(BOUND_SPECS):
        constraint_path = RAW_DIR / filename

        if not constraint_path.exists():
            raise FileNotFoundError(f"Missing ALP-photon constraint file:\n  {constraint_path}")

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
            if label_positions_axes is not None and filename in label_positions_axes:
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
                plot_y = label_y
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
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 6.2),
    )

    draw_photon_constraints(
        axis,
        draw_labels=True,
        label_positions_axes=(PHOTON_STANDALONE_LABEL_POSITIONS_AXES),
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlabel(r"$m_a$ [GeV]")
    axis.set_ylabel(r"$g_{a\gamma\gamma}$ " r"[GeV$^{-1}$]")
    axis.set_title("Existing constraints on ALP-photon")
    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    axis.grid(False)
    figure.tight_layout()

    output_path = PLOT_DIR / "photon_constraints_foresee_style.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
