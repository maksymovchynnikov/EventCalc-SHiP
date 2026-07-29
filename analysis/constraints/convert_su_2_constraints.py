from pathlib import Path

import numpy as np

from table_builders.ALP_SU2L.constants import ALPHA_SU2


COUPLING_CONVERSION_FACTOR = np.pi / ALPHA_SU2


CONSTRAINTS_DIR = Path(__file__).resolve().parent
RAW_DIR = CONSTRAINTS_DIR / "raw" / "alp_su2l"
CONVERTED_DIR = CONSTRAINTS_DIR / "converted" / "alp_su2l"


def convert_constraint_file(input_path: Path) -> Path:
    """Convert one FORESEE ALP-W constraint to the EventCalc convention."""
    data = np.loadtxt(
        input_path,
        comments="#",
        ndmin=2,
    )

    if data.shape[1] < 2:
        raise ValueError(f"{input_path.name} contains fewer than two columns.")

    if not np.all(np.isfinite(data[:, :2])):
        raise ValueError(f"{input_path.name} contains non-finite masses or couplings.")

    converted_data = data.copy()
    # Column 0: ALP mass [GeV], unchanged.
    # Column 1: coupling [GeV^-1], converted.
    converted_data[:, 1] *= COUPLING_CONVERSION_FACTOR
    output_path = CONVERTED_DIR / input_path.name

    header = (
        "ALP-W constraint converted from the FORESEE coupling convention\n"
        "column 1: m_a [GeV]\n"
        "column 2: c_W/f_a [GeV^-1] in the EventCalc convention\n"
        f"conversion factor: pi/alpha_SU2 = "
        f"{COUPLING_CONVERSION_FACTOR:.12e}"
    )

    np.savetxt(output_path, converted_data, fmt="%.12e", header=header)

    return output_path


def main() -> None:
    """Convert all bounds_*.txt files in the raw directory."""
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(RAW_DIR.glob("bounds_*.txt"))

    if not input_paths:
        raise FileNotFoundError(f"No bounds_*.txt files found in {RAW_DIR}")

    print(f"alpha_SU2 = {ALPHA_SU2:.12e}")
    print(f"Coupling conversion factor pi/alpha_SU2 = {COUPLING_CONVERSION_FACTOR:.12e}")
    print()

    for input_path in input_paths:
        output_path = convert_constraint_file(input_path)
        print(f"Converted {input_path.name} -> {output_path.relative_to(CONSTRAINTS_DIR)}")

    print(f"\nConverted {len(input_paths)} constraint files.")


if __name__ == "__main__":
    main()
