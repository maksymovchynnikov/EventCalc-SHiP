"""Convert FORESEE ALP-W couplings to EventCalc c_W/f_a convention."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from table_builders.ALP_SU2L.constants import ALPHA_SU2
from analysis2.cache import atomic_output_path
from analysis2.config import PROFILES
from analysis2.paths import profile_output_dir

COUPLING_CONVERSION_FACTOR = np.pi / ALPHA_SU2


def convert_constraint(input_path: Path, output_path: Path) -> Path:
    data = np.loadtxt(input_path, comments="#", ndmin=2)
    if data.shape[1] < 2 or not np.all(np.isfinite(data[:, :2])):
        raise ValueError(f"invalid constraint polygon {input_path}")
    data[:, 1] *= COUPLING_CONVERSION_FACTOR
    header = (
        "ALP-W constraint converted from FORESEE\ncolumn 1: m_a [GeV]\n"
        "column 2: c_W/f_a [GeV^-1] (EventCalc)\n"
        f"conversion factor pi/alpha_SU2 = {COUPLING_CONVERSION_FACTOR:.12e}"
    )
    with atomic_output_path(output_path) as temporary:
        np.savetxt(temporary, data, fmt="%.12e", header=header)
    return output_path


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    root = profile_output_dir(args.profile, "constraints")
    input_dir = args.input_dir or root / "raw/alp_su2l"
    output_dir = root / "converted/alp_su2l"
    paths = sorted(input_dir.glob("bounds_*.txt"))
    if not paths:
        raise FileNotFoundError(f"no constraint files in {input_dir}")
    for path in paths:
        convert_constraint(path, output_dir / path.name)
    print(f"Converted {len(paths)} files to {output_dir}")


if __name__ == "__main__":
    main()
