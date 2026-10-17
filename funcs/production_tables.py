"""Lossless storage of rectangular mass/angle/energy production tables."""
from pathlib import Path

import numpy as np
import pandas as pd


def distribution_path(path):
    """Resolve a production table, retaining support for fresh text exports."""
    path = Path(path)
    if path.is_file() or path.suffix not in {".txt", ".npz"}:
        return path
    return path.with_suffix(".npz" if path.suffix == ".txt" else ".txt")


def read_distribution(path):
    """Return the usual four-column DataFrame without changing float bits."""
    path = distribution_path(path)
    if path.suffix == ".txt":
        return pd.read_csv(path, header=None, sep="\t")
    with np.load(path, allow_pickle=False) as data:
        axes = [data[key] for key in ("mass", "theta", "energy")]
        density = data["density"]
    if (any(a.dtype != np.float64 or a.ndim != 1 or not len(a)
            or not np.all(np.isfinite(a)) or not np.all(np.diff(a) > 0)
            for a in axes)
            or density.dtype != np.float64
            or density.shape != tuple(len(a) for a in axes)
            or not np.all(np.isfinite(density))):
        raise ValueError(f"Invalid production grid: {path}")
    # Pandas' original text reader stores contiguous columns, not rows.
    table = np.empty((density.size, 4), dtype=np.float64, order="F")
    for i, axis in enumerate(axes):
        shape = [1, 1, 1]
        shape[i] = len(axis)
        table[:, i].reshape(density.shape)[:] = axis.reshape(shape)
    table[:, 3] = density.reshape(-1)
    return pd.DataFrame(table, copy=False)


def pack_distribution(path, *, replace_text=False):
    """Pack a text export; remove it only after a bit-identical round trip."""
    path = Path(path)
    if path.suffix != ".txt" or not path.name.startswith("DoubleDistr-"):
        raise ValueError("Expected a DoubleDistr-*.txt production export")
    if path.name.startswith("DoubleDistr-ALP-photon"):
        raise ValueError("ALP-photon tables must remain in text format")
    target = path.with_suffix(".npz")
    # Use the same parser as generation, including its float-rounding rules.
    original = pd.read_csv(path, header=None, sep="\t").to_numpy()
    if original.dtype != np.float64 or original.ndim != 2 or original.shape[1] != 4:
        raise ValueError(f"Expected four float64 columns: {path}")
    axes = [np.unique(original[:, i]) for i in range(3)]
    shape = tuple(len(axis) for axis in axes)
    if len(original) != int(np.prod(shape)):
        raise ValueError(f"Not a complete rectangular grid: {path}")
    for i, axis in enumerate(axes):
        axis_shape = [1, 1, 1]
        axis_shape[i] = len(axis)
        expected = np.broadcast_to(axis.reshape(axis_shape), shape)
        if not np.array_equal(original[:, i].reshape(shape).view(np.uint64),
                              expected.view(np.uint64)):
            raise ValueError(f"Unexpected grid order: {path}")
    # Exclusive creation prevents an accidental overwrite of an existing table.
    with target.open("xb") as stream:
        np.savez_compressed(stream, mass=axes[0], theta=axes[1], energy=axes[2],
                            density=original[:, 3].reshape(shape))
    restored = read_distribution(target).to_numpy()
    if not np.array_equal(original.view(np.uint64), restored.view(np.uint64)):
        raise ValueError(f"Production table failed lossless verification: {target}")
    if replace_text:
        path.unlink()
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="+", type=Path)
    parser.add_argument("--replace-text", action="store_true",
                        help="Remove each text file after bit-identical verification")
    args = parser.parse_args()
    for table in args.tables:
        print(pack_distribution(table, replace_text=args.replace_text))
