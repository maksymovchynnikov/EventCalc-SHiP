"""Generic exact-array and documented CSV-roundtrip comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


CSV_RTOL = 1.0e-14
CSV_ATOL = 5.0e-15
REPORT_COLUMNS = (
    "category",
    "artifact",
    "quantity",
    "comparison_mode",
    "reference_count",
    "current_count",
    "max_abs_difference",
    "max_relative_difference",
    "absolute_tolerance",
    "relative_tolerance",
    "status",
    "details",
)


def comparison_row(
    category: str,
    artifact: str,
    quantity: str,
    mode: str,
    *,
    reference_count: int = 0,
    current_count: int = 0,
    maximum_absolute: float = np.nan,
    maximum_relative: float = np.nan,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    status: str,
    details: str = "",
) -> dict:
    return dict(
        zip(
            REPORT_COLUMNS,
            (
                category,
                artifact,
                quantity,
                mode,
                reference_count,
                current_count,
                maximum_absolute,
                maximum_relative,
                absolute_tolerance,
                relative_tolerance,
                status,
                details,
            ),
        )
    )


def mismatch_record(category: str, artifact: str, details: str) -> dict:
    return comparison_row(
        category,
        artifact,
        "structure",
        "exact_structure",
        status="genuine_mismatch",
        details=details,
    )


def numeric_record(
    category: str,
    artifact: str,
    quantity: str,
    reference,
    current,
    *,
    mode: str,
    allow_csv_tolerance: bool,
    require_same_dtype: bool = False,
) -> dict:
    left, right = np.asarray(reference), np.asarray(current)
    if left.shape != right.shape:
        return mismatch_record(category, artifact, f"{quantity}: shapes differ")
    if require_same_dtype and left.dtype != right.dtype:
        return mismatch_record(category, artifact, f"{quantity}: dtypes differ")
    left_float, right_float = left.astype(float), right.astype(float)
    finite = np.isfinite(left_float) & np.isfinite(right_float)
    same_finite_state = np.array_equal(
        np.isfinite(left_float), np.isfinite(right_float)
    ) and np.array_equal(np.isnan(left_float), np.isnan(right_float))
    difference = np.abs(left_float[finite] - right_float[finite])
    relative = difference / np.maximum(
        np.abs(left_float[finite]), np.finfo(float).tiny
    )
    maximum_absolute = float(difference.max(initial=0.0))
    maximum_relative = float(relative.max(initial=0.0))
    exact = same_finite_state and np.array_equal(left, right, equal_nan=True)
    close = same_finite_state and np.allclose(
        left_float,
        right_float,
        rtol=CSV_RTOL,
        atol=CSV_ATOL,
        equal_nan=True,
    )
    if exact:
        status = "exact_agreement"
    elif allow_csv_tolerance and close:
        status = "csv_roundtrip_agreement"
    else:
        status = "genuine_mismatch"
    return comparison_row(
        category,
        artifact,
        quantity,
        mode,
        reference_count=left.size,
        current_count=right.size,
        maximum_absolute=maximum_absolute,
        maximum_relative=maximum_relative,
        absolute_tolerance=CSV_ATOL if allow_csv_tolerance else 0.0,
        relative_tolerance=CSV_RTOL if allow_csv_tolerance else 0.0,
        status=status,
    )


def text_record(category, artifact, quantity, reference, current, mode) -> dict:
    left = np.asarray(reference, dtype=object)
    right = np.asarray(current, dtype=object)
    equal = left.shape == right.shape and all(
        (a == b) or (pd.isna(a) and pd.isna(b))
        for a, b in zip(left.ravel(), right.ravel())
    )
    return comparison_row(
        category,
        artifact,
        quantity,
        mode,
        reference_count=left.size,
        current_count=right.size,
        status="exact_agreement" if equal else "genuine_mismatch",
    )


def compare_npz_files(
    reference_path: Path,
    current_path: Path,
    *,
    category: str,
    artifact: str,
    expected_current_metadata: Mapping[str, object] | None = None,
) -> list[dict]:
    """Compare frozen arrays exactly and validate an optional schema extension.

    With ``expected_current_metadata`` every key in the frozen archive is a
    required scientific key.  The current archive may add exactly the named
    scalar metadata keys; their dtype and value are checked exactly and the
    extension is reported separately from the scientific comparison.
    """
    if not reference_path.is_file() or not current_path.is_file():
        missing = "reference" if not reference_path.is_file() else "current"
        return [mismatch_record(category, artifact, f"missing {missing} NPZ artifact")]
    with np.load(reference_path, allow_pickle=False) as first, np.load(
        current_path, allow_pickle=False
    ) as second:
        reference_keys, current_keys = set(first.files), set(second.files)
        metadata = dict(expected_current_metadata or {})
        metadata_keys = set(metadata)
        scientific_status = (
            "exact_agreement"
            if reference_keys == current_keys - metadata_keys
            else "genuine_mismatch"
        )
        rows = [
            comparison_row(
                category,
                artifact,
                "scientific_key_set",
                "exact_npz_scientific",
                reference_count=len(reference_keys),
                current_count=len(current_keys - metadata_keys),
                status=scientific_status,
                details=(
                    ""
                    if scientific_status == "exact_agreement"
                    else "missing or unexpected scientific NPZ keys"
                ),
            )
        ]
        if metadata:
            current_metadata_keys = current_keys & metadata_keys
            metadata_status = (
                "validated_metadata_extension"
                if current_metadata_keys == metadata_keys
                else "genuine_mismatch"
            )
            rows.append(
                comparison_row(
                    category,
                    artifact,
                    "metadata_key_set",
                    "validated_npz_metadata",
                    reference_count=0,
                    current_count=len(current_metadata_keys),
                    status=metadata_status,
                    details=(
                        ", ".join(sorted(metadata_keys))
                        if metadata_status == "validated_metadata_extension"
                        else "missing or unexpected NPZ metadata keys"
                    ),
                )
            )
        for name in sorted(reference_keys & current_keys):
            rows.append(
                numeric_record(
                    category,
                    artifact,
                    name,
                    first[name],
                    second[name],
                    mode="exact_npz",
                    allow_csv_tolerance=False,
                    require_same_dtype=True,
                )
            )
        for name, expected in sorted(metadata.items()):
            if name not in second:
                rows.append(
                    mismatch_record(
                        category,
                        artifact,
                        f"missing required metadata key {name!r}",
                    )
                )
                continue
            expected_array = np.asarray(expected)
            current_array = second[name]
            exact = (
                expected_array.shape == current_array.shape
                and expected_array.dtype == current_array.dtype
                and np.array_equal(expected_array, current_array)
            )
            rows.append(
                comparison_row(
                    category,
                    artifact,
                    f"metadata:{name}",
                    "validated_npz_metadata",
                    reference_count=0,
                    current_count=current_array.size,
                    status=(
                        "validated_metadata_extension"
                        if exact
                        else "genuine_mismatch"
                    ),
                    details="expected scalar dtype and value" if not exact else "",
                )
            )
    return rows


def compare_csv_files(
    reference_path: Path,
    current_path: Path,
    *,
    category: str,
    artifact: str,
    keys: tuple[str, ...],
) -> list[dict]:
    """Compare keyed CSV tables with tolerance only for floating columns."""
    if not reference_path.is_file() or not current_path.is_file():
        missing = "reference" if not reference_path.is_file() else "current"
        return [mismatch_record(category, artifact, f"missing {missing} CSV artifact")]
    first, second = pd.read_csv(reference_path), pd.read_csv(current_path)
    if set(first.columns) != set(second.columns):
        return [mismatch_record(category, artifact, "CSV column sets differ")]
    if any(key not in first for key in keys):
        return [mismatch_record(category, artifact, "CSV comparison keys are missing")]
    if first.duplicated(list(keys)).any() or second.duplicated(list(keys)).any():
        return [mismatch_record(category, artifact, "CSV contains duplicate keys")]
    if len(first) != len(second):
        return [mismatch_record(category, artifact, "CSV row counts differ")]
    first = first.sort_values(list(keys), ignore_index=True)
    second = second.sort_values(list(keys), ignore_index=True)
    rows = [
        comparison_row(
            category,
            artifact,
            "table_structure",
            "csv_roundtrip",
            reference_count=len(first),
            current_count=len(second),
            status="exact_agreement",
        )
    ]
    for column in first.columns:
        if is_numeric_dtype(first[column]) and is_numeric_dtype(second[column]):
            floating = pd.api.types.is_float_dtype(
                first[column]
            ) or pd.api.types.is_float_dtype(second[column])
            rows.append(
                numeric_record(
                    category,
                    artifact,
                    column,
                    first[column].to_numpy(),
                    second[column].to_numpy(),
                    mode="csv_roundtrip",
                    allow_csv_tolerance=floating,
                )
            )
        else:
            rows.append(
                text_record(
                    category,
                    artifact,
                    column,
                    first[column].to_numpy(),
                    second[column].to_numpy(),
                    "csv_roundtrip",
                )
            )
    return rows
