"""Runtime repository paths and portable path serialization helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent

# Existing expensive runtime data remain in their historical on-disk namespace.
# This preserves cache/checkpoint/result reuse across the source-package rename.
DATA_ROOT = REPOSITORY_ROOT / "analysis2"
CACHE_ROOT = DATA_ROOT / "cache"
OUTPUT_ROOT = DATA_ROOT / "outputs"


EXTERNAL_PATH_PREFIX = "external:"


def repository_relative_path(path: str | Path) -> str:
    """Return a POSIX path relative to the repository, rejecting outsiders."""
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"path is outside the repository: {resolved}") from error
    return relative.as_posix() or "."


def portable_path(path: str | Path) -> str:
    """Serialize repository paths portably without exposing host home paths."""
    resolved = Path(path).expanduser().resolve()
    try:
        return repository_relative_path(resolved)
    except ValueError:
        digest = hashlib.sha256(resolved.as_posix().encode()).hexdigest()
        return f"{EXTERNAL_PATH_PREFIX}{digest}:{resolved.name}"


def profile_cache_dir(profile: str, stage: str | None = None) -> Path:
    path = CACHE_ROOT / profile
    return path if stage is None else path / stage


def profile_output_dir(profile: str, stage: str | None = None) -> Path:
    path = OUTPUT_ROOT / profile
    return path if stage is None else path / stage
