"""Deterministic, metadata-validated cache with atomic NPZ/JSON writes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Iterator, Mapping

import numpy as np

from .paths import REPOSITORY_ROOT, portable_path, profile_cache_dir

CACHE_FORMAT_VERSION = 1


def _jsonable(value, *, nonfinite_to_none: bool = False):
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, nonfinite_to_none=nonfinite_to_none)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, nonfinite_to_none=nonfinite_to_none) for item in value]
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), nonfinite_to_none=nonfinite_to_none)
    if isinstance(value, float) and not np.isfinite(value):
        if nonfinite_to_none:
            return None
        raise ValueError("Non-finite floats are not valid cache identity values")
    return value


def canonical_json(value) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def cache_key(identity: Mapping) -> str:
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def file_fingerprint(path: Path, checksum_limit: int = 1_000_000) -> dict:
    path = path.resolve()
    stat = path.stat()
    result = {
        "path": portable_path(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= checksum_limit:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=False,
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and replace the target on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class CacheStore:
    def __init__(self, profile: str, root: Path | None = None, enabled: bool = True):
        self.profile = profile
        self.root = (root or profile_cache_dir(profile)).resolve()
        self.enabled = enabled
        self.git_commit = git_commit()
        self._counters = {"hits": 0, "misses": 0, "writes": 0, "rejected": 0}

    def counter_snapshot(self) -> dict[str, int]:
        """Return an independent snapshot suitable for runtime reports."""
        return dict(self._counters)

    def paths(self, kind: str, identity: Mapping) -> tuple[Path, Path, str]:
        key = cache_key(identity)
        base = self.root / kind / key
        return base.with_suffix(".npz"), base.with_suffix(".json"), key

    def load(
        self, kind: str, identity: Mapping,
        validator: Callable[[dict[str, np.ndarray], dict], None] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict] | None:
        if not self.enabled:
            self._counters["misses"] += 1
            return None
        array_path, metadata_path, key = self.paths(kind, identity)
        if not array_path.exists() or not metadata_path.exists():
            self._counters["misses"] += 1
            return None
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("cache_format_version") != CACHE_FORMAT_VERSION:
                raise ValueError("cache-format version differs")
            if metadata.get("cache_key") != key:
                raise ValueError("cache-key metadata differs")
            if metadata.get("profile") != self.profile:
                raise ValueError("cache profile differs")
            if canonical_json(metadata.get("identity")) != canonical_json(identity):
                raise ValueError("identity metadata differs")
            for name, value in identity.items():
                if name in metadata and canonical_json(metadata[name]) != canonical_json(value):
                    raise ValueError(f"top-level metadata differs for {name}")
            with np.load(array_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            if validator:
                validator(arrays, metadata)
        except (OSError, AttributeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            self._counters["misses"] += 1
            self._counters["rejected"] += 1
            print(f"CACHE REJECTED [{kind}] {key[:12]}: {error}")
            return None
        self._counters["hits"] += 1
        print(f"CACHE LOADED   [{kind}] {key[:12]}")
        return arrays, metadata

    def save(self, kind: str, identity: Mapping, arrays: Mapping[str, np.ndarray], metadata: Mapping) -> dict:
        if not self.enabled:
            print(f"CACHE DISABLED [{kind}] {cache_key(identity)[:12]}: calculated, not stored")
            return dict(metadata)
        array_path, metadata_path, key = self.paths(kind, identity)
        full_metadata = {
            **_jsonable(metadata, nonfinite_to_none=True), "identity": _jsonable(identity), "cache_key": key,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "profile": self.profile, "git_commit": self.git_commit,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with atomic_output_path(array_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
        with atomic_output_path(metadata_path) as temporary:
            temporary.write_text(json.dumps(full_metadata, indent=2, sort_keys=True) + "\n")
        self._counters["writes"] += 1
        print(f"CACHE WRITTEN  [{kind}] {key[:12]}")
        return full_metadata
