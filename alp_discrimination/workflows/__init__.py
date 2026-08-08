"""Small helpers shared by command-line orchestration modules."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from alp_discrimination.cache import CacheStore, atomic_output_path, git_commit
from alp_discrimination.config import PROFILES, AnalysisConfig, get_config
from alp_discrimination.paths import portable_path

if TYPE_CHECKING:
    from alp_discrimination.eventcalc_adapter import EventCalcAdapter


def add_profile_cache_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument("--force", action="store_true", help="Recompute requested cache entries")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write caches")


def config_and_adapter(args: Namespace) -> tuple[AnalysisConfig, "EventCalcAdapter"]:
    from alp_discrimination.eventcalc_adapter import EventCalcAdapter

    config = get_config(args.profile)
    cache = CacheStore(config.name, enabled=not args.no_cache)
    return config, EventCalcAdapter(config, cache=cache, force=args.force)


def write_dataframe(data: pd.DataFrame, path: Path) -> Path:
    with atomic_output_path(path) as temporary:
        data.to_csv(temporary, index=False)
    return path


def write_manifest(
    config: AnalysisConfig,
    workflow: str,
    output_dir: Path,
    *,
    elapsed_seconds: float | None = None,
    cache_stats: dict[str, int] | None = None,
    artifacts: list[Path] | None = None,
    extra: dict | None = None,
) -> Path:
    payload = {
        "workflow": workflow, "profile": config.name, "selection_name": config.selection_name,
        "configuration": asdict(config), "git_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = float(elapsed_seconds)
    if cache_stats is not None:
        payload["cache_stats"] = dict(cache_stats)
    if artifacts is not None:
        payload["artifacts"] = [portable_path(path) for path in artifacts]
    if extra:
        payload.update(extra)
    path = output_dir / "manifest.json"
    with atomic_output_path(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m").replace("+", "")


def require_columns(data: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
