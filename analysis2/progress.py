"""Small progress/ETA helpers for long ALP-SU2L analysis stages."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import threading
import time


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not float(seconds) < float("inf"):
        return "unknown"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


class ProgressMeter:
    """Throughput-based ETA for a finite set of comparable jobs."""

    def __init__(self, *, total: int, label: str = "progress") -> None:
        self.total = int(total)
        self.label = str(label)
        self.started = time.perf_counter()

    def message(self, completed: int) -> str:
        completed = int(completed)
        elapsed = max(0.0, time.perf_counter() - self.started)
        remaining_jobs = max(0, self.total - completed)
        if completed <= 0:
            return f"elapsed={format_duration(elapsed)}, ETA=learning"
        rate = elapsed / float(completed)
        eta = rate * float(remaining_jobs)
        finish = datetime.now() + timedelta(seconds=eta)
        return (
            f"elapsed={format_duration(elapsed)}, "
            f"ETA={format_duration(eta)}, finish~{finish:%H:%M}"
        )


class CheckpointMonitor:
    """Monitor truth/seed checkpoint files while a direct pilot call blocks."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        truth_table: Path,
        seeds_per_truth: int,
        label: str,
        poll_seconds: float = 20.0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.truth_table = Path(truth_table)
        self.seeds_per_truth = int(seeds_per_truth)
        self.label = str(label)
        self.poll_seconds = float(poll_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._initial = 0

    def _completed(self) -> int:
        if not self.checkpoint_dir.is_dir():
            return 0
        return sum(1 for _ in self.checkpoint_dir.glob("*.csv"))

    def _total(self) -> int | None:
        if not self.truth_table.is_file():
            return None
        try:
            with self.truth_table.open("r", encoding="utf-8") as handle:
                rows = max(0, sum(1 for _ in handle) - 1)
        except OSError:
            return None
        return rows * self.seeds_per_truth

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started = time.perf_counter()
        self._initial = self._completed()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds + 1.0))

    def _run(self) -> None:
        last = None
        while not self._stop.wait(self.poll_seconds):
            completed = self._completed()
            total = self._total()
            if total is None or total <= 0:
                continue
            if completed == last:
                continue
            last = completed
            elapsed = max(0.0, time.perf_counter() - self._started)
            new = max(0, completed - self._initial)
            remaining = max(0, total - completed)
            if new > 0:
                eta = elapsed / float(new) * float(remaining)
                finish = datetime.now() + timedelta(seconds=eta)
                eta_text = (
                    f"ETA={format_duration(eta)}, finish~{finish:%H:%M}"
                )
            else:
                eta_text = "ETA=learning"
            print(
                f"PROGRESS {self.label}: {completed}/{total} checkpoints | "
                f"elapsed={format_duration(elapsed)}, {eta_text}",
                flush=True,
            )
