"""Intermediate reporting / pruning protocol for CREDO trials.

A trainer accepts an optional :class:`SearchReporter`. After each lightweight
evaluation it calls ``report(step, metrics)`` and then checks ``should_prune()``;
if pruning is requested it raises :class:`TrialPrunedError` so the optimizer can
stop the trial early. This keeps the optimizer (Optuna/Ray/controller) fully
decoupled from the dynamical model -- the trainer only sees this small Protocol.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .metrics import CREDOTrialMetrics


class TrialPrunedError(RuntimeError):
    """Raised inside the training loop when the reporter requests pruning."""

    def __init__(self, epoch: int | None = None, message: str | None = None) -> None:
        self.epoch = epoch
        super().__init__(message or f"trial pruned at epoch {epoch}")


@runtime_checkable
class SearchReporter(Protocol):
    """Minimal callback the trainer uses to report progress and ask to prune."""

    def report(self, step: int, metrics: CREDOTrialMetrics) -> None: ...

    def should_prune(self) -> bool: ...


class NoOpReporter:
    """Default reporter: records nothing and never prunes."""

    def report(self, step: int, metrics: CREDOTrialMetrics) -> None:  # noqa: D401
        return None

    def should_prune(self) -> bool:
        return False


class RecordingReporter:
    """In-memory reporter useful for tests and offline analysis."""

    def __init__(self, prune_after: int | None = None) -> None:
        self.history: list[tuple[int, CREDOTrialMetrics]] = []
        self._prune_after = prune_after

    def report(self, step: int, metrics: CREDOTrialMetrics) -> None:
        self.history.append((step, metrics))

    def should_prune(self) -> bool:
        if self._prune_after is None:
            return False
        return len(self.history) >= self._prune_after


__all__ = [
    "NoOpReporter",
    "RecordingReporter",
    "SearchReporter",
    "TrialPrunedError",
]
