"""Training-side pruning signal.

A trainer raises :class:`TrainingPruned` when an injected reporter requests early
stopping (after persisting the pruned checkpoint and history). This lives in
``credo.training`` so the trainers never import ``credo.search``; the search
layer recognizes it duck-typed via the ``_credo_pruned`` marker and translates
it into its own ``TrialPrunedError`` (so a pruned trial is reported as pruned,
never scored as a short completed run).
"""
from __future__ import annotations


class TrainingPruned(RuntimeError):
    """Raised inside a training loop when the reporter requests pruning."""

    # Duck-typed marker so the search layer can translate without importing this
    # module (which would pull torch via credo.training).
    _credo_pruned = True

    def __init__(self, epoch: int | None = None) -> None:
        self.epoch = None if epoch is None else int(epoch)
        super().__init__(f"training pruned at epoch {self.epoch}")


__all__ = ["TrainingPruned"]
