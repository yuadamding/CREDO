"""Validated views for multi-time CREDO trajectory training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch

from .core import MeasureKey, SparseTrajectoryProblem, TrajectoryProblem


TrajectoryLike = TrajectoryProblem | SparseTrajectoryProblem


def embedding_id_for_measure_key(key: MeasureKey) -> str:
    """Map a finite-measure key to the perturbation embedding id."""
    if isinstance(key, tuple):
        _, perturbation_id = key
        return str(perturbation_id)
    return str(key)


@dataclass
class TrajectoryView:
    """Source/target view of a trajectory problem.

    The view is deliberately key-aware:

    - ``measure_keys`` index observed finite measures and may be tuples.
    - ``embedding_ids`` index model perturbation embeddings and are strings.
    """

    trajectory: TrajectoryLike
    source_label: str
    target_labels: list[str]
    measure_keys: list[MeasureKey] | None = None
    sparse_missing: str = "mask"
    target_support_cache: Dict[str, Dict[str, Dict[MeasureKey, torch.Tensor]]] = field(
        default_factory=dict, init=False
    )
    target_logw_cache: Dict[str, Dict[str, Dict[MeasureKey, torch.Tensor]]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        if self.sparse_missing not in {"mask", "error"}:
            raise ValueError("sparse_missing must be 'mask' or 'error'.")
        labels = set(self.trajectory.time_labels)
        if self.source_label not in labels:
            raise KeyError(f"Unknown source_label {self.source_label!r}.")
        unknown_targets = [label for label in self.target_labels if label not in labels]
        if unknown_targets:
            raise KeyError(f"Unknown target_labels: {unknown_targets}")
        if not self.target_labels:
            raise ValueError("TrajectoryView requires at least one target label.")

        source_tau = self.trajectory.tau(self.source_label)
        target_taus = [self.trajectory.tau(label) for label in self.target_labels]
        if any(tau <= source_tau for tau in target_taus):
            raise ValueError("All target_labels must occur after source_label.")
        if any(b <= a for a, b in zip(target_taus[:-1], target_taus[1:])):
            raise ValueError("target_labels must be strictly increasing by tau.")

        source_keys = self._available_keys(self.source_label)
        if not source_keys:
            raise ValueError(f"No source keys available at {self.source_label!r}.")
        if self.measure_keys is None:
            self.measure_keys = sorted(source_keys, key=str)
        else:
            self.measure_keys = list(self.measure_keys)
            missing_source = [key for key in self.measure_keys if key not in source_keys]
            if missing_source:
                raise KeyError(f"measure_keys missing at source_label: {missing_source[:5]}")

        if self.sparse_missing == "error":
            missing = {
                label: sorted(set(self.measure_keys) - self._available_keys(label), key=str)
                for label in self.target_labels
            }
            missing = {label: keys for label, keys in missing.items() if keys}
            if missing:
                raise ValueError(f"Trajectory target keys are incomplete: {missing}")

        for label in self.target_labels:
            active = self.active_keys(label)
            if not active:
                raise ValueError(f"No active target keys for target label {label!r}.")

    @property
    def time_labels(self) -> list[str]:
        return [self.source_label] + list(self.target_labels)

    @property
    def observed_taus(self) -> list[float]:
        return [self.trajectory.tau(label) for label in self.time_labels]

    @property
    def source_keys(self) -> list[MeasureKey]:
        return list(self.measure_keys or [])

    @property
    def embedding_ids(self) -> dict[MeasureKey, str]:
        return {key: embedding_id_for_measure_key(key) for key in self.source_keys}

    @property
    def embedding_id_list(self) -> list[str]:
        return [embedding_id_for_measure_key(key) for key in self.source_keys]

    @property
    def target_keys_by_time(self) -> dict[str, set[MeasureKey]]:
        return {label: set(self.active_keys(label)) for label in self.target_labels}

    def _available_keys(self, time_label: str) -> set[MeasureKey]:
        if hasattr(self.trajectory, "available_keys"):
            return set(self.trajectory.available_keys(time_label))  # type: ignore[attr-defined]
        return set(self.trajectory.measures[time_label].keys())

    def active_keys(self, time_label: str) -> list[MeasureKey]:
        available = self._available_keys(time_label)
        return [key for key in self.source_keys if key in available]

    def checkpoint_taus(self, labels: Sequence[str] | None = None) -> list[float]:
        selected = self.time_labels if labels is None else list(labels)
        return [self.trajectory.tau(label) for label in selected]

    def target_tensors(
        self,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[dict[str, dict[MeasureKey, torch.Tensor]], dict[str, dict[MeasureKey, torch.Tensor]]]:
        """Return target support/log-weight tensors for every target label."""
        cache_key = f"{torch.device(device)}:{dtype}"
        if cache_key in self.target_support_cache:
            return self.target_support_cache[cache_key], self.target_logw_cache[cache_key]

        target_support: dict[str, dict[MeasureKey, torch.Tensor]] = {}
        target_logw: dict[str, dict[MeasureKey, torch.Tensor]] = {}
        for label in self.target_labels:
            target_support[label] = {}
            target_logw[label] = {}
            for key in self.active_keys(label):
                mu = self.trajectory.get(label, key)
                support, weights = mu.to_torch(device=str(device), dtype=dtype)
                target_support[label][key] = support
                target_logw[label][key] = torch.log(weights + 1e-30)

        self.target_support_cache[cache_key] = target_support
        self.target_logw_cache[cache_key] = target_logw
        return target_support, target_logw


__all__ = [
    "TrajectoryLike",
    "TrajectoryView",
    "embedding_id_for_measure_key",
]
