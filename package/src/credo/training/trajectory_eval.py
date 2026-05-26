"""Evaluation helpers for trajectory training."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch

from ..data.core import MeasureKey
from ..data.trajectory_view import TrajectoryView, embedding_id_for_measure_key
from ..models.weighted_sde import ParticleRollout


@dataclass
class TrajectoryEvaluation:
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def rollout_metrics_by_key_time(
    view: TrajectoryView,
    rollout: ParticleRollout,
    checkpoint_indices: dict[str, int],
    measure_keys: list[MeasureKey],
    endpoint_logs: dict[str, torch.Tensor] | None = None,
) -> pd.DataFrame:
    """Build a compact per-key/time prediction table."""
    rows = []
    endpoint_logs = endpoint_logs or {}
    if rollout.log_m0 is None:
        raise ValueError("rollout_metrics_by_key_time requires rollout.log_m0.")

    for g, key in enumerate(measure_keys):
        source_mu = view.trajectory.get(view.source_label, key)
        for label in view.target_labels:
            if key not in view.trajectory.measures[label]:
                continue
            target_mu = view.trajectory.get(label, key)
            idx = checkpoint_indices[label]
            pred_log_mass = rollout.log_m0[g] + torch.logsumexp(rollout.logw_steps[idx, g], dim=0)
            target_log_mass = pred_log_mass.new_tensor(float(target_mu.total_mass)).log()
            if isinstance(key, tuple):
                sample_id, perturbation_id = key
            else:
                sample_id, perturbation_id = "", str(key)
            rows.append(
                {
                    "measure_key": str(key),
                    "sample_id": str(sample_id),
                    "perturbation_id": str(perturbation_id),
                    "embedding_id": embedding_id_for_measure_key(key),
                    "time_label": label,
                    "tau": float(view.trajectory.tau(label)),
                    "normalized_tau": float(view.trajectory.tau(label)),
                    "physical_time": float(view.trajectory.time_axis.physical(label)),
                    "source_physical_time": float(view.trajectory.time_axis.physical(view.source_label)),
                    "interval_physical_duration": float(
                        view.trajectory.time_axis.physical(label)
                        - view.trajectory.time_axis.physical(view.source_label)
                    ),
                    "n_source_cells": int(source_mu.n_atoms),
                    "n_target_cells": int(target_mu.n_atoms),
                    "source_mass": float(source_mu.total_mass),
                    "target_mass": float(target_mu.total_mass),
                    "predicted_mass": float(pred_log_mass.detach().exp().cpu()),
                    "log_mass_error": float(
                        (pred_log_mass - target_log_mass).detach().cpu()
                    ),
                    "endpoint_loss": float(endpoint_logs.get(f"endpoint/{label}", torch.tensor(float("nan")))),
                    "geom_loss": float(endpoint_logs.get(f"endpoint/{label}/geom_mean", torch.tensor(float("nan")))),
                    "mass_loss": float(endpoint_logs.get(f"endpoint/{label}/mass_mean", torch.tensor(float("nan")))),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["TrajectoryEvaluation", "rollout_metrics_by_key_time"]
