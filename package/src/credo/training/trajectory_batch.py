"""Batch helpers for multi-time trajectory training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from ..data.core import MeasureKey
from ..data.trajectory_view import TrajectoryView, embedding_id_for_measure_key
from ..models.context import ContextState, GroupStatistics
from ..models.simulator import initialise_particles_from_trajectory
from ..models.weighted_sde import ParticleRollout


@dataclass(frozen=True)
class TrajectoryBatch:
    measure_keys: list[MeasureKey]
    embedding_ids: list[str]
    source_label: str
    target_labels: list[str]
    tau_grid: torch.Tensor
    checkpoint_indices: dict[str, int]


def embedding_ids_for_measure_keys(measure_keys: list[MeasureKey]) -> list[str]:
    return [embedding_id_for_measure_key(key) for key in measure_keys]


class TargetBalancedTrajectorySampler:
    """Build target-gene batches while retaining all guide/donor views per target."""

    def __init__(
        self,
        view: TrajectoryView,
        *,
        genes_per_batch: int = 32,
        controls_per_batch: int = 16,
        max_active_measure_keys: int = 0,
        seed: int = 0,
    ) -> None:
        if genes_per_batch < 1:
            raise ValueError("genes_per_batch must be >= 1.")
        if controls_per_batch < 0:
            raise ValueError("controls_per_batch must be >= 0.")
        if max_active_measure_keys < 0:
            raise ValueError("max_active_measure_keys must be >= 0.")
        self.view = view
        self.genes_per_batch = int(genes_per_batch)
        self.controls_per_batch = int(controls_per_batch)
        self.max_active_measure_keys = int(max_active_measure_keys)
        self.seed = int(seed)
        self.control_keys = list(view.control_measure_keys)
        control_set = set(self.control_keys)
        self.keys_by_target: dict[str, list[MeasureKey]] = {}
        for key in view.source_keys:
            if key in control_set:
                continue
            self.keys_by_target.setdefault(view.target_gene(key), []).append(key)
        if not self.keys_by_target and not self.control_keys:
            raise ValueError("Trajectory sampler has no target or control measure keys.")
        if not self.keys_by_target and self.controls_per_batch == 0:
            raise ValueError("A control-only trajectory batch requires controls_per_batch > 0.")
        selected_control_count = min(len(self.control_keys), self.controls_per_batch)
        if self.max_active_measure_keys and selected_control_count > self.max_active_measure_keys:
            raise ValueError(
                "Selected controls exceed max_active_measure_keys before target genes are added."
            )

    @property
    def target_genes(self) -> list[str]:
        return sorted(self.keys_by_target)

    def batches(self, sweep: int = 0) -> Iterator[list[MeasureKey]]:
        rng = np.random.default_rng(self.seed + int(sweep))
        genes = np.asarray(self.target_genes, dtype=object)
        if genes.size:
            rng.shuffle(genes)

        controls = list(self.control_keys)
        if self.controls_per_batch and len(controls) > self.controls_per_batch:
            selected = rng.choice(len(controls), size=self.controls_per_batch, replace=False)
            controls = [controls[int(idx)] for idx in sorted(selected.tolist())]
        elif self.controls_per_batch == 0:
            controls = []

        if not genes.size:
            yield controls
            return

        current: list[MeasureKey] = list(controls)
        n_current_genes = 0
        for gene in genes.tolist():
            gene_keys = self.keys_by_target[str(gene)]
            would_exceed_keys = (
                self.max_active_measure_keys > 0
                and len(current) + len(gene_keys) > self.max_active_measure_keys
            )
            would_exceed_genes = n_current_genes >= self.genes_per_batch
            if n_current_genes > 0 and (would_exceed_keys or would_exceed_genes):
                yield current
                current = list(controls)
                n_current_genes = 0
            if self.max_active_measure_keys and len(current) + len(gene_keys) > self.max_active_measure_keys:
                raise ValueError(
                    f"Target {gene!r} has too many guide/donor views to fit "
                    "max_active_measure_keys without splitting a target."
                )
            current.extend(gene_keys)
            n_current_genes += 1
        if n_current_genes > 0:
            yield current


class TrajectoryContextBank:
    """Detached full-catalog group statistics with differentiable active-key replacement."""

    def __init__(
        self,
        *,
        context_aggregator: torch.nn.Module,
        initial_statistics: GroupStatistics,
        context_group_index: torch.Tensor,
        n_steps: int,
        momentum: float = 0.9,
    ) -> None:
        if n_steps < 1:
            raise ValueError("TrajectoryContextBank n_steps must be >= 1.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("TrajectoryContextBank momentum must be in [0, 1).")
        n_keys = initial_statistics.log_n_g.shape[0]
        if context_group_index.shape != (n_keys,):
            raise ValueError("context_group_index must have shape [n_keys].")
        self.context_aggregator = context_aggregator
        self.context_group_index = context_group_index.detach().clone().long()
        self.log_n_steps = initial_statistics.log_n_g.detach().clone().repeat(n_steps, 1)
        self.eta_steps = initial_statistics.eta_g.detach().clone().repeat(n_steps, 1, 1)
        self.phi_steps = initial_statistics.phi_g.detach().clone().repeat(n_steps, 1, 1)
        self.momentum = float(momentum)

    def _merged_state(
        self,
        *,
        step_index: int,
        z: torch.Tensor,
        logw: torch.Tensor,
        log_m0: torch.Tensor,
        active_key_indices: torch.Tensor,
        replace_active: bool,
    ) -> ContextState:
        active = active_key_indices.to(device=z.device, dtype=torch.long)
        log_n = self.log_n_steps[step_index].to(z).detach()
        eta = self.eta_steps[step_index].to(z).detach()
        phi = self.phi_steps[step_index].to(z).detach()
        if replace_active:
            active_stats, _, _ = self.context_aggregator.summarize_groups(z, logw, log_m0)
            log_n = log_n.index_copy(0, active, active_stats.log_n_g)
            eta = eta.index_copy(0, active, active_stats.eta_g)
            phi = phi.index_copy(0, active, active_stats.phi_g)
        full_state = self.context_aggregator.context_from_group_statistics(
            GroupStatistics(log_n_g=log_n, eta_g=eta, phi_g=phi),
            context_group_index=self.context_group_index.to(z.device),
        )
        context = full_state.context.index_select(0, active)
        q = full_state.q.index_select(0, active)
        s = full_state.s.index_select(0, active)
        return ContextState(
            q=q,
            s=s,
            context=context,
            mass_g=full_state.mass_g.index_select(0, active),
            freq_g=full_state.freq_g.index_select(0, active),
            log_mass_g=full_state.log_mass_g.index_select(0, active),
            log_total_mass=full_state.log_total_mass,
            base_context=context,
            growth_context=context,
        )

    def override(
        self,
        active_key_indices: torch.Tensor,
        *,
        replace_active: bool = True,
    ):
        active = active_key_indices.detach().clone()

        def provider(*, step_index, z, logw, log_m0, **_):
            return self._merged_state(
                step_index=int(step_index),
                z=z,
                logw=logw,
                log_m0=log_m0,
                active_key_indices=active,
                replace_active=replace_active,
            )

        return provider

    @torch.no_grad()
    def update(
        self,
        rollout: ParticleRollout,
        active_key_indices: torch.Tensor,
        *,
        hard: bool = False,
    ) -> None:
        active = active_key_indices.to(device=self.log_n_steps.device, dtype=torch.long)
        if rollout.log_m0 is None:
            raise ValueError("TrajectoryContextBank.update requires rollout.log_m0.")
        n_steps = min(self.log_n_steps.shape[0], rollout.z_steps.shape[0] - 1)
        blend = 0.0 if hard else self.momentum
        for step in range(n_steps):
            stats, _, _ = self.context_aggregator.summarize_groups(
                rollout.z_steps[step],
                rollout.logw_steps[step],
                rollout.log_m0,
            )
            for bank, current in (
                (self.log_n_steps, stats.log_n_g),
                (self.eta_steps, stats.eta_g),
                (self.phi_steps, stats.phi_g),
            ):
                old = bank[step].index_select(0, active)
                value = blend * old + (1.0 - blend) * current.to(old)
                bank[step].index_copy_(0, active, value)


__all__ = [
    "TrajectoryBatch",
    "TargetBalancedTrajectorySampler",
    "TrajectoryContextBank",
    "embedding_ids_for_measure_keys",
    "initialise_particles_from_trajectory",
]
