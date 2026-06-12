"""Shared context providers for single-time CREDO effect paths."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

import torch

from ..data.core import EndpointProblem
from ..data.single_time import SingleTimeContextProtocol, SingleTimeProblem
from .full_model import FullDynamicsModel
from .simulator import embedding_ids_from_endpoint, initialise_particles_from_measures


ContextTau = float | Literal["auto", "source", "target", "midpoint"]


def _detach_cached_value(value: Any) -> Any:
    """Detach tensors before storing fixed context across training epochs."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        init_kwargs = {
            item.name: _detach_cached_value(getattr(value, item.name))
            for item in fields(value)
            if item.init
        }
        detached = type(value)(**init_kwargs)
        for item in fields(value):
            if not item.init:
                setattr(detached, item.name, _detach_cached_value(getattr(value, item.name)))
        return detached
    if isinstance(value, dict):
        return {key: _detach_cached_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_cached_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_cached_value(item) for item in value)
    return value


def resolve_single_time_context_tau(
    protocol: SingleTimeContextProtocol,
    context_tau: ContextTau = "auto",
) -> float:
    """Resolve single-time context tau onto the non-physical effect axis."""
    if isinstance(context_tau, (float, int)):
        return float(context_tau)
    if context_tau == "source":
        return 0.0
    if context_tau == "target":
        return 1.0
    if context_tau == "midpoint":
        return 0.5
    if context_tau != "auto":
        raise ValueError("context_tau must be a float or one of auto/source/target/midpoint.")
    if protocol == "observed_snapshot":
        return 1.0
    if protocol == "source_reference":
        return 0.0
    return 0.0


@dataclass
class SingleTimeContextProvider:
    """Build static context overrides for single-time effect-path protocols."""

    problem: SingleTimeProblem
    n_particles: int
    device: str = "cpu"
    protocol: SingleTimeContextProtocol | None = None
    context_tau: ContextTau = "auto"
    context_override: Any = None
    endpoint: EndpointProblem | None = None
    context_sampling: Literal["fixed", "epoch_resample"] = "fixed"
    seed_offset: int = 50_000
    _cached_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _cached_context: Any = field(default=None, init=False, repr=False)

    def build(
        self,
        model: FullDynamicsModel,
        *,
        seed: int = 0,
        perturbation_ids: list[str] | None = None,
    ) -> Any:
        selected = self.protocol or self.problem.context_protocol
        if selected == "self_consistent":
            return None
        if selected == "clamped_external":
            if self.context_override is None:
                raise ValueError("context_protocol='clamped_external' requires context_override.")
            return self.context_override
        if self.context_sampling not in {"fixed", "epoch_resample"}:
            raise ValueError("context_sampling must be 'fixed' or 'epoch_resample'.")

        endpoint = self.endpoint or self.problem.to_effect_endpoint_problem()
        pids = perturbation_ids or endpoint.perturbation_ids
        missing = [pid for pid in pids if pid not in endpoint.initial or pid not in endpoint.terminal]
        if missing:
            raise KeyError(f"Context perturbation_ids missing from single-time endpoint: {missing}")
        tau_value = resolve_single_time_context_tau(selected, self.context_tau)
        cache_key = (selected, float(tau_value), tuple(pids), int(self.n_particles))
        if self.context_sampling == "fixed" and self._cached_key == cache_key:
            return self._cached_context

        measures = endpoint.terminal if selected == "observed_snapshot" else endpoint.initial
        z_ctx, lw_ctx, lm_ctx = initialise_particles_from_measures(
            measures,
            pids,
            self.n_particles,
            self.device,
            seed=int(seed) + int(self.seed_offset),
        )
        grad_enabled = self.context_sampling != "fixed"
        with torch.set_grad_enabled(grad_enabled):
            _, ctx = model.step(
                z=z_ctx,
                tau=torch.tensor(tau_value, dtype=z_ctx.dtype, device=z_ctx.device),
                logw=lw_ctx,
                log_m0=lm_ctx,
                perturbation_ids=pids,
                embedding_ids=embedding_ids_from_endpoint(endpoint, pids),
            )
        if self.context_sampling == "fixed":
            ctx = _detach_cached_value(ctx)
            self._cached_key = cache_key
            self._cached_context = ctx
        return ctx


__all__ = [
    "ContextTau",
    "SingleTimeContextProvider",
    "resolve_single_time_context_tau",
]
