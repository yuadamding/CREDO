"""Counterfactual utilities for single-time CREDO effect paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from ..data.single_time import SingleTimeContextProtocol, SingleTimeProblem
from .full_model import FullDynamicsModel
from .single_time_context import ContextGradientMode, ContextTau, SingleTimeContextProvider
from .simulator import (
    CounterfactualResult,
    _control_embedding_context,
    embedding_ids_from_endpoint,
    initialise_particles_from_measures,
)
from .weighted_sde import WeightedParticleSimulator


@dataclass
class SingleTimeCounterfactualEngine:
    """Same-reference-source counterfactuals for a SingleTimeProblem."""

    model: FullDynamicsModel
    simulator: WeightedParticleSimulator
    n_particles: int = 256
    device: str = "cpu"

    @torch.no_grad()
    def context_override_from_problem(
        self,
        problem: SingleTimeProblem,
        *,
        protocol: Optional[SingleTimeContextProtocol] = None,
        context_tau: ContextTau = "auto",
        context_sampling: str = "fixed",
        context_gradient_mode: ContextGradientMode = "recompute_no_grad",
        seed: int = 0,
        context_override: Any = None,
    ) -> Any:
        """Build the static context override for the requested single-time protocol."""
        return SingleTimeContextProvider(
            problem=problem,
            n_particles=self.n_particles,
            device=self.device,
            protocol=protocol,
            context_tau=context_tau,
            context_override=context_override,
            endpoint=problem.to_effect_endpoint_problem(view_level="view"),
            context_sampling=context_sampling,  # type: ignore[arg-type]
            context_gradient_mode=context_gradient_mode,
        ).build(self.model, seed=seed)

    @torch.no_grad()
    def run(
        self,
        problem: SingleTimeProblem,
        perturbation_ids: Optional[List[str]] = None,
        *,
        seed: int = 0,
        common_noise: bool = True,
        control_rollout_mode: str = "reference_consistent",
        context_protocol: Optional[SingleTimeContextProtocol] = None,
        context_tau: ContextTau = "auto",
        context_sampling: str = "fixed",
        context_gradient_mode: ContextGradientMode = "recompute_no_grad",
        context_override: Any = None,
    ) -> List[CounterfactualResult]:
        """Run factual vs. reference branches from the same matched control source."""
        if control_rollout_mode not in {"reference_consistent", "zero_centered"}:
            raise ValueError(
                "control_rollout_mode must be 'reference_consistent' or 'zero_centered'."
            )
        endpoint = problem.to_effect_endpoint_problem(view_level="view")
        all_pids = endpoint.perturbation_ids
        control_measure_keys = set(endpoint.metadata.get("control_measure_keys", []))
        measure_to_embedding = endpoint.metadata.get("measure_to_embedding", {})
        measure_to_original = endpoint.metadata.get("measure_to_original_perturbation", {})

        def _resolve_targets(requested: Optional[List[str]]) -> list[str]:
            if requested is None:
                return [pid for pid in all_pids if pid not in control_measure_keys]
            resolved: list[str] = []
            unknown: list[str] = []
            for item in requested:
                key = str(item)
                if key in all_pids:
                    resolved.append(key)
                    continue
                matches = [
                    pid for pid in all_pids
                    if str(measure_to_embedding.get(pid, pid)) == key
                    or str(measure_to_original.get(pid, pid)) == key
                ]
                if matches:
                    resolved.extend(matches)
                else:
                    unknown.append(key)
            if unknown:
                raise KeyError(f"Unknown single-time perturbation_ids: {unknown}")
            return list(dict.fromkeys(resolved))

        target_pids = _resolve_targets(perturbation_ids)
        all_embedding_ids = embedding_ids_from_endpoint(endpoint, all_pids)

        z0_all, lw0_all, lm0_all = initialise_particles_from_measures(
            endpoint.initial,
            all_pids,
            self.n_particles,
            self.device,
            seed=seed,
        )
        noise_seed = int(seed) + 10_000
        noise_steps = None
        if common_noise:
            noise_steps = self.simulator.sample_noise_like(
                z0_all,
                self.simulator.n_steps,
                seed=noise_seed,
            )
        selected_protocol = context_protocol or problem.context_protocol
        selected_context_override = self.context_override_from_problem(
            problem,
            protocol=selected_protocol,
            context_tau=context_tau,
            context_sampling=context_sampling,
            context_gradient_mode=context_gradient_mode,
            seed=seed,
            context_override=context_override,
        )
        factual_all = self.simulator.rollout(
            z0=z0_all,
            logw0=lw0_all,
            model=self.model,
            log_m0=lm0_all,
            perturbation_ids=all_pids,
            embedding_ids=all_embedding_ids,
            noise_steps=noise_steps,
            return_noise_used=common_noise,
            context_override=selected_context_override,
        )

        reference_by_embedding: dict[str, Any] = {}
        for target_embedding_id in dict.fromkeys(embedding_ids_from_endpoint(endpoint, target_pids)):
            with _control_embedding_context(self.model, target_embedding_id, mode=control_rollout_mode):
                reference_by_embedding[target_embedding_id] = self.simulator.rollout(
                    z0=z0_all.clone(),
                    logw0=lw0_all.clone(),
                    model=self.model,
                    log_m0=lm0_all.clone(),
                    perturbation_ids=all_pids,
                    embedding_ids=all_embedding_ids,
                    noise_steps=noise_steps.clone() if noise_steps is not None else None,
                    return_noise_used=common_noise,
                    context_override=selected_context_override,
                )

        results: list[CounterfactualResult] = []
        for pid in target_pids:
            target_embedding_id = embedding_ids_from_endpoint(endpoint, [pid])[0]
            reference_all = reference_by_embedding[target_embedding_id]
            idx = all_pids.index(pid)
            metadata = {
                **endpoint.metadata,
                "counterfactual_type": "single_time_effect_path",
                "target_measure_key": pid,
                "target_perturbation_id": endpoint.metadata.get(
                    "measure_to_original_perturbation",
                    {},
                ).get(pid, pid),
                "target_embedding_id": target_embedding_id,
                "same_reference_source": True,
                "same_start": True,
                "same_start_semantics": "constructed_reference_source",
                "same_noise": bool(common_noise),
                "context_protocol": selected_protocol,
                "context_tau": context_tau,
                "context_sampling": context_sampling,
                "context_gradient_mode": context_gradient_mode,
                "initial_seed": int(seed),
                "noise_seed": noise_seed if common_noise else None,
                "factual_full_context_reused": True,
                "reference_rollout_cache_key": target_embedding_id,
                "reference_rollouts_cached_by_embedding": True,
                "rollout_control_semantics": control_rollout_mode,
            }
            results.append(
                CounterfactualResult(
                    perturbation_id=pid,
                    rollout_perturb=factual_all.slice_group(idx),
                    rollout_control=reference_all.slice_group(idx),
                    metadata=metadata,
                )
            )
        return results


__all__ = ["SingleTimeCounterfactualEngine"]
