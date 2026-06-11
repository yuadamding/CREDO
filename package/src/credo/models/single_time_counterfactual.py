"""Counterfactual utilities for single-time CREDO effect paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from ..data.single_time import SingleTimeContextProtocol, SingleTimeProblem
from .full_model import FullDynamicsModel
from .simulator import CounterfactualResult, _control_embedding_context, initialise_particles_from_measures
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
        seed: int = 0,
        context_override: Any = None,
    ) -> Any:
        """Build the static context override for the requested single-time protocol."""
        selected = protocol or problem.context_protocol
        if selected == "self_consistent":
            return None
        if selected == "clamped_external":
            if context_override is None:
                raise ValueError("context_protocol='clamped_external' requires context_override.")
            return context_override
        endpoint = problem.to_effect_endpoint_problem()
        measures = endpoint.terminal if selected == "observed_snapshot" else endpoint.initial
        z_ctx, lw_ctx, lm_ctx = initialise_particles_from_measures(
            measures,
            endpoint.perturbation_ids,
            self.n_particles,
            self.device,
            seed=int(seed) + 50_000,
        )
        _, ctx = self.model.step(
            z=z_ctx,
            tau=torch.tensor(0.0, dtype=z_ctx.dtype, device=z_ctx.device),
            logw=lw_ctx,
            log_m0=lm_ctx,
            perturbation_ids=endpoint.perturbation_ids,
        )
        return ctx

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
        context_override: Any = None,
    ) -> List[CounterfactualResult]:
        """Run factual vs. reference branches from the same matched control source."""
        if control_rollout_mode not in {"reference_consistent", "zero_centered"}:
            raise ValueError(
                "control_rollout_mode must be 'reference_consistent' or 'zero_centered'."
            )
        endpoint = problem.to_effect_endpoint_problem()
        all_pids = endpoint.perturbation_ids
        target_pids = perturbation_ids or problem.perturbation_ids
        unknown = [pid for pid in target_pids if pid not in all_pids]
        if unknown:
            raise KeyError(f"Unknown single-time perturbation_ids: {unknown}")

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
            seed=seed,
            context_override=context_override,
        )
        factual_all = self.simulator.rollout(
            z0=z0_all,
            logw0=lw0_all,
            model=self.model,
            log_m0=lm0_all,
            perturbation_ids=all_pids,
            noise_steps=noise_steps,
            return_noise_used=common_noise,
            context_override=selected_context_override,
        )

        results: list[CounterfactualResult] = []
        for pid in target_pids:
            with _control_embedding_context(self.model, pid, mode=control_rollout_mode):
                reference_all = self.simulator.rollout(
                    z0=z0_all.clone(),
                    logw0=lw0_all.clone(),
                    model=self.model,
                    log_m0=lm0_all.clone(),
                    perturbation_ids=all_pids,
                    noise_steps=noise_steps.clone() if noise_steps is not None else None,
                    return_noise_used=common_noise,
                    context_override=selected_context_override,
                )
            idx = all_pids.index(pid)
            metadata = {
                **endpoint.metadata,
                "counterfactual_type": "single_time_effect_path",
                "target_perturbation_id": pid,
                "same_reference_source": True,
                "same_start": True,
                "same_noise": bool(common_noise),
                "context_protocol": selected_protocol,
                "initial_seed": int(seed),
                "noise_seed": noise_seed if common_noise else None,
                "factual_full_context_reused": True,
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
