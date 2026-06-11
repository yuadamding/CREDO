"""Training adapter for single-time CREDO effect-path problems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ..config.schema import RunConfig
from ..data.single_time import SingleTimeProblem
from ..losses.single_time import (
    control_null_effect_loss,
    guide_concordance_effect_loss,
    minimal_effect_action_loss,
)
from ..models.full_model import FullDynamicsModel
from ..models.single_time_context import SingleTimeContextProvider
from .trainer import Trainer, TrainingHistory


@dataclass
class SingleTimeTrainingHistory:
    """Training history plus explicit single-time claim metadata."""

    history: TrainingHistory
    claim_report: dict[str, object]


class SingleTimeTrainer:
    """Compact adapter from ``SingleTimeProblem`` to the endpoint trainer.

    The underlying objective is still CREDO's finite-measure endpoint objective,
    but the endpoint is a non-physical control-reference -> observed effect
    axis.  Use ``claim_report`` in downstream outputs to avoid longitudinal
    dynamics claims from one-snapshot data.
    """

    def __init__(
        self,
        model: FullDynamicsModel,
        config: RunConfig,
        problem: SingleTimeProblem,
        *,
        count_data: Optional[dict] = None,
        output_dir: str = "outputs",
        ema_decay: float = 0.995,
        warmup_epochs: int = 50,
    ) -> None:
        self.problem = problem
        self.endpoint = problem.to_effect_endpoint_problem()
        self._control_embedding_ids = {
            view.embedding_id
            for view in problem.views
            if view.is_control
        }
        self._target_ids_by_pid = dict(self.endpoint.metadata.get("target_ids", {}))
        single_time_config = config.single_time
        selected_protocol = (
            single_time_config.context_protocol
            if single_time_config.enabled
            else problem.context_protocol
        )
        self.context_provider = SingleTimeContextProvider(
            problem=problem,
            n_particles=config.simulation.n_particles,
            device=config.resolve_device(),
            protocol=selected_protocol,
            context_tau=single_time_config.context_tau,
        )
        self.trainer = Trainer(
            model=model,
            config=config,
            endpoint=self.endpoint,
            supported_pids=self.endpoint.perturbation_ids,
            count_data=count_data,
            output_dir=output_dir,
            ema_decay=ema_decay,
            warmup_epochs=warmup_epochs,
            particle_sampling="measure_weights",
            context_override_provider=self._context_override_for_training,
            extra_loss_callback=self._single_time_extra_loss,
        )

    @property
    def claim_report(self) -> dict[str, object]:
        return self.problem.claim_report()

    def _context_override_for_training(
        self,
        *,
        model: FullDynamicsModel,
        perturbation_ids: list[str],
        epoch: int,
        **_: object,
    ) -> object:
        return self.context_provider.build(
            model,
            seed=self.trainer.config.training.seed + int(epoch),
            perturbation_ids=perturbation_ids,
        )

    def _single_time_extra_loss(
        self,
        *,
        rollout,
        perturbation_ids: list[str],
        **_: object,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stc = self.trainer.config.single_time
        device = rollout.terminal_logw.device
        loss = torch.tensor(0.0, device=device)
        metrics: Dict[str, float] = {}

        log_mass0 = rollout.log_m0.to(device=device, dtype=rollout.terminal_logw.dtype)
        log_mass_terminal = log_mass0 + torch.logsumexp(rollout.terminal_logw, dim=1)
        effect_scores = log_mass_terminal - log_mass0

        if stc.lambda_control_null > 0:
            control_mask = torch.tensor(
                [pid in self._control_embedding_ids for pid in perturbation_ids],
                device=device,
                dtype=torch.bool,
            )
            if not bool(control_mask.any()):
                raise ValueError(
                    "lambda_control_null > 0 requires control views. "
                    "Use multiple controls or enough cells for control-cell split calibration."
                )
            value = control_null_effect_loss(effect_scores, control_mask)
            loss = loss + float(stc.lambda_control_null) * value
            metrics["single_time_control_null"] = float(value.detach().item())

        if stc.lambda_minimal_action > 0:
            value = minimal_effect_action_loss(
                drift_steps=rollout.drift_steps,
                sigma_steps=rollout.sigma_steps,
                growth_steps=rollout.growth_steps,
            )
            loss = loss + float(stc.lambda_minimal_action) * value
            metrics["single_time_minimal_action"] = float(value.detach().item())

        if stc.lambda_guide_concordance > 0:
            target_ids = [
                str(self._target_ids_by_pid.get(pid, pid))
                for pid in perturbation_ids
            ]
            value = guide_concordance_effect_loss(effect_scores, target_ids)
            loss = loss + float(stc.lambda_guide_concordance) * value
            metrics["single_time_guide_concordance"] = float(value.detach().item())

        metrics["loss_single_time"] = float(loss.detach().item())
        return loss, metrics

    def train(self, stage: str = "all", n_epochs: Optional[int] = None) -> SingleTimeTrainingHistory:
        history = self.trainer.train(stage=stage, n_epochs=n_epochs)
        return SingleTimeTrainingHistory(history=history, claim_report=self.claim_report)


__all__ = ["SingleTimeTrainer", "SingleTimeTrainingHistory"]
