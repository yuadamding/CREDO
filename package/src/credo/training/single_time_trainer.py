"""Training adapter for single-time CREDO effect-path problems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config.schema import RunConfig
from ..data.single_time import SingleTimeProblem
from ..models.full_model import FullDynamicsModel
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
        self.trainer = Trainer(
            model=model,
            config=config,
            endpoint=self.endpoint,
            supported_pids=self.endpoint.perturbation_ids,
            count_data=count_data,
            output_dir=output_dir,
            ema_decay=ema_decay,
            warmup_epochs=warmup_epochs,
        )

    @property
    def claim_report(self) -> dict[str, object]:
        return self.problem.claim_report()

    def train(self, stage: str = "all", n_epochs: Optional[int] = None) -> SingleTimeTrainingHistory:
        history = self.trainer.train(stage=stage, n_epochs=n_epochs)
        return SingleTimeTrainingHistory(history=history, claim_report=self.claim_report)


__all__ = ["SingleTimeTrainer", "SingleTimeTrainingHistory"]
