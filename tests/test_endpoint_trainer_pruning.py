from __future__ import annotations

import numpy as np
import pytest

from credo.config.schema import ModelConfig, RunConfig, SimulationConfig, TrainingConfig
from credo.data.core import EndpointProblem, FiniteMeasure, TimeAxis
from credo.models.full_model import FullDynamicsModel
from credo.training.pruning import TrainingPruned
from credo.training.trainer import Trainer


pytestmark = pytest.mark.integration


def _tiny_endpoint(perturbation_ids: list[str]) -> EndpointProblem:
    support = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
    weights = np.ones(3, dtype=np.float32)
    measure = FiniteMeasure(support=support, weights=weights, total_mass=float(weights.sum()))
    return EndpointProblem(
        initial={pid: measure for pid in perturbation_ids},
        terminal={pid: measure for pid in perturbation_ids},
        time_axis=TimeAxis(["t0", "t1"], [0.0, 1.0]),
        perturbation_ids=perturbation_ids,
    )


def _tiny_model(perturbation_ids: list[str]) -> FullDynamicsModel:
    return FullDynamicsModel(
        perturbation_ids=perturbation_ids,
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=4,
        n_programs=3,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        control_mode="soft_ref",
    )


def _tiny_config(tmp_path, *, multi_gpu: bool = False) -> RunConfig:
    return RunConfig(
        output_dir=str(tmp_path),
        device="cpu",
        multi_gpu_devices=["cuda:0", "cuda:1"] if multi_gpu else [],
        model=ModelConfig(
            embedding_dim=4, n_programs=3, mediator_dim=2, hidden_dim=8, depth=1,
            ecological_growth=False,
        ),
        simulation=SimulationConfig(n_particles=3, n_steps=1),
        training=TrainingConfig(
            epochs=5,
            control_ref_warmup_epochs=0,
            lambda_count=0.0,
            lambda_weak=0.0,
            lambda_reg_net=0.0,
            lambda_reg_diffusion=0.0,
            lambda_reg_embed=0.0,
            lambda_reg_growth_bias=0.0,
        ),
    )


def test_endpoint_trainer_reporter_prunes_and_raises(tmp_path) -> None:
    pids = ["ctrl", "gene_a"]

    class _Reporter:
        def __init__(self) -> None:
            self.calls = 0

        def report(self, epoch, metrics):
            self.calls += 1

        def should_prune(self):
            return self.calls >= 2

    trainer = Trainer(
        model=_tiny_model(pids),
        config=_tiny_config(tmp_path),
        endpoint=_tiny_endpoint(pids),
        supported_pids=pids,
        output_dir=str(tmp_path),
        ema_decay=0.0,
        warmup_epochs=1,
        reporter=_Reporter(),
    )
    # Pruning raises TrainingPruned so the trial is reported as pruned, not scored
    # as a short completed run.
    with pytest.raises(TrainingPruned):
        trainer.train()
    assert trainer._pruned_epoch is not None
    # Pruned history uses the same filename as completed history.
    assert (tmp_path / "training_history.csv").exists()


def test_endpoint_trainer_reporter_rejects_multi_gpu(tmp_path) -> None:
    pids = ["ctrl"]

    class _Reporter:
        def report(self, epoch, metrics):
            pass

        def should_prune(self):
            return False

    with pytest.raises(NotImplementedError, match="multi-GPU"):
        Trainer(
            model=_tiny_model(["ctrl", "gene_a"]),
            config=_tiny_config(tmp_path, multi_gpu=True),
            endpoint=_tiny_endpoint(pids),
            supported_pids=pids,
            output_dir=str(tmp_path),
            ema_decay=0.0,
            reporter=_Reporter(),
        )
