from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from credo.config.schema import RunConfig
from credo.data.core import (
    CellStateTable,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    SparseTrajectoryProblem,
    TimeAxis,
)
from credo.data.trajectory_view import TrajectoryView, embedding_id_for_measure_key
from credo.models.full_model import FullDynamicsModel
from credo.models.trajectory_counterfactual import TrajectoryCounterfactualEngine, _weighted_energy_distance
from credo.models.weighted_sde import WeightedParticleSimulator
from credo.training.trajectory_batch import initialise_particles_from_trajectory
from credo.training.trajectory_trainer import TrajectoryTrainer


pytestmark = pytest.mark.integration


def _toy_study() -> PerturbSeqDynamicsData:
    labels = ["90m", "6h", "10h"]
    rows = []
    latent = []
    rng = np.random.default_rng(3)
    for sample in ["D1", "D2"]:
        for pid, is_ctrl in [("ctrl", True), ("LPS__mono", False)]:
            for time_i, label in enumerate(labels):
                if sample == "D2" and pid == "LPS__mono" and label == "6h":
                    continue
                for cell_i in range(3):
                    rows.append(
                        {
                            "cell_id": f"{sample}_{pid}_{label}_{cell_i}",
                            "perturbation_id": pid,
                            "time_label": label,
                            "sample_id": sample,
                            "is_control": is_ctrl,
                        }
                    )
                    shift = 0.0 if is_ctrl else float(time_i)
                    latent.append(rng.normal(loc=shift, scale=0.05, size=2))
    cell_df = pd.DataFrame(rows)
    mass_df = (
        cell_df.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
        .size()
        .astype(float)
        .rename("mass")
        .reset_index()
    )
    return PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=[1.5, 6.0, 10.0]),
        catalog=PerturbationCatalog(["ctrl", "LPS__mono"], ["ctrl"]),
        cell_state=CellStateTable(cell_df[["cell_id", "perturbation_id", "time_label", "sample_id"]], np.asarray(latent, dtype=np.float32)),
        mass_table=MassTable(mass_df),
    )


def _tiny_config(tmp_path) -> RunConfig:
    cfg = RunConfig(output_dir=str(tmp_path), device="cpu")
    cfg.latent.dim = 2
    cfg.model.embedding_dim = 2
    cfg.model.n_programs = 2
    cfg.model.mediator_dim = 2
    cfg.model.hidden_dim = 16
    cfg.model.depth = 1
    cfg.model.ecological_growth = False
    cfg.model.control_mode = "soft_ref"
    cfg.simulation.n_particles = 4
    cfg.training.epochs = 1
    cfg.training.seed = 11
    cfg.training.lambda_weak = 0.0
    cfg.training.lambda_count = 0.0
    cfg.training.lambda_reg_net = 0.0
    cfg.training.lambda_reg_diffusion = 0.0
    cfg.training.lambda_reg_embed = 0.0
    cfg.training.lambda_reg_growth_bias = 0.0
    cfg.training.sinkhorn_max_iter = 8
    cfg.training.sinkhorn_epsilon = 0.2
    cfg.training.log_every = 1
    cfg.trajectory_training.source_label = "90m"
    cfg.trajectory_training.target_labels = ["6h", "10h"]
    cfg.trajectory_training.steps_per_interval = 1
    cfg.trajectory_training.endpoint_time_weights = {"6h": 0.5, "10h": 1.0}
    cfg.trajectory_training.normalize_time_weights = True
    cfg.trajectory_training.key_mode = "sample_aware"
    cfg.trajectory_training.sparse_missing = "mask"
    return cfg


def _model() -> FullDynamicsModel:
    return FullDynamicsModel(
        perturbation_ids=["ctrl", "LPS__mono"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=16,
        depth=1,
        ecological_growth=False,
        control_ref_penalty=0.0,
    )


def test_trajectory_view_sparse_keys_and_embedding_mapping() -> None:
    trajectory = _toy_study().to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    view = TrajectoryView(
        trajectory=trajectory,
        source_label="90m",
        target_labels=["6h", "10h"],
        sparse_missing="mask",
    )

    missing_6h_key = ("D2", "LPS__mono")
    assert missing_6h_key in view.source_keys
    assert missing_6h_key not in view.active_keys("6h")
    assert missing_6h_key in view.active_keys("10h")
    assert embedding_id_for_measure_key(missing_6h_key) == "LPS__mono"


def test_initialise_particles_from_trajectory_sample_aware_shapes() -> None:
    trajectory = _toy_study().to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    keys = [("D1", "ctrl"), ("D1", "LPS__mono")]
    z0, logw0, log_m0 = initialise_particles_from_trajectory(
        trajectory,
        "90m",
        keys,
        n_particles=5,
        device="cpu",
        dtype=torch.float32,
        seed=7,
    )

    assert z0.shape == (2, 5, 2)
    assert logw0.shape == (2, 5)
    assert log_m0.shape == (2,)
    assert torch.allclose(log_m0.exp(), torch.tensor([3.0, 3.0]))
    assert torch.allclose(logw0.exp().sum(dim=1), torch.ones(2))


def test_trajectory_trainer_one_epoch_full_start(tmp_path) -> None:
    study = _toy_study()
    trajectory = study.to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    cfg = _tiny_config(tmp_path)
    trainer = TrajectoryTrainer(
        model=_model(),
        config=cfg,
        trajectory=trajectory,
        source_label="90m",
        target_labels=["6h", "10h"],
        output_dir=str(tmp_path),
        ema_decay=0.0,
    )

    history = trainer.train()

    assert history.epochs == [1]
    assert (tmp_path / "checkpoint_last.pt").exists()
    assert (tmp_path / "trajectory_config.json").exists()
    pred = pd.read_csv(tmp_path / "predicted_metrics_by_key_time.csv")
    assert {"6h", "10h"}.issubset(set(pred["time_label"]))
    coverage = pd.read_csv(tmp_path / "target_coverage_by_time.csv")
    d2_mono_6h = coverage[
        (coverage["time_label"] == "6h")
        & (coverage["measure_key"] == "('D2', 'LPS__mono')")
    ]
    assert not d2_mono_6h.empty
    assert bool(d2_mono_6h["active"].iloc[0]) is False


def test_trajectory_trainer_rejects_validation_tau_mismatch(tmp_path) -> None:
    study = _toy_study()
    trajectory = study.to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    validation = SparseTrajectoryProblem(
        measures=trajectory.measures,
        catalog=trajectory.catalog,
        time_axis=TimeAxis(labels=["90m", "6h", "10h"], physical_times=[1.5, 6.5, 10.0]),
        time_labels=trajectory.time_labels,
    )
    cfg = _tiny_config(tmp_path)

    with pytest.raises(ValueError, match="tau mismatch"):
        TrajectoryTrainer(
            model=_model(),
            config=cfg,
            trajectory=trajectory,
            validation_trajectory=validation,
            source_label="90m",
            target_labels=["6h", "10h"],
            output_dir=str(tmp_path),
            ema_decay=0.0,
        )


def test_trajectory_trainer_evaluate_uses_eval_particles(tmp_path) -> None:
    study = _toy_study()
    trajectory = study.to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    cfg = _tiny_config(tmp_path)
    cfg.simulation.n_particles = 3
    cfg.eval.n_eval_particles = 7
    trainer = TrajectoryTrainer(
        model=_model(),
        config=cfg,
        trajectory=trajectory,
        source_label="90m",
        target_labels=["6h", "10h"],
        output_dir=str(tmp_path),
        ema_decay=0.0,
    )

    seen: list[tuple[int, bool]] = []
    original_rollout = trainer._rollout

    def wrapped_rollout(view, *, n_particles: int, seed: int, training: bool):
        seen.append((n_particles, training))
        return original_rollout(view, n_particles=n_particles, seed=seed, training=training)

    trainer._rollout = wrapped_rollout  # type: ignore[method-assign]
    trainer.evaluate(epoch=0)

    assert seen[-1] == (7, False)


def test_trajectory_trainer_records_validation_source(tmp_path) -> None:
    study = _toy_study()
    trajectory = study.to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    cfg = _tiny_config(tmp_path)

    # No validation trajectory -> self-eval on the training view, with a warning
    # that held-out validation (the configured default) is unavailable.
    trainer = TrajectoryTrainer(
        model=_model(),
        config=cfg,
        trajectory=trajectory,
        source_label="90m",
        target_labels=["6h", "10h"],
        output_dir=str(tmp_path),
        ema_decay=0.0,
    )
    with pytest.warns(RuntimeWarning, match="train_self_eval"):
        result = trainer.evaluate(epoch=0)
    assert result["metrics"]["validation_source"] == "train_self_eval"

    # With a held-out validation trajectory -> labeled held_out.
    trainer_val = TrajectoryTrainer(
        model=_model(),
        config=cfg,
        trajectory=trajectory,
        source_label="90m",
        target_labels=["6h", "10h"],
        validation_trajectory=trajectory,
        output_dir=str(tmp_path),
        ema_decay=0.0,
    )
    result_val = trainer_val.evaluate(epoch=0)
    assert result_val["metrics"]["validation_source"] == "held_out"


def test_trajectory_trainer_reporter_reports_and_prunes(tmp_path) -> None:
    study = _toy_study()
    trajectory = study.to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    cfg = _tiny_config(tmp_path)
    cfg.training.epochs = 5

    class _Reporter:
        def __init__(self) -> None:
            self.calls: list[tuple[int, dict]] = []

        def report(self, epoch, metrics):
            self.calls.append((epoch, dict(metrics)))

        def should_prune(self):
            return len(self.calls) >= 2  # prune after the second report

    reporter = _Reporter()
    trainer = TrajectoryTrainer(
        model=_model(),
        config=cfg,
        trajectory=trajectory,
        source_label="90m",
        target_labels=["6h", "10h"],
        output_dir=str(tmp_path),
        ema_decay=0.0,
        reporter=reporter,
    )
    trainer.train()

    # The reporter was invoked each epoch and pruning stopped training early.
    assert len(reporter.calls) == 2
    assert trainer._pruned_epoch == 2
    assert (tmp_path / "checkpoint_pruned.pt").exists()
    # The per-epoch mapping carries the diagnostics the optimizer needs.
    _, first_metrics = reporter.calls[0]
    assert "loss_end" in first_metrics


def test_trajectory_trainer_count_data_key_order_must_match(tmp_path) -> None:
    study = _toy_study()
    trajectory = study.to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    cfg = _tiny_config(tmp_path)
    cfg.training.lambda_count = 0.1
    count_data = {
        "key_level": "measure_key",
        "key_order": [("wrong", "ctrl")],
        "exposures": {},
        "count_matrices": {},
        "n_totals": {},
    }

    with pytest.raises(ValueError, match="key_order"):
        TrajectoryTrainer(
            model=_model(),
            config=cfg,
            trajectory=trajectory,
            source_label="90m",
            target_labels=["6h", "10h"],
            count_data=count_data,
            output_dir=str(tmp_path),
            ema_decay=0.0,
        )


def test_trajectory_counterfactual_same_start_same_noise() -> None:
    study = _toy_study()
    trajectory = study.to_sparse_trajectory_problem(
        by_sample=True,
        time_labels=["90m", "6h", "10h"],
    )
    simulator = WeightedParticleSimulator(n_steps=2, store_history=True)
    tau_grid = torch.tensor([0.0, (6.0 - 1.5) / (10.0 - 1.5), 1.0])
    engine = TrajectoryCounterfactualEngine(
        model=_model(),
        simulator=simulator,
        n_particles=4,
        device="cpu",
    )

    result = engine.run(
        trajectory,
        source_label="90m",
        target_labels=["6h", "10h"],
        measure_key=("D1", "LPS__mono"),
        tau_grid=tau_grid,
        common_noise=True,
        clamp_context=True,
        seed=17,
    )

    assert torch.allclose(result.factual.z_steps[0], result.reference.z_steps[0])
    assert torch.allclose(result.factual.logw_steps[0], result.reference.logw_steps[0])
    assert torch.allclose(result.factual.log_m0, result.reference.log_m0)
    assert result.factual_clamped is not None
    assert torch.allclose(result.factual_clamped.tau_steps, tau_grid)
    assert set(result.metrics_by_time["target_label"]) == {"6h", "10h"}
    assert "weighted_mean_shift_l2_fact_vs_ref" in result.metrics_by_time.columns
    assert "energy_distance_fact_vs_ref" in result.metrics_by_time.columns


def test_trajectory_energy_distance_detects_same_mean_distribution_shift() -> None:
    factual = torch.tensor([[-1.0], [1.0]])
    reference = torch.tensor([[0.0], [0.0]])
    logw = torch.zeros(2)
    mean_f = (torch.softmax(logw, dim=0)[:, None] * factual).sum(dim=0)
    mean_r = (torch.softmax(logw, dim=0)[:, None] * reference).sum(dim=0)

    assert torch.linalg.norm(mean_f - mean_r).item() == 0.0
    assert _weighted_energy_distance(factual, logw, reference, logw) > 0.1


def test_trajectory_energy_distance_chunked_matches_full_pairwise() -> None:
    z_a = torch.linspace(-1.0, 1.0, 9).reshape(9, 1)
    z_b = torch.linspace(-0.5, 1.5, 7).reshape(7, 1)
    logw_a = torch.linspace(-0.2, 0.2, 9)
    logw_b = torch.linspace(0.3, -0.1, 7)

    full = _weighted_energy_distance(z_a, logw_a, z_b, logw_b, chunk_size=100)
    chunked = _weighted_energy_distance(z_a, logw_a, z_b, logw_b, chunk_size=3)

    assert np.isclose(full, chunked)
