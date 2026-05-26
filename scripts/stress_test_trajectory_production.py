#!/usr/bin/env python
"""Randomized production-layer stress checks for trajectory CREDO.

This complements ``stress_test_trajectory_core.py`` by exercising the newer
training/evaluation layer: sparse sample-aware keys, measure-key/embedding-id
separation, checkpoint endpoint diagnostics, one-epoch trainer smoke cases, and
time-indexed same-start counterfactuals.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "src"))

from credo.config.schema import RunConfig  # noqa: E402
from credo.data.core import (  # noqa: E402
    CellStateTable,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
)
from credo.data.trajectory_view import TrajectoryView, embedding_id_for_measure_key  # noqa: E402
from credo.losses.multitime import MultiTimeEndpointLoss, checkpoint_indices_for_taus, make_observed_tau_grid  # noqa: E402
from credo.models.full_model import FullDynamicsModel  # noqa: E402
from credo.models.trajectory_counterfactual import TrajectoryCounterfactualEngine  # noqa: E402
from credo.models.weighted_sde import ParticleRollout, WeightedParticleSimulator  # noqa: E402
from credo.training.trajectory_batch import initialise_particles_from_trajectory  # noqa: E402
from credo.training.trajectory_trainer import TrajectoryTrainer  # noqa: E402


class ConstantComponentLoss(nn.Module):
    """Cheap endpoint-loss stand-in for randomized bookkeeping checks."""

    def component_dict(self, pred_z, pred_logw_abs, target_support, target_logw, perturbation_ids):
        active = [key for key in perturbation_ids if key in target_support]
        loss = pred_z.new_tensor(float(len(active)))
        components = {
            key: {
                "geom": pred_z.new_tensor(0.25),
                "mass": pred_z.new_tensor(0.75),
                "total": pred_z.new_tensor(1.0),
            }
            for key in active
        }
        return loss, components


def make_random_study(seed: int, *, small: bool = False) -> PerturbSeqDynamicsData:
    rng = np.random.default_rng(seed)
    n_times = 3 if small else int(rng.integers(3, 6))
    n_samples = 2 if small else int(rng.integers(2, 5))
    n_nonctrl = 1 if small else int(rng.integers(1, 5))
    latent_dim = 2 if small else int(rng.integers(1, 6))
    labels = [f"t{idx}" for idx in range(n_times)]
    physical_times = np.cumsum(rng.uniform(0.4, 3.0, size=n_times)).tolist()
    pids = ["ctrl"] + [f"pert_{idx}" for idx in range(n_nonctrl)]
    samples = [f"D{idx}" for idx in range(n_samples)]

    rows: list[dict] = []
    latent: list[np.ndarray] = []
    mass_rows: list[dict] = []
    guaranteed_key = (samples[0], pids[1])
    for sample_i, sample_id in enumerate(samples):
        for pid_i, pid in enumerate(pids):
            for time_i, label in enumerate(labels):
                present = True
                if time_i > 0 and pid != "ctrl" and (sample_id, pid) != guaranteed_key:
                    present = bool(rng.random() > 0.25)
                if not present:
                    continue
                n_cells = int(rng.integers(2, 7 if small else 10))
                mass = float(rng.uniform(0.05, 4.0))
                mass_rows.append(
                    {
                        "perturbation_id": pid,
                        "time_label": label,
                        "sample_id": sample_id,
                        "mass": mass,
                    }
                )
                center = (0.0 if pid == "ctrl" else 0.5 * pid_i) + float(time_i) * 0.15
                center += sample_i * 0.03
                for cell_i in range(n_cells):
                    rows.append(
                        {
                            "cell_id": f"{seed}_{sample_id}_{pid}_{label}_{cell_i}",
                            "perturbation_id": pid,
                            "time_label": label,
                            "sample_id": sample_id,
                        }
                    )
                    latent.append(rng.normal(center, 0.1, size=latent_dim).astype(np.float32))

    return PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=physical_times),
        catalog=PerturbationCatalog(pids, ["ctrl"]),
        cell_state=CellStateTable(pd.DataFrame(rows), np.asarray(latent, dtype=np.float32)),
        mass_table=MassTable(pd.DataFrame(mass_rows)),
    )


def make_model(pids: list[str], latent_dim: int, seed: int) -> FullDynamicsModel:
    torch.manual_seed(seed)
    return FullDynamicsModel(
        perturbation_ids=pids,
        control_ids=["ctrl"],
        latent_dim=latent_dim,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=1,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        control_ref_penalty=0.0,
    )


def tiny_config(output_dir: str, seed: int, latent_dim: int) -> RunConfig:
    cfg = RunConfig(output_dir=output_dir, device="cpu")
    cfg.latent.dim = latent_dim
    cfg.model.embedding_dim = 2
    cfg.model.n_programs = 2
    cfg.model.mediator_dim = 1
    cfg.model.hidden_dim = 8
    cfg.model.depth = 1
    cfg.model.ecological_growth = False
    cfg.model.control_mode = "soft_ref"
    cfg.simulation.n_particles = 3
    cfg.training.epochs = 1
    cfg.training.seed = seed
    cfg.training.lambda_weak = 0.0
    cfg.training.lambda_count = 0.0
    cfg.training.lambda_reg_net = 0.0
    cfg.training.lambda_reg_diffusion = 0.0
    cfg.training.lambda_reg_embed = 0.0
    cfg.training.lambda_reg_growth_bias = 0.0
    cfg.training.sinkhorn_max_iter = 2
    cfg.training.sinkhorn_epsilon = 0.2
    cfg.training.log_every = 1
    cfg.trajectory_training.steps_per_interval = 1
    cfg.trajectory_training.normalize_time_weights = True
    cfg.trajectory_training.sparse_missing = "mask"
    cfg.trajectory_training.key_mode = "sample_aware"
    return cfg


def assert_finite_tensor(tensor: torch.Tensor, label: str, seed: int) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{label} has non-finite values for seed={seed}")


def production_case(seed: int) -> None:
    data = make_random_study(seed)
    trajectory = data.to_sparse_trajectory_problem(by_sample=True)
    source_label = trajectory.time_labels[0]
    target_labels = trajectory.time_labels[1:]
    view = TrajectoryView(
        trajectory=trajectory,
        source_label=source_label,
        target_labels=target_labels,
        sparse_missing="mask",
    )
    missing_any = any(
        len(set(view.source_keys) - set(view.active_keys(label))) > 0
        for label in target_labels
    )
    if missing_any:
        try:
            TrajectoryView(
                trajectory=trajectory,
                source_label=source_label,
                target_labels=target_labels,
                sparse_missing="error",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"sparse_missing='error' failed to reject missing keys seed={seed}")

    embedding_ids = [embedding_id_for_measure_key(key) for key in view.source_keys]
    if not set(embedding_ids).issubset(set(trajectory.perturbation_ids)):
        raise AssertionError(f"embedding ids not in trajectory perturbation ids seed={seed}")

    n_particles = 2 + seed % 6
    z0, logw0, log_m0 = initialise_particles_from_trajectory(
        trajectory,
        source_label,
        view.source_keys,
        n_particles=n_particles,
        seed=seed,
    )
    assert z0.shape[0] == len(view.source_keys)
    assert logw0.shape == (len(view.source_keys), n_particles)
    assert_finite_tensor(z0, "z0", seed)
    assert_finite_tensor(log_m0, "log_m0", seed)
    if not torch.allclose(logw0.exp().sum(dim=1), torch.ones(len(view.source_keys)), atol=1e-6):
        raise AssertionError(f"relative initial weights do not sum to 1 seed={seed}")
    for g, key in enumerate(view.source_keys):
        expected_mass = float(trajectory.get(source_label, key).total_mass)
        got_mass = float(log_m0[g].exp())
        if not np.isclose(got_mass, expected_mass, rtol=1e-5, atol=1e-6):
            raise AssertionError(f"log_m0 mismatch seed={seed}, key={key}: {got_mass} != {expected_mass}")

    tau_grid = make_observed_tau_grid(view.observed_taus, steps_per_interval=1 + seed % 3)
    checkpoint_indices = checkpoint_indices_for_taus(tau_grid, view.time_labels, view.observed_taus)
    target_indices = {label: checkpoint_indices[label] for label in target_labels}
    target_support, target_logw = view.target_tensors(device="cpu", dtype=torch.float32)
    repeated_z = z0.unsqueeze(0).expand(len(tau_grid), -1, -1, -1).contiguous()
    repeated_logw = logw0.unsqueeze(0).expand(len(tau_grid), -1, -1).contiguous()
    rollout = ParticleRollout(
        z_steps=repeated_z,
        logw_steps=repeated_logw,
        tau_steps=tau_grid,
        log_m0=log_m0,
    )
    time_weights = {label: float(1 + idx) for idx, label in enumerate(target_labels)}
    loss_fn = MultiTimeEndpointLoss(
        ConstantComponentLoss(),
        time_weights=time_weights,
        reduction="mean",
        normalize_time_weights=True,
    )
    loss, logs = loss_fn(
        rollout,
        checkpoint_indices=target_indices,
        target_support_by_time=target_support,
        target_logw_by_time=target_logw,
        prediction_keys=view.source_keys,
        embedding_ids=embedding_ids,
    )
    if not torch.allclose(loss, torch.tensor(1.0), atol=1e-6):
        raise AssertionError(f"normalized constant endpoint loss mismatch seed={seed}: {loss}")
    for label in target_labels:
        expected_active = len(view.active_keys(label))
        expected_missing = len(view.source_keys) - expected_active
        if int(logs[f"endpoint/{label}/n_active_keys"]) != expected_active:
            raise AssertionError(f"active key log mismatch seed={seed}, label={label}")
        if int(logs[f"endpoint/{label}/n_missing_keys"]) != expected_missing:
            raise AssertionError(f"missing key log mismatch seed={seed}, label={label}")


def trainer_case(seed: int) -> None:
    data = make_random_study(seed, small=True)
    trajectory = data.to_sparse_trajectory_problem(by_sample=True)
    source_label = trajectory.time_labels[0]
    target_labels = trajectory.time_labels[1:]
    with TemporaryDirectory(prefix="credo_traj_stress_") as tmp:
        cfg = tiny_config(tmp, seed=seed, latent_dim=data.latent_dim)
        cfg.trajectory_training.source_label = source_label
        cfg.trajectory_training.target_labels = target_labels
        cfg.trajectory_training.endpoint_time_weights = {target_labels[-1]: 1.0}
        model = make_model(trajectory.perturbation_ids, data.latent_dim, seed=10_000 + seed)
        trainer = TrajectoryTrainer(
            model=model,
            config=cfg,
            trajectory=trajectory,
            source_label=source_label,
            target_labels=target_labels,
            output_dir=tmp,
            ema_decay=0.0,
        )
        history = trainer.train()
        if history.epochs != [1]:
            raise AssertionError(f"trainer did not record one epoch seed={seed}")
        hist = history.to_dataframe()
        if not np.isfinite(hist["loss_total"].to_numpy()).all():
            raise AssertionError(f"trainer produced non-finite loss seed={seed}")
        for filename in ["checkpoint_last.pt", "trajectory_config.json", "predicted_metrics_by_key_time.csv"]:
            if not (Path(tmp) / filename).exists():
                raise AssertionError(f"missing trainer artifact {filename} seed={seed}")


def counterfactual_case(seed: int) -> None:
    data = make_random_study(seed, small=True)
    trajectory = data.to_sparse_trajectory_problem(by_sample=True)
    source_label = trajectory.time_labels[0]
    target_labels = trajectory.time_labels[1:]
    measure_key = ("D0", "pert_0")
    if measure_key not in trajectory.measures[source_label]:
        return
    tau_grid = make_observed_tau_grid(trajectory.observed_taus, steps_per_interval=1)
    simulator = WeightedParticleSimulator(n_steps=len(tau_grid) - 1, store_history=True)
    model = make_model(trajectory.perturbation_ids, data.latent_dim, seed=20_000 + seed)
    engine = TrajectoryCounterfactualEngine(
        model=model,
        simulator=simulator,
        n_particles=3,
        device="cpu",
    )
    result = engine.run(
        trajectory,
        source_label=source_label,
        target_labels=target_labels,
        measure_key=measure_key,
        tau_grid=tau_grid,
        common_noise=True,
        clamp_context=True,
        seed=seed,
    )
    if not torch.equal(result.factual.z_steps[0], result.reference.z_steps[0]):
        raise AssertionError(f"counterfactual z0 differs seed={seed}")
    if not torch.equal(result.factual.logw_steps[0], result.reference.logw_steps[0]):
        raise AssertionError(f"counterfactual logw0 differs seed={seed}")
    if not torch.equal(result.factual.log_m0, result.reference.log_m0):
        raise AssertionError(f"counterfactual log_m0 differs seed={seed}")
    if result.factual_clamped is None or not torch.equal(result.factual_clamped.tau_steps, tau_grid):
        raise AssertionError(f"clamped trajectory did not preserve tau grid seed={seed}")
    if set(result.metrics_by_time["target_label"]) != set(target_labels):
        raise AssertionError(f"counterfactual metrics missing target labels seed={seed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--trainer-cases", type=int, default=100)
    parser.add_argument("--counterfactual-cases", type=int, default=300)
    args = parser.parse_args()

    for seed in range(args.cases):
        production_case(seed)
        if (seed + 1) % 100 == 0:
            print(f"production passed {seed + 1} cases", flush=True)

    for seed in range(args.counterfactual_cases):
        counterfactual_case(seed)
        if (seed + 1) % 50 == 0:
            print(f"counterfactual passed {seed + 1} cases", flush=True)

    for seed in range(args.trainer_cases):
        trainer_case(seed)
        if (seed + 1) % 25 == 0:
            print(f"trainer passed {seed + 1} cases", flush=True)

    print(
        "all production-layer stress checks passed: "
        f"{args.cases} production, "
        f"{args.counterfactual_cases} counterfactual, "
        f"{args.trainer_cases} trainer"
    )


if __name__ == "__main__":
    main()
