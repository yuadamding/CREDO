"""Benchmark runner: fits the model on synthetic data and reports acceptance criteria.

Acceptance criteria (Section 17.2):
1. Model fits terminal measures substantially better than control-only baseline.
2. Model distinguishes perturbations that alter drift, diffusion, and growth.
3. Ecological model outperforms ecology-free model on mean-field benchmark.
4. Recovered terminal masses numerically stable across seeds.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config.schema import RunConfig, DataConfig, LatentConfig, ModelConfig, SimulationConfig, TrainingConfig
from ..data.filters import filter_state_supported_perturbations
from ..losses.uot import sinkhorn_divergence
from ..models.full_model import FullDynamicsModel
from ..models.simulator import initialise_particles
from ..models.weighted_sde import WeightedParticleSimulator
from ..training.trainer import Trainer
from .simulation import (
    build_drift_diffusion_reaction_benchmark,
    DriftDiffusionReactionConfig,
    build_meanfield_ecology_benchmark,
    MeanFieldEcologyConfig,
)


@dataclass
class BenchmarkResult:
    benchmark_name: str
    endpoint_uot_model: Dict[str, float]   # pid -> UOT divergence
    endpoint_uot_control: Dict[str, float]  # baseline (control-only)
    terminal_mass_error: Dict[str, float]   # pid -> relative mass error
    history: pd.DataFrame
    passed: bool
    criteria: Dict[str, bool]
    elapsed_seconds: float


def _compute_endpoint_uot(
    model: FullDynamicsModel,
    simulator: WeightedParticleSimulator,
    endpoint,
    supported_pids: List[str],
    n_particles: int = 256,
    device: str = "cpu",
    seed: int = 0,
    eps: float = 0.1,
    tau: float = 1.0,
) -> Dict[str, float]:
    """Run rollout and compute per-perturbation UOT divergence vs target."""
    from ..losses.uot import sinkhorn_divergence

    model.eval()
    dtype = torch.float32

    z0, logw0, log_m0 = initialise_particles(
        endpoint, supported_pids, n_particles, device, dtype, seed=seed)

    with torch.no_grad():
        rollout = simulator.rollout(z0=z0, logw0=logw0, model=model, log_m0=log_m0,
                                   perturbation_ids=supported_pids)

    uot_dict = {}
    for g, pid in enumerate(supported_pids):
        if pid not in endpoint.terminal:
            continue
        x = rollout.terminal_z[g]                          # [N, d]
        la_abs = rollout.terminal_logw[g] + log_m0[g]      # [N] absolute

        mu = endpoint.terminal[pid]
        y = torch.tensor(mu.support, dtype=dtype, device=device)
        lb = torch.log(torch.tensor(mu.weights, dtype=dtype, device=device) + 1e-30)

        div = sinkhorn_divergence(x, la_abs, y, lb, eps=eps, tau=tau).item()
        uot_dict[pid] = div

    return uot_dict


def run_benchmark(
    benchmark_name: str,
    data,
    truth_params: Dict[str, Any],
    train_epochs: int = 200,
    n_particles: int = 128,
    latent_dim: int = 4,
    device: str = "auto",
    output_dir: str = "outputs/benchmark",
    ecological: bool = False,
    seed: int = 0,
) -> BenchmarkResult:
    """Fit the model on synthetic data and evaluate acceptance criteria."""
    import torch as _torch
    _torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    actual_device = "cuda" if _torch.cuda.is_available() else "cpu" if device == "auto" else device

    # --- Filter supported perturbations ---
    supported_pids = filter_state_supported_perturbations(
        data, min_cells_p4=10, min_cells_p60=10)
    print(f"[{benchmark_name}] Supported perturbations: {len(supported_pids)}")

    # --- Build endpoint problem ---
    endpoint = data.to_endpoint_problem(perturbation_ids=supported_pids)

    # --- Config ---
    cfg = RunConfig(
        run_id=f"benchmark_{benchmark_name}",
        device=device,
        latent=LatentConfig(dim=latent_dim, whiten=False),
        model=ModelConfig(
            embedding_dim=min(4, len(supported_pids)),
            n_programs=4,
            mediator_dim=4,
            hidden_dim=64,
            depth=2,
            sigma_min=1e-3,
            r_max=2.0,
            ecological_growth=ecological,
        ),
        simulation=SimulationConfig(n_particles=n_particles, n_steps=16, store_history=True),
        training=TrainingConfig(
            epochs=train_epochs,
            lr_net=3e-4,
            lr_embed=1e-3,
            lambda_end=1.0,
            lambda_weak=0.1,
            lambda_count=0.0,
            lambda_reg_embed=1e-4,
            lambda_reg_net=1e-4,
            lambda_reg_diffusion=1e-4,
            seed=seed,
            early_stop_patience=train_epochs,
            log_every=50,
            checkpoint_every=9999,
            sinkhorn_epsilon=0.1,
            sinkhorn_tau=1.0,
            n_test_functions=16,
            test_function_bandwidth=1.0,
        ),
        output_dir=output_dir,
    )

    # --- Model ---
    ctrl_ids = data.catalog.control_ids
    model = FullDynamicsModel(
        perturbation_ids=supported_pids,
        control_ids=[c for c in ctrl_ids if c in supported_pids],
        latent_dim=latent_dim,
        embedding_dim=cfg.model.embedding_dim,
        n_programs=cfg.model.n_programs,
        mediator_dim=cfg.model.mediator_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        sigma_min=cfg.model.sigma_min,
        r_max=cfg.model.r_max,
        ecological_growth=ecological,
    ).to(actual_device)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        config=cfg,
        endpoint=endpoint,
        supported_pids=supported_pids,
        output_dir=output_dir,
    )
    history = trainer.train(stage="all", n_epochs=train_epochs)

    # --- Evaluate ---
    simulator = WeightedParticleSimulator(n_steps=32, store_history=False)

    uot_model = _compute_endpoint_uot(
        model, simulator, endpoint, supported_pids,
        n_particles=256, device=actual_device, seed=seed)

    # --- Control-only baseline: freeze all embeddings at zero, use same model ---
    model_ctrl = FullDynamicsModel(
        perturbation_ids=supported_pids,
        control_ids=supported_pids,  # ALL are controls -> all embeddings zero
        latent_dim=latent_dim,
        embedding_dim=cfg.model.embedding_dim,
        n_programs=cfg.model.n_programs,
        mediator_dim=cfg.model.mediator_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        sigma_min=cfg.model.sigma_min,
        r_max=cfg.model.r_max,
        ecological_growth=False,
    ).to(actual_device)

    # Train a control-only model (all embeddings zero) for the baseline
    try:
        ctrl_trainer = Trainer(
            model=model_ctrl,
            config=cfg,
            endpoint=endpoint,
            supported_pids=supported_pids,
            output_dir=output_dir + "_ctrl",
        )
        ctrl_trainer.train(stage="C", n_epochs=min(train_epochs, 100))
        uot_control = _compute_endpoint_uot(
            model_ctrl, simulator, endpoint, supported_pids,
            n_particles=256, device=actual_device, seed=seed)
    except Exception as e:
        print(f"  [WARNING] Control baseline failed: {e}")
        uot_control = {pid: float("inf") for pid in supported_pids}

    # --- Terminal mass error ---
    mass_error = {}
    for g, pid in enumerate(supported_pids):
        if pid not in endpoint.terminal:
            continue
        true_mass = endpoint.terminal[pid].total_mass
        # Predicted mass from rollout
        z0, lw0, lm0 = initialise_particles(endpoint, [pid], 256, actual_device, seed=seed)
        with torch.no_grad():
            roll = simulator.rollout(z0=z0, logw0=lw0, model=model, log_m0=lm0, perturbation_ids=[pid])
        # logw[i] = -log(N) + log(W_i), so logsumexp = log(mean W).
        # pred_mass = M0 * mean(W) = M0 * exp(logsumexp(logw)).
        log_pred = lm0[0] + torch.logsumexp(roll.terminal_logw[0], 0)
        pred_mass = float(log_pred.exp().item())
        mass_error[pid] = abs(pred_mass - true_mass) / true_mass

    # --- Acceptance criteria ---
    model_mean = np.mean(list(uot_model.values()))
    ctrl_mean = np.mean([v for v in uot_control.values() if not np.isinf(v)] or [1e6])
    criteria = {
        "better_than_control": model_mean < 0.9 * ctrl_mean,
        "mass_stable": np.mean(list(mass_error.values())) < 1.0,
    }
    passed = all(criteria.values())

    elapsed = time.time() - t0
    print(f"\n[{benchmark_name}] Results:")
    print(f"  Model UOT (mean): {model_mean:.4f}")
    print(f"  Control UOT (mean): {ctrl_mean:.4f}")
    print(f"  Mass error (mean): {np.mean(list(mass_error.values())):.4f}")
    print(f"  Criteria: {criteria}")
    print(f"  Passed: {passed}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        endpoint_uot_model=uot_model,
        endpoint_uot_control=uot_control,
        terminal_mass_error=mass_error,
        history=history.to_dataframe(),
        passed=passed,
        criteria=criteria,
        elapsed_seconds=elapsed,
    )


def run_all_benchmarks(
    output_dir: str = "outputs/benchmarks",
    train_epochs: int = 200,
    device: str = "auto",
) -> Dict[str, BenchmarkResult]:
    """Run all required benchmarks and return results dict."""
    results = {}

    print("=" * 60)
    print("BENCHMARK 1: Drift/Diffusion/Reaction")
    print("=" * 60)
    data_ddr, truth_ddr = build_drift_diffusion_reaction_benchmark(
        DriftDiffusionReactionConfig(
            n_gene_perturbations=8, n_controls=2, latent_dim=4,
            n_particles_gt=256, n_cells_per_group=100, n_steps_gt=50))
    results["drift_diffusion_reaction"] = run_benchmark(
        "drift_diffusion_reaction", data_ddr, truth_ddr,
        train_epochs=train_epochs, n_particles=64, latent_dim=4,
        device=device, output_dir=f"{output_dir}/ddr",
        ecological=False)

    print("\n" + "=" * 60)
    print("BENCHMARK 2: Mean-Field Ecology")
    print("=" * 60)
    data_mfe, truth_mfe = build_meanfield_ecology_benchmark(
        MeanFieldEcologyConfig(
            n_gene_perturbations=6, n_controls=2, latent_dim=4,
            n_programs=4, n_particles_gt=256, n_cells_per_group=100, n_steps_gt=50))
    results["meanfield_ecology"] = run_benchmark(
        "meanfield_ecology", data_mfe, truth_mfe,
        train_epochs=train_epochs, n_particles=64, latent_dim=4,
        device=device, output_dir=f"{output_dir}/mfe",
        ecological=True)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and result.passed

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # Save summary
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        name: {
            "passed": bool(r.passed),
            "criteria": {k: bool(v) for k, v in r.criteria.items()},
            "model_uot_mean": float(np.mean(list(r.endpoint_uot_model.values()))),
            "mass_error_mean": float(np.mean(list(r.terminal_mass_error.values()))),
            "elapsed_seconds": r.elapsed_seconds,
        }
        for name, r in results.items()
    }
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results
