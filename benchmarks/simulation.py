"""Synthetic benchmark simulation suite.

Generates ground-truth datasets for three benchmark tasks:

1. SingleScreenDriftDiffusionBenchmark
   - Pure drift + diffusion, no growth
   - Used to verify state transport and UOT loss

2. SingleScreenGrowthBenchmark
   - Drift + diffusion + growth
   - Perturbations alter drift target, diffusion scale, or growth rate

3. MeanFieldEcologyBenchmark
   - Full system with ecological context coupling
   - Perturbations that thrive in different ecological niches

All benchmarks produce PerturbSeqDynamicsData objects with SimulationTruth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..data.core import (
    TimeAxis, PerturbationCatalog, CellStateTable, MassTable,
    FiniteMeasure, EndpointProblem, SimulationTruth, PerturbSeqDynamicsData,
    ExposureTable, ReplicateCountTable,
)


# ---------------------------------------------------------------------------
# Ground-truth simulator: Euler-Maruyama on synthetic dynamics
# ---------------------------------------------------------------------------

def _ou_step(
    z: np.ndarray,    # [G, N, d]
    kappa: np.ndarray,  # [G] mean-reversion
    theta: np.ndarray,  # [G, d] mean
    sigma: np.ndarray,  # [G, d] diagonal diffusion
    growth: np.ndarray,  # [G] scalar growth rate
    logw: np.ndarray,   # [G, N]
    dt: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """One OU+growth Euler-Maruyama step."""
    G, N, d = z.shape
    drift = kappa[:, None, None] * (theta[:, None, :] - z)  # [G, N, d]
    noise = rng.standard_normal(z.shape)
    z_new = z + drift * dt + sigma[:, None, :] * math.sqrt(dt) * noise
    logw_new = logw + growth[:, None] * dt
    return z_new, logw_new


def simulate_ground_truth(
    n_perturbations: int,
    n_controls: int,
    n_particles: int,
    latent_dim: int,
    n_steps: int,
    kappa: np.ndarray,   # [G]
    theta: np.ndarray,   # [G, d]  mean-reversion targets
    sigma: np.ndarray,   # [G, d]  diffusion
    growth: np.ndarray,  # [G]    growth rates
    initial_z: np.ndarray,  # [G, N, d]
    initial_mass: np.ndarray,  # [G]
    rng: np.random.Generator,
    store_paths: bool = False,
) -> Dict[str, Any]:
    """Run OU+growth ground-truth simulation.

    Returns dict with:
      - terminal_z: [G, N, d]
      - terminal_logw: [G, N]
      - terminal_mass: [G]
      - paths: [G, N, K+1, d] if store_paths else None
      - context_traj: None (no ecology in this simple version)
    """
    G = n_perturbations
    z = initial_z.copy()
    logw = np.zeros((G, n_particles))
    dt = 1.0 / n_steps

    paths = [z.copy()] if store_paths else None

    for _ in range(n_steps):
        z, logw = _ou_step(z, kappa, theta, sigma, growth, logw, dt, rng)
        if store_paths and paths is not None:
            paths.append(z.copy())

    # Compute terminal mass: M0 * mean(exp(logw))
    # log_mass = log(M0) + logsumexp(logw) - log(N)
    log_m0 = np.log(initial_mass)
    log_mass_terminal = log_m0 + np.array([
        np.log(np.sum(np.exp(logw[g] - logw[g].max()))) + logw[g].max() - np.log(n_particles)
        for g in range(G)
    ])
    terminal_mass = np.exp(log_mass_terminal)

    return {
        "terminal_z": z,
        "terminal_logw": logw,
        "terminal_mass": terminal_mass,
        "paths": np.stack(paths, axis=2) if store_paths and paths else None,
        "context_traj": None,
    }


# ---------------------------------------------------------------------------
# Benchmark 1: Drift / Diffusion / Reaction
# ---------------------------------------------------------------------------

@dataclass
class DriftDiffusionReactionConfig:
    """Configuration for the core drift-diffusion-reaction benchmark."""
    n_gene_perturbations: int = 10
    n_controls: int = 2
    latent_dim: int = 4
    n_particles_gt: int = 512     # ground-truth particles
    n_cells_per_group: int = 200  # subsampled cells in output
    n_steps_gt: int = 100
    noise_std: float = 0.05       # measurement noise on cells
    seed: int = 0

    # Ground-truth parameter ranges
    kappa_range: Tuple[float, float] = (0.5, 2.0)
    theta_range: Tuple[float, float] = (-2.0, 2.0)
    sigma_range: Tuple[float, float] = (0.1, 0.5)
    growth_range: Tuple[float, float] = (-0.5, 0.5)
    initial_mass: float = 1000.0


def build_drift_diffusion_reaction_benchmark(
    cfg: Optional[DriftDiffusionReactionConfig] = None,
) -> Tuple[PerturbSeqDynamicsData, Dict[str, Any]]:
    """Generate the core drift/diffusion/growth benchmark dataset.

    Returns
    -------
    data: PerturbSeqDynamicsData
    truth_params: dict of ground-truth parameters for evaluation
    """
    if cfg is None:
        cfg = DriftDiffusionReactionConfig()

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    G = cfg.n_gene_perturbations + cfg.n_controls
    d = cfg.latent_dim
    N = cfg.n_particles_gt
    K = cfg.n_steps_gt

    # --- Control and perturbation ids ---
    ctrl_ids = [f"ctrl_{i}" for i in range(cfg.n_controls)]
    gene_ids = [f"gene_{i}" for i in range(cfg.n_gene_perturbations)]
    all_ids = ctrl_ids + gene_ids

    # --- Ground-truth parameters ---
    # Controls: moderate, centered
    kappa_ctrl = rng.uniform(0.8, 1.2, size=cfg.n_controls)
    theta_ctrl = np.zeros((cfg.n_controls, d))
    sigma_ctrl = rng.uniform(0.15, 0.25, size=(cfg.n_controls, d))
    growth_ctrl = np.zeros(cfg.n_controls)

    # Gene perturbations: vary drift target, diffusion, growth
    kappa_gene = rng.uniform(*cfg.kappa_range, size=cfg.n_gene_perturbations)
    theta_gene = rng.uniform(*cfg.theta_range, size=(cfg.n_gene_perturbations, d))
    sigma_gene = rng.uniform(*cfg.sigma_range, size=(cfg.n_gene_perturbations, d))
    growth_gene = rng.uniform(*cfg.growth_range, size=cfg.n_gene_perturbations)

    kappa = np.concatenate([kappa_ctrl, kappa_gene])
    theta = np.vstack([theta_ctrl, theta_gene])
    sigma = np.vstack([sigma_ctrl, sigma_gene])
    growth = np.concatenate([growth_ctrl, growth_gene])

    truth_params = {
        "kappa": kappa.tolist(),
        "theta": theta.tolist(),
        "sigma": sigma.tolist(),
        "growth": growth.tolist(),
        "perturbation_ids": all_ids,
        "control_ids": ctrl_ids,
    }

    # --- Initial conditions (P4) ---
    z0 = rng.standard_normal((G, N, d)) * 0.3  # near origin at P4
    init_mass = np.full(G, cfg.initial_mass)

    # --- Run ground-truth simulation ---
    gt = simulate_ground_truth(
        n_perturbations=G,
        n_controls=cfg.n_controls,
        n_particles=N,
        latent_dim=d,
        n_steps=K,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        growth=growth,
        initial_z=z0,
        initial_mass=init_mass,
        rng=rng,
        store_paths=True,
    )

    # --- Build CellStateTable (subsample particles as "cells") ---
    n_cells = cfg.n_cells_per_group
    cell_rows = []
    latent_list = []
    cell_id_counter = 0

    for label, z_data, mass_data in [
        ("P4", z0, init_mass),
        ("P60", gt["terminal_z"], gt["terminal_mass"]),
    ]:
        for g, pid in enumerate(all_ids):
            idx = rng.choice(N, size=n_cells, replace=True)
            z_cells = z_data[g][idx] + rng.standard_normal((n_cells, d)) * cfg.noise_std
            latent_list.append(z_cells)
            for i in range(n_cells):
                cell_rows.append({
                    "cell_id": f"cell_{cell_id_counter}",
                    "perturbation_id": pid,
                    "time_label": label,
                    "sample_id": "pooled",
                })
                cell_id_counter += 1

    cell_df = pd.DataFrame(cell_rows)
    latent_array = np.vstack(latent_list)
    cell_table = CellStateTable(df=cell_df, latent=latent_array)

    # --- MassTable ---
    mass_rows = []
    for label, mass_data in [("P4", init_mass), ("P60", gt["terminal_mass"])]:
        for g, pid in enumerate(all_ids):
            mass_rows.append({
                "perturbation_id": pid,
                "time_label": label,
                "sample_id": "pooled",
                "mass": float(mass_data[g]),
            })
    mass_table = MassTable(df=pd.DataFrame(mass_rows))

    # --- ExposureTable (uniform T0 exposure) ---
    exp_rows = [{"perturbation_id": pid, "library_batch": "batch0", "exposure": 1.0 / G}
                for pid in all_ids]
    exposure_table = ExposureTable(df=pd.DataFrame(exp_rows))

    # --- Replicate count table (single replicate) ---
    count_rows = []
    for label, mass_data in [("P4", init_mass), ("P60", gt["terminal_mass"])]:
        total_n = int(mass_data.sum())
        for g, pid in enumerate(all_ids):
            count_rows.append({
                "sample_id": "rep0",
                "time_label": label,
                "library_batch": "batch0",
                "perturbation_id": pid,
                "count": int(mass_data[g]),
                "n_total_sample": total_n,
            })
    rep_counts = ReplicateCountTable(df=pd.DataFrame(count_rows))

    # --- Assemble study object ---
    time_axis = TimeAxis(labels=["P4", "P60"], physical_times=[4.0, 60.0])
    catalog = PerturbationCatalog(perturbation_ids=all_ids, control_ids=ctrl_ids)

    sim_truth = SimulationTruth(
        truth_params=truth_params,
        hidden_paths=gt["paths"],
        simulator_config={"type": "DriftDiffusionReaction", **cfg.__dict__},
    )

    data = PerturbSeqDynamicsData(
        time_axis=time_axis,
        catalog=catalog,
        cell_state=cell_table,
        mass_table=mass_table,
        exposure_table=exposure_table,
        replicate_counts=rep_counts,
        truth=sim_truth,
    )

    return data, truth_params


# ---------------------------------------------------------------------------
# Benchmark 2: Mean-field ecology
# ---------------------------------------------------------------------------

@dataclass
class MeanFieldEcologyConfig:
    """Configuration for the mean-field ecology benchmark."""
    n_gene_perturbations: int = 8
    n_controls: int = 2
    latent_dim: int = 4
    n_programs: int = 4
    n_particles_gt: int = 512
    n_cells_per_group: int = 200
    n_steps_gt: int = 100
    noise_std: float = 0.05
    seed: int = 42
    ecology_strength: float = 0.5
    initial_mass: float = 1000.0


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def simulate_meanfield_ground_truth(
    G: int,
    N: int,
    d: int,
    K: int,
    n_programs: int,
    kappa: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    growth_base: np.ndarray,
    # Ecological payoff: eta_weights [G, K], payoff_matrix [K, K]
    eta_weights: np.ndarray,   # [G, d, K] -> program encoder weights
    payoff_matrix: np.ndarray,  # [K, K]
    initial_z: np.ndarray,
    initial_mass: np.ndarray,
    rng: np.random.Generator,
    ecology_strength: float = 0.5,
) -> Dict[str, Any]:
    """Mean-field simulation with ecological context coupling."""
    z = initial_z.copy()
    logw = np.zeros((G, N))
    dt = 1.0 / K
    log_m0 = np.log(initial_mass)

    context_traj = []

    for step in range(K):
        # Compute per-perturbation mass
        log_mass_g = log_m0 + np.array([
            np.log(np.sum(np.exp(logw[g] - logw[g].max()))) + logw[g].max() - np.log(N)
            for g in range(G)
        ])
        log_total = np.log(np.sum(np.exp(log_mass_g - log_mass_g.max()))) + log_mass_g.max()
        freq_g = np.exp(log_mass_g - log_total)  # [G]

        # Compute per-perturbation program averages
        # eta(z): [G, N, K]
        eta_z = np.zeros((G, N, n_programs))
        for g in range(G):
            raw = z[g] @ eta_weights[g]  # [N, K]
            eta_z[g] = _softmax(raw)

        # Normalised weights within perturbation
        w_norm = np.zeros((G, N))
        for g in range(G):
            lw_shifted = logw[g] - logw[g].max()
            w_norm[g] = np.exp(lw_shifted) / np.exp(lw_shifted).sum()

        # Population program composition q [K]
        eta_g_mean = np.einsum("gn, gnk -> gk", w_norm, eta_z)  # [G, K]
        q = np.einsum("g, gk -> k", freq_g, eta_g_mean)          # [K]

        context_traj.append(q.copy())

        # Ecological growth: Phi_{g,i} = eta(z_{g,i}) . P q
        Pq = payoff_matrix @ q  # [K]
        phi = np.einsum("gnk, k -> gn", eta_z, Pq)  # [G, N]

        # Update particles
        growth = growth_base[:, None] + ecology_strength * phi  # [G, N]
        drift = kappa[:, None, None] * (theta[:, None, :] - z)
        noise = rng.standard_normal(z.shape)
        z = z + drift * dt + sigma[:, None, :] * math.sqrt(dt) * noise
        logw = logw + growth * dt

    log_mass_term = log_m0 + np.array([
        np.log(np.sum(np.exp(logw[g] - logw[g].max()))) + logw[g].max() - np.log(N)
        for g in range(G)
    ])

    return {
        "terminal_z": z,
        "terminal_logw": logw,
        "terminal_mass": np.exp(log_mass_term),
        "context_traj": np.array(context_traj),  # [K, K_prog]
        "paths": None,
    }


def build_meanfield_ecology_benchmark(
    cfg: Optional[MeanFieldEcologyConfig] = None,
) -> Tuple[PerturbSeqDynamicsData, Dict[str, Any]]:
    """Generate the mean-field ecology benchmark dataset."""
    if cfg is None:
        cfg = MeanFieldEcologyConfig()

    rng = np.random.default_rng(cfg.seed)
    G = cfg.n_gene_perturbations + cfg.n_controls
    d = cfg.latent_dim
    K_prog = cfg.n_programs
    N = cfg.n_particles_gt

    ctrl_ids = [f"ctrl_{i}" for i in range(cfg.n_controls)]
    gene_ids = [f"gene_{i}" for i in range(cfg.n_gene_perturbations)]
    all_ids = ctrl_ids + gene_ids

    # Parameters
    kappa = rng.uniform(0.5, 1.5, size=G)
    theta = rng.uniform(-1.5, 1.5, size=(G, d))
    sigma = rng.uniform(0.1, 0.3, size=(G, d))
    growth_base = rng.uniform(-0.3, 0.3, size=G)

    # Ecological payoff
    payoff_matrix = rng.standard_normal((K_prog, K_prog)) * 0.5

    # Per-perturbation program encoder weights (random linear heads)
    eta_weights = rng.standard_normal((G, d, K_prog)) * 0.5  # [G, d, K]

    z0 = rng.standard_normal((G, N, d)) * 0.3
    init_mass = np.full(G, cfg.initial_mass)

    gt = simulate_meanfield_ground_truth(
        G=G, N=N, d=d, K=cfg.n_steps_gt,
        n_programs=K_prog,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        growth_base=growth_base,
        eta_weights=eta_weights,
        payoff_matrix=payoff_matrix,
        initial_z=z0,
        initial_mass=init_mass,
        rng=rng,
        ecology_strength=cfg.ecology_strength,
    )

    n_cells = cfg.n_cells_per_group
    cell_rows, latent_list = [], []
    cell_id_counter = 0

    for label, z_data in [("P4", z0), ("P60", gt["terminal_z"])]:
        for g, pid in enumerate(all_ids):
            idx = rng.choice(N, size=n_cells, replace=True)
            z_cells = z_data[g][idx] + rng.standard_normal((n_cells, d)) * cfg.noise_std
            latent_list.append(z_cells)
            for _ in range(n_cells):
                cell_rows.append({
                    "cell_id": f"cell_{cell_id_counter}",
                    "perturbation_id": pid,
                    "time_label": label,
                    "sample_id": "pooled",
                })
                cell_id_counter += 1

    cell_df = pd.DataFrame(cell_rows)
    cell_table = CellStateTable(df=cell_df, latent=np.vstack(latent_list))

    mass_rows = []
    for label, mass_data in [("P4", init_mass), ("P60", gt["terminal_mass"])]:
        for g, pid in enumerate(all_ids):
            mass_rows.append({
                "perturbation_id": pid, "time_label": label,
                "sample_id": "pooled", "mass": float(mass_data[g]),
            })
    mass_table = MassTable(df=pd.DataFrame(mass_rows))

    exp_rows = [{"perturbation_id": pid, "library_batch": "batch0", "exposure": 1.0 / G}
                for pid in all_ids]
    exposure_table = ExposureTable(df=pd.DataFrame(exp_rows))

    time_axis = TimeAxis(labels=["P4", "P60"], physical_times=[4.0, 60.0])
    catalog = PerturbationCatalog(perturbation_ids=all_ids, control_ids=ctrl_ids)

    truth_params = {
        "kappa": kappa.tolist(), "theta": theta.tolist(),
        "sigma": sigma.tolist(), "growth_base": growth_base.tolist(),
        "payoff_matrix": payoff_matrix.tolist(),
        "perturbation_ids": all_ids, "control_ids": ctrl_ids,
        "context_trajectory": gt["context_traj"].tolist() if gt["context_traj"] is not None else None,
    }

    sim_truth = SimulationTruth(
        truth_params=truth_params,
        context_trajectories=gt["context_traj"],
        simulator_config={"type": "MeanFieldEcology", **cfg.__dict__},
    )

    data = PerturbSeqDynamicsData(
        time_axis=time_axis,
        catalog=catalog,
        cell_state=cell_table,
        mass_table=mass_table,
        exposure_table=exposure_table,
        truth=sim_truth,
    )

    return data, truth_params
