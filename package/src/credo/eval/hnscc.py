"""HNSCC-specific evaluation helpers for CREDO runs."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..data.core import EndpointProblem, FiniteMeasure
from ..losses.uot import sinkhorn_divergence
from ..models.full_model import FullDynamicsModel
from ..models.simulator import initialise_particles
from ..models.weighted_sde import WeightedParticleSimulator


def cap_measure_atoms(
    measure: FiniteMeasure,
    *,
    max_atoms: Optional[int],
    seed: int,
) -> FiniteMeasure:
    if max_atoms is None or measure.n_atoms <= max_atoms:
        return measure
    rng = np.random.default_rng(seed)
    idx = rng.choice(measure.n_atoms, size=max_atoms, replace=False)
    support = measure.support[idx]
    weights = np.full(max_atoms, measure.total_mass / max_atoms, dtype=np.float32)
    return FiniteMeasure(support=support, weights=weights, total_mass=measure.total_mass)


def cap_endpoint_problem_terminal(
    endpoint: EndpointProblem,
    *,
    max_terminal_atoms: Optional[int],
    seed: int,
) -> EndpointProblem:
    if max_terminal_atoms is None:
        return endpoint
    terminal = {}
    for i, pid in enumerate(endpoint.perturbation_ids):
        terminal[pid] = cap_measure_atoms(
            endpoint.terminal[pid],
            max_atoms=max_terminal_atoms,
            seed=seed + i,
        )
    return EndpointProblem(
        initial=endpoint.initial,
        terminal=terminal,
        time_axis=endpoint.time_axis,
        perturbation_ids=endpoint.perturbation_ids,
    )


@torch.no_grad()
def evaluate_endpoint_problem(
    model: FullDynamicsModel,
    endpoint: EndpointProblem,
    perturbation_ids: list[str],
    control_ids: set[str],
    *,
    device: str,
    n_particles: int,
    n_steps: int,
    target_particles: int,
    seed: int,
    eps: float,
    tau: float,
) -> pd.DataFrame:
    simulator = WeightedParticleSimulator(n_steps=n_steps, store_history=False)
    dtype = torch.float32
    model.eval()

    z0, logw0, log_m0 = initialise_particles(
        endpoint,
        perturbation_ids,
        n_particles=n_particles,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    rollout = simulator.rollout(
        z0,
        logw0,
        model,
        log_m0,
        perturbation_ids=perturbation_ids,
    )

    rows = []
    rng = np.random.default_rng(seed)
    for g, pid in enumerate(perturbation_ids):
        mu = endpoint.terminal[pid]
        if len(mu.support) > target_particles:
            idx = rng.choice(len(mu.support), size=target_particles, replace=False)
            target_support = mu.support[idx]
            target_weights = np.full(len(idx), mu.total_mass / len(idx), dtype=np.float32)
        else:
            target_support = mu.support
            target_weights = mu.weights

        y = torch.tensor(target_support, dtype=dtype, device=device)
        lb = torch.log(torch.tensor(target_weights, dtype=dtype, device=device) + 1e-30)

        la_abs = rollout.terminal_logw[g] + log_m0[g]
        div = sinkhorn_divergence(rollout.terminal_z[g], la_abs, y, lb, eps=eps, tau=tau)

        log_pred = log_m0[g] + torch.logsumexp(rollout.terminal_logw[g], dim=0)
        mass_pred = float(log_pred.exp().item())
        mass_true = float(mu.total_mass)
        mass_err = abs(mass_pred - mass_true) / mass_true if mass_true > 0 else 0.0

        rows.append(
            {
                "perturbation_id": pid,
                "uot": float(div.item()),
                "mass_pred": mass_pred,
                "mass_true": mass_true,
                "mass_rel_error": mass_err,
                "is_control": pid in control_ids,
                "n_init_atoms": int(endpoint.initial[pid].n_atoms),
                "n_term_atoms": int(endpoint.terminal[pid].n_atoms),
                "n_term_atoms_eval": int(len(target_support)),
            }
        )
    return pd.DataFrame(rows)


def summarize_eval(df: pd.DataFrame) -> dict:
    summary = {
        "n_perturbations": int(len(df)),
        "mean_uot": float(df["uot"].mean()),
        "median_uot": float(df["uot"].median()),
        "mean_mass_rel_error": float(df["mass_rel_error"].mean()),
        "median_mass_rel_error": float(df["mass_rel_error"].median()),
    }
    if "is_control" in df.columns and df["is_control"].any():
        ctrl = df[df["is_control"]]
        summary["n_controls"] = int(len(ctrl))
        summary["control_mean_uot"] = float(ctrl["uot"].mean())
        summary["control_mean_mass_rel_error"] = float(ctrl["mass_rel_error"].mean())
    non_ctrl = df[~df["is_control"]]
    if len(non_ctrl) > 0:
        summary["n_non_controls"] = int(len(non_ctrl))
        summary["non_control_mean_uot"] = float(non_ctrl["uot"].mean())
        summary["non_control_mean_mass_rel_error"] = float(non_ctrl["mass_rel_error"].mean())
    return summary


def build_true_terminal_state_table(
    obs: pd.DataFrame,
    *,
    perturbation_ids: list[str],
    target_time: float,
    state_key: str,
) -> pd.DataFrame:
    sub = obs.loc[pd.to_numeric(obs["Time point"], errors="coerce").eq(float(target_time))].copy()
    table = pd.crosstab(sub["perturbation_id"], sub[state_key]).astype(np.float32)
    table = table.reindex(perturbation_ids, fill_value=0.0)
    return table


def _state_distribution_from_assignments(
    assignments: np.ndarray,
    weights: np.ndarray,
    n_states: int,
) -> np.ndarray:
    dist = np.bincount(assignments, weights=weights, minlength=n_states).astype(np.float64)
    total = dist.sum()
    if total > 0:
        dist /= total
    return dist


@torch.no_grad()
def evaluate_state_compositions(
    model: FullDynamicsModel,
    endpoint: EndpointProblem,
    perturbation_ids: list[str],
    *,
    state_labels: list[str],
    state_centroids: np.ndarray,
    true_state_table: pd.DataFrame,
    device: str,
    n_particles: int,
    n_steps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulator = WeightedParticleSimulator(n_steps=n_steps, store_history=False)
    dtype = torch.float32
    model.eval()

    z0, logw0, log_m0 = initialise_particles(
        endpoint,
        perturbation_ids,
        n_particles=n_particles,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    rollout = simulator.rollout(
        z0,
        logw0,
        model,
        log_m0,
        perturbation_ids=perturbation_ids,
    )

    centers = torch.tensor(state_centroids, dtype=dtype, device=device)
    summary_rows = []
    dist_rows = []

    for g, pid in enumerate(perturbation_ids):
        pred_z = rollout.terminal_z[g]
        pred_weights = torch.softmax(rollout.terminal_logw[g], dim=0).detach().cpu().numpy()
        sq_dist = ((pred_z.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=-1)
        assignments = sq_dist.argmin(dim=1).detach().cpu().numpy()
        pred_dist = _state_distribution_from_assignments(assignments, pred_weights, len(state_labels))

        truth = true_state_table.loc[pid].reindex(state_labels, fill_value=0.0).to_numpy(dtype=np.float64)
        truth_total = truth.sum()
        if truth_total > 0:
            truth /= truth_total

        tv = 0.5 * np.abs(pred_dist - truth).sum()
        pred_idx = int(pred_dist.argmax())
        truth_idx = int(truth.argmax()) if truth_total > 0 else -1

        init_mass = float(endpoint.initial[pid].total_mass)
        true_mass = float(endpoint.terminal[pid].total_mass)
        pred_mass = float((log_m0[g] + torch.logsumexp(rollout.terminal_logw[g], dim=0)).exp().item())
        pred_expansion = pred_mass / init_mass if init_mass > 0 else np.nan
        true_expansion = true_mass / init_mass if init_mass > 0 else np.nan

        summary_rows.append(
            {
                "perturbation_id": pid,
                "state_tv": float(tv),
                "dominant_state_pred": state_labels[pred_idx],
                "dominant_state_true": state_labels[truth_idx] if truth_idx >= 0 else pd.NA,
                "dominant_state_match": bool(pred_idx == truth_idx) if truth_idx >= 0 else pd.NA,
                "pred_expansion_ratio": float(pred_expansion),
                "true_expansion_ratio": float(true_expansion),
                "expansion_ratio_gap": float(pred_expansion - true_expansion),
            }
        )
        for i, state in enumerate(state_labels):
            dist_rows.append(
                {
                    "perturbation_id": pid,
                    "state": state,
                    "pred_fraction": float(pred_dist[i]),
                    "true_fraction": float(truth[i]),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(dist_rows)


def summarize_state_metrics(df: pd.DataFrame) -> dict:
    match = df["dominant_state_match"].dropna()
    return {
        "n_perturbations": int(len(df)),
        "mean_state_tv": float(df["state_tv"].mean()),
        "median_state_tv": float(df["state_tv"].median()),
        "dominant_state_accuracy": float(match.mean()) if len(match) > 0 else None,
        "mean_abs_expansion_ratio_gap": float(df["expansion_ratio_gap"].abs().mean()),
        "median_abs_expansion_ratio_gap": float(df["expansion_ratio_gap"].abs().median()),
    }
