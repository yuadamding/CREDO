"""CounterfactualEngine: simulate under perturbation vs control embedding.

Also exports helper to initialise particles from an EndpointProblem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.core import EndpointProblem, FiniteMeasure
from .full_model import FullDynamicsModel
from .weighted_sde import WeightedParticleSimulator, ParticleRollout


def initialise_particles(
    endpoint: EndpointProblem,
    perturbation_ids: List[str],
    n_particles: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample initial particles from the P4 support.

    Returns
    -------
    z0: [G, N, d]
    logw0: [G, N]  (uniform, sum = log(total_mass))
    log_m0: [G]
    """
    if seed is not None:
        torch.manual_seed(seed)

    G = len(perturbation_ids)
    d = next(iter(endpoint.initial.values())).latent_dim
    z0 = torch.zeros(G, n_particles, d, dtype=dtype, device=device)
    logw0 = torch.zeros(G, n_particles, dtype=dtype, device=device)
    log_m0 = torch.zeros(G, dtype=dtype, device=device)

    for g, pid in enumerate(perturbation_ids):
        mu: FiniteMeasure = endpoint.initial[pid]
        support = torch.tensor(mu.support, dtype=dtype, device=device)  # [n_atoms, d]
        n_atoms = len(support)
        # Sample with replacement
        idx = torch.randint(0, n_atoms, (n_particles,), device=device)
        z0[g] = support[idx]
        total_mass = mu.total_mass
        logw0[g] = torch.full((n_particles,), -np.log(n_particles), dtype=dtype, device=device)
        log_m0[g] = torch.tensor(np.log(total_mass), dtype=dtype, device=device)

    return z0, logw0, log_m0


@dataclass
class CounterfactualResult:
    """Paired simulation outputs for a single perturbation."""
    perturbation_id: str
    rollout_perturb: ParticleRollout
    rollout_control: ParticleRollout
    rollout_clamped: Optional[ParticleRollout] = None  # context clamped to control

    def terminal_mass_diff(self) -> float:
        logw_p = self.rollout_perturb.terminal_logw.squeeze(0)
        logw_c = self.rollout_control.terminal_logw.squeeze(0)
        if self.rollout_perturb.log_m0 is None or self.rollout_control.log_m0 is None:
            raise ValueError("Counterfactual mass comparison requires rollout.log_m0.")
        log_mass_p = self.rollout_perturb.log_m0.squeeze(0) + torch.logsumexp(logw_p, 0)
        log_mass_c = self.rollout_control.log_m0.squeeze(0) + torch.logsumexp(logw_c, 0)
        mass_p = float(log_mass_p.exp().item())
        mass_c = float(log_mass_c.exp().item())
        return mass_p - mass_c

    def terminal_mean_diff(self) -> torch.Tensor:
        def _mean(rollout: ParticleRollout) -> torch.Tensor:
            z = rollout.terminal_z.squeeze(0)     # [N, d]
            logw = rollout.terminal_logw.squeeze(0)  # [N]
            w = torch.softmax(logw, 0)
            return (w.unsqueeze(-1) * z).sum(0)
        return _mean(self.rollout_perturb) - _mean(self.rollout_control)


class CounterfactualEngine:
    """Simulate under perturbation vs control for each requested perturbation.

    Parameters
    ----------
    model: trained FullDynamicsModel
    simulator: WeightedParticleSimulator
    n_particles: number of particles per simulation
    device: torch device string
    """

    def __init__(
        self,
        model: FullDynamicsModel,
        simulator: WeightedParticleSimulator,
        n_particles: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.simulator = simulator
        self.n_particles = n_particles
        self.device = device

    @torch.no_grad()
    def run(
        self,
        endpoint: EndpointProblem,
        perturbation_ids: List[str],
        clamp_context: bool = False,
        seed: int = 0,
    ) -> List[CounterfactualResult]:
        """Run counterfactual simulations.

        Parameters
        ----------
        endpoint: provides P4 initial conditions
        perturbation_ids: which perturbations to analyse
        clamp_context: if True, also run with context fixed to control trajectory
        """
        results = []

        for pid in perturbation_ids:
            if pid not in endpoint.initial:
                continue

            # --- Perturbation rollout ---
            z0p, lw0p, lm0p = initialise_particles(
                endpoint, [pid], self.n_particles, self.device, seed=seed)
            rollout_p = self.simulator.rollout(
                z0=z0p,
                logw0=lw0p,
                model=self.model,
                log_m0=lm0p,
                perturbation_ids=[pid],
            )

            # --- Control rollout with the same perturbation-specific initial measure ---
            z0c, lw0c, lm0c = z0p.clone(), lw0p.clone(), lm0p.clone()

            # Temporarily patch embedding to zero for control rollout
            with _zero_embedding_context(self.model, pid):
                rollout_c = self.simulator.rollout(
                    z0=z0c,
                    logw0=lw0c,
                    model=self.model,
                    log_m0=lm0c,
                    perturbation_ids=[pid],
                )

            result = CounterfactualResult(
                perturbation_id=pid,
                rollout_perturb=rollout_p,
                rollout_control=rollout_c,
            )
            results.append(result)

        return results


class _zero_embedding_context:
    """Context manager that temporarily makes one perturbation's effective embedding zero."""

    def __init__(self, model: FullDynamicsModel, pid: str) -> None:
        self.model = model
        self.pid = pid
        self._saved_embedding = None
        self._saved_reference = None

    def __enter__(self) -> None:
        emb = self.model.embedding
        if self.pid in emb._nc_to_local and emb.embeddings is not None:
            local_idx = emb._nc_to_local[self.pid]
            self._saved_embedding = emb.embeddings[local_idx].clone()
            with torch.no_grad():
                if emb.reference_embedding is not None:
                    emb.embeddings[local_idx].copy_(-emb.reference_embedding.detach())
                else:
                    emb.embeddings[local_idx].zero_()
        elif self.pid in emb.all_control_ids and emb.reference_embedding is not None:
            self._saved_reference = emb.reference_embedding.clone()
            with torch.no_grad():
                emb.reference_embedding.zero_()

    def __exit__(self, *args: object) -> None:
        emb = self.model.embedding
        if self._saved_embedding is not None and self.pid in emb._nc_to_local:
            local_idx = emb._nc_to_local[self.pid]
            with torch.no_grad():
                emb.embeddings[local_idx].copy_(self._saved_embedding)
        if self._saved_reference is not None and emb.reference_embedding is not None:
            with torch.no_grad():
                emb.reference_embedding.copy_(self._saved_reference)
