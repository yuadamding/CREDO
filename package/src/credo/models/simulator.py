"""CounterfactualEngine: simulate under perturbation vs control embedding.

Also exports helper to initialise particles from an EndpointProblem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.core import EndpointProblem, FiniteMeasure, TrajectoryProblem
from .full_model import FullDynamicsModel
from .weighted_sde import WeightedParticleSimulator, ParticleRollout


def initialise_particles_from_measures(
    measures: Dict[str, FiniteMeasure],
    perturbation_ids: List[str],
    n_particles: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample particles from a perturbation-keyed finite-measure dictionary.

    Returns
    -------
    z0: [G, N, d]
    logw0: [G, N]  relative log-weights, normalised to total 1
    log_m0: [G]
    """
    if seed is not None:
        torch.manual_seed(seed)

    G = len(perturbation_ids)
    d = next(iter(measures.values())).latent_dim
    z0 = torch.zeros(G, n_particles, d, dtype=dtype, device=device)
    logw0 = torch.zeros(G, n_particles, dtype=dtype, device=device)
    log_m0 = torch.zeros(G, dtype=dtype, device=device)

    for g, pid in enumerate(perturbation_ids):
        mu: FiniteMeasure = measures[pid]
        support = torch.tensor(mu.support, dtype=dtype, device=device)  # [n_atoms, d]
        probs = torch.tensor(mu.normalized_weights, dtype=dtype, device=device)
        idx = torch.multinomial(probs, n_particles, replacement=True)
        z0[g] = support[idx]
        total_mass = mu.total_mass
        logw0[g] = torch.full((n_particles,), -np.log(n_particles), dtype=dtype, device=device)
        log_m0[g] = torch.tensor(np.log(total_mass), dtype=dtype, device=device)

    return z0, logw0, log_m0


def initialise_particles(
    endpoint: EndpointProblem,
    perturbation_ids: List[str],
    n_particles: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample initial particles from the endpoint initial measure.

    This preserves the original endpoint code path exactly: cells are sampled
    uniformly from the support with ``torch.randint``.  Trajectory-specific
    helpers may use finite-measure weights, but legacy P4/P60 training keeps
    identical seeded starts.
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
        idx = torch.randint(0, n_atoms, (n_particles,), device=device)
        z0[g] = support[idx]
        total_mass = mu.total_mass
        logw0[g] = torch.full((n_particles,), -np.log(n_particles), dtype=dtype, device=device)
        log_m0[g] = torch.tensor(np.log(total_mass), dtype=dtype, device=device)

    return z0, logw0, log_m0


def initialise_particles_from_trajectory(
    trajectory: TrajectoryProblem,
    source_label: str,
    perturbation_ids: List[str],
    n_particles: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample particles from a pooled TrajectoryProblem checkpoint."""
    measures = trajectory.measures[source_label]
    if not all(isinstance(key, str) for key in measures):
        raise ValueError("initialise_particles_from_trajectory expects pooled perturbation-id keys")
    return initialise_particles_from_measures(
        {str(key): value for key, value in measures.items()},
        perturbation_ids,
        n_particles,
        device=device,
        dtype=dtype,
        seed=seed,
    )


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
        control_rollout_mode: str = "reference_consistent",
    ) -> List[CounterfactualResult]:
        """Run counterfactual simulations.

        Parameters
        ----------
        endpoint: provides P4 initial conditions
        perturbation_ids: which perturbations to analyse
        clamp_context: if True, also run with context fixed to control trajectory
        control_rollout_mode:
            - ``reference_consistent``: for ``soft_ref``, keep the shared
              reference embedding and set only the perturbation residual to zero
            - ``zero_centered``: force the full effective embedding to zero as a
              diagnostic rollout
        """
        results = []

        if control_rollout_mode not in {"reference_consistent", "zero_centered"}:
            raise ValueError(
                "control_rollout_mode must be 'reference_consistent' or 'zero_centered'."
            )

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

            # Reference-consistent soft-ref semantics keep a_ref and zero only
            # the perturbation residual; full zeroing is left as a diagnostic.
            with _control_embedding_context(self.model, pid, mode=control_rollout_mode):
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


class _control_embedding_context:
    """Temporarily patch one perturbation's control embedding semantics.

    For ``soft_ref``:
    - ``reference_consistent`` keeps the shared reference embedding and zeros
      only the perturbation-specific residual
    - ``zero_centered`` forces the full effective embedding to zero as an
      optional diagnostic
    """

    def __init__(self, model: FullDynamicsModel, pid: str, mode: str = "reference_consistent") -> None:
        self.model = model
        self.pid = pid
        self.mode = mode
        self._saved_embedding = None
        self._saved_reference = None

    def __enter__(self) -> None:
        emb = self.model.embedding
        if self.pid in emb._nc_to_local and emb.embeddings is not None:
            local_idx = emb._nc_to_local[self.pid]
            self._saved_embedding = emb.embeddings[local_idx].clone()
            with torch.no_grad():
                if emb.reference_embedding is not None and self.mode == "zero_centered":
                    emb.embeddings[local_idx].copy_(-emb.reference_embedding.detach())
                else:
                    emb.embeddings[local_idx].zero_()
        elif (
            self.mode == "zero_centered"
            and self.pid in emb.all_control_ids
            and emb.reference_embedding is not None
        ):
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


class _zero_embedding_context(_control_embedding_context):
    """Backward-compatible zero-centered diagnostic embedding context."""

    def __init__(self, model: FullDynamicsModel, pid: str) -> None:
        super().__init__(model, pid, mode="zero_centered")
