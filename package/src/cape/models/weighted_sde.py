"""Weighted particle SDE simulator (Euler-Maruyama).

Implements the particle system:
    dZ_tau = v_g(Z, tau, c) dtau + Sigma_g(Z, tau, c) dW_tau
    d/dtau log w_tau = r_g(Z, tau, c)

All mass computations use absolute log-weights and log-space reductions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .full_model import FullDynamicsModel


@dataclass
class ParticleRollout:
    """Result of a WeightedParticleSimulator.rollout() call."""
    z_steps: torch.Tensor       # [K+1, G, N, d]
    logw_steps: torch.Tensor    # [K+1, G, N]
    tau_steps: torch.Tensor     # [K+1]
    log_m0: Optional[torch.Tensor] = None   # [G] initial log-mass per perturbation
    # Optional cached values
    drift_steps: Optional[torch.Tensor] = None      # [K, G, N, d]
    sigma_steps: Optional[torch.Tensor] = None      # [K, G, N, d]
    growth_steps: Optional[torch.Tensor] = None     # [K, G, N]
    context_steps: Optional[torch.Tensor] = None    # [K+1, C]

    @property
    def n_steps(self) -> int:
        return self.z_steps.shape[0] - 1

    @property
    def G(self) -> int:
        return self.z_steps.shape[1]

    @property
    def N(self) -> int:
        return self.z_steps.shape[2]

    @property
    def terminal_z(self) -> torch.Tensor:
        return self.z_steps[-1]   # [G, N, d]

    @property
    def terminal_logw(self) -> torch.Tensor:
        return self.logw_steps[-1]  # [G, N]


class WeightedParticleSimulator(nn.Module):
    """Euler-Maruyama rollout for the weighted-particle representation.

    Parameters
    ----------
    n_steps: number of Euler-Maruyama steps
    store_history: whether to keep drift/sigma/growth at every step
    """

    def __init__(
        self,
        n_steps: int = 24,
        store_history: bool = False,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.store_history = store_history

    def rollout(
        self,
        z0: torch.Tensor,        # [G, N, d]
        logw0: torch.Tensor,     # [G, N]
        model: "FullDynamicsModel",
        log_m0: torch.Tensor,    # [G]  log initial total mass
        tau_start: float = 0.0,
        tau_end: float = 1.0,
        perturbation_ids: Optional[List[str]] = None,
    ) -> ParticleRollout:
        """Run the full Euler-Maruyama rollout.

        Parameters
        ----------
        z0: initial particle positions [G, N, d]
        logw0: initial log-weights [G, N]; should satisfy exp(logw0).sum(1) = 1
        model: FullDynamicsModel with .step() method
        log_m0: log of initial total mass per perturbation [G]
        tau_start, tau_end: time interval in normalized coordinates
        """
        device = z0.device
        dtype = z0.dtype
        G, N, d = z0.shape
        K = self.n_steps
        dtau = (tau_end - tau_start) / K

        tau_steps = torch.linspace(tau_start, tau_end, K + 1, device=device, dtype=dtype)

        # Storage
        z_list = [z0]
        logw_list = [logw0]
        drift_list, sigma_list, growth_list, ctx_list = [], [], [], []

        z = z0.clone()
        logw = logw0.clone()

        for k in range(K):
            tau_k = tau_steps[k]

            # Get coefficients and context from model
            coeffs, ctx = model.step(
                z=z,
                tau=tau_k,
                logw=logw,
                log_m0=log_m0,
                perturbation_ids=perturbation_ids,
            )

            v = coeffs.drift      # [G, N, d]
            sigma = coeffs.sigma_diag  # [G, N, d]
            r = coeffs.growth     # [G, N]

            if self.store_history:
                # Keep rollout coefficients attached so weak-form and rollout
                # regularizers can backpropagate through the simulator.
                drift_list.append(v)
                sigma_list.append(sigma)
                growth_list.append(r)
                ctx_list.append(ctx.context.detach())

            # Euler-Maruyama update
            noise = torch.randn_like(z)  # [G, N, d]
            z = z + v * dtau + sigma * (dtau ** 0.5) * noise

            # Log-weight update (Euler for log-weight ODE)
            logw = logw + r * dtau

            z_list.append(z)
            logw_list.append(logw)

        z_steps = torch.stack(z_list, dim=0)     # [K+1, G, N, d]
        logw_steps = torch.stack(logw_list, dim=0)  # [K+1, G, N]

        result = ParticleRollout(
            z_steps=z_steps,
            logw_steps=logw_steps,
            tau_steps=tau_steps,
            log_m0=log_m0.detach().clone(),
        )
        if self.store_history and drift_list:
            result.drift_steps = torch.stack(drift_list, dim=0)
            result.sigma_steps = torch.stack(sigma_list, dim=0)
            result.growth_steps = torch.stack(growth_list, dim=0)
            result.context_steps = torch.stack(ctx_list, dim=0)

        return result

    @staticmethod
    def effective_sample_size(logw: torch.Tensor) -> torch.Tensor:
        """ESS per perturbation. logw: [G, N] -> [G]."""
        log_norm = logw - torch.logsumexp(logw, dim=-1, keepdim=True)
        log_ess = -torch.logsumexp(2 * log_norm, dim=-1)
        return log_ess.exp()
