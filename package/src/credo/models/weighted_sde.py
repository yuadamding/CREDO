"""Weighted particle SDE simulator (Euler-Maruyama).

Implements the particle system:
    dZ_tau = v_g(Z, tau, c) dtau + Sigma_g(Z, tau, c) dW_tau
    d/dtau log w_tau = r_g(Z, tau, c)

All mass computations use absolute log-weights and log-space reductions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
    context_steps: Optional[torch.Tensor] = None    # [K, C], context for tau_k -> tau_{k+1}
    base_context_steps: Optional[torch.Tensor] = None  # [K, C], drift/sigma context
    growth_context_steps: Optional[torch.Tensor] = None  # [K, C] or [K, G, C], growth context
    context_diagnostics: Optional[Dict[str, torch.Tensor]] = None  # scalar diagnostics stacked over K
    causal_edge_scores_steps: Optional[torch.Tensor] = None  # [K, G, M]
    causal_baseline_edge_scores_steps: Optional[torch.Tensor] = None  # [K, G, M]
    causal_residual_edge_scores_steps: Optional[torch.Tensor] = None  # [K, G, M]
    causal_residual_edge_magnitude_steps: Optional[torch.Tensor] = None  # [K, G, M]
    causal_mediator_tokens_steps: Optional[torch.Tensor] = None  # [K, M, H]
    causal_growth_context_steps: Optional[torch.Tensor] = None  # [K, C] or [K, G, C]
    causal_delta_steps: Optional[torch.Tensor] = None  # [K, C] or [K, G, C]
    noise_steps: Optional[torch.Tensor] = None      # [K, G, N, d] innovations used by rollout
    ess_steps: Optional[torch.Tensor] = None  # [K+1, G]
    ess_frac_steps: Optional[torch.Tensor] = None  # [K+1, G]
    logw_range_steps: Optional[torch.Tensor] = None  # [K+1, G]
    max_weight_frac_steps: Optional[torch.Tensor] = None  # [K+1, G]

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

    def slice_group(self, group_index: int) -> "ParticleRollout":
        """Return a one-group view/copy preserving global context trajectories."""
        idx = int(group_index)
        if idx < 0 or idx >= self.G:
            raise IndexError(f"group_index {idx} out of range for G={self.G}")

        def _slice_optional(value: Optional[torch.Tensor], group_dim: int) -> Optional[torch.Tensor]:
            if value is None:
                return None
            slicer = [slice(None)] * value.ndim
            slicer[group_dim] = slice(idx, idx + 1)
            return value[tuple(slicer)]

        def _slice_context_optional(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if value.ndim == 3:
                return value[:, idx:idx + 1, :]
            return value

        return ParticleRollout(
            z_steps=self.z_steps[:, idx:idx + 1],
            logw_steps=self.logw_steps[:, idx:idx + 1],
            tau_steps=self.tau_steps,
            log_m0=None if self.log_m0 is None else self.log_m0[idx:idx + 1],
            drift_steps=_slice_optional(self.drift_steps, 1),
            sigma_steps=_slice_optional(self.sigma_steps, 1),
            growth_steps=_slice_optional(self.growth_steps, 1),
            context_steps=self.context_steps,
            base_context_steps=_slice_context_optional(self.base_context_steps),
            growth_context_steps=_slice_context_optional(self.growth_context_steps),
            context_diagnostics=self.context_diagnostics,
            causal_edge_scores_steps=_slice_optional(self.causal_edge_scores_steps, 1),
            causal_baseline_edge_scores_steps=_slice_optional(self.causal_baseline_edge_scores_steps, 1),
            causal_residual_edge_scores_steps=_slice_optional(self.causal_residual_edge_scores_steps, 1),
            causal_residual_edge_magnitude_steps=_slice_optional(self.causal_residual_edge_magnitude_steps, 1),
            causal_mediator_tokens_steps=self.causal_mediator_tokens_steps,
            causal_growth_context_steps=_slice_context_optional(self.causal_growth_context_steps),
            causal_delta_steps=_slice_context_optional(self.causal_delta_steps),
            noise_steps=_slice_optional(self.noise_steps, 1),
            ess_steps=_slice_optional(self.ess_steps, 1),
            ess_frac_steps=_slice_optional(self.ess_frac_steps, 1),
            logw_range_steps=_slice_optional(self.logw_range_steps, 1),
            max_weight_frac_steps=_slice_optional(self.max_weight_frac_steps, 1),
        )


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
        tau_grid: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        noise_steps: Optional[torch.Tensor] = None,
        return_noise_used: bool = False,
        intervention: Optional[object] = None,
        context_override: Any = None,
    ) -> ParticleRollout:
        """Run the full Euler-Maruyama rollout.

        Parameters
        ----------
        z0: initial particle positions [G, N, d]
        logw0: initial log-weights [G, N]; should satisfy exp(logw0).sum(1) = 1
        model: FullDynamicsModel with .step() method
        log_m0: log of initial total mass per perturbation [G]
        tau_start, tau_end: time interval in normalized coordinates
        tau_grid: optional explicit grid.  When provided, it must begin at
            ``tau_start`` and end at ``tau_end`` and may use non-uniform steps.
        generator: optional random generator for standard normal innovations.
        noise_steps: optional explicit standard normal innovations with shape
            ``[K, G, N, d]``.  They are multiplied by ``sqrt(dtau)`` inside
            the Euler-Maruyama update and are useful for same-noise
            counterfactuals.
        return_noise_used: if true, attach the exact standard normal
            innovations consumed by the rollout to the returned object.
        context_override: optional external context used at every step, or a
            per-step sequence/tensor with length ``K``.  This is intended for
            clamped single-time effect paths and diagnostics; the default
            remains self-consistent generated-particle context.
        """
        device = z0.device
        dtype = z0.dtype
        G, N, d = z0.shape
        if generator is not None and noise_steps is not None:
            raise ValueError("Pass either generator or noise_steps, not both")

        if tau_grid is None:
            K = self.n_steps
            dtau_uniform = (tau_end - tau_start) / K
            tau_steps = torch.linspace(tau_start, tau_end, K + 1, device=device, dtype=dtype)
        else:
            tau_steps = tau_grid.to(device=device, dtype=dtype)
            if tau_steps.ndim != 1:
                raise ValueError("tau_grid must be a 1D tensor")
            if len(tau_steps) < 2:
                raise ValueError("tau_grid must contain at least two points")
            expected_start = torch.as_tensor(tau_start, device=device, dtype=dtype)
            expected_end = torch.as_tensor(tau_end, device=device, dtype=dtype)
            if not torch.isclose(tau_steps[0], expected_start):
                raise ValueError("tau_grid[0] must equal tau_start")
            if not torch.isclose(tau_steps[-1], expected_end):
                raise ValueError("tau_grid[-1] must equal tau_end")
            if not torch.all(tau_steps[1:] > tau_steps[:-1]):
                raise ValueError("tau_grid must be strictly increasing")
            K = len(tau_steps) - 1
            dtau_uniform = None

        if noise_steps is not None:
            expected_noise_shape = (K, G, N, d)
            noise_steps = noise_steps.to(device=device, dtype=dtype)
            if tuple(noise_steps.shape) != expected_noise_shape:
                raise ValueError(
                    f"noise_steps must have shape {expected_noise_shape}, "
                    f"got {tuple(noise_steps.shape)}"
                )

        # Storage
        z_list = [z0]
        logw_list = [logw0]
        drift_list, sigma_list, growth_list = [], [], []
        ctx_list, base_ctx_list, growth_ctx_list = [], [], []
        causal_edge_scores_list = []
        causal_baseline_edge_scores_list = []
        causal_residual_edge_scores_list = []
        causal_residual_edge_magnitude_list = []
        causal_mediator_tokens_list = []
        causal_growth_context_list = []
        causal_delta_list = []
        diagnostics: dict[str, list[torch.Tensor]] = {}
        noise_used_list = []

        z = z0.clone()
        logw = logw0.clone()
        ess_list = [self.effective_sample_size(logw).detach()]
        ess_frac_list = [self.ess_fraction(logw).detach()]
        logw_range_list = [self.log_weight_range(logw).detach()]
        max_weight_frac_list = [self.max_weight_fraction(logw).detach()]

        for k in range(K):
            tau_k = tau_steps[k]
            dtau = dtau_uniform if dtau_uniform is not None else tau_steps[k + 1] - tau_steps[k]
            sqrt_dtau = (dtau ** 0.5) if dtau_uniform is not None else torch.sqrt(dtau)

            # Get coefficients and context from model
            step_kwargs = {
                "z": z,
                "tau": tau_k,
                "logw": logw,
                "log_m0": log_m0,
                "perturbation_ids": perturbation_ids,
            }
            if intervention is not None:
                step_kwargs["intervention"] = intervention
            selected_context_override = self._context_override_at_step(context_override, k)
            if selected_context_override is not None:
                step_kwargs["context_override"] = selected_context_override
            coeffs, ctx = model.step(**step_kwargs)

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
                base_context = getattr(ctx, "base_context", None)
                if base_context is None:
                    base_context = ctx.context
                growth_context = getattr(ctx, "growth_context", None)
                if growth_context is None:
                    growth_context = ctx.context
                base_ctx_list.append(base_context.detach())
                growth_ctx_list.append(growth_context.detach())
                edge_scores = getattr(ctx, "edge_scores_gm", None)
                if edge_scores is not None:
                    causal_edge_scores_list.append(edge_scores)
                    causal_growth_context_list.append(growth_context)
                    causal_delta_list.append(self._causal_delta(growth_context, ctx.context))
                baseline_edge_scores = getattr(ctx, "baseline_edge_scores_gm", None)
                if baseline_edge_scores is not None:
                    causal_baseline_edge_scores_list.append(baseline_edge_scores)
                residual_edge_scores = getattr(ctx, "residual_edge_scores_gm", None)
                if residual_edge_scores is not None:
                    causal_residual_edge_scores_list.append(residual_edge_scores)
                residual_edge_magnitude = getattr(ctx, "residual_edge_magnitude_gm", None)
                if residual_edge_magnitude is not None:
                    causal_residual_edge_magnitude_list.append(residual_edge_magnitude)
                mediator_tokens = getattr(ctx, "mediator_tokens", None)
                if mediator_tokens is not None:
                    causal_mediator_tokens_list.append(mediator_tokens)
                ctx_diagnostics = getattr(ctx, "diagnostics", None)
                if ctx_diagnostics is not None:
                    for name, value in ctx_diagnostics.__dict__.items():
                        if value is not None:
                            diagnostics.setdefault(name, []).append(value.detach())

            # Euler-Maruyama update
            if noise_steps is not None:
                noise = noise_steps[k]
            elif generator is not None:
                noise = torch.randn(z.shape, dtype=dtype, device=device, generator=generator)
            else:
                noise = torch.randn_like(z)  # [G, N, d]
            if return_noise_used:
                noise_used_list.append(noise.detach().clone())
            z = z + v * dtau + sigma * sqrt_dtau * noise

            # Log-weight update (Euler for log-weight ODE)
            logw = logw + r * dtau

            z_list.append(z)
            logw_list.append(logw)
            ess_list.append(self.effective_sample_size(logw).detach())
            ess_frac_list.append(self.ess_fraction(logw).detach())
            logw_range_list.append(self.log_weight_range(logw).detach())
            max_weight_frac_list.append(self.max_weight_fraction(logw).detach())

        z_steps = torch.stack(z_list, dim=0)     # [K+1, G, N, d]
        logw_steps = torch.stack(logw_list, dim=0)  # [K+1, G, N]

        result = ParticleRollout(
            z_steps=z_steps,
            logw_steps=logw_steps,
            tau_steps=tau_steps,
            log_m0=log_m0.detach().clone(),
            ess_steps=torch.stack(ess_list, dim=0),
            ess_frac_steps=torch.stack(ess_frac_list, dim=0),
            logw_range_steps=torch.stack(logw_range_list, dim=0),
            max_weight_frac_steps=torch.stack(max_weight_frac_list, dim=0),
        )
        if self.store_history and drift_list:
            result.drift_steps = torch.stack(drift_list, dim=0)
            result.sigma_steps = torch.stack(sigma_list, dim=0)
            result.growth_steps = torch.stack(growth_list, dim=0)
            result.context_steps = torch.stack(ctx_list, dim=0)
            result.base_context_steps = torch.stack(base_ctx_list, dim=0)
            result.growth_context_steps = torch.stack(growth_ctx_list, dim=0)
            if diagnostics:
                result.context_diagnostics = {
                    name: torch.stack(values, dim=0)
                    for name, values in diagnostics.items()
                    if values
                }
            if causal_edge_scores_list:
                result.causal_edge_scores_steps = torch.stack(causal_edge_scores_list, dim=0)
            if causal_baseline_edge_scores_list:
                result.causal_baseline_edge_scores_steps = torch.stack(causal_baseline_edge_scores_list, dim=0)
            if causal_residual_edge_scores_list:
                result.causal_residual_edge_scores_steps = torch.stack(causal_residual_edge_scores_list, dim=0)
            if causal_residual_edge_magnitude_list:
                result.causal_residual_edge_magnitude_steps = torch.stack(causal_residual_edge_magnitude_list, dim=0)
            if causal_mediator_tokens_list:
                result.causal_mediator_tokens_steps = torch.stack(causal_mediator_tokens_list, dim=0)
            if causal_growth_context_list:
                result.causal_growth_context_steps = torch.stack(causal_growth_context_list, dim=0)
            if causal_delta_list:
                result.causal_delta_steps = torch.stack(causal_delta_list, dim=0)
        if return_noise_used:
            result.noise_steps = torch.stack(noise_used_list, dim=0)

        return result

    @staticmethod
    def _context_override_at_step(context_override: Any, step_index: int) -> Any:
        """Select a static or step-indexed context override."""
        if context_override is None:
            return None
        if torch.is_tensor(context_override):
            if context_override.ndim >= 2:
                return context_override[step_index]
            return context_override
        if isinstance(context_override, dict):
            if "context_steps" in context_override:
                selected = dict(context_override)
                selected["context"] = context_override["context_steps"][step_index]
                selected.pop("context_steps", None)
                if "base_context_steps" in context_override:
                    selected["base_context"] = context_override["base_context_steps"][step_index]
                    selected.pop("base_context_steps", None)
                if "growth_context_steps" in context_override:
                    selected["growth_context"] = context_override["growth_context_steps"][step_index]
                    selected.pop("growth_context_steps", None)
                return selected
            return context_override
        if isinstance(context_override, (list, tuple)):
            return context_override[step_index]
        return context_override

    @staticmethod
    def _causal_delta(growth_context: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return the group-specific causal delta from global context."""
        if growth_context.ndim == 2 and context.ndim == 1:
            return growth_context - context[None, :]
        return growth_context - context

    @staticmethod
    def sample_noise_like(
        z0: torch.Tensor,
        n_steps: int,
        *,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample standard normal innovations with shape ``[n_steps, *z0.shape]``."""
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both")
        if generator is None and seed is not None:
            generator = torch.Generator(device=z0.device)
            generator.manual_seed(int(seed))
        return torch.randn(
            (int(n_steps),) + tuple(z0.shape),
            dtype=z0.dtype,
            device=z0.device,
            generator=generator,
        )

    @staticmethod
    def sample_noise_for_tau_grid(
        z0: torch.Tensor,
        tau_grid: torch.Tensor,
        *,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample standard normal innovations for ``len(tau_grid) - 1`` steps."""
        if tau_grid.ndim != 1 or len(tau_grid) < 2:
            raise ValueError("tau_grid must be a 1D tensor with at least two points")
        return WeightedParticleSimulator.sample_noise_like(
            z0,
            len(tau_grid) - 1,
            seed=seed,
            generator=generator,
        )

    @staticmethod
    def effective_sample_size(logw: torch.Tensor) -> torch.Tensor:
        """ESS per perturbation. logw: [G, N] -> [G]."""
        return WeightedParticleSimulator.log_effective_sample_size(logw).exp()

    @staticmethod
    def log_effective_sample_size(logw: torch.Tensor) -> torch.Tensor:
        """Log effective sample size per perturbation. logw: [G, N] -> [G]."""
        logw32 = logw.float()
        log_norm = logw32 - torch.logsumexp(logw32, dim=-1, keepdim=True)
        return -torch.logsumexp(2 * log_norm, dim=-1)

    @staticmethod
    def ess_fraction(logw: torch.Tensor) -> torch.Tensor:
        """ESS divided by particle count per perturbation. logw: [G, N] -> [G]."""
        return WeightedParticleSimulator.effective_sample_size(logw) / float(logw.shape[-1])

    @staticmethod
    def max_weight_fraction(logw: torch.Tensor) -> torch.Tensor:
        """Largest normalised particle weight per perturbation. logw: [G, N] -> [G]."""
        logw32 = logw.float()
        return torch.exp(logw32.max(dim=-1).values - torch.logsumexp(logw32, dim=-1))

    @staticmethod
    def log_weight_range(logw: torch.Tensor) -> torch.Tensor:
        """Range of relative log-weights per perturbation. logw: [G, N] -> [G]."""
        logw32 = logw.float()
        return logw32.max(dim=-1).values - logw32.min(dim=-1).values
