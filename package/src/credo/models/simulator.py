"""CounterfactualEngine: simulate under perturbation vs control embedding.

Also exports helper to initialise particles from an EndpointProblem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from ..data.core import EndpointProblem, FiniteMeasure, TrajectoryProblem
from .full_model import FullDynamicsModel
from .interventions import CausalAttentionIntervention
from .weighted_sde import WeightedParticleSimulator, ParticleRollout


def _make_generator(seed: Optional[int], device: torch.device | str) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _stable_seed_offset(text: str, modulus: int = 1_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def embedding_ids_from_endpoint(endpoint: EndpointProblem, perturbation_ids: List[str]) -> List[str]:
    """Resolve model embedding IDs for endpoint measure keys."""
    mapping = endpoint.metadata.get("measure_to_embedding")
    if not isinstance(mapping, dict):
        mapping = endpoint.metadata.get("embedding_ids")
    if not isinstance(mapping, dict):
        return list(perturbation_ids)
    return [str(mapping.get(pid, pid)) for pid in perturbation_ids]


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
    generator = _make_generator(seed, torch.device(device))

    G = len(perturbation_ids)
    d = next(iter(measures.values())).latent_dim
    z0 = torch.zeros(G, n_particles, d, dtype=dtype, device=device)
    logw0 = torch.zeros(G, n_particles, dtype=dtype, device=device)
    log_m0 = torch.zeros(G, dtype=dtype, device=device)

    for g, pid in enumerate(perturbation_ids):
        mu: FiniteMeasure = measures[pid]
        support = torch.tensor(mu.support, dtype=dtype, device=device)  # [n_atoms, d]
        probs = torch.tensor(mu.normalized_weights, dtype=dtype, device=device)
        idx = torch.multinomial(probs, n_particles, replacement=True, generator=generator)
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
    sampling: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample initial particles from the endpoint initial measure.

    ``sampling="uniform"`` preserves the original endpoint code path exactly:
    cells are sampled uniformly from support with ``torch.randint``.  Use
    ``sampling="legacy_uniform"`` when this legacy path should fail fast on a
    non-uniform finite measure, or ``sampling="measure_weights"`` to sample
    atoms by their finite-measure weights.
    """
    if sampling not in {"uniform", "legacy_uniform", "measure_weights"}:
        raise ValueError("sampling must be 'uniform', 'legacy_uniform', or 'measure_weights'.")
    generator = _make_generator(seed, torch.device(device))

    G = len(perturbation_ids)
    d = next(iter(endpoint.initial.values())).latent_dim
    z0 = torch.zeros(G, n_particles, d, dtype=dtype, device=device)
    logw0 = torch.zeros(G, n_particles, dtype=dtype, device=device)
    log_m0 = torch.zeros(G, dtype=dtype, device=device)

    for g, pid in enumerate(perturbation_ids):
        mu: FiniteMeasure = endpoint.initial[pid]
        support = torch.tensor(mu.support, dtype=dtype, device=device)  # [n_atoms, d]
        n_atoms = len(support)
        if sampling == "legacy_uniform":
            weights = np.asarray(mu.normalized_weights, dtype=np.float64)
            expected = np.full_like(weights, 1.0 / max(1, len(weights)))
            if not np.allclose(weights, expected, rtol=1e-5, atol=1e-8):
                raise ValueError(
                    "legacy_uniform sampling requested for a non-uniform finite measure; "
                    "use sampling='measure_weights' to respect atom weights."
                )
        if sampling == "measure_weights":
            probs = torch.tensor(mu.normalized_weights, dtype=dtype, device=device)
            idx = torch.multinomial(probs, n_particles, replacement=True, generator=generator)
        else:
            idx = torch.randint(0, n_atoms, (n_particles,), device=device, generator=generator)
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


@torch.no_grad()
def rollout_with_clamped_context(
    model: FullDynamicsModel,
    z0: torch.Tensor,
    logw0: torch.Tensor,
    log_m0: torch.Tensor,
    perturbation_ids: List[str],
    context_steps: torch.Tensor,
    *,
    embedding_ids: Optional[List[str]] = None,
    base_context_steps: Optional[torch.Tensor] = None,
    growth_context_steps: Optional[torch.Tensor] = None,
    n_steps: Optional[int] = None,
    tau_start: float = 0.0,
    tau_end: float = 1.0,
    tau_grid: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    noise_steps: Optional[torch.Tensor] = None,
    return_noise_used: bool = False,
) -> ParticleRollout:
    """Roll out dynamics while reusing a fixed context trajectory.

    ``context_steps[k]`` is the context used for the transition
    ``tau_steps[k] -> tau_steps[k + 1]`` and therefore has length ``K``, not
    ``K+1``.  This helper is used for context-clamped counterfactuals.
    """
    if z0.shape[:2] != logw0.shape:
        raise ValueError("z0 and logw0 shapes are inconsistent")
    if z0.shape[0] != len(perturbation_ids):
        raise ValueError("perturbation_ids length must match z0.shape[0]")
    if embedding_ids is not None and len(embedding_ids) != len(perturbation_ids):
        raise ValueError("embedding_ids length must match perturbation_ids length")
    if generator is not None and noise_steps is not None:
        raise ValueError("Pass either generator or noise_steps, not both")

    device = z0.device
    dtype = z0.dtype
    if tau_grid is None:
        K = int(n_steps if n_steps is not None else context_steps.shape[0])
        if K < 1:
            raise ValueError("n_steps must be >= 1")
        tau_steps = torch.linspace(tau_start, tau_end, K + 1, device=device, dtype=dtype)
    else:
        tau_steps = tau_grid.to(device=device, dtype=dtype)
        if tau_steps.ndim != 1 or len(tau_steps) < 2:
            raise ValueError("tau_grid must be a 1D tensor with at least two points")
        expected_start = torch.as_tensor(tau_start, device=device, dtype=dtype)
        expected_end = torch.as_tensor(tau_end, device=device, dtype=dtype)
        if not torch.isclose(tau_steps[0], expected_start):
            raise ValueError("tau_grid[0] must equal tau_start")
        if not torch.isclose(tau_steps[-1], expected_end):
            raise ValueError("tau_grid[-1] must equal tau_end")
        if not torch.all(tau_steps[1:] > tau_steps[:-1]):
            raise ValueError("tau_grid must be strictly increasing")
        K = len(tau_steps) - 1
        if n_steps is not None and int(n_steps) != K:
            raise ValueError("n_steps must match len(tau_grid) - 1")

    if context_steps.shape[0] < K:
        raise ValueError(
            f"Clamped context has {context_steps.shape[0]} steps, but rollout requested {K} steps."
        )
    expected_context_width = model.context_agg.context_dim

    def _validate_context_steps(name: str, value: torch.Tensor) -> None:
        if value.shape[0] < K:
            raise ValueError(f"{name} has {value.shape[0]} steps, but rollout requested {K} steps.")
        if value.ndim == 2 and value.shape[1] == expected_context_width:
            return
        if (
            value.ndim == 3
            and value.shape[1] == z0.shape[0]
            and value.shape[2] == expected_context_width
        ):
            return
        if value.ndim == 3:
            raise ValueError(
                f"{name} must have shape [K, context_dim] or [K, G, context_dim] "
                f"with G={z0.shape[0]} and context_dim={expected_context_width}; "
                f"got {tuple(value.shape)}"
            )
        else:
            raise ValueError(
                f"{name} must have shape [K, context_dim] with "
                f"context_dim={expected_context_width}; got {tuple(value.shape)}"
            )

    _validate_context_steps("context_steps", context_steps)
    if base_context_steps is not None:
        _validate_context_steps("base_context_steps", base_context_steps)
    if growth_context_steps is not None:
        _validate_context_steps("growth_context_steps", growth_context_steps)
    if (
        getattr(model, "transformer_growth_only", False)
        and getattr(model, "meanfield_context_agg", None) is not None
        and (base_context_steps is None or growth_context_steps is None)
    ):
        raise ValueError(
            "transformer_growth_only clamped rollouts require both "
            "base_context_steps and growth_context_steps."
        )
    if noise_steps is not None:
        expected_noise_shape = (K,) + tuple(z0.shape)
        noise_steps = noise_steps.to(device=device, dtype=dtype)
        if tuple(noise_steps.shape) != expected_noise_shape:
            raise ValueError(
                f"noise_steps must have shape {expected_noise_shape}, got {tuple(noise_steps.shape)}"
            )

    z = z0.clone()
    logw = logw0.clone()
    z_list = [z]
    logw_list = [logw]
    drift_list = []
    sigma_list = []
    growth_list = []
    used_context = []
    used_base_context = []
    used_growth_context = []
    noise_used_list = []

    n_programs = model.context_agg.n_programs
    embed_pids = embedding_ids or perturbation_ids
    for k in range(K):
        tau_k = tau_steps[k]
        dtau = tau_steps[k + 1] - tau_steps[k]
        context = context_steps[k].to(device=device, dtype=dtype)
        base_context = (
            base_context_steps[k].to(device=device, dtype=dtype)
            if base_context_steps is not None
            else context
        )
        growth_context = (
            growth_context_steps[k].to(device=device, dtype=dtype)
            if growth_context_steps is not None
            else context
        )
        # q/s are global finite-measure ecological summaries.  CEA may supply
        # group-specific growth contexts [G, C], but those contain an additive
        # causal mediator delta and do not redefine the global q/s payoff input.
        global_ecology_context = (
            growth_context
            if growth_context_steps is not None and growth_context.ndim == 1
            else context
        )
        q = global_ecology_context[:n_programs]
        s = global_ecology_context[n_programs:]
        a = model.embedding(embed_pids)
        b = model.embedding.growth_intercepts(embed_pids)
        eta_z, _ = model.context_agg.encode_particles(z)
        coeffs = model.coeff_nets(
            z=z,
            tau=tau_k,
            context=base_context,
            a=a,
            growth_intercept=b,
            eta_z=eta_z,
            q=q,
            s=s,
            growth_context=growth_context,
        )

        drift_list.append(coeffs.drift)
        sigma_list.append(coeffs.sigma_diag)
        growth_list.append(coeffs.growth)
        used_context.append(context.detach())
        used_base_context.append(base_context.detach())
        used_growth_context.append(growth_context.detach())

        if noise_steps is not None:
            noise = noise_steps[k]
        elif generator is not None:
            noise = torch.randn(z.shape, dtype=dtype, device=device, generator=generator)
        else:
            noise = torch.randn_like(z)
        if return_noise_used:
            noise_used_list.append(noise.detach().clone())
        z = z + coeffs.drift * dtau + coeffs.sigma_diag * torch.sqrt(dtau) * noise
        logw = logw + coeffs.growth * dtau
        z_list.append(z)
        logw_list.append(logw)

    rollout = ParticleRollout(
        z_steps=torch.stack(z_list, dim=0),
        logw_steps=torch.stack(logw_list, dim=0),
        tau_steps=tau_steps,
        log_m0=log_m0.detach().clone(),
        drift_steps=torch.stack(drift_list, dim=0),
        sigma_steps=torch.stack(sigma_list, dim=0),
        growth_steps=torch.stack(growth_list, dim=0),
        context_steps=torch.stack(used_context, dim=0),
        base_context_steps=torch.stack(used_base_context, dim=0),
        growth_context_steps=torch.stack(used_growth_context, dim=0),
    )
    if return_noise_used:
        rollout.noise_steps = torch.stack(noise_used_list, dim=0)
    return rollout


@dataclass
class CounterfactualResult:
    """Paired simulation outputs for a single perturbation."""
    perturbation_id: str
    rollout_perturb: ParticleRollout
    rollout_control: ParticleRollout
    rollout_clamped: Optional[ParticleRollout] = None  # factual dynamics with context clamped to control
    rollout_control_clamped: Optional[ParticleRollout] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def rollout_factual(self) -> ParticleRollout:
        """Alias for the factual branch."""
        return self.rollout_perturb

    @property
    def rollout_intervention(self) -> ParticleRollout:
        """Alias for intervention branches stored in ``rollout_control``."""
        return self.rollout_control

    def terminal_log_mass_diff(self) -> float:
        """Terminal factual-reference log-mass contrast."""
        logw_p = self.rollout_perturb.terminal_logw.squeeze(0)
        logw_c = self.rollout_control.terminal_logw.squeeze(0)
        if self.rollout_perturb.log_m0 is None or self.rollout_control.log_m0 is None:
            raise ValueError("Counterfactual mass comparison requires rollout.log_m0.")
        log_mass_p = self.rollout_perturb.log_m0.squeeze(0) + torch.logsumexp(logw_p, 0)
        log_mass_c = self.rollout_control.log_m0.squeeze(0) + torch.logsumexp(logw_c, 0)
        return float((log_mass_p - log_mass_c).item())

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
        common_noise: bool = True,
        allow_partial_context: bool = False,
        min_context_fraction: float = 0.95,
    ) -> List[CounterfactualResult]:
        """Run counterfactual simulations.

        Parameters
        ----------
        endpoint: provides P4 initial conditions
        perturbation_ids: which perturbations to analyse
        clamp_context: if True, also run with context fixed to the control trajectory.
            This requires ``self.simulator.store_history=True`` so control
            context checkpoints are available.
        control_rollout_mode:
            - ``reference_consistent``: for ``soft_ref``, keep the shared
              reference embedding and set only the perturbation residual to zero
            - ``zero_centered``: force the full effective embedding to zero as a
              diagnostic rollout
        common_noise: if True, factual and reference branches reuse the same
            standard normal innovation stream after sharing the same initial
            particles.
        """
        results = []

        if control_rollout_mode not in {"reference_consistent", "zero_centered"}:
            raise ValueError(
                "control_rollout_mode must be 'reference_consistent' or 'zero_centered'."
            )
        if clamp_context and not self.simulator.store_history:
            raise ValueError("clamp_context=True requires a simulator with store_history=True.")

        if getattr(self.model, "context_kind", "mlp") in {"transformer", "causal_attention"}:
            return self._run_global_context_counterfactual(
                endpoint=endpoint,
                perturbation_ids=perturbation_ids,
                clamp_context=clamp_context,
                seed=seed,
                control_rollout_mode=control_rollout_mode,
                common_noise=common_noise,
                allow_partial_context=allow_partial_context,
                min_context_fraction=min_context_fraction,
            )

        for pid in perturbation_ids:
            if pid not in endpoint.initial:
                continue
            embed_pid = embedding_ids_from_endpoint(endpoint, [pid])[0]

            # --- Perturbation rollout ---
            z0p, lw0p, lm0p = initialise_particles(
                endpoint, [pid], self.n_particles, self.device, seed=seed)
            branch_seed = int(seed) + 10_000 + _stable_seed_offset(pid)
            noise_steps = None
            if common_noise:
                noise_steps = self.simulator.sample_noise_like(
                    z0p, self.simulator.n_steps, seed=branch_seed
                )
            rollout_p = self.simulator.rollout(
                z0=z0p,
                logw0=lw0p,
                model=self.model,
                log_m0=lm0p,
                perturbation_ids=[pid],
                embedding_ids=[embed_pid],
                noise_steps=noise_steps,
            )

            # --- Control rollout with the same perturbation-specific initial measure ---
            z0c, lw0c, lm0c = z0p.clone(), lw0p.clone(), lm0p.clone()

            # Reference-consistent soft-ref semantics keep a_ref and zero only
            # the perturbation residual; full zeroing is left as a diagnostic.
            with _control_embedding_context(self.model, embed_pid, mode=control_rollout_mode):
                rollout_c = self.simulator.rollout(
                    z0=z0c,
                    logw0=lw0c,
                    model=self.model,
                    log_m0=lm0c,
                    perturbation_ids=[pid],
                    embedding_ids=[embed_pid],
                    noise_steps=noise_steps,
                )

            rollout_clamped = None
            rollout_control_clamped = None
            if clamp_context:
                if rollout_c.context_steps is None:
                    raise ValueError("Control rollout did not store context_steps for clamped context.")
                tau_grid = rollout_c.tau_steps.detach()
                tau_start = float(tau_grid[0].item())
                tau_end = float(tau_grid[-1].item())
                rollout_clamped = rollout_with_clamped_context(
                    model=self.model,
                    z0=z0p,
                    logw0=lw0p,
                    log_m0=lm0p,
                    perturbation_ids=[pid],
                    embedding_ids=[embed_pid],
                    context_steps=rollout_c.context_steps,
                    base_context_steps=rollout_c.base_context_steps,
                    growth_context_steps=rollout_c.growth_context_steps,
                    tau_start=tau_start,
                    tau_end=tau_end,
                    tau_grid=tau_grid,
                    noise_steps=noise_steps,
                )
                with _control_embedding_context(self.model, embed_pid, mode=control_rollout_mode):
                    rollout_control_clamped = rollout_with_clamped_context(
                        model=self.model,
                        z0=z0c,
                        logw0=lw0c,
                        log_m0=lm0c,
                        perturbation_ids=[pid],
                        embedding_ids=[embed_pid],
                        context_steps=rollout_c.context_steps,
                        base_context_steps=rollout_c.base_context_steps,
                        growth_context_steps=rollout_c.growth_context_steps,
                        tau_start=tau_start,
                        tau_end=tau_end,
                        tau_grid=tau_grid,
                        noise_steps=noise_steps,
                    )

            result = CounterfactualResult(
                perturbation_id=pid,
                rollout_perturb=rollout_p,
                rollout_control=rollout_c,
                rollout_clamped=rollout_clamped,
                rollout_control_clamped=rollout_control_clamped,
            )
            results.append(result)

        return results

    def _full_context_perturbation_ids(
        self,
        endpoint: EndpointProblem,
        *,
        allow_partial_context: bool = False,
        min_context_fraction: float = 0.95,
    ) -> Tuple[List[str], Dict[str, object]]:
        model_pids = list(self.model.perturbation_ids)
        available = [pid for pid in model_pids if pid in endpoint.initial]
        missing = [pid for pid in model_pids if pid not in endpoint.initial]
        n_model = max(1, len(model_pids))
        context_fraction = len(available) / float(n_model)
        if len(available) < 2:
            raise ValueError(
                "Global-context counterfactuals require full-context rollout with at least "
                "two perturbations, or an explicit clamped context trajectory. "
                "Single-perturbation ecology is degenerate."
            )
        if missing and not allow_partial_context:
            raise ValueError(
                "Global-context counterfactual context is partial: "
                f"{len(available)}/{len(model_pids)} perturbations available. "
                "Pass allow_partial_context=True only for an explicit diagnostic."
            )
        if context_fraction < float(min_context_fraction) and not allow_partial_context:
            raise ValueError(
                f"Global-context counterfactual context coverage {context_fraction:.3f} "
                f"is below the required {float(min_context_fraction):.3f}."
            )
        metadata: Dict[str, object] = {
            "context_n_available": len(available),
            "context_n_model": len(model_pids),
            "context_fraction": context_fraction,
            "allow_partial_context": bool(allow_partial_context),
            "min_context_fraction": float(min_context_fraction),
            "context_missing_perturbations": missing,
        }
        return available, metadata

    def _run_global_context_counterfactual(
        self,
        *,
        endpoint: EndpointProblem,
        perturbation_ids: List[str],
        clamp_context: bool,
        seed: int,
        control_rollout_mode: str,
        common_noise: bool,
        allow_partial_context: bool,
        min_context_fraction: float,
    ) -> List[CounterfactualResult]:
        """Run counterfactuals inside the full ecological context."""
        all_pids, context_metadata = self._full_context_perturbation_ids(
            endpoint,
            allow_partial_context=allow_partial_context,
            min_context_fraction=min_context_fraction,
        )
        results: List[CounterfactualResult] = []

        z0_all, lw0_all, lm0_all = initialise_particles(
            endpoint,
            all_pids,
            self.n_particles,
            self.device,
            seed=seed,
        )
        all_embedding_ids = embedding_ids_from_endpoint(endpoint, all_pids)
        noise_seed = int(seed) + 10_000
        noise_steps = None
        if common_noise:
            noise_steps = self.simulator.sample_noise_like(
                z0_all,
                self.simulator.n_steps,
                seed=noise_seed,
            )

        rollout_p_all = self.simulator.rollout(
            z0=z0_all,
            logw0=lw0_all,
            model=self.model,
            log_m0=lm0_all,
            perturbation_ids=all_pids,
            embedding_ids=all_embedding_ids,
            noise_steps=noise_steps,
            return_noise_used=common_noise,
        )

        for pid in perturbation_ids:
            if pid not in endpoint.initial or pid not in all_pids:
                continue
            embed_pid = embedding_ids_from_endpoint(endpoint, [pid])[0]

            with _control_embedding_context(self.model, embed_pid, mode=control_rollout_mode):
                rollout_c_all = self.simulator.rollout(
                    z0=z0_all.clone(),
                    logw0=lw0_all.clone(),
                    model=self.model,
                    log_m0=lm0_all.clone(),
                    perturbation_ids=all_pids,
                    embedding_ids=all_embedding_ids,
                    noise_steps=noise_steps.clone() if noise_steps is not None else None,
                    return_noise_used=common_noise,
                )

            rollout_clamped = None
            rollout_control_clamped = None
            if clamp_context:
                if (
                    rollout_c_all.context_steps is None
                    or rollout_c_all.base_context_steps is None
                    or rollout_c_all.growth_context_steps is None
                ):
                    raise ValueError("Control rollout did not store all context_steps for clamped context.")
                tau_grid = rollout_c_all.tau_steps.detach()
                tau_start = float(tau_grid[0].item())
                tau_end = float(tau_grid[-1].item())
                rollout_clamped_all = rollout_with_clamped_context(
                    model=self.model,
                    z0=z0_all,
                    logw0=lw0_all,
                    log_m0=lm0_all,
                    perturbation_ids=all_pids,
                    embedding_ids=all_embedding_ids,
                    context_steps=rollout_c_all.context_steps,
                    base_context_steps=rollout_c_all.base_context_steps,
                    growth_context_steps=rollout_c_all.growth_context_steps,
                    tau_start=tau_start,
                    tau_end=tau_end,
                    tau_grid=tau_grid,
                    noise_steps=noise_steps,
                    return_noise_used=common_noise,
                )
                with _control_embedding_context(self.model, embed_pid, mode=control_rollout_mode):
                    rollout_control_clamped_all = rollout_with_clamped_context(
                        model=self.model,
                        z0=z0_all.clone(),
                        logw0=lw0_all.clone(),
                        log_m0=lm0_all.clone(),
                        perturbation_ids=all_pids,
                        embedding_ids=all_embedding_ids,
                        context_steps=rollout_c_all.context_steps,
                        base_context_steps=rollout_c_all.base_context_steps,
                        growth_context_steps=rollout_c_all.growth_context_steps,
                        tau_start=tau_start,
                        tau_end=tau_end,
                        tau_grid=tau_grid,
                        noise_steps=noise_steps.clone() if noise_steps is not None else None,
                        return_noise_used=common_noise,
                    )
                target_idx = all_pids.index(pid)
                rollout_clamped = rollout_clamped_all.slice_group(target_idx)
                rollout_control_clamped = rollout_control_clamped_all.slice_group(target_idx)

            target_idx = all_pids.index(pid)
            metadata = dict(context_metadata)
            metadata.update(
                {
                    "context_kind": getattr(self.model, "context_kind", "mlp"),
                    "target_perturbation_id": pid,
                    "counterfactual_seed_mode": "global_common" if common_noise else "global_initial_only",
                    "same_start": True,
                    "same_noise": bool(common_noise),
                    "initial_seed": int(seed),
                    "noise_seed": noise_seed if common_noise else None,
                    "factual_full_context_reused": True,
                }
            )
            results.append(
                CounterfactualResult(
                    perturbation_id=pid,
                    rollout_perturb=rollout_p_all.slice_group(target_idx),
                    rollout_control=rollout_c_all.slice_group(target_idx),
                    rollout_clamped=rollout_clamped,
                    rollout_control_clamped=rollout_control_clamped,
                    metadata=metadata,
                )
            )

        return results

    @torch.no_grad()
    def run_mediator_ablation(
        self,
        endpoint: EndpointProblem,
        perturbation_ids: List[str],
        mediator_ids: List[int],
        seed: int = 0,
        common_noise: bool = True,
        allow_partial_context: bool = False,
        min_context_fraction: float = 0.95,
        ablate_global_mediator: bool = False,
        edge_protocol: str = "ablate_effective_edges",
        context_protocol: Literal["self_consistent", "global_context_clamped"] = "self_consistent",
    ) -> List[CounterfactualResult]:
        """Run same-start/same-noise CEA mediator or edge ablations."""
        if getattr(self.model, "context_kind", "mlp") != "causal_attention":
            raise ValueError("run_mediator_ablation requires context_kind='causal_attention'.")
        if not getattr(getattr(self.model, "context_agg", None), "use_sparse_edges", True):
            raise ValueError(
                "CEA mediator interventions require causal_sparse_edges=True; "
                "dense mediator attention is not intervention-addressable."
            )
        if context_protocol not in {"self_consistent", "global_context_clamped"}:
            raise ValueError("context_protocol must be 'self_consistent' or 'global_context_clamped'.")
        if context_protocol == "global_context_clamped":
            raise NotImplementedError(
                "Global-context-clamped CEA edge ablation is not implemented yet; "
                "use context_protocol='self_consistent'."
            )
        if edge_protocol not in {"ablate_residual_edges", "ablate_effective_edges", "ablate_baseline_edges"}:
            raise ValueError(
                "edge_protocol must be 'ablate_residual_edges', "
                "'ablate_effective_edges', or 'ablate_baseline_edges'."
            )
        if (
            edge_protocol == "ablate_residual_edges"
            and getattr(self.model.context_agg, "residual_policy", "edges_only") != "edges_only"
        ):
            raise ValueError(
                "Residual-edge ablation is only a clean causal estimand when "
                "causal_residual_policy='edges_only'. Use effective-edge ablation "
                "or label the run as predictive/diagnostic."
            )
        all_pids, context_metadata = self._full_context_perturbation_ids(
            endpoint,
            allow_partial_context=allow_partial_context,
            min_context_fraction=min_context_fraction,
        )
        z0_all, lw0_all, lm0_all = initialise_particles(
            endpoint,
            all_pids,
            self.n_particles,
            self.device,
            seed=seed,
        )
        noise_seed = int(seed) + 10_000
        noise_steps = None
        if common_noise:
            noise_steps = self.simulator.sample_noise_like(
                z0_all,
                self.simulator.n_steps,
                seed=noise_seed,
            )

        factual_all = self.simulator.rollout(
            z0=z0_all,
            logw0=lw0_all,
            model=self.model,
            log_m0=lm0_all,
            perturbation_ids=all_pids,
            embedding_ids=embedding_ids_from_endpoint(endpoint, all_pids),
            noise_steps=noise_steps,
            return_noise_used=common_noise,
        )

        results: List[CounterfactualResult] = []
        for pid in perturbation_ids:
            if pid not in endpoint.initial or pid not in all_pids:
                continue
            target_idx = all_pids.index(pid)
            for mediator_id in mediator_ids:
                if ablate_global_mediator:
                    intervention = CausalAttentionIntervention(
                        protocol=edge_protocol,
                        ablate_mediator_ids=[int(mediator_id)],
                    )
                else:
                    intervention = CausalAttentionIntervention(
                        protocol=edge_protocol,
                        ablate_group_mediator_edges=[(target_idx, int(mediator_id))],
                    )
                ablated_all = self.simulator.rollout(
                    z0=z0_all.clone(),
                    logw0=lw0_all.clone(),
                    model=self.model,
                    log_m0=lm0_all.clone(),
                    perturbation_ids=all_pids,
                    embedding_ids=embedding_ids_from_endpoint(endpoint, all_pids),
                    noise_steps=noise_steps.clone() if noise_steps is not None else None,
                    return_noise_used=common_noise,
                    intervention=intervention,
                )
                metadata = dict(context_metadata)
                metadata.update(
                    {
                        "counterfactual_type": "mediator_ablation",
                        "context_kind": "causal_attention",
                        "same_start": True,
                        "same_noise": bool(common_noise),
                        "target_perturbation_id": pid,
                        "mediator_id": int(mediator_id),
                        "ablation_scope": "global_mediator" if ablate_global_mediator else "group_edge",
                        "rollout_control_semantics": "intervention_not_control_reference",
                        "intervention_type": "mediator_ablation",
                        "edge_protocol": edge_protocol,
                        "context_protocol": context_protocol,
                        "initial_seed": int(seed),
                        "noise_seed": noise_seed if common_noise else None,
                    }
                )
                results.append(
                    CounterfactualResult(
                        perturbation_id=pid,
                        rollout_perturb=factual_all.slice_group(target_idx),
                        rollout_control=ablated_all.slice_group(target_idx),
                        metadata=metadata,
                    )
                )
        return results

    @torch.no_grad()
    def run_residual_edge_ablation(
        self,
        endpoint: EndpointProblem,
        perturbation_ids: List[str],
        mediator_ids: List[int],
        seed: int = 0,
        common_noise: bool = True,
        allow_partial_context: bool = False,
        min_context_fraction: float = 0.95,
        ablate_global_mediator: bool = False,
        context_protocol: Literal["self_consistent", "global_context_clamped"] = "self_consistent",
    ) -> List[CounterfactualResult]:
        """Run CEA ablations that remove perturbation-residual mediator edges only."""
        return self.run_mediator_ablation(
            endpoint,
            perturbation_ids,
            mediator_ids,
            seed=seed,
            common_noise=common_noise,
            allow_partial_context=allow_partial_context,
            min_context_fraction=min_context_fraction,
            ablate_global_mediator=ablate_global_mediator,
            edge_protocol="ablate_residual_edges",
            context_protocol=context_protocol,
        )


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
