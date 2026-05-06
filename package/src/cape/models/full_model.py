"""Full dynamics model wrapping embeddings, context, and coefficients.

FullDynamicsModel.step() is called by the simulator at each time step.
It computes the ecological context, then returns Coefficients and ContextState.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .embeddings import PerturbationEmbedding, TimeEmbedding
from .context import ContextAggregator, ContextState
from .coefficients import CoefficientNetworks, Coefficients


class FullDynamicsModel(nn.Module):
    """Top-level model combining all sub-modules.

    Parameters
    ----------
    perturbation_ids: all perturbation ids (defines ordering)
    control_ids: subset that are biological controls
    latent_dim: d
    embedding_dim: r
    n_programs: K
    mediator_dim: L
    context_dim: C = K + L (identity context by default)
    hidden_dim, depth: MLP architecture
    n_time_freqs: Fourier frequencies for time
    sigma_min: minimum diffusion std
    r_max: max growth rate
    n_payoff_ranks: ecological payoff ranks
    ecological_growth: enable ecological growth term
    control_mode: ``anchored`` (exact zero controls), ``free`` (learn controls),
        or ``soft_ref`` (shared control reference + zero control residual)
    """

    def __init__(
        self,
        perturbation_ids: List[str],
        control_ids: List[str],
        latent_dim: int = 16,
        embedding_dim: int = 8,
        n_programs: int = 8,
        mediator_dim: int = 8,
        hidden_dim: int = 128,
        depth: int = 3,
        activation_checkpointing: bool = False,
        n_time_freqs: int = 4,
        sigma_min: float = 1e-3,
        r_max: float = 3.0,
        n_payoff_ranks: int = 4,
        ecological_growth: bool = True,
        use_growth_intercept: bool = True,
        shared_guide_embedding: bool = False,
        program_centroids: Optional[torch.Tensor] = None,
        program_assignment_scale: float = 1.0,
        control_mode: str = "soft_ref",
        control_ref_penalty: float = 5e-4,
    ) -> None:
        super().__init__()
        self.perturbation_ids = perturbation_ids
        self.control_ids = set(control_ids)
        self.control_mode = control_mode
        self.anchor_controls = control_mode == "anchored"
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        if program_centroids is not None:
            program_centroids = torch.as_tensor(program_centroids, dtype=torch.float32)
            if program_centroids.ndim != 2 or program_centroids.shape[1] != latent_dim:
                raise ValueError(
                    "program_centroids must have shape [n_programs, latent_dim], "
                    f"got {tuple(program_centroids.shape)} with latent_dim={latent_dim}."
                )
            self.n_programs = int(program_centroids.shape[0])
        else:
            self.n_programs = n_programs
        self.mediator_dim = mediator_dim

        context_dim = self.n_programs + mediator_dim  # identity context

        self.embedding = PerturbationEmbedding(
            perturbation_ids=perturbation_ids,
            control_ids=list(control_ids),
            embedding_dim=embedding_dim,
            control_mode=control_mode,
            control_ref_penalty=control_ref_penalty,
            use_growth_intercept=use_growth_intercept,
            shared_guide_embedding=shared_guide_embedding,
        )

        self.context_agg = ContextAggregator(
            latent_dim=latent_dim,
            n_programs=self.n_programs,
            mediator_dim=mediator_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            use_identity_context=True,
            fixed_program_centroids=program_centroids,
            program_assignment_scale=program_assignment_scale,
            activation_checkpointing=activation_checkpointing,
        )

        self.coeff_nets = CoefficientNetworks(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation_checkpointing=activation_checkpointing,
            n_time_freqs=n_time_freqs,
            sigma_min=sigma_min,
            r_max=r_max,
            n_programs=self.n_programs,
            n_payoff_ranks=n_payoff_ranks,
            ecological_growth=ecological_growth,
        )

    def get_embeddings(self, perturbation_ids: Optional[List[str]] = None) -> torch.Tensor:
        pids = perturbation_ids or self.perturbation_ids
        return self.embedding(pids)  # [G, r]

    def step(
        self,
        z: torch.Tensor,      # [G, N, d]
        tau: torch.Tensor,    # scalar
        logw: torch.Tensor,   # [G, N]
        log_m0: torch.Tensor, # [G]
        perturbation_ids: Optional[List[str]] = None,
    ) -> Tuple[Coefficients, ContextState]:
        """One step of the dynamics: compute context and coefficients."""
        pids = perturbation_ids or self.perturbation_ids
        a = self.embedding(pids)   # [G, r]
        b_g = self.embedding.growth_intercepts(pids)  # [G]

        ctx_state = self.context_agg(z, logw, a, log_m0)
        ctx = ctx_state.context    # [C]

        # Get program scores for ecology (if enabled)
        eta_z = self.context_agg.encoder.eta(z)   # [G, N, K]
        q = ctx_state.q                             # [K]
        s = ctx_state.s                             # [L]

        coeffs = self.coeff_nets(
            z=z,
            tau=tau,
            context=ctx,
            a=a,
            growth_intercept=b_g,
            eta_z=eta_z,
            q=q,
            s=s,
        )

        return coeffs, ctx_state

    def regularization(self, lambda_embed: float = 0.0) -> torch.Tensor:
        reg = self.embedding.regularization(lambda_embed=lambda_embed)
        reg = reg + self.coeff_nets.regularization()
        return reg

    def growth_bias_regularization(self, lambda_growth_bias: float = 0.0) -> torch.Tensor:
        return self.embedding.growth_bias_regularization(lambda_growth_bias=lambda_growth_bias)

    def freeze_embeddings(self) -> None:
        """Freeze perturbation embeddings (for control warm-start stage)."""
        if self.embedding.embeddings is not None:
            self.embedding.embeddings.requires_grad_(False)
        self.freeze_control_reference()

    def unfreeze_embeddings(self) -> None:
        if self.embedding.embeddings is not None:
            self.embedding.embeddings.requires_grad_(True)
        self.unfreeze_control_reference()

    def freeze_control_reference(self) -> None:
        self.embedding.freeze_reference()

    def unfreeze_control_reference(self) -> None:
        self.embedding.unfreeze_reference()

    def freeze_ecology(self) -> None:
        if self.coeff_nets.ecology is not None:
            for p in self.coeff_nets.ecology.parameters():
                p.requires_grad_(False)

    def unfreeze_ecology(self) -> None:
        if self.coeff_nets.ecology is not None:
            for p in self.coeff_nets.ecology.parameters():
                p.requires_grad_(True)
