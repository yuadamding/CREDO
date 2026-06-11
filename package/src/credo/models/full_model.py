"""Full dynamics model wrapping embeddings, context, and coefficients.

FullDynamicsModel.step() is called by the simulator at each time step.
It computes the ecological context, then returns Coefficients and ContextState.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .embeddings import PerturbationEmbedding, TimeEmbedding
from .context import ContextAggregator, ContextState
from .coefficients import CoefficientNetworks, Coefficients
from .causal_context import CausalEcologicalAttentionContext
from .transformer_context import MassAwareTransformerContextAggregator
from .interventions import CausalAttentionIntervention


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
        context_kind: Literal["mlp", "transformer", "causal_attention"] = "mlp",
        transformer_token_dim: int = 64,
        transformer_heads: int = 4,
        transformer_within_layers: int = 1,
        transformer_cross_layers: int = 1,
        transformer_inducing: int = 8,
        transformer_dropout: float = 0.05,
        mass_attention_temperature: float = 0.5,
        transformer_growth_only: bool = True,
        causal_token_dim: int = 64,
        causal_heads: int = 4,
        causal_n_mediators: int = 12,
        causal_dropout: float = 0.05,
        causal_mass_attention_temperature: float = 0.5,
        causal_growth_only: bool = True,
        causal_sparse_edges: bool = True,
        causal_residual_policy: Literal["edges_only", "tokens_and_edges"] = "edges_only",
    ) -> None:
        super().__init__()
        self.perturbation_ids = perturbation_ids
        self.control_ids = set(control_ids)
        self.control_mode = control_mode
        self.context_kind = context_kind
        if context_kind == "causal_attention" and not causal_sparse_edges:
            raise ValueError(
                "causal_attention requires causal_sparse_edges=True. "
                "Dense mediator attention is not intervention-addressable CEA."
            )
        self.causal_growth_only = bool(causal_growth_only)
        self.transformer_growth_only = (
            bool(causal_growth_only)
            if context_kind == "causal_attention"
            else bool(transformer_growth_only)
        )
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

        if context_kind == "mlp":
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
            self.meanfield_context_agg = None
        elif context_kind == "transformer":
            self.context_agg = MassAwareTransformerContextAggregator(
                latent_dim=latent_dim,
                embedding_dim=embedding_dim,
                n_programs=self.n_programs,
                mediator_dim=mediator_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                token_dim=transformer_token_dim,
                n_heads=transformer_heads,
                n_within_layers=transformer_within_layers,
                n_cross_layers=transformer_cross_layers,
                n_inducing=transformer_inducing,
                dropout=transformer_dropout,
                fixed_program_centroids=program_centroids,
                program_assignment_scale=program_assignment_scale,
                activation_checkpointing=activation_checkpointing,
                mass_attention_temperature=mass_attention_temperature,
            )
            if self.transformer_growth_only:
                self.meanfield_context_agg = ContextAggregator(
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
                self.meanfield_context_agg.encoder = self.context_agg.program_encoder
            else:
                self.meanfield_context_agg = None
        elif context_kind == "causal_attention":
            self.context_agg = CausalEcologicalAttentionContext(
                latent_dim=latent_dim,
                embedding_dim=embedding_dim,
                n_programs=self.n_programs,
                mediator_dim=mediator_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                token_dim=causal_token_dim,
                n_heads=causal_heads,
                n_mediators=causal_n_mediators,
                dropout=causal_dropout,
                mass_attention_temperature=causal_mass_attention_temperature,
                fixed_program_centroids=program_centroids,
                program_assignment_scale=program_assignment_scale,
                activation_checkpointing=activation_checkpointing,
                use_sparse_edges=causal_sparse_edges,
                residual_policy=causal_residual_policy,
            )
            if self.transformer_growth_only:
                self.meanfield_context_agg = ContextAggregator(
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
                self.meanfield_context_agg.encoder = self.context_agg.program_encoder
            else:
                self.meanfield_context_agg = None
        else:
            raise ValueError(f"Unknown context_kind {context_kind!r}")

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

    def _coerce_context_override(
        self,
        context_override: Any,
        *,
        z: torch.Tensor,
        logw: torch.Tensor,
        a: torch.Tensor,
        log_m0: torch.Tensor,
        tau: torch.Tensor,
    ) -> ContextState:
        """Convert an external context override into a ContextState."""
        def _move_optional(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.to(device=z.device, dtype=z.dtype)
            return value

        if isinstance(context_override, ContextState):
            return ContextState(
                q=_move_optional(context_override.q),
                s=_move_optional(context_override.s),
                context=_move_optional(context_override.context),
                mass_g=_move_optional(context_override.mass_g),
                freq_g=_move_optional(context_override.freq_g),
                log_mass_g=_move_optional(context_override.log_mass_g),
                log_total_mass=_move_optional(context_override.log_total_mass),
                diagnostics=context_override.diagnostics,
                base_context=_move_optional(context_override.base_context),
                growth_context=_move_optional(context_override.growth_context),
            )
        if isinstance(context_override, dict):
            if "context_state" in context_override:
                state = context_override["context_state"]
                if not isinstance(state, ContextState):
                    raise TypeError("context_override['context_state'] must be a ContextState.")
                return state
            if "context" not in context_override:
                raise KeyError("context_override dict must contain 'context' or 'context_state'.")
            context = context_override["context"]
            q = context_override.get("q")
            s = context_override.get("s")
            base_context = context_override.get("base_context")
            growth_context = context_override.get("growth_context")
        else:
            context = context_override
            q = None
            s = None
            base_context = None
            growth_context = None

        if not torch.is_tensor(context):
            raise TypeError("context_override must be a ContextState, dict, or torch.Tensor.")
        context = context.to(device=z.device, dtype=z.dtype)
        if context.ndim != 1:
            raise ValueError("Tensor context_override must have shape [context_dim].")
        expected = self.n_programs + self.mediator_dim
        if q is None or s is None:
            if context.shape[-1] != expected:
                raise ValueError(
                    "Tensor context_override without explicit q/s requires identity context "
                    f"width {expected}, got {context.shape[-1]}."
                )
            q = context[: self.n_programs]
            s = context[self.n_programs :]
        q = q.to(device=z.device, dtype=z.dtype) if torch.is_tensor(q) else torch.as_tensor(q, device=z.device, dtype=z.dtype)
        s = s.to(device=z.device, dtype=z.dtype) if torch.is_tensor(s) else torch.as_tensor(s, device=z.device, dtype=z.dtype)
        if base_context is not None:
            base_context = base_context.to(device=z.device, dtype=z.dtype)
        if growth_context is not None:
            growth_context = growth_context.to(device=z.device, dtype=z.dtype)

        # Preserve mass diagnostics from the current generated particles while
        # replacing the ecological feature context.
        logw_abs32 = log_m0.float()[:, None] + logw.float()
        log_mass_g = torch.logsumexp(logw_abs32, dim=1)
        log_total_mass = torch.logsumexp(log_mass_g, dim=0)
        freq_g = torch.exp(log_mass_g - log_total_mass).to(dtype=z.dtype)
        mass_g = torch.exp(torch.clamp(log_mass_g, min=-30.0, max=30.0)).to(dtype=z.dtype)
        return ContextState(
            q=q,
            s=s,
            context=context,
            mass_g=mass_g,
            freq_g=freq_g,
            log_mass_g=log_mass_g,
            log_total_mass=log_total_mass,
            base_context=base_context,
            growth_context=growth_context,
        )

    def step(
        self,
        z: torch.Tensor,      # [G, N, d]
        tau: torch.Tensor,    # scalar
        logw: torch.Tensor,   # [G, N]
        log_m0: torch.Tensor, # [G]
        perturbation_ids: Optional[List[str]] = None,
        intervention: Optional[CausalAttentionIntervention] = None,
        context_override: Any = None,
    ) -> Tuple[Coefficients, ContextState]:
        """One step of the dynamics: compute context and coefficients."""
        pids = perturbation_ids or self.perturbation_ids
        a = self.embedding(pids)   # [G, r]
        delta = self.embedding.residuals(pids)
        b_g = self.embedding.growth_intercepts(pids)  # [G]

        if context_override is not None:
            ctx_state = self._coerce_context_override(
                context_override,
                z=z,
                logw=logw,
                a=a,
                log_m0=log_m0,
                tau=tau,
            )
        elif self.context_kind == "causal_attention":
            ctx_state = self.context_agg(
                z,
                logw,
                a,
                log_m0,
                tau=tau,
                residual=delta,
                intervention=intervention,
            )
        else:
            ctx_state = self.context_agg(z, logw, a, log_m0, tau=tau)
        ctx = ctx_state.context    # [C]
        base_context = getattr(ctx_state, "base_context", None)
        if base_context is None:
            base_context = ctx
        growth_context = getattr(ctx_state, "growth_context", None)
        if self.context_kind == "causal_attention" and not self.causal_growth_only and growth_context is not None:
            base_context = growth_context
        if context_override is None and self.transformer_growth_only and self.meanfield_context_agg is not None:
            base_state = self.meanfield_context_agg(z, logw, a, log_m0, tau=tau)
            base_context = base_state.context
            if growth_context is None:
                growth_context = ctx
        ctx_state.base_context = base_context
        ctx_state.growth_context = growth_context if growth_context is not None else base_context

        # Get program scores for ecology (if enabled)
        eta_z, _ = self.context_agg.encode_particles(z)   # [G, N, K]
        q = ctx_state.q                             # [K]
        s = ctx_state.s                             # [L]

        coeffs = self.coeff_nets(
            z=z,
            tau=tau,
            context=base_context,
            a=a,
            growth_intercept=b_g,
            eta_z=eta_z,
            q=q,
            s=s,
            growth_context=ctx_state.growth_context,
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
