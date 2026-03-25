from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from camfnd.data.contract import PerturbationCatalog
from camfnd.models.embeddings import ControlAnchoredEmbeddingStore
from camfnd.models.context_map import ContextMapConfig, OccupancyContextMap


# ---------------------------------------------------------------------------
# Shared scalar field used by both Stage-I and Stage-II models
# ---------------------------------------------------------------------------

class ControlAnchoredScalarField(nn.Module):
    """Scalar beta + <B, a_g> with an exact zero control anchor via a_g = 0."""

    def __init__(self, embedding_dim: int, init_baseline: float = 0.0) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.baseline = nn.Parameter(torch.tensor([[init_baseline]], dtype=torch.float64))
        self.modulation = nn.Parameter(torch.zeros(1, self.embedding_dim, dtype=torch.float64))

    def forward(self, embedding: Tensor, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        if embedding.ndim != 1 or embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding must have shape [{self.embedding_dim}], got {tuple(embedding.shape)}"
            )
        value = self.baseline + (self.modulation * embedding[None, :]).sum(dim=1, keepdim=True)
        value = value.repeat(batch_size, 1)
        baseline = self.baseline.repeat(batch_size, 1)
        modulation = self.modulation.repeat(batch_size, 1)
        return value, baseline, modulation


# ---------------------------------------------------------------------------
# Stage-I model (no context coupling)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stage1CoefficientConfig:
    embedding_dim: int = 2
    hidden_dim: int = 16  # retained for API compatibility
    depth: int = 1        # retained for API compatibility
    time_frequencies: int = 0  # retained for API compatibility
    sigma_min: float = 0.02
    r_max: float = 2.0
    shared_diffusion: bool = False
    use_growth: bool = True

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive.")
        if self.r_max <= 0:
            raise ValueError("r_max must be positive.")


class ControlAnchoredStage1Model(nn.Module):
    """Structured no-context Step-3 model specialized to the Stage-I benchmark.

    The fields are deliberately simple in Step 3:
    - drift uses a shared mean-reversion rate kappa and perturbation-specific theta_g
    - diffusion uses perturbation-specific sigma_g
    - growth uses perturbation-specific r_g

    This keeps the benchmark interpretable, guarantees an exact control anchor,
    and makes the required ablations mechanistically meaningful.
    """

    def __init__(self, catalog: PerturbationCatalog, config: Stage1CoefficientConfig) -> None:
        super().__init__()
        catalog.validate()
        config.validate()
        self.catalog = catalog
        self.config = config
        self.embedding_store = ControlAnchoredEmbeddingStore(catalog, embedding_dim=config.embedding_dim)
        self.theta_field = ControlAnchoredScalarField(config.embedding_dim, init_baseline=0.0)
        self.sigma_field = ControlAnchoredScalarField(config.embedding_dim, init_baseline=0.15)
        self.growth_field = ControlAnchoredScalarField(config.embedding_dim, init_baseline=0.0)
        self.kappa_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def coefficients(self, z: Tensor, t: Tensor, perturbation_id: str) -> Dict[str, Tensor]:
        if z.ndim != 2 or z.shape[1] != 1:
            raise ValueError("Step 3 reference implementation currently supports z with shape [N, 1].")
        batch_size = z.shape[0]
        embedding = self.embedding_store.forward_one(perturbation_id)
        theta_raw, theta_baseline, theta_mod = self.theta_field(embedding, batch_size)

        if self.config.shared_diffusion:
            zero_embedding = torch.zeros_like(embedding)
            sigma_raw, sigma_baseline, sigma_mod = self.sigma_field(zero_embedding, batch_size)
        else:
            sigma_raw, sigma_baseline, sigma_mod = self.sigma_field(embedding, batch_size)
        diffusion = torch.nn.functional.softplus(sigma_raw) + self.config.sigma_min

        if self.config.use_growth:
            growth_raw, growth_baseline, growth_mod = self.growth_field(embedding, batch_size)
            growth = self.config.r_max * torch.tanh(growth_raw)
        else:
            growth_raw = torch.zeros(batch_size, 1, dtype=z.dtype, device=z.device)
            growth_baseline = torch.zeros_like(growth_raw)
            growth_mod = torch.zeros(batch_size, self.config.embedding_dim, dtype=z.dtype, device=z.device)
            growth = torch.zeros_like(growth_raw)

        kappa = torch.nn.functional.softplus(self.kappa_raw)
        drift = kappa * (theta_raw - z)
        return {
            "drift": drift,
            "diffusion": diffusion,
            "growth": growth,
            "theta": theta_raw,
            "sigma_raw": sigma_raw,
            "growth_raw": growth_raw,
            "theta_baseline": theta_baseline,
            "theta_modulation": theta_mod,
            "sigma_baseline": sigma_baseline,
            "sigma_modulation": sigma_mod,
            "growth_baseline": growth_baseline,
            "growth_modulation": growth_mod,
            "kappa": kappa.reshape(1, 1),
        }

    def regularization_terms(self) -> Dict[str, Tensor]:
        reg_emb = self.embedding_store.regularization()
        reg_mod = torch.stack([
            (self.theta_field.modulation ** 2).mean(),
            (self.sigma_field.modulation ** 2).mean(),
            (self.growth_field.modulation ** 2).mean(),
        ]).mean()
        reg_nn = torch.stack([
            (self.theta_field.baseline ** 2).mean(),
            (self.sigma_field.baseline ** 2).mean(),
            (self.growth_field.baseline ** 2).mean(),
            self.kappa_raw.pow(2),
        ]).mean()
        return {
            "emb": reg_emb,
            "mod": reg_mod,
            "disp": reg_nn,
            "nn": reg_nn,
        }

    def control_anchor_is_exact(self) -> bool:
        return self.embedding_store.control_anchor_is_exact(atol=0.0)


# ---------------------------------------------------------------------------
# Stage-II model (with screen-level context coupling)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stage2CoefficientConfig:
    embedding_dim: int = 4
    sigma_min: float = 0.02
    r_max: float = 2.0
    shared_diffusion: bool = False
    use_growth: bool = True
    use_context: bool = True
    context_map: ContextMapConfig = ContextMapConfig()

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError('embedding_dim must be positive.')
        if self.sigma_min <= 0:
            raise ValueError('sigma_min must be positive.')
        if self.r_max <= 0:
            raise ValueError('r_max must be positive.')
        self.context_map.validate()


class ControlAnchoredStage2Model(nn.Module):
    """Structured context-aware Stage-II model.

    The model keeps Step 3's exact control anchor and separate drift / diffusion / growth heads,
    then adds a sample-aware screen context term through
        drift = kappa * (theta_g - z) + eta * c_{s,t}.
    """

    def __init__(self, catalog: PerturbationCatalog, config: Stage2CoefficientConfig) -> None:
        super().__init__()
        catalog.validate()
        config.validate()
        self.catalog = catalog
        self.config = config
        self.embedding_store = ControlAnchoredEmbeddingStore(catalog, embedding_dim=config.embedding_dim)
        self.theta_field = ControlAnchoredScalarField(config.embedding_dim, init_baseline=0.0)
        self.sigma_field = ControlAnchoredScalarField(config.embedding_dim, init_baseline=0.15)
        self.growth_field = ControlAnchoredScalarField(config.embedding_dim, init_baseline=0.0)
        self.kappa_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.eta_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.context_map = OccupancyContextMap(config.context_map)

    def context_values(self, particles) -> Dict[str, Tensor]:
        if not self.config.use_context:
            sample_ids = sorted({state.sample_id for state in particles.values()})
            device = next(iter(particles.values())).z.device
            dtype = next(iter(particles.values())).z.dtype
            return {sample_id: torch.zeros((), dtype=dtype, device=device) for sample_id in sample_ids}
        return self.context_map(particles)

    def coefficients(self, z: Tensor, t: Tensor, perturbation_id: str, context_scalar: Tensor) -> Dict[str, Tensor]:
        if z.ndim != 2 or z.shape[1] != 1:
            raise ValueError('Step 4 reference implementation currently supports z with shape [N, 1].')
        batch_size = z.shape[0]
        embedding = self.embedding_store.forward_one(perturbation_id)
        theta_raw, theta_baseline, theta_mod = self.theta_field(embedding, batch_size)

        if self.config.shared_diffusion:
            zero_embedding = torch.zeros_like(embedding)
            sigma_raw, sigma_baseline, sigma_mod = self.sigma_field(zero_embedding, batch_size)
        else:
            sigma_raw, sigma_baseline, sigma_mod = self.sigma_field(embedding, batch_size)
        diffusion = torch.nn.functional.softplus(sigma_raw) + self.config.sigma_min

        if self.config.use_growth:
            growth_raw, growth_baseline, growth_mod = self.growth_field(embedding, batch_size)
            growth = self.config.r_max * torch.tanh(growth_raw)
        else:
            growth_raw = torch.zeros(batch_size, 1, dtype=z.dtype, device=z.device)
            growth_baseline = torch.zeros_like(growth_raw)
            growth_mod = torch.zeros(batch_size, self.config.embedding_dim, dtype=z.dtype, device=z.device)
            growth = torch.zeros_like(growth_raw)

        kappa = torch.nn.functional.softplus(self.kappa_raw)
        if self.config.use_context:
            eta = torch.nn.functional.softplus(self.eta_raw)
        else:
            eta = torch.zeros((), dtype=z.dtype, device=z.device)
        context_term = context_scalar.reshape(1, 1).repeat(batch_size, 1)
        drift = kappa * (theta_raw - z) + eta * context_term
        return {
            'drift': drift,
            'diffusion': diffusion,
            'growth': growth,
            'theta': theta_raw,
            'sigma_raw': sigma_raw,
            'growth_raw': growth_raw,
            'theta_baseline': theta_baseline,
            'theta_modulation': theta_mod,
            'sigma_baseline': sigma_baseline,
            'sigma_modulation': sigma_mod,
            'growth_baseline': growth_baseline,
            'growth_modulation': growth_mod,
            'kappa': kappa.reshape(1, 1),
            'eta': eta.reshape(1, 1),
            'context': context_term,
        }

    def regularization_terms(self) -> Dict[str, Tensor]:
        reg_emb = self.embedding_store.regularization()
        reg_mod = torch.stack([
            (self.theta_field.modulation ** 2).mean(),
            (self.sigma_field.modulation ** 2).mean(),
            (self.growth_field.modulation ** 2).mean(),
        ]).mean()
        reg_nn = torch.stack([
            (self.theta_field.baseline ** 2).mean(),
            (self.sigma_field.baseline ** 2).mean(),
            (self.growth_field.baseline ** 2).mean(),
            self.kappa_raw.pow(2),
            self.eta_raw.pow(2),
        ]).mean()
        reg_context = self.context_map.regularization()
        return {
            'emb': reg_emb,
            'mod': reg_mod,
            'disp': reg_nn,
            'nn': reg_nn,
            'context': reg_context,
        }

    def control_anchor_is_exact(self) -> bool:
        return self.embedding_store.control_anchor_is_exact(atol=0.0)
