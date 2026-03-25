from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from camfnd.numerics.particles_torch import TorchParticleState


def _make_mlp(input_dim: int, hidden_dim: int, depth: int, output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")

    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(int(depth)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True, slots=True)
class MeanFieldContextConfig:
    latent_dim: int
    summary_dim: int = 16
    context_dim: int = 8
    summary_hidden_dim: int = 32
    summary_depth: int = 2
    context_hidden_dim: int = 32
    context_depth: int = 2
    use_context: bool = True

    def validate(self) -> None:
        for name in (
            "latent_dim",
            "summary_dim",
            "context_dim",
            "summary_hidden_dim",
            "summary_depth",
            "context_hidden_dim",
            "context_depth",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")


class MeanFieldContextMap(nn.Module):
    """Mass-weighted, sample-wise context map for the full model path.

    The map is permutation invariant within each sample and returns a context vector
    `c_t(sample_id)` built from a weighted mean of bounded particle features.
    """

    def __init__(self, config: MeanFieldContextConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.summary_net = _make_mlp(
            input_dim=config.latent_dim,
            hidden_dim=config.summary_hidden_dim,
            depth=config.summary_depth,
            output_dim=config.summary_dim,
        )
        self.context_net = _make_mlp(
            input_dim=config.summary_dim,
            hidden_dim=config.context_hidden_dim,
            depth=config.context_depth,
            output_dim=config.context_dim,
        )
        self.double()

    def summary_features(self, z: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"z must have shape [N, {self.config.latent_dim}], got {tuple(z.shape)}."
            )
        # Keep pooled summaries bounded, matching the PDF's emphasis on bounded context inputs.
        return torch.tanh(self.summary_net(z))

    def forward(self, states: Dict[object, TorchParticleState]) -> Dict[str, Tensor]:
        if not self.config.use_context:
            by_sample = sorted({state.sample_id for state in states.values()})
            device = next(iter(states.values())).z.device
            dtype = next(iter(states.values())).z.dtype
            zero = torch.zeros(self.config.context_dim, dtype=dtype, device=device)
            return {sample_id: zero.clone() for sample_id in by_sample}

        by_sample: Dict[str, list[TorchParticleState]] = {}
        for state in states.values():
            by_sample.setdefault(state.sample_id, []).append(state)

        contexts: Dict[str, Tensor] = {}
        for sample_id, group in by_sample.items():
            pooled_num = None
            pooled_den = None
            for state in group:
                atom_weights = state.atom_weights().reshape(-1, 1)
                features = self.summary_features(state.z)
                num = (atom_weights * features).sum(dim=0)
                den = atom_weights.sum()
                pooled_num = num if pooled_num is None else pooled_num + num
                pooled_den = den if pooled_den is None else pooled_den + den
            pooled = pooled_num / torch.clamp(pooled_den, min=1e-12)
            contexts[sample_id] = torch.tanh(self.context_net(pooled.reshape(1, -1))).reshape(-1)
        return contexts

    def regularization(self) -> Tensor:
        return torch.stack([
            parameter.pow(2).mean() for parameter in self.parameters()
        ]).mean()
