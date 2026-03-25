from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from camfnd.numerics.particles_torch import TorchParticleState


@dataclass(frozen=True, slots=True)
class ContextMapConfig:
    sharpness: float = 6.0
    bias: float = 0.0
    learnable: bool = False

    def validate(self) -> None:
        if self.sharpness <= 0:
            raise ValueError("sharpness must be positive.")


class OccupancyContextMap(nn.Module):
    """Screen-level soft right-occupancy context map.

    For a screen/sample s, this computes
        c_{s,t} = (1 / Mbar_s(t)) * sum_g int h(z) mu^g_{s,t}(dz)
    with h(z) = sigmoid(sharpness * (z - bias)).

    This is the interpretable context summary used by the Stage-II benchmark.
    """

    def __init__(self, config: ContextMapConfig | None = None) -> None:
        super().__init__()
        config = config or ContextMapConfig()
        config.validate()
        self.config = config
        if config.learnable:
            sharpness_raw = torch.log(torch.expm1(torch.tensor(float(config.sharpness), dtype=torch.float64)))
            self.sharpness_raw = nn.Parameter(sharpness_raw.clone())
            self.bias = nn.Parameter(torch.tensor(float(config.bias), dtype=torch.float64))
        else:
            self.register_buffer('sharpness_raw', torch.log(torch.expm1(torch.tensor(float(config.sharpness), dtype=torch.float64))), persistent=True)
            self.register_buffer('bias', torch.tensor(float(config.bias), dtype=torch.float64), persistent=True)

    @property
    def sharpness(self) -> Tensor:
        return torch.nn.functional.softplus(self.sharpness_raw)

    def occupancy(self, z: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[1] != 1:
            raise ValueError('OccupancyContextMap expects z with shape [N, 1].')
        return torch.sigmoid(self.sharpness * (z - self.bias))

    def forward(self, states: Dict[object, TorchParticleState]) -> Dict[str, Tensor]:
        by_sample: Dict[str, list[TorchParticleState]] = {}
        for state in states.values():
            by_sample.setdefault(state.sample_id, []).append(state)

        contexts: Dict[str, Tensor] = {}
        for sample_id, group in by_sample.items():
            numerator = None
            denominator = None
            for state in group:
                atom_weights = state.atom_weights().reshape(-1, 1)
                occ = self.occupancy(state.z)
                num = (atom_weights * occ).sum()
                den = atom_weights.sum()
                numerator = num if numerator is None else numerator + num
                denominator = den if denominator is None else denominator + den
            contexts[sample_id] = numerator / torch.clamp(denominator, min=1e-12)
        return contexts

    def regularization(self) -> Tensor:
        if not self.config.learnable:
            return torch.zeros((), dtype=torch.float64, device=self.bias.device)
        return (self.bias ** 2) + 1e-2 * (self.sharpness_raw ** 2)
