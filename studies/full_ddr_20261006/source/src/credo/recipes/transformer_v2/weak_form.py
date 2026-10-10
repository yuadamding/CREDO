"""Frozen normalized-law weak-form residual used by transformer-SDE v2."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...runtime import LossReport, RuntimeState


class GaussianRBFTestFunctions:
    def __init__(self, centers: torch.Tensor, bandwidth: float) -> None:
        self.centers = centers
        self.bandwidth = bandwidth

    def _terms(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        difference = z.unsqueeze(-2) - self.centers
        squared_distance = difference.square().sum(-1)
        psi = torch.exp(-squared_distance / (2 * self.bandwidth**2))
        return difference, psi

    def psi(self, z: torch.Tensor) -> torch.Tensor:
        return self._terms(z)[1]

    def generator(
        self,
        z: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
        centered_growth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        difference, psi = self._terms(z)
        h2 = self.bandwidth**2
        drift_term = -(psi / h2) * (difference * drift.unsqueeze(-2)).sum(-1)
        diffusion_squared = diffusion.square()
        hessian = (diffusion_squared.unsqueeze(-2) * difference.square()).sum(
            -1
        ) / h2**2 - diffusion_squared.sum(-1).unsqueeze(-1) / h2
        diffusion_term = 0.5 * psi * hessian
        growth_term = centered_growth.unsqueeze(-1) * psi
        return psi, drift_term + diffusion_term + growth_term


class WeakFormLoss(nn.Module):
    def __init__(
        self,
        n_test_functions: int = 12,
        bandwidth: float = 1.0,
        latent_dim: int = 50,
    ) -> None:
        super().__init__()
        self.n_test_functions = int(n_test_functions)
        self.bandwidth = float(bandwidth)
        self.latent_dim = int(latent_dim)
        self.register_buffer("_centers", torch.zeros(self.n_test_functions, self.latent_dim))
        self._centers_initialized = False

    def refresh(self, reference: torch.Tensor) -> None:
        flattened = reference.detach().float().reshape(-1, reference.shape[-1])
        minimum = flattened.min(0).values
        maximum = flattened.max(0).values
        self._centers = minimum + torch.rand(
            self.n_test_functions,
            self.latent_dim,
            device=reference.device,
        ) * (maximum - minimum)
        subset = flattened[
            torch.randperm(flattened.shape[0], device=flattened.device)[
                : min(512, flattened.shape[0])
            ]
        ]
        distances = (subset.unsqueeze(1) - self._centers.unsqueeze(0)).square().sum(-1)
        self._adaptive_bandwidth = max(distances.median().sqrt().item(), 1.0)
        self._centers_initialized = True

    def forward(
        self,
        z_steps: torch.Tensor,
        logw_steps: torch.Tensor,
        drift_steps: torch.Tensor,
        diffusion_steps: torch.Tensor,
        growth_steps: torch.Tensor,
        axis_grid: torch.Tensor,
        *,
        refresh_centers: bool = True,
    ) -> torch.Tensor:
        z_steps = z_steps.float()
        logw_steps = logw_steps.float()
        drift_steps = drift_steps.float()
        diffusion_steps = diffusion_steps.float()
        growth_steps = growth_steps.float()
        axis_grid = axis_grid.float()
        if refresh_centers or not self._centers_initialized:
            self.refresh(z_steps[0])
        tests = GaussianRBFTestFunctions(
            self._centers.to(z_steps),
            getattr(self, "_adaptive_bandwidth", self.bandwidth),
        )
        normalized = (logw_steps - torch.logsumexp(logw_steps, dim=-1, keepdim=True)).exp()
        previous = torch.einsum("gn,gnm->gm", normalized[0], tests.psi(z_steps[0]))
        residual = z_steps.new_zeros(())
        for step in range(len(axis_grid) - 1):
            growth = growth_steps[step]
            centered_growth = growth - (normalized[step] * growth).sum(-1, keepdim=True)
            _, generator = tests.generator(
                z_steps[step],
                drift_steps[step],
                diffusion_steps[step],
                centered_growth,
            )
            expected_generator = torch.einsum("gn,gnm->gm", normalized[step], generator)
            following = torch.einsum(
                "gn,gnm->gm", normalized[step + 1], tests.psi(z_steps[step + 1])
            )
            derivative = (following - previous) / (axis_grid[step + 1] - axis_grid[step])
            residual = residual + (derivative - expected_generator).square().mean()
            previous = following
        return residual / (len(axis_grid) - 1)


class WeakFormResidual:
    name = "weak_form_residual"
    requires = frozenset({"drift", "diffusion", "growth"})

    def __init__(self, loss: WeakFormLoss) -> None:
        self.loss = loss

    def compute(self, rollout, study, runtime_state: RuntimeState) -> LossReport:
        del study
        value = self.loss(
            rollout.z_steps,
            rollout.logw_steps,
            rollout.drift_steps,
            rollout.diffusion_steps,
            rollout.growth_steps,
            rollout.axis_grid,
            refresh_centers=bool(runtime_state.values.get("refresh_weak_centers", True)),
        )
        return LossReport(self.name, value)


__all__ = ["GaussianRBFTestFunctions", "WeakFormLoss", "WeakFormResidual"]
