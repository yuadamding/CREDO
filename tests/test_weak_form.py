from __future__ import annotations

import pytest
import torch

from credo.losses.weak_form import WeakFormLoss


pytestmark = pytest.mark.unit


def test_constant_growth_does_not_move_normalized_law() -> None:
    torch.manual_seed(0)
    loss_fn = WeakFormLoss(n_test_functions=4, bandwidth=1.0, latent_dim=2)
    z0 = torch.randn(1, 6, 2)
    z_steps = torch.stack([z0, z0.clone()], dim=0)
    logw_steps = torch.zeros(2, 1, 6)
    drift_steps = torch.zeros(1, 1, 6, 2)
    sigma_steps = torch.zeros(1, 1, 6, 2)
    growth_steps = torch.full((1, 1, 6), 3.0)
    tau_steps = torch.tensor([0.0, 1.0])

    loss = loss_fn(
        z_steps,
        logw_steps,
        drift_steps,
        sigma_steps,
        growth_steps,
        tau_steps,
    )

    assert float(loss) < 1e-8
