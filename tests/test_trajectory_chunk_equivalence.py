from __future__ import annotations

import pytest
import torch

from credo.models.full_model import FullDynamicsModel
from credo.models.weighted_sde import WeightedParticleSimulator


pytestmark = pytest.mark.unit


def _model() -> FullDynamicsModel:
    return FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        context_kind="none",
        ecological_growth=False,
        control_mode="soft_ref",
    )


def _objective(rollout) -> torch.Tensor:
    return (
        rollout.terminal_z.square().sum()
        + 0.3 * rollout.terminal_logw.square().sum()
        + 0.1 * rollout.sigma_steps.square().sum()
    )


def test_intrinsic_full_rollout_equals_sum_of_measure_chunks() -> None:
    torch.manual_seed(4)
    full_model = _model()
    chunk_model = _model()
    chunk_model.load_state_dict(full_model.state_dict())
    simulator = WeightedParticleSimulator(n_steps=2, store_history=True)
    z0 = torch.randn(2, 3, 2)
    logw0 = torch.full((2, 3), -torch.log(torch.tensor(3.0)))
    log_m0 = torch.tensor([-0.2, 0.3])
    tau_grid = torch.tensor([0.0, 0.4, 1.0])
    noise = torch.randn(2, 2, 3, 2)

    full = simulator.rollout(
        z0,
        logw0,
        full_model,
        log_m0,
        tau_grid=tau_grid,
        perturbation_ids=["view_a", "view_b"],
        embedding_ids=["gene_a", "gene_b"],
        noise_steps=noise,
    )
    full_loss = _objective(full)
    full_loss.backward()

    chunks = []
    chunk_loss = torch.zeros(())
    for index, (view_id, embedding_id) in enumerate(
        [("view_a", "gene_a"), ("view_b", "gene_b")]
    ):
        rollout = simulator.rollout(
            z0[index:index + 1],
            logw0[index:index + 1],
            chunk_model,
            log_m0[index:index + 1],
            tau_grid=tau_grid,
            perturbation_ids=[view_id],
            embedding_ids=[embedding_id],
            noise_steps=noise[:, index:index + 1],
        )
        chunks.append(rollout)
        chunk_loss = chunk_loss + _objective(rollout)
    chunk_loss.backward()

    assert torch.allclose(
        full.terminal_z,
        torch.cat([rollout.terminal_z for rollout in chunks], dim=0),
        atol=1e-6,
    )
    assert torch.allclose(
        full.terminal_logw,
        torch.cat([rollout.terminal_logw for rollout in chunks], dim=0),
        atol=1e-6,
    )
    full_gradients = dict(full_model.named_parameters())
    chunk_gradients = dict(chunk_model.named_parameters())
    checked = 0
    for name in full_gradients:
        if not any(token in name for token in ["drift_head", "sigma_head", "embedding.embeddings"]):
            continue
        grad_full = full_gradients[name].grad
        grad_chunk = chunk_gradients[name].grad
        assert grad_full is not None and grad_chunk is not None
        assert torch.allclose(grad_full, grad_chunk, rtol=1e-5, atol=1e-6), name
        checked += 1
    assert checked > 0
