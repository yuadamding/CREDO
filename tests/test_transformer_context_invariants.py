from __future__ import annotations

import torch
import pytest

from credo.config.schema import ModelConfig
from credo.models.full_model import FullDynamicsModel
from credo.models.simulator import rollout_with_clamped_context
from credo.models.transformer_context import MassAwareTransformerContextAggregator
from credo.models.weighted_sde import WeightedParticleSimulator


def _aggregator() -> MassAwareTransformerContextAggregator:
    agg = MassAwareTransformerContextAggregator(
        latent_dim=2,
        embedding_dim=3,
        n_programs=4,
        mediator_dim=3,
        context_dim=7,
        hidden_dim=8,
        token_dim=16,
        n_heads=4,
        n_within_layers=1,
        n_cross_layers=1,
        n_inducing=3,
        dropout=0.0,
        mass_attention_temperature=0.7,
    )
    agg.eval()
    return agg


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)
    z = torch.randn(3, 5, 2)
    logw = torch.randn(3, 5) * 0.2 - torch.log(torch.tensor(5.0))
    a = torch.randn(3, 3)
    log_m0 = torch.tensor([-0.4, 1.2, 0.3])
    return z, logw, a, log_m0


def test_transformer_context_is_particle_permutation_invariant() -> None:
    agg = _aggregator()
    z, logw, a, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=torch.tensor(0.25))
    perms = [torch.tensor([2, 0, 4, 1, 3]), torch.tensor([4, 3, 2, 1, 0]), torch.tensor([1, 3, 0, 4, 2])]
    z_perm = torch.stack([z[g, perm] for g, perm in enumerate(perms)], dim=0)
    logw_perm = torch.stack([logw[g, perm] for g, perm in enumerate(perms)], dim=0)
    out_perm = agg(z_perm, logw_perm, a, log_m0, tau=torch.tensor(0.25))

    assert torch.allclose(out.q, out_perm.q, atol=1e-5)
    assert torch.allclose(out.s, out_perm.s, atol=1e-5)
    assert torch.allclose(out.context, out_perm.context, atol=1e-5)
    assert torch.allclose(out.mass_g, out_perm.mass_g, atol=1e-6)
    assert torch.allclose(out.freq_g, out_perm.freq_g, atol=1e-6)


def test_transformer_context_uses_absolute_mass_reductions() -> None:
    agg = _aggregator()
    z, logw, a, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0)
    expected_mass = torch.exp(log_m0 + torch.logsumexp(logw, dim=1))
    expected_freq = torch.softmax(torch.log(expected_mass), dim=0)

    assert torch.allclose(out.mass_g, expected_mass, atol=1e-6)
    assert torch.allclose(out.freq_g, expected_freq, atol=1e-6)
    assert torch.isclose(out.q.sum(), torch.tensor(1.0), atol=1e-6)


def test_transformer_context_invariant_to_stabilized_logw_shift() -> None:
    agg = _aggregator()
    z, logw, a, log_m0 = _inputs()
    shifts = torch.tensor([5.0, -2.0, 1.3])

    original = agg(z, logw, a, log_m0, tau=0.5)
    stabilized = agg(z, logw + shifts[:, None], a, log_m0 - shifts, tau=0.5)

    assert torch.allclose(original.q, stabilized.q, atol=1e-5)
    assert torch.allclose(original.s, stabilized.s, atol=1e-5)
    assert torch.allclose(original.context, stabilized.context, atol=1e-5)
    assert torch.allclose(original.mass_g, stabilized.mass_g, atol=1e-6)
    assert torch.allclose(original.freq_g, stabilized.freq_g, atol=1e-6)


def test_transformer_context_backpropagates_through_context_features() -> None:
    agg = _aggregator()
    agg.train()
    z, logw, a, log_m0 = _inputs()
    z = z.requires_grad_(True)
    logw = logw.requires_grad_(True)
    a = a.requires_grad_(True)
    log_m0 = log_m0.requires_grad_(True)

    out = agg(z, logw, a, log_m0, tau=torch.tensor(0.75))
    loss = out.context.square().sum() + 0.01 * out.mass_g.log().square().sum()
    loss.backward()

    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert logw.grad is not None and torch.isfinite(logw.grad).all()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert log_m0.grad is not None and torch.isfinite(log_m0.grad).all()
    parameter_grads = [p.grad for p in agg.parameters() if p.grad is not None]
    assert parameter_grads
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)


def test_transformer_context_mass_sensitivity_changes_frequency() -> None:
    agg = _aggregator()
    z, logw, a, log_m0 = _inputs()

    original = agg(z, logw, a, log_m0)
    shifted = agg(z, logw, a, log_m0 + torch.tensor([0.0, 2.0, 0.0]))

    assert shifted.mass_g[1] > original.mass_g[1]
    assert shifted.freq_g[1] > original.freq_g[1]
    assert torch.allclose(shifted.mass_g[[0, 2]], original.mass_g[[0, 2]], atol=1e-6)


def test_model_config_rejects_invalid_transformer_attention_shape() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(context_kind="transformer", transformer_token_dim=10, transformer_heads=4)


def test_full_dynamics_model_uses_transformer_context_backend() -> None:
    model = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=4,
        n_programs=3,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=True,
        control_mode="soft_ref",
        context_kind="transformer",
        transformer_token_dim=16,
        transformer_heads=4,
        transformer_within_layers=1,
        transformer_cross_layers=1,
        transformer_inducing=3,
        transformer_dropout=0.0,
        transformer_growth_only=True,
    )
    model.eval()
    z = torch.randn(2, 4, 2)
    logw = torch.full((2, 4), -torch.log(torch.tensor(4.0)))
    log_m0 = torch.tensor([0.0, 0.7])

    coeffs, ctx = model.step(
        z=z,
        tau=torch.tensor(0.2),
        logw=logw,
        log_m0=log_m0,
        perturbation_ids=["ctrl", "gene_a"],
    )

    assert coeffs.drift.shape == (2, 4, 2)
    assert coeffs.sigma_diag.shape == (2, 4, 2)
    assert coeffs.growth.shape == (2, 4)
    assert ctx.context.shape == (5,)
    assert torch.isfinite(ctx.context).all()
    assert torch.allclose(ctx.mass_g, torch.exp(log_m0), atol=1e-6)


def test_full_dynamics_model_uses_transformer_context_for_all_coefficients() -> None:
    model = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=4,
        n_programs=3,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=True,
        control_mode="soft_ref",
        context_kind="transformer",
        transformer_token_dim=16,
        transformer_heads=4,
        transformer_within_layers=1,
        transformer_cross_layers=1,
        transformer_inducing=3,
        transformer_dropout=0.0,
        transformer_growth_only=False,
    )
    model.eval()
    z = torch.randn(2, 4, 2)
    logw = torch.full((2, 4), -torch.log(torch.tensor(4.0)))
    log_m0 = torch.tensor([0.0, 0.7])

    coeffs, ctx = model.step(
        z=z,
        tau=torch.tensor(0.2),
        logw=logw,
        log_m0=log_m0,
        perturbation_ids=["ctrl", "gene_a"],
    )

    assert coeffs.drift.shape == (2, 4, 2)
    assert coeffs.sigma_diag.shape == (2, 4, 2)
    assert coeffs.growth.shape == (2, 4)
    assert torch.isfinite(coeffs.drift).all()
    assert torch.isfinite(coeffs.sigma_diag).all()
    assert torch.isfinite(coeffs.growth).all()
    assert torch.isfinite(ctx.context).all()


def test_transformer_growth_only_clamped_rollout_smoke() -> None:
    model = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=4,
        n_programs=3,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=True,
        control_mode="soft_ref",
        context_kind="transformer",
        transformer_token_dim=16,
        transformer_heads=4,
        transformer_within_layers=1,
        transformer_cross_layers=1,
        transformer_inducing=3,
        transformer_dropout=0.0,
        transformer_growth_only=True,
    )
    model.eval()
    z0 = torch.randn(1, 4, 2)
    logw0 = torch.full((1, 4), -torch.log(torch.tensor(4.0)))
    log_m0 = torch.tensor([0.3])
    simulator = WeightedParticleSimulator(n_steps=2, store_history=True)
    rollout = simulator.rollout(
        z0=z0,
        logw0=logw0,
        model=model,
        log_m0=log_m0,
        perturbation_ids=["gene_a"],
    )

    clamped = rollout_with_clamped_context(
        model=model,
        z0=z0,
        logw0=logw0,
        log_m0=log_m0,
        perturbation_ids=["gene_a"],
        context_steps=rollout.context_steps,
        n_steps=2,
    )

    assert clamped.terminal_z.shape == (1, 4, 2)
    assert clamped.terminal_logw.shape == (1, 4)
    assert torch.isfinite(clamped.terminal_z).all()
    assert torch.isfinite(clamped.terminal_logw).all()
