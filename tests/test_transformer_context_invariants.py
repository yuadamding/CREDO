from __future__ import annotations

import numpy as np
import torch
import pytest

from credo.config.schema import ModelConfig, RunConfig, SimulationConfig, TrainingConfig
from credo.data.core import EndpointProblem, FiniteMeasure, TimeAxis
from credo.models.full_model import FullDynamicsModel
from credo.models.simulator import _control_embedding_context, rollout_with_clamped_context
from credo.models.transformer_context import MassAwareTransformerContextAggregator
from credo.models.weighted_sde import WeightedParticleSimulator
from credo.training.trainer import Trainer


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


def test_transformer_context_is_group_permutation_invariant() -> None:
    agg = _aggregator()
    z, logw, a, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=0.4)
    perm = torch.tensor([2, 0, 1])
    out_perm = agg(z[perm], logw[perm], a[perm], log_m0[perm], tau=0.4)

    assert torch.allclose(out.context, out_perm.context, atol=1e-5)
    assert torch.allclose(out.q, out_perm.q, atol=1e-5)
    assert torch.allclose(out.s, out_perm.s, atol=1e-5)
    assert torch.allclose(out_perm.mass_g, out.mass_g[perm], atol=1e-6)
    assert torch.allclose(out_perm.freq_g, out.freq_g[perm], atol=1e-6)


def test_global_mass_shift_does_not_change_transformer_context() -> None:
    agg = _aggregator()
    z, logw, a, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0)
    shifted = agg(z, logw, a, log_m0 + 7.0)

    assert torch.allclose(out.context, shifted.context, atol=1e-5)
    assert torch.allclose(out.q, shifted.q, atol=1e-5)
    assert torch.allclose(out.s, shifted.s, atol=1e-5)
    assert torch.allclose(out.freq_g, shifted.freq_g, atol=1e-6)
    assert torch.allclose(
        shifted.mass_g,
        out.mass_g * torch.exp(torch.tensor(7.0)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_low_mass_extra_group_has_small_context_effect() -> None:
    agg = _aggregator()
    z, logw, a, log_m0 = _inputs()
    out = agg(z, logw, a, log_m0)

    N = z.shape[1]
    z_extra = torch.cat([z, torch.randn(1, N, z.shape[2])], dim=0)
    logw_extra = torch.cat(
        [
            logw,
            torch.full((1, N), -torch.log(torch.tensor(float(N)))),
        ],
        dim=0,
    )
    a_extra = torch.cat([a, torch.randn(1, a.shape[1])], dim=0)
    log_m0_extra = torch.cat([log_m0, torch.tensor([-50.0])], dim=0)

    out_extra = agg(z_extra, logw_extra, a_extra, log_m0_extra)

    assert torch.allclose(out.context, out_extra.context, atol=1e-3)
    assert out_extra.freq_g[-1] < 1e-20


def test_all_transformer_context_parameters_get_gradients() -> None:
    agg = _aggregator()
    agg.train()
    z, logw, a, log_m0 = _inputs()

    out = agg(
        z.requires_grad_(True),
        logw.requires_grad_(True),
        a.requires_grad_(True),
        log_m0.requires_grad_(True),
        tau=0.3,
    )
    loss = out.context.square().sum() + out.mass_g.log().square().sum()
    loss.backward()

    missing = [
        name
        for name, param in agg.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not missing, missing
    assert all(torch.isfinite(param.grad).all() for param in agg.parameters())


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


def test_growth_only_transformer_does_not_route_transformer_context_to_drift_sigma() -> None:
    torch.manual_seed(31)
    model_a = FullDynamicsModel(
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
    model_b = FullDynamicsModel(
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
    model_b.load_state_dict(model_a.state_dict())
    with torch.no_grad():
        for param in model_b.context_agg.parameters():
            param.add_(0.5 * torch.randn_like(param))

    z = torch.randn(2, 4, 2)
    logw = torch.full((2, 4), -torch.log(torch.tensor(4.0)))
    log_m0 = torch.tensor([0.0, 0.7])
    coeffs_a, _ = model_a.step(z, torch.tensor(0.2), logw, log_m0, ["ctrl", "gene_a"])
    coeffs_b, _ = model_b.step(z, torch.tensor(0.2), logw, log_m0, ["ctrl", "gene_a"])

    assert torch.allclose(coeffs_a.drift, coeffs_b.drift, atol=1e-6)
    assert torch.allclose(coeffs_a.sigma_diag, coeffs_b.sigma_diag, atol=1e-6)
    assert not torch.allclose(coeffs_a.growth, coeffs_b.growth, atol=1e-5)


def test_transformer_counterfactual_rollouts_share_start_and_noise() -> None:
    torch.manual_seed(41)
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
    with torch.no_grad():
        local_idx = model.embedding._nc_to_local["gene_a"]
        model.embedding.embeddings[local_idx].copy_(torch.tensor([0.5, -0.3, 0.1, 0.2]))

    z0 = torch.randn(1, 4, 2)
    logw0 = torch.full((1, 4), -torch.log(torch.tensor(4.0)))
    log_m0 = torch.tensor([0.2])
    noise_steps = torch.randn(2, 1, 4, 2)
    simulator = WeightedParticleSimulator(n_steps=2, store_history=True)

    factual = simulator.rollout(
        z0=z0,
        logw0=logw0,
        model=model,
        log_m0=log_m0,
        perturbation_ids=["gene_a"],
        noise_steps=noise_steps,
        return_noise_used=True,
    )
    with _control_embedding_context(model, "gene_a", mode="reference_consistent"):
        reference_embedding = model.embedding(["gene_a"])[0]
        assert torch.allclose(reference_embedding, model.embedding.reference_embedding)
        reference = simulator.rollout(
            z0=z0.clone(),
            logw0=logw0.clone(),
            model=model,
            log_m0=log_m0.clone(),
            perturbation_ids=["gene_a"],
            noise_steps=noise_steps.clone(),
            return_noise_used=True,
        )

    assert torch.equal(factual.z_steps[0], reference.z_steps[0])
    assert torch.equal(factual.logw_steps[0], reference.logw_steps[0])
    assert torch.equal(factual.log_m0, reference.log_m0)
    assert factual.noise_steps is not None
    assert reference.noise_steps is not None
    assert torch.equal(factual.noise_steps, reference.noise_steps)


def test_chunked_transformer_training_uses_full_g_context(tmp_path) -> None:
    support = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
    weights = np.ones(3, dtype=np.float32)
    measure = FiniteMeasure(support=support, weights=weights, total_mass=float(weights.sum()))
    endpoint = EndpointProblem(
        initial={"ctrl": measure, "gene_a": measure, "gene_b": measure},
        terminal={"ctrl": measure, "gene_a": measure, "gene_b": measure},
        time_axis=TimeAxis(["t0", "t1"], [0.0, 1.0]),
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
    )
    model = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
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
    cfg = RunConfig(
        run_id="chunked-transformer-test",
        output_dir=str(tmp_path),
        device="cpu",
        model=ModelConfig(
            context_kind="transformer",
            transformer_token_dim=16,
            transformer_heads=4,
            transformer_within_layers=1,
            transformer_cross_layers=1,
            transformer_inducing=3,
            transformer_dropout=0.0,
            transformer_growth_only=True,
        ),
        simulation=SimulationConfig(n_particles=3, n_steps=1, store_history=True),
        training=TrainingConfig(
            epochs=1,
            max_active_perturbations=1,
            lambda_count=0.0,
            lambda_weak=0.0,
            lambda_reg_net=0.0,
            lambda_reg_diffusion=0.0,
            lambda_reg_embed=0.0,
            lambda_reg_growth_bias=0.0,
            lr_transformer=1e-4,
        ),
    )
    trainer = Trainer(
        model=model,
        config=cfg,
        endpoint=endpoint,
        supported_pids=["ctrl", "gene_a", "gene_b"],
        output_dir=str(tmp_path),
        ema_decay=0.0,
        warmup_epochs=1,
    )
    optimizer = trainer._build_optimizer(stage="all")
    lrs = sorted({group["lr"] for group in optimizer.param_groups})

    metrics = trainer._one_epoch_chunked(
        optimizer=optimizer,
        epoch=0,
        stage="all",
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
    )

    assert cfg.training.lr_transformer in lrs
    assert metrics["perturbation_batch_size"] == 1
    assert np.isfinite(metrics["loss_total"])
