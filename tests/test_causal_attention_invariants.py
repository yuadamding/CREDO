from __future__ import annotations

import numpy as np
import pytest
import torch

from credo.config.schema import ModelConfig, RunConfig, SimulationConfig, TrainingConfig
from credo.data.core import EndpointProblem, FiniteMeasure, TimeAxis
from credo.models import (
    CausalAttentionIntervention,
    CausalEcologicalAttentionContext,
    CounterfactualEngine,
    FullDynamicsModel,
    WeightedParticleSimulator,
)
from credo.training.trainer import Trainer


def _aggregator() -> CausalEcologicalAttentionContext:
    agg = CausalEcologicalAttentionContext(
        latent_dim=2,
        embedding_dim=3,
        n_programs=4,
        mediator_dim=3,
        context_dim=7,
        hidden_dim=12,
        token_dim=16,
        n_heads=4,
        n_mediators=5,
        dropout=0.0,
        mass_attention_temperature=0.5,
    )
    agg.eval()
    return agg


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(19)
    z = torch.randn(3, 6, 2)
    logw = torch.randn(3, 6) * 0.15 - torch.log(torch.tensor(6.0))
    a = torch.randn(3, 3)
    residual = torch.randn(3, 3) * 0.2
    residual[0].zero_()
    log_m0 = torch.tensor([-0.2, 0.8, 0.1])
    return z, logw, a, residual, log_m0


def _measure(total_mass: float = 3.0) -> FiniteMeasure:
    return FiniteMeasure(
        support=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        weights=np.ones(3, dtype=np.float32),
        total_mass=total_mass,
    )


def _endpoint() -> EndpointProblem:
    measure = _measure()
    return EndpointProblem(
        initial={"ctrl": measure, "gene_a": measure, "gene_b": measure},
        terminal={"ctrl": measure, "gene_a": measure, "gene_b": measure},
        time_axis=TimeAxis(["t0", "t1"], [0.0, 1.0]),
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
    )


def _model(*, causal_growth_only: bool = True) -> FullDynamicsModel:
    model = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=4,
        n_programs=3,
        mediator_dim=2,
        hidden_dim=12,
        depth=1,
        ecological_growth=True,
        control_mode="soft_ref",
        context_kind="causal_attention",
        causal_token_dim=16,
        causal_heads=4,
        causal_n_mediators=4,
        causal_dropout=0.0,
        causal_mass_attention_temperature=0.5,
        causal_growth_only=causal_growth_only,
    )
    model.eval()
    return model


def test_causal_attention_context_uses_absolute_mass_reductions() -> None:
    agg = _aggregator()
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=0.3, residual=residual)
    expected_log_mass = log_m0.float() + torch.logsumexp(logw.float(), dim=1)
    expected_freq = torch.softmax(expected_log_mass, dim=0)

    assert out.context.shape == (7,)
    assert out.growth_context.shape == (3, 7)
    assert out.edge_scores_gm.shape == (3, 5)
    assert out.log_mass_g is not None
    assert torch.allclose(out.log_mass_g, expected_log_mass, atol=1e-6)
    assert torch.allclose(out.freq_g, expected_freq, atol=1e-6)
    assert torch.isclose(out.q.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isfinite(out.growth_context).all()


def test_causal_attention_particle_and_group_permutation_equivariance() -> None:
    agg = _aggregator()
    z, logw, a, residual, log_m0 = _inputs()
    out = agg(z, logw, a, log_m0, tau=0.4, residual=residual)

    particle_perm = torch.tensor([4, 1, 5, 0, 2, 3])
    group_perm = torch.tensor([2, 0, 1])
    out_perm = agg(
        z[group_perm][:, particle_perm],
        logw[group_perm][:, particle_perm],
        a[group_perm],
        log_m0[group_perm],
        tau=0.4,
        residual=residual[group_perm],
    )

    assert torch.allclose(out.context, out_perm.context, atol=1e-5)
    assert torch.allclose(out.q, out_perm.q, atol=1e-5)
    assert torch.allclose(out.s, out_perm.s, atol=1e-5)
    assert torch.allclose(out_perm.mass_g, out.mass_g[group_perm], atol=1e-6)
    assert torch.allclose(out_perm.freq_g, out.freq_g[group_perm], atol=1e-6)
    assert torch.allclose(out_perm.growth_context, out.growth_context[group_perm], atol=1e-5)
    assert torch.allclose(out_perm.edge_scores_gm, out.edge_scores_gm[group_perm], atol=1e-5)


def test_causal_attention_intervention_can_ablate_single_edge() -> None:
    agg = _aggregator()
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(
        z,
        logw,
        a,
        log_m0,
        tau=0.5,
        residual=residual,
        intervention=CausalAttentionIntervention(
            protocol="ablate_edges",
            ablate_group_mediator_edges=[(1, 2)],
        ),
    )

    assert out.edge_scores_gm[1, 2].item() == pytest.approx(0.0, abs=1e-30)
    assert torch.isfinite(out.context).all()


def test_all_causal_attention_context_parameters_receive_gradients() -> None:
    agg = _aggregator()
    agg.train()
    z, logw, a, residual, log_m0 = _inputs()
    z = z.requires_grad_(True)
    logw = logw.requires_grad_(True)
    a = a.requires_grad_(True)
    residual = residual.requires_grad_(True)
    log_m0 = log_m0.requires_grad_(True)

    out = agg(z, logw, a, log_m0, tau=0.5, residual=residual)
    loss = (
        out.context.square().sum()
        + out.growth_context.square().mean()
        + out.edge_scores_gm.square().mean()
        + out.mediator_tokens.square().mean()
    )
    loss.backward()

    missing = [
        name
        for name, parameter in agg.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, missing
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert logw.grad is not None and torch.isfinite(logw.grad).all()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert log_m0.grad is not None and torch.isfinite(log_m0.grad).all()


def test_soft_reference_residuals_are_zero_for_controls() -> None:
    model = _model()
    with torch.no_grad():
        model.embedding.reference_embedding.fill_(0.3)
        model.embedding.embeddings[model.embedding._nc_to_local["gene_a"]].fill_(0.2)

    effective = model.embedding(["ctrl", "gene_a"])
    residual = model.embedding.residuals(["ctrl", "gene_a"])

    assert torch.allclose(residual[0], torch.zeros_like(residual[0]))
    assert torch.allclose(effective[0], model.embedding.reference_embedding)
    assert torch.allclose(effective[1], model.embedding.reference_embedding + residual[1])


def test_full_dynamics_model_uses_causal_attention_growth_context() -> None:
    model = _model(causal_growth_only=True)
    z = torch.randn(3, 5, 2)
    logw = torch.full((3, 5), -torch.log(torch.tensor(5.0)))
    log_m0 = torch.tensor([0.0, 0.3, -0.2])

    coeffs, ctx = model.step(
        z=z,
        tau=torch.tensor(0.2),
        logw=logw,
        log_m0=log_m0,
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
    )

    assert coeffs.drift.shape == (3, 5, 2)
    assert coeffs.growth.shape == (3, 5)
    assert ctx.base_context.shape == (5,)
    assert ctx.growth_context.shape == (3, 5)
    assert torch.isfinite(coeffs.growth).all()


def test_causal_attention_all_coefficients_uses_group_context() -> None:
    model = _model(causal_growth_only=False)
    z = torch.randn(3, 5, 2)
    logw = torch.full((3, 5), -torch.log(torch.tensor(5.0)))
    log_m0 = torch.tensor([0.0, 0.3, -0.2])

    _, ctx = model.step(
        z=z,
        tau=torch.tensor(0.2),
        logw=logw,
        log_m0=log_m0,
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
    )

    assert ctx.base_context.shape == (3, 5)
    assert ctx.growth_context.shape == (3, 5)


def test_causal_attention_counterfactual_requires_full_context_by_default() -> None:
    endpoint = _endpoint()
    model = _model()
    partial_endpoint = EndpointProblem(
        initial={"ctrl": endpoint.initial["ctrl"], "gene_a": endpoint.initial["gene_a"]},
        terminal={"ctrl": endpoint.terminal["ctrl"], "gene_a": endpoint.terminal["gene_a"]},
        time_axis=endpoint.time_axis,
        perturbation_ids=["ctrl", "gene_a"],
    )
    engine = CounterfactualEngine(
        model=model,
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=4,
    )

    with pytest.raises(ValueError, match="partial"):
        engine.run(partial_endpoint, ["gene_a"], seed=7)


def test_causal_attention_counterfactual_uses_full_context_same_start_noise() -> None:
    endpoint = _endpoint()
    model = _model()
    engine = CounterfactualEngine(
        model=model,
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=4,
    )

    result = engine.run(endpoint, ["gene_a"], clamp_context=True, seed=7, common_noise=True)[0]

    assert result.metadata["context_kind"] == "causal_attention"
    assert result.metadata["counterfactual_seed_mode"] == "global_common"
    assert result.metadata["same_start"] is True
    assert result.metadata["same_noise"] is True
    assert result.metadata["context_fraction"] == pytest.approx(1.0)
    assert result.rollout_clamped is not None
    assert result.rollout_control_clamped is not None
    assert torch.equal(result.rollout_perturb.z_steps[0], result.rollout_control.z_steps[0])
    assert torch.equal(result.rollout_perturb.logw_steps[0], result.rollout_control.logw_steps[0])
    assert torch.equal(result.rollout_perturb.noise_steps, result.rollout_control.noise_steps)


def test_causal_attention_mediator_ablation_is_same_start_same_noise() -> None:
    endpoint = _endpoint()
    model = _model()
    engine = CounterfactualEngine(
        model=model,
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=4,
    )

    result = engine.run_mediator_ablation(endpoint, ["gene_a"], [1], seed=11)[0]

    assert result.metadata["counterfactual_type"] == "mediator_ablation"
    assert result.metadata["mediator_id"] == 1
    assert result.metadata["same_start"] is True
    assert result.metadata["same_noise"] is True
    assert torch.equal(result.rollout_perturb.z_steps[0], result.rollout_control.z_steps[0])
    assert torch.equal(result.rollout_perturb.noise_steps, result.rollout_control.noise_steps)


def test_model_config_rejects_invalid_causal_attention_shape() -> None:
    with pytest.raises(ValueError, match="causal_token_dim"):
        ModelConfig(context_kind="causal_attention", causal_token_dim=10, causal_heads=4)


def test_chunked_causal_attention_training_uses_global_context(tmp_path) -> None:
    endpoint = _endpoint()
    model = _model()
    cfg = RunConfig(
        device="cpu",
        latent={"dim": 2},
        model={
            "context_kind": "causal_attention",
            "embedding_dim": 4,
            "n_programs": 3,
            "mediator_dim": 2,
            "causal_token_dim": 16,
            "causal_heads": 4,
            "causal_n_mediators": 4,
            "hidden_dim": 12,
            "depth": 1,
        },
        simulation=SimulationConfig(n_particles=4, n_steps=2, store_history=True),
        training=TrainingConfig(
            lambda_weak=0.0,
            lambda_count=0.0,
            lambda_reg_net=0.0,
            lambda_reg_diffusion=0.0,
            max_active_perturbations=1,
            sinkhorn_max_iter=5,
            lr_transformer=1e-4,
        ),
    )
    trainer = Trainer(model, cfg, endpoint, ["ctrl", "gene_a", "gene_b"], output_dir=str(tmp_path))
    optimizer = trainer._build_optimizer("all")

    metrics = trainer._one_epoch_chunked(
        optimizer=optimizer,
        epoch=1,
        stage="all",
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
        seed_offset=0,
    )

    assert metrics["perturbation_batch_size"] == 1
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
