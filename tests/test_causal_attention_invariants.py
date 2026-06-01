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
    MassGraphMaskedCrossAttention,
    ParticleRollout,
    PerturbationEmbedding,
    WeightedParticleSimulator,
)
from credo.losses.causal_attention import (
    context_smoothness_loss,
    control_edge_null_loss,
    edge_sparsity_loss,
    guide_concordance_loss,
    mediator_orthogonality_loss,
)
from credo.training.trainer import Trainer


def _aggregator(
    *,
    residual_policy: str = "edges_only",
) -> CausalEcologicalAttentionContext:
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
        residual_policy=residual_policy,
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


def test_all_masked_attention_rows_fail_loudly() -> None:
    attn = MassGraphMaskedCrossAttention(dim=8, heads=2, dropout=0.0)
    query = torch.randn(1, 2, 8)
    key = torch.randn(1, 3, 8)
    value = torch.randn(1, 3, 8)
    graph_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    graph_mask[:, 1, :] = False

    with pytest.raises(ValueError, match="At least one key"):
        attn(query, key, value, graph_mask=graph_mask)


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
    assert out.diagnostics is not None
    assert torch.isfinite(out.diagnostics.effective_edge_mean)
    assert torch.isfinite(out.diagnostics.baseline_edge_mean)
    assert torch.isfinite(out.diagnostics.residual_edge_sparsity_loss)


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


def test_clamped_edge_scores_reject_simultaneous_ablation() -> None:
    logits = torch.randn(3, 5)
    intervention = CausalAttentionIntervention(
        protocol="clamp_edges",
        ablate_mediator_ids=[1],
        clamp_edge_scores_gm=torch.full((3, 5), 0.5),
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        intervention.apply_effective_logits(logits)


def test_residual_edge_scores_are_signed_deltas_from_baseline() -> None:
    agg = _aggregator()
    with torch.no_grad():
        agg.residual_edge_score.weight.fill_(0.4)
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=0.5, residual=residual)

    expected_delta = out.edge_scores_gm - out.baseline_edge_scores_gm
    assert torch.allclose(out.residual_edge_scores_gm, expected_delta, atol=1e-6)
    assert torch.allclose(out.residual_edge_magnitude_gm, expected_delta.abs(), atol=1e-6)
    assert torch.allclose(
        out.residual_edge_scores_gm[0],
        torch.zeros_like(out.residual_edge_scores_gm[0]),
        atol=1e-6,
    )


def test_residual_edge_ablation_preserves_baseline_usage() -> None:
    agg = _aggregator()
    with torch.no_grad():
        agg.residual_edge_score.weight.fill_(0.4)
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=0.5, residual=residual)
    ablated = agg(
        z,
        logw,
        a,
        log_m0,
        tau=0.5,
        residual=residual,
        intervention=CausalAttentionIntervention(
            protocol="ablate_residual_edges",
            ablate_group_mediator_edges=[(1, 2)],
        ),
    )

    assert torch.allclose(ablated.baseline_edge_scores_gm, out.baseline_edge_scores_gm, atol=1e-6)
    assert ablated.residual_edge_scores_gm[1, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(ablated.edge_scores_gm[1, 2], ablated.baseline_edge_scores_gm[1, 2], atol=1e-6)


def test_edges_only_tokenizers_do_not_leak_residual_or_effective_embedding() -> None:
    agg = _aggregator(residual_policy="edges_only")
    with torch.no_grad():
        agg.residual_edge_score.weight.fill_(0.4)
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=0.5, residual=residual)
    changed = agg(
        z,
        logw,
        a + 10.0,
        log_m0,
        tau=0.5,
        residual=-3.0 * residual,
    )

    assert torch.allclose(out.baseline_edge_scores_gm, changed.baseline_edge_scores_gm, atol=1e-6)
    assert torch.allclose(out.mediator_tokens, changed.mediator_tokens, atol=1e-6)
    assert not torch.allclose(out.edge_scores_gm, changed.edge_scores_gm, atol=1e-6)


def test_tokens_and_edges_policy_allows_residual_conditioned_baseline_features() -> None:
    agg = _aggregator(residual_policy="tokens_and_edges")
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=0.5, residual=residual)
    changed = agg(z, logw, a, log_m0, tau=0.5, residual=-3.0 * residual)

    assert not torch.allclose(out.baseline_edge_scores_gm, changed.baseline_edge_scores_gm, atol=1e-6)


def test_all_edges_ablated_gives_zero_causal_delta() -> None:
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
            protocol="ablate_mediators",
            ablate_mediator_ids=list(range(agg.n_mediators)),
        ),
    )

    assert torch.allclose(out.edge_scores_gm, torch.zeros_like(out.edge_scores_gm))
    assert torch.allclose(
        out.growth_context,
        out.context.unsqueeze(0).expand_as(out.growth_context),
        atol=1e-6,
    )


def test_all_edges_ablated_stays_exact_after_attention_bias_training() -> None:
    agg = _aggregator()
    with torch.no_grad():
        agg.global_med_to_group.out_proj.bias.fill_(2.0)
        agg.group_context_scale.fill_(10.0)
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(
        z,
        logw,
        a,
        log_m0,
        tau=0.5,
        residual=residual,
        intervention=CausalAttentionIntervention(
            protocol="ablate_effective_edges",
            ablate_mediator_ids=list(range(agg.n_mediators)),
        ),
    )

    assert torch.allclose(out.edge_scores_gm, torch.zeros_like(out.edge_scores_gm))
    assert torch.allclose(
        out.growth_context,
        out.context.unsqueeze(0).expand_as(out.growth_context),
        atol=1e-6,
    )


def test_all_effective_edges_ablated_gives_zero_causal_delta() -> None:
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
            protocol="ablate_effective_edges",
            ablate_mediator_ids=list(range(agg.n_mediators)),
        ),
    )

    assert torch.allclose(out.edge_scores_gm, torch.zeros_like(out.edge_scores_gm))
    assert torch.allclose(
        out.growth_context,
        out.context.unsqueeze(0).expand_as(out.growth_context),
        atol=1e-6,
    )


def test_causal_attention_splits_baseline_and_residual_edges() -> None:
    agg = _aggregator()
    z, logw, a, residual, log_m0 = _inputs()

    out = agg(z, logw, a, log_m0, tau=0.5, residual=residual)

    assert out.baseline_edge_scores_gm.shape == out.edge_scores_gm.shape
    assert out.residual_edge_scores_gm.shape == out.edge_scores_gm.shape
    assert torch.allclose(
        out.residual_edge_scores_gm[0],
        torch.zeros_like(out.residual_edge_scores_gm[0]),
        atol=1e-8,
    )
    assert torch.isfinite(out.baseline_edge_scores_gm).all()
    assert torch.isfinite(out.edge_scores_gm).all()


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
    assert a.grad is None
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


def test_shared_guide_residuals_are_zero_for_controls() -> None:
    embedding = PerturbationEmbedding(
        perturbation_ids=["ctrl", "guide_a"],
        control_ids=["ctrl"],
        embedding_dim=3,
        control_mode="soft_ref",
        shared_guide_embedding=True,
    )
    with torch.no_grad():
        embedding.shared_embedding.fill_(0.5)

    residual = embedding.residuals(["ctrl", "guide_a"])

    assert torch.allclose(residual[0], torch.zeros_like(residual[0]))
    assert torch.allclose(residual[1], torch.full_like(residual[1], 0.5))


def test_slice_group_slices_group_specific_context_and_causal_edges() -> None:
    rollout = ParticleRollout(
        z_steps=torch.zeros(3, 3, 2, 2),
        logw_steps=torch.zeros(3, 3, 2),
        tau_steps=torch.linspace(0.0, 1.0, 3),
        context_steps=torch.zeros(2, 5),
        base_context_steps=torch.randn(2, 3, 5),
        growth_context_steps=torch.randn(2, 3, 5),
        causal_edge_scores_steps=torch.randn(2, 3, 4),
        causal_residual_edge_scores_steps=torch.randn(2, 3, 4),
        causal_residual_edge_magnitude_steps=torch.randn(2, 3, 4),
        causal_mediator_tokens_steps=torch.randn(2, 4, 6),
        causal_growth_context_steps=torch.randn(2, 3, 5),
    )

    sliced = rollout.slice_group(1)

    assert sliced.context_steps.shape == (2, 5)
    assert sliced.base_context_steps.shape == (2, 1, 5)
    assert sliced.growth_context_steps.shape == (2, 1, 5)
    assert sliced.causal_edge_scores_steps.shape == (2, 1, 4)
    assert sliced.causal_residual_edge_scores_steps.shape == (2, 1, 4)
    assert sliced.causal_residual_edge_magnitude_steps.shape == (2, 1, 4)
    assert sliced.causal_mediator_tokens_steps.shape == (2, 4, 6)
    assert sliced.causal_growth_context_steps.shape == (2, 1, 5)


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
    assert result.metadata["edge_protocol"] == "ablate_effective_edges"
    assert result.metadata["same_start"] is True
    assert result.metadata["same_noise"] is True
    assert torch.equal(result.rollout_perturb.z_steps[0], result.rollout_control.z_steps[0])
    assert torch.equal(result.rollout_perturb.noise_steps, result.rollout_control.noise_steps)


def test_causal_attention_residual_edge_ablation_counterfactual() -> None:
    endpoint = _endpoint()
    model = _model()
    engine = CounterfactualEngine(
        model=model,
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=4,
    )

    result = engine.run_residual_edge_ablation(endpoint, ["gene_a"], [1], seed=11)[0]

    assert result.metadata["counterfactual_type"] == "mediator_ablation"
    assert result.metadata["edge_protocol"] == "ablate_residual_edges"
    assert result.metadata["rollout_control_semantics"] == "intervention_not_control_reference"
    assert result.metadata["same_start"] is True
    assert result.metadata["same_noise"] is True
    assert torch.equal(result.rollout_perturb.z_steps[0], result.rollout_control.z_steps[0])
    assert torch.equal(result.rollout_perturb.noise_steps, result.rollout_control.noise_steps)


def test_model_config_rejects_invalid_causal_attention_shape() -> None:
    with pytest.raises(ValueError, match="causal_token_dim"):
        ModelConfig(context_kind="causal_attention", causal_token_dim=10, causal_heads=4)


def test_causal_attention_defaults_are_claim_grade_edges_only() -> None:
    model_cfg = ModelConfig(context_kind="causal_attention")
    training_cfg = TrainingConfig()

    assert model_cfg.causal_residual_policy == "edges_only"
    assert training_cfg.causal_loss_start_epoch == 100
    assert training_cfg.causal_loss_ramp_epochs == 200


def test_causal_losses_backpropagate() -> None:
    edge_scores = torch.rand(3, 4, requires_grad=True)
    residual_edges = torch.rand(3, 4, requires_grad=True)
    mediator_tokens = torch.randn(4, 6, requires_grad=True)
    context_steps = torch.randn(3, 3, 5, requires_grad=True)
    tau_steps = torch.linspace(0.0, 1.0, 4)
    control_mask = torch.tensor([True, False, False])

    loss = (
        control_edge_null_loss(residual_edges, control_mask)
        + guide_concordance_loss(residual_edges, ["ctrl", "gene", "gene"])
        + edge_sparsity_loss(edge_scores)
        + mediator_orthogonality_loss(mediator_tokens)
        + context_smoothness_loss(context_steps, tau_steps)
    )
    loss.backward()

    for tensor in (edge_scores, residual_edges, mediator_tokens, context_steps):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_causal_guide_loss_requires_explicit_target_map(tmp_path) -> None:
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
            lambda_causal_guide=1e-3,
            lambda_reg_net=0.0,
            lambda_reg_diffusion=0.0,
            sinkhorn_max_iter=5,
        ),
    )
    trainer = Trainer(model, cfg, endpoint, ["ctrl", "gene_a", "gene_b"], output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="target_ids_by_pid"):
        trainer._one_epoch(
            optimizer=trainer._build_optimizer("all"),
            epoch=0,
            stage="all",
            perturbation_ids=["ctrl", "gene_a", "gene_b"],
        )


def test_count_loss_order_mismatch_fails_loudly(tmp_path) -> None:
    endpoint = _endpoint()
    pids = ["ctrl", "gene_a", "gene_b"]
    count_data = {
        "perturbation_ids": ["gene_b", "gene_a", "ctrl"],
        "exposures": np.ones(3, dtype=np.float32),
        "counts": np.asarray([[5.0, 7.0, 9.0]], dtype=np.float32),
        "n_totals": np.asarray([21.0], dtype=np.float32),
    }

    for max_active in (0, 1):
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
                lambda_count=0.3,
                lambda_reg_net=0.0,
                lambda_reg_diffusion=0.0,
                max_active_perturbations=max_active,
                sinkhorn_max_iter=5,
            ),
        )
        trainer = Trainer(
            model,
            cfg,
            endpoint,
            pids,
            count_data=count_data,
            output_dir=str(tmp_path / f"count-order-{max_active}"),
        )

        with pytest.raises(ValueError, match="Count loss perturbation order mismatch"):
            trainer._one_epoch(
                optimizer=trainer._build_optimizer("all"),
                epoch=0,
                stage="all",
                perturbation_ids=pids,
            )


def test_causal_guide_loss_rejects_identity_target_map(tmp_path) -> None:
    endpoint = _endpoint()
    pids = ["ctrl", "gene_a", "gene_b"]
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
            lambda_causal_guide=1e-3,
            lambda_reg_net=0.0,
            lambda_reg_diffusion=0.0,
            sinkhorn_max_iter=5,
        ),
    )
    count_data = {"target_ids_by_pid": {pid: pid for pid in pids}}
    trainer = Trainer(
        model,
        cfg,
        endpoint,
        pids,
        count_data=count_data,
        output_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="non-identity target map"):
        trainer._one_epoch(
            optimizer=trainer._build_optimizer("all"),
            epoch=0,
            stage="all",
            perturbation_ids=pids,
        )


def test_causal_attention_optimizer_uses_causal_lr(tmp_path) -> None:
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
        training=TrainingConfig(lr_causal_attention=2e-5, causal_attention_weight_decay=3e-4),
    )
    trainer = Trainer(model, cfg, endpoint, ["ctrl", "gene_a", "gene_b"], output_dir=str(tmp_path))
    optimizer = trainer._build_optimizer("all")

    causal_group_sizes = [
        len(group["params"])
        for group in optimizer.param_groups
        if abs(float(group["lr"]) - 2e-5) < 1e-12
    ]
    assert causal_group_sizes


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
    assert torch.isfinite(torch.tensor(metrics["loss_causal"]))
    assert np.isfinite(metrics["edge_sparsity"])
    assert np.isfinite(metrics["mediator_orthogonality"])


def test_causal_training_history_records_loss_and_diagnostics(tmp_path) -> None:
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
            log_every=10,
            checkpoint_every=100,
            early_stop_patience=5,
            lr_transformer=1e-4,
        ),
    )
    trainer = Trainer(model, cfg, endpoint, ["ctrl", "gene_a", "gene_b"], output_dir=str(tmp_path))

    history = trainer.train(stage="all", n_epochs=1)
    frame = history.to_dataframe()

    assert "loss_causal" in frame.columns
    assert "edge_sparsity" in frame.columns
    assert "effective_edge_mean" in frame.columns
    assert "baseline_edge_mean" in frame.columns
    assert "residual_edge_sparsity_loss" in frame.columns
    assert "mediator_orthogonality" in frame.columns
    assert np.isfinite(frame.loc[0, "loss_causal"])
    assert np.isfinite(frame.loc[0, "edge_sparsity"])
    assert (tmp_path / "training_history.csv").exists()
