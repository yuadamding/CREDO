from __future__ import annotations

import numpy as np
import pytest
import torch

from credo.data.core import FiniteMeasure, PerturbationCatalog, TimeAxis, TrajectoryProblem
from credo.models.context import ContextAggregator, GroupStatistics
from credo.models.embeddings import PerturbationEmbedding
from credo.models.full_model import FullDynamicsModel
from credo.models.simulator import _control_embedding_context
from credo.models.weighted_sde import WeightedParticleSimulator
from credo.training.trainer import _uses_global_ecological_context
from credo.training.trajectory_batch import initialise_particles_from_trajectory


pytestmark = pytest.mark.semantic


def test_soft_ref_effective_embeddings_are_reference_plus_residual() -> None:
    emb = PerturbationEmbedding(
        perturbation_ids=["ctrl_a", "ctrl_b", "gene_x"],
        control_ids=["ctrl_a", "ctrl_b"],
        embedding_dim=2,
        control_mode="soft_ref",
    )
    with torch.no_grad():
        emb.reference_embedding.copy_(torch.tensor([1.0, 2.0]))
        emb.embeddings[emb._nc_to_local["gene_x"]].copy_(torch.tensor([3.0, 4.0]))

    out = emb(["ctrl_a", "ctrl_b", "gene_x"])

    assert torch.allclose(out[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(out[1], torch.tensor([1.0, 2.0]))
    assert torch.allclose(out[2], torch.tensor([4.0, 6.0]))
    assert torch.allclose(out[0], out[1])
    emb.assert_soft_ref_invariants()


def test_soft_ref_invariant_assertion_catches_shared_guide_bypass() -> None:
    emb = PerturbationEmbedding(
        perturbation_ids=["ctrl", "gene_x"],
        control_ids=["ctrl"],
        embedding_dim=2,
        control_mode="soft_ref",
        shared_guide_embedding=True,
    )

    try:
        emb.assert_soft_ref_invariants()
    except AssertionError as exc:
        assert "shared_guide_embedding" in str(exc)
    else:
        raise AssertionError("Expected soft-ref invariant assertion to fail")


def test_context_uses_absolute_log_m0_mass() -> None:
    context = ContextAggregator(
        latent_dim=1,
        n_programs=2,
        mediator_dim=1,
        context_dim=3,
        fixed_program_centroids=torch.tensor([[0.0], [10.0]]),
    )
    z = torch.tensor([[[0.0], [0.2]], [[9.8], [10.0]]])
    logw = torch.full((2, 2), -np.log(2.0))
    a = torch.zeros(2, 1)

    equal_mass = context(z, logw, a, torch.tensor([0.0, 0.0]))
    expanded_second = context(z, logw, a, torch.tensor([0.0, np.log(9.0)], dtype=torch.float32))

    assert expanded_second.freq_g[1] > equal_mass.freq_g[1]
    assert not torch.allclose(equal_mass.q, expanded_second.q)


def test_context_invariant_to_stabilized_logw_shift_with_restored_mass() -> None:
    context = ContextAggregator(
        latent_dim=1,
        n_programs=2,
        mediator_dim=1,
        context_dim=3,
        fixed_program_centroids=torch.tensor([[0.0], [10.0]]),
    )
    z = torch.tensor([[[0.0], [0.2]], [[9.8], [10.0]]])
    logw = torch.tensor([[-0.2, -1.7], [-0.3, -1.4]])
    log_m0 = torch.tensor([0.4, -0.1])
    shifts = torch.tensor([5.0, -3.0])
    a = torch.zeros(2, 1)

    original = context(z, logw, a, log_m0)
    stabilized = context(z, logw + shifts[:, None], a, log_m0 - shifts)

    assert torch.allclose(original.freq_g, stabilized.freq_g, atol=1e-6)
    assert torch.allclose(original.q, stabilized.q, atol=1e-6)
    assert torch.allclose(original.s, stabilized.s, atol=1e-6)


def test_context_extreme_log_masses_remain_finite() -> None:
    context = ContextAggregator(
        latent_dim=1,
        n_programs=2,
        mediator_dim=1,
        context_dim=3,
        fixed_program_centroids=torch.tensor([[0.0], [10.0]]),
    )
    z = torch.tensor([[[0.0], [0.2]], [[9.8], [10.0]]])
    logw = torch.full((2, 2), -np.log(2.0))
    a = torch.zeros(2, 1)

    out = context(z, logw, a, torch.tensor([-100.0, 100.0]))

    assert torch.isfinite(out.freq_g).all()
    assert torch.isfinite(out.q).all()
    assert torch.isfinite(out.s).all()
    assert out.freq_g[1] > 1.0 - 1e-6


def test_reference_counterfactual_context_keeps_soft_reference_embedding() -> None:
    model = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_x"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        control_mode="soft_ref",
        control_ref_penalty=0.0,
    )
    with torch.no_grad():
        model.embedding.reference_embedding.copy_(torch.tensor([1.0, 2.0]))
        model.embedding.embeddings[model.embedding._nc_to_local["gene_x"]].copy_(
            torch.tensor([3.0, 4.0])
        )

    factual = model.embedding(["gene_x"])[0]
    with _control_embedding_context(model, "gene_x", mode="reference_consistent"):
        reference = model.embedding(["gene_x"])[0]
    restored = model.embedding(["gene_x"])[0]

    assert torch.allclose(factual, torch.tensor([4.0, 6.0]))
    assert torch.allclose(reference, torch.tensor([1.0, 2.0]))
    assert not torch.allclose(reference, torch.zeros_like(reference))
    assert torch.allclose(restored, factual)


def test_ecological_growth_requires_global_context_under_sharding() -> None:
    ecological = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_x"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=True,
        context_kind="mlp",
    )
    non_ecological = FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_x"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        context_kind="mlp",
    )

    assert _uses_global_ecological_context(ecological)
    assert not _uses_global_ecological_context(non_ecological)


def test_mlp_ecological_chunked_rollout_matches_full_global_context() -> None:
    torch.manual_seed(71)
    pids = ["ctrl", "gene_a", "gene_b"]
    model = FullDynamicsModel(
        perturbation_ids=pids,
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=3,
        n_programs=3,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        ecological_growth=True,
        control_mode="soft_ref",
        context_kind="mlp",
    )
    model.eval()
    z_full = torch.randn(3, 4, 2)
    logw_full = torch.full((3, 4), -torch.log(torch.tensor(4.0)))
    log_m0_full = torch.tensor([0.0, 0.2, -0.4])
    noise_steps = torch.randn(2, 3, 4, 2)

    simulator = WeightedParticleSimulator(n_steps=2, store_history=True)
    full = simulator.rollout(
        z0=z_full,
        logw0=logw_full,
        model=model,
        log_m0=log_m0_full,
        perturbation_ids=pids,
        noise_steps=noise_steps,
    )

    chunks = [
        {
            "pids": pids[:1],
            "z": z_full[:1].clone(),
            "logw": logw_full[:1].clone(),
            "log_m0": log_m0_full[:1].clone(),
        },
        {
            "pids": pids[1:],
            "z": z_full[1:].clone(),
            "logw": logw_full[1:].clone(),
            "log_m0": log_m0_full[1:].clone(),
        },
    ]
    tau_steps = torch.linspace(0.0, 1.0, 3)
    for step_idx in range(2):
        tau_k = tau_steps[step_idx]
        dtau = tau_steps[step_idx + 1] - tau_steps[step_idx]
        stats_parts = []
        cache = []
        for chunk in chunks:
            a_local = model.embedding(chunk["pids"])
            b_local = model.embedding.growth_intercepts(chunk["pids"])
            eta_local, phi_local = model.context_agg.encode_particles(chunk["z"])
            stats_local, _, _ = model.context_agg.summarize_groups(
                chunk["z"],
                chunk["logw"],
                chunk["log_m0"],
                eta=eta_local,
                phi=phi_local,
            )
            stats_parts.append(stats_local)
            cache.append({"a": a_local, "b": b_local, "eta": eta_local})
        global_context = model.context_agg.context_from_group_statistics(
            GroupStatistics(
                log_n_g=torch.cat([stats.log_n_g for stats in stats_parts], dim=0),
                eta_g=torch.cat([stats.eta_g for stats in stats_parts], dim=0),
                phi_g=torch.cat([stats.phi_g for stats in stats_parts], dim=0),
            )
        )

        start = 0
        for chunk, cached in zip(chunks, cache):
            stop = start + len(chunk["pids"])
            coeffs = model.coeff_nets(
                z=chunk["z"],
                tau=tau_k,
                context=global_context.context,
                a=cached["a"],
                growth_intercept=cached["b"],
                eta_z=cached["eta"],
                q=global_context.q,
                s=global_context.s,
            )
            chunk_noise = noise_steps[step_idx, start:stop]
            chunk["z"] = chunk["z"] + coeffs.drift * dtau + coeffs.sigma_diag * torch.sqrt(dtau) * chunk_noise
            chunk["logw"] = chunk["logw"] + coeffs.growth * dtau
            start = stop

    assert torch.allclose(torch.cat([chunk["z"] for chunk in chunks], dim=0), full.terminal_z, atol=1e-6)
    assert torch.allclose(
        torch.cat([chunk["logw"] for chunk in chunks], dim=0),
        full.terminal_logw,
        atol=1e-6,
    )


def test_particle_initialization_respects_nonuniform_measure_weights() -> None:
    source = FiniteMeasure(
        support=np.asarray([[0.0], [10.0]], dtype=np.float32),
        weights=np.asarray([0.9, 0.1], dtype=np.float32),
        total_mass=1.0,
    )
    target = FiniteMeasure(
        support=np.asarray([[0.0], [10.0]], dtype=np.float32),
        weights=np.asarray([0.9, 0.1], dtype=np.float32),
        total_mass=1.0,
    )
    trajectory = TrajectoryProblem(
        measures={"t0": {"gene": source}, "t1": {"gene": target}},
        catalog=PerturbationCatalog(["gene"], ["gene"]),
        time_axis=TimeAxis(["t0", "t1"], [0.0, 1.0]),
        time_labels=["t0", "t1"],
    )

    z0, logw0, log_m0 = initialise_particles_from_trajectory(
        trajectory,
        "t0",
        ["gene"],
        n_particles=20_000,
        seed=17,
    )

    empirical_left_mass = (z0[0, :, 0] < 5.0).float().mean()
    assert 0.87 < float(empirical_left_mass) < 0.93
    assert torch.allclose(logw0.exp().sum(dim=1), torch.ones(1))
    assert torch.allclose(log_m0.exp(), torch.ones(1))
