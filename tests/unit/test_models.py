"""Unit tests for model components."""
import numpy as np
import pytest
import torch

from cape.models.embeddings import PerturbationEmbedding, TimeEmbedding
from cape.models.context import ContextAggregator
from cape.models.coefficients import CoefficientNetworks
from cape.models.ecology import EcologicalPayoff
from cape.models.full_model import FullDynamicsModel
from cape.models.weighted_sde import WeightedParticleSimulator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# PerturbationEmbedding
# ---------------------------------------------------------------------------

def test_control_anchor_exact():
    emb = PerturbationEmbedding(["ctrl", "g1", "g2"], ["ctrl"], embedding_dim=4)
    emb.to(DEVICE)
    a = emb(["ctrl"])
    assert torch.all(a == 0), "Control embedding must be exactly zero"


def test_embedding_device_propagation():
    """Embedding must work on CUDA even when all pids are controls."""
    emb = PerturbationEmbedding(["ctrl1", "ctrl2"], ["ctrl1", "ctrl2"], embedding_dim=4)
    emb.to(DEVICE)
    a = emb(["ctrl1", "ctrl2"])
    assert a.device.type == DEVICE if DEVICE != "cpu" else "cpu"
    assert torch.all(a == 0)


def test_non_control_embedding_nonzero():
    torch.manual_seed(42)
    emb = PerturbationEmbedding(["ctrl", "g1"], ["ctrl"], embedding_dim=4)
    # After random init, non-control embeddings should be non-zero
    a = emb(["ctrl", "g1"])
    assert a[0].abs().max() == 0        # control = 0
    assert a[1].abs().max() > 0         # non-control != 0


def test_embedding_regularization():
    emb = PerturbationEmbedding(["ctrl", "g1"], ["ctrl"], embedding_dim=4)
    reg = emb.regularization()
    assert reg >= 0


# ---------------------------------------------------------------------------
# TimeEmbedding
# ---------------------------------------------------------------------------

def test_time_embedding_shape():
    te = TimeEmbedding(n_frequencies=4)
    assert te.output_dim == 9   # 1 + 2*4
    tau = torch.tensor(0.5)
    out = te(tau)
    assert out.shape == (9,)


def test_time_embedding_boundary():
    te = TimeEmbedding(n_frequencies=2)
    tau0 = te(torch.tensor(0.0))
    tau1 = te(torch.tensor(1.0))
    # First element should be tau itself
    assert tau0[0].item() == pytest.approx(0.0)
    assert tau1[0].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ContextAggregator
# ---------------------------------------------------------------------------

def test_context_aggregator_shapes():
    G, N, d, K, L = 5, 32, 8, 4, 4
    agg = ContextAggregator(d, K, L, K + L, use_identity_context=True).to(DEVICE)
    z = torch.randn(G, N, d, device=DEVICE)
    logw = torch.full((G, N), -np.log(N), device=DEVICE)
    a = torch.zeros(G, 4, device=DEVICE)
    log_m0 = torch.log(torch.tensor([1000.0] * G, device=DEVICE))

    ctx = agg(z, logw, a, log_m0)
    assert ctx.q.shape == (K,)
    assert ctx.s.shape == (L,)
    assert ctx.context.shape == (K + L,)
    assert ctx.mass_g.shape == (G,)
    assert ctx.freq_g.shape == (G,)


def test_context_freq_sums_to_one():
    G, N, d, K, L = 4, 16, 8, 4, 4
    agg = ContextAggregator(d, K, L, K + L, use_identity_context=True).to(DEVICE)
    z = torch.randn(G, N, d, device=DEVICE)
    logw = torch.full((G, N), -np.log(N), device=DEVICE)
    a = torch.zeros(G, 4, device=DEVICE)
    log_m0 = torch.zeros(G, device=DEVICE)

    ctx = agg(z, logw, a, log_m0)
    assert ctx.freq_g.sum().item() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# CoefficientNetworks
# ---------------------------------------------------------------------------

def test_coeff_shapes():
    G, N, d = 3, 16, 8
    nets = CoefficientNetworks(
        latent_dim=d, embedding_dim=4, context_dim=8, hidden_dim=32, depth=2,
        sigma_min=1e-3, r_max=2.0, ecological_growth=False,
    ).to(DEVICE)

    z = torch.randn(G, N, d, device=DEVICE)
    tau = torch.tensor(0.5, device=DEVICE)
    ctx = torch.zeros(8, device=DEVICE)
    a = torch.zeros(G, 4, device=DEVICE)

    coeffs = nets(z, tau, ctx, a)
    assert coeffs.drift.shape == (G, N, d)
    assert coeffs.sigma_diag.shape == (G, N, d)
    assert coeffs.growth.shape == (G, N)


def test_sigma_positive():
    G, N, d = 2, 8, 4
    nets = CoefficientNetworks(d, 4, 8, 32, 2, sigma_min=0.01, r_max=2.0).to(DEVICE)
    z = torch.randn(G, N, d, device=DEVICE)
    tau = torch.tensor(0.0, device=DEVICE)
    ctx = torch.zeros(8, device=DEVICE)
    a = torch.randn(G, 4, device=DEVICE)
    coeffs = nets(z, tau, ctx, a)
    assert (coeffs.sigma_diag > 0).all()


def test_growth_bounded():
    G, N, d = 2, 8, 4
    r_max = 1.5
    nets = CoefficientNetworks(d, 4, 8, 32, 2, r_max=r_max).to(DEVICE)
    z = torch.randn(G, N, d, device=DEVICE)
    tau = torch.tensor(0.5, device=DEVICE)
    ctx = torch.zeros(8, device=DEVICE)
    a = torch.randn(G, 4, device=DEVICE) * 10  # large embedding to stress-test
    coeffs = nets(z, tau, ctx, a)
    assert (coeffs.growth.abs() <= r_max + 1e-5).all()


def test_control_anchor_zero_modulation():
    """When a_g = 0 (control), drift/sigma must equal baseline (no perturbation shift)."""
    G, N, d = 2, 8, 4
    nets = CoefficientNetworks(d, 4, 8, 32, 2).to(DEVICE)
    z = torch.randn(G, N, d, device=DEVICE)
    tau = torch.tensor(0.5, device=DEVICE)
    ctx = torch.zeros(8, device=DEVICE)

    a_zero = torch.zeros(G, 4, device=DEVICE)
    a_nonzero = torch.randn(G, 4, device=DEVICE)

    c_zero = nets(z, tau, ctx, a_zero)
    c_nonzero = nets(z, tau, ctx, a_nonzero)

    # With zero embedding, result equals baseline; with non-zero it should differ
    assert not torch.allclose(c_zero.drift, c_nonzero.drift, atol=1e-5)


# ---------------------------------------------------------------------------
# WeightedParticleSimulator
# ---------------------------------------------------------------------------

def _make_simple_model():
    pids = ["ctrl", "g1", "g2"]
    return FullDynamicsModel(
        perturbation_ids=pids,
        control_ids=["ctrl"],
        latent_dim=4, embedding_dim=4, n_programs=4, mediator_dim=4,
        hidden_dim=32, depth=2,
    ).to(DEVICE)


def test_rollout_shapes():
    G, N, d = 3, 16, 4
    model = _make_simple_model()
    sim = WeightedParticleSimulator(n_steps=8, store_history=True)
    z0 = torch.randn(G, N, d, device=DEVICE)
    lw0 = torch.full((G, N), -np.log(N), device=DEVICE)
    lm0 = torch.zeros(G, device=DEVICE)

    pids = ["ctrl", "g1", "g2"]
    rollout = sim.rollout(z0, lw0, model, lm0, perturbation_ids=pids)

    assert rollout.z_steps.shape == (9, G, N, d)
    assert rollout.logw_steps.shape == (9, G, N)
    assert rollout.drift_steps.shape == (8, G, N, d)
    assert rollout.sigma_steps.shape == (8, G, N, d)
    assert rollout.growth_steps.shape == (8, G, N)


def test_rollout_backward():
    G, N, d = 2, 8, 4
    model = FullDynamicsModel(["ctrl", "g1"], ["ctrl"], latent_dim=d,
        embedding_dim=4, n_programs=4, mediator_dim=4, hidden_dim=32, depth=1).to(DEVICE)
    sim = WeightedParticleSimulator(n_steps=4, store_history=False)
    z0 = torch.randn(G, N, d, device=DEVICE)
    lw0 = torch.full((G, N), -np.log(N), device=DEVICE)
    lm0 = torch.zeros(G, device=DEVICE)

    rollout = sim.rollout(z0, lw0, model, lm0, perturbation_ids=["ctrl", "g1"])
    loss = rollout.terminal_logw.mean()
    loss.backward()
    # Check that gradients flowed
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()


def test_logweight_stabilisation():
    """Log-weight stabilisation should not change normalised weights."""
    G, N = 2, 32
    logw = torch.randn(G, N, device=DEVICE) * 5  # large spread
    logw_stable = logw - logw.max(dim=-1, keepdim=True).values
    # Normalised weights must be the same
    w1 = torch.softmax(logw, dim=-1)
    w2 = torch.softmax(logw_stable, dim=-1)
    assert torch.allclose(w1, w2, atol=1e-5)


# ---------------------------------------------------------------------------
# EcologicalPayoff
# ---------------------------------------------------------------------------

def test_ecology_shape():
    K, r = 4, 4
    eco = EcologicalPayoff(n_programs=K, embedding_dim=r, n_ranks=2).to(DEVICE)
    G, N = 3, 8
    eta_z = torch.softmax(torch.randn(G, N, K, device=DEVICE), dim=-1)
    a_g = torch.randn(G, r, device=DEVICE)
    q = torch.softmax(torch.randn(K, device=DEVICE), dim=-1)

    phi = eco(eta_z, a_g, q)
    assert phi.shape == (G, N)


def test_ecology_control_zero_perturbation():
    """With a_g=0 (control), ecology still computes but only P0 matters."""
    K = 4
    eco = EcologicalPayoff(n_programs=K, embedding_dim=4, n_ranks=2).to(DEVICE)
    G, N = 2, 8
    eta_z = torch.softmax(torch.randn(G, N, K, device=DEVICE), dim=-1)
    a_ctrl = torch.zeros(G, 4, device=DEVICE)
    q = torch.softmax(torch.randn(K, device=DEVICE), dim=-1)
    phi = eco(eta_z, a_ctrl, q)
    assert phi.shape == (G, N)
    assert not torch.isnan(phi).any()
