"""Unit tests for loss functions."""
import numpy as np
import pytest
import torch

from cape.losses.uot import UOTLoss, sinkhorn_divergence
from cape.losses.weak_form import WeakFormLoss, GaussianRBFTestFunctions
from cape.losses.counts import CountLikelihood, integrated_fitness


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# UOT Loss
# ---------------------------------------------------------------------------

def test_uot_nonnegative():
    """Sinkhorn divergence must be non-negative."""
    x = torch.randn(20, 4, device=DEVICE)
    y = torch.randn(15, 4, device=DEVICE)
    la = torch.log(torch.softmax(torch.randn(20, device=DEVICE), dim=0)) + 5.0
    lb = torch.log(torch.softmax(torch.randn(15, device=DEVICE), dim=0)) + 5.0
    div = sinkhorn_divergence(x, la, y, lb, eps=0.1, tau=1.0)
    assert div.item() >= -1e-4  # allow small numerical error


def test_uot_zero_for_identical():
    """Divergence between identical measures should be near zero."""
    x = torch.randn(10, 4, device=DEVICE)
    la = torch.log(torch.ones(10, device=DEVICE) / 10) + np.log(500)  # total mass 500
    div = sinkhorn_divergence(x, la, x, la, eps=0.1, tau=1.0)
    assert div.item() == pytest.approx(0.0, abs=1e-3)


def test_uot_loss_module_shapes():
    G, N, d = 3, 16, 4
    loss_fn = UOTLoss(eps=0.1, tau=1.0, use_geomloss=True)
    pred_z = torch.randn(G, N, d, device=DEVICE, requires_grad=True)
    # Absolute log-weights (requires grad to test backward)
    pred_logw = torch.full((G, N), -np.log(N) + np.log(1000),
                            device=DEVICE, requires_grad=True)
    pids = ["ctrl", "g1", "g2"]
    tgt_sup = {pid: torch.randn(8, d, device=DEVICE) for pid in pids}
    tgt_lw = {pid: torch.log(torch.ones(8, device=DEVICE) / 8 * 1000) for pid in pids}

    total, per_pid = loss_fn(pred_z, pred_logw, tgt_sup, tgt_lw, pids)
    assert total.item() >= 0
    assert len(per_pid) == G
    total.backward()


def test_uot_mass_penalty_large_discrepancy():
    """Large mass discrepancy should produce large mass penalty."""
    x = torch.randn(10, 4, device=DEVICE)
    # Two measures with same geometry but very different masses
    la_small = torch.log(torch.softmax(torch.randn(10, device=DEVICE), dim=0)) + 0.0    # mass~1
    la_large = torch.log(torch.softmax(torch.randn(10, device=DEVICE), dim=0)) + 10.0   # mass~e^10

    div = sinkhorn_divergence(x, la_small, x, la_large, eps=0.1, tau=1.0)
    assert div.item() > 1.0  # should be large due to mass penalty


# ---------------------------------------------------------------------------
# Gaussian RBF test functions
# ---------------------------------------------------------------------------

def test_rbf_psi_shape():
    centers = torch.randn(8, 4, device=DEVICE)
    rbf = GaussianRBFTestFunctions(centers, bandwidth=1.0)
    z = torch.randn(3, 16, 4, device=DEVICE)  # [G, N, d]
    psi = rbf.psi(z)
    assert psi.shape == (3, 16, 8)


def test_rbf_psi_range():
    centers = torch.zeros(1, 4, device=DEVICE)
    rbf = GaussianRBFTestFunctions(centers, bandwidth=1.0)
    z = torch.zeros(1, 1, 4, device=DEVICE)   # exactly at center
    psi = rbf.psi(z)
    assert psi.item() == pytest.approx(1.0, abs=1e-5)


def test_rbf_gradient_analytic():
    """Compare autograd gradient with analytic formula."""
    centers = torch.zeros(1, 2, device=DEVICE)
    rbf = GaussianRBFTestFunctions(centers, bandwidth=1.0)

    z = torch.tensor([[[0.5, 0.3]]], device=DEVICE, requires_grad=True)
    psi = rbf.psi(z)  # [1, 1, 1]
    psi.backward()
    autograd_grad = z.grad[0, 0].cpu()  # [2]

    analytic_grad = rbf.grad_psi(z.detach())[0, 0, 0].cpu()  # [2]
    np.testing.assert_allclose(autograd_grad.numpy(), analytic_grad.numpy(), rtol=1e-4)


# ---------------------------------------------------------------------------
# WeakFormLoss
# ---------------------------------------------------------------------------

def test_weak_form_loss_nonnegative():
    G, N, d, K_time = 3, 16, 4, 8
    wfl = WeakFormLoss(n_test_functions=4, bandwidth=1.0, latent_dim=d).to(DEVICE)

    z_steps = torch.randn(K_time + 1, G, N, d, device=DEVICE)
    logw_steps = torch.full((K_time + 1, G, N), -np.log(N), device=DEVICE)
    drift_steps = torch.zeros(K_time, G, N, d, device=DEVICE)
    sigma_steps = torch.ones(K_time, G, N, d, device=DEVICE) * 0.1
    growth_steps = torch.zeros(K_time, G, N, device=DEVICE)
    tau_steps = torch.linspace(0, 1, K_time + 1, device=DEVICE)

    loss = wfl(z_steps, logw_steps, drift_steps, sigma_steps, growth_steps, tau_steps)
    assert loss.item() >= 0


def test_weak_form_loss_zero_for_stationary():
    """For a zero-drift, zero-diffusion, zero-growth system, residual should be near zero."""
    G, N, d, K_time = 2, 32, 4, 8
    wfl = WeakFormLoss(n_test_functions=8, bandwidth=1.0, latent_dim=d).to(DEVICE)

    # Particles don't move; weights don't change
    z_fixed = torch.randn(1, G, N, d, device=DEVICE).expand(K_time + 1, -1, -1, -1).clone()
    logw_fixed = torch.full((K_time + 1, G, N), -np.log(N), device=DEVICE)
    drift_zero = torch.zeros(K_time, G, N, d, device=DEVICE)
    sigma_zero = torch.zeros(K_time, G, N, d, device=DEVICE)
    growth_zero = torch.zeros(K_time, G, N, device=DEVICE)
    tau_steps = torch.linspace(0, 1, K_time + 1, device=DEVICE)

    loss = wfl(z_fixed, logw_fixed, drift_zero, sigma_zero, growth_zero, tau_steps)
    # With zero dynamics and stationary particles, the time derivative of E[w*psi] = 0
    # and the generator value = 0, so residual = 0
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# CountLikelihood
# ---------------------------------------------------------------------------

def test_integrated_fitness_zero_growth():
    G, N, K = 3, 16, 8
    growth_steps = torch.zeros(K, G, N, device=DEVICE)
    logw_steps = torch.full((K + 1, G, N), -np.log(N), device=DEVICE)
    tau_steps = torch.linspace(0, 1, K + 1, device=DEVICE)
    zeta = integrated_fitness(growth_steps, logw_steps, tau_steps)
    assert zeta.shape == (G,)
    assert zeta.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_count_likelihood_shape():
    G, N, K, S = 4, 16, 8, 3
    lik = CountLikelihood(use_dirichlet_multinomial=True).to(DEVICE)

    growth_steps = torch.randn(K, G, N, device=DEVICE) * 0.1
    logw_steps = torch.full((K + 1, G, N), -np.log(N), device=DEVICE)
    tau_steps = torch.linspace(0, 1, K + 1, device=DEVICE)
    exposures = torch.ones(G, device=DEVICE) / G
    counts = torch.randint(0, 100, (S, G), dtype=torch.float32, device=DEVICE)
    n_totals = counts.sum(-1)

    loss = lik(growth_steps, logw_steps, tau_steps, exposures, counts, n_totals)
    assert loss.item() > 0  # negative log-likelihood should be positive
    loss.backward()
