from __future__ import annotations

from types import SimpleNamespace

import torch

from credo.models.weighted_sde import WeightedParticleSimulator


class _ConstantDynamics:
    def step(self, z, tau, logw, log_m0, perturbation_ids=None):
        coeffs = SimpleNamespace(
            drift=torch.ones_like(z),
            sigma_diag=torch.zeros_like(z),
            growth=torch.full_like(logw, 2.0),
        )
        ctx = SimpleNamespace(context=torch.zeros(1, dtype=z.dtype, device=z.device))
        return coeffs, ctx


class _LegacyReferenceDynamics:
    def step(self, z, tau, logw, log_m0, perturbation_ids=None):
        drift = 0.1 * z + tau.reshape(1, 1, 1)
        sigma = torch.full_like(z, 0.05)
        growth = 0.2 + 0.1 * logw
        coeffs = SimpleNamespace(drift=drift, sigma_diag=sigma, growth=growth)
        ctx = SimpleNamespace(context=torch.zeros(1, dtype=z.dtype, device=z.device))
        return coeffs, ctx


def _legacy_uniform_rollout(z0, logw0, model, log_m0, *, n_steps):
    dtau = 1.0 / n_steps
    tau_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=z0.device, dtype=z0.dtype)
    z_list = [z0]
    logw_list = [logw0]
    drift_list = []
    sigma_list = []
    growth_list = []
    z = z0.clone()
    logw = logw0.clone()
    for k in range(n_steps):
        coeffs, _ = model.step(z=z, tau=tau_steps[k], logw=logw, log_m0=log_m0)
        drift_list.append(coeffs.drift)
        sigma_list.append(coeffs.sigma_diag)
        growth_list.append(coeffs.growth)
        noise = torch.randn_like(z)
        z = z + coeffs.drift * dtau + coeffs.sigma_diag * (dtau**0.5) * noise
        logw = logw + coeffs.growth * dtau
        z_list.append(z)
        logw_list.append(logw)
    return {
        "z_steps": torch.stack(z_list, dim=0),
        "logw_steps": torch.stack(logw_list, dim=0),
        "tau_steps": tau_steps,
        "drift_steps": torch.stack(drift_list, dim=0),
        "sigma_steps": torch.stack(sigma_list, dim=0),
        "growth_steps": torch.stack(growth_list, dim=0),
    }


def test_default_rollout_matches_legacy_uniform_reference() -> None:
    simulator = WeightedParticleSimulator(n_steps=5, store_history=True)
    model = _LegacyReferenceDynamics()
    z0 = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2) / 10.0
    logw0 = torch.zeros(1, 6)
    log_m0 = torch.zeros(1)

    torch.manual_seed(11)
    rollout = simulator.rollout(z0=z0, logw0=logw0, model=model, log_m0=log_m0)
    torch.manual_seed(11)
    expected = _legacy_uniform_rollout(z0, logw0, model, log_m0, n_steps=5)

    assert torch.equal(rollout.tau_steps, expected["tau_steps"])
    assert torch.equal(rollout.z_steps, expected["z_steps"])
    assert torch.equal(rollout.logw_steps, expected["logw_steps"])
    assert torch.equal(rollout.drift_steps, expected["drift_steps"])
    assert torch.equal(rollout.sigma_steps, expected["sigma_steps"])
    assert torch.equal(rollout.growth_steps, expected["growth_steps"])


def test_rollout_accepts_nonuniform_tau_grid() -> None:
    simulator = WeightedParticleSimulator(n_steps=99, store_history=True)
    tau_grid = torch.tensor([0.0, 0.2, 0.7, 1.0])
    z0 = torch.zeros(1, 5, 2)
    logw0 = torch.zeros(1, 5)
    log_m0 = torch.zeros(1)

    rollout = simulator.rollout(
        z0=z0,
        logw0=logw0,
        model=_ConstantDynamics(),
        log_m0=log_m0,
        tau_grid=tau_grid,
    )

    assert torch.equal(rollout.tau_steps, tau_grid)
    assert rollout.z_steps.shape[0] == len(tau_grid)
    assert rollout.growth_steps.shape[0] == len(tau_grid) - 1
    assert torch.allclose(rollout.terminal_z, torch.ones_like(rollout.terminal_z))
    assert torch.allclose(rollout.terminal_logw, torch.full_like(rollout.terminal_logw, 2.0))
