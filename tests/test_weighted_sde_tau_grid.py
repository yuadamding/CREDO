from __future__ import annotations

from types import SimpleNamespace

import pytest
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
    assert rollout.context_steps.shape[0] == len(tau_grid) - 1
    assert torch.allclose(rollout.terminal_z, torch.ones_like(rollout.terminal_z))
    assert torch.allclose(rollout.terminal_logw, torch.full_like(rollout.terminal_logw, 2.0))


def test_rollout_accepts_explicit_noise_without_consuming_global_rng() -> None:
    simulator = WeightedParticleSimulator(n_steps=3, store_history=False)
    z0 = torch.zeros(1, 4, 2)
    logw0 = torch.zeros(1, 4)
    log_m0 = torch.zeros(1)
    noise_steps = torch.zeros(3, 1, 4, 2)

    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    rollout = simulator.rollout(
        z0=z0,
        logw0=logw0,
        model=_ConstantDynamics(),
        log_m0=log_m0,
        noise_steps=noise_steps,
    )
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert torch.allclose(rollout.terminal_z, torch.ones_like(rollout.terminal_z))


def test_rollout_can_return_exact_noise_used() -> None:
    simulator = WeightedParticleSimulator(n_steps=3, store_history=False)
    z0 = torch.zeros(1, 4, 2)
    logw0 = torch.zeros(1, 4)
    log_m0 = torch.zeros(1)
    noise_steps = torch.arange(24, dtype=torch.float32).reshape(3, 1, 4, 2) / 100.0

    rollout = simulator.rollout(
        z0=z0,
        logw0=logw0,
        model=_ConstantDynamics(),
        log_m0=log_m0,
        noise_steps=noise_steps,
        return_noise_used=True,
    )

    assert rollout.noise_steps is not None
    assert torch.equal(rollout.noise_steps, noise_steps)


def test_rollout_rejects_bad_noise_shape() -> None:
    simulator = WeightedParticleSimulator(n_steps=3, store_history=False)
    z0 = torch.zeros(1, 4, 2)
    logw0 = torch.zeros(1, 4)
    log_m0 = torch.zeros(1)

    with pytest.raises(ValueError, match="noise_steps must have shape"):
        simulator.rollout(
            z0=z0,
            logw0=logw0,
            model=_ConstantDynamics(),
            log_m0=log_m0,
            noise_steps=torch.zeros(2, 1, 4, 2),
        )


def test_sample_noise_for_tau_grid_uses_grid_step_count_without_global_rng() -> None:
    z0 = torch.zeros(1, 4, 2)
    tau_grid = torch.tensor([0.0, 0.2, 0.7, 1.0])

    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    noise = WeightedParticleSimulator.sample_noise_for_tau_grid(z0, tau_grid, seed=5)
    rng_after = torch.random.get_rng_state()

    assert noise.shape == (len(tau_grid) - 1, 1, 4, 2)
    assert torch.equal(rng_before, rng_after)


def test_effective_sample_size_detects_weight_degeneracy() -> None:
    logw_uniform = torch.zeros(1, 5)
    logw_degenerate = torch.tensor([[0.0, -20.0, -20.0, -20.0, -20.0]])

    assert torch.allclose(
        WeightedParticleSimulator.ess_fraction(logw_uniform),
        torch.ones(1),
    )
    assert float(WeightedParticleSimulator.ess_fraction(logw_degenerate)) < 0.21
    assert float(WeightedParticleSimulator.max_weight_fraction(logw_degenerate)) > 0.999
    assert float(WeightedParticleSimulator.log_weight_range(logw_degenerate)) == 20.0


def test_rollout_records_weight_diagnostics_for_every_step() -> None:
    simulator = WeightedParticleSimulator(n_steps=3, store_history=False)
    z0 = torch.zeros(1, 4, 2)
    logw0 = torch.zeros(1, 4)
    log_m0 = torch.zeros(1)

    rollout = simulator.rollout(
        z0=z0,
        logw0=logw0,
        model=_ConstantDynamics(),
        log_m0=log_m0,
        noise_steps=torch.zeros(3, 1, 4, 2),
    )

    assert rollout.ess_steps is not None
    assert rollout.ess_frac_steps is not None
    assert rollout.logw_range_steps is not None
    assert rollout.max_weight_frac_steps is not None
    assert rollout.ess_steps.shape == (4, 1)
    assert torch.allclose(rollout.ess_frac_steps, torch.ones_like(rollout.ess_frac_steps))
    assert torch.allclose(rollout.logw_range_steps, torch.zeros_like(rollout.logw_range_steps))
    assert torch.allclose(
        rollout.max_weight_frac_steps,
        torch.full_like(rollout.max_weight_frac_steps, 0.25),
    )
