from __future__ import annotations

import numpy as np
import pytest
import torch

from credo.data.core import EndpointProblem, FiniteMeasure, TimeAxis
from credo.models.full_model import FullDynamicsModel
from credo.models.simulator import CounterfactualEngine, rollout_with_clamped_context
from credo.models.weighted_sde import WeightedParticleSimulator


pytestmark = pytest.mark.semantic


def _endpoint() -> EndpointProblem:
    measure = FiniteMeasure(
        support=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        weights=np.ones(3, dtype=np.float32),
        total_mass=3.0,
    )
    return EndpointProblem(
        initial={"pert": measure, "ctrl": measure},
        terminal={"pert": measure, "ctrl": measure},
        time_axis=TimeAxis(["t0", "t1"], [0.0, 1.0]),
        perturbation_ids=["pert", "ctrl"],
    )


def _model() -> FullDynamicsModel:
    torch.manual_seed(0)
    return FullDynamicsModel(
        perturbation_ids=["pert", "ctrl"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=1,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        control_mode="soft_ref",
    )


def test_counterfactual_engine_clamp_context_requires_history() -> None:
    engine = CounterfactualEngine(
        model=_model(),
        simulator=WeightedParticleSimulator(n_steps=2, store_history=False),
        n_particles=4,
    )

    with pytest.raises(ValueError, match="store_history=True"):
        engine.run(_endpoint(), ["pert"], clamp_context=True)


def test_counterfactual_engine_clamp_context_returns_same_start_branches() -> None:
    engine = CounterfactualEngine(
        model=_model(),
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=5,
    )

    result = engine.run(_endpoint(), ["pert"], clamp_context=True, seed=7)[0]

    assert result.rollout_clamped is not None
    assert result.rollout_control_clamped is not None
    assert result.rollout_control.context_steps is not None
    assert result.rollout_control.context_steps.shape[0] == result.rollout_control.n_steps

    start_z = result.rollout_perturb.z_steps[0]
    start_logw = result.rollout_perturb.logw_steps[0]
    start_log_m0 = result.rollout_perturb.log_m0
    for rollout in [
        result.rollout_control,
        result.rollout_clamped,
        result.rollout_control_clamped,
    ]:
        assert torch.equal(rollout.z_steps[0], start_z)
        assert torch.equal(rollout.logw_steps[0], start_logw)
        assert torch.equal(rollout.log_m0, start_log_m0)

    assert torch.allclose(result.rollout_control.z_steps, result.rollout_control_clamped.z_steps)
    assert torch.allclose(result.rollout_control.logw_steps, result.rollout_control_clamped.logw_steps)
    assert torch.equal(result.rollout_control.tau_steps, result.rollout_control_clamped.tau_steps)
    assert isinstance(result.terminal_log_mass_diff(), float)


def test_counterfactual_common_noise_does_not_mutate_global_rng() -> None:
    engine = CounterfactualEngine(
        model=_model(),
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=5,
    )

    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    engine.run(_endpoint(), ["pert"], clamp_context=True, seed=7, common_noise=True)
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)


def test_rollout_with_clamped_context_preserves_nonuniform_tau_grid() -> None:
    model = _model()
    tau_grid = torch.tensor([0.0, 0.2, 0.7, 1.0])
    z0 = torch.zeros(1, 4, 2)
    logw0 = torch.full((1, 4), -np.log(4.0))
    log_m0 = torch.zeros(1)
    context_steps = torch.zeros(3, 3)

    rollout = rollout_with_clamped_context(
        model=model,
        z0=z0,
        logw0=logw0,
        log_m0=log_m0,
        perturbation_ids=["pert"],
        context_steps=context_steps,
        tau_start=0.0,
        tau_end=1.0,
        tau_grid=tau_grid,
    )

    assert torch.equal(rollout.tau_steps, tau_grid)
    assert rollout.context_steps.shape[0] == len(tau_grid) - 1


def test_rollout_with_clamped_context_returns_consumed_noise() -> None:
    model = _model()
    tau_grid = torch.tensor([0.0, 0.2, 0.7, 1.0])
    z0 = torch.zeros(1, 4, 2)
    logw0 = torch.full((1, 4), -np.log(4.0))
    log_m0 = torch.zeros(1)
    context_steps = torch.zeros(3, 3)
    noise_steps = torch.arange(3 * 1 * 4 * 2, dtype=torch.float32).reshape(3, 1, 4, 2) / 100.0

    rollout = rollout_with_clamped_context(
        model=model,
        z0=z0,
        logw0=logw0,
        log_m0=log_m0,
        perturbation_ids=["pert"],
        context_steps=context_steps,
        tau_start=0.0,
        tau_end=1.0,
        tau_grid=tau_grid,
        noise_steps=noise_steps,
        return_noise_used=True,
    )

    assert rollout.noise_steps is not None
    assert torch.equal(rollout.noise_steps, noise_steps)


def test_rollout_with_clamped_context_rejects_bad_context_width() -> None:
    model = _model()
    z0 = torch.zeros(1, 4, 2)
    logw0 = torch.full((1, 4), -np.log(4.0))
    log_m0 = torch.zeros(1)

    with pytest.raises(ValueError, match="context_steps must have shape"):
        rollout_with_clamped_context(
            model=model,
            z0=z0,
            logw0=logw0,
            log_m0=log_m0,
            perturbation_ids=["pert"],
            context_steps=torch.zeros(2, 2),
            n_steps=2,
        )
