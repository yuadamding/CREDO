from __future__ import annotations

import numpy as np
import pytest
import torch

from credo.data.core import EndpointProblem, FiniteMeasure, TimeAxis
from credo.models.full_model import FullDynamicsModel
from credo.models.simulator import CounterfactualEngine
from credo.models.weighted_sde import WeightedParticleSimulator


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
