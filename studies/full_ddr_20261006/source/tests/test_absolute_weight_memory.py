"""Synthetic value/gradient and retained-storage regression for capacity OOM."""
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from credo.contracts import Axis, FiniteMeasure
from credo.recipes.compact_sde_v3 import objective
from credo.recipes.compact_sde_v3.model import CREDOModel
from credo.recipes.compact_sde_v3.particles import ParticleRollout
from credo.recipes.compact_sde_v3.training import CatalogBank
from tests.test_particle_mass_increment import make_data, make_rollout


def inputs():
    roll = make_rollout(torch.tensor([[0.4, -0.5], [-0.2, 0.7]], dtype=torch.float64), steps=4)
    roll = replace(roll,
        z_steps=(roll.z_steps.detach().clone() + torch.tensor(
            [-0.1, 0.2], dtype=torch.float64)[None, None, :, None]).requires_grad_(),
        logw_steps=roll.logw_steps.detach().clone().requires_grad_(),
        log_m0=roll.log_m0.detach().clone().requires_grad_())
    data = make_data()
    data.axis = Axis(kind="physical", labels=("early", "mid", "late"),
                     values=(8.0, 28.0, 48.0), source="early")
    data.measures["mid"] = {key: FiniteMeasure(
        np.array([[-0.2], [0.3]]), np.array([0.4, 0.6]) * mass, mass)
        for key, mass in zip(data.measure_ids, (2.2, 2.7), strict=True)}
    return roll, data


def legacy_function():
    """The exact old allocation at the otherwise unchanged call site."""
    source = inspect.getsource(objective.checkpoint_geometry_mass_loss)
    old = "rollout.log_m0[local_index] + rollout.logw_steps[step, local_index]"
    assert source.count(old) == 1
    source = source.replace(old, "rollout.absolute_log_weight_steps[step, local_index]")
    namespace = dict(objective.__dict__)
    exec(compile(source, "<synthetic-legacy-checkpoint>", "exec"), namespace)
    return namespace["checkpoint_geometry_mass_loss"]


@pytest.mark.parametrize("compute_geometry", [False, True])
@pytest.mark.parametrize("include_mass", [False, True])
def test_checkpoint_values_and_gradients_match_legacy(compute_geometry, include_mass):
    roll, data = inputs()
    kwargs = dict(mass_weight=0.3, include_mass=include_mass,
                  validation_source="synthetic", compute_geometry=compute_geometry,
                  collect_rows=False)
    current = objective.checkpoint_geometry_mass_loss(roll, data, **kwargs)
    legacy = legacy_function()(roll, data, **kwargs)
    torch.testing.assert_close(current.total, legacy.total, rtol=0, atol=0)
    assert current.observation_count == legacy.observation_count == 4
    if compute_geometry or include_mass:
        tensors = (roll.z_steps, roll.logw_steps, roll.log_m0)
        current_grad = torch.autograd.grad(
            current.total, tensors, allow_unused=True, retain_graph=True)
        legacy_grad = torch.autograd.grad(legacy.total, tensors, allow_unused=True)
        for first, second in zip(current_grad, legacy_grad, strict=True):
            if first is None:
                assert second is None
            else:
                torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_saved_autograd_storage_does_not_duplicate_full_trajectory():
    roll, data = inputs()
    # The production capacity path is float32. In float64 the existing mass
    # reduction casts each vector, so that cast can hide the retained base.
    roll = replace(roll, logw_steps=roll.logw_steps.float(), log_m0=roll.log_m0.float())
    full_bytes = roll.logw_steps.untyped_storage().nbytes()

    def retained(function):
        stores = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            stores[storage.data_ptr()] = storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = function(roll, data, mass_weight=1.0, include_mass=True,
                validation_source="synthetic", compute_geometry=False, collect_rows=False)
        # Keep result alive until the inventory is complete.
        assert torch.isfinite(result.total)
        return sum(size == full_bytes for size in stores.values())

    assert retained(legacy_function()) == 4
    assert retained(objective.checkpoint_geometry_mass_loss) == 0


def test_terminal_and_bank_paths_never_allocate_full_absolute_trajectory(monkeypatch):
    roll = make_rollout(torch.tensor([[0.4, -0.5], [-0.2, 0.7]], dtype=torch.float64), steps=4)
    expected_terminal = roll.absolute_log_weight_steps[-1]
    data = make_data()
    model = CREDOModel(embedding_ids=("control", "target"), control_embedding_ids=("control",),
        latent_dim=1, embedding_dim=2, n_programs=2, hidden_dim=8, context_mode="none").double()
    bank = CatalogBank.empty(data, model, 4, device=torch.device("cpu"), dtype=torch.float64,
                             mass_increment="particle_log_mass")
    bank.update_from_rollout(roll, model, data, full_refresh=True)
    expected_bank = {key: value.clone() for key, value in bank.tensors.items()}
    def forbidden(_):
        raise AssertionError("Full trajectory must not be allocated for a slice")
    monkeypatch.setattr(ParticleRollout, "absolute_log_weight_steps", property(forbidden))
    torch.testing.assert_close(roll.terminal_absolute_log_weight, expected_terminal, rtol=0, atol=0)
    bank.update_from_rollout(roll, model, data, full_refresh=True)
    for key, value in bank.tensors.items():
        torch.testing.assert_close(value, expected_bank[key], rtol=0, atol=0)
    objective.checkpoint_geometry_mass_loss(roll, data, mass_weight=1.0, include_mass=True,
        validation_source="synthetic", compute_geometry=False, collect_rows=False)
