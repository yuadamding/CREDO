"""Full-support padding regression; synthetic data only, no GPU claims."""
# ruff: noqa: F811
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gpu_native import BatchedGeometry, DeviceLawCache, GPUOptions, geometry_admission, geometry_width
from gpu_native_qualification import full_support_qualification_ids
from test_gpu_native import data, model_for, bounded_threads  # noqa: F401
from credo.recipes.compact_sde_v3.objective import checkpoint_geometry
from credo.recipes.compact_sde_v3.particles import axis_grid, rollout, sample_noise


@pytest.mark.parametrize("support,expected", [(2048, 2048), (2300, 2300), (2782, 2782), (2948, 2948), (3584, 3584)])
def test_real_shape_metadata_admitted_without_extra_workspace(support, expected):
    assert geometry_width(support, 512, GPUOptions()) == expected


def test_genuinely_oversized_law_still_fails_closed():
    with pytest.raises(RuntimeError, match="exact_elements"):
        geometry_width(3585, 512, GPUOptions())


def test_admission_sees_all_laws_not_only_first_batch():
    cache = SimpleNamespace(ids=("first", "count_only", "last"), target_lengths=(2, 0, 2948),
        options=GPUOptions(), data=None, require=lambda data: None)
    receipt = geometry_admission(cache, 512)
    assert receipt["paired_laws"] == 2 and receipt["dropped_cells"] == 0
    assert receipt["exact_width_laws"] == [dict(measure_id="last", support=2948, width=2948, pairwise_elements=11971600)]
    assert full_support_qualification_ids(cache, cache.ids, 2, 512) == ("last", "first")
    assert full_support_qualification_ids(cache, ("first",), 1, 512) == ("first",)
    cache.target_lengths = (2300, 2782, 2948)
    with pytest.raises(RuntimeError, match="no silent omission"):
        full_support_qualification_ids(cache, cache.ids, 2, 512)
    cache.target_lengths = (2, 0, 3585)
    with pytest.raises(RuntimeError, match="last"):
        geometry_admission(cache, 512)


@pytest.mark.parametrize("support", [5, 6, 7])
def test_exact_width_value_and_model_gradients(data, support):
    # Keep support real within this synthetic empirical law; do not truncate it.
    torch.manual_seed(51)
    measure_id = next(iter(data.measures["Stim48hr"]))
    endpoint = data.measures["Stim48hr"][measure_id]
    weights = np.arange(1, support+1, dtype=float)
    weights *= endpoint.total_mass / weights.sum()
    endpoint = replace(endpoint, support=np.random.default_rng(7).normal(size=(support, data.latent_dim)),
        weights=weights)
    laws = {name: dict(values) for name, values in data.measures.items()}
    laws["Stim48hr"][measure_id] = endpoint
    changed = replace(data, measures=laws)
    # Exact (4+support)^2 fits, padded (4+8)^2 does not.
    cache = DeviceLawCache(changed, device="cpu", options=GPUOptions(maximum_geometry_elements=(4+support)**2))
    model = model_for(changed)
    initial = cache.sample((measure_id,), 4, seed=8)
    grid = axis_grid(changed.axis, 2, device="cpu", dtype=torch.float32)
    result = rollout(model, initial, grid, noise=sample_noise(initial, grid, seed=9))
    actual, n = BatchedGeometry(cache)(result)
    reference = checkpoint_geometry(result.terminal_z[0], result.terminal_absolute_log_weight[0],
        torch.tensor(np.array(endpoint.support), dtype=torch.float32),
        torch.tensor(np.array(endpoint.weights), dtype=torch.float32).clamp_min(1e-30).log())
    assert n == 1
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    parameters = tuple(model.parameters())
    a = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(reference, parameters, allow_unused=True)
    for observed, expected in zip(a, b, strict=True):
        if expected is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-5)
