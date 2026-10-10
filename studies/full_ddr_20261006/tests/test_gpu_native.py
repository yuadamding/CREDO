"""Handwritten fixtures only: no real data, remote jobs or qualification reuse."""
# ruff: noqa: E402
from dataclasses import replace
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]

import numpy as np
import pytest
import torch

from credo.recipes.compact_sde_v3.model import CREDOModel
from credo.recipes.compact_sde_v3.objective import checkpoint_geometry
from credo.recipes.compact_sde_v3.particles import axis_grid, rollout, sample_initial_particles, sample_noise
from gpu_native import BatchedGeometry, DeviceLawCache, GPUOptions, TensorDynamics, geomloss_backend, sinkhorn_batched, tensor_rollout
from test_terminal_fitting import setup_run


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    geomloss_backend.cache_clear()
    torch.set_num_threads(1)
    yield
    geomloss_backend.cache_clear()
    torch.set_num_threads(old)


@pytest.fixture
def data(tmp_path):
    problem, _ = setup_run(tmp_path)
    return problem.training


def model_for(data, *, trunk_kind="mlp"):
    return CREDOModel(embedding_ids=data.embedding_ids, control_embedding_ids=data.control_embedding_ids,
                      latent_dim=data.latent_dim, hidden_dim=16, embedding_dim=4, n_programs=2,
                      context_mode="none", trunk_kind=trunk_kind,
                      trunk_depth=2 if trunk_kind == "mlp" else 3)


def test_cache_laws_identical_and_constructed_once(data):
    cache = DeviceLawCache(data, device="cpu")
    for i, key in enumerate(cache.ids):
        offset, length = cache.source_offsets_host[i], cache.source_lengths[i]
        law = data.measures["Stim8hr"][key]
        torch.testing.assert_close(cache.source_z[offset:offset+length], torch.tensor(np.array(law.support)), rtol=0, atol=0)
        torch.testing.assert_close(cache.source_w[offset:offset+length], torch.tensor(law.normalized_weights, dtype=torch.float32), rtol=0, atol=0)
        if cache.target_lengths[i]:
            law = data.measures["Stim48hr"][key]
            offset, length = cache.target_offsets_host[i], cache.target_lengths[i]
            reference = torch.tensor(np.array(law.weights), dtype=torch.float32).clamp_min(1e-30).log()
            torch.testing.assert_close(cache.target_logw[offset:offset+length], reference, rtol=0, atol=0)
    assert cache.allocated_bytes <= cache.options.maximum_cache_bytes
    cache.close()
    with pytest.raises(ValueError, match="closed"):
        cache.sample(cache.ids, 4, seed=1)


def test_cache_budget_and_role_rejected_before_packing(data, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("packing entered before admission")
    monkeypatch.setattr(DeviceLawCache, "_pack", forbidden)
    with pytest.raises(RuntimeError, match="inventory"):
        DeviceLawCache(data, device="cpu", options=GPUOptions(maximum_cache_bytes=1))
    meta = data.measure_meta.copy()
    meta["sample_id"] = "D2"
    with pytest.raises(ValueError, match="D3/D4"):
        DeviceLawCache(replace(data, measure_meta=meta), device="cpu")


def test_cache_rejects_different_data_and_unknown_sources(data):
    cache = DeviceLawCache(data, device="cpu")
    with pytest.raises(ValueError, match="different"):
        cache.require(replace(data))
    with pytest.raises(KeyError, match="Unknown"):
        cache.sample(("D2::unknown",), 4, seed=1)
    with pytest.raises(ValueError, match="distinct"):
        cache.sample((cache.ids[0], cache.ids[0]), 4, seed=1)


def test_sampler_reproducible_and_never_selects_padding(data):
    cache = DeviceLawCache(data, device="cpu", options=GPUOptions(maximum_sampler_elements=16))
    first = cache.sample(cache.ids, 400, seed=100)
    second = cache.sample(cache.ids, 400, seed=100)
    torch.testing.assert_close(first.z, second.z, rtol=0, atol=0)
    for row, key in enumerate(cache.ids):
        source = torch.tensor(np.array(data.measures["Stim8hr"][key].support), dtype=torch.float32)
        assert bool((first.z[row, :, None, :] == source[None, :, :]).all(-1).any(-1).all())
    assert first.measure_ids == cache.ids
    torch.testing.assert_close(first.log_m0, cache.source_log_mass, rtol=0, atol=0)
    torch.testing.assert_close(first.logw.exp().sum(-1), torch.ones(len(cache.ids)))


def test_batched_sampler_retains_nonuniform_probability_law(data):
    cache = DeviceLawCache(data, device="cpu")
    first = cache.ids[0]
    source = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
    # Replace only this synthetic cache's tensors; do not invent measured cells.
    cache.source_z = torch.cat((source, cache.source_z))
    cache.source_w = torch.cat((torch.tensor([.1, .3, .6]), cache.source_w))
    cache.source_offsets += 3
    cache.source_offsets[0] = 0
    lengths = list(cache.source_lengths)
    lengths[0] = 3
    cache.source_lengths = tuple(lengths)
    cache.source_lengths_device[0] = 3
    sample = cache.sample((first,), 20000, seed=9)
    counts = sample.z[0, :, 0].long().bincount(minlength=3)
    assert bool((counts.float()/20000 - torch.tensor([.1, .3, .6])).abs().lt(.015).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("padding", [0, 3])
def test_batched_sinkhorn_matches_individual_schedules_values_and_gradients(dtype, padding):
    from importlib import import_module
    divergence = import_module(geomloss_backend().identity["divergence_module"])
    epsilon_schedule, max_diameter = divergence.epsilon_schedule, divergence.max_diameter

    torch.manual_seed(18)
    x = torch.randn(4, 8, 3, dtype=dtype, requires_grad=True)
    y = torch.randn(4, 5, 3, dtype=dtype)
    # Different diameters, with the same number of annealing iterations.
    x.data[1] *= 1.2
    x.data[2] *= 1.4
    weights = torch.randn(4, 8, dtype=dtype, requires_grad=True)
    observed = torch.randn(4, 5, dtype=dtype)
    schedules = [epsilon_schedule(2, max_diameter(x[i], y[i]), .1**.5, .5) for i in range(4)]
    scalar = torch.stack([checkpoint_geometry(x[i], weights[i], y[i], observed[i]) for i in range(4)])
    scalar_grad = torch.autograd.grad(scalar.sum(), (x, weights), retain_graph=True)
    result = []
    # Groups may have different iteration counts; never replace their schedules.
    for length in sorted({len(s) for s in schedules}):
        rows = [i for i, schedule in enumerate(schedules) if len(schedule) == length]
        ys = y[rows]
        bs = observed[rows]
        if padding:
            ys = torch.cat((ys, ys[:, :1].expand(-1, padding, -1)), 1)
            bs = torch.cat((bs, torch.full((len(rows), padding), -torch.inf, dtype=dtype)), 1)
        values = sinkhorn_batched(x[rows], weights[rows], ys, bs, [schedules[i] for i in rows])
        result.extend((row, values[j]) for j, row in enumerate(rows))
    batched = torch.stack([value for _, value in sorted(result)])
    batch_grad = torch.autograd.grad(batched.sum(), (x, weights))
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(batched, scalar, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(batch_grad, scalar_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("grad_mode", [True, False])
def test_geomloss_grad_mode_restored_even_on_failure(grad_mode, monkeypatch):
    backend = geomloss_backend()

    def failed(*args, **kwargs):
        torch.set_grad_enabled(not grad_mode)
        raise RuntimeError("synthetic solver failure")
    monkeypatch.setattr(backend, "sinkhorn_loop", failed)
    with torch.set_grad_enabled(grad_mode):
        x = torch.randn(1, 3, 2)
        with pytest.raises(RuntimeError, match="synthetic"):
            sinkhorn_batched(x, torch.zeros(1, 3), x, torch.zeros(1, 3), [[1., .1]])
        assert torch.is_grad_enabled() is grad_mode


@pytest.mark.parametrize("growth", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("trunk_kind", ["mlp", "residual_mlp"])
def test_tensor_rollout_values_and_parameter_gradients_match(growth, compiled, trunk_kind, data):
    torch.manual_seed(1)
    original = model_for(data, trunk_kind=trunk_kind)
    original.growth_enabled = growth
    optimized = copy.deepcopy(original)
    state = sample_initial_particles(data, data.measure_ids, 4, seed=3)
    grid = axis_grid(data.axis, 2, device="cpu", dtype=torch.float32)
    noise = sample_noise(state, grid, seed=16)
    reference = rollout(original, state, grid, noise=noise)
    dynamics = TensorDynamics(optimized)
    if compiled:
        # Fullgraph capture on CPU; NOT an Inductor/H100 speed qualification.
        dynamics = torch.compile(dynamics, backend="eager", fullgraph=True, dynamic=True)
    indices = optimized._residual_indices(state.embedding_ids)
    result = tensor_rollout(optimized, dynamics, state, grid, noise, indices)
    for name in ("z_steps", "logw_steps", "drift_steps", "sigma_steps", "growth_steps", "context_steps"):
        torch.testing.assert_close(getattr(result, name), getattr(reference, name), rtol=0, atol=0)
    (reference.z_steps.square().sum()+reference.logw_steps.square().sum()).backward()
    (result.z_steps.square().sum()+result.logw_steps.square().sum()).backward()
    for (key, expected), (other, actual) in zip(original.named_parameters(), optimized.named_parameters(), strict=True):
        assert key == other
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-5)


def test_ragged_geometry_matches_scalar_and_count_only_support(data):
    cache = DeviceLawCache(data, device="cpu")
    model = model_for(data)
    state = cache.sample(cache.ids, 8, seed=37)
    grid = axis_grid(data.axis, 2, device="cpu", dtype=torch.float32)
    result = rollout(model, state, grid, noise=sample_noise(state, grid, seed=123))
    geometry = BatchedGeometry(cache)
    actual, n = geometry(result)
    values = []
    for row, key in enumerate(cache.ids):
        target = data.measures["Stim48hr"].get(key)
        if target is not None:
            values.append(checkpoint_geometry(result.terminal_z[row], result.terminal_absolute_log_weight[row],
                torch.tensor(np.array(target.support), dtype=torch.float32),
                torch.tensor(np.array(target.weights), dtype=torch.float32).clamp_min(1e-30).log()))
    reference = torch.stack(values).mean()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    assert n == len(values) == 3
    missing = [i for i, key in enumerate(cache.ids) if key not in data.measures["Stim48hr"]]
    assert len(missing) == 1
    zero, n = geometry(result.slice_measure(missing[0]))
    assert float(zero.detach()) == n == 0


def test_geometry_bound_rejects_instead_of_subsampling(data):
    cache = DeviceLawCache(data, device="cpu", options=GPUOptions(maximum_geometry_elements=1))
    model = model_for(data)
    state = cache.sample(cache.ids, 4, seed=37)
    grid = axis_grid(data.axis, 2, device="cpu", dtype=torch.float32)
    result = rollout(model, state, grid, noise=sample_noise(state, grid, seed=123))
    with pytest.raises(RuntimeError, match="full-support geometry"):
        BatchedGeometry(cache)(result)


@pytest.mark.parametrize("key,value", [("maximum_cache_bytes", 0), ("maximum_geometry_guides", True),
                                     ("precision", "unknown"), ("optimizer_backend", "unknown")])
def test_execution_policy_validation(key, value):
    with pytest.raises(ValueError):
        GPUOptions(**{key: value})
