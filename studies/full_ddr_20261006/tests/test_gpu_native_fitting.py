"""Synthetic end-to-end fits, counts, guards and deferred-assessment tests."""
# ruff: noqa: E402
from pathlib import Path
from dataclasses import replace
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]

import pytest
import torch

from architecture_fitting import complete_count_loss
from credo.recipes.compact_sde_v3.training import Trainer
from credo.runtime import TrainingEngine
from gpu_native import DeviceLawCache, GPUOptions, POLICY
from gpu_native_fitting import GPUFittingRecipe, GPUFittingTrainer
from test_architecture_fitting import execution, normalizers
from test_terminal_fitting import setup_run


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("optimizer", ["single_tensor", "foreach"])
@pytest.mark.parametrize("compiled", [False, True])
def test_full_staged_fit_and_publication_never_scores_holdout(tmp_path, monkeypatch, optimizer, compiled):
    problem, config = setup_run(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Holdout scorer entered fitting")
    monkeypatch.setattr(Trainer, "_evaluate_ids", forbidden)
    monkeypatch.setattr(Trainer, "_validation_count_loss", forbidden)
    events = []
    options = GPUOptions(optimizer_backend=optimizer, compile_dynamics=compiled, compiler_backend="eager", timing_interval=1)
    recipe = GPUFittingRecipe(normalizers(), execution(), options=options, progress=events.append)
    trainer = TrainingEngine().fit(recipe, problem, config, device="cpu")
    state_ids = trainer._training_ids_for_stage(trainer.training_plan.stages[0])
    all_ids = trainer._training_ids_for_stage(trainer.training_plan.stages[1])
    assert trainer.optimizer_updates == trainer.attempted_optimizer_updates == math.ceil(len(state_ids)/2)+math.ceil(len(all_ids)/2)
    assert trainer.training_guide_exposures == len(state_ids)+len(all_ids)
    assert trainer.metrics.empty and all(not r["validation_monitored"] for r in trainer.history_rows)
    assert all(v > 0 for v in trainer.head_gradient_maxima.values())
    assert trainer.execution_trace[-1]["event"] == POLICY
    assert trainer.execution_trace[-1]["sampler_bitwise_scalar_replay"] is False
    completed = [r for r in events if r["event"] == "scientific_update_completed"]
    assert len(completed) == trainer.optimizer_updates
    assert all(r["timing_backend"] == "cpu_reference_clock" for r in completed)
    assert all("geometry" in r["component_milliseconds"] for r in completed)
    trainer.save()
    saved = torch.load(config.output / "state/checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved["completed_epochs"] == 2
    assert saved["execution_trace"][-1]["event"] == POLICY
    for name, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(saved["model_state"][name], tensor, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        trainer.evaluate()
    with pytest.raises(ValueError, match="different"):
        trainer._rollout_ids(problem.validation.measure_ids, particles=4, seed=3,
            provider=trainer._provider(trainer.training_plan.stages[0]), data=problem.validation)
    cache = trainer._law_cache
    trainer.close()
    assert cache.closed


def test_cached_complete_count_value_and_active_gradients_match(tmp_path):
    problem, config = setup_run(tmp_path)
    trainer = TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution()), problem, config, device="cpu")
    # Keep the opposite donor wholly inactive in this batch; both blocks remain.
    ids = (trainer.data.measure_ids[0],)
    result = trainer._rollout_ids(ids, particles=4, seed=180, provider=trainer._provider(trainer.training_plan.stages[1]))
    original = complete_count_loss(result, trainer.data, trainer.bank, trainer.log_count_concentration)
    optimized = trainer._law_cache.count_loss(result, trainer.bank)
    torch.testing.assert_close(optimized, original, rtol=0, atol=0)
    names = [(name, p) for name, p in trainer.model.named_parameters() if p.requires_grad]
    first = torch.autograd.grad(original, [p for _, p in names], allow_unused=True, retain_graph=True)
    second = torch.autograd.grad(optimized, [p for _, p in names], allow_unused=True)
    for actual, expected in zip(second, first, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    trainer.close()


def test_shared_cache_reused_without_repacking_and_not_closed_by_fit(tmp_path, monkeypatch):
    problem, config = setup_run(tmp_path)
    options = GPUOptions()
    cache = DeviceLawCache(problem.training, device="cpu", options=options)
    def forbidden(*args, **kwargs):
        raise AssertionError("Retained law cache was repacked")
    monkeypatch.setattr(DeviceLawCache, "_pack", forbidden)
    first = TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), cache=cache), problem, config, device="cpu")
    first.close()
    assert not cache.closed
    second_config = config.model_copy(update={"output": tmp_path / "second"})
    second = TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), cache=cache), problem, second_config, device="cpu")
    second.close()
    assert not cache.closed
    cache.close()


def test_wrong_cache_object_and_policy_rejected(tmp_path):
    problem, config = setup_run(tmp_path)
    cache = DeviceLawCache(problem.training, device="cpu")
    changed = replace(problem, training=replace(problem.training))
    with pytest.raises(ValueError, match="different"):
        TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), cache=cache), changed, config, device="cpu")
    with pytest.raises(ValueError, match="policy differ"):
        TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), cache=cache,
            options=GPUOptions(maximum_geometry_guides=1)), problem, config, device="cpu")
    cache.close()


def test_missing_endpoint_count_only_step_stays_available(tmp_path):
    from credo.io import RunConfig
    problem, config = setup_run(tmp_path)
    raw = config.model_dump()
    raw["recipe_config"]["training"]["measures_per_batch"] = 1
    config = RunConfig.model_validate(raw)
    events = []
    trainer = TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), progress=events.append),
                                  problem, config, device="cpu")
    missing = [r for r in events if r.get("geometry_backend") == "not_evaluated_no_state_observations"]
    assert len(missing) == 1 and missing[0]["stage"] == "mass"
    assert trainer.optimizer_updates == trainer.training_guide_exposures == 7
    trainer.close()


def test_post_step_failure_retains_applied_count(tmp_path, monkeypatch):
    problem, config = setup_run(tmp_path)
    events = []
    def failed(self, *, after_step=False, **kwargs):
        if after_step:
            raise RuntimeError("synthetic post-step memory failure")
        return {}
    monkeypatch.setattr(GPUFittingTrainer, "_memory_guard", failed)
    with pytest.raises(RuntimeError, match="post-step memory"):
        TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), progress=events.append), problem, config, device="cpu")
    assert events[-1]["event"] == "optimizer_step_applied" and events[-1]["optimizer_updates"] == 1
    assert not config.output.exists()


def test_nonfinite_parameter_failure_is_not_converted_to_success(tmp_path, monkeypatch):
    problem, config = setup_run(tmp_path)
    events = []
    original = torch.optim.Adam.step
    def damaged(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(torch.nan)
        return output
    monkeypatch.setattr(torch.optim.Adam, "step", damaged)
    with pytest.raises(RuntimeError, match="nonfinite parameters"):
        TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), progress=events.append), problem, config, device="cpu")
    assert events[-1]["event"] == "optimizer_step_applied" and events[-1]["optimizer_updates"] == 1
    assert not config.output.exists()


@pytest.mark.parametrize("options", [GPUOptions(optimizer_backend="fused"), GPUOptions(precision="tf32"),
                                     GPUOptions(precision="bf16_dynamics")])
def test_accelerated_policies_do_not_fallback_to_cpu(tmp_path, options):
    problem, config = setup_run(tmp_path)
    with pytest.raises(ValueError, match="CUDA qualification"):
        TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(), options=options), problem, config, device="cpu")


def test_compilation_failure_remains_failure(tmp_path, monkeypatch):
    problem, config = setup_run(tmp_path)
    def forbidden(*args, **kwargs):
        raise RuntimeError("synthetic compilation failure")
    monkeypatch.setattr(torch, "compile", forbidden)
    with pytest.raises(RuntimeError, match="compilation"):
        TrainingEngine().fit(GPUFittingRecipe(normalizers(), execution(),
            options=GPUOptions(compile_dynamics=True)), problem, config, device="cpu")
    assert not config.output.exists()


def test_foreach_adam_agrees_with_scalar_backend_over_repeated_steps():
    generator = torch.Generator().manual_seed(83)
    scalar = [torch.nn.Parameter(torch.randn(shape, generator=generator)) for shape in ((8, 5), (8,), (3, 2))]
    batched = [torch.nn.Parameter(p.detach().clone()) for p in scalar]
    first = torch.optim.Adam(scalar, lr=.001, foreach=False, fused=False)
    second = torch.optim.Adam(batched, lr=.001, foreach=True, fused=False)
    for _ in range(20):
        for left, right in zip(scalar, batched, strict=True):
            gradient = torch.randn(left.shape, generator=generator)
            left.grad, right.grad = gradient, gradient.clone()
        first.step()
        second.step()
    for left, right in zip(scalar, batched, strict=True):
        torch.testing.assert_close(right, left, rtol=2e-6, atol=2e-6)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(second.state[right][key], first.state[left][key], rtol=2e-6, atol=2e-6)
