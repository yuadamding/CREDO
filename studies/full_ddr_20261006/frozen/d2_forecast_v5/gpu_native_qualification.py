"""Real fitting-input qualification. No training, selection, or fallback.

Every forward/backward and disposable Adam operation is budgeted before use.
Cold compile, warm execution and all-catalog refresh have separate accounting.
"""
from __future__ import annotations

import gc
import math
import statistics
import time

import numpy as np
import torch

from architecture_fitting import complete_count_loss, count_sampling_surrogate, validate_normalizers
from credo.recipes.compact_sde_v3.objective import checkpoint_geometry, rollout_regularization
from credo.recipes.compact_sde_v3.particles import axis_grid, rollout, sample_initial_particles, sample_noise
from credo.recipes.compact_sde_v3.recipe import recipe
from credo.recipes.compact_sde_v3.training import CatalogBank
from gpu_native import BatchedGeometry, GPUOptions, TensorDynamics, tensor_rollout

PROFILES = ("scalar_reference", "GPU_native_eager_foreach", "GPU_native_compiled_foreach")
ATOL = RTOL = 2e-5


class WorkBudget:
    def __init__(self, *, full=90, partial=12, steps=18, refresh=3, chunks=549):
        self.limits = dict(full=full, partial=partial, steps=steps, refresh=refresh, chunks=chunks)
        self.used = dict.fromkeys(self.limits, 0)

    def charge(self, key):
        if self.used[key] >= self.limits[key]:
            raise RuntimeError("Qualification work cap exhausted: " + key)
        self.used[key] += 1  # Attempted work remains charged even on failure.

    def complete(self):
        if self.used != self.limits:
            raise RuntimeError("Qualification coverage incomplete")


def execution_options(profile, *, compiler_backend="inductor"):
    if profile not in PROFILES[1:]:
        raise ValueError("Only qualified GPU-native profiles can train")
    return GPUOptions(compile_dynamics=profile == PROFILES[2], optimizer_backend="foreach",
        precision="strict_fp32", compiler_backend=compiler_backend)


def memory_guard(device, parameters=(), *, first_step=False):
    if device.type != "cuda":
        return {}  # Synthetic unit tests only; public worker rejects CPU.
    free, total = torch.cuda.mem_get_info(device)
    allocated, reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
    extra = 2 * sum(p.numel()*p.element_size() for p in parameters) if first_step else 0
    if allocated+extra > 75_000_000_000 or reserved+extra > 78_000_000_000 or free-extra < 4_000_000_000:
        raise RuntimeError("Optimizer-inclusive qualification memory admission failed")
    return dict(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, free_bytes=free,
        total_bytes=total, estimated_first_Adam_bytes=extra)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _comparison_failure(label, name, reason, *, progress=None, **details):
    record = dict(event="numerical_agreement_failed", comparison=label,
        tensor=name, reason=reason, atol=ATOL, rtol=RTOL, **details)
    if progress is not None:
        progress(record)
    suffix = "" if not details else " " + repr(details)
    raise RuntimeError(label + " " + reason + ": " + name + suffix)


def compare(actual, expected, label, *, progress=None):
    if set(actual) != set(expected):
        _comparison_failure(label, "inventory", "tensor inventory differs", progress=progress)
    maxima = []
    for name in expected:
        a, b = actual[name], expected[name]
        if a is None or b is None:
            if a is not b:
                _comparison_failure(label, name, "gradient availability differs", progress=progress)
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            _comparison_failure(label, name, "shape/dtype differs", progress=progress)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            _comparison_failure(label, name, "nonfinite", progress=progress)
        if not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
            # Host float64 diagnostic arithmetic avoids overflow when a finite
            # FP32 discrepancy is divided by a near-zero reference value.
            aa, bb = a.detach().cpu().double(), b.detach().cpu().double()
            difference = (aa-bb).abs()
            tolerance = ATOL + RTOL*bb.abs()
            ratio = difference/tolerance
            _comparison_failure(label, name, "allclose failed", progress=progress,
                maximum_absolute_difference=float(difference.max()),
                maximum_tolerance_ratio=float(ratio.max()),
                maximum_relative_tolerance_excess=float((ratio-1).clamp_min(0).max()),
                failed_elements=int((difference > tolerance).sum()), elements=a.numel())
        maxima.append(float((a-b).abs().max()) if a.numel() else 0.)
    return dict(passed=True, maximum_absolute_difference=max(maxima, default=0.), atol=ATOL, rtol=RTOL)


def exact_snapshot(actual, expected, label, *, progress=None):
    """Attest equal inputs to a backend test, rather than approximate equality."""
    if set(actual) != set(expected):
        _comparison_failure(label, "inventory", "tensor inventory differs", progress=progress)
    for name, a in actual.items():
        b = expected[name]
        if a is None or b is None:
            equal = a is b
        else:
            equal = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        if not equal:
            _comparison_failure(label, name, "exact identity failed", progress=progress)
    return dict(passed=True, comparison="exact_tensor_equality", tensors=len(actual))


def install_gradients(model, snapshot):
    """Copy common gradients for the disposable backend test only."""
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(snapshot):
        raise RuntimeError("Disposable optimizer gradient inventory differs")
    for name, parameter in parameters.items():
        gradient = snapshot[name]
        if gradient is None:
            parameter.grad = None
        else:
            if parameter.shape != gradient.shape or parameter.dtype != gradient.dtype:
                raise RuntimeError("Disposable optimizer gradient shape/dtype differs: " + name)
            parameter.grad = gradient.to(device=parameter.device).clone()


def optimizer_snapshot(model, optimizer):
    """Record moments and step counters as well as the resulting parameters."""
    result = {}
    for name, parameter in model.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if not torch.is_tensor(value):
                raise RuntimeError("Unexpected non-tensor Adam state: " + key)
            result[name + "/" + key] = value.detach().cpu().clone()
    return result


def first_adam_sensitivity(initial, actual, expected, *, lr=.001, eps=1e-8):
    """Describe independent-gradient sensitivity without an optimizer call.

    At fresh plain Adam state the real-arithmetic first step is
    ``theta - lr*g/(abs(g)+eps)``. This quantity can differ considerably for
    gradients that pass their own absolute tolerance near zero. It is not an
    optimizer-backend equivalence test and does not change any numerical gate.
    """
    maximum, maximum_ratio, failures, sign_changes, elements = 0., 0., 0, 0, 0
    worst = None
    for name, gradient in actual.items():
        reference = expected[name]
        if gradient is None and reference is None:
            continue
        if gradient is None or reference is None:
            raise RuntimeError("Sensitivity gradient availability differs: " + name)
        a, b, start = gradient.double(), reference.double(), initial[name].double()
        first, second = lr*a/(a.abs()+eps), lr*b/(b.abs()+eps)
        difference = (first-second).abs()
        ratio = difference/(ATOL+RTOL*(start-second).abs())
        value = float(difference.max()) if difference.numel() else 0.
        if value > maximum:
            maximum, worst = value, name
        maximum_ratio = max(maximum_ratio, float(ratio.max()) if ratio.numel() else 0.)
        failures += int((ratio > 1).sum())
        sign_changes += int(((a*b) < 0).sum())
        elements += a.numel()
    return dict(status="descriptive_independent_gradient_sensitivity",
        formula="fresh_plain_Adam_real_arithmetic_theta_minus_lr_g_over_abs_g_plus_eps",
        arithmetic="float64_from_saved_FP32_gradients", learning_rate=lr, epsilon=eps,
        maximum_absolute_parameter_difference=maximum, worst_tensor=worst,
        maximum_tolerance_ratio=maximum_ratio, elements_exceeding_parameter_tolerance=failures,
        gradient_sign_changes=sign_changes, elements=elements,
        additional_optimizer_steps=0, affects_backend_acceptance=False)


def gradient_snapshot(model):
    return {name: None if p.grad is None else p.grad.detach().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def parameters_snapshot(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def head_gradients(gradients, stage):
    maxima = {head: max((float(value.abs().max()) for name, value in gradients.items()
        if value is not None and name.startswith(prefix)), default=0.)
        for head, prefix in (("drift", "drift_"), ("diffusion", "sigma_"), ("reaction", "growth_"))}
    required = ("drift", "diffusion") if stage == "state" else ("drift", "diffusion", "reaction")
    if any(not math.isfinite(maxima[key]) or maxima[key] <= 0 for key in required):
        raise RuntimeError("Required stage head gradients absent/nonfinite")
    return maxima


def scalar_geometry(result, data):
    values = []
    for row, key in enumerate(result.measure_ids):
        target = data.measures["Stim48hr"].get(key)
        if target is not None:
            z = torch.tensor(np.array(target.support, copy=True), device=result.z_steps.device, dtype=torch.float32)
            logw = torch.tensor(np.array(target.weights, copy=True), device=z.device, dtype=torch.float32).log()
            values.append(checkpoint_geometry(result.terminal_z[row], result.log_m0[row]+result.logw_steps[-1, row],
                z, logw, epsilon=.1))
    return (torch.stack(values).mean(), len(values)) if values else (result.z_steps.sum()*0., 0)


def balanced_loss(model, result, data, cache, bank, normalizers, contract, stage, *, scalar):
    geometry, n = scalar_geometry(result, data) if scalar else BatchedGeometry(cache)(result)
    count = complete_count_loss(result, data, bank, None) if scalar and stage == "mass" else (
        cache.count_loss(result, bank) if stage == "mass" else geometry.new_zeros(()))
    scaled = count_sampling_surrogate(count, population=len(data.measure_ids), batch=len(result.measure_ids))
    settings = contract["loss"]
    regularizer = rollout_regularization(result, drift_weight=settings["action_drift"],
        diffusion_weight=settings["action_diffusion"], growth_weight=settings["action_growth"])
    regularizer = regularizer + model.regularization(coefficient=settings["parameter_regularization"])
    loss = .5/normalizers["state"]*geometry + .5/normalizers["count"]*scaled + regularizer
    return loss, (geometry, count, regularizer), n


def one_call(model, profile, ids, *, data, cache, bank, grid, dynamics, normalizers, contract,
             stage, budget, kind, seed, common=None, particles=512):
    budget.charge(kind)
    model.zero_grad(set_to_none=True)
    synchronize(cache.device)
    started = time.perf_counter()
    checkpoints = []
    def mark(name):
        if cache.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()
        checkpoints.append((name, event))
    mark("start")
    if common is None:
        initial = (sample_initial_particles(data, ids, particles, device=cache.device, seed=seed)
            if profile == PROFILES[0] else cache.sample(ids, particles, seed=seed))
        noise = sample_noise(initial, grid, seed=seed+1_000_003)
    else:
        initial, noise = common
    mark("sampling")
    result = (rollout(model, initial, grid, noise=noise) if profile == PROFILES[0] else
        tensor_rollout(model, dynamics, initial, grid, noise, model._residual_indices(initial.embedding_ids)))
    mark("rollout")
    loss, pieces, observations = balanced_loss(model, result, data, cache, bank, normalizers, contract,
        stage, scalar=profile == PROFILES[0])
    mark("objective")
    loss.backward()
    mark("backward")
    synchronize(cache.device)
    elapsed = time.perf_counter()-started
    values = torch.stack((loss.detach(), *(x.detach() for x in pieces))).cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError("Nonfinite qualification objective")
    if cache.device.type == "cuda":
        component = {name: left.elapsed_time(right)/1000 for (_, left), (name, right)
            in zip(checkpoints, checkpoints[1:])}
    else:
        component = {name: right-left for (_, left), (name, right) in zip(checkpoints, checkpoints[1:])}
    memory = memory_guard(cache.device)
    return dict(elapsed_seconds=elapsed, component_seconds=component, loss=float(values[0]),
        geometry=float(values[1]), count_ce=float(values[2]), regularization=float(values[3]),
        observed_state_laws=observations, guides=len(ids), **memory), values


@torch.no_grad()
def refresh(model, dynamics, *, data, cache, grid, bank, budget, progress, seed=0):
    budget.charge("refresh")
    synchronize(cache.device)
    started = time.perf_counter()
    bank.reset_coverage()
    bank.refresh_fixed_background(model)
    bank.refresh_particles = 32
    for chunk, start in enumerate(range(0, len(data.measure_ids), 256)):
        budget.charge("chunks")
        initial = cache.sample(data.measure_ids[start:start+256], 32, seed=seed+chunk)
        result = tensor_rollout(model, dynamics, initial, grid, sample_noise(initial, grid, seed=seed+chunk+2_000_003),
            model._residual_indices(initial.embedding_ids))
        bank.update_from_rollout(result, model, data, full_refresh=True)
        if chunk % 20 == 0:
            progress(dict(event="qualification_refresh_progress", chunks=budget.used["chunks"]))
        del initial, result
    bank.last_full_refresh_epoch = 0
    bank.assert_complete()
    synchronize(cache.device)
    return time.perf_counter()-started


def choose_profile(rows, refresh_seconds, *, state_updates=1680, mass_updates=880, refresh_passes=22):
    """One execution policy for all architectures; no accuracy or holdout scores."""
    projected = {}
    for profile in PROFILES[1:]:
        cost = 0.
        for candidate in refresh_seconds:
            by_stage = {row["stage"]: row for row in rows if row["candidate_id"] == candidate
                and row["profile"] == profile}
            if set(by_stage) != {"state", "mass"}:
                raise RuntimeError("Missing qualified timing profile")
            cost += 3*(state_updates*by_stage["state"]["median_warm_seconds"]
                +mass_updates*by_stage["mass"]["median_warm_seconds"]+refresh_passes*refresh_seconds[candidate])
            # One cold compile per stage/fit; its overhead is not hidden.
            cost += 3*sum(max(0., row["calls"][0]["elapsed_seconds"]-row["median_warm_seconds"])
                for row in by_stage.values())
        projected[profile] = cost
    return min(PROFILES[1:], key=lambda key: projected[key]), projected


def full_support_qualification_ids(cache, eligible, full_guides, particles):
    """Force every exact-width exception into the SAME bounded full batch.

    No effect/outcome values or performance scores select this challenge set.
    The remaining IDs retain their canonical order; exposure is disposable.
    """
    from gpu_native import geometry_admission
    admission = geometry_admission(cache, particles)
    eligible = tuple(eligible)
    allowed = set(eligible)
    required = tuple(row["measure_id"] for row in admission["exact_width_laws"]
                     if row["measure_id"] in allowed)
    if len(required) > full_guides:
        raise RuntimeError("Exact-width challenge inventory exceeds qualification batch; no silent omission")
    required_set = set(required)
    ids = required + tuple(key for key in eligible if key not in required_set)[:full_guides-len(required)]
    if len(ids) != full_guides or len(set(ids)) != full_guides:
        raise RuntimeError("Distinct full-support qualification shape unavailable")
    return ids


def qualify(problem, candidates, config, contract, normalizers, cache, *, progress, budget,
            full_guides=1086, partial_state=488, partial_mass=111, particles=512,
            steps=8, compiler_backend="inductor", require_full_support_challenge=False):
    validate_normalizers(normalizers)
    data, rows, partial_rows, refresh_seconds = problem.training, [], [], {}
    paired = tuple(key for key in data.measure_ids if key in data.measures["Stim48hr"])
    grid = axis_grid(data.axis, steps, device=cache.device, dtype=torch.float32)
    for candidate in candidates:
        cid = candidate["candidate_id"]
        cfg = config(candidate, 0)
        torch.manual_seed(0)
        original = recipe.build_model(problem, cfg).to(cache.device)
        pristine = {key: value.detach().cpu().clone() for key, value in original.state_dict().items()}
        original.set_phase("mass")
        bank = CatalogBank.empty(data, original, steps, device=cache.device, dtype=torch.float32,
            momentum=0., mass_increment="particle_log_mass")
        refresh_seconds[cid] = refresh(original, TensorDynamics(original), data=data, cache=cache,
            grid=grid, bank=bank, budget=budget, progress=progress)
        del original
        references = {}
        for profile in PROFILES:
            model = recipe.build_model(problem, cfg).to(cache.device)
            model.load_state_dict(pristine)
            dynamics = TensorDynamics(model)
            if profile == PROFILES[2]:
                torch.compiler.reset()
                dynamics = torch.compile(dynamics, backend=compiler_backend, fullgraph=True, dynamic=True)
            for stage, eligible, partial_size in (("state", paired, partial_state), ("mass", data.measure_ids, partial_mass)):
                model.load_state_dict(pristine)
                model.set_phase(stage)
                if cache.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(cache.device)
                ids = (full_support_qualification_ids(cache, eligible, full_guides, particles)
                       if require_full_support_challenge else tuple(eligible[:full_guides]))
                if len(ids) != full_guides:
                    raise RuntimeError("Qualification full shape unavailable")
                calls = []
                for repetition in range(4):
                    value, _ = one_call(model, profile, ids, data=data, cache=cache, bank=bank,
                        grid=grid, dynamics=dynamics, normalizers=normalizers, contract=contract,
                        stage=stage, budget=budget, kind="full", seed=29+repetition, particles=particles)
                    calls.append(value)
                    progress(dict(event="qualification_call", candidate_id=cid, profile=profile,
                        stage=stage, repetition=repetition, budget=dict(budget.used), **value))
                common = sample_initial_particles(data, ids, particles, device=cache.device, seed=71)
                common = (common, sample_noise(common, grid, seed=1_000_074))
                numeric, values = one_call(model, profile, ids, data=data, cache=cache, bank=bank,
                    grid=grid, dynamics=dynamics, normalizers=normalizers, contract=contract,
                    stage=stage, budget=budget, kind="full", seed=71, common=common, particles=particles)
                gradients = gradient_snapshot(model)
                heads = head_gradients(gradients, stage)
                parameters = [p for p in model.parameters() if p.requires_grad]
                norm = torch.nn.utils.clip_grad_norm_(parameters, 10., foreach=profile != PROFILES[0], error_if_nonfinite=True)
                clipped = gradient_snapshot(model)
                initial_parameters = parameters_snapshot(model)
                def agreement_progress(record):
                    progress(dict(record, candidate_id=cid, profile=profile, stage=stage,
                        budget=dict(budget.used)))
                agreement = dict(reference=True)
                sensitivity = None
                if profile != PROFILES[0]:
                    ref = references[stage]
                    # Independent forward/backward and clipping checks retain
                    # the bound tolerances. Adam itself must receive IDENTICAL
                    # gradients, as required by the original step contract.
                    agreement = dict(
                        loss=compare({"values": values}, {"values": ref["values"]}, "objective", progress=agreement_progress),
                        gradients=compare(gradients, ref["gradients"], "gradients", progress=agreement_progress),
                        clipped=compare(clipped, ref["clipped"], "clipped gradients", progress=agreement_progress),
                        norm=compare({"norm": norm.detach().cpu()}, {"norm": ref["norm"]}, "clip norm", progress=agreement_progress))
                    agreement["optimizer_initial_parameters"] = exact_snapshot(initial_parameters,
                        ref["initial_parameters"], "Adam initial parameters", progress=agreement_progress)
                    sensitivity = first_adam_sensitivity(
                        initial_parameters, clipped, ref["clipped"])
                    agreement_progress(dict(event="independent_gradient_Adam_sensitivity",
                        **sensitivity))
                    install_gradients(model, ref["clipped"])
                    agreement["optimizer_input_gradients"] = exact_snapshot(gradient_snapshot(model),
                        ref["clipped"], "Adam common clipped gradients", progress=agreement_progress)
                optimizer = torch.optim.Adam(parameters, lr=.001, betas=(.9, .999), eps=1e-8,
                    weight_decay=0., amsgrad=False, foreach=profile != PROFILES[0], fused=False)
                if optimizer.state:
                    raise RuntimeError("Disposable Adam must start from empty optimizer state")
                agreement["optimizer_initial_state"] = dict(passed=True, state="identical_empty_fresh_plain_Adam")
                memory_guard(cache.device, parameters, first_step=True)
                budget.charge("steps")
                optimizer.step()
                synchronize(cache.device)
                memory = memory_guard(cache.device)
                updated = parameters_snapshot(model)
                optimizer_state = optimizer_snapshot(model, optimizer)
                result = dict(candidate_id=cid, profile=profile, stage=stage, calls=calls,
                    median_warm_seconds=statistics.median(row["elapsed_seconds"] for row in calls[1:]),
                    numerical_call=numeric, preclip_norm=float(norm), head_gradients=heads,
                    optimizer_state_bytes=sum(value.numel()*value.element_size() for state in optimizer.state.values()
                        for value in state.values() if torch.is_tensor(value)), **memory)
                if require_full_support_challenge:
                    from gpu_native import geometry_admission
                    required = [row["measure_id"] for row in geometry_admission(cache, particles)["exact_width_laws"]]
                    if not set(required).issubset(ids):
                        raise RuntimeError("Full-shape qualification omitted an exact-width support law")
                    result["full_support_challenge"] = dict(exact_width_measure_ids=required,
                        full_batch_guides=len(ids), all_required_included=True)
                if profile == PROFILES[0]:
                    references[stage] = dict(values=values.clone(), gradients=gradients, clipped=clipped,
                        norm=norm.detach().cpu(), initial_parameters=initial_parameters,
                        updated=updated, optimizer_state=optimizer_state)
                else:
                    agreement["Adam"] = compare(updated, ref["updated"],
                        "disposable Adam parameters", progress=agreement_progress)
                    agreement["Adam_state"] = compare(optimizer_state, ref["optimizer_state"],
                        "disposable Adam state", progress=agreement_progress)
                result["numerical_agreement"] = agreement
                if sensitivity is not None:
                    result["independent_gradient_Adam_sensitivity"] = sensitivity
                agreement_progress(dict(event="qualification_numerical_agreement", numerical_agreement=agreement))
                rows.append(result)
                del optimizer, gradients, clipped, updated, optimizer_state, initial_parameters, common, parameters
                model.load_state_dict(pristine)
                model.zero_grad(set_to_none=True)
                if profile != PROFILES[1]:
                    ids_partial = tuple(eligible[-partial_size:])
                    initial = sample_initial_particles(data, ids_partial, particles, device=cache.device, seed=97)
                    common = (initial, sample_noise(initial, grid, seed=1_000_100))
                    value, values = one_call(model, profile, ids_partial, data=data, cache=cache, bank=bank,
                        grid=grid, dynamics=dynamics, normalizers=normalizers, contract=contract,
                        stage=stage, budget=budget, kind="partial", seed=97, common=common, particles=particles)
                    gradients = gradient_snapshot(model)
                    if profile == PROFILES[0]:
                        references[stage]["partial"] = (values.clone(), gradients)
                    else:
                        old_values, old_gradients = references[stage]["partial"]
                        value["numerical_agreement"] = dict(loss=compare({"values": values}, {"values": old_values}, "partial loss", progress=agreement_progress),
                            gradients=compare(gradients, old_gradients, "partial gradients", progress=agreement_progress))
                    partial_rows.append(dict(candidate_id=cid, profile=profile, stage=stage, **value))
                    del initial, common, gradients
                model.zero_grad(set_to_none=True)
            del model, dynamics
            gc.collect()
            if cache.device.type == "cuda":
                torch.cuda.empty_cache()
        del bank, pristine, references
    budget.complete()
    selected, cost = choose_profile(rows, refresh_seconds)
    return dict(status="qualified_gpu_native_full_ddr_execution", profiles=rows, partial_profiles=partial_rows,
        refresh_seconds=refresh_seconds, selected_profile=selected, timing_projection_seconds=cost,
        budget_used=dict(budget.used), scientific_optimizer_updates=0, disposable_optimizer_steps=budget.used["steps"],
        numerical_tolerance=dict(atol=ATOL, rtol=RTOL), sampler_bitwise_historical_replay=False)
