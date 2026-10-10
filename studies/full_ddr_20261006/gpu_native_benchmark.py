"""Bounded synthetic benchmark; no cohort reads, scheduler calls or model selection.

CPU output is explicitly CPU evidence. CUDA requires an H100 and a separately
admitted worker: invoking this file never requests or allocates a cluster GPU.
The common-particle comparison checks numerical changes separately from the new
batched sampler's throughput. Cold compile time is not hidden in warm timings.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source/src"))

import numpy as np
import pandas as pd
import torch

from architecture_fitting import complete_count_loss
from credo.contracts import Axis, FiniteMeasure
from credo.recipes.compact_sde_v3.model import CREDOModel
from credo.recipes.compact_sde_v3.objective import CountBlock, checkpoint_geometry, rollout_regularization
from credo.recipes.compact_sde_v3.particles import axis_grid, rollout, sample_initial_particles, sample_noise
from gpu_native import BatchedGeometry, DeviceLawCache, GPUOptions, POLICY, SAMPLER, TensorDynamics, tensor_rollout


def synthetic_data(*, groups, latent_dim, support, seed, catalog=26504):
    """Synthetic D3/D4 labels; NOT any actual donor's observations."""
    rng = np.random.default_rng(seed)
    ids, rows, source, endpoint = [], [], {}, {}
    for i in range(groups):
        donor = "D3" if i < groups//2 else "D4"
        key = donor+"::synthetic-"+str(i)
        ids.append(key)
        target = "NTC" if i in (0, groups//2) else "target-"+str(i)
        rows.append(dict(measure_id=key, sample_id=donor, embedding_id=target, context_group_id=donor))
        n = max(2, support-(i % max(1, support//2)))
        z = rng.normal(size=(n, latent_dim)).astype(np.float32)
        w = rng.uniform(.1, 1., size=n)
        mass = .5+i/100
        source[key] = FiniteMeasure(z, mass*w/w.sum(), mass)
        if i % 7 != 3:
            m = max(2, support-(i % max(1, support//3)))
            z = rng.normal(loc=.15, size=(m, latent_dim)).astype(np.float32)
            w = rng.uniform(.1, 1., size=m)
            endpoint[key] = FiniteMeasure(z, mass*w/w.sum(), mass)
    blocks = []
    for donor in ("D3", "D4"):
        indices = [i for i, row in enumerate(rows) if row["sample_id"] == donor]
        counts = rng.integers(0, 100, size=catalog).astype(np.float32)
        exposure = rng.uniform(.5, 100, size=catalog).astype(np.float32)
        n = len(indices)
        blocks.append(CountBlock(donor, "Stim48hr", indices, exposure[:n], counts[:n],
            background_series_ids=tuple("synthetic-complement-"+str(i) for i in range(catalog-n)),
            background_fitness=np.zeros(catalog-n, dtype=np.float32),
            background_exposure=exposure[n:], background_counts=counts[n:], conditioning_policy="preserve_background"))
    return SimpleNamespace(measure_ids=tuple(ids), latent_dim=latent_dim,
        axis=Axis("physical", ("Stim8hr", "Stim48hr"), (8., 48.), "Stim8hr"),
        measures={"Stim8hr": source, "Stim48hr": endpoint}, measure_meta=pd.DataFrame(rows),
        count_blocks=tuple(blocks), embedding_ids=tuple(dict.fromkeys(row["embedding_id"] for row in rows)),
        control_embedding_ids=("NTC",))


class SyntheticDetachedBank:
    """Only the detached-rest count-loss interface; not catalog-refresh evidence."""
    def __init__(self, groups, device):
        self.fitness = torch.zeros(groups, device=device)

    def fitness_for_active(self, *, time_label, active_indices, active_fitness):
        if time_label != "Stim48hr":
            raise ValueError("Unexpected count time")
        return self.fitness.detach().clone().index_copy(0, active_indices, active_fitness)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(call, *, device, repetitions):
    _synchronize(device)
    start = time.perf_counter()
    first = call()
    _synchronize(device)
    cold = time.perf_counter()-start
    del first
    warm = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = call()
        _synchronize(device)
        warm.append(time.perf_counter()-start)
        del output
    return dict(cold_seconds=cold, warm_seconds=warm, median_warm_seconds=statistics.median(warm))


def run_benchmark(*, device="cpu", groups=32, particles=64, steps=4, support=48,
                  hidden_dim=128, embedding_dim=8, trunk_kind="mlp", trunk_depth=2,
                  repetitions=3, seed=29, options=None):
    options = options or GPUOptions()
    device = torch.device(device)
    if (not 2 <= groups <= 1086 or not 2 <= particles <= 512 or not 1 <= steps <= 8
            or not 2 <= support <= 512 or not 1 <= repetitions <= 5):
        raise ValueError("Synthetic benchmark workload exceeds bound")
    if (not 8 <= hidden_dim <= 512 or not 1 <= embedding_dim <= 32
            or trunk_kind not in {"mlp", "residual_mlp"} or not 1 <= trunk_depth <= 6):
        raise ValueError("Synthetic benchmark architecture exceeds bound")
    if device.type == "cpu" and (groups > 64 or particles > 128 or steps > 4 or support > 128):
        raise ValueError("CPU reference benchmark exceeds its small-fixture bound")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Explicit CPU or CUDA device required")
    if device.type == "cpu" and options.precision != "strict_fp32":
        raise ValueError("Accelerated precision requires CUDA; no fallback")
    hardware = dict(device_type=device.type, required_hardware_identity_matches=False)
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        if "H100" not in prop.name or prop.total_memory < 79_000_000_000:
            raise RuntimeError("This benchmark requires the separately admitted H100-80GB worker")
        hardware.update(name=prop.name, total_memory_bytes=prop.total_memory,
                        required_hardware_identity_matches=True)
    old_threads = torch.get_num_threads()
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1 if device.type == "cpu" else 8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = None
    try:
        data = synthetic_data(groups=groups, latent_dim=48, support=support, seed=seed)
        cache_started = time.perf_counter()
        cache = DeviceLawCache(data, device=device, options=options)
        _synchronize(device)
        cache_seconds = time.perf_counter()-cache_started
        torch.manual_seed(seed)
        original = CREDOModel(embedding_ids=data.embedding_ids, control_embedding_ids=data.control_embedding_ids,
            latent_dim=data.latent_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim, context_mode="none",
            trunk_kind=trunk_kind, trunk_depth=trunk_depth).to(device)
        optimized = copy.deepcopy(original)
        initial = sample_initial_particles(data, data.measure_ids, particles, device=device, seed=seed)
        grid = axis_grid(data.axis, steps, device=device, dtype=torch.float32)
        noise = sample_noise(initial, grid, seed=seed+100)
        lookup = optimized._residual_indices(initial.embedding_ids)
        dynamics = TensorDynamics(optimized, precision=options.precision)
        if options.compile_dynamics:
            torch.compiler.reset()
            dynamics = torch.compile(dynamics, backend=options.compiler_backend, fullgraph=True, dynamic=True)
        geometry = BatchedGeometry(cache)
        bank = SyntheticDetachedBank(groups, device)
        concentration = torch.tensor(1., device=device)

        def reference():
            original.zero_grad(set_to_none=True)
            result = rollout(original, initial, grid, noise=noise)
            values = []
            for row, key in enumerate(data.measure_ids):
                target = data.measures["Stim48hr"].get(key)
                if target is not None:
                    values.append(checkpoint_geometry(result.terminal_z[row],
                        result.log_m0[row]+result.logw_steps[-1, row],
                        torch.tensor(target.support, device=device),
                        torch.tensor(target.weights, dtype=torch.float32, device=device).clamp_min(1e-30).log()))
            loss = torch.stack(values).mean()+.02*complete_count_loss(result, data, bank, concentration)
            loss = loss+rollout_regularization(result)
            loss.backward()
            return loss.detach()

        def fast():
            optimized.zero_grad(set_to_none=True)
            result = tensor_rollout(optimized, dynamics, initial, grid, noise, lookup)
            g, _ = geometry(result)
            loss = g+.02*cache.count_loss(result, bank)+rollout_regularization(result)
            loss.backward()
            return loss.detach()

        # Common states/noise separate numerical agreement from sampler changes.
        reference_timing = timed(reference, device=device, repetitions=repetitions)
        reference_value = float(reference())
        reference_grads = {key: p.grad.detach().cpu().clone() for key, p in original.named_parameters() if p.grad is not None}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = options.precision == "tf32"
        torch.backends.cudnn.allow_tf32 = options.precision == "tf32"
        fast_timing = timed(fast, device=device, repetitions=repetitions)
        fast_value = float(fast())
        max_gradient_error = 0.
        fast_grads = {key: p.grad.detach().cpu() for key, p in optimized.named_parameters() if p.grad is not None}
        gradient_set_agreement = reference_grads.keys() == fast_grads.keys()
        finite_gradients = all(bool(g.isfinite().all()) for g in (*reference_grads.values(), *fast_grads.values()))
        gradient_agreement = gradient_set_agreement and finite_gradients
        for name in reference_grads.keys() & fast_grads.keys():
            expected, actual = reference_grads[name], fast_grads[name]
            if finite_gradients:
                max_gradient_error = max(max_gradient_error, float((actual-expected).abs().max()))
            gradient_agreement &= torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
        finite_objectives = bool(np.isfinite(reference_value) and np.isfinite(fast_value))
        objective_error = abs(reference_value-fast_value) if finite_objectives else None
        sampling_reference = timed(lambda: sample_initial_particles(data, data.measure_ids, particles,
            device=device, seed=seed), device=device, repetitions=repetitions)
        sampling_fast = timed(lambda: cache.sample(data.measure_ids, particles, seed=seed),
                              device=device, repetitions=repetitions)
        source_files = ("gpu_native.py", "gpu_native_fitting.py", "gpu_native_benchmark.py")
        return dict(schema="credo.gpu_native.synthetic_component_benchmark.v1", utc=datetime.now(timezone.utc).isoformat(),
            status="synthetic_benchmark_complete", execution_policy=POLICY, source_sampler=SAMPLER,
            scope="synthetic_primitives_not_full_cohort_training_or_memory_admission", real_data_reads=0,
            scientific_optimizer_updates=0, optimizer_steps=0, sampler_bitwise_scalar_replay=False,
            full_cohort_training_qualified=False,
            environment=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                             cuda_runtime=torch.version.cuda, numpy=np.__version__),
            options=asdict(options), hardware=hardware, groups=groups, particles=particles, steps=steps,
            support=support, seed=seed,
            architecture=dict(hidden_dim=hidden_dim, embedding_dim=embedding_dim,
                              trunk_kind=trunk_kind, trunk_depth=trunk_depth),
            count_catalogs=[dict(donor=b.context_group_id, categories=len(b.measure_indices)+len(b.background_series_ids),
                observed_zero_categories=int((b.counts == 0).sum()+(b.background_counts == 0).sum()))
                for b in data.count_blocks],
            cache_bytes=cache.allocated_bytes, cache_construction_seconds=cache_seconds,
            reference=reference_timing, optimized=fast_timing, sampling_reference=sampling_reference,
            sampling_optimized=sampling_fast,
            objective_and_backward_speed_ratio=reference_timing["median_warm_seconds"]/fast_timing["median_warm_seconds"],
            sampler_speed_ratio=sampling_reference["median_warm_seconds"]/sampling_fast["median_warm_seconds"],
            reference_objective=reference_value if np.isfinite(reference_value) else None,
            optimized_objective=fast_value if np.isfinite(fast_value) else None,
            absolute_objective_error=objective_error,
            maximum_absolute_gradient_error=max_gradient_error if finite_gradients and gradient_set_agreement else None,
            finite_objectives=finite_objectives, finite_gradients=finite_gradients,
            gradient_parameter_set_agreement=gradient_set_agreement,
            numerical_agreement=(finite_objectives and objective_error <= 1e-5+1e-5*abs(reference_value)) and gradient_agreement,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            geometry_batches=geometry.last_batches,
            sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files})
    finally:
        if cache is not None:
            cache.close()
        torch.set_num_threads(old_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--support", type=int, default=48)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--compile-dynamics", action="store_true")
    parser.add_argument("--precision", choices=("strict_fp32", "tf32", "bf16_dynamics"), default="strict_fp32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir() or args.output.is_symlink()):
        raise FileExistsError("New absent output under an existing approved directory required")
    result = run_benchmark(device=args.device, groups=args.groups, particles=args.particles, steps=args.steps,
        support=args.support, repetitions=args.repetitions,
        options=GPUOptions(compile_dynamics=args.compile_dynamics, precision=args.precision))
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if args.output is not None:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    print(raw.decode(), end="")
    return 0 if result["numerical_agreement"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
