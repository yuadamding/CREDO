"""Real main-CREDO forward/backward capacity probe; never an optimizer fit."""
from __future__ import annotations

import gc
import math
import time
from unittest.mock import patch
import numpy as np
import torch

from credo.recipes.compact_sde_v3.particles import (
    NoContextProvider, axis_grid, rollout, sample_initial_particles, sample_noise,
)
from credo.recipes.compact_sde_v3.objective import total_objective
from credo.recipes.compact_sde_v3 import objective as objective_module
from credo.recipes.compact_sde_v3.training import CatalogBank
from credo.recipes.compact_sde_v3.recipe import recipe


def qualification_config(model):
    from credo.recipes.compact_sde_v3.recipe import CompactV3Config
    return CompactV3Config.model_validate({
        "model": dict(model, context="none"),
        "training": {"epochs": {"state": 1, "mass": 1, "context": 0},
                     "particles": 512, "steps_per_interval": 8,
                     "measures_per_batch": 16, "checkpoint_selection": "last",
                     "catalog_bank_momentum": 0.0, "catalog_refresh_particles": 32},
        "loss": {"mass": 0.0, "count": 1.0, "count_likelihood": "categorical_ce",
                 "mass_increment": "particle_log_mass"},
        "validation": {"fraction": 0.0, "representation_scope": "shared"},
    })


def probe(problem, config, measure_ids, *, device="cuda", seed=0):
    """One actual full-DDR objective + backward, all weights unchanged."""
    data = problem.training
    torch.manual_seed(seed)
    model = recipe.build_model(problem, config).to(device=device, dtype=torch.float32)
    model.set_trainable_tags(("reference_embedding", "residual_embeddings", "shared_dynamics",
                              "drift", "diffusion", "growth"), growth_enabled=True, context_enabled=False)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    grid = axis_grid(data.axis, config.training.steps_per_interval, device=device, dtype=torch.float32)
    bank = CatalogBank.empty(data, model, len(grid) - 1, device=torch.device(device),
                            dtype=torch.float32, momentum=0.0, mass_increment="particle_log_mass")
    # Populate a REAL detached all-guide bank with the same finite-step mass
    # calculation used in training. With ecology disabled, complete-group
    # rollout is unnecessary: independent guide batches give identical laws.
    with torch.no_grad():
        model.eval()
        ids = tuple(data.measure_ids)
        for start_index in range(0, len(ids), 256):
            initial = sample_initial_particles(data, ids[start_index:start_index + 256],
                config.training.catalog_refresh_particles, device=torch.device(device),
                dtype=torch.float32, seed=seed + start_index)
            refreshed = rollout(model, initial, grid, context_provider=NoContextProvider(),
                noise=sample_noise(initial, grid, seed=seed + start_index + 2_000_003))
            bank.update_from_rollout(refreshed, model, data, full_refresh=True)
        bank.assert_complete()
        del initial, refreshed
        model.train()
    capacity = {}
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        capacity = dict(device_total_bytes=total, free_bytes_before_objective=free,
            baseline_reserved_bytes=torch.cuda.memory_reserved(),
            reserved_admission_ceiling_bytes=min(78_000_000_000,
                free + torch.cuda.memory_reserved() - 4_000_000_000))
    start = time.perf_counter()
    state = sample_initial_particles(data, measure_ids, config.training.particles,
                                     device=torch.device(device), dtype=torch.float32, seed=seed)
    result = rollout(model, state, grid, context_provider=NoContextProvider(),
                     noise=sample_noise(state, grid, seed=seed + 1_000_003))
    backends = set()
    original_geometry = objective_module.checkpoint_geometry
    def observed_geometry(*args, **kwargs):
        return original_geometry(*args, **kwargs, backend_observer=backends.add)
    with patch.object(objective_module, "checkpoint_geometry", observed_geometry):
        objective = total_objective(result, data, mass_weight=0.0, count_weight=config.loss.count,
            include_mass=True, log_concentration=torch.tensor(np.log(100.0), device=device),
            fitness_bank=bank, time_labels=data.axis.labels[1:], collect_rows=False,
            count_likelihood="categorical_ce", mass_increment="particle_log_mass")
    if len(backends) != 1:
        raise RuntimeError("Actual geometry backend missing or mixed")
    loss = objective.total + model.regularization(coefficient=config.loss.parameter_regularization)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite full-DDR qualification objective")
    loss.backward()
    gradients = {name: float(parameter.grad.detach().abs().max())
                 for name, parameter in model.named_parameters() if parameter.grad is not None}
    if not gradients or any(not np.isfinite(value) for value in gradients.values()):
        raise RuntimeError("Invalid full-DDR qualification gradients")
    if any(not torch.equal(before[name], tensor) for name, tensor in model.state_dict().items()):
        raise RuntimeError("Qualification unexpectedly changed model parameters or buffers")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    record = dict(groups=len(measure_ids), particles=config.training.particles,
        geometry_backend=next(iter(backends)), **capacity,
        steps=config.training.steps_per_interval, observed_cells=sum(len(data.measures[data.axis.labels[-1]][key].support)
             for key in measure_ids), parameter_updates=0,
        geometry=float(objective.checkpoint.geometry.detach()), count_ce=float(objective.count.detach()),
        loss=float(loss.detach()), gradients=gradients, elapsed_seconds=time.perf_counter() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(device).startswith("cuda") else None)
    del objective, result, state, loss, model, before, bank
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return record


def next_batch(records, *, available, initial_groups=16, target_bytes=70_000_000_000,
               cap_bytes=75_000_000_000, reserved_ceiling_bytes=78_000_000_000):
    """Quadratic-or-steeper guarded planning heuristic, never an OOM retry.

    Actual allocator peaks still require measurement. The old linear per-group
    extrapolation is deliberately not used; a 10% prediction cushion and live
    free-memory headroom constrain every increase.
    """
    if available < 1 or initial_groups < 1 or min(target_bytes, cap_bytes, reserved_ceiling_bytes) <= 0:
        raise ValueError("Invalid capacity admission")
    if not records:
        return min(initial_groups, available)
    for row in records:
        if not 0 < row["peak_allocated_bytes"] <= row["peak_reserved_bytes"] or row["groups"] < 1:
            raise ValueError("Invalid measured memory record")
    last = records[-1]
    if last["peak_allocated_bytes"] >= 65_000_000_000 or last["groups"] >= available:
        return None
    exponent = 2.0
    for earlier, later in zip(records, records[1:]):
        if later["groups"] <= earlier["groups"]:
            raise ValueError("Capacity ladder must increase")
        exponent = max(exponent, math.log(later["peak_reserved_bytes"] / earlier["peak_reserved_bytes"])
                       / math.log(later["groups"] / earlier["groups"]))
    ceiling = min(reserved_ceiling_bytes, last.get("reserved_admission_ceiling_bytes", reserved_ceiling_bytes))
    permitted = min(target_bytes, cap_bytes, ceiling / 1.10)
    admitted = int(last["groups"] * (permitted / last["peak_reserved_bytes"]) ** (1 / exponent))
    proposal = min(available, last["groups"] * 2, admitted)
    return proposal if proposal > last["groups"] else None


def qualify(problem, model, *, max_probes=10, initial_groups=16, progress=lambda _: None):
    config = qualification_config(model)
    data = problem.training
    downstream = data.measures[data.axis.labels[-1]]
    order = tuple(sorted((key for key in data.measure_ids if key in downstream),
                         key=lambda key: (-len(downstream[key].support), key)))
    if len(order) < 16:
        raise ValueError("Insufficient fitting support for the prescribed capacity ladder")
    records = []
    if not 1 <= max_probes <= 10 or not 16 <= initial_groups <= 4096:
        raise ValueError("Bounded capacity workload required")
    for _ in range(max_probes):
        free, _ = torch.cuda.mem_get_info()
        live_ceiling = min(78_000_000_000, free + torch.cuda.memory_reserved() - 4_000_000_000)
        if live_ceiling <= 0:
            raise RuntimeError("Insufficient live GPU memory headroom")
        groups = next_batch(records, available=min(len(order), 4096), initial_groups=initial_groups,
                            reserved_ceiling_bytes=live_ceiling)
        if groups is None:
            break
        # An attempted call consumes budget even if it fails before a result.
        progress(dict(event="probe_started", probe_index=len(records) + 1,
                      groups=groups, particles=512, steps=8, parameter_updates=0))
        row = probe(problem, config, order[:groups])
        if (row["peak_reserved_bytes"] > min(78_000_000_000, row["reserved_admission_ceiling_bytes"])
                or row["peak_allocated_bytes"] > 75_000_000_000):
            raise RuntimeError("Qualification exceeded reserved-memory safety ceiling")
        records.append(row)
        progress(dict(row, event="probe_completed", probe_index=len(records)))
    if not records:
        raise RuntimeError("No qualified full-DDR batch")
    return dict(probes=records, measured_groups=records[-1]["groups"], available_groups=min(len(order), 4096),
                target_vram_reached=65_000_000_000 <= records[-1]["peak_allocated_bytes"] <= 75_000_000_000,
                productive_memory_only=True, scientific_optimizer_updates=0,
                geometry_backend=records[-1]["geometry_backend"],
                probe_attempts=len(records), initial_groups=initial_groups,
                stopping_reason=("target_band" if records[-1]["peak_allocated_bytes"] >= 65_000_000_000
                    else "probe_budget" if len(records) == max_probes else "support_or_memory_admission"))
