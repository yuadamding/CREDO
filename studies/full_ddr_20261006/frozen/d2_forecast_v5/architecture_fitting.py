"""Bounded, fitting-only full-DDR execution with shared training scales.

The minibatch count value retains the complete catalog. Its active-coordinate
gradient is scaled for uniform guide sampling; the detached remainder is still
an approximation, not an exact current-parameter full-catalog gradient.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import time
from unittest.mock import patch

import numpy as np
import torch

from credo.recipes.compact_sde_v3 import objective as objective_module
from credo.recipes.compact_sde_v3.objective import (
    _count_block_nll, checkpoint_geometry, integrated_fitness_curve, total_objective,
)
from terminal_fitting import TerminalFittingRecipe, TerminalFittingTrainer

POLICY = "balanced_fitting_scales_active_coordinate_count_v1"


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def count_sampling_surrogate(value, *, population: int, batch: int):
    """Preserve a full-catalog score's value while scaling its active gradient."""
    if not 1 <= batch <= population:
        raise ValueError("Invalid active-coordinate sampling population")
    return value + (population / batch - 1.0) * (value - value.detach())


def complete_count_loss(result, data, bank, concentration):
    """All fitting donor blocks, including a donor with no active batch guides."""
    curve = integrated_fitness_curve(result, mass_increment="particle_log_mass")
    losses = []
    for block in data.count_blocks:
        if block.time_label != "Stim48hr":
            raise ValueError("Only endpoint counts belong to fitting")
        full = bank.fitness_for_active(time_label=block.time_label,
            active_indices=result.measure_indices, active_fitness=curve[-1])
        indices = block.measure_indices.to(device=full.device, dtype=torch.long)
        losses.append(_count_block_nll(block, full.index_select(0, indices), concentration,
                                      count_likelihood="categorical_ce"))
    if len(losses) != 2:
        raise ValueError("Two complete D3/D4 count blocks required")
    return torch.stack(losses).mean()


@torch.no_grad()
def fitting_normalizers(problem, *, particles=512, seed=104729, catalog_size=26504,
                        device="cuda", progress=lambda _: None):
    """One source-persistence pass; no model, holdout scores or optimizer calls."""
    data = problem.training
    if set(data.measure_meta.sample_id.astype(str)) != {"D3", "D4"}:
        raise ValueError("Only fitting donors may determine training scales")
    device = torch.device(device)
    endpoint = data.measures["Stim48hr"]
    ids = tuple(key for key in data.measure_ids if key in endpoint)
    if not ids:
        raise ValueError("No fitting state pairs")
    total = 0.0
    backend = set()
    for index, key in enumerate(ids):
        # Reuse exactly the source sampler's multinomial law/RNG without
        # reconstructing its full metadata index once for EACH guide (O(G²)).
        source = data.measures[data.axis.source][key]
        source_z = torch.as_tensor(np.array(source.support, copy=True), device=device, dtype=torch.float32)
        source_w = torch.as_tensor(source.normalized_weights, device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(seed + index)
        indices = torch.multinomial(source_w, particles, replacement=True, generator=generator)
        predicted = source_z[indices]
        predicted_logw = torch.full((particles,), -float(np.log(particles)), device=device, dtype=torch.float32)
        target = endpoint[key]
        z = torch.as_tensor(np.array(target.support, copy=True), device=device, dtype=torch.float32)
        w = torch.as_tensor(np.array(target.weights, copy=True), device=device, dtype=torch.float32)
        score = checkpoint_geometry(predicted, predicted_logw, z, w.log(),
                                    epsilon=0.1, backend_observer=backend.add)
        value = float(score)
        if not math.isfinite(value) or value < -1e-5:
            raise RuntimeError("Invalid fitting persistence geometry")
        total += max(0.0, value)
        if (index + 1) % 1000 == 0 or index + 1 == len(ids):
            progress(dict(event="normalizer_state_progress", laws_completed=index + 1,
                          laws_total=len(ids), optimizer_updates=0))
        del source_z, source_w, indices, predicted, predicted_logw, z, w, score
    count_rows = []
    for block in data.count_blocks:
        if block.time_label != "Stim48hr":
            raise ValueError("Unexpected count role in fitting scales")
        exposure = np.concatenate((block.exposure.numpy(), block.background_exposure.numpy())).astype(np.float64)
        counts = np.concatenate((block.counts.numpy(), block.background_counts.numpy())).astype(np.float64)
        if len(counts) != catalog_size or not np.all(exposure > 0) or counts.sum() <= 0:
            raise ValueError("Complete captured denominator required")
        truth = counts / counts.sum()
        source = exposure / exposure.sum()
        positive = truth > 0
        entropy = float(-(truth[positive] * np.log(truth[positive])).sum())
        ce = float(-(truth[positive] * np.log(source[positive])).sum())
        count_rows.append(dict(donor=str(block.context_group_id), ce=ce, entropy=entropy, kl=ce-entropy))
    if len(count_rows) != 2 or backend != {"geomloss_tensorized"}:
        raise RuntimeError("Incomplete fitting normalizer or changed geometry backend")
    result = dict(policy=POLICY, state=float(total / len(ids)), count=float(np.mean([r["kl"] for r in count_rows])),
                  state_laws=len(ids), particles=particles, seed=seed,
                  geometry_backend=next(iter(backend)), count_blocks=count_rows,
                  fitting_donors=["D3", "D4"], model_forward_calls=0, optimizer_updates=0)
    validate_normalizers(result)
    return result


def validate_normalizers(normalizers):
    if normalizers.get("policy") != POLICY or normalizers.get("fitting_donors") != ["D3", "D4"]:
        raise ValueError("Fitting normalizer identity differs")
    if any(not math.isfinite(normalizers[k]) or normalizers[k] <= 0 for k in ("state", "count")):
        raise ValueError("Positive finite fitting normalizers required; no epsilon fallback")


class ArchitectureFittingTrainer(TerminalFittingTrainer):
    """Same staged recipe, with explicit balanced scales and update accounting."""

    def _fit(self):
        self.optimizer_updates = 0
        self.attempted_optimizer_updates = 0
        self.training_guide_exposures = 0
        self.optimizer_memory_admissions = []
        self.head_gradient_maxima = {"drift": 0.0, "diffusion": 0.0, "reaction": 0.0}
        self._progress = getattr(self, "_progress", lambda _: None)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        super()._fit()
        self.execution_trace.append(dict(event=POLICY, optimizer_updates=self.optimizer_updates,
            attempted_optimizer_updates=self.attempted_optimizer_updates,
            training_guide_exposures=self.training_guide_exposures,
            optimizer_memory_admissions=self.optimizer_memory_admissions))

    def _optimizer(self, spec, parameters):
        if spec.kind != "adam" or spec.weight_decay != 0 or spec.parameter_learning_rates or spec.parameter_weight_decays:
            raise ValueError("Only the frozen plain Adam optimizer is admitted")
        return torch.optim.Adam(parameters, lr=spec.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0.0, amsgrad=False, foreach=False, fused=False, capturable=False)

    def _memory_guard(self, *, optimizer, first_step, after_step=False):
        if self.device.type != "cuda":
            return {}
        torch.cuda.synchronize(self.device)
        policy = self.objective_map["checkpoint_geometry"].config["execution"]
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        parameter_bytes = sum(p.numel() * p.element_size() for group in optimizer.param_groups for p in group["params"])
        extra = 2 * parameter_bytes if first_step and not after_step else 0
        if (allocated + extra > policy["maximum_allocated_bytes"]
                or reserved + extra > policy["maximum_reserved_bytes"]
                or free - extra < policy["minimum_live_headroom_bytes"]):
            raise RuntimeError("Optimizer-inclusive H100 memory admission failed; no retry")
        return dict(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                    free_bytes=free, total_bytes=total, optimizer_parameter_bytes=parameter_bytes,
                    first_step_adam_additional_bytes_estimate=extra)

    def _train_epoch(self, stage, optimizer, phase_epoch):
        del phase_epoch
        self.model.train()
        training_ids = self._training_ids_for_stage(stage)
        seed = self.training_plan.seed + self.completed_epochs * 10_000
        geometry_sum = count_sum = regularization_sum = 0.0
        observations = batches = 0
        execution = self.objective_map["checkpoint_geometry"].config["execution"]
        for batch_index, batch_ids in enumerate(self._batches(training_ids, seed=seed,
                batch_size=stage.batching.measures_per_batch, order=stage.batching.order)):
            self.attempted_optimizer_updates += 1
            if self.attempted_optimizer_updates > execution["maximum_optimizer_updates_per_fit"]:
                raise RuntimeError("Fitting update cap exceeded")
            self._progress(dict(event="scientific_update_started", stage=stage.name, epoch=self.completed_epochs,
                batch=batch_index, attempted_optimizer_updates=self.attempted_optimizer_updates,
                optimizer_updates=self.optimizer_updates, guides=len(batch_ids)))
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            self.bank.tick()
            result = self._rollout_ids(batch_ids, particles=self.training_plan.particles,
                seed=seed + batch_index, provider=self._provider(stage), grid=self.training_grid)
            backend = set()
            original = objective_module.checkpoint_geometry

            def geometry(*args, **kwargs):
                return original(*args, **kwargs, backend_observer=backend.add)

            action = self._objective(stage, "rollout_action")
            action_config = {} if action is None else dict(action.config)
            with patch.object(objective_module, "checkpoint_geometry", geometry):
                objective = total_objective(result, self.data,
                    geometry_weight=self._objective_weight(stage, "checkpoint_geometry"),
                    mass_weight=0.0, count_weight=0.0, include_mass=False,
                    log_concentration=self.log_count_concentration, time_labels=self.train_time_labels,
                    action_weights=tuple(float(action_config.get(key, 0.0)) for key in
                        ("drift_weight", "diffusion_weight", "growth_weight")),
                    collect_rows=False, count_likelihood="categorical_ce", mass_increment="particle_log_mass")
            if ((not backend and objective.checkpoint.observation_count != 0)
                    or (not backend and stage.name != "mass")
                    or (backend and self.device.type == "cuda" and backend != {"geomloss_tensorized"})):
                raise RuntimeError("Geometry backend missing or unqualified")
            count = result.z_steps.new_zeros(())
            if stage.name == "mass":
                count = complete_count_loss(result, self.data, self.bank, self.log_count_concentration)
            scaled_count = count_sampling_surrogate(count, population=len(training_ids), batch=len(batch_ids))
            regularizer = self.model.regularization(coefficient=self._objective_weight(stage, "model_regularization"))
            loss = objective.total + self._objective_weight(stage, "grouped_count_likelihood") * scaled_count + regularizer
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite fitting loss")
            loss.backward()
            heads = {}
            for head, prefix in (("drift", "drift_"), ("diffusion", "sigma_"), ("reaction", "growth_")):
                values = [p.grad.detach().abs().max() for name, p in self.model.named_parameters()
                          if name.startswith(prefix) and p.grad is not None]
                heads[head] = float(torch.stack(values).max()) if values else 0.0
                self.head_gradient_maxima[head] = max(self.head_gradient_maxima[head], heads[head])
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.training_plan.gradient_clip_norm,
                                                 error_if_nonfinite=True)
            first_step = not optimizer.state
            self._memory_guard(optimizer=optimizer, first_step=first_step)
            optimizer.step()
            self.optimizer_updates += 1
            self.training_guide_exposures += len(batch_ids)
            # Persist the applied-step count before any post-step guard can
            # fail; a failed fit must not claim its parameters were unchanged.
            self._progress(dict(event="optimizer_step_applied", stage=stage.name,
                optimizer_updates=self.optimizer_updates,
                attempted_optimizer_updates=self.attempted_optimizer_updates))
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise RuntimeError("Optimizer produced nonfinite parameters")
            memory = self._memory_guard(optimizer=optimizer, first_step=first_step, after_step=True)
            if first_step:
                self.optimizer_memory_admissions.append(dict(stage=stage.name, optimizer_updates=self.optimizer_updates,
                    optimizer_state_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values()
                                              for v in state.values() if torch.is_tensor(v)), **memory))
            if stage.name == "mass":
                self.bank.update_from_rollout(result, self.model, self.data, full_refresh=False)
            n = objective.checkpoint.observation_count
            g, c, r = float(objective.checkpoint.geometry.detach()), float(count.detach()), float((objective.regularization + regularizer).detach())
            geometry_sum += g * n
            count_sum += c
            regularization_sum += r
            observations += n
            batches += 1
            self._progress(dict(event="scientific_update_completed", stage=stage.name, epoch=self.completed_epochs,
                batch=batch_index, optimizer_updates=self.optimizer_updates, guides=len(batch_ids), geometry=g,
                count_ce=c, objective=float(loss.detach()), preclip_gradient_norm=float(norm),
                head_gradient_maxima=heads,
                geometry_backend=next(iter(backend)) if backend else "not_evaluated_no_state_observations",
                elapsed_seconds=time.perf_counter()-start, **memory))
            del result, objective, loss, count, scaled_count, regularizer, norm
        g = geometry_sum / max(1, observations)
        c = count_sum / max(1, batches)
        r = regularization_sum / max(1, batches)
        return dict(train_objective=self._objective_weight(stage, "checkpoint_geometry")*g
                    +self._objective_weight(stage, "grouped_count_likelihood")*c+r,
                    train_geometry=g, train_log_mass_error=0.0, train_count_loss=c,
                    train_observations=observations, train_batches=batches)


class ArchitectureFittingRecipe(TerminalFittingRecipe):
    def __init__(self, normalizers, execution, *, progress=lambda _: None):
        validate_normalizers(normalizers)
        self.normalizers, self.execution, self.progress = normalizers, execution, progress

    def build_objectives(self, study, config):
        output = []
        for term in super().build_objectives(study, config):
            if term.name == "checkpoint_geometry":
                term = replace(term, weight=0.5/self.normalizers["state"], config=dict(term.config,
                    policy=POLICY, normalizers_sha256=hashlib.sha256(canonical(self.normalizers)).hexdigest(),
                    normalizers=self.normalizers, execution=self.execution))
            elif term.name == "grouped_count_likelihood":
                term = replace(term, weight=0.5/self.normalizers["count"], config=dict(term.config,
                    active_coordinate_sampling_scale="joint_population/actual_batch", all_fitting_blocks=True))
            output.append(term)
        return tuple(output)

    def execute_training(self, study, *, model, plan, objectives, run_config, **kwargs):
        # The callback is execution plumbing, not model/config selection. Bind it
        # on a temporary subclass so fitting has it before from_plan calls _fit.
        callback = self.progress

        class ThisFit(ArchitectureFittingTrainer):
            _progress = staticmethod(callback)

        return ThisFit.from_plan(study, model, run_config, plan, objectives, **kwargs)
