"""Separately versioned GPU-first trainer, not a hot patch or launch command.

This retains staged exposure, fresh stage optimizers, balanced frozen scales,
complete-catalog count normalization, approximate detached-bank semantics,
terminal publication and no holdout scoring. Source sampling has an explicitly
new batched RNG policy; mixed-source model reuse is NOT authorized here.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import time

import torch

from architecture_fitting import ArchitectureFittingRecipe, ArchitectureFittingTrainer, count_sampling_surrogate
from credo.recipes.compact_sde_v3.objective import rollout_regularization
from credo.recipes.compact_sde_v3.particles import NoContextProvider, sample_noise
from gpu_native import BatchedGeometry, DeviceLawCache, GPUOptions, POLICY, SAMPLER, TensorDynamics, tensor_rollout
from terminal_fitting import REFRESH_GUIDES, _validate_architecture_fit


class StepTiming:
    """CUDA events at a bounded cadence; no per-component GPU synchronization."""

    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.events = []

    def mark(self, label):
        if not self.enabled:
            return
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()
        self.events.append((label, event))

    def summary(self):
        if not self.enabled:
            return {}
        if self.device.type == "cuda":
            # The post-step diagnostic readback already drained the same stream.
            values = {name: left.elapsed_time(right) for (_, left), (name, right)
                      in zip(self.events, self.events[1:], strict=False)}
        else:
            values = {name: 1000 * (right-left) for (_, left), (name, right)
                      in zip(self.events, self.events[1:], strict=False)}
        return dict(component_milliseconds=values,
                    timing_backend="cuda_events" if self.device.type == "cuda" else "cpu_reference_clock")


class GPUFittingTrainer(ArchitectureFittingTrainer):
    _gpu_options = GPUOptions()
    _prepared_cache = None

    def _fit(self):
        _validate_architecture_fit(self.problem, self.config)
        options = self._gpu_options
        if self.device.type != "cuda" and (options.precision != "strict_fp32" or options.optimizer_backend == "fused"):
            raise ValueError("Accelerated precision/fused Adam require explicit CUDA qualification")
        self._owns_cache = self._prepared_cache is None
        self._law_cache = None
        old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            self._law_cache = self._prepared_cache or DeviceLawCache(self.data, device=self.device, options=options)
            self._law_cache.require(self.data)
            if self._law_cache.device != self.device or self._law_cache.options != options:
                raise ValueError("Prepared device cache and execution policy differ")
            self._geometry = BatchedGeometry(self._law_cache)
            self._residual_indices = torch.tensor(
                [self.model._residual_index.get(key, -1)+1 for key in self._law_cache.embedding_ids],
                device=self.device, dtype=torch.long)
            dynamics = TensorDynamics(self.model, precision=options.precision)
            if options.compile_dynamics:
                # One sequential model owns each process-local compiler generation.
                # Do not accumulate identity/stage guards across all nine models,
                # increase the compiler's limit, or fall back silently to eager.
                torch.compiler.reset()
                dynamics = torch.compile(dynamics, backend=options.compiler_backend, fullgraph=True, dynamic=True)
            self._tensor_dynamics = dynamics
            torch.backends.cuda.matmul.allow_tf32 = options.precision == "tf32"
            torch.backends.cudnn.allow_tf32 = options.precision == "tf32"
            super()._fit()
            self.execution_trace.append(dict(event=POLICY, execution_options=asdict(options), sampler=SAMPLER,
                sampler_bitwise_scalar_replay=False, cache_bytes=self._law_cache.allocated_bytes,
                state_law_count=len(self._law_cache.ids), optimizer_updates=self.optimizer_updates,
                assessment_performed=False, full_support_geometry=True,
                per_guide_sinkhorn_schedule=True, gpu_qualification_claimed=False))
        except BaseException:
            if self._owns_cache and self._law_cache is not None:
                self._law_cache.close()
            raise
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn

    def _rollout_ids(self, measure_ids, *, particles, seed, provider, data=None, grid=None):
        selected_data = self.data if data is None else data
        self._law_cache.require(selected_data)
        if not isinstance(provider, NoContextProvider):
            raise ValueError("GPU fitting cannot activate ecological context")
        if grid is not None and grid is not self.training_grid:
            raise ValueError("Only the bound fitting grid may be integrated")
        state = self._law_cache.sample(measure_ids, particles, seed=seed)
        noise = sample_noise(state, self.training_grid, seed=seed+1_000_003)
        return tensor_rollout(self.model, self._tensor_dynamics, state, self.training_grid, noise,
                              self._residual_indices[state.measure_indices])

    def _optimizer(self, spec, parameters):
        if spec.kind != "adam" or spec.weight_decay != 0 or spec.parameter_learning_rates or spec.parameter_weight_decays:
            raise ValueError("Only the bound plain Adam estimator is admitted")
        backend = self._gpu_options.optimizer_backend
        return torch.optim.Adam(parameters, lr=spec.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0., amsgrad=False, foreach=backend == "foreach",
                                fused=backend == "fused", capturable=False)

    def _memory_guard(self, *, optimizer, first_step, after_step=False):
        if self.device.type != "cuda":
            return {}
        # The existing pre-/post-step finite diagnostic readback already orders
        # the stream. Do not add a redundant cuda.synchronize() at both guards.
        policy = self.objective_map["checkpoint_geometry"].config["execution"]
        free, total = torch.cuda.mem_get_info(self.device)
        allocated, reserved = torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)
        parameter_bytes = sum(p.numel()*p.element_size() for group in optimizer.param_groups for p in group["params"])
        extra = 2*parameter_bytes if first_step and not after_step else 0
        if (allocated+extra > policy["maximum_allocated_bytes"]
                or reserved+extra > policy["maximum_reserved_bytes"]
                or free-extra < policy["minimum_live_headroom_bytes"]):
            raise RuntimeError("Optimizer-inclusive H100 memory admission failed; no retry")
        return dict(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, free_bytes=free,
                    total_bytes=total, optimizer_parameter_bytes=parameter_bytes,
                    first_step_adam_additional_bytes_estimate=extra)

    def _train_epoch(self, stage, optimizer, phase_epoch):
        del phase_epoch
        self.model.train()
        ids = self._training_ids_for_stage(stage)
        seed = self.training_plan.seed+self.completed_epochs*10_000
        geometry_sum = count_sum = regularization_sum = 0.
        observations = batches = 0
        execution = self.objective_map["checkpoint_geometry"].config["execution"]
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        head_parameters = {head: tuple(p for name, p in self.model.named_parameters() if name.startswith(prefix))
                           for head, prefix in (("drift", "drift_"), ("diffusion", "sigma_"), ("reaction", "growth_"))}
        for batch_index, batch_ids in enumerate(self._batches(ids, seed=seed,
                batch_size=stage.batching.measures_per_batch, order=stage.batching.order)):
            self.attempted_optimizer_updates += 1
            if self.attempted_optimizer_updates > execution["maximum_optimizer_updates_per_fit"]:
                raise RuntimeError("Fitting update cap exceeded")
            self._progress(dict(event="scientific_update_started", stage=stage.name, epoch=self.completed_epochs,
                batch=batch_index, attempted_optimizer_updates=self.attempted_optimizer_updates,
                optimizer_updates=self.optimizer_updates, guides=len(batch_ids), execution_policy=POLICY))
            started = time.perf_counter()
            timing = StepTiming(self.device, self.optimizer_updates % self._gpu_options.timing_interval == 0)
            timing.mark("start")
            optimizer.zero_grad(set_to_none=True)
            self.bank.tick()
            result = self._rollout_ids(batch_ids, particles=self.training_plan.particles,
                seed=seed+batch_index, provider=self._provider(stage))
            timing.mark("sampling_and_rollout")
            geometry, n = self._geometry(result)
            timing.mark("geometry")
            count = self._law_cache.count_loss(result, self.bank) if stage.name == "mass" else geometry.new_zeros(())
            scaled_count = count_sampling_surrogate(count, population=len(ids), batch=len(batch_ids))
            action = self._objective(stage, "rollout_action")
            action_config = {} if action is None else dict(action.config)
            regularizer = rollout_regularization(result, **{key: float(action_config.get(key, 0.))
                for key in ("drift_weight", "diffusion_weight", "growth_weight")})
            regularizer = regularizer+self.model.regularization(coefficient=self._objective_weight(stage, "model_regularization"))
            loss = self._objective_weight(stage, "checkpoint_geometry")*geometry
            loss = loss+self._objective_weight(stage, "grouped_count_likelihood")*scaled_count+regularizer
            timing.mark("count_and_regularization")
            loss.backward()
            timing.mark("backward")
            heads = []
            for values in head_parameters.values():
                gradients = [p.grad.detach().abs().max() for p in values if p.grad is not None]
                heads.append(torch.stack(gradients).max() if gradients else loss.new_zeros(()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.training_plan.gradient_clip_norm,
                error_if_nonfinite=False, foreach=self._gpu_options.optimizer_backend != "single_tensor")
            # One scalar guard replaces separate host reads of loss and norm.
            if not bool(torch.stack((loss.detach(), norm)).isfinite().all()):
                raise RuntimeError("Nonfinite fitting loss or gradient; optimizer not applied")
            first_step = not optimizer.state
            self._memory_guard(optimizer=optimizer, first_step=first_step)
            timing.mark("gradient_checks_and_clip")
            optimizer.step()
            self.optimizer_updates += 1
            self.training_guide_exposures += len(batch_ids)
            # Applied-step evidence remains durable even if a post-step guard fails.
            self._progress(dict(event="optimizer_step_applied", stage=stage.name,
                optimizer_updates=self.optimizer_updates, attempted_optimizer_updates=self.attempted_optimizer_updates))
            timing.mark("optimizer")
            finite_parameters = torch.stack([p.detach().isfinite().all() for p in parameters]).all()
            if stage.name == "mass":
                self.bank.update_from_rollout(result, self.model, self.data, full_refresh=False)
            timing.mark("bank_and_parameter_check")
            # One packed diagnostic transfer, including the post-Adam finite guard,
            # drains all work above. No per-parameter/three-head scalar round trips.
            values = torch.stack((geometry.detach(), count.detach(), regularizer.detach(), loss.detach(), norm,
                                  *heads, finite_parameters.to(loss))).detach().cpu().tolist()
            if values[-1] != 1.:
                raise RuntimeError("Optimizer produced nonfinite parameters")
            g, c, r, objective, gradient_norm = values[:5]
            head_values = dict(zip(head_parameters, values[5:8], strict=True))
            for key, value in head_values.items():
                self.head_gradient_maxima[key] = max(self.head_gradient_maxima[key], value)
            memory = self._memory_guard(optimizer=optimizer, first_step=first_step, after_step=True)
            if first_step:
                self.optimizer_memory_admissions.append(dict(stage=stage.name, optimizer_updates=self.optimizer_updates,
                    optimizer_state_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values()
                                              for v in state.values() if torch.is_tensor(v)), **memory))
            geometry_sum += g*n
            count_sum += c
            regularization_sum += r
            observations += n
            batches += 1
            self._progress(dict(event="scientific_update_completed", stage=stage.name, epoch=self.completed_epochs,
                batch=batch_index, optimizer_updates=self.optimizer_updates, guides=len(batch_ids), geometry=g,
                count_ce=c, objective=objective, preclip_gradient_norm=gradient_norm, head_gradient_maxima=head_values,
                geometry_backend="geomloss_batched_per_guide_schedule" if n else "not_evaluated_no_state_observations",
                geometry_batches=self._geometry.last_batches, execution_policy=POLICY, sampler=SAMPLER,
                elapsed_seconds=time.perf_counter()-started, **memory, **timing.summary()))
            del result, geometry, count, regularizer, scaled_count, loss, norm, heads, finite_parameters
        g, c, r = geometry_sum/max(1, observations), count_sum/max(1, batches), regularization_sum/max(1, batches)
        return dict(train_objective=self._objective_weight(stage, "checkpoint_geometry")*g
                    +self._objective_weight(stage, "grouped_count_likelihood")*c+r,
                    train_geometry=g, train_log_mass_error=0., train_count_loss=c,
                    train_observations=observations, train_batches=batches)

    @torch.no_grad()
    def _refresh_bank_for(self, data, bank, *, epoch, grid=None):
        if data is not self.data or bank is not self.bank or self.model.context_mode != "none":
            raise ValueError("Only the fitting bank may be refreshed")
        if grid is not None and grid is not self.training_grid:
            raise ValueError("Refresh grid differs from bound fitting grid")
        if bank.has_fixed_background or self.model.context_enabled:
            raise ValueError("GPU fitting refresh cannot substitute for ecology")
        was_training = self.model.training
        self.model.eval()
        try:
            bank.reset_coverage()
            bank.refresh_fixed_background(self.model)
            particles = self._catalog_refresh_particle_count()
            bank.refresh_particles = particles
            for chunk, start in enumerate(range(0, len(data.measure_ids), REFRESH_GUIDES)):
                ids = data.measure_ids[start:start+REFRESH_GUIDES]
                initial = self._law_cache.sample(ids, particles, seed=self.training_plan.seed+epoch*1009+chunk)
                noise = sample_noise(initial, self.training_grid,
                    seed=self.training_plan.seed+epoch*1009+chunk+2_000_003)
                result = tensor_rollout(self.model, self._tensor_dynamics, initial, self.training_grid, noise,
                                         self._residual_indices[initial.measure_indices])
                bank.update_from_rollout(result, self.model, data, full_refresh=True)
                del result, initial, noise
            bank.last_full_refresh_epoch = int(epoch)
            bank.assert_complete()
        finally:
            self.model.train(was_training)

    def close(self):
        if getattr(self, "_owns_cache", False):
            self._law_cache.close()
        super().close()

    def _manifest(self):
        manifest = super()._manifest()
        manifest.update(study_execution_policy=POLICY, gpu_execution_options=asdict(self._gpu_options),
                        source_sampler=SAMPLER, sampler_bitwise_scalar_replay=False)
        return manifest


class GPUFittingRecipe(ArchitectureFittingRecipe):
    def __init__(self, normalizers, execution, *, options=None, cache=None, progress=lambda _: None):
        super().__init__(normalizers, execution, progress=progress)
        self.options, self.cache = options or GPUOptions(), cache

    def build_objectives(self, study, config):
        return tuple(replace(term, config=dict(term.config, gpu_execution_policy=POLICY,
            gpu_execution_options=asdict(self.options), source_sampler=SAMPLER))
            if term.name == "checkpoint_geometry" else term for term in super().build_objectives(study, config))

    def execute_training(self, study, *, model, plan, objectives, run_config, **kwargs):
        class ThisFit(GPUFittingTrainer):
            _gpu_options = self.options
            _prepared_cache = self.cache
            _progress = staticmethod(self.progress)
        return ThisFit.from_plan(study, model, run_config, plan, objectives, **kwargs)
