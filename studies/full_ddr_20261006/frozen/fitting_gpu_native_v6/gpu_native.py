"""GPU-first, fitting-only primitives; never imported by the frozen v1 worker.

The normalized Sinkhorn objective retains EACH guide's original annealing
schedule. Calling SamplesLoss once on a batch would instead choose one global
diameter and change the finite-iteration objective. Ragged laws are padded only
with zero-probability atoms, never subsampled. The original scalar implementation
remains the independent numerical reference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.metadata import version
import inspect
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from credo.recipes.compact_sde_v3.objective import integrated_fitness_curve
from credo.recipes.compact_sde_v3.particles import ParticleRollout, ParticleState

POLICY = "gpu_native_fitting_cache_per_guide_sinkhorn_v1"
SAMPLER = "batched_padded_multinomial_v1"


@lru_cache(maxsize=1)
def geomloss_backend():
    """Bind internal helpers to the SAME backend as public scalar SamplesLoss.

    GeomLoss 0.3.1 moved the scalar implementation into ``_legacy``; 0.2.6
    exposes it directly. Resolve from the public class, never by catching an
    arbitrary import failure and silently trying a different estimator.
    This metadata-only check is safe before opening any cohort inputs.
    """
    from geomloss import SamplesLoss

    scalar_module = SamplesLoss.__module__
    if scalar_module not in {"geomloss.samples_loss", "geomloss._legacy.samples_loss"}:
        raise RuntimeError("Unqualified GeomLoss scalar backend: " + scalar_module)
    package = scalar_module.rsplit(".", 1)[0]
    scalar = import_module(scalar_module)
    divergence = import_module(package + ".sinkhorn_divergence")
    utilities = import_module(package + ".utils")
    samples = import_module(package + ".sinkhorn_samples")
    if (scalar.SamplesLoss is not SamplesLoss
            or scalar.routines["sinkhorn"]["tensorized"] is not samples.sinkhorn_tensorized):
        raise RuntimeError("GeomLoss public and tensorized backend identities differ")
    helpers = {}
    for name in ("log_weights", "sinkhorn_cost", "sinkhorn_loop", "epsilon_schedule"):
        helper = getattr(divergence, name)
        if not callable(helper) or helper.__module__ != divergence.__name__:
            raise RuntimeError("Unqualified GeomLoss helper: " + name)
        helpers[name] = helper
    for name in ("log_weights", "sinkhorn_cost", "sinkhorn_loop"):
        if samples.sinkhorn_tensorized.__globals__.get(name) is not helpers[name]:
            raise RuntimeError("GeomLoss batched/scalar helper identities differ: " + name)
    squared_distances = utilities.squared_distances
    if not callable(squared_distances) or squared_distances.__module__ != utilities.__name__:
        raise RuntimeError("Unqualified GeomLoss squared-distance helper")
    try:
        inspect.signature(helpers["sinkhorn_loop"]).bind(*(None,) * 9, debias=True)
        inspect.signature(helpers["sinkhorn_cost"]).bind(*(None,) * 8, batch=True, debias=True, potentials=False)
        inspect.signature(helpers["epsilon_schedule"]).bind(*(None,) * 4)
    except TypeError as error:
        raise RuntimeError("GeomLoss internal call signature changed; no fallback") from error
    return SimpleNamespace(**helpers, squared_distances=squared_distances,
        identity=dict(version=version("geomloss"), scalar_module=scalar_module,
                      divergence_module=divergence.__name__, utilities_module=utilities.__name__))


@dataclass(frozen=True)
class GPUOptions:
    """Explicit execution choices, not permission to launch or change a run."""

    maximum_cache_bytes: int = 8 * 1024**3
    maximum_sampler_elements: int = 16 * 1024**2
    maximum_geometry_elements: int = 16 * 1024**2
    maximum_geometry_guides: int = 64
    optimizer_backend: str = "foreach"
    precision: str = "strict_fp32"
    compile_dynamics: bool = False
    compiler_backend: str = "inductor"
    timing_interval: int = 20

    def __post_init__(self):
        for name in ("maximum_cache_bytes", "maximum_sampler_elements",
                     "maximum_geometry_elements", "maximum_geometry_guides", "timing_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Positive integer {name} required")
        if self.optimizer_backend not in {"single_tensor", "foreach", "fused"}:
            raise ValueError("Unknown optimizer backend")
        if self.precision not in {"strict_fp32", "tf32", "bf16_dynamics"}:
            raise ValueError("Unknown precision policy")
        if not isinstance(self.compile_dynamics, bool):
            raise ValueError("compile_dynamics must be boolean")
        if self.compiler_backend not in {"inductor", "eager"}:
            raise ValueError("Compiler backend must be explicit")


class DeviceLawCache:
    """One bounded device copy of fitting source/endpoint laws and count inputs.

    This object is tied to one exact TrajectoryData object. It never discovers
    files, reads RNA, encodes states, or accesses the validation problem. It may
    be shared by sequential fits while their prepared-input lease remains open.
    """

    def __init__(self, data, *, device, options=None):
        self.data = data
        self.device = torch.device(device)
        self.options = options or GPUOptions()
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only explicit CPU reference or CUDA execution is admitted")
        if tuple(data.axis.labels) != ("Stim8hr", "Stim48hr"):
            raise ValueError("Only the bound stimulated fitting interval is admitted")
        metadata = data.measure_meta.set_index("measure_id")
        if set(metadata.sample_id.astype(str)) != {"D3", "D4"}:
            raise ValueError("Cache may read fitting D3/D4 only")
        self.ids = tuple(data.measure_ids)
        self.lookup = {key: index for index, key in enumerate(self.ids)}
        if len(self.lookup) != len(self.ids):
            raise ValueError("Duplicate fitting measure")
        if any(not key.startswith(str(metadata.loc[key, "sample_id"]) + "::") for key in self.ids):
            raise ValueError("Measure identity and donor disagree")
        self.embedding_ids = tuple(str(metadata.loc[key, "embedding_id"]) for key in self.ids)
        self.context_ids = tuple(str(metadata.loc[key, "context_group_id"]) for key in self.ids)
        self.source_lengths = tuple(len(data.measures["Stim8hr"][key].support) for key in self.ids)
        self.target_lengths = tuple(len(data.measures["Stim48hr"][key].support)
                                    if key in data.measures["Stim48hr"] else 0 for key in self.ids)
        if min(self.source_lengths, default=0) < 1:
            raise ValueError("Every retained fitting source must contain cells")
        self.allocated_bytes = 0
        self._tensors = []
        self.closed = False
        self.source_offsets_host = np.cumsum((0,) + self.source_lengths, dtype=np.int64)[:-1]
        self.target_offsets_host = np.cumsum((0,) + self.target_lengths, dtype=np.int64)[:-1]
        # Admit the complete persistent inventory before any large allocation.
        cells = sum(self.source_lengths) + sum(self.target_lengths)
        required = cells * (data.latent_dim + 1) * 4 + len(self.ids) * (8 * 4 + 4)
        required += sum((len(b.counts) + len(b.background_counts)) * 12
                        + len(b.measure_indices) * 8 for b in data.count_blocks)
        if required > self.options.maximum_cache_bytes:
            raise RuntimeError("Device-law cache exceeds its bound inventory; no fallback")
        try:
            self.source_z, self.source_w = self._pack("Stim8hr", normalized=True)
            self.target_z, self.target_logw = self._pack("Stim48hr", normalized=False)
            self.source_offsets = self._store(torch.from_numpy(self.source_offsets_host))
            self.target_offsets = self._store(torch.from_numpy(self.target_offsets_host))
            self.source_lengths_device = self._store(torch.tensor(self.source_lengths, dtype=torch.long))
            self.target_lengths_device = self._store(torch.tensor(self.target_lengths, dtype=torch.long))
            masses = [math.log(data.measures["Stim8hr"][key].total_mass) for key in self.ids]
            self.source_log_mass = self._store(torch.tensor(masses, dtype=torch.float32))
            self.count_blocks = []
            for block in data.count_blocks:
                if block.time_label != "Stim48hr":
                    raise ValueError("Unexpected count role")
                exposure = torch.cat((block.exposure, block.background_exposure))
                counts = torch.cat((block.counts, block.background_counts))
                if not bool(torch.isfinite(exposure).all()) or not bool((exposure > 0).all()):
                    raise ValueError("Complete source exposures must be positive and finite")
                if not bool(torch.isfinite(counts).all()) or not bool((counts >= 0).all()) or float(counts.sum()) <= 0:
                    raise ValueError("Complete observed counts must be nonnegative with a positive total")
                self.count_blocks.append(dict(
                    indices=self._store(block.measure_indices),
                    log_exposure=self._store(exposure.log()),
                    frequencies=self._store(counts / counts.sum()),
                    background_fitness=self._store(block.background_fitness),
                ))
            if len(self.count_blocks) != 2:
                raise ValueError("Two complete fitting-donor count blocks required")
        except BaseException:
            self.close()
            raise

    def _store(self, tensor):
        tensor = tensor.detach().to(device=self.device).contiguous()
        size = tensor.numel() * tensor.element_size()
        if self.allocated_bytes + size > self.options.maximum_cache_bytes:
            raise RuntimeError("Device-law cache allocation budget exceeded")
        self._tensors.append(tensor)
        self.allocated_bytes += size
        return tensor

    def _pack(self, label, *, normalized):
        lengths = self.source_lengths if normalized else self.target_lengths
        support = torch.empty((sum(lengths), self.data.latent_dim), dtype=torch.float32)
        weights = torch.empty(sum(lengths), dtype=torch.float32)
        start = 0
        for key, length in zip(self.ids, lengths, strict=True):
            if not length:
                continue
            law = self.data.measures[label][key]
            z = np.asarray(law.support)
            w = np.asarray(law.normalized_weights if normalized else law.weights)
            if z.shape != (length, self.data.latent_dim) or w.shape != (length,):
                raise ValueError("Malformed cached law")
            if not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
                raise ValueError("Invalid cached state/weight support")
            support[start:start + length].copy_(torch.tensor(z))
            weights[start:start + length].copy_(torch.tensor(w))
            start += length
        support = self._store(support)
        # This is exactly the reference endpoint clamp, before logarithm.
        weights = self._store(weights if normalized else weights.clamp_min(1e-30).log())
        return support, weights

    def require(self, data):
        if self.closed or data is not self.data:
            raise ValueError("Cache is closed or belongs to a different prepared object")

    def indices(self, ids):
        self.require(self.data)
        ids = tuple(ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Nonempty distinct measure IDs required")
        try:
            indices = tuple(self.lookup[key] for key in ids)
        except KeyError as error:
            raise KeyError("Unknown fitting source measure") from error
        return indices, torch.tensor(indices, device=self.device, dtype=torch.long)

    def sample(self, ids, particles, *, seed):
        """Batch multinomial draws; same probability law, NEW explicit RNG stream.

        It is intentionally not described as bitwise replay of the scalar
        sampler. All source support is retained; zero-probability padding is
        never sampled. Tiles have a bound independent of guide imbalance.
        """
        if isinstance(particles, bool) or not isinstance(particles, int) or particles < 2:
            raise ValueError("At least two particles required")
        ids = tuple(ids)
        host, indices = self.indices(ids)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        outputs = []
        start = 0
        while start < len(ids):
            width = self.source_lengths[host[start]]
            stop = start + 1
            if width > self.options.maximum_sampler_elements:
                raise RuntimeError("Single source exceeds sampler workspace budget")
            while stop < len(ids):
                proposal = max(width, self.source_lengths[host[stop]])
                if (stop - start + 1) * proposal > self.options.maximum_sampler_elements:
                    break
                width = proposal
                stop += 1
            active = indices[start:stop]
            offsets = self.source_offsets[active, None]
            length = self.source_lengths_device[active, None]
            columns = torch.arange(width, device=self.device)[None, :]
            valid = columns < length
            absolute = offsets + torch.minimum(columns, length - 1)
            probabilities = torch.where(valid, self.source_w[absolute], 0.0)
            selected = torch.multinomial(probabilities, particles, replacement=True, generator=generator)
            outputs.append(self.source_z[offsets + selected])
            start = stop
        return ParticleState(z=torch.cat(outputs),
            logw=torch.full((len(ids), particles), -math.log(particles), device=self.device),
            log_m0=self.source_log_mass[indices], measure_ids=ids,
            embedding_ids=tuple(self.embedding_ids[i] for i in host),
            context_group_ids=tuple(self.context_ids[i] for i in host), measure_indices=indices,
            residual_scale=torch.ones(len(ids), device=self.device))

    def targets(self, host_indices, *, width):
        self.require(self.data)
        if not host_indices or any(i < 0 or i >= len(self.ids) or self.target_lengths[i] < 1
                                   or self.target_lengths[i] > width for i in host_indices):
            raise ValueError("Full nonempty endpoint support required")
        indices = torch.tensor(host_indices, device=self.device, dtype=torch.long)
        lengths = self.target_lengths_device[indices, None]
        offsets = self.target_offsets[indices, None]
        columns = torch.arange(width, device=self.device)[None, :]
        valid = columns < lengths
        absolute = offsets + torch.minimum(columns, lengths - 1)
        support = self.target_z[absolute]
        log_weight = torch.where(valid, self.target_logw[absolute], -torch.inf)
        return support, log_weight, valid

    def count_loss(self, result, bank):
        self.require(self.data)
        curve = integrated_fitness_curve(result, mass_increment="particle_log_mass")
        full = bank.fitness_for_active(time_label="Stim48hr", active_indices=result.measure_indices,
                                       active_fitness=curve[-1])
        losses = []
        for block in self.count_blocks:
            fitness = torch.cat((full[block["indices"]], block["background_fitness"]))
            losses.append(-(block["frequencies"] * (block["log_exposure"] + fitness).log_softmax(0)).sum())
        return torch.stack(losses).mean()

    def close(self):
        self.closed = True
        self._tensors.clear()
        # Release persistent device references without resetting CUDA's allocator.
        for name in ("source_z", "source_w", "target_z", "target_logw", "source_offsets", "target_offsets",
                     "source_lengths_device", "target_lengths_device", "source_log_mass", "count_blocks"):
            if hasattr(self, name):
                delattr(self, name)


def softmin_batched(eps, cost, log_potential):
    """Per-guide temperatures, not a batch-global annealing temperature."""
    return -eps * (log_potential[:, None, :] - cost / eps[:, :, None]).logsumexp(2)


def sinkhorn_batched(x, log_a, y, log_b, schedules):
    """Upstream tensorized solver with independent scalar-guide schedules.

    Only groups with equal schedule LENGTH are combined. Values remain different
    for each guide. Right-side detach and envelope gradients match GeomLoss.
    The upstream loop changes global grad mode; restore it even on failure.
    """
    backend = geomloss_backend()
    log_weights, sinkhorn_cost, sinkhorn_loop = backend.log_weights, backend.sinkhorn_cost, backend.sinkhorn_loop
    squared_distances = backend.squared_distances

    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[-1] != y.shape[-1]:
        raise ValueError("Matching batched point clouds required")
    if log_a.shape != x.shape[:2] or log_b.shape != y.shape[:2]:
        raise ValueError("Weights and supports disagree")
    if len(schedules) != x.shape[0] or not schedules or len({len(s) for s in schedules}) != 1:
        raise ValueError("One equal-length annealing schedule per guide required")
    if any(any(not math.isfinite(eps) or eps <= 0 for eps in s) for s in schedules):
        raise ValueError("Positive finite temperatures required")
    if len({s[-1] for s in schedules}) != 1:
        raise ValueError("All guides must use the same terminal epsilon")
    a, b = log_a.softmax(-1), log_b.softmax(-1)
    c_xy = squared_distances(x, y.detach()) / 2
    c_yx = squared_distances(y, x.detach()) / 2
    c_xx = squared_distances(x, x.detach()) / 2
    c_yy = squared_distances(y, y.detach()) / 2
    schedule = torch.tensor(np.asarray(schedules).T.copy(), device=x.device, dtype=x.dtype)
    eps_list = [row[:, None] for row in schedule]
    grad_enabled = torch.is_grad_enabled()
    try:
        f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(softmin_batched, log_weights(a), log_weights(b),
            c_xx, c_yy, c_xy, c_yx, eps_list, None, debias=True)
        return sinkhorn_cost(schedules[0][-1], None, a, b, f_aa, g_bb, g_ab, f_ba,
                             batch=True, debias=True, potentials=False)
    finally:
        torch.set_grad_enabled(grad_enabled)


def geometry_width(length, particles, options):
    """Retain full support; padding is optional, the workspace bound is not.

    Most laws share power-of-two buckets. If padding alone would make even a
    singleton inadmissible, use its exact length with the SAME dense solver.
    A truly oversized law still fails closed, before scientific updates.
    """
    if type(length) is not int or length < 1 or type(particles) is not int or particles < 1:
        raise ValueError("Positive integer geometry support and particle counts required")
    padded = 1 << (length - 1).bit_length()
    width = padded if (particles + padded)**2 <= options.maximum_geometry_elements else length
    if (particles + width)**2 > options.maximum_geometry_elements:
        raise RuntimeError("Single full-support geometry exceeds workspace budget: "
            f"support={length}, particles={particles}, exact_elements={(particles + length)**2}, "
            f"maximum={options.maximum_geometry_elements}")
    return width


def geometry_admission(cache, particles):
    """Metadata-only coverage of EVERY fitting law, not a sampled minibatch."""
    cache.require(cache.data)
    exceptions, maximum, paired = [], 0, 0
    for key, length in zip(cache.ids, cache.target_lengths, strict=True):
        if not length:
            continue
        try:
            width = geometry_width(length, particles, cache.options)
        except RuntimeError as error:
            raise RuntimeError(f"Geometry admission failed for {key}: {error}") from error
        paired += 1
        maximum = max(maximum, (particles + width)**2)
        if width != 1 << (length - 1).bit_length():
            exceptions.append(dict(measure_id=key, support=length, width=width,
                pairwise_elements=(particles + width)**2))
    return dict(status="all_fitting_geometry_support_admitted", particles=particles,
        fitting_laws=len(cache.ids), paired_laws=paired,
        maximum_support=max(cache.target_lengths), maximum_singleton_elements=maximum,
        maximum_geometry_elements=cache.options.maximum_geometry_elements,
        exact_width_laws=exceptions, dropped_cells=0, solver="unchanged_geomloss_tensorized",
        observation_weights_changed=False)


class BatchedGeometry:
    """Ragged full-support losses with bounded buckets and one diameter readback."""

    def __init__(self, cache, *, epsilon=0.1):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Positive finite Sinkhorn epsilon required")
        self.cache, self.epsilon = cache, epsilon
        self.last_batches = 0

    def __call__(self, result):
        epsilon_schedule = geomloss_backend().epsilon_schedule

        cache = self.cache
        cache.require(cache.data)
        selected = [(row, cache.lookup[key]) for row, key in enumerate(result.measure_ids)
                    if cache.target_lengths[cache.lookup[key]] > 0]
        self.last_batches = 0
        if not selected:
            return result.z_steps.sum() * 0, 0
        # Endpoint bounds are computed in shape buckets, then transferred ONCE.
        # The predicted/observed min/max reproduces max_diameter for each guide.
        buckets = defaultdict(list)
        particles = result.terminal_z.shape[1]
        for row, index in selected:
            width = geometry_width(cache.target_lengths[index], particles, cache.options)
            buckets[width].append((row, index))
        diameter_parts = []
        bucket_specs = []
        for width, members in buckets.items():
            per_guide = particles**2 + 2 * particles * width + width**2
            batch_limit = min(cache.options.maximum_geometry_guides,
                              cache.options.maximum_geometry_elements // per_guide)
            if batch_limit < 1:
                raise RuntimeError("Single full-support geometry exceeds workspace budget")
            for start in range(0, len(members), batch_limit):
                chunk = members[start:start + batch_limit]
                rows = torch.tensor([r for r, _ in chunk], device=cache.device)
                y, _, valid = cache.targets([i for _, i in chunk], width=width)
                x = result.terminal_z[rows].detach()
                low_y = y.masked_fill(~valid[..., None], torch.inf).amin(1)
                high_y = y.masked_fill(~valid[..., None], -torch.inf).amax(1)
                lower = torch.minimum(x.amin(1), low_y)
                upper = torch.maximum(x.amax(1), high_y)
                diameter_parts.append((upper - lower).norm(dim=-1))
                bucket_specs.append((width, chunk))
        diameters = torch.cat(diameter_parts).detach().cpu().tolist()
        if any(not math.isfinite(d) or d <= 0 for d in diameters):
            raise RuntimeError("Degenerate Sinkhorn annealing diameter; no silent fallback")
        offset, chunks, values, value_rows = 0, defaultdict(list), [], []
        for width, members in bucket_specs:
            for row, index in members:
                schedule = epsilon_schedule(2, diameters[offset], self.epsilon**0.5, 0.5)
                chunks[(width, len(schedule))].append((row, index, schedule))
                offset += 1
        for (width, _), members in chunks.items():
            per_guide = particles**2 + 2 * particles * width + width**2
            limit = min(cache.options.maximum_geometry_guides,
                        cache.options.maximum_geometry_elements // per_guide)
            for start in range(0, len(members), limit):
                chunk = members[start:start + limit]
                rows = torch.tensor([r for r, _, _ in chunk], device=cache.device)
                y, log_b, _ = cache.targets([i for _, i, _ in chunk], width=width)
                x = result.terminal_z[rows]
                log_a = result.log_m0[rows, None] + result.logw_steps[-1, rows]
                values.append(sinkhorn_batched(x, log_a, y, log_b, [s for _, _, s in chunk]))
                value_rows.extend(r for r, _, _ in chunk)
                self.last_batches += 1
        # Preserve guide order when reducing; no bucket-size or capture weighting.
        combined = torch.cat(values)
        ordering = torch.tensor(np.argsort(value_rows).copy(), device=cache.device)
        return combined[ordering].mean(), len(selected)


class TensorDynamics(nn.Module):
    """Tensor-only dynamics boundary; no string lookup inside integration steps."""

    def __init__(self, model, *, precision="strict_fp32"):
        super().__init__()
        if model.context_mode != "none":
            raise ValueError("No ecological-context substitution is permitted")
        self.model, self.precision = model, precision

    def forward(self, z, time, effective, intercept):
        model = self.model
        with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16,
                            enabled=self.precision == "bf16_dynamics"):
            embedded_time = model.time_embedding(time)
            common = torch.cat((z, embedded_time.reshape(1, 1, -1).expand(z.shape[0], z.shape[1], -1)), -1)
            hidden = model.trunk(common)
            drift = model._modulated_head(hidden, effective, model.drift_reference, model.drift_residual, model.latent_dim)
            sigma = torch.nn.functional.softplus(model._modulated_head(hidden, effective,
                model.sigma_reference, model.sigma_residual, model.latent_dim)) + model.sigma_min
            if model.growth_enabled:
                raw = model.growth_reference(hidden).squeeze(-1)
                raw = raw + torch.einsum("gnr,gr->gn", model.growth_residual(hidden), effective)
                growth = model.growth_max * torch.tanh(raw + intercept[:, None])
            else:
                growth = z.new_zeros(z.shape[:2])
        # Keep integration, weights, objective and parameter master copies FP32.
        return drift.float(), sigma.float(), growth.float()


def tensor_rollout(model, dynamics, initial, grid, noise, residual_indices):
    """Same Euler–Maruyama equations; effective target tensors computed once."""
    if model.context_mode != "none" or grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("Bound no-context integration grid required")
    if noise.shape != (len(grid)-1,) + initial.z.shape:
        raise ValueError("Noise shape differs from the prescribed integration")
    if residual_indices.shape != (initial.z.shape[0],):
        raise ValueError("One residual lookup index per guide required")
    residual = torch.cat((model.residual_embedding.new_zeros(1, model.embedding_dim), model.residual_embedding))[residual_indices]
    effective = model.reference_embedding[None, :] + residual * initial.residual_scale[:, None]
    intercept = torch.cat((model.growth_bias.new_zeros(1), model.growth_bias))[residual_indices] * initial.residual_scale
    z, logw = initial.z.clone(), initial.logw.clone()
    zs, ws, drifts, sigmas, growths = [z], [logw], [], [], []
    for step in range(len(grid)-1):
        drift, sigma, growth = dynamics(z, grid[step], effective, intercept)
        dt = grid[step+1] - grid[step]
        z = z + drift * dt + sigma * dt.sqrt() * noise[step]
        logw = logw + growth * dt
        zs.append(z)
        ws.append(logw)
        drifts.append(drift)
        sigmas.append(sigma)
        growths.append(growth)
    return ParticleRollout(z_steps=torch.stack(zs), logw_steps=torch.stack(ws), log_m0=initial.log_m0,
        axis_grid=grid, measure_ids=initial.measure_ids, embedding_ids=initial.embedding_ids,
        context_group_ids=initial.context_group_ids, measure_indices=initial.measure_indices,
        residual_scale=initial.residual_scale, drift_steps=torch.stack(drifts), sigma_steps=torch.stack(sigmas),
        growth_steps=torch.stack(growths), context_steps=z.new_zeros(len(grid)-1, z.shape[0], model.n_programs),
        noise_steps=noise)
