"""Integrated first-order frontier objective; no optimizer, sampler or file I/O.

Inputs are explicit fitting-only tensors, fixed source/noise and pool identities.
Scientific coefficients/scales/pooling are supplied through a named protocol;
this module does not freeze an unapproved auxiliary matrix or infer observations.
The count prepass also decodes targeting laws. Shared control derivatives are
accumulated over ALL contrasts and replayed through each fixed pool component.
Only scalar derivative proxies have zero displayed value: the actual molecular
objective is returned separately and added to the reported complete objective.

Local regularizers retain comparator_models.py:336-358 equations. DIRECT mixture
actions integrate using proposal correction, not reaction-normalized weights.
DDR actions average their eight steps. No ecological payoff constant is created.
All microbatch reductions use full logical denominators. A failed call may leave
partial gradients: caller MUST NOT step. Cache writes, clipping, optimizer steps,
accepted-update accounting, checkpointing and scientific admission are external.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor

from .count_objective import CatalogBlock, FitnessCache
from .geometry import ObservedLaw, RaggedSinkhornGeometry
from .logical_batch import (LocalForward, LogicalBatchResult, LogicalMicrobatch,
    LossReduction, accumulate_logical_batch, _module_identity, _tensor_identity,
    _rng_state, _same_rng, _cache_identity)
from .measure_ops import multiscale_mmd2
from .models import FrontierModel
from .molecular import _composition
from .molecular_adjoint import molecular_adjoint
from .observation import SparseR48


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Explicit finite positive scale or nonnegative coefficient required")


@dataclass(frozen=True)
class ObjectiveProtocol:
    protocol_id: str
    family: str
    training_donors: tuple[str, ...]
    state_coefficient: float
    count_coefficient: float
    shape_coefficient: float
    endpoint_coefficient: float
    effect_coefficient: float
    state_scale: float
    count_scale: float
    shape_scale: float
    endpoint_scale: float
    effect_scale: float
    deterministic_action: float
    stochastic_action: float
    reaction_action: float
    reference_embedding_penalty: float
    target_embedding_penalty: float
    target_growth_penalty: float
    shape_bandwidths: tuple[float, ...]

    def __post_init__(self):
        if (not isinstance(self.protocol_id, str) or not self.protocol_id or self.family not in {f"F{i}" for i in range(6)}
                or type(self.training_donors) is not tuple or len(self.training_donors) not in (2, 3)
                or self.training_donors != tuple(sorted(set(self.training_donors)))
                or not set(self.training_donors) <= {"D1", "D2", "D3", "D4"}):
            raise ValueError("Named family protocol and canonical two/three-donor fitting subset required")
        for name in ("state_coefficient", "count_coefficient", "shape_coefficient", "endpoint_coefficient", "effect_coefficient",
                     "deterministic_action", "stochastic_action", "reaction_action", "reference_embedding_penalty",
                     "target_embedding_penalty", "target_growth_penalty"):
            _number(getattr(self, name))
        for name in ("state_scale", "count_scale", "shape_scale", "endpoint_scale", "effect_scale"):
            _number(getattr(self, name), positive=True)
        if type(self.shape_bandwidths) is not tuple or not self.shape_bandwidths:
            raise ValueError("Fixed fitting-only multiscale bandwidth tuple required")
        for width in self.shape_bandwidths:
            _number(width, positive=True)


@dataclass(frozen=True)
class ObjectiveExecution:
    microbatch_laws: int
    control_microbatch_laws: int
    gene_block: int
    decoder_row_block: int
    shape_block: int
    maximum_molecular_elements: int
    replay_atol: float
    replay_rtol: float

    def __post_init__(self):
        for name in ("microbatch_laws", "control_microbatch_laws", "gene_block", "decoder_row_block", "shape_block", "maximum_molecular_elements"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Explicit positive objective execution bounds required")
        _number(self.replay_atol)
        _number(self.replay_rtol)


def molecular_working_element_bound(*, targets, donors, components, genes,
                                   target_microbatch, control_microbatch):
    """Conservative peak admission for objective-level full-gene buffers.

    All units are FP64-equivalent elements (eight bytes each), including the
    caller's target/control truth. This is NOT total GPU memory: model/decoder
    activations, particles, geometry, optimizer state and allocator workspaces
    require their separate bounds and measured H100 qualification.

    The deliberately additive envelope covers stages that do not all peak at
    once. Per target, twelve gene vectors cover truth, retained first-pass
    predictions, their stack, the adjoint prediction clone and derivative,
    transformed predictions/truth, endpoint/contrast errors, indexed controls,
    and derivative arithmetic/validation temporaries. Ten per donor cover
    truth, pooled predictions, the adjoint clone/derivative, transforms/errors
    and pooling/derivative temporaries. Four per control component cover the
    first-pass list/stack and weighted pooling/copy buffers. Twelve per active
    target/control replay row cover current/stacked compositions, indexed
    expected compositions/derivatives, allclose/difference/product intermediates
    and first-order proxy/backward buffers. Eight additional gene vectors cover
    a single composition's returned, concatenated, detached/cloned and reduction
    scratch buffers. Boolean validation arrays are charged as FP64 as well.

    In particular, the old two-vectors-per-target estimate omitted even the
    retained first-pass array plus TWO separate adjoint arrays. Do not replace
    this envelope with only the final adjoint size or raise a caller's cap to
    make an oversized logical batch pass.
    """
    values = (targets, donors, components, genes, target_microbatch, control_microbatch)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Positive exact molecular dimensions and microbatch bounds required")
    replay_targets = min(targets, target_microbatch)
    replay_controls = min(components, control_microbatch)
    vector_rows = (12 * targets + 10 * donors + 4 * components
                   + 12 * replay_targets + 12 * replay_controls + 8)
    return vector_rows * genes


@dataclass(frozen=True)
class LogicalTrainingInputs:
    law_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    source: Tensor
    noise: Tensor
    cache_indices: Tensor
    observations: tuple[ObservedLaw | None, ...]
    geometry_weights: Tensor
    action_weights: Tensor


@dataclass(frozen=True)
class PooledControlInputs:
    """Fixed RNA-state mixture; not captured-frequency or reaction-mass pooling.

    Components can be original NTC source laws or one admitted pooled source
    law per donor. Weights sum to one WITHIN each donor and remain fixed. The
    owner must bind which physical population this represents before fitting.
    """
    donor_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    component_donor_indices: Tensor
    component_target_ids: tuple[str, ...]
    source: Tensor
    noise: Tensor
    mixture_weights: Tensor
    observed_compositions: Tensor
    reduction_weights: Tensor


@dataclass(frozen=True)
class MolecularTrainingInputs:
    """Fixed targeting reduction and one full pooled-control reduction.

    Under uniform without-replacement law sampling, the provider supplies
    normalized sampled target weights and ``(N/B) * sum(sampled global target
    weights)`` as the multiplier. This restores the fixed target/donor-macro
    estimator without scaling the separately evaluated donor control mean.
    It also scales the control-side derivative contributed by target contrasts.
    The provider owns N, B and global weights; none are inferred here.
    """
    target_law_indices: Tensor
    observed_targets: Tensor
    target_weights: Tensor
    controls: PooledControlInputs
    target_estimator_multiplier: float = 1.0

    def __post_init__(self):
        _number(self.target_estimator_multiplier, positive=True)


@dataclass
class TrainingObjectiveResult:
    displayed_objective: Tensor
    logical_result: LogicalBatchResult
    molecular_objective: Tensor | None
    molecular_terms: dict
    state_observations: int
    target_molecular_observations: int
    control_molecular_observations: int
    control_component_first_passes: int
    control_component_replays: int
    model_forward_calls: int
    decoder_law_calls: int
    observed_weight_clamp_counts: tuple[int, ...]
    protocol_id: str
    optimizer_calls: int = 0
    cache_writes: int = 0


def _fixed(value, *, shape, reference, nonnegative=False, normalized=False):
    if (not isinstance(value, Tensor) or value.shape != shape or value.device != reference.device
            or value.dtype != reference.dtype or value.requires_grad or not bool(torch.isfinite(value).all())
            or (nonnegative and bool((value < 0).any()))):
        raise ValueError("Aligned finite fixed tensor required")
    tolerance = 2e-6 if value.dtype == torch.float32 else 1e-10
    if normalized and not torch.allclose(value.sum(), value.new_tensor(1), atol=tolerance, rtol=0):
        raise ValueError("Fixed logical weights must sum to one")


def _distinct(values, length):
    return (type(values) is tuple and len(values) == length and len(set(values)) == length
            and all(isinstance(value, str) and value for value in values))


def _inputs(model, inputs, protocol):
    if not isinstance(model, FrontierModel) or not isinstance(inputs, LogicalTrainingInputs) or not isinstance(protocol, ObjectiveProtocol):
        raise ValueError("Explicit frontier model, logical inputs and protocol required")
    if model.config.family != protocol.family or model.phase not in ("state", "joint"):
        raise ValueError("Family/phase differs from the objective protocol")
    source = inputs.source
    if (not isinstance(source, Tensor) or source.ndim != 3 or source.shape[-1] != 48 or min(source.shape[:2]) < 1
            or source.dtype != torch.float32 or source.requires_grad or not bool(torch.isfinite(source).all())
            or any(p.device != source.device or p.dtype != source.dtype for p in model.parameters())):
        raise ValueError("Fixed FP32 source particles and model on one device required")
    n = len(source)
    _fixed(inputs.noise, shape=(8, *source.shape), reference=source)
    if (not _distinct(inputs.law_ids, n) or len(inputs.target_ids) != n or len(inputs.donor_ids) != n
            or any(not isinstance(value, str) or not value for value in inputs.target_ids)
            or not set(inputs.donor_ids) <= set(protocol.training_donors) or len(inputs.observations) != n):
        raise ValueError("Bound law, actual target, fitting donor and observation routing required")
    index = inputs.cache_indices
    if (not isinstance(index, Tensor) or index.shape != (n,) or index.dtype != torch.int64 or index.device != source.device
            or index.unique().numel() != n or bool((index < 0).any())):
        raise ValueError("Distinct nonnegative cache routing required")
    _fixed(inputs.geometry_weights, shape=(n,), reference=source, nonnegative=True)
    _fixed(inputs.action_weights, shape=(n,), reference=source, nonnegative=True, normalized=True)
    for row, law in enumerate(inputs.observations):
        if law is not None and (not isinstance(law, ObservedLaw) or law.law_id != inputs.law_ids[row]):
            raise ValueError("Observed geometry identity mismatch")
    available = [row for row, law in enumerate(inputs.observations) if law is not None]
    denominator = inputs.geometry_weights[available].sum().item()
    if protocol.state_coefficient > 0 and denominator <= 0:
        raise ValueError("Positive state objective has no positive observed-law denominator")
    return available, denominator


def _molecular_inputs(model, inputs, molecular, decoder, protocol, execution):
    if not isinstance(molecular, MolecularTrainingInputs) or not isinstance(molecular.controls, PooledControlInputs) or not isinstance(decoder, SparseR48):
        raise ValueError("Explicit molecular targets, pooled controls and frozen fold decoder required")
    _number(molecular.target_estimator_multiplier, positive=True)
    if (decoder.training or any(p.requires_grad for p in decoder.parameters())
            or decoder.fold.training_donors != protocol.training_donors or decoder.device != inputs.source.device
            or decoder.first.weight.dtype != torch.float32 or decoder.latent_dim != 48):
        raise ValueError("Decoder must be frozen on this exact fitting subset/device/latent axis")
    decoder._check_scaling()
    controls = molecular.controls
    n = len(inputs.source)
    rows = molecular.target_law_indices
    if (not isinstance(rows, Tensor) or rows.dtype != torch.int64 or rows.device != inputs.source.device
            or rows.ndim != 1 or not rows.numel() or rows.unique().numel() != rows.numel()
            or bool(((rows < 0) | (rows >= n)).any())):
        raise ValueError("Distinct supported logical targeting-law indices required")
    if any(inputs.target_ids[row] in model.control_ids for row in rows.tolist()):
        raise ValueError("NTC laws are not targeting molecular observations")
    _composition(molecular.observed_targets, ndim=2, name="fixed targeting truth")
    if (molecular.observed_targets.dtype != torch.float64 or molecular.observed_targets.device != inputs.source.device
            or molecular.observed_targets.requires_grad or molecular.observed_targets.shape != (len(rows), decoder.features)):
        raise ValueError("Fixed full-gene FP64 target means required for blocked decoder accumulation")
    _fixed(molecular.target_weights, shape=(len(rows),), reference=molecular.observed_targets, nonnegative=True, normalized=True)
    if controls.donor_ids != protocol.training_donors:
        raise ValueError("One pooled NTC endpoint for each fitting donor required")
    c, d = len(controls.source), len(controls.donor_ids)
    if (controls.source.ndim != 3 or controls.source.shape[-1] != 48 or controls.source.shape[1] != inputs.source.shape[1]
            or c < 1 or not _distinct(controls.component_ids, c) or len(controls.component_target_ids) != c
            or not set(controls.component_target_ids) <= set(model.control_ids)):
        raise ValueError("Bound NTC components with the common particle budget required")
    _fixed(controls.source, shape=controls.source.shape, reference=inputs.source)
    _fixed(controls.noise, shape=(8, *controls.source.shape), reference=inputs.source)
    routing = controls.component_donor_indices
    if (not isinstance(routing, Tensor) or routing.dtype != torch.int64 or routing.shape != (c,)
            or routing.device != inputs.source.device or bool(((routing < 0) | (routing >= d)).any())):
        raise ValueError("Exact control component-to-donor routing required")
    _composition(controls.observed_compositions, ndim=2, name="fixed pooled control truth")
    if (controls.observed_compositions.dtype != torch.float64 or controls.observed_compositions.device != inputs.source.device
            or controls.observed_compositions.requires_grad or controls.observed_compositions.shape != (d, decoder.features)):
        raise ValueError("Fixed full-gene FP64 control means required")
    _fixed(controls.mixture_weights, shape=(c,), reference=controls.observed_compositions, nonnegative=True)
    for donor in range(d):
        weights = controls.mixture_weights[routing == donor]
        if not len(weights) or not torch.allclose(weights.sum(), weights.new_tensor(1), atol=1e-10, rtol=0):
            raise ValueError("Every donor needs an original complete normalized control pool")
    _fixed(controls.reduction_weights, shape=(d,), reference=controls.observed_compositions, nonnegative=True, normalized=True)
    required_elements = molecular_working_element_bound(targets=len(rows), donors=d,
        components=c, genes=decoder.features, target_microbatch=execution.microbatch_laws,
        control_microbatch=execution.control_microbatch_laws)
    if required_elements > execution.maximum_molecular_elements:
        raise ValueError("Molecular working-buffer envelope exceeds its unchanged explicit element budget: "
                         f"required={required_elements}, cap={execution.maximum_molecular_elements}")
    target_donors = torch.tensor([controls.donor_ids.index(inputs.donor_ids[row]) for row in rows.tolist()],
                                device=inputs.source.device, dtype=torch.int64)
    return target_donors


def _action_sums(model, result, law_weights):
    if model.config.dynamics == "direct":
        if result.anchor_displacement is None:
            raise ValueError("Direct training requires the differentiable zero-noise anchor")
        base = result.mixture_log_correction.exp() / result.states.shape[1]
        deterministic = (base * result.anchor_displacement.square().mean(-1)).sum(-1)
        stochastic = (base * (result.displacement - result.anchor_displacement).square().mean(-1)).sum(-1)
        reaction = (base * result.reaction_log_increment.square()).sum(-1)
    else:
        deterministic = result.drift_steps.square().mean((0, 2, 3))
        stochastic = result.sigma_steps.square().mean((0, 2, 3))
        reaction = result.reaction_steps.square().mean((0, 2))
    return {"deterministic_action": (law_weights * deterministic).sum(),
            "stochastic_action": (law_weights * stochastic).sum(),
            "reaction_action": (law_weights * reaction).sum()}


def _global_regularizer(model, protocol):
    result = protocol.reference_embedding_penalty * model.reference_embedding.square().mean()
    if model.config.target_residual:
        result = result + protocol.target_embedding_penalty * (model.target_codes @ model.target_projection).square().mean()
        if model.phase == "joint":
            result = result + protocol.target_growth_penalty * model.target_growth_intercepts.square().mean()
    return result


def accumulate_training_objective(model, inputs, *, protocol, execution, geometry,
                                  blocks: Sequence[CatalogBlock] | None, cache: FitnessCache | None,
                                  sampling_scale: float, decoder=None, molecular=None) -> TrainingObjectiveResult:
    """Backward a complete logical objective, retaining existing parameter grads.

    The decoder's blocked composition returns FP64; molecular adjoints stay FP64.
    Their zero-valued proxies are cast to model FP32 only at the scalar interface.
    True displayed molecular losses retain FP64. Control components are replayed
    exactly once and are NOT extra catalog/state/action observations.
    """
    available, denominator = _inputs(model, inputs, protocol)
    if not isinstance(execution, ObjectiveExecution) or not isinstance(geometry, RaggedSinkhornGeometry):
        raise ValueError("Explicit objective execution and retained geometry required")
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("Ordinary enabled autograd required")
    joint = model.phase == "joint"
    count_weight = protocol.count_coefficient / protocol.count_scale if joint else 0.
    shape_weight = protocol.shape_coefficient / protocol.shape_scale if joint else 0.
    molecular_enabled = joint and (protocol.endpoint_coefficient > 0 or protocol.effect_coefficient > 0)
    if count_weight > 0:
        if (not blocks or not isinstance(cache, FitnessCache) or any(not isinstance(block, CatalogBlock) for block in blocks)
                or tuple(block.donor_id for block in blocks) != protocol.training_donors):
            raise ValueError("Count catalog donors differ from the bound fitting subset")
        owners = torch.full((len(cache.values),), -1, device=inputs.source.device, dtype=torch.int64)
        for donor, block in enumerate(blocks):
            block.validate(cache)
            owners.index_fill_(0, block.cache_indices, donor)
        if bool((inputs.cache_indices >= len(cache.values)).any()):
            raise ValueError("Logical law cache index lies outside the full modeled catalog")
        expected_owners = torch.tensor([protocol.training_donors.index(donor) for donor in inputs.donor_ids],
                                      device=inputs.source.device, dtype=torch.int64)
        if not torch.equal(owners[inputs.cache_indices], expected_owners):
            raise ValueError("Logical source donor and complete-catalog cache routing disagree")
    target_donors = _molecular_inputs(model, inputs, molecular, decoder, protocol, execution) if molecular_enabled else None
    source = inputs.source
    n = len(source)
    cuda = (source.device.index,) if source.device.type == "cuda" else ()
    rng = _rng_state(cuda)
    model_stamp = _module_identity(model)
    decoder_stamp = _module_identity(decoder) if molecular_enabled else None
    cache_stamp = _cache_identity(cache)
    tensors = [source, inputs.noise, inputs.cache_indices, inputs.geometry_weights, inputs.action_weights]
    for law in inputs.observations:
        if law is not None:
            tensors.extend((law.support, law.weights))
    if count_weight > 0:
        for block in blocks:
            tensors.extend((block.log_source, block.endpoint_counts, block.modeled_indices, block.cache_indices))
    if molecular_enabled:
        controls = molecular.controls
        tensors.extend((molecular.target_law_indices, molecular.observed_targets, molecular.target_weights,
            controls.source, controls.noise, controls.component_donor_indices, controls.mixture_weights,
            controls.observed_compositions, controls.reduction_weights))
    tensor_stamp = tuple(_tensor_identity(value) for value in tensors)

    def unchanged():
        if (_module_identity(model) != model_stamp or _cache_identity(cache) != cache_stamp
                or (molecular_enabled and _module_identity(decoder) != decoder_stamp)
                or tuple(_tensor_identity(value) for value in tensors) != tensor_stamp
                or not _same_rng(rng, _rng_state(cuda))):
            raise RuntimeError("Training objective mutated inputs/model/cache/RNG; discard partial gradients and do not step")

    calls = {"model": 0, "decoder": 0}
    def rollout(z, noise, ids, anchor):
        unchanged()
        calls["model"] += 1
        output = model(z, noise, target_ids=ids, compute_anchor=anchor)
        unchanged()
        return output

    def composition(result, row):
        calls["decoder"] += 1
        return decoder.weighted_composition(result.states[row], result.weights[row],
            gene_block=execution.gene_block, row_block=execution.decoder_row_block,
            standardized=True, recompute_backward=True)

    target_first, control_first, control_chunks, adjoint = [], [], [], None
    target_lookup = {}
    if molecular_enabled:
        target_lookup = {row: index for index, row in enumerate(molecular.target_law_indices.tolist())}
        target_first = [None] * len(target_lookup)
        with torch.no_grad():
            for start in range(0, len(controls.source), execution.control_microbatch_laws):
                stop = min(start + execution.control_microbatch_laws, len(controls.source))
                result = rollout(controls.source[start:stop], controls.noise[:, start:stop], controls.component_target_ids[start:stop], False)
                control_first.extend(composition(result, row).detach().clone() for row in range(stop - start))
                control_chunks.append((start, stop))
                del result
        control_first = torch.stack(control_first)
        pooled_first = control_first.new_zeros((len(controls.donor_ids), decoder.features)).index_add(
            0, controls.component_donor_indices, control_first * controls.mixture_weights[:, None])

    reductions = {
        "state": LossReduction(protocol.state_coefficient / protocol.state_scale, denominator if denominator > 0 else 1.),
        "deterministic_action": LossReduction(protocol.deterministic_action, 1.),
        "stochastic_action": LossReduction(protocol.stochastic_action, 1.),
        "reaction_action": LossReduction(protocol.reaction_action if joint else 0., 1.),
    }
    if shape_weight > 0:
        if denominator <= 0:
            raise ValueError("Enabled shape objective has no observed-state denominator")
        reductions["shape"] = LossReduction(shape_weight, denominator)
    if molecular_enabled:
        reductions["molecular_proxy"] = LossReduction(1., 1.)
    chunks = []
    for start in range(0, n, execution.microbatch_laws):
        stop = min(n, start + execution.microbatch_laws)
        rows = torch.arange(start, stop, device=source.device, dtype=torch.int64)
        chunks.append(LogicalMicrobatch(f"laws-{start}-{stop}", inputs.cache_indices[start:stop],
            {"source": source[start:stop], "noise": inputs.noise[:, start:stop], "rows": rows}))
    clamp_counts = [0] * n
    target_replayed = []

    def get_adjoint():
        nonlocal adjoint
        if adjoint is None:
            if any(value is None for value in target_first):
                raise RuntimeError("Incomplete molecular first pass; no target/control zero fill")
            adjoint = molecular_adjoint(torch.stack(target_first), molecular.observed_targets,
                pooled_first, controls.observed_compositions, target_donor_indices=target_donors,
                target_weights=molecular.target_weights, control_weights=controls.reduction_weights,
                target_coefficient=molecular.target_estimator_multiplier * (.5 * protocol.endpoint_coefficient / protocol.endpoint_scale),
                control_coefficient=.5 * protocol.endpoint_coefficient / protocol.endpoint_scale,
                effect_coefficient=molecular.target_estimator_multiplier * (protocol.effect_coefficient / protocol.effect_scale))
        return adjoint

    def forward(batch, *, phase, include_local_losses):
        unchanged()
        rows = batch.payload["rows"].tolist()
        ids = tuple(inputs.target_ids[row] for row in rows)
        result = rollout(batch.payload["source"], batch.payload["noise"], ids, include_local_losses)
        losses = {}
        if molecular_enabled:
            positions = [(local, target_lookup[row]) for local, row in enumerate(rows) if row in target_lookup]
            if not include_local_losses:
                for local, target_index in positions:
                    if target_first[target_index] is not None:
                        raise RuntimeError("Target first pass duplicated a logical row")
                    target_first[target_index] = composition(result, local).detach().clone()
            else:
                plan = get_adjoint()
                proxy = source.new_zeros(())
                if positions:
                    index = torch.tensor([target for _, target in positions], device=source.device, dtype=torch.int64)
                    predicted = torch.stack([composition(result, local) for local, _ in positions])
                    raw = plan.replay_proxy(role="target", indices=index, current_compositions=predicted,
                        atol=execution.replay_atol, rtol=execution.replay_rtol)
                    proxy = (raw - raw.detach()).to(source.dtype)
                    target_replayed.extend(index.tolist())
                losses["molecular_proxy"] = proxy
        if include_local_losses:
            observed = tuple(inputs.observations[row] for row in rows)
            geo = geometry(result.states, torch.log_softmax(result.log_weights, -1),
                law_ids=tuple(inputs.law_ids[row] for row in rows), observations=observed)
            losses["state"] = geo.weighted_sum(inputs.geometry_weights[batch.payload["rows"]])
            for local, row in enumerate(rows):
                clamp_counts[row] = geo.observed_weight_clamp_counts[local]
            losses.update(_action_sums(model, result, inputs.action_weights[batch.payload["rows"]]))
            if shape_weight > 0:
                value = result.states.sum() * 0
                for local, row in enumerate(rows):
                    law = inputs.observations[row]
                    if law is not None:
                        value = value + inputs.geometry_weights[row] * multiscale_mmd2(result.states[local], result.weights[local],
                            law.support, law.weights / law.weights.sum(), protocol.shape_bandwidths,
                            block=execution.shape_block, recompute_backward=True)
                losses["shape"] = value
        unchanged()
        return LocalForward(result.log_mass_increment, losses,
            {"identity": batch.identity, "laws": len(rows), "state_observations": sum(inputs.observations[row] is not None for row in rows),
             "phase": phase, "protocol_id": protocol.protocol_id})

    # Normally the existing count prepass performs this collection. Explicitly
    # molecular-only synthetic protocols still need one no-grad target pass.
    if molecular_enabled and count_weight == 0:
        with torch.no_grad():
            for chunk in chunks:
                forward(chunk, phase=model.phase, include_local_losses=False)
    result = accumulate_logical_batch(model, chunks, forward, phase=model.phase,
        local_reductions=reductions, count_coefficient=count_weight, sampling_scale=sampling_scale,
        blocks=blocks, cache=cache, global_regularizer=lambda: _global_regularizer(model, protocol),
        global_regularizer_coefficient=1., replay_atol=execution.replay_atol, replay_rtol=execution.replay_rtol)
    unchanged()
    control_replayed = []
    if molecular_enabled:
        plan = get_adjoint()
        if sorted(target_replayed) != list(range(len(target_first))):
            raise RuntimeError("Target replay did not cover each logical molecular law exactly once; do not step")
        for start, stop in control_chunks:
            current = rollout(controls.source[start:stop], controls.noise[:, start:stop], controls.component_target_ids[start:stop], False)
            q = torch.stack([composition(current, row) for row in range(stop - start)])
            if not torch.allclose(q.detach(), control_first[start:stop], atol=execution.replay_atol, rtol=execution.replay_rtol):
                raise RuntimeError("Control component replay differs from first pass; discard gradients and do not step")
            derivative = plan.control_derivatives[controls.component_donor_indices[start:stop]] * controls.mixture_weights[start:stop, None]
            proxy = ((q - q.detach()) * derivative).sum()
            proxy.backward()
            control_replayed.extend(range(start, stop))
            del current, q, proxy
            unchanged()
        if control_replayed != list(range(len(controls.source))):
            raise RuntimeError("Control replay was not exactly once; do not step")
        # Existing *_mse fields are normalized sampled-law means. The explicit
        # *_estimator fields, not those unscaled means, enter the actual loss.
        multiplier = molecular.target_estimator_multiplier
        molecular_terms = {"target_mse": plan.target_mse, "control_mse": plan.control_mse,
            "effect_mse": plan.effect_mse, "target_estimator_multiplier": multiplier,
            "target_mse_estimator": multiplier * plan.target_mse,
            "effect_mse_estimator": multiplier * plan.effect_mse}
        displayed = result.displayed_objective.double() + plan.objective
    else:
        molecular_terms, displayed = {}, result.displayed_objective
    unchanged()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise RuntimeError("Nonfinite integrated objective gradients; do not step")
    return TrainingObjectiveResult(displayed, result, None if not molecular_enabled else plan.objective,
        molecular_terms, len(available), 0 if not molecular_enabled else len(target_first),
        0 if not molecular_enabled else len(controls.donor_ids),
        0 if not molecular_enabled else len(controls.source), len(control_replayed), calls["model"], calls["decoder"],
        tuple(clamp_counts), protocol.protocol_id)
