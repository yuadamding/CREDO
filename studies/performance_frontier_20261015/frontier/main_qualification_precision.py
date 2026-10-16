"""Compact paired objective/gradient checks for a future admitted CUDA run.

No CLI, optimizer, scheduler, biological input, or qualification release exists
here. The same CPU-generated tensors and initial weights are copied to CUDA;
device-specific random-number streams are NOT treated as paired inputs. This
checks arithmetic and accumulation, not full-shape memory or predictive quality.
The separate owner probe retains the full biological-axis dimensions.
"""

import copy
from dataclasses import asdict, fields, is_dataclass

import torch

from .coordination import _sha
from .count_objective import FitnessCache
from .geometry import GeometryOptions, RaggedSinkhornGeometry
from .main_qualification_fixture import FixtureDimensions, QualificationProvider
from .main_qualification_plan import QualificationCase, cases
from .models import build_model
from .objective_registry import ObjectiveRules, FittingScaleBinding, build_objective_protocol
from .observation import ObservationFold, SparseR48
from .observation_worker import _cpu_tree, _exact_tree, _fp32_comparison
from .retention import identity_sha256
from .training_objective import ObjectiveExecution, accumulate_training_objective


# Prospective, immutable thresholds; no outcome-dependent relaxation. Their
# numeric values match the existing FP32 check, but its pass is NOT reused.
TOLERANCES = dict(loss_atol=2e-5, loss_rtol=2e-5,
                  gradient_atol=2e-5, gradient_rtol=2e-4)
PHASES = ("state", "joint")
MICROBATCHES = (1, 4)


def precision_dimensions():
    return FixtureDimensions("synthetic_cpu_only", 31, 113, 8, 16, 8, 16, (1, 3, 7, 16))


def precision_plan():
    return dict(schema="credo.frontier.main-paired-precision-plan.v1",
        status="proposal_not_launch_authority", input_dimensions=asdict(precision_dimensions()),
        cases=[dict(configuration_id=c.configuration_id, training_donors=c.training_donors,
                    phase=phase) for c in cases() for phase in PHASES],
        evaluations_per_case=["cpu_microbatch_1", "cuda_microbatch_1", "cuda_microbatch_4"],
        planned_objective_backward_evaluations=288, optimizer_calls=0, scientific_updates=0,
        tolerances=dict(TOLERANCES), cpu_reference_threads=2, seed=419,
        same_inputs_and_initial_weights=True, real_biological_payload_access=False,
        full_worker_entry_verified=False, full_shape_memory_qualified=False,
        main_worker_qualification_pass=False, automatic_retry=False)


def _move(value, device):
    """Copy the complete typed fixture, including FP64 truth and zero weights."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if is_dataclass(value):
        return type(value)(**{f.name: _move(getattr(value, f.name), device) for f in fields(value)})
    if type(value) is tuple:
        return tuple(_move(v, device) for v in value)
    if type(value) is dict:
        return {k: _move(v, device) for k, v in value.items()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("Unregistered fabricated-input field")


def build_precision_fixture(case, phase, *, rules, source_sha256):
    """Small CPU-only fixture; no sampling position represents a fitted update."""
    if type(case) is not QualificationCase or phase not in PHASES or type(rules) is not ObjectiveRules:
        raise ValueError("Exact registered precision case, phase and objective rules required")
    case.__post_init__()
    _sha(source_sha256)
    if torch.get_num_threads() > 2:
        raise ValueError("Compact CPU reference requires at most two threads")
    if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled("cpu"):
        raise ValueError("Explicit FP32 construction without CPU autocast required")
    dims = precision_dimensions()
    provider = QualificationProvider(case, dims, device="cpu")
    # Position 1 selects a distinct deterministic fixture. No optimizer exists.
    if phase == "joint":
        provider.accepted(accepted_updates=1)
    batch = provider.sample(accepted_updates=provider.position, phase=phase)
    blocks, cache = provider.count_inputs()
    cache.apply_full_refresh(.03 * torch.sin(torch.arange(len(cache.values)).float()),
        update_index=provider.position, particles=32)
    private = {} if case.configuration_id[:2] not in ("F3", "F5") else dict(
        target_ids=provider.target_ids, support_factors=[(i % 4)/3 for i in range(len(provider.target_ids))])
    model = build_model(case.configuration_id, seed=419, **private)
    model.set_phase(phase)
    decoder = SparseR48(ObservationFold("fabricated-precision", case.training_donors,
        identity_sha256(["fabricated-precision-axis", dims.genes])), features=dims.genes,
        hidden_dims=(512, 128), maximum_rows=8, seed=419)
    decoder.scaling_ready.fill_(True)
    decoder.freeze()
    scales = FittingScaleBinding(fitting_donors=case.training_donors, rules_sha256=rules.identity,
        observation_artifact_sha256=provider.inventory.observation_sha256,
        fitting_input_sha256=provider.inventory.input_sha256,
        support_inventory_sha256=provider.inventory.manifest_sha256,
        scale_artifact_sha256=identity_sha256(["fabricated-precision-unit-scales", provider.input_identity]),
        producer_sha256=source_sha256, sampler_sha256=provider.inventory.sampler_sha256,
        values={k: 1. for k in ("state", "count", "shape", "endpoint", "effect")})
    protocol = build_objective_protocol(case.configuration_id, rules=rules, scales=scales,
        fitting_donors=case.training_donors,
        observation_artifact_sha256=provider.inventory.observation_sha256,
        fitting_input_sha256=provider.inventory.input_sha256)
    return dict(model=model, decoder=decoder, batch=batch, blocks=blocks, cache=cache,
                protocol=protocol, input_identity=provider.input_identity)


def evaluate_precision_fixture(fixture, *, device, microbatch_laws, geometry_options):
    """One actual forward/backward, with copied parameters and zero updates.

    CPU use is only the compact synthetic reference. The CUDA caller must be
    independently admitted. This function neither grants nor asserts admission.
    """
    device = torch.device(device)
    if (device not in (torch.device("cpu"), torch.device("cuda:0"))
            or microbatch_laws not in MICROBATCHES or type(microbatch_laws) is not int
            or type(geometry_options) is not GeometryOptions
            or fixture["batch"].inputs.source.shape != (16, 8, 48)):
        raise ValueError("Exact compact precision shape/device/microbatch required")
    if torch.is_autocast_enabled(device.type):
        raise ValueError("Paired precision must not silently use autocast")
    model = copy.deepcopy(fixture["model"]).to(device)
    decoder = copy.deepcopy(fixture["decoder"]).to(device)
    before, decoder_before = _cpu_tree(model.state_dict()), _cpu_tree(decoder.state_dict())
    if (not _exact_tree(before, fixture["model"].state_dict())
            or not _exact_tree(decoder_before, fixture["decoder"].state_dict())):
        raise ValueError("Paired initialization changed during device transfer")
    batch, blocks = _move(fixture["batch"], device), _move(fixture["blocks"], device)
    original = fixture["cache"]
    cache = FitnessCache.from_state_dict(original.state_dict(), expected_binding=original.binding,
        expected_size=len(original.values), expected_dtype=original.values.dtype, device=device)
    cache_before = cache.state_dict()
    execution = ObjectiveExecution(microbatch_laws, microbatch_laws, 7, 4, 8, 100000, 3e-6, 3e-6)
    result = accumulate_training_objective(model, batch.inputs, protocol=fixture["protocol"],
        execution=execution, geometry=RaggedSinkhornGeometry(options=geometry_options),
        blocks=blocks, cache=cache, sampling_scale=batch.sampling_scale,
        decoder=decoder, molecular=batch.molecular)
    if (result.optimizer_calls != 0 or result.cache_writes != 0
            or not _exact_tree(before, _cpu_tree(model.state_dict()))
            or not _exact_tree(decoder_before, _cpu_tree(decoder.state_dict()))
            or not _exact_tree(cache_before, cache.state_dict())
            or any(p.grad is not None for p in decoder.parameters())):
        raise RuntimeError("Precision evaluation mutated parameters/cache or differentiated the frozen decoder")
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in model.named_parameters()}
    present = [g for g in gradients.values() if g is not None]
    if not present or any(not bool(torch.isfinite(g).all()) for g in present) or not any(bool(g.any()) for g in present):
        raise ValueError("Actual finite nontrivial model gradients required")
    logical = result.logical_result
    values = {"objective": result.displayed_objective, "global_regularizer": logical.global_regularizer,
              "active_log_mass": logical.pre_update_log_mass}
    for label, mapping in (("local", logical.local_loss_sums), ("weighted", logical.local_weighted_terms),
                           ("molecular", result.molecular_terms)):
        values.update({label+"/"+k: v for k, v in mapping.items()})
    for key in ("count_ce", "donor_count_ce", "count_gradient_coefficients"):
        if getattr(logical, key) is not None:
            values[key] = getattr(logical, key)
    values = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor)
                  else torch.tensor(v, dtype=torch.float64)) for k, v in values.items()}
    if any(not bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError("Nonfinite reported precision objective/channel")
    return dict(values=values, gradients=gradients, requires_grad={k: p.requires_grad for k, p in model.named_parameters()},
        parameter_count=sum(p.numel() for p in model.parameters()),
        active_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        active_indices=logical.active_indices.detach().cpu().clone(),
        coverage={k: getattr(result, k) for k in ("state_observations", "target_molecular_observations",
            "control_molecular_observations", "control_component_first_passes", "control_component_replays",
            "observed_weight_clamp_counts")}, optimizer_calls=0, cache_writes=0,
        execution=asdict(execution), same_initial_weights=True)


def compare_evaluations(reference, actual):
    """All channels and parameters are compared; absent gradients stay visible."""
    for key in ("coverage", "requires_grad", "parameter_count", "active_parameter_count"):
        if reference[key] != actual[key]:
            raise ValueError("Paired identity/coverage differs: " + key)
    if not torch.equal(reference["active_indices"], actual["active_indices"]):
        raise ValueError("Paired catalog routing differs")
    if reference["values"].keys() != actual["values"].keys() or reference["gradients"].keys() != actual["gradients"].keys():
        raise ValueError("Paired objective/parameter inventory differs")
    values = {name: _fp32_comparison(value, actual["values"][name],
        atol=TOLERANCES["loss_atol"], rtol=TOLERANCES["loss_rtol"])
        for name, value in reference["values"].items()}
    gradients, absent = {}, []
    for name, value in reference["gradients"].items():
        other = actual["gradients"][name]
        if (value is None) != (other is None):
            raise ValueError("Paired gradient participation differs: " + name)
        if value is None:
            absent.append(name)
        else:
            gradients[name] = _fp32_comparison(value, other,
                atol=TOLERANCES["gradient_atol"], rtol=TOLERANCES["gradient_rtol"])
    passed = all(v["passed"] for v in (*values.values(), *gradients.values()))
    return dict(status="passed" if passed else "failed", channels=values, parameter_gradients=gradients,
        absent_gradients=absent, all_parameter_gradients_checked=True,
        compared_gradient_elements=sum(v["elements"] for v in gradients.values()),
        maximum_gradient_absolute_error=max(v["maximum_absolute_error"] for v in gradients.values()),
        maximum_channel_absolute_error=max(v["maximum_absolute_error"] for v in values.values()),
        maximum_tolerance_normalized_error=max(v["maximum_tolerance_normalized_error"]
                                             for v in (*values.values(), *gradients.values())))


def run_paired_precision_case(case, phase, *, rules, geometry_options, source_sha256):
    """Future CUDA producer only; no pass can be obtained from CPU replay alone."""
    if not torch.cuda.is_available():
        raise RuntimeError("Paired main precision requires an admitted actual H100; no CPU fallback")
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0)
            or torch.cuda.get_device_name(0) != "NVIDIA H100 80GB HBM3"
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("Bound deterministic FP32 single-H100 configuration required")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        fixture = build_precision_fixture(case, phase, rules=rules, source_sha256=source_sha256)
        cpu = evaluate_precision_fixture(fixture, device="cpu", microbatch_laws=1, geometry_options=geometry_options)
    finally:
        torch.set_num_threads(previous_threads)
    cuda_one = evaluate_precision_fixture(fixture, device="cuda:0", microbatch_laws=1, geometry_options=geometry_options)
    cuda_four = evaluate_precision_fixture(fixture, device="cuda:0", microbatch_laws=4, geometry_options=geometry_options)
    paired, accumulated = compare_evaluations(cpu, cuda_one), compare_evaluations(cuda_one, cuda_four)
    return dict(schema="credo.frontier.main-paired-precision-case.v1",
        status="passed" if paired["status"] == accumulated["status"] == "passed" else "failed",
        case=asdict(case), phase=phase, input_identity=fixture["input_identity"],
        source_sha256=source_sha256, rules_sha256=rules.identity,
        precision_plan_sha256=identity_sha256(precision_plan()), geometry=asdict(geometry_options),
        input_dimensions=asdict(precision_dimensions()), tolerances=dict(TOLERANCES),
        cpu_cuda=paired, cuda_microbatch_accumulation=accumulated,
        parameter_count=cpu["parameter_count"], active_parameter_count=cpu["active_parameter_count"],
        coverage=cpu["coverage"], optimizer_calls=0, scientific_updates=0,
        same_inputs_and_initial_weights=True, real_biological_payload_access=False,
        full_worker_entry_verified=False, full_shape_memory_qualified=False,
        main_worker_qualification_pass=False)
