"""Real CUDA owner-case producer for a separately admitted qualification.

No CLI, scheduler, source/authority admission or full-worker pass is provided.
The outer qualification controller must admit the exact plan, count all actual
calls including failed attempts, enforce its aggregate 16-GiB budget, and
publish failures. These owner results alone cannot release scientific fitting.
"""

from dataclasses import asdict
import copy
import gc
import random
import time

import numpy as np
import torch

from .checkpoints import TrajectoryLease
from .coordination import _sha
from .configuration import get_configuration
from .fitting_resources import FittingResourceGuard, FittingResourcePolicy, check_process_resources, output_usage
from .geometry import RaggedSinkhornGeometry, GeometryOptions
from .main_qualification_fixture import FixtureDimensions, QualificationProvider
from .main_qualification_plan import (QualificationCase, SCOPE, MAXIMUM_OPTIMIZER_CALLS,
                                      plan_identity, qualification_schedule)
from .models import build_model
from .objective_registry import ObjectiveRules, FittingScaleBinding, build_objective_protocol
from .observation import ObservationFold, SparseR48
from .observation_worker import _atomic_json, _cpu_tree, _exact_tree
from .retention import STUDY_ID, identity_sha256, _path
from .training_objective import ObjectiveExecution
from .training_owner import TrainingOwner, ObservationBinding, observation_state_digest, owner_binding


MIB, GIB = 1024**2, 1024**3


class DisposableBudget:
    """In-process accounting only, not a durable launch/resource reservation."""
    def __init__(self):
        self.calls = 0
        self.failed = False
        self.completed_cases = set()
        self.started_cases = set()

    def admit(self, case):
        if (type(case) is not QualificationCase or self.failed or case.identifier in self.started_cases
                or self.calls + case.optimizer_calls > MAXIMUM_OPTIMIZER_CALLS):
            raise ValueError("Case already attempted, failed or outside the finite disposable budget")
        self.started_cases.add(case.identifier)


def _seed():
    random.seed(419); np.random.seed(419); torch.manual_seed(419); torch.cuda.manual_seed_all(419)


def _components(case, rules, execution, geometry, source_sha256, identity_label):
    _seed()
    dims = FixtureDimensions()
    provider = QualificationProvider(case, dims, device="cuda:0")
    config = get_configuration(case.configuration_id)
    private = {} if not config.target_residual else dict(target_ids=provider.target_ids,
        support_factors=[(i % 4)/3 for i in range(len(provider.target_ids))])
    model = build_model(case.configuration_id, seed=419, **private).to("cuda:0")
    fold = ObservationFold("fabricated-"+"-".join(case.training_donors), case.training_donors,
        identity_sha256(["fabricated-axis", dims.genes]))
    decoder = SparseR48(fold, seed=419)
    decoder.scaling_ready.fill_(True)
    decoder.freeze(); decoder.to("cuda:0")
    observation = ObservationBinding(provider.inventory.observation_sha256, observation_state_digest(decoder))
    scales = FittingScaleBinding(fitting_donors=case.training_donors, rules_sha256=rules.identity,
        observation_artifact_sha256=provider.inventory.observation_sha256,
        fitting_input_sha256=provider.inventory.input_sha256,
        support_inventory_sha256=provider.inventory.manifest_sha256,
        scale_artifact_sha256=identity_sha256(["fabricated-unit-scales", provider.input_identity]),
        producer_sha256=source_sha256, sampler_sha256=provider.inventory.sampler_sha256,
        values={name: 1. for name in ("state", "count", "shape", "endpoint", "effect")})
    protocol = build_objective_protocol(case.configuration_id, rules=rules, scales=scales,
        fitting_donors=case.training_donors, observation_artifact_sha256=observation.artifact_sha256,
        fitting_input_sha256=provider.inventory.input_sha256)
    blocks, cache = provider.count_inputs()
    arguments = dict(model=model, provider=provider, protocol=protocol, execution=execution, geometry=geometry,
        blocks=blocks, cache=cache, schedule=qualification_schedule(), decoder=decoder,
        observation_binding=observation, maximum_checkpoint_bytes=64*MIB)
    bound = owner_binding(model=model, protocol=protocol, execution=execution, geometry=geometry,
        inventory=provider.inventory, schedule=arguments["schedule"], decoder=decoder,
        observation_binding=observation, maximum_checkpoint_bytes=64*MIB)
    identity = dict(trajectory_id=identity_label, fold_id=fold.fold_id, seed=419,
        input_sha256=provider.inventory.input_sha256, configuration_sha256=bound["configuration_sha256"],
        source_sha256=source_sha256, training_owner=bound)
    return identity, arguments


def _snapshot(owner):
    # Resource guard charges checkpoint-like CPU copies before taking them.
    owner.resource_guard.check("before_checkpoint")
    return dict(model=_cpu_tree(owner.model.state_dict()), optimizer=_cpu_tree(owner.optimizer.state_dict()),
        cache=owner.cache.state_dict(), provider=copy.deepcopy(owner.provider.state_dict()),
        counters=(owner.accepted_updates, owner.optimizer_calls, owner.joint_entries),
        metrics=copy.deepcopy(owner.last_metrics), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        cpu_rng=torch.get_rng_state().clone(), cuda_rng=torch.cuda.get_rng_state(0).cpu())


def _same_snapshot(left, right):
    # NumPy state contains an ndarray, unlike the checkpoint tensor tree helper.
    a, b = left["numpy_rng"], right["numpy_rng"]
    same_numpy = (a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:])
    return same_numpy and _exact_tree({k:v for k,v in left.items() if k != "numpy_rng"},
                                    {k:v for k,v in right.items() if k != "numpy_rng"})


def run_owner_case(case, root, *, budget, rules, execution, geometry_options, source_sha256):
    """Perform one pre-admitted full-shape case; failures latch this batch.

    This entry deliberately cannot call the scientific worker or manufacture
    its qualification receipt. A caller must provide trusted launch authority.
    """
    if (type(case) is not QualificationCase or type(budget) is not DisposableBudget or type(rules) is not ObjectiveRules
            or type(execution) is not ObjectiveExecution or type(geometry_options) is not GeometryOptions):
        raise TypeError("Exact admitted qualification components required")
    _sha(source_sha256)
    root = _path(root)
    if STUDY_ID not in root.parts or root.name != case.identifier or not root.parent.is_dir() or root.exists():
        raise ValueError("Prepared absent exact qualification-case directory required")
    policy = FittingResourcePolicy(SCOPE, 256*MIB, 64*MIB, 256*MIB, 8*MIB, 200*GIB, 8*GIB)
    check_process_resources(policy, "cuda:0")
    budget.admit(case)
    calls_before = budget.calls
    root.mkdir(mode=0o700)
    timings, records = [], []
    def trajectory(label, stops, resume=None, capture=False):
        identity, args = _components(case, rules, execution, geometry, source_sha256, label)
        owner = None
        start = 0 if resume is None else resume["accepted_updates"]
        try:
            with TrajectoryLease(root/label, identity, label+"-worker-"+str(start)) as lease:
                guard = FittingResourceGuard(lease, policy)
                owner = TrainingOwner(lease=lease, resource_guard=guard, resume=resume, **args)
                initial_model = _cpu_tree(owner.model.state_dict())
                for stop in stops:
                    torch.cuda.synchronize()
                    began = time.monotonic()
                    owner.run_stage(stop)
                    torch.cuda.synchronize()
                    timings.append(dict(label=label, accepted_target=stop, elapsed_seconds=time.monotonic()-began))
                if (_exact_tree(initial_model, _cpu_tree(owner.model.state_dict()))
                        or not owner.last_metrics["gradient_norm_before_clip"] > 0
                        or owner.last_metrics["phase"] != "joint"
                        or owner.last_metrics["count_ce"] is None):
                    raise ValueError("Disposable main case did not apply a finite nontrivial joint update")
                del initial_model
                checkpoint = copy.deepcopy(owner.last_checkpoint)
                state = owner._state()
                records.append(dict(label=label, accepted=owner.accepted_updates,
                    scientific_updates=state["scientific_updates"], disposable_calls=owner.optimizer_calls-start,
                    checkpoint=checkpoint, resources=guard.record(), last_metrics=owner.last_metrics,
                    active_parameters=sum(p.numel() for p in owner.model.parameters() if p.requires_grad)))
                saved = _snapshot(owner) if capture else None
            return checkpoint, saved
        finally:
            if owner is not None:
                budget.calls += owner.optimizer_calls-start
            del owner, args
            gc.collect()
            torch.cuda.empty_cache()
    try:
        geometry = RaggedSinkhornGeometry(options=geometry_options)
        checkpoint, _ = trajectory("primary", (2,))
        if case.restart_comparison:
            _, resumed = trajectory("primary", (3,), checkpoint, capture=True)
            _, continuous = trajectory("continuous-reference", (2, 3), capture=True)
            if not _same_snapshot(resumed, continuous):
                raise ValueError("CUDA checkpoint continuation differs from fresh continuous reference")
            del resumed, continuous
        if budget.calls-calls_before != case.optimizer_calls or budget.calls > MAXIMUM_OPTIMIZER_CALLS:
            raise ValueError("Actual disposable calls differ from the finite case plan")
        for row in records:
            if row["scientific_updates"] != 0:
                raise ValueError("Synthetic CUDA work was mislabelled scientific")
        result = dict(schema="credo.frontier.main-cuda-owner-case.v1", status="owner_case_passed",
            case=asdict(case), plan_sha256=plan_identity(), source_sha256=source_sha256,
            input_dimensions=asdict(FixtureDimensions()), execution=asdict(execution), geometry=asdict(geometry_options),
            optimizer_calls=budget.calls-calls_before, scientific_updates=0,
            real_biological_payload_access=False, restart_verified=case.restart_comparison,
            records=records, timings=timings,
            timing_scope="Owner stages include qualification checkpoint cadence; not steady-state scientific throughput",
            output=output_usage(root),
            full_worker_entry_verified=False, main_worker_qualification_pass=False,
            automatic_retry=False, automatic_successor=False)
        _atomic_json(root, "OWNER_CASE_COMPLETE.json", result)
        budget.completed_cases.add(case.identifier)
        return result
    except BaseException as error:
        budget.failed = True
        _atomic_json(root, "OWNER_CASE_FAILED.json", dict(exception_type=type(error).__name__,
            actual_disposable_optimizer_calls=budget.calls-calls_before, scientific_updates=0,
            automatic_retry=False, automatic_successor=False))
        raise
