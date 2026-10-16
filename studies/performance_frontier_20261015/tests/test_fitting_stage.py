"""Small actual CPU fitting stages using fabricated native/observation caches.

At most24 Adam invocations per invocation of this file; zero scientific steps.
Saved normalizers are scalar fixtures from the existing factory tests. Numerical
normalization has its separate tested join; this suite tests stage ownership.
"""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from frontier import fitting_stage as module
from frontier.cache_storage import load_fold_cache
from frontier.checkpoints import TrajectoryLease
from frontier.fitting_factory import prepare_fitting_components
from frontier.fitting_stage import FittingStage, task_boundaries
from frontier.planner import (DONORS, TrajectoryKey, TrainingTask, family_plan,
                              finalist_plan, outer_plan, screen_plan)
from frontier.prediction_plan import required_before_continuation
from frontier.retention import STUDY_ID
from frontier.training_owner import OwnerSchedule
from test_fitting_factory import package as factory_fixture


ORIGINAL_STEP = torch.optim.Adam.step
TOTAL_CALLS = 0


@pytest.fixture
def package(tmp_path, monkeypatch, request):
    generator = factory_fixture.__wrapped__(tmp_path/"fixture", monkeypatch)
    args, _ = next(generator)
    calls = []
    def counted(optimizer, *a, **kw):
        global TOTAL_CALLS
        TOTAL_CALLS += 1
        if TOTAL_CALLS > 24:
            raise RuntimeError("Local fitting-stage disposable24-step cap exceeded")
        calls.append(1)
        return ORIGINAL_STEP(optimizer, *a, **kw)
    monkeypatch.setattr(torch.optim.Adam, "step", counted)
    study = tmp_path/STUDY_ID; study.mkdir()
    (study/"trajectories").mkdir()
    args = dict(args, schedule=OwnerSchedule("synthetic_cpu_only", (2, 4, 6), 1, 2, 32, 1, 2, 4, 4),
                maximum_checkpoint_bytes=64*1024**2)
    cache = args["cache"]
    publication = dict(cache.publication_reference)
    try:
        yield dict(args=args, generator=generator, study=study, calls=calls,
                   cache_limits=cache.limits, publication=publication)
    finally:
        generator.close()
        request.node.user_properties.append(("disposable_cpu_optimizer_calls", len(calls)))


def plans():
    first = screen_plan()
    second = family_plan({d: tuple(f"F{i}-V1" for i in range(6)) for d in DONORS})
    third = finalist_plan(second, {d: ("F0-V1", "F1-V1", "F4-V1") for d in DONORS})
    return first, second, third


def task(plan, config="F0-V1"):
    return next(t for t in plan.training if t.key == TrajectoryKey(("D3", "D4"), config, 0))


def prepared(package, task, cache=None):
    args = dict(package["args"], configuration_id=task.key.configuration_id,
        model_seed=task.key.seed, trajectory_id=task.key.storage_identifier)
    if cache is not None: args["cache"] = cache
    return prepare_fitting_components(**args)


def lifecycle(package, stack, plan, t, *, cache=None, resume=None, previous=None, edges=()):
    p = prepared(package, t, cache)
    lease = stack.enter_context(TrajectoryLease(package["study"]/"trajectories"/t.key.storage_identifier,
                                               p.identity, "fabricated-stage"))
    return stack.enter_context(FittingStage(prepared=p, lease=lease, plan=plan, task=t,
        study_root=package["study"], resume=resume, previous_stage=previous,
        verified_complete_edge_ids=edges))


def reopen(package):
    ref = package["publication"]
    return load_fold_cache(ref["root"], expected_manifest_sha256=ref["manifest_sha256"],
        expected_complete_sha256=ref["complete_sha256"], output_cap_bytes=1024**2,
        limits=package["cache_limits"])


def test_real_fixture_stage_resume_and_rolling_retention(package):
    first, second, third = plans()
    with ExitStack() as stack:
        stage = lifecycle(package, stack, first, task(first))
        stage.run()
        package["generator"].close()  # Complete input exit BEFORE stage/lease exit.
    assert stage.receipt and stage.owner.accepted_updates == 2
    for plan, prior in ((second, first), (third, second)):
        resume = stage.owner.last_checkpoint
        t = task(plan)
        with ExitStack() as stack:
            with reopen(package) as cache:
                stage = lifecycle(package, stack, plan, t, cache=cache, resume=resume, previous=prior,
                    edges=required_before_continuation(prior, t))
                stage.run()
        assert stage.receipt and stage.owner.accepted_updates == stage.targets[-1]
    assert len(package["calls"]) == 6
    assert stage.owner.joint_entries == 1 and stage.owner.refresh_updates == [1, 3, 5]
    directory = stage.lease.root
    present = [n for n in range(1, 7) if (directory/f"checkpoints/update-{n:09d}/checkpoint.pt").exists()]
    assert present == [2, 4, 5, 6]
    assert all((directory/f"checkpoints/update-{n:09d}/COMPLETE.json").exists() for n in range(1, 7))
    retired = list((directory/"retention/transactions").glob("*/DELETED.json"))
    assert len(retired) == 2
    for name in ("screen", "family", "finalist"):
        record = json.loads((directory/f"FITTING_STAGE_{name}.json").read_text())
        assert record["scientific_updates_this_task"] == 0
        assert record["optimizer_calls_this_task"] == 2
        assert record["checkpoint"]["maximum_checkpoint_bytes"] == 64*1024**2
        assert record["cache_verification"]["exit_full_hash_verified"] is True
        assert record["query_outcomes_opened"] is False
        assert record["prior_forecast_files_verified_by_this_component"] is False
    assert not list(directory.glob("*_FAILED.json"))


def test_ddr_molecular_fixture_stage_uses_same_lifecycle(package):
    first = screen_plan()
    with ExitStack() as stack:
        stage = lifecycle(package, stack, first, task(first, "F4-V1"))
        stage.run()
        package["generator"].close()
    assert len(package["calls"]) == 2 and stage.receipt
    assert stage.owner.last_metrics["phase"] == "joint"
    assert stage.owner.last_metrics["molecular_terms"]


def test_final_cache_failure_marks_numerically_complete_owner_failed(package):
    first = screen_plan()
    with pytest.raises(ValueError):
        with ExitStack() as stack:
            stage = lifecycle(package, stack, first, task(first))
            stage.run()
            (Path(package["publication"]["root"])/"p000.states.npy").touch()
            package["generator"].close()
    assert len(package["calls"]) == 2
    assert stage.owner.status == "failed" and stage.owner.accepted_updates == 2
    assert stage.failure_path.exists() and not stage.path.exists()
    assert json.loads(stage.owner.journal.read_text())["failure"]["stage_lifecycle_failure"] is True


def test_completion_publication_failure_invalidates_owner_without_replay(package, monkeypatch):
    first = screen_plan()
    original = module._write_once
    def changed(fd, name, *a, **kw):
        result = original(fd, name, *a, **kw)
        if name == "FITTING_STAGE_screen.json": raise OSError("fabricated final readback failure")
        return result
    monkeypatch.setattr(module, "_write_once", changed)
    with pytest.raises(OSError, match="fabricated"):
        with ExitStack() as stack:
            stage = lifecycle(package, stack, first, task(first))
            stage.run()
            package["generator"].close()
    assert len(package["calls"]) == 2 and stage.owner.status == "failed"
    assert stage.path.exists() and stage.failure_path.exists()
    assert stage.receipt is None


def test_retirement_failure_stops_owner_and_does_not_delete_invented_targets(package, monkeypatch):
    first, second, _ = plans()
    with ExitStack() as stack:
        stage = lifecycle(package, stack, first, task(first))
        stage.run(); package["generator"].close()
    resume = stage.owner.last_checkpoint
    def fail(*_a, **_kw): raise OSError("fabricated retention failure")
    monkeypatch.setattr("frontier.retention.commit_retirement", fail)
    t = task(second)
    with pytest.raises(OSError, match="retention failure"):
        with ExitStack() as stack:
            with reopen(package) as cache:
                stage = lifecycle(package, stack, second, t, cache=cache, resume=resume,
                    previous=first, edges=required_before_continuation(first, t))
                stage.run()
    assert len(package["calls"]) == 3
    assert stage.owner.accepted_updates == 3 and stage.owner.status == "failed"
    assert stage.owner.failure["preceding_numerical_failure"]["error_type"] == "OSError"
    assert not stage.path.exists() and stage.failure_path.exists()
    assert (stage.lease.root/"checkpoints/update-000000001/checkpoint.pt").exists()
    assert list((stage.lease.root/"retention/transactions").glob("*/PREPARED.json"))
    assert not list((stage.lease.root/"retention/transactions").glob("*/DELETED.json"))


def test_no_run_or_wrong_context_order_cannot_publish_success(package):
    first = screen_plan()
    with pytest.raises(ValueError, match="cache exit"):
        with ExitStack() as stack:
            stage = lifecycle(package, stack, first, task(first))
            # Deliberately neither run nor close the cache before stage exit.
    assert len(package["calls"]) == 0 and stage.owner.status == "failed"
    assert not stage.path.exists()


@pytest.mark.parametrize("kind", ["seed", "trajectory", "task", "resume", "missing_forecasts"])
def test_wrong_task_or_missing_continuation_rejected_without_steps(package, kind):
    first, second, _ = plans()
    plan = second if kind == "missing_forecasts" else first
    t = task(plan)
    p = prepared(package, t)
    if kind == "seed": p.identity["seed"] = 1
    if kind == "trajectory": p.identity["trajectory_id"] = "not-the-task"
    if kind == "task": t = task(first, "F1-V1")
    resume = {"accepted_updates": 2} if kind in ("resume", "missing_forecasts") else None
    with TrajectoryLease(package["study"]/"trajectories"/"wrong", p.identity, "wrong") as lease:
        with pytest.raises(ValueError):
            FittingStage(prepared=p, lease=lease, plan=plan, task=t, study_root=package["study"],
                resume=resume, previous_stage=first if kind == "missing_forecasts" else None)
    assert len(package["calls"]) == 0


def test_scientific_and_outer_boundaries_are_exact_not_fixture_training_claims():
    first, second, third = plans()
    for plan in (first, second, third):
        t = task(plan)
        start, stops = task_boundaries(t, OwnerSchedule())
        assert start == t.start_updates and stops[-1] == t.target_updates
    outer = outer_plan(third)
    assert task_boundaries(outer.training[0], OwnerSchedule()) == (0, (1024, 3072, 9216))
    tiny = OwnerSchedule("synthetic_cpu_only", (2, 4, 6), 1, 2, 32, 1, 2, 4, 4)
    assert task_boundaries(outer.training[0], tiny) == (0, (2, 4, 6))
    with pytest.raises(ValueError, match="boundary"):
        task_boundaries(task(first), replace(tiny, stages=(2, 4)))


def test_scientific_continuation_rejects_bare_edge_ids_before_owner_open(package,monkeypatch):
    first,second,_=plans()
    t=task(second);p=prepared(package,t)
    # A negative admission test only: change schedule metadata on fabricated
    # prepared components, and require rejection before any owner/optimizer.
    monkeypatch.setattr(p,"arguments",dict(p.arguments,schedule=OwnerSchedule()))
    monkeypatch.setattr(p,"open_owner",lambda *_a,**_k:pytest.fail("No owner may open"))
    with TrajectoryLease(package["study"]/"trajectories"/t.key.storage_identifier,p.identity,"bare-ids") as lease:
        with pytest.raises(ValueError,match="actual verified forecast-file custody"):
            FittingStage(prepared=p,lease=lease,plan=second,task=t,study_root=package["study"],
                previous_stage=first,resume=dict(accepted_updates=1024,checkpoint_sha256="a"*64),
                verified_complete_edge_ids=required_before_continuation(first,t))
    assert package["calls"]==[]
