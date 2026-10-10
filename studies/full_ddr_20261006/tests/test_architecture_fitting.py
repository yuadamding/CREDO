"""Synthetic-only fitting objective, budgets, lifecycle and no-holdout tests."""
# ruff: noqa: E402
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]

import pytest
import torch
from credo.recipes.compact_sde_v3.training import Trainer
from credo.runtime import TrainingEngine
from architecture_fitting import (
    ArchitectureFittingRecipe, POLICY, complete_count_loss, count_sampling_surrogate, fitting_normalizers, validate_normalizers,
)
import fitting_launch as builder
import fitting_operate as operator
import fitting_worker as worker
import fitting_control as controller
from test_terminal_fitting import setup_run


def normalizers():
    return dict(policy=POLICY, state=2.0, count=0.25, fitting_donors=["D3", "D4"])


def execution():
    return dict(maximum_allocated_bytes=75_000_000_000, maximum_reserved_bytes=78_000_000_000,
                minimum_live_headroom_bytes=4_000_000_000, maximum_optimizer_updates_per_fit=20)


@pytest.mark.parametrize("published,current,updates,attempts,expected", [
    (False, dict(candidate_id="ddr_mlp128_e8", seed=1), 3, 4, (13, 14)),
    (True, dict(candidate_id="ddr_mlp128_e8", seed=0), 10, 10, (10, 10)),
    (False, None, 0, 0, (10, 10)),
])
def test_failure_totals_do_not_double_count_published_current_fit(published, current, updates, attempts, expected):
    fits = [dict(candidate_id="ddr_mlp128_e8", seed=0)]
    record = worker.failure_accounting(fits, current, 10, 10, updates, attempts)
    assert record["current_fit_already_published"] is published
    assert (record["scientific_optimizer_updates"], record["attempted_optimizer_updates"]) == expected


def test_value_preserving_gradient_scale_and_partial_batch():
    x = torch.tensor(2.5, requires_grad=True)
    result = count_sampling_surrogate(x.square(), population=7, batch=2)
    assert float(result.detach()) == 6.25
    result.backward()
    assert float(x.grad) == 17.5
    y = torch.tensor(1.5, requires_grad=True)
    count_sampling_surrogate(y, population=7, batch=1).backward()
    assert float(y.grad) == 7
    with pytest.raises(ValueError):
        count_sampling_surrogate(y, population=2, batch=3)


@pytest.mark.parametrize("key,value", [("state", 0), ("count", 0), ("count", -0.5), ("state", float("nan"))])
def test_undefined_training_scale_is_not_repaired(key, value):
    raw = normalizers()
    raw[key] = value
    with pytest.raises(ValueError):
        validate_normalizers(raw)


def test_actual_fit_balanced_scales_both_stages_and_no_holdout(tmp_path, monkeypatch):
    problem, config = setup_run(tmp_path)
    events = []
    recipe = ArchitectureFittingRecipe(normalizers(), execution(), progress=events.append)
    def forbidden(*args, **kwargs):
        raise AssertionError("Outcome scorer entered fitting")
    monkeypatch.setattr(Trainer, "_evaluate_ids", forbidden)
    monkeypatch.setattr(Trainer, "_validation_count_loss", forbidden)
    trainer = TrainingEngine().fit(recipe, problem, config, device="cpu")
    state_ids = trainer._training_ids_for_stage(trainer.training_plan.stages[0])
    mass_ids = trainer._training_ids_for_stage(trainer.training_plan.stages[1])
    assert trainer.optimizer_updates == trainer.attempted_optimizer_updates == math.ceil(len(state_ids)/2)+math.ceil(len(mass_ids)/2)
    assert trainer.training_guide_exposures == len(state_ids)+len(mass_ids)
    assert trainer.metrics.empty and all(not r["validation_monitored"] for r in trainer.history_rows)
    assert trainer.objective_map["checkpoint_geometry"].weight == 0.25
    assert trainer.objective_map["grouped_count_likelihood"].weight == 2.0
    assert len([r for r in events if r["event"] == "optimizer_step_applied"]) == trainer.optimizer_updates
    assert len([r for r in events if r["event"] == "scientific_update_completed"]) == trainer.optimizer_updates
    for head in ("drift_", "sigma_", "growth_"):
        assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for name, p in trainer.model.named_parameters() if name.startswith(head))
    trainer.save()
    checkpoint = torch.load(config.output / "state/checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["execution_trace"][-1]["event"] == POLICY
    assert checkpoint["execution_trace"][-1]["optimizer_updates"] == trainer.optimizer_updates


def test_failed_post_step_guard_preserves_actual_applied_count(tmp_path, monkeypatch):
    from architecture_fitting import ArchitectureFittingTrainer
    problem, config = setup_run(tmp_path)
    events = []
    def fail(self, *, after_step=False, **kwargs):
        if after_step:
            raise RuntimeError("synthetic post-step failure")
        return {}
    monkeypatch.setattr(ArchitectureFittingTrainer, "_memory_guard", fail)
    with pytest.raises(RuntimeError, match="post-step failure"):
        TrainingEngine().fit(ArchitectureFittingRecipe(normalizers(), execution(), progress=events.append),
                             problem, config, device="cpu")
    assert events[-1]["event"] == "optimizer_step_applied"
    assert events[-1]["optimizer_updates"] == 1
    assert not config.output.exists()


def test_all_fitting_count_blocks_even_when_a_donor_has_no_active_guides(tmp_path):
    problem, config = setup_run(tmp_path)
    trainer = TrainingEngine().fit(ArchitectureFittingRecipe(normalizers(), execution()), problem, config, device="cpu")
    result = trainer._rollout_ids((trainer.data.measure_ids[0],), particles=4, seed=100,
                                 provider=trainer._provider(trainer.training_plan.stages[1]))
    from credo.recipes.compact_sde_v3.objective import _count_block_nll, integrated_fitness_curve
    curve = integrated_fitness_curve(result, mass_increment="particle_log_mass")
    manual = []
    for block in trainer.data.count_blocks:
        full = trainer.bank.fitness_for_active(time_label=block.time_label, active_indices=result.measure_indices,
                                               active_fitness=curve[-1])
        manual.append(_count_block_nll(block, full[block.measure_indices], trainer.log_count_concentration,
                                       count_likelihood="categorical_ce"))
    actual = complete_count_loss(result, trainer.data, trainer.bank, trainer.log_count_concentration)
    torch.testing.assert_close(actual, torch.stack(manual).mean(), rtol=0, atol=0)
    assert len(manual) == 2


def test_fitting_normalizer_reproducible_complete_catalog_and_no_model_calls(tmp_path):
    problem, _ = setup_run(tmp_path)
    # Validation laws cannot contribute to either common training scale.
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError("D2 outcomes entered training normalization")
    from types import SimpleNamespace
    problem = SimpleNamespace(training=problem.training, validation=Forbidden())
    first = fitting_normalizers(problem, particles=4, catalog_size=5, device="cpu")
    second = fitting_normalizers(problem, particles=4, catalog_size=5, device="cpu")
    assert first == second
    assert first["state_laws"] == 3 and first["model_forward_calls"] == first["optimizer_updates"] == 0
    assert first["fitting_donors"] == ["D3", "D4"]
    assert first["geometry_backend"] == "geomloss_tensorized"
    assert len(first["count_blocks"]) == 2 and first["state"] > 0 and first["count"] > 0


def test_missing_endpoint_guide_retained_for_count_only_update(tmp_path):
    from credo.io import RunConfig
    problem, config = setup_run(tmp_path)
    raw = config.model_dump()
    raw["recipe_config"]["training"]["measures_per_batch"] = 1
    config = RunConfig.model_validate(raw)
    events = []
    trainer = TrainingEngine().fit(ArchitectureFittingRecipe(normalizers(), execution(), progress=events.append),
                                  problem, config, device="cpu")
    count_only = [r for r in events if r.get("geometry_backend") == "not_evaluated_no_state_observations"]
    assert len(count_only) == 1 and count_only[0]["stage"] == "mass"
    assert trainer.optimizer_updates == 7 and trainer.training_guide_exposures == 7


def test_worker_entry_reaches_exact_writer_guard_without_opening_data(payload, monkeypatch):
    plan = json.loads(payload["PLAN.json"])
    monkeypatch.setattr(worker, "load_plan", lambda *args, **kwargs: (
        Path(plan["bundle_root"]), plan, hashlib.sha256(payload["PLAN.json"]).hexdigest()))
    seen = []
    monkeypatch.setattr(worker, "verify_bundle", lambda *args: seen.append(True))
    monkeypatch.setattr(worker.os, "getuid", lambda: 1)
    with pytest.raises(RuntimeError, match="writer differs"):
        worker.main([])
    assert seen == [True]


@pytest.fixture(scope="module")
def payload():
    return builder.assemble()


def test_immutable_fitting_package_exact_budget_and_roles(payload):
    plan = json.loads(payload["PLAN.json"])
    worker.validate_plan(plan, json.loads(payload["FITTING_CONTRACT.json"]))
    assert len(plan["fits"]) == 9 and plan["maximum_optimizer_updates"] == 23760
    assert "memory_qualification.py" not in payload and plan["capacity_probe_budget_remaining"] == 0
    job = json.loads(payload["gpu-job.json"])
    assert job["metadata"]["name"] == builder.RUN_ID
    assert job["spec"]["suspend"] and job["spec"]["backoffLimit"] == 0
    assert "activeDeadlineSeconds" not in job["spec"]
    assert "activeDeadlineSeconds" not in job["spec"]["template"]["spec"]
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["requests"]["memory"] == "200Gi"
    assert container["command"][-1] == builder.HOME_RUN+"/bundle/worker.py"
    assert builder.RUN_ID.encode() in payload["publication.py"]
    assert builder.capacity_builder.RUN_ID.encode() not in payload["publication.py"]
    implementation = json.loads(payload["IMPLEMENTATION.json"])
    assert implementation == {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()
                               if n.endswith(".py") and n not in {"control.py", "publication.py"}}
    ast.parse(payload["control.py"], feature_version=(3, 6))
    ast.parse(payload["worker.py"], feature_version=(3, 12))
    compile(operator.CAPACITY_AUDIT, "<capacity-audit>", "exec")


@pytest.mark.parametrize("key,value", [("maximum_fits", 10), ("D2_scores_computed", True),
                                       ("D1_endpoint_access", True), ("maximum_optimizer_updates", 24000)])
def test_fitting_scope_inflation_rejected(payload, key, value):
    plan = json.loads(payload["PLAN.json"])
    plan[key] = value
    with pytest.raises(ValueError):
        worker.validate_plan(plan)


def test_exclusive_fitting_build_does_not_relabel_capacity_receipt(tmp_path):
    result = builder.build(tmp_path / "fit")
    assert result["run_id"] == builder.RUN_ID
    saved = json.loads((tmp_path / "fit/BUILD.json").read_text())
    assert saved == result
    assert saved["schema"] == "credo.all4_full_ddr.fitting_bundle.v1"
    with pytest.raises(FileExistsError):
        builder.build(tmp_path / "fit")


@pytest.fixture
def completion_fixture(tmp_path, monkeypatch, payload):
    from types import SimpleNamespace
    from prepare_launch import canonical
    original_lstat = Path.lstat
    output, control = tmp_path / "worker-output", tmp_path / "controller/gpu"
    output.mkdir()
    control.mkdir(parents=True)
    plan = json.loads(payload["PLAN.json"])
    plan["worker_output"] = str(output)
    plan_sha = "f"*64
    pairs, updates = 1000, 920
    files = {}

    def put(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else canonical(value)
        path.write_bytes(raw)
        ref = dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        files[name] = ref
        return ref

    put("STARTUP.json", dict(status="startup_guard_passed"))
    put("COMPILED_STUDY.json", dict(fitting_state_pairs=pairs, fitting_series=46809,
        fitting_targets=12555, exact_optimizer_updates_per_fit=updates, exact_optimizer_updates_total=9*updates))
    scale = dict(normalizers(), state_laws=pairs, particles=512, seed=104729,
                 geometry_backend="geomloss_tensorized", model_forward_calls=0, optimizer_updates=0)
    norm = put("NORMALIZERS.json", scale)
    fits = []
    for item in plan["fits"]:
        parent = "fits/"+item["candidate_id"]+"/seed-"+str(item["seed"])
        checkpoint = parent+"/state/checkpoint.pt"
        ref = put(checkpoint, b"synthetic-checkpoint-identity-not-a-neural-model")
        row = dict(item, status="terminal_fitting_complete_pending_input_lease_exit", checkpoint_path=checkpoint,
            checkpoint=ref, optimizer_updates=updates, attempted_optimizer_updates=updates,
            epochs_completed=60, guide_exposures=40*pairs+20*46809, particles=512, steps=8,
            parameter_count=10, peak_allocated_bytes=100, peak_reserved_bytes=200,
            optimizer_memory_admissions=[dict(stage=stage, optimizer_state_bytes=20,
                peak_allocated_bytes=100, peak_reserved_bytes=200) for stage in ("state", "mass")],
            empty_assessment_table_rows=dict(metrics=0, predictions=0, counterfactuals=0),
            head_gradient_maxima=dict(drift=1.0, diffusion=1.0, reaction=1.0),
            normalizers_sha256=norm["sha256"], fitting_contract_sha256=plan["fitting_contract_sha256"],
            scientific_assessment="not_performed_fitting_only")
        put(parent+"/FIT_COMPLETE.json", row)
        fits.append(row)
    complete = dict(status="completed_full_ddr_architecture_fitting", run_id=plan["run_id"], plan_sha256=plan_sha,
        pod_uid="pod", implementation_sha256=plan["implementation_sha256"],
        fitting_contract_sha256=plan["fitting_contract_sha256"], capacity_binding_sha256=plan["capacity_binding_sha256"],
        input_lease_exit_verified=True, scientific_training_complete=True, scientific_promotion=False,
        query_deployment_performed=False, assessment_performed=False, scientific_acceptance="not_assessed_fitting_only",
        D1_endpoint_access=False, D2_scores_computed=False, LARRY_access=False,
        scientific_optimizer_updates=9*updates, attempted_optimizer_updates=9*updates, fits=fits,
        files=[dict(path=name, **ref) for name, ref in sorted(files.items())],
        output_bytes=sum(ref["size_bytes"] for ref in files.values()))
    (output / "COMPLETE.json").write_bytes(canonical(complete))
    (control / "startup-validated.json").write_bytes(canonical(dict(uid="job", actual_pod_audit_passed=True,
        startup_sha256=files["STARTUP.json"]["sha256"])))

    def checked(path, retain=False):
        raw = Path(path).read_bytes()
        return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)), raw if retain else b""

    def owned_lstat(path):
        info = original_lstat(path)
        return SimpleNamespace(st_mode=info.st_mode, st_uid=307469)

    monkeypatch.setattr(controller.base, "checked_file", checked)
    monkeypatch.setattr(controller.base, "core", SimpleNamespace(sha=lambda path: plan_sha), raising=False)
    monkeypatch.setattr(Path, "lstat", owned_lstat)
    return output, control, plan, complete, canonical


def test_terminal_validator_with_actual_separate_evidence_roots(completion_fixture):
    _, control, plan, _, _ = completion_fixture
    assert controller.validate_result(plan, dict(uid="job"), control, [dict(metadata=dict(uid="pod"))])


@pytest.mark.parametrize("change", ["missing_seed", "extra_updates", "assessment", "unverified_inputs",
                                    "changed_file", "wrong_control_root"])
def test_terminal_validator_rejects_partial_or_misbound_publication(completion_fixture, change):
    output, control, plan, complete, canonical = completion_fixture
    if change == "missing_seed":
        complete["fits"].pop()
    elif change == "extra_updates":
        complete["scientific_optimizer_updates"] += 1
    elif change == "assessment":
        complete["assessment_performed"] = True
    elif change == "unverified_inputs":
        complete["input_lease_exit_verified"] = False
    elif change == "changed_file":
        (output / "NORMALIZERS.json").write_bytes(b"{}\n")
    elif change == "wrong_control_root":
        control = output
    (output / "COMPLETE.json").write_bytes(canonical(complete))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        controller.validate_result(plan, dict(uid="job"), control, [dict(metadata=dict(uid="pod"))])
