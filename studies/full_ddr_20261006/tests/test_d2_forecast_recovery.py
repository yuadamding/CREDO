"""Synthetic metadata tests only: no cohort arrays, CUDA or remote activity."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import d2_forecast_control as controller
import d2_forecast_worker as worker
from d2_forecast_reuse import SOURCE_KEYS


def qualification(candidate="ddr_res256_e8", seed=0):
    trunk = "mlp" if candidate == "ddr_mlp128_e8" else "residual_mlp"
    return dict(status="passed", production_qualified=True, agreement_passed=True,
        scope="source_only_scalar_vs_production_not_endpoint_precision", atol=2e-5, rtol=1e-5,
        guides=16, compared_guides=16, particles=8192, steps=8, seed=seed, parameter_updates=0,
        endpoint_observations_read=False,
        numerical_policy=dict(policy="architecture_fixed_v4_mlp_batch16_residual_single", trunk_kind=trunk,
            source_chunk_maximum_guides=16, dynamics_maximum_guides=16 if trunk == "mlp" else 1,
            precision="strict_fp32", automatic_fallback=False, diagnostic_only=False))


def test_failure_report_published_before_gate_propagates(tmp_path):
    report = dict(status="failed", fields={"state": {"maximum_absolute_difference": 0.01}})
    class Disagreement(RuntimeError):
        pass
    failure = Disagreement("mismatch")
    failure.report = report
    def run():
        raise failure
    with pytest.raises(Disagreement):
        worker.publish_qualification(tmp_path, "model", run)
    assert json.loads((tmp_path / "model--qualification.json").read_text()) == report
    assert not (tmp_path / "COMPLETE.json").exists()


def test_diagnostic_report_cannot_qualify(tmp_path):
    with pytest.raises(RuntimeError, match="unresolved numerical"):
        worker.publish_qualification(tmp_path, "model", lambda: dict(status="diagnostic_only"))


@pytest.mark.parametrize("new,retained,completion,valid", [
    (239, 108, 1, True), (240, 108, 1, False), (240, 121, 0, False), (241, 0, 0, False)])
def test_retained_and_new_caps_both_apply(new, retained, completion, valid):
    plan = dict(maximum_new_output_bytes=240, maximum_output_bytes=360,
                combined_phase_output_ceiling_bytes=400)
    if valid:
        worker.check_combined_output(new, retained, plan, completion)
    else:
        with pytest.raises(RuntimeError, match="Retained plus new"):
            worker.check_combined_output(new, retained, plan, completion)


@pytest.mark.parametrize("candidate", ["ddr_mlp128_e8", "ddr_res256_e8", "ddr_res256_e16"])
def test_controller_accepts_only_bound_architecture_policy(candidate):
    checkpoint = dict(candidate_id=candidate, seed=1)
    good = qualification(candidate, 1)
    controller.validate_qualification(good, checkpoint)
    for key, value in (("atol", 3e-5), ("production_qualified", False), ("guides", 1),
                       ("scope", "legacy_batch16_diagnosis_not_a_production_certificate")):
        wrong = copy.deepcopy(good)
        wrong[key] = value
        with pytest.raises(RuntimeError, match="architecture-bound"):
            controller.validate_qualification(wrong, checkpoint)
    wrong = copy.deepcopy(good)
    wrong["numerical_policy"]["automatic_fallback"] = True
    with pytest.raises(RuntimeError, match="architecture-bound"):
        controller.validate_qualification(wrong, checkpoint)


def test_worker_and_controller_seal_views_are_identical():
    row = dict(candidate_id="ddr_mlp128_e8", seed=0, storage="external_reused",
        path="/original/bank", seal={"sha256": "s"}, bank_identity={"sha256": "b"},
        reuse_seal_path="reuse-seals/model.json", array_identity={"sha256": "a"})
    assert worker.seal_row(row) == controller.seal_row(row)
    row.update(storage="new", path="banks/model")
    assert worker.seal_row(row) == controller.seal_row(row)
    assert "reuse_seal_path" not in worker.seal_row(row)


def test_compatibility_and_publication_order_are_bound():
    control = (ROOT / "d2_forecast_control.py").read_text()
    ast.parse(control, feature_version=(3, 6))
    source = (ROOT / "d2_forecast_worker.py").read_text()
    ast.parse(source, feature_version=(3, 12))
    assert source.index('publish_json(output / "ALL_MODELS_QUALIFIED.json"') < source.index('bank = write_bank(')
    assert source.index('event="source_lease_exit_verified"') < source.index('reused_seals = publish_reuse_seals(')
    assert source.index('Checkpoint changed after prediction') < source.index('reused_seals = publish_reuse_seals(')
    assert 'fresh_lease_exit_verified=True' in source


@pytest.fixture
def controller_case(tmp_path, monkeypatch):
    """Small stand-in bytes for all nine banks; helper integrity tested separately."""
    output, bundle, control_root = (tmp_path / name for name in ("output", "bundle", "control"))
    for directory in (output, bundle, control_root):
        directory.mkdir()
    def publish(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
    def checked(path, retain=False):
        raw = Path(path).read_bytes()
        return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)), raw if retain else b""
    checkpoints = [dict(candidate_id=candidate, seed=seed, sha256=candidate+str(seed))
        for candidate in ("ddr_mlp128_e8", "ddr_res256_e8", "ddr_res256_e16") for seed in range(3)]
    handoff = dict(parent={"completion_sha256": "parent"}, checkpoints=checkpoints,
                   query_measure_ids=["D2::synthetic"])
    publish(bundle / "D2_HANDOFF.json", handoff)
    publish(bundle / "PLAN.json", {"test": "synthetic"})
    manifest = {"synthetic": True}
    publish(bundle / "EXTERNAL_REUSE.json", manifest)
    manifest_id = checked(bundle / "EXTERNAL_REUSE.json")[0]
    plan = dict(worker_output=str(output), run_id="synthetic", implementation_sha256="impl",
        handoff_sha256="handoff", external_reuse_manifest_sha256=manifest_id["sha256"],
        maximum_new_output_bytes=240*1024**3, maximum_output_bytes=360*1024**3,
        combined_phase_output_ceiling_bytes=400*1024**3)
    source = {key: "synthetic" for key in SOURCE_KEYS}
    source.update(modeled_measure_ids=handoff["query_measure_ids"], modeled_guide_indices=[0])
    publish(output / "REFERENCES.json", source)
    publish(output / "STARTUP.json", {"startup": True})
    publish(control_root / "startup-validated.json", dict(uid="job", startup_sha256=checked(output / "STARTUP.json")[0]["sha256"],
        actual_pod_audit_passed=True))
    external, rows = [], []
    for index, binding in enumerate(checkpoints):
        candidate, seed = binding["candidate_id"], binding["seed"]
        label = candidate+"--seed-"+str(seed)
        q = qualification(candidate, seed)
        loader = {"candidate_id": candidate, "seed": seed}
        publish(output / (label+"--qualification.json"), q)
        publish(output / (label+"--loader.json"), loader)
        bank = dict(modeled_measure_ids=handoff["query_measure_ids"], prefixes=[4096, 8192],
            steps_per_interval=8, endpoint_observations_read=False, parameter_updates=0,
            checkpoint_sha256=binding["sha256"], implementation_sha256="impl")
        row = dict(candidate_id=candidate, seed=seed, checkpoint_sha256=binding["sha256"],
                   qualification=q, loader=loader)
        if index < 3:
            directory = tmp_path / "old" / label
            directory.mkdir(parents=True)
            (directory / "forecast.h5").write_bytes(b"not a real array")
            publish(directory / "BANK.json", bank)
            row.update(storage="external_reused", path=str(directory),
                bank_identity=checked(directory / "BANK.json")[0], array_identity=checked(directory / "forecast.h5")[0])
            old = dict(row, bank=bank)
            external.append(old)
            seal_path = "reuse-seals/"+label+".json"
            publish(output / seal_path, {"synthetic_reuse": label})
            row.update(reuse_seal_path=seal_path, reuse_seal=checked(output / seal_path)[0],
                       seal=checked(output / seal_path)[0])
        else:
            directory = output / "banks" / label
            directory.mkdir(parents=True)
            (directory / "forecast.h5").write_bytes(b"not a real array")
            array = checked(directory / "forecast.h5")[0]
            bank["artifacts"] = {"forecast.h5": dict(sha256=array["sha256"], bytes=array["size_bytes"])}
            publish(directory / "BANK.json", bank)
            input_check_path = output / (label+"--input-check.json")
            publish(input_check_path, dict(inputs_verified_after_prediction=True, parameters_unchanged=True))
            publish(directory / "COMPLETE.json", dict(status="sealed_forecast_bank",
                bank_sha256=checked(directory / "BANK.json")[0]["sha256"],
                input_check_sha256=checked(input_check_path)[0]["sha256"]))
            row.update(storage="new", path="banks/"+label, bank_identity=checked(directory / "BANK.json")[0],
                       seal=checked(directory / "COMPLETE.json")[0])
        rows.append(row)
    publish(output / "ALL_MODELS_QUALIFIED.json", dict(
        status="all_nine_current_production_paths_qualified_before_new_bulk_arrays",
        checkpoints=[dict(candidate_id=r["candidate_id"], seed=r["seed"], qualification=checked(output /
            (r["candidate_id"]+"--seed-"+str(r["seed"])+"--qualification.json"))[0]) for r in rows]))
    publish(output / "ddr_res256_e8--seed-0--legacy16-diagnostic.json",
            dict(status="diagnostic_only", production_qualified=False))
    publish(output / "PREDICTION_SEAL.json", dict(status="all_nine_forecasts_and_source_references_sealed",
        banks=[worker.seal_row(row) for row in rows], references=checked(output / "REFERENCES.json")[0],
        endpoint_observations_read=False))
    files = [dict(path=path.relative_to(output).as_posix(), **checked(path)[0])
        for path in sorted(output.rglob("*")) if path.is_file()]
    size = sum(row["size_bytes"] for row in files)
    complete = dict(status="completed_D2_frozen_forecasts", run_id="synthetic", plan_sha256=checked(bundle / "PLAN.json")[0]["sha256"],
        pod_uid="pod", implementation_sha256="impl", handoff_sha256="handoff",
        input_lease_exit_verified=True, predictions_sealed=True, parameter_updates=0,
        query_deployment_performed=True, assessment_performed=False, D2_scores_computed=False,
        D1_access=False, D1_endpoint_access=False, LARRY_access=False, scientific_promotion=False,
        scientific_acceptance="not_assessed_forecasts_only", new_bank_count=6, reused_bank_count=3,
        all_model_qualification_before_bulk=True, external_reuse_manifest_sha256=manifest_id["sha256"],
        parent_completion_sha256="parent", banks=rows, files=files, output_bytes=size,
        retained_output_bytes=1000, combined_forecast_output_bytes=size+1000)
    publish(output / "COMPLETE.json", complete)
    monkeypatch.setattr(controller.base, "checked_file", checked)
    monkeypatch.setattr(controller.base, "BASE", bundle)
    monkeypatch.setattr(controller.base, "core", SimpleNamespace(sha=lambda path: checked(path)[0]["sha256"]), raising=False)
    monkeypatch.setattr(controller, "validate_reuse", lambda *a, **kw: dict(banks=external, retained_bytes=1000))
    monkeypatch.setattr(controller, "expected_reuse_seal", lambda manifest, validated, row:
        {"synthetic_reuse": row["candidate_id"]+"--seed-"+str(row["seed"])})
    # Only the owner view is replaced; actual membership and bytes remain real.
    real_lstat = Path.lstat
    monkeypatch.setattr(Path, "lstat", lambda self: SimpleNamespace(
        st_uid=307469, st_mode=real_lstat(self).st_mode))
    return plan, dict(uid="job"), control_root, [{"metadata": {"uid": "pod"}}], output, complete, publish


def test_controller_checks_mixed_external_and_new_bank_completion(controller_case):
    plan, binding, root, owned, *_ = controller_case
    assert controller.validate_result(plan, binding, root, owned)


@pytest.mark.parametrize("mutation", ["combined_bytes", "external_path", "external_seal", "storage", "qualification", "extra_file"])
def test_controller_rejects_recovery_identity_and_budget_changes(controller_case, mutation):
    plan, binding, root, owned, output, complete, publish = controller_case
    if mutation == "combined_bytes":
        complete["combined_forecast_output_bytes"] += 1
    elif mutation == "external_path":
        complete["banks"][0]["path"] = "/different/bank"
    elif mutation == "external_seal":
        complete["banks"][0]["seal"] = {"sha256": "wrong", "size_bytes": 0}
    elif mutation == "storage":
        complete["banks"][0]["storage"] = "new"
    elif mutation == "qualification":
        complete["banks"][8]["qualification"]["production_qualified"] = False
    elif mutation == "extra_file":
        (output / "unaccounted").write_bytes(b"extra")
    publish(output / "COMPLETE.json", complete)
    with pytest.raises(RuntimeError):
        controller.validate_result(plan, binding, root, owned)
