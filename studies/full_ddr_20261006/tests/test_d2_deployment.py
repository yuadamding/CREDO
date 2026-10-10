"""Synthetic-only tests of the newly admitted frozen-model deployment path."""
# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
import copy
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]

import numpy as np
import pytest
import torch

from credo.runtime import TrainingEngine
from credo.recipes.compact_sde_v3.training import Trainer
from d2_forecast import iter_batched_forecasts, qualify_forecast
from frozen_ddr import load_frozen_model
from prepare_worker import file_identity
from terminal_fitting import recipe
from test_full_forecast import setup
from test_terminal_fitting import setup_run


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    problem, config = setup_run(tmp_path)
    trainer = TrainingEngine().fit(recipe, problem, config, device="cpu")
    path = trainer.save() / "state/checkpoint.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    binding = dict(file_identity(path, owner=os.getuid()), candidate_id="synthetic", seed=7,
        completed_epochs=2, compiled_problem_hash=payload["compiled_problem_hash"],
        parameter_count=sum(p.numel() for p in trainer.model.parameters()),
        architecture=payload["architecture"])
    handoff = dict(expected_train_measure_ids=list(payload["train_measure_ids"]),
        query_measure_ids=list(payload["validation_measure_ids"]),
        expected_embedding_ids=sorted(payload["architecture"]["embedding_ids"]))
    return path, binding, handoff, payload


def test_inference_loader_has_no_training_or_assessment_path(checkpoint, monkeypatch):
    path, binding, handoff, payload = checkpoint
    def forbidden(*args, **kwargs):
        raise AssertionError("Training or endpoint evaluation was entered")
    monkeypatch.setattr(Trainer, "load", forbidden)
    monkeypatch.setattr(Trainer, "evaluate", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    rng = torch.get_rng_state().clone()
    model, receipt = load_frozen_model(path, binding, handoff, device="cpu", expected_owner=os.getuid())
    assert torch.equal(rng, torch.get_rng_state())
    assert not model.training and model.growth_enabled and not model.context_enabled
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert receipt["parameter_updates"] == 0
    assert not receipt["evaluation_performed"] and not receipt["optimizer_constructed"]
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, payload["model_state"][key])


@pytest.mark.parametrize("change", ["seed", "hash", "epochs", "support", "architecture"])
def test_loader_rejects_identity_changes(checkpoint, change):
    path, binding, handoff, _ = checkpoint
    binding, handoff = copy.deepcopy(binding), copy.deepcopy(handoff)
    if change == "seed": binding["seed"] = 8
    if change == "hash": binding["sha256"] = "0" * 64
    if change == "epochs": binding["completed_epochs"] = 3
    if change == "support": handoff["query_measure_ids"] = []
    if change == "architecture": binding["architecture"]["hidden_dim"] = 17
    with pytest.raises(ValueError):
        load_frozen_model(path, binding, handoff, device="cpu", expected_owner=os.getuid())


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_batched_matches_scalar_and_retains_order_independent_streams(batch):
    from full_forecast import iter_forecasts
    torch.set_num_threads(1)
    _, sources, model = setup()
    model.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    expected = list(iter_forecasts(model, sources, prefixes=(2, 4), steps_per_interval=2, seed=7))
    actual = list(iter_batched_forecasts(model, sources, prefixes=(2, 4), steps_per_interval=2,
                                       seed=7, batch_guides=batch))
    for left, right in zip(expected, actual, strict=True):
        assert left.measure_id == right.measure_id
        for name in ("state", "log_weight", "mass_increment", "conditional_features"):
            assert np.allclose(getattr(left, name), getattr(right, name), rtol=1e-5, atol=2e-5)
    reordered = replace(sources, measure_ids=sources.measure_ids[::-1], guide_indices=sources.guide_indices[::-1])
    reverse = {r.measure_id: r for r in iter_batched_forecasts(model, reordered, prefixes=(2, 4),
                                                           steps_per_interval=2, seed=7, batch_guides=batch)}
    assert all(np.allclose(r.state, reverse[r.measure_id].state, atol=2e-5, rtol=1e-5) for r in actual)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_cpu_smoke_cannot_claim_real_cuda_qualification():
    _, sources, model = setup()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        qualify_forecast(model, sources, seed=0, device="cpu")


@pytest.fixture(scope="module")
def deployment_payload():
    import d2_forecast_launch as launch
    launch.bound_shared()
    return launch.assemble()


def test_exact_production_bundle_and_approved_resources(deployment_payload):
    import ast
    import json
    from d2_forecast_worker import validate_plan
    from d2_handoff import PARENT_PLAN
    from prepare_launch import sha
    p = deployment_payload
    plan, handoff, authority = (json.loads(p[name]) for name in ("PLAN.json", "D2_HANDOFF.json", "D2_AUTHORITY.json"))
    validate_plan(plan, handoff, authority)
    assert handoff["parent"]["plan_sha256"] == PARENT_PLAN
    assert len(handoff["query_measure_ids"]) == 23421
    assert all(key.startswith("D2::") for key in handoff["query_measure_ids"])
    assert len(handoff["checkpoints"]) == 9
    assert handoff["forecast"]["nine_bank_admission_bytes"] < 360*1024**3
    assert plan["maximum_output_bytes"] == 360*1024**3
    assert authority["maximum_output_bytes"] == plan["combined_phase_output_ceiling_bytes"] == 400*1024**3
    assert plan["later_assessment_requires_fresh_remaining_quota_and_combined_cap_check"]
    assert plan["handoff_sha256"] == sha(p["D2_HANDOFF.json"])
    job = json.loads(p["gpu-job.json"])
    pod = job["spec"]["template"]["spec"]
    assert "activeDeadlineSeconds" not in pod and "activeDeadlineSeconds" not in job["spec"]
    assert job["spec"]["backoffLimit"] == 0 and job["spec"]["suspend"]
    for kind in ("requests", "limits"):
        r = pod["containers"][0]["resources"][kind]
        assert (r["cpu"], r["memory"], r["nvidia.com/gpu"]) == ("24", "200Gi", "1")
    assert "PHASE_CHAIN.json" not in p
    for name in ("control.py", "publication.py"):
        ast.parse(p[name], feature_version=(3, 6))
    for name in ("worker.py", "d2_forecast.py", "frozen_ddr.py"):
        ast.parse(p[name], feature_version=(3, 12))
    assert b"Trainer.load" not in p["frozen_ddr.py"]
    # The on-disk source integration tests must exercise the exact production
    # reader and type contracts imported from the completed fitting bundle.
    for name in ("latent_support.py", "development_inputs.py",
                 "source/src/credo/contracts.py", "source/src/credo/data/support.py"):
        assert p[name] == (ROOT / name).read_bytes()
    recovery = json.loads(p["FAILED_V3_RECONCILIATION.json"])
    assert sha(p["FAILED_V3_RECONCILIATION.json"]) == plan["failed_v3_reconciliation_sha256"]
    assert recovery["failure"]["written_banks"] == 3
    assert recovery["failure"]["D2_scores_computed"] is False
    assert plan["run_id"] == "credo-full-ddr-d2-forecast-20261010-v5"
    assert plan["maximum_new_output_bytes"] == 240*1024**3
    assert plan["retained_prior_output_bytes"] + plan["maximum_new_output_bytes"] < plan["maximum_output_bytes"]
    assert sha(p["EXTERNAL_REUSE.json"]) == plan["external_reuse_manifest_sha256"]
    assert (plan["new_bank_count"], plan["reused_bank_count"]) == (6, 3)
    assert plan["all_model_qualification_before_bulk"] is True
    assert plan["production_batch_guides_by_architecture"] == {"mlp": 16, "residual_mlp": 1}
    ast.parse(p["d2_forecast_reuse.py"], feature_version=(3, 6))
    assert authority["manual_relaunch_user_instruction"] == "fix all and relaunch"


def test_recovery_metadata_does_not_enter_ssh_argv():
    from d2_forecast_operate import process_reconciliation
    import json
    recovery = json.loads((ROOT / "d2_forecast_v3/RECOVERY_RECONCILIATION_20261010.json").read_text())
    compact = process_reconciliation(recovery)
    assert set(compact) == {"job", "uid", "host", "process"}
    assert len(json.dumps(compact).encode()) < 4096
    assert compact["process"] == recovery["process"]


@pytest.mark.parametrize("key,value", [("D1_access", True), ("parameter_updates", 1),
    ("D2_scores_computed", True), ("batch_guides", 32), ("automatic_retries", 1),
    ("maximum_output_bytes", 401*1024**3), ("per_bank_output_bytes", 50*1024**3)])
def test_worker_scope_rejects_expansion(deployment_payload, key, value):
    import json
    from d2_forecast_worker import validate_plan
    plan, handoff, authority = (json.loads(deployment_payload[name]) for name in
                               ("PLAN.json", "D2_HANDOFF.json", "D2_AUTHORITY.json"))
    plan[key] = value
    with pytest.raises(ValueError):
        validate_plan(plan, handoff, authority)


@pytest.mark.parametrize("caller_fails", [False, True])
def test_source_lease_opens_only_early_source_and_closes(monkeypatch, caller_fails):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import d2_forecast as module
    from credo.data.support import EmpiricalLaw
    opened, closed = [], []
    class FakeStore:
        def __init__(self, binding, access, **kwargs):
            self.binding = binding
            opened.append(binding.source)
        @contextmanager
        def cached_snapshot(self, **kwargs):
            yield self
        def support_record(self, source, index):
            assert source == "D2_Stim8hr"
            return dict(captured_cells=2 if index == 0 else 0)
        def ref(self, source, index):
            assert (source, index) == ("D2_Stim8hr", 0)
            return source, index
        def read(self, ref):
            # Match the actual SupportStore protocol, not its downstream type.
            return EmpiricalLaw(np.arange(96).reshape(2, 48), np.array([.25, .75]))
        def close(self):
            closed.append(self.binding.source)
    def binding(inputs, entry):
        assert entry["source_ids"] == ["D2_Stim8hr"]
        return SimpleNamespace(source=entry["source_ids"][0], representation="frozen")
    monkeypatch.setattr(module, "frozen_binding", binding)
    monkeypatch.setattr(module, "FrozenLatentSupportStore", FakeStore)
    inputs = dict(stores=[dict(source_ids=["D2_Stim8hr"], completion_sha256="a"*64),
                         dict(source_ids=["D2_Stim48hr"], completion_sha256="b"*64)],
                  maximum_snapshot_bytes=1024, representation_sha256="c"*64)
    metadata = dict(canonical_guides=[dict(guide_index=i, sgRNA="guide"+str(i),
                    is_design_ntc=i==0, primary_target_id="target"+str(i)) for i in range(26504)])
    handoff = dict(query_inventory_sha256="d"*64, query_records=[dict(guide_index=0,
        measure_id="D2::guide-index-0", guide_id="guide0", embedding_id="shared-NTC-reference", is_control=True)])
    def use_source():
        with module.d2_source_lease(inputs, metadata, handoff) as sources:
            assert sources.axis.kind == "physical" and sources.axis.source == "Stim8hr"
            mass = 2.5/(2+.5*26504)
            assert sources.catalog.source_frequency[0] == pytest.approx(mass)
            assert len(sources.catalog.counts) == 26504
            assert sources.measure_ids == ("D2::guide-index-0",)
            law = sources.measures["Stim8hr"][sources.measure_ids[0]]
            np.testing.assert_array_equal(law.support, np.arange(96).reshape(2, 48))
            np.testing.assert_allclose(law.normalized_weights, [.25, .75], rtol=0, atol=1e-15)
            assert law.weights.sum() == pytest.approx(mass)
            assert law.total_mass == pytest.approx(mass)
            if caller_fails:
                raise RuntimeError("synthetic caller failure")
    if caller_fails:
        with pytest.raises(RuntimeError, match="synthetic caller failure"):
            use_source()
    else:
        use_source()
    assert opened == closed == ["D2_Stim8hr"]


@pytest.fixture
def disk_source(tmp_path):
    from test_latent_support import fixture
    binding = fixture(tmp_path, source="D2_Stim8hr", guide_count=26504, dimension=48)
    # Exercise the real frozen_binding as well as the real HDF5/parquet store.
    inputs = dict(stores=[dict(source_ids=["D2_Stim8hr"], root=str(binding.latent.path.parent),
                              store_id=binding.store_id, completion_sha256=binding.completion.sha256,
                              retained_cells=5, geometry_cells=3),
                         dict(source_ids=["D2_Stim48hr"], root=str(tmp_path / "FORBIDDEN_ENDPOINT"))],
                  maximum_snapshot_bytes=4096, guide_count=26504, dimension=48,
                  fitted_root=str(binding.representation_record.path.parent),
                  fitted_complete_sha256=binding.representation_record.sha256,
                  representation_id=binding.representation.representation_id,
                  representation_sha256=binding.representation.artifact_sha256,
                  model_sha256=binding.representation.model_sha256)
    metadata = dict(canonical_guides=[dict(guide_index=i, sgRNA="guide"+str(i),
                        is_design_ntc=i==0, primary_target_id="target"+str(i)) for i in range(26504)])
    handoff = dict(query_inventory_sha256="d"*64, query_records=[dict(guide_index=i,
        measure_id="D2::guide-index-"+str(i), guide_id="guide"+str(i),
        embedding_id="shared-NTC-reference" if i == 0 else "target::target1", is_control=i==0)
        for i in (0, 1)])
    return binding, inputs, metadata, handoff


def test_real_disk_source_to_batched_bank_and_post_lease_seal(disk_source, tmp_path):
    import h5py
    import d2_forecast as module
    from credo.recipes.compact_sde_v3.model import CREDOModel
    from encode_sources import publish_json, sha_file
    from full_forecast import iter_forecasts, seal_bank, write_bank
    binding, inputs, metadata, handoff = disk_source
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = CREDOModel(embedding_ids=["shared-NTC-reference", "target::target1"],
                      control_embedding_ids=["shared-NTC-reference"], latent_dim=48,
                      hidden_dim=8, embedding_dim=2, n_programs=2, context_mode="none")
    model.set_phase("mass")
    model.requires_grad_(False)
    destination = tmp_path / "synthetic_bank"
    with module.d2_source_lease(inputs, metadata, handoff) as sources:
        assert tuple(sources.measures) == ("Stim8hr",)
        assert sources.catalog.counts[:4] == (3, 1, 1, 0)
        assert sum(sources.catalog.counts) == 5
        # A captured zero-RNA cell remains in the count, never in state support.
        law = sources.measures["Stim8hr"][sources.measure_ids[0]]
        assert law.support.shape == (2, 48)
        np.testing.assert_array_equal(law.support[:, :2], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(law.normalized_weights, [.5, .5])
        assert law.total_mass == pytest.approx(3.5 / (5 + .5 * 26504))
        expected = list(iter_forecasts(model, sources, prefixes=(2, 4), steps_per_interval=2, seed=7))
        actual = list(iter_batched_forecasts(model, sources, prefixes=(2, 4), steps_per_interval=2,
                                            seed=7, batch_guides=16))
        for left, right in zip(expected, actual, strict=True):
            for field in ("state", "log_weight", "mass_increment", "conditional_features"):
                np.testing.assert_allclose(getattr(left, field), getattr(right, field), rtol=1e-5, atol=2e-5)
        bank = write_bank(destination, sources, iter(actual), prefixes=(2, 4), checkpoint_sha256="a"*64,
                          implementation_sha256="b"*64, seed=7, steps_per_interval=2,
                          maximum_output_bytes=16*1024**2)
        assert not (destination / "COMPLETE.json").exists()
    check = tmp_path / "input_check.json"
    publish_json(check, dict(schema="credo.all4_full_ddr.forecast_input_check.v1",
        bank_sha256=sha_file(destination / "BANK.json"), inputs_verified_after_prediction=True,
        parameters_unchanged=True, parameter_updates=0, endpoint_observations_read=False,
        **{k: bank[k] for k in ("source_artifact_sha256", "support_sha256", "representation_sha256",
                                "checkpoint_sha256")}))
    assert seal_bank(destination, input_check_receipt=check, input_check_sha256=sha_file(check))["status"] == "sealed_forecast_bank"
    with h5py.File(destination / "forecast.h5", "r") as handle:
        assert handle["terminal_state"].shape == (2, 4, 48)
        probability = handle["catalog_probability"][:]
        assert probability.shape == (2, 26504)
        np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-12)
        assert np.all(probability > 0)
        # Positive count without state and observed zero both stay in complement.
        np.testing.assert_allclose(probability[:, 2] / probability[:, 3], 3, rtol=1e-12)
    assert all(p.grad is None for p in model.parameters())


def test_real_disk_lease_rejects_input_drift_before_seal(disk_source):
    import d2_forecast as module
    from latent_support import LatentSupportError
    binding, inputs, metadata, handoff = disk_source
    with pytest.raises(LatentSupportError, match="changed"):
        with module.d2_source_lease(inputs, metadata, handoff):
            with binding.support.path.open("ab") as stream:
                stream.write(b"synthetic-drift")
