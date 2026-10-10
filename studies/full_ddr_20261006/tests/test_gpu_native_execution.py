"""Changed input, work-budget, qualification and two-phase launch boundaries."""
# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]

import pytest
import torch

from credo.recipes.compact_sde_v3.recipe import CompactV3Config
from gpu_native import DeviceLawCache
from gpu_native_benchmark import synthetic_data
from gpu_native_inputs import compile_fitting_qualification, fitting_entries, prepared_fitting
from gpu_native_recovery_launch import assemble_phase, roots
from gpu_native_qualification import WorkBudget, choose_profile, compare, execution_options, qualify
from gpu_native_worker import validate_plan
from native_study import build_native_study
from test_native_study import fixture


def test_fitting_only_metadata_projection_excludes_selection():
    contract = json.loads((ROOT / "DEVELOPMENT_INPUTS.json").read_text())
    rows = fitting_entries(contract)
    assert len(rows) == 3
    assert all(row["phase"] == "fitting" and all(source.startswith(("D3_", "D4_")) for source in row["source_ids"]) for row in rows)


def test_projection_does_not_even_hash_D2_payload(monkeypatch):
    import gpu_native_inputs as module
    opened = []
    def binding(contract, entry):
        assert entry["phase"] == "fitting"
        assert not any(s.startswith("D2") for s in entry["source_ids"])
        opened.append(entry["store_id"])
        return SimpleNamespace(representation="synthetic")
    class Store:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass
    monkeypatch.setattr(module, "frozen_binding", binding)
    monkeypatch.setattr(module, "FrozenLatentSupportStore", Store)
    def union(*args):
        raise RuntimeError("stop before synthetic payload reads")
    monkeypatch.setattr(module, "DevelopmentLatentSupportStore", union)
    with pytest.raises(RuntimeError, match="before synthetic"):
        with prepared_fitting(json.loads((ROOT / "DEVELOPMENT_INPUTS.json").read_text()), {},
                authority_sha256="a"*64, input_sha256="b"*64, metadata_sha256="c"*64):
            pass
    assert opened == [row["store_id"] for row in fitting_entries(json.loads((ROOT / "DEVELOPMENT_INPUTS.json").read_text()))]


def test_fitting_only_real_native_compiler_preserves_count_background():
    values = fixture()
    values["observations"] = [row for row in values["observations"] if row.donor in {"D3", "D4"}]
    values["support_records"] = {key: row for key, row in values["support_records"].items() if key[0].startswith(("D3", "D4"))}
    bundle = build_native_study(**values)
    config = CompactV3Config.model_validate(dict(model=dict(context="none"),
        validation=dict(fraction=0., representation_scope="shared"),
        loss=dict(mass=0., count=1., count_likelihood="categorical_ce", mass_increment="particle_log_mass")))
    problem = compile_fitting_qualification(bundle, config)
    assert set(problem.training.measure_meta.sample_id) == {"D3", "D4"}
    assert not any(key.startswith(("D1", "D2")) for key in problem.training.measure_ids)
    assert len(problem.training.count_blocks) == 2
    assert all(len(row.counts)+len(row.background_counts) == len(values["guides"]) for row in problem.training.count_blocks)


def test_work_caps_precede_execution_and_failure_remains_charged():
    budget = WorkBudget(full=1, partial=0, steps=0, refresh=0, chunks=0)
    budget.charge("full")
    with pytest.raises(RuntimeError, match="exhausted"):
        budget.charge("full")
    assert budget.used["full"] == 1
    with pytest.raises(RuntimeError):
        budget.charge("steps")
    budget.complete()


@pytest.mark.parametrize("profile", ["scalar_reference", "unknown", "fused", "BF16"])
def test_only_bound_GPU_native_profiles_train(profile):
    with pytest.raises(ValueError):
        execution_options(profile)


@pytest.mark.parametrize("value", [torch.tensor([float("nan")]), torch.tensor([1.001])])
def test_numerical_failure_is_not_a_timing_fallback(value):
    with pytest.raises(RuntimeError):
        compare({"x": value}, {"x": torch.ones(1)}, "synthetic")


def test_timing_selection_one_profile_all_architectures_and_cold_overhead():
    rows = []
    for candidate in ("a", "b", "c"):
        for profile, warm, cold in (("GPU_native_eager_foreach", 2., 2.), ("GPU_native_compiled_foreach", 1., 100.)):
            for stage in ("state", "mass"):
                rows.append(dict(candidate_id=candidate, profile=profile, stage=stage,
                    median_warm_seconds=warm, calls=[dict(elapsed_seconds=cold)]))
    selected, costs = choose_profile(rows, dict(a=10., b=10., c=10.))
    assert selected == "GPU_native_compiled_foreach"
    assert costs["GPU_native_compiled_foreach"] == 3*3*(2560+220+198)
    with pytest.raises(RuntimeError, match="Missing"):
        choose_profile(rows[:-1], dict(a=10., b=10., c=10.))


def test_complete_small_synthetic_protocol_no_scientific_training():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    data = synthetic_data(groups=4, latent_dim=2, support=3, seed=8)
    data.context_backgrounds = ()
    problem = SimpleNamespace(training=data, embedding_ids=data.embedding_ids,
        control_embedding_ids=data.control_embedding_ids, latent_dim=data.latent_dim)
    cache = DeviceLawCache(data, device="cpu")
    candidates = [dict(candidate_id="synthetic-"+str(i), model=dict(hidden_dim=8, embedding_dim=2,
        trunk_kind="mlp" if i == 0 else "residual_mlp", trunk_depth=2)) for i in range(3)]
    def config_for(candidate, seed):
        return CompactV3Config.model_validate(dict(model=dict(candidate["model"], context="none"),
            training=dict(seed=seed)))
    scales = dict(policy="balanced_fitting_scales_active_coordinate_count_v1", fitting_donors=["D3", "D4"], state=1., count=1.)
    contract = json.loads((ROOT / "FITTING_CONTRACT.json").read_text())
    budget = WorkBudget(chunks=3)
    try:
        result = qualify(problem, candidates, config_for, contract, scales, cache,
            progress=lambda row: None, budget=budget, full_guides=2, partial_state=1, partial_mass=1,
            particles=4, steps=1, compiler_backend="eager")
    finally:
        cache.close()
        torch.set_num_threads(old_threads)
    assert result["budget_used"] == dict(full=90, partial=12, steps=18, refresh=3, chunks=3)
    assert result["scientific_optimizer_updates"] == 0 and result["disposable_optimizer_steps"] == 18
    assert len(result["profiles"]) == 18 and len(result["partial_profiles"]) == 12
    assert all(len(row["calls"]) == 4 for row in result["profiles"])


def test_pair_builder_is_new_source_bound_not_old_capacity_reuse():
    from prepare_launch import sha
    q = assemble_phase("qualification")
    f = assemble_phase("fitting", qualification_plan_sha=sha(q["PLAN.json"]))
    authority = json.loads(q["EXECUTION_AUTHORITY.json"])
    contract = json.loads(q["GPU_CONTRACT.json"])
    for phase, payload in (("qualification", q), ("fitting", f)):
        plan = json.loads(payload["PLAN.json"])
        validate_plan(plan, contract, authority)
        assert plan["worker_output"] == roots(phase)[0]+"/output"
        assert plan["worker_sha256"] == sha(payload["worker.py"])
        assert plan["implementation_sha256"] == sha(payload["IMPLEMENTATION.json"])
        job = json.loads(payload["gpu-job.json"])
        assert job["spec"]["suspend"] is True and job["spec"]["backoffLimit"] == 0
        assert "activeDeadlineSeconds" not in job["spec"]
        assert job["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert json.loads(f["PLAN.json"])["qualification_binding"]["plan_sha256"] == sha(q["PLAN.json"])
    assert json.loads(f["PLAN.json"])["qualification_binding"]["control"] == roots("qualification")[1]+"/gpu"
    assert "PHASE_CHAIN.json" not in f
    changed = copy.deepcopy(json.loads(f["PLAN.json"]))
    changed["maximum_optimizer_updates"] += 1
    with pytest.raises(ValueError):
        validate_plan(changed, contract, authority)


@pytest.mark.parametrize("case", ["passed", "passed_after_deletion_wait", "unqualified", "unclean", "create_failure"])
def test_actual_qualification_to_successor_call_site_never_retries(tmp_path, monkeypatch, case):
    import gpu_native_control as module
    calls = []
    chain = dict(qualification_plan_sha256="q", successor_bundle=str(tmp_path / "fit"),
        successor_inventory_sha256="i", successor_plan_sha256="f", successor_run_id="fitting")
    terminal = dict(status="failed" if case == "unqualified" else "passed", uid="bound")
    cleanup = dict(status="complete", uid="bound", reads=[dict(job_absent=True, owned_pods=0)]*(1 if case == "unclean" else 2))
    if case == "passed_after_deletion_wait":
        cleanup["reads"].insert(0, dict(job_absent=False, owned_pods=0))
    def read(path):
        value = {"PHASE_CHAIN.json": chain, "terminal.json": terminal, "cleanup.json": cleanup,
            "INVENTORY.json": {"control.py": dict(sha256="source", size_bytes=1)}}[Path(path).name]
        return value, {}
    def sha(path):
        return "i" if Path(path).name == "INVENTORY.json" else ("f" if Path(path).parent.name == "fit" else "q")
    monkeypatch.setattr(module, "read", read)
    monkeypatch.setattr(module.base, "core", SimpleNamespace(sha=sha, publish=lambda path, value: None), raising=False)
    monkeypatch.setattr(module.base, "checked_file", lambda path: (dict(sha256="source", size_bytes=1), b"x"))
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if case == "create_failure" else 0,
            stdout=b'{"status":"healthy_detached_activated","job":"fitting","uid":"successor"}\n', stderr=b"synthetic failure")
    monkeypatch.setattr(module.subprocess, "run", run)
    if case in {"passed", "passed_after_deletion_wait"}:
        module.launch_successor({}, dict(uid="bound"), tmp_path)
        assert [call[2] for call in calls] == ["create", "launch"]
        assert all(call[-2:] == ["--plan", str(tmp_path / "fit/PLAN.json")] for call in calls)
    else:
        with pytest.raises(RuntimeError):
            module.launch_successor({}, dict(uid="bound"), tmp_path)
        assert len(calls) == (1 if case == "create_failure" else 0)
