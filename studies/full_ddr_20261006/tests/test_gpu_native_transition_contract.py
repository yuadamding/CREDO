"""Verify prepared transition accounting, source bindings and non-launch scope."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    return json.loads((ROOT / "gpu_native_transition_v1/QUALIFICATION_RELAUNCH_CONTRACT.json").read_bytes())


def test_prepared_contract_cannot_be_mistaken_for_launch_authority():
    contract = load()
    assert contract["status"] == "prepared_not_launch_authorized"
    assert not contract["authority"]["qualification_submission_authorized"]
    assert not contract["authority"]["replacement_training_authorized"]
    assert not contract["authority"]["automatic_successor_launch_authorized"]
    assert contract["resources"]["automatic_retries"] == 0
    assert contract["resources"]["elapsed_time_deadline_seconds"] is None
    assert not contract["information_contract"]["LARRY_access"]


def test_full_shape_partial_shape_refresh_and_update_arithmetic():
    contract = load()
    information, q, replacement = (contract[key] for key in ("information_contract", "qualification", "proposed_replacement"))
    n, batch = information["fitting_source_laws"], replacement["guides_per_batch"]
    pairs = information["fitting_endpoint_laws"]
    assert pairs % batch == q["partial_state_guides"]
    assert n % batch == q["partial_mass_guides"]
    assert n % 256 == q["refresh_partial_guides"]
    assert q["maximum_refresh_chunks"] == 3*((n+255)//256)
    updates = 40*((pairs+batch-1)//batch)+20*((n+batch-1)//batch)
    assert updates == replacement["exact_updates_per_fit"] == 2560
    assert 40*pairs+20*n == replacement["guide_exposures_per_fit"]
    assert updates*replacement["maximum_fits"] == replacement["maximum_new_scientific_optimizer_updates"]
    assert q["maximum_total_loss_backward_calls"] == q["maximum_full_shape_loss_backward_calls"]+q["maximum_partial_shape_loss_backward_calls"] == 102
    assert q["maximum_disposable_optimizer_steps"] == 3*3*2


def test_consumed_parent_work_not_ignored_or_reallocated():
    contract = load()
    parent, new = contract["parent"], contract["proposed_replacement"]
    assert parent["unused_original_allowance"] == parent["original_update_ceiling"]-parent["consumed_updates_upper_bound"]
    combined = parent["consumed_updates_upper_bound"]+new["maximum_new_scientific_optimizer_updates"]
    assert combined == new["maximum_combined_scientific_attempts_including_parent_upper_bound"]
    assert combined-parent["original_update_ceiling"] == new["additional_scientific_update_allowance_above_original_ceiling"]
    assert combined+contract["qualification"]["maximum_disposable_optimizer_steps"] == new["maximum_combined_optimizer_operations_including_disposable_steps"]
    assert not new["reuse_old_completed_fits"] and not new["warm_start_from_old_models"]
    assert not parent["unpublished_model_reuse"] and not parent["optimizer_resume"]


def test_exact_stop_and_historical_source_bindings():
    contract = load()
    root = ROOT / "gpu_native_transition_v1"
    parent = contract["parent"]
    raw = (root / parent["stop_result"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == parent["stop_result_sha256"]
    assert len(raw) == parent["stop_result_size_bytes"]
    stop = json.loads(raw)
    assert stop["status"] == "user_stopped_exact_job" and stop["uid"] == parent["job_uid"]
    assert stop["cleanup"]["reads"][-2:] == [dict(job_absent=True, owned_pods=0)]*2
    assert stop["output_files_deleted"] == stop["new_jobs"] == 0
    assert not stop["input_lease_exit_verified"] and not stop["old_pipeline_completion_claimed"]
    assert stop["update_accounting"]["attempted_updates_upper_bound"] == parent["consumed_updates_upper_bound"]
    for name, expected in contract["source_sha256"].items():
        # Recovery changes live helpers, not the immutable source that produced
        # the original prepared contract and failed qualification. The benchmark
        # was not deployed; its unchanged local source remains its bound copy.
        path = ROOT / ("gpu_native_qualification_v1/bundle/" + name) if name != "gpu_native_benchmark.py" else ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_each_input_binding_matches_the_preserved_parent_bundle_bytes():
    bindings = load()["information_contract"]["parent_input_bindings"]
    for entry in bindings.values():
        if isinstance(entry, dict):
            path = ROOT / "fitting_v1/bundle" / Path(entry["path"]).name
            assert len(entry["sha256"]) == 64
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
