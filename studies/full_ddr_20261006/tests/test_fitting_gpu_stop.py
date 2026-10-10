"""Data-free stop scope and interrupted-step accounting; no remote calls."""
# ruff: noqa: E402
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from fitting_gpu_stop import account_progress, publish, REMOTE_STOP


def fit(seed):
    return dict(candidate_id="example", seed=seed, optimizer_updates=2560)


def test_unfinished_attempt_not_fabricated_as_completed_or_resumable():
    fits = [fit(0), fit(1), fit(2)]
    records = [dict(fit=dict(candidate_id="next", seed=0), optimizer_updates=2137,
                    attempted_optimizer_updates=2138)]
    value = account_progress(fits, records)
    assert value["durably_recorded_optimizer_updates"] == 9817
    assert value["attempted_updates_upper_bound"] == 9818
    assert not value["exact_unpublished_optimizer_completion_verified"]
    assert not value["reuse_unpublished_model"] and not value["optimizer_resume"]


def test_published_fit_not_double_counted_and_partial_batch_accounted():
    fits = [fit(0)]
    records = [dict(fit=dict(candidate_id="example", seed=0), optimizer_updates=2560)]
    value = account_progress(fits, records)
    assert value["durably_recorded_optimizer_updates"] == value["attempted_updates_upper_bound"] == 2560
    assert value["unpublished_fit"] is None


def test_new_fit_latest_counter_not_previous_fit_counter():
    records = [dict(fit=dict(candidate_id="old", seed=2), optimizer_updates=2560),
               dict(fit=dict(candidate_id="next", seed=0), optimizer_updates=0)]
    assert account_progress([fit(0)], records)["durably_recorded_optimizer_updates"] == 2560


def test_exclusive_local_receipt_never_overwrites(tmp_path):
    path = tmp_path / "receipt.json"
    publish(path, dict(status="retained"))
    raw = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish(path, dict(status="replaced"))
    assert path.read_bytes() == raw


def test_host_payload_syntax_and_write_scope():
    compile(REMOTE_STOP, "<host stop>", "exec")
    assert "core.cleanup(binding,root)" in REMOTE_STOP
    assert "not in ('get','logs','delete')" in REMOTE_STOP
    assert "input_lease_exit_verified=False" in REMOTE_STOP
    assert "checkpoint bytes differ" in REMOTE_STOP
    assert "unlink" not in REMOTE_STOP and "rmtree" not in REMOTE_STOP
