import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_status import summarize
from prepare_operate import RUN_ID


def records():
    job = {"metadata": {"uid": "job1", "annotations": {"seadragon_run_id": RUN_ID}},
           "spec": {"template": {"spec": {"containers": [{"resources": {
               "requests": {"nvidia.com/gpu": "1"}, "limits": {"nvidia.com/gpu": "1"}}}]}}}}
    pod = {"metadata": {"uid": "pod1", "name": "pod1", "ownerReferences": [
        {"uid": "job1", "name": RUN_ID, "controller": True}]},
        "status": {"phase": "Pending", "conditions": [{"type": "PodScheduled", "status": "False"}]}}
    return job, pod


def test_pending_not_gpu_usage():
    job, pod = records()
    result = summarize(job, [pod], "job1")
    assert result["ready_gpu_requests"] == 0
    assert result["pods"][0]["scheduled"] == "False"


def test_ready_requires_matching_startup_for_hardware():
    job, pod = records()
    pod["status"] = {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]}
    assert summarize(job, [pod], "job1")["hardware_verified_gpus"] == 0
    startup = {"pod_uid": "pod1", "status": "startup_guard_passed"}
    assert summarize(job, [pod], "job1", startup)["hardware_verified_gpus"] == 1


@pytest.mark.parametrize("what", ["uid", "owner", "gpu"])
def test_wrong_identity_and_gpu_rejected(what):
    job, pod = map(copy.deepcopy, records())
    if what == "uid":
        job["metadata"]["uid"] = "other"
    elif what == "owner":
        pod["metadata"]["ownerReferences"] = []
    else:
        job["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "2"
    with pytest.raises(ValueError):
        summarize(job, [pod], "job1")
