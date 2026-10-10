"""Exact-identity qualification lifecycle; host Python 3.6 compatible.

The supervise command is attach-only: its Kubernetes wrapper forbids create.
Static probe is neutral, explicitly zero-GPU/masked, and never imports science.
"""

import argparse
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from runtime_policy import validate_plan as validate_runtime_policy, validate_job as validate_runtime_job, validate_pod as validate_runtime_pod

KUBECTL = "/risapps/noarch/kubectl/1.28.4/bin/kubectl"
CONTEXT = "yding4_yn-gpu-workload@kubernetes-admin@kubernetes"
NAMESPACE = "yn-gpu-workload"
MODE = None


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, value):
    data = value if isinstance(value, bytes) else canonical(value)
    temporary = path.with_name(path.name + ".publishing")
    fd = os.open(
        str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(str(temporary), str(path), follow_symlinks=False)
    temporary.unlink()


def kube(*args):
    if MODE == "supervise" and args[0] not in ("get", "logs", "patch", "delete"):
        raise RuntimeError("Attach-only supervisor cannot create/replace workloads")
    command = [KUBECTL, "--request-timeout=30s", "--context", CONTEXT, "-n", NAMESPACE]
    command += list(args)
    tries = 4 if args[0] in ("get", "logs") else 1
    for attempt in range(tries):
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45
            )
            if result.returncode == 0:
                return result.stdout.decode()
            error = result.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            error = "context deadline exceeded: bounded subprocess timeout"
        transient = any(
            m in error.lower()
            for m in (
                "etcdserver: leader changed",
                "context deadline exceeded",
                "i/o timeout",
                "connection reset",
                "tls handshake timeout",
                "service unavailable",
                "unexpected eof",
            )
        )
        if not transient or attempt == tries - 1:
            raise RuntimeError(error)
        time.sleep(2 ** (attempt + 1))


def get_job(name):
    raw = kube("get", "job/" + name, "--ignore-not-found", "-o", "json")
    return json.loads(raw) if raw.strip() else None


def pods(name, uid):
    values = json.loads(kube("get", "pods", "-l", "job-name=" + name, "-o", "json"))
    rows = values.get("items", [])
    for pod in rows:
        if not any(
            o.get("uid") == uid and o.get("controller")
            for o in pod["metadata"].get("ownerReferences", [])
        ):
            raise RuntimeError("Exact-name selector returned another owner's Pod")
    return rows


def job_contract(job):
    validate_runtime_job(job)
    spec = job["spec"]
    return dict(
        selector=spec["selector"],
        template=spec["template"],
        pod_spec_sha256=digest(spec["template"]["spec"]),
        execution_shape={k: spec.get(k) for k in ('completionMode','completions','parallelism','backoffLimit')},
    )


def audit(job, manifest):
    """Bind admission defaults; reject changes to every supplied execution field."""
    validate_runtime_job(manifest)
    validate_runtime_job(job)

    def supplied_equal(expected, observed, path=""):
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                raise RuntimeError("Admission changed type at " + path)
            for key, value in expected.items():
                if key not in observed:
                    raise RuntimeError("Admission removed " + path + "/" + key)
                supplied_equal(value, observed[key], path + "/" + key)
        elif isinstance(expected, list):
            if not isinstance(observed, list) or len(expected) != len(observed):
                raise RuntimeError("Admission changed list at " + path)
            for index, value in enumerate(expected):
                supplied_equal(value, observed[index], path + "/" + str(index))
        elif expected != observed:
            raise RuntimeError("Admission changed protected field " + path)

    # Exact observed site mapping, not an in-place rewrite of the rendered
    # manifest. Re-submitting cpu=0 changed memory to zero in attempt A2.
    expected = copy.deepcopy(manifest)
    container = expected["spec"]["template"]["spec"]["containers"][0]
    if container["command"] == [
        "/opt/credo/bin/python",
        "-c",
        "print('CREDO_NEUTRAL_IMAGE_STARTED')",
    ]:
        env = {e["name"]: e.get("value") for e in container["env"]}
        if env.get("NVIDIA_VISIBLE_DEVICES") != "none":
            raise RuntimeError("Neutral probe is not explicitly masked")
        for key in ("requests", "limits"):
            resources = container["resources"][key]
            if (
                resources["cpu"] != "1"
                or resources["memory"] != "4Gi"
                or resources["nvidia.com/gpu"] != "0"
            ):
                raise RuntimeError("Unqualified neutral resource request")
            resources["cpu"] = "0"
    supplied_equal(expected, job)
    pod = job["spec"]["template"]["spec"]
    allowed_pod = set(manifest["spec"]["template"]["spec"]) | {
        "dnsPolicy",
        "schedulerName",
        "priorityClassName",
        "priority",
        "preemptionPolicy",
        "tolerations",
        "serviceAccount",
        "serviceAccountName",
        "nodeSelector",
        "affinity",
    }
    if set(pod) - allowed_pod:
        raise RuntimeError(
            "Unexpected admitted Pod fields: " + str(sorted(set(pod) - allowed_pod))
        )
    for c in pod["containers"]:
        if any(
            k in c
            for k in ("lifecycle", "livenessProbe", "startupProbe", "readinessProbe")
        ):
            raise RuntimeError("Unexpected executable admission mutation")
    # Full admitted object is retained for human review before create. Actual
    # execution is bound by canonical template and selector after real creation.


def verify(job, binding, suspended=None):
    validate_runtime_job(job)
    if job is None or job["metadata"]["uid"] != binding["uid"]:
        raise RuntimeError("Bound exact Job absent or replaced")
    if (
        job["metadata"].get("annotations", {}).get("seadragon_run_id")
        != binding["run_id"]
    ):
        raise RuntimeError("Run-ID drift")
    if job_contract(job) != binding["contract"]:
        raise RuntimeError("Admitted selector/template drift")
    if suspended is not None and job["spec"].get("suspend") is not suspended:
        raise RuntimeError("Suspension state drift")


def capture(job, receipt_root):
    values = pods(job["metadata"]["name"], job["metadata"]["uid"])
    publish(receipt_root / "terminal-job.json", job)
    publish(receipt_root / "terminal-pods.json", dict(items=values))
    for i, pod in enumerate(values):
        try:
            log = kube(
                "logs",
                "pod/" + pod["metadata"]["name"],
                "--all-containers=true",
                "--tail=4000",
                "--limit-bytes=1500000",
            )
        except Exception as error:
            log = "LOG_CAPTURE_UNAVAILABLE: " + repr(error)
        publish(receipt_root / ("pod-%02d.log" % i), log.encode())
    return values


def cleanup(binding, root, create_scoped=False):
    name, uid = binding["job"], binding["uid"]
    for attempt in range(3):
        current = get_job(name)
        if current is None:
            break
        if current["metadata"]["uid"] != uid:
            raise RuntimeError("Cleanup UID differs")
        if not create_scoped:
            if (
                current["metadata"].get("annotations", {}).get("seadragon_run_id")
                != binding["run_id"]
            ):
                raise RuntimeError("Cleanup run-ID differs")
        options = dict(
            apiVersion="v1",
            kind="DeleteOptions",
            propagationPolicy="Foreground",
            preconditions=dict(
                uid=uid, resourceVersion=current["metadata"]["resourceVersion"]
            ),
        )
        path = root / ("delete-%02d.json" % attempt)
        publish(path, options)
        try:
            kube(
                "delete",
                "--raw",
                "/apis/batch/v1/namespaces/" + NAMESPACE + "/jobs/" + name,
                "-f",
                str(path),
            )
            break
        except Exception as error:
            # Only an explicit resourceVersion conflict permits a new mutation.
            if "Conflict" not in str(error) and "object has been modified" not in str(
                error
            ):
                publish(
                    root / "cleanup-attention.json", dict(error=repr(error), uid=uid)
                )
                raise
            time.sleep(1 + attempt)
    absence = []
    for _ in range(90):
        job, owned = get_job(name), pods(name, uid)
        if job is not None and job["metadata"]["uid"] != uid:
            raise RuntimeError("Cleanup name reused")
        absence.append(dict(job_absent=job is None, owned_pods=len(owned)))
        if len(absence) >= 2 and all(
            x == dict(job_absent=True, owned_pods=0) for x in absence[-2:]
        ):
            publish(
                root / "cleanup.json",
                dict(status="complete", job=name, uid=uid, reads=absence),
            )
            return
        time.sleep(1)
    raise RuntimeError("Cleanup did not establish two absence reads")


def activate(binding, root):
    for attempt in range(3):
        current = get_job(binding["job"])
        verify(current, binding, True)
        if pods(binding["job"], binding["uid"]):
            raise RuntimeError("Suspended Job already owns Pods")
        patch = [
            dict(op="test", path=p, value=v)
            for p, v in (
                ("/metadata/uid", binding["uid"]),
                ("/metadata/resourceVersion", current["metadata"]["resourceVersion"]),
                ("/metadata/annotations/seadragon_run_id", binding["run_id"]),
                ("/spec/suspend", True),
                ("/spec/selector", binding["contract"]["selector"]),
                ("/spec/template", binding["contract"]["template"]),
            )
        ]
        patch += [dict(op='test',path='/spec/'+key,value=value)
                  for key,value in binding['contract'].get('execution_shape',{}).items()
                  if value is not None]
        patch.append(dict(op="replace", path="/spec/suspend", value=False))
        try:
            result = json.loads(
                kube(
                    "patch",
                    "job/" + binding["job"],
                    "--type=json",
                    "-p",
                    canonical(patch).decode(),
                    "-o",
                    "json",
                )
            )
        except Exception as error:
            now = get_job(binding["job"])
            verify(now, binding)
            if now["spec"].get("suspend") is not True:
                publish(
                    root / "activation-attention.json",
                    dict(uid=binding["uid"], error=repr(error)),
                )
                raise RuntimeError(
                    "Ambiguous activation; retained, not ordinary success"
                )
            if attempt == 2 or not (
                "Conflict" in str(error) or "test failed" in str(error)
            ):
                raise
            time.sleep(1 + attempt)
            continue
        verify(result, binding, False)
        publish(
            root / "activation.json",
            dict(
                status="activated", uid=binding["uid"], binding_sha256=digest(binding)
            ),
        )
        return


def complete(job):
    values = job.get("status", {}).get("conditions", [])
    success = any(
        x.get("type") == "Complete" and x.get("status") == "True" for x in values
    )
    failed = any(
        x.get("type") == "Failed" and x.get("status") == "True" for x in values
    )
    return success, failed


def audit_runtime_pod(pod, binding, plan):
    validate_runtime_pod(pod)
    expected = binding["contract"]["template"]["spec"]
    observed = copy.deepcopy(pod["spec"])
    if binding['contract'].get('execution_shape',{}).get('completionMode') == 'Indexed':
        index = pod['metadata'].get('annotations', {}).get('batch.kubernetes.io/job-completion-index')
        if index not in ('0', '1', '2'):
            raise RuntimeError('Indexed Pod has no valid fixed arm identity')
        if 'hostname' in observed:
            if observed.pop('hostname') != binding['job'] + '-' + index:
                raise RuntimeError('Unexpected Indexed Pod hostname')
    node = observed.pop("nodeName", None)
    if not isinstance(node, str) or not node or "/" in node:
        raise RuntimeError("Actual worker node identity missing")
    # Runtime-only defaults are retained in the raw Pod receipt, not mistaken
    # for a second source of command/env/mount authority.
    wrapped = dict(spec=dict(template=dict(spec=observed)))
    audit(wrapped, dict(spec=dict(template=dict(spec=expected))))
    container = observed["containers"][0]
    allowed_container = set(expected["containers"][0]) | {
        "terminationMessagePath",
        "terminationMessagePolicy",
    }
    if set(container) - allowed_container:
        raise RuntimeError("Unqualified actual container fields")
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if len(statuses) != 1 or statuses[0].get("name") != "worker":
        raise RuntimeError("Exact worker container status missing")
    if not statuses[0].get("imageID", "").endswith(plan["image"].split("@", 1)[1]):
        raise RuntimeError("Actual pulled image digest differs")
    if pod["metadata"].get("namespace") != NAMESPACE:
        raise RuntimeError("Actual Pod namespace differs")
    if not any(
        o.get("uid") == binding["uid"] and o.get("controller")
        for o in pod["metadata"].get("ownerReferences", [])
    ):
        raise RuntimeError("Actual Pod owner differs")


def release_startup(plan, binding, root, current, owned=None):
    path = Path(plan["worker_output"]) / "STARTUP.json"
    if not path.is_file():
        return
    if path.is_symlink() or path.stat().st_uid != 307469:
        raise RuntimeError("Worker startup receipt path/owner differs")
    startup = json.loads(path.read_text())
    owned = pods(binding["job"], binding["uid"]) if owned is None else owned
    if len(owned) != 1:
        raise RuntimeError("Startup requires exactly one UID-owned Pod")
    pod = owned[0]
    # Preserve actual admission even when subsequent checks reject it.
    publish(root / "startup-pod.json", pod)
    audit_runtime_pod(pod, binding, plan)
    gpu = startup.get("gpu", {})
    plan_sha = sha(Path(plan["bundle_root"]) / "PLAN.json")
    if (
        startup.get("status") != "startup_guard_passed"
        or startup.get("run_id") != plan["run_id"]
        or startup.get("plan_sha256") != plan_sha
        or startup.get("pod_uid") != pod["metadata"]["uid"]
        or startup.get("implementation_sha256") != plan["implementation_sha256"]
        or startup.get("image") != plan["image"]
        or gpu.get("count") != 1
        or "H100" not in gpu.get("name", "")
        or "MIG" in gpu.get("name", "")
        or gpu.get("total_memory_bytes", 0) < 75e9
        or gpu.get("compute_capability") != "9.0"
        or startup.get("writer_uid") != 307469
        or startup.get("writer_gid") != 600815
        or startup.get("writer_canary_removed") is not True
        or startup.get("bundle_inventory_verified") is not True
        or startup.get("required_imports_passed") is not True
        or startup.get("tests_started") is not False
    ):
        raise RuntimeError("In-worker startup evidence differs")
    value = dict(
        status="startup_released",
        plan_sha256=plan_sha,
        startup_sha256=sha(path),
        uid=binding["uid"],
        pod_uid=pod["metadata"]["uid"],
        actual_pod_audit_passed=True,
        actual_pod_sha256=digest(pod),
        tests_permitted=True,
        scientific_promotion=False,
    )
    if "parent_gate_path" in plan:
        value["parent_gate_sha256"] = sha(Path(plan["parent_gate_path"]))
    publish(root / "startup-validated.json", value)
    publish(Path(plan["worker_output"]) / "STARTUP_RELEASE.json", value)


def validate_throughput(plan, summary):
    output = Path(plan["worker_output"])
    path = output / "THROUGHPUT.json"
    if (summary.get("status") != "completed_matched_measurement"
            or summary.get("matched_integrity_passed") is not True
            or summary.get("throughput_sha256") != sha(path)):
        return False
    value = json.loads(path.read_text())
    if (value.get("status") != summary["status"]
            or value.get("schema_id") != "credo.external_matched_cache_throughput"
            or value.get("run_id") != plan["run_id"]
            or value.get("implementation_sha256") != plan["implementation_sha256"]
            or value.get("specification_sha256") != plan["parent_specification_sha256"]
            or value.get("measured_profile_identity") != plan["measured_profile_identity"]
            or value.get("protocol") != plan["benchmark"]
            or value.get("comparison") != summary.get("comparison")
            or any(value.get(k) is not True for k in (
                "matched_integrity_passed", "training_weights_discarded",
                "full_audit_at_every_evaluation", "optimizer_state_parity_checked"))
            or any(value.get(k) is not False for k in (
                "calibration_completed", "count_gate_evaluated", "scientific_promotion",
                "automatic_calibration"))):
        return False
    if len(value.get("arms", [])) != 4:
        return False
    for index, item in enumerate(value["arms"]):
        arm_path = output / ("ARM_%02d.json" % index)
        if (item.get("arm") != index or item.get("mode") != plan["benchmark"]["arms"][index]
                or item.get("sha256") != sha(arm_path)):
            return False
        arm = json.loads(arm_path.read_text())
        if (arm.get("mode") != item["mode"] or arm.get("arm") != index
                or arm.get("train", {}).get("updates") != dict(autoencoder=1050, count_factor=1050)
                or [v["batches"] for v in arm.get("traces", [])] != [607, 1050, 607]):
            return False
    return True


def validate_pipeline(plan, summary):
    output = Path(plan["worker_output"])
    value = json.loads((output / "PIPELINE.json").read_text())
    gate = summary.get("count_gate_passed")
    expected = "completed_count_representation" if gate is True else "completed_count_gate_failed"
    if (type(gate) is not bool or summary.get("status") != expected
            or value.get("status") != expected
            or summary.get("pipeline_sha256") != sha(output / "PIPELINE.json")
            or summary.get("report_sha256") != sha(output / "RUN_REPORT.md")
            or value.get("run_id") != plan["run_id"]
            or value.get("specification_sha256") != plan["parent_specification_sha256"]
            or value.get("implementation_sha256") != plan["implementation_sha256"]
            or value.get("count_gate", {}).get("count_gate_passed") is not gate
            or value.get("calibration_completed") is not True
            or value.get("refit_completed") is not gate
            or value.get("frozen_exports_completed") is not gate
            or value.get("scientific_promotion") is not False
            or value.get("protected_endpoint_counts_opened") is not False
            or value.get("parent_gate_sha256") != sha(Path(plan["parent_gate_path"]))):
        return False
    expected_stages = {"calibration", "fitted", "latent-fitting", "latent-query"} if gate else {"calibration"}
    if set(value.get("stages", {})) != expected_stages:
        return False
    for name, record in value["stages"].items():
        if record.get("manifest_sha256") != sha(output / name / "COMPLETE.json"):
            return False
    if not gate and any((output / n).exists() for n in ("fitted", "refit-checkpoints", "latent-fitting", "latent-query")):
        return False
    return True


def monitor(plan, binding, root, static):
    raise RuntimeError("Application must install its no-runtime-deadline monitor")


def process_identity(pid):
    parts = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
    return dict(
        pid=pid,
        state=parts[0],
        pgrp=int(parts[2]),
        session=int(parts[3]),
        tty=int(parts[4]),
        start_ticks=int(parts[19]),
        cmdline_sha256=sha(Path("/proc/%d/cmdline" % pid)),
    )


def stable_process(pid):
    value = process_identity(pid)
    if (
        value.pop("state") == "Z"
        or value["pgrp"] != pid
        or value["session"] != pid
        or value["tty"] != 0
    ):
        raise RuntimeError("Detached process identity invalid")
    return value


def main():
    global MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dryrun", "create", "launch", "supervise"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("gpu",), default="gpu")
    args = parser.parse_args()
    MODE = args.mode
    plan = json.loads(args.plan.read_text())
    validate_runtime_policy(plan)
    bundle = Path(plan["bundle_root"])
    if sha(bundle / "control.py") != plan["control_sha256"]:
        raise RuntimeError("Lifecycle source identity differs")
    root = Path(plan["control_root"]) / args.stage
    manifest_path = bundle / (args.stage + "-job.json")
    manifest = json.loads(manifest_path.read_text())
    name = manifest["metadata"]["name"]
    if args.mode == "dryrun":
        root.mkdir(mode=0o700, parents=False)
        if get_job(name) is not None:
            raise RuntimeError("Exact name already exists")
        dry = json.loads(
            kube("create", "--dry-run=server", "-o", "json", "-f", str(manifest_path))
        )
        publish(root / "dry-run-object.json", dry)
        audit(dry, manifest)
        publish(
            root / "dry-run.json",
            dict(
                manifest_sha256=sha(manifest_path),
                admitted_sha256=digest(dry),
                plan_sha256=sha(args.plan),
                status="passed",
            ),
        )
        print(json.dumps(dry, sort_keys=True))
        return
    if args.mode == "create":
        dry = json.loads((root / "dry-run.json").read_text())
        if dry["manifest_sha256"] != sha(manifest_path) or dry["plan_sha256"] != sha(
            args.plan
        ):
            raise RuntimeError("Dry-run binding differs")
        if plan.get(
            "startup_gate"
        ) != "inline_h100_controller_release_v1" or not plan.get(
            "startup_gate_authority"
        ):
            raise RuntimeError("User-authorized in-worker startup gate is required")
        if get_job(name) is not None:
            raise RuntimeError("Refusing existing exact Job")
        publish(
            root / "create-intent.json",
            dict(job=name, manifest_sha256=sha(manifest_path)),
        )
        try:
            created = json.loads(kube("create", "-o", "json", "-f", str(manifest_path)))
        except Exception as error:
            observations = []
            until = time.monotonic() + 45
            while time.monotonic() < until:
                try:
                    observations.append(get_job(name))
                except Exception:
                    observations.append(dict(state="unknown"))
                time.sleep(3)
            publish(
                root / "create-ambiguous.json",
                dict(error=repr(error), observations=observations),
            )
            raise RuntimeError("Create outcome uncertain; no retry permitted")
        binding = dict(
            job=name,
            uid=created["metadata"]["uid"],
            run_id=manifest["metadata"]["annotations"]["seadragon_run_id"],
        )
        publish(root / "create-response.json", created)
        try:
            audit(created, manifest)
            binding["contract"] = job_contract(created)
            reads = []
            for _ in range(2):
                current = get_job(name)
                verify(current, binding, True)
                if pods(name, binding["uid"]):
                    raise RuntimeError("Suspended Job has Pods")
                reads.append(current["metadata"]["resourceVersion"])
                time.sleep(1)
            binding["read_resource_versions"] = reads
            binding["plan_sha256"] = sha(args.plan)
            publish(root / "live-binding.json", binding)
            publish(
                root / "arm.json", dict(binding_sha256=digest(binding), status="armed")
            )
            print(json.dumps(binding, sort_keys=True))
        except Exception as error:
            publish(
                root / "create-failure.json",
                dict(uid=binding["uid"], error=repr(error)),
            )
            cleanup(binding, root, create_scoped=True)
            raise
        return
    binding = json.loads((root / "live-binding.json").read_text())
    if binding["plan_sha256"] != sha(args.plan):
        raise RuntimeError("Live binding plan differs")
    if args.mode == "static-run":
        if args.stage != "static":
            raise RuntimeError("Static mode requires static stage")
        activate(binding, root)
        monitor(plan, binding, root, True)
        print("STATIC_QUALIFICATION_PASSED")
    elif args.mode == "supervise":
        try:
            verify(get_job(name), binding, True)
            publish(
                root / "supervisor-ready.json",
                dict(status="ready_attach_only", uid=binding["uid"]),
            )
            if "parent" in plan:
                from parent_gate import wait_for_parent
                try:
                    wait_for_parent(plan, binding, root, sys.modules[__name__])
                    # Commit the zero-GPU handoff before activating even when
                    # parent completion races the initial launcher handshake.
                    until = time.monotonic() + 90
                    while not (root / "launch-success.json").exists():
                        if time.monotonic() >= until:
                            raise RuntimeError("Deferred launcher did not commit handoff")
                        time.sleep(1)
                except Exception as error:
                    publish(root / "parent-gate-failure.json", dict(error=repr(error), uid=binding["uid"]))
                    current = get_job(name)
                    verify(current, binding, True)
                    if pods(name, binding["uid"]):
                        raise RuntimeError("Unexpected worker before parent acceptance")
                    cleanup(binding, root)
                    raise
            activate(binding, root)
            monitor(plan, binding, root, False)
        except Exception as error:
            publish(
                root / "supervisor-attention.json",
                dict(
                    error=repr(error),
                    uid=binding["uid"],
                    status="attention_required_no_duplicate_launch",
                    cleanup_deferred=not (root / "cleanup.json").is_file(),
                ),
            )
            raise
    elif args.mode == "launch":
        python = Path("/usr/libexec/platform-python3.6")
        for path in (python, Path(KUBECTL)):
            if (
                path.is_symlink()
                or not path.is_file()
                or not os.access(str(path), os.X_OK)
            ):
                raise RuntimeError("Bound detached tool path invalid")
        config_path = os.environ.get("KUBECONFIG", "")
        info = os.lstat(config_path)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RuntimeError("Kubernetes configuration metadata invalid")
        verify(get_job(name), binding, True)
        command = [
            str(python),
            str(bundle / "control.py"),
            "supervise",
            "--plan",
            str(args.plan),
        ]
        publish(
            root / "spawn-intent.json",
            dict(
                command=command,
                plan_sha256=sha(args.plan),
                control_sha256=plan["control_sha256"],
            ),
        )
        fd = os.open(
            str(root / "supervisor.log"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(fd, "wb") as log:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=dict(
                    PATH="/usr/bin:/bin",
                    LANG="C.UTF-8",
                    PYTHONUNBUFFERED="1",
                    KUBECONFIG=config_path,
                ),
            )
        identity = stable_process(child.pid)
        publish(root / "process-identity.json", identity)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if ((root / "retained-suspended.json").is_file()
                    and (root / "supervisor-ready.json").is_file()
                    and not (root / "activation.json").exists()):
                current = get_job(name)
                verify(current, binding, True)
                if pods(name, binding["uid"]) or stable_process(child.pid) != identity:
                    raise RuntimeError("Deferred activation binding changed")
                if (root / "supervisor-attention.json").exists():
                    raise RuntimeError("Deferred supervisor already reports attention")
                value = dict(status="healthy_detached_retained_suspended_waiting_parent",
                             job=name, uid=binding["uid"], process=identity,
                             binding_sha256=digest(binding),
                             retained_sha256=sha(root / "retained-suspended.json"),
                             readiness_sha256=sha(root / "supervisor-ready.json"))
                publish(root / "launch-success.json", value)
                print(json.dumps(value, sort_keys=True))
                return
            if (root / "activation.json").is_file() and (
                root / "supervisor-ready.json"
            ).is_file():
                if stable_process(child.pid) != identity:
                    raise RuntimeError("Detached identity changed")
                publish(
                    root / "launch-success.json",
                    dict(
                        status="healthy_detached_activated",
                        process=identity,
                        binding_sha256=digest(binding),
                        activation_sha256=sha(root / "activation.json"),
                        readiness_sha256=sha(root / "supervisor-ready.json"),
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            status="healthy_detached_activated",
                            job=name,
                            uid=binding["uid"],
                            process=identity,
                        )
                    )
                )
                return
            if child.poll() is not None:
                raise RuntimeError(
                    "Supervisor exited before handoff; inspect exact receipts"
                )
            time.sleep(1)
        publish(
            root / "launch-attention.json",
            dict(status="attach_required", process=identity),
        )
        raise RuntimeError("Handshake deadline; retained, no duplicate launch")


if __name__ == "__main__":
    main()
