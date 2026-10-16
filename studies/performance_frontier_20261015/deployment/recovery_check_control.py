"""Distinct retention-check adapter to the pinned exact-UID lifecycle.

The host reads JSON and hashes bytes only. Numerical imports, checkpoint loads,
scientific qualification certificates and automatic successors are forbidden.
"""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

try:
    from .common import CORE_SHA, IMAGE, identity, read, read_json
    from .recovery_check_common import same, reference, read_reference, verify_bundle, validate_gpu, validate_resources
except ImportError:
    from common import CORE_SHA, IMAGE, identity, read, read_json
    from recovery_check_common import same, reference, read_reference, verify_bundle, validate_gpu, validate_resources


BASE = Path(__file__).absolute().parent
core = None
OWNER_UID = 307469


def inventory(root, limit):
    """Bounded host byte inventory with before/after membership verification."""
    root = Path(root)
    def signature(info):
        return tuple(getattr(info, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_uid',
                                               'st_mode', 'st_nlink', 'st_mtime_ns', 'st_ctime_ns'))
    def scan():
        for parent in (root,) + tuple(root.parents):
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise ValueError('Unsafe output root or ancestor')
        if root.stat().st_uid != OWNER_UID:
            raise ValueError('Output root owner differs')
        rows, size = {}, 0
        def failed(error):
            raise error
        for directory, folders, files in os.walk(str(root), followlinks=False, onerror=failed):
            for name in sorted(folders + files):
                path = Path(directory)/name
                info = path.lstat()
                if info.st_uid != OWNER_UID:
                    raise ValueError('Output owner differs')
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError('Regular single-link output required')
                if name.endswith('FAILED.json') or name.startswith('.publishing'):
                    raise ValueError('Failure or incomplete publication retained')
                if name.endswith('.log') and info.st_size > 4*1024**2:
                    raise ValueError('Output log limit exceeded')
                size += info.st_size
                if size > limit or len(rows) >= 20000:
                    raise ValueError('Output byte/file limit exceeded')
                rows[path.relative_to(root).as_posix()] = signature(info)
        if not rows:
            raise ValueError('Empty output inventory')
        return rows
    initial = scan()
    result = {}
    for name, before in initial.items():
        path, hasher = root/name, hashlib.sha256()
        with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
            if signature(os.fstat(stream.fileno())) != before:
                raise ValueError('Output changed before read')
            while True:
                chunk = stream.read(2*1024**2)
                if not chunk:
                    break
                hasher.update(chunk)
            if signature(os.fstat(stream.fileno())) != before:
                raise ValueError('Output changed during read')
        result[name] = dict(sha256=hasher.hexdigest(), size_bytes=before[2])
    if scan() != initial:
        raise ValueError('Output inventory changed during readback')
    return result


def release_startup(plan, binding, root, current, owned=None):
    root, handshake = Path(root), Path(plan['worker_output'])
    if (root/'startup-validated.json').exists() or not (handshake/'STARTUP.json').is_file():
        return
    worker, admission = verify_bundle(plan, BASE)
    raw = read(handshake/'STARTUP.json')
    startup = json.loads(raw)
    owned = core.pods(binding['job'], binding['uid']) if owned is None else owned
    if len(owned) != 1:
        raise ValueError('One exact-owned scalar qualification Pod required')
    pod = owned[0]
    core.publish(root/'startup-pod.json', pod)
    core.audit_runtime_pod(pod, binding, plan)
    plan_sha = core.sha(BASE/'PLAN.json')
    same(startup, dict(schema='credo.frontier.retention-check-startup.v1', status='startup_guard_passed',
        run_id=plan['run_id'], plan_sha256=plan_sha, worker_contract_sha256=plan['worker_contract']['sha256'],
        image=IMAGE, pod_uid=pod['metadata']['uid'], pod_name=pod['metadata']['name'],
        writer_canary_removed=True, uid=307469, gid=600815, real_data_opened=False, optimizer_calls=0))
    validate_gpu(startup['gpu'])
    validate_resources(startup['resources'])
    # Reuse the exact fileset/aggregate-headroom calculation, not its observation
    # numerical authority. The legacy receipt schema describes capacity only.
    from capacity import collect_capacity, validate_capacity
    capacity = collect_capacity(plan, plan_sha, binding, admission)
    core.publish(root/'capacity-admission.json', capacity)
    validate_capacity(capacity, plan, plan_sha, binding, admission)
    value = dict(schema='credo.frontier.retention-check-startup-release.v1', status='startup_released',
        run_id=plan['run_id'], lifecycle_plan_sha256=plan_sha, worker_contract_sha256=plan['worker_contract']['sha256'],
        startup_sha256=identity(raw)['sha256'], job_uid=binding['uid'],
        pod_uid=pod['metadata']['uid'], pod_name=pod['metadata']['name'], image=IMAGE,
        admission_sha256=worker['admission']['sha256'], source_manifest_sha256=worker['source_manifest']['sha256'],
        phase_authority_sha256=worker['authority']['sha256'], journal_event_sha256=admission['journal_event_sha256'],
        capacity_receipt_sha256=admission['capacity_receipt_sha256'],
        startup_capacity_sha256=core.sha(root/'capacity-admission.json'),
        actual_pod_audit_passed=True, actual_pod_sha256=core.digest(pod),
        storage_and_fileset_quota_passed=True, aggregate_reservation_verified=True, cgroup_resources_verified=True,
        optimizer_calls_authorized=0, checkpoint_intake_sha256=worker['parent_intake']['sha256'],
        scientific_fit_authorized=False, real_biological_payload_access=False,
        authorized_cpus=24, authorized_memory_bytes=200*1024**3, threads=8, gpus=1,
        automatic_retry=False, automatic_successor=False)
    core.publish(root/'startup-validated.json', value)
    core.publish(handshake/'STARTUP_RELEASE.json', value)


def validate_result(plan, binding, root, owned):
    worker, _ = verify_bundle(plan, BASE)
    if len(owned) != 1:
        raise ValueError('One terminal exact-owned Pod required')
    core.audit_runtime_pod(owned[0], binding, plan)
    handshake, output, root = Path(plan['worker_output']), Path(plan['scientific_output']), Path(root)
    release_raw = read(root/'startup-validated.json')
    if read(handshake/'STARTUP_RELEASE.json') != release_raw:
        raise ValueError('Released bytes changed')
    same(json.loads(release_raw),dict(job_uid=binding['uid'],pod_uid=owned[0]['metadata']['uid'],
        worker_contract_sha256=plan['worker_contract']['sha256'],optimizer_calls_authorized=0))
    same(read_json(handshake/'WORKER_EXIT.json'),dict(returncode=0,run_id=plan['run_id']))
    files=inventory(output,plan['maximum_output_bytes'])
    hfiles=inventory(handshake,4*1024**2)
    total=sum(r['size_bytes'] for r in files.values())+sum(r['size_bytes'] for r in hfiles.values())
    if total>plan['maximum_output_bytes']: raise ValueError('Combined check output cap exceeded')
    complete=read_json(output/'COMPLETE.json',files['COMPLETE.json']['sha256'])
    same(complete,dict(schema='credo.frontier.retention-check-completion.v1',status='passed',
        run_id=plan['run_id'],contract=plan['worker_contract'],
        release=reference(handshake/'STARTUP_RELEASE.json',release_raw),
        source_manifest_sha256=worker['source_manifest']['sha256'],
        parent_intake_sha256=worker['parent_intake']['sha256'],optimizer_calls=0,scientific_updates=0,
        consumed_updates_preserved=5120,remaining_scientific_updates_not_authorized=74880,
        parent_writes=False,raw_data_access=False,training_continuation_qualified=False,
        scientific_relaunch_authorized=False,automatic_retry=False,automatic_successor=False,canary_passed=True))
    if [r['pair'] for r in complete['parents']] != ['D1D2','D1D3','D1D4','D2D3']:
        raise ValueError('Exact four checkpoint restorations required')
    for row in complete['parents']:
        same(row,dict(status='exact_state_restoration_verified_on_cuda',consumed_updates=1280,
            remaining_update_cap=18720,optimizer_calls=0,scientific_updates=0,parent_changed=False,
            training_data_opened=False,model_optimizer_rng_sampler_and_audit_values_equal=True,
            training_continuation_qualified=False))
        if read_json(output/row['pair']/'RESTORE_VERIFIED.json')!=row:
            raise ValueError('Per-parent restoration evidence differs')
    same(read_json(output/'storage_canary/COMPLETE.json'),dict(status='passed',optimizer_calls=0,
        scientific_updates=0,real_checkpoint_reads=0,quota_reclaim_claimed=False))
    core.publish(root/'output-readback.json',dict(schema='credo.frontier.retention-check-readback.v1',
        status='passed',run_id=plan['run_id'],files=files,handshake_files=hfiles,total_bytes=total,
        complete_sha256=files['COMPLETE.json']['sha256'],no_model_loading=True,
        scientific_launch_authority=False,automatic_successor=False))
    return files['COMPLETE.json']['sha256']


def monitor(plan, binding, root, static=False):
    if static:
        raise ValueError('Reuse only separately validated neutral platform evidence')
    root = Path(root)
    while True:
        current = core.get_job(binding['job'])
        core.verify(current, binding, False)
        passed, failed = core.complete(current)
        if not (root/'startup-validated.json').exists() and not failed:
            try:
                release_startup(plan, binding, root, current)
            except Exception as error:
                core.publish(root/'startup-failure.json', dict(error=repr(error), uid=binding['uid']))
                core.capture(current, root); core.cleanup(binding, root)
                raise
        if passed or failed:
            owned = core.capture(current, root)
            accepted, reason, complete_sha = False, None, None
            try:
                if passed and not failed:
                    complete_sha = validate_result(plan, binding, root, owned)
                    accepted = True
            except Exception as error:
                reason = repr(error)
            core.publish(root/'terminal.json', dict(status='passed' if accepted else 'failed',
                job=binding['job'], uid=binding['uid'], run_id=plan['run_id'], kubernetes_complete=passed,
                application_pass=accepted, reason=reason, mode=plan['mode'],
                lifecycle_plan_sha256=core.sha(BASE/'PLAN.json'), result_complete_sha256=complete_sha,
                scientific_launch_authority=False, automatic_successor=False))
            core.cleanup(binding, root)
            if not accepted:
                raise RuntimeError('Qualification failed; retain evidence without retry')
            return
        time.sleep(30)


def main():
    global core
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', choices=('dryrun', 'create', 'launch', 'supervise'))
    parser.add_argument('--plan', required=True)
    args, _ = parser.parse_known_args()
    if Path(args.plan) != BASE/'PLAN.json':
        raise ValueError('Exact immutable lifecycle plan required')
    plan = read_json(args.plan)
    verify_bundle(plan, BASE)
    from gpu_v2_static import validate
    validate(BASE, plan)
    read(BASE/'lifecycle_core.py', CORE_SHA)
    spec = importlib.util.spec_from_file_location('retention_check_retained_lifecycle', str(BASE/'lifecycle_core.py'))
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    core.monitor, core.release_startup = monitor, release_startup
    core.main()


if __name__ == '__main__':
    main()
