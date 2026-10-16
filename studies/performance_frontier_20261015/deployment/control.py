"""Observation adapter for the pinned exact-UID lifecycle; host Python 3.6."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from common import CORE_SHA, IMAGE, identity, read, read_json, verify_bundle

BASE = Path(__file__).absolute().parent
core = None
PARENT_OWNER_UID = 307469


def release_startup(plan, binding, root, current, owned=None):
    root = Path(root)
    handshake = Path(plan['worker_output'])
    if (root / 'startup-validated.json').exists() or not (handshake / 'STARTUP.json').is_file():
        return
    worker, admission = verify_bundle(plan, BASE)
    startup_raw = read(handshake / 'STARTUP.json')
    startup = read_json(handshake / 'STARTUP.json')
    owned = core.pods(binding['job'], binding['uid']) if owned is None else owned
    if len(owned) != 1:
        raise RuntimeError('Exactly one owned scalar observation Pod required')
    pod = owned[0]
    core.publish(root / 'startup-pod.json', pod)
    core.audit_runtime_pod(pod, binding, plan)
    plan_sha = core.sha(BASE / 'PLAN.json')
    expected = dict(schema='credo.frontier.observation-startup.v1', status='startup_guard_passed',
                    run_id=plan['run_id'], plan_sha256=plan_sha, worker_contract_sha256=plan['worker_contract']['sha256'],
                    image=IMAGE, pod_uid=pod['metadata']['uid'], pod_name=pod['metadata']['name'],
                    writer_canary_removed=True, uid=307469, gid=600815, real_data_opened=False, optimizer_calls=0)
    if 'recovery_contract' in plan:
        expected['recovery_contract_sha256'] = plan['recovery_contract']['sha256']
    if any(startup.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Observation startup identity differs')
    gpu = startup.get('gpu', {})
    if (gpu.get('count') != 1 or gpu.get('compute_capability') != [9, 0]
            or 'H100' not in gpu.get('name', '') or 'MIG' in gpu.get('name', '')
            or not 75 * 1024 ** 3 <= gpu.get('total_memory_bytes', 0) <= 82 * 1024 ** 3):
        raise RuntimeError('Full H100 startup hardware differs')
    from capacity import collect_capacity, validate_capacity
    capacity = collect_capacity(plan, plan_sha, binding, admission)
    core.publish(root / 'capacity-admission.json', capacity)
    validate_capacity(capacity, plan, plan_sha, binding, admission)
    value = dict(status='startup_released', run_id=plan['run_id'], lifecycle_plan_sha256=plan_sha,
                 worker_contract_sha256=plan['worker_contract']['sha256'], startup_sha256=identity(startup_raw)['sha256'],
                 uid=binding['uid'], pod_uid=pod['metadata']['uid'], actual_pod_audit_passed=True,
                 actual_pod_sha256=core.digest(pod), storage_and_fileset_quota_passed=True,
                 capacity_sha256=core.sha(root / 'capacity-admission.json'), mode=plan['mode'],
                 automatic_successor=False)
    if 'recovery_contract' in plan:
        value['recovery_contract_sha256'] = plan['recovery_contract']['sha256']
    core.publish(root / 'startup-validated.json', value)
    core.publish(handshake / 'STARTUP_RELEASE.json', value)


def inventory(root, limit):
    """Terminal byte readback only. Never torch-load a checkpoint on a login host."""
    root = Path(root)
    result, total = {}, 0
    for directory, folders, files in os.walk(str(root), followlinks=False):
        for name in sorted(folders + files):
            path = Path(directory) / name
            before = path.lstat()
            if before.st_uid != 307469 or not (stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode)):
                raise ValueError('Unexpected output type/owner')
            if not stat.S_ISREG(before.st_mode):
                continue
            total += before.st_size
            if total > limit:
                raise ValueError('Observation output allowance exceeded')
            digest = hashlib.sha256()
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
                if signature(os.fstat(stream.fileno())) != signature(before):
                    raise ValueError('Output changed before readback')
                while True:
                    chunk = stream.read(8 * 1024 ** 2)
                    if not chunk:
                        break
                    digest.update(chunk)
                if signature(os.fstat(stream.fileno())) != signature(before):
                    raise ValueError('Output changed during readback')
            if signature(path.lstat()) != signature(before):
                raise ValueError('Output changed after readback')
            result[path.relative_to(root).as_posix()] = dict(sha256=digest.hexdigest(), size_bytes=before.st_size)
    if not result:
        raise ValueError('Empty observation output inventory')
    return result


def read_reference(reference):
    """Read only exact, bounded metadata; never deserialize numerical payloads."""
    if set(reference) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('Exact metadata reference required')
    raw = read(reference['path'], reference['sha256'])
    if type(reference['size_bytes']) is not int or len(raw) != reference['size_bytes']:
        raise ValueError('Metadata reference size differs')
    return json.loads(raw)


def input_projection(snapshot):
    """Portable immutable bindings, excluding worker-local inode/device IDs."""
    return {**{key: snapshot[key] for key in ('schema', 'dataset_root', 'access_sha256',
        'metadata_index_sha256', 'source_ids', 'payload_hashes_verified_on_consumption',
        'all_payloads_consumed_claimed')},
        'members': {name: {'sha256': item['sha256'], 'size_bytes': item['file_identity']['st_size']}
                    for name, item in snapshot['members'].items()}}


def parent_inventory(root, recorded):
    """Check retained parent membership/portable stats without opening payloads.

    Backend-local device/inode numbers are not compared across producers. Full
    signatures are compared across two scans on this same host instead.
    """
    root = Path(root)
    keys = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    portable = keys[2:]
    def scan():
        for ancestor in (root,) + tuple(root.parents):
            if not stat.S_ISDIR(ancestor.lstat().st_mode):
                raise ValueError('Unsafe retained parent root')
        rows = {}
        def failed(error):
            raise error
        for directory, folders, files in os.walk(str(root), followlinks=False, onerror=failed):
            for name in sorted(folders + files):
                path = Path(directory)/name
                info = path.lstat()
                if info.st_uid != PARENT_OWNER_UID:
                    raise ValueError('Retained parent owner differs')
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError('Retained parent member is not owned regular single-link')
                if len(rows) >= 20000:
                    raise ValueError('Retained parent inventory exceeds bound')
                rows[path.relative_to(root).as_posix()] = {k: getattr(info, k) for k in keys}
        return rows
    current = scan()
    if (set(current) != set(recorded) or any(
            {k: row[k] for k in portable} != {k: recorded[name][k] for k in portable}
            for name, row in current.items())):
        raise ValueError('Retained parent inventory changed')
    if scan() != current:
        raise ValueError('Retained parent changed during inventory')
    return sum(row['st_size'] for row in current.values())


def validate_continuation_result(plan, worker, complete, files):
    """Validate mixed-source lineage and its lifetime budget, not a fresh fit."""
    recovery = read_reference(plan['recovery_contract'])
    if (recovery['fit_contract'] != plan['worker_contract']
            or recovery['consumed_updates'] != 1280 or recovery['maximum_new_updates'] != 18720
            or recovery['maximum_total_updates'] != 20000
            or recovery['combined_output_limit_bytes'] != 4*1024**3
            or recovery['parent_retained_output_bytes'] != plan['parent_retained_output_bytes']):
        raise ValueError('Continuation execution/storage bindings differ')
    parent = read_reference(recovery['parent_intake'])['fits'][recovery['pair']]
    parent_root = Path(parent['identity']['path']).parent
    expected_identity = dict(parent['identity']['value'], trajectory_id=worker['trajectory_id'],
        source_sha256=worker['source_manifest_sha256'], worker_contract_sha256=plan['worker_contract']['sha256'],
        recovery_parent_checkpoint_sha256=parent['checkpoint_payload']['sha256'],
        recovery_authority_sha256=recovery['authority']['sha256'])
    expected = dict(identity=expected_identity, accepted_updates=20000, attempted_updates=20000,
        scientific_updates=20000, resume_from_accepted_updates=1280, consumed_parent_updates=1280,
        new_scientific_updates_this_owner=18720, source_continuation_contract=plan['recovery_contract'],
        mixed_source_continuation=True)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError('Mixed-source completion lineage/update accounting differs')
    for key in ('accepted_updates', 'attempted_updates', 'scientific_updates', 'resume_from_accepted_updates',
                'consumed_parent_updates', 'new_scientific_updates_this_owner'):
        if type(complete[key]) is not int:
            raise ValueError('Integer continuation accounting required')
    output = Path(plan['scientific_output'])
    input_ref, exit_ref = complete['input_lease'], complete['input_exit']
    if input_ref['path'] != str(output/'INPUT_LEASE.json'):
        raise ValueError('Continuation input lease escaped child root')
    exit_path = Path(exit_ref['path'])
    if (exit_path.parent != output/'input-exits'
            or re.fullmatch(r'[0-9a-f]{32}\.json', exit_path.name) is None):
        raise ValueError('Exact claim-specific continuation exit required')
    for ref in (input_ref, exit_ref):
        name = Path(ref['path']).relative_to(output).as_posix()
        if files.get(name) != {k: ref[k] for k in ('sha256', 'size_bytes')}:
            raise ValueError('Continuation input evidence differs from output readback')
    claim = exit_path.stem
    def child_metadata(name):
        ref = files.get(name)
        if ref is None:
            raise ValueError('Claim metadata absent from child readback')
        return read_reference(dict(path=str(output/name), **ref))
    claim_value = child_metadata('claims/%s.json' % claim)
    released = child_metadata('claims/%s.released.json' % claim)
    if (claim_value.get('claim_id') != claim or claim_value.get('worker_id') != worker['worker_id']
            or claim_value.get('trajectory_id') != worker['trajectory_id']
            or released != dict(claim_id=claim, exception_type=None, automatic_restart_authorized=False)):
        raise ValueError('Continuation exit claim was not successfully released')
    lease, exit_value = read_reference(input_ref), read_reference(exit_ref)
    previous_lease = read_reference(recovery['parent_input_lease'])
    if (recovery['parent_input_lease']['path'] != str(parent_root/'INPUT_LEASE.json')
            or input_projection(previous_lease) != input_projection(lease)
            or lease.get('access_sha256') != worker['input_access_sha256']
            or lease.get('dataset_root') != worker['dataset_root']
            or lease.get('metadata_index_sha256') != worker['metadata_index_sha256']
            or lease.get('payload_hashes_verified_on_consumption') is not True
            or lease.get('all_payloads_consumed_claimed') is not False):
        raise ValueError('Continuation input differs from parent/fold binding')
    lease_sha = hashlib.sha256(json.dumps(lease, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if exit_value != dict(schema='credo.frontier.native-input-lease-exit.v1',status='unchanged',
            input_lease_sha256=lease_sha, access_sha256=worker['input_access_sha256'],
            checked_members=len(lease['members']), additional_count_array_scans=0,
            same_worker_identity_check=True):
        raise ValueError('Continuation same-worker input exit differs')
    retained = read_reference(recovery['parent_inventory'])['fits'][recovery['pair']]
    parent_bytes = parent_inventory(parent_root, retained['inventory'])
    if parent_bytes != retained['parent_output_bytes'] or parent_bytes != plan['parent_retained_output_bytes']:
        raise ValueError('Retained parent byte accounting differs')
    child_bytes = sum(row['size_bytes'] for row in files.values())
    if (plan['combined_output_limit_bytes'] != 4*1024**3
            or parent_bytes + child_bytes > plan['combined_output_limit_bytes']
            or plan['maximum_output_bytes'] + parent_bytes != plan['combined_output_limit_bytes']):
        raise ValueError('Combined parent/child output allowance exceeded')
    return dict(parent_output_bytes=parent_bytes, child_output_bytes=child_bytes,
        combined_output_bytes=parent_bytes+child_bytes, combined_output_limit_bytes=plan['combined_output_limit_bytes'],
        source_continuation_contract=plan['recovery_contract'], new_scientific_updates_this_owner=18720,
        cumulative_scientific_updates=20000, parent_payloads_opened=False)


def validate_result(plan, binding, root, owned):
    worker, _ = verify_bundle(plan, BASE)
    if len(owned) != 1:
        raise RuntimeError('One terminal owned Pod required')
    handshake, output = Path(plan['worker_output']), Path(plan['scientific_output'])
    if (handshake / 'ENTRY_FAILED.json').exists() or (output / 'WORKER_FAILED.json').exists():
        raise RuntimeError('Worker failure receipt present')
    if read_json(handshake / 'WORKER_EXIT.json') != dict(returncode=0, run_id=plan['run_id']):
        raise RuntimeError('Worker did not exit successfully')
    release = read_json(Path(root) / 'startup-validated.json')
    if (read_json(handshake / 'STARTUP_RELEASE.json') != release
            or release.get('uid') != binding['uid'] or release.get('pod_uid') != owned[0]['metadata']['uid']):
        raise RuntimeError('Release differs from exact completed Pod')
    if ('recovery_contract' in plan
            and release.get('recovery_contract_sha256') != plan['recovery_contract']['sha256']):
        raise RuntimeError('Continuation release binding differs')
    name = 'QUALIFICATION_COMPLETE.json' if plan['mode'] == 'qualify' else 'COMPLETE.json'
    complete = read_json(output / name)
    if plan['mode'] == 'qualify':
        expected = dict(schema='credo.frontier.observation-cuda-qualification.v1', status='passed',
                        source_manifest_sha256=worker['source_manifest_sha256'], protocol_sha256=worker['protocol_sha256'],
                        runtime_image=IMAGE, scientific_updates=0, actual_disposable_optimizer_calls=13,
                        real_count_payloads_opened=False, real_input_mount_qualified=False,
                        restart_equivalence='exact_model_optimizer_sampler_rng', shape=[2048, 18129], nonzeros_per_cell=4096)
        paired = complete.get('paired_fp32_check', {})
        paired_ref = complete.get('paired_fp32_receipt', {})
        if (paired.get('schema') != 'credo.frontier.observation-paired-fp32.v1' or paired.get('status') != 'passed'
                or paired_ref.get('path') != str(output / 'PAIRED_FP32_CHECK.json')
                or read_json(paired_ref['path'], paired_ref.get('sha256')) != paired):
            raise RuntimeError('Paired FP32 CPU/CUDA qualification missing or changed')
    else:
        expected = dict(schema='credo.frontier.observation-worker-completion.v1', status='complete', scope='scientific',
                        accepted_updates=20000, scientific_updates=20000, actual_disposable_optimizer_calls=0,
                        main_dynamics_fitted=False, held_out_donor_outcomes_opened=False)
        lease = complete.get('identity', {})
        if lease.get('worker_contract_sha256') != plan['worker_contract']['sha256'] or lease.get('trajectory_id') != worker['trajectory_id']:
            raise RuntimeError('Completed fit identity differs')
        # The numerical owner must check file identities on exit as well as on
        # consumption. The host never fabricates this from metadata alone.
        if ('recovery_contract' not in plan
                and (not (output / 'INPUT_LEASE.json').is_file() or not (output / 'INPUT_EXIT.json').is_file())):
            raise RuntimeError('Same-worker input lease/exit evidence missing')
    if any(complete.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Observation completion scope differs')
    files = inventory(output, plan['maximum_output_bytes'])
    extra = {} if 'recovery_contract' not in plan else validate_continuation_result(plan, worker, complete, files)
    core.publish(Path(root) / 'output-readback.json', dict(schema='credo.frontier.observation-output-readback.v1',
                 status='passed', run_id=plan['run_id'], files=files, total_bytes=sum(r['size_bytes'] for r in files.values()),
                 complete_sha256=files[name]['sha256'], no_model_loading=True, **extra))
    return files[name]['sha256']


def monitor(plan, binding, root, static=False):
    if static:
        raise RuntimeError('Neutral proof is reused, not rerun')
    root = Path(root)
    while True:
        current = core.get_job(binding['job'])
        core.verify(current, binding, False)
        passed, failed = core.complete(current)
        if not (root / 'startup-validated.json').exists() and not failed:
            try:
                release_startup(plan, binding, root, current)
            except Exception as error:
                core.publish(root / 'startup-failure.json', dict(error=repr(error), uid=binding['uid']))
                core.capture(current, root)
                core.cleanup(binding, root)
                raise
        if passed or failed:
            owned = core.capture(current, root)
            application, reason, complete_sha = False, None, None
            try:
                if passed and not failed:
                    complete_sha = validate_result(plan, binding, root, owned)
                    application = True
            except Exception as error:
                reason = repr(error)
            core.publish(root / 'terminal.json', dict(status='passed' if application else 'failed',
                         job=binding['job'], uid=binding['uid'], run_id=plan['run_id'], kubernetes_complete=passed,
                         application_pass=application, reason=reason, mode=plan['mode'],
                         lifecycle_plan_sha256=core.sha(BASE / 'PLAN.json'), result_complete_sha256=complete_sha,
                         automatic_successor=False))
            core.cleanup(binding, root)
            if not application:
                raise RuntimeError('Observation worker failed; preserve evidence, no retry')
            return
        time.sleep(30)


def main():
    global core
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', choices=('dryrun', 'create', 'launch', 'supervise'))
    parser.add_argument('--plan', required=True)
    args, _ = parser.parse_known_args()
    if Path(args.plan) != BASE / 'PLAN.json':
        raise RuntimeError('Exact immutable lifecycle plan required')
    plan = read_json(args.plan)
    verify_bundle(plan, BASE)
    from gpu_v2_static import validate
    validate(BASE, plan)
    read(BASE / 'lifecycle_core.py', CORE_SHA)
    spec = importlib.util.spec_from_file_location('observation_retained_lifecycle', str(BASE / 'lifecycle_core.py'))
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    core.monitor, core.release_startup = monitor, release_startup
    core.main()


if __name__ == '__main__':
    main()
