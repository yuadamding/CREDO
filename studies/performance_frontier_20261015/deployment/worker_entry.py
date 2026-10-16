"""Bound worker handshake followed by the observation CLI; no scheduler client."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import IMAGE, checked_path, identity, publish, read, read_json, verify_bundle


def canary(directory):
    path = directory / 'writer-canary.json'
    raw = dict(uid=os.getuid(), gid=os.getgid(), purpose='create-readback-unlink-before-work')
    receipt = publish(path, raw)
    read(path, receipt['sha256'])
    path.unlink()
    if path.exists():
        raise RuntimeError('Writer canary was not removed')


def command_for(plan, worker):
    if 'recovery_contract' in plan:
        if plan['mode'] != 'fit':
            raise ValueError('Source-changing continuation is fitting-only')
        recovery = plan['recovery_contract']
        return ['/opt/credo/bin/python', '-m', 'frontier.observation_continuation',
                '--contract', plan['worker_contract']['path'], '--contract-sha256', plan['worker_contract']['sha256'],
                '--recovery-contract', recovery['path'], '--recovery-contract-sha256', recovery['sha256']]
    return ['/opt/credo/bin/python', '-m', 'frontier.observation_worker', plan['mode'],
            '--contract', plan['worker_contract']['path'], '--contract-sha256', plan['worker_contract']['sha256']]


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(argv)
    plan = read_json(args.plan, args.plan_sha256)
    bundle = checked_path(Path(__file__).absolute().parent)
    worker, admission = verify_bundle(plan, bundle)
    if (os.getuid(), os.getgid()) != (307469, 600815):
        raise RuntimeError('Worker ownership differs')
    if os.environ.get('CREDO_RUN_ID') != plan['run_id'] or os.environ.get('CREDO_V4_EXECUTION_IMAGE_DIGEST') != IMAGE.split('@')[1]:
        raise RuntimeError('Worker environment identity differs')
    pod_name, pod_uid = os.environ.get('CREDO_POD_NAME'), os.environ.get('CREDO_POD_UID')
    if not pod_name or not pod_uid:
        raise RuntimeError('Actual Pod identity missing')
    scientific = checked_path(plan['scientific_output'], exists=False)
    if scientific.exists():
        raise RuntimeError('Scientific output already exists; no automatic retry')
    handshake = checked_path(plan['worker_output'], exists=False)
    handshake.mkdir(mode=0o700)
    try:
        canary(handshake)
        # Separate process: worker CUDA configuration still precedes its own
        # context initialization. This process does not open any real arrays.
        script = "import json,torch; p=torch.cuda.get_device_properties(0); print(json.dumps(dict(count=torch.cuda.device_count(),name=p.name,total_memory_bytes=p.total_memory,compute_capability=list(torch.cuda.get_device_capability(0)))))"
        result = subprocess.run([sys.executable, '-c', script], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True, timeout=90)
        gpu = json.loads(result.stdout.decode())
        if (gpu.get('count') != 1 or 'H100' not in gpu.get('name', '') or 'MIG' in gpu.get('name', '')
                or gpu.get('compute_capability') != [9, 0] or gpu.get('total_memory_bytes', 0) < 75 * 1024 ** 3):
            raise RuntimeError('Full H100 startup guard failed')
        startup = dict(schema='credo.frontier.observation-startup.v1', status='startup_guard_passed',
                       run_id=plan['run_id'], plan_sha256=args.plan_sha256,
                       worker_contract_sha256=plan['worker_contract']['sha256'], image=IMAGE,
                       pod_name=pod_name, pod_uid=pod_uid, writer_canary_removed=True,
                       uid=os.getuid(), gid=os.getgid(), gpu=gpu, real_data_opened=False, optimizer_calls=0)
        if 'recovery_contract' in plan:
            startup['recovery_contract_sha256'] = plan['recovery_contract']['sha256']
        startup_id = publish(handshake / 'STARTUP.json', startup)
        release_path = handshake / 'STARTUP_RELEASE.json'
        while not release_path.exists():
            time.sleep(5)  # No startup or overall runtime deadline.
        release = read_json(release_path)
        expected = dict(status='startup_released', run_id=plan['run_id'], lifecycle_plan_sha256=args.plan_sha256,
                        worker_contract_sha256=plan['worker_contract']['sha256'], startup_sha256=startup_id['sha256'],
                        pod_uid=pod_uid, actual_pod_audit_passed=True, storage_and_fileset_quota_passed=True,
                        mode=plan['mode'], automatic_successor=False)
        if 'recovery_contract' in plan:
            expected['recovery_contract_sha256'] = plan['recovery_contract']['sha256']
        if any(release.get(key) != value for key, value in expected.items()):
            raise RuntimeError('Controller release identity differs')
        verify_bundle(plan, bundle)
        environment = dict(os.environ, PYTHONPATH=worker['source_root'])
        command = command_for(plan, worker)
        publish(handshake / 'WORKER_INVOCATION.json', dict(command=command, source_root=worker['source_root'],
                contract_sha256=plan['worker_contract']['sha256'], release_sha256=identity(read(release_path))['sha256']))
        # No retry, timeout or shell interpolation. Worker owns its output lease.
        completed = subprocess.run(command, env=environment, cwd=worker['source_root'])
        publish(handshake / 'WORKER_EXIT.json', dict(returncode=completed.returncode, run_id=plan['run_id']))
        if completed.returncode:
            raise RuntimeError('Observation worker failed; no automatic retry')
        return 0
    except BaseException as error:
        publish(handshake / 'ENTRY_FAILED.json', dict(exception_type=type(error).__name__, automatic_retry=False,
                run_id=plan['run_id']))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
