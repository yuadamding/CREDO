"""Actual-Pod guard then the distinct main qualification; no scheduler client."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .common import IMAGE, checked_path, identity, publish, read, read_json
    from .main_qualification_common import same, verify_bundle, validate_gpu, validate_resources
except ImportError:
    from common import IMAGE, checked_path, identity, publish, read, read_json
    from main_qualification_common import same, verify_bundle, validate_gpu, validate_resources


def resources(root=Path('/sys/fs/cgroup')):
    """Read only fixed self-cgroup limits, never another process or workload."""
    def integer(path):
        raw = read(path, maximum=4096).decode().strip()
        if not raw.isdecimal():
            raise ValueError('Finite explicit cgroup limit required')
        return int(raw)
    if (root/'cgroup.controllers').is_file():
        quota, period = read(root/'cpu.max', maximum=4096).decode().split()
        if not quota.isdecimal() or not period.isdecimal() or int(period) == 0:
            raise ValueError('Finite explicit CPU quota required')
        quota, period = int(quota), int(period)
        memory, version = integer(root/'memory.max'), 2
    else:
        quota = integer(root/'cpu/cpu.cfs_quota_us')
        period = integer(root/'cpu/cpu.cfs_period_us')
        memory, version = integer(root/'memory/memory.limit_in_bytes'), 1
    if period <= 0 or quota != 24*period:
        raise ValueError('Actual CPU quota differs from 24 cores')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '8':
            raise ValueError('Bound application thread environment differs')
    result = dict(cgroup_version=version, cpu_limit=24, memory_limit_bytes=memory,
                  application_threads=8, affinity_cpus=len(os.sched_getaffinity(0)))
    validate_resources(result)
    return result


def command_for(plan, worker, release_path, release_sha):
    return ['/opt/credo/bin/python', '-m', 'frontier.main_qualification_worker', 'run',
        '--contract', plan['worker_contract']['path'], '--contract-sha256', plan['worker_contract']['sha256'],
        '--release', str(release_path), '--release-sha256', release_sha]


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(argv)
    bundle = checked_path(Path(__file__).absolute().parent)
    if Path(args.plan) != bundle/'PLAN.json':
        raise ValueError('Exact packaged plan required')
    plan = read_json(args.plan, args.plan_sha256)
    worker, admission = verify_bundle(plan, bundle)
    if ((os.getuid(), os.getgid()) != (307469, 600815)
            or os.environ.get('CREDO_RUN_ID') != plan['run_id']
            or os.environ.get('CREDO_V4_EXECUTION_IMAGE_DIGEST') != IMAGE.split('@')[1]):
        raise ValueError('Actual worker identity differs')
    pod_name, pod_uid = os.environ.get('CREDO_POD_NAME'), os.environ.get('CREDO_POD_UID')
    if not pod_name or not pod_uid:
        raise ValueError('Actual Pod identity missing')
    if checked_path(plan['scientific_output'], exists=False).exists():
        raise ValueError('Output already exists; no automatic retry')
    handshake = checked_path(plan['worker_output'], exists=False)
    handshake.mkdir(mode=0o700)
    try:
        canary = handshake/'writer-canary.json'
        proof = publish(canary, dict(uid=os.getuid(), gid=os.getgid()))
        read(canary, proof['sha256']); canary.unlink()
        if canary.exists():
            raise ValueError('Writer canary removal failed')
        limits = resources()
        # Fresh process releases its CUDA context before the qualification starts.
        probe = ('import json,torch; p=torch.cuda.get_device_properties(0); '
                 'print(json.dumps(dict(count=torch.cuda.device_count(),name=p.name,'
                 'total_memory_bytes=p.total_memory,compute_capability=list(torch.cuda.get_device_capability(0)))))')
        result = subprocess.run([sys.executable, '-c', probe], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True, timeout=90)
        gpu = json.loads(result.stdout)
        validate_gpu(gpu)
        startup = dict(schema='credo.frontier.main-qualification-startup.v1', status='startup_guard_passed',
            run_id=plan['run_id'], plan_sha256=args.plan_sha256, worker_contract_sha256=plan['worker_contract']['sha256'],
            image=IMAGE, pod_name=pod_name, pod_uid=pod_uid, uid=os.getuid(), gid=os.getgid(),
            writer_canary_removed=True, gpu=gpu, resources=limits, real_data_opened=False, optimizer_calls=0)
        startup_id = publish(handshake/'STARTUP.json', startup)
        release_path = handshake/'STARTUP_RELEASE.json'
        while not release_path.exists():
            time.sleep(5)  # No whole-job/startup deadline or automatic retry.
        release_raw = read(release_path)
        release = json.loads(release_raw)
        same(release, dict(schema='credo.frontier.main-qualification-startup-release.v1',
            status='startup_released', run_id=plan['run_id'], lifecycle_plan_sha256=args.plan_sha256,
            worker_contract_sha256=plan['worker_contract']['sha256'], startup_sha256=startup_id['sha256'],
            pod_name=pod_name, pod_uid=pod_uid, image=IMAGE, admission_sha256=worker['admission']['sha256'],
            actual_pod_audit_passed=True, cgroup_resources_verified=True, scientific_fit_authorized=False))
        verify_bundle(plan, bundle)
        command = command_for(plan, worker, release_path, identity(release_raw)['sha256'])
        publish(handshake/'WORKER_INVOCATION.json', dict(command=command,
            contract_sha256=plan['worker_contract']['sha256'], release_sha256=identity(release_raw)['sha256']))
        completed = subprocess.run(command, env=dict(os.environ, PYTHONPATH=worker['source_root']),
                                   cwd=worker['source_root'])
        publish(handshake/'WORKER_EXIT.json', dict(returncode=completed.returncode, run_id=plan['run_id']))
        if completed.returncode:
            raise RuntimeError('Main qualification failed; no automatic retry')
        return 0
    except BaseException as error:
        publish(handshake/'ENTRY_FAILED.json', dict(exception_type=type(error).__name__,
            run_id=plan['run_id'], automatic_retry=False))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
