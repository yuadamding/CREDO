"""Observation-only capacity admission, using retained exact-fileset parsing."""
import importlib.util
from pathlib import Path
from common import GIB, PROJECT, read

METADATA_SHA = 'a847dc3a31716a68f333155a07ca128842d6e04dbd25c8833b1de6608e695bf1'


def helpers():
    path = Path(__file__).absolute().parent / 'capacity_metadata.py'
    read(path, METADATA_SHA)
    spec = importlib.util.spec_from_file_location('observation_capacity_metadata', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assess(snapshot, plan, plan_sha, binding, admission):
    quota = helpers().parse_quota(snapshot['quota_output'])
    limits = [quota[key] for key in ('soft_bytes', 'hard_bytes') if quota[key] > 0]
    free = min(limits) - quota['used_bytes'] - quota['in_doubt_bytes'] if limits else -1
    file_limits = [quota[key] for key in ('soft_files', 'hard_files') if quota[key] > 0]
    free_files = min(file_limits) - quota['used_files'] - quota['in_doubt_files'] if file_limits else None
    # Conservative: reserve the FULL aggregate reservation even if some bytes
    # are already present in filesystem usage. Never use dated free-space alone.
    required = admission['aggregate_reserved_output_bytes'] + 16 * GIB
    reasons = []
    if snapshot.get('project_root') != PROJECT or snapshot.get('fileset') != 'h_bcb-yding4':
        reasons.append('project_fileset_mismatch')
    if snapshot.get('quota_block_size_bytes') != 1024 or snapshot.get('remote_writes') is not False or snapshot.get('numerical_payloads_opened') is not False:
        reasons.append('invalid_metadata_snapshot')
    if quota['soft_bytes'] <= 0 or quota['block_grace'] != 'none' or quota['file_grace'] != 'none':
        reasons.append('quota_not_clear')
    if free < required or snapshot.get('filesystem_available_bytes', -1) < required:
        reasons.append('insufficient_aggregate_headroom')
    if snapshot.get('filesystem_available_inodes', -1) < 20000 or (free_files is not None and free_files < 20000):
        reasons.append('insufficient_inodes')
    return dict(schema='credo.frontier.observation-capacity.v1', status='passed' if not reasons else 'held',
                reasons=reasons, run_id=plan['run_id'], plan_sha256=plan_sha, uid=binding.get('uid'),
                job=binding.get('job', plan['run_id']), snapshot=snapshot, quota=quota,
                required_free_bytes=required, fileset_headroom_bytes=free,
                admission_sha256=plan['admission']['sha256'], automatic_successor=False)


def collect_capacity(plan, plan_sha, binding, admission):
    module = helpers()
    module.PROJECT = PROJECT
    module.assess = lambda snapshot, ignored_plan, ignored_sha, ignored_binding: snapshot
    return assess(module.collect_capacity(plan, plan_sha, binding), plan, plan_sha, binding, admission)


def validate_capacity(receipt, plan, plan_sha, binding, admission):
    expected = assess(receipt['snapshot'], plan, plan_sha, binding, admission)
    if receipt != expected or expected['status'] != 'passed':
        raise ValueError('Observation capacity gate failed')
    return receipt
