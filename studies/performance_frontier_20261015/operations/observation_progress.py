"""Bounded metadata-only continuation progress; source is remote Python 3.6 safe.

Execute this source in an already exact-UID/plan-verified status scope, then call
observation_progress(plan). No scheduler, checkpoint, array or log access occurs.
The reported milestone is a lower bound, not terminal scientific acceptance.
"""

import hashlib
import json
import os
import stat


MAXIMUM_JSON_BYTES = 1024 * 1024
MILESTONES = tuple(sorted(set(range(256, 20001, 256)) | {20000}, reverse=True))


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate metadata key')
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError('Nonfinite metadata number: ' + value)


def read_progress_json(path, maximum=MAXIMUM_JSON_BYTES, owner_uid=307469):
    """Read one regular, owned, nonsymlink file, bounded before/after opening."""
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return None
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != owner_uid
            or not 0 < before.st_size <= maximum):
        raise ValueError('Progress metadata type/owner/size differs: ' + path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(descriptor)
        fields = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        signature = lambda info: tuple(getattr(info, key) for key in fields)
        if signature(opened) != signature(before):
            raise ValueError('Progress metadata changed during open')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        if (len(raw) != before.st_size or signature(os.fstat(descriptor)) != signature(before)
                or signature(os.lstat(path)) != signature(before)):
            raise ValueError('Progress metadata changed during read')
    finally:
        os.close(descriptor)
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_object,
                       parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError('Progress metadata must be a JSON object')
    return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw), value=value)


def observation_progress(plan, reader=None, owner_uid=307469):
    """Return latest validated declared milestone, or unavailable (never zero).

    An injected reader takes (path, maximum) and returns sha256/size_bytes/value
    or None. Callers must bind the plan itself before invoking this helper.
    """
    if reader is None:
        reader = lambda path, maximum: read_progress_json(path, maximum, owner_uid)
    output = plan['scientific_output']
    bundle = plan['bundle_root']
    if (plan.get('mode') != 'fit' or plan.get('maximum_new_scientific_updates') != 18720
            or not isinstance(output, str) or not output.startswith('/')
            or os.path.normpath(output) != output or os.path.normpath(bundle) != bundle):
        raise ValueError('Expected bound observation continuation plan')
    result = dict(schema='credo.frontier.observation-progress-status.v1',
                  run_id=plan['run_id'], status='unavailable', remote_writes=False,
                  metadata_only=True, terminal_acceptance=False,
                  accepted_updates=None, new_scientific_updates_this_owner=None,
                  consumed_parent_updates=1280, target_updates=20000)

    def bound(name, filename):
        reference = plan[name]
        if reference['path'] != bundle + '/' + filename:
            raise ValueError('Bound metadata path differs: ' + name)
        record = reader(reference['path'], MAXIMUM_JSON_BYTES)
        if record is not None and any(record[key] != reference[key] for key in ('sha256', 'size_bytes')):
            raise ValueError('Bound metadata bytes differ: ' + name)
        return record

    worker_record = bound('worker_contract', 'WORKER_CONTRACT.json')
    if worker_record is None:
        result['reason'] = 'worker_contract_unavailable'
        return result
    worker = worker_record['value']
    expected_worker = dict(schema='credo.frontier.observation-worker-contract.v1',
        worker_id=plan['run_id'], mode='fit', output_root=output,
        source_manifest=plan['source_manifest']['path'],
        source_manifest_sha256=plan['source_manifest']['sha256'])
    if any(worker.get(key) != value for key, value in expected_worker.items()):
        raise ValueError('Worker contract identity differs')
    source = bound('source_manifest', 'SOURCE_MANIFEST.json')
    if source is None:
        result['reason'] = 'source_manifest_unavailable'
        return result
    if source['value'].get('schema') != 'credo.frontier.observation-worker-source.v1':
        raise ValueError('Source manifest schema differs')
    identity_record = reader(output + '/IDENTITY.json', MAXIMUM_JSON_BYTES)
    if identity_record is None:
        result['reason'] = 'identity_unavailable'
        return result
    identity = identity_record['value']
    expected_identity = dict(trajectory_id=worker['trajectory_id'], fold_id=worker['fold_id'], seed=0,
        input_sha256=worker['input_access_sha256'], source_sha256=plan['source_manifest']['sha256'],
        worker_contract_sha256=plan['worker_contract']['sha256'])
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError('Output identity differs')
    for updates in MILESTONES:
        if updates <= 1280:
            continue
        path = output + '/progress/update-%09d.json' % updates
        record = reader(path, MAXIMUM_JSON_BYTES)
        if record is None:
            continue
        progress = record['value']
        expected = dict(schema='credo.frontier.observation-worker-progress.v1', identity=identity,
            consumed_parent_updates=1280, accepted_updates=updates, attempted_updates=updates,
            actual_optimizer_calls=updates, new_scientific_updates_this_owner=updates-1280,
            source_continuation_contract=plan['recovery_contract'])
        if any(progress.get(key) != value for key, value in expected.items()):
            raise ValueError('Published continuation milestone differs: ' + path)
        for key in ('consumed_parent_updates', 'accepted_updates', 'attempted_updates',
                    'actual_optimizer_calls', 'new_scientific_updates_this_owner'):
            if type(progress[key]) is not int:
                raise ValueError('Progress counter must be an integer: ' + key)
        result.update(status='published_milestone', accepted_updates=updates,
            attempted_updates=updates, actual_optimizer_calls=updates,
            new_scientific_updates_this_owner=updates-1280,
            remaining_updates_to_target=20000-updates, at_utc=progress.get('at_utc'),
            progress=dict(path=path, sha256=record['sha256'], size_bytes=record['size_bytes']))
        return result
    result['reason'] = 'no_post_parent_milestone_published'
    return result
