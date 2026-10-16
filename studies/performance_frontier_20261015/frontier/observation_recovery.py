"""Explicit, source-changing observation checkpoint import; never a launcher.

The failed parent remains read-only. A fresh child identity and immutable import
record identify mixed-source continuation. Only provenance/artifact references
change; numerical model, optimizer, RNG, sampler and audit values are retained.
Ordinary exact-source resume guards are not relaxed. Runtime resume must still
be qualified separately before any optimizer call.
"""

from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

import torch

from .checkpoints import SCHEMA, _BoundedCheckpointWriter, _cpu_tree, _runtime
from .retention import _path, _read_file, _sha, _signature, _sync, _write_once


SCHEMA_IMPORT = 'credo.frontier.observation-source-import.v1'
CONSUMED = 1280
MAXIMUM = 20000
AUDITS = (256, 512, 1024)
PAYLOAD_LIMIT = 256 * 1024**2
TOTAL_IMPORT_LIMIT = 512 * 1024**2


def _json_record(reference, expected_path):
    if reference.get('path') != str(expected_path):
        raise ValueError('Parent evidence path differs')
    _, _, raw = _read_file(expected_path, expected_sha=reference['sha256'],
                          size=reference['size_bytes'], maximum_bytes=4*1024**2)
    value = json.loads(raw)
    if value != reference['value']:
        raise ValueError('Parent evidence bytes/value differ')
    return value


def _tensor_record(reference, expected_path):
    if reference.get('path') != str(expected_path) or not 0 < reference['size_bytes'] <= PAYLOAD_LIMIT:
        raise ValueError('Exact bounded parent tensor path required')
    path = _path(expected_path)
    _, signature, _ = _read_file(path, expected_sha=reference['sha256'], size=reference['size_bytes'])
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        if _signature(os.fstat(stream.fileno())) != signature:
            raise ValueError('Parent tensor changed before restricted load')
        # Rehash this same handle: do not trust an earlier path-only hash when
        # deserializing a changed artifact. No unrestricted pickle is admitted.
        digest = hashlib.sha256()
        while block := stream.read(1024**2):
            digest.update(block)
        if digest.hexdigest() != reference['sha256']:
            raise ValueError('Parent tensor changed during load hashing')
        stream.seek(0)
        payload = torch.load(stream, map_location='cpu', weights_only=True)
        if (_signature(os.fstat(stream.fileno())) != signature
                or _signature(path.lstat()) != signature):
            raise ValueError('Parent tensor changed during restricted load')
    _cpu_tree(payload)  # Primitive containers and finite tensors only.
    return payload


def _publish_tensor(path, payload):
    with path.open('xb') as stream:
        writer = _BoundedCheckpointWriter(stream, PAYLOAD_LIMIT)
        try:
            torch.save(payload, writer)
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    _sync(path.parent)
    digest, signature, _ = _read_file(path)
    return {'relative_uri':path.name, 'sha256':digest, 'size_bytes':signature['st_size']}


def _identity(parent, child, checkpoint_sha, authority_sha):
    allowed = {'trajectory_id', 'source_sha256', 'worker_contract_sha256'}
    additions = {'recovery_parent_checkpoint_sha256', 'recovery_authority_sha256'}
    if set(child) != set(parent) | additions:
        raise ValueError('Child must carry complete parent identity and exact recovery lineage')
    if any(child[k] != v for k,v in parent.items() if k not in allowed):
        raise ValueError('Scientific identity changes are outside retention recovery')
    if any(child[k] == parent[k] for k in allowed):
        raise ValueError('Fresh trajectory/source/worker contract required')
    if (child['recovery_parent_checkpoint_sha256'] != _sha(checkpoint_sha)
            or child['recovery_authority_sha256'] != _sha(authority_sha)):
        raise ValueError('Recovery lineage binding differs')


def _validate_state(payload, parent_identity, scope):
    if (payload.get('schema') != SCHEMA or payload.get('identity') != parent_identity
            or payload.get('resume_capable') is not True or payload.get('runtime') != _runtime()
            or payload.get('execution_scope') != scope or payload.get('phase') != 'observation_fit'
            or payload.get('accepted_updates') != CONSUMED or payload.get('attempted_updates') != CONSUMED
            or payload.get('last_attempt_status') != 'accepted'):
        raise ValueError('Parent is not the exact healthy 1280-update observation checkpoint')
    cache = payload['count_cache']
    if (cache.get('schema') != 'credo.frontier.observation-engine.v1'
            or cache.get('count_cache_active') is not False or cache.get('optimizer_calls') != CONSUMED
            or [r['accepted_updates'] for r in cache['audit_history']] != list(AUDITS)
            or [r['accepted_updates'] for r in cache['training_history']] != list(range(1, CONSUMED+1))
            or payload['sampler_state']['position'] != CONSUMED*2048
            or sum(cache['exposure_by_source'].values()) != CONSUMED*2048
            or cache['selected_candidate'] != min(cache['audit_history'],
                                                   key=lambda r:(r['risk'],r['accepted_updates']))):
        raise ValueError('Observation history, selection or consumed-work accounting differs')
    _sha(cache['specification_sha256'])
    return cache


def _assert_parent_lock(path, descriptor, expected):
    """Keep the read lease bound to the existing named parent lock inode."""
    _path(path)
    for info in (os.fstat(descriptor), path.lstat()):
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_nlink != 1 or _signature(info) != expected):
            raise ValueError('Parent lock identity changed while importing')


def import_parent(lease, parent, *, authority_sha256, execution_scope='scientific'):
    """Import into an empty leased child; zero optimizer calls, no parent writes.

    ``parent`` is one byte-intake fit entry, including exact failure, checkpoint,
    candidate and incomplete-retirement records. Actual scheduler absence,
    admission/capacity and launch authority remain external prerequisites. This
    operation does not claim numerical resume compatibility or qualify a GPU.
    """
    lease.assert_active()
    _sha(authority_sha256)
    if execution_scope not in ('scientific', 'synthetic_cpu_only'):
        raise ValueError('Explicit import scope required')
    if (set(p.name for p in lease.root.iterdir()) != {'IDENTITY.json','OWNER.lock','claims','checkpoints'}
            or any((lease.root/'checkpoints').iterdir())):
        raise ValueError('Recovery requires a fresh empty child; no overwrite or retry')
    root = _path(Path(parent['identity']['path']).parent, directory=True)
    if root == lease.root or root.is_relative_to(lease.root) or lease.root.is_relative_to(root):
        raise ValueError('Disjoint parent and child roots required')
    if parent['consumed_updates'] != CONSUMED or parent['remaining_update_cap'] != MAXIMUM-CONSUMED:
        raise ValueError('Consumed updates cannot be reset')
    parent_identity = parent['identity']['value']
    _identity(parent_identity, lease.identity, parent['checkpoint_payload']['sha256'], authority_sha256)
    lock_path = _path(root/'OWNER.lock')
    expected_lock = _signature(lock_path.lstat())
    descriptor = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        check_parent_lock = lambda: _assert_parent_lock(lock_path, descriptor, expected_lock)
        check_parent_lock()
        # Linux NFS clients emulate flock using whole-file fcntl locks: an
        # exclusive lock requires write-open. The parent is deliberately
        # read-only, so take a shared lock, which still excludes its trainer's
        # exclusive lease. This is cooperative writer exclusion, not a defense
        # against actors that bypass locking or replace the lock path.
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        check_parent_lock()
        result = _import_locked(lease, root, parent, authority_sha256,
                                execution_scope, check_parent_lock)
        check_parent_lock()
        return result
    finally:
        os.close(descriptor)


def _import_locked(lease, root, parent, authority_sha, scope, check_parent_lock):
    ident = _json_record(parent['identity'], root/'IDENTITY.json')
    failure = _json_record(parent['failure'], root/'WORKER_FAILED.json')
    checkpoint = root/'checkpoints/update-000001280'
    complete = _json_record(parent['checkpoint_completion'], checkpoint/'COMPLETE.json')
    prepared = _json_record(parent['incomplete_retirement'],
                           root/'retention/transactions/after-update-000001280/PREPARED.json')
    if (failure['identity'] != ident or failure['exception_type'] != 'ValueError'
            or failure['exception_message'] != 'Unlinked payload inode readback failed'
            or failure['counters'] != dict(accepted_updates=CONSUMED, attempted_updates=CONSUMED,
                                           optimizer_calls=CONSUMED)
            or complete.get('schema') != SCHEMA or complete['identity'] != ident or complete['execution_scope'] != scope
            or complete['checkpoint_sha256'] != parent['checkpoint_payload']['sha256']
            or complete['checkpoint_bytes'] != parent['checkpoint_payload']['size_bytes']
            or complete['accepted_updates'] != CONSUMED or complete['attempted_updates'] != CONSUMED
            or complete['phase'] != 'observation_fit' or prepared['identity'] != ident
            or [r['relative_uri'] for r in prepared['candidates']] != ['checkpoints/update-000000768/checkpoint.pt']):
        raise ValueError('Parent failure/checkpoint/retirement outside this recovery')
    if (os.path.lexists(root/'checkpoints/update-000000768/checkpoint.pt')
            or os.path.lexists(root/'retention/transactions/after-update-000001280/DELETED.json')):
        raise ValueError('Parent retirement changed; do not rewrite or silently reconcile')
    payload = _tensor_record(parent['checkpoint_payload'], checkpoint/'checkpoint.pt')
    cache = _validate_state(payload, ident, scope)
    exports = parent['candidate_payloads']
    if len(exports) != len(AUDITS) or len({r['path'] for r in exports}) != len(AUDITS):
        raise ValueError('All prior technical-holdout candidates are required')
    source_candidates = []
    for report in cache['audit_history']:
        expected = root/f"observation-candidate-{report['accepted_updates']:09d}.pt"
        references = [r for r in exports if r['path'] == str(expected)]
        if len(references) != 1:
            raise ValueError('Exact candidate artifact missing')
        reference = references[0]
        if (report['artifact']['relative_uri'] != expected.name
                or any(report['artifact'][k] != reference[k] for k in ('sha256','size_bytes'))):
            raise ValueError('Candidate differs from saved selection history')
        candidate = _tensor_record(reference, expected)
        if (candidate.get('schema') != 'credo.frontier.observation-candidate.v1'
                or candidate.get('inference_only') is not True or candidate.get('identity') != ident
                or candidate.get('audit') != {k:v for k,v in report.items() if k != 'artifact'}):
            raise ValueError('Candidate payload provenance/audit differs')
        source_candidates.append((reference, candidate))
    intent = {'schema':SCHEMA_IMPORT, 'parent_identity':ident, 'child_identity':lease.identity,
              'parent_checkpoint':parent['checkpoint_payload'], 'authority_sha256':authority_sha,
              'parent_failure_sha256':parent['failure']['sha256'],
              'parent_prepared_retirement_sha256':parent['incomplete_retirement']['sha256'],
              'consumed_updates':CONSUMED, 'remaining_update_cap':MAXIMUM-CONSUMED,
              'maximum_updates_total':MAXIMUM, 'execution_scope':scope,
              'parent_changed':False, 'ordinary_resume_guards_changed':False,
              'maximum_import_bytes':TOTAL_IMPORT_LIMIT, 'scientific_optimizer_calls':0}
    check_parent_lock()
    _write_once(lease.root/'IMPORT_PREPARED.json', intent)
    imported = []
    total = 0
    # Deepcopy just provenance containers; leave tensors unchanged before their
    # restricted serialization. This does not reinitialize any trained state.
    new_cache = deepcopy(cache)
    for index,(reference,candidate) in enumerate(source_candidates):
        candidate = {**candidate, 'identity':lease.identity}
        new_ref = _publish_tensor(lease.root/Path(reference['path']).name, candidate)
        total += new_ref['size_bytes']
        if total > TOTAL_IMPORT_LIMIT-PAYLOAD_LIMIT:
            raise ValueError('Candidate import exceeds total byte reserve')
        new_cache['audit_history'][index]['artifact'] = new_ref
        imported.append({'parent':reference, 'child':new_ref})
    new_cache['selected_candidate'] = min(new_cache['audit_history'],
                                         key=lambda r:(r['risk'],r['accepted_updates']))
    child_payload = {**payload, 'identity':lease.identity, 'count_cache':new_cache}
    child_dir = lease.root/'checkpoints/update-000001280'
    child_dir.mkdir()
    child_ref = _publish_tensor(child_dir/'checkpoint.pt', child_payload)
    total += child_ref['size_bytes']
    if total > TOTAL_IMPORT_LIMIT:
        raise ValueError('Checkpoint import exceeds total byte reserve')
    child_complete = {**complete, 'identity':lease.identity, 'checkpoint_sha256':child_ref['sha256'],
                      'checkpoint_bytes':child_ref['size_bytes'], 'claim_id':lease.claim_id,
                      'worker_id':lease.worker_id}
    _write_once(child_dir/'COMPLETE.json', child_complete)
    _sync(child_dir.parent)
    result = {**intent, 'status':'imported_not_numerically_resumed', 'candidate_imports':imported,
              'checkpoint_directory':str(child_dir), 'checkpoint_sha256':child_ref['sha256'],
              'payload_bytes_written':total, 'numerical_resume_verified':False,
              'scientific_relaunch_authorized':False}
    check_parent_lock()
    _write_once(lease.root/'IMPORT_COMPLETE.json', result)
    return result
