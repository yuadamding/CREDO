"""Zero-update storage canary for an explicitly admitted worker output root.

Creates only bounded literal fixtures. It exercises the old open-unlink readback
without assuming what an actual mount returns, then executes real corrected
prepare/commit retirement. This is not GPU or scientific fitting qualification.
"""

import os
from pathlib import Path

from .checkpoints import SCHEMA, TrajectoryLease
from .retention import (STUDY_ID, RetentionPolicy, _path, _read_file, _signature,
                        _sync, _write_once, bind_policy, commit_retirement,
                        identity_sha256, prepare_retirement)


def run_canary(output_root):
    root = _path(Path(output_root).absolute())
    if root.exists():
        raise FileExistsError('Fresh canary root required; no retry or overwrite')
    _path(root.parent, directory=True)
    root.mkdir()
    # A disposable observation of the failed invariant, not a replay of an
    # actual checkpoint or an attempt to alter failed retirement history.
    legacy = root/'open-unlink-literal.bin'
    legacy.write_bytes(b'credo-retention-canary\n')
    descriptor = os.open(legacy, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = _signature(os.fstat(descriptor))
        os.unlink(legacy)
        _sync(root)
        after = _signature(os.fstat(descriptor))
        absent_while_open = not os.path.lexists(legacy)
        old_check = (after['st_nlink']==0 and after['st_ino']==before['st_ino']
                     and after['st_size']==before['st_size'])
    finally:
        os.close(descriptor)
    _sync(root)
    legacy_result = {'before':before, 'after_unlink_while_open':after,
                     'pathname_absent_while_open':absent_while_open,
                     'pathname_absent_after_close':not os.path.lexists(legacy),
                     'historical_open_inode_check_passed':old_check}
    _write_once(root/'LEGACY_INVARIANT_OBSERVATION.json', legacy_result)
    fixture_study = root/STUDY_ID
    fixture_study.mkdir()
    identity = dict(trajectory_id='literal-storage-canary',fold_id='fabricated',seed=0,
                    source_sha256='a'*64,input_sha256='b'*64,configuration_sha256='c'*64,
                    fixture_note='Literal file labels, not executed optimizer updates')
    with TrajectoryLease(fixture_study/'trajectory',identity,'storage-canary') as lease:
        for label in range(6):
            directory = lease.root/'checkpoints'/f'update-{label:09d}'
            directory.mkdir()
            payload = directory/'checkpoint.pt'
            payload.write_bytes(bytes([label])*2048)
            digest,signature,_ = _read_file(payload)
            _write_once(directory/'COMPLETE.json',dict(schema=SCHEMA,identity=identity,
                accepted_updates=label,attempted_updates=label,milestone=label==1,
                checkpoint_sha256=digest,checkpoint_bytes=signature['st_size'],
                phase='literal_storage_fixture',execution_scope='synthetic_cpu_only'))
        policy = RetentionPolicy(str(fixture_study),str(lease.root),identity_sha256(identity),(2,))
        bind_policy(lease,policy)
        prepared = prepare_retirement(lease,policy)
        deleted = commit_retirement(lease,policy,prepared)
        if [r['accepted_updates'] for r in deleted['removed']] != [0,3]:
            raise ValueError('Literal retirement set differs')
        if prepare_retirement(lease,policy) is not None:
            raise ValueError('Retirement did not reconcile on next inventory')
    result = dict(schema='credo.frontier.retention-storage-canary.v1',status='passed',
                  legacy_observation=legacy_result,corrected_retirement=deleted,
                  optimizer_calls=0,scientific_updates=0,real_checkpoint_reads=0,
                  gpu_runtime_qualified=False,quota_reclaim_claimed=False,
                  automatic_successor=False)
    _write_once(root/'COMPLETE.json',result)
    return result
