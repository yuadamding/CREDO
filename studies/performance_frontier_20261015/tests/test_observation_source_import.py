"""Fabricated checkpoint labels only: no optimizer.step, biological input or GPU."""

from copy import deepcopy
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from frontier.checkpoints import TrajectoryLease, load_checkpoint, save_checkpoint
from frontier.observation_recovery import import_parent
from frontier import observation_recovery as recovery


def ref(path, value=False):
    raw = path.read_bytes()
    result = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    if value:
        result['value'] = json.loads(raw)
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return ref(path, True)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left,right)
    if isinstance(left, dict):
        return left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):
        return type(left) is type(right) and len(left)==len(right) and all(map(lambda p:equal(*p),zip(left,right)))
    return left==right


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *a, **k:pytest.fail('Zero optimizer-call test'))
    ident = dict(trajectory_id='fabricated-parent', fold_id='subset-D1-D2', seed=0,
                 source_sha256='a'*64, configuration_sha256='b'*64,input_sha256='c'*64,
                 worker_contract_sha256='d'*64)
    model = torch.nn.Linear(3,2)
    optimizer = torch.optim.AdamW(model.parameters())
    root = tmp_path/'parent'
    audits, exports = [], []
    with TrajectoryLease(root,ident,'fixture-parent') as lease:
        for update,risk in ((256,3.0),(512,1.0),(1024,2.0)):
            report = dict(accepted_updates=update,risk=risk)
            path = root/f'observation-candidate-{update:09d}.pt'
            torch.save(dict(schema='credo.frontier.observation-candidate.v1', inference_only=True,
                            identity=ident,audit=report,model_state=model.state_dict()),path)
            reference = ref(path)
            exports.append(reference)
            audits.append({**report,'artifact':dict(relative_uri=path.name,sha256=reference['sha256'],
                                                    size_bytes=reference['size_bytes'])})
        state = dict(schema='credo.frontier.observation-engine.v1',count_cache_active=False,
                     specification_sha256='e'*64,optimizer_calls=1280,audit_history=audits,
                     training_history=[dict(accepted_updates=n,loss=1.0) for n in range(1,1281)],
                     exposure_by_source={'fabricated':1280*2048},selected_candidate=audits[1])
        ck = save_checkpoint(lease,model=model,optimizer=optimizer,scheduler=None,accepted_updates=1280,
             attempted_updates=1280,phase='observation_fit',sampler_state=dict(position=1280*2048),
             count_cache=state,execution_scope='synthetic_cpu_only')
    failure = write_json(root/'WORKER_FAILED.json',dict(identity=ident,exception_type='ValueError',
               exception_message='Unlinked payload inode readback failed',counters=dict(accepted_updates=1280,
               attempted_updates=1280,optimizer_calls=1280)))
    prepared = write_json(root/'retention/transactions/after-update-000001280/PREPARED.json',
                         dict(identity=ident,candidates=[dict(relative_uri='checkpoints/update-000000768/checkpoint.pt')]))
    parent = dict(identity=ref(root/'IDENTITY.json',True),failure=failure,
                  checkpoint_completion=ref(Path(ck['directory'])/'COMPLETE.json',True),
                  checkpoint_payload=ref(Path(ck['directory'])/'checkpoint.pt'),candidate_payloads=exports,
                  incomplete_retirement=prepared,consumed_updates=1280,remaining_update_cap=18720)
    child_ident = {**ident,'trajectory_id':'fabricated-child','source_sha256':'f'*64,
                   'worker_contract_sha256':'1'*64,'recovery_authority_sha256':'2'*64,
                   'recovery_parent_checkpoint_sha256':parent['checkpoint_payload']['sha256']}
    return root,parent,child_ident,model,optimizer


def test_import_preserves_numerical_state_parent_history_and_exact_resume(source,tmp_path):
    root,parent,ident,model,optimizer = source
    before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    old = torch.load(parent['checkpoint_payload']['path'],weights_only=True)
    with TrajectoryLease(tmp_path/'child',ident,'fixture-child') as lease:
        result = import_parent(lease,parent,authority_sha256='2'*64,execution_scope='synthetic_cpu_only')
        new = torch.load(Path(result['checkpoint_directory'])/'checkpoint.pt',weights_only=True)
        for key in old.keys()-{'identity','count_cache'}:
            assert equal(old[key],new[key]),key
        for key in old['count_cache'].keys()-{'audit_history','selected_candidate'}:
            assert equal(old['count_cache'][key],new['count_cache'][key]),key
        assert new['count_cache']['selected_candidate']['accepted_updates']==512
        for a,b in zip(old['count_cache']['audit_history'],new['count_cache']['audit_history']):
            assert {k:v for k,v in a.items() if k!='artifact'}=={k:v for k,v in b.items() if k!='artifact'}
            imported = torch.load(lease.root/b['artifact']['relative_uri'],weights_only=True)
            original = torch.load(root/a['artifact']['relative_uri'],weights_only=True)
            assert imported['identity']==ident and equal(imported['model_state'],original['model_state'])
        state = load_checkpoint(lease,result['checkpoint_directory'],expected_sha256=result['checkpoint_sha256'],
                                model=model,optimizer=optimizer,scheduler=None,execution_scope='synthetic_cpu_only')
        assert state['accepted_updates']==1280 and result['remaining_update_cap']==18720
        assert result['scientific_optimizer_calls']==0 and result['scientific_relaunch_authorized'] is False
        with pytest.raises(ValueError,match='fresh empty'):
            import_parent(lease,parent,authority_sha256='2'*64,execution_scope='synthetic_cpu_only')
    after = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before==after


@pytest.mark.parametrize('damage',['source_unchanged','fold','authority','checkpoint_hash','budget_reset',
                                  'candidate_missing','candidate_corrupt','failure_hash','retirement_republished'])
def test_malformed_import_is_rejected_without_parent_mutation(source,tmp_path,damage):
    root,parent,ident,model,optimizer = source
    parent,ident = deepcopy(parent),dict(ident)
    if damage=='source_unchanged':ident['source_sha256']=parent['identity']['value']['source_sha256']
    elif damage=='fold':ident['fold_id']='subset-D3-D4'
    elif damage=='authority':ident['recovery_authority_sha256']='3'*64
    elif damage=='checkpoint_hash':parent['checkpoint_payload']['sha256']='4'*64
    elif damage=='budget_reset':parent['consumed_updates']=0
    elif damage=='candidate_missing':parent['candidate_payloads'].pop()
    elif damage=='candidate_corrupt':Path(parent['candidate_payloads'][0]['path']).write_bytes(b'bad')
    elif damage=='failure_hash':parent['failure']['sha256']='5'*64
    else:(root/'retention/transactions/after-update-000001280/DELETED.json').write_text('{}')
    before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with TrajectoryLease(tmp_path/'child',ident,'fixture-child') as lease:
        with pytest.raises(ValueError):
            import_parent(lease,parent,authority_sha256='2'*64,execution_scope='synthetic_cpu_only')
        assert not (lease.root/'IMPORT_COMPLETE.json').exists()
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_active_parent_lock_prevents_import(source,tmp_path):
    root,parent,ident,model,optimizer = source
    with TrajectoryLease(root,parent['identity']['value'],'other-fixture-owner'):
        with TrajectoryLease(tmp_path/'child',ident,'fixture-child') as lease:
            with pytest.raises(BlockingIOError):
                import_parent(lease,parent,authority_sha256='2'*64,execution_scope='synthetic_cpu_only')
            assert not (lease.root/'IMPORT_PREPARED.json').exists()


def test_parent_lock_is_read_only_shared_and_nonblocking(source,tmp_path,monkeypatch):
    root,parent,ident,model,optimizer = source
    original_flock = fcntl.flock
    parent_inode = (root/'OWNER.lock').stat().st_ino
    seen = []

    def nfs_like_flock(descriptor, operation):
        if os.fstat(descriptor).st_ino == parent_inode:
            access = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
            if access == os.O_RDONLY and operation & fcntl.LOCK_EX:
                raise OSError(errno.EBADF, 'NFS exclusive lock requires write-open')
            assert access == os.O_RDONLY
            assert operation == fcntl.LOCK_SH | fcntl.LOCK_NB
            seen.append(operation)
        return original_flock(descriptor, operation)

    with TrajectoryLease(tmp_path/'child',ident,'fixture-child') as lease:
        monkeypatch.setattr(fcntl, 'flock', nfs_like_flock)
        result = import_parent(lease,parent,authority_sha256='2'*64,execution_scope='synthetic_cpu_only')
        assert result['scientific_optimizer_calls'] == 0
    assert seen == [fcntl.LOCK_SH | fcntl.LOCK_NB]


def test_parent_read_lease_excludes_new_exclusive_writer(source,tmp_path,monkeypatch):
    root,parent,ident,model,optimizer = source
    original_import = recovery._import_locked
    checked = []

    def while_parent_locked(*args, **kwargs):
        descriptor = os.open(root/'OWNER.lock', os.O_RDWR | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checked.append(True)
        finally:
            os.close(descriptor)
        return original_import(*args, **kwargs)

    monkeypatch.setattr(recovery, '_import_locked', while_parent_locked)
    with TrajectoryLease(tmp_path/'child',ident,'fixture-child') as lease:
        import_parent(lease,parent,authority_sha256='2'*64,execution_scope='synthetic_cpu_only')
    assert checked == [True]


@pytest.mark.parametrize('stage', ['before_open', 'after_lock', 'after_read', 'during_publication'])
def test_parent_lock_replacement_is_rejected(source,tmp_path,monkeypatch,stage):
    root,parent,ident,model,optimizer = source
    path = root/'OWNER.lock'
    parent_inode = path.stat().st_ino
    replaced = []

    def replace_lock():
        if not replaced:
            path.rename(root/'fixture-displaced-parent.lock')
            path.write_bytes(b'')
            replaced.append(True)

    original_open, original_flock = os.open, fcntl.flock
    original_read, original_publish = recovery._tensor_record, recovery._publish_tensor

    def before_open(filename, flags, *args, **kwargs):
        if Path(filename) == path:
            replace_lock()
        return original_open(filename, flags, *args, **kwargs)

    def after_lock(descriptor, operation):
        result = original_flock(descriptor, operation)
        if os.fstat(descriptor).st_ino == parent_inode:
            replace_lock()
        return result

    def after_read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        replace_lock()
        return result

    def during_publication(*args, **kwargs):
        result = original_publish(*args, **kwargs)
        replace_lock()
        return result

    with TrajectoryLease(tmp_path/'child',ident,'fixture-child') as lease:
        if stage == 'before_open':
            monkeypatch.setattr(os, 'open', before_open)
        elif stage == 'after_lock':
            monkeypatch.setattr(fcntl, 'flock', after_lock)
        elif stage == 'after_read':
            monkeypatch.setattr(recovery, '_tensor_record', after_read)
        else:
            monkeypatch.setattr(recovery, '_publish_tensor', during_publication)
        with pytest.raises(ValueError,match='Parent lock identity changed'):
            import_parent(lease,parent,authority_sha256='2'*64,execution_scope='synthetic_cpu_only')
        assert not (lease.root/'IMPORT_COMPLETE.json').exists()
        if stage != 'during_publication':
            assert not (lease.root/'IMPORT_PREPARED.json').exists()
    assert replaced == [True]
