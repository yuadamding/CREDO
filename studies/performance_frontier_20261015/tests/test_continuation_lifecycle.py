"""Fabricated metadata only: no model loading, optimizer, GPU or remote calls."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from deployment import common


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def adapter(name, monkeypatch):
    monkeypatch.setitem(sys.modules, 'common', common)
    spec = importlib.util.spec_from_file_location('fixture_continuation_'+name, ROOT/'deployment'/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = common.encoded(value)
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def signatures(root):
    fields = ('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
    return {p.relative_to(root).as_posix(): {k:getattr(p.stat(),k) for k in fields}
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def context(tmp_path, monkeypatch):
    control = adapter('control', monkeypatch)
    monkeypatch.setattr(control, 'PARENT_OWNER_UID', os.getuid())
    parent, child, bundle = (tmp_path/name for name in ('parent','child','bundle'))
    for path in (parent,child,bundle): path.mkdir()
    lease = dict(schema='credo.frontier.native-input-lease.v1', dataset_root='/fixture/counts',
        access_sha256='b'*64, metadata_index_sha256='c'*64, source_ids=['source1'],
        payload_hashes_verified_on_consumption=True, all_payloads_consumed_claimed=False,
        members={'fixture':dict(sha256='d'*64,file_identity=dict(st_size=7,st_dev=1,st_ino=2))})
    old_lease = write(parent/'INPUT_LEASE.json',lease)
    parent_id = dict(trajectory_id='old',fold_id='subset-D1-D2',seed=0,input_sha256='b'*64,
        configuration_sha256='e'*64,source_sha256='f'*64,worker_contract_sha256='1'*64)
    parent_record = dict(identity=dict(path=str(parent/'IDENTITY.json'),value=parent_id),
        checkpoint_payload=dict(path=str(parent/'checkpoints/update-000001280/checkpoint.pt'),
                                sha256='2'*64,size_bytes=123))
    intake = write(bundle/'PARENT_INTAKE.json',dict(fits={'D1D2':parent_record}))
    recorded = signatures(parent)
    parent_bytes = sum(r['st_size'] for r in recorded.values())
    inventory_ref = write(bundle/'PARENT_INVENTORY.json',dict(fits={'D1D2':dict(
        inventory=recorded,parent_output_bytes=parent_bytes)}))
    fit_ref = write(bundle/'WORKER_CONTRACT.json',dict(fixture=True))
    worker = dict(trajectory_id='new',worker_id='worker-new',source_manifest_sha256='3'*64,
        input_access_sha256='b'*64,dataset_root='/fixture/counts',metadata_index_sha256='c'*64)
    recovery = dict(fit_contract=fit_ref,pair='D1D2',parent_intake=intake,parent_inventory=inventory_ref,
        parent_input_lease=old_lease,authority=dict(path='/fixture/authority',sha256='4'*64,size_bytes=1),
        consumed_updates=1280,maximum_new_updates=18720,maximum_total_updates=20000,
        parent_retained_output_bytes=parent_bytes,combined_output_limit_bytes=4*GIB)
    recovery_ref = write(bundle/'RECOVERY.json',recovery)
    # Worker-local physical identities differ legitimately from parent intake.
    new_lease = deepcopy(lease);new_lease['members']['fixture']['file_identity'].update(st_dev=99,st_ino=101)
    input_ref = write(child/'INPUT_LEASE.json',new_lease)
    claim='5'*32
    exit_value = dict(schema='credo.frontier.native-input-lease-exit.v1',status='unchanged',
        input_lease_sha256=hashlib.sha256(json.dumps(new_lease,sort_keys=True,separators=(',',':'),
            ensure_ascii=False,allow_nan=False).encode()).hexdigest(),access_sha256='b'*64,
        checked_members=1,additional_count_array_scans=0,same_worker_identity_check=True)
    exit_ref = write(child/'input-exits'/(claim+'.json'),exit_value)
    write(child/'claims'/(claim+'.json'),dict(claim_id=claim,worker_id='worker-new',trajectory_id='new'))
    write(child/'claims'/(claim+'.released.json'),dict(claim_id=claim,exception_type=None,automatic_restart_authorized=False))
    ident=dict(parent_id,trajectory_id='new',source_sha256='3'*64,worker_contract_sha256=fit_ref['sha256'],
        recovery_parent_checkpoint_sha256='2'*64,recovery_authority_sha256='4'*64)
    complete=dict(identity=ident,accepted_updates=20000,attempted_updates=20000,scientific_updates=20000,
        resume_from_accepted_updates=1280,consumed_parent_updates=1280,new_scientific_updates_this_owner=18720,
        source_continuation_contract=recovery_ref,mixed_source_continuation=True,input_lease=input_ref,input_exit=exit_ref)
    files={p.relative_to(child).as_posix():common.identity(p.read_bytes()) for p in child.rglob('*') if p.is_file()}
    plan=dict(mode='fit',worker_contract=fit_ref,recovery_contract=recovery_ref,
        scientific_output=str(child),maximum_output_bytes=4*GIB-parent_bytes,
        combined_output_limit_bytes=4*GIB,parent_retained_output_bytes=parent_bytes)
    return control,plan,worker,complete,files,parent,child,recovery


def test_continuation_command_is_explicit_and_normal_path_unchanged(context,monkeypatch):
    _,plan,worker,*_=context
    entry=adapter('worker_entry',monkeypatch)
    command=entry.command_for(plan,worker)
    assert command[:3]==['/opt/credo/bin/python','-m','frontier.observation_continuation']
    assert command[-4:]==['--recovery-contract',plan['recovery_contract']['path'],
                          '--recovery-contract-sha256',plan['recovery_contract']['sha256']]
    normal=dict(plan);normal.pop('recovery_contract')
    assert entry.command_for(normal,worker)[:4]==['/opt/credo/bin/python','-m','frontier.observation_worker','fit']
    with pytest.raises(ValueError):entry.command_for(dict(plan,mode='qualify'),worker)


def test_continuation_accepts_claim_exit_and_cumulative_accounting(context):
    control,plan,worker,complete,files,parent,child,_=context
    assert not (child/'INPUT_EXIT.json').exists()
    result=control.validate_continuation_result(plan,worker,complete,files)
    assert result['new_scientific_updates_this_owner']==18720
    assert result['cumulative_scientific_updates']==20000
    assert result['combined_output_bytes']==result['parent_output_bytes']+result['child_output_bytes']
    assert result['parent_payloads_opened'] is False


@pytest.mark.parametrize('key,value',[('accepted_updates',18720),('attempted_updates',20001),
    ('resume_from_accepted_updates',0),('consumed_parent_updates',0),
    ('new_scientific_updates_this_owner',20000),('mixed_source_continuation',False)])
def test_wrong_completion_accounting_rejected(context,key,value):
    control,plan,worker,complete,files,*_=context
    changed=deepcopy(complete);changed[key]=value
    with pytest.raises(ValueError):control.validate_continuation_result(plan,worker,changed,files)


@pytest.mark.parametrize('damage',['lineage','input_hash','exit_path','exit_hash','claim_failure',
                                  'parent_changed','combined_cap','readback_size'])
def test_continuation_evidence_failures_rejected(context,damage):
    control,plan,worker,complete,files,parent,child,recovery=context
    plan,complete,files=deepcopy(plan),deepcopy(complete),deepcopy(files)
    if damage=='lineage':complete['identity']['recovery_parent_checkpoint_sha256']='a'*64
    elif damage=='input_hash':complete['input_lease']['sha256']='a'*64
    elif damage=='exit_path':complete['input_exit']['path']=str(child/'INPUT_EXIT.json')
    elif damage=='exit_hash':complete['input_exit']['sha256']='a'*64
    elif damage=='claim_failure':
        write(child/'claims'/('5'*32+'.released.json'),dict(claim_id='5'*32,exception_type='RuntimeError',automatic_restart_authorized=False))
    elif damage=='parent_changed':(parent/'unexpected.txt').write_text('changed')
    elif damage=='combined_cap':plan['maximum_output_bytes']+=1
    else:files['INPUT_LEASE.json']['size_bytes']+=1
    with pytest.raises((ValueError,RuntimeError)):
        control.validate_continuation_result(plan,worker,complete,files)


def test_parent_inventory_portable_ids_but_exact_content_stats(context):
    control,_,_,_,_,parent,*_=context
    recorded=signatures(parent)
    for item in recorded.values():item.update(st_dev=111,st_ino=222)
    assert control.parent_inventory(parent,recorded)==sum(x['st_size'] for x in recorded.values())
    recorded['INPUT_LEASE.json']['st_mtime_ns']+=1
    with pytest.raises(ValueError):control.parent_inventory(parent,recorded)


def test_controller_accepts_real_claim_layout_without_fabricating_root_exit(context,monkeypatch,tmp_path):
    control,plan,worker,complete,files,parent,child,_=context
    handshake=tmp_path/'handshake';handshake.mkdir()
    controller=tmp_path/'controller';controller.mkdir()
    plan=dict(plan,run_id='fixture-worker',worker_output=str(handshake))
    complete=dict(complete,schema='credo.frontier.observation-worker-completion.v1',status='complete',
        scope='scientific',actual_disposable_optimizer_calls=0,main_dynamics_fitted=False,
        held_out_donor_outcomes_opened=False)
    complete_ref=write(child/'COMPLETE.json',complete)
    files['COMPLETE.json']={k:complete_ref[k] for k in ('sha256','size_bytes')}
    release=dict(uid='job',pod_uid='pod',recovery_contract_sha256=plan['recovery_contract']['sha256'])
    write(controller/'startup-validated.json',release)
    write(handshake/'STARTUP_RELEASE.json',release)
    write(handshake/'WORKER_EXIT.json',dict(returncode=0,run_id='fixture-worker'))
    monkeypatch.setattr(control,'verify_bundle',lambda *args:(worker,{}))
    monkeypatch.setattr(control,'inventory',lambda *args:files)
    monkeypatch.setattr(control,'core',SimpleNamespace(publish=write))
    result=control.validate_result(plan,dict(uid='job'),controller,[dict(metadata=dict(uid='pod'))])
    assert result==complete_ref['sha256']
    readback=json.loads((controller/'output-readback.json').read_text())
    assert readback['status']=='passed' and readback['cumulative_scientific_updates']==20000
    assert not (child/'INPUT_EXIT.json').exists()


def test_changed_adapters_remain_python36_host_metadata_only():
    for name in ('control.py','worker_entry.py'):
        source=(ROOT/'deployment'/name).read_text()
        ast.parse(source,feature_version=(3,6))
        assert 'import torch' not in source
