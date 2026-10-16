"""Local metadata and small tensor checks; zero optimizer calls and no real loads."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
from deployment import recovery_check_build as build, recovery_check_common as common
from deployment import recovery_check_entry as entry, recovery_check_control as control
from deployment.common import encoded, identity
from frontier.observation_recovery_check import exact_tree, forbidden_step
from test_observation_deployment import RETAINED

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def packaged(tmp_path,monkeypatch):
    home=tmp_path/'home'; scratch=tmp_path/'scratch'
    for module in (build,common):
        monkeypatch.setattr(module,'PROJECT',str(home));monkeypatch.setattr(module,'CONTROL',str(scratch))
    authority=json.loads((ROOT/'receipts/OBSERVATION_RETENTION_CHECK_AUTHORITY_20261016_V1.json').read_bytes())
    authority.update(run_id=common.RUN,phase=common.PHASE,fixture_only_not_real_authority=True)
    authority_raw=encoded(authority)
    intake=(ROOT/'receipts/OBSERVATION_RETENTION_RECOVERY_BYTE_INTAKE_20261016_V1.json').read_bytes()
    _,source=build.source_payload(ROOT)
    admission=dict(schema='credo.frontier.retention-check-admission.v1',status='reserved',run_id=common.RUN,
        work_binding=dict(kind='qualification',identity=[identity(authority_raw)['sha256']],start_updates=0,target_updates=0),
        phase_authority_sha256=identity(authority_raw)['sha256'],source_manifest_sha256=identity(source)['sha256'],
        scientific_update_reservation=0,storage_reserved_bytes=4*1024**3,disposable_optimizer_call_limit=0,
        automatic_retry=False,automatic_successor=False,journal_event_sha256='a'*64,
        journal_identity_sha256='b'*64,capacity_receipt_sha256='c'*64,
        live_h100_reservations=1,aggregate_reserved_output_bytes=32*1024**3)
    payload,archive,report=build.construct(ROOT,RETAINED,authority_raw,intake,admission)
    bundle=home/common.PHASE/'bundle'
    for n,raw in payload.items():
        p=bundle/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    return payload,archive,report,bundle


def test_real_surface_and_full_bound_source(packaged):
    payload,archive,report,bundle=packaged
    plan=json.loads(payload['PLAN.json']);worker,admission=common.verify_bundle(plan,bundle)
    assert plan['maximum_output_bytes']==4*1024**3
    assert plan['maximum_new_scientific_updates']==plan['planned_disposable_optimizer_steps']==0
    assert worker['optimizer_calls']==0
    assert 'dataset_root' not in worker
    assert set(json.loads(payload['SOURCE_MANIFEST.json'])['files'])==set(build.MEMBERS)
    job=json.loads(payload['gpu-job.json']);pod=job['spec']['template']['spec']
    assert job['spec']['suspend'] and job['spec']['backoffLimit']==0
    assert 'activeDeadlineSeconds' not in job['spec'] and 'activeDeadlineSeconds' not in pod
    assert len(pod['containers'])==1
    for term in pod['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']:
        row=next(r for r in term['matchExpressions'] if r['key']=='nvidia.com/gpu.product')
        assert row['values']==['NVIDIA-H100-80GB-HBM3']
    command=entry.command_for(plan,worker,bundle.parent/'handshake/STARTUP_RELEASE.json','a'*64)
    assert command[2]=='frontier.observation_recovery_check'
    assert 'fit' not in command and '--resume-directory' not in command
    for n,raw in payload.items():
        if n.endswith('.py') and not n.startswith('source/'):
            ast.parse(raw,feature_version=(3,6))


@pytest.mark.parametrize('field,value',[
    ('optimizer_calls',1),('parent_writes',True),('raw_data_access',True),
    ('automatic_successor',True),('scientific_relaunch_authorized',True),('output_limit_bytes',8*1024**3)])
def test_authority_cannot_expand(packaged,field,value):
    authority=json.loads(packaged[0]['RECOVERY_CHECK_AUTHORITY.json']);authority[field]=value
    with pytest.raises(ValueError):common.validate_authority(authority)


def test_modified_source_is_rejected(packaged):
    payload,_,_,bundle=packaged
    path=bundle/'source/frontier/retention.py';path.write_bytes(path.read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError):common.verify_bundle(json.loads(payload['PLAN.json']),bundle)


def test_tree_bitwise_and_no_optimizer():
    value=dict(a=torch.tensor([1.,2.]),b=(3,[None,True]))
    exact_tree(deepcopy(value),value)
    changed=deepcopy(value);changed['a'][0]+=0.001
    with pytest.raises(ValueError):exact_tree(changed,value)
    with pytest.raises(RuntimeError,match='forbidden'):forbidden_step()


def test_scientific_code_unchanged_except_retirement():
    frozen=ROOT/'observation_d1d2_v2_launch/bundle/source'
    for name in build.SOURCE_MEMBERS:
        if name!='frontier/retention.py':assert (ROOT/name).read_bytes()==(frozen/name).read_bytes()


def test_worker_contains_no_update_or_raw_data_path():
    tree=ast.parse((ROOT/'frontier/observation_recovery_check.py').read_bytes())
    calls=[node.func for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not any(isinstance(f,ast.Attribute) and f.attr in ('step','fit','backward','train_to') for f in calls)
    assert not any(isinstance(f,ast.Name) and f.id in ('NativeInputIndex','ObservationRows','ObservationFitter') for f in calls)


def test_controller_terminal_scope(packaged,monkeypatch):
    payload,_,_,bundle=packaged
    plan=json.loads(payload['PLAN.json'])
    worker,admission=common.verify_bundle(plan,bundle)
    # A Kubernetes completion is insufficient: no output/release means fail closed.
    monkeypatch.setattr(control,'BASE',bundle)
    monkeypatch.setattr(control,'core',type('Core',(),{'audit_runtime_pod':staticmethod(lambda *args:None)})())
    with pytest.raises(FileNotFoundError):
        control.validate_result(plan,dict(uid='u'),bundle.parent/'controller',[dict(metadata=dict(uid='p'))])


def test_previous_attempt_remains_status_readable_but_not_rebuildable():
    old=ROOT/'observation_retention_check_v1_launch/bundle'
    plan=json.loads((old/'PLAN.json').read_bytes())
    common.validate_plan(plan)
    authority=(old/'RECOVERY_CHECK_AUTHORITY.json').read_bytes()
    common.validate_authority(json.loads(authority))
    with pytest.raises(ValueError):
        build.construct(ROOT,RETAINED,authority,(old/'PARENT_INTAKE.json').read_bytes(),
                        json.loads((old/'ADMISSION.json').read_bytes()))
