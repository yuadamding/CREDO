"""Disposable metadata packaging only; no journal mutation, model load or job."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile

import pytest

from deployment import common, continuation_build, continuation_replacement_build as build
from frontier.coordination import WorkBinding


ROOT=Path(__file__).resolve().parents[1]


def admission(pair):
    old=ROOT/('observation_'+pair.lower()+'_v2_launch')/'bundle'
    worker=json.loads((old/'WORKER_CONTRACT.json').read_bytes())
    inventory=json.loads((ROOT/'receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json').read_bytes())
    _,source=continuation_build.source_manifest(ROOT)
    return dict(schema='credo.frontier.observation-admission-export.v1',
        run_id='credo-frontier-obs-'+pair.lower()+'-20261016-v4',status='reserved',
        work_binding=WorkBinding.representation(worker['training_donors'],20000,1280).to_dict(),
        storage_reserved_bytes=4*common.GIB-inventory['fits'][pair]['parent_output_bytes'],
        scientific_update_reservation=18720,aggregate_reserved_output_bytes=64*common.GIB,
        live_h100_reservations=4,journal_event_sha256='a'*64,journal_identity_sha256='b'*64,
        capacity_receipt_sha256='c'*64,phase_authority_sha256=common.identity((old/'OBSERVATION_PHASE_CONTRACT.json').read_bytes())['sha256'],
        continuation_authority_sha256=common.identity((ROOT/'OBSERVATION_CONTINUATION_V3_AUTHORITY.json').read_bytes())['sha256'],
        source_manifest_sha256=common.identity(source)['sha256'],
        replacement_authority_sha256=common.identity((ROOT/build.AUTHORITY).read_bytes())['sha256'],
        zero_update_evidence_sha256='d'*64,replacement_intake_sha256=common.identity((ROOT/build.INTAKE).read_bytes())['sha256'],
        replaces_continuation_task_id='credo-frontier-obs-'+pair.lower()+'-20261016-v3',
        fixture_only_not_launch_authority=True)


@pytest.fixture
def factory(tmp_path,monkeypatch):
    for module in (common,continuation_build,build):
        monkeypatch.setattr(module,'PROJECT',str(tmp_path/'home'))
        monkeypatch.setattr(module,'CONTROL',str(tmp_path/'scratch'))
    def construct(pair='D1D2',bound=None):
        payload,archive,report=build.construct(ROOT,pair,admission(pair) if bound is None else bound)
        return payload,archive,report
    return construct


@pytest.mark.parametrize('pair',build.PAIRS)
def test_four_replacements_have_fresh_identities_unchanged_scientific_sources_and_budgets(factory,pair):
    payload,archive,report=factory(pair)
    plan=json.loads(payload['PLAN.json']);worker=json.loads(payload['WORKER_CONTRACT.json'])
    recovery=json.loads(payload['CONTINUATION_CONTRACT.json'])
    bound=json.loads(payload['ADMISSION.json'])
    old=ROOT/('observation_'+pair.lower()+'_v3_launch')/'bundle'
    old_worker=json.loads((old/'WORKER_CONTRACT.json').read_bytes())
    assert plan['run_id']==report['run_id']==worker['worker_id']==bound['run_id']=='credo-frontier-obs-'+pair.lower()+'-20261016-v4'
    assert worker['trajectory_id'].endswith('-v4-recovery')
    assert plan['phase_directory']=='observation_'+pair.lower()+'_v4'
    assert '/'+plan['phase_directory']+'/' in worker['output_root']
    assert plan['maximum_new_scientific_updates']==report['maximum_new_scientific_updates']==18720
    assert recovery['consumed_updates']==1280 and recovery['maximum_total_updates']==20000
    assert plan['maximum_output_bytes']+recovery['parent_retained_output_bytes']==4*common.GIB
    for name in ('metadata_root','dataset_root','fold_id','training_donors','metadata_index_sha256',
                 'rna_axis_sha256','input_access_sha256','protocol_sha256','runtime_image',
                 'qualification_receipt','qualification_sha256','source_manifest_sha256'):
        assert worker[name]==old_worker[name],name
    for name,raw in payload.items():
        if name.startswith(('source/','retained_reader/')):
            assert raw==(old/name).read_bytes(),name
    assert recovery['authority']['sha256']==bound['continuation_authority_sha256']
    assert recovery['authority']['sha256']!=plan['replacement_authority']['sha256']
    assert common.identity(payload['REPLACEMENT_AUTHORITY.json'])['sha256']==bound['replacement_authority_sha256']
    assert report['remote_submission'] is False and report['automatic_retry'] is False
    assert common.identity(archive)['sha256']==report['archive_sha256']
    common.validate_plan(plan);common.validate_admission(plan,bound)


def test_exact_copy_origins_and_complete_worker_dependency_closure(factory):
    payload,_,_=factory()
    plan=json.loads(payload['PLAN.json']);recovery=json.loads(payload['CONTINUATION_CONTRACT.json'])
    closure=json.loads(payload['WORKER_READ_DEPENDENCIES.json'])
    origins=json.loads(payload['RECOVERY_CHECK_METADATA_ORIGINS.json'])
    bundle=Path(plan['bundle_root'])
    for field,key,name in (('recovery_check_readback','output-readback.json','RECOVERY_CHECK_OUTPUT_READBACK.json'),
                           ('recovery_check_terminal','terminal.json','RECOVERY_CHECK_TERMINAL.json')):
        assert recovery[field]['path']==str(bundle/name)
        record=origins['copies'][key]
        assert record['origin']['path'].startswith('/rsrch8/scratch/')
        assert {k:record['origin'][k] for k in ('sha256','size_bytes')}==common.identity(payload[name])
        assert record['worker_visible_copy']==recovery[field]
    for category in ('metadata_references','checkpoint_payload_references','input_payload_references'):
        for label,item in closure[category].items():
            assert not item['path'].startswith('/rsrch8/scratch/'),label
            assert Path(item['path']).is_absolute()
            if item['path'].startswith(str(bundle)+'/'):
                name=Path(item['path']).relative_to(bundle).as_posix()
                assert common.identity(payload[name])['sha256']==item['sha256']
    for category in ('stat_and_read_roots','lock_paths','expected_absent_paths'):
        assert all(not path.startswith('/rsrch8/scratch/') for path in closure[category].values())
    assert len(closure['checkpoint_payload_references'])==4
    assert len([name for name in closure['metadata_references'] if name.startswith('native_metadata.')])==6
    assert closure['eager_checkpoint_or_native_payload_hashing_authorized'] is False
    assert len(closure['input_payload_references'])>100
    for name in ('parent.identity','parent.failure','parent.checkpoint_completion','parent.incomplete_retirement',
                 'checked_worker.source_manifest','original.worker_source','original.qualification'):
        assert name in closure['metadata_references']


@pytest.mark.parametrize('damage',['direct_scratch','nested_parent_checkpoint','dataset_root','native_member_escape','copy_bytes'])
def test_transitive_dependency_closure_rejects_bad_worker_paths_or_copy(factory,damage):
    payload,_,_=factory();payload=dict(payload);plan=json.loads(payload['PLAN.json'])
    if damage=='direct_scratch':
        recovery=json.loads(payload['CONTINUATION_CONTRACT.json'])
        recovery['recovery_check_terminal']['path']='/rsrch8/scratch/forbidden.json'
        payload['CONTINUATION_CONTRACT.json']=common.encoded(recovery)
        plan['recovery_contract']=build.ref(Path(plan['bundle_root'])/'CONTINUATION_CONTRACT.json',payload['CONTINUATION_CONTRACT.json'])
    elif damage=='nested_parent_checkpoint':
        parent=json.loads(payload['PARENT_INTAKE.json'])
        parent['fits']['D1D2']['checkpoint_payload']['path']='/rsrch8/scratch/forbidden.pt'
        payload['PARENT_INTAKE.json']=common.encoded(parent)
        recovery=json.loads(payload['CONTINUATION_CONTRACT.json'])
        recovery['parent_intake']=build.ref(Path(plan['bundle_root'])/'PARENT_INTAKE.json',payload['PARENT_INTAKE.json'])
        payload['CONTINUATION_CONTRACT.json']=common.encoded(recovery)
        plan['recovery_contract']=build.ref(Path(plan['bundle_root'])/'CONTINUATION_CONTRACT.json',payload['CONTINUATION_CONTRACT.json'])
    elif damage=='dataset_root':
        worker=json.loads(payload['WORKER_CONTRACT.json']);worker['dataset_root']='/rsrch8/scratch/forbidden'
        payload['WORKER_CONTRACT.json']=common.encoded(worker)
    elif damage=='native_member_escape':
        parent=json.loads(payload['PARENT_INVENTORY.json'])
        members=parent['fits']['D1D2']['parent_input_lease']['value']['members']
        members['../forbidden.npz']=deepcopy(next(iter(members.values())))
        payload['PARENT_INVENTORY.json']=common.encoded(parent)
    else:payload['RECOVERY_CHECK_TERMINAL.json']+=b'\n'
    with pytest.raises(ValueError):build.worker_dependency_closure(ROOT,payload,plan)


@pytest.mark.parametrize('field,value',[('replacement_authority_sha256','0'*64),
    ('zero_update_evidence_sha256','invalid'),('replacement_intake_sha256',None),
    ('replaces_continuation_task_id','another-task'),('scientific_update_reservation',20000),
    ('live_h100_reservations',5)])
def test_replacement_admission_does_not_expand_or_detach_authority(factory,field,value):
    bound=admission('D1D2');bound[field]=value
    with pytest.raises(ValueError):factory(bound=bound)


def test_archive_inventory_rebuilt_once_and_no_frozen_package_mutation(factory):
    retained=ROOT/'observation_d1d2_v3_launch/bundle/CONTINUATION_CONTRACT.json'
    before=retained.read_bytes()
    payload,archive,_=factory()
    assert retained.read_bytes()==before
    inventory=json.loads(payload['INVENTORY.json'])
    assert set(inventory)==set(payload)-{'INVENTORY.json'}
    assert all(common.identity(payload[name])==item for name,item in inventory.items())
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as stream:
        assert set(member.name for member in stream.getmembers())=={'bundle/'+name for name in payload}
    job=json.loads(payload['gpu-job.json'])
    assert job['metadata']['name'].endswith('-v4') and job['spec']['suspend'] is True
    assert job['spec']['backoffLimit']==0 and 'activeDeadlineSeconds' not in job['spec']
