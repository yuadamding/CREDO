"""Small fabricated metadata and lexical payload references; zero model work."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from deployment import common


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def context(tmp_path, monkeypatch):
    home = tmp_path/'home'
    project = home/common.STUDY
    bundle = project/'observation_d1d2_v4'/'bundle'
    bundle.mkdir(parents=True)
    monkeypatch.setattr(common,'VISIBLE_HOME',str(home))
    monkeypatch.setattr(common,'PROJECT',str(project))
    def save(name,value):
        path = bundle/name
        path.parent.mkdir(parents=True,exist_ok=True)
        raw = common.encoded(value)
        path.write_bytes(raw)
        return dict(path=str(path),**common.identity(raw))
    plan = dict(mode='fit',run_id='credo-frontier-obs-d1d2-20261016-v4',bundle_root=str(bundle))
    for key in ('worker_contract','source_manifest','protocol','phase_authority','parent_authority','admission','recovery_contract'):
        plan[key] = save(key+'.json',{})
    authority = json.loads((ROOT/'OBSERVATION_CONTINUATION_V4_REPLACEMENT_AUTHORITY.json').read_text())
    plan['replacement_authority'] = save('REPLACEMENT_AUTHORITY.json',authority)
    plan['replacement_intake'] = save('REPLACEMENT_INTAKE.json',dict(
        schema='credo.frontier.observation-continuation-replacement-intake.v1',status='passed',
        authority_sha256=plan['replacement_authority']['sha256'],prior_failure_closure_sha256='9'*64))
    copies = {}
    for key in ('output-readback.json','terminal.json'):
        copied = save('copied-'+key,dict(status='passed',fixture=key))
        copies[key] = dict(origin=dict(copied,path='/rsrch8/scratch/controller/'+key),worker_visible_copy=copied)
    plan['recovery_metadata_origins'] = save('ORIGINS.json',dict(copies=copies))
    worker = dict(training_donors=['D1','D2'],study_root=str(project),source_root=str(bundle/'source'),
        reader_source_root=str(bundle/'reader'),metadata_root=str(home/'inputs'),dataset_root=str(home/'inputs'),
        output_root=str(bundle.parent/'output'))
    metadata = {'plan.'+key:value for key,value in plan.items() if isinstance(value,dict)}
    metadata.update({'copy.'+key:item['worker_visible_copy'] for key,item in copies.items()})
    payload = dict(path=str(project/'parent'/'checkpoint.pt'),sha256='1'*64,size_bytes=1000000)
    closure = dict(schema='credo.frontier.worker-read-dependency-closure.v1',
        policy='home_visible_metadata_preflight_payload_verification_on_existing_consumption_path',
        run_id=plan['run_id'],worker_visible_roots=[str(project),str(home/'inputs')],metadata_references=metadata,
        checkpoint_payload_references={'parent':payload},
        input_payload_references={'array':dict(path=str(home/'inputs'/'array.npy'),sha256='2'*64,size_bytes=1000000)},
        stat_and_read_roots={'worker.'+k:worker[k] for k in
            ('study_root','source_root','reader_source_root','metadata_root','dataset_root','output_root')},
        lock_paths={'parent.owner':str(project/'parent'/'OWNER.lock')},
        expected_absent_paths={'parent.retired':str(project/'parent'/'retired.pt')},
        provenance_only_origins=copies,runtime_image_libraries_outside_study_closure=True,
        eager_checkpoint_or_native_payload_hashing_authorized=False,optimizer_calls=0)
    admission = dict(replacement_authority_sha256=plan['replacement_authority']['sha256'],
        replaces_continuation_task_id='credo-frontier-obs-d1d2-20261016-v3',zero_update_evidence_sha256='9'*64,
        replacement_intake_sha256=plan['replacement_intake']['sha256'])
    def publish():
        plan['worker_read_dependencies'] = save('WORKER_READ_DEPENDENCIES.json',closure)
        inventory = {p.relative_to(bundle).as_posix():common.identity(p.read_bytes()) for p in bundle.rglob('*')
            if p.is_file() and p.name != 'INVENTORY.json'}
        save('INVENTORY.json',inventory)
    publish()
    return plan,worker,closure,admission,save,publish


def test_reads_all_exact_metadata_but_never_payloads_locks_or_origins(context,monkeypatch):
    plan,worker,closure,_,_,_=context
    reads=[]
    original=common.read
    def tracked(path,*args,**kwargs):
        reads.append(str(path))
        assert not str(path).endswith(('.pt','.npy','OWNER.lock'))
        assert not str(path).startswith('/rsrch8/scratch/')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(common,'read',tracked)
    assert common.verify_worker_read_dependencies(plan,worker)==closure
    assert set(item['path'] for item in closure['metadata_references'].values()).issubset(reads)
    assert not Path(closure['checkpoint_payload_references']['parent']['path']).exists()


@pytest.mark.parametrize('damage',['missing','hash','size','scratch','duplicate','omitted_plan','manifest_inventory',
    'wrong_run','eager_payloads','optimizer','root','payload_escape','lock_escape','origin_copy','unknown_key'])
def test_dependency_corruption_is_rejected_before_release(context,damage):
    plan,worker,closure,_,save,publish=context
    reference=closure['metadata_references']['plan.protocol']
    if damage=='missing':Path(reference['path']).unlink()
    elif damage=='hash':reference['sha256']='f'*64
    elif damage=='size':reference['size_bytes']+=1
    elif damage=='scratch':reference['path']='/rsrch8/scratch/invisible/protocol.json'
    elif damage=='duplicate':closure['metadata_references']['conflict']=dict(reference,sha256='f'*64)
    elif damage=='omitted_plan':del closure['metadata_references']['plan.protocol']
    elif damage=='wrong_run':closure['run_id']='other'
    elif damage=='eager_payloads':closure['eager_checkpoint_or_native_payload_hashing_authorized']=True
    elif damage=='optimizer':closure['optimizer_calls']=1
    elif damage=='root':closure['worker_visible_roots']=['/']
    elif damage=='payload_escape':closure['checkpoint_payload_references']['parent']['path']='/other/model.pt'
    elif damage=='lock_escape':closure['lock_paths']['parent.owner']='/rsrch8/scratch/OWNER.lock'
    elif damage=='origin_copy':closure['provenance_only_origins']['terminal.json']['origin']['sha256']='f'*64
    elif damage=='unknown_key':closure['extra']=True
    publish()
    if damage=='manifest_inventory':
        path=Path(plan['bundle_root'])/'INVENTORY.json'
        inventory=json.loads(path.read_text());del inventory['WORKER_READ_DEPENDENCIES.json']
        save('INVENTORY.json',inventory)
    with pytest.raises((ValueError,FileNotFoundError)):
        common.verify_worker_read_dependencies(plan,worker)


def test_optional_reference_size_and_duplicate_same_bytes_are_permitted(context):
    plan,worker,closure,_,_,publish=context
    ref=closure['metadata_references']['plan.protocol']
    closure['metadata_references']['optional_size']={k:v for k,v in ref.items() if k!='size_bytes'}
    publish()
    assert common.verify_worker_read_dependencies(plan,worker)==closure


def test_bound_recovery_metadata_cannot_be_silently_omitted(context):
    plan,worker,closure,_,save,publish=context
    other=save('external.json',{'fixture':True})
    recovery=save('recovery_contract.json',dict(parent_input_lease=other))
    plan['recovery_contract']=recovery
    closure['metadata_references']['plan.recovery_contract']=recovery
    publish()
    with pytest.raises(ValueError,match='recovery dependency'):
        common.verify_worker_read_dependencies(plan,worker)
    closure['metadata_references']['recovery.parent_input_lease']=other
    publish()
    assert common.verify_worker_read_dependencies(plan,worker)==closure


def test_replacement_authority_is_exactly_bound_to_admission(context):
    plan,worker,_,admission,_,_=context
    assert common.validate_replacement(plan,admission,worker)['status']=='approved'
    changed=dict(admission,replacement_authority_sha256='0'*64)
    with pytest.raises(ValueError,match='authority/admission'):common.validate_replacement(plan,changed,worker)
    changed=dict(admission,replaces_continuation_task_id='wrong')
    with pytest.raises(ValueError,match='v3-to-v4'):common.validate_replacement(plan,changed,worker)


@pytest.mark.parametrize('key,value',[('status','proposed'),('maximum_new_updates_each',20000),
    ('automatic_retries',1),('maximum_concurrent_h100',5),('main_model_fitting',True),
    ('numerical_worker_sources_changed',True),('local_optimizer_calls_authorized',1),
    ('elapsed_time_deadline_seconds',86400),('LARRY_access',True)])
def test_rehashed_authority_scope_changes_are_rejected(context,key,value):
    plan,worker,_,admission,save,_=context
    authority=json.loads(Path(plan['replacement_authority']['path']).read_text())
    authority[key]=value
    plan['replacement_authority']=save('REPLACEMENT_AUTHORITY.json',authority)
    admission['replacement_authority_sha256']=plan['replacement_authority']['sha256']
    with pytest.raises(ValueError,match='authority scope'):common.validate_replacement(plan,admission,worker)


def test_common_stays_host_python36_and_has_no_numerical_or_scheduler_imports():
    source=(ROOT/'deployment/common.py').read_text()
    ast.parse(source,feature_version=(3,6))
    for text in ('import torch','import numpy','subprocess','AdmissionJournal'):
        assert text not in source
