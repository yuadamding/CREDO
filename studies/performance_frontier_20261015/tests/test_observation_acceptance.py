"""Fabricated terminal metadata only; no fitting, GPU, SSH or model loading."""
import ast
import copy
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_main_worker import no_numerics
from frontier.coordination import AdmissionJournal, WorkBinding, LIVE


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'observation_d1d2_v2_launch'


@pytest.fixture
def api():
    def load(name):
        spec = importlib.util.spec_from_file_location(name, ROOT/'operations'/f'{name}.py')
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    transport = load('transport')
    with patch.dict(sys.modules, transport=transport):
        observation = load('observation')
        with patch.dict(sys.modules, observation=observation):
            status = load('status_observation')
            with patch.dict(sys.modules, status_observation=status):
                return load('accept_observation')


def pack(api, case):
    """Serialize fabricated writer bytes and their dependent metadata hashes."""
    build, plan, worker, protocol, uid, values = case
    root = plan['scientific_output']
    def item(name):
        raw = (json.dumps(values[name], sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        path = (plan['control_root']+'/gpu/' if name in api.CONTROL_METADATA else root+'/')+name
        return {**api.reference(raw, path), 'text':raw.decode()}
    complete = values['COMPLETE.json']
    for field, name in (('input_lease','INPUT_LEASE.json'), ('input_exit','INPUT_EXIT.json')):
        complete[field] = {k:v for k,v in item(name).items() if k!='text'}
    values['WORKER_STARTED.json']['input_lease'] = complete['input_lease']
    files = {'OWNER.lock':dict(sha256=hashlib.sha256(b'').hexdigest(), size_bytes=0),
             'claims/fixture.released.json':dict(sha256='f'*64,size_bytes=40)}
    for row in complete['audit_history']:
        entry=row['artifact']; files[entry['relative_uri']]={k:entry[k] for k in ('sha256','size_bytes')}
    entry=complete['selected_export']['publication']
    files['SELECTED_OBSERVATION.pt']={k:entry[k] for k in ('sha256','size_bytes')}
    for name in api.OUTPUT_METADATA:
        entry=item(name); files[name]={k:entry[k] for k in ('sha256','size_bytes')}
    readback=values['output-readback.json']
    readback.update(files=files,total_bytes=sum(f['size_bytes'] for f in files.values()),
                    complete_sha256=files['COMPLETE.json']['sha256'])
    values['terminal.json']['result_complete_sha256']=files['COMPLETE.json']['sha256']
    collected=dict(schema='credo.frontier.observation-terminal-metadata.v1',
        files={entry['path']:entry for entry in (item(n) for n in api.OUTPUT_METADATA+api.CONTROL_METADATA)},
        remote_writes=False,model_payloads_opened=False,count_payloads_opened=False)
    records={n:None for n in ('supervisor-failed.json','startup-failure.json','handshake/ENTRY_FAILED.json','output/WORKER_FAILED.json')}
    for name in api.CONTROL_METADATA+('COMPLETE.json','WORKER_STARTED.json'):
        key=name if name in api.CONTROL_METADATA else 'output/'+name
        records[key]=dict(sha256=item(name)['sha256'],value=copy.deepcopy(values[name]))
    records['handshake/WORKER_EXIT.json']=dict(sha256='a'*64,value=dict(returncode=0,run_id=build['run_id']))
    snapshot=dict(schema='credo.frontier.exact-observation-status.v1',run_id=build['run_id'],uid=uid,
        plan_sha256=build['plan_sha256'],remote_writes=False,job=None,pods=[],records=records)
    return snapshot,collected


@pytest.fixture
def case(api):
    _, build, plan, uid, worker, protocol = api.local_context(PACKAGE)
    spec, spec_sha=api.specification(protocol)
    identity=dict(trajectory_id=worker['trajectory_id'],fold_id=worker['fold_id'],seed=0,
        input_sha256=worker['input_access_sha256'],configuration_sha256=spec_sha,
        source_sha256=worker['source_manifest_sha256'],worker_contract_sha256=plan['worker_contract']['sha256'])
    lease=dict(schema='credo.frontier.native-input-lease.v1',dataset_root=worker['dataset_root'],
        access_sha256=worker['input_access_sha256'],metadata_index_sha256=worker['metadata_index_sha256'],
        source_ids=['fabricated-α'],members={'fabricated.json':dict(sha256='a'*64,file_identity=dict(st_size=1))},
        payload_hashes_verified_on_consumption=True,all_payloads_consumed_claimed=False)
    exit_record=dict(schema='credo.frontier.native-input-lease-exit.v1',status='unchanged',
        input_lease_sha256=api.digest(lease,ensure_ascii=False),access_sha256=worker['input_access_sha256'],
        checked_members=1,additional_count_array_scans=0,same_worker_identity_check=True)
    selected=dict(relative_uri='SELECTED_OBSERVATION.pt',sha256='b'*64,size_bytes=128)
    compatibility=dict(status='compatible',source_manifest_sha256=worker['source_manifest_sha256'],
        protocol_sha256=worker['protocol_sha256'],qualification_receipt_sha256=worker['qualification_sha256'])
    audits=[dict(accepted_updates=u,risk=1.,sum_normalized_cross_entropy=4.,positive_cell_directions=4,
        positive_RNA_cells=2,thinning_seed=1,independent_donors=False,artifact=dict(
            relative_uri='observation-candidate-%09d.pt'%u,sha256='c'*64,size_bytes=128)) for u in spec['audit_updates']]
    complete=dict(schema='credo.frontier.observation-worker-completion.v1',status='complete',identity=identity,
        scope='scientific',accepted_updates=20000,attempted_updates=20000,scientific_updates=20000,
        actual_disposable_optimizer_calls=0,main_dynamics_fitted=False,held_out_donor_outcomes_opened=False,
        resume_from_accepted_updates=0,qualification_compatibility=compatibility,audit_history=audits,
        selected_export=dict(publication=selected,completed_updates=20000,selected_updates=256,optimizer_resume=False,
            technical_holdout_excluded_from_scaling=True,scaling=dict(fold_id=worker['fold_id'],positive_RNA_cells=12,
                population_variance_ddof=0,scale_floor=0.01,scale_floor_rule='max(1e-6,fraction*median_positive_fitting_std)')))
    values={'INPUT_LEASE.json':lease,'INPUT_EXIT.json':exit_record,'SELECTED_OBSERVATION.pt.json':selected,
        'COMPLETE.json':complete,'WORKER_STARTED.json':dict(identity=identity,scope='scientific',qualification_compatibility=compatibility),
        'output-readback.json':dict(schema='credo.frontier.observation-output-readback.v1',status='passed',run_id=build['run_id'],no_model_loading=True),
        'terminal.json':dict(status='passed',job=build['run_id'],uid=uid,run_id=build['run_id'],kubernetes_complete=True,
            application_pass=True,reason=None,mode='fit',lifecycle_plan_sha256=build['plan_sha256'],automatic_successor=False),
        'cleanup.json':dict(status='complete',job=build['run_id'],uid=uid,reads=[dict(job_absent=True,owned_pods=0)]*2),
        'startup-validated.json':dict(status='startup_released',uid=uid,run_id=build['run_id'],
            worker_contract_sha256=plan['worker_contract']['sha256'],lifecycle_plan_sha256=build['plan_sha256'],
            actual_pod_audit_passed=True,storage_and_fileset_quota_passed=True)}
    return build,plan,worker,protocol,uid,values


def evaluate(api, case):
    return api.validate(*case[:5],*pack(api,case))


def test_complete_metadata_yields_exact_loader_admission_without_loading(api,case,monkeypatch):
    import torch
    from frontier.observation import ObservationFold
    from frontier.observation_training import ObservationTrainingSpec
    from frontier.observation_loader import ObservationFile,ObservationArtifactAdmission
    from frontier.native_inputs import _hash
    monkeypatch.setattr(torch,'load',lambda *_a,**_k:pytest.fail('No checkpoint load permitted'))
    result=evaluate(api,case)
    assert result['status']=='metadata_accepted' and result['scientific_updates']==20000
    assert result['selected_updates']==256 and not result['main_model_launch_authorized']
    assert result['downstream_loader_byte_and_tensor_checks_required']
    assert result['storage_reservation_bytes']==4*api.GIB and not result['storage_reclamation_claimed']
    assert api.specification(case[3])[1]==ObservationTrainingSpec().identity
    assert case[-1]['INPUT_EXIT.json']['input_lease_sha256']==_hash(case[-1]['INPUT_LEASE.json'])
    plan=case[1]
    metadata={name:api.reference((PACKAGE/'bundle'/filename).read_bytes(),plan[name]['path']) for name,filename in
              (('worker_contract','WORKER_CONTRACT.json'),('protocol','OBSERVATION_PROTOCOL.json'))}
    admission=ObservationArtifactAdmission(**{k:ObservationFile(**v) for k,v in
        {**{k:result[k] for k in ('artifact','receipt','completion')},**metadata}.items()},
        fold=ObservationFold(result['fold']['fold_id'],tuple(result['fold']['training_donors']),result['fold']['feature_axis_sha256']),
        specification=ObservationTrainingSpec(),trajectory_identity=result['trajectory_identity'])
    assert admission.artifact.path==plan['scientific_output']+'/SELECTED_OBSERVATION.pt'


@pytest.mark.parametrize('name,key,value',[
 ('terminal.json','uid','other'),('terminal.json','application_pass',False),
 ('terminal.json','automatic_successor',True),('cleanup.json','reads',[dict(job_absent=True,owned_pods=0)]),
 ('startup-validated.json','actual_pod_audit_passed',False),
 ('COMPLETE.json','attempted_updates',20001),('COMPLETE.json','accepted_updates',19999),
 ('COMPLETE.json','accepted_updates',20000.0),
 ('COMPLETE.json','scope','synthetic_cpu_only'),('COMPLETE.json','held_out_donor_outcomes_opened',True),
 ('COMPLETE.json','resume_from_accepted_updates',256),('COMPLETE.json','audit_history',[]),
 ('INPUT_EXIT.json','same_worker_identity_check',False),('INPUT_EXIT.json','input_lease_sha256','d'*64),
 ('INPUT_EXIT.json','checked_members',2),('INPUT_EXIT.json','additional_count_array_scans',1),
 ('INPUT_LEASE.json','all_payloads_consumed_claimed',True),
 ('INPUT_LEASE.json','metadata_index_sha256','d'*64),('INPUT_LEASE.json','dataset_root','/unrelated'),
 ('WORKER_STARTED.json','scope','synthetic_cpu_only'),('output-readback.json','no_model_loading',False),
])
def test_semantic_drift_fails_even_when_rehashed(api,case,name,key,value):
    case[-1][name][key]=value
    with pytest.raises(ValueError): evaluate(api,case)


@pytest.mark.parametrize('mutation',[
 lambda c:c['selected_export'].update(selected_updates=512),
 lambda c:c['selected_export'].update(optimizer_resume=True),
 lambda c:c['selected_export'].update(technical_holdout_excluded_from_scaling=False),
 lambda c:c['selected_export']['scaling'].update(fold_id='subset-D3-D4'),
 lambda c:c['selected_export']['scaling'].update(scale_floor=-0.01),
 lambda c:c['audit_history'][0].update(risk=3.),
 lambda c:c['audit_history'][0].update(independent_donors=True),
 lambda c:c['audit_history'][0].update(positive_RNA_cells=1),
 lambda c:c['audit_history'][0]['artifact'].update(relative_uri='../foreign.pt'),
 lambda c:c['qualification_compatibility'].update(qualification_receipt_sha256='d'*64),
 lambda c:c['identity'].update(fold_id='subset-D3-D4'),
])
def test_selected_audit_and_input_identity_semantics(api,case,mutation):
    mutation(case[-1]['COMPLETE.json'])
    with pytest.raises(ValueError): evaluate(api,case)


@pytest.mark.parametrize('key,value',[('job',{'status':{'active':1}}),('pods',[{'uid':'still-here'}]),('uid','other'),('remote_writes',True)])
def test_nonterminal_or_mismatched_snapshot_never_accepts(api,case,key,value):
    snapshot,collected=pack(api,case);snapshot[key]=value
    with pytest.raises(ValueError):api.validate(*case[:5],snapshot,collected)


def test_changed_metadata_bytes_and_extra_paths_fail(api,case):
    snapshot,collected=pack(api,case)
    path=next(iter(collected['files']))
    collected['files'][path]['text']+=' '
    with pytest.raises(ValueError,match='bytes'):api.validate(*case[:5],snapshot,collected)
    snapshot,collected=pack(api,case);collected['files']['/unrelated']=collected['files'][path]
    with pytest.raises(ValueError,match='collection'):api.validate(*case[:5],snapshot,collected)


def test_strict_json_rejects_duplicate_keys_and_nonfinite(api):
    for raw in ('{"x":1,"x":2}','{"x":NaN}','[]'):
        with pytest.raises(ValueError):api.strict_json(raw)


def test_qualification_package_cannot_enter_fit_intake(api):
    with pytest.raises(ValueError,match='Only a scientific observation fit'):
        api.local_context(ROOT/'observation_qualification_launch_v2')


def test_collect_script_is_host_compatible_readonly_and_checks_type(api,tmp_path,monkeypatch):
    ast.parse(api.COLLECT_CODE,feature_version=(3,6))
    path=tmp_path/'metadata.json';path.write_text('{"fabricated":true}')
    original=Path.lstat
    def owned(path,*args,**kw):
        s=original(path,*args,**kw)
        return SimpleNamespace(**{k:getattr(s,k) for k in dir(s) if k.startswith('st_') and k!='st_uid'},st_uid=307469)
    monkeypatch.setattr(Path,'lstat',owned)
    stream=io.StringIO()
    with redirect_stdout(stream):exec(api.COLLECT_CODE,dict(PATHS=[str(path)]))
    result=json.loads(stream.getvalue().split('RECEIPT=',1)[1])
    assert result['files'][str(path)]['text']=='{"fabricated":true}' and result['remote_writes'] is False
    link=tmp_path/'link.json';link.symlink_to(path)
    with pytest.raises(ValueError,match='type/owner'):exec(api.COLLECT_CODE,dict(PATHS=[str(link)]))


@pytest.fixture
def integrated(api,case,tmp_path,monkeypatch):
    monkeypatch.setattr(api,'ROOT',tmp_path)
    (tmp_path/'AUTHORITY.json').write_text('{"fixture":true}')
    bundle=tmp_path/'package/bundle';bundle.mkdir(parents=True)
    # Small existing frozen protocol/contract METADATA only, not data/checkpoints.
    for filename in ('WORKER_CONTRACT.json','OBSERVATION_PROTOCOL.json'):
        (bundle/filename).write_bytes((PACKAGE/'bundle'/filename).read_bytes())
    work=WorkBinding.representation(tuple(case[2]['training_donors']))
    admission=dict(work_binding=work.to_dict(),phase_authority_sha256='a'*64)
    raw=json.dumps(admission).encode();(bundle/'ADMISSION.json').write_bytes(raw)
    case[1]['admission']['sha256']=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(api,'local_context',lambda path:(bundle.parent,*case[:2],case[4],*case[2:4]))
    calls=[]
    def remote(code,values):
        calls.append(code)
        snapshot,collected=pack(api,case)
        return snapshot if code==api.STATUS_CODE else collected
    monkeypatch.setattr(api,'remote',remote)
    def record(path,value):
        with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False)
    monkeypatch.setattr(api,'record',record)
    options=dict(study_identity_sha256=api.sha(tmp_path/'AUTHORITY.json'))
    with AdmissionJournal(tmp_path/'admission',**options) as journal:
        journal.reserve(case[0]['run_id'],worker_id=case[0]['run_id'],kind='representation',scientific_updates=20000,
            output_bytes=4*api.GIB,admission_sha256='a'*64,capacity_receipt_sha256='b'*64,
            fileset_headroom_bytes=100*api.GIB,work_binding=work)
        journal.reconcile(case[0]['run_id'],status='submitted',exact_scheduler_id=case[4],evidence_sha256='c'*64)
    return tmp_path,calls,options


def test_collect_and_reconcile_real_temporary_journal_no_remote_writes(api,case,integrated):
    root,calls,options=integrated
    intake,receipt=root/'intake.json',root/'reconciled.json'
    selected=api.collect(root/'package',intake)
    assert len(calls)==2 and selected['selected_updates']==256
    with AdmissionJournal(root/'admission',**options) as journal:
        assert journal.snapshot()['tasks'][case[0]['run_id']]['status']=='submitted'
    result=api.reconcile(intake,receipt)
    assert result['live_h100_reservations']==0 and result['storage_reclaimed_bytes']==0
    with AdmissionJournal(root/'admission',**options) as journal:
        state=journal.snapshot();sequence=state['sequence']
        task=state['tasks'][case[0]['run_id']]
        assert task['status']=='complete' and task['storage_reserved_bytes']==4*api.GIB
        assert task['scientific_update_reservation']==20000
    recovered=api.reconcile(intake,root/'explicit-reconciliation.json')
    assert recovered['action']=='reused_exact_committed_terminal_event'
    with AdmissionJournal(root/'admission',**options) as journal:assert journal.snapshot()['sequence']==sequence
    assert len(calls)==2  # Journal reconciliation performs no network operation.
    with pytest.raises(ValueError,match='New local'):api.reconcile(intake,receipt)


def test_running_job_stops_intake_before_detail_reads_and_publication(api,case,integrated,monkeypatch):
    root,calls,_=integrated
    def active(code,values):
        calls.append(code)
        snapshot,_=pack(api,case);snapshot['job']={'active':1}
        return snapshot
    monkeypatch.setattr(api,'remote',active)
    with pytest.raises(ValueError,match='absence'):api.collect(root/'package',root/'intake.json')
    assert len(calls)==1 and not (root/'intake.json').exists()


def test_changed_intake_cannot_free_reservation(api,case,integrated):
    root,_,options=integrated
    intake=root/'intake.json';api.collect(root/'package',intake)
    value=json.loads(intake.read_text());value['accepted']['selected_updates']=512
    intake.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='acceptance differs'):api.reconcile(intake,root/'reconciled.json')
    with AdmissionJournal(root/'admission',**options) as journal:
        assert journal.snapshot()['tasks'][case[0]['run_id']]['status']=='submitted'
    assert not (root/'reconciled.json').exists()


def test_interrupted_reconciliation_publication_reuses_only_same_committed_event(api,case,integrated,monkeypatch):
    root,calls,options=integrated
    intake=root/'intake.json';api.collect(root/'package',intake)
    writer=api.record
    monkeypatch.setattr(api,'record',lambda *_a,**_k:(_ for _ in ()).throw(OSError('fabricated receipt failure')))
    with pytest.raises(OSError,match='fabricated'):api.reconcile(intake,root/'reconciled.json')
    with AdmissionJournal(root/'admission',**options) as journal:
        state=journal.snapshot();assert state['tasks'][case[0]['run_id']]['status']=='complete'
    monkeypatch.setattr(api,'record',writer)
    result=api.reconcile(intake,root/'explicit-recovery.json')
    assert result['action']=='reused_exact_committed_terminal_event'
    with AdmissionJournal(root/'admission',**options) as journal:assert journal.snapshot()==state
    assert len(calls)==2 and result['storage_reclaimed_bytes']==0


def test_failed_task_is_not_promoted_from_completion_metadata(api,case,integrated):
    root,_,options=integrated
    intake=root/'intake.json';api.collect(root/'package',intake)
    with AdmissionJournal(root/'admission',**options) as journal:
        journal.reconcile(case[0]['run_id'],status='failed',exact_scheduler_id=case[4],evidence_sha256='d'*64)
    with pytest.raises(ValueError,match='Failed/cancelled'):api.reconcile(intake,root/'reconciled.json')


def test_collected_readback_hash_cannot_be_substituted(api,case):
    snapshot,collected=pack(api,case)
    snapshot['records']['output-readback.json']['sha256']='d'*64
    with pytest.raises(ValueError,match='changed after exact status'):api.validate(*case[:5],snapshot,collected)
