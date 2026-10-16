"""Fabricated JSON only: no remote call, ledger transition, tensor or optimizer."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLAIM = 'a'*32
EXIT = 'input-exits/'+CLAIM+'.json'
OWNER = 'claims/'+CLAIM+'.json'
RELEASE = 'claims/'+CLAIM+'.released.json'


@pytest.fixture
def api():
    def load(name):
        spec = importlib.util.spec_from_file_location(name,ROOT/'operations'/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    with patch.dict(sys.modules):
        for name in ('transport','observation','status_observation','accept_observation'):
            sys.modules[name] = load(name)
        module = load('accept_observation_continuation')
        # No real transport or receipt write is permitted in these tests.
        module.remote = lambda *_a,**_k: pytest.fail('Remote call forbidden')
        module.record = lambda *_a,**_k: pytest.fail('Receipt publication forbidden')
        yield module


@pytest.fixture
def case(api):
    ref = lambda path,c:dict(path=path,sha256=c*64,size_bytes=123)
    bundle,output,control = '/example/v3/bundle','/example/v3/output','/example/control/v3'
    parent_root = '/example/v2/output'
    parent_bytes = 1124766153
    plan = dict(mode='fit',bundle_root=bundle,scientific_output=output,control_root=control,
        maximum_new_scientific_updates=18720,maximum_output_bytes=4*api.GIB-parent_bytes,
        parent_retained_output_bytes=parent_bytes,combined_output_limit_bytes=4*api.GIB,
        worker_contract=ref(bundle+'/WORKER_CONTRACT.json','1'),
        protocol=ref(bundle+'/OBSERVATION_PROTOCOL.json','2'),
        recovery_contract=ref(bundle+'/CONTINUATION_CONTRACT.json','3'))
    build = dict(run_id='fixture-v3',plan_sha256='4'*64,source_manifest_sha256='5'*64)
    worker = dict(mode='fit',worker_id=build['run_id'],trajectory_id='observation-D1-D2-v3',fold_id='subset-D1-D2',
        training_donors=['D1','D2'],output_root=output,output_limit_bytes=plan['maximum_output_bytes'],
        source_manifest_sha256=build['source_manifest_sha256'],protocol_sha256=plan['protocol']['sha256'],
        protocol_path=plan['protocol']['path'],rna_axis_sha256='6'*64,metadata_index_sha256='7'*64,
        input_access_sha256='8'*64,qualification_sha256='9'*64,dataset_root='/example/dataset')
    protocol = dict(architecture_dimensions=[18129,1024,256,48],logical_batch_positive_training_cells=1024,
        maximum_accepted_updates_each=20000,audit_candidates_accepted_updates=[256,512,1024,2048,4096,8192,12288,16384,20000],
        checkpoint_interval_accepted_updates=256,maximum_dense_working_estimate_bytes=2*api.GIB,
        RNA_feature_axis_identity=worker['rna_axis_sha256'],metadata_index_identity=worker['metadata_index_sha256'],
        training_subsets=[['D1','D2']])
    recovery = dict(schema='credo.frontier.observation-continuation-contract.v1',mode='explicit_source_change_continuation',
        pair='D1D2',consumed_updates=1280,maximum_new_updates=18720,maximum_total_updates=20000,
        combined_output_limit_bytes=4*api.GIB,fit_contract=plan['worker_contract'],parent_retained_output_bytes=parent_bytes,
        authority=ref(bundle+'/CONTINUATION_AUTHORITY.json','a'),parent_intake=ref(bundle+'/PARENT_INTAKE.json','b'),
        parent_inventory=ref(bundle+'/PARENT_INVENTORY.json','c'),
        parent_fit_contract=ref('/example/v2/bundle/WORKER_CONTRACT.json','d'),
        parent_source_manifest=ref('/example/v2/bundle/SOURCE_MANIFEST.json','e'),
        parent_input_lease=ref(parent_root+'/INPUT_LEASE.json','f'),
        recovery_check_completion=ref('/example/check/output/COMPLETE.json','0'))
    _,spec_sha = api.specification(protocol)
    original = dict(trajectory_id='observation-D1-D2',fold_id=worker['fold_id'],seed=0,
        input_sha256=worker['input_access_sha256'],configuration_sha256=spec_sha,
        source_sha256=recovery['parent_source_manifest']['sha256'],worker_contract_sha256=recovery['parent_fit_contract']['sha256'])
    parents = dict(fits=dict(D1D2=dict(identity=dict(path=parent_root+'/IDENTITY.json',value=original),
        checkpoint_payload=ref(parent_root+'/checkpoint-000001280.pt','c'))))
    identity = dict(original,trajectory_id=worker['trajectory_id'],source_sha256=worker['source_manifest_sha256'],
        worker_contract_sha256=plan['worker_contract']['sha256'],
        recovery_parent_checkpoint_sha256=parents['fits']['D1D2']['checkpoint_payload']['sha256'],
        recovery_authority_sha256=recovery['authority']['sha256'])
    lease = dict(schema='credo.frontier.native-input-lease.v1',dataset_root=worker['dataset_root'],
        access_sha256=worker['input_access_sha256'],metadata_index_sha256=worker['metadata_index_sha256'],
        source_ids=['fabricated-α'],members={'fixture':dict(sha256='f'*64,file_identity=dict(st_size=123,st_dev=1,st_ino=2))},
        payload_hashes_verified_on_consumption=True,all_payloads_consumed_claimed=False)
    inventory = {'fixture.pt':dict(st_size=parent_bytes)}
    inventory_sha = hashlib.sha256((json.dumps(inventory,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    retained = dict(schema='credo.frontier.observation-continuation-intake.v1',status='passed',
        authority_sha256=recovery['authority']['sha256'],fits=dict(D1D2=dict(parent_output_bytes=parent_bytes,
        old_job_absent=True,new_job_absent=True,owned_pods_absent=True,new_roots_absent=True,
        inventory=inventory,inventory_sha256=inventory_sha,
        parent_input_lease=dict(recovery['parent_input_lease'],value=copy.deepcopy(lease)))))
    compatibility = dict(schema='credo.frontier.observation-source-continuation-compatibility.v1',status='compatible',
        policy='unchanged_numerical_sources_plus_actual_mount_retirement_and_exact_cuda_restore_v1',
        original_qualification_sha256=worker['qualification_sha256'],original_source_manifest_sha256=original['source_sha256'],
        continuation_source_manifest_sha256=worker['source_manifest_sha256'],
        recovery_check_sha256=recovery['recovery_check_completion']['sha256'],
        continuation_contract_sha256=plan['recovery_contract']['sha256'],numerical_sources_unchanged=api.NUMERICAL.copy(),
        mixed_source_continuation=True,fresh_numerical_qualification_performed=False)
    publication = dict(relative_uri='SELECTED_OBSERVATION.pt',sha256='b'*64,size_bytes=128)
    audits = [dict(accepted_updates=u,risk=1.,sum_normalized_cross_entropy=4.,positive_cell_directions=4,
        positive_RNA_cells=2,thinning_seed=1,independent_donors=False,
        artifact=dict(relative_uri='observation-candidate-%09d.pt'%u,sha256='c'*64,size_bytes=128))
        for u in protocol['audit_candidates_accepted_updates']]
    complete = dict(schema='credo.frontier.observation-worker-completion.v1',status='complete',identity=identity,
        scope='scientific',accepted_updates=20000,attempted_updates=20000,scientific_updates=20000,
        actual_disposable_optimizer_calls=0,main_dynamics_fitted=False,held_out_donor_outcomes_opened=False,
        resume_from_accepted_updates=1280,consumed_parent_updates=1280,new_scientific_updates_this_owner=18720,
        source_continuation_contract=plan['recovery_contract'],mixed_source_continuation=True,
        qualification_compatibility=compatibility,audit_history=audits,
        selected_export=dict(publication=publication,completed_updates=20000,selected_updates=256,
            optimizer_resume=False,technical_holdout_excluded_from_scaling=True,scaling=dict(fold_id=worker['fold_id'],
            positive_RNA_cells=12,population_variance_ddof=0,scale_floor=0.01,
            scale_floor_rule='max(1e-6,fraction*median_positive_fitting_std)')))
    uid = 'fixture-uid'
    values = {'COMPLETE.json':complete,'WORKER_STARTED.json':dict(identity=identity,scope='scientific',
        qualification_compatibility=copy.deepcopy(compatibility)),'INPUT_LEASE.json':lease,
        EXIT:dict(schema='credo.frontier.native-input-lease-exit.v1',status='unchanged',
            input_lease_sha256=api.digest(lease,ensure_ascii=False),access_sha256=worker['input_access_sha256'],
            checked_members=1,additional_count_array_scans=0,same_worker_identity_check=True),
        OWNER:dict(claim_id=CLAIM,worker_id=worker['worker_id'],trajectory_id=worker['trajectory_id']),
        RELEASE:dict(claim_id=CLAIM,exception_type=None,automatic_restart_authorized=False),
        'SELECTED_OBSERVATION.pt.json':publication,
        'output-readback.json':dict(schema='credo.frontier.observation-output-readback.v1',status='passed',
            run_id=build['run_id'],no_model_loading=True,parent_output_bytes=parent_bytes,
            combined_output_limit_bytes=4*api.GIB,source_continuation_contract=plan['recovery_contract'],
            new_scientific_updates_this_owner=18720,cumulative_scientific_updates=20000,parent_payloads_opened=False),
        'terminal.json':dict(status='passed',job=build['run_id'],uid=uid,run_id=build['run_id'],kubernetes_complete=True,
            application_pass=True,reason=None,mode='fit',lifecycle_plan_sha256=build['plan_sha256'],automatic_successor=False),
        'cleanup.json':dict(status='complete',job=build['run_id'],uid=uid,reads=[dict(job_absent=True,owned_pods=0)]*2),
        'startup-validated.json':dict(status='startup_released',uid=uid,run_id=build['run_id'],
            worker_contract_sha256=plan['worker_contract']['sha256'],lifecycle_plan_sha256=build['plan_sha256'],
            recovery_contract_sha256=plan['recovery_contract']['sha256'],actual_pod_audit_passed=True,
            storage_and_fileset_quota_passed=True)}
    return (build,plan,worker,protocol,recovery,parents,retained,uid),values


def pack(api,case):
    context,values = case
    build,plan,worker,protocol,recovery,parents,retained,uid = context
    def item(name):
        raw = (json.dumps(values[name],sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        path = (plan['control_root']+'/gpu/' if name in api.CONTROL_METADATA else plan['scientific_output']+'/')+name
        return dict(api.reference(raw,path),text=raw.decode())
    complete = values['COMPLETE.json']
    complete['input_lease'] = {k:v for k,v in item('INPUT_LEASE.json').items() if k != 'text'}
    complete['input_exit'] = {k:v for k,v in item(EXIT).items() if k != 'text'}
    values['WORKER_STARTED.json']['input_lease'] = complete['input_lease']
    names = api.STATIC_OUTPUT+(EXIT,OWNER,RELEASE)
    files = {'OWNER.lock':dict(sha256=hashlib.sha256(b'').hexdigest(),size_bytes=0)}
    for row in complete['audit_history']:
        entry = row['artifact'];files[entry['relative_uri']] = {k:entry[k] for k in ('sha256','size_bytes')}
    entry = complete['selected_export']['publication']
    files['SELECTED_OBSERVATION.pt'] = {k:entry[k] for k in ('sha256','size_bytes')}
    for name in names:
        entry = item(name);files[name] = {k:entry[k] for k in ('sha256','size_bytes')}
    total = sum(entry['size_bytes'] for entry in files.values())
    values['output-readback.json'].update(files=files,total_bytes=total,child_output_bytes=total,
        combined_output_bytes=total+plan['parent_retained_output_bytes'],complete_sha256=files['COMPLETE.json']['sha256'])
    values['terminal.json']['result_complete_sha256'] = files['COMPLETE.json']['sha256']
    collected = dict(schema='credo.frontier.observation-terminal-metadata.v1',remote_writes=False,
        model_payloads_opened=False,count_payloads_opened=False,
        files={entry['path']:entry for entry in (item(n) for n in names+api.CONTROL_METADATA)})
    records = {name:None for name in ('supervisor-failed.json','startup-failure.json',
        'handshake/ENTRY_FAILED.json','output/WORKER_FAILED.json')}
    for name in api.CONTROL_METADATA+('COMPLETE.json','WORKER_STARTED.json'):
        key = name if name in api.CONTROL_METADATA else 'output/'+name
        records[key] = dict(sha256=item(name)['sha256'],value=copy.deepcopy(values[name]))
    records['handshake/WORKER_EXIT.json'] = dict(value=dict(returncode=0,run_id=build['run_id']))
    snapshot = dict(schema='credo.frontier.exact-observation-status.v1',run_id=build['run_id'],uid=uid,
        plan_sha256=build['plan_sha256'],remote_writes=False,job=None,pods=[],records=records)
    return snapshot,collected


def evaluate(api,case):
    return api.validate(*case[0],*pack(api,case))


def test_accepts_exact_mixed_lineage_and_parent_candidate_without_numerical_execution(api,case):
    result = evaluate(api,case)
    assert result['status'] == 'metadata_accepted'
    assert result['scientific_updates'] == 20000
    assert result['consumed_parent_updates'] == 1280
    assert result['new_scientific_updates_this_owner'] == 18720
    assert result['selected_updates'] == 256  # Original parent candidate remains eligible.
    assert result['input_exit']['path'].endswith('/'+EXIT)
    assert result['storage_reservation_bytes']+result['parent_retained_output_bytes'] == 4*api.GIB
    assert result['mixed_source_continuation'] and result['downstream_loader_byte_and_tensor_checks_required']
    assert not any(result[k] for k in ('tensors_loaded','numerical_replay','journal_reconciled',
        'main_model_launch_authorized','storage_reclamation_claimed'))


@pytest.mark.parametrize('name,key,value',[
    ('COMPLETE.json','resume_from_accepted_updates',0),('COMPLETE.json','consumed_parent_updates',0),
    ('COMPLETE.json','new_scientific_updates_this_owner',20000),('COMPLETE.json','new_scientific_updates_this_owner',18720.0),
    ('COMPLETE.json','scientific_updates',18720),('COMPLETE.json','actual_disposable_optimizer_calls',1),
    ('COMPLETE.json','mixed_source_continuation',False),('COMPLETE.json','held_out_donor_outcomes_opened',True),
    ('COMPLETE.json','main_dynamics_fitted',True),('COMPLETE.json','audit_history',[]),
    ('WORKER_STARTED.json','scope','synthetic'),('INPUT_LEASE.json','all_payloads_consumed_claimed',True),
    (EXIT,'same_worker_identity_check',False),(EXIT,'checked_members',2),(EXIT,'additional_count_array_scans',1),
    (OWNER,'worker_id','another-worker'),(RELEASE,'exception_type','RuntimeError'),
    (RELEASE,'automatic_restart_authorized',True),('output-readback.json','parent_payloads_opened',True),
    ('output-readback.json','parent_output_bytes',123),('output-readback.json','no_model_loading',False),
    ('startup-validated.json','recovery_contract_sha256','e'*64),('terminal.json','application_pass',False),
    ('cleanup.json','reads',[dict(job_absent=True,owned_pods=0)])])
def test_rehashed_semantic_changes_fail(api,case,name,key,value):
    case[1][name][key] = value
    with pytest.raises(ValueError):evaluate(api,case)


@pytest.mark.parametrize('mutation',[
    lambda c:c['identity'].update(recovery_parent_checkpoint_sha256='a'*64),
    lambda c:c['identity'].update(source_sha256='a'*64),
    lambda c:c['qualification_compatibility'].update(fresh_numerical_qualification_performed=True),
    lambda c:c['qualification_compatibility'].update(original_source_manifest_sha256='a'*64),
    lambda c:c['qualification_compatibility'].update(numerical_sources_unchanged=[]),
    lambda c:c['selected_export'].update(selected_updates=512),
    lambda c:c['selected_export'].update(optimizer_resume=True),
    lambda c:c['selected_export']['scaling'].update(scale_floor=-1.),
    lambda c:c['audit_history'][0].update(risk=2.),
    lambda c:c['audit_history'][0].update(independent_donors=True),
])
def test_no_false_same_source_qualification_or_reselection(api,case,mutation):
    mutation(case[1]['COMPLETE.json'])
    with pytest.raises(ValueError):evaluate(api,case)


def test_parent_portable_input_projection_allows_backend_inode_changes_only(api,case):
    lease = case[1]['INPUT_LEASE.json']
    lease['members']['fixture']['file_identity'].update(st_dev=33,st_ino=44)
    case[1][EXIT]['input_lease_sha256'] = api.digest(lease,ensure_ascii=False)
    assert evaluate(api,case)['status'] == 'metadata_accepted'
    lease['members']['fixture']['sha256'] = 'a'*64
    case[1][EXIT]['input_lease_sha256'] = api.digest(lease,ensure_ascii=False)
    with pytest.raises(ValueError,match='original fitting inputs'):evaluate(api,case)


def test_wrong_input_exit_path_rejected_before_collection(api,case):
    snapshot,_ = pack(api,case)
    for path in ('/unrelated/x.json',case[0][1]['scientific_output']+'/INPUT_EXIT.json',
                 case[0][1]['scientific_output']+'/input-exits/../'+CLAIM+'.json'):
        snapshot['records']['output/COMPLETE.json']['value']['input_exit']['path'] = path
        with pytest.raises(ValueError,match='claim-specific'):api.metadata_paths(case[0][1],snapshot)


def test_combined_storage_and_parent_intake_drift_rejected(api,case):
    context,_ = case
    context[1]['maximum_output_bytes'] = 4*api.GIB
    with pytest.raises(ValueError):evaluate(api,case)
    context[1]['maximum_output_bytes'] = context[2]['output_limit_bytes']
    context[6]['fits']['D1D2']['inventory']['fixture.pt']['st_size'] += 1
    with pytest.raises(ValueError,match='Parent inventory'):evaluate(api,case)


def test_collection_rejects_running_job_without_followup_reads(api,case,tmp_path,monkeypatch):
    context,_ = case
    monkeypatch.setattr(api,'ROOT',tmp_path)
    monkeypatch.setattr(api,'local_context',lambda _:(tmp_path/'package',*context))
    snapshot,_ = pack(api,case)
    snapshot['job'] = {'status':{'active':1}}
    calls = []
    def remote(code,values):
        calls.append(code)
        assert code == api.STATUS_CODE
        return snapshot
    monkeypatch.setattr(api,'remote',remote)
    with pytest.raises(ValueError,match='absence snapshot'):api.collect(tmp_path/'package',tmp_path/'intake.json')
    assert len(calls) == 1 and not (tmp_path/'intake.json').exists()


def test_byte_or_inventory_drift_rejected(api,case):
    snapshot,collected = pack(api,case)
    first = next(iter(collected['files'].values()))
    first['text'] += ' '
    with pytest.raises(ValueError,match='bytes'):api.validate(*case[0],snapshot,collected)
    snapshot,collected = pack(api,case)
    collected['files']['/unrelated'] = first
    with pytest.raises(ValueError,match='collection'):api.validate(*case[0],snapshot,collected)


def test_successful_collector_only_records_intake_never_reconciles(api,case,tmp_path,monkeypatch):
    monkeypatch.setattr(api,'ROOT',tmp_path)
    monkeypatch.setattr(api,'local_context',lambda _:(tmp_path/'package',*case[0]))
    snapshot,collected = pack(api,case)
    calls,records = [],[]
    def remote(code,values):
        calls.append((code,values))
        return snapshot if code == api.STATUS_CODE else collected
    monkeypatch.setattr(api,'remote',remote)
    monkeypatch.setattr(api,'record',lambda path,value:records.append((path,value)))
    result = api.collect(tmp_path/'package',tmp_path/'intake.json')
    assert len(calls) == 2 and len(records) == 1
    assert calls[1][1]['PATHS'] == api.metadata_paths(case[0][1],snapshot)
    assert not result['journal_reconciled'] and not records[0][1]['remote_writes']
    assert records[0][1]['accepted'] == result
