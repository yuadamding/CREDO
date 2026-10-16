"""Metadata-only continuation checks; no data, checkpoint loads or optimizer calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frontier import observation_continuation as continuation
from frontier import observation_worker as worker
from frontier.observation_training import ObservationTrainingSpec


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_optimizer(monkeypatch):
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        monkeypatch.setattr(cls, 'step', lambda *a, **k:pytest.fail('No optimizer calls authorized'))


def publish(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def file_ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def package(tmp_path, monkeypatch):
    def save(name, value):
        return publish(tmp_path/name, value)
    parent_root, child_root = tmp_path/'parent', tmp_path/'child'
    parent_root.mkdir(); child_root.mkdir()
    source_manifest = save('sources.json', dict(schema='credo.frontier.observation-worker-source.v1',
        files={name:file_ref(ROOT/name)['sha256'] for name in worker.SOURCE_MEMBERS}))
    original_sources_path = ROOT/'observation_d1d2_v2_launch/bundle/SOURCE_MANIFEST.json'
    old_sources = json.loads(original_sources_path.read_text())
    original = dict(mode='fit', output_limit_bytes=continuation.COMBINED_CAP,
        metadata_root='/fixture/metadata', dataset_root='/fixture/data', fold_id='subset-D1-D2',
        training_donors=['D1','D2'], metadata_index_sha256='a'*64, rna_axis_sha256='b'*64,
        input_access_sha256='c'*64, protocol_sha256='d'*64, runtime_image='fixture@sha256:'+'e'*64,
        qualification_receipt='/fixture/qualification.json', qualification_sha256='f'*64,
        source_manifest=str(original_sources_path), source_manifest_sha256=file_ref(original_sources_path)['sha256'],
        source_root=str(ROOT/'observation_d1d2_v2_launch/bundle/source'), output_root=str(parent_root))
    original_ref = save('original-contract.json', original)
    parent_identity = dict(source_sha256=original['source_manifest_sha256'], worker_contract_sha256=original_ref['sha256'])
    identity_ref = publish(parent_root/'IDENTITY.json', parent_identity)
    parent = dict(identity={**identity_ref, 'value':parent_identity}, checkpoint_payload=dict(
        path=str(parent_root/'checkpoints/update-000001280/checkpoint.pt'),sha256='1'*64,size_bytes=12))
    snapshot = dict(schema='credo.frontier.native-input-lease.v1', dataset_root='/fixture/data',
        access_sha256='c'*64, metadata_index_sha256='a'*64, source_ids=['D1_Stim8hr'],
        payload_hashes_verified_on_consumption=True, all_payloads_consumed_claimed=False,
        members={'fixture':dict(sha256='2'*64,file_identity=dict(st_size=7, st_ino=123, st_dev=45))})
    input_ref = publish(parent_root/'INPUT_LEASE.json', snapshot)
    retained = worker._output_bytes(parent_root)
    contract = dict(original, source_root=str(ROOT), source_manifest=source_manifest['path'],
        source_manifest_sha256=source_manifest['sha256'], output_root=str(child_root),
        output_limit_bytes=continuation.COMBINED_CAP-retained)
    fit_ref = save('fit-contract.json', contract)
    intake_ref = save('parent-intake.json',dict(fits={'D1D2':parent}))
    authority_ref = save('authority.json',dict(schema='credo.frontier.observation-continuation-authority.v1',
        status='approved', pairs=list(continuation.PAIRS), consumed_updates_each=1280,
        maximum_new_updates_each=18720, maximum_total_updates_each=20000,
        combined_output_limit_bytes_each=continuation.COMBINED_CAP, maximum_concurrent_h100=4,
        automatic_retries=0, elapsed_time_deadline_seconds=None, original_workload_expanded=False,
        main_model_fitting=False))
    inventory={path.relative_to(parent_root).as_posix():{key:getattr(path.stat(),key) for key in
        ('st_dev','st_ino','st_size','st_uid','st_mode','st_nlink','st_mtime_ns','st_ctime_ns')}
        for path in parent_root.iterdir()}
    inventory_raw=(json.dumps(inventory,sort_keys=True,separators=(',',':'))+'\n').encode()
    parent_inventory_ref=save('inventory.json',dict(schema='credo.frontier.observation-continuation-intake.v1',
        status='passed',authority_sha256=authority_ref['sha256'],fits={'D1D2':dict(inventory=inventory,
        inventory_sha256=hashlib.sha256(inventory_raw).hexdigest(),parent_output_bytes=retained,
        parent_input_lease={**input_ref,'value':snapshot},old_job_absent=True,new_job_absent=True,
        owned_pods_absent=True,new_roots_absent=True,terminal=dict(value=dict(status='failed',uid='fixture')),
        cleanup=dict(value=dict(status='complete',uid='fixture')))}))
    check_root = tmp_path/'checked'
    check_root.mkdir(); (check_root/'D1D2').mkdir()
    check_source = save('check-sources.json',dict(files={
        **{name:dict(sha256=sha) for name,sha in source_manifest_value(contract).items()},
        'frontier/observation_worker.py':dict(sha256=old_sources['files']['frontier/observation_worker.py']),
        'frontier/observation_recovery.py':dict(sha256=file_ref(ROOT/'frontier/observation_recovery.py')['sha256'])}))
    check_worker = dict(run_id='fixture-check',source_manifest=check_source, parent_intake=intake_ref,
        runtime_image=original['runtime_image'],expected_runtime=worker._runtime(),output_root=str(check_root))
    check_worker_ref = save('check-worker.json',check_worker)
    restore = dict(pair='D1D2',status='exact_state_restoration_verified_on_cuda',
        parent_checkpoint=parent['checkpoint_payload'],consumed_updates=1280,remaining_update_cap=18720,
        optimizer_calls=0,scientific_updates=0,parent_changed=False,training_data_opened=False,
        model_optimizer_rng_sampler_and_audit_values_equal=True,training_continuation_qualified=False)
    restore_ref = publish(check_root/'D1D2/RESTORE_VERIFIED.json',restore)
    completion = dict(schema='credo.frontier.retention-check-completion.v1',status='passed',
        run_id='fixture-check',contract=check_worker_ref,source_manifest_sha256=check_source['sha256'],
        parent_intake_sha256=intake_ref['sha256'],optimizer_calls=0,scientific_updates=0,
        canary_passed=True,parent_writes=False,training_continuation_qualified=False,parents=[restore])
    completion_ref = publish(check_root/'COMPLETE.json',completion)
    readback_ref = save('readback.json',dict(status='passed',run_id='fixture-check',
        complete_sha256=completion_ref['sha256'],files={
            'COMPLETE.json':{k:completion_ref[k] for k in ('sha256','size_bytes')},
            'D1D2/RESTORE_VERIFIED.json':{k:restore_ref[k] for k in ('sha256','size_bytes')}}))
    terminal_ref = save('terminal.json',dict(status='passed',application_pass=True,kubernetes_complete=True,
        run_id='fixture-check',result_complete_sha256=completion_ref['sha256']))
    recovery = dict(schema=continuation.SCHEMA, mode='explicit_source_change_continuation',fit_contract=fit_ref,
        authority=authority_ref,pair='D1D2',parent_intake=intake_ref,parent_inventory=parent_inventory_ref,parent_fit_contract=original_ref,
        parent_source_manifest=file_ref(original_sources_path),parent_input_lease=input_ref,
        recovery_check_worker_contract=check_worker_ref,recovery_check_completion=completion_ref,
        recovery_check_restore=restore_ref,recovery_check_readback=readback_ref,recovery_check_terminal=terminal_ref,
        parent_retained_output_bytes=retained,combined_output_limit_bytes=continuation.COMBINED_CAP,
        consumed_updates=1280,maximum_new_updates=18720,maximum_total_updates=20000,
        extra_sources={name:file_ref(ROOT/name)['sha256'] for name in continuation.EXTRA})
    qualification = dict(status='passed',hardware={})
    calls=[]
    def original_only(contract_arg,hardware_arg):
        assert contract_arg==original
        calls.append(contract_arg)
        return qualification
    monkeypatch.setattr(worker,'verify_qualification',original_only)
    monkeypatch.setattr(worker,'_qualification_hardware_compatibility',lambda *args:dict(status='compatible'))
    return SimpleNamespace(recovery=recovery, contract=contract, snapshot=snapshot,parent=parent,
        parent_root=parent_root, child_root=child_root, qualification=qualification,calls=calls,
        save=save, original=original)


def source_manifest_value(contract):
    return json.loads(Path(contract['source_manifest']).read_text())['files']


def test_explicit_reuse_preserves_original_qualification_identity(package):
    ref=package.save('recovery.json',package.recovery)
    recovery,parent,qualification,compatibility=continuation.load_recovery(ref,package.contract,{})
    assert qualification==package.qualification and parent==package.parent
    assert len(package.calls)==1
    assert compatibility['original_source_manifest_sha256'] != compatibility['continuation_source_manifest_sha256']
    assert compatibility['mixed_source_continuation'] is True
    assert compatibility['fresh_numerical_qualification_performed'] is False


@pytest.mark.parametrize('damage',['updates_reset','expanded_cap','changed_input','wrong_parent',
                                  'failed_check','unbound_extra','increased_parent_bytes'])
def test_recovery_contract_rejects_scope_or_evidence_changes(package,damage):
    value=deepcopy(package.recovery)
    if damage=='updates_reset': value['consumed_updates']=0
    elif damage=='expanded_cap': value['combined_output_limit_bytes']+=1
    elif damage=='changed_input': package.contract['input_access_sha256']='0'*64
    elif damage=='wrong_parent': value['parent_fit_contract']['sha256']='0'*64
    elif damage=='failed_check':
        record=continuation.read_reference(value['recovery_check_completion']);record['status']='failed'
        value['recovery_check_completion']=package.save('failed.json',record)
    elif damage=='unbound_extra': value['extra_sources']['frontier/observation_recovery.py']='0'*64
    else: (package.parent_root/'unexpected.bin').write_bytes(b'x')
    ref=package.save('recovery.json',value)
    with pytest.raises(ValueError): continuation.load_recovery(ref,package.contract,{})
    assert package.calls==[]


def test_parent_input_comparison_preserves_worker_local_stat_distinction(package):
    fresh=deepcopy(package.snapshot)
    fresh['members']['fixture']['file_identity']['st_ino']=999
    fresh['members']['fixture']['file_identity']['st_dev']=888
    continuation.verify_input_snapshot(package.recovery,fresh)
    fresh['members']['fixture']['sha256']='0'*64
    with pytest.raises(ValueError,match='input hashes/sizes/roles'):
        continuation.verify_input_snapshot(package.recovery,fresh)


def test_combined_capacity_does_not_reset_parent_allowance(package):
    ref=package.save('recovery.json',package.recovery)
    (package.child_root/'child.txt').write_bytes(b'abc')
    result=continuation.combined_capacity(package.child_root,package.contract,ref,reserve=5)
    assert result['combined_output_bytes']==package.recovery['parent_retained_output_bytes']+3
    assert result['combined_output_limit_bytes']==4*1024**3
    with pytest.raises(RuntimeError,match='allowance'):
        continuation.combined_capacity(package.child_root,package.contract,ref,reserve=4*1024**3)
    (package.parent_root/'added.txt').write_bytes(b'x')
    with pytest.raises(ValueError,match='Retained parent'):
        continuation.combined_capacity(package.child_root,package.contract,ref)


def test_worker_change_scope_rejects_training_or_ordinary_qualification_edits():
    old=(ROOT/'observation_d1d2_v2_launch/bundle/source/frontier/observation_worker.py').read_text()
    current=(ROOT/'frontier/observation_worker.py').read_text()
    continuation._worker_change_scope(old,current)
    with pytest.raises(ValueError,match='fitter/retention'):
        continuation._worker_change_scope(old,current.replace('owner.run_until(target)','owner.run_until(target + 1)'))
    with pytest.raises(ValueError,match='outside'):
        continuation._worker_change_scope(old,current+'\ndef surprise():\n    return True\n')


def test_scientific_continuation_cannot_enter_cpu_fixture():
    lease=SimpleNamespace(assert_active=lambda:None)
    with pytest.raises(ValueError,match='CPU fixture'):
        worker.run_fitting_core(lease,None,ObservationTrainingSpec.fixture(),{},device='cpu',qualification={},
                                continuation=dict(path='/fixture',sha256='0'*64,size_bytes=1))


@pytest.mark.parametrize('fail', [False, True])
def test_continuation_lifecycle_routes_only_remaining_work_without_optimizer_calls(tmp_path,monkeypatch,fail):
    """Mocked bookkeeping only, not a GPU run or synthetic training qualification."""
    root=tmp_path/'child'; root.mkdir()
    lease=SimpleNamespace(root=root,identity={'fixture':'counter-routing-only'},assert_active=lambda:None)
    spec=ObservationTrainingSpec()
    contract=dict(mode='fit',study_root=str(tmp_path),output_limit_bytes=1024**3,
                  qualification_receipt='/fixture/q',qualification_sha256='a'*64)
    context=dict(path='/fixture/recovery',sha256='b'*64,size_bytes=1)
    calls=[]
    owner=SimpleNamespace(accepted_updates=1280,attempted_updates=1280,optimizer_calls=1280,
        model=torch.nn.Linear(1,1),exposure_by_source={'fixture':1280*2048},last_checkpoint={},
        selected_candidate={},audit_history=[],_audit_and_publish=lambda:None,_checkpoint=lambda:None)
    def run_until(target):
        calls.append(target)
        if fail:
            raise RuntimeError('fabricated continuation failure before any update')
        # No model or optimizer operations: only exercise control bookkeeping.
        owner.accepted_updates=owner.attempted_updates=owner.optimizer_calls=target
        owner.exposure_by_source={'fixture':target*2048}
    owner.run_until=run_until
    owner.finalize_selected=lambda:(None,{'fixture':'selected'})
    monkeypatch.setattr(worker.ObservationFitter,'resume',lambda *a,**k:owner)
    monkeypatch.setattr(continuation,'verify_for_execution',lambda *a,**k:{'fixture':'validated-by-separate-tests'})
    monkeypatch.setattr(continuation,'combined_capacity',lambda *a,**k:{'fixture':'combined-cap'})
    monkeypatch.setattr(worker,'_input_lease_receipt',lambda *a,**k:{'fixture':'new-child-input'})
    monkeypatch.setattr(worker,'_input_exit_receipt',lambda *a,**k:{'fixture':'claim-specific-exit'})
    monkeypatch.setattr(worker,'RetentionPolicy',lambda *a,**k:None)
    monkeypatch.setattr(worker,'bind_policy',lambda *a,**k:None)
    monkeypatch.setattr(worker,'prepare_retirement',lambda *a,**k:None)
    monkeypatch.setattr(worker,'_resources',lambda *a,**k:{'fixture':'no-cuda-call'})
    arguments=dict(device='cuda:0',qualification={'status':'passed','hardware':{}},
        resume={'directory':'/fixture/imported','sha256':'c'*64},hardware={},continuation=context)
    if fail:
        with pytest.raises(RuntimeError,match='fabricated continuation'):
            worker.run_fitting_core(lease,None,spec,contract,**arguments)
        result=json.loads((root/'WORKER_FAILED.json').read_text())
        assert result['counters']==dict(accepted_updates=1280,attempted_updates=1280,optimizer_calls=1280)
        assert calls==[1536] and not (root/'COMPLETE.json').exists()
    else:
        result,_=worker.run_fitting_core(lease,None,spec,contract,**arguments)
        assert calls[0]==1536 and calls[-1]==20000 and all(target>1280 for target in calls)
        assert result['resume_from_accepted_updates']==result['consumed_parent_updates']==1280
        assert result['new_scientific_updates_this_owner']==18720
        assert result['scientific_updates']==20000 and result['mixed_source_continuation'] is True
        assert result['source_continuation_contract']==context
