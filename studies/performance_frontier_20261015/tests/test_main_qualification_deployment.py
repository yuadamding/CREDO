"""Fabricated lifecycle/metadata tests only; no remote, CUDA or optimizer work."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import pytest

from deployment import main_qualification_build as build
from deployment import main_qualification_common as common
from deployment import main_qualification_control as control
from deployment import main_qualification_entry as entry
from deployment.common import encoded, identity, publish
from frontier.coordination import WorkBinding
from frontier.main_qualification_contract import settings, validate_parent
from frontier.main_qualification_plan import plan_identity
from frontier.main_qualification_artifacts import file_reference
from frontier.main_qualification_contract import decode
from frontier.retention import identity_sha256, STUDY_ID
from test_main_worker import no_numerics
from test_native_inputs import READER
from test_observation_deployment import RETAINED, module


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    project = tmp_path/STUDY_ID
    scratch = tmp_path/'control'/STUDY_ID
    for m in (build, common):
        monkeypatch.setattr(m, 'PROJECT', str(project))
        monkeypatch.setattr(m, 'CONTROL', str(scratch))
    source = json.loads((ROOT/'main_qualification_source_20261016_v2/PACKAGE.json').read_bytes())
    authority = dict(schema='credo.frontier.main-qualification-authority.v1', status='authorized',
        run_id='credo-frontier-main-qualification-test-v1', plan_sha256=plan_identity(),
        settings_sha256=identity_sha256(settings()), source_manifest_sha256=source['source_manifest_sha256'],
        worker_source_manifest_sha256=source['worker_source_manifest_sha256'],
        planned_disposable_optimizer_calls=111, maximum_disposable_optimizer_calls=112,
        scientific_updates=0, real_biological_payload_access=False, main_scientific_fitting=False,
        automatic_retry=False, automatic_successor=False, elapsed_time_deadline=None,
        fixture_only_not_real_authority=True)
    admission = dict(schema='credo.frontier.main-qualification-admission.v1', status='reserved',
        run_id=authority['run_id'], work_binding=WorkBinding.other('qualification', plan_identity()).to_dict(),
        phase_authority_sha256=identity(encoded(authority))['sha256'],
        source_manifest_sha256=source['source_manifest_sha256'],
        worker_source_manifest_sha256=source['worker_source_manifest_sha256'],
        scientific_update_reservation=0, storage_reserved_bytes=16*GIB, disposable_optimizer_call_limit=112,
        automatic_retry=False, automatic_successor=False, journal_event_sha256='a'*64,
        journal_identity_sha256='b'*64, capacity_receipt_sha256='c'*64,
        aggregate_reserved_output_bytes=32*GIB, live_h100_reservations=2)
    payload, archive, report = build.construct(ROOT, READER, RETAINED, 'main_qualification_v1',
                                              authority['run_id'], authority, admission)
    plan = json.loads(payload['PLAN.json'])
    bundle = Path(plan['bundle_root'])
    for name, raw in payload.items():
        path = bundle/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    worker = json.loads(payload['WORKER_CONTRACT.json'])
    monkeypatch.setattr(control, 'BASE', bundle)
    monkeypatch.setattr(control, 'OWNER_UID', os.getuid())
    return SimpleNamespace(bundle=bundle, payload=payload, archive=archive, report=report,
        plan=plan, worker=worker, authority=authority, admission=admission,
        output=Path(plan['scientific_output']), handshake=Path(plan['worker_output']),
        root=Path(plan['control_root'])/'gpu')


def gpu():
    return dict(count=1, name='NVIDIA H100 80GB HBM3', compute_capability=[9,0], total_memory_bytes=80*GIB)


def limits():
    return dict(cgroup_version=2, cpu_limit=24, memory_limit_bytes=200*GIB,
                application_threads=8, affinity_cpus=64)


def core_for(p, monkeypatch):
    p.root.mkdir(parents=True)
    p.handshake.mkdir()
    p.binding = dict(job=p.plan['run_id'], uid='fake-job-uid')
    p.pod = dict(metadata=dict(name='fake-pod', uid='fake-pod-uid'))
    p.calls = []
    core = SimpleNamespace(pods=lambda *args:[p.pod], publish=publish,
        sha=lambda path:identity(path.read_bytes())['sha256'], digest=common.semantic_sha,
        audit_runtime_pod=lambda *args:p.calls.append('pod-audit'))
    monkeypatch.setattr(control, 'core', core)
    capacity = SimpleNamespace(collect_capacity=lambda *args:dict(fixture=True, status='passed'),
                              validate_capacity=lambda *args:p.calls.append('capacity-audit'))
    monkeypatch.setitem(sys.modules, 'capacity', capacity)
    startup = dict(schema='credo.frontier.main-qualification-startup.v1', status='startup_guard_passed',
        run_id=p.plan['run_id'], plan_sha256=p.report['plan_sha256'], worker_contract_sha256=p.plan['worker_contract']['sha256'],
        image=common.IMAGE, pod_uid='fake-pod-uid', pod_name='fake-pod',
        writer_canary_removed=True, uid=307469, gid=600815, real_data_opened=False, optimizer_calls=0,
        gpu=gpu(), resources=limits())
    publish(p.handshake/'STARTUP.json', startup)
    return core


def terminal_fixture(p, monkeypatch):
    core_for(p, monkeypatch)
    control.release_startup(p.plan, p.binding, p.root, {})
    p.output.mkdir()
    release = p.handshake/'STARTUP_RELEASE.json'
    publish(p.handshake/'WORKER_EXIT.json', dict(returncode=0, run_id=p.plan['run_id']))
    publish(p.output/'EXECUTION_COMPLETE.json', dict(fixture_only_not_numerical=True))
    (p.output/'opaque.pt').write_bytes(b'not a trained or serialized checkpoint')
    files = control.inventory(p.output, 16*GIB)
    verified = dict(schema='credo.frontier.main-qualification-terminal-readback.v1',
        status='complete_numerical_publication_inventory_verified', contract=p.plan['worker_contract'],
        execution=common.reference(p.output/'EXECUTION_COMPLETE.json', (p.output/'EXECUTION_COMPLETE.json').read_bytes()),
        precision_cases=96, owner_cases=48, worker_cases=3, owner_terminal_checkpoint_payloads=52,
        owner_all_checkpoint_payloads=104, worker_all_checkpoint_payloads=7, disposable_optimizer_calls_reported=111,
        scientific_updates=0, checkpoint_deserializations=0, numerical_replay=False, real_biological_payload_access=False,
        scheduler_terminal_verified=False, main_worker_qualification_pass=False, scientific_launch_authority=False,
        source_manifest_sha256=p.worker['source_manifest']['sha256'],
        worker_source_manifest_sha256=p.worker['worker_source_manifest']['sha256'], files=files,
        total_bytes=sum(r['size_bytes'] for r in files.values()))
    publish(p.output/'TERMINAL_READBACK.json', verified)
    (p.output/'READBACK.log').write_bytes(b'fixture log\n')
    publish(p.output/'READBACK_EXIT.json', dict(returncode=0, automatic_retry=False))
    complete = dict(schema='credo.frontier.main-qualification-execution.v1',
        status='components_complete_after_independent_terminal_readback', contract=p.plan['worker_contract'],
        release=common.reference(release, release.read_bytes()),
        source_manifest_sha256=p.worker['source_manifest']['sha256'],
        worker_source_manifest_sha256=p.worker['worker_source_manifest']['sha256'],
        reported_disposable_optimizer_calls=111, maximum_disposable_optimizer_calls=112,
        scientific_updates=0, real_biological_payload_access=False, scientific_launch_authority=False,
        main_worker_qualification_pass=False, automatic_retry=False, automatic_successor=False,
        all_parent_source_and_metadata_exits_verified=True, execution_record=verified['execution'],
        terminal_readback=common.reference(p.output/'TERMINAL_READBACK.json', (p.output/'TERMINAL_READBACK.json').read_bytes()))
    publish(p.output/'COMPLETE.json', complete)


def test_bundle_matches_numerical_contract_and_pinned_surface(packaged):
    p = packaged
    worker, admission = common.verify_bundle(p.plan, p.bundle)
    assert worker == p.worker and admission == p.admission
    decode(p.payload['WORKER_CONTRACT.json'])
    manifest = json.loads(p.payload['gpu-job.json'])
    assert manifest['spec']['suspend'] and manifest['spec']['backoffLimit'] == 0
    pod = manifest['spec']['template']['spec']
    old = json.loads((RETAINED/'gpu-job.json').read_bytes())['spec']['template']['spec']
    for k in ('volumes', 'securityContext', 'automountServiceAccountToken', 'enableServiceLinks'):
        assert pod[k] == old[k]
    for k in ('image', 'resources', 'securityContext', 'volumeMounts'):
        assert pod['containers'][0][k] == old['containers'][0][k]
    assert pod['containers'][0]['command'][-1] == p.report['plan_sha256']
    assert 'activeDeadlineSeconds' not in pod and 'activeDeadlineSeconds' not in manifest['spec']
    assert not p.report['cuda_qualification'] and not p.report['scheduler_submitted']
    static = module(p.bundle/'gpu_v2_static.py', 'main_qualification_test_static')
    assert static.validate(p.bundle, p.plan)
    for name in (*build.ADAPTERS, 'lifecycle_core.py', 'runtime_policy.py'):
        ast.parse(p.payload[name].decode(), feature_version=(3,6))


def test_actual_packaged_host_imports_in_fresh_cpu_process(packaged):
    script = "import sys;sys.path.insert(0,sys.argv[1]);import control,worker_entry;print('imports-only-no-execution')"
    result=subprocess.run([sys.executable,'-I','-c',script,str(packaged.bundle)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=True)
    assert result.stdout.strip()==b'imports-only-no-execution'
    assert not packaged.output.exists() and not packaged.handshake.exists()


@pytest.mark.parametrize('section,key,value', [
    ('plan','mode','qualify'), ('plan','maximum_new_scientific_updates',1),
    ('plan','elapsed_time_deadline_seconds',3600), ('plan','gpu_ceiling',2),
    ('authority','main_scientific_fitting',True), ('authority','planned_disposable_optimizer_calls',112),
    ('authority','schema','credo.frontier.observation-phase-authority.v1'),
    ('admission','live_h100_reservations',5), ('admission','scientific_update_reservation',1),
    ('admission','aggregate_reserved_output_bytes',385*GIB), ('admission','live_h100_reservations',True)])
def test_cross_scope_and_excess_resources_rejected(packaged, section, key, value):
    p = packaged
    changed = dict(getattr(p,section), **{key:value})
    with pytest.raises(ValueError):
        if section == 'plan': common.validate_plan(changed)
        elif section == 'authority': common.validate_authority(p.worker, changed)
        else: common.validate_admission(p.worker, changed)


def test_source_drift_rejected_before_worker_start(packaged):
    p = packaged
    (p.bundle/'source/frontier/models.py').write_bytes(b'changed')
    with pytest.raises(ValueError): common.verify_bundle(p.plan, p.bundle)


def test_actual_release_satisfies_parent_contract(packaged, monkeypatch):
    p = packaged
    core_for(p, monkeypatch)
    control.release_startup(p.plan,p.binding,p.root,{})
    assert p.calls == ['pod-audit','capacity-audit']
    release = json.loads((p.handshake/'STARTUP_RELEASE.json').read_bytes())
    record, refs = decode(p.payload['WORKER_CONTRACT.json'])
    with monkeypatch.context() as m:
        m.setattr(os,'getuid',lambda:307469);m.setattr(os,'getgid',lambda:600815)
        validate_parent(record,refs,file_reference(p.bundle/'WORKER_CONTRACT.json'),p.authority,p.admission,
            release,dict(CREDO_RUN_ID=p.plan['run_id'],CREDO_POD_NAME='fake-pod',CREDO_POD_UID='fake-pod-uid',
                CREDO_V4_EXECUTION_IMAGE_DIGEST=common.IMAGE.split('@')[1]), build.RUNTIME)
    assert release['scientific_fit_authorized'] is False
    before=(p.handshake/'STARTUP_RELEASE.json').read_bytes()
    control.release_startup(p.plan,p.binding,p.root,{})
    assert (p.handshake/'STARTUP_RELEASE.json').read_bytes()==before


def test_missing_cgroup_proof_does_not_release(packaged, monkeypatch):
    p = packaged
    core_for(p,monkeypatch)
    path=p.handshake/'STARTUP.json';v=json.loads(path.read_bytes());v['resources']['cpu_limit']=8
    path.write_bytes(encoded(v))
    with pytest.raises(ValueError):control.release_startup(p.plan,p.binding,p.root,{})
    assert not (p.handshake/'STARTUP_RELEASE.json').exists()


def test_terminal_acceptance_is_byte_readback_not_scientific_permission(packaged,monkeypatch):
    p=packaged;terminal_fixture(p,monkeypatch)
    sha=control.validate_result(p.plan,p.binding,p.root,[p.pod])
    assert sha==identity((p.output/'COMPLETE.json').read_bytes())['sha256']
    receipt=json.loads((p.root/'output-readback.json').read_bytes())
    assert receipt['no_model_loading'] and not receipt['scientific_launch_authority']
    assert set(receipt['files'])=={'opaque.pt','EXECUTION_COMPLETE.json','TERMINAL_READBACK.json',
                                  'READBACK.log','READBACK_EXIT.json','COMPLETE.json'}


@pytest.mark.parametrize('change',['bytes','extra','failure','exit','release','inventory'])
def test_terminal_rejects_incomplete_or_changed_publication(packaged,monkeypatch,change):
    p=packaged;terminal_fixture(p,monkeypatch)
    if change=='bytes':(p.output/'opaque.pt').write_bytes(b'changed')
    elif change=='extra':(p.output/'unexpected').write_bytes(b'extra')
    elif change=='failure':(p.output/'FAILED.json').write_bytes(b'{}')
    elif change=='exit':(p.handshake/'WORKER_EXIT.json').write_bytes(encoded(dict(returncode=1,run_id=p.plan['run_id'])))
    elif change=='release':(p.handshake/'STARTUP_RELEASE.json').write_bytes(b'{}')
    else:(p.output/'TERMINAL_READBACK.json').write_bytes(b'{}')
    with pytest.raises((ValueError,KeyError)):control.validate_result(p.plan,p.binding,p.root,[p.pod])
    assert not (p.root/'output-readback.json').exists()


@pytest.mark.parametrize('version',[1,2])
def test_cgroup_guard_uses_actual_limits(tmp_path,monkeypatch,version):
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(n,'8')
    monkeypatch.setattr(os,'sched_getaffinity',lambda _:set(range(24)))
    if version==2:
        (tmp_path/'cgroup.controllers').write_text('cpu memory')
        (tmp_path/'cpu.max').write_text('2400000 100000')
        m=tmp_path/'memory.max';m.write_text(str(200*GIB))
    else:
        (tmp_path/'cpu').mkdir();(tmp_path/'memory').mkdir()
        (tmp_path/'cpu/cpu.cfs_quota_us').write_text('2400000')
        (tmp_path/'cpu/cpu.cfs_period_us').write_text('100000')
        m=tmp_path/'memory/memory.limit_in_bytes';m.write_text(str(200*GIB))
    assert entry.resources(tmp_path)['cgroup_version']==version
    m.write_text(str(199*GIB))
    with pytest.raises(ValueError):entry.resources(tmp_path)


def test_worker_command_is_distinct_and_release_bound(packaged):
    p=packaged
    cmd=entry.command_for(p.plan,p.worker,p.handshake/'STARTUP_RELEASE.json','f'*64)
    assert cmd[2:4]==['frontier.main_qualification_worker','run']
    assert cmd[-2:]==['--release-sha256','f'*64]
    assert 'frontier.observation_worker' not in cmd


@pytest.mark.parametrize('exit_code',[0,1])
def test_packaged_entry_releases_once_and_never_retries(packaged,monkeypatch,exit_code):
    p=packaged
    p.root.mkdir(parents=True)
    binding=dict(job=p.plan['run_id'],uid='fake-job-uid')
    pod=dict(metadata=dict(name='fake-pod',uid='fake-pod-uid'))
    monkeypatch.setattr(control,'core',SimpleNamespace(pods=lambda *args:[pod],publish=publish,
        sha=lambda path:identity(path.read_bytes())['sha256'],digest=common.semantic_sha,
        audit_runtime_pod=lambda *args:None))
    monkeypatch.setitem(sys.modules,'capacity',SimpleNamespace(
        collect_capacity=lambda *args:dict(fixture=True,status='passed'),validate_capacity=lambda *args:None))
    monkeypatch.setattr(entry,'__file__',str(p.bundle/'worker_entry.py'))
    monkeypatch.setattr(entry,'resources',limits)
    monkeypatch.setattr(os,'getuid',lambda:307469);monkeypatch.setattr(os,'getgid',lambda:600815)
    for k,v in dict(CREDO_RUN_ID=p.plan['run_id'],CREDO_POD_NAME='fake-pod',CREDO_POD_UID='fake-pod-uid',
                    CREDO_V4_EXECUTION_IMAGE_DIGEST=common.IMAGE.split('@')[1]).items():monkeypatch.setenv(k,v)
    def publish_then_release(path,value):
        ref=publish(path,value)
        if path.name=='STARTUP.json':control.release_startup(p.plan,binding,p.root,{})
        return ref
    monkeypatch.setattr(entry,'publish',publish_then_release)
    commands=[]
    def mocked_run(command,**kwargs):
        commands.append((command,kwargs))
        if command[1]=='-c':return SimpleNamespace(stdout=json.dumps(gpu()).encode(),returncode=0)
        assert command[2:4]==['frontier.main_qualification_worker','run']
        assert 'timeout' not in kwargs and kwargs['env']['PYTHONPATH']==p.worker['source_root']
        assert kwargs['cwd']==p.worker['source_root']
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(entry.subprocess,'run',mocked_run)
    args=['--plan',str(p.bundle/'PLAN.json'),'--plan-sha256',p.report['plan_sha256']]
    if exit_code:
        with pytest.raises(RuntimeError,match='no automatic retry'):entry.main(args)
    else:assert entry.main(args)==0
    assert len(commands)==2
    assert json.loads((p.handshake/'WORKER_EXIT.json').read_bytes())['returncode']==exit_code
    assert (p.handshake/'ENTRY_FAILED.json').exists()==bool(exit_code)
    assert not p.output.exists()  # Worker computation was explicitly mocked.


@pytest.mark.parametrize('passed,failed,readback_ok',[(True,False,True),(True,False,False),(False,True,False)])
def test_terminal_monitor_captures_before_cleanup_without_successor(packaged,monkeypatch,passed,failed,readback_ok):
    p=packaged;core_for(p,monkeypatch)
    publish(p.root/'startup-validated.json',dict(fixture=True))
    actions=[]
    c=control.core
    c.get_job=lambda job:dict(fixture=True)
    c.verify=lambda *args:actions.append('verify')
    c.complete=lambda job:(passed,failed)
    c.capture=lambda *args:actions.append('capture') or [p.pod]
    c.cleanup=lambda *args:actions.append('cleanup')
    def readback(*args):
        actions.append('readback')
        if not readback_ok:raise ValueError('injected terminal mismatch')
        return 'e'*64
    monkeypatch.setattr(control,'validate_result',readback)
    if passed and not failed and readback_ok:control.monitor(p.plan,p.binding,p.root)
    else:
        with pytest.raises(RuntimeError,match='without retry'):control.monitor(p.plan,p.binding,p.root)
    assert actions==(['verify','capture','readback','cleanup'] if passed else ['verify','capture','cleanup'])
    terminal=json.loads((p.root/'terminal.json').read_bytes())
    assert terminal['application_pass']==bool(passed and not failed and readback_ok)
    assert terminal['automatic_successor'] is False and terminal['scientific_launch_authority'] is False


def test_host_inventory_rejects_links_and_byte_limit(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'OWNER_UID',os.getuid())
    p=tmp_path/'file';p.write_bytes(b'opaque')
    assert control.inventory(tmp_path,6)=={'file':identity(b'opaque')}
    with pytest.raises(ValueError):control.inventory(tmp_path,5)
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):control.inventory(tmp_path,1024)


def test_authority_preserves_original_approved_bytes(packaged):
    p=packaged
    raw=(json.dumps(p.authority,indent=2)+'\n').encode()
    admission=dict(p.admission,phase_authority_sha256=identity(raw)['sha256'])
    payload,_,_=build.construct(ROOT,READER,RETAINED,'main_qualification_v1',p.plan['run_id'],raw,admission)
    assert payload['MAIN_QUALIFICATION_AUTHORITY.json']==raw
    assert json.loads(payload['WORKER_CONTRACT.json'])['authority']['sha256']==identity(raw)['sha256']


@pytest.fixture
def operators(monkeypatch):
    transport=module(ROOT/'operations/transport.py','main_qual_test_transport')
    monkeypatch.setitem(sys.modules,'transport',transport)
    retained=module(ROOT/'operations/observation.py','main_qual_test_retained_ops')
    monkeypatch.setitem(sys.modules,'observation',retained)
    operator=module(ROOT/'operations/main_qualification.py','main_qual_test_ops')
    reserve=module(ROOT/'operations/reserve_main_qualification.py','main_qual_test_reserve')
    return operator,reserve,retained


def test_operator_reads_exact_package_without_remote_work(packaged,operators,monkeypatch):
    p=packaged;op,reserve,retained=operators
    monkeypatch.setattr(op,'ROOT',p.bundle.parent.parent)
    package=p.bundle.parent
    (package/'BUILD.json').write_bytes(encoded(p.report))
    (package/'bundle.tar.gz').write_bytes(p.archive)
    actual=op.package(package)
    assert actual==(package,p.report,p.plan)
    (p.bundle/'unexpected.py').write_bytes(b'pass')
    with pytest.raises(ValueError,match='inventory'):op.package(package)


@pytest.mark.parametrize('action',['preflight','stage','closeout','dryrun','create','launch'])
def test_operator_routes_only_after_live_reservation_gate(packaged,operators,monkeypatch,action):
    p=packaged;op,reserve,retained=operators
    monkeypatch.setattr(op,'package',lambda _:(p.bundle.parent,p.report,p.plan))
    calls=[]
    monkeypatch.setattr(retained,'session_gate',lambda _:calls.append('session'))
    def admitted(*args):calls.append('admission');return p.admission
    monkeypatch.setattr(retained,'admission_gate',admitted)
    for name in ('preflight','stage','closeout'):
        monkeypatch.setattr(retained,name,lambda *a,n=name:calls.append(n) or dict(fixture=True))
    monkeypatch.setattr(retained,'lifecycle',lambda *a:calls.append(a[-1]) or dict(fixture=True))
    args=[action,'--package','fixture','--session-receipt','fixture']
    op.main(args)
    assert calls==['session','admission',action]
    def rejected(*args):raise ValueError('live reservation changed')
    monkeypatch.setattr(retained,'admission_gate',rejected)
    with pytest.raises(ValueError,match='live reservation'):op.main(args)
    assert calls==['session','admission',action,'session']


@pytest.mark.parametrize('occupied',[0,4])
def test_reservation_uses_real_temporary_journal_and_blocks_fifth_worker(packaged,operators,monkeypatch,occupied):
    p=packaged;op,reserve,retained=operators
    project=p.bundle.parent.parent
    monkeypatch.setattr(reserve,'ROOT',project)
    real_closure=reserve.source_closure
    monkeypatch.setattr(reserve,'source_closure',lambda root,**kw:real_closure(ROOT,**kw))
    monkeypatch.setattr(reserve,'record',lambda path,v:publish(path,v))
    (project/'AUTHORITY.json').write_bytes(encoded(dict(fixture_only_not_real_authority=True)))
    cap=project/'capacity.json'
    cap.write_bytes(encoded(dict(schema='credo.frontier.storage-intake.v1',study_id=STUDY_ID,
        fileset='h_bcb-yding4',remote_writes=False,numerical_payloads_opened=False,
        filesystem_available_inodes=50000,headroom_bytes=400*GIB,filesystem_available_bytes=400*GIB)))
    if occupied:
        with reserve.AdmissionJournal(project/'admission',study_identity_sha256=reserve.sha(project/'AUTHORITY.json')) as j:
            for index in range(occupied):
                j.reserve('prior-'+str(index),worker_id='prior-'+str(index),kind='qualification',scientific_updates=0,
                    output_bytes=GIB,admission_sha256='a'*64,capacity_receipt_sha256='b'*64,
                    fileset_headroom_bytes=400*GIB,work_binding=WorkBinding.other('qualification',str(index+1)*64))
    out=project/'new-admission.json'
    if occupied:
        with pytest.raises(ValueError):reserve.reserve(p.bundle/'MAIN_QUALIFICATION_AUTHORITY.json',cap,out)
        assert not out.exists()
    else:
        result=reserve.reserve(p.bundle/'MAIN_QUALIFICATION_AUTHORITY.json',cap,out)
        common.validate_admission(p.worker,result)
        assert result['live_h100_reservations']==1 and not result['scheduler_submitted']
        with pytest.raises(ValueError):reserve.reserve(p.bundle/'MAIN_QUALIFICATION_AUTHORITY.json',cap,out)
