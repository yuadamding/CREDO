"""One explicitly approved zero-work replacement of the failed v3 continuations."""
import argparse
import json
from pathlib import Path
import sys
from transport import ROOT, remote, record, capacity
import observation as ops
from intake_observation_continuation import CODE as PARENT_CODE
sys.path.insert(0, str(ROOT))
from deployment.common import GIB, identity, read
from deployment.continuation_build import PAIRS, source_manifest
from frontier.coordination import AdmissionJournal, LIVE

AUTHORITY = ROOT/'OBSERVATION_CONTINUATION_V4_REPLACEMENT_AUTHORITY.json'
INTAKE = ROOT/'receipts/OBSERVATION_CONTINUATION_V4_REPLACEMENT_INTAKE_20261016.json'
CAPACITY = ROOT/'receipts/OBSERVATION_CONTINUATION_V4_CAPACITY_20261016.json'
FAILURE = ROOT/'receipts/OBSERVATION_CONTINUATION_V3_FAILURE_CLOSURE_20261016.json'
BATCH = ROOT/'observation_continuation_v4_admission'


def authority():
    value = json.loads(read(AUTHORITY))
    expected = dict(schema='credo.frontier.observation-continuation-replacement-authority.v1',
        status='approved', user_instruction='yes', pairs=list(PAIRS), failed_generation='v3',
        replacement_generation='v4', per_pair_explicit_replacement_attempts=1,
        consumed_updates_each=1280, maximum_new_updates_each=18720, maximum_total_updates_each=20000,
        maximum_new_updates_all_four=74880, required_failed_child_new_updates=0,
        combined_output_limit_bytes_each=4*GIB, maximum_concurrent_h100=4,
        automatic_retries=0, automatic_successor=False, elapsed_time_deadline_seconds=None,
        original_workload_expanded=False, main_model_fitting=False, numerical_worker_sources_changed=False)
    if any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact explicit replacement authority differs')
    return value


INTAKE_CODE = PARENT_CODE.split('result={}\nfor pair,row in ROWS.items():')[0] + r'''
import socket
if socket.gethostname()!=EXPECTED_BACKEND:
 raise RuntimeError('Old supervisor absence requires its observed approved backend')
result={}
for pair,row in ROWS.items():
 for name in (row['run_id'],row['new_run_id']):
  if query(['get','job/'+name,'--ignore-not-found=true','-o','json']) is not None or query(['get','pods','-l','job-name='+name,'-o','json'])['items']:
   raise RuntimeError('Exact failed child or replacement Job/Pods exist')
 for path in row['new_roots']:
  if os.path.lexists(path):raise RuntimeError('Replacement root already exists')
 if os.path.lexists(row['output_root']):raise RuntimeError('Failed child has output; zero-work replacement is not established')
 if metadata(row['bundle_root']+'/PLAN.json')['sha256']!=row['plan_sha256']:
  raise RuntimeError('Failed child plan changed')
 terminal=metadata(row['control_root']+'/gpu/terminal.json')
 cleanup=metadata(row['control_root']+'/gpu/cleanup.json')
 if terminal['value']['uid']!=row['uid'] or terminal['value']['status']!='failed' or cleanup['value']['uid']!=row['uid'] or cleanup['value']['status']!='complete':
  raise RuntimeError('Failed child terminal/cleanup differs')
 log=Path(row['control_root'])/'gpu/pod-00.log'
 info=log.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>1500000:
  raise RuntimeError('Exact bounded failure log differs')
 raw=log.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=row['failure_log_sha256']:
  raise RuntimeError('Frozen pre-input failure log changed')
 launch=metadata(row['control_root']+'/gpu/launch-success.json')
 process=launch['value']['process']
 proc=Path('/proc')/str(process['pid'])
 process_absence='pid_absent'
 try:
  info=proc.stat(); parts=(proc/'stat').read_text().rsplit(')',1)[1].split()
  current=dict(pid=process['pid'],pgrp=int(parts[2]),session=int(parts[3]),tty=int(parts[4]),start_ticks=int(parts[19]))
  if info.st_uid==307469 and all(current[k]==process[k] for k in current):
   current['cmdline_sha256']=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()
   if current==process and parts[0]!='Z':raise RuntimeError('Exact old supervisor still live')
   process_absence='bound_process_not_live'
  else:process_absence='pid_not_bound_process'
 except (FileNotFoundError,ProcessLookupError):pass
 parent=inventory(Path(row['original_output_root']))
 expected=row['parent_inventory']
 portable=lambda x:{p:{k:v for k,v in s.items() if k not in ('st_dev','st_ino')} for p,s in x.items()}
 if portable(parent)!=portable(expected) or inventory(Path(row['original_output_root']))!=parent:
  raise RuntimeError('Original retained parent changed')
 result[pair]=dict(run_id=row['run_id'],uid=row['uid'],new_run_id=row['new_run_id'],
  terminal=terminal,cleanup=cleanup,failed_job_absent=True,failed_owned_pods_absent=True,
  new_job_absent=True,new_roots_absent=True,failed_scientific_output_absent=True,new_scientific_updates=0,
  zero_update_basis='unchanged pre-input failure trace plus absent scientific output',
  failure_log_sha256=row['failure_log_sha256'],supervisor_absence=process_absence,
  supervisor_backend=socket.gethostname(),supervisor_binding=process,
  parent_inventory_unchanged=True,parent_output_bytes=sum(s['st_size'] for s in parent.values()))
print('RECEIPT='+json.dumps(dict(schema='credo.frontier.observation-continuation-replacement-intake.v1',
 status='passed',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 authority_sha256=AUTHORITY_SHA,prior_failure_closure_sha256=FAILURE_SHA,
 fits=result,remote_writes=False,numerical_payloads_opened=False)))
'''


def intake():
    authority()
    if INTAKE.exists() or CAPACITY.exists():raise ValueError('Intake receipt exists; reconcile, never overwrite')
    retained=json.loads(read(ROOT/'receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json'))
    closure=json.loads(read(FAILURE));rows={}
    for item in closure['fits']:
        pair=item['pair']; path,build,plan=ops.package(ROOT/('observation_'+pair.lower()+'_v3_launch'))
        prior=json.loads(read(item['snapshot']['path'],item['snapshot']['sha256']))
        if pair=='D1D2':
            log=json.loads(read(ROOT/'receipts/OBSERVATION_CONTINUATION_V3_D1D2_FAILURE_TRACE_20261016.json'))['records']['pod-00.log']
        else:log=prior['failure_log']
        _,_,original=ops.package(ROOT/('observation_'+pair.lower()+'_v2_launch'))
        phase='observation_'+pair.lower()+'_v4'
        rows[pair]=dict(run_id=build['run_id'],uid=item['uid'],plan_sha256=build['plan_sha256'],
            bundle_root=plan['bundle_root'],output_root=plan['scientific_output'],control_root=plan['control_root'],
            new_run_id='credo-frontier-obs-'+pair.lower()+'-20261016-v4',
            new_roots=[str(Path(plan['control_root']).parent/phase),str(Path(plan['scientific_output']).parent.parent/phase)],
            failure_log_sha256=log['sha256'],original_output_root=original['scientific_output'],
            parent_inventory=retained['fits'][pair]['inventory'])
    if set(rows)!=set(PAIRS):raise ValueError('Four exact failed children required')
    result=remote(INTAKE_CODE,dict(ROWS=rows,EXPECTED_BACKEND='ldragon3',AUTHORITY_SHA=ops.sha(AUTHORITY),FAILURE_SHA=ops.sha(FAILURE)))
    record(INTAKE,result)
    cap=capacity();record(CAPACITY,cap)
    return dict(status='fresh_zero_work_closure_passed',headroom_GiB=cap['headroom_bytes']/GIB,failed_children=4)


def checked_intake():
    authority(); proof=json.loads(read(INTAKE));cap=json.loads(read(CAPACITY))
    if (proof.get('status')!='passed' or proof['authority_sha256']!=ops.sha(AUTHORITY)
            or proof['prior_failure_closure_sha256']!=ops.sha(FAILURE) or set(proof['fits'])!=set(PAIRS)
            or cap['schema']!='credo.frontier.storage-intake.v1' or cap['fileset']!='h_bcb-yding4'
            or cap['study_id']!=ROOT.name or cap['filesystem_available_inodes']<20000):
        raise ValueError('Fresh exact closure or capacity differs')
    for row in proof['fits'].values():
        if (row['new_scientific_updates']!=0 or not all(row[k] is True for k in
                ('failed_job_absent','failed_owned_pods_absent','new_job_absent','new_roots_absent',
                 'failed_scientific_output_absent','parent_inventory_unchanged'))):
            raise ValueError('Zero-work child or unchanged original parent not established')
    return proof,cap


def reserve():
    if BATCH.exists():raise ValueError('Replacement admission already exists; reconcile, never repeat')
    proof,cap=checked_intake();_,manifest=source_manifest(ROOT)
    packages={pair:ops.package(ROOT/('observation_'+pair.lower()+'_v3_launch')) for pair in PAIRS}
    with AdmissionJournal(ROOT/'admission',study_identity_sha256=ops.sha(ROOT/'AUTHORITY.json')) as journal:
        state=journal.snapshot()
        for pair,(_,build,_) in packages.items():
            old=state['tasks'][build['run_id']]
            if (old['status']!='failed' or old['scheduler_id']!=proof['fits'][pair]['uid']
                    or old['evidence_sha256']!=ops.sha(FAILURE)):
                raise ValueError('Exact immutable failed child journal binding differs')
        BATCH.mkdir(mode=0o700)
        record(BATCH/'BATCH_INTENT.json',dict(authority_sha256=ops.sha(AUTHORITY),intake_sha256=ops.sha(INTAKE),
            capacity_sha256=ops.sha(CAPACITY),predecessor_journal_sha256=state['last_sha256'],
            consumed_updates=5120,remaining_updates=74880,automatic_retry=False))
        for pair,(_,build,_) in packages.items():
            run=proof['fits'][pair]['new_run_id']
            state=journal.replace_failed_continuation(build['run_id'],run,
                authority_sha256=ops.sha(AUTHORITY),zero_update_evidence_sha256=ops.sha(FAILURE),
                admission_sha256=build['phase_authority_sha256'],capacity_receipt_sha256=ops.sha(CAPACITY),
                fileset_headroom_bytes=min(cap['headroom_bytes'],cap['filesystem_available_bytes']))
        exports=[]
        for pair,(path,build,_) in packages.items():
            run=proof['fits'][pair]['new_run_id'];task=state['tasks'][run]
            old=json.loads(read(path/'bundle/ADMISSION.json'))
            admission=dict(old,run_id=run,status='reserved',
                storage_reserved_bytes=task['storage_reserved_bytes'],scientific_update_reservation=18720,
                aggregate_reserved_output_bytes=sum(t['storage_reserved_bytes'] for t in state['tasks'].values()),
                live_h100_reservations=sum(t['status'] in LIVE for t in state['tasks'].values()),
                journal_event_sha256=state['last_sha256'],source_manifest_sha256=identity(manifest)['sha256'],
                capacity_receipt_sha256=ops.sha(CAPACITY),scheduler_submitted=False,
                work_binding=task['work_binding'],continues_task_id=task['continues_task_id'],
                continuation_binding=task['continuation_binding'],replaces_continuation_task_id=build['run_id'],
                replacement_authority_sha256=ops.sha(AUTHORITY),zero_update_evidence_sha256=ops.sha(FAILURE),
                replacement_intake_sha256=ops.sha(INTAKE))
            target=BATCH/(pair+'.json');record(target,admission)
            exports.append(dict(pair=pair,path=str(target),sha256=ops.sha(target)))
        result=dict(status='reserved_not_launched',fits=exports,credited_parent_updates=5120,new_updates=74880,
                    live_h100_reservations=4,journal_event_sha256=state['last_sha256'])
        record(BATCH/'BATCH_RESERVED.json',result)
    return result


def build_all():
    from deployment.continuation_replacement_build import construct
    from deployment.build import write_package
    batch=json.loads(read(BATCH/'BATCH_RESERVED.json'))
    for row in batch['fits']:
        a=json.loads(read(row['path'],row['sha256']))
        payload,archive,report=construct(ROOT,row['pair'],a)
        write_package(ROOT/('observation_'+row['pair'].lower()+'_v4_launch'),payload,archive,report)
    return dict(status='built_not_launched',fits=4)


def gate(path,build,plan):
    a=ops.admission_gate(path,build,plan)
    if (a['replacement_authority_sha256']!=ops.sha(AUTHORITY)
            or a['replacement_intake_sha256']!=ops.sha(INTAKE)
            or a['zero_update_evidence_sha256']!=ops.sha(FAILURE)):
        raise ValueError('Exact current replacement approval/evidence differs')
    with AdmissionJournal(ROOT/'admission',study_identity_sha256=ops.sha(ROOT/'AUTHORITY.json')) as journal:
        task=journal.snapshot()['tasks'][build['run_id']]
        for key in ('continues_task_id','continuation_binding','replaces_continuation_task_id',
                    'replacement_authority_sha256','zero_update_evidence_sha256'):
            if task.get(key)!=a.get(key):raise ValueError('Replacement reservation differs: '+key)
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('intake','reserve','build','preflight','stage','closeout','dryrun','create','launch'))
    p.add_argument('--pair',choices=PAIRS);p.add_argument('--session-receipt')
    a=p.parse_args()
    if a.action=='intake':result=intake()
    elif a.action=='reserve':result=reserve()
    elif a.action=='build':result=build_all()
    else:
        if a.pair is None or a.session_receipt is None:p.error('Exact pair and current session receipt required')
        path,build,plan=ops.package(ROOT/('observation_'+a.pair.lower()+'_v4_launch'))
        ops.session_gate(a.session_receipt);admission=gate(path,build,plan)
        if a.action=='preflight':result=ops.preflight(path,build,plan,admission)
        elif a.action=='stage':result=ops.stage(path,build,plan)
        elif a.action=='closeout':result=ops.closeout(path,build)
        else:result=ops.lifecycle(path,build,plan,a.action)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
