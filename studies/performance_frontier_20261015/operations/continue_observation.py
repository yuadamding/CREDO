"""Explicit remaining-work admission/build/launch; no automatic retry or successor."""
import argparse
import json
from pathlib import Path
import sys
from transport import ROOT,record
import observation as ops
sys.path.insert(0,str(ROOT))
from deployment.common import GIB,read,encoded,identity
from deployment.continuation_build import PAIRS,construct,source_manifest
from deployment.build import write_package
from frontier.coordination import AdmissionJournal,LIVE

BATCH=ROOT/'observation_continuation_v3_admission'
AUTHORITY=ROOT/'OBSERVATION_CONTINUATION_V3_AUTHORITY.json'
INTAKE=ROOT/'receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json'
CAPACITY=ROOT/'receipts/OBSERVATION_CONTINUATION_V3_CAPACITY_20261016.json'
CHECK=ROOT/'receipts/RETENTION_CHECK_V2_STATUS_20261016T161210Z.json'


def reserve():
    if BATCH.exists():raise ValueError('Batch already has admission state; reconcile, never repeat')
    authority=json.loads(read(AUTHORITY));intake=json.loads(read(INTAKE));cap=json.loads(read(CAPACITY))
    check=json.loads(read(CHECK));r=check['records'];complete=r['output/COMPLETE.json']
    if (authority.get('status')!='approved' or authority.get('user_instruction')!='proceed with all left'
            or authority.get('maximum_new_updates_all_four')!=74880
            or intake.get('status')!='passed' or intake['authority_sha256']!=ops.sha(AUTHORITY)
            or set(intake['fits'])!=set(PAIRS) or check['job'] is not None or check['pods']
            or r['terminal.json']['value']['status']!='passed' or r['cleanup.json']['value']['status']!='complete'
            or r['output-readback.json']['value']['complete_sha256']!=complete['sha256']
            or complete['sha256']!=authority['recovery_check_complete_sha256']):
        raise ValueError('Explicit authority, parent intake and successful recovery closure required')
    if (cap.get('schema')!='credo.frontier.storage-intake.v1' or cap['fileset']!='h_bcb-yding4'
            or cap['study_id']!=ROOT.name or cap['remote_writes'] is not False
            or cap['numerical_payloads_opened'] is not False or cap['filesystem_available_inodes']<20000):
        raise ValueError('Fresh exact fileset capacity required')
    _,manifest=source_manifest(ROOT)
    parent_intake=json.loads(read(ROOT/'receipts/OBSERVATION_RETENTION_RECOVERY_BYTE_INTAKE_20261016_V1.json'))
    with AdmissionJournal(ROOT/'admission',study_identity_sha256=ops.sha(ROOT/'AUTHORITY.json')) as journal:
        state=journal.snapshot();old_check=state['tasks'][check['run_id']]
        if old_check['status'] in LIVE:
            state=journal.reconcile(check['run_id'],status='complete',exact_scheduler_id=check['uid'],evidence_sha256=ops.sha(CHECK))
        elif old_check['status']!='complete':raise ValueError('Recovery-check reservation is not successful')
        rows=[]
        for pair in PAIRS:
            prior,build,plan=ops.package(ROOT/('observation_'+pair.lower()+'_v2_launch'))
            old=state['tasks'][build['run_id']];proof=intake['fits'][pair]
            if (old['status']!='failed' or old['scheduler_id']!=proof['terminal']['value']['uid']
                    or not all(proof[k] for k in ('old_job_absent','new_job_absent','owned_pods_absent','new_roots_absent'))):
                raise ValueError('Failed parent identity/closure differs')
            rows.append((pair,prior,build,proof,old['evidence_sha256']))
        BATCH.mkdir(mode=0o700)
        record(BATCH/'BATCH_INTENT.json',dict(authority_sha256=ops.sha(AUTHORITY),intake_sha256=ops.sha(INTAKE),
            source_manifest_sha256=identity(manifest)['sha256'],predecessor_journal_sha256=state['last_sha256'],
            consumed_updates=5120,maximum_new_updates=74880,automatic_retry=False))
        for pair,prior,build,proof,failure_sha in rows:
            run='credo-frontier-obs-'+pair.lower()+'-20261016-v3'
            state=journal.continue_failed_representation(build['run_id'],run,accepted_updates=1280,
                authority_sha256=ops.sha(AUTHORITY),failure_evidence_sha256=failure_sha,
                checkpoint_sha256=parent_intake['fits'][pair]['checkpoint_payload']['sha256'],
                recovery_completion_sha256=complete['sha256'],retained_parent_bytes=proof['parent_output_bytes'],
                retained_inventory_sha256=proof['inventory_sha256'],admission_sha256=build['phase_authority_sha256'],
                capacity_receipt_sha256=ops.sha(CAPACITY),fileset_headroom_bytes=min(cap['headroom_bytes'],cap['filesystem_available_bytes']),
                output_bytes=4*GIB-proof['parent_output_bytes'])
        exports=[]
        for pair,prior,build,proof,_ in rows:
            run='credo-frontier-obs-'+pair.lower()+'-20261016-v3';task=state['tasks'][run]
            old_admission=json.loads(read(prior/'bundle/ADMISSION.json'))
            a=dict(schema='credo.frontier.observation-admission-export.v1',run_id=run,status='reserved',
                storage_reserved_bytes=task['storage_reserved_bytes'],scientific_update_reservation=18720,
                aggregate_reserved_output_bytes=sum(t['storage_reserved_bytes'] for t in state['tasks'].values()),
                live_h100_reservations=sum(t['status'] in LIVE for t in state['tasks'].values()),
                journal_event_sha256=state['last_sha256'],journal_identity_sha256=ops.sha(ROOT/'admission/IDENTITY.json'),
                phase_authority_sha256=build['phase_authority_sha256'],source_manifest_sha256=identity(manifest)['sha256'],
                continuation_authority_sha256=ops.sha(AUTHORITY),continues_task_id=build['run_id'],
                continuation_binding=task['continuation_binding'],capacity_receipt_sha256=ops.sha(CAPACITY),
                work_binding=task['work_binding'],scheduler_submitted=False,
                **{k:old_admission[k] for k in ('qualification_sha256','qualified_hardware','qualified_peak_reserved_bytes')})
            file=BATCH/(pair+'.json');record(file,a);exports.append(dict(pair=pair,path=str(file),sha256=ops.sha(file)))
        result=dict(status='reserved_not_launched',fits=exports,new_updates=74880,consumed_updates=5120,
                    live_h100_reservations=4,journal_event_sha256=state['last_sha256'])
        record(BATCH/'BATCH_RESERVED.json',result)
    return result


def build_all():
    batch=json.loads(read(BATCH/'BATCH_RESERVED.json'))
    for row in batch['fits']:
        admission=json.loads(read(row['path'],row['sha256']))
        payload,archive,report=construct(ROOT,row['pair'],admission)
        write_package(ROOT/('observation_'+row['pair'].lower()+'_v3_launch'),payload,archive,report)
    return dict(status='built_not_launched',fits=4)


def gate(path,build,plan):
    a=ops.admission_gate(path,build,plan)
    if a.get('continuation_authority_sha256')!=ops.sha(AUTHORITY):raise ValueError('Continuation authority differs')
    with AdmissionJournal(ROOT/'admission',study_identity_sha256=ops.sha(ROOT/'AUTHORITY.json')) as journal:
        task=journal.snapshot()['tasks'][build['run_id']]
        if (task.get('continues_task_id')!=a['continues_task_id']
                or task.get('continuation_binding')!=a['continuation_binding']):
            raise ValueError('Exact consumed-work continuation reservation missing')
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('reserve','build','preflight','stage','closeout','dryrun','create','launch'))
    p.add_argument('--pair',choices=PAIRS)
    p.add_argument('--session-receipt')
    a=p.parse_args()
    if a.action=='reserve':result=reserve()
    elif a.action=='build':result=build_all()
    else:
        if a.pair is None or a.session_receipt is None:p.error('Exact pair and current session receipt required')
        path,build,plan=ops.package(ROOT/('observation_'+a.pair.lower()+'_v3_launch'))
        ops.session_gate(a.session_receipt);admission=gate(path,build,plan)
        if a.action=='preflight':result=ops.preflight(path,build,plan,admission)
        elif a.action=='stage':result=ops.stage(path,build,plan)
        elif a.action=='closeout':result=ops.closeout(path,build)
        else:result=ops.lifecycle(path,build,plan,a.action)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
