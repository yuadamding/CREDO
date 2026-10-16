"""Reserve one externally authorized disposable main qualification; never retry.

Existing Pending/reserved workers consume the same four-H100 admission cap.
This command is not run merely because the implementation exists.
"""
import argparse
import json
from pathlib import Path
import sys

from transport import ROOT, record
from observation import sha

sys.path.insert(0, str(ROOT))
from deployment.common import encoded, identity, read
from deployment.main_qualification_common import validate_authority
from frontier.coordination import AdmissionJournal, WorkBinding, LIVE
from frontier.main_qualification_contract import settings
from frontier.main_qualification_plan import plan_identity
from frontier.main_worker_source import source_closure, manifest_payload, QUALIFICATION_ENTRY


def reserve(authority_path, capacity_path, output):
    authority_path, capacity_path, output = map(lambda p:Path(p).absolute(), (authority_path,capacity_path,output))
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError('New study-owned admission export required; never reserve twice')
    authority_raw = read(authority_path)
    authority = json.loads(authority_raw)
    capacity = json.loads(read(capacity_path))
    if (capacity.get('schema') != 'credo.frontier.storage-intake.v1'
            or capacity.get('study_id') != ROOT.name or capacity.get('fileset') != 'h_bcb-yding4'
            or capacity.get('remote_writes') is not False or capacity.get('numerical_payloads_opened') is not False
            or capacity.get('filesystem_available_inodes',0) < 20000):
        raise ValueError('Current exact-fileset metadata capacity required')
    source = manifest_payload(source_closure(ROOT,entry=QUALIFICATION_ENTRY),entry=QUALIFICATION_ENTRY)
    child = manifest_payload(source_closure(ROOT))
    worker = dict(run_id=authority.get('run_id'), plan_sha256=plan_identity(), settings=settings(),
                  source_manifest=identity(source), worker_source_manifest=identity(child))
    validate_authority(worker,authority)
    phase_sha = identity(authority_raw)['sha256']
    work = WorkBinding.other('qualification',plan_identity())
    with AdmissionJournal(ROOT/'admission',study_identity_sha256=sha(ROOT/'AUTHORITY.json')) as journal:
        state = journal.reserve(worker['run_id'],worker_id=worker['run_id'],kind='qualification',scientific_updates=0,
            output_bytes=16*1024**3,admission_sha256=phase_sha,capacity_receipt_sha256=sha(capacity_path),
            fileset_headroom_bytes=min(capacity['headroom_bytes'],capacity['filesystem_available_bytes']),work_binding=work)
        result = dict(schema='credo.frontier.main-qualification-admission.v1',status='reserved',run_id=worker['run_id'],
            work_binding=work.to_dict(),phase_authority_sha256=phase_sha,
            source_manifest_sha256=identity(source)['sha256'],worker_source_manifest_sha256=identity(child)['sha256'],
            scientific_update_reservation=0,storage_reserved_bytes=16*1024**3,disposable_optimizer_call_limit=112,
            automatic_retry=False,automatic_successor=False,journal_event_sha256=state['last_sha256'],
            journal_identity_sha256=sha(ROOT/'admission/IDENTITY.json'),capacity_receipt_sha256=sha(capacity_path),
            aggregate_reserved_output_bytes=sum(t['storage_reserved_bytes'] for t in state['tasks'].values()),
            live_h100_reservations=sum(t['status'] in LIVE for t in state['tasks'].values()),scheduler_submitted=False)
        record(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('authority','capacity-receipt','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(reserve(args.authority,args.capacity_receipt,args.output),sort_keys=True))


if __name__=='__main__':
    main()
