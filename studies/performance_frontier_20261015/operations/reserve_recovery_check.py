"""Reserve the approved zero-update check; never replace a scientific budget."""
import argparse
import json
from pathlib import Path
import sys
from transport import ROOT,record
from observation import sha
sys.path.insert(0,str(ROOT))
from deployment.common import identity,read
from deployment.recovery_check_common import RUN,PHASE,same,validate_authority
from deployment.recovery_check_build import source_payload
from frontier.coordination import AdmissionJournal,WorkBinding,LIVE


def reserve(authority_path,capacity_path,output):
    output=Path(output).absolute()
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('New study receipt required')
    authority_raw=read(Path(authority_path).absolute())
    authority=validate_authority(json.loads(authority_raw))
    same(authority,dict(run_id=RUN,phase=PHASE))
    capacity=json.loads(read(Path(capacity_path).absolute()))
    if (capacity.get('schema')!='credo.frontier.storage-intake.v1' or capacity.get('study_id')!=ROOT.name
            or capacity.get('fileset')!='h_bcb-yding4' or capacity.get('remote_writes') is not False
            or capacity.get('numerical_payloads_opened') is not False
            or capacity.get('filesystem_available_inodes',0)<20000):
        raise ValueError('Current exact-fileset capacity required')
    _,source=source_payload(ROOT)
    phase_sha=identity(authority_raw)['sha256']
    work=WorkBinding.other('qualification',phase_sha)
    with AdmissionJournal(ROOT/'admission',study_identity_sha256=sha(ROOT/'AUTHORITY.json')) as journal:
        state=journal.reserve(RUN,worker_id=RUN,kind='qualification',scientific_updates=0,
            output_bytes=4*1024**3,admission_sha256=phase_sha,capacity_receipt_sha256=sha(capacity_path),
            fileset_headroom_bytes=min(capacity['headroom_bytes'],capacity['filesystem_available_bytes']),work_binding=work)
        result=dict(schema='credo.frontier.retention-check-admission.v1',status='reserved',run_id=RUN,
            work_binding=work.to_dict(),phase_authority_sha256=phase_sha,
            source_manifest_sha256=identity(source)['sha256'],scientific_update_reservation=0,
            storage_reserved_bytes=4*1024**3,disposable_optimizer_call_limit=0,
            automatic_retry=False,automatic_successor=False,journal_event_sha256=state['last_sha256'],
            journal_identity_sha256=sha(ROOT/'admission/IDENTITY.json'),capacity_receipt_sha256=sha(capacity_path),
            aggregate_reserved_output_bytes=sum(t['storage_reserved_bytes'] for t in state['tasks'].values()),
            live_h100_reservations=sum(t['status'] in LIVE for t in state['tasks'].values()),scheduler_submitted=False)
        record(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('authority','capacity','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();print(json.dumps(reserve(args.authority,args.capacity,args.output),sort_keys=True))


if __name__=='__main__':main()
