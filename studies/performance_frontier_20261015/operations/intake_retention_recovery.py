"""Read-only intake of the four exact failed observation attempts.

Hash retained resume/candidate bytes without importing torch or changing parent
files. This is integrity evidence, NOT a numerical resume or relaunch approval.
Caller must verify the approved remote chain once before invoking this command.
"""

import argparse
import datetime
import json
from pathlib import Path

from observation import package
from status_observation import STATUS_CODE
from transport import remote, record


INTAKE_CODE = r'''
import contextlib,datetime,hashlib,io,json,os,stat
from pathlib import Path
def signature(s):
 return tuple(getattr(s,k) for k in ('st_dev','st_ino','st_size','st_uid','st_nlink','st_mtime_ns','st_ctime_ns'))
def artifact(path,parse=False):
 p=Path(path)
 for parent in (p,)+tuple(p.parents):
  if parent.is_symlink():raise RuntimeError('Symlink in exact recovery input')
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode) or before.st_uid!=307469 or before.st_nlink!=1:
  raise RuntimeError('Recovery input not owned regular single-link file')
 if before.st_size>(4*1024**2 if parse else 256*1024**2):raise RuntimeError('Recovery input exceeds bound')
 digest=hashlib.sha256();parts=[]
 with os.fdopen(os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  if signature(os.fstat(f.fileno()))!=signature(before):raise RuntimeError('Changed during open')
  while True:
   block=f.read(1024**2)
   if not block:break
   digest.update(block)
   if parse:parts.append(block)
  if signature(os.fstat(f.fileno()))!=signature(before) or signature(p.lstat())!=signature(before):
   raise RuntimeError('Changed during hash')
 result={'path':str(p),'sha256':digest.hexdigest(),'size_bytes':before.st_size}
 if parse:result['value']=json.loads(b''.join(parts))
 return result
results={}
for b in BINDINGS:
 if b.get('prior_status') is None:
  output=io.StringIO()
  with contextlib.redirect_stdout(output):exec(STATUS_CODE,dict(b))
  lines=[s[8:] for s in output.getvalue().splitlines() if s.startswith('RECEIPT=')]
  if len(lines)!=1:raise RuntimeError('Ambiguous exact status')
  status=json.loads(lines[0])
 else:status=b['prior_status']
 if status['job'] is not None or status['pods']:raise RuntimeError('Parent still has scheduler ownership')
 cleanup=status['records']['cleanup.json']
 failed=status['records']['output/WORKER_FAILED.json']
 terminal=status['records']['terminal.json']
 if (cleanup is None or cleanup['value']['status']!='complete' or terminal['value']['status']!='failed'
     or cleanup['value']['uid']!=b['UID'] or status['uid']!=b['UID']):raise RuntimeError('Parent closure differs')
 root=Path(b['OUTPUT'])
 failure=artifact(root/'WORKER_FAILED.json',True)
 if failure['sha256']!=failed['sha256']:raise RuntimeError('Failure changed')
 f=failure['value']
 if (f['counters']!={'accepted_updates':1280,'attempted_updates':1280,'optimizer_calls':1280}
     or f['exception_message']!='Unlinked payload inode readback failed'):raise RuntimeError('Not this recovery scope')
 ident=artifact(root/'IDENTITY.json',True)
 if ident['value']!=f['identity']:raise RuntimeError('Parent identity differs')
 latest=root/'checkpoints/update-000001280'
 complete=artifact(latest/'COMPLETE.json',True);c=complete['value']
 if (c['identity']!=f['identity'] or c['accepted_updates']!=1280 or c['attempted_updates']!=1280
     or c['execution_scope']!='scientific' or c['phase']!='observation_fit'):raise RuntimeError('Checkpoint receipt differs')
 payload=artifact(latest/'checkpoint.pt')
 if payload['sha256']!=c['checkpoint_sha256'] or payload['size_bytes']!=c['checkpoint_bytes']:
  raise RuntimeError('Checkpoint payload integrity failure')
 prepared=artifact(root/'retention/transactions/after-update-000001280/PREPARED.json',True)
 candidates=prepared['value']['candidates']
 if len(candidates)!=1 or candidates[0]['relative_uri']!='checkpoints/update-000000768/checkpoint.pt':
  raise RuntimeError('Retirement differs from this recovery scope')
 if (os.path.lexists(str(root/candidates[0]['relative_uri'])) or
     os.path.lexists(str(root/'retention/transactions/after-update-000001280/DELETED.json'))):
  raise RuntimeError('Retirement state changed')
 exports=[artifact(root/('observation-candidate-%09d.pt'%n)) for n in (256,512,1024)]
 progress=artifact(root/'progress/update-000001024.json',True)
 selected=progress['value']['selected_candidate']['artifact']
 matches=[x for x in exports if x['path']==str(root/selected['relative_uri'])]
 if len(matches)!=1 or any(matches[0][k]!=selected[k] for k in ('sha256','size_bytes')):
  raise RuntimeError('Selected candidate integrity failure')
 results[b['PAIR']]={'status':status,'failure':failure,'identity':ident,'checkpoint_completion':complete,
  'checkpoint_payload':payload,'candidate_payloads':exports,'progress_1024':progress,
  'incomplete_retirement':prepared,'retired_768_payload_absent':True,'deletion_publication_absent':True,
  'consumed_updates':1280,'remaining_update_cap':18720,'checkpoint_hash_verified':True,
  'tensor_deserialization_performed':False,'numerical_resume_verified':False}
print('RECEIPT='+json.dumps({'schema':'credo.frontier.retention-recovery-byte-intake.v1',
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fits':results,
 'remote_writes':False,'scientific_relaunch_authorized':False,'total_consumed_updates':5120,
 'total_remaining_updates_if_resume_validates':74880,'scientific_optimizer_calls':0}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--d2d3-status', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('Intake receipt already exists; do not overwrite')
    prior = json.loads(args.d2d3_status.read_bytes())
    age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(prior['at_utc'])
    # Completed failure/cleanup evidence is reusable; this does not certify a
    # launch-time live capacity check or absence of unrelated successors.
    if age.total_seconds() < 0:
        raise ValueError('Future-dated status')
    bindings = []
    for pair in ('D1D2', 'D1D3', 'D1D4', 'D2D3'):
        path, build, plan = package(Path('observation_' + pair.lower() + '_v2_launch'))
        bound = json.loads((path/'CREATE.json').read_bytes())['records']['live-binding.json']
        if bound['job'] != build['run_id'] or bound['plan_sha256'] != build['plan_sha256']:
            raise ValueError('Local exact binding differs')
        bindings.append(dict(PAIR=pair, BUNDLE=plan['bundle_root'], OUTPUT=plan['scientific_output'],
                             PLAN_SHA=build['plan_sha256'], JOB=build['run_id'], UID=bound['uid'],
                             prior_status=prior if pair == 'D2D3' else None))
    result = remote(INTAKE_CODE, dict(BINDINGS=bindings, STATUS_CODE=STATUS_CODE))
    record(args.receipt, result)
    print(json.dumps({'receipt':str(args.receipt), 'at_utc':result['at_utc'],
                      'fits':{k:{'checkpoint_hash_verified':v['checkpoint_hash_verified'],
                                 'consumed_updates':v['consumed_updates'],
                                 'remaining_update_cap':v['remaining_update_cap']} for k,v in result['fits'].items()},
                      'remote_writes':False, 'numerical_resume_verified':False}))


if __name__ == '__main__':
    main()
