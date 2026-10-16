"""Read-only exact failed-parent closure and retained-byte intake; no tensor load."""
from pathlib import Path
import json
import sys
from transport import ROOT, remote, record, capacity
from observation import package, sha

CODE = r'''
import os,stat,json,hashlib,datetime,subprocess
from pathlib import Path
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
def query(args):
 r=subprocess.run(k+args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if r.returncode:raise RuntimeError(r.stderr.decode()[-2000:])
 return json.loads(r.stdout) if r.stdout.strip() else None
def metadata(path):
 p=Path(path); s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_uid!=307469 or s.st_size>32*1024**2:raise RuntimeError('Unsafe metadata')
 raw=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw),value=json.loads(raw))
def inventory(root):
 out={}
 for directory,folders,files in os.walk(str(root),followlinks=False):
  for name in sorted(folders+files):
   p=Path(directory)/name;s=p.lstat()
   if s.st_uid!=307469 or not(stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):raise RuntimeError('Unsafe parent output')
   if stat.S_ISREG(s.st_mode):
    if s.st_nlink!=1:raise RuntimeError('Linked output')
    out[p.relative_to(root).as_posix()]={key:getattr(s,key) for key in ('st_dev','st_ino','st_size','st_uid','st_mode','st_nlink','st_mtime_ns','st_ctime_ns')}
   if len(out)>20000:raise RuntimeError('Parent inventory bound')
 return out
result={}
for pair,row in ROWS.items():
 for name in (row['run_id'],row['new_run_id']):
  if query(['get','job/'+name,'--ignore-not-found=true','-o','json']) is not None or query(['get','pods','-l','job-name='+name,'-o','json'])['items']:
   raise RuntimeError('Exact predecessor or continuation job exists')
 for p in row['new_roots']:
  if os.path.lexists(p):raise RuntimeError('Continuation root exists')
 terminal=metadata(row['control_root']+'/gpu/terminal.json')
 cleanup=metadata(row['control_root']+'/gpu/cleanup.json')
 if terminal['value']['uid']!=row['uid'] or terminal['value']['status']!='failed' or cleanup['value']['uid']!=row['uid'] or cleanup['value']['status']!='complete':raise RuntimeError('Parent closure differs')
 root=Path(row['output_root'])
 initial=inventory(root); final=inventory(root)
 if initial!=final:raise RuntimeError('Parent changed during byte intake')
 total=sum(s['st_size'] for s in initial.values())
 if not 0<total<2*1024**3:raise RuntimeError('Parent exceeds conservative continuation allowance')
 raw=(json.dumps(initial,sort_keys=True,separators=(',',':'))+'\n').encode()
 result[pair]=dict(parent_output_bytes=total,inventory=initial,inventory_sha256=hashlib.sha256(raw).hexdigest(),
  parent_input_lease=metadata(root/'INPUT_LEASE.json'),terminal=terminal,cleanup=cleanup,
  old_job_absent=True,new_job_absent=True,owned_pods_absent=True,new_roots_absent=True)
print('RECEIPT='+json.dumps(dict(schema='credo.frontier.observation-continuation-intake.v1',status='passed',
 at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authority_sha256=AUTHORITY_SHA,
 fits=result,remote_writes=False,numerical_payloads_opened=False)))
'''


def main():
    rows={}
    for pair in ('D1D2','D1D3','D1D4','D2D3'):
        path,build,plan=package(ROOT/('observation_'+pair.lower()+'_v2_launch'))
        binding=json.loads((path/'CREATE.json').read_bytes())['records']['live-binding.json']
        rows[pair]=dict(run_id=build['run_id'],uid=binding['uid'],control_root=plan['control_root'],
            output_root=plan['scientific_output'],new_run_id='credo-frontier-obs-'+pair.lower()+'-20261016-v3',
            new_roots=[str(Path(plan[k]).parent/('observation_'+pair.lower()+'_v3'))
                       for k in ('control_root',)] +
                      [str(Path(plan['scientific_output']).parent.parent/('observation_'+pair.lower()+'_v3'))])
    result=remote(CODE,dict(ROWS=rows,AUTHORITY_SHA=sha(ROOT/'OBSERVATION_CONTINUATION_V3_AUTHORITY.json')))
    record(ROOT/'receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json',result)
    cap=capacity();record(ROOT/'receipts/OBSERVATION_CONTINUATION_V3_CAPACITY_20261016.json',cap)
    print(json.dumps(dict(parent_bytes={p:r['parent_output_bytes'] for p,r in result['fits'].items()},
                          headroom_GiB=cap['headroom_bytes']/1024**3)))


if __name__=='__main__':main()
