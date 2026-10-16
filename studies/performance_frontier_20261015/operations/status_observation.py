"""One read-only exact-UID observation snapshot; no polling or scheduler writes."""

import argparse
import json
from pathlib import Path

from observation import package
from transport import remote, record


STATUS_CODE = r'''
import datetime,hashlib,json,os,socket,stat,subprocess
from pathlib import Path
def read(path, maximum=4*1024**2):
 p=Path(path)
 if not os.path.lexists(str(p)):return None
 info=p.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>maximum:
  raise RuntimeError('Exact status receipt type/owner/size differs: '+str(p))
 raw=p.read_bytes()
 return {'sha256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
plan_record=read(BUNDLE+'/PLAN.json')
if plan_record['sha256']!=PLAN_SHA:raise RuntimeError('Remote plan changed')
plan=plan_record['value']
root=Path(plan['control_root'])/'gpu'
binding=read(root/'live-binding.json')
if binding is None or binding['value']['uid']!=UID or binding['value']['job']!=JOB:
 raise RuntimeError('Exact Job binding absent or changed')
config='/rsrch8/home/bcb/yding4/.kube/config'
info=os.lstat(config)
if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or stat.S_IMODE(info.st_mode)!=0o600:
 raise RuntimeError('Kubeconfig metadata differs')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
def query(arguments):
 result=subprocess.run(k+arguments,env=dict(os.environ,KUBECONFIG=config),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if result.returncode:raise RuntimeError(result.stderr.decode()[-2000:])
 return json.loads(result.stdout) if result.stdout.strip() else None
job=query(['get','job/'+JOB,'--ignore-not-found=true','-o','json'])
if job is not None and job['metadata']['uid']!=UID:raise RuntimeError('Job name now owned by another UID')
pods=query(['get','pods','-l','job-name='+JOB,'-o','json'])['items']
for p in pods:
 if not any(o.get('controller') and o.get('uid')==UID for o in p['metadata'].get('ownerReferences',[])):
  raise RuntimeError('Exact Pod selector contains a different owner')
records={}
for name in ['launch-success.json','startup-validated.json','startup-failure.json','terminal.json','cleanup.json','supervisor-failed.json','output-readback.json']:
 records[name]=read(root/name,32*1024**2)
for name in ['STARTUP.json','ENTRY_FAILED.json','WORKER_EXIT.json']:
 records['handshake/'+name]=read(Path(plan['worker_output'])/name)
for name in ['WORKER_STARTED.json','WORKER_FAILED.json','PAIRED_FP32_CHECK.json','QUALIFICATION_COMPLETE.json','COMPLETE.json']:
 records['output/'+name]=read(Path(plan['scientific_output'])/name)
supervisor={'observed_backend':socket.gethostname(),'status':'not_observed_on_this_backend'}
launch=records['launch-success.json']
if launch is not None:
 expected=launch['value']['process']
 proc=Path('/proc')/str(expected['pid'])
 try:
  info=proc.stat()
  parts=(proc/'stat').read_text().rsplit(')',1)[1].split()
  current=dict(pid=expected['pid'],pgrp=int(parts[2]),session=int(parts[3]),tty=int(parts[4]),start_ticks=int(parts[19]))
  # Do not read another/reused process's command line. An alias can land on a
  # different approved backend, so local absence alone does not mean dead.
  if info.st_uid==307469 and all(current[k]==expected[k] for k in current):
   current['cmdline_sha256']=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()
   supervisor.update(status='matching_live_process' if current==expected and parts[0]!='Z' else 'bound_process_not_live',identity=current)
 except (FileNotFoundError,ProcessLookupError):
  pass
snapshot=dict(schema='credo.frontier.exact-observation-status.v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 run_id=JOB,uid=UID,plan_sha256=PLAN_SHA,remote_writes=False,
 job=None if job is None else dict(uid=job['metadata']['uid'],suspend=job['spec'].get('suspend'),status=job.get('status',{})),
 pods=[dict(name=p['metadata']['name'],uid=p['metadata']['uid'],node=p['spec'].get('nodeName'),status=p.get('status',{})) for p in pods],
 records=records,supervisor=supervisor)
print('RECEIPT='+json.dumps(snapshot))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("Use a new dated status receipt; never overwrite history")
    path, build, plan = package(args.package)
    created = json.loads((path / "CREATE.json").read_bytes())
    binding = created["records"]["live-binding.json"]
    if binding["job"] != build["run_id"] or binding["plan_sha256"] != build["plan_sha256"]:
        raise ValueError("Local creation receipt differs")
    result = remote(STATUS_CODE, dict(BUNDLE=plan["bundle_root"], PLAN_SHA=build["plan_sha256"],
                                    JOB=build["run_id"], UID=binding["uid"]))
    record(args.receipt, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
