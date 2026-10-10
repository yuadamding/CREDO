"""One exact D2 forecast snapshot; no scheduler mutation, scans or polling."""
from datetime import datetime, timezone
import json

import d2_forecast_operate as operation
import prepare_operate as ops
from prepare_status import summarize


REMOTE = r'''
import json,os,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
def read(path):
 path=Path(path)
 if not os.path.lexists(str(path)):return None
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>16*1024**2:raise RuntimeError('Unqualified exact receipt')
 return json.loads(path.read_text())
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
def query(args):
 result=subprocess.run(k+args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if result.returncode:raise RuntimeError('Exact Job query failed')
 return json.loads(result.stdout) if result.stdout.strip() else None
job=query(['get','job',JOB,'--ignore-not-found=true','-o','json'])
pods=query(['get','pods','-l','job-name='+JOB,'-o','json'])['items']
if job is not None and (job['metadata']['uid']!=UID or job['metadata'].get('annotations',{}).get('seadragon_run_id')!=JOB):raise RuntimeError('Job identity mismatch')
if not all(any(o.get('uid')==UID and o.get('controller') and o.get('name')==JOB for o in p['metadata'].get('ownerReferences',[])) for p in pods):raise RuntimeError('Unowned Pod')
receipts={}
for name in ('launch-success.json','startup-validated.json','supervisor-attention.json','terminal.json','cleanup.json'):
 value=read(Path(CONTROL)/'gpu'/name)
 if value is not None:receipts[name]=value
outputs={}
for name in ('STARTUP.json','FAILURE.json','COMPLETE.json','PREDICTION_SEAL.json'):
 value=read(Path(OUTPUT)/name)
 if value is not None:outputs[name]={k:v for k,v in value.items() if k!='files'}
path=Path(OUTPUT)/'PROGRESS.jsonl'
last=None
if os.path.lexists(str(path)):
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469:raise RuntimeError('Unqualified progress')
 with path.open('rb') as f:
  f.seek(max(0,info.st_size-16384));lines=f.read().decode().splitlines()
 if lines:
  try:last=json.loads(lines[-1])
  except ValueError:last={'status':'publication_in_progress'}
print('RECEIPT='+json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),job=job,pods=pods,receipts=receipts,outputs=outputs,last_progress=last)))
'''


def main():
    operation.bind_context()
    build, plan = ops.verified_build()
    created = json.loads(ops.read_regular(ops.BASE / "CREATE_RESULT.json"))["result"]
    if created["plan_sha256"] != build["plan_sha256"]:
        raise ValueError("Exact creation and plan identity required")
    value = ops.remote(REMOTE, dict(JOB=ops.RUN_ID, UID=created["uid"], CONTROL=ops.CONTROL_ROOT,
                                   OUTPUT=plan["worker_output"]), timeout=150)
    if value["job"] is not None:
        value.update(summarize(value.pop("job"), value.pop("pods"), created["uid"], value["outputs"].get("STARTUP.json")))
    elif value["pods"] or value["receipts"].get("cleanup.json", {}).get("status") != "complete":
        raise RuntimeError("Absent Job without completed cleanup; reconcile exact receipts")
    name = "STATUS_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    ops.record(name, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
