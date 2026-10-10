"""One read-only snapshot for the receipt-bound scalar preparation Job."""
from datetime import datetime, timezone
import json

import prepare_operate as ops


def summarize(job, pods, expected_uid, startup=None):
    if (job["metadata"]["uid"] != expected_uid
            or job["metadata"].get("annotations", {}).get("seadragon_run_id") != ops.RUN_ID):
        raise ValueError("Job identity mismatch")
    workers = job["spec"]["template"]["spec"]["containers"]
    if (len(workers) != 1 or any(str(workers[0]["resources"][kind].get("nvidia.com/gpu")) != "1"
                               for kind in ("requests", "limits"))):
        raise ValueError("Admitted GPU request mismatch")
    states = []
    for pod in pods:
        if not any(row.get("uid") == expected_uid and row.get("name") == ops.RUN_ID and row.get("controller")
                   for row in pod["metadata"].get("ownerReferences", [])):
            raise ValueError("Unowned Pod")
        status = pod.get("status", {})
        conditions = {row["type"]: row for row in status.get("conditions", [])}
        ready = status.get("phase") == "Running" and conditions.get("Ready", {}).get("status") == "True"
        scheduled = conditions.get("PodScheduled", {}).get("status")
        hardware = bool(ready and startup and startup.get("pod_uid") == pod["metadata"]["uid"]
                        and startup.get("status") == "startup_guard_passed")
        states.append(dict(name=pod["metadata"]["name"], uid=pod["metadata"]["uid"],
            phase=status.get("phase"), ready=ready, scheduled=scheduled,
            scheduling_reason=conditions.get("PodScheduled", {}).get("reason"),
            scheduling_message=conditions.get("PodScheduled", {}).get("message"),
            restarts=sum(c.get("restartCount", 0) for c in status.get("containerStatuses", [])),
            container_states=[c.get("state", {}) for c in status.get("containerStatuses", [])],
            hardware_verified=hardware))
    return dict(job_uid=expected_uid, suspend=job["spec"].get("suspend", False),
                job_status=job.get("status", {}), pods=states,
                ready_gpu_requests=sum(p["ready"] for p in states),
                hardware_verified_gpus=sum(p["hardware_verified"] for p in states))


def main():
    build, plan = ops.verified_build()
    created = json.loads(ops.read_regular(ops.BASE / "CREATE_RESULT.json"))["result"]
    if created["plan_sha256"] != build["plan_sha256"]:
        raise ValueError("Creation receipt belongs to another plan")
    code = r'''
import json,os,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
def read(path):
 path=Path(path)
 if not os.path.lexists(str(path)):return None
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>16*1024**2:raise RuntimeError('Unqualified receipt')
 return json.loads(path.read_text())
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
def query(args):
 result=subprocess.run(k+args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if result.returncode:raise RuntimeError('Exact scalar Job query failed')
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
for name in ('STARTUP.json','FAILED.json','COMPLETE.json'):
 value=read(Path(OUTPUT)/name)
 if value is not None:outputs[name]={k:v for k,v in value.items() if k!='files'}
path=Path(OUTPUT)/'PROGRESS.jsonl'
last=None
if os.path.lexists(str(path)):
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469:raise RuntimeError('Unqualified progress file')
 with path.open('rb') as f:
  f.seek(max(0,info.st_size-8192));lines=f.read().decode().splitlines()
 if lines:
  try:last=json.loads(lines[-1])
  except ValueError:last={'status':'publication_in_progress'}
print('RECEIPT='+json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),job=job,pods=pods,
 receipts=receipts,outputs=outputs,last_progress=last)))
'''
    value = ops.remote(code, dict(JOB=ops.RUN_ID, UID=created["uid"], CONTROL=ops.CONTROL_ROOT,
                                 OUTPUT=plan["worker_output"]), timeout=150)
    if value["job"] is not None:
        summary = summarize(value.pop("job"), value.pop("pods"), created["uid"], value["outputs"].get("STARTUP.json"))
        value.update(summary)
    else:
        if value["pods"] or value["receipts"].get("cleanup.json", {}).get("status") != "complete":
            raise RuntimeError("Absent Job without completed cleanup; inspect exact receipts")
    name = "STATUS_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    ops.record(name, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
