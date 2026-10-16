"""One read-only exact-UID recovery-check snapshot; no automatic polling."""
import argparse
import json
from pathlib import Path
from recovery_check import package
from status_observation import STATUS_CODE
from transport import remote,record

# Same exact-job identity and process checks; only the bounded receipt inventory
# and descriptive schema change. No new scheduler permissions or broad reads.
CHECK_STATUS_CODE=STATUS_CODE.replace(
    "['WORKER_STARTED.json','WORKER_FAILED.json','PAIRED_FP32_CHECK.json','QUALIFICATION_COMPLETE.json','COMPLETE.json']",
    "['CHECK_FAILED.json','COMPLETE.json','storage_canary/COMPLETE.json','D1D2/RESTORE_VERIFIED.json',"
    "'D1D3/RESTORE_VERIFIED.json','D1D4/RESTORE_VERIFIED.json','D2D3/RESTORE_VERIFIED.json']"
).replace('credo.frontier.exact-observation-status.v1','credo.frontier.exact-retention-check-status.v1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True);parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    if Path(args.receipt).exists():raise ValueError('New immutable receipt required')
    path,build,plan=package(args.package)
    binding=json.loads((path/'CREATE.json').read_bytes())['records']['live-binding.json']
    if binding['job']!=build['run_id'] or binding['plan_sha256']!=build['plan_sha256']:
        raise ValueError('Local exact Job binding differs')
    value=remote(CHECK_STATUS_CODE,dict(BUNDLE=plan['bundle_root'],PLAN_SHA=build['plan_sha256'],
        JOB=build['run_id'],UID=binding['uid']))
    record(args.receipt,value)
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
