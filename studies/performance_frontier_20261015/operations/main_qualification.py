"""Separate main-qualification operator using retained serialized transport.

No action on import. Every action requires external approval, a live journal
reservation and the current session preflight; this is not a main-fit launcher.
"""
import argparse
import json
from pathlib import Path
import sys

from transport import ROOT
import observation as retained

sys.path.insert(0, str(ROOT))
from deployment.main_qualification_common import validate_plan, validate_authority, validate_admission


def package(path):
    path = Path(path).absolute()
    if not path.is_relative_to(ROOT) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Exact local main-qualification package required')
    build = json.loads((path/'BUILD.json').read_bytes())
    plan = json.loads((path/'bundle/PLAN.json').read_bytes())
    validate_plan(plan)
    if (build.get('schema') != 'credo.frontier.main-qualification-build.v1'
            or build['plan_sha256'] != retained.sha(path/'bundle/PLAN.json')
            or build['archive_sha256'] != retained.sha(path/'bundle.tar.gz')
            or build['inventory_sha256'] != retained.sha(path/'bundle/INVENTORY.json')
            or build['run_id'] != plan['run_id'] or build['phase'] != plan['phase_directory']):
        raise ValueError('Main-qualification build identity differs')
    inventory = json.loads((path/'bundle/INVENTORY.json').read_bytes())
    found = set()
    for p in (path/'bundle').rglob('*'):
        if p.is_symlink() or not (p.is_dir() or p.is_file()):
            raise ValueError('Unexpected package member type')
        if p.is_file(): found.add(p.relative_to(path/'bundle').as_posix())
    if found != set(inventory) | {'INVENTORY.json'}:
        raise ValueError('Package inventory differs')
    for name, ref in inventory.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe package member')
        p = path/'bundle'/name
        if retained.sha(p) != ref['sha256'] or p.stat().st_size != ref['size_bytes']:
            raise ValueError('Package bytes changed')
    worker = json.loads((path/'bundle/WORKER_CONTRACT.json').read_bytes())
    for ref in [plan['worker_contract'], *[worker[k] for k in ('source_manifest','worker_source_manifest',
            'observation_protocol','normalization_protocol','rules','authority','admission')]]:
        if Path(ref['path']).parent != Path(plan['bundle_root']):
            raise ValueError('Bound metadata escaped the package')
        local = path/'bundle'/Path(ref['path']).name
        if retained.sha(local) != ref['sha256'] or local.stat().st_size != ref['size_bytes']:
            raise ValueError('Bound metadata identity differs')
    authority = json.loads((path/'bundle/MAIN_QUALIFICATION_AUTHORITY.json').read_bytes())
    admission = json.loads((path/'bundle/ADMISSION.json').read_bytes())
    validate_authority(worker, authority)
    validate_admission(worker, admission)
    if (build['source_manifest_sha256'] != worker['source_manifest']['sha256']
            or build['worker_source_manifest_sha256'] != worker['worker_source_manifest']['sha256']
            or build['phase_authority_sha256'] != worker['authority']['sha256']):
        raise ValueError('Build/reservation source authority differs')
    return path, build, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight','stage','closeout','dryrun','create','launch'))
    parser.add_argument('--package', required=True)
    parser.add_argument('--session-receipt', required=True)
    args = parser.parse_args(argv)
    path, build, plan = package(args.package)
    retained.session_gate(args.session_receipt)
    # Exact work binding, zero updates, source/authority hashes and current
    # aggregate reservation are checked under the existing journal lock.
    admission = retained.admission_gate(path, build, plan)
    if args.action == 'preflight': result = retained.preflight(path, build, plan, admission)
    elif args.action == 'stage': result = retained.stage(path, build, plan)
    elif args.action == 'closeout': result = retained.closeout(path, build)
    else: result = retained.lifecycle(path, build, plan, args.action)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
