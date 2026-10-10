"""Stdlib-only exact new-study scope around the unchanged publisher."""
import hashlib
import json
from pathlib import Path
import deployment

EXPECTED = '1c88af69851bb00028488d714bb4fdef014cf3f3cfd20c081cd470c9d7df0ae2'
ROOTS = ('/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/fitting_gpu_native_v6', '/rsrch8/home/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/fitting_gpu_native_v6')

def validated_roots(plan):
    if (plan.get("control_root") != ROOTS[0]
            or plan.get("bundle_root") != ROOTS[1]+"/bundle"
            or plan.get("worker_output") != ROOTS[1]+"/output"
            or plan.get("run_id") != 'credo-full-ddr-fitting-gpu-native-20261009-v6'):
        raise ValueError("New study preparation roots/identity differ")
    return tuple(Path(value) for value in ROOTS)

def publish_archive(raw, build, **kwargs):
    path = Path(__file__).with_name("deployment.py")
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Reused publication helper changed")
    deployment.validated_roots = validated_roots
    return deployment.publish_archive(raw, build, **kwargs)
