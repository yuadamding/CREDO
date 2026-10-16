"""Resource checks for one already admitted leased training trajectory.

No scheduler/quota reservation, deadline, retry, cleanup or numerical adaptation
is performed. Filesystem free bytes are NOT fileset-quota evidence. Host RSS and
CUDA allocator measurements are process observations, not proof that a future
operation cannot fail. Snapshot allowance is an explicit conservative planning
bound, not a measured peak. Parent worker admission must enforce aggregate
reservations, the actual mount and its quota, cgroup resources and qualification.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import resource
import stat

import torch

from .checkpoints import TrajectoryLease
from .retention import STUDY_ID, identity_sha256, _path


MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class FittingResourcePolicy:
    execution_scope: str
    output_limit_bytes: int
    checkpoint_limit_bytes: int
    snapshot_reserve_bytes: int
    metadata_reserve_bytes: int
    maximum_rss_bytes: int
    minimum_gpu_headroom_bytes: int

    def __post_init__(self):
        for name in ("output_limit_bytes", "checkpoint_limit_bytes", "snapshot_reserve_bytes",
                     "metadata_reserve_bytes", "maximum_rss_bytes"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError("Explicit positive integer fitting resource allowances required")
        if (self.metadata_reserve_bytes<8*MIB
                or self.checkpoint_limit_bytes+self.metadata_reserve_bytes>self.output_limit_bytes
                or 2*self.checkpoint_limit_bytes+self.metadata_reserve_bytes>self.snapshot_reserve_bytes
                or self.snapshot_reserve_bytes>self.maximum_rss_bytes):
            raise ValueError("Output and conservative snapshot allowances do not fit")
        if type(self.minimum_gpu_headroom_bytes) is not int:
            raise ValueError("Exact GPU headroom byte count required")
        if self.execution_scope=="synthetic_cpu_only":
            if (self.output_limit_bytes>GIB or self.maximum_rss_bytes>8*GIB
                    or self.checkpoint_limit_bytes>64*MIB or self.minimum_gpu_headroom_bytes!=0):
                raise ValueError("Synthetic fitting resource scope exceeded")
        elif self.execution_scope=="scientific":
            if (self.output_limit_bytes>384*GIB or self.maximum_rss_bytes>200*GIB
                    or self.minimum_gpu_headroom_bytes!=8*GIB):
                raise ValueError("Scientific worker resource ceiling/headroom changed")
        elif self.execution_scope=="synthetic_cuda_qualification":
            if (self.output_limit_bytes>16*GIB or self.maximum_rss_bytes>200*GIB
                    or self.minimum_gpu_headroom_bytes!=8*GIB):
                raise ValueError("Disposable CUDA qualification resource ceiling/headroom changed")
        else:
            raise ValueError("Explicit scientific or fabricated CPU resource scope required")

    @property
    def identity(self):
        return identity_sha256(dict(schema="credo.frontier.fitting-resource-policy.v1",**asdict(self)))


def process_resources(device):
    """Own process only; no unrelated jobs, volumes or allocations inspected."""
    dev=torch.device(device)
    result=dict(maximum_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                threads=torch.get_num_threads())
    if dev.type=="cuda":
        if dev!=torch.device("cuda:0") or torch.cuda.device_count()!=1:
            raise ValueError("One explicit CUDA device required")
        properties=torch.cuda.get_device_properties(dev)
        result.update(gpu_name=properties.name, total_gpu_bytes=properties.total_memory,
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(dev),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(dev),
            reserved_gpu_bytes=torch.cuda.memory_reserved(dev), allocated_gpu_bytes=torch.cuda.memory_allocated(dev))
    return result


def check_process_resources(policy,device):
    """Reusable own-process admission before inputs and at update boundaries.

    This is not allocator forecasting, cgroup admission or fileset quota proof.
    Production CUDA configuration and source/runtime qualification precede it.
    """
    if type(policy) is not FittingResourcePolicy:
        raise TypeError("Exact fitting resource policy required")
    policy.__post_init__()
    device=torch.device(device)
    if policy.execution_scope=="synthetic_cpu_only" and device.type!="cpu":
        raise ValueError("Fabricated resource guard is CPU-only with two threads")
    if policy.execution_scope in ("scientific","synthetic_cuda_qualification") and device!=torch.device("cuda:0"):
        raise RuntimeError("Scientific resource checks require the admitted cuda:0")
    sample=process_resources(device)
    if any(type(sample.get(k)) is not int or sample[k]<0 for k in ("maximum_rss_bytes","threads")):
        raise ValueError("Invalid own-process resource sample")
    if sample["maximum_rss_bytes"]>policy.maximum_rss_bytes:
        raise RuntimeError(f"Observed worker RSS {sample['maximum_rss_bytes']} exceeds host allowance {policy.maximum_rss_bytes}")
    if policy.execution_scope in ("scientific","synthetic_cuda_qualification"):
        if (device!=torch.device("cuda:0") or sample["threads"]!=8 or "H100" not in sample.get("gpu_name","")
                or not 75*GIB<=sample.get("total_gpu_bytes",0)<=82*GIB
                or sample["total_gpu_bytes"]-sample.get("peak_reserved_gpu_bytes",sample["total_gpu_bytes"])<policy.minimum_gpu_headroom_bytes):
            raise RuntimeError("Observed H100 threads/device/headroom violates admitted resources")
    elif device.type!="cpu" or sample["threads"]>2:
        raise ValueError("Fabricated resource guard is CPU-only with two threads")
    return sample


def output_usage(root):
    """Exact leased trajectory only, logical file sizes; never read payloads."""
    root=_path(root,directory=True)
    if root.lstat().st_uid!=os.geteuid():
        raise ValueError("Worker output root ownership changed")
    def failed_scan(error):
        # os.walk otherwise ignores inaccessible subtrees, undercounting usage.
        raise error
    total=0
    for directory,folders,files in os.walk(root,followlinks=False,onerror=failed_scan):
        for name in (*folders,*files):
            path=_path(Path(directory)/name)
            info=path.lstat()
            if info.st_uid!=os.geteuid():
                raise ValueError("Worker output ownership changed")
            if stat.S_ISREG(info.st_mode) and info.st_nlink==1:
                total+=info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("Unexpected linked or nonregular output member")
    fs=os.statvfs(root)
    return dict(output_bytes=total, filesystem_available_bytes=fs.f_bavail*fs.f_frsize,
                fileset_quota_checked_by_this_component=False)


def _tensor_bytes(value):
    if isinstance(value,torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value,dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def snapshot_tensor_bytes(owner):
    """Count metadata dimensions without CPU copies or CUDA synchronization.

    Count duplicate references conservatively. The actual checkpoint writer
    bounds serialization separately. RNG/Python/container bookkeeping and
    transient copies are charged to the explicit snapshot/metadata reserve.
    FitnessCache.state_dict is deliberately not called: it already copies.
    """
    return (_tensor_bytes(owner.model.state_dict())+_tensor_bytes(owner.optimizer.state_dict())
            +_tensor_bytes((owner.cache.values,owner.cache.seen,owner.cache.age))
            +_tensor_bytes(owner.provider.state_dict()))


class FittingResourceGuard:
    """One immutable policy under one live lease, checked around actual updates."""
    def __init__(self,lease,policy):
        if type(lease) is not TrajectoryLease or type(policy) is not FittingResourcePolicy:
            raise TypeError("Actual trajectory lease and explicit resource policy required")
        lease.assert_active();policy.__post_init__()
        root=_path(lease.root,directory=True)
        if STUDY_ID not in root.parts or root.name==STUDY_ID:
            raise ValueError("Resource checks require one exact new-study trajectory")
        info=root.lstat()
        self.lease,self.policy=lease,policy
        self.identity=policy.identity
        self._root_identity=(info.st_dev,info.st_ino,info.st_uid)
        self._trajectory_identity=identity_sha256(lease.identity)
        self._owner=None
        self.checks=0
        self.last=None

    def bind(self,owner):
        if (self._owner is not None or owner.lease is not self.lease
                or owner.schedule.execution_scope!=self.policy.execution_scope
                or owner.maximum_checkpoint_bytes!=self.policy.checkpoint_limit_bytes):
            raise ValueError("Resource guard must bind one exact owner/scope/checkpoint allowance")
        self._owner=owner
        self.check("before_owner")

    def _bound(self):
        self.lease.assert_active()
        self.policy.__post_init__()
        root=_path(self.lease.root,directory=True)
        info=root.lstat()
        if (self.policy.identity!=self.identity or self._owner is None
                or (info.st_dev,info.st_ino,info.st_uid)!=self._root_identity
                or identity_sha256(self.lease.identity)!=self._trajectory_identity
                or self._owner.maximum_checkpoint_bytes!=self.policy.checkpoint_limit_bytes
                or self._owner.schedule.execution_scope!=self.policy.execution_scope):
            raise ValueError("Bound fitting resource policy/owner/output root changed")

    def check(self,event):
        if event not in ("before_owner","before_update","after_update","before_checkpoint","after_checkpoint","stage_complete"):
            raise ValueError("Unknown fitting resource boundary")
        self._bound()
        owner,policy=self._owner,self.policy
        sample=check_process_resources(policy,owner.device)
        reserve=policy.metadata_reserve_bytes
        if event=="before_checkpoint":
            tensor_bytes=snapshot_tensor_bytes(owner)
            if (tensor_bytes>policy.checkpoint_limit_bytes
                    or 2*tensor_bytes+policy.metadata_reserve_bytes>policy.snapshot_reserve_bytes
                    or sample["maximum_rss_bytes"]+policy.snapshot_reserve_bytes>policy.maximum_rss_bytes):
                raise RuntimeError("Checkpoint snapshot lacks admitted tensor/host headroom")
            sample.update(snapshot_tensor_bytes=tensor_bytes,
                          conservative_snapshot_reserve_bytes=policy.snapshot_reserve_bytes)
            reserve+=policy.checkpoint_limit_bytes
        # Scan only at entry/update and checkpoint boundaries, never inside a
        # law/particle/geometry kernel. No payload hashing or payload copies.
        usage=output_usage(self.lease.root)
        if usage["output_bytes"]+reserve>policy.output_limit_bytes or usage["filesystem_available_bytes"]<reserve:
            raise RuntimeError("Worker output allowance or filesystem free bytes cannot cover next write: "
                f"used={usage['output_bytes']}, reserve={reserve}, cap={policy.output_limit_bytes}, "
                f"filesystem_free={usage['filesystem_available_bytes']}")
        self.checks+=1
        self.last=dict(event=event,accepted_updates=owner.accepted_updates,
            policy_sha256=self.identity,policy=asdict(policy),checks=self.checks,reserved_next_write_bytes=reserve,
            output_limit_bytes=policy.output_limit_bytes,**sample,**usage)
        return dict(self.last)

    def record(self):
        # Failure journals must remain writable even when the guard just
        # rejected a changed policy or exhausted capacity. This is the last
        # successful observation, not a new resource admission.
        return None if self.last is None else dict(self.last)
