"""Bounded terminal output byte inventory, not a numerical or scheduler pass."""

from hashlib import sha256
import os
from pathlib import Path
import stat

from .observation_loader import ObservationFile, _path, _verified_file
from .main_worker_contract import MAXIMUM_BYTES, parse


def signature(info):
    return tuple(getattr(info,k) for k in ("st_dev","st_ino","st_mode","st_nlink","st_uid",
                                          "st_gid","st_size","st_mtime_ns","st_ctime_ns"))


class OutputInventory:
    """Read exact outputs once, then detect additions/removals/changes on exit.

    Does not keep thousands of checkpoint descriptors open or deserialize them.
    The caller must separately establish terminal ownership before acceptance.
    """
    def __init__(self,root,*,limit,excluded=()):
        self.root=_path(root)
        self.limit=limit
        self.excluded=frozenset(excluded)
        if (type(limit) is not int or not 0<limit<=16*1024**3
                or not self.excluded<=frozenset(("READBACK.log","READBACK_EXIT.json","TERMINAL_READBACK.json","COMPLETE.json"))):
            raise ValueError("Exact qualification output scope required")
        self.files,self.signatures,self.total_bytes={}, {}, 0
        initial=self._scan()
        for name,before in initial.items():
            path=self.root/name
            digest=sha256()
            with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),"rb") as stream:
                if signature(os.fstat(stream.fileno()))!=before:
                    raise ValueError("Output changed before inventory read")
                while chunk:=stream.read(2*1024**2): digest.update(chunk)
                if signature(os.fstat(stream.fileno()))!=before:
                    raise ValueError("Output changed during inventory read")
            if signature(path.lstat())!=before:raise ValueError("Output changed after inventory read")
            self.files[name]=dict(sha256=digest.hexdigest(),size_bytes=before[6])
        self.signatures=initial
        self.total_bytes=sum(x["size_bytes"] for x in self.files.values())
        self.verify_exit()

    def _scan(self):
        root_info=self.root.lstat()
        if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid!=os.geteuid():
            raise ValueError("Exact owned output directory required")
        result,total={},0
        def failed(error):raise error
        for directory,folders,files in os.walk(self.root,followlinks=False,onerror=failed):
            for name in sorted((*folders,*files)):
                path=_path(Path(directory)/name);info=path.lstat()
                if info.st_uid!=os.geteuid():raise ValueError("Output owner differs")
                if stat.S_ISDIR(info.st_mode):continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
                    raise ValueError("Output must be regular single-link files")
                relative=path.relative_to(self.root).as_posix()
                if relative in self.excluded:continue
                if name.endswith("FAILED.json") or name.startswith(".publishing-"):
                    raise ValueError("Retained failure or incomplete publication prevents acceptance")
                if name.endswith(".log") and info.st_size>4*1024**2:
                    raise ValueError("Qualification process log exceeded its byte limit")
                total+=info.st_size
                if total>self.limit or len(result)>=20000:
                    raise ValueError("Qualification output bytes or file count exceeded")
                result[relative]=signature(info)
        if not result:raise ValueError("Empty qualification output inventory")
        return result

    def reference(self,path,*,maximum=MAXIMUM_BYTES):
        path=_path(path)
        if not path.is_relative_to(self.root):raise ValueError("Output reference escaped the exact run")
        name=path.relative_to(self.root).as_posix()
        item=self.files.get(name)
        if item is None or not 0<item["size_bytes"]<=maximum:
            raise ValueError("Required bounded publication missing from output inventory")
        return ObservationFile(str(path),**item)

    def read(self,path,reference=None):
        actual=self.reference(path)
        if reference is not None and actual!=reference:
            raise ValueError("Declared output hash/size/path differs from inventory")
        with _verified_file(actual,maximum=MAXIMUM_BYTES) as stream:
            return parse(stream.read())

    def verify_exit(self):
        if self._scan()!=self.signatures:raise ValueError("Output inventory changed during terminal verification")
