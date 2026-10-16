"""One foreground child with bounded on-disk output, no deadline or retry."""

import os
from pathlib import Path
import subprocess


def run_process(command, *, cwd, env, log_path, maximum_log_bytes=4*1024**2):
    if (type(command) is not list or not command or any(type(x) is not str for x in command)
            or type(maximum_log_bytes) is not int or not 1 <= maximum_log_bytes <= 4*1024**2):
        raise ValueError("Explicit command and at most four MiB log allowance required")
    path=Path(log_path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError("Exact regular owned log path required")
    with path.open("xb") as log:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,shell=False)
        try:
            written=0
            while True:
                block=os.read(process.stdout.fileno(),64*1024)
                if not block: break
                if written+len(block)>maximum_log_bytes:
                    log.write(block[:maximum_log_bytes-written]);log.flush();os.fsync(log.fileno())
                    raise RuntimeError("Child log exceeds the declared byte allowance; no retry")
                log.write(block);written+=len(block)
            result=process.wait()
            log.flush();os.fsync(log.fileno())
            return result
        finally:
            # Only our own still-live direct child, and only after an actual
            # read/write/resource failure or external interruption, is stopped.
            if process.poll() is None:
                process.kill();process.wait()
            process.stdout.close()
