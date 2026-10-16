"""Tiny local I/O subprocesses; no optimizer, CUDA, data or remote operations."""

import os
from pathlib import Path
import sys

import pytest

from frontier.bounded_process import run_process


def launch(tmp_path, code, **kwargs):
    return run_process([sys.executable, "-c", code], cwd=str(tmp_path), env=dict(os.environ),
                       log_path=tmp_path/"child.log", **kwargs)


def test_stdout_stderr_and_nonzero_exit_are_retained(tmp_path):
    assert launch(tmp_path, "import os; os.write(1,b'out'); os.write(2,b'err'); raise SystemExit(7)") == 7
    assert (tmp_path/"child.log").read_bytes() == b"outerr"
    with pytest.raises(FileExistsError): launch(tmp_path, "raise AssertionError('must not start')")


def test_log_limit_stops_only_its_child_without_retry(tmp_path):
    with pytest.raises(RuntimeError, match="byte allowance"):
        launch(tmp_path, "import os\nwhile True: os.write(1,b'x'*1024)", maximum_log_bytes=257)
    assert (tmp_path/"child.log").read_bytes() == b"x"*257


@pytest.mark.parametrize("limit", [True, 0, -1, 4*1024**2+1])
def test_invalid_limit_before_process(tmp_path, limit):
    with pytest.raises(ValueError): launch(tmp_path, "pass", maximum_log_bytes=limit)
    assert not (tmp_path/"child.log").exists()


def test_symlink_log_not_followed(tmp_path):
    target=tmp_path/"retained";target.write_bytes(b"keep")
    (tmp_path/"child.log").symlink_to(target)
    with pytest.raises(ValueError): launch(tmp_path, "pass")
    assert target.read_bytes()==b"keep"
