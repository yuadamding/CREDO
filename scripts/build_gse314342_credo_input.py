#!/usr/bin/env python3
"""Forward to the workspace-level GSE314342 converter."""
from __future__ import annotations

from pathlib import Path
import runpy


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_gse314342_credo_input.py"
if not SCRIPT.is_file():
    raise FileNotFoundError(
        "The cohort converter belongs at /home/yding1995/opscc_sc/scripts/"
        "build_gse314342_credo_input.py."
    )
runpy.run_path(str(SCRIPT), run_name="__main__")
