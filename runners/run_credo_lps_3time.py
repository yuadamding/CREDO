"""LPS 90m -> 6h -> 10h preset for the CREDO trajectory runner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runners.run_credo_trajectory import main as trajectory_main


DEFAULT_DATA = "../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad"
DEFAULT_OUTPUT = "runs/lps_90m_6h_10h_credo3"


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    defaults = {
        "--data-path": DEFAULT_DATA,
        "--output-dir": DEFAULT_OUTPUT,
        "--source-label": "90m",
        "--target-labels": "6h,10h",
        "--physical-times": "90m:1.5,6h:6.0,10h:10.0",
        "--key-mode": "sample_aware",
        "--latent-source": "vae",
        "--vae-layer": "counts",
        "--endpoint-time-weights": "6h:0.5,10h:1.0",
        "--steps-per-interval": "12",
    }
    prefixed: list[str] = []
    for flag, value in defaults.items():
        if not _has_flag(args, flag):
            prefixed.extend([flag, value])
    trajectory_main(prefixed + args)


if __name__ == "__main__":
    main()
