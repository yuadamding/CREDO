"""Cohort runner for the Rest -> Stim8hr -> Stim48hr GSE314342 trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from .run_credo_trajectory import main as trajectory_main
except ImportError:  # Direct script execution.
    from run_credo_trajectory import main as trajectory_main


COHORT_DEFAULTS = [
    "--source-label", "Rest",
    "--target-labels", "Stim8hr,Stim48hr",
    "--physical-times", "Rest:0,Stim8hr:8,Stim48hr:48",
    "--perturbation-col", "guide_id",
    "--mass-mode", "group_total",
    "--mass-scope", "full_obs",
    "--latent-source", "obsm",
    "--latent-key", "X_credo",
    "--context-protocol", "none",
    "--n-particles", "64",
    "--eval-particles", "256",
    "--max-active-measure-keys", "256",
    "--max-train-target-atoms", "32",
    "--max-eval-target-atoms", "64",
    "--endpoint-time-weights", "Stim8hr:1,Stim48hr:1",
    "--sinkhorn-tau", "0.25",
    "--lambda-weak", "0.01",
    "--epochs", "40",
    "--eval-every", "10",
    "--stage", "geometry",
]


def configured_args(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="")
    known, remaining = parser.parse_known_args(argv)
    if not known.config:
        return COHORT_DEFAULTS + remaining
    payload = json.loads(Path(known.config).read_text(encoding="utf-8"))
    config_args = payload.get("args")
    if not isinstance(config_args, list) or not all(isinstance(item, str) for item in config_args):
        raise TypeError("GSE314342 config must contain a string list named 'args'.")
    return COHORT_DEFAULTS + config_args + remaining


if __name__ == "__main__":
    trajectory_main(configured_args(sys.argv[1:]))
