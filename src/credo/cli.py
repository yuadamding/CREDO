"""The single CREDO command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .io import load_config, load_data, validate_inputs
from .training import Trainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credo")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a strict run config and its data")
    validate.add_argument("config", type=Path)

    run = commands.add_parser("run", help="fit the fixed CREDO pipeline")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--device")
    run.add_argument("--seed", type=int)

    summarize = commands.add_parser("summarize", help="summarize durable run artifacts")
    summarize.add_argument("run_dir", type=Path)
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    updates = config.model_dump()
    changed = False
    if args.output_dir is not None:
        updates["output"] = args.output_dir.expanduser().resolve()
        changed = True
    if args.seed is not None:
        updates["training"]["seed"] = args.seed
        changed = True
    if changed:
        config = type(config).model_validate(updates)
    data = load_data(config)
    trainer = Trainer.fit(data, None, config, device=args.device)
    output = trainer.save()
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


def _summarize(run_dir: Path) -> int:
    directory = run_dir.expanduser().resolve()
    with (directory / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    metrics = pd.read_parquet(directory / "metrics.parquet")
    history = pd.read_parquet(directory / "history.parquet")
    summary = {
        "package_version": manifest.get("package_version"),
        "axis": manifest.get("axis"),
        "mass_semantics": manifest.get("mass_semantics"),
        "validation_source": manifest.get("validation_split", {}).get("source"),
        "epochs": int(len(history)),
        "metric_rows": int(len(metrics)),
        "mean_geometry": float(metrics["geometry"].mean()),
        "mean_log_mass_error": float(metrics["log_mass_error"].mean()),
        "minimum_ess_fraction": float(metrics["ess_fraction"].min()),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_inputs(args.config), indent=2))
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "summarize":
        return _summarize(args.run_dir)
    raise AssertionError(f"Unhandled command {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
