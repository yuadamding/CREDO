"""CLI entry point: train the model or run benchmarks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the P4/P60 PINN model")
    sub = parser.add_subparsers(dest="command")

    # benchmark sub-command
    bench = sub.add_parser("benchmark", help="Run synthetic benchmarks")
    bench.add_argument("--epochs", type=int, default=200)
    bench.add_argument("--device", type=str, default="auto")
    bench.add_argument("--output-dir", type=str, default="outputs/benchmarks")

    args = parser.parse_args()

    if args.command == "benchmark":
        from cape.benchmarks.benchmark_runner import run_all_benchmarks
        run_all_benchmarks(
            output_dir=args.output_dir,
            train_epochs=args.epochs,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
