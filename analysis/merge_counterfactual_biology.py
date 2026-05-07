#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-fold CREDO counterfactual biology tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="counterfactual_biology_effects.csv files.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for raw in args.inputs:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "source_file" not in frame.columns:
            frame["source_file"] = str(path)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
