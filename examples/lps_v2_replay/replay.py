from __future__ import annotations

import argparse
import json
from pathlib import Path

from credo.recipes.transformer_v2.replay import replay_lps_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--study-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", action="append")
    parser.add_argument("--particles", type=int, default=640)
    parser.add_argument("--steps-per-interval", type=int, default=24)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = replay_lps_bundle(
        args.bundle_root,
        args.study_source,
        args.output,
        folds=None if args.fold is None else tuple(args.fold),
        particles=args.particles,
        steps_per_interval=args.steps_per_interval,
        noise_seed=args.noise_seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
