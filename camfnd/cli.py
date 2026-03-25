#!/usr/bin/env python3
"""CAMFND command-line interface.

Usage examples
--------------
Run the full pipeline with defaults and save outputs:

    python -m camfnd --output-dir ./camfnd_outputs

Run only specific steps:

    python -m camfnd --steps 1 2
    python -m camfnd --steps 3 4 --output-dir ./outputs

Use compact (faster) benchmark configs:

    python -m camfnd --fast --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STEP_NAMES = {
    1: "data_contract",
    2: "simulator_validation",
    3: "single_screen_model",
    4: "multiscreen_context_model",
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="camfnd",
        description="Control-Anchored Mean-Field Neural Differential Equations — CAMFND pipeline runner",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        metavar="N",
        help="Which benchmark phases to run by legacy numeric id (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for output CSVs and JSON evaluation files.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller benchmark configs for faster iteration.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step progress output.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    steps = sorted(set(args.steps))
    verbose = not args.quiet

    # Import lazily to avoid slow PyTorch import when just asking for --help.
    from camfnd.data.multiscreen_benchmark import Stage2BenchmarkConfig, generate_multiscreen_dataset
    from camfnd.data.single_screen_benchmark import Stage1BenchmarkConfig, generate_single_screen_dataset
    from camfnd.training.multiscreen_context_model import Stage2TrainConfig
    from camfnd.training.single_screen_model import Stage1TrainConfig

    if args.fast:
        s1_cfg = Stage1BenchmarkConfig(n_obs_p4=32, n_obs_p60=32)
        s2_cfg = Stage2BenchmarkConfig(seed=29, n_obs_p4=32, n_obs_p60=32, n_truth_particles=512, n_steps=32)
        s1_train = Stage1TrainConfig(epochs=20)
        s2_train = Stage2TrainConfig(epochs=15)
    else:
        s1_cfg = Stage1BenchmarkConfig()
        s2_cfg = Stage2BenchmarkConfig(seed=29, n_obs_p4=32, n_obs_p60=32, n_truth_particles=1024, n_steps=48)
        s1_train = Stage1TrainConfig()
        s2_train = Stage2TrainConfig()

    results = {}

    if 1 in steps or 2 in steps:
        single_screen_dataset = generate_single_screen_dataset(s1_cfg)

    if 1 in steps:
        from camfnd.evaluation.data_contract import evaluate_data_contract
        if verbose:
            print("Running data_contract ...")
        ev = evaluate_data_contract(single_screen_dataset)
        results[STEP_NAMES[1]] = ev.to_dict()
        if verbose:
            print(json.dumps(ev.to_dict(), indent=2))
        if args.output_dir:
            s1_out = args.output_dir / STEP_NAMES[1]
            s1_out.mkdir(parents=True, exist_ok=True)
            single_screen_dataset.cells.obs.to_csv(s1_out / "cells_obs.csv", index=False)
            single_screen_dataset.masses.table.to_csv(s1_out / "masses.csv", index=False)
            ev.count_summary.to_csv(s1_out / "count_summary.csv", index=False)
            ev.empirical_terminal_summary.to_csv(s1_out / "empirical_terminal_summary.csv", index=False)
            ev.analytic_comparison.to_csv(s1_out / "analytic_comparison.csv", index=False)
            (s1_out / "evaluation.json").write_text(json.dumps(ev.to_dict(), indent=2))

    if 2 in steps:
        from camfnd.evaluation.simulator_validation import evaluate_simulator_validation
        if verbose:
            print("Running simulator_validation ...")
        ev = evaluate_simulator_validation(single_screen_dataset)
        results[STEP_NAMES[2]] = ev.to_dict()
        if verbose:
            print(json.dumps(ev.to_dict(), indent=2))
        if args.output_dir:
            s2_out = args.output_dir / STEP_NAMES[2]
            s2_out.mkdir(parents=True, exist_ok=True)
            ev.default_run_summary.to_csv(s2_out / "default_run_summary.csv", index=False)
            ev.default_analytic_comparison.to_csv(s2_out / "default_analytic_comparison.csv", index=False)
            ev.convergence_table.to_csv(s2_out / "convergence_table.csv", index=False)
            (s2_out / "evaluation.json").write_text(json.dumps(ev.to_dict(), indent=2))

    if 3 in steps:
        if 1 not in steps:
            single_screen_dataset = generate_single_screen_dataset(s1_cfg)
        from camfnd.evaluation.single_screen_model import evaluate_single_screen_model
        if verbose:
            print("Running single_screen_model ...")
        ev = evaluate_single_screen_model(single_screen_dataset, full_config=s1_train)
        results[STEP_NAMES[3]] = ev.to_dict()
        if verbose:
            print(json.dumps(ev.to_dict(), indent=2))
        if args.output_dir:
            s3_out = args.output_dir / STEP_NAMES[3]
            s3_out.mkdir(parents=True, exist_ok=True)
            ev.summary_table.to_csv(s3_out / "summary_table.csv", index=False)
            ev.full_result.history.to_csv(s3_out / "full_history.csv", index=False)
            ev.full_result.final_simulation.summary.to_csv(s3_out / "full_terminal_summary.csv", index=False)
            (s3_out / "evaluation.json").write_text(json.dumps(ev.to_dict(), indent=2))

    if 4 in steps:
        multiscreen_dataset = generate_multiscreen_dataset(s2_cfg)
        from camfnd.evaluation.multiscreen_context_model import evaluate_multiscreen_context_model
        if verbose:
            print("Running multiscreen_context_model ...")
        ev = evaluate_multiscreen_context_model(multiscreen_dataset, full_config=s2_train)
        results[STEP_NAMES[4]] = ev.to_dict()
        if verbose:
            print(json.dumps(ev.to_dict(), indent=2))
        if args.output_dir:
            s4_out = args.output_dir / STEP_NAMES[4]
            s4_out.mkdir(parents=True, exist_ok=True)
            ev.summary_table.to_csv(s4_out / "summary_table.csv", index=False)
            ev.full_result.history.to_csv(s4_out / "full_history.csv", index=False)
            ev.no_context_result.history.to_csv(s4_out / "no_context_history.csv", index=False)
            ev.full_result.final_loss_table.to_csv(s4_out / "full_terminal_summary.csv", index=False)
            ev.full_result.final_simulation.context_summary.to_csv(s4_out / "full_context_summary.csv", index=False)
            if multiscreen_dataset.truth and multiscreen_dataset.truth.context_trajectories is not None:
                multiscreen_dataset.truth.context_trajectories.to_csv(s4_out / "truth_context_trajectories.csv", index=False)
            (s4_out / "evaluation.json").write_text(json.dumps(ev.to_dict(), indent=2))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))

    all_pass = all(v.get("ok", False) for v in results.values())
    if verbose:
        print(f"\nAll pass: {all_pass}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
