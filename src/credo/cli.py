"""The single CREDO command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .io import load_config, load_data, validate_inputs
from .registry import get_recipe
from .runtime import TrainingEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credo")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a strict run config and its data")
    validate.add_argument("config", type=Path)

    for name in ("train", "run"):
        train = commands.add_parser(name, help="fit a released recipe from one strict config")
        train.add_argument("config", type=Path)
        train.add_argument("--output-dir", type=Path)
        train.add_argument("--device")
        train.add_argument("--seed", type=int)

    evaluate = commands.add_parser("evaluate", help="evaluate a native checkpoint or import bundle")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--study-source", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--particles", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--device", default="cpu")

    contrast = commands.add_parser(
        "counterfactual", help="run a same-start, same-noise reference contrast"
    )
    contrast.add_argument("checkpoint", type=Path)
    contrast.add_argument("--measure-id", required=True)
    contrast.add_argument("--config", type=Path)
    contrast.add_argument("--study-source", type=Path)
    contrast.add_argument("--output", type=Path)
    contrast.add_argument("--particles", type=int)
    contrast.add_argument("--seed", type=int)
    contrast.add_argument("--device", default="cpu")
    contrast.add_argument(
        "--context-policy", choices=["self_consistent", "clamped"], default="self_consistent"
    )

    replay = commands.add_parser("replay", help="replay a transformer-v2 OOF archive")
    replay.add_argument("--bundle-root", required=True, type=Path)
    replay.add_argument("--study-source", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--particles", type=int, default=640)
    replay.add_argument("--steps-per-interval", type=int, default=24)
    replay.add_argument("--noise-seed", type=int, default=0)
    replay.add_argument("--device")

    summarize = commands.add_parser("summarize", help="summarize durable run artifacts")
    summarize.add_argument("run_dir", type=Path)

    importer = commands.add_parser(
        "import-checkpoint", help="normalize a historical checkpoint for inference"
    )
    importer.add_argument("--format", required=True, choices=["legacy-v2-transformer"])
    importer.add_argument("--checkpoint", required=True, type=Path)
    importer.add_argument("--run-config", required=True, type=Path)
    importer.add_argument("--representation", required=True, type=Path)
    importer.add_argument("--latents", required=True, type=Path)
    importer.add_argument("--output", required=True, type=Path)
    importer.add_argument("--vae-metadata", type=Path)
    importer.add_argument("--catalog", type=Path)
    importer.add_argument("--control-id", action="append")
    importer.add_argument("--study-source", type=Path)
    importer.add_argument("--state", choices=["raw", "ema"], default="raw")
    importer.add_argument("--device", default="cpu")
    return parser


def _train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    updates = config.model_dump()
    changed = False
    if args.output_dir is not None:
        updates["output"] = args.output_dir.expanduser().resolve()
        changed = True
    if args.seed is not None:
        training = updates.get("recipe_config", {}).get("training")
        if not isinstance(training, dict) or "seed" not in training:
            raise ValueError(f"Recipe {config.recipe!r} does not expose a training.seed override.")
        training["seed"] = args.seed
        changed = True
    if changed:
        config = type(config).model_validate(updates)
    data = load_data(config)
    recipe = get_recipe(config.recipe)
    trainer = TrainingEngine().fit(recipe, data, config, device=args.device)
    output = trainer.save()
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


def _load_runtime(args: argparse.Namespace):
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        from .recipes.transformer_v2.importer import load_imported_bundle
        from .recipes.transformer_v2.replay import load_lps_replay_study

        if args.study_source is None:
            raise ValueError("Imported transformer-v2 bundles require --study-source.")
        run = load_imported_bundle(checkpoint, device=args.device)
        study = load_lps_replay_study(run, args.study_source)
        run.study = study
        return run, study
    if args.config is None:
        raise ValueError("Native compact checkpoints require --config.")
    from .training import Trainer

    config = load_config(args.config)
    study = load_data(config)
    return Trainer.load(checkpoint, study, config, device=args.device), study


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate

    run, study = _load_runtime(args)
    options = {"study": study, "particles": args.particles, "seed": args.seed}
    if getattr(run, "recipe_id", None) == "credo.transformer_sde_v2":
        options["device"] = args.device
    metrics = evaluate(run, **options)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(output, index=False)
    print(json.dumps({"status": "evaluated", "rows": len(metrics)}, indent=2))
    return 0


def _counterfactual(args: argparse.Namespace) -> int:
    from .counterfactual import counterfactual

    run, study = _load_runtime(args)
    options = {
        "context_policy": args.context_policy,
        "particles": args.particles,
        "seed": args.seed,
        "study": study,
        "device": args.device,
    }
    frame = counterfactual(run, args.measure_id, **options)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
    print(json.dumps({"status": "evaluated", "rows": len(frame)}, indent=2))
    return 0


def _replay(args: argparse.Namespace) -> int:
    from .recipes.transformer_v2.replay import replay_lps_bundle

    summary = replay_lps_bundle(
        args.bundle_root,
        args.study_source,
        args.output,
        particles=args.particles,
        steps_per_interval=args.steps_per_interval,
        noise_seed=args.noise_seed,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
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


def _import_checkpoint(args: argparse.Namespace) -> int:
    from .recipes.transformer_v2.importer import import_legacy_checkpoint

    run = import_legacy_checkpoint(
        args.checkpoint,
        args.run_config,
        args.representation,
        args.latents,
        output=args.output,
        vae_metadata=args.vae_metadata,
        catalog=args.catalog,
        controls=args.control_id,
        study_source=args.study_source,
        model_state=args.state,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": "imported",
                "mode": run.envelope.mode.value,
                "recipe": f"{run.recipe_id}@{run.recipe_version}",
                "state": run.model_state,
                "output": str(args.output.expanduser().resolve()),
                "source_checkpoint_sha256": run.envelope.import_provenance[
                    "source_checkpoint_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_inputs(args.config), indent=2))
        return 0
    if args.command in {"train", "run"}:
        return _train(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "counterfactual":
        return _counterfactual(args)
    if args.command == "replay":
        return _replay(args)
    if args.command == "summarize":
        return _summarize(args.run_dir)
    if args.command == "import-checkpoint":
        return _import_checkpoint(args)
    raise AssertionError(f"Unhandled command {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
