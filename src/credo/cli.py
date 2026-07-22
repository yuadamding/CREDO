"""The single CREDO command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .contracts import SplitSpec
from .data import open_study
from .io import load_config, validate_inputs
from .registry import get_recipe
from .runtime import TrainingEngine, validate_view_for_recipe


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

    evaluate = commands.add_parser("evaluate", help="evaluate a native checkpoint")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--particles", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--device", default="cpu")

    contrast = commands.add_parser(
        "counterfactual", help="run a same-start, same-noise reference contrast"
    )
    contrast.add_argument("checkpoint", type=Path)
    contrast.add_argument("--measure-id", required=True)
    contrast.add_argument("--config", required=True, type=Path)
    contrast.add_argument("--output", type=Path)
    contrast.add_argument("--particles", type=int)
    contrast.add_argument("--seed", type=int)
    contrast.add_argument("--device", default="cpu")
    contrast.add_argument(
        "--context-policy", choices=["self_consistent", "clamped"], default="self_consistent"
    )

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
    recipe = get_recipe(config.recipe)
    study = open_study(config, verify="semantic")
    try:
        trainer = TrainingEngine().fit(recipe, study.view(), config, device=args.device)
        output = trainer.save()
    finally:
        study.close()
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


def _load_runtime(args: argparse.Namespace):
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        raise ValueError(
            "Core CLI evaluation accepts native checkpoints only. Imported transformer-v2 "
            "bundles require a canonical compiled study through the Python API; cohort replay "
            "belongs in its external adapter workflow."
        )
    config = load_config(args.config)
    recipe = get_recipe(config.recipe)
    semantic_study = open_study(config, verify="semantic")
    try:
        view = semantic_study.view()
        representation_scope = getattr(
            getattr(config.recipe_config, "validation", None),
            "representation_scope",
            "shared",
        )
        split = SplitSpec(
            strategy="none",
            representation_scope=representation_scope,
            split_id=f"study-view:{view.semantic_hash()[:12]}",
        )
        validate_view_for_recipe(
            view,
            split,
            recipe.requirements(config.recipe_config),
        ).raise_for_errors()
        compiled_study = recipe.compile_study(view, split, config.recipe_config)
        from .training import Trainer

        run = Trainer.load(checkpoint, compiled_study, config, device=args.device)
    except Exception:
        semantic_study.close()
        raise
    return run, compiled_study, semantic_study


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate

    run, study, owner = _load_runtime(args)
    try:
        options = {"study": study, "particles": args.particles, "seed": args.seed}
        if getattr(run, "recipe_id", None) == "credo.transformer_sde_v2":
            options["device"] = args.device
        metrics = evaluate(run, **options)
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_parquet(output, index=False)
    finally:
        owner.close()
    print(json.dumps({"status": "evaluated", "rows": len(metrics)}, indent=2))
    return 0


def _counterfactual(args: argparse.Namespace) -> int:
    from .counterfactual import counterfactual

    run, study, owner = _load_runtime(args)
    try:
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
    finally:
        owner.close()
    print(json.dumps({"status": "evaluated", "rows": len(frame)}, indent=2))
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
    if args.command == "summarize":
        return _summarize(args.run_dir)
    if args.command == "import-checkpoint":
        return _import_checkpoint(args)
    raise AssertionError(f"Unhandled command {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
