"""The single CREDO command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .data import open_study
from .data.splits import validate_representation_scope, validate_split_plan
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

    evaluate = commands.add_parser("evaluate", help="evaluate a run manifest or checkpoint")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--particles", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--device", default="cpu")

    contrast = commands.add_parser(
        "counterfactual", help="run a same-start, same-noise reference contrast"
    )
    contrast.add_argument("checkpoint", type=Path)
    contrast.add_argument("--measure-id", "--series-id", dest="measure_id", required=True)
    contrast.add_argument("--config", type=Path)
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
    importer.add_argument("--bind-config", type=Path)
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
    study = open_study(config if config.study is None else config.study, verify="semantic")
    try:
        trainer = TrainingEngine().fit(recipe, config.view(study), config, device=args.device)
        output = trainer.save()
    finally:
        study.close()
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


def _load_runtime(args: argparse.Namespace):
    checkpoint = args.checkpoint.expanduser().resolve()
    run_manifest = checkpoint / "run.json" if checkpoint.is_dir() else checkpoint
    if run_manifest.name == "run.json" and run_manifest.is_file():
        from .artifacts import open_run

        run = open_run(run_manifest, device=args.device)
        study = getattr(run, "data", None) or getattr(run, "study", None)
        return run, study, None
    if args.config is None:
        raise ValueError("A legacy checkpoint requires --config; run.json is self-describing.")
    config = load_config(args.config)
    recipe = get_recipe(config.recipe)
    semantic_study = open_study(
        config if config.study is None else config.study,
        verify="semantic",
    )
    try:
        view = config.view(semantic_study)
        split = recipe.plan_split(view, config.recipe_config)
        validate_split_plan(view, split)
        validate_representation_scope(view, split)
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
        if owner is not None:
            owner.close()
        close = getattr(run, "close", None)
        if owner is None and callable(close):
            close()
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
        if owner is not None:
            owner.close()
        close = getattr(run, "close", None)
        if owner is None and callable(close):
            close()
    print(json.dumps({"status": "evaluated", "rows": len(frame)}, indent=2))
    return 0


def _summarize(run_dir: Path) -> int:
    from .artifacts import _verified_run_manifest

    manifest, directory = _verified_run_manifest(run_dir)

    def output(name: str) -> Path:
        relative = manifest["outputs"][name]
        if relative not in manifest["artifacts"]:
            raise ValueError(f"Run output {name!r} is absent from the artifact catalog.")
        return directory / relative

    metrics = pd.read_parquet(output("metrics"))
    predictions = pd.read_parquet(output("predictions"))
    diagnostics = pd.read_parquet(output("diagnostics"))
    history = pd.read_parquet(output("history"))
    geometry = metrics.loc[metrics["metric_name"].eq("sinkhorn_divergence"), "value"]
    mass_error = metrics.loc[metrics["metric_name"].eq("log_abundance_squared_error"), "value"]
    ess = diagnostics.loc[diagnostics["diagnostic_name"].eq("particle_ess_fraction"), "value"]
    summary = {
        "package_version": manifest.get("package_version"),
        "axis": manifest.get("axis"),
        "mass_semantics": manifest.get("mass_semantics"),
        "validation_source": manifest.get("validation_split", {}).get("source"),
        "epochs": int(len(history)),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "mean_geometry": float(geometry.mean()),
        "mean_log_mass_error": float(mass_error.mean()),
        "minimum_ess_fraction": float(ess.min()),
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
    if args.bind_config is not None:
        from .artifacts import bind_run_study

        bind_run_study(args.output / "run.json", args.bind_config)
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
