"""Train CREDO on one-snapshot Perturb-seq as a non-physical effect axis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "package"
sys.path.insert(0, str(ROOT / "src"))

from credo.config.schema import RunConfig
from credo.data import build_single_time_problem_from_anndata
from credo.models import FullDynamicsModel
from credo.training import SingleTimeTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-key", default="X_pca")
    parser.add_argument("--perturbation-col", default="perturbation_id")
    parser.add_argument("--guide-col", default="guide_id")
    parser.add_argument("--target-gene-col", default="target_gene")
    parser.add_argument("--control-col", default="is_control")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--batch-col", default="batch_id")
    parser.add_argument("--embedding-level", choices=["perturbation", "guide", "target_gene"], default="target_gene")
    parser.add_argument(
        "--view-key-level",
        choices=["perturbation", "guide", "sample_perturbation", "sample_guide"],
        default="sample_guide",
        help="Finite-measure view key. Use guide/sample_guide to preserve guide-level views.",
    )
    parser.add_argument("--view-level", choices=["view", "embedding"], default="view")
    parser.add_argument("--mass-mode", choices=["cell_count", "unit_mass", "obs_column", "unavailable"], default="unit_mass")
    parser.add_argument("--mass-value-col", default=None)
    parser.add_argument("--mass-claim-grade", choices=["auto", "none", "diagnostic", "claim_grade"], default="auto")
    parser.add_argument("--reference-scope", choices=["auto", "sample", "batch", "global"], default="auto")
    parser.add_argument(
        "--context-protocol",
        choices=["observed_snapshot", "source_reference", "self_consistent", "clamped_external"],
        default="observed_snapshot",
    )
    parser.add_argument("--context-sampling", choices=["fixed", "epoch_resample"], default="fixed")
    parser.add_argument(
        "--context-gradient-mode",
        choices=["detached_cache", "recompute_no_grad", "recompute_with_grad"],
        default="recompute_no_grad",
    )
    parser.add_argument("--context-tau", default="auto")
    parser.add_argument("--min-cells", type=int, default=1)
    parser.add_argument("--control-split-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--n-particles", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--lr-net", type=float, default=3e-4)
    parser.add_argument("--lr-embed", type=float, default=1e-3)
    parser.add_argument("--lambda-weak", type=float, default=0.0)
    parser.add_argument("--lambda-reg-net", type=float, default=1e-4)
    parser.add_argument("--lambda-reg-diffusion", type=float, default=1e-4)
    parser.add_argument("--lambda-reg-embed", type=float, default=1e-4)
    parser.add_argument("--lambda-control-null", type=float, default=0.0)
    parser.add_argument("--lambda-minimal-action", type=float, default=1e-4)
    parser.add_argument("--lambda-guide-concordance", type=float, default=0.0)

    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--mediator-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--control-mode", choices=["anchored", "free", "soft_ref"], default="soft_ref")
    parser.add_argument("--ecology-on", dest="ecological_growth", action="store_true")
    parser.add_argument("--ecology-off", dest="ecological_growth", action="store_false")
    parser.set_defaults(ecological_growth=True)
    return parser.parse_args(argv)


def _parse_context_tau(value: str) -> str | float:
    if value in {"auto", "source", "target", "midpoint"}:
        return value
    return float(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_single_time_problem_from_anndata(
        args.data_path,
        latent_key=args.latent_key,
        perturbation_col=args.perturbation_col,
        guide_col=args.guide_col or None,
        target_gene_col=args.target_gene_col or None,
        embedding_level=args.embedding_level,
        view_key_level=args.view_key_level,
        control_col=args.control_col,
        sample_col=args.sample_col or None,
        batch_col=args.batch_col or None,
        mass_mode=args.mass_mode,
        mass_value_col=args.mass_value_col,
        mass_claim_grade=args.mass_claim_grade,
        reference_scope=args.reference_scope,
        context_protocol=args.context_protocol,
        min_cells=args.min_cells,
        control_split_seed=args.control_split_seed,
    )
    latent_dim = problem.views[0].target.latent_dim
    config = RunConfig(
        device=args.device,
        latent={"dim": latent_dim, "key": args.latent_key},
        simulation={
            "n_particles": args.n_particles,
            "n_steps": args.n_steps,
            "store_history": True,
        },
        training={
            "epochs": args.epochs,
            "seed": args.seed,
            "precision": args.precision,
            "lr_net": args.lr_net,
            "lr_embed": args.lr_embed,
            "lambda_count": 0.0,
            "lambda_weak": args.lambda_weak,
            "lambda_reg_net": args.lambda_reg_net,
            "lambda_reg_diffusion": args.lambda_reg_diffusion,
            "lambda_reg_embed": args.lambda_reg_embed,
            "max_active_perturbations": 0,
        },
        single_time={
            "enabled": True,
            "view_level": args.view_level,
            "view_key_level": args.view_key_level,
            "context_protocol": args.context_protocol,
            "context_sampling": args.context_sampling,
            "context_gradient_mode": args.context_gradient_mode,
            "context_tau": _parse_context_tau(args.context_tau),
            "mass_mode": args.mass_mode,
            "mass_claim_grade": args.mass_claim_grade,
            "lambda_control_null": args.lambda_control_null,
            "lambda_minimal_action": args.lambda_minimal_action,
            "lambda_guide_concordance": args.lambda_guide_concordance,
        },
    )
    model = FullDynamicsModel(
        perturbation_ids=problem.catalog.perturbation_ids,
        control_ids=problem.catalog.control_ids,
        latent_dim=latent_dim,
        embedding_dim=args.embedding_dim,
        n_programs=args.n_programs,
        mediator_dim=args.mediator_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        ecological_growth=args.ecological_growth,
        control_mode=args.control_mode,
    )
    trainer = SingleTimeTrainer(
        model=model,
        config=config,
        problem=problem,
        output_dir=str(output_dir),
        warmup_epochs=0,
    )
    result = trainer.train(n_epochs=args.epochs)

    result.history.to_dataframe().to_csv(output_dir / "training_history.csv", index=False)
    with (output_dir / "single_time_claim_report.json").open("w") as handle:
        json.dump(result.claim_report, handle, indent=2, sort_keys=True)
    with (output_dir / "single_time_problem_summary.json").open("w") as handle:
        json.dump(
            {
                "n_views": len(problem.views),
                "n_embeddings": len(problem.catalog.perturbation_ids),
                "control_ids": list(problem.catalog.control_ids),
                "view_key_level": args.view_key_level,
                "embedding_level": args.embedding_level,
                "effect_axis_is_physical_time": False,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
