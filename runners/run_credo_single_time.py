"""Train CREDO on one-snapshot Perturb-seq as a non-physical effect axis."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent / "package"
sys.path.insert(0, str(ROOT / "src"))

from credo.config.schema import RunConfig
from credo.data import build_single_time_problem_from_anndata, validate_anndata_schema
from credo.losses import EndpointGeometryMassLoss
from credo.models import FullDynamicsModel, SingleTimeCounterfactualEngine, WeightedParticleSimulator
from credo.training import SingleTimeTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-key", default="X_pca")
    parser.add_argument("--strict-data-schema", action="store_true")
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
    parser.add_argument(
        "--effect-vector-components",
        default="delta_log_mass,latent_mean_shift",
        help=(
            "Comma-separated single-time effect components for control-null and "
            "guide-concordance regularizers. Supported: delta_log_mass, "
            "latent_mean_shift, latent_variance_shift."
        ),
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


def _parse_effect_vector_components(value: str) -> tuple[str, ...]:
    allowed = {"delta_log_mass", "latent_mean_shift", "latent_variance_shift"}
    components = tuple(part.strip() for part in value.split(",") if part.strip())
    if not components:
        raise ValueError("--effect-vector-components must include at least one component.")
    unknown = sorted(set(components) - allowed)
    if unknown:
        raise ValueError(f"Unsupported --effect-vector-components value(s): {unknown}.")
    if len(set(components)) != len(components):
        raise ValueError("--effect-vector-components contains duplicates.")
    return components


def _weighted_mean(z: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logw, dim=0)
    return (weights.unsqueeze(-1) * z).sum(dim=0)


def _weighted_variance(z: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logw, dim=0)
    mean = (weights.unsqueeze(-1) * z).sum(dim=0)
    return (weights.unsqueeze(-1) * (z - mean).square()).sum(dim=0)


def _log_mass(log_m0: torch.Tensor | None, logw: torch.Tensor) -> torch.Tensor:
    if log_m0 is None:
        raise ValueError("single-time effect reporting requires rollout.log_m0.")
    return log_m0.squeeze(0).to(logw.device, dtype=logw.dtype) + torch.logsumexp(logw.squeeze(0), dim=0)


def _measure_mean_and_var(measure) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support = torch.as_tensor(measure.support, dtype=torch.float32)
    weights = torch.as_tensor(measure.weights, dtype=torch.float32)
    logw = torch.log(weights.clamp_min(1e-30))
    mean = _weighted_mean(support, logw)
    var = _weighted_variance(support, logw)
    return mean, var, logw


def _write_single_time_effect_outputs(
    *,
    output_dir: Path,
    problem,
    model: FullDynamicsModel,
    config: RunConfig,
    context_tau: str | float,
) -> None:
    endpoint = problem.to_effect_endpoint_problem(view_level="view")
    simulator = WeightedParticleSimulator(
        n_steps=config.simulation.n_steps,
        store_history=True,
    )
    engine = SingleTimeCounterfactualEngine(
        model=model,
        simulator=simulator,
        n_particles=config.simulation.n_particles,
        device=config.resolve_device(),
    )
    model.eval()
    results = engine.run(
        problem,
        perturbation_ids=list(endpoint.perturbation_ids),
        seed=config.training.seed,
        common_noise=True,
        context_protocol=config.single_time.context_protocol,
        context_tau=context_tau,
    )
    loss_fn = EndpointGeometryMassLoss(
        eps=config.training.sinkhorn_epsilon,
        tau=config.training.sinkhorn_tau,
        max_iter=config.training.sinkhorn_max_iter,
    )

    effect_rows = []
    endpoint_rows = []
    for result in results:
        pid = result.perturbation_id
        metadata = result.metadata
        target = endpoint.terminal[pid]
        factual = result.rollout_perturb
        reference = result.rollout_control

        fact_z = factual.terminal_z.squeeze(0).detach().float().cpu()
        fact_logw = factual.terminal_logw.squeeze(0).detach().float().cpu()
        ref_z = reference.terminal_z.squeeze(0).detach().float().cpu()
        ref_logw = reference.terminal_logw.squeeze(0).detach().float().cpu()
        fact_log_mass = _log_mass(factual.log_m0, factual.terminal_logw).detach().float().cpu()
        ref_log_mass = _log_mass(reference.log_m0, reference.terminal_logw).detach().float().cpu()
        fact_mean = _weighted_mean(fact_z, fact_logw)
        ref_mean = _weighted_mean(ref_z, ref_logw)
        fact_var = _weighted_variance(fact_z, fact_logw)
        ref_var = _weighted_variance(ref_z, ref_logw)
        mean_diff = fact_mean - ref_mean
        var_diff = fact_var - ref_var

        target_mean, target_var, target_logw = _measure_mean_and_var(target)
        pred_logw_abs = fact_logw + factual.log_m0.squeeze(0).detach().float().cpu()
        components = loss_fn.component_dict(
            pred_z=fact_z.unsqueeze(0),
            pred_logw_abs=pred_logw_abs.unsqueeze(0),
            target_support={pid: torch.as_tensor(target.support, dtype=torch.float32)},
            target_logw={pid: target_logw},
            perturbation_ids=[pid],
        )[1][pid]

        base = {
            "view_id": pid,
            "original_perturbation_id": metadata.get("target_perturbation_id", pid),
            "guide_id": endpoint.metadata.get("measure_to_guide", {}).get(pid),
            "target_gene": endpoint.metadata.get("measure_to_target_gene", {}).get(pid),
            "embedding_id": metadata.get("target_embedding_id"),
            "is_control": pid in set(endpoint.metadata.get("control_measure_keys", [])),
            "claim_level": endpoint.metadata.get("claim_level"),
            "mass_claim_grade": endpoint.metadata.get("abundance_claim_grade"),
            "context_protocol": config.single_time.context_protocol,
            "context_gradient_mode": config.single_time.context_gradient_mode,
            "effect_axis_is_physical_time": False,
        }
        effect_rows.append(
            {
                **base,
                "delta_log_mass": float((fact_log_mass - ref_log_mass).item()),
                "delta_mass": float((fact_log_mass.exp() - ref_log_mass.exp()).item()),
                "latent_mean_shift_norm": float(mean_diff.norm().item()),
                "latent_variance_shift_norm": float(var_diff.norm().item()),
                "terminal_factual_log_mass": float(fact_log_mass.item()),
                "terminal_reference_log_mass": float(ref_log_mass.item()),
                "same_reference_source": metadata.get("same_reference_source"),
                "same_noise": metadata.get("same_noise"),
            }
        )
        endpoint_rows.append(
            {
                **base,
                "endpoint_geom_mass": float(components["total"].detach().cpu().item()),
                "endpoint_sinkhorn": float(components["geom"].detach().cpu().item()),
                "endpoint_mass_penalty": float(components["mass"].detach().cpu().item()),
                "mass_error": float(
                    (
                        components["log_mass_pred"] - components["log_mass_target"]
                    ).detach().cpu().item()
                ),
                "target_log_mass": float(components["log_mass_target"].detach().cpu().item()),
                "pred_log_mass": float(components["log_mass_pred"].detach().cpu().item()),
                "target_mean_shift_norm": float((fact_mean - target_mean).norm().item()),
                "target_variance_shift_norm": float((fact_var - target_var).norm().item()),
            }
        )

    base_columns = [
        "view_id",
        "original_perturbation_id",
        "guide_id",
        "target_gene",
        "embedding_id",
        "is_control",
        "claim_level",
        "mass_claim_grade",
        "context_protocol",
        "context_gradient_mode",
        "effect_axis_is_physical_time",
    ]
    effect_columns = base_columns + [
        "delta_log_mass",
        "delta_mass",
        "latent_mean_shift_norm",
        "latent_variance_shift_norm",
        "terminal_factual_log_mass",
        "terminal_reference_log_mass",
        "same_reference_source",
        "same_noise",
    ]
    endpoint_columns = base_columns + [
        "endpoint_geom_mass",
        "endpoint_sinkhorn",
        "endpoint_mass_penalty",
        "mass_error",
        "target_log_mass",
        "pred_log_mass",
        "target_mean_shift_norm",
        "target_variance_shift_norm",
    ]
    effects = pd.DataFrame(effect_rows, columns=effect_columns)
    endpoints = pd.DataFrame(endpoint_rows, columns=endpoint_columns)
    effects.to_csv(output_dir / "single_time_effects.csv", index=False)
    endpoints.to_csv(output_dir / "single_time_endpoint_metrics.csv", index=False)

    if effects.empty:
        control_null = pd.DataFrame(columns=effects.columns)
        guide_summary = pd.DataFrame(
            columns=[
                "target_gene",
                "n_views",
                "delta_log_mass_std",
                "latent_mean_shift_norm_std",
                "latent_variance_shift_norm_std",
            ]
        )
    else:
        control_null = effects.loc[effects["is_control"]].copy()
        guide = effects.loc[~effects["is_control"]].copy()
        guide = guide[guide["target_gene"].notna()]
        guide_summary = (
            guide.groupby("target_gene", dropna=False)
            .agg(
                n_views=("view_id", "count"),
                delta_log_mass_std=("delta_log_mass", "std"),
                latent_mean_shift_norm_std=("latent_mean_shift_norm", "std"),
                latent_variance_shift_norm_std=("latent_variance_shift_norm", "std"),
            )
            .reset_index()
            .fillna(0.0)
            if not guide.empty
            else pd.DataFrame(
                columns=[
                    "target_gene",
                    "n_views",
                    "delta_log_mass_std",
                    "latent_mean_shift_norm_std",
                    "latent_variance_shift_norm_std",
                ]
            )
        )
    control_null.to_csv(output_dir / "single_time_control_null.csv", index=False)
    guide_summary.to_csv(output_dir / "single_time_guide_concordance.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_tau = _parse_context_tau(args.context_tau)
    effect_vector_components = _parse_effect_vector_components(args.effect_vector_components)
    if args.view_key_level in {"guide", "sample_guide"} and args.view_level == "embedding":
        warnings.warn(
            "Guide-level finite-measure views will be pooled by embedding because "
            "--view-level=embedding. Use --view-level=view for guide-level effect "
            "and concordance outputs.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.context_gradient_mode == "detached_cache":
        warnings.warn(
            "context_gradient_mode='detached_cache' freezes model-derived context "
            "features. Prefer recompute_no_grad for trainable-context biology.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.strict_data_schema:
        schema_report = validate_anndata_schema(
            args.data_path,
            schema="single_time",
            latent_key=args.latent_key,
            strict=True,
        )
        if not bool(schema_report.get("ok")):
            errors = schema_report.get("errors") or [schema_report.get("error", "unknown schema error")]
            raise ValueError("single-time AnnData schema validation failed: " + "; ".join(map(str, errors)))

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
            "effect_vector_components": effect_vector_components,
            "context_protocol": args.context_protocol,
            "context_sampling": args.context_sampling,
            "context_gradient_mode": args.context_gradient_mode,
            "context_tau": context_tau,
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
                "effect_vector_components": list(effect_vector_components),
                "effect_axis_is_physical_time": False,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    _write_single_time_effect_outputs(
        output_dir=output_dir,
        problem=problem,
        model=model,
        config=config,
        context_tau=context_tau,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
