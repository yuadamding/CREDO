"""Train CREDO on one-snapshot Perturb-seq as a non-physical effect axis."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
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


def _single_time_schema_column_map(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "control": args.control_col,
        "sample": args.sample_col or None,
        "batch": args.batch_col or None,
        "guide": args.guide_col or None,
        "perturbation": args.perturbation_col,
        "target_gene": args.target_gene_col or None,
    }


def _single_time_extra_schema_columns(args: argparse.Namespace) -> list[str]:
    columns = [args.control_col]
    if args.view_key_level in {"guide", "sample_guide"} and args.guide_col:
        columns.append(args.guide_col)
    elif args.perturbation_col:
        columns.append(args.perturbation_col)
    if args.embedding_level == "target_gene" and args.target_gene_col:
        columns.append(args.target_gene_col)
    return list(dict.fromkeys(column for column in columns if column))


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


def _weight_diagnostics(logw: torch.Tensor) -> dict[str, float]:
    weights = torch.softmax(logw, dim=0)
    n_particles = max(1, int(weights.numel()))
    ess = 1.0 / weights.square().sum().clamp_min(1e-30)
    return {
        "terminal_ess_frac": float((ess / n_particles).item()),
        "max_weight_frac": float(weights.max().item()),
        "logw_range": float((logw.max() - logw.min()).item()),
    }


def _vector_rows(
    *,
    base: dict[str, object],
    values: torch.Tensor,
    value_column: str,
) -> list[dict[str, object]]:
    return [
        {
            **base,
            "latent_dim": int(idx),
            value_column: float(value.item()),
        }
        for idx, value in enumerate(values)
    ]


def _control_null_summary(control_null: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "n_controls",
        "mean",
        "std",
        "abs_p95",
    ]
    if control_null.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for metric in [
        "diagnostic_delta_log_mass",
        "latent_mean_shift_norm",
        "latent_variance_shift_norm",
    ]:
        values = pd.to_numeric(control_null.get(metric), errors="coerce").dropna()
        if values.empty:
            rows.append({"metric": metric, "n_controls": 0, "mean": pd.NA, "std": pd.NA, "abs_p95": pd.NA})
        else:
            rows.append(
                {
                    "metric": metric,
                    "n_controls": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std()) if values.shape[0] > 1 else pd.NA,
                    "abs_p95": float(values.abs().quantile(0.95)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _git_sha() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _write_single_time_provenance(
    *,
    output_dir: Path,
    config: RunConfig,
    argv: list[str] | None,
) -> None:
    command_parts = ["python", "runners/run_credo_single_time.py", *(argv if argv is not None else sys.argv[1:])]
    (output_dir / "single_time_command.txt").write_text(
        " ".join(shlex.quote(str(part)) for part in command_parts) + "\n",
        encoding="utf-8",
    )
    (output_dir / "single_time_git_sha.txt").write_text(_git_sha() + "\n", encoding="utf-8")
    with (output_dir / "single_time_resolved_config.json").open("w") as handle:
        json.dump(config.model_dump(mode="json"), handle, indent=2, sort_keys=True)


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
        context_sampling=config.single_time.context_sampling,
        context_gradient_mode=config.single_time.context_gradient_mode,
    )
    loss_fn = EndpointGeometryMassLoss(
        eps=config.training.sinkhorn_epsilon,
        tau=config.training.sinkhorn_tau,
        max_iter=config.training.sinkhorn_max_iter,
    )

    effect_rows = []
    endpoint_rows = []
    mean_shift_rows = []
    variance_shift_rows = []
    report_view_level = "view"
    training_view_level = config.single_time.view_level
    report_is_posthoc = training_view_level != report_view_level
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
        weight_diag = _weight_diagnostics(fact_logw)

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
            "sample_id": endpoint.metadata.get("measure_to_sample_id", {}).get(pid),
            "batch_id": endpoint.metadata.get("measure_to_batch_id", {}).get(pid),
            "embedding_id": metadata.get("target_embedding_id"),
            "is_control": pid in set(endpoint.metadata.get("control_measure_keys", [])),
            "claim_level": endpoint.metadata.get("claim_level"),
            "mass_claim_grade": endpoint.metadata.get("abundance_claim_grade"),
            "abundance_claim_grade": endpoint.metadata.get("abundance_claim_grade"),
            "abundance_claimable": endpoint.metadata.get("abundance_claim_grade") == "claim_grade",
            "training_view_level": training_view_level,
            "report_view_level": report_view_level,
            "report_is_posthoc_view_level": report_is_posthoc,
            "context_protocol": config.single_time.context_protocol,
            "context_gradient_mode": config.single_time.context_gradient_mode,
            "context_sampling": config.single_time.context_sampling,
            "training_context_sampling": config.single_time.context_sampling,
            "training_context_gradient_mode": config.single_time.context_gradient_mode,
            "report_context_sampling": metadata.get("context_sampling"),
            "report_context_gradient_mode": metadata.get("context_gradient_mode"),
            "effect_axis_is_physical_time": False,
        }
        delta_log_mass = float((fact_log_mass - ref_log_mass).item())
        delta_mass = float((fact_log_mass.exp() - ref_log_mass.exp()).item())
        mass_claimable = base["abundance_claimable"]
        effect_rows.append(
            {
                **base,
                "diagnostic_delta_log_mass": delta_log_mass,
                "diagnostic_delta_mass": delta_mass,
                "delta_log_mass": delta_log_mass,
                "delta_mass": delta_mass,
                "abundance_delta_log_mass_claimable": delta_log_mass if mass_claimable else pd.NA,
                "abundance_delta_mass_claimable": delta_mass if mass_claimable else pd.NA,
                "latent_mean_shift_norm": float(mean_diff.norm().item()),
                "latent_variance_shift_norm": float(var_diff.norm().item()),
                "terminal_factual_log_mass": float(fact_log_mass.item()),
                "terminal_reference_log_mass": float(ref_log_mass.item()),
                **weight_diag,
                "same_reference_source": metadata.get("same_reference_source"),
                "same_noise": metadata.get("same_noise"),
            }
        )
        mean_shift_rows.extend(_vector_rows(base=base, values=mean_diff, value_column="latent_mean_shift"))
        variance_shift_rows.extend(
            _vector_rows(base=base, values=var_diff, value_column="latent_variance_shift")
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
        "sample_id",
        "batch_id",
        "embedding_id",
        "is_control",
        "claim_level",
        "mass_claim_grade",
        "abundance_claim_grade",
        "abundance_claimable",
        "training_view_level",
        "report_view_level",
        "report_is_posthoc_view_level",
        "context_protocol",
        "context_sampling",
        "context_gradient_mode",
        "training_context_sampling",
        "training_context_gradient_mode",
        "report_context_sampling",
        "report_context_gradient_mode",
        "effect_axis_is_physical_time",
    ]
    effect_columns = base_columns + [
        "diagnostic_delta_log_mass",
        "diagnostic_delta_mass",
        "delta_log_mass",
        "delta_mass",
        "abundance_delta_log_mass_claimable",
        "abundance_delta_mass_claimable",
        "latent_mean_shift_norm",
        "latent_variance_shift_norm",
        "terminal_factual_log_mass",
        "terminal_reference_log_mass",
        "terminal_ess_frac",
        "max_weight_frac",
        "logw_range",
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
    mean_shifts = pd.DataFrame(
        mean_shift_rows,
        columns=base_columns + ["latent_dim", "latent_mean_shift"],
    )
    variance_shifts = pd.DataFrame(
        variance_shift_rows,
        columns=base_columns + ["latent_dim", "latent_variance_shift"],
    )
    effects.to_csv(output_dir / "single_time_effects.csv", index=False)
    endpoints.to_csv(output_dir / "single_time_endpoint_metrics.csv", index=False)
    mean_shifts.to_csv(output_dir / "single_time_latent_mean_shift_by_dim.csv", index=False)
    variance_shifts.to_csv(output_dir / "single_time_latent_variance_shift_by_dim.csv", index=False)

    if effects.empty:
        control_null = pd.DataFrame(columns=effects.columns)
        guide_summary = pd.DataFrame(
            columns=[
                "target_gene",
                "n_views",
                "n_guides",
                "n_samples",
                "guide_concordance_evaluable",
                "delta_log_mass_std",
                "latent_mean_shift_norm_std",
                "latent_variance_shift_norm_std",
            ]
        )
    else:
        control_null = effects.loc[effects["is_control"]].copy()
        guide = effects.loc[~effects["is_control"]].copy()
        guide = guide[guide["target_gene"].notna()]
        if not guide.empty:
            guide_summary = (
                guide.groupby("target_gene", dropna=False)
                .agg(
                    n_views=("view_id", "count"),
                    n_guides=("guide_id", "nunique"),
                    n_samples=("sample_id", "nunique"),
                    delta_log_mass_std=("diagnostic_delta_log_mass", "std"),
                    latent_mean_shift_norm_std=("latent_mean_shift_norm", "std"),
                    latent_variance_shift_norm_std=("latent_variance_shift_norm", "std"),
                )
                .reset_index()
            )
            guide_summary["guide_concordance_evaluable"] = (
                (guide_summary["n_views"] >= 2)
                & ((guide_summary["n_guides"] >= 2) | (guide_summary["n_samples"] >= 2))
            )
        else:
            guide_summary = pd.DataFrame(
                columns=[
                    "target_gene",
                    "n_views",
                    "n_guides",
                    "n_samples",
                    "guide_concordance_evaluable",
                    "delta_log_mass_std",
                    "latent_mean_shift_norm_std",
                    "latent_variance_shift_norm_std",
                ]
            )
    control_null.to_csv(output_dir / "single_time_control_null.csv", index=False)
    _control_null_summary(control_null).to_csv(
        output_dir / "single_time_control_null_summary.csv",
        index=False,
    )
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
            obs_columns=_single_time_extra_schema_columns(args),
            column_map=_single_time_schema_column_map(args),
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
    _write_single_time_provenance(output_dir=output_dir, config=config, argv=argv)
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
