#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from credo.data.hnscc import build_study_from_split  # noqa: E402
from credo.models.expression_vae import VAEArtifactBundle  # noqa: E402
from credo.models.simulator import _control_embedding_context, initialise_particles  # noqa: E402
from credo.models.weighted_sde import WeightedParticleSimulator  # noqa: E402

from hnscc_biology_common import infer_target_gene  # noqa: E402
from run_counterfactual_biology import (  # noqa: E402
    _build_model,
    _checkpoint_path,
    _load_obs_latent,
    _load_split,
    _program_centroids,
    _read_json,
    _resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a decoded CREDO perturbation module by comparing same-start "
            "factual and reference terminal rollouts, then decoding terminal VAE "
            "latents back to the VAE expression-gene panel."
        )
    )
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Trained with-guide fold directories.")
    parser.add_argument("--data-path", default=None, help="h5ad data path. Defaults to config.json data_path.")
    parser.add_argument("--perturbation", default="Lrp1", help="Perturbation id or target gene, e.g. Lrp1.")
    parser.add_argument("--source-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--n-particles", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=28)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-batch-size", type=int, default=4096)
    parser.add_argument("--min-concordance", type=float, default=0.75)
    parser.add_argument(
        "--min-folds",
        type=int,
        default=0,
        help="Minimum fold coverage for up/down module calls. Default 0 requires all supplied folds.",
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _resolve_perturbation_id(requested: str, candidates: list[str]) -> str:
    if requested in candidates:
        return requested
    by_lower = {pid.lower(): pid for pid in candidates}
    if requested.lower() in by_lower:
        return by_lower[requested.lower()]

    wanted_gene = infer_target_gene(requested).upper()
    matches = [pid for pid in candidates if infer_target_gene(pid).upper() == wanted_gene]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Could not find perturbation {requested!r}. "
            f"Available examples: {', '.join(candidates[:10])}"
        )
    raise ValueError(
        f"Perturbation {requested!r} matched multiple ids: {', '.join(matches)}. "
        "Pass an exact perturbation id."
    )


def _terminal_log_mass(log_m0: torch.Tensor, logw: torch.Tensor) -> float:
    return float((log_m0[0] + torch.logsumexp(logw, dim=0)).item())


@torch.no_grad()
def _decode_weighted_mean(
    z_std: torch.Tensor,
    logw: torch.Tensor,
    *,
    bundle: VAEArtifactBundle,
    vae: torch.nn.Module,
    device: str,
    batch_size: int,
) -> np.ndarray:
    z_np = z_std.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if bundle.latent_standardization is not None:
        z_np = bundle.latent_standardization.inverse(z_np).astype(np.float32, copy=False)

    weights = torch.softmax(logw.detach().float(), dim=0).cpu().numpy().astype(np.float32, copy=False)
    n_genes = len(bundle.gene_names)
    out = np.zeros(n_genes, dtype=np.float64)
    batch_size = max(int(batch_size), 1)
    for start in range(0, z_np.shape[0], batch_size):
        stop = min(start + batch_size, z_np.shape[0])
        z_batch = torch.from_numpy(z_np[start:stop]).to(device=device)
        decoded = vae.decode(z_batch).detach().float().cpu().numpy()
        out += weights[start:stop].astype(np.float64) @ decoded.astype(np.float64)
    return out.astype(np.float32)


def _run_one_fold(args: argparse.Namespace, run_dir: Path, device: str, fold_i: int) -> tuple[pd.DataFrame, dict]:
    config = _read_json(run_dir / "config.json")
    results = _read_json(run_dir / "results_summary.json")
    data_path = args.data_path or config.get("data_path")
    if not data_path:
        raise ValueError(f"Data path must be supplied or present in {run_dir / 'config.json'}")

    obs, latent = _load_obs_latent(run_dir, config, data_path)
    original_split = (
        _load_split(run_dir, obs, "train")
        if (run_dir / "split_assignments.csv").exists()
        else pd.Series("train", index=obs.index)
    )
    source_split = _load_split(run_dir, obs, args.source_split) if args.source_split != "all" else pd.Series(
        "analysis", index=obs.index, dtype="object"
    )
    split_name = "analysis" if args.source_split == "all" else args.source_split

    program_centroids = _program_centroids(config, obs, latent, original_split)
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    data = build_study_from_split(
        obs,
        latent,
        split=source_split,
        split_name=split_name,
        mass_value_col=config.get("mass_value_col"),
        mass_scope=config.get("mass_scope", "subset_only"),
        mass_mode=data_cfg.get("mass_mode") or config.get("mass_mode") or config.get("train_mass_mode", "auto"),
    )

    controls = set(config.get("control_ids", []))
    supported = [pid for pid in config["supported_perturbations"] if pid not in controls and pid in data.catalog.perturbation_ids]
    pid = _resolve_perturbation_id(args.perturbation, supported)
    endpoint = data.to_endpoint_problem([pid], initial_label="P4", terminal_label="P60")

    model = _build_model(config, latent.shape[1], program_centroids, device)
    checkpoint = _checkpoint_path(run_dir, results)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    vae_dir = run_dir / "vae_artifact"
    if not vae_dir.exists():
        raise FileNotFoundError(f"Missing VAE artifact directory: {vae_dir}")
    bundle, vae = VAEArtifactBundle.load(vae_dir, device=device)
    vae.to(device)
    vae.eval()

    simulator = WeightedParticleSimulator(n_steps=args.n_steps, store_history=False)
    seed = int(args.seed + fold_i)
    z0, logw0, log_m0 = initialise_particles(
        endpoint,
        [pid],
        n_particles=args.n_particles,
        device=device,
        seed=seed,
    )

    torch.manual_seed(seed)
    factual = simulator.rollout(z0, logw0, model, log_m0, perturbation_ids=[pid])
    with _control_embedding_context(model, pid, mode="reference_consistent"):
        torch.manual_seed(seed)
        reference = simulator.rollout(z0.clone(), logw0.clone(), model, log_m0.clone(), perturbation_ids=[pid])

    fact_expr = _decode_weighted_mean(
        factual.terminal_z[0],
        factual.terminal_logw[0],
        bundle=bundle,
        vae=vae,
        device=device,
        batch_size=args.decode_batch_size,
    )
    ref_expr = _decode_weighted_mean(
        reference.terminal_z[0],
        reference.terminal_logw[0],
        bundle=bundle,
        vae=vae,
        device=device,
        batch_size=args.decode_batch_size,
    )

    delta = fact_expr - ref_expr
    fold_id = f"fold_{config.get('split', {}).get('fold_index', fold_i)}"
    table = pd.DataFrame(
        {
            "gene": bundle.gene_names,
            "fold_id": fold_id,
            "perturbation_id": pid,
            "target_gene": infer_target_gene(pid),
            "decoded_factual_expr": fact_expr,
            "decoded_reference_expr": ref_expr,
            "delta_decoded_expr_fact_vs_ref": delta,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
        }
    )
    meta = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "fold_id": fold_id,
        "perturbation_id": pid,
        "target_gene": infer_target_gene(pid),
        "n_p4": int(endpoint.initial[pid].n_atoms),
        "n_p60": int(endpoint.terminal[pid].n_atoms),
        "n_particles": int(args.n_particles),
        "n_steps": int(args.n_steps),
        "seed": seed,
        "source_split": args.source_split,
        "log_mass_factual": _terminal_log_mass(factual.log_m0, factual.terminal_logw[0]),
        "log_mass_reference": _terminal_log_mass(reference.log_m0, reference.terminal_logw[0]),
        "vae_gene_count": len(bundle.gene_names),
    }
    meta["delta_log_mass_fact_vs_ref"] = meta["log_mass_factual"] - meta["log_mass_reference"]
    return table, meta


def _aggregate(per_fold: pd.DataFrame, min_concordance: float, min_folds: int) -> pd.DataFrame:
    grouped = per_fold.groupby(["gene", "perturbation_id", "target_gene"], dropna=False)
    # Count independent runs by run_dir, not fold_id: fold_id is the CV fold index
    # (fold_0..fold_3) and collides across seeds, so a 3-seed x 4-fold panel has only
    # 4 distinct fold_id values but 12 distinct run_dir values.
    fold_key = "run_dir" if "run_dir" in per_fold.columns else "fold_id"
    out = grouped.agg(
        n_folds=(fold_key, "nunique"),
        mean_delta_decoded_expr=("delta_decoded_expr_fact_vs_ref", "mean"),
        median_delta_decoded_expr=("delta_decoded_expr_fact_vs_ref", "median"),
        std_delta_decoded_expr=("delta_decoded_expr_fact_vs_ref", "std"),
        mean_factual_expr=("decoded_factual_expr", "mean"),
        mean_reference_expr=("decoded_reference_expr", "mean"),
    ).reset_index()

    signs = grouped["delta_decoded_expr_fact_vs_ref"].agg(
        pos_concordance=lambda x: float((x > 0).mean()),
        neg_concordance=lambda x: float((x < 0).mean()),
    ).reset_index()
    out = out.merge(signs, on=["gene", "perturbation_id", "target_gene"], how="left")
    out["abs_mean_delta_decoded_expr"] = out["mean_delta_decoded_expr"].abs()
    required_folds = int(min_folds) if int(min_folds) > 0 else int(out["n_folds"].max())
    out["min_folds_required"] = required_folds
    out["fold_coverage_pass"] = out["n_folds"] >= required_folds
    out["module_direction"] = "unstable"
    out.loc[
        out["fold_coverage_pass"]
        & (out["mean_delta_decoded_expr"] > 0)
        & (out["pos_concordance"] >= min_concordance),
        "module_direction",
    ] = "up"
    out.loc[
        out["fold_coverage_pass"]
        & (out["mean_delta_decoded_expr"] < 0)
        & (out["neg_concordance"] >= min_concordance),
        "module_direction",
    ] = "down"
    out = out.sort_values(
        ["module_direction", "abs_mean_delta_decoded_expr"],
        ascending=[True, False],
    )
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    per_fold_tables = []
    fold_meta = []
    for fold_i, raw_run_dir in enumerate(args.run_dirs):
        table, meta = _run_one_fold(args, Path(raw_run_dir), device, fold_i)
        per_fold_tables.append(table)
        fold_meta.append(meta)

    per_fold = pd.concat(per_fold_tables, ignore_index=True)
    agg = _aggregate(
        per_fold,
        min_concordance=float(args.min_concordance),
        min_folds=int(args.min_folds),
    )

    prefix = infer_target_gene(args.perturbation).lower()
    per_fold_path = out_dir / f"{prefix}_credo_decoded_module_by_fold.csv"
    agg_path = out_dir / f"{prefix}_credo_decoded_module_by_gene.csv"
    up_path = out_dir / f"{prefix}_credo_decoded_module_up_genes.txt"
    down_path = out_dir / f"{prefix}_credo_decoded_module_down_genes.txt"
    meta_path = out_dir / f"{prefix}_credo_decoded_module_metadata.json"

    per_fold.to_csv(per_fold_path, index=False)
    agg.to_csv(agg_path, index=False)
    top_n = max(int(args.top_n), 0)
    up = agg[agg["module_direction"].eq("up")].sort_values("mean_delta_decoded_expr", ascending=False)
    down = agg[agg["module_direction"].eq("down")].sort_values("mean_delta_decoded_expr", ascending=True)
    up.head(top_n)["gene"].to_csv(up_path, index=False, header=False)
    down.head(top_n)["gene"].to_csv(down_path, index=False, header=False)

    metadata = {
        "args": vars(args),
        "device": device,
        "n_run_dirs": len(args.run_dirs),
        "folds": fold_meta,
        "outputs": {
            "per_fold": str(per_fold_path),
            "by_gene": str(agg_path),
            "up_genes": str(up_path),
            "down_genes": str(down_path),
        },
        "notes": [
            "Decoded values are VAE reconstructions of library-normalized log1p expression.",
            "The module is factual terminal expression minus same-start reference terminal expression.",
            "The gene universe is the saved VAE expression-gene panel, not necessarily all genes.",
        ],
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(agg_path)
    print(up_path)
    print(down_path)


if __name__ == "__main__":
    main()
