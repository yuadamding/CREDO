"""
Train and evaluate CAPE on the HNSCC P4/P60 perturb-seq dataset.

This uses CAPE's full perturb-seq structure:
  - FullDynamicsModel
  - WeightedParticleSimulator
  - endpoint UOT loss + weak-form loss
  - ecological growth enabled by default

Run with:
  conda run -n ml1 python run_cape_hnscc_full.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cape.config.schema import LatentConfig, ModelConfig, RunConfig, SimulationConfig, TrainingConfig
from cape.data.core import (
    CellStateTable,
    EndpointProblem,
    FiniteMeasure,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
)
from cape.data.filters import filter_state_supported_perturbations
from cape.losses.uot import sinkhorn_divergence
from cape.models.full_model import FullDynamicsModel
from cape.models.simulator import initialise_particles
from cape.models.weighted_sde import WeightedParticleSimulator
from cape.training.trainer import Trainer


DEFAULT_DATA = (
    "/home/yding1995/opscc_sc/scDiffeq/hnscc/"
    "GSE235325_P4P60_scdiffeq_compatible.h5ad"
)
DEFAULT_OUTPUT = "/home/yding1995/opscc_sc/CAPE/outputs/hnscc_cape_full"
DEFAULT_WTA_COLUMN = "Library"
DEFAULT_TRAIN_WTAS = [
    "wta4",
    "wta5",
    "wta6",
    "wta7",
    "wta9",
    "wta13",
    "wta14",
    "wta15",
    "wta16",
    "wta17",
    "wta18",
]
DEFAULT_TEST_WTAS = ["wta8", "wta10", "wta11", "wta12"]
P4 = "P4"
P60 = "P60"
TIME_MAP = {4.0: P4, 60.0: P60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full CAPE training on HNSCC perturb-seq data.")
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wta-column", default=DEFAULT_WTA_COLUMN)
    parser.add_argument("--train-wtas", default=",".join(DEFAULT_TRAIN_WTAS))
    parser.add_argument("--test-wtas", default=",".join(DEFAULT_TEST_WTAS))
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--mediator-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-particles", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--eval-particles", type=int, default=384)
    parser.add_argument("--eval-steps", type=int, default=24)
    parser.add_argument("--eval-target-particles", type=int, default=768)
    parser.add_argument("--max-train-target-atoms", type=int, default=768)
    parser.add_argument("--n-test-functions", type=int, default=12)
    parser.add_argument("--lambda-weak", type=float, default=0.1)
    parser.add_argument("--min-cells-p4", type=int, default=20)
    parser.add_argument("--min-cells-p60", type=int, default=20)
    parser.add_argument("--guide-confident-only", dest="guide_confident_only", action="store_true")
    parser.add_argument("--include-nonconfident", dest="guide_confident_only", action="store_false")
    parser.set_defaults(guide_confident_only=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _clean_perturbation_ids(obs: pd.DataFrame) -> pd.Series:
    if "perturbation_gene" in obs.columns:
        pid = obs["perturbation_gene"].astype(str).copy()
    else:
        pid = obs["target_gene"].astype(str).copy()
    pid = pid.replace({"": "ctrl", "nan": "ctrl", "None": "ctrl"})
    pid.loc[obs["is_control"].astype(bool).to_numpy()] = "ctrl"
    return pid


def _time_labels(obs: pd.DataFrame) -> pd.Series:
    vals = pd.to_numeric(obs["Time point"], errors="coerce")
    labels = vals.map(TIME_MAP)
    if labels.isna().any():
        missing = sorted(vals[labels.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected time points in dataset: {missing}")
    return labels


def load_hnscc(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    latent = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return obs, latent


def parse_wta_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_wta_split(obs: pd.DataFrame, wta_column: str, train_wtas: list[str], test_wtas: list[str]) -> dict:
    available = sorted(obs[wta_column].astype(str).unique().tolist())
    train_set = set(train_wtas)
    test_set = set(test_wtas)
    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(f"Train/test WTA overlap detected: {overlap}")
    missing = sorted((train_set | test_set) - set(available))
    if missing:
        raise ValueError(f"Unknown WTAs in requested split: {missing}")
    uncovered = sorted(set(available) - train_set - test_set)
    if uncovered:
        raise ValueError(f"Some WTAs are not assigned to train or test: {uncovered}")
    return {
        "available_wtas": available,
        "train_wtas": sorted(train_set),
        "test_wtas": sorted(test_set),
    }


def build_study(
    obs: pd.DataFrame,
    latent: np.ndarray,
    split_name: str,
    wta_column: str,
    selected_wtas: list[str],
    guide_confident_only: bool,
) -> PerturbSeqDynamicsData:
    mask = obs[wta_column].astype(str).isin(selected_wtas).to_numpy()
    if guide_confident_only and "guide_confident" in obs.columns:
        mask &= obs["guide_confident"].fillna(False).to_numpy(dtype=bool)

    sub_obs = obs.loc[mask].copy()
    sub_latent = latent[mask]
    if len(sub_obs) == 0:
        raise ValueError(f"No cells left for split={split_name!r}")

    sub_obs["perturbation_id"] = _clean_perturbation_ids(sub_obs)
    sub_obs["time_label"] = _time_labels(sub_obs)
    sub_obs["sample_id"] = (
        sub_obs["Library"].astype(str).replace({"": "pooled", "nan": "pooled", "None": "pooled"})
    )
    sub_obs["cell_id"] = sub_obs["cell_id"].astype(str)

    cell_df = sub_obs[["cell_id", "perturbation_id", "time_label", "sample_id"]].copy()
    mass_df = (
        cell_df.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
        .size()
        .rename("mass")
        .reset_index()
    )

    perturbation_ids = sorted(cell_df["perturbation_id"].unique().tolist())
    control_ids = sorted(sub_obs.loc[sub_obs["is_control"].astype(bool), "perturbation_id"].unique().tolist())
    if not control_ids:
        raise ValueError("No control perturbations found after filtering.")

    return PerturbSeqDynamicsData(
        time_axis=TimeAxis.p4_p60(),
        catalog=PerturbationCatalog(perturbation_ids=perturbation_ids, control_ids=control_ids),
        cell_state=CellStateTable(df=cell_df.reset_index(drop=True), latent=sub_latent),
        mass_table=MassTable(df=mass_df),
    )


def supported_intersection(
    train_data: PerturbSeqDynamicsData,
    test_data: PerturbSeqDynamicsData,
    min_cells_p4: int,
    min_cells_p60: int,
) -> list[str]:
    train_supported = set(
        filter_state_supported_perturbations(
            train_data, min_cells_p4=min_cells_p4, min_cells_p60=min_cells_p60
        )
    )
    test_supported = set(
        filter_state_supported_perturbations(
            test_data, min_cells_p4=min_cells_p4, min_cells_p60=min_cells_p60
        )
    )
    control_ids = set(train_data.catalog.control_ids) & set(test_data.catalog.control_ids)
    supported = sorted((train_supported & test_supported) | control_ids)
    if not supported:
        raise ValueError("No perturbations have sufficient support in both train and test.")
    return supported


def _cap_measure_atoms(
    measure: FiniteMeasure,
    *,
    max_atoms: int | None,
    seed: int,
) -> FiniteMeasure:
    if max_atoms is None or measure.n_atoms <= max_atoms:
        return measure
    rng = np.random.default_rng(seed)
    idx = rng.choice(measure.n_atoms, size=max_atoms, replace=False)
    support = measure.support[idx]
    weights = np.full(max_atoms, measure.total_mass / max_atoms, dtype=np.float32)
    return FiniteMeasure(support=support, weights=weights, total_mass=measure.total_mass)


def cap_endpoint_problem_terminal(
    endpoint: EndpointProblem,
    *,
    max_terminal_atoms: int | None,
    seed: int,
) -> EndpointProblem:
    if max_terminal_atoms is None:
        return endpoint
    initial = endpoint.initial
    terminal = {}
    for i, pid in enumerate(endpoint.perturbation_ids):
        terminal[pid] = _cap_measure_atoms(
            endpoint.terminal[pid],
            max_atoms=max_terminal_atoms,
            seed=seed + i,
        )
    return EndpointProblem(
        initial=initial,
        terminal=terminal,
        time_axis=endpoint.time_axis,
        perturbation_ids=endpoint.perturbation_ids,
    )


@torch.no_grad()
def evaluate_endpoint_problem(
    model: FullDynamicsModel,
    endpoint,
    perturbation_ids: list[str],
    control_ids: set[str],
    *,
    device: str,
    n_particles: int,
    n_steps: int,
    target_particles: int,
    seed: int,
    eps: float,
    tau: float,
) -> pd.DataFrame:
    simulator = WeightedParticleSimulator(n_steps=n_steps, store_history=False)
    dtype = torch.float32
    model.eval()

    z0, logw0, log_m0 = initialise_particles(
        endpoint, perturbation_ids, n_particles=n_particles, device=device, dtype=dtype, seed=seed
    )
    rollout = simulator.rollout(z0, logw0, model, log_m0, perturbation_ids=perturbation_ids)

    rows = []
    rng = np.random.default_rng(seed)
    for g, pid in enumerate(perturbation_ids):
        mu = endpoint.terminal[pid]
        if len(mu.support) > target_particles:
            idx = rng.choice(len(mu.support), size=target_particles, replace=False)
            target_support = mu.support[idx]
            target_weights = np.full(len(idx), mu.total_mass / len(idx), dtype=np.float32)
        else:
            target_support = mu.support
            target_weights = mu.weights

        y = torch.tensor(target_support, dtype=dtype, device=device)
        lb = torch.log(torch.tensor(target_weights, dtype=dtype, device=device) + 1e-30)

        la_abs = rollout.terminal_logw[g] + log_m0[g]
        div = sinkhorn_divergence(rollout.terminal_z[g], la_abs, y, lb, eps=eps, tau=tau)

        log_pred = log_m0[g] + torch.logsumexp(rollout.terminal_logw[g], dim=0)
        mass_pred = float(log_pred.exp().item())
        mass_true = float(mu.total_mass)
        mass_err = abs(mass_pred - mass_true) / mass_true if mass_true > 0 else 0.0

        rows.append(
            {
                "perturbation_id": pid,
                "uot": float(div.item()),
                "mass_pred": mass_pred,
                "mass_true": mass_true,
                "mass_rel_error": mass_err,
                "is_control": pid in control_ids,
                "n_init_atoms": int(endpoint.initial[pid].n_atoms),
                "n_term_atoms": int(endpoint.terminal[pid].n_atoms),
                "n_term_atoms_eval": int(len(target_support)),
            }
        )
    return pd.DataFrame(rows)


def summarize_eval(df: pd.DataFrame) -> dict:
    summary = {
        "n_perturbations": int(len(df)),
        "mean_uot": float(df["uot"].mean()),
        "median_uot": float(df["uot"].median()),
        "mean_mass_rel_error": float(df["mass_rel_error"].mean()),
        "median_mass_rel_error": float(df["mass_rel_error"].median()),
    }
    if "is_control" in df.columns and df["is_control"].any():
        ctrl = df[df["is_control"]]
        summary["n_controls"] = int(len(ctrl))
        summary["control_mean_uot"] = float(ctrl["uot"].mean())
        summary["control_mean_mass_rel_error"] = float(ctrl["mass_rel_error"].mean())
    non_ctrl = df[~df["is_control"]]
    if len(non_ctrl) > 0:
        summary["n_non_controls"] = int(len(non_ctrl))
        summary["non_control_mean_uot"] = float(non_ctrl["uot"].mean())
        summary["non_control_mean_mass_rel_error"] = float(non_ctrl["mass_rel_error"].mean())
    return summary


def save_text(path: Path, text: str) -> None:
    path.write_text(text)


def peak_gpu_stats_mb() -> dict | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_torch()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HNSCC data from {args.data_path}")
    obs, latent = load_hnscc(args.data_path)
    print(f"Loaded obs={obs.shape} latent={latent.shape}")

    train_wtas = parse_wta_list(args.train_wtas)
    test_wtas = parse_wta_list(args.test_wtas)
    split_meta = validate_wta_split(obs, args.wta_column, train_wtas, test_wtas)

    train_data = build_study(
        obs,
        latent,
        split_name="train",
        wta_column=args.wta_column,
        selected_wtas=train_wtas,
        guide_confident_only=args.guide_confident_only,
    )
    test_data = build_study(
        obs,
        latent,
        split_name="test",
        wta_column=args.wta_column,
        selected_wtas=test_wtas,
        guide_confident_only=args.guide_confident_only,
    )
    supported_pids = supported_intersection(
        train_data, test_data, min_cells_p4=args.min_cells_p4, min_cells_p60=args.min_cells_p60
    )
    control_ids = [pid for pid in train_data.catalog.control_ids if pid in supported_pids]
    if not control_ids:
        raise ValueError("No supported control perturbations remain after intersection.")

    print(
        f"Train cells={train_data.cell_state.n_cells} | Test cells={test_data.cell_state.n_cells} | "
        f"Supported perturbations={len(supported_pids)} | Controls={control_ids}"
    )

    train_ep_full = train_data.to_endpoint_problem(perturbation_ids=supported_pids, initial_label=P4, terminal_label=P60)
    test_ep = test_data.to_endpoint_problem(perturbation_ids=supported_pids, initial_label=P4, terminal_label=P60)
    train_ep = cap_endpoint_problem_terminal(
        train_ep_full,
        max_terminal_atoms=args.max_train_target_atoms,
        seed=args.seed,
    )

    latent_dim = train_data.latent_dim
    cfg = RunConfig(
        device="auto",
        output_dir=str(output_dir),
        latent=LatentConfig(dim=latent_dim, whiten=False),
        model=ModelConfig(
            embedding_dim=args.embedding_dim,
            n_programs=args.n_programs,
            mediator_dim=args.mediator_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            ecological_growth=True,
        ),
        simulation=SimulationConfig(
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            store_history=True,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            lr_net=3e-4,
            lr_embed=1e-3,
            lambda_end=1.0,
            lambda_weak=args.lambda_weak,
            lambda_count=0.0,
            lambda_reg_embed=1e-4,
            lambda_reg_net=1e-4,
            lambda_reg_diffusion=1e-4,
            seed=args.seed,
            early_stop_patience=args.epochs,
            log_every=25,
            checkpoint_every=100,
            sinkhorn_epsilon=0.1,
            sinkhorn_tau=1.0,
            n_test_functions=args.n_test_functions,
            test_function_bandwidth=1.0,
        ),
    )

    model = FullDynamicsModel(
        perturbation_ids=supported_pids,
        control_ids=control_ids,
        latent_dim=latent_dim,
        embedding_dim=args.embedding_dim,
        n_programs=args.n_programs,
        mediator_dim=args.mediator_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        ecological_growth=True,
    ).to(cfg.resolve_device())

    trainer = Trainer(model, cfg, train_ep, supported_pids, output_dir=str(output_dir))

    meta = {
        "data_path": args.data_path,
        "guide_confident_only": args.guide_confident_only,
        "wta_column": args.wta_column,
        "train_wtas": split_meta["train_wtas"],
        "test_wtas": split_meta["test_wtas"],
        "available_wtas": split_meta["available_wtas"],
        "n_particles": args.n_particles,
        "n_steps": args.n_steps,
        "eval_particles": args.eval_particles,
        "eval_steps": args.eval_steps,
        "eval_target_particles": args.eval_target_particles,
        "n_test_functions": args.n_test_functions,
        "lambda_weak": args.lambda_weak,
        "max_train_target_atoms": args.max_train_target_atoms,
        "train_cells": int(train_data.cell_state.n_cells),
        "test_cells": int(test_data.cell_state.n_cells),
        "supported_perturbations": supported_pids,
        "control_ids": control_ids,
        "config": cfg.model_dump(),
    }
    (output_dir / "config.json").write_text(json.dumps(meta, indent=2))
    save_text(output_dir / "supported_perturbations.txt", "\n".join(supported_pids) + "\n")
    save_text(
        output_dir / "wta_split_manifest.txt",
        "train_wtas\n" + "\n".join(split_meta["train_wtas"]) + "\n\n" +
        "test_wtas\n" + "\n".join(split_meta["test_wtas"]) + "\n",
    )
    train_data.summary().to_csv(output_dir / "train_study_summary.csv", index=False)
    test_data.summary().to_csv(output_dir / "test_study_summary.csv", index=False)

    print(f"Training CAPE for {args.epochs} epochs ...")
    train_peak_gpu_mb = None
    eval_peak_gpu_mb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = trainer.train(stage="all", n_epochs=args.epochs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        train_peak_gpu_mb = peak_gpu_stats_mb()
    train_time_s = time.time() - t0
    print(f"Training finished in {train_time_s:.1f}s")

    best_ckpt = output_dir / "checkpoint_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=cfg.resolve_device())
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint from {best_ckpt}")

    eval_device = cfg.resolve_device()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    train_eval = evaluate_endpoint_problem(
        model,
        train_ep,
        supported_pids,
        set(control_ids),
        device=eval_device,
        n_particles=args.eval_particles,
        n_steps=args.eval_steps,
        target_particles=args.eval_target_particles,
        seed=args.seed,
        eps=cfg.training.sinkhorn_epsilon,
        tau=cfg.training.sinkhorn_tau,
    )
    test_eval = evaluate_endpoint_problem(
        model,
        test_ep,
        supported_pids,
        set(control_ids),
        device=eval_device,
        n_particles=args.eval_particles,
        n_steps=args.eval_steps,
        target_particles=args.eval_target_particles,
        seed=args.seed,
        eps=cfg.training.sinkhorn_epsilon,
        tau=cfg.training.sinkhorn_tau,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        eval_peak_gpu_mb = peak_gpu_stats_mb()

    train_eval.to_csv(output_dir / "train_endpoint_metrics.csv", index=False)
    test_eval.to_csv(output_dir / "test_endpoint_metrics.csv", index=False)
    history.to_dataframe().to_csv(output_dir / "training_history_export.csv", index=False)

    train_summary = summarize_eval(train_eval)
    test_summary = summarize_eval(test_eval)
    results = {
        "train_time_s": round(train_time_s, 1),
        "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "n_supported_perturbations": len(supported_pids),
        "guide_confident_only": args.guide_confident_only,
        "data_path": args.data_path,
        "output_dir": str(output_dir),
        "train_peak_gpu_mb": train_peak_gpu_mb,
        "eval_peak_gpu_mb": eval_peak_gpu_mb,
    }
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2))

    md = "\n".join(
        [
            "# CAPE HNSCC P4/P60 Run",
            "",
            f"Output dir: `{output_dir}`",
            f"Data path: `{args.data_path}`",
            "",
            f"- Guide-confident only: `{args.guide_confident_only}`",
            f"- WTA column: `{args.wta_column}`",
            f"- Train WTAs: `{', '.join(split_meta['train_wtas'])}`",
            f"- Test WTAs: `{', '.join(split_meta['test_wtas'])}`",
            f"- Train particles / steps: `{args.n_particles}` / `{args.n_steps}`",
            f"- Eval particles / steps: `{args.eval_particles}` / `{args.eval_steps}`",
            f"- Eval target atoms per perturbation: `{args.eval_target_particles}`",
            f"- Weak-form test functions: `{args.n_test_functions}`",
            f"- Weak loss weight: `{args.lambda_weak}`",
            f"- Max train target atoms per perturbation: `{args.max_train_target_atoms}`",
            f"- Supported perturbations: `{len(supported_pids)}`",
            f"- Control ids: `{', '.join(control_ids)}`",
            f"- Train time (s): `{train_time_s:.1f}`",
            f"- Train peak GPU allocated / reserved (MB): "
            f"`{train_peak_gpu_mb['allocated_mb']:.1f}` / `{train_peak_gpu_mb['reserved_mb']:.1f}`"
            if train_peak_gpu_mb is not None
            else "- Train peak GPU allocated / reserved (MB): `n/a`",
            f"- Eval peak GPU allocated / reserved (MB): "
            f"`{eval_peak_gpu_mb['allocated_mb']:.1f}` / `{eval_peak_gpu_mb['reserved_mb']:.1f}`"
            if eval_peak_gpu_mb is not None
            else "- Eval peak GPU allocated / reserved (MB): `n/a`",
            "",
            "## Train Endpoint Summary",
            "",
            f"- Mean UOT: `{train_summary['mean_uot']:.4f}`",
            f"- Median UOT: `{train_summary['median_uot']:.4f}`",
            f"- Mean mass rel error: `{train_summary['mean_mass_rel_error']:.4f}`",
            "",
            "## Test Endpoint Summary",
            "",
            f"- Mean UOT: `{test_summary['mean_uot']:.4f}`",
            f"- Median UOT: `{test_summary['median_uot']:.4f}`",
            f"- Mean mass rel error: `{test_summary['mean_mass_rel_error']:.4f}`",
            "",
        ]
    )
    save_text(output_dir / "summary.md", md)
    print(md)


if __name__ == "__main__":
    main()
