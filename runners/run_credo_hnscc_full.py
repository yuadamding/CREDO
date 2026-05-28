"""
Train and evaluate CREDO on the HNSCC P4/P60 perturb-seq dataset.

This runner now favors a compact, biology-aligned setup:
  - stratified random train/test split by default
  - optional WTA split for stress-testing batch generalization
  - optional fixed epithelial-state centroids as the program basis
  - endpoint UOT + state-composition evaluation

Run with:
  conda run -n credo-hnscc python runners/run_credo_hnscc_full.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent / "package"
sys.path.insert(0, str(ROOT / "src"))

import credo
from credo.config.schema import LatentConfig, ModelConfig, RunConfig, SimulationConfig, TrainingConfig, VAEConfig
from credo.data.hnscc import (
    DEFAULT_LATENT_KEY,
    DEFAULT_RANDOM_STRATIFY_COLS,
    DEFAULT_STATE_KEY,
    DEFAULT_TEST_WTAS,
    DEFAULT_TRAIN_WTAS,
    DEFAULT_WTA_COLUMN,
    build_split_summary,
    build_study_from_split,
    build_study_from_vae_latent,
    build_vae_latent,
    compute_state_centroids,
    load_hnscc,
    load_hnscc_obs,
    make_random_kfold_split,
    make_random_split,
    make_wta_split,
    parse_list_arg,
    prepare_hnscc_obs,
    supported_intersection,
)
from credo.eval.hnscc import (
    build_true_terminal_state_table,
    cap_endpoint_problem_terminal,
    evaluate_endpoint_problem,
    evaluate_state_compositions,
    summarize_eval,
    summarize_state_metrics,
)
from credo.models.full_model import FullDynamicsModel
from credo.training.trainer import Trainer


DEFAULT_DATA_CANDIDATES = [
    "../inputs/hnscc/GSE235325_P4P60_allgenes_allcells_latest_states.h5ad",
]
DEFAULT_OUTPUT = "runs/hnscc_credo_full_random"
BASELINE_SUPPORTED_PERTURBATIONS = 121
BASELINE_TRAIN_PARTICLES = 128
BASELINE_TEST_FUNCTIONS = 12
VRAM_COMPLEXITY_HEADROOM = 0.70


def default_data_path() -> str:
    for candidate in DEFAULT_DATA_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return DEFAULT_DATA_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full CREDO training on HNSCC perturb-seq data.")
    parser.add_argument("--data-path", default=default_data_path())
    parser.add_argument("--latent-source", choices=["obsm", "vae"], default="vae")
    parser.add_argument("--latent-key", default=DEFAULT_LATENT_KEY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--multi-gpu-devices", default="")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cpu-interop-threads", type=int, default=0)

    parser.add_argument("--split-strategy", choices=["random", "random_kfold", "wta"], default="random")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--random-stratify-cols",
        default=",".join(col for col in DEFAULT_RANDOM_STRATIFY_COLS if col != DEFAULT_STATE_KEY),
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-fold-index", type=int, default=0)
    parser.add_argument("--wta-column", default=DEFAULT_WTA_COLUMN)
    parser.add_argument("--train-wtas", default=",".join(DEFAULT_TRAIN_WTAS))
    parser.add_argument("--test-wtas", default=",".join(DEFAULT_TEST_WTAS))

    parser.add_argument("--state-key", default=None)
    parser.add_argument("--use-state-centroids", dest="use_state_centroids", action="store_true")
    parser.add_argument("--learned-programs", dest="use_state_centroids", action="store_false")
    parser.add_argument("--program-assignment-scale", type=float, default=2.0)
    parser.set_defaults(use_state_centroids=False)
    parser.add_argument("--shared-guide-embedding", dest="shared_guide_embedding", action="store_true")
    parser.add_argument("--distinct-guide-embedding", dest="shared_guide_embedding", action="store_false")
    parser.set_defaults(shared_guide_embedding=False)
    parser.add_argument(
        "--control-mode",
        choices=["anchored", "free", "soft_ref"],
        default="soft_ref",
    )
    parser.add_argument("--control-anchor", dest="control_mode", action="store_const", const="anchored")
    parser.add_argument("--control-free", dest="control_mode", action="store_const", const="free")
    parser.add_argument("--soft-control-ref", dest="control_mode", action="store_const", const="soft_ref")
    parser.add_argument("--lambda-control-ref", type=float, default=5e-4)
    parser.add_argument("--control-ref-warmup-epochs", type=int, default=150)
    parser.add_argument("--training-schedule", choices=["joint", "staged"], default="staged")
    parser.add_argument("--stage-c-epochs", type=int, default=150)
    parser.add_argument("--stage-d-epochs", type=int, default=150)
    parser.add_argument("--ecology-on", dest="ecological_growth", action="store_true")
    parser.add_argument("--ecology-off", dest="ecological_growth", action="store_false")
    parser.set_defaults(ecological_growth=True)
    parser.add_argument("--growth-intercept-on", dest="use_growth_intercept", action="store_true")
    parser.add_argument("--growth-intercept-off", dest="use_growth_intercept", action="store_false")
    parser.set_defaults(use_growth_intercept=True)
    parser.add_argument("--activation-checkpointing", dest="activation_checkpointing", action="store_true")
    parser.add_argument("--no-activation-checkpointing", dest="activation_checkpointing", action="store_false")
    parser.set_defaults(activation_checkpointing=False)

    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--mediator-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--context-kind", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--transformer-token-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-within-layers", type=int, default=2)
    parser.add_argument("--transformer-cross-layers", type=int, default=2)
    parser.add_argument("--transformer-inducing", type=int, default=16)
    parser.add_argument("--transformer-dropout", type=float, default=0.05)
    parser.add_argument("--mass-attention-temperature", type=float, default=1.0)
    parser.add_argument("--transformer-growth-only", dest="transformer_growth_only", action="store_true")
    parser.add_argument("--transformer-all-coefficients", dest="transformer_growth_only", action="store_false")
    parser.set_defaults(transformer_growth_only=False)
    parser.add_argument("--n-particles", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--eval-particles", type=int, default=384)
    parser.add_argument("--eval-steps", type=int, default=24)
    parser.add_argument("--eval-target-particles", type=int, default=768)
    parser.add_argument("--max-train-target-atoms", type=int, default=768)
    parser.add_argument("--n-test-functions", type=int, default=12)
    parser.add_argument("--lambda-weak", type=float, default=0.1)
    parser.add_argument("--lambda-reg-growth-bias", type=float, default=1e-4)
    parser.add_argument("--max-active-perturbations", type=int, default=0)
    parser.add_argument("--budget-headroom", type=float, default=VRAM_COMPLEXITY_HEADROOM)
    parser.add_argument("--auto-scale-budget", dest="auto_scale_budget", action="store_true")
    parser.add_argument("--no-auto-scale-budget", dest="auto_scale_budget", action="store_false")
    parser.add_argument("--min-cells-p4", type=int, default=20)
    parser.add_argument("--min-cells-p60", type=int, default=20)
    parser.add_argument("--mass-value-col", default=None)
    parser.add_argument("--mass-scope", choices=["full_obs", "subset_only"], default="subset_only")
    parser.add_argument(
        "--mass-mode",
        choices=["auto", "count", "group_total", "per_cell_contribution"],
        default="auto",
        help=(
            "Mass semantics for --mass-value-col. count ignores the column; "
            "group_total uses one repeated group total; per_cell_contribution sums values; "
            "auto refuses ambiguous repeated group totals."
        ),
    )
    parser.add_argument("--guide-confident-only", dest="guide_confident_only", action="store_true")
    parser.add_argument("--include-nonconfident", dest="guide_confident_only", action="store_false")
    parser.add_argument("--expression-gene-mask-col", default="hv_gene")
    parser.add_argument("--expression-top-genes", type=int, default=2000)
    parser.add_argument("--vae-allow-empty-gene-mask-fallback", dest="vae_allow_empty_gene_mask_fallback", action="store_true")
    parser.add_argument("--no-vae-allow-empty-gene-mask-fallback", dest="vae_allow_empty_gene_mask_fallback", action="store_false")
    parser.add_argument("--vae-latent-dim", type=int, default=50)
    parser.add_argument("--vae-hidden-dim", type=int, default=512)
    parser.add_argument("--vae-depth", type=int, default=2)
    parser.add_argument("--vae-dropout", type=float, default=0.1)
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--vae-batch-size", type=int, default=1024)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-weight-decay", type=float, default=1e-6)
    parser.add_argument("--vae-kl-weight", type=float, default=1e-3)
    parser.add_argument("--vae-kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--vae-val-frac", type=float, default=0.1)
    parser.add_argument("--vae-early-stop-patience", type=int, default=15)
    parser.add_argument("--vae-grad-clip", type=float, default=1.0)
    parser.add_argument("--vae-layer", type=str, default=None)
    parser.add_argument("--vae-use-raw", dest="vae_use_raw", action="store_true")
    parser.add_argument("--no-vae-use-raw", dest="vae_use_raw", action="store_false")
    parser.add_argument("--vae-target-sum", type=float, default=1e4)
    parser.add_argument("--vae-strict-layer", dest="vae_strict_layer", action="store_true")
    parser.add_argument("--no-vae-strict-layer", dest="vae_strict_layer", action="store_false")
    parser.add_argument("--vae-strict-counts", dest="vae_strict_counts", action="store_true")
    parser.add_argument("--no-vae-strict-counts", dest="vae_strict_counts", action="store_false")
    parser.add_argument("--vae-encode-batch-size", type=int, default=4096)
    parser.add_argument("--expression-workers", type=int, default=0)
    parser.add_argument("--expression-chunk-size", type=int, default=1024)
    parser.add_argument("--vae-batch-aware-hvg", dest="vae_batch_aware_hvg", action="store_true")
    parser.add_argument("--no-vae-batch-aware-hvg", dest="vae_batch_aware_hvg", action="store_false")
    parser.add_argument("--vae-hvg-batch-col", type=str, default=DEFAULT_WTA_COLUMN)
    parser.add_argument("--vae-hvg-time-col", type=str, default="Time point")
    parser.add_argument("--vae-hvg-min-cells-per-batch", type=int, default=256)
    parser.add_argument("--vae-allow-full-gene-scan", dest="vae_allow_full_gene_scan", action="store_true")
    parser.add_argument("--no-vae-allow-full-gene-scan", dest="vae_allow_full_gene_scan", action="store_false")
    parser.add_argument("--vae-preload-dense-max-gb", type=float, default=4.0)
    parser.add_argument("--vae-reuse-artifact", dest="vae_reuse_artifact", action="store_true")
    parser.add_argument("--no-vae-reuse-artifact", dest="vae_reuse_artifact", action="store_false")
    parser.add_argument("--vae-use-amp", dest="vae_use_amp", action="store_true")
    parser.add_argument("--no-vae-use-amp", dest="vae_use_amp", action="store_false")
    parser.add_argument("--vae-amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.set_defaults(guide_confident_only=True)
    parser.set_defaults(auto_scale_budget=True)
    parser.set_defaults(vae_use_raw=False)
    parser.set_defaults(vae_allow_empty_gene_mask_fallback=False)
    parser.set_defaults(vae_strict_layer=True)
    parser.set_defaults(vae_strict_counts=True)
    parser.set_defaults(vae_batch_aware_hvg=True)
    parser.set_defaults(vae_allow_full_gene_scan=False)
    parser.set_defaults(vae_reuse_artifact=True)
    parser.set_defaults(vae_use_amp=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(cpu_threads: int = 0, cpu_interop_threads: int = 0) -> None:
    if cpu_threads and cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    if cpu_interop_threads and cpu_interop_threads > 0:
        try:
            torch.set_num_interop_threads(cpu_interop_threads)
        except RuntimeError:
            pass
    torch.set_float32_matmul_precision("high")
    deterministic = os.environ.get("CREDO_DETERMINISTIC", "0").lower() in {"1", "true", "yes", "on"}
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def save_text(path: Path, text: str) -> None:
    path.write_text(text)


def resolve_git_sha(repo_root: Path) -> str | None:
    for candidate in [repo_root, *repo_root.parents]:
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        sha = result.stdout.strip()
        if sha:
            return sha
    return None


def resolve_git_dirty(repo_root: Path) -> bool | None:
    for candidate in [repo_root, *repo_root.parents]:
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        return bool(result.stdout.strip())
    return None


def file_metadata(path: str | None) -> dict:
    if not path:
        return {"path": None, "exists": False}
    data_path = Path(path).expanduser()
    exists = data_path.exists()
    out = {
        "path": str(data_path),
        "resolved_path": str(data_path.resolve()) if exists else None,
        "exists": exists,
    }
    if exists:
        stat = data_path.stat()
        out.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    if os.environ.get("CREDO_DATA_SHA256"):
        out["sha256"] = os.environ["CREDO_DATA_SHA256"]
    return out


def software_versions_manifest(args: argparse.Namespace, *, git_sha: str | None, git_dirty: bool | None) -> dict:
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(idx))),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return {
        "package_name": "credo",
        "package_version": credo.__version__,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "data_file": file_metadata(args.data_path),
        "data_sha256_note": "Set CREDO_DATA_SHA256 to record a precomputed full data-file SHA256 without re-reading the shared H5AD in every parallel job.",
    }


def parse_device_list_arg(raw: str) -> list[str]:
    devices: list[str] = []
    for item in str(raw).split(","):
        dev = item.strip()
        if not dev:
            continue
        if dev.isdigit():
            dev = f"cuda:{dev}"
        elif dev == "cuda":
            dev = "cuda:0"
        devices.append(dev)
    return devices


def normalize_optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "na"}:
        return None
    return text


def resolve_random_stratify_cols(
    raw_cols: str,
    *,
    state_key: str | None,
    obs: pd.DataFrame,
) -> list[str]:
    cols = parse_list_arg(raw_cols)
    if state_key is None:
        cols = [col for col in cols if col != DEFAULT_STATE_KEY]
    missing = [col for col in cols if col not in obs.columns]
    if missing:
        raise KeyError(
            "Requested random stratify columns are not present after preprocessing: "
            + ", ".join(repr(col) for col in missing)
        )
    return cols


def _resolve_peak_devices(devices: list[str] | None) -> list[str]:
    if not torch.cuda.is_available():
        return []
    if devices:
        resolved = [str(device) for device in devices if str(device).startswith("cuda")]
        if resolved:
            return resolved
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def reset_peak_gpu_stats(devices: list[str] | None = None) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    for device in _resolve_peak_devices(devices):
        torch.cuda.reset_peak_memory_stats(device=device)


def synchronize_devices(devices: list[str] | None = None) -> None:
    if not torch.cuda.is_available():
        return
    for device in _resolve_peak_devices(devices):
        torch.cuda.synchronize(device=device)


def peak_gpu_stats_mb(devices: list[str] | None = None) -> dict | None:
    if not torch.cuda.is_available():
        return None
    per_device: list[dict] = []
    for device in _resolve_peak_devices(devices):
        per_device.append(
            {
                "device": device,
                "allocated_mb": float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)),
                "reserved_mb": float(torch.cuda.max_memory_reserved(device=device) / (1024 ** 2)),
            }
        )
    return {
        "allocated_mb": float(sum(item["allocated_mb"] for item in per_device)),
        "reserved_mb": float(sum(item["reserved_mb"] for item in per_device)),
        "per_device": per_device,
    }


def format_peak_gpu_line(label: str, stats: dict | None) -> str:
    if stats is None:
        return f"- {label} peak GPU allocated / reserved (MB): `n/a`"
    per_device = stats.get("per_device", [])
    if not per_device:
        return (
            f"- {label} peak GPU allocated / reserved (MB): "
            f"`{stats['allocated_mb']:.1f}` / `{stats['reserved_mb']:.1f}`"
        )
    detail = ", ".join(
        f"{item['device']}=`{item['allocated_mb']:.1f}`/`{item['reserved_mb']:.1f}`"
        for item in per_device
    )
    return (
        f"- {label} peak GPU allocated / reserved (MB): "
        f"`{stats['allocated_mb']:.1f}` / `{stats['reserved_mb']:.1f}` total"
        f" ({detail})"
    )


def build_split(
    obs: pd.DataFrame,
    args: argparse.Namespace,
    *,
    random_stratify_cols: list[str],
):
    if args.split_strategy == "random":
        return make_random_split(
            obs,
            train_frac=args.train_frac,
            seed=args.seed,
            stratify_cols=random_stratify_cols,
        )
    if args.split_strategy == "random_kfold":
        return make_random_kfold_split(
            obs,
            n_folds=args.cv_folds,
            fold_index=args.cv_fold_index,
            seed=args.seed,
            stratify_cols=random_stratify_cols,
        )
    return make_wta_split(
        obs,
        wta_column=args.wta_column,
        train_wtas=parse_list_arg(args.train_wtas),
        test_wtas=parse_list_arg(args.test_wtas),
    )


def calibrate_train_budget(args: argparse.Namespace, n_supported_pids: int) -> dict:
    max_complexity = (
        BASELINE_SUPPORTED_PERTURBATIONS * BASELINE_TRAIN_PARTICLES * BASELINE_TEST_FUNCTIONS
    )
    target_complexity = int(np.floor(args.budget_headroom * max_complexity))
    requested = n_supported_pids * args.n_particles * max(args.n_test_functions, 1)
    eff_particles = args.n_particles
    eff_test_functions = args.n_test_functions

    if args.auto_scale_budget and requested > target_complexity:
        scale = (target_complexity / requested) ** 0.5
        eff_particles = max(64, int(np.floor(args.n_particles * scale / 8.0) * 8))
        eff_test_functions = max(6, int(np.floor(args.n_test_functions * scale)))
        while (
            n_supported_pids * eff_particles * eff_test_functions > target_complexity
            and eff_particles > 64
        ):
            eff_particles -= 8
        while (
            n_supported_pids * eff_particles * eff_test_functions > target_complexity
            and eff_test_functions > 4
        ):
            eff_test_functions -= 1

    return {
        "effective_n_particles": int(eff_particles),
        "effective_n_test_functions": int(eff_test_functions),
        "requested_complexity": int(requested),
        "max_complexity": int(max_complexity),
        "target_complexity": int(target_complexity),
        "headroom_fraction": float(args.budget_headroom),
        "auto_scale_budget": bool(args.auto_scale_budget),
        "budget_scaled": bool(
            eff_particles != args.n_particles or eff_test_functions != args.n_test_functions
        ),
    }


def format_split_lines(split_meta: dict) -> list[str]:
    lines = [f"- Split strategy: `{split_meta['split_strategy']}`"]
    if split_meta["split_strategy"] == "random":
        lines.extend(
            [
                f"- Train fraction: `{split_meta['train_frac']}`",
                f"- Split seed: `{split_meta['seed']}`",
                f"- Random stratify cols: `{', '.join(split_meta['stratify_cols']) or 'none'}`",
            ]
        )
    elif split_meta["split_strategy"] == "random_kfold":
        lines.extend(
            [
                f"- CV folds: `{split_meta['n_folds']}`",
                f"- CV test fold index: `{split_meta['fold_index']}`",
                f"- Split seed: `{split_meta['seed']}`",
                f"- Random stratify cols: `{', '.join(split_meta['stratify_cols']) or 'none'}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- WTA column: `{split_meta['wta_column']}`",
                f"- Train WTAs: `{', '.join(split_meta['train_wtas'])}`",
                f"- Test WTAs: `{', '.join(split_meta['test_wtas'])}`",
            ]
        )
    return lines


def resolve_training_schedule(args: argparse.Namespace) -> list[tuple[str, int]]:
    total_epochs = max(int(args.epochs), 0)
    if args.training_schedule == "joint":
        return [("all", total_epochs)] if total_epochs > 0 else []

    stage_c = max(int(args.stage_c_epochs), 0)
    stage_d = max(int(args.stage_d_epochs), 0)
    stage_c = min(stage_c, total_epochs)
    stage_d = min(stage_d, max(total_epochs - stage_c, 0))
    remaining = max(total_epochs - stage_c - stage_d, 0)

    # The control-only warm-start only makes sense when controls are explicitly
    # represented as the zero-effect baseline.
    if args.control_mode == "free":
        stage_d += stage_c
        stage_c = 0

    # When ecology is disabled, fold the final stage into the ecology-off warm-start.
    if not args.ecological_growth:
        stage_d += remaining
        remaining = 0

    schedule: list[tuple[str, int]] = []
    if stage_c > 0:
        schedule.append(("C", stage_c))
    if stage_d > 0:
        schedule.append(("D", stage_d))
    if remaining > 0:
        schedule.append(("E", remaining))
    return schedule


def format_training_schedule(schedule: list[tuple[str, int]]) -> str:
    if not schedule:
        return "none"
    return ", ".join(f"{stage}:{epochs}" for stage, epochs in schedule)


def main() -> None:
    args = parse_args()
    state_key = normalize_optional_name(args.state_key)
    args.state_key = state_key
    set_seed(args.seed)
    configure_torch(
        cpu_threads=args.cpu_threads,
        cpu_interop_threads=args.cpu_interop_threads,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HNSCC data from {args.data_path}")
    if args.latent_source == "vae":
        raw_obs = load_hnscc_obs(args.data_path)
        raw_latent = None
        expr_gene_names: list[str] = []
        expr_meta: dict = {}
        print(f"Loaded obs={raw_obs.shape} for VAE latent construction")
    else:
        raw_obs, raw_latent = load_hnscc(args.data_path, latent_key=args.latent_key)
        raw_expr = None
        expr_gene_names = []
        expr_meta = {}
        print(f"Loaded obs={raw_obs.shape} latent={raw_latent.shape}")

    obs, kept_positions = prepare_hnscc_obs(
        raw_obs,
        guide_confident_only=args.guide_confident_only,
        state_key=state_key,
    )

    random_stratify_cols = resolve_random_stratify_cols(
        args.random_stratify_cols,
        state_key=state_key,
        obs=obs,
    )
    split_result = build_split(obs, args, random_stratify_cols=random_stratify_cols)
    split = split_result.split
    split_meta = split_result.metadata
    split_meta["stratify_cols"] = list(random_stratify_cols) if args.split_strategy in {"random", "random_kfold"} else split_meta.get("stratify_cols")
    multi_gpu_devices = parse_device_list_arg(args.multi_gpu_devices)
    primary_device = multi_gpu_devices[0] if multi_gpu_devices else "auto"
    git_sha = resolve_git_sha(ROOT)
    git_dirty = resolve_git_dirty(ROOT)
    software_versions = software_versions_manifest(args, git_sha=git_sha, git_dirty=git_dirty)
    state_eval_enabled = state_key is not None

    if args.use_state_centroids and not state_eval_enabled:
        raise ValueError("Fixed state centroids require a non-empty --state-key.")

    vae_summary = None
    vae_bundle = None
    vae_result = None
    if args.latent_source == "vae":
        vae_device = primary_device if primary_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Building split-safe VAE latent on training cells only "
            f"(expression_workers={args.expression_workers}, "
            f"expression_chunk_size={args.expression_chunk_size}) ..."
        )
        vae_result = build_vae_latent(
            args.data_path,
            split=split,
            obs=obs,
            kept_positions=kept_positions,
            latent_dim=args.vae_latent_dim,
            layer=args.vae_layer,
            use_raw=args.vae_use_raw,
            gene_mask_col=args.expression_gene_mask_col,
            n_genes=args.expression_top_genes,
            allow_empty_gene_mask_fallback=args.vae_allow_empty_gene_mask_fallback,
            batch_aware_hvg=args.vae_batch_aware_hvg,
            hvg_batch_col=args.vae_hvg_batch_col,
            hvg_time_col=args.vae_hvg_time_col,
            hvg_min_cells_per_batch=args.vae_hvg_min_cells_per_batch,
            allow_full_gene_scan=args.vae_allow_full_gene_scan,
            target_sum=args.vae_target_sum,
            strict_layer=args.vae_strict_layer,
            strict_counts=args.vae_strict_counts,
            vae_hidden_dim=args.vae_hidden_dim,
            vae_depth=args.vae_depth,
            vae_dropout=args.vae_dropout,
            vae_epochs=args.vae_epochs,
            vae_batch_size=args.vae_batch_size,
            vae_lr=args.vae_lr,
            vae_weight_decay=args.vae_weight_decay,
            vae_kl_weight=args.vae_kl_weight,
            vae_kl_warmup_epochs=args.vae_kl_warmup_epochs,
            vae_val_frac=args.vae_val_frac,
            vae_early_stop_patience=args.vae_early_stop_patience,
            vae_grad_clip=args.vae_grad_clip,
            vae_seed=args.seed,
            encode_batch_size=args.vae_encode_batch_size,
            expression_workers=args.expression_workers,
            expression_chunk_size=args.expression_chunk_size,
            preload_dense_max_gb=args.vae_preload_dense_max_gb,
            reuse_artifact=args.vae_reuse_artifact,
            vae_use_amp=args.vae_use_amp,
            vae_amp_dtype=args.vae_amp_dtype,
            device=vae_device,
            state_key=state_key,
            compute_centroids=args.use_state_centroids,
            save_dir=str(output_dir / "vae_artifact"),
            commit_sha=git_sha,
        )
        latent = vae_result.latent
        vae_bundle = vae_result.bundle
        vae_summary = vae_bundle.training_summary
        expr_gene_names = list(vae_bundle.gene_names)
        expr_meta = {
            "requested_layer": vae_bundle.requested_layer,
            "layer": vae_bundle.source_layer,
            "selected_gene_indices": vae_bundle.selected_gene_indices,
            "split_manifest_hash": vae_bundle.split_manifest_hash,
        }
        (output_dir / "expression_vae_genes.txt").write_text("\n".join(expr_gene_names) + "\n")
        print(
            f"VAE trained: {vae_summary.epochs_trained} epochs, "
            f"early_stopped={vae_summary.early_stopped}, "
            f"best_val={vae_summary.best_val_loss:.4f} at epoch {vae_summary.best_epoch}"
        )
    else:
        latent = raw_latent[kept_positions]

    if args.latent_source == "vae":
        train_data, test_data = build_study_from_vae_latent(
            vae_result,
            obs,
            split,
            mass_value_col=args.mass_value_col,
            mass_scope=args.mass_scope,
            mass_mode=args.mass_mode,
        )
    else:
        train_data = build_study_from_split(
            obs,
            latent,
            split=split,
            split_name="train",
            mass_value_col=args.mass_value_col,
            mass_scope=args.mass_scope,
            mass_mode=args.mass_mode,
        )
        test_data = build_study_from_split(
            obs,
            latent,
            split=split,
            split_name="test",
            mass_value_col=args.mass_value_col,
            mass_scope=args.mass_scope,
            mass_mode=args.mass_mode,
        )
    supported_pids = supported_intersection(
        train_data,
        test_data,
        min_cells_p4=args.min_cells_p4,
        min_cells_p60=args.min_cells_p60,
    )
    control_ids = [pid for pid in train_data.catalog.control_ids if pid in supported_pids]
    if not control_ids:
        raise ValueError("No supported control perturbations remain after intersection.")

    train_mask = split.eq("train").to_numpy()
    train_states: list[str] = []
    train_centroids: np.ndarray | None = None
    centroid_counts: pd.DataFrame | None = None
    if state_eval_enabled:
        train_states, train_centroids, centroid_counts = compute_state_centroids(
            obs.loc[train_mask],
            latent[train_mask],
            state_key=state_key,
        )
    resolved_n_programs = len(train_states) if args.use_state_centroids else args.n_programs
    if args.use_state_centroids:
        if args.latent_source == "vae":
            program_centroids = vae_result.program_centroids
        else:
            program_centroids = train_centroids
    else:
        program_centroids = None
    budget = calibrate_train_budget(args, len(supported_pids))
    train_particles = budget["effective_n_particles"]
    train_test_functions = budget["effective_n_test_functions"]
    training_schedule = resolve_training_schedule(args)
    if not training_schedule:
        raise ValueError("Resolved training schedule is empty; increase --epochs.")

    print(
        f"Train cells={train_data.cell_state.n_cells} | Test cells={test_data.cell_state.n_cells} | "
        f"Supported perturbations={len(supported_pids)} | Controls={control_ids}"
    )
    print(f"Training schedule: {format_training_schedule(training_schedule)}")
    if multi_gpu_devices:
        print(f"Multi-GPU training devices: {', '.join(multi_gpu_devices)}")
    if budget["budget_scaled"]:
        print(
            "Adjusted training budget for VRAM target: "
            f"particles {args.n_particles}->{train_particles}, "
            f"test_functions {args.n_test_functions}->{train_test_functions}"
        )

    train_ep_full = train_data.to_endpoint_problem(
        perturbation_ids=supported_pids,
        initial_label="P4",
        terminal_label="P60",
    )
    test_ep = test_data.to_endpoint_problem(
        perturbation_ids=supported_pids,
        initial_label="P4",
        terminal_label="P60",
    )
    train_ep = cap_endpoint_problem_terminal(
        train_ep_full,
        max_terminal_atoms=args.max_train_target_atoms,
        seed=args.seed,
    )

    train_true_state_table = None
    test_true_state_table = None
    if state_eval_enabled:
        train_true_state_table = build_true_terminal_state_table(
            obs.loc[train_mask],
            perturbation_ids=supported_pids,
            target_time=60.0,
            state_key=state_key,
        )
        test_true_state_table = build_true_terminal_state_table(
            obs.loc[~train_mask],
            perturbation_ids=supported_pids,
            target_time=60.0,
            state_key=state_key,
        )

    latent_dim = train_data.latent_dim
    cfg = RunConfig(
        git_sha=git_sha,
        device=primary_device,
        multi_gpu_devices=multi_gpu_devices,
        output_dir=str(output_dir),
        latent=LatentConfig(
            source="vae" if args.latent_source == "vae" else "pca",
            key="X_vae" if args.latent_source == "vae" else args.latent_key,
            dim=latent_dim,
            whiten=False,
            vae=VAEConfig(
                hidden_dim=args.vae_hidden_dim,
                depth=args.vae_depth,
                dropout=args.vae_dropout,
                epochs=args.vae_epochs,
                batch_size=args.vae_batch_size,
                learning_rate=args.vae_lr,
                weight_decay=args.vae_weight_decay,
                kl_weight=args.vae_kl_weight,
                kl_warmup_epochs=args.vae_kl_warmup_epochs,
                val_frac=args.vae_val_frac,
                early_stop_patience=args.vae_early_stop_patience,
                grad_clip=args.vae_grad_clip,
                seed=args.seed,
                layer=args.vae_layer,
                use_raw=args.vae_use_raw,
                n_genes=args.expression_top_genes,
                gene_mask_col=args.expression_gene_mask_col,
                allow_empty_gene_mask_fallback=args.vae_allow_empty_gene_mask_fallback,
                target_sum=args.vae_target_sum,
                strict_layer=args.vae_strict_layer,
                strict_counts=args.vae_strict_counts,
                batch_aware_hvg=args.vae_batch_aware_hvg,
                hvg_batch_col=args.vae_hvg_batch_col,
                hvg_time_col=args.vae_hvg_time_col,
                hvg_min_cells_per_batch=args.vae_hvg_min_cells_per_batch,
                allow_full_gene_scan=args.vae_allow_full_gene_scan,
                expression_workers=args.expression_workers,
                expression_chunk_size=args.expression_chunk_size,
                preload_dense_max_gb=args.vae_preload_dense_max_gb,
                reuse_artifact=args.vae_reuse_artifact,
                use_amp=args.vae_use_amp,
                amp_dtype=args.vae_amp_dtype,
            ),
        ),
        data={
            "min_cells_p4": args.min_cells_p4,
            "min_cells_p60": args.min_cells_p60,
            "mass_value_col": args.mass_value_col,
            "mass_scope": args.mass_scope,
            "mass_mode": args.mass_mode,
        },
        model=ModelConfig(
            embedding_dim=args.embedding_dim,
            n_programs=resolved_n_programs,
            mediator_dim=args.mediator_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            activation_checkpointing=args.activation_checkpointing,
            ecological_growth=args.ecological_growth,
            use_growth_intercept=args.use_growth_intercept,
            control_mode=args.control_mode,
            control_ref_penalty=args.lambda_control_ref,
            context_kind=args.context_kind,
            transformer_token_dim=args.transformer_token_dim,
            transformer_heads=args.transformer_heads,
            transformer_within_layers=args.transformer_within_layers,
            transformer_cross_layers=args.transformer_cross_layers,
            transformer_inducing=args.transformer_inducing,
            transformer_dropout=args.transformer_dropout,
            mass_attention_temperature=args.mass_attention_temperature,
            transformer_growth_only=args.transformer_growth_only,
        ),
        simulation=SimulationConfig(
            n_particles=train_particles,
            n_steps=args.n_steps,
            store_history=True,
        ),
        training=TrainingConfig(
            precision=args.precision,
            epochs=args.epochs,
            lr_net=3e-4,
            lr_embed=1e-3,
            lambda_end=1.0,
            lambda_weak=args.lambda_weak,
            lambda_count=0.0,
            lambda_reg_embed=1e-4,
            lambda_reg_growth_bias=args.lambda_reg_growth_bias,
            lambda_reg_net=1e-4,
            lambda_reg_diffusion=1e-4,
            training_schedule=args.training_schedule,
            stage_c_epochs=args.stage_c_epochs,
            stage_d_epochs=args.stage_d_epochs,
            max_active_perturbations=args.max_active_perturbations,
            control_ref_warmup_epochs=args.control_ref_warmup_epochs,
            seed=args.seed,
            early_stop_patience=args.epochs,
            log_every=25,
            checkpoint_every=100,
            sinkhorn_epsilon=0.1,
            sinkhorn_tau=1.0,
            n_test_functions=train_test_functions,
            test_function_bandwidth=1.0,
        ),
    )

    model = FullDynamicsModel(
        perturbation_ids=supported_pids,
        control_ids=control_ids,
        latent_dim=latent_dim,
        embedding_dim=args.embedding_dim,
        n_programs=resolved_n_programs,
        mediator_dim=args.mediator_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        activation_checkpointing=args.activation_checkpointing,
        ecological_growth=args.ecological_growth,
        use_growth_intercept=args.use_growth_intercept,
        shared_guide_embedding=args.shared_guide_embedding,
        program_centroids=program_centroids,
        program_assignment_scale=args.program_assignment_scale,
        control_mode=args.control_mode,
        control_ref_penalty=args.lambda_control_ref,
        context_kind=args.context_kind,
        transformer_token_dim=args.transformer_token_dim,
        transformer_heads=args.transformer_heads,
        transformer_within_layers=args.transformer_within_layers,
        transformer_cross_layers=args.transformer_cross_layers,
        transformer_inducing=args.transformer_inducing,
        transformer_dropout=args.transformer_dropout,
        mass_attention_temperature=args.mass_attention_temperature,
        transformer_growth_only=args.transformer_growth_only,
    ).to(cfg.resolve_device())

    trainer = Trainer(model, cfg, train_ep, supported_pids, output_dir=str(output_dir))

    split_summary = build_split_summary(obs, split=split, state_key=state_key)
    split_assignments_dict = {
        "cell_id": obs["cell_id"].astype(str).to_numpy(),
        "split": split.to_numpy(),
        "Time point": obs["Time point"].to_numpy(),
        "perturbation_id": obs["perturbation_id"].to_numpy(),
    }
    if state_eval_enabled:
        split_assignments_dict[state_key] = obs[state_key].to_numpy()
    split_assignments = pd.DataFrame(split_assignments_dict)

    meta = {
        "method": "CREDO",
        "data_path": args.data_path,
        "latent_source": args.latent_source,
        "latent_key": args.latent_key if args.latent_source == "obsm" else None,
        "expression_gene_mask_col": args.expression_gene_mask_col if args.latent_source == "vae" else None,
        "expression_top_genes": args.expression_top_genes if args.latent_source == "vae" else None,
        "expression_selected_genes": expr_gene_names if args.latent_source == "vae" else None,
        "expression_selected_gene_indices": expr_meta.get("selected_gene_indices") if args.latent_source == "vae" else None,
        "expression_requested_layer": expr_meta.get("requested_layer") if args.latent_source == "vae" else None,
        "expression_resolved_layer": expr_meta.get("layer") if args.latent_source == "vae" else None,
        "expression_split_manifest_hash": expr_meta.get("split_manifest_hash") if args.latent_source == "vae" else None,
        "expression_encoder": {
            **vae_summary.__dict__,
            "kl_warmup_epochs": args.vae_kl_warmup_epochs,
            "val_frac": args.vae_val_frac,
            "early_stop_patience": args.vae_early_stop_patience,
            "latent_standardized": True,
            "vae_layer": args.vae_layer,
            "vae_use_raw": args.vae_use_raw,
            "allow_empty_gene_mask_fallback": args.vae_allow_empty_gene_mask_fallback,
            "target_sum": args.vae_target_sum,
            "strict_layer": args.vae_strict_layer,
            "strict_counts": args.vae_strict_counts,
            "batch_aware_hvg": args.vae_batch_aware_hvg,
            "hvg_batch_col": args.vae_hvg_batch_col,
            "hvg_time_col": args.vae_hvg_time_col,
            "hvg_min_cells_per_batch": args.vae_hvg_min_cells_per_batch,
            "allow_full_gene_scan": args.vae_allow_full_gene_scan,
            "preload_dense_max_gb": args.vae_preload_dense_max_gb,
            "reuse_artifact": args.vae_reuse_artifact,
            "vae_use_amp": args.vae_use_amp,
            "vae_amp_dtype": args.vae_amp_dtype,
            "expression_workers": args.expression_workers,
            "expression_chunk_size": args.expression_chunk_size,
        } if vae_summary is not None else None,
        "guide_confident_only": args.guide_confident_only,
        "split": split_meta,
        "state_key": state_key,
        "state_eval_enabled": state_eval_enabled,
        "use_state_centroids": args.use_state_centroids,
        "shared_guide_embedding": args.shared_guide_embedding,
        "embedding_dim": args.embedding_dim,
        "mediator_dim": args.mediator_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "n_programs": resolved_n_programs,
        "control_mode": args.control_mode,
        "control_anchor": args.control_mode == "anchored",
        "lambda_control_ref": args.lambda_control_ref,
        "control_ref_warmup_epochs": args.control_ref_warmup_epochs,
        "control_ref_warmup_active": bool(
            args.training_schedule == "joint" and args.control_mode == "soft_ref"
        ),
        "ecological_growth": args.ecological_growth,
        "use_growth_intercept": args.use_growth_intercept,
        "activation_checkpointing": args.activation_checkpointing,
        "training_schedule": args.training_schedule,
        "stage_c_epochs": args.stage_c_epochs,
        "stage_d_epochs": args.stage_d_epochs,
        "resolved_training_schedule": [
            {"stage": stage, "epochs": epochs} for stage, epochs in training_schedule
        ],
        "program_assignment_scale": args.program_assignment_scale if args.use_state_centroids else None,
        "resolved_n_programs": resolved_n_programs,
        "requested_n_particles": args.n_particles,
        "effective_n_particles": train_particles,
        "n_steps": args.n_steps,
        "eval_particles": args.eval_particles,
        "eval_steps": args.eval_steps,
        "eval_target_particles": args.eval_target_particles,
        "requested_n_test_functions": args.n_test_functions,
        "effective_n_test_functions": train_test_functions,
        "max_active_perturbations": args.max_active_perturbations,
        "vram_budget": budget,
        "budget_headroom": args.budget_headroom,
        "lambda_weak": args.lambda_weak,
        "lambda_reg_growth_bias": args.lambda_reg_growth_bias,
        "precision": args.precision,
        "cpu_threads": args.cpu_threads,
        "cpu_interop_threads": args.cpu_interop_threads,
        "multi_gpu_devices": multi_gpu_devices,
        "max_train_target_atoms": args.max_train_target_atoms,
        "mass_value_col": args.mass_value_col,
        "mass_scope": args.mass_scope,
        "mass_mode": args.mass_mode,
        "requested_mass_mode": args.mass_mode,
        "train_mass_mode": train_data.mass_table.df.attrs.get("mass_mode"),
        "test_mass_mode": test_data.mass_table.df.attrs.get("mass_mode"),
        "train_mass_mode_resolution_reason": train_data.mass_table.df.attrs.get("mass_mode_resolution_reason"),
        "test_mass_mode_resolution_reason": test_data.mass_table.df.attrs.get("mass_mode_resolution_reason"),
        "train_cells": int(train_data.cell_state.n_cells),
        "test_cells": int(test_data.cell_state.n_cells),
        "supported_perturbations": supported_pids,
        "control_ids": control_ids,
        "state_labels": train_states if state_eval_enabled else None,
        "config": cfg.model_dump(),
        "software_versions": software_versions,
    }
    (output_dir / "config.json").write_text(json.dumps(meta, indent=2))
    (output_dir / "software_versions.json").write_text(json.dumps(software_versions, indent=2))
    split_result.manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    if centroid_counts is not None:
        centroid_counts.to_csv(output_dir / "state_reference.csv", index=False)
    save_text(output_dir / "supported_perturbations.txt", "\n".join(supported_pids) + "\n")
    train_data.summary().to_csv(output_dir / "train_study_summary.csv", index=False)
    test_data.summary().to_csv(output_dir / "test_study_summary.csv", index=False)

    print(f"Training CREDO for {args.epochs} epochs ...")
    train_peak_gpu_mb = None
    eval_peak_gpu_mb = None
    if torch.cuda.is_available():
        reset_peak_gpu_stats(multi_gpu_devices or [cfg.resolve_device()])
    t0 = time.time()
    history = trainer.history
    for stage_name, stage_epochs in training_schedule:
        print(f"Training stage {stage_name} for {stage_epochs} epochs ...")
        history = trainer.train(stage=stage_name, n_epochs=stage_epochs)
    if torch.cuda.is_available():
        synchronize_devices(multi_gpu_devices or [cfg.resolve_device()])
        train_peak_gpu_mb = peak_gpu_stats_mb(multi_gpu_devices or [cfg.resolve_device()])
    train_time_s = time.time() - t0
    print(f"Training finished in {train_time_s:.1f}s")

    # Prefer EMA checkpoint for evaluation (smoother, better generalization)
    ema_ckpt = output_dir / "checkpoint_best_ema.pt"
    best_ckpt = output_dir / "checkpoint_best.pt"
    evaluated_ckpt = None
    if ema_ckpt.exists():
        ckpt = torch.load(ema_ckpt, map_location=cfg.resolve_device())
        model.load_state_dict(ckpt["model_state"])
        evaluated_ckpt = ema_ckpt
        print(f"Loaded EMA checkpoint from {ema_ckpt}")
    elif best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=cfg.resolve_device())
        model.load_state_dict(ckpt["model_state"])
        evaluated_ckpt = best_ckpt
        print(f"Loaded best checkpoint from {best_ckpt}")

    eval_device = cfg.resolve_device()
    if torch.cuda.is_available():
        reset_peak_gpu_stats(multi_gpu_devices or [eval_device])

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
    train_state_metrics = None
    test_state_metrics = None
    train_state_dist = None
    test_state_dist = None
    if state_eval_enabled:
        train_state_metrics, train_state_dist = evaluate_state_compositions(
            model,
            train_ep,
            supported_pids,
            state_labels=train_states,
            state_centroids=train_centroids,
            true_state_table=train_true_state_table,
            device=eval_device,
            n_particles=args.eval_particles,
            n_steps=args.eval_steps,
            seed=args.seed,
        )
        test_state_metrics, test_state_dist = evaluate_state_compositions(
            model,
            test_ep,
            supported_pids,
            state_labels=train_states,
            state_centroids=train_centroids,
            true_state_table=test_true_state_table,
            device=eval_device,
            n_particles=args.eval_particles,
            n_steps=args.eval_steps,
            seed=args.seed,
        )
    if torch.cuda.is_available():
        synchronize_devices(multi_gpu_devices or [eval_device])
        eval_peak_gpu_mb = peak_gpu_stats_mb(multi_gpu_devices or [eval_device])

    train_eval.to_csv(output_dir / "train_endpoint_metrics.csv", index=False)
    test_eval.to_csv(output_dir / "test_endpoint_metrics.csv", index=False)
    if train_state_metrics is not None:
        train_state_metrics.to_csv(output_dir / "train_state_metrics.csv", index=False)
    if test_state_metrics is not None:
        test_state_metrics.to_csv(output_dir / "test_state_metrics.csv", index=False)
    if train_state_dist is not None:
        train_state_dist.to_csv(output_dir / "train_state_distributions.csv", index=False)
    if test_state_dist is not None:
        test_state_dist.to_csv(output_dir / "test_state_distributions.csv", index=False)
    history.to_dataframe().to_csv(output_dir / "training_history.csv", index=False)
    history.to_dataframe().to_csv(output_dir / "training_history_export.csv", index=False)

    train_summary = summarize_eval(train_eval)
    test_summary = summarize_eval(test_eval)
    train_state_summary = summarize_state_metrics(train_state_metrics) if train_state_metrics is not None else None
    test_state_summary = summarize_state_metrics(test_state_metrics) if test_state_metrics is not None else None

    results = {
        "method": "CREDO",
        "package_name": "credo",
        "package_version": credo.__version__,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "train_time_s": round(train_time_s, 1),
        "best_checkpoint": str(evaluated_ckpt) if evaluated_ckpt is not None else None,
        "training_best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
        "evaluated_checkpoint": str(evaluated_ckpt) if evaluated_ckpt is not None else None,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "train_state_summary": train_state_summary,
        "test_state_summary": test_state_summary,
        "n_supported_perturbations": len(supported_pids),
        "guide_confident_only": args.guide_confident_only,
        "data_path": args.data_path,
        "latent_source": args.latent_source,
        "latent_key": args.latent_key if args.latent_source == "obsm" else None,
        "output_dir": str(output_dir),
        "state_key": state_key,
        "state_eval_enabled": state_eval_enabled,
        "use_state_centroids": args.use_state_centroids,
        "shared_guide_embedding": args.shared_guide_embedding,
        "control_mode": args.control_mode,
        "control_anchor": args.control_mode == "anchored",
        "lambda_control_ref": args.lambda_control_ref,
        "control_ref_warmup_epochs": args.control_ref_warmup_epochs,
        "control_ref_warmup_active": bool(
            args.training_schedule == "joint" and args.control_mode == "soft_ref"
        ),
        "ecological_growth": args.ecological_growth,
        "training_schedule": args.training_schedule,
        "stage_c_epochs": args.stage_c_epochs,
        "stage_d_epochs": args.stage_d_epochs,
        "epochs": args.epochs,
        "lambda_weak": args.lambda_weak,
        "lambda_count": cfg.training.lambda_count,
        "mass_supervision": "endpoint_geometry_plus_log_mass",
        "mass_mode": args.mass_mode,
        "requested_mass_mode": args.mass_mode,
        "train_mass_mode": train_data.mass_table.df.attrs.get("mass_mode"),
        "test_mass_mode": test_data.mass_table.df.attrs.get("mass_mode"),
        "train_mass_mode_resolution_reason": train_data.mass_table.df.attrs.get("mass_mode_resolution_reason"),
        "test_mass_mode_resolution_reason": test_data.mass_table.df.attrs.get("mass_mode_resolution_reason"),
        "lambda_reg_growth_bias": args.lambda_reg_growth_bias,
        "max_active_perturbations": args.max_active_perturbations,
        "use_growth_intercept": args.use_growth_intercept,
        "embedding_dim": args.embedding_dim,
        "mediator_dim": args.mediator_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "resolved_training_schedule": [
            {"stage": stage, "epochs": epochs} for stage, epochs in training_schedule
        ],
        "resolved_n_programs": resolved_n_programs,
        "requested_n_particles": args.n_particles,
        "effective_n_particles": train_particles,
        "n_steps": args.n_steps,
        "eval_particles": args.eval_particles,
        "eval_steps": args.eval_steps,
        "eval_target_particles": args.eval_target_particles,
        "train_peak_gpu_mb": train_peak_gpu_mb,
        "eval_peak_gpu_mb": eval_peak_gpu_mb,
    }
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2))

    train_gpu_line = format_peak_gpu_line("Train", train_peak_gpu_mb)
    eval_gpu_line = format_peak_gpu_line("Eval", eval_peak_gpu_mb)

    md_lines = [
        "# CREDO HNSCC P4/P60 Run",
        "",
        f"Output dir: `{output_dir}`",
        f"Data path: `{args.data_path}`",
        f"- Latent source: `{args.latent_source}`",
        f"- Latent key: `{args.latent_key}`" if args.latent_source == "obsm" else None,
        f"- Expression genes for VAE: `{len(expr_gene_names)}`" if args.latent_source == "vae" else None,
        "",
        f"- Guide-confident only: `{args.guide_confident_only}`",
        *format_split_lines(split_meta),
        f"- State key: `{state_key if state_key is not None else 'disabled'}`",
        f"- State evaluation enabled: `{state_eval_enabled}`",
        f"- Program basis: `{'fixed state centroids' if args.use_state_centroids else 'learned latent programs'}`",
        f"- Shared guide embedding: `{args.shared_guide_embedding}`",
        f"- Control mode: `{args.control_mode}`",
        f"- Control reference penalty: `{args.lambda_control_ref}`",
        f"- Control reference warmup epochs: `{args.control_ref_warmup_epochs}`",
        f"- Control reference warmup active: `{args.training_schedule == 'joint' and args.control_mode == 'soft_ref'}`",
        f"- Ecological growth enabled: `{args.ecological_growth}`",
        f"- Growth intercept enabled: `{args.use_growth_intercept}`",
        f"- Training schedule: `{args.training_schedule}`",
        f"- Resolved training stages: `{format_training_schedule(training_schedule)}`",
        f"- Resolved program count: `{resolved_n_programs}`",
        f"- Train particles / steps: `{train_particles}` / `{args.n_steps}`",
        f"- Eval particles / steps: `{args.eval_particles}` / `{args.eval_steps}`",
        f"- Eval target atoms per perturbation: `{args.eval_target_particles}`",
        f"- Weak-form test functions: `{train_test_functions}`",
        f"- Max active perturbations per train step: "
        f"`{args.max_active_perturbations if args.max_active_perturbations > 0 else 'all'}`",
        f"- Auto-scale VRAM budget: `{args.auto_scale_budget}`",
        f"- Budget headroom fraction: `{args.budget_headroom}`",
        f"- VRAM budget scaled: `{budget['budget_scaled']}`",
        f"- Weak loss weight: `{args.lambda_weak}`",
        f"- Growth-bias regularization: `{args.lambda_reg_growth_bias}`",
        f"- Precision: `{args.precision}`",
        f"- CPU threads / interop threads: `{args.cpu_threads}` / `{args.cpu_interop_threads}`",
        f"- Max train target atoms per perturbation: `{args.max_train_target_atoms}`",
        f"- Mass scope: `{args.mass_scope}`",
        f"- Requested mass mode: `{args.mass_mode}`",
        f"- Train mass mode: `{train_data.mass_table.df.attrs.get('mass_mode')}`",
        f"- Test mass mode: `{test_data.mass_table.df.attrs.get('mass_mode')}`",
        f"- Supported perturbations: `{len(supported_pids)}`",
        f"- Control ids: `{', '.join(control_ids)}`",
        f"- Train time (s): `{train_time_s:.1f}`",
        train_gpu_line,
        eval_gpu_line,
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
        "## Train State Summary" if train_state_summary is not None else "## State Summary",
        "",
        f"- Mean state TV: `{train_state_summary['mean_state_tv']:.4f}`" if train_state_summary is not None else "- Skipped because no state_key was provided.",
        f"- Median state TV: `{train_state_summary['median_state_tv']:.4f}`" if train_state_summary is not None else None,
        (
            f"- Dominant-state accuracy: `{train_state_summary['dominant_state_accuracy']:.4f}`"
            if train_state_summary is not None and train_state_summary["dominant_state_accuracy"] is not None
            else "- Dominant-state accuracy: `n/a`"
        ) if train_state_summary is not None else None,
        f"- Mean abs expansion-ratio gap: `{train_state_summary['mean_abs_expansion_ratio_gap']:.4f}`" if train_state_summary is not None else None,
        "",
        "## Test State Summary" if test_state_summary is not None else None,
        "",
        f"- Mean state TV: `{test_state_summary['mean_state_tv']:.4f}`" if test_state_summary is not None else None,
        f"- Median state TV: `{test_state_summary['median_state_tv']:.4f}`" if test_state_summary is not None else None,
        (
            f"- Dominant-state accuracy: `{test_state_summary['dominant_state_accuracy']:.4f}`"
            if test_state_summary is not None and test_state_summary["dominant_state_accuracy"] is not None
            else "- Dominant-state accuracy: `n/a`"
        ) if test_state_summary is not None else None,
        f"- Mean abs expansion-ratio gap: `{test_state_summary['mean_abs_expansion_ratio_gap']:.4f}`" if test_state_summary is not None else None,
        "",
    ]
    md = "\n".join(line for line in md_lines if line is not None)
    save_text(output_dir / "summary.md", md)
    print(md)


if __name__ == "__main__":
    main()
