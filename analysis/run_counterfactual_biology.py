#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "src"))

from credo.data.hnscc import (  # noqa: E402
    build_study_from_split,
    compute_state_centroids,
    load_hnscc,
    load_hnscc_obs,
    prepare_hnscc_obs,
)
from credo.models.full_model import FullDynamicsModel  # noqa: E402
from credo.models.simulator import (  # noqa: E402
    _control_embedding_context,
    embedding_ids_from_endpoint,
    initialise_particles,
    rollout_with_clamped_context,
)
from credo.models.weighted_sde import ParticleRollout, WeightedParticleSimulator  # noqa: E402

from hnscc_biology_common import infer_target_gene  # noqa: E402

EXPLICIT_MASS_MODES = {"count", "group_total", "per_cell_contribution"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factual-vs-reference CREDO rollouts for biological effect mining."
    )
    parser.add_argument("--run-dir", required=True, help="Single trained run directory.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--source-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--n-particles", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=28)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-perturbations", type=int, default=0)
    parser.add_argument("--perturbations", default="", help="Comma-separated perturbation ids.")
    parser.add_argument(
        "--full-context-endpoint",
        action="store_true",
        help="Build the endpoint from all available perturbations while only emitting selected rows.",
    )
    parser.add_argument(
        "--include-controls-for-null",
        action="store_true",
        help="Also emit control-guide same-start counterfactuals for metric-specific null calibration.",
    )
    parser.add_argument("--context-clamped", action="store_true")
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="Store only initial and terminal particles; disables action and clamped-context summaries.",
    )
    parser.add_argument(
        "--allow-partial-context",
        action="store_true",
        help="Allow global-context rollouts when a small fraction of model perturbations lack endpoints.",
    )
    parser.add_argument(
        "--min-context-fraction",
        type=float,
        default=0.95,
        help="Minimum model perturbation fraction required when --allow-partial-context is set.",
    )
    parser.add_argument("--fold-id", default=None, help="Optional fold label written to the output table.")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _first_non_null(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _requested_mass_mode_for_counterfactual(config: dict, results: dict, source_split: str) -> str:
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    requested = _first_non_null(
        results.get("requested_mass_mode"),
        config.get("requested_mass_mode"),
        data_cfg.get("requested_mass_mode"),
    )
    requested_s = str(requested).strip().lower() if requested is not None else ""
    if requested_s not in EXPLICIT_MASS_MODES:
        split_hint = _first_non_null(
            results.get(f"{source_split}_mass_mode"),
            config.get(f"{source_split}_mass_mode"),
            results.get("resolved_mass_mode"),
            config.get("resolved_mass_mode"),
        )
        raise ValueError(
            "Counterfactual biology requires explicit requested_mass_mode "
            f"({sorted(EXPLICIT_MASS_MODES)}); got {requested!r} "
            f"with resolved split hint {split_hint!r}."
        )
    return requested_s


def _resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    return "cuda" if torch.cuda.is_available() else "cpu"


def _checkpoint_path(run_dir: Path, results: dict) -> Path:
    candidates: list[Path] = []
    for key in ("evaluated_checkpoint", "best_checkpoint", "training_best_checkpoint"):
        value = results.get(key)
        if value:
            path = Path(value)
            candidates.extend([path, run_dir / path.name])
    candidates.extend([run_dir / "checkpoint_best_ema.pt", run_dir / "checkpoint_best.pt"])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found for {run_dir}")


def _load_obs_latent(run_dir: Path, config: dict, data_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    state_key = config.get("state_key")
    guide_confident_only = bool(config.get("guide_confident_only", True))
    latent_source = config.get("latent_source", "vae")

    if latent_source == "vae":
        raw_obs = load_hnscc_obs(data_path)
        obs, _ = prepare_hnscc_obs(
            raw_obs,
            guide_confident_only=guide_confident_only,
            state_key=state_key,
        )
        latent_path = run_dir / "vae_artifact" / "latent_all_std.npy"
        if not latent_path.exists():
            raise FileNotFoundError(
                f"Missing {latent_path}; rerun with VAE artifact caching or use --latent-source obsm runs."
            )
        latent = np.load(latent_path)
    elif latent_source == "expression":
        raw_obs = load_hnscc_obs(data_path)
        obs, _ = prepare_hnscc_obs(
            raw_obs,
            guide_confident_only=guide_confident_only,
            state_key=state_key,
        )
        latent_path = run_dir / "expression_artifact" / "latent_all_std.npy"
        if not latent_path.exists():
            raise FileNotFoundError(
                f"Missing {latent_path}; rerun the raw-expression run to cache expression_artifact."
            )
        latent = np.load(latent_path)
    else:
        raw_obs, raw_latent = load_hnscc(data_path, latent_key=config.get("latent_key"))
        obs, kept_positions = prepare_hnscc_obs(
            raw_obs,
            guide_confident_only=guide_confident_only,
            state_key=state_key,
        )
        latent = raw_latent[kept_positions]
    if len(obs) != len(latent):
        raise ValueError(f"obs/latent length mismatch: {len(obs)} vs {len(latent)}")
    return obs, latent


def _load_split(run_dir: Path, obs: pd.DataFrame, source_split: str) -> pd.Series:
    if source_split == "all":
        return pd.Series("analysis", index=obs.index, dtype="object")
    split_path = run_dir / "split_assignments.csv"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    split_df = pd.read_csv(split_path)
    if "cell_id" not in split_df.columns or "split" not in split_df.columns:
        raise KeyError("split_assignments.csv must contain cell_id and split columns.")
    if len(split_df) == len(obs) and split_df["cell_id"].astype(str).to_numpy().tolist() == obs["cell_id"].astype(str).to_numpy().tolist():
        split = pd.Series(split_df["split"].astype(str).to_numpy(), index=obs.index)
    else:
        split_map = dict(zip(split_df["cell_id"].astype(str), split_df["split"].astype(str)))
        split = obs["cell_id"].astype(str).map(split_map)
    if split.isna().any():
        raise ValueError("Some obs cells are missing from split_assignments.csv.")
    if not split.eq(source_split).any():
        raise ValueError(f"No cells found for split={source_split!r}.")
    return split


def _program_centroids(config: dict, obs: pd.DataFrame, latent: np.ndarray, split: pd.Series) -> np.ndarray | None:
    if not bool(config.get("use_state_centroids", False)):
        return None
    state_key = config.get("state_key")
    if not state_key:
        return None
    train_mask = split.eq("train").to_numpy()
    if not train_mask.any():
        train_mask = np.ones(len(obs), dtype=bool)
    _, centroids, _ = compute_state_centroids(obs.loc[train_mask], latent[train_mask], state_key=state_key)
    return centroids


def _build_model(config: dict, latent_dim: int, program_centroids: np.ndarray | None, device: str) -> FullDynamicsModel:
    model_cfg = config.get("config", {}).get("model", {}) if isinstance(config.get("config"), dict) else {}

    def cfg(name: str, default):
        return model_cfg.get(name, config.get(name, default))

    model = FullDynamicsModel(
        perturbation_ids=list(config["supported_perturbations"]),
        control_ids=list(config["control_ids"]),
        latent_dim=latent_dim,
        embedding_dim=int(cfg("embedding_dim", 8)),
        n_programs=int(config.get("resolved_n_programs", cfg("n_programs", 8))),
        mediator_dim=int(cfg("mediator_dim", 8)),
        hidden_dim=int(cfg("hidden_dim", 128)),
        depth=int(cfg("depth", 3)),
        activation_checkpointing=False,
        n_time_freqs=int(cfg("time_frequencies", 4)),
        sigma_min=float(cfg("sigma_min", 1e-3)),
        r_max=float(cfg("r_max", 3.0)),
        n_payoff_ranks=int(cfg("n_payoff_ranks", 4)),
        ecological_growth=bool(cfg("ecological_growth", True)),
        use_growth_intercept=bool(cfg("use_growth_intercept", True)),
        shared_guide_embedding=bool(config.get("shared_guide_embedding", False)),
        program_centroids=program_centroids,
        program_assignment_scale=float(config.get("program_assignment_scale") or 1.0),
        control_mode=str(cfg("control_mode", config.get("control_mode", "soft_ref"))),
        control_ref_penalty=float(cfg("control_ref_penalty", config.get("lambda_control_ref", 5e-4))),
        context_kind=str(cfg("context_kind", "causal_attention")),
        transformer_token_dim=int(cfg("transformer_token_dim", 64)),
        transformer_heads=int(cfg("transformer_heads", 4)),
        transformer_within_layers=int(cfg("transformer_within_layers", 1)),
        transformer_cross_layers=int(cfg("transformer_cross_layers", 1)),
        transformer_inducing=int(cfg("transformer_inducing", 8)),
        transformer_dropout=float(cfg("transformer_dropout", 0.05)),
        mass_attention_temperature=float(cfg("mass_attention_temperature", 0.5)),
        transformer_growth_only=bool(cfg("transformer_growth_only", True)),
        causal_token_dim=int(cfg("causal_token_dim", 64)),
        causal_heads=int(cfg("causal_heads", 4)),
        causal_n_mediators=int(cfg("causal_n_mediators", 12)),
        causal_dropout=float(cfg("causal_dropout", 0.05)),
        causal_mass_attention_temperature=float(cfg("causal_mass_attention_temperature", 0.5)),
        causal_growth_only=bool(cfg("causal_growth_only", True)),
        causal_sparse_edges=bool(cfg("causal_sparse_edges", True)),
        causal_residual_policy=str(cfg("causal_residual_policy", "edges_only")),
    ).to(device)
    return model


def _terminal_log_mass(rollout: ParticleRollout) -> torch.Tensor:
    if rollout.log_m0 is None:
        raise ValueError("Rollout is missing log_m0.")
    return rollout.log_m0 + torch.logsumexp(rollout.terminal_logw, dim=1)


def _weighted_mean(z: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(logw, dim=-1)
    return (w.unsqueeze(-1) * z).sum(dim=-2)


def _weighted_energy_distance(
    z_a: torch.Tensor,
    logw_a: torch.Tensor,
    z_b: torch.Tensor,
    logw_b: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> float:
    w_a = torch.softmax(logw_a, dim=-1)
    w_b = torch.softmax(logw_b, dim=-1)

    def weighted_distance_sum(x: torch.Tensor, wx: torch.Tensor, y: torch.Tensor, wy: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=x.dtype, device=x.device)
        step = max(int(chunk_size), 1)
        for start in range(0, x.shape[0], step):
            stop = min(start + step, x.shape[0])
            dist = torch.cdist(x[start:stop], y)
            total = total + (wx[start:stop, None] * wy[None, :] * dist).sum()
        return total

    cross = weighted_distance_sum(z_a, w_a, z_b, w_b)
    self_a = weighted_distance_sum(z_a, w_a, z_a, w_a)
    self_b = weighted_distance_sum(z_b, w_b, z_b, w_b)
    return float(torch.clamp(2.0 * cross - self_a - self_b, min=0.0).item())


def _entropy(logw: torch.Tensor) -> float:
    w = torch.softmax(logw, dim=-1)
    return float((-(w * torch.log(w + 1e-30)).sum(dim=-1)).item())


def _tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _materialize_rollout(rollout: ParticleRollout) -> ParticleRollout:
    diagnostics = None
    if rollout.context_diagnostics is not None:
        diagnostics = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in rollout.context_diagnostics.items()
        }
    return ParticleRollout(
        z_steps=rollout.z_steps.detach().clone(),
        logw_steps=rollout.logw_steps.detach().clone(),
        tau_steps=rollout.tau_steps.detach().clone(),
        log_m0=_clone_optional_tensor(rollout.log_m0),
        drift_steps=_clone_optional_tensor(rollout.drift_steps),
        sigma_steps=_clone_optional_tensor(rollout.sigma_steps),
        growth_steps=_clone_optional_tensor(rollout.growth_steps),
        context_steps=_clone_optional_tensor(rollout.context_steps),
        base_context_steps=_clone_optional_tensor(rollout.base_context_steps),
        growth_context_steps=_clone_optional_tensor(rollout.growth_context_steps),
        context_diagnostics=diagnostics,
        causal_edge_scores_steps=_clone_optional_tensor(rollout.causal_edge_scores_steps),
        causal_baseline_edge_scores_steps=_clone_optional_tensor(rollout.causal_baseline_edge_scores_steps),
        causal_residual_edge_scores_steps=_clone_optional_tensor(rollout.causal_residual_edge_scores_steps),
        causal_residual_edge_magnitude_steps=_clone_optional_tensor(rollout.causal_residual_edge_magnitude_steps),
        causal_mediator_tokens_steps=_clone_optional_tensor(rollout.causal_mediator_tokens_steps),
        causal_growth_context_steps=_clone_optional_tensor(rollout.causal_growth_context_steps),
        causal_delta_steps=_clone_optional_tensor(rollout.causal_delta_steps),
        noise_steps=_clone_optional_tensor(rollout.noise_steps),
        ess_steps=_clone_optional_tensor(rollout.ess_steps),
        ess_frac_steps=_clone_optional_tensor(rollout.ess_frac_steps),
        logw_range_steps=_clone_optional_tensor(rollout.logw_range_steps),
        max_weight_frac_steps=_clone_optional_tensor(rollout.max_weight_frac_steps),
    )


def _counterfactual_context_metadata(
    model: FullDynamicsModel,
    endpoint,
    *,
    allow_partial_context: bool = False,
    min_context_fraction: float = 0.95,
) -> dict:
    model_pids = list(model.perturbation_ids)
    available = [pid for pid in model_pids if pid in endpoint.initial]
    missing = [pid for pid in model_pids if pid not in endpoint.initial]
    fraction = len(available) / float(max(1, len(model_pids)))
    return {
        "context_n_available": len(available),
        "context_n_model": len(model_pids),
        "context_fraction": fraction,
        "allow_partial_context": bool(allow_partial_context),
        "min_context_fraction": float(min_context_fraction),
        "context_missing_perturbations": missing,
    }


def _select_counterfactual_pids(
    available: list[str],
    control_ids: set[str],
    requested: list[str],
    *,
    include_controls_for_null: bool,
    max_perturbations: int,
) -> list[str]:
    available_set = set(available)
    requested_set = set(requested)
    noncontrols = [pid for pid in available if pid not in control_ids]
    if requested:
        noncontrols = [pid for pid in noncontrols if pid in requested_set]
    if max_perturbations > 0:
        noncontrols = noncontrols[:max_perturbations]
    controls = []
    if include_controls_for_null:
        controls = [pid for pid in available if pid in control_ids]
        if requested:
            controls = [pid for pid in controls if pid in requested_set]
    selected = [*noncontrols, *controls]
    missing = sorted(requested_set - available_set)
    if missing:
        raise KeyError(f"Requested perturbations not available in study: {missing}")
    return selected


def _action_deltas(fact: dict, ref: dict) -> dict:
    out = {}
    for key in ("growth_action", "drift_action", "diffusion_action"):
        if key in fact and key in ref:
            out[f"delta_{key}_fact_vs_ref"] = fact[key] - ref[key]
    return out


def _action_summary(rollout: ParticleRollout) -> dict:
    if rollout.growth_steps is None or rollout.drift_steps is None or rollout.sigma_steps is None:
        return {}
    dt = 1.0 / max(int(rollout.n_steps), 1)
    growth = []
    drift = []
    diffusion = []
    for k in range(rollout.n_steps):
        w = torch.softmax(rollout.logw_steps[k, 0], dim=-1)
        growth.append((w * rollout.growth_steps[k, 0]).sum())
        drift.append((w * torch.linalg.norm(rollout.drift_steps[k, 0], dim=-1)).sum())
        diffusion.append((w * (rollout.sigma_steps[k, 0] ** 2).sum(dim=-1)).sum())
    return {
        "growth_action": float((torch.stack(growth).sum() * dt).item()),
        "drift_action": float((torch.stack(drift).sum() * dt).item()),
        "diffusion_action": float((torch.stack(diffusion).sum() * dt).item()),
    }


@torch.no_grad()
def _program_fractions(model: FullDynamicsModel, rollout: ParticleRollout) -> torch.Tensor:
    encoder = getattr(model.context_agg, "program_encoder", None)
    if encoder is None:
        encoder = getattr(model.context_agg, "encoder", None)
    if encoder is None:
        raise AttributeError(
            f"{type(model.context_agg).__name__} does not expose a program encoder for biology summaries."
        )
    eta = encoder.eta(rollout.terminal_z)[0]
    w = torch.softmax(rollout.terminal_logw[0], dim=-1)
    return (w.unsqueeze(-1) * eta).sum(dim=0)


@torch.no_grad()
def _program_summary(model: FullDynamicsModel, rollout: ParticleRollout, labels: list[str] | None) -> dict:
    q = _program_fractions(model, rollout).detach().cpu().numpy()
    idx = int(q.argmax())
    out = {
        "dominant_program_index": idx,
        "dominant_program_fraction": float(q[idx]),
        "program_entropy": float(-(q * np.log(q + 1e-30)).sum()),
    }
    if labels and idx < len(labels):
        out["dominant_program_label"] = labels[idx]
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.terminal_only and args.context_clamped:
        raise ValueError("--terminal-only cannot be combined with --context-clamped.")
    run_dir = Path(args.run_dir)
    config = _read_json(run_dir / "config.json")
    results = _read_json(run_dir / "results_summary.json")
    data_path = args.data_path or config.get("data_path")
    if not data_path:
        raise ValueError("Data path must be supplied or present in config.json.")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "biology"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    obs, latent = _load_obs_latent(run_dir, config, data_path)
    original_split = _load_split(run_dir, obs, "train") if (run_dir / "split_assignments.csv").exists() else pd.Series("train", index=obs.index)
    source_split = _load_split(run_dir, obs, args.source_split)
    split_name = "analysis" if args.source_split == "all" else args.source_split
    if args.source_split == "all":
        source_split = pd.Series("analysis", index=obs.index, dtype="object")

    program_centroids = _program_centroids(config, obs, latent, original_split)
    requested_mass_mode = _requested_mass_mode_for_counterfactual(config, results, split_name)
    data = build_study_from_split(
        obs,
        latent,
        split=source_split,
        split_name=split_name,
        mass_value_col=config.get("mass_value_col"),
        mass_scope=config.get("mass_scope", "subset_only"),
        mass_mode=requested_mass_mode,
    )
    available_supported = [pid for pid in config["supported_perturbations"] if pid in data.catalog.perturbation_ids]
    requested = [pid.strip() for pid in args.perturbations.split(",") if pid.strip()]
    controls = set(config.get("control_ids", []))
    supported = _select_counterfactual_pids(
        available_supported,
        controls,
        requested,
        include_controls_for_null=args.include_controls_for_null,
        max_perturbations=args.max_perturbations,
    )
    if not supported:
        raise ValueError("No perturbations selected for counterfactual analysis.")
    endpoint_pids = available_supported if args.full_context_endpoint else supported
    endpoint = data.to_endpoint_problem(endpoint_pids, initial_label="P4", terminal_label="P60")
    missing_selected_endpoints = [pid for pid in supported if pid not in endpoint.initial]
    if missing_selected_endpoints:
        raise KeyError(f"Requested perturbations lack complete endpoints: {missing_selected_endpoints}")

    model = _build_model(config, latent.shape[1], program_centroids, device)
    checkpoint = _checkpoint_path(run_dir, results)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if bool(getattr(model.embedding, "shared_guide_embedding", False)):
        raise ValueError(
            "Counterfactual biology requires distinct guide embeddings. "
            "shared_guide_embedding uses one effective embedding for every perturbation, "
            "so per-perturbation factual-vs-reference effects are not identifiable."
        )
    simulator = WeightedParticleSimulator(n_steps=args.n_steps, store_history=not bool(args.terminal_only))
    context_metadata = _counterfactual_context_metadata(
        model,
        endpoint,
        allow_partial_context=bool(args.allow_partial_context),
        min_context_fraction=float(args.min_context_fraction),
    )
    row_context_metadata = {
        key: value for key, value in context_metadata.items()
        if key != "context_missing_perturbations"
    }
    use_global_context = getattr(model, "context_kind", "mlp") in {"transformer", "causal_attention"}
    global_all_pids: list[str] = []
    global_pid_to_idx: dict[str, int] = {}
    global_embedding_ids: list[str] = []
    global_z0_all = None
    global_lw0_all = None
    global_lm0_all = None
    global_noise_steps = None
    global_noise_seed = int(args.seed) + 10_000
    global_factual_by_pid = {}
    global_metadata_by_pid = {}
    if use_global_context:
        model_pids = list(model.perturbation_ids)
        global_all_pids = [pid for pid in model_pids if pid in endpoint.initial]
        missing = [pid for pid in model_pids if pid not in endpoint.initial]
        context_fraction = len(global_all_pids) / float(max(1, len(model_pids)))
        if len(global_all_pids) < 2:
            raise ValueError("Global-context counterfactuals require at least two perturbations.")
        if context_fraction < float(args.min_context_fraction) or (missing and not args.allow_partial_context):
            raise ValueError(
                "Global-context counterfactual context is incomplete: "
                f"{len(global_all_pids)}/{len(model_pids)} perturbations available."
            )
        global_pid_to_idx = {pid: idx for idx, pid in enumerate(global_all_pids)}
        global_z0_all, global_lw0_all, global_lm0_all = initialise_particles(
            endpoint,
            global_all_pids,
            n_particles=args.n_particles,
            device=device,
            seed=args.seed,
        )
        global_embedding_ids = embedding_ids_from_endpoint(endpoint, global_all_pids)
        global_noise_steps = simulator.sample_noise_like(
            global_z0_all,
            args.n_steps,
            seed=global_noise_seed,
        )
        factual_all = simulator.rollout(
            z0=global_z0_all,
            logw0=global_lw0_all,
            model=model,
            log_m0=global_lm0_all,
            perturbation_ids=global_all_pids,
            embedding_ids=global_embedding_ids,
            noise_steps=global_noise_steps,
            return_noise_used=not bool(args.terminal_only),
            terminal_only=bool(args.terminal_only),
        )
        for pid in supported:
            idx = global_pid_to_idx[pid]
            global_factual_by_pid[pid] = _materialize_rollout(factual_all.slice_group(idx))
            global_metadata_by_pid[pid] = {
                "context_kind": getattr(model, "context_kind", "mlp"),
                "target_perturbation_id": pid,
                "counterfactual_seed_mode": "global_common",
                "same_start": True,
                "same_noise": True,
                "initial_seed": int(args.seed),
                "noise_seed": global_noise_seed,
                "factual_full_context_reused": True,
            }
        del factual_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    state_labels = config.get("state_labels") if bool(config.get("use_state_centroids", False)) else None
    split_meta = config.get("split", {})
    fold_id = args.fold_id
    if fold_id is None:
        fold_index = split_meta.get("fold_index")
        fold_id = f"fold_{fold_index}" if fold_index is not None else run_dir.name
    rows = []
    for i, pid in enumerate(supported):
        if use_global_context:
            if pid not in global_factual_by_pid:
                raise RuntimeError(f"Missing global-context factual result for {pid!r}.")
            factual = global_factual_by_pid[pid]
            target_idx = global_pid_to_idx[pid]
            embed_pid = embedding_ids_from_endpoint(endpoint, [pid])[0]
            with _control_embedding_context(model, embed_pid, mode="reference_consistent"):
                reference_all = simulator.rollout(
                    z0=global_z0_all.clone(),
                    logw0=global_lw0_all.clone(),
                    model=model,
                    log_m0=global_lm0_all.clone(),
                    perturbation_ids=global_all_pids,
                    embedding_ids=global_embedding_ids,
                    noise_steps=global_noise_steps.clone(),
                    return_noise_used=not bool(args.terminal_only),
                    terminal_only=bool(args.terminal_only),
                )
            reference = _materialize_rollout(reference_all.slice_group(target_idx))
            del reference_all
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            clamped = None
            reference_clamped = None
            if not args.terminal_only and (factual.noise_steps is None or reference.noise_steps is None):
                raise RuntimeError("Global-context counterfactual did not return noise_steps for provenance.")
            if args.context_clamped:
                if (
                    reference.context_steps is None
                    or reference.base_context_steps is None
                    or reference.growth_context_steps is None
                ):
                    raise ValueError("Reference rollout did not store target context steps for clamped context.")
                tau_grid = reference.tau_steps.detach()
                tau_start = float(tau_grid[0].item())
                tau_end = float(tau_grid[-1].item())
                clamped = rollout_with_clamped_context(
                    model=model,
                    z0=factual.z_steps[0].clone(),
                    logw0=factual.logw_steps[0].clone(),
                    log_m0=factual.log_m0.clone(),
                    perturbation_ids=[pid],
                    embedding_ids=[embed_pid],
                    context_steps=reference.context_steps,
                    base_context_steps=reference.base_context_steps,
                    growth_context_steps=reference.growth_context_steps,
                    tau_start=tau_start,
                    tau_end=tau_end,
                    tau_grid=tau_grid,
                    noise_steps=factual.noise_steps.clone(),
                    return_noise_used=True,
                )
                with _control_embedding_context(model, embed_pid, mode="reference_consistent"):
                    reference_clamped = rollout_with_clamped_context(
                        model=model,
                        z0=reference.z_steps[0].clone(),
                        logw0=reference.logw_steps[0].clone(),
                        log_m0=reference.log_m0.clone(),
                        perturbation_ids=[pid],
                        embedding_ids=[embed_pid],
                        context_steps=reference.context_steps,
                        base_context_steps=reference.base_context_steps,
                        growth_context_steps=reference.growth_context_steps,
                        tau_start=tau_start,
                        tau_end=tau_end,
                        tau_grid=tau_grid,
                        noise_steps=reference.noise_steps.clone(),
                        return_noise_used=True,
                    )

            fact_z0_hash = _tensor_sha256(factual.z_steps[0])
            ref_z0_hash = _tensor_sha256(reference.z_steps[0])
            fact_logw0_hash = _tensor_sha256(factual.logw_steps[0])
            ref_logw0_hash = _tensor_sha256(reference.logw_steps[0])
            fact_log_m0_hash = _tensor_sha256(factual.log_m0)
            ref_log_m0_hash = _tensor_sha256(reference.log_m0)
            fact_noise_hash = (
                _tensor_sha256(factual.noise_steps)
                if factual.noise_steps is not None
                else "not_stored_terminal_only"
            )
            ref_noise_hash = (
                _tensor_sha256(reference.noise_steps)
                if reference.noise_steps is not None
                else "not_stored_terminal_only"
            )
            fact_log_mass = float(_terminal_log_mass(factual)[0].item())
            ref_log_mass = float(_terminal_log_mass(reference)[0].item())
            fact_mean = _weighted_mean(factual.terminal_z[0], factual.terminal_logw[0])
            ref_mean = _weighted_mean(reference.terminal_z[0], reference.terminal_logw[0])
            mean_shift_l2 = float(torch.linalg.norm(fact_mean - ref_mean).item())
            energy_distance = _weighted_energy_distance(
                factual.terminal_z[0],
                factual.terminal_logw[0],
                reference.terminal_z[0],
                reference.terminal_logw[0],
            )
            fact_program = _program_summary(model, factual, state_labels)
            ref_program = _program_summary(model, reference, state_labels)
            fact_program_fractions = _program_fractions(model, factual)
            ref_program_fractions = _program_fractions(model, reference)
            fact_actions = _action_summary(factual)
            ref_actions = _action_summary(reference)
            terminal_entropy_factual = _entropy(factual.terminal_logw[0])
            terminal_entropy_reference = _entropy(reference.terminal_logw[0])
            row = {
                "perturbation_id": pid,
                "target_gene": infer_target_gene(pid),
                "sgRNA_id": pid,
                "is_control": pid in controls,
                "fold_id": fold_id,
                "run_dir": str(run_dir),
                "source_split": args.source_split,
                "n_p4": int(endpoint.initial[pid].n_atoms),
                "n_p60": int(endpoint.terminal[pid].n_atoms),
                "log_mass_factual": fact_log_mass,
                "log_mass_reference": ref_log_mass,
                "delta_log_mass_fact_vs_ref": fact_log_mass - ref_log_mass,
                "mass_ratio_fact_vs_ref": float(np.exp(fact_log_mass - ref_log_mass)),
                "weighted_mean_shift_l2_fact_vs_ref": mean_shift_l2,
                "energy_distance_fact_vs_ref": energy_distance,
                "geometry_shift_l2": mean_shift_l2,
                "legacy_geom_shift_fact_vs_ref": mean_shift_l2,
                "geom_shift_fact_vs_ref": mean_shift_l2,
                "geometry_metric": "weighted_mean_l2",
                "control_rollout_mode": "reference_consistent",
                "counterfactual_context_protocol": "global_full_context",
                "initial_particles_sha256_factual": fact_z0_hash,
                "initial_particles_sha256_reference": ref_z0_hash,
                "initial_logw_sha256_factual": fact_logw0_hash,
                "initial_logw_sha256_reference": ref_logw0_hash,
                "initial_log_m0_sha256_factual": fact_log_m0_hash,
                "initial_log_m0_sha256_reference": ref_log_m0_hash,
                "noise_seed": global_metadata_by_pid[pid].get("noise_seed"),
                "noise_sha256_factual": fact_noise_hash,
                "noise_sha256_reference": ref_noise_hash,
                "same_initial_particles": fact_z0_hash == ref_z0_hash,
                "same_initial_logw": fact_logw0_hash == ref_logw0_hash,
                "same_initial_log_m0": fact_log_m0_hash == ref_log_m0_hash,
                "common_noise": True if args.terminal_only else fact_noise_hash == ref_noise_hash,
                "terminal_entropy_factual": terminal_entropy_factual,
                "terminal_entropy_reference": terminal_entropy_reference,
                "terminal_state_entropy_fact": terminal_entropy_factual,
                "terminal_state_entropy_ref": terminal_entropy_reference,
                "terminal_entropy_delta_fact_vs_ref": terminal_entropy_factual - terminal_entropy_reference,
                "dominant_program_factual": fact_program.get("dominant_program_label", fact_program["dominant_program_index"]),
                "dominant_program_reference": ref_program.get("dominant_program_label", ref_program["dominant_program_index"]),
                "program_fraction_shift_abs": abs(
                    fact_program["dominant_program_fraction"] - ref_program["dominant_program_fraction"]
                ),
                "program_occupancy_tv_fact_vs_ref": float(
                    0.5 * torch.abs(fact_program_fractions - ref_program_fractions).sum().item()
                ),
            }
            row.update(row_context_metadata)
            for key, value in global_metadata_by_pid[pid].items():
                if key != "context_missing_perturbations" and isinstance(value, (str, int, float, bool, type(None))):
                    row.setdefault(key, value)
            for key, value in fact_actions.items():
                row[key] = value
                row[f"{key}_fact"] = value
            for key, value in ref_actions.items():
                row[f"{key}_ref"] = value
            row.update(_action_deltas(fact_actions, ref_actions))
            if clamped is not None:
                clamped_mean = _weighted_mean(clamped.terminal_z[0], clamped.terminal_logw[0])
                context_geom = float(torch.linalg.norm(fact_mean - clamped_mean).item())
                row["log_mass_clamped_context"] = float(_terminal_log_mass(clamped)[0].item())
                context_mass = row["log_mass_factual"] - row["log_mass_clamped_context"]
                row["context_dependence"] = context_geom
                row["context_dependence_geom"] = context_geom
                row["delta_log_mass_self_vs_clamped"] = context_mass
                row["context_dependence_mass"] = abs(context_mass)
                row["terminal_state_entropy_clamped"] = _entropy(clamped.terminal_logw[0])
                if clamped.noise_steps is None:
                    raise RuntimeError("Clamped counterfactual rollout did not return noise_steps for provenance.")
                clamped_noise_hash = _tensor_sha256(clamped.noise_steps)
                row["noise_sha256_clamped_context"] = clamped_noise_hash
                row["clamped_same_noise_as_factual"] = clamped_noise_hash == fact_noise_hash
                row["clamped_same_initial_particles"] = _tensor_sha256(clamped.z_steps[0]) == fact_z0_hash
                row["clamped_same_initial_logw"] = _tensor_sha256(clamped.logw_steps[0]) == fact_logw0_hash
                row["clamped_same_initial_log_m0"] = _tensor_sha256(clamped.log_m0) == fact_log_m0_hash
                for key, value in _action_summary(clamped).items():
                    row[f"{key}_clamped_context"] = value
            if reference_clamped is not None:
                reference_clamped_mean = _weighted_mean(
                    reference_clamped.terminal_z[0],
                    reference_clamped.terminal_logw[0],
                )
                row["log_mass_reference_clamped_context"] = float(_terminal_log_mass(reference_clamped)[0].item())
                row["geom_shift_ref_vs_ref_clamped"] = float(torch.linalg.norm(ref_mean - reference_clamped_mean).item())
                row["delta_log_mass_ref_vs_ref_clamped"] = row["log_mass_reference"] - row["log_mass_reference_clamped_context"]
                row["terminal_state_entropy_reference_clamped"] = _entropy(reference_clamped.terminal_logw[0])
                if reference_clamped.noise_steps is None:
                    raise RuntimeError("Reference-clamped counterfactual rollout did not return noise_steps for provenance.")
                row["noise_sha256_reference_clamped_context"] = _tensor_sha256(reference_clamped.noise_steps)
            if args.context_clamped and clamped is not None:
                for flag in [
                    "clamped_same_noise_as_factual",
                    "clamped_same_initial_particles",
                    "clamped_same_initial_logw",
                    "clamped_same_initial_log_m0",
                ]:
                    if not bool(row[flag]):
                        raise RuntimeError(f"Clamped counterfactual invariant failed for {pid}: {flag}")
            for flag in ["same_initial_particles", "same_initial_logw", "same_initial_log_m0", "common_noise"]:
                if not bool(row[flag]):
                    raise RuntimeError(f"Counterfactual invariant failed for {pid}: {flag}")
            rows.append(row)
            continue

        seed = int(args.seed + i)
        z0, logw0, log_m0 = initialise_particles(
            endpoint,
            [pid],
            n_particles=args.n_particles,
            device=device,
            seed=seed,
        )
        noise_seed = seed + 100_000
        noise_steps = simulator.sample_noise_like(z0, args.n_steps, seed=noise_seed)
        noise_steps_original = noise_steps.clone()
        factual = simulator.rollout(
            z0,
            logw0,
            model,
            log_m0,
            perturbation_ids=[pid],
            noise_steps=noise_steps,
            return_noise_used=True,
        )
        ref_noise_steps = noise_steps.clone()
        with _control_embedding_context(model, pid, mode="reference_consistent"):
            reference = simulator.rollout(
                z0.clone(),
                logw0.clone(),
                model,
                log_m0.clone(),
                perturbation_ids=[pid],
                noise_steps=ref_noise_steps,
                return_noise_used=True,
            )

        clamped = None
        reference_clamped = None
        if args.context_clamped and reference.context_steps is not None:
            if (
                getattr(model, "transformer_growth_only", False)
                and getattr(model, "meanfield_context_agg", None) is not None
                and (reference.base_context_steps is None or reference.growth_context_steps is None)
            ):
                raise ValueError("Reference rollout did not store base/growth context steps for clamped context.")
            tau_grid = reference.tau_steps.detach()
            tau_start = float(tau_grid[0].item())
            tau_end = float(tau_grid[-1].item())
            clamped_noise_steps = noise_steps.clone()
            clamped = rollout_with_clamped_context(
                model=model,
                z0=z0.clone(),
                logw0=logw0.clone(),
                log_m0=log_m0.clone(),
                perturbation_ids=[pid],
                context_steps=reference.context_steps,
                base_context_steps=reference.base_context_steps,
                growth_context_steps=reference.growth_context_steps,
                tau_start=tau_start,
                tau_end=tau_end,
                tau_grid=tau_grid,
                noise_steps=clamped_noise_steps,
                return_noise_used=True,
            )
            reference_clamped_noise_steps = noise_steps.clone()
            with _control_embedding_context(model, pid, mode="reference_consistent"):
                reference_clamped = rollout_with_clamped_context(
                    model=model,
                    z0=z0.clone(),
                    logw0=logw0.clone(),
                    log_m0=log_m0.clone(),
                    perturbation_ids=[pid],
                    context_steps=reference.context_steps,
                    base_context_steps=reference.base_context_steps,
                    growth_context_steps=reference.growth_context_steps,
                    tau_start=tau_start,
                    tau_end=tau_end,
                    tau_grid=tau_grid,
                    noise_steps=reference_clamped_noise_steps,
                    return_noise_used=True,
                )

        fact_z0_hash = _tensor_sha256(factual.z_steps[0])
        ref_z0_hash = _tensor_sha256(reference.z_steps[0])
        fact_logw0_hash = _tensor_sha256(factual.logw_steps[0])
        ref_logw0_hash = _tensor_sha256(reference.logw_steps[0])
        fact_log_m0_hash = _tensor_sha256(factual.log_m0)
        ref_log_m0_hash = _tensor_sha256(reference.log_m0)
        if factual.noise_steps is None or reference.noise_steps is None:
            raise RuntimeError("Counterfactual rollout did not return noise_steps for provenance.")
        fact_noise_hash = _tensor_sha256(factual.noise_steps)
        ref_noise_hash = _tensor_sha256(reference.noise_steps)
        fact_log_mass = float(_terminal_log_mass(factual)[0].item())
        ref_log_mass = float(_terminal_log_mass(reference)[0].item())
        fact_mean = _weighted_mean(factual.terminal_z[0], factual.terminal_logw[0])
        ref_mean = _weighted_mean(reference.terminal_z[0], reference.terminal_logw[0])
        mean_shift_l2 = float(torch.linalg.norm(fact_mean - ref_mean).item())
        energy_distance = _weighted_energy_distance(
            factual.terminal_z[0],
            factual.terminal_logw[0],
            reference.terminal_z[0],
            reference.terminal_logw[0],
        )
        fact_program = _program_summary(model, factual, state_labels)
        ref_program = _program_summary(model, reference, state_labels)
        program_shift = abs(fact_program["dominant_program_fraction"] - ref_program["dominant_program_fraction"])
        fact_program_fractions = _program_fractions(model, factual)
        ref_program_fractions = _program_fractions(model, reference)
        program_occupancy_tv = float(0.5 * torch.abs(fact_program_fractions - ref_program_fractions).sum().item())
        fact_actions = _action_summary(factual)
        ref_actions = _action_summary(reference)
        action_deltas = _action_deltas(fact_actions, ref_actions)
        terminal_entropy_factual = _entropy(factual.terminal_logw[0])
        terminal_entropy_reference = _entropy(reference.terminal_logw[0])
        row = {
            "perturbation_id": pid,
            "target_gene": infer_target_gene(pid),
            "sgRNA_id": pid,
            "is_control": pid in controls,
            "fold_id": fold_id,
            "run_dir": str(run_dir),
            "source_split": args.source_split,
            "n_p4": int(endpoint.initial[pid].n_atoms),
            "n_p60": int(endpoint.terminal[pid].n_atoms),
            "log_mass_factual": fact_log_mass,
            "log_mass_reference": ref_log_mass,
            "delta_log_mass_fact_vs_ref": fact_log_mass - ref_log_mass,
            "mass_ratio_fact_vs_ref": float(np.exp(fact_log_mass - ref_log_mass)),
            "weighted_mean_shift_l2_fact_vs_ref": mean_shift_l2,
            "energy_distance_fact_vs_ref": energy_distance,
            "geometry_shift_l2": mean_shift_l2,
            "legacy_geom_shift_fact_vs_ref": mean_shift_l2,
            "geom_shift_fact_vs_ref": mean_shift_l2,
            "geometry_metric": "weighted_mean_l2",
            "control_rollout_mode": "reference_consistent",
            "initial_particles_sha256_factual": fact_z0_hash,
            "initial_particles_sha256_reference": ref_z0_hash,
            "initial_logw_sha256_factual": fact_logw0_hash,
            "initial_logw_sha256_reference": ref_logw0_hash,
            "initial_log_m0_sha256_factual": fact_log_m0_hash,
            "initial_log_m0_sha256_reference": ref_log_m0_hash,
            "noise_seed": noise_seed,
            "noise_sha256_factual": fact_noise_hash,
            "noise_sha256_reference": ref_noise_hash,
            "same_initial_particles": fact_z0_hash == ref_z0_hash,
            "same_initial_logw": fact_logw0_hash == ref_logw0_hash,
            "same_initial_log_m0": fact_log_m0_hash == ref_log_m0_hash,
            "common_noise": fact_noise_hash == ref_noise_hash,
            "terminal_entropy_factual": terminal_entropy_factual,
            "terminal_entropy_reference": terminal_entropy_reference,
            "terminal_state_entropy_fact": terminal_entropy_factual,
            "terminal_state_entropy_ref": terminal_entropy_reference,
            "terminal_entropy_delta_fact_vs_ref": terminal_entropy_factual - terminal_entropy_reference,
            "dominant_program_factual": fact_program.get("dominant_program_label", fact_program["dominant_program_index"]),
            "dominant_program_reference": ref_program.get("dominant_program_label", ref_program["dominant_program_index"]),
            "program_fraction_shift_abs": program_shift,
            "program_occupancy_tv_fact_vs_ref": program_occupancy_tv,
        }
        row.update(row_context_metadata)
        for key, value in fact_actions.items():
            row[key] = value
            row[f"{key}_fact"] = value
        for key, value in ref_actions.items():
            row[f"{key}_ref"] = value
        row.update(action_deltas)
        if clamped is not None:
            clamped_mean = _weighted_mean(clamped.terminal_z[0], clamped.terminal_logw[0])
            context_geom = float(torch.linalg.norm(fact_mean - clamped_mean).item())
            row["log_mass_clamped_context"] = float(_terminal_log_mass(clamped)[0].item())
            context_mass = row["log_mass_factual"] - row["log_mass_clamped_context"]
            row["context_dependence"] = context_geom
            row["context_dependence_geom"] = context_geom
            row["delta_log_mass_self_vs_clamped"] = row["log_mass_factual"] - row["log_mass_clamped_context"]
            row["context_dependence_mass"] = abs(context_mass)
            row["terminal_state_entropy_clamped"] = _entropy(clamped.terminal_logw[0])
            if clamped.noise_steps is None:
                raise RuntimeError("Clamped counterfactual rollout did not return noise_steps for provenance.")
            clamped_noise_hash = _tensor_sha256(clamped.noise_steps)
            row["noise_sha256_clamped_context"] = clamped_noise_hash
            row["clamped_same_noise_as_factual"] = clamped_noise_hash == fact_noise_hash
            row["clamped_same_initial_particles"] = _tensor_sha256(clamped.z_steps[0]) == fact_z0_hash
            row["clamped_same_initial_logw"] = _tensor_sha256(clamped.logw_steps[0]) == fact_logw0_hash
            row["clamped_same_initial_log_m0"] = _tensor_sha256(clamped.log_m0) == fact_log_m0_hash
            clamped_actions = _action_summary(clamped)
            for key, value in clamped_actions.items():
                row[f"{key}_clamped_context"] = value
        if reference_clamped is not None:
            reference_clamped_mean = _weighted_mean(
                reference_clamped.terminal_z[0],
                reference_clamped.terminal_logw[0],
            )
            row["log_mass_reference_clamped_context"] = float(_terminal_log_mass(reference_clamped)[0].item())
            row["geom_shift_ref_vs_ref_clamped"] = float(torch.linalg.norm(ref_mean - reference_clamped_mean).item())
            row["delta_log_mass_ref_vs_ref_clamped"] = row["log_mass_reference"] - row["log_mass_reference_clamped_context"]
            row["terminal_state_entropy_reference_clamped"] = _entropy(reference_clamped.terminal_logw[0])
            if reference_clamped.noise_steps is None:
                raise RuntimeError("Reference-clamped counterfactual rollout did not return noise_steps for provenance.")
            row["noise_sha256_reference_clamped_context"] = _tensor_sha256(reference_clamped.noise_steps)
        if not torch.equal(noise_steps, noise_steps_original):
            raise RuntimeError("Counterfactual rollout mutated explicit noise_steps.")
        if not torch.equal(ref_noise_steps, noise_steps_original):
            raise RuntimeError("Reference counterfactual rollout mutated explicit noise_steps.")
        if args.context_clamped and clamped is not None:
            for flag in [
                "clamped_same_noise_as_factual",
                "clamped_same_initial_particles",
                "clamped_same_initial_logw",
                "clamped_same_initial_log_m0",
            ]:
                if not bool(row[flag]):
                    raise RuntimeError(f"Clamped counterfactual invariant failed for {pid}: {flag}")
        for flag in ["same_initial_particles", "same_initial_logw", "same_initial_log_m0", "common_noise"]:
            if not bool(row[flag]):
                raise RuntimeError(f"Counterfactual invariant failed for {pid}: {flag}")
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("delta_log_mass_fact_vs_ref", ascending=False)
    out.to_csv(output_dir / "counterfactual_biology_effects.csv", index=False)
    manifest = {
        "run_dir": str(run_dir),
        "source_split": args.source_split,
        "n_particles": int(args.n_particles),
        "n_steps": int(args.n_steps),
        "seed": int(args.seed),
        "context_clamped": bool(args.context_clamped),
        "include_controls_for_null": bool(args.include_controls_for_null),
        "requested_mass_mode": requested_mass_mode,
        "resolved_mass_mode": data.mass_table.df.attrs.get("mass_mode"),
        "mass_mode_resolution_reason": data.mass_table.df.attrs.get("mass_mode_resolution_reason"),
        "control_rollout_mode": "reference_consistent",
        "same_initial_particles": bool(
            (out["initial_particles_sha256_factual"] == out["initial_particles_sha256_reference"]).all()
        ),
        "same_initial_logw": bool(
            (out["initial_logw_sha256_factual"] == out["initial_logw_sha256_reference"]).all()
        ),
        "same_initial_log_m0": bool(
            (out["initial_log_m0_sha256_factual"] == out["initial_log_m0_sha256_reference"]).all()
        ),
        "common_noise": bool((out["noise_sha256_factual"] == out["noise_sha256_reference"]).all()),
        "clamped_same_noise_as_factual": (
            bool(out["clamped_same_noise_as_factual"].all())
            if "clamped_same_noise_as_factual" in out.columns
            else None
        ),
        "n_counterfactual_rows": int(len(out)),
        **context_metadata,
        "reference_protocol": "delta_zero_soft_ref_reference_consistent",
        "geometry_metric": "weighted_mean_l2",
        "geometry_metric_note": (
            "weighted_mean_shift_l2_fact_vs_ref is a weighted terminal mean shift, "
            "not a full distributional Sinkhorn/Wasserstein distance."
        ),
    }
    (output_dir / "counterfactual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "counterfactual_biology_effects.csv")


if __name__ == "__main__":
    main()
