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

from cape.data.hnscc import (  # noqa: E402
    build_study_from_split,
    compute_state_centroids,
    load_hnscc,
    load_hnscc_obs,
    prepare_hnscc_obs,
)
from cape.models.full_model import FullDynamicsModel  # noqa: E402
from cape.models.simulator import _control_embedding_context, initialise_particles  # noqa: E402
from cape.models.weighted_sde import ParticleRollout, WeightedParticleSimulator  # noqa: E402

from hnscc_biology_common import infer_target_gene  # noqa: E402


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
    parser.add_argument("--context-clamped", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


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
    model = FullDynamicsModel(
        perturbation_ids=list(config["supported_perturbations"]),
        control_ids=list(config["control_ids"]),
        latent_dim=latent_dim,
        embedding_dim=int(config.get("embedding_dim", 8)),
        n_programs=int(config.get("resolved_n_programs", config.get("n_programs", 8))),
        mediator_dim=int(config.get("mediator_dim", 8)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        depth=int(config.get("depth", 3)),
        activation_checkpointing=False,
        ecological_growth=bool(config.get("ecological_growth", True)),
        use_growth_intercept=bool(config.get("use_growth_intercept", True)),
        shared_guide_embedding=bool(config.get("shared_guide_embedding", False)),
        program_centroids=program_centroids,
        program_assignment_scale=float(config.get("program_assignment_scale") or 1.0),
        control_mode=str(config.get("control_mode", "soft_ref")),
        control_ref_penalty=float(config.get("lambda_control_ref", 5e-4)),
    ).to(device)
    return model


def _terminal_log_mass(rollout: ParticleRollout) -> torch.Tensor:
    if rollout.log_m0 is None:
        raise ValueError("Rollout is missing log_m0.")
    return rollout.log_m0 + torch.logsumexp(rollout.terminal_logw, dim=1)


def _weighted_mean(z: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(logw, dim=-1)
    return (w.unsqueeze(-1) * z).sum(dim=-2)


def _entropy(logw: torch.Tensor) -> float:
    w = torch.softmax(logw, dim=-1)
    return float((-(w * torch.log(w + 1e-30)).sum(dim=-1)).item())


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
def _rollout_clamped_context(
    model: FullDynamicsModel,
    z0: torch.Tensor,
    logw0: torch.Tensor,
    log_m0: torch.Tensor,
    pid: str,
    context_steps: torch.Tensor,
    *,
    n_steps: int,
) -> ParticleRollout:
    device = z0.device
    dtype = z0.dtype
    tau_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    dtau = 1.0 / max(n_steps, 1)
    z = z0.clone()
    logw = logw0.clone()
    z_list = [z]
    logw_list = [logw]
    drift_list = []
    sigma_list = []
    growth_list = []

    for k in range(n_steps):
        tau_k = tau_steps[k]
        context = context_steps[k].to(device=device, dtype=dtype)
        n_programs = model.context_agg.n_programs
        q = context[:n_programs]
        s = context[n_programs:]
        a = model.embedding([pid])
        b = model.embedding.growth_intercepts([pid])
        eta_z = model.context_agg.encoder.eta(z)
        coeffs = model.coeff_nets(
            z=z,
            tau=tau_k,
            context=context,
            a=a,
            growth_intercept=b,
            eta_z=eta_z,
            q=q,
            s=s,
        )
        drift_list.append(coeffs.drift)
        sigma_list.append(coeffs.sigma_diag)
        growth_list.append(coeffs.growth)
        noise = torch.randn_like(z)
        z = z + coeffs.drift * dtau + coeffs.sigma_diag * (dtau ** 0.5) * noise
        logw = logw + coeffs.growth * dtau
        z_list.append(z)
        logw_list.append(logw)

    return ParticleRollout(
        z_steps=torch.stack(z_list, dim=0),
        logw_steps=torch.stack(logw_list, dim=0),
        tau_steps=tau_steps,
        log_m0=log_m0.detach().clone(),
        drift_steps=torch.stack(drift_list, dim=0),
        sigma_steps=torch.stack(sigma_list, dim=0),
        growth_steps=torch.stack(growth_list, dim=0),
        context_steps=context_steps.detach().clone(),
    )


@torch.no_grad()
def _program_summary(model: FullDynamicsModel, rollout: ParticleRollout, labels: list[str] | None) -> dict:
    eta = model.context_agg.encoder.eta(rollout.terminal_z)[0]
    w = torch.softmax(rollout.terminal_logw[0], dim=-1)
    q = (w.unsqueeze(-1) * eta).sum(dim=0).detach().cpu().numpy()
    idx = int(q.argmax())
    out = {
        "dominant_program_index": idx,
        "dominant_program_fraction": float(q[idx]),
        "program_entropy": float(-(q * np.log(q + 1e-30)).sum()),
    }
    if labels and idx < len(labels):
        out["dominant_program_label"] = labels[idx]
    return out


def main() -> None:
    args = parse_args()
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
    data = build_study_from_split(
        obs,
        latent,
        split=source_split,
        split_name=split_name,
        mass_value_col=config.get("mass_value_col"),
        mass_scope=config.get("mass_scope", "subset_only"),
    )
    supported = [pid for pid in config["supported_perturbations"] if pid in data.catalog.perturbation_ids]
    requested = [pid.strip() for pid in args.perturbations.split(",") if pid.strip()]
    if requested:
        supported = [pid for pid in supported if pid in set(requested)]
    controls = set(config.get("control_ids", []))
    supported = [pid for pid in supported if pid not in controls]
    if args.max_perturbations > 0:
        supported = supported[: args.max_perturbations]
    if not supported:
        raise ValueError("No non-control perturbations selected for counterfactual analysis.")
    endpoint = data.to_endpoint_problem(supported, initial_label="P4", terminal_label="P60")

    model = _build_model(config, latent.shape[1], program_centroids, device)
    checkpoint = _checkpoint_path(run_dir, results)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    simulator = WeightedParticleSimulator(n_steps=args.n_steps, store_history=True)

    state_labels = config.get("state_labels") if bool(config.get("use_state_centroids", False)) else None
    rows = []
    for i, pid in enumerate(supported):
        seed = int(args.seed + i)
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

        clamped = None
        if args.context_clamped and reference.context_steps is not None:
            torch.manual_seed(seed)
            clamped = _rollout_clamped_context(
                model,
                z0.clone(),
                logw0.clone(),
                log_m0.clone(),
                pid,
                reference.context_steps,
                n_steps=args.n_steps,
            )

        fact_log_mass = float(_terminal_log_mass(factual)[0].item())
        ref_log_mass = float(_terminal_log_mass(reference)[0].item())
        fact_mean = _weighted_mean(factual.terminal_z[0], factual.terminal_logw[0])
        ref_mean = _weighted_mean(reference.terminal_z[0], reference.terminal_logw[0])
        geom_shift = float(torch.linalg.norm(fact_mean - ref_mean).item())
        fact_program = _program_summary(model, factual, state_labels)
        ref_program = _program_summary(model, reference, state_labels)
        program_shift = abs(fact_program["dominant_program_fraction"] - ref_program["dominant_program_fraction"])
        fact_actions = _action_summary(factual)
        row = {
            "perturbation_id": pid,
            "target_gene": infer_target_gene(pid),
            "source_split": args.source_split,
            "n_p4": int(endpoint.initial[pid].n_atoms),
            "n_p60": int(endpoint.terminal[pid].n_atoms),
            "log_mass_factual": fact_log_mass,
            "log_mass_reference": ref_log_mass,
            "delta_log_mass_fact_vs_ref": fact_log_mass - ref_log_mass,
            "mass_ratio_fact_vs_ref": float(np.exp(fact_log_mass - ref_log_mass)),
            "geometry_shift_l2": geom_shift,
            "geom_shift_fact_vs_ref": geom_shift,
            "terminal_entropy_factual": _entropy(factual.terminal_logw[0]),
            "terminal_entropy_reference": _entropy(reference.terminal_logw[0]),
            "dominant_program_factual": fact_program.get("dominant_program_label", fact_program["dominant_program_index"]),
            "dominant_program_reference": ref_program.get("dominant_program_label", ref_program["dominant_program_index"]),
            "program_fraction_shift_abs": program_shift,
        }
        for key, value in fact_actions.items():
            row[key] = value
            row[f"{key}_fact"] = value
        if clamped is not None:
            clamped_mean = _weighted_mean(clamped.terminal_z[0], clamped.terminal_logw[0])
            context_geom = float(torch.linalg.norm(fact_mean - clamped_mean).item())
            row["log_mass_clamped_context"] = float(_terminal_log_mass(clamped)[0].item())
            context_mass = row["log_mass_factual"] - row["log_mass_clamped_context"]
            row["context_dependence"] = context_geom
            row["context_dependence_geom"] = context_geom
            row["delta_log_mass_self_vs_clamped"] = row["log_mass_factual"] - row["log_mass_clamped_context"]
            row["context_dependence_mass"] = abs(context_mass)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("delta_log_mass_fact_vs_ref", ascending=False)
    out.to_csv(output_dir / "counterfactual_biology_effects.csv", index=False)
    print(output_dir / "counterfactual_biology_effects.csv")


if __name__ == "__main__":
    main()
