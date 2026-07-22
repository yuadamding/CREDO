"""Deterministic inference replay for imported transformer-v2 runs."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ...contracts import Axis, CREDOStudy, FiniteMeasure, MassSemantics
from ...objective import checkpoint_geometry_mass_loss
from ...particles import (
    ParticleRollout,
    ParticleState,
    sample_initial_particles,
    weight_diagnostics,
)
from .importer import (
    ImportedTransformerV2Run,
    import_legacy_checkpoint,
    sha256_file,
)
from .model import FullDynamicsModel


def historical_axis_grid(
    axis: Axis,
    steps_per_interval: int,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Reproduce v2's piecewise torch.linspace checkpoint grid."""
    pieces = []
    for start, stop in zip(axis.normalized_values[:-1], axis.normalized_values[1:], strict=False):
        segment = torch.linspace(
            start,
            stop,
            steps_per_interval + 1,
            device=device,
            dtype=torch.float32,
        )
        if pieces:
            segment = segment[1:]
        pieces.append(segment)
    return torch.cat(pieces)


def load_lps_replay_study(
    run: ImportedTransformerV2Run,
    study_source: str | Path,
) -> CREDOStudy:
    """Reconstruct held-out finite measures from preserved rows and latent cache."""
    import anndata as ad

    source_path = Path(study_source).expanduser().resolve()
    adata = ad.read_h5ad(source_path, backed="r")
    obs = adata.obs.copy().reset_index(drop=True)
    latents = np.load(run.latents_path, mmap_mode="r")
    if len(obs) != len(latents):
        raise ValueError("Study rows and preserved latent cache are not aligned.")
    required = {
        "sample_id",
        "time_label",
        "perturbation_id",
        "embedding_id",
        "is_control",
        "mass_value",
    }
    missing = required - set(obs)
    if missing:
        raise ValueError(f"Replay study is missing obs columns: {sorted(missing)}")
    for column in ("sample_id", "time_label", "perturbation_id", "embedding_id"):
        obs[column] = obs[column].astype(str)
    obs["mass_value"] = pd.to_numeric(obs["mass_value"], errors="raise")
    axis_payload = run.envelope.study_contract["axis"]
    axis = Axis(
        kind="physical",
        source=str(axis_payload["source"]),
        labels=tuple(str(value) for value in axis_payload["labels"]),
        values=tuple(float(value) for value in axis_payload["values"]),
    )
    validation_samples = set(run.split.validation_values or ())
    selected = obs["sample_id"].isin(validation_samples) & obs["time_label"].isin(axis.labels)
    positions = np.flatnonzero(selected.to_numpy())
    scoped = obs.iloc[positions].copy()
    scoped["_position"] = positions
    grouped = scoped.groupby(
        ["sample_id", "perturbation_id", "time_label"],
        observed=True,
        sort=False,
    ).indices
    source_pairs = sorted(
        {
            (str(sample_id), str(perturbation_id))
            for sample_id, perturbation_id, label in grouped
            if str(label) == axis.source and str(perturbation_id) in set(run.model.perturbation_ids)
        },
        key=str,
    )
    if not source_pairs:
        raise ValueError("No held-out source measures match the imported embedding catalog.")

    measures: dict[str, dict[str, FiniteMeasure]] = {label: {} for label in axis.labels}
    metadata_rows: list[dict[str, Any]] = []
    for sample_id, perturbation_id in source_pairs:
        source_rows = scoped.iloc[grouped[(sample_id, perturbation_id, axis.source)]]
        embedding_values = source_rows["embedding_id"].unique().tolist()
        if len(embedding_values) != 1:
            raise ValueError("One replay measure maps to multiple embedding IDs.")
        embedding_id = str(embedding_values[0])
        measure_id = str((sample_id, perturbation_id))
        is_control = bool(source_rows["is_control"].astype(bool).all())
        metadata_rows.append(
            {
                "measure_id": measure_id,
                "sample_id": sample_id,
                "perturbation_id": perturbation_id,
                "guide_id": perturbation_id,
                "embedding_id": embedding_id,
                "target_gene": embedding_id,
                "context_group_id": run.split.split_id or "legacy-validation",
                "is_control": is_control,
            }
        )
        for label in axis.labels:
            key = (sample_id, perturbation_id, label)
            if key not in grouped:
                continue
            rows = scoped.iloc[grouped[key]]
            row_positions = rows["_position"].to_numpy(dtype=np.int64)
            total_mass = float(rows["mass_value"].sum())
            weights = np.full(len(rows), total_mass / len(rows), dtype=np.float64)
            measures[label][measure_id] = FiniteMeasure(
                np.asarray(latents[row_positions], dtype=np.float32),
                weights,
                total_mass,
            )
    representation = run.representation
    all_samples = tuple(sorted(obs["sample_id"].unique()))
    if representation.included_samples != all_samples:
        representation = replace(representation, included_samples=all_samples)
    study = CREDOStudy(
        axis=axis,
        measures=measures,
        measure_meta=pd.DataFrame(metadata_rows),
        mass_semantics=MassSemantics.RELATIVE_WITHIN_GROUP,
        metadata={
            "input_paths": {
                "study_source": str(source_path),
                "latents": str(run.latents_path),
            },
            "input_hashes": {
                "latents": sha256_file(run.latents_path),
            },
            "dataset": {"name": "archived_lps_90m_6h_10h"},
            "mass_denominators": [
                f"{sample_id}::{label}::all_cell_states"
                for sample_id in sorted(validation_samples)
                for label in axis.labels
            ],
        },
        representation=representation,
    )
    run.study = study
    return study


def _sample_noise(
    state: ParticleState,
    grid: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=state.z.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (len(grid) - 1,) + tuple(state.z.shape),
        device=state.z.device,
        dtype=state.z.dtype,
        generator=generator,
    )


@torch.no_grad()
def rollout_transformer_v2(
    model: FullDynamicsModel,
    initial_state: ParticleState,
    grid: torch.Tensor,
    *,
    noise: torch.Tensor,
) -> ParticleRollout:
    integration_grid = grid.to(device=initial_state.z.device, dtype=initial_state.z.dtype)
    expected = (len(integration_grid) - 1,) + tuple(initial_state.z.shape)
    if tuple(noise.shape) != expected:
        raise ValueError(f"noise must have shape {expected}.")
    z = initial_state.z.clone()
    logw = initial_state.logw.clone()
    z_steps = [z]
    logw_steps = [logw]
    drift_steps = []
    diffusion_steps = []
    growth_steps = []
    context_steps = []
    for step in range(len(integration_grid) - 1):
        coefficients, context = model.step(
            z,
            integration_grid[step],
            logw,
            initial_state.log_m0,
            list(initial_state.embedding_ids),
        )
        dt = integration_grid[step + 1] - integration_grid[step]
        z = z + coefficients.drift * dt + coefficients.sigma_diag * torch.sqrt(dt) * noise[step]
        logw = logw + coefficients.growth * dt
        z_steps.append(z)
        logw_steps.append(logw)
        drift_steps.append(coefficients.drift)
        diffusion_steps.append(coefficients.sigma_diag)
        growth_steps.append(coefficients.growth)
        context_steps.append(context.context.detach())
    return ParticleRollout(
        z_steps=torch.stack(z_steps),
        logw_steps=torch.stack(logw_steps),
        log_m0=initial_state.log_m0,
        axis_grid=grid.to(device=initial_state.z.device, dtype=torch.float32),
        measure_ids=initial_state.measure_ids,
        embedding_ids=initial_state.embedding_ids,
        context_group_ids=initial_state.context_group_ids,
        measure_indices=initial_state.measure_indices,
        residual_scale=initial_state.residual_scale,
        drift_steps=torch.stack(drift_steps),
        sigma_steps=torch.stack(diffusion_steps),
        growth_steps=torch.stack(growth_steps),
        context_steps=torch.stack(context_steps),
        noise_steps=noise,
    )


def _float_rollout(rollout: ParticleRollout) -> ParticleRollout:
    return ParticleRollout(
        z_steps=rollout.z_steps.float(),
        logw_steps=rollout.logw_steps.float(),
        log_m0=rollout.log_m0.float(),
        axis_grid=rollout.axis_grid.float(),
        measure_ids=rollout.measure_ids,
        embedding_ids=rollout.embedding_ids,
        context_group_ids=rollout.context_group_ids,
        measure_indices=rollout.measure_indices,
        residual_scale=rollout.residual_scale.float(),
        drift_steps=rollout.drift_steps.float(),
        sigma_steps=rollout.sigma_steps.float(),
        growth_steps=rollout.growth_steps.float(),
        context_steps=rollout.context_steps.float(),
        noise_steps=rollout.noise_steps.float(),
    )


@torch.no_grad()
def evaluate_replay(
    run: ImportedTransformerV2Run,
    study: CREDOStudy,
    *,
    particles: int = 640,
    steps_per_interval: int = 24,
    seed: int = 0,
    noise_seed: int | None = None,
    device: str | torch.device | None = None,
    compute_geometry: bool = True,
) -> tuple[pd.DataFrame, ParticleRollout]:
    """Run one deterministic held-out replay through the common metric contract."""
    run.require("evaluate")
    if particles < 2 or steps_per_interval < 1 or seed < 0:
        raise ValueError("Replay particles, integration steps, and seed are invalid.")
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    run.model.to(selected_device, dtype=torch.float32).eval()
    grid = historical_axis_grid(study.axis, steps_per_interval, device=selected_device)
    source = sample_initial_particles(
        study,
        study.measure_ids,
        particles,
        device=selected_device,
        dtype=dtype,
        seed=seed,
    )
    resolved_noise_seed = seed + 1_000_003 if noise_seed is None else int(noise_seed)
    noise = _sample_noise(source, grid, seed=resolved_noise_seed)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if selected_device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        particle_rollout = rollout_transformer_v2(run.model, source, grid, noise=noise)
    float_rollout = _float_rollout(particle_rollout)
    checkpoint = checkpoint_geometry_mass_loss(
        float_rollout,
        study,
        mass_weight=1.0,
        include_mass=True,
        validation_source="held_out",
        sinkhorn_epsilon=0.1,
    )
    rows = checkpoint.rows
    if not compute_geometry:
        for row in rows:
            row["geometry"] = float("nan")
    order = {value: index for index, value in enumerate(study.measure_ids)}
    time_order = {value: index for index, value in enumerate(study.axis.labels)}
    frame = pd.DataFrame(rows)
    frame["_measure_order"] = frame["measure_id"].map(order)
    frame["_time_order"] = frame["time_label"].map(time_order)
    frame = frame.sort_values(["_measure_order", "_time_order"]).drop(
        columns=["_measure_order", "_time_order"]
    )
    frame.insert(0, "recipe_id", run.recipe_id)
    frame.insert(1, "recipe_version", run.recipe_version)
    frame.insert(2, "representation_id", run.representation.representation_id)
    frame.insert(3, "split_id", run.split.split_id)
    frame["evaluation_particles"] = int(particles)
    frame["integration_steps"] = int(len(grid) - 1)
    frame["evaluation_seed"] = int(seed)
    frame["noise_seed"] = int(resolved_noise_seed)
    return frame.reset_index(drop=True), particle_rollout


def compare_with_archive(
    metrics: pd.DataFrame,
    archived_metrics: str | Path,
) -> dict[str, Any]:
    archived_path = Path(archived_metrics).expanduser().resolve()
    archived = pd.read_csv(archived_path, dtype={"sample_id": str})
    expected = archived[["measure_key", "time_label"]].rename(columns={"measure_key": "measure_id"})
    observed = metrics[["measure_id", "time_label"]]
    ordering_match = expected.reset_index(drop=True).equals(observed.reset_index(drop=True))
    joined = metrics.merge(
        archived,
        left_on=["measure_id", "time_label"],
        right_on=["measure_key", "time_label"],
        validate="one_to_one",
    )
    archived_log_mass = np.log(joined["predicted_mass"].to_numpy(dtype=float))
    current_log_mass = joined["predicted_log_mass"].to_numpy(dtype=float)
    difference = current_log_mass - archived_log_mass
    rank_correlation = float(
        pd.Series(current_log_mass).rank().corr(pd.Series(archived_log_mass).rank())
    )
    max_difference = float(np.max(np.abs(difference))) if len(difference) else float("nan")
    mean_difference = float(np.mean(np.abs(difference))) if len(difference) else float("nan")
    if ordering_match and max_difference <= 1e-6:
        agreement = "exact"
    elif ordering_match and max_difference <= 0.1:
        agreement = "tolerance-level"
    elif rank_correlation >= 0.9:
        agreement = "rank-level"
    else:
        agreement = "qualitative"
    geometry_by_time = (
        joined.groupby("time_label", observed=True)
        .agg(replayed_geometry=("geometry", "mean"), archived_geometry=("geom_loss", "first"))
        .reset_index()
    )
    return {
        "archived_metrics_sha256": sha256_file(archived_path),
        "archived_rows": int(len(archived)),
        "replayed_rows": int(len(metrics)),
        "row_count_match": len(archived) == len(metrics),
        "measure_order_match": ordering_match,
        "predicted_log_mass_max_abs_difference": max_difference,
        "predicted_log_mass_mean_abs_difference": mean_difference,
        "predicted_log_mass_rank_correlation": rank_correlation,
        "geometry_by_time": geometry_by_time.to_dict(orient="records"),
        "agreement": agreement,
    }


@torch.no_grad()
def counterfactual_replay(
    run: ImportedTransformerV2Run,
    study: CREDOStudy,
    measure_id: str,
    *,
    context_policy: str = "self_consistent",
    same_noise: bool = True,
    n_particles: int | None = None,
    seed: int | None = None,
    steps_per_interval: int = 24,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Exact full-group, same-start, same-noise v2 reference contrast."""
    from ...counterfactual import _energy_distance, _weighted_mean

    run.require("counterfactual")
    if context_policy != "self_consistent":
        raise ValueError(
            "transformer-v2 currently exposes exact self_consistent full-group context only."
        )
    if not same_noise:
        raise ValueError("CREDO reference counterfactuals require same_noise=True.")
    if measure_id not in study.measure_ids:
        raise KeyError(f"Unknown measure_id {measure_id!r}.")
    particles = 640 if n_particles is None else int(n_particles)
    if particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if seed is None:
        import hashlib

        seed = int(hashlib.sha256(measure_id.encode()).hexdigest()[:8], 16) % 1_000_000
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    run.model.to(selected_device, dtype=torch.float32).eval()
    grid = historical_axis_grid(study.axis, steps_per_interval, device=selected_device)
    source = sample_initial_particles(
        study,
        study.measure_ids,
        particles,
        device=selected_device,
        dtype=dtype,
        seed=seed,
    )
    noise = _sample_noise(source, grid, seed=seed + 1_000_003)
    if not torch.equal(source.z, source.z.clone()) or not torch.equal(noise, noise.clone()):
        raise AssertionError("Counterfactual source or noise cloning changed values.")
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if selected_device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        factual = rollout_transformer_v2(run.model, source, grid, noise=noise)

    local_index = study.measure_ids.index(measure_id)
    embedding_id = source.embedding_ids[local_index]
    embedding = run.model.embedding
    before = None if embedding.embeddings is None else embedding.embeddings.detach().clone()
    before_growth = (
        None if embedding.growth_bias is None else embedding.growth_bias.detach().clone()
    )
    residual_index = embedding._nc_to_local.get(embedding_id)
    try:
        if residual_index is not None:
            with torch.no_grad():
                embedding.embeddings[residual_index].zero_()
                if embedding.growth_bias is not None:
                    embedding.growth_bias[residual_index].zero_()
        if before is not None:
            after = embedding.embeddings.detach()
            unselected = torch.ones(len(after), dtype=torch.bool, device=after.device)
            if residual_index is not None:
                unselected[residual_index] = False
                if not torch.equal(after[residual_index], torch.zeros_like(after[residual_index])):
                    raise AssertionError("Selected perturbation residual was not removed.")
            if not torch.equal(before[unselected], after[unselected]):
                raise AssertionError("Counterfactual changed an unselected perturbation residual.")
        if before_growth is not None:
            after_growth = embedding.growth_bias.detach()
            unselected_growth = torch.ones(
                len(after_growth), dtype=torch.bool, device=after_growth.device
            )
            if residual_index is not None:
                unselected_growth[residual_index] = False
                if not torch.equal(
                    after_growth[residual_index],
                    torch.zeros_like(after_growth[residual_index]),
                ):
                    raise AssertionError("Selected perturbation growth residual was not removed.")
            if not torch.equal(before_growth[unselected_growth], after_growth[unselected_growth]):
                raise AssertionError("Counterfactual changed an unselected growth residual.")
        with autocast:
            reference = rollout_transformer_v2(run.model, source, grid, noise=noise)
    finally:
        if before is not None:
            with torch.no_grad():
                embedding.embeddings.copy_(before)
        if before_growth is not None:
            with torch.no_grad():
                embedding.growth_bias.copy_(before_growth)
    if (
        not torch.equal(factual.z_steps[0], reference.z_steps[0])
        or not torch.equal(factual.logw_steps[0], reference.logw_steps[0])
        or not torch.equal(factual.log_m0, reference.log_m0)
    ):
        raise AssertionError("Counterfactual branches did not use identical source particles.")
    if not torch.equal(factual.noise_steps, reference.noise_steps):
        raise AssertionError("Counterfactual branches did not use identical Brownian noise.")

    from ...particles import checkpoint_indices

    checkpoints = checkpoint_indices(study.axis, factual.axis_grid)
    factual_diagnostics = weight_diagnostics(factual.logw_steps)
    reference_diagnostics = weight_diagnostics(reference.logw_steps)
    rows = []
    for label in study.axis.labels[1:]:
        step = checkpoints[label]
        factual_support = factual.z_steps[step, local_index].float()
        reference_support = reference.z_steps[step, local_index].float()
        factual_weight = factual.absolute_log_weight_steps[step, local_index].float()
        reference_weight = reference.absolute_log_weight_steps[step, local_index].float()
        rows.append(
            {
                "recipe_id": run.recipe_id,
                "recipe_version": run.recipe_version,
                "representation_id": run.representation.representation_id,
                "split_id": run.split.split_id,
                "measure_id": measure_id,
                "time_label": label,
                "context_policy": context_policy,
                "delta_log_mass": float(
                    (
                        torch.logsumexp(factual_weight, dim=0)
                        - torch.logsumexp(reference_weight, dim=0)
                    ).cpu()
                ),
                "mean_shift_l2": float(
                    torch.linalg.vector_norm(
                        _weighted_mean(factual_support, factual_weight)
                        - _weighted_mean(reference_support, reference_weight)
                    ).cpu()
                ),
                "energy_distance": float(
                    _energy_distance(
                        factual_support,
                        factual_weight,
                        reference_support,
                        reference_weight,
                    ).cpu()
                ),
                "context_dependence_shift": float(
                    torch.linalg.vector_norm(
                        factual.context_steps[:step].float()
                        - reference.context_steps[:step].float(),
                        dim=-1,
                    )
                    .mean()
                    .cpu()
                ),
                "factual_ess": float(factual_diagnostics["ess"][step, local_index].cpu()),
                "reference_ess": float(reference_diagnostics["ess"][step, local_index].cpu()),
                "eval_particles": particles,
                "integration_steps": len(grid) - 1,
                "counterfactual_seed": seed,
                "checkpoint_sha256": run.envelope.import_provenance["source_checkpoint_sha256"],
            }
        )
    return pd.DataFrame(rows)


def write_replay(
    output: str | Path,
    metrics: pd.DataFrame,
    comparison: dict[str, Any],
) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(destination / "metrics.parquet", index=False)
    (destination / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def replay_lps_bundle(
    bundle_root: str | Path,
    study_source: str | Path,
    output: str | Path,
    *,
    folds: tuple[str, ...] | None = None,
    particles: int = 640,
    steps_per_interval: int = 24,
    noise_seed: int = 0,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Replay selected archived folds and write one standardized OOF table."""
    bundle = Path(bundle_root).expanduser().resolve()
    selected = folds or tuple(path.name for path in sorted(bundle.glob("fold*")))
    if not selected:
        raise ValueError("No replay folds were selected.")
    destination = Path(output).expanduser().resolve()
    resolved_study_source = Path(study_source).expanduser().resolve()
    study_source_hash = sha256_file(resolved_study_source)
    all_metrics = []
    fold_reports: dict[str, Any] = {}
    for fold_name in selected:
        fold = bundle / fold_name
        metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
        imported = import_legacy_checkpoint(
            fold / "checkpoint_best.pt",
            fold / "run_config.json",
            fold / "vae_artifact/vae_state_dict.pt",
            fold / "vae_artifact/latent_all_std.npy",
            output=destination / fold_name / "imported",
            model_state="raw",
        )
        if set(imported.split.train_values or ()) & set(imported.split.validation_values or ()):
            raise AssertionError("Held-out samples appear in dynamics training measures.")
        study = load_lps_replay_study(imported, resolved_study_source)
        evaluation_seed = 100_000 + int(metadata["selected_epoch"])
        metrics, _ = evaluate_replay(
            imported,
            study,
            particles=particles,
            steps_per_interval=steps_per_interval,
            seed=evaluation_seed,
            noise_seed=noise_seed,
            device=device,
        )
        metrics.insert(4, "fold", int(metadata["fold"]))
        comparison = compare_with_archive(metrics, fold / "predicted_metrics_by_key_time.csv")
        write_replay(destination / fold_name, metrics, comparison)
        all_metrics.append(metrics)
        fold_reports[fold_name] = {
            **comparison,
            "dynamics_train_samples": list(imported.split.train_values or ()),
            "held_out_samples": list(imported.split.validation_values or ()),
            "representation_scope": imported.split.representation_scope,
            "representation_fit_scope": imported.representation.fit_scope,
            "evaluation_seed": evaluation_seed,
            "noise_seed": noise_seed,
        }
        imported.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    oof = pd.concat(all_metrics, ignore_index=True)
    oof.to_parquet(destination / "oof_metrics.parquet", index=False)
    archived_rows = sum(report["archived_rows"] for report in fold_reports.values())
    summary = {
        "recipe_id": "credo.transformer_sde_v2",
        "recipe_version": "2.0",
        "bundle_root": str(bundle),
        "study_source": str(resolved_study_source),
        "study_source_sha256": study_source_hash,
        "folds": list(selected),
        "particles": particles,
        "steps_per_interval": steps_per_interval,
        "oof_rows": int(len(oof)),
        "archived_oof_rows": int(archived_rows),
        "oof_row_count_match": len(oof) == archived_rows,
        "all_measure_orders_match": all(
            report["measure_order_match"] for report in fold_reports.values()
        ),
        "agreement": {fold: report["agreement"] for fold, report in fold_reports.items()},
        "fold_reports": fold_reports,
    }
    (destination / "replay_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "compare_with_archive",
    "counterfactual_replay",
    "evaluate_replay",
    "historical_axis_grid",
    "load_lps_replay_study",
    "rollout_transformer_v2",
    "replay_lps_bundle",
    "write_replay",
]
