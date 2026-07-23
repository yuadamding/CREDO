"""One same-start, same-noise reference counterfactual engine."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import torch

from .particles import (
    ClampedContextProvider,
    NoContextProvider,
    SelfConsistentContextProvider,
    checkpoint_indices,
    rollout,
    sample_initial_particles,
    sample_noise,
    weight_diagnostics,
)

if TYPE_CHECKING:
    from .training import Trainer


COMMON_COUNTERFACTUAL_COLUMNS = (
    "recipe_id",
    "recipe_version",
    "implementation_hash",
    "representation_id",
    "split_id",
    "measure_id",
    "time_label",
    "context_policy",
    "delta_log_mass",
    "mean_shift_l2",
    "energy_distance",
    "context_dependence_shift",
    "factual_ess",
    "reference_ess",
    "evaluation_particles",
    "integration_steps",
    "evaluation_seed",
    "noise_seed",
    "checkpoint_sha256",
    "package_version",
    "git_sha",
)
COUNTERFACTUAL_COLUMNS = list(COMMON_COUNTERFACTUAL_COLUMNS)


def validate_counterfactual_result(result: Any) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame):
        raise TypeError("A CREDO counterfactual runtime must return a pandas DataFrame.")
    missing = set(COMMON_COUNTERFACTUAL_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"Counterfactual result omitted common columns: {sorted(missing)}")
    key = [
        "recipe_id",
        "recipe_version",
        "split_id",
        "measure_id",
        "time_label",
        "context_policy",
        "evaluation_particles",
        "evaluation_seed",
        "noise_seed",
    ]
    if result.duplicated(key).any():
        raise ValueError("Counterfactual result contains duplicate recipe/measure/checkpoint rows.")
    if (result["evaluation_particles"] < 2).any():
        raise ValueError("Counterfactual result contains an invalid particle count.")
    if (
        (result["integration_steps"] < 1).any()
        or (result["evaluation_seed"] < 0).any()
        or (result["noise_seed"] < 0).any()
    ):
        raise ValueError("Counterfactual result contains invalid integration or seed provenance.")
    return result


def _weighted_mean(support: torch.Tensor, log_weight: torch.Tensor) -> torch.Tensor:
    return (torch.softmax(log_weight.float(), dim=0).to(support)[:, None] * support).sum(0)


def _energy_distance(
    factual_support: torch.Tensor,
    factual_log_weight: torch.Tensor,
    reference_support: torch.Tensor,
    reference_log_weight: torch.Tensor,
) -> torch.Tensor:
    factual_weight = torch.softmax(factual_log_weight.float(), dim=0).to(factual_support)
    reference_weight = torch.softmax(reference_log_weight.float(), dim=0).to(reference_support)
    cross = torch.cdist(factual_support, reference_support)
    factual_self = torch.cdist(factual_support, factual_support)
    reference_self = torch.cdist(reference_support, reference_support)
    cross_term = torch.einsum("i,ij,j->", factual_weight, cross, reference_weight)
    factual_term = torch.einsum("i,ij,j->", factual_weight, factual_self, factual_weight)
    reference_term = torch.einsum("i,ij,j->", reference_weight, reference_self, reference_weight)
    return (2 * cross_term - factual_term - reference_term).clamp_min(0)


@torch.no_grad()
def counterfactual(
    run: Trainer,
    measure_id: str,
    *,
    context_policy: Literal["self_consistent", "clamped"] = "self_consistent",
    same_noise: bool = True,
    n_particles: int | None = None,
    particles: int | None = None,
    seed: int | None = None,
    study: Any = None,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Compare one measure with the same-start continuation after removing its residual."""
    if n_particles is not None and particles is not None:
        raise ValueError("Specify only one of particles and n_particles.")
    resolved_particles = particles if particles is not None else n_particles
    require = getattr(run, "require", None)
    if callable(require):
        require("counterfactual")
    runtime_method = getattr(run, "counterfactual_runtime", None)
    if callable(runtime_method):
        runtime_options = {}
        if device is not None:
            runtime_options["device"] = device
        return validate_counterfactual_result(
            runtime_method(
                measure_id,
                context_policy=context_policy,
                same_noise=same_noise,
                n_particles=resolved_particles,
                seed=seed,
                study=study,
                **runtime_options,
            )
        )
    if device is not None and torch.device(device) != run.device:
        raise ValueError("Compact counterfactual device must match the loaded runtime device.")
    if study is not None and study is not run.data:
        raise ValueError("External counterfactual data must be loaded as a separate run.")
    if context_policy not in {"self_consistent", "clamped"}:
        raise ValueError("context_policy must be 'self_consistent' or 'clamped'.")
    if not same_noise:
        raise ValueError("CREDO reference counterfactuals require same_noise=True.")
    metadata = run.data.measure_meta.set_index("measure_id")
    if measure_id not in metadata.index:
        raise KeyError(f"Unknown measure_id {measure_id!r}.")
    group_id = metadata.loc[measure_id, "context_group_id"]
    if run.model.context_enabled:
        group_ids = tuple(
            value
            for value in run.data.measure_ids
            if metadata.loc[value, "context_group_id"] == group_id
        )
    else:
        group_ids = (measure_id,)
    local_index = group_ids.index(measure_id)
    particle_count = (
        run.settings.evaluation.particles if resolved_particles is None else int(resolved_particles)
    )
    if particle_count < 2:
        raise ValueError("n_particles must be at least 2.")
    if seed is None:
        digest = hashlib.sha256(measure_id.encode("utf-8")).hexdigest()
        seed = run.training_plan.seed + int(digest[:8], 16) % 1_000_000
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    source = sample_initial_particles(
        run.data,
        group_ids,
        particle_count,
        device=run.device,
        dtype=run.dtype,
        seed=seed,
    )
    factual_scale = torch.ones(len(group_ids), device=run.device, dtype=run.dtype)
    reference_scale = factual_scale.clone()
    reference_scale[local_index] = 0
    reference_mask = torch.zeros(len(group_ids), device=run.device, dtype=torch.bool)
    reference_mask[local_index] = True
    run.model.assert_reference_branch(source.embedding_ids, reference_mask)
    factual_source = source.with_residual_scale(factual_scale)
    reference_source = source.with_residual_scale(reference_scale)
    same_source = (
        torch.equal(factual_source.z, reference_source.z)
        and torch.equal(factual_source.logw, reference_source.logw)
        and torch.equal(factual_source.log_m0, reference_source.log_m0)
        and factual_source.measure_ids == reference_source.measure_ids
    )
    if not same_source:
        raise AssertionError("Counterfactual branches must have identical source particles.")
    noise = sample_noise(factual_source, run.grid, seed=seed + 1_000_003)
    reference_noise = noise
    if not torch.equal(noise, reference_noise):
        raise AssertionError("Counterfactual branches must use identical noise.")

    run.model.eval()
    if run.model.context_enabled:
        self_consistent = SelfConsistentContextProvider()
    else:
        self_consistent = NoContextProvider()
    reference_rollout = rollout(
        run.model,
        reference_source,
        run.grid,
        context_provider=self_consistent,
        noise=reference_noise,
    )
    factual_provider = self_consistent
    if context_policy == "clamped":
        factual_provider = ClampedContextProvider(reference_rollout.context_steps)
    factual_rollout = rollout(
        run.model,
        factual_source,
        run.grid,
        context_provider=factual_provider,
        noise=noise,
    )

    checkpoint = checkpoint_indices(run.data.axis, run.grid)
    factual_diagnostics = weight_diagnostics(factual_rollout.logw_steps)
    reference_diagnostics = weight_diagnostics(reference_rollout.logw_steps)
    evaluator = _evaluator_provenance(run)
    from .evaluation import compact_split_id
    from .training import _compact_recipe_contract

    recipe_contract = _compact_recipe_contract()
    rows = []
    for label in run.data.axis.labels[1:]:
        step = checkpoint[label]
        factual_support = factual_rollout.z_steps[step, local_index]
        reference_support = reference_rollout.z_steps[step, local_index]
        factual_weight = factual_rollout.absolute_log_weight_steps[step, local_index]
        reference_weight = reference_rollout.absolute_log_weight_steps[step, local_index]
        factual_mass = torch.logsumexp(factual_weight.float(), dim=0)
        reference_mass = torch.logsumexp(reference_weight.float(), dim=0)
        mean_shift = torch.linalg.vector_norm(
            _weighted_mean(factual_support, factual_weight)
            - _weighted_mean(reference_support, reference_weight)
        )
        if context_policy == "self_consistent":
            context_shift = torch.linalg.vector_norm(
                factual_rollout.context_steps[:step, local_index]
                - reference_rollout.context_steps[:step, local_index],
                dim=-1,
            ).mean()
        else:
            context_shift = factual_support.new_zeros(())
        rows.append(
            {
                "recipe_id": recipe_contract["id"],
                "recipe_version": recipe_contract["version"],
                "implementation_hash": recipe_contract["implementation_hash"],
                "representation_id": run.data.representation.representation_id,
                "split_id": compact_split_id(run),
                "measure_id": measure_id,
                "time_label": label,
                "context_policy": context_policy,
                "delta_log_mass": float((factual_mass - reference_mass).cpu()),
                "mean_shift_l2": float(mean_shift.cpu()),
                "energy_distance": float(
                    _energy_distance(
                        factual_support,
                        factual_weight,
                        reference_support,
                        reference_weight,
                    ).cpu()
                ),
                "context_dependence_shift": float(context_shift.cpu()),
                "factual_ess": float(factual_diagnostics["ess"][step, local_index].cpu()),
                "reference_ess": float(reference_diagnostics["ess"][step, local_index].cpu()),
                "evaluation_particles": particle_count,
                "integration_steps": len(run.grid) - 1,
                "evaluation_seed": seed,
                "noise_seed": seed + 1_000_003,
                **evaluator,
            }
        )
    frame = pd.DataFrame(rows, columns=COUNTERFACTUAL_COLUMNS)
    run.counterfactual_rows.extend(rows)
    _persist_if_saved(run)
    return validate_counterfactual_result(frame)


def _evaluator_provenance(run: Trainer) -> dict[str, str | None]:
    from . import __version__
    from .training import _git_state

    git_sha, _ = _git_state()
    return {
        "checkpoint_sha256": run.checkpoint_sha256,
        "package_version": __version__,
        "git_sha": git_sha,
    }


def _persist_if_saved(run: Trainer) -> None:
    output = Path(run.config.output)
    manifest_path = output / "run.json"
    if not manifest_path.exists():
        return
    path = output / "tables/counterfactuals.parquet"
    current = pd.DataFrame(run.counterfactual_rows, columns=COUNTERFACTUAL_COLUMNS)
    if path.exists():
        existing = pd.read_parquet(path)
        unknown = set(existing.columns) - set(COUNTERFACTUAL_COLUMNS)
        if unknown:
            raise ValueError(
                f"Existing counterfactuals.parquet has unknown columns: {sorted(unknown)}"
            )
        for column in COUNTERFACTUAL_COLUMNS:
            if column not in existing:
                existing[column] = pd.NA
        existing = existing[COUNTERFACTUAL_COLUMNS]
        if not existing.empty:
            current = pd.concat((existing, current), ignore_index=True)
    key = [
        "measure_id",
        "time_label",
        "context_policy",
        "evaluation_particles",
        "integration_steps",
        "evaluation_seed",
        "noise_seed",
        "checkpoint_sha256",
        "package_version",
        "git_sha",
    ]
    current = current.drop_duplicates(key, keep="last").reset_index(drop=True)
    run.counterfactual_rows = current.to_dict(orient="records")
    current.to_parquet(path, index=False)
    from .artifacts import write_compact_run_json

    write_compact_run_json(run)
