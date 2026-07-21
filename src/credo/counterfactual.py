"""One same-start, same-noise reference counterfactual engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
) -> pd.DataFrame:
    """Compare one perturbation with its residual removed in its full group."""
    if context_policy not in {"self_consistent", "clamped"}:
        raise ValueError("context_policy must be 'self_consistent' or 'clamped'.")
    if not same_noise:
        raise ValueError("CREDO reference counterfactuals require same_noise=True.")
    metadata = run.data.measure_meta.set_index("measure_id")
    if measure_id not in metadata.index:
        raise KeyError(f"Unknown measure_id {measure_id!r}.")
    if bool(metadata.loc[measure_id, "is_control"]):
        raise ValueError("Reference counterfactuals require a non-control measure.")
    group_id = metadata.loc[measure_id, "context_group_id"]
    group_ids = tuple(
        value
        for value in run.data.measure_ids
        if metadata.loc[value, "context_group_id"] == group_id
    )
    local_index = group_ids.index(measure_id)
    digest = hashlib.sha256(measure_id.encode("utf-8")).hexdigest()
    seed = run.config.training.seed + int(digest[:8], 16) % 1_000_000
    source = sample_initial_particles(
        run.data,
        group_ids,
        run.config.training.eval_particles,
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
            }
        )
    frame = pd.DataFrame(rows)
    run.counterfactual_rows.extend(rows)
    _persist_if_saved(run)
    return frame


def _persist_if_saved(run: Trainer) -> None:
    output = Path(run.config.output)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return
    columns = [
        "measure_id",
        "time_label",
        "context_policy",
        "delta_log_mass",
        "mean_shift_l2",
        "energy_distance",
        "context_dependence_shift",
        "factual_ess",
        "reference_ess",
    ]
    path = output / "counterfactuals.parquet"
    current = pd.DataFrame(run.counterfactual_rows, columns=columns)
    if path.exists():
        existing = pd.read_parquet(path)
        if existing.columns.tolist() != columns:
            raise ValueError("Existing counterfactuals.parquet has an incompatible schema.")
        if not existing.empty:
            current = pd.concat((existing, current), ignore_index=True)
    key = ["measure_id", "time_label", "context_policy"]
    current = current.drop_duplicates(key, keep="last").reset_index(drop=True)
    run.counterfactual_rows = current.to_dict(orient="records")
    current.to_parquet(path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counterfactual_status"] = "evaluated"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
