"""Update-based trainer with immutable checkpoint generations."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy import sparse

from ..canonical import atomic_json, canonical_json_bytes, contract_id, sha256_bytes
from ..compile import load_compiled_problem
from ..contracts import (
    ArtifactRef,
    CheckpointManifest,
    CompiledRunContract,
    ResolvedConfig,
    RunIntent,
    SelectionManifest,
    SemanticStudySnapshot,
    TrainingConfig,
)
from ..errors import ResumeMismatchError
from ..model import CountSDEModel, DynamicPoolBank
from ..objectives import exact_count_loss
from ..persistence import (
    load_tensor_file,
    publish_directory,
    save_tensor_file,
    verify_directory,
)
from ..runtime_identity import environment_lock_hash, implementation_tree_hash
from ..store import CountStore


@dataclass
class _GeneDecoderData:
    row_ids: np.ndarray[Any, Any]
    latents: np.ndarray[Any, Any]
    counts: sparse.csr_matrix
    library_sizes: np.ndarray[Any, Any]
    validation_row_ids: np.ndarray[Any, Any]
    validation_latents: np.ndarray[Any, Any]
    validation_counts: sparse.csr_matrix
    validation_library_sizes: np.ndarray[Any, Any]


@dataclass(frozen=True)
class _StateSplit:
    fit_indices: torch.Tensor
    validation_indices: torch.Tensor
    fit_series_hash: str
    validation_series_hash: str


def _state_split(
    arrays: dict[str, np.ndarray[Any, Any]], config: ResolvedConfig, device: torch.device
) -> _StateSplit:
    """Create a deterministic target-stratified training-only state split."""

    terminal = arrays["terminal_z"]
    evaluable = np.isfinite(terminal).all(axis=1)
    target = arrays["target_index"].astype(np.int64, copy=False)
    series = arrays["series_ids"].astype(str, copy=False)
    validation: list[int] = []
    if config.training.state_validation_fraction:
        for target_value in sorted(set(map(int, target[evaluable]))):
            positions = np.where(evaluable & (target == target_value))[0].tolist()
            if len(positions) < 2:
                continue
            requested = max(
                1,
                int(round(len(positions) * config.training.state_validation_fraction)),
            )
            count = min(
                len(positions) - 1,
                config.training.state_validation_max_per_target,
                requested,
            )
            ranked = sorted(
                positions,
                key=lambda index: hashlib.sha256(
                    f"{config.training.state_split_seed}:{target_value}:{series[index]}".encode()
                ).digest(),
            )
            validation.extend(ranked[:count])
    validation_array = np.asarray(sorted(validation), dtype=np.int64)
    validation_set = set(map(int, validation_array))
    fit_array = np.asarray(
        [index for index in np.where(evaluable)[0] if int(index) not in validation_set],
        dtype=np.int64,
    )
    if not len(fit_array):
        raise ValueError("The state training split is empty.")
    if config.training.state_validation_fraction and not len(validation_array):
        raise ValueError("The configured state validation split is empty.")

    def series_hash(indices: np.ndarray[Any, Any]) -> str:
        payload = "\n".join(series[indices].tolist()).encode() + b"\n"
        return sha256_bytes(payload)

    return _StateSplit(
        fit_indices=torch.from_numpy(fit_array).to(device),
        validation_indices=torch.from_numpy(validation_array).to(device),
        fit_series_hash=series_hash(fit_array),
        validation_series_hash=series_hash(validation_array),
    )


def _all_state_split(
    arrays: dict[str, np.ndarray[Any, Any]], config: ResolvedConfig, device: torch.device
) -> _StateSplit:
    """Return the post-selection refit information set: every evaluable train series."""

    terminal = arrays["terminal_z"]
    indices = np.where(np.isfinite(terminal).all(axis=1))[0].astype(np.int64)
    series = arrays["series_ids"].astype(str, copy=False)
    if not len(indices):
        raise ValueError("The post-selection state-refit set is empty.")
    payload = "\n".join(series[indices].tolist()).encode() + b"\n"
    return _StateSplit(
        fit_indices=torch.from_numpy(indices).to(device),
        validation_indices=torch.empty(0, dtype=torch.long, device=device),
        fit_series_hash=sha256_bytes(payload),
        validation_series_hash=sha256_bytes(b"\n"),
    )


def _gene_decoder_data(
    root: Path, workspace: Path, config: ResolvedConfig
) -> _GeneDecoderData | None:
    if not config.training.gene_decoder_batch_size:
        return None
    snapshot = SemanticStudySnapshot.model_validate_json(
        (workspace / "compiled/snapshot.json").read_text()
    )
    row_ids = np.asarray(
        [row for series in snapshot.series for row in (*series.source_rows, *series.terminal_rows)],
        dtype=np.int64,
    )
    with h5py.File(workspace / "prepared/latents.h5", "r") as handle:
        available = handle["row_ids"][:]
        z = handle["z"][:]
    lookup = {int(row): index for index, row in enumerate(available)}
    try:
        positions = np.asarray([lookup[int(row)] for row in row_ids], dtype=np.int64)
    except KeyError as error:
        raise ResumeMismatchError("Gene decoder row is absent from the latent cache.") from error
    latents = np.asarray(z[positions], dtype=np.float32)
    validation_count = min(
        config.training.gene_decoder_validation_max_rows,
        int(round(len(row_ids) * config.training.gene_decoder_validation_fraction)),
    )
    if config.training.gene_decoder_validation_fraction and validation_count == 0:
        validation_count = 1
    generator = np.random.default_rng(config.training.seed + 4_117_919)
    permutation = generator.permutation(len(row_ids))
    validation_positions = np.sort(permutation[:validation_count])
    training_positions = np.sort(permutation[validation_count:])
    # Materialize the immutable decoder row universe once.  Random HDF5 CSR
    # reads otherwise rescan every source-row window at every update, which is
    # correct but prohibitively I/O-bound for multi-thousand-update training.
    # The run contract already bounds this cohort-local cache through the Pod
    # memory request; gene counts remain sparse and are never moved wholesale
    # to the GPU.
    selected = CountStore((root / config.count_store).resolve()).rows(row_ids)
    counts = selected.matrix.tocsr()
    library_sizes = selected.library_sizes.astype(np.float32, copy=False)
    return _GeneDecoderData(
        row_ids=row_ids[training_positions],
        latents=latents[training_positions],
        counts=counts[training_positions].tocsr(),
        library_sizes=library_sizes[training_positions],
        validation_row_ids=row_ids[validation_positions],
        validation_latents=latents[validation_positions],
        validation_counts=counts[validation_positions].tocsr(),
        validation_library_sizes=library_sizes[validation_positions],
    )


def _gene_decoder_loss(
    model: CountSDEModel,
    data: _GeneDecoderData,
    config: ResolvedConfig,
    update: int,
    device: torch.device,
) -> torch.Tensor:
    generator = np.random.default_rng(config.training.seed + update * 1_000_003)
    size = min(config.training.gene_decoder_batch_size, len(data.row_ids))
    positions = generator.choice(len(data.row_ids), size=size, replace=False)
    matrix = data.counts[positions].tocsr()
    state = torch.from_numpy(data.latents[positions]).to(device)
    logits = model.decode_logits(state)
    selected = _csr_selected_logit_sum(logits, matrix)
    totals = torch.from_numpy(data.library_sizes[positions]).to(device)
    return torch.mean(torch.logsumexp(logits, dim=1) - selected / totals.clamp_min(1.0))


def _csr_selected_logit_sum(logits: torch.Tensor, matrix: sparse.csr_matrix) -> torch.Tensor:
    """Deterministically reduce observed-count logits by CSR row.

    CUDA ``scatter_add_`` uses atomic updates for repeated row indices.  A
    length-described segment reduction preserves CSR order and permits the
    deterministic-algorithm runtime gate used by release training.
    """

    matrix = matrix.tocsr()
    lengths = torch.from_numpy(np.diff(matrix.indptr).astype(np.int64, copy=False)).to(
        logits.device
    )
    columns = torch.from_numpy(matrix.indices.astype(np.int64, copy=False)).to(logits.device)
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False)).to(logits.device)
    rows = torch.repeat_interleave(torch.arange(matrix.shape[0], device=logits.device), lengths)
    contributions = logits[rows, columns] * values
    return torch.segment_reduce(contributions, reduce="sum", lengths=lengths)


@torch.no_grad()
def _gene_decoder_validation_diagnostics(
    model: CountSDEModel,
    data: _GeneDecoderData | None,
    device: torch.device,
) -> dict[str, float | str]:
    if data is None or not len(data.validation_row_ids):
        return {}
    total_loss = 0.0
    total_rows = 0
    for start in range(0, len(data.validation_row_ids), 1_024):
        stop = min(start + 1_024, len(data.validation_row_ids))
        matrix = data.validation_counts[start:stop].tocsr()
        state = torch.from_numpy(data.validation_latents[start:stop]).to(device)
        logits = model.decode_logits(state)
        selected = _csr_selected_logit_sum(logits, matrix)
        totals = torch.from_numpy(data.validation_library_sizes[start:stop]).to(device)
        losses = torch.logsumexp(logits, dim=1) - selected / totals.clamp_min(1.0)
        total_loss += float(losses.sum().cpu())
        total_rows += len(losses)
    return {
        "gene_decoder_validation_cross_entropy": total_loss / total_rows,
        "gene_decoder_validation_rows": float(total_rows),
        "gene_decoder_validation_rows_hash": sha256_bytes(
            np.asarray(data.validation_row_ids, dtype="<i8").tobytes()
        ),
    }


def _device(value: str | torch.device | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return result


def _validate_pilot_device(config: ResolvedConfig, device: torch.device) -> None:
    expected = config.training.pilot_device_type
    if config.model.source_target_interaction_rank and device.type != expected:
        raise ResumeMismatchError(
            f"Pilot device differs from calibrated device: expected {expected}, got {device.type}."
        )


def _tensor_problem(
    arrays: dict[str, np.ndarray[Any, Any]], device: torch.device
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in arrays.items():
        if value.dtype.kind in "USO":
            continue
        tensor = torch.from_numpy(value)
        if value.dtype.kind == "f":
            tensor = tensor.to(torch.float32)
        result[name] = tensor.to(device)
    return result


def _named_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    parameter_names: dict[torch.Tensor, str] = {
        parameter: name for name, parameter in model.named_parameters()
    }
    for parameter, state in optimizer.state.items():
        name = parameter_names[parameter]
        for key, value in state.items():
            if torch.is_tensor(value):
                tensors[f"state::{name}::{key}"] = value
            elif isinstance(value, (int, float, bool)):
                tensors[f"state::{name}::{key}"] = torch.tensor(value)
            else:
                raise TypeError(f"Unsupported optimizer state {name}.{key}: {type(value)}")
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        row = {key: value for key, value in group.items() if key != "params"}
        row["params"] = [parameter_names[parameter] for parameter in group["params"]]
        groups.append(row)
    return tensors, {"schema_version": 1, "optimizer": type(optimizer).__name__, "groups": groups}


def _restore_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    tree: dict[str, Any],
) -> None:
    named = dict(model.named_parameters())
    if tree.get("optimizer") != type(optimizer).__name__:
        raise ResumeMismatchError("Optimizer type changed across resume.")
    for key, value in tensors.items():
        _, parameter_name, state_name = key.split("::", 2)
        if parameter_name not in named:
            raise ResumeMismatchError(f"Optimizer parameter missing on resume: {parameter_name}.")
        optimizer.state[named[parameter_name]][state_name] = value.to(named[parameter_name].device)


def _checkpoint(
    workspace: Path,
    training_root: Path,
    contract: CompiledRunContract,
    model: CountSDEModel,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    loss: float,
    diagnostics: dict[str, float | str] | None,
    parent_checkpoint_id: str | None,
    selection_source_checkpoint_id: str | None = None,
    artifact_training_root: Path | None = None,
) -> CheckpointManifest:
    generation_root = training_root / "checkpoints" / f"generation-{update:09d}"
    reference_training_root = artifact_training_root or training_root
    reference_generation_root = reference_training_root / "checkpoints" / f"generation-{update:09d}"
    manifest_holder: dict[str, CheckpointManifest] = {}

    def writer(temp: Path) -> None:
        save_tensor_file(temp / "model.safetensors", model.state_dict())
        optimizer_tensors, optimizer_tree = _named_optimizer_state(model, optimizer)
        save_tensor_file(temp / "optimizer.safetensors", optimizer_tensors)
        (temp / "optimizer-tree.json").write_bytes(canonical_json_bytes(optimizer_tree) + b"\n")
        rng_tensors: dict[str, torch.Tensor] = {"torch_cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            for index, cuda_state in enumerate(torch.cuda.get_rng_state_all()):
                rng_tensors[f"torch_cuda_{index}"] = cuda_state
        save_tensor_file(temp / "rng.safetensors", rng_tensors)
        sampler = {
            "schema_version": 1,
            "next_update": update,
            "random_policy": "counter_derived",
            "prefetch": "quiesced",
        }
        (temp / "sampler.json").write_bytes(canonical_json_bytes(sampler) + b"\n")
        training_state = {
            "schema_version": 1,
            "update": update,
            "stage": "joint",
            "loss": loss,
            "diagnostics": diagnostics or {},
        }
        (temp / "training-state.json").write_bytes(canonical_json_bytes(training_state) + b"\n")

        def future(name: str, schema: str, media: str) -> ArtifactRef:
            path = temp / name
            return ArtifactRef(
                schema_id=schema,
                schema_version=1,
                sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
                media_type=media,
                relative_uri=(reference_generation_root / name).relative_to(workspace).as_posix(),
            )

        payload = {
            "schema_version": 1,
            "checkpoint_id": "pending",
            "compiled_run_id": contract.compiled_run_id,
            "generation": update,
            "update": update,
            "stage": "joint",
            "parent_checkpoint_id": parent_checkpoint_id,
            "selection_source_checkpoint_id": selection_source_checkpoint_id,
            "model": future(
                "model.safetensors", "credo.model_state", "application/x-safetensors"
            ).model_dump(mode="json"),
            "optimizer": future(
                "optimizer.safetensors", "credo.optimizer_state", "application/x-safetensors"
            ).model_dump(mode="json"),
            "optimizer_tree": future(
                "optimizer-tree.json", "credo.optimizer_tree", "application/json"
            ).model_dump(mode="json"),
            "rng": future(
                "rng.safetensors", "credo.rng_state", "application/x-safetensors"
            ).model_dump(mode="json"),
            "sampler": future("sampler.json", "credo.sampler_state", "application/json").model_dump(
                mode="json"
            ),
            "training_state": future(
                "training-state.json", "credo.training_state", "application/json"
            ).model_dump(mode="json"),
        }
        payload["checkpoint_id"] = contract_id(payload, id_field="checkpoint_id")
        manifest = CheckpointManifest.model_validate(payload)
        manifest_holder["value"] = manifest
        (temp / "checkpoint.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(generation_root, writer)
    latest = {
        "schema_version": 1,
        "generation": update,
        "checkpoint_id": manifest_holder["value"].checkpoint_id,
        "relative_uri": generation_root.relative_to(training_root).as_posix(),
    }
    atomic_json(training_root / "checkpoints" / "latest.json", latest)
    return manifest_holder["value"]


def _deterministic_prediction(
    model: CountSDEModel,
    source: torch.Tensor,
    duration: torch.Tensor,
    target: torch.Tensor,
    pool: torch.Tensor,
    control: torch.Tensor,
    grid_steps: torch.Tensor,
    *,
    effect_mode: str = "factual",
) -> torch.Tensor:
    """Use the same physical Euler grid as deterministic inference."""

    prediction = torch.empty_like(source)
    for steps_value in torch.unique(grid_steps).tolist():
        local = torch.where(grid_steps == steps_value)[0]
        z = source[local]
        dt = duration[local] / int(steps_value)
        for _ in range(int(steps_value)):
            z = model.state_step(
                z,
                dt,
                target[local],
                pool[local],
                control[local],
                total_steps=int(steps_value),
                source_z=source[local],
                effect_mode=effect_mode,
            )
        prediction[local] = z
    return prediction


@torch.no_grad()
def _initialize_state_channels(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    fit_indices: torch.Tensor,
    config: ResolvedConfig,
) -> dict[str, float]:
    """Initialize the nested constant-drift submodel at its closed-form optimum."""

    evaluable = torch.zeros(
        len(problem["terminal_z"]), dtype=torch.bool, device=problem["terminal_z"].device
    )
    evaluable[fit_indices] = True
    evaluable &= torch.isfinite(problem["terminal_z"]).all(dim=1)
    if model.config.terminal_anchor_drift:
        terminal = problem["terminal_z"][evaluable]
        source = problem["source_z"][evaluable]
        target = problem["target_index"][evaluable].long()
        control = problem["is_control"][evaluable].bool()
        if model.config.source_target_interaction_rank:
            targeting_targets = torch.unique(target[~control])
            target_terminal_means = [
                terminal[(target == int(value)) & ~control].mean(dim=0)
                for value in targeting_targets.tolist()
            ]
            model.terminal_anchor.copy_(
                torch.stack(target_terminal_means).mean(dim=0)
                if target_terminal_means
                else terminal.mean(dim=0)
            )
        else:
            # Preserve the established non-pilot recipe exactly. Equal-target
            # M0 is part of the pooled interaction estimand, not a silent
            # migration of earlier state models.
            model.terminal_anchor.copy_(terminal.mean(dim=0))
        model.target_anchor_offset.zero_()
        model.target_anchor_gate.zero_()
        weight = model.config.target_anchor_weight
        if weight:
            for target_value in torch.unique(target[~control]).tolist():
                local = (target == target_value) & ~control
                offset = terminal[local].mean(dim=0) - model.terminal_anchor
                model.target_anchor_offset[int(target_value)].copy_(weight * offset)
                model.target_anchor_gate[int(target_value)] = weight
        if model.config.adaptive_target_anchor:
            source_support = problem["source_counts"][evaluable].float()
            terminal_support = problem["terminal_counts"][evaluable].float()
            reliability = torch.minimum(source_support, terminal_support)
            reliability = reliability.clamp(
                min=1.0, max=float(config.training.support_weight_cap)
            ).pow(config.training.support_weight_power)
            for target_value in torch.unique(target[~control]).tolist():
                local = (target == target_value) & ~control
                local_terminal = terminal[local]
                if len(local_terminal) < 2:
                    continue
                local_reliability = reliability[local]
                total_weight = local_reliability.sum()
                other_weight = (total_weight - local_reliability).clamp_min(1e-12)
                weighted_sum = (local_reliability[:, None] * local_terminal).sum(dim=0)
                other_mean = (
                    weighted_sum - local_reliability[:, None] * local_terminal
                ) / other_weight[:, None]
                # The expanded expression above is deliberately explicit: each
                # guide is predicted only from the other training guides for its
                # target.  No outer guide endpoint enters this gate.
                residual = local_terminal - model.terminal_anchor
                candidate = other_mean - model.terminal_anchor
                row_weight = local_reliability / local_reliability.sum().clamp_min(1e-12)
                denominator = (row_weight[:, None] * candidate.square()).sum()
                if denominator <= 0:
                    continue
                numerator = (row_weight[:, None] * residual * candidate).sum()
                gate = torch.clamp(
                    numerator / denominator,
                    min=0.0,
                    max=model.config.adaptive_target_anchor_max_weight,
                )
                baseline_error = residual.square().mean(dim=1)
                candidate_error = (residual - gate * candidate).square().mean(dim=1)
                guide_gain = baseline_error - candidate_error
                if torch.any(guide_gain < model.config.adaptive_target_anchor_min_guide_gain):
                    continue
                target_mean = (local_reliability[:, None] * local_terminal).sum(
                    dim=0
                ) / local_reliability.sum().clamp_min(1e-12)
                model.target_anchor_gate[int(target_value)] = gate
                model.target_anchor_offset[int(target_value)].copy_(
                    gate * (target_mean - model.terminal_anchor)
                )
        if model.config.source_target_interaction_rank:
            model.source_target_center.zero_()
            residual_rows: list[torch.Tensor] = []
            residual_weights: list[torch.Tensor] = []
            for target_value in targeting_targets.tolist():
                local = (target == int(target_value)) & ~control
                local_source = source[local]
                local_center = local_source.mean(dim=0)
                model.source_target_center[int(target_value)].copy_(local_center)
                residual_rows.append(local_source - local_center)
                residual_weights.append(
                    torch.full(
                        (len(local_source),),
                        1.0 / (len(targeting_targets) * len(local_source)),
                        dtype=source.dtype,
                        device=source.device,
                    )
                )
            if not residual_rows:
                raise ValueError("Source-target interaction has no targeting training guides.")
            centered = torch.cat(residual_rows, dim=0)
            weights = torch.cat(residual_weights)
            center = torch.stack(
                [model.source_target_center[int(value)] for value in targeting_targets.tolist()]
            ).mean(dim=0)
            covariance = centered.T @ (centered * weights[:, None])
            average_variance = torch.diagonal(covariance).mean().clamp_min(1e-6)
            ridge = model.config.source_target_whitening_ridge * average_variance
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            # Conventional ridge whitening, not an eigenvalue floor.
            inverse_root = torch.rsqrt(eigenvalues.clamp_min(0.0) + ridge)
            whitener = (eigenvectors * inverse_root.unsqueeze(0)) @ eigenvectors.T
            model.source_interaction_center.copy_(center)
            model.source_interaction_whitener.copy_(whitener)
            model.source_target_main_offset.zero_()
            assert model.source_target_main_weight is not None
            model.source_target_main_weight.zero_()
            for target_value in torch.unique(target[~control]).tolist():
                local = (target == int(target_value)) & ~control
                model.source_target_main_offset[int(target_value)].copy_(
                    terminal[local].mean(dim=0) - model.terminal_anchor
                )
            model.refresh_source_target_interaction_mean(source, target, control)
        prediction = _deterministic_prediction(
            model,
            problem["source_z"][evaluable],
            problem["duration"][evaluable],
            target,
            problem["pool_index"][evaluable].long(),
            control,
            problem["grid_steps"][evaluable].long(),
        )
        rmse = torch.sqrt(torch.mean((prediction - terminal) ** 2))
        return {
            "analytic_terminal_anchor_rmse": float(rmse.cpu()),
            "source_carryover_alpha": model.config.source_carryover_alpha,
            "target_anchor_weight": model.config.target_anchor_weight,
            "adaptive_target_anchor_active_targets": float(
                torch.count_nonzero(model.target_anchor_gate).cpu()
            ),
            "adaptive_target_anchor_mean_gate": float(
                model.target_anchor_gate[model.target_anchor_gate > 0].mean().cpu()
            )
            if torch.any(model.target_anchor_gate > 0)
            else 0.0,
        }
    rate = problem["terminal_z"][evaluable] - problem["source_z"][evaluable]
    rate = rate / problem["duration"][evaluable, None]
    target = problem["target_index"][evaluable].long()
    control = problem["is_control"][evaluable].bool()
    base_rows = rate[control] if torch.any(control) else rate
    model.base_drift.copy_(base_rows.mean(dim=0))
    model.target_drift.zero_()
    for target_value in torch.unique(target[~control]).tolist():
        local = (target == target_value) & ~control
        model.target_drift[int(target_value)].copy_(rate[local].mean(dim=0) - model.base_drift)
    prediction = problem["source_z"][evaluable] + problem["duration"][evaluable, None] * (
        model.base_drift[None, :] + model.target_drift[target] * (~control).float()[:, None]
    )
    rmse = torch.sqrt(torch.mean((prediction - problem["terminal_z"][evaluable]) ** 2))
    return {"closed_form_initial_rmse": float(rmse.cpu())}


@torch.no_grad()
def _fit_shrunk_target_main_weight(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    fit_indices: torch.Tensor,
    config: ResolvedConfig,
    *,
    materialize: bool,
) -> float:
    """Fit one bounded target-main coefficient without interaction parameters.

    The objective is targeting-only and target-balanced.  The positive ridge
    penalty is applied to the scalar coefficient, and the result is bounded to
    the preregistered interpolation interval.  No validation endpoint is used.
    """

    if model.source_target_main_weight is None:
        raise RuntimeError("Shrunk target fitting requires a source-target model.")
    terminal = problem["terminal_z"][fit_indices]
    target = problem["target_index"][fit_indices].long()
    control = problem["is_control"][fit_indices].bool()
    numerators: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []
    for target_value in torch.unique(target[~control]).tolist():
        local = (target == int(target_value)) & ~control
        local_terminal = terminal[local]
        if len(local_terminal) < 2:
            continue
        residual = local_terminal - model.terminal_anchor
        # Fit shrinkage on sister-guide leave-one-out target means.  Using the
        # same row in its target mean would force alpha toward one and would
        # not estimate generalization to another guide of the known target.
        other_mean = (local_terminal.sum(dim=0) - local_terminal) / (len(local_terminal) - 1)
        candidate = other_mean - model.terminal_anchor
        numerators.append((residual * candidate).mean())
        denominators.append(candidate.square().mean())
    if not numerators:
        alpha = torch.zeros((), device=terminal.device)
    else:
        numerator = torch.stack(numerators).mean()
        denominator = torch.stack(denominators).mean()
        alpha = torch.clamp(
            numerator / (denominator + config.training.source_target_main_penalty),
            min=0.0,
            max=model.config.source_target_main_max_weight,
        )
    if materialize:
        model.source_target_main_weight.copy_(alpha)
    return float(alpha.cpu())


def _target_balanced_mse(
    error: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average series MSE within targets, then weight every target equally."""

    series_mse = error.square().mean(dim=1)
    target_losses: list[torch.Tensor] = []
    for target_value in torch.unique(target).tolist():
        local = target == target_value
        values = series_mse[local]
        if reliability is None:
            target_losses.append(values.mean())
        else:
            weights = reliability[local]
            weights = weights / weights.sum().clamp_min(1e-12)
            target_losses.append((weights * values).sum())
    if not target_losses:
        raise ValueError("Target-balanced state loss has no evaluable targets.")
    return torch.stack(target_losses).mean()


_TARGET_MAIN_BASELINE_CODES = {
    "shrunk_target_only": 0,
    "empirical_bayes_target": 1,
    "target_terminal": 2,
}


def _empirical_bayes_target_gates(
    terminal: torch.Tensor,
    target: torch.Tensor,
    control: torch.Tensor,
    global_terminal: torch.Tensor,
) -> dict[int, torch.Tensor]:
    target_values = torch.unique(target[~control]).tolist()
    means = {
        int(value): terminal[(target == int(value)) & ~control].mean(dim=0)
        for value in target_values
    }
    if not means:
        return {}
    tau2 = torch.stack(
        [(value - global_terminal).square().mean() for value in means.values()]
    ).mean()
    gates: dict[int, torch.Tensor] = {}
    for value in target_values:
        local = terminal[(target == int(value)) & ~control]
        sigma2 = (local - local.mean(dim=0)).square().mean()
        gates[int(value)] = tau2 / (tau2 + sigma2 / len(local) + 1e-12)
    return gates


@torch.no_grad()
def _noninteraction_validation_scores(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    state_split: _StateSplit,
    config: ResolvedConfig,
) -> dict[str, float | str]:
    """Score the frozen pooled noninteraction family on one inner split."""

    fit = state_split.fit_indices
    validation = state_split.validation_indices
    validation_control = problem["is_control"][validation].bool()
    primary = ~validation_control
    if not torch.any(primary):
        raise ValueError("Pooled baseline selection requires noncontrol validation guides.")
    fit_source = problem["source_z"][fit]
    fit_terminal = problem["terminal_z"][fit]
    fit_target = problem["target_index"][fit].long()
    fit_control = problem["is_control"][fit].bool()
    validation_source = problem["source_z"][validation][primary]
    validation_terminal = problem["terminal_z"][validation][primary]
    validation_target = problem["target_index"][validation].long()[primary]
    global_terminal = model.terminal_anchor
    target_means = {
        int(value): fit_terminal[(fit_target == int(value)) & ~fit_control].mean(dim=0)
        for value in torch.unique(fit_target[~fit_control]).tolist()
    }
    if any(int(value) not in target_means for value in validation_target.tolist()):
        raise ValueError("An inner validation target has no training-guide target mean.")

    target_terminal = torch.stack([target_means[int(value)] for value in validation_target])
    alpha = _fit_shrunk_target_main_weight(model, problem, fit, config, materialize=False)
    shrunk = global_terminal + alpha * (target_terminal - global_terminal)
    gates = _empirical_bayes_target_gates(fit_terminal, fit_target, fit_control, global_terminal)
    empirical_bayes = torch.stack(
        [
            global_terminal + gates[int(value)] * (target_means[int(value)] - global_terminal)
            for value in validation_target
        ]
    )
    target_delta_means = {
        int(value): (
            fit_terminal[(fit_target == int(value)) & ~fit_control]
            - fit_source[(fit_target == int(value)) & ~fit_control]
        ).mean(dim=0)
        for value in torch.unique(fit_target[~fit_control]).tolist()
    }
    target_delta = validation_source + torch.stack(
        [target_delta_means[int(value)] for value in validation_target]
    )

    train = ~fit_control
    train_source = fit_source[train].to(torch.float64)
    train_terminal = fit_terminal[train].to(torch.float64)
    train_target = fit_target[train]
    target_values = tuple(sorted(set(map(int, train_target.tolist()))))
    target_lookup = {value: index for index, value in enumerate(target_values)}
    train_one_hot = torch.zeros(
        len(train_target), len(target_values), dtype=torch.float64, device=train_source.device
    )
    train_one_hot[
        torch.arange(len(train_target), device=train_source.device),
        torch.tensor(
            [target_lookup[int(value)] for value in train_target],
            device=train_source.device,
        ),
    ] = 1.0
    design = torch.cat(
        [
            torch.ones(len(train_source), 1, dtype=torch.float64, device=train_source.device),
            train_source,
            train_one_hot,
        ],
        dim=1,
    )
    multiplicity = torch.stack([(train_target == int(value)).sum() for value in train_target]).to(
        torch.float64
    )
    row_weight = multiplicity.reciprocal()
    weighted_design = design * torch.sqrt(row_weight)[:, None]
    weighted_terminal = train_terminal * torch.sqrt(row_weight)[:, None]
    penalty = torch.eye(design.shape[1], dtype=torch.float64, device=design.device)
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        weighted_design.T @ weighted_design + config.training.noninteraction_linear_ridge * penalty,
        weighted_design.T @ weighted_terminal,
    )
    validation_one_hot = torch.zeros(
        len(validation_target),
        len(target_values),
        dtype=torch.float64,
        device=design.device,
    )
    validation_one_hot[
        torch.arange(len(validation_target), device=design.device),
        torch.tensor(
            [target_lookup[int(value)] for value in validation_target], device=design.device
        ),
    ] = 1.0
    validation_design = torch.cat(
        [
            torch.ones(len(validation_source), 1, dtype=torch.float64, device=design.device),
            validation_source.to(torch.float64),
            validation_one_hot,
        ],
        dim=1,
    )
    linear = (validation_design @ coefficients).to(validation_terminal.dtype)

    predictions = {
        "shrunk_target_only": shrunk,
        "empirical_bayes_target": empirical_bayes,
        "target_terminal": target_terminal,
        "target_delta": target_delta,
        "linear_source_plus_target": linear,
    }
    scores = {
        name: float(
            torch.sqrt(
                _target_balanced_mse(prediction - validation_terminal, validation_target)
            ).cpu()
        )
        for name, prediction in predictions.items()
    }
    target_order = ("shrunk_target_only", "empirical_bayes_target", "target_terminal")
    all_order = (*target_order, "target_delta", "linear_source_plus_target")
    best_target = min(target_order, key=lambda name: (scores[name], target_order.index(name)))
    best_all = min(all_order, key=lambda name: (scores[name], all_order.index(name)))
    return {
        **{f"state_validation_noninteraction_{name}_rmse": score for name, score in scores.items()},
        "state_validation_shrunk_target_only_rmse": scores["shrunk_target_only"],
        "state_validation_best_target_only_rmse": scores[best_target],
        "state_validation_best_target_only_baseline": best_target,
        "state_validation_best_noninteraction_rmse": scores[best_all],
        "state_validation_best_noninteraction_baseline": best_all,
    }


@torch.no_grad()
def _materialize_target_main_baseline(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    fit_indices: torch.Tensor,
    config: ResolvedConfig,
    baseline: str,
) -> None:
    """Materialize the selected deployable M1 without validation endpoints."""

    if baseline not in _TARGET_MAIN_BASELINE_CODES:
        raise ValueError(f"Unsupported deployable target-main baseline: {baseline}.")
    assert model.source_target_main_weight is not None
    if baseline == "shrunk_target_only":
        _fit_shrunk_target_main_weight(model, problem, fit_indices, config, materialize=True)
    else:
        model.source_target_main_weight.fill_(1.0)
        if baseline == "empirical_bayes_target":
            terminal = problem["terminal_z"][fit_indices]
            target = problem["target_index"][fit_indices].long()
            control = problem["is_control"][fit_indices].bool()
            gates = _empirical_bayes_target_gates(terminal, target, control, model.terminal_anchor)
            for target_value, gate in gates.items():
                model.source_target_main_offset[target_value].mul_(gate)
    model.source_target_main_baseline_code.fill_(_TARGET_MAIN_BASELINE_CODES[baseline])


def _support_reliability(
    problem: dict[str, torch.Tensor], indices: torch.Tensor, config: ResolvedConfig
) -> torch.Tensor | None:
    if not config.training.support_weight_power:
        return None
    support = torch.minimum(problem["source_counts"][indices], problem["terminal_counts"][indices])
    support = support.float().clamp(min=1.0, max=float(config.training.support_weight_cap))
    return support.pow(config.training.support_weight_power)


def _state_objective(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    config: ResolvedConfig,
    indices: torch.Tensor,
    prediction: torch.Tensor,
    *,
    train_state: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return empirical target-balanced MSE and the exact optimization objective.

    This is the single state-objective implementation used by minibatch
    training, complete-fit diagnostics, and post-selection refitting.  The
    analytic target-main coefficient uses scalar ridge, so the configured
    target-main penalty is applied to that same scalar here.
    """

    source = problem["source_z"][indices]
    duration = problem["duration"][indices]
    target = problem["target_index"][indices].long()
    pool = problem["pool_index"][indices].long()
    control = problem["is_control"][indices].bool()
    error = prediction - problem["terminal_z"][indices]
    reliability = _support_reliability(problem, indices, config)
    if model.config.source_target_interaction_rank and torch.any(~control):
        # Perturbation targets are the primary pooled units. Controls remain a
        # secondary audited population and do not receive one target's worth
        # of influence in the scientific pilot objective.
        empirical = _target_balanced_mse(
            error[~control],
            target[~control],
            reliability[~control] if reliability is not None else None,
        )
    else:
        empirical = _target_balanced_mse(error, target, reliability)
    state_parameters: list[torch.Tensor] = [
        model.base_drift,
        model.target_drift,
        model.target_selection,
    ]
    if model.config.source_conditioned_anchor and train_state:
        assert model.source_anchor_hidden is not None
        assert model.source_anchor_output is not None
        state_parameters.extend(
            [
                model.source_anchor_hidden.weight,
                model.source_anchor_hidden.bias,
                model.source_anchor_output.weight,
            ]
        )
    if model.config.source_target_interaction_rank and train_state:
        assert model.source_interaction_projection is not None
        assert model.target_interaction_embedding is not None
        assert model.source_target_output is not None
        state_parameters.extend(
            [
                model.source_interaction_projection.weight,
                model.target_interaction_embedding,
                model.source_target_output.weight,
            ]
        )
    if model.config.state_dependent_drift:
        state_parameters.extend([model.state_drift_hidden.weight, model.state_drift_output.weight])
    if model.config.shared_diffusion:
        state_parameters.append(model.log_diffusion)
    if config.intent is RunIntent.COUNT_CONTEXT:
        state_parameters.extend(
            [model.state_pool_projection, model.state_mass_context, model.context_to_state]
        )
    generic_regularization = 1e-4 * sum(parameter.square().mean() for parameter in state_parameters)
    objective = empirical + generic_regularization
    if config.training.target_drift_penalty:
        noncontrol_targets = torch.unique(target[~control])
        if len(noncontrol_targets):
            displacement = duration.mean() * model.target_drift[noncontrol_targets]
            objective = (
                objective + config.training.target_drift_penalty * displacement.square().mean()
            )
    if config.training.source_drift_penalty and model.config.state_dependent_drift:
        displacement = duration[:, None] * model.state_drift_output(
            torch.tanh(model.state_drift_hidden(source))
        )
        objective = objective + config.training.source_drift_penalty * displacement.square().mean()
    if (
        config.training.source_drift_penalty
        and model.config.source_conditioned_anchor
        and train_state
    ):
        assert model.source_anchor_hidden is not None
        assert model.source_anchor_output is not None
        displacement = model.config.source_anchor_residual_scale * torch.tanh(
            model.source_anchor_output(torch.tanh(model.source_anchor_hidden(source)))
        )
        objective = objective + config.training.source_drift_penalty * displacement.square().mean()
    if model.config.source_target_interaction_rank and train_state:
        assert model.source_target_main_weight is not None
        target_only = _deterministic_prediction(
            model,
            source,
            duration,
            target,
            pool,
            control,
            problem["grid_steps"][indices].long(),
            effect_mode="target_only",
        )
        interaction_displacement = prediction - target_only
        interaction_penalty = (
            _target_balanced_mse(
                interaction_displacement[~control],
                target[~control],
            )
            if torch.any(~control)
            else interaction_displacement.square().mean()
        )
        objective = (
            objective
            + config.training.source_target_main_penalty * model.source_target_main_weight.square()
            + config.training.source_target_interaction_penalty * interaction_penalty
        )
    return empirical, objective


@torch.no_grad()
def _training_diagnostics(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    state_split: _StateSplit,
    config: ResolvedConfig,
    decoder_data: _GeneDecoderData | None = None,
) -> dict[str, float | str]:
    if model.config.source_target_interaction_rank:
        fit = state_split.fit_indices
        model.refresh_source_target_interaction_mean(
            problem["source_z"][fit],
            problem["target_index"][fit].long(),
            problem["is_control"][fit].bool(),
        )
    evaluable = torch.isfinite(problem["terminal_z"]).all(dim=1)
    indices = torch.where(evaluable)[0]
    prediction = _deterministic_prediction(
        model,
        problem["source_z"][indices],
        problem["duration"][indices],
        problem["target_index"][indices].long(),
        problem["pool_index"][indices].long(),
        problem["is_control"][indices].bool(),
        problem["grid_steps"][indices].long(),
    )
    error = prediction - problem["terminal_z"][indices]
    diagnostics: dict[str, float | str] = {
        "full_state_rmse": float(torch.sqrt(torch.mean(error.square())).cpu()),
        "finite_prediction_fraction": float(torch.isfinite(prediction).float().mean().cpu()),
        "state_fit_series_hash": state_split.fit_series_hash,
        "state_validation_series_hash": state_split.validation_series_hash,
        **_gene_decoder_validation_diagnostics(model, decoder_data, problem["source_z"].device),
    }
    for label, local_indices in (
        ("fit", state_split.fit_indices),
        ("validation", state_split.validation_indices),
    ):
        if not len(local_indices):
            continue
        local_prediction = _deterministic_prediction(
            model,
            problem["source_z"][local_indices],
            problem["duration"][local_indices],
            problem["target_index"][local_indices].long(),
            problem["pool_index"][local_indices].long(),
            problem["is_control"][local_indices].bool(),
            problem["grid_steps"][local_indices].long(),
        )
        local_error = local_prediction - problem["terminal_z"][local_indices]
        local_mse, local_objective = _state_objective(
            model,
            problem,
            config,
            local_indices,
            local_prediction,
            train_state=True,
        )
        diagnostics[f"state_{label}_target_balanced_rmse"] = float(torch.sqrt(local_mse).cpu())
        diagnostics[f"state_{label}_unregularized_target_balanced_state_mse"] = float(
            local_mse.cpu()
        )
        diagnostics[f"state_{label}_regularized_optimization_objective"] = float(
            local_objective.cpu()
        )
        diagnostics[f"state_{label}_series"] = float(len(local_indices))
        local_control = problem["is_control"][local_indices].bool()
        if torch.any(~local_control):
            targeting_error = local_error[~local_control]
            targeting_target = problem["target_index"][local_indices][~local_control].long()
            targeting_mse = _target_balanced_mse(targeting_error, targeting_target)
            diagnostics[f"state_{label}_targeting_target_balanced_rmse"] = float(
                torch.sqrt(targeting_mse).cpu()
            )
    if len(state_split.validation_indices):
        fit = state_split.fit_indices
        validation = state_split.validation_indices
        validation_target = problem["target_index"][validation].long()
        validation_control = problem["is_control"][validation].bool()
        global_terminal = model.terminal_anchor
        global_prediction = global_terminal.expand(len(validation), -1)
        targeting = ~validation_control
        if torch.any(targeting):
            # The pooled noninteraction registry is part of the dev18
            # source-target interaction contract.  Legacy state models do not
            # own the target-main buffers needed to materialize those
            # baselines and must retain their existing validation path.
            if model.config.source_target_interaction_rank:
                baseline_evidence = _noninteraction_validation_scores(
                    model, problem, state_split, config
                )
                diagnostics.update(baseline_evidence)
            validation_terminal = problem["terminal_z"][validation]
            global_error = global_prediction[targeting] - validation_terminal[targeting]
            global_mse = _target_balanced_mse(global_error, validation_target[targeting])
            diagnostics["state_validation_global_null_rmse"] = float(torch.sqrt(global_mse).cpu())
            if model.config.source_target_interaction_rank:
                target_prediction = _deterministic_prediction(
                    model,
                    problem["source_z"][validation],
                    problem["duration"][validation],
                    validation_target,
                    problem["pool_index"][validation].long(),
                    validation_control,
                    problem["grid_steps"][validation].long(),
                    effect_mode="target_only",
                )
                target_displacement = target_prediction - global_prediction
                target_error = target_prediction[targeting] - validation_terminal[targeting]
                target_mse = _target_balanced_mse(target_error, validation_target[targeting])
                target_rmse = float(torch.sqrt(target_mse).cpu())
                diagnostics["state_validation_materialized_target_only_rmse"] = target_rmse
                full_prediction = _deterministic_prediction(
                    model,
                    problem["source_z"][validation],
                    problem["duration"][validation],
                    validation_target,
                    problem["pool_index"][validation].long(),
                    validation_control,
                    problem["grid_steps"][validation].long(),
                )
                full_error = full_prediction[targeting] - validation_terminal[targeting]
                full_mse = _target_balanced_mse(full_error, validation_target[targeting])
                full_rmse = float(torch.sqrt(full_mse).cpu())
                interaction_displacement = full_prediction - target_prediction
                if model.source_target_main_weight is None:
                    raise RuntimeError("Target-main diagnostics require a fitted target channel.")
                diagnostics.update(
                    {
                        "state_validation_full_interaction_rmse": full_rmse,
                        "state_validation_interaction_incremental_gain": target_rmse - full_rmse,
                        "interaction_displacement_rms": float(
                            torch.sqrt(
                                _target_balanced_mse(
                                    interaction_displacement[targeting],
                                    validation_target[targeting],
                                )
                            ).cpu()
                        ),
                        "target_main_displacement_rms": float(
                            torch.sqrt(
                                _target_balanced_mse(
                                    target_displacement[targeting],
                                    validation_target[targeting],
                                )
                            ).cpu()
                        ),
                        "state_fit_shrunk_target_main_weight": float(
                            model.source_target_main_weight.cpu()
                        ),
                    }
                )
    return diagnostics


def _write_selection(
    training_root: Path,
    contract: CompiledRunContract,
    config: ResolvedConfig,
    checkpoints: list[CheckpointManifest],
) -> dict[str, Any]:
    null_guarded = config.training.checkpoint_selection == "minimum_state_validation_null_guarded"
    eligible_updates = scientific_checkpoint_updates(
        tuple(checkpoint.update for checkpoint in checkpoints), config.training
    )
    eligible_checkpoints = [
        checkpoint for checkpoint in checkpoints if checkpoint.update in eligible_updates
    ]
    if null_guarded:
        expected_candidates = (0, *config.training.state_checkpoint_updates)
        actual_candidates = tuple(item.update for item in eligible_checkpoints)
        if actual_candidates != expected_candidates:
            raise RuntimeError(
                "Null-guarded selection candidates differ from the calibrated schedule: "
                f"expected {expected_candidates}, observed {actual_candidates}."
            )
    if config.training.checkpoint_selection in {
        "minimum_gene_decoder_validation",
        "minimum_state_validation",
        "minimum_state_validation_null_guarded",
    }:
        diagnostic_key = (
            "gene_decoder_validation_cross_entropy"
            if config.training.checkpoint_selection == "minimum_gene_decoder_validation"
            else (
                "state_validation_full_interaction_rmse"
                if null_guarded
                else "state_validation_target_balanced_rmse"
            )
        )
        scored = []
        for checkpoint in eligible_checkpoints:
            generation = training_root / "checkpoints" / f"generation-{checkpoint.update:09d}"
            state = json.loads((generation / "training-state.json").read_text())
            score = state.get("diagnostics", {}).get(diagnostic_key)
            if score is not None:
                scored.append((float(score), checkpoint.update, checkpoint.checkpoint_id))
        if not scored:
            raise RuntimeError("No validation-scored checkpoint is available for selection.")
        score, update, checkpoint_id = min(scored)
        selection_details: dict[str, Any] = {}
        if null_guarded:
            null_rows = [row for row in scored if row[1] == 0]
            trained_rows = [row for row in scored if row[1] > 0]
            if len(null_rows) != 1 or not trained_rows:
                raise RuntimeError(
                    "Null-guarded selection requires update 0 and trained candidates."
                )
            null_generation = training_root / "checkpoints/generation-000000000"
            null_state = json.loads((null_generation / "training-state.json").read_text())
            null_score_value = null_state.get("diagnostics", {}).get(
                "state_validation_global_null_rmse"
            )
            if null_score_value is None:
                raise RuntimeError("Null-guarded selection lacks the global-terminal null score.")
            null_score = float(null_score_value)
            _, _, null_checkpoint_id = null_rows[0]
            best_trained_score, best_trained_update, best_trained_checkpoint_id = min(trained_rows)
            best_generation = (
                training_root / "checkpoints" / f"generation-{best_trained_update:09d}"
            )
            best_state = json.loads((best_generation / "training-state.json").read_text())
            best_diagnostics = best_state.get("diagnostics", {})
            target_score = best_diagnostics.get("state_validation_best_target_only_rmse")
            target_baseline = best_diagnostics.get("state_validation_best_target_only_baseline")
            noninteraction_score = best_diagnostics.get("state_validation_best_noninteraction_rmse")
            noninteraction_baseline = best_diagnostics.get(
                "state_validation_best_noninteraction_baseline"
            )
            shrunk_score = best_diagnostics.get("state_validation_shrunk_target_only_rmse")
            if any(
                value is None
                for value in (
                    target_score,
                    target_baseline,
                    noninteraction_score,
                    noninteraction_baseline,
                    shrunk_score,
                )
            ):
                raise RuntimeError(
                    "Null-guarded selection lacks complete noninteraction baselines."
                )
            target_score = float(target_score)
            noninteraction_score = float(noninteraction_score)
            shrunk_score = float(shrunk_score)
            target_required = config.training.state_validation_target_minimum_improvement
            interaction_required = config.training.state_validation_interaction_minimum_improvement
            interaction_improvement = noninteraction_score - best_trained_score
            target_improvement = null_score - target_score
            overall_interaction_improvement = null_score - best_trained_score
            if (
                interaction_improvement >= interaction_required
                and overall_interaction_improvement >= target_required
            ):
                score, update, checkpoint_id = (
                    best_trained_score,
                    best_trained_update,
                    best_trained_checkpoint_id,
                )
                selected_family = "target_plus_source_target_interaction"
            elif target_improvement >= target_required:
                score, update, checkpoint_id = target_score, 0, null_checkpoint_id
                selected_family = "selected_training_only_target_main"
            else:
                score, update, checkpoint_id = null_score, 0, null_checkpoint_id
                selected_family = "global_terminal_null"
            selection_details = {
                "global_null_score": null_score,
                "shrunk_target_only_score": shrunk_score,
                "best_target_only_score": target_score,
                "best_target_only_baseline": target_baseline,
                "best_noninteraction_score": noninteraction_score,
                "best_noninteraction_baseline": noninteraction_baseline,
                "interaction_score": best_trained_score,
                "interaction_incremental_gain": interaction_improvement,
                "target_incremental_gain": target_improvement,
                "target_minimum_required_improvement": target_required,
                "interaction_minimum_required_improvement": interaction_required,
                "selection_calibration_hash": contract.state_selection_calibration_hash,
                "selected_family": selected_family,
            }
        metric = diagnostic_key
    else:
        selected_update = (
            config.training.selected_update
            if config.training.selected_update is not None
            else config.training.max_updates
        )
        matches = [item for item in checkpoints if item.update == selected_update]
        if not matches:
            raise RuntimeError("The configured selected checkpoint was not persisted.")
        update, checkpoint_id = matches[0].update, matches[0].checkpoint_id
        score, metric = None, "configured_final_update"
        selection_details = {"selected_family": "configured_checkpoint"}
    if config.training.checkpoint_selection == "minimum_gene_decoder_validation":
        selection_details = {**selection_details, "selected_family": "gene_decoder_selected"}
    elif config.training.checkpoint_selection == "minimum_state_validation":
        selection_details = {**selection_details, "selected_family": "state_validation_selected"}
    payload = {
        "schema_version": 1,
        "selection_id": "pending",
        "compiled_run_id": contract.compiled_run_id,
        "policy": config.training.checkpoint_selection,
        "metric": metric,
        "score": score,
        "selected_update": update,
        "selected_checkpoint_id": checkpoint_id,
        "selected_checkpoint_relative_uri": (f"training/checkpoints/generation-{update:09d}"),
        "candidate_updates": tuple(item.update for item in eligible_checkpoints),
        **selection_details,
    }
    normalized = SelectionManifest.model_construct(**payload).model_dump(mode="json")
    normalized["selection_id"] = contract_id(normalized, id_field="selection_id")
    selection = SelectionManifest.model_validate(normalized)
    serialized = selection.model_dump(mode="json")
    atomic_json(training_root / "selection.json", serialized)
    return serialized


def scientific_checkpoint_updates(
    persisted_updates: tuple[int, ...], training: TrainingConfig
) -> tuple[int, ...]:
    """Return only preregistered scientific selection candidates.

    A disposable capacity probe can persist update 1 in the same lineage, but
    it is excluded unless update 1 is explicitly part of the frozen scientific
    checkpoint grid. The order of the persisted parent chain is preserved.
    """

    null_guarded = training.checkpoint_selection == "minimum_state_validation_null_guarded"
    if null_guarded:
        allowed = {0, *training.state_checkpoint_updates}
        return tuple(update for update in persisted_updates if update in allowed)
    return tuple(
        update
        for update in persisted_updates
        if update > 0
        and (update % training.checkpoint_every == 0 or update == training.max_updates)
    )


def _loss(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    config: ResolvedConfig,
    update: int,
    decoder_data: _GeneDecoderData | None,
    state_split: _StateSplit,
) -> torch.Tensor:
    phase = (
        "state"
        if config.intent is RunIntent.COUNT_STATE or update % 2 == 1
        else "complete_count_block"
    )
    candidates = state_split.fit_indices
    if config.training.state_full_batch:
        indices = candidates
    else:
        generator = torch.Generator(device=problem["source_z"].device)
        generator.manual_seed(config.training.seed + update)
        order = candidates[
            torch.randperm(len(candidates), generator=generator, device=candidates.device)
        ]
        indices = order[: config.training.state_batch_size]
    if config.model.source_target_interaction_rank:
        model.refresh_source_target_interaction_mean(
            problem["source_z"][candidates],
            problem["target_index"][candidates].long(),
            problem["is_control"][candidates].bool(),
        )
    source = problem["source_z"][indices]
    duration = problem["duration"][indices]
    target = problem["target_index"][indices]
    pool = problem["pool_index"][indices]
    control = problem["is_control"][indices].bool()
    context_mode = "pool_dynamic" if config.intent is RunIntent.COUNT_CONTEXT else "source_fixed"
    pool_state_mean: torch.Tensor | None = None
    pool_log_mass: torch.Tensor | None = None
    if config.intent is RunIntent.COUNT_CONTEXT:
        bank = DynamicPoolBank.from_series(
            problem["source_z"],
            problem["source_counts"].float() + config.source_smoothing,
            problem["pool_index"],
            pool_count=config.model.pool_count,
        )
        pool_state_mean, pool_log_mass = bank.state_mean, bank.log_mass
    if config.intent is RunIntent.COUNT_CONTEXT:
        prediction = source + duration[:, None] * model.drift(
            target,
            pool,
            control,
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
            z=source,
        )
    else:
        prediction = _deterministic_prediction(
            model, source, duration, target, pool, control, problem["grid_steps"][indices]
        )
    if phase == "state":
        train_state = decoder_data is None or config.training.train_state_with_gene_decoder
        empirical, optimization = _state_objective(
            model,
            problem,
            config,
            indices,
            prediction,
            train_state=train_state,
        )
        # Decoder-only training must leave the state channel completely absent
        # from the optimizer graph. A zero state gradient is insufficient
        # because AdamW still decays a parameter once its gradient is materialized.
        loss = optimization if train_state else empirical.detach()
        if decoder_data is not None:
            decoder_loss = _gene_decoder_loss(
                model, decoder_data, config, update, problem["source_z"].device
            )
            loss = loss + config.training.gene_decoder_loss_weight * decoder_loss
    else:
        # Exact count events use complete pool blocks and connect only to the
        # scalar-fitness channel. State-dynamics tensors are structurally absent
        # from this graph, so a count update cannot reopen drift or diffusion.
        raw = model.raw_fitness(
            problem["target_index"],
            problem["pool_index"],
            problem["is_control"].bool(),
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
        )
        count_loss = exact_count_loss(
            problem["terminal_counts"],
            problem["source_counts"],
            raw,
            problem["duration"],
            problem["pool_index"],
            model.log_concentration,
        )
        count_parameters = [
            model.pool_reference_fitness,
            model.target_fitness,
            model.log_concentration,
        ]
        if config.model.source_efficacy_sensitivity:
            count_parameters.append(model.guide_efficacy_logit)
        if config.intent is RunIntent.COUNT_CONTEXT:
            count_parameters.extend(
                [
                    model.fitness_pool_projection,
                    model.fitness_mass_context,
                    model.target_fitness_context,
                ]
            )
        loss = count_loss / max(float(problem["terminal_counts"].sum()), 1.0)
        loss = loss + 1e-4 * sum(parameter.square().mean() for parameter in count_parameters)
    return loss


def _new_model_optimizer(
    config: ResolvedConfig, device: torch.device
) -> tuple[CountSDEModel, torch.optim.Optimizer]:
    torch.manual_seed(config.training.initialization_seed)
    np.random.seed(config.training.initialization_seed)
    random.seed(config.training.initialization_seed)
    torch.use_deterministic_algorithms(config.training.deterministic)
    if torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
        torch.backends.cudnn.deterministic = config.training.deterministic
        torch.backends.cudnn.benchmark = not config.training.deterministic
    model = CountSDEModel(config.model, config.intent).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        betas=config.training.optimizer_betas,
        eps=config.training.optimizer_epsilon,
        weight_decay=config.training.optimizer_weight_decay,
    )
    return model, optimizer


def _should_checkpoint(update: int, end: int, config: ResolvedConfig) -> bool:
    return bool(
        update in config.training.state_checkpoint_updates
        or update % config.training.checkpoint_every == 0
        or update == end
    )


@torch.no_grad()
def _complete_state_objective_values(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    state_split: _StateSplit,
    config: ResolvedConfig,
) -> tuple[float, float]:
    indices = state_split.fit_indices
    prediction = _deterministic_prediction(
        model,
        problem["source_z"][indices],
        problem["duration"][indices],
        problem["target_index"][indices].long(),
        problem["pool_index"][indices].long(),
        problem["is_control"][indices].bool(),
        problem["grid_steps"][indices].long(),
    )
    empirical, optimization = _state_objective(
        model,
        problem,
        config,
        indices,
        prediction,
        train_state=True,
    )
    return float(empirical.cpu()), float(optimization.cpu())


@torch.no_grad()
def _exact_refit_state_objective(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    state_split: _StateSplit,
    config: ResolvedConfig,
) -> float:
    """Compatibility helper returning the canonical regularized objective."""

    return _complete_state_objective_values(model, problem, state_split, config)[1]


def _post_selection_refit(
    workspace: Path,
    training_root: Path,
    contract: CompiledRunContract,
    arrays: dict[str, np.ndarray[Any, Any]],
    config: ResolvedConfig,
    device: torch.device,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Refit the chosen update budget on every outer-training state series."""

    if not config.training.post_selection_state_refit:
        return selection
    if config.training.gene_decoder_batch_size:
        raise RuntimeError("Post-selection state refitting forbids decoder training.")
    refit_root = training_root / "refit"
    state_path = training_root / "refit-state.json"
    selected_update = int(selection["selected_update"])
    selected_family = str(selection["selected_family"])
    inner_checkpoint_id = str(selection["selected_checkpoint_id"])
    if refit_root.exists():
        verify_directory(refit_root)
        refit_receipt = json.loads((refit_root / "refit.json").read_text())
        if (
            refit_receipt.get("compiled_run_id") != contract.compiled_run_id
            or refit_receipt.get("selection_source_checkpoint_id") != inner_checkpoint_id
            or refit_receipt.get("selected_family") != selected_family
        ):
            raise ResumeMismatchError("Committed post-selection refit belongs to another decision.")
    else:
        atomic_json(
            state_path,
            {
                "schema_version": 1,
                "state": "REFIT_PLANNED",
                "compiled_run_id": contract.compiled_run_id,
                "selection_source_checkpoint_id": inner_checkpoint_id,
                "selected_family": selected_family,
                "selected_update": selected_update,
            },
        )
        atomic_json(
            state_path,
            {
                "schema_version": 1,
                "state": "REFIT_RUNNING",
                "compiled_run_id": contract.compiled_run_id,
                "selection_source_checkpoint_id": inner_checkpoint_id,
                "selected_family": selected_family,
                "selected_update": selected_update,
            },
        )

        def writer(temp: Path) -> None:
            model, optimizer = _new_model_optimizer(config, device)
            problem = _tensor_problem(arrays, device)
            state_split = _all_state_split(arrays, config, device)
            initialization = _initialize_state_channels(
                model, problem, state_split.fit_indices, config
            )
            if selected_family in {
                "shrunk_sister_guide_target_terminal",
                "selected_training_only_target_main",
                "target_plus_source_target_interaction",
            }:
                _materialize_target_main_baseline(
                    model,
                    problem,
                    state_split.fit_indices,
                    config,
                    str(selection["best_target_only_baseline"]),
                )
            if selected_family == "target_plus_source_target_interaction" and selected_update:
                for update in range(1, selected_update + 1):
                    optimizer.zero_grad(set_to_none=True)
                    loss = _loss(model, problem, config, update, None, state_split)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite refit loss at update {update}.")
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
            empirical_value, loss_value = _complete_state_objective_values(
                model, problem, state_split, config
            )
            diagnostics = {
                **initialization,
                **_training_diagnostics(model, problem, state_split, config, None),
                "post_selection_refit": "all_outer_training_state_series",
                "inner_selected_update": float(selected_update),
                "selected_family": selected_family,
            }
            checkpoint = _checkpoint(
                workspace,
                temp,
                contract,
                model,
                optimizer,
                update=selected_update,
                loss=loss_value,
                diagnostics=diagnostics,
                parent_checkpoint_id=None,
                selection_source_checkpoint_id=inner_checkpoint_id,
                artifact_training_root=refit_root,
            )
            refit_receipt = {
                "schema_version": 1,
                "state": "REFIT_COMMITTED",
                "compiled_run_id": contract.compiled_run_id,
                "selection_source_checkpoint_id": inner_checkpoint_id,
                "inner_selected_update": selected_update,
                "selected_family": selected_family,
                "refit_checkpoint_id": checkpoint.checkpoint_id,
                "refit_series_hash": state_split.fit_series_hash,
                "refit_series": int(len(state_split.fit_indices)),
                "unregularized_target_balanced_state_mse": empirical_value,
                "regularized_optimization_objective": loss_value,
            }
            (temp / "refit.json").write_bytes(canonical_json_bytes(refit_receipt) + b"\n")

        publish_directory(refit_root, writer)
        refit_receipt = json.loads((refit_root / "refit.json").read_text())

    updated = {
        **selection,
        "selection_id": "pending",
        "inner_selected_update": selected_update,
        "inner_selected_checkpoint_id": inner_checkpoint_id,
        "selected_checkpoint_id": str(refit_receipt["refit_checkpoint_id"]),
        "refit_checkpoint_id": str(refit_receipt["refit_checkpoint_id"]),
        "selected_checkpoint_relative_uri": (
            f"training/refit/checkpoints/generation-{selected_update:09d}"
        ),
        "post_selection_refit": True,
        "refit_series_hash": str(refit_receipt["refit_series_hash"]),
    }
    updated["candidate_updates"] = tuple(updated["candidate_updates"])
    normalized = SelectionManifest.model_construct(**updated).model_dump(mode="json")
    normalized["selection_id"] = contract_id(normalized, id_field="selection_id")
    manifest = SelectionManifest.model_validate(normalized)
    serialized = manifest.model_dump(mode="json")
    atomic_json(training_root / "selection.json", serialized)
    atomic_json(
        state_path,
        {
            "schema_version": 1,
            "state": "REFIT_COMMITTED",
            "compiled_run_id": contract.compiled_run_id,
            "selection_id": manifest.selection_id,
            "refit_checkpoint_id": manifest.refit_checkpoint_id,
        },
    )
    return serialized


def train_model(
    config_path: Path,
    *,
    device: str | torch.device | None = None,
    stop_after: int | None = None,
    initial_checkpoint: Path | None = None,
) -> Path:
    from ..prepare.pipeline import load_config

    requested_config = load_config(config_path)
    if initial_checkpoint is not None and requested_config.model.source_target_interaction_rank:
        raise ResumeMismatchError(
            "Null-guarded source-target pilots cannot fork from a checkpoint."
        )
    root = config_path.parent.resolve()
    workspace = (root / requested_config.workspace).resolve()
    verify_directory(workspace / "compiled")
    contract, arrays = load_compiled_problem(workspace)
    config = ResolvedConfig.model_validate_json(
        (workspace / "compiled" / "config.json").read_text()
    )
    if config.model_dump(mode="json") != requested_config.model_dump(mode="json"):
        raise ResumeMismatchError("External configuration changed after compilation.")
    if contract.implementation_tree_hash != implementation_tree_hash():
        raise ResumeMismatchError("Implementation changed after compilation.")
    if contract.environment_lock_hash != environment_lock_hash():
        raise ResumeMismatchError("Environment changed after compilation.")
    training_root = workspace / "training"
    if training_root.exists():
        raise FileExistsError("Training attempt already exists; use resume.")
    (training_root / "checkpoints").mkdir(parents=True)
    (training_root / "intent.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "compiled_run_id": contract.compiled_run_id,
                "config_hash": contract.resolved_config_hash,
            }
        )
        + b"\n"
    )
    selected_device = _device(device)
    _validate_pilot_device(config, selected_device)
    model, optimizer = _new_model_optimizer(config, selected_device)
    fork_parent: str | None = None
    if initial_checkpoint is not None:
        verify_directory(initial_checkpoint)
        parent = CheckpointManifest.model_validate_json(
            (initial_checkpoint / "checkpoint.json").read_text()
        )
        state = load_tensor_file(initial_checkpoint / "model.safetensors", device=selected_device)
        model.load_state_dict(state, strict=True)
        fork_parent = parent.checkpoint_id
    problem = _tensor_problem(arrays, selected_device)
    state_split = _state_split(arrays, config, selected_device)
    decoder_data = _gene_decoder_data(root, workspace, config)
    initialization = (
        _initialize_state_channels(model, problem, state_split.fit_indices, config)
        if initial_checkpoint is None
        else {}
    )
    end = min(stop_after or config.training.max_updates, config.training.max_updates)
    checkpoint: CheckpointManifest | None = None
    loss_value = float("nan")
    if config.training.analytic_fit:
        diagnostics = {
            **initialization,
            **_training_diagnostics(model, problem, state_split, config, decoder_data),
        }
        checkpoint = _checkpoint(
            workspace,
            training_root,
            contract,
            model,
            optimizer,
            update=1,
            loss=float(diagnostics["full_state_rmse"]) ** 2,
            diagnostics=diagnostics,
            parent_checkpoint_id=fork_parent,
        )
        _write_selection(training_root, contract, config, [checkpoint])
        return training_root
    checkpoints: list[CheckpointManifest] = []
    if config.training.checkpoint_selection == "minimum_state_validation_null_guarded":
        diagnostics = {
            **initialization,
            **_training_diagnostics(model, problem, state_split, config, decoder_data),
        }
        checkpoint = _checkpoint(
            workspace,
            training_root,
            contract,
            model,
            optimizer,
            update=0,
            loss=float(diagnostics["full_state_rmse"]) ** 2,
            diagnostics=diagnostics,
            parent_checkpoint_id=fork_parent,
        )
        checkpoints.append(checkpoint)
        if config.model.source_target_interaction_rank:
            _materialize_target_main_baseline(
                model,
                problem,
                state_split.fit_indices,
                config,
                str(diagnostics["state_validation_best_target_only_baseline"]),
            )
    for update in range(1, end + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, problem, config, update, decoder_data, state_split)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {update}.")
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if _should_checkpoint(update, end, config):
            diagnostics = {
                **initialization,
                **_training_diagnostics(model, problem, state_split, config, decoder_data),
            }
            checkpoint = _checkpoint(
                workspace,
                training_root,
                contract,
                model,
                optimizer,
                update=update,
                loss=loss_value,
                diagnostics=diagnostics,
                parent_checkpoint_id=checkpoint.checkpoint_id if checkpoint else fork_parent,
            )
            checkpoints.append(checkpoint)
    if end == config.training.max_updates:
        selection = _write_selection(training_root, contract, config, checkpoints)
        _post_selection_refit(
            workspace,
            training_root,
            contract,
            arrays,
            config,
            selected_device,
            selection,
        )
    return training_root


def _load_latest(
    workspace: Path,
    config: ResolvedConfig,
    device: torch.device,
) -> tuple[
    CompiledRunContract,
    dict[str, np.ndarray[Any, Any]],
    CountSDEModel,
    torch.optim.Optimizer,
    CheckpointManifest,
]:
    contract, arrays = load_compiled_problem(workspace)
    training_root = workspace / "training"
    intent = json.loads((training_root / "intent.json").read_text())
    if (
        intent["compiled_run_id"] != contract.compiled_run_id
        or intent["config_hash"] != contract.resolved_config_hash
    ):
        raise ResumeMismatchError("Compiled contract changed across resume.")
    latest = json.loads((training_root / "checkpoints" / "latest.json").read_text())
    generation = training_root / latest["relative_uri"]
    manifest = CheckpointManifest.model_validate_json((generation / "checkpoint.json").read_text())
    model, optimizer = _new_model_optimizer(config, device)
    state = load_tensor_file(generation / "model.safetensors", device=device)
    model.load_state_dict(state, strict=True)
    opt_tensors = load_tensor_file(generation / "optimizer.safetensors", device=device)
    opt_tree = json.loads((generation / "optimizer-tree.json").read_text())
    _restore_optimizer(model, optimizer, opt_tensors, opt_tree)
    rng = load_tensor_file(generation / "rng.safetensors")
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if device.type == "cuda":
        states = [rng[key].cpu() for key in sorted(rng) if key.startswith("torch_cuda_")]
        if states:
            torch.cuda.set_rng_state_all(states)
    return contract, arrays, model, optimizer, manifest


def resume_training(config_path: Path, *, device: str | torch.device | None = None) -> Path:
    from ..prepare.pipeline import load_config

    requested_config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / requested_config.workspace).resolve()
    verify_directory(workspace / "compiled")
    compiled_config = ResolvedConfig.model_validate_json(
        (workspace / "compiled" / "config.json").read_text()
    )
    if compiled_config.model_dump(mode="json") != requested_config.model_dump(mode="json"):
        raise ResumeMismatchError("External configuration changed after compilation.")
    contract, _ = load_compiled_problem(workspace)
    if contract.implementation_tree_hash != implementation_tree_hash():
        raise ResumeMismatchError("Implementation changed after compilation.")
    if contract.environment_lock_hash != environment_lock_hash():
        raise ResumeMismatchError("Environment changed after compilation.")
    config = compiled_config
    selected_device = _device(device)
    _validate_pilot_device(config, selected_device)
    contract, arrays, model, optimizer, previous = _load_latest(workspace, config, selected_device)
    if previous.compiled_run_id != contract.compiled_run_id:
        raise ResumeMismatchError("Checkpoint belongs to another compiled run.")
    if previous.update >= config.training.max_updates:
        selection_path = workspace / "training/selection.json"
        if not selection_path.exists():
            raise ResumeMismatchError("Training is complete but checkpoint selection is absent.")
        selection = json.loads(selection_path.read_text())
        if config.training.post_selection_state_refit and not selection.get(
            "post_selection_refit", False
        ):
            _post_selection_refit(
                workspace,
                workspace / "training",
                contract,
                arrays,
                config,
                selected_device,
                selection,
            )
            return workspace / "training"
        raise ResumeMismatchError("Training is already complete under the compiled budget.")
    problem = _tensor_problem(arrays, selected_device)
    state_split = _state_split(arrays, config, selected_device)
    decoder_data = _gene_decoder_data(root, workspace, config)
    if config.model.source_target_interaction_rank and previous.update == 0:
        previous_state = json.loads(
            (
                workspace / "training/checkpoints/generation-000000000/training-state.json"
            ).read_text()
        )
        _materialize_target_main_baseline(
            model,
            problem,
            state_split.fit_indices,
            config,
            str(previous_state["diagnostics"]["state_validation_best_target_only_baseline"]),
        )
    training_root = workspace / "training"
    checkpoint = previous
    checkpoints: list[CheckpointManifest] = []
    for generation in sorted((training_root / "checkpoints").glob("generation-*")):
        checkpoints.append(
            CheckpointManifest.model_validate_json((generation / "checkpoint.json").read_text())
        )
    for update in range(previous.update + 1, config.training.max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, problem, config, update, decoder_data, state_split)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {update}.")
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if _should_checkpoint(update, config.training.max_updates, config):
            diagnostics = _training_diagnostics(model, problem, state_split, config, decoder_data)
            checkpoint = _checkpoint(
                workspace,
                training_root,
                contract,
                model,
                optimizer,
                update=update,
                loss=float(loss.detach().cpu()),
                diagnostics=diagnostics,
                parent_checkpoint_id=checkpoint.checkpoint_id,
            )
            checkpoints.append(checkpoint)
    selection = _write_selection(training_root, contract, config, checkpoints)
    _post_selection_refit(
        workspace,
        training_root,
        contract,
        arrays,
        config,
        selected_device,
        selection,
    )
    return training_root


def load_training_state(
    config_path: Path, *, device: str | torch.device = "cpu"
) -> tuple[CompiledRunContract, dict[str, np.ndarray[Any, Any]], CountSDEModel, CheckpointManifest]:
    from ..prepare.pipeline import load_config

    config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / config.workspace).resolve()
    contract, arrays, model, _, checkpoint = _load_latest(workspace, config, _device(device))
    if config.training.checkpoint_selection in {
        "minimum_gene_decoder_validation",
        "minimum_state_validation",
        "minimum_state_validation_null_guarded",
    }:
        selection_manifest = SelectionManifest.model_validate_json(
            (workspace / "training/selection.json").read_text()
        )
        selection = selection_manifest.model_dump(mode="json")
        if selection_manifest.compiled_run_id != contract.compiled_run_id:
            raise ResumeMismatchError("Checkpoint selection belongs to another compiled run.")
        selected_update = int(selection["selected_update"])
        selected_checkpoint_id = str(selection["selected_checkpoint_id"])
        selection_relative = selection.get("selected_checkpoint_relative_uri")
    else:
        selected_update = (
            config.training.selected_update
            if config.training.selected_update is not None
            else config.training.max_updates
        )
        selected_checkpoint_id = None
        selection_relative = None
    if checkpoint.update != selected_update or (
        selected_checkpoint_id is not None and checkpoint.checkpoint_id != selected_checkpoint_id
    ):
        generation = (
            workspace / selection_relative
            if selection_relative
            else workspace / "training/checkpoints" / f"generation-{selected_update:09d}"
        )
        verify_directory(generation)
        checkpoint = CheckpointManifest.model_validate_json(
            (generation / "checkpoint.json").read_text()
        )
        if checkpoint.compiled_run_id != contract.compiled_run_id:
            raise ResumeMismatchError("Selected checkpoint belongs to another compiled run.")
        state = load_tensor_file(generation / "model.safetensors", device=device)
        model.load_state_dict(state, strict=True)
    return contract, arrays, model, checkpoint
