"""Immutable inference bundles and bounded streaming operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..canonical import canonical_json_bytes, contract_id, sha256_file
from ..compile import load_compiled_problem
from ..contracts import (
    ArtifactRef,
    CompiledRunContract,
    CounterfactualDesign,
    InferenceBundleManifest,
    ResolvedConfig,
    ResolvedRunCapabilities,
    SelectionManifest,
)
from ..errors import CapabilityError, IntegrityError
from ..model import CountSDEModel
from ..numerics import rollout, rollout_with_context_schedule
from ..persistence import load_tensor_file, publish_directory, save_tensor_file, verify_directory
from ..runtime_identity import (
    environment_lock_hash,
    implementation_tree_hash,
    recipe_distribution_hash,
)
from ..training import load_training_state


def _future_ref(
    workspace: Path,
    final: Path,
    temporary: Path,
    *,
    schema_id: str,
    media_type: str,
) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema_id,
        schema_version=1,
        sha256=sha256_file(temporary),
        size_bytes=temporary.stat().st_size,
        media_type=media_type,
        relative_uri=final.relative_to(workspace).as_posix(),
    )


def _verify_ref(workspace: Path, reference: ArtifactRef) -> Path:
    path = workspace / reference.relative_uri
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"Referenced artifact is absent or not regular: {path}.")
    if path.stat().st_size != reference.size_bytes or sha256_file(path) != reference.sha256:
        raise IntegrityError(f"Referenced artifact bytes do not match: {path}.")
    return path


def finalize_inference(config_path: Path) -> Path:
    import yaml

    root = config_path.parent.resolve()
    raw = yaml.safe_load(config_path.read_text())
    workspace = (root / raw["workspace"]).resolve()
    config = ResolvedConfig.model_validate_json(
        (workspace / "compiled" / "config.json").read_text()
    )
    contract, arrays, model, checkpoint = load_training_state(config_path, device="cpu")
    selection = SelectionManifest.model_validate_json(
        (workspace / "training" / "selection.json").read_text()
    )
    if selection.selected_checkpoint_id != checkpoint.checkpoint_id:
        raise IntegrityError("Selection manifest does not bind the finalized checkpoint.")
    destination = workspace / "inference"
    manifest_holder: dict[str, InferenceBundleManifest] = {}

    def writer(temp: Path) -> None:
        save_tensor_file(temp / "model.safetensors", model.state_dict())
        (temp / "run-contract.json").write_bytes(
            canonical_json_bytes(contract.model_dump(mode="json")) + b"\n"
        )
        (temp / "selection-manifest.json").write_bytes(
            canonical_json_bytes(selection.model_dump(mode="json")) + b"\n"
        )
        seed_plan = {
            "schema_version": 1,
            "seed": config.evaluation.seed,
            "particles": config.evaluation.particles,
            "steps": config.evaluation.steps,
            "maximum_physical_step": config.evaluation.max_step_duration,
            "resolved_steps": arrays["grid_steps"].tolist(),
            "common_noise": True,
        }
        (temp / "evaluation-seed-plan.json").write_bytes(canonical_json_bytes(seed_plan) + b"\n")
        output_schema = {
            "schema_version": 1,
            "selected_family": selection.selected_family,
            "terminal_state": "weighted_particle_mean",
            "gene_output": (
                "composition" if contract.capabilities.decode_gene_composition else None
            ),
            "mass": "relative_within_pool" if contract.capabilities.predict_relative_mass else None,
        }
        (temp / "output-schema.json").write_bytes(canonical_json_bytes(output_schema) + b"\n")
        final = destination
        payload = {
            "schema_version": 1,
            "run_id": "pending",
            "compiled_run_id": contract.compiled_run_id,
            "selected_checkpoint_id": checkpoint.checkpoint_id,
            "recipe_id": contract.recipe_id,
            "recipe_version": contract.recipe_version,
            "selected_family": selection.selected_family,
            "model": _future_ref(
                workspace,
                final / "model.safetensors",
                temp / "model.safetensors",
                schema_id="credo.inference_model",
                media_type="application/x-safetensors",
            ).model_dump(mode="json"),
            "run_contract": _future_ref(
                workspace,
                final / "run-contract.json",
                temp / "run-contract.json",
                schema_id="credo.compiled_run",
                media_type="application/json",
            ).model_dump(mode="json"),
            "selection": _future_ref(
                workspace,
                final / "selection-manifest.json",
                temp / "selection-manifest.json",
                schema_id="credo.selection_manifest",
                media_type="application/json",
            ).model_dump(mode="json"),
            "evaluation_seed_plan": _future_ref(
                workspace,
                final / "evaluation-seed-plan.json",
                temp / "evaluation-seed-plan.json",
                schema_id="credo.evaluation_seed",
                media_type="application/json",
            ).model_dump(mode="json"),
            "output_schema": _future_ref(
                workspace,
                final / "output-schema.json",
                temp / "output-schema.json",
                schema_id="credo.output_schema",
                media_type="application/json",
            ).model_dump(mode="json"),
            "capabilities": contract.capabilities.model_dump(mode="json"),
        }
        payload["run_id"] = contract_id(payload, id_field="run_id")
        manifest = InferenceBundleManifest.model_validate(payload)
        manifest_holder["value"] = manifest
        (temp / "inference.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(destination, writer)
    return destination


@dataclass
class V4Run:
    workspace: Path
    manifest: InferenceBundleManifest
    contract: CompiledRunContract
    config: ResolvedConfig
    arrays: dict[str, np.ndarray[Any, Any]]
    model: CountSDEModel
    device: torch.device

    @property
    def capabilities(self) -> ResolvedRunCapabilities:
        return self.manifest.capabilities

    def _problem(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, value in self.arrays.items():
            if value.dtype.kind in "USO":
                continue
            result[name] = torch.from_numpy(value).to(self.device)
        return result

    def terminal(
        self,
        *,
        particles: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        effect_mode: str = "factual",
        context_mode: str = "source_fixed",
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        problem = self._problem()
        requested_particles = particles or self.config.evaluation.particles
        requested_seed = self.config.evaluation.seed if seed is None else seed
        step_plan = (
            torch.full_like(problem["grid_steps"], steps)
            if steps is not None
            else problem["grid_steps"]
        )
        z_parts: list[torch.Tensor] = []
        weight_parts: list[torch.Tensor] = []
        mass_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        for group_steps in torch.unique(step_plan).tolist():
            indices = torch.where(step_plan == group_steps)[0]
            z_group, weights_group, mass_group = rollout(
                self.model,
                problem["source_z"][indices].float(),
                problem["duration"][indices].float(),
                problem["target_index"][indices].long(),
                problem["pool_index"][indices].long(),
                problem["is_control"][indices].bool(),
                problem["source_counts"][indices].float() + 0.5,
                particles=requested_particles,
                steps=int(group_steps),
                seed=requested_seed + int(group_steps) * 1_000_003,
                effect_mode=effect_mode,
                context_mode=context_mode,
            )
            index_parts.append(indices)
            z_parts.append(z_group)
            weight_parts.append(weights_group)
            mass_parts.append(mass_group)
        order = torch.argsort(torch.cat(index_parts))
        z = torch.cat(z_parts)[order]
        weights = torch.cat(weight_parts)[order]
        mass = torch.cat(mass_parts)[order]
        mean = (z * weights.unsqueeze(-1)).sum(dim=1)
        return mean.cpu().numpy(), mass.cpu().numpy(), weights.cpu().numpy()

    def decode_composition(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if not self.capabilities.decode_gene_composition:
            raise NotImplementedError(
                "This representation has no validated gene-composition decoder."
            )
        values = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device)
        rows: list[np.ndarray[Any, Any]] = []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                logits = self.model.decode_logits(values[start : start + 256])
                rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(rows, axis=0)

    def predict(self, output: Path) -> Path:
        mean, mass, _ = self.terminal()
        payload: dict[str, np.ndarray[Any, Any]] = {
            "series_ids": self.arrays["series_ids"],
            "terminal_mean": mean,
            "relative_mass": mass,
        }
        if self.capabilities.decode_gene_composition:
            payload["gene_composition"] = self.decode_composition(mean)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            np.savez(handle, **payload)  # type: ignore[arg-type]
        if output.stat().st_size > self.config.evaluation.output_bytes_limit:
            output.unlink()
            raise OSError("Prediction exceeded the compiled output quota.")
        return output

    def counterfactual(self, design: CounterfactualDesign, output: Path) -> Path:
        if design.series_index >= len(self.arrays["series_ids"]):
            raise ValueError("Counterfactual series index is outside the compiled catalog.")
        if any(branch.context_mode != "source_fixed" for branch in design.branches):
            if not self.capabilities.dynamic_context_counterfactual:
                raise CapabilityError("Dynamic context was not compiled for this run.")
        problem = self._problem()
        factual_schedule = None
        reference_schedule = None
        if any(branch.context_mode != "source_fixed" for branch in design.branches):
            _, _, _, factual_schedule = rollout_with_context_schedule(
                self.model,
                problem["source_z"].float(),
                problem["duration"].float(),
                problem["target_index"].long(),
                problem["pool_index"].long(),
                problem["is_control"].bool(),
                problem["source_counts"].float() + 0.5,
                particles=design.particles,
                steps=design.steps,
                seed=design.seed,
                effect_mode="factual",
            )
            _, _, _, reference_schedule = rollout_with_context_schedule(
                self.model,
                problem["source_z"].float(),
                problem["duration"].float(),
                problem["target_index"].long(),
                problem["pool_index"].long(),
                problem["is_control"].bool(),
                problem["source_counts"].float() + 0.5,
                particles=design.particles,
                steps=design.steps,
                seed=design.seed,
                effect_mode="reference",
            )
        rows: list[dict[str, Any]] = []
        for branch in design.branches:
            schedule = (
                factual_schedule
                if branch.context_mode == "pool_dynamic"
                else reference_schedule
                if branch.context_mode == "reference_dynamic"
                else None
            )
            z, weights, mass_tensor = rollout(
                self.model,
                problem["source_z"].float(),
                problem["duration"].float(),
                problem["target_index"].long(),
                problem["pool_index"].long(),
                problem["is_control"].bool(),
                problem["source_counts"].float() + 0.5,
                particles=design.particles,
                steps=design.steps,
                seed=design.seed,
                effect_mode=branch.effect_mode,
                context_mode=branch.context_mode,
                context_schedule=schedule,
            )
            mean = (z * weights.unsqueeze(-1)).sum(dim=1).cpu().numpy()
            mass = mass_tensor.cpu().numpy()
            index = design.series_index
            rows.append(
                {
                    "branch_id": branch.branch_id,
                    "effect_mode": branch.effect_mode,
                    "context_mode": branch.context_mode,
                    "terminal_mean": mean[index].tolist(),
                    "relative_mass": float(mass[index]),
                }
            )
        by_mode = {(row["effect_mode"], row["context_mode"]): row for row in rows}
        contrasts: dict[str, Any] = {}
        required = {
            ("factual", "pool_dynamic"),
            ("reference", "pool_dynamic"),
            ("factual", "reference_dynamic"),
            ("reference", "reference_dynamic"),
        }
        if required <= set(by_mode):
            factual_pool = by_mode[("factual", "pool_dynamic")]
            reference_pool = by_mode[("reference", "pool_dynamic")]
            factual_reference = by_mode[("factual", "reference_dynamic")]
            reference_reference = by_mode[("reference", "reference_dynamic")]
            pool_state = np.asarray(factual_pool["terminal_mean"]) - np.asarray(
                reference_pool["terminal_mean"]
            )
            reference_state = np.asarray(factual_reference["terminal_mean"]) - np.asarray(
                reference_reference["terminal_mean"]
            )
            pool_mass = factual_pool["relative_mass"] - reference_pool["relative_mass"]
            reference_mass = (
                factual_reference["relative_mass"] - reference_reference["relative_mass"]
            )
            contrasts = {
                "delta_pool": {
                    "terminal_mean": pool_state.tolist(),
                    "relative_mass": pool_mass,
                },
                "delta_reference": {
                    "terminal_mean": reference_state.tolist(),
                    "relative_mass": reference_mass,
                },
                "delta_context_susceptibility": {
                    "terminal_mean": (pool_state - reference_state).tolist(),
                    "relative_mass": pool_mass - reference_mass,
                },
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            canonical_json_bytes({"schema_version": 1, "branches": rows, "contrasts": contrasts})
            + b"\n"
        )
        if len(payload) > self.config.evaluation.output_bytes_limit:
            raise OSError("Counterfactual exceeded the compiled output quota.")
        with output.open("xb") as handle:
            handle.write(payload)
        return output


def open_inference_run(
    path: Path, *, device: str | torch.device = "cpu", verify: str = "full"
) -> V4Run:
    if verify in {"content", "reload", "full"}:
        verify_directory(path)
    manifest = InferenceBundleManifest.model_validate_json((path / "inference.json").read_text())
    contract = CompiledRunContract.model_validate_json((path / "run-contract.json").read_text())
    workspace = path.parent
    for reference in (
        manifest.model,
        manifest.run_contract,
        manifest.selection,
        manifest.evaluation_seed_plan,
        manifest.output_schema,
    ):
        _verify_ref(workspace, reference)
    if manifest.compiled_run_id != contract.compiled_run_id:
        raise IntegrityError("Inference parent contract mismatch.")
    selection = SelectionManifest.model_validate_json(
        (path / "selection-manifest.json").read_text()
    )
    if (
        selection.compiled_run_id != contract.compiled_run_id
        or selection.selected_checkpoint_id != manifest.selected_checkpoint_id
        or selection.selected_family != manifest.selected_family
    ):
        raise IntegrityError("Inference selection binding mismatch.")
    if contract.implementation_tree_hash != implementation_tree_hash():
        raise IntegrityError("Installed V4 implementation differs from the compiled run.")
    if contract.recipe_wheel_hash != recipe_distribution_hash():
        raise IntegrityError("Recipe package identity differs from the compiled run.")
    if contract.environment_lock_hash != environment_lock_hash():
        raise IntegrityError("Installed environment identity differs from the compiled run.")
    if verify in {"content", "reload", "full"}:
        verify_directory(workspace / "compiled")
        verify_directory(workspace / "prepared")
    loaded_contract, arrays = load_compiled_problem(workspace)
    if loaded_contract.compiled_run_id != contract.compiled_run_id:
        raise IntegrityError("Workspace compiled contract differs from inference bundle.")
    config = ResolvedConfig.model_validate_json(
        (workspace / "compiled" / "config.json").read_text()
    )
    selected = torch.device(device)
    model = CountSDEModel(config.model, config.intent).to(selected)
    state = load_tensor_file(path / "model.safetensors", device=selected)
    model.load_state_dict(state, strict=True)
    if model.source_target_main_weight is not None:
        alpha = float(model.source_target_main_weight.detach().cpu())
        if not 0.0 <= alpha <= model.config.source_target_main_max_weight:
            raise IntegrityError("Deployed target-main shrinkage weight is outside its bound.")
    model.eval()
    return V4Run(workspace, manifest, contract, config, arrays, model, selected)
