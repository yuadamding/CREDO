"""Sparse-safe scale-only representation preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from scipy import sparse
from scipy.sparse.linalg import svds

from ..canonical import canonical_json_bytes, contract_id, sha256_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    FeatureKey,
    InformationSet,
    PreparedRepresentation,
    ResolvedConfig,
)
from ..errors import ContractError
from ..persistence import publish_directory, save_tensor_file
from ..store import CountStore


def load_config(path: Path) -> ResolvedConfig:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ContractError("Configuration must be a YAML mapping.")
    return ResolvedConfig.model_validate(payload)


def _ref_for_future(
    workspace: Path,
    final_path: Path,
    temporary_path: Path,
    *,
    schema_id: str,
    media_type: str,
) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema_id,
        schema_version=1,
        sha256=sha256_file(temporary_path),
        size_bytes=temporary_path.stat().st_size,
        media_type=media_type,
        relative_uri=final_path.relative_to(workspace).as_posix(),
    )


def _library_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.astype(np.float64).tocsr(copy=True)
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    factors = np.divide(10_000.0, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = sparse.diags(factors) @ matrix
    return matrix.tocsr()


def _apply_multiplicative_offsets(
    matrix: sparse.csr_matrix,
    batch_ids: np.ndarray[Any, Any],
    offsets: dict[str, list[float]],
) -> sparse.csr_matrix:
    """Apply ``X * exp(-beta_batch)`` without changing the zero pattern."""

    coo = matrix.tocoo(copy=True)
    for batch, values in offsets.items():
        row_mask = batch_ids[coo.row] == batch
        if np.any(row_mask):
            beta = np.asarray(values, dtype=np.float64)
            coo.data[row_mask] *= np.exp(-beta[coo.col[row_mask]])
    return coo.tocsr()


def _fit_corrected_view(
    config_root: Path,
    config: ResolvedConfig,
    fit_rows: np.ndarray[Any, Any],
    fit_matrix: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Fit a conservative matched-reference multiplicative offset.

    Metadata is a JSON mapping with ``row_ids``, ``batch_ids``, ``protected_ids``
    and ``is_reference``. Full rank, minimum support, connected overlap, and
    conditioning are enforced. Failure is an explicit identity fallback.
    """

    if config.input_view == "identity_library_normalized":
        return fit_matrix, {"selected": "identity_library_normalized", "reason": "requested"}
    metadata = json.loads((config_root / str(config.correction_metadata)).read_text())
    ids = np.asarray(metadata["row_ids"], dtype=np.int64)
    batches = np.asarray(metadata["batch_ids"], dtype=str)
    protected = np.asarray(metadata["protected_ids"], dtype=str)
    reference = np.asarray(metadata["is_reference"], dtype=bool)
    if not (len(ids) == len(batches) == len(protected) == len(reference)):
        raise ContractError("Correction metadata columns have unequal lengths.")
    lookup = {int(row_id): index for index, row_id in enumerate(ids)}
    if any(int(row_id) not in lookup for row_id in fit_rows):
        raise ContractError("Correction metadata does not cover every fit row.")
    positions = np.asarray([lookup[int(row_id)] for row_id in fit_rows], dtype=np.int64)
    batches, protected, reference = batches[positions], protected[positions], reference[positions]
    selected = reference
    unique_batches = sorted(set(batches[selected]))
    unique_protected = sorted(set(protected[selected]))
    counts_ok = all(np.sum(selected & (batches == value)) >= 32 for value in unique_batches)
    counts_ok &= all(np.sum(selected & (protected == value)) >= 32 for value in unique_protected)
    columns = [np.ones(len(fit_rows))]
    columns += [(batches == value).astype(float) for value in unique_batches[1:]]
    columns += [(protected == value).astype(float) for value in unique_protected[1:]]
    design = np.stack(columns, axis=1)[selected]
    rank = int(np.linalg.matrix_rank(design)) if design.size else 0
    condition = (
        float(np.linalg.cond(design)) if design.size and rank == design.shape[1] else float("inf")
    )
    overlap = {
        (batch, protected_value)
        for batch, protected_value in zip(batches[selected], protected[selected], strict=True)
    }
    connected = all(
        any((batch, protected_value) in overlap for protected_value in unique_protected)
        for batch in unique_batches
    ) and all(
        any((batch, value) in overlap for batch in unique_batches) for value in unique_protected
    )
    validation = metadata.get("validation", {})
    validation_gate = (
        float(validation.get("reconstruction_delta", float("inf"))) <= 0.0
        and float(validation.get("state_prediction_delta", float("inf"))) <= 0.0
        and float(validation.get("technical_separability_delta", float("inf"))) < 0.0
        and float(validation.get("protected_program_correlation", float("-inf"))) >= 0.95
    )
    gate = (
        counts_ok and rank == design.shape[1] and condition <= 1e4 and connected and validation_gate
    )
    audit = {
        "requested": "matched_control_offset_v1",
        "full_rank": rank == design.shape[1],
        "rank": rank,
        "columns": int(design.shape[1]),
        "condition_number": condition if np.isfinite(condition) else None,
        "minimum_support": counts_ok,
        "overlap_connected": connected,
        "validation": validation,
        "validation_gate": validation_gate,
        "corrected_view_model": (
            "X_enc[i,j] = library_normalize(X_raw)[i,j] * exp(-beta[batch(i),j])"
        ),
        "zero_preserving": True,
        "selected": "matched_control_offset_v1" if gate else "identity_library_normalized",
        "reason": "all_gates_pass" if gate else "fail_closed_identity",
    }
    if not gate:
        return fit_matrix, audit
    dense_reference = fit_matrix[selected].toarray()
    global_mean = dense_reference.mean(axis=0)
    offsets: dict[str, list[float]] = {}
    corrected = fit_matrix.tolil(copy=True)
    for batch in unique_batches:
        mask = selected & (batches == batch)
        batch_mean = fit_matrix[mask].toarray().mean(axis=0)
        offset = np.log(batch_mean + 1e-8) - np.log(global_mean + 1e-8)
        offsets[batch] = offset.tolist()
    audit["analytic_offsets"] = offsets
    corrected = _apply_multiplicative_offsets(corrected.tocsr(), batches, offsets)
    return corrected, audit


def _apply_frozen_view(
    config_root: Path,
    config: ResolvedConfig,
    row_ids: np.ndarray[Any, Any],
    matrix: sparse.csr_matrix,
    audit: dict[str, Any],
) -> sparse.csr_matrix:
    if audit["selected"] != "matched_control_offset_v1":
        return matrix
    metadata = json.loads((config_root / str(config.correction_metadata)).read_text())
    ids = np.asarray(metadata["row_ids"], dtype=np.int64)
    batches = np.asarray(metadata["batch_ids"], dtype=str)
    lookup = {int(row_id): index for index, row_id in enumerate(ids)}
    if any(int(row_id) not in lookup for row_id in row_ids):
        raise ContractError("Correction metadata does not cover every encoded row.")
    ordered_batches = batches[[lookup[int(row_id)] for row_id in row_ids]]
    return _apply_multiplicative_offsets(matrix, ordered_batches, audit["analytic_offsets"])


def _randomized_right_singular_vectors(
    matrix: sparse.csr_matrix,
    rank: int,
    *,
    seed: int = 0,
    oversample: int = 8,
) -> np.ndarray[Any, Any]:
    """Return a deterministic fixed-work approximation for a large sparse matrix.

    ARPACK can require hundreds of full sparse passes for the full-gene cohort.
    This range finder uses four sparse products and one small dense SVD, keeping
    preparation bounded while preserving an orthonormal right-singular basis.
    """

    width = min(rank + oversample, min(matrix.shape))
    generator = np.random.default_rng(seed)
    probe = generator.standard_normal((matrix.shape[1], width), dtype=np.float32)
    sample = np.asarray(matrix @ probe, dtype=np.float32)
    basis, _ = np.linalg.qr(sample, mode="reduced")
    # One power iteration materially improves the leading subspace without an
    # open-ended convergence loop.
    dual = np.asarray(matrix.T @ basis, dtype=np.float32)
    sample = np.asarray(matrix @ dual, dtype=np.float32)
    basis, _ = np.linalg.qr(sample, mode="reduced")
    compressed = np.asarray(basis.T @ matrix, dtype=np.float32)
    _, _, right = np.linalg.svd(compressed, full_matrices=False)
    return np.asarray(right[:rank], dtype=np.float32)


def _fit_components(
    matrix: sparse.csr_matrix, state_dim: int, *, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    squared = matrix.copy()
    squared.data **= 2
    scale = np.sqrt(np.asarray(squared.mean(axis=0)).reshape(-1))
    scale[scale < 1e-8] = 1.0
    scaled = matrix @ sparse.diags(1.0 / scale)
    available = min(scaled.shape)
    k = min(state_dim, max(1, available - 1))
    if k == 1 and available <= 1:
        components = np.ones((1, scaled.shape[1]), dtype=np.float64)
        components /= np.linalg.norm(components)
    elif scaled.nnz >= 10_000_000:
        components = _randomized_right_singular_vectors(scaled.astype(np.float32), k, seed=seed)
    else:
        _, singular, vt = svds(scaled.astype(np.float64), k=k, which="LM", random_state=seed)
        order = np.argsort(singular)[::-1]
        components = vt[order]
    if components.shape[0] < state_dim:
        components = np.pad(components, ((0, state_dim - components.shape[0]), (0, 0)))
    projected = np.asarray(scaled @ components.T, dtype=np.float64)
    total_energy = float(scaled.multiply(scaled).sum())
    explained_energy = float(np.square(projected).sum())
    ratio = explained_energy / total_energy if total_energy > 0 else 0.0
    return components.astype(np.float32), scale.astype(np.float32), ratio


def prepare_representation(config_path: Path) -> Path:
    config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / config.workspace).resolve()
    destination = workspace / "prepared"
    count_path = (root / config.count_store).resolve()
    store = CountStore(count_path)
    store_manifest = store.verify(full=True)
    information = InformationSet.model_validate_json((root / config.information_set).read_text())
    features = tuple(
        FeatureKey.model_validate(row)
        for row in json.loads((root / config.feature_index).read_text())
    )
    if len(features) != store_manifest.features:
        raise ContractError("Feature contract does not match count store.")
    logical_feature_hash = sha256_bytes(
        canonical_json_bytes([feature.model_dump(mode="json") for feature in features])
    )
    if logical_feature_hash != store_manifest.feature_index_hash:
        raise ContractError("Ordered feature contract differs from the count store.")
    if config.model.gene_decoder_features not in {0, len(features)}:
        raise ContractError("Gene decoder width differs from the ordered feature contract.")
    allowed_fit_ids = np.asarray(information.fit_rows, dtype=np.int64)
    fit_ids = allowed_fit_ids
    if len(fit_ids) > config.representation_fit_max_rows:
        generator = np.random.default_rng(config.representation_seed)
        positions = generator.choice(
            len(fit_ids), size=config.representation_fit_max_rows, replace=False
        )
        fit_ids = fit_ids[np.sort(positions)]
    fit_matrix = _library_normalize(store.rows(fit_ids).matrix)
    fit_matrix, correction_audit = _fit_corrected_view(root, config, fit_ids, fit_matrix)
    components, scale, explained_energy_ratio = _fit_components(
        fit_matrix, config.model.state_dim, seed=config.representation_seed
    )
    # Encode only rows admitted by the frozen information set. This avoids
    # silently materializing unrelated cells and keeps large stores bounded.
    all_ids = np.unique(
        np.asarray(
            (*information.fit_rows, *information.query_rows, *information.protected_rows),
            dtype=np.int64,
        )
    )
    prepared_id_holder: dict[str, str] = {}

    def writer(temp: Path) -> None:
        (temp / "information_set.json").write_bytes(
            canonical_json_bytes(information.model_dump(mode="json")) + b"\n"
        )
        (temp / "fit-selection.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "allowed_fit_rows_hash": sha256_bytes(
                        np.asarray(allowed_fit_ids, dtype="<i8").tobytes()
                    ),
                    "selected_row_ids": fit_ids.tolist(),
                    "maximum_rows": config.representation_fit_max_rows,
                    "seed": config.representation_seed,
                    "decomposition_seed": config.representation_seed,
                    "explained_scaled_energy_ratio": explained_energy_ratio,
                }
            )
            + b"\n"
        )
        (temp / "feature_index.json").write_bytes(
            canonical_json_bytes([feature.model_dump(mode="json") for feature in features]) + b"\n"
        )
        (temp / "input_view.json").write_bytes(canonical_json_bytes(correction_audit) + b"\n")
        save_tensor_file(
            temp / "encoder.safetensors",
            {"components": torch.from_numpy(components), "scale": torch.from_numpy(scale)},
        )
        save_tensor_file(
            temp / "decoder.safetensors",
            {"components": torch.from_numpy(components), "scale": torch.from_numpy(scale)},
        )
        with h5py.File(temp / "latents.h5", "x") as handle:
            handle.create_dataset("row_ids", data=all_ids.astype(np.int64), compression="gzip")
            latent_output = handle.create_dataset(
                "z",
                shape=(len(all_ids), config.model.state_dim),
                dtype=np.float32,
                chunks=(
                    min(config.representation_encode_batch_size, len(all_ids)),
                    config.model.state_dim,
                ),
                compression="gzip",
                shuffle=True,
            )
            for start in range(0, len(all_ids), config.representation_encode_batch_size):
                end = min(start + config.representation_encode_batch_size, len(all_ids))
                batch_ids = all_ids[start:end]
                normalized = _library_normalize(store.rows(batch_ids).matrix)
                normalized = _apply_frozen_view(
                    root, config, batch_ids, normalized, correction_audit
                )
                encoded = normalized @ sparse.diags(1.0 / scale) @ components.T
                latent_output[start:end] = np.asarray(encoded, dtype=np.float32)
            handle.flush()
        final = destination
        count_ref = ArtifactRef(
            schema_id="credo.count_store",
            schema_version=1,
            sha256=store_manifest.content_sha256,
            size_bytes=count_path.stat().st_size,
            media_type="application/x-hdf5",
            relative_uri=count_path.relative_to(workspace).as_posix(),
        )
        payload = {
            "schema_version": 1,
            "prepared_id": "pending",
            "cache_generation_id": "pending",
            "count_store": count_ref.model_dump(mode="json"),
            "information_set": _ref_for_future(
                workspace,
                final / "information_set.json",
                temp / "information_set.json",
                schema_id="credo.information_set",
                media_type="application/json",
            ).model_dump(mode="json"),
            "fit_selection": _ref_for_future(
                workspace,
                final / "fit-selection.json",
                temp / "fit-selection.json",
                schema_id="credo.representation_fit_selection",
                media_type="application/json",
            ).model_dump(mode="json"),
            "input_view": _ref_for_future(
                workspace,
                final / "input_view.json",
                temp / "input_view.json",
                schema_id="credo.input_view",
                media_type="application/json",
            ).model_dump(mode="json"),
            "feature_index": _ref_for_future(
                workspace,
                final / "feature_index.json",
                temp / "feature_index.json",
                schema_id="credo.feature_index",
                media_type="application/json",
            ).model_dump(mode="json"),
            "encoder_state": _ref_for_future(
                workspace,
                final / "encoder.safetensors",
                temp / "encoder.safetensors",
                schema_id="credo.encoder",
                media_type="application/x-safetensors",
            ).model_dump(mode="json"),
            "decoder_state": _ref_for_future(
                workspace,
                final / "decoder.safetensors",
                temp / "decoder.safetensors",
                schema_id="credo.decoder",
                media_type="application/x-safetensors",
            ).model_dump(mode="json"),
            "latent_cache": _ref_for_future(
                workspace,
                final / "latents.h5",
                temp / "latents.h5",
                schema_id="credo.latent_cache",
                media_type="application/x-hdf5",
            ).model_dump(mode="json"),
            "fit_rows_hash": sha256_bytes(np.asarray(fit_ids, dtype="<i8").tobytes()),
            "validation_rows_hash": sha256_bytes(
                np.asarray(information.validation_rows, dtype="<i8").tobytes()
            ),
            "state_dim": config.model.state_dim,
        }
        representation_core = {
            key: value
            for key, value in payload.items()
            if key not in {"prepared_id", "cache_generation_id", "count_store", "latent_cache"}
        }
        payload["prepared_id"] = contract_id(representation_core)
        payload["cache_generation_id"] = contract_id(
            {
                "prepared_id": payload["prepared_id"],
                "count_store": payload["count_store"],
                "latent_cache": payload["latent_cache"],
            }
        )
        manifest = PreparedRepresentation.model_validate(payload)
        prepared_id_holder["value"] = manifest.prepared_id
        (temp / "prepared.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(destination, writer)
    return destination


def load_prepared_arrays(
    workspace: Path,
) -> tuple[PreparedRepresentation, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    prepared_root = workspace / "prepared"
    manifest = PreparedRepresentation.model_validate_json(
        (prepared_root / "prepared.json").read_text()
    )
    for reference in (
        manifest.encoder_state,
        manifest.decoder_state,
        manifest.latent_cache,
        manifest.input_view,
        manifest.information_set,
        manifest.fit_selection,
        manifest.feature_index,
        manifest.count_store,
    ):
        path = workspace / reference.relative_uri
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"Prepared artifact is absent or not regular: {path}.")
        if path.stat().st_size != reference.size_bytes or sha256_file(path) != reference.sha256:
            raise ContractError(f"Prepared artifact bytes do not match: {path}.")
    with h5py.File(prepared_root / "latents.h5", "r") as handle:
        row_ids = handle["row_ids"][:]
        latents = handle["z"][:]
    return manifest, row_ids, latents
