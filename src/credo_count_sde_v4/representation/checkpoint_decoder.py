"""Leakage-safe checkpoint-conditioned multinomial decoder for G04.

The observation model is

    p_i = softmax(b_checkpoint(i) + W z_i).

At dimension zero, ``W`` has no columns and the model is exactly the
checkpoint-conditioned global-frequency null.  Guide, target, and held-out
donor labels are intentionally absent from every fitting and prediction API.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import sparse

from ..canonical import contract_id
from ..contracts import CheckpointMultinomialDecoderContract
from ..errors import ContractError


def _fit_row_hash(row_ids: np.ndarray[Any, Any]) -> str:
    values = np.asarray(row_ids, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class CheckpointMultinomialDecoder:
    """One immutable checkpoint-intercept decoder state."""

    contract: CheckpointMultinomialDecoderContract
    checkpoint_intercepts: np.ndarray[Any, Any]
    latent_weights: np.ndarray[Any, Any]

    def __post_init__(self) -> None:
        intercepts = np.asarray(self.checkpoint_intercepts, dtype=np.float64)
        weights = np.asarray(self.latent_weights, dtype=np.float64)
        expected_intercepts = (len(self.contract.checkpoints), self.contract.features)
        expected_weights = (self.contract.features, self.contract.latent_dimension)
        if intercepts.shape != expected_intercepts or weights.shape != expected_weights:
            raise ContractError("Checkpoint decoder arrays disagree with the frozen contract.")
        if not np.isfinite(intercepts).all() or not np.isfinite(weights).all():
            raise ContractError("Checkpoint decoder arrays must be finite.")
        object.__setattr__(self, "checkpoint_intercepts", intercepts.copy())
        object.__setattr__(self, "latent_weights", weights.copy())

    def state_dict(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return the complete numerical state without target/guide parameters."""

        return {
            "checkpoint_intercepts": self.checkpoint_intercepts.copy(),
            "latent_weights": self.latent_weights.copy(),
        }

    def probabilities(
        self, z: np.ndarray[Any, Any], checkpoints: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Evaluate probabilities in input order for declared checkpoints."""

        latent = np.asarray(z, dtype=np.float64)
        labels = np.asarray(checkpoints, dtype=str)
        if latent.ndim != 2 or latent.shape != (
            len(labels),
            self.contract.latent_dimension,
        ):
            raise ContractError("Latent rows or dimensions disagree with decoder inputs.")
        checkpoint_index = {value: index for index, value in enumerate(self.contract.checkpoints)}
        try:
            indices = np.asarray([checkpoint_index[value] for value in labels], dtype=np.int64)
        except KeyError as error:
            raise ContractError(f"Unknown decoder checkpoint: {error.args[0]}.") from error
        logits = self.checkpoint_intercepts[indices] + latent @ self.latent_weights.T
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ContractError("Checkpoint decoder produced invalid probabilities.")
        return cast(np.ndarray[Any, Any], probabilities)


def fit_checkpoint_frequency_null(
    training_counts: sparse.spmatrix,
    *,
    training_checkpoints: np.ndarray[Any, Any],
    training_row_ids: np.ndarray[Any, Any],
    checkpoint_order: tuple[str, ...],
    physical_time_hours: tuple[float, ...],
    pseudocount: float = 0.5,
) -> CheckpointMultinomialDecoder:
    """Fit the dimension-zero null from permitted outer-training rows only."""

    matrix = sparse.csr_matrix(training_counts)
    labels = np.asarray(training_checkpoints, dtype=str)
    row_ids = np.asarray(training_row_ids, dtype=np.int64)
    if matrix.shape[0] != len(labels) or matrix.shape[0] != len(row_ids):
        raise ContractError("Training counts, checkpoints, and row IDs must align.")
    if matrix.shape[1] <= 0 or matrix.shape[0] <= 0:
        raise ContractError("Checkpoint null requires a nonempty count matrix.")
    if len(np.unique(row_ids)) != len(row_ids):
        raise ContractError("Checkpoint null training row IDs must be unique.")
    if not np.issubdtype(matrix.dtype, np.number) or np.any(matrix.data < 0):
        raise ContractError("Checkpoint null requires nonnegative counts.")
    if not np.isfinite(matrix.data).all() or not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ContractError("Checkpoint null counts and pseudocount must be finite and valid.")
    checkpoints = tuple(checkpoint_order)
    if (
        not checkpoints
        or len(checkpoints) != len(set(checkpoints))
        or set(labels) != set(checkpoints)
    ):
        raise ContractError("Checkpoint order must be unique and cover training labels exactly.")
    if len(physical_time_hours) != len(checkpoints):
        raise ContractError("Checkpoint order and physical times must align.")
    intercepts = np.empty((len(checkpoints), matrix.shape[1]), dtype=np.float64)
    for index, checkpoint in enumerate(checkpoints):
        positions = np.where(labels == checkpoint)[0]
        if not len(positions):
            raise ContractError(f"Checkpoint {checkpoint} has no permitted training rows.")
        totals = np.asarray(matrix[positions].sum(axis=0), dtype=np.float64).reshape(-1)
        totals += pseudocount
        probabilities = totals / totals.sum()
        intercepts[index] = np.log(probabilities)
    payload = {
        "schema_version": 2,
        "decoder_contract_id": "pending",
        "equation": "softmax(checkpoint_intercept + latent_weights @ z)",
        "intercept_axis": "checkpoint",
        "dimension_zero_null": "checkpoint_global_frequency",
        "checkpoints": checkpoints,
        "physical_time_hours": physical_time_hours,
        "checkpoint_order_hash": contract_id(
            {"checkpoints": checkpoints, "physical_time_hours": physical_time_hours}
        ),
        "features": matrix.shape[1],
        "latent_dimension": 0,
        "pseudocount": pseudocount,
        "fit_row_ids_hash": _fit_row_hash(row_ids),
        "heldout_donor_outcomes_used": False,
        "guide_parameters": False,
        "target_parameters": False,
        "heldout_donor_parameters": False,
    }
    payload["decoder_contract_id"] = contract_id(payload, id_field="decoder_contract_id")
    contract = CheckpointMultinomialDecoderContract.model_validate(payload)
    return CheckpointMultinomialDecoder(
        contract=contract,
        checkpoint_intercepts=intercepts,
        latent_weights=np.empty((matrix.shape[1], 0), dtype=np.float64),
    )
