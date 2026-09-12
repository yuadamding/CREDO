"""Score-matched E2 composition primitives and small label-free architectures.

This module does not fit a cohort, choose a split, sample observations, or write
an artifact. The caller binds source/count/feature identities, donor partitions,
the fixed observation model and the complete training/evaluation protocol.
M0 is analytic; M1/M2 default to rank 8 and hidden width 128, without PCA.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from scipy import sparse
from torch import Tensor, nn

from ..errors import ContractError
from .state_information import validate_count_csr


def _labels(values: Sequence[str], size: int, name: str) -> tuple[str, ...]:
    if not isinstance(values, (Sequence, np.ndarray)) or isinstance(values, (str, bytes)):
        raise ContractError(f"{name} requires an ordered sequence of string labels.")
    result = tuple(values)
    if len(result) != size or any(not isinstance(x, str) or not x.strip() for x in result):
        raise ContractError(f"{name} requires aligned nonempty string labels.")
    return result


def _real(values: np.ndarray[Any, Any], name: str) -> np.ndarray[Any, Any]:
    array = np.asarray(values)
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ContractError(f"{name} requires real, non-boolean numbers.")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError(f"{name} must be finite.")
    return result


def _normalized_weights(values: np.ndarray[Any, Any], size: int) -> np.ndarray[Any, Any]:
    weights = _real(values, "weights")
    if weights.shape != (size,) or np.any(weights < 0) or not np.any(weights > 0):
        raise ContractError("Weights must align, be nonnegative and have positive total weight.")
    weights = weights / weights.max()
    return weights / weights.sum()


def _seed_ids(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(values, (Sequence, np.ndarray)) or isinstance(values, (str, bytes)):
        raise ContractError(f"{name} requires an ordered seed sequence.")
    result = tuple(values)
    if any(
        isinstance(x, (bool, np.bool_))
        or not isinstance(x, (int, np.integer))
        or not 0 <= x < 2**64
        for x in result
    ) or len(result) != len(set(result)):
        raise ContractError(f"{name} requires unique unsigned 64-bit integer seeds.")
    return tuple(map(int, result))


def strict_seed_aggregate(
    seed_ids: Sequence[int], values: Sequence[float | None], *, expected_seeds: Sequence[int]
) -> dict[str, Any]:
    """Require every prespecified seed to be present and finite for any summary.

    Missing, None, NaN and infinite scores make mean/std/min/max all None. The
    sample standard deviation is also undefined for a complete singleton. It
    describes technical seed variability, not independent donor replication.
    Unexpected/duplicate seeds and malformed values are errors, not omissions.
    """
    expected = _seed_ids(expected_seeds, "expected_seeds")
    present = _seed_ids(seed_ids, "seed_ids")
    if not expected or not set(present).issubset(expected):
        raise ContractError("Expected seeds must be nonempty and cover every present seed.")
    if not isinstance(values, (Sequence, np.ndarray)) or isinstance(values, (str, bytes)):
        raise ContractError("Seed values require an ordered sequence.")
    if len(values) != len(present):
        raise ContractError("Seed values and identities must align.")
    scores: dict[int, float | None] = {}
    for seed, value in zip(present, values, strict=True):
        if value is None:
            scores[seed] = None
        elif isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ContractError("Seed scores must be real numbers or explicitly undefined.")
        else:
            scores[seed] = float(value) if np.isfinite(value) else None
    missing = [seed for seed in expected if seed not in scores]
    undefined = [seed for seed in expected if seed in scores and scores[seed] is None]
    complete = not missing and not undefined
    result = {
        "expected_count": len(expected),
        "present_count": len(present),
        "defined_count": sum(value is not None for value in scores.values()),
        "expected_seeds": list(expected),
        "missing_seeds": missing,
        "undefined_seeds": undefined,
        "all_defined": complete,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
    }
    if complete:
        array = np.asarray([scores[seed] for seed in expected], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            average = float(array.mean())
            deviation = float(array.std(ddof=1)) if len(array) > 1 else None
        if not np.isfinite(average) or (deviation is not None and not np.isfinite(deviation)):
            raise ContractError("Seed summary overflowed finite float64 arithmetic.")
        result.update(mean=average, std=deviation, min=float(array.min()), max=float(array.max()))
    return result


def hierarchical_cell_weights(
    donor_ids: Sequence[str],
    checkpoint_ids: Sequence[str],
    target_ids: Sequence[str],
    guide_ids: Sequence[str],
    *,
    is_control: np.ndarray[Any, Any] | None = None,
    role: Literal["targeting", "controls"] = "targeting",
) -> np.ndarray[Any, Any]:
    """Weights equal donor, checkpoint, target, guide, then cell; sum one.

    Only the chosen role receives nonzero weights. Controls are a separate
    donor/checkpoint/guide/cell hierarchy, never silently mixed into targeting
    training. This function does not infer absent experimental units; the
    caller must validate its complete expected donor/time/target/guide scope.
    IDs determine weights only and never enter the composition architecture.
    """
    if not isinstance(donor_ids, (Sequence, np.ndarray)) or isinstance(donor_ids, (str, bytes)):
        raise ContractError("donor_ids requires an ordered label sequence.")
    n = len(donor_ids)
    donors = _labels(donor_ids, n, "donor_ids")
    times = _labels(checkpoint_ids, n, "checkpoint_ids")
    targets = _labels(target_ids, n, "target_ids")
    guides = _labels(guide_ids, n, "guide_ids")
    controls = np.zeros(n, dtype=bool) if is_control is None else np.asarray(is_control)
    if (
        controls.dtype != np.dtype(bool)
        or controls.shape != (n,)
        or role not in {"targeting", "controls"}
    ):
        raise ContractError("Control roles require aligned boolean flags and a declared role.")
    selected = np.flatnonzero(controls if role == "controls" else ~controls)
    if not len(selected):
        raise ContractError("The selected training/scoring role has no cells.")
    guide_targets: dict[str, set[str]] = defaultdict(set)
    for i in selected:
        guide_targets[guides[i]].add(targets[i])
    if any(len(values) != 1 for values in guide_targets.values()):
        raise ContractError("A guide has contradictory target identities.")
    times_by_donor: dict[str, set[str]] = defaultdict(set)
    targets_by_time: dict[tuple[str, str], set[str]] = defaultdict(set)
    guides_by_target: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    keys = {}
    for i in selected:
        target = targets[i] if role == "targeting" else "__control_reference__"
        key = (donors[i], times[i], target, guides[i])
        keys[i] = key
        times_by_donor[key[0]].add(key[1])
        targets_by_time[key[:2]].add(key[2])
        guides_by_target[key[:3]].add(key[3])
    cells_by_guide = Counter(keys.values())
    weights = np.zeros(n, dtype=np.float64)
    for i, key in keys.items():
        weights[i] = 1 / (
            len(times_by_donor)
            * len(times_by_donor[key[0]])
            * len(targets_by_time[key[:2]])
            * len(guides_by_target[key[:3]])
            * cells_by_guide[key]
        )
    return weights / weights.sum()


def score_matched_composition(
    counts: sparse.csr_matrix, weights: np.ndarray[Any, Any], *, uniform_mixture: float = 1e-8
) -> np.ndarray[Any, Any]:
    """Weighted mean cell RNA compositions, with fixed uniform full-gene support.

    This is not UMI-pooled gene frequency. Positive-weight zero-depth rows make
    the estimator undefined and are rejected; zero-weight rows never affect it.
    The caller supplies only permitted fitting counts and the intended objective
    weights. Full RNA feature order/denominator must be preserved upstream.
    """
    matrix = validate_count_csr(counts)
    weight = _normalized_weights(weights, matrix.shape[0])
    if (
        isinstance(uniform_mixture, (bool, np.bool_))
        or not isinstance(uniform_mixture, (int, float, np.integer, np.floating))
        or not np.isfinite(uniform_mixture)
        or not 0 < uniform_mixture < 1
    ):
        raise ContractError(
            "Uniform support mixture must be finite and strictly between zero and one."
        )
    totals = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    if np.any((weight > 0) & (totals <= 0)):
        raise ContractError("A positive-weight cell has zero RNA depth.")
    inverse_depth_weight = np.divide(weight, totals, out=np.zeros_like(weight), where=totals > 0)
    composition = np.asarray(matrix.T @ inverse_depth_weight, dtype=np.float64).reshape(-1)
    composition /= composition.sum()
    probability = (1 - uniform_mixture) * composition + uniform_mixture / matrix.shape[1]
    return probability / probability.sum()


def _composition(values: np.ndarray[Any, Any], features: int | None = None) -> np.ndarray[Any, Any]:
    probability = _real(values, "composition")
    if (
        probability.ndim != 1
        or not len(probability)
        or (features is not None and len(probability) != features)
        or np.any(probability <= 0)
        or not np.isclose(probability.sum(), 1, atol=1e-10, rtol=0)
    ):
        raise ContractError("Composition must be strictly positive, aligned and normalized.")
    return probability


def smoothed_reference_delta_components(
    first_counts: sparse.csr_matrix,
    heldback_counts: sparse.csr_matrix,
    prior: np.ndarray[Any, Any],
    *,
    prior_total_umi: float = 64.0,
) -> dict[str, np.ndarray[Any, Any]]:
    """Exact missing-A/detected-A contributions to CE(smoothed A)-CE(prior).

    Outputs are per B UMI. Missing-A contribution is its B mass fraction times
    log1p(A_depth/prior_total_umi); detected-A contribution may have either sign.
    Zero-B rows retain NaN scores/fractions, and raw depth/count coverage remains
    explicit. The historical smoothing constant is not fitted by this function.
    """
    first = validate_count_csr(first_counts)
    heldback = validate_count_csr(heldback_counts)
    if first.shape != heldback.shape:
        raise ContractError("A and B count matrices must share exact row/feature dimensions.")
    base = _composition(prior, first.shape[1])
    if (
        isinstance(prior_total_umi, (bool, np.bool_))
        or not isinstance(prior_total_umi, (int, float, np.integer, np.floating))
        or not np.isfinite(prior_total_umi)
        or prior_total_umi <= 0
    ):
        raise ContractError("Prior UMI exposure must be finite and positive.")
    n = first.shape[0]
    result = {
        name: np.full(n, np.nan)
        for name in (
            "missing_a_b_fraction",
            "missing_a_contribution",
            "detected_a_contribution",
            "total_delta",
            "direct_delta",
            "decomposition_residual",
        )
    }
    result["a_depth"] = np.asarray(first.sum(axis=1), dtype=np.int64).reshape(-1)
    result["b_depth"] = np.asarray(heldback.sum(axis=1), dtype=np.int64).reshape(-1)
    result["missing_a_b_count"] = np.zeros(n, dtype=np.int64)
    result["detected_a_b_count"] = np.zeros(n, dtype=np.int64)
    for row in range(n):
        a_left, a_right = first.indptr[row : row + 2]
        b_left, b_right = heldback.indptr[row : row + 2]
        a_genes = first.indices[a_left:a_right]
        b_genes = heldback.indices[b_left:b_right]
        b_values = heldback.data[b_left:b_right].astype(np.float64)
        if result["b_depth"][row] == 0:
            continue
        positions = np.searchsorted(a_genes, b_genes)
        exists = positions < len(a_genes)
        exists[exists] &= a_genes[positions[exists]] == b_genes[exists]
        a_values = np.zeros(len(b_genes), dtype=np.float64)
        a_values[exists] = first.data[a_left:a_right][positions[exists]]
        missing = a_values == 0
        missing_count = int(b_values[missing].sum())
        result["missing_a_b_count"][row] = missing_count
        result["detected_a_b_count"][row] = int(b_values[~missing].sum())
        fraction = missing_count / result["b_depth"][row]
        missing_contribution = fraction * np.log1p(result["a_depth"][row] / prior_total_umi)
        log_ratio = (
            np.log(base[b_genes])
            - np.log(a_values + prior_total_umi * base[b_genes])
            + np.log(result["a_depth"][row] + prior_total_umi)
        )
        detected_contribution = (
            float(b_values[~missing] @ log_ratio[~missing]) / result["b_depth"][row]
        )
        direct = float(b_values @ log_ratio) / result["b_depth"][row]
        result["missing_a_b_fraction"][row] = fraction
        result["missing_a_contribution"][row] = missing_contribution
        result["detected_a_contribution"][row] = detected_contribution
        result["total_delta"][row] = missing_contribution + detected_contribution
        result["direct_delta"][row] = direct
        result["decomposition_residual"][row] = result["total_delta"][row] - direct
    return result


class CountCompositionModel(nn.Module):
    """Small count-composition family with no donor/target/guide/time inputs.

    Input is the entire ordered RNA gene vector of raw nonnegative counts.
    The encoder sees log1p(1e4*A/sum(A)); zero-A rows return current intercept
    predictions. Output logits are base log-frequency + common intercept offset
    + cell-dependent residual. M0 is an analytic, fixed intercept; M1/M2 can
    refit the common offset without weight decay. Decoder final biases are absent
    and final weights start at zero, exactly nesting the base intercept at init.
    This class has no depth/dispersion likelihood parameters: the caller applies
    the same frozen conditional-multinomial observation treatment to every model.
    """

    base_log_composition: Tensor
    force_latent_ablation: Tensor
    encoder: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        base_composition: np.ndarray[Any, Any],
        *,
        family: Literal["M0", "M1", "M2"] = "M0",
        latent_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        base = _composition(base_composition)
        if family not in {"M0", "M1", "M2"}:
            raise ContractError("Unknown count-composition family.")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in (latent_dim, hidden_dim)
        ):
            raise ContractError("Architecture dimensions must be positive integers.")
        self.family = family
        self.features = len(base)
        self.latent_dim = 0 if family == "M0" else int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer(
            "base_log_composition", torch.as_tensor(np.log(base), dtype=torch.float32)
        )
        self.register_buffer("force_latent_ablation", torch.tensor(False))
        self.intercept_offset = nn.Parameter(
            torch.zeros(self.features), requires_grad=family != "M0"
        )
        if family == "M0":
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
        elif family == "M1":
            self.encoder = nn.Linear(self.features, self.latent_dim, bias=False)
            self.decoder = nn.Linear(self.latent_dim, self.features, bias=False)
            nn.init.zeros_(self.decoder.weight)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.features, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.features, bias=False),
            )
            nn.init.zeros_(cast(nn.Linear, self.decoder[-1]).weight)

    def _count_input(self, counts: Tensor) -> Tensor:
        if (
            not isinstance(counts, Tensor)
            or counts.layout != torch.strided
            or counts.ndim != 2
            or counts.shape[1] != self.features
            or counts.shape[0] == 0
            or counts.dtype == torch.bool
            or counts.is_complex()
        ):
            raise ContractError("Count model requires an aligned dense raw-count tensor.")
        if (
            not bool(torch.isfinite(counts).all())
            or bool(torch.any(counts < 0))
            or not bool(torch.all(counts == torch.floor(counts)))
        ):
            raise ContractError(
                "Count encoder requires finite nonnegative integer-valued raw counts."
            )
        # A tensor-dtype comparison rounds int32 max up to 2**31 in float32.
        # Compare the observed maximum as a Python scalar against the exact cap.
        if counts.max().item() > np.iinfo(np.int32).max:
            raise ContractError("Count encoder input exceeds the int32 value contract.")
        values = counts.to(
            device=self.base_log_composition.device, dtype=self.base_log_composition.dtype
        )
        if not bool(torch.isfinite(values.sum(dim=1)).all()):
            raise ContractError("Count encoder RNA-depth sum is nonfinite.")
        return values

    def _encode_values(self, values: Tensor) -> Tensor:
        if self.family == "M0":
            return values.new_zeros((len(values), 0))
        totals = values.sum(dim=1, keepdim=True)
        normalized = torch.log1p(1e4 * (values / totals.clamp_min(1)))
        return self.encoder(normalized)

    def encode(self, counts: Tensor) -> Tensor:
        """Encode only the supplied observed count snapshot, never metadata IDs."""
        return self._encode_values(self._count_input(counts))

    def logits(self, counts: Tensor, *, ablate_latent: bool = False) -> Tensor:
        """Remove the entire residual for ablation, preserving fitted intercept."""
        if not isinstance(ablate_latent, bool):
            raise ContractError("ablate_latent must be boolean.")
        values = self._count_input(counts)
        baseline = self.base_log_composition + self.intercept_offset
        logits = baseline.unsqueeze(0).expand(len(values), -1)
        if self.family != "M0" and not ablate_latent and not bool(self.force_latent_ablation):
            residual = self.decoder(self._encode_values(values))
            logits = logits + residual * (values.sum(dim=1, keepdim=True) > 0)
        return logits

    def forward(self, counts: Tensor, *, ablate_latent: bool = False) -> Tensor:
        return torch.softmax(self.logits(counts, ablate_latent=ablate_latent), dim=-1)

    def intercept_probabilities(self) -> Tensor:
        return torch.softmax(self.base_log_composition + self.intercept_offset, dim=-1)

    def optimizer_parameter_groups(self, weight_decay: float = 1e-4) -> list[dict[str, Any]]:
        """AdamW-style groups; the common intercept is never weight-decayed."""
        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, (int, float))
            or not np.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ContractError("Weight decay must be finite and nonnegative.")
        regularized = [
            parameter
            for name, parameter in self.named_parameters()
            if name != "intercept_offset" and parameter.requires_grad
        ]
        result = []
        if regularized:
            result.append({"params": regularized, "weight_decay": float(weight_decay)})
        if self.intercept_offset.requires_grad:
            result.append({"params": [self.intercept_offset], "weight_decay": 0.0})
        return result

    def frozen_latent_ablation(self) -> CountCompositionModel:
        """Detached frozen copy, retaining fitted intercept and all saved buffers.

        External observation sampling must use the identical model-independent
        family, depth and seed contract. This method cannot silently replace it.
        """
        model = copy.deepcopy(self)
        model.force_latent_ablation.fill_(True)
        model.requires_grad_(False)
        model.eval()
        return model

    def configuration(self) -> dict[str, Any]:
        """Pure metadata; enclosing persisted model/receipt must hash its bytes."""
        return {
            "schema_version": 1,
            "family": self.family,
            "features": self.features,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "encoder_transform": "log1p(10000*observed_RNA_counts/observed_RNA_total)",
            "composition": "softmax(base_log_frequency+common_intercept_offset+state_residual)",
            "identity_inputs": [],
            "observation_parameters": [],
            "analytic_intercept_only": self.family == "M0",
            "force_latent_ablation": bool(self.force_latent_ablation),
        }
