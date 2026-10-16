"""Fresh fold-native count observation model; no files or historical weights loaded.

Architecture follows the retained count-native R48 encoder/decoder. The new
fold/row-hash policies below are explicit new-study primitives, not historical
checkpoint equivalence. CPU first projection stays sparse; CUDA may densify
only an admitted minibatch. Molecular decoding uses one full-gene softmax
denominator even when output genes are evaluated in blocks.

Checkpoint selection, data ownership, row completeness and H100 qualification
belong to a future admitted owner. A donor label check is not source provenance.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable

import numpy as np
from scipy import sparse
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ObservationFold:
    fold_id: str
    training_donors: tuple[str, ...]
    feature_axis_sha256: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.fold_id):
            raise ValueError("Explicit safe fold identity required")
        if (type(self.training_donors) is not tuple or len(self.training_donors) not in (2, 3)
                or tuple(sorted(set(self.training_donors))) != self.training_donors
                or not set(self.training_donors).issubset({"D1", "D2", "D3", "D4"})):
            raise ValueError("Sorted two- or three-donor training subset required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.feature_axis_sha256):
            raise ValueError("Exact RNA feature-axis SHA-256 required")

    def admit_training(self, donors, conditions, rows):
        if len(donors) != rows or len(conditions) != rows:
            raise ValueError("One donor/condition identity per training row required")
        if not set(donors).issubset(self.training_donors):
            raise ValueError("Held-out donor in observation fitting or normalization")
        if not set(conditions).issubset({"Stim8hr", "Stim48hr"}):
            raise ValueError("Only Stim8hr/Stim48hr fitting observations admitted; Rest is excluded")


def _row_ids(row_ids, rows):
    if len(row_ids) != rows or any(not isinstance(item, str) or not item for item in row_ids):
        raise ValueError("Explicit nonempty, source-scoped row IDs required")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Duplicate row IDs within the supplied batch")
    return tuple(row_ids)


def technical_holdout(row_ids, *, fold: ObservationFold, seed=0, fraction=0.05):
    """Stable row-hash split, independent of outcomes and input iteration order.

    Fraction is expected, not an exact per-guide allocation. The owner must
    inventory actual support and guarantee that row IDs are globally scoped.
    """
    _row_ids(row_ids, len(row_ids))
    if type(seed) is not int or seed < 0 or not 0 < fraction < 1:
        raise ValueError("Nonnegative seed and interior technical holdout fraction required")
    threshold = int(fraction * (1 << 256))
    return np.asarray([int(hashlib.sha256(json.dumps(
        ["frontier-observation-holdout-v1", fold.fold_id, seed, row_id], separators=(",", ":")
    ).encode()).hexdigest(), 16) < threshold for row_id in row_ids], dtype=bool)


def validate_counts(counts, *, features, maximum_rows):
    if not sparse.issparse(counts) or counts.format != "csr":
        raise TypeError("Sparse canonical CSR counts required; dense cohort input is not admitted")
    if counts.ndim != 2 or counts.shape[1] != features or not 0 < counts.shape[0] <= maximum_rows:
        raise ValueError("Count feature axis or bounded minibatch size mismatch")
    if not counts.has_canonical_format or not counts.has_sorted_indices:
        raise ValueError("CSR columns must be sorted and duplicate-free")
    if (counts.dtype.kind not in "iu" or np.any(counts.data < 0)
            or (counts.nnz and int(counts.data.max()) > np.iinfo(np.int32).max)):
        raise ValueError("Nonnegative int32-range integer molecule counts required")
    return counts


def symmetric_umi_halves(counts, row_ids, *, seed=0, namespace="frontier-umi-v1", maximum_rows=2048):
    """Binomial(UMI, 1/2) plus exact complement; independent local per-row RNG.

    Reordering or minibatch partitioning cannot change a cell's halves. Each
    half's own RNA depth defines its encoder input; the other half is a target.
    """
    validate_counts(counts, features=counts.shape[1], maximum_rows=maximum_rows)
    ids = _row_ids(row_ids, counts.shape[0])
    if type(seed) is not int or seed < 0 or not isinstance(namespace, str) or not namespace:
        raise ValueError("Explicit thinning seed and namespace required")
    values = np.empty(counts.nnz, dtype=np.int32)
    for row, row_id in enumerate(ids):
        key = json.dumps([namespace, seed, row_id], separators=(",", ":")).encode()
        generator = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(hashlib.sha256(key).digest()[:16], "big")))
        start, stop = counts.indptr[row:row + 2]
        values[start:stop] = generator.binomial(counts.data[start:stop], 0.5)
    other = counts.data.astype(np.int64) - values
    first = sparse.csr_matrix((values, counts.indices.copy(), counts.indptr.copy()), shape=counts.shape)
    second = sparse.csr_matrix((other.astype(np.int32), counts.indices.copy(), counts.indptr.copy()), shape=counts.shape)
    first.eliminate_zeros()
    second.eliminate_zeros()
    return first, second


class SparseR48(nn.Module):
    """18,129 -> 512 -> 128 -> 48 and mirrored LayerNorm/GELU decoder.

    Smaller dimensions exist only to qualify interfaces on synthetic fixtures.
    Constructor initializes fresh weights and buffers; it never opens a path.
    """

    def __init__(self, fold: ObservationFold, *, features=18129, hidden_dims=(512, 128),
                 latent_dim=48, maximum_rows=2048, maximum_dense_bytes=1073741824, seed=0):
        super().__init__()
        if not isinstance(fold, ObservationFold):
            raise TypeError("Explicit fold identity required")
        if (any(type(value) is not int or value < 1 for value in
                (features, latent_dim, maximum_rows, maximum_dense_bytes, *hidden_dims))
                or len(hidden_dims) != 2 or type(seed) is not int or seed < 0):
            raise ValueError("Positive dimensions/budgets and nonnegative initialization seed required")
        self.fold, self.features, self.latent_dim = fold, features, latent_dim
        self.hidden_dims, self.maximum_rows = tuple(hidden_dims), maximum_rows
        self.maximum_dense_bytes = maximum_dense_bytes
        first, second = hidden_dims
        with torch.random.fork_rng(devices=[]):
            # Do not call torch.manual_seed here: it also seeds CUDA generators.
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            options = {"device": "cpu", "dtype": torch.float32}
            self.first = nn.Linear(features, first, **options)
            self.encoder = nn.Sequential(nn.LayerNorm(first, **options), nn.GELU(), nn.Linear(first, second, **options),
                                         nn.LayerNorm(second, **options), nn.GELU(), nn.Linear(second, latent_dim, **options))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, second, **options), nn.LayerNorm(second, **options), nn.GELU(),
                                         nn.Linear(second, first, **options), nn.LayerNorm(first, **options), nn.GELU(),
                                         nn.Linear(first, features, **options))
        self.register_buffer("center", torch.zeros(latent_dim, device="cpu", dtype=torch.float32))
        self.register_buffer("scale", torch.ones(latent_dim, device="cpu", dtype=torch.float32))
        self.register_buffer("scaling_ready", torch.tensor(False, device="cpu"))

    @property
    def device(self):
        return self.first.weight.device

    def _budget(self, rows, *, decoder=False):
        if not 0 < rows <= self.maximum_rows:
            raise ValueError("Minibatch row limit exceeded")
        # Conservative activation estimate; not a claim about peak allocator use.
        factor = 8 if decoder else 1
        estimate = rows * (factor * self.features + 16 * sum(self.hidden_dims) + 8 * self.latent_dim) * 4
        if estimate > self.maximum_dense_bytes:
            raise ValueError("Admitted minibatch dense-working budget exceeded")

    def raw_encode(self, counts):
        counts = validate_counts(counts, features=self.features, maximum_rows=self.maximum_rows)
        self._budget(counts.shape[0])
        if self.first.weight.dtype != torch.float32:
            raise ValueError("Observation model's qualified path requires FP32 weights")
        rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        depth = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
        normalized = np.log1p(10000 * counts.data.astype(np.float64) / np.maximum(depth[rows], 1)).astype(np.float32)
        indices = torch.as_tensor(np.stack((rows, counts.indices)), dtype=torch.int64, device=self.device)
        values = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        matrix = torch.sparse_coo_tensor(indices, values, counts.shape, device=self.device, is_coalesced=True)
        if self.device.type == "cpu":
            projected = torch.sparse.mm(matrix, self.first.weight.T)
        elif self.device.type == "cuda":
            projected = torch.mm(matrix.to_dense(), self.first.weight.T)
        else:
            raise ValueError("Only explicit CPU or CUDA observation execution supported")
        return self.encoder(projected + self.first.bias)

    def encode(self, counts):
        self._check_scaling()
        return (self.raw_encode(counts) - self.center) / self.scale

    def _check_scaling(self):
        if not bool(self.scaling_ready) or not bool(torch.isfinite(self.center).all()) or not bool(torch.isfinite(self.scale).all()) or bool((self.scale <= 0).any()):
            raise ValueError("Fitting-only fold center and positive scale are not finalized")

    def _raw_state(self, state, *, standardized):
        if (state.ndim != 2 or state.shape[1] != self.latent_dim or state.device != self.device
                or state.dtype != torch.float32 or not bool(torch.isfinite(state).all())):
            raise ValueError("Finite aligned FP32 latent states required")
        if standardized:
            self._check_scaling()
            return state * self.scale + self.center
        return state

    def probabilities(self, state, *, standardized=True):
        self._budget(len(state), decoder=True)
        return torch.softmax(self.decoder(self._raw_state(state, standardized=standardized)), dim=-1)

    def freeze(self):
        """Freeze parameters, not differentiation through the decoder's inputs."""
        self.eval().requires_grad_(False)
        return self

    def symmetric_count_loss(self, counts, *, row_ids, donors, conditions, thinning_seed=0):
        counts = validate_counts(counts, features=self.features, maximum_rows=self.maximum_rows)
        self.fold.admit_training(donors, conditions, counts.shape[0])
        self._budget(counts.shape[0], decoder=True)
        first, second = symmetric_umi_halves(counts, row_ids, seed=thinning_seed,
            namespace=f"frontier-umi-v1:{self.fold.fold_id}", maximum_rows=self.maximum_rows)
        summed, scored = self.first.weight.sum() * 0, 0
        for inputs, targets in ((first, second), (second, first)):
            log_probability = torch.log_softmax(self.decoder(self.raw_encode(inputs)), dim=-1)
            rows = np.repeat(np.arange(targets.shape[0]), np.diff(targets.indptr))
            values = torch.as_tensor(targets.data.astype(np.float32), device=self.device)
            row = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
            columns = torch.as_tensor(targets.indices, dtype=torch.int64, device=self.device)
            totals = torch.zeros(len(targets.indptr) - 1, device=self.device).index_add(0, row, values)
            nll = torch.zeros_like(totals).index_add(0, row, -values * log_probability[row, columns])
            positive = totals > 0
            summed = summed + (nll[positive] / totals[positive]).sum()
            scored += int(positive.sum())
        return {"loss": summed / scored if scored else None, "scored_cell_directions": scored,
                "zero_scored_directions": 2 * counts.shape[0] - scored,
                "normalization": "equal_positive_target_cell_direction", "halves": (first, second)}

    def fit_scaling(self, batches: Iterable, *, scale_floor_fraction=0.01):
        """Stream raw fitting encodings; zero-RNA rows do not define geometry.

        Each item is (CSR minibatch, donor labels, condition labels). Caller must
        establish complete authorized fitting coverage and checkpoint identity.
        """
        if not 0 < scale_floor_fraction <= 1 or bool(self.scaling_ready):
            raise ValueError("Scaling is a once-only fitting publication with a positive floor fraction")
        count = 0
        mean = torch.zeros(self.latent_dim, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        with torch.no_grad():
            for counts, donors, conditions in batches:
                validate_counts(counts, features=self.features, maximum_rows=self.maximum_rows)
                self.fold.admit_training(donors, conditions, counts.shape[0])
                positive = np.asarray(counts.sum(axis=1)).ravel() > 0
                z = self.raw_encode(counts).detach().cpu().double()[positive]
                if not len(z):
                    continue
                if not bool(torch.isfinite(z).all()):
                    raise ValueError("Nonfinite fitting latent state")
                batch_mean = z.mean(0)
                difference, total = batch_mean - mean, count + len(z)
                m2 += (z - batch_mean).square().sum(0) + difference.square() * count * len(z) / total
                mean += difference * len(z) / total
                count = total
        if count < 2:
            raise ValueError("At least two positive-RNA fitting rows required for scaling")
        std = (m2.clamp_min(0) / count).sqrt()
        positive_std = std[std > 0].numpy()
        floor = max(float(np.median(positive_std)) * scale_floor_fraction if len(positive_std) else 0, 1e-6)
        with torch.no_grad():
            self.center.copy_(mean.to(self.device, torch.float32))
            self.scale.copy_(std.clamp_min(floor).to(self.device, torch.float32))
            self.scaling_ready.fill_(True)
        return {"positive_RNA_cells": count, "population_variance_ddof": 0,
                "scale_floor": floor, "fold_id": self.fold.fold_id,
                "scale_floor_rule": "max(1e-6,fraction*median_positive_fitting_std)"}

    def weighted_composition(self, state, weights, *, gene_block=1024, row_block=None,
                             standardized=True, recompute_backward=True):
        """Stream weighted full-gene compositions with a global denominator.

        There are two gene-block passes per row block: global log-sum-exp, then
        normalized weighted sums. With gradients enabled, checkpointing
        recomputes block operations in backward to avoid retaining every
        particle-by-gene activation. Without recomputation, blocks bound each
        forward allocation but autograd may retain O(particles*genes) total
        intermediates. Actual memory/throughput must be profiled on H100.
        Parameters must be frozen, while state and weight gradients are retained.
        """
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("Molecular observation parameters must be frozen")
        row_block = min(self.maximum_rows, 2048) if row_block is None else row_block
        if (type(gene_block) is not int or gene_block < 1 or type(row_block) is not int
                or not 0 < row_block <= self.maximum_rows or not len(state)):
            raise ValueError("Positive bounded gene/row blocks required")
        if (weights.shape != (len(state),) or weights.device != self.device
                or not weights.is_floating_point() or not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any()) or not bool(weights.sum() > 0)):
            raise ValueError("Aligned finite nonnegative weights with positive total required")
        raw = self._raw_state(state, standardized=standardized)
        weights = weights.double() / weights.double().sum()
        output = self.decoder[-1]
        ranges = [(start, min(start + gene_block, self.features)) for start in range(0, self.features, gene_block)]
        totals = [torch.zeros(stop - start, dtype=torch.float64, device=self.device) for start, stop in ranges]

        def reduce_lse(hidden, weight, bias):
            return torch.logsumexp(F.linear(hidden, weight, bias).double(), dim=-1)

        def reduce_probability(hidden, weight, bias, log_denominator, importance):
            return (torch.exp(F.linear(hidden, weight, bias).double() - log_denominator[:, None])
                    * importance[:, None]).sum(0)

        def apply(function, *arguments):
            if recompute_backward and torch.is_grad_enabled() and any(value.requires_grad for value in arguments):
                return checkpoint(function, *arguments, use_reentrant=False)
            return function(*arguments)

        for row_start in range(0, len(raw), row_block):
            row_stop = min(row_start + row_block, len(raw))
            self._budget(row_stop - row_start)
            hidden = self.decoder[:-1](raw[row_start:row_stop])
            log_denominator = torch.full((len(hidden),), -torch.inf, dtype=torch.float64, device=self.device)
            for start, stop in ranges:
                log_denominator = torch.logaddexp(log_denominator,
                    apply(reduce_lse, hidden, output.weight[start:stop], output.bias[start:stop]))
            if not bool(torch.isfinite(log_denominator).all()):
                raise ValueError("Nonfinite full-gene decoder denominator")
            for index, (start, stop) in enumerate(ranges):
                totals[index] = totals[index] + apply(reduce_probability, hidden, output.weight[start:stop],
                    output.bias[start:stop], log_denominator, weights[row_start:row_stop])
        return torch.cat(totals)
