"""Negative-binomial static/checkpoint-conditional biological-program head."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..errors import ContractError


@dataclass(frozen=True)
class ProgramBatch:
    """Aligned dense minibatch inputs for the count-linked head."""

    counts: Tensor
    library_size: Tensor
    state: Tensor
    sample_index: Tensor
    checkpoint_index: Tensor
    target_index: Tensor
    guide_index: Tensor

    def validate(self, *, genes: int, state_dimension: int) -> None:
        rows = self.counts.shape[0]
        if self.counts.ndim != 2 or self.counts.shape[1] != genes or rows == 0:
            raise ContractError("Program counts have invalid rows or gene dimension.")
        if self.state.shape != (rows, state_dimension):
            raise ContractError("Program state covariates do not align with counts.")
        for name in (
            "library_size",
            "sample_index",
            "checkpoint_index",
            "target_index",
            "guide_index",
        ):
            if getattr(self, name).shape != (rows,):
                raise ContractError(f"Program batch {name} does not align with counts.")
        if not torch.isfinite(self.counts).all() or torch.any(self.counts < 0):
            raise ContractError("Program counts must be finite and nonnegative.")
        if not torch.all(self.counts == torch.floor(self.counts)):
            raise ContractError("Program likelihood requires raw integer counts.")
        if not torch.isfinite(self.library_size).all() or torch.any(self.library_size <= 0):
            raise ContractError("Library sizes must be finite and positive.")
        observed = self.counts.sum(dim=1)
        if not torch.allclose(observed, self.library_size.to(observed), atol=0, rtol=0):
            raise ContractError("Library-size offset must equal each observed raw-count total.")

    def subset(self, indices: Tensor) -> ProgramBatch:
        return ProgramBatch(
            counts=self.counts[indices],
            library_size=self.library_size[indices],
            state=self.state[indices],
            sample_index=self.sample_index[indices],
            checkpoint_index=self.checkpoint_index[indices],
            target_index=self.target_index[indices],
            guide_index=self.guide_index[indices],
        )

    def to(self, device: torch.device | str) -> ProgramBatch:
        return ProgramBatch(**{name: value.to(device) for name, value in vars(self).items()})


def negative_binomial_log_prob(counts: Tensor, mean: Tensor, dispersion: Tensor) -> Tensor:
    """NB2 log probability with ``Var[Y]=mu+mu^2/theta``."""

    if counts.shape != mean.shape or dispersion.shape != (counts.shape[1],):
        raise ContractError("NB count, mean, and gene-dispersion shapes differ.")
    theta = dispersion.unsqueeze(0)
    log_theta_mu = torch.log(theta + mean)
    return (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + counts * (torch.log(mean) - log_theta_mu)
    )


class CountLinkedProgramHead(nn.Module):
    """Sparse hierarchical program head with sample/state/checkpoint effects."""

    guide_to_target: Tensor
    control_guide_mask: Tensor
    normalized_checkpoint_times: Tensor
    target_descriptors: Tensor | None
    target_effect: nn.Parameter | None
    target_descriptor_weight: nn.Parameter | None

    def __init__(
        self,
        *,
        genes: int,
        programs: int,
        state_dimension: int,
        samples: int,
        checkpoints: int,
        targets: int,
        guides: int,
        guide_to_target: Tensor,
        control_guide_indices: tuple[int, ...],
        checkpoint_times: Tensor,
        target_descriptors: Tensor | None = None,
    ) -> None:
        super().__init__()
        if min(genes, programs, samples, checkpoints, targets, guides) <= 0:
            raise ContractError("Program-head dimensions must be positive.")
        mapping = guide_to_target.to(dtype=torch.long)
        if mapping.shape != (guides,) or torch.any(mapping < 0) or torch.any(mapping >= targets):
            raise ContractError("Guide-to-target mapping is invalid.")
        controls = torch.zeros(guides, dtype=torch.bool)
        if not control_guide_indices:
            raise ContractError("Program head requires at least one control guide.")
        controls[list(control_guide_indices)] = True
        times = checkpoint_times.to(dtype=torch.float32)
        if times.shape != (checkpoints,) or not torch.all(times[1:] > times[:-1]):
            raise ContractError("Checkpoint physical times must be complete and increasing.")
        time_range = (times[-1] - times[0]).clamp_min(1e-8)
        normalized_times = (times - times[0]) / time_range
        self.genes = genes
        self.programs = programs
        self.state_dimension = state_dimension
        self.samples = samples
        self.checkpoints = checkpoints
        self.targets = targets
        self.guides = guides
        self.register_buffer("guide_to_target", mapping)
        self.register_buffer("control_guide_mask", controls)
        self.register_buffer("normalized_checkpoint_times", normalized_times)

        self.gene_intercept = nn.Parameter(torch.zeros(genes))
        self.sample_gene = nn.Parameter(torch.zeros(samples, genes))
        self.state_gene = nn.Parameter(torch.zeros(state_dimension, genes))
        self.loading_raw = nn.Parameter(torch.randn(genes, programs) / math.sqrt(genes))
        self.checkpoint_reference = nn.Parameter(torch.zeros(2, programs))
        self.state_program = nn.Parameter(torch.zeros(state_dimension, programs))
        if target_descriptors is None:
            self.target_effect = nn.Parameter(torch.zeros(targets, 2, programs))
            self.target_descriptor_weight = None
            self.register_buffer("target_descriptors", None)
        else:
            descriptors = target_descriptors.to(dtype=torch.float32)
            if descriptors.ndim != 2 or descriptors.shape[0] != targets:
                raise ContractError("Target descriptors must have one row per target.")
            self.target_effect = None
            self.target_descriptor_weight = nn.Parameter(
                torch.zeros(descriptors.shape[1], 2, programs)
            )
            self.register_buffer("target_descriptors", descriptors)
        self.guide_deviation = nn.Parameter(torch.zeros(guides, 2, programs))
        self.guide_efficiency_logit = nn.Parameter(torch.zeros(guides))
        self.dispersion_raw = nn.Parameter(torch.zeros(genes))

    @property
    def normalized_loadings(self) -> Tensor:
        return cast(
            Tensor,
            self.loading_raw
            / torch.linalg.vector_norm(self.loading_raw, dim=0, keepdim=True).clamp_min(1e-8),
        )

    @property
    def dispersion(self) -> Tensor:
        return F.softplus(self.dispersion_raw) + 1e-4

    @property
    def latent_guide_scale(self) -> Tensor:
        """Unidentified latent multiplier, NOT measured biological guide efficiency.

        The historical state-dict key is retained for checkpoint compatibility.
        A centered-deviation successor requires a separately versioned model.
        """
        return torch.sigmoid(self.guide_efficiency_logit)

    def _time_basis(self, checkpoint_index: Tensor) -> Tensor:
        times = self.normalized_checkpoint_times[checkpoint_index]
        return torch.stack((torch.ones_like(times), times), dim=1)

    def _target_activity(self, target_index: Tensor, time_basis: Tensor) -> Tensor:
        if self.target_effect is not None:
            return torch.einsum("nb,nbk->nk", time_basis, self.target_effect[target_index])
        assert self.target_descriptors is not None
        assert self.target_descriptor_weight is not None
        descriptor_effect = torch.einsum(
            "nd,dbk->nbk", self.target_descriptors[target_index], self.target_descriptor_weight
        )
        return torch.einsum("nb,nbk->nk", time_basis, descriptor_effect)

    def program_activity(self, batch: ProgramBatch) -> Tensor:
        """Return reference + target + latent-scaled guide activity (not efficacy)."""

        checkpoint = batch.checkpoint_index.to(dtype=torch.long)
        target = batch.target_index.to(dtype=torch.long)
        guide = batch.guide_index.to(dtype=torch.long)
        time_basis = self._time_basis(checkpoint)
        perturbation_mask = (~self.control_guide_mask[guide]).to(batch.counts.dtype).unsqueeze(1)
        state_gate = (
            torch.tanh(batch.state @ self.state_program)
            if self.state_dimension
            else batch.counts.new_zeros((len(guide), self.programs))
        )
        reference = time_basis @ self.checkpoint_reference + state_gate
        target_activity = self._target_activity(target, time_basis) * (1.0 + 0.25 * state_gate)
        efficiency = self.latent_guide_scale[guide].unsqueeze(1)
        guide_activity = torch.einsum("nb,nbk->nk", time_basis, self.guide_deviation[guide]) * (
            1.0 + 0.25 * state_gate
        )
        return reference + perturbation_mask * (target_activity + efficiency * guide_activity)

    def log_mean(self, batch: ProgramBatch) -> Tensor:
        batch.validate(genes=self.genes, state_dimension=self.state_dimension)
        sample = batch.sample_index.to(dtype=torch.long)
        if torch.any(sample < 0) or torch.any(sample >= self.samples):
            raise ContractError("Sample index is outside the program-head contract.")
        if torch.any(batch.checkpoint_index < 0) or torch.any(
            batch.checkpoint_index >= self.checkpoints
        ):
            raise ContractError("Checkpoint index is outside the program-head contract.")
        guide = batch.guide_index.to(dtype=torch.long)
        target = batch.target_index.to(dtype=torch.long)
        if torch.any(guide < 0) or torch.any(guide >= self.guides):
            raise ContractError("Guide index is outside the program-head contract.")
        if not torch.equal(self.guide_to_target[guide], target):
            raise ContractError("Batch guide/target labels contradict the frozen hierarchy.")
        baseline = self.gene_intercept + self.sample_gene[sample]
        if self.state_dimension:
            baseline = baseline + batch.state @ self.state_gene
        program = self.program_activity(batch) @ self.normalized_loadings.T
        return torch.log(batch.library_size).unsqueeze(1) + baseline + program

    def mean(self, batch: ProgramBatch) -> Tensor:
        return torch.exp(self.log_mean(batch).clamp(min=-20.0, max=20.0))

    def reference_mean(self, batch: ProgramBatch) -> Tensor:
        """Predict the masked reference without consuming observed control outcomes."""
        control = torch.where(self.control_guide_mask)[0][0]
        guide = torch.full_like(batch.guide_index, int(control))
        reference = replace(batch, guide_index=guide, target_index=self.guide_to_target[guide])
        return self.mean(reference)

    def negative_log_likelihood(self, batch: ProgramBatch) -> Tensor:
        log_prob = negative_binomial_log_prob(batch.counts, self.mean(batch), self.dispersion)
        return -log_prob.sum(dim=1).mean()

    def regularization(
        self,
        *,
        loading_l1_weight: float,
        guide_deviation_l2_weight: float,
        target_effect_l2_weight: float,
    ) -> Tensor:
        target_parameter = (
            self.target_effect if self.target_effect is not None else self.target_descriptor_weight
        )
        assert target_parameter is not None
        return (
            loading_l1_weight * self.normalized_loadings.abs().mean()
            + guide_deviation_l2_weight * self.guide_deviation.square().mean()
            + target_effect_l2_weight * target_parameter.square().mean()
            + 1e-4 * self.sample_gene.square().mean()
        )


@dataclass(frozen=True)
class ProgramFitConfig:
    """Deterministic optimization settings selected without protected outcomes."""

    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    loading_l1_weight: float = 1e-2
    guide_deviation_l2_weight: float = 1e-2
    target_effect_l2_weight: float = 1e-3
    max_epochs: int = 300
    patience: int = 30
    minimum_delta: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 0
    minibatch_size: int = 256
    maximum_panel_genes: int = 2048
    maximum_host_payload_bytes: int = 512 * 1024**2


@dataclass(frozen=True)
class ProgramFitResult:
    """Fitted state plus leakage-safe optimization trace."""

    model: CountLinkedProgramHead
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int
    stopped_early: bool


def iter_program_predictions(
    model: CountLinkedProgramHead,
    batches: Iterable[ProgramBatch],
    *,
    device: torch.device | str,
    maximum_batch_rows: int,
    maximum_output_bytes: int,
    include_reference: bool = True,
) -> Iterator[tuple[ProgramBatch, Tensor, Tensor | None]]:
    """Predict bounded host batches; callers consume each result before advancing.

    The byte limit covers prediction tensors, not allocator/workspace peak memory.
    Observed evaluator-reference rows must never be passed to this predictor.
    """
    if min(maximum_batch_rows, maximum_output_bytes) <= 0:
        raise ContractError("Prediction batch and output limits must be positive.")
    model.eval()
    for host_batch in batches:
        rows = len(host_batch.counts)
        needed = rows * model.genes * model.gene_intercept.element_size()
        needed *= 2 if include_reference else 1
        if rows > maximum_batch_rows or needed > maximum_output_bytes:
            raise ContractError("Prediction chunk exceeds its evaluation-output budget.")
        if any(value.device.type != "cpu" for value in vars(host_batch).values()):
            raise ContractError("Prediction input batches must remain on the CPU host.")
        batch = host_batch.to(device)
        with torch.no_grad():
            mean = model.mean(batch)
            reference = model.reference_mean(batch) if include_reference else None
        yield batch, mean, reference


def fit_program_head(
    model: CountLinkedProgramHead,
    *,
    training: ProgramBatch,
    validation: ProgramBatch,
    config: ProgramFitConfig,
    device: torch.device | str = "cpu",
) -> ProgramFitResult:
    """Fit one head using only training and inner-validation rows."""

    if (
        min(
            config.max_epochs,
            config.patience,
            config.minibatch_size,
            config.maximum_panel_genes,
            config.maximum_host_payload_bytes,
        )
        <= 0
    ):
        raise ContractError("Program optimization counts must be positive.")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device_value = torch.device(device)
    # This component accepts a bounded dense host panel, NOT a full-cohort CSR stream.
    if (
        model.genes > config.maximum_panel_genes
        or sum(
            value.numel() * value.element_size()
            for data in (training, validation)
            for value in vars(data).values()
        )
        > config.maximum_host_payload_bytes
    ):
        raise ContractError("Program component exceeds its small-panel host payload budget.")
    if any(
        value.device.type != "cpu"
        for data in (training, validation)
        for value in vars(data).values()
    ):
        raise ContractError("Program component input batches must remain on the CPU host.")
    model = model.to(device_value)
    training.validate(genes=model.genes, state_dimension=model.state_dimension)
    validation.validate(genes=model.genes, state_dimension=model.state_dimension)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_trace: list[float] = []
    validation_trace: list[float] = []
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(config.max_epochs):
        model.train()
        permutation = torch.randperm(len(training.counts), generator=generator)
        epoch_losses: list[float] = []
        for start in range(0, len(permutation), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            batch = training.subset(indices).to(device_value)
            optimizer.zero_grad(set_to_none=True)
            loss = model.negative_log_likelihood(batch) + model.regularization(
                loading_l1_weight=config.loading_l1_weight,
                guide_deviation_l2_weight=config.guide_deviation_l2_weight,
                target_effect_l2_weight=config.target_effect_l2_weight,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        train_trace.append(sum(epoch_losses) / len(epoch_losses))
        model.eval()
        with torch.no_grad():
            total = 0.0
            batches = (
                validation.subset(
                    torch.arange(start, min(start + config.minibatch_size, len(validation.counts)))
                )
                for start in range(0, len(validation.counts), config.minibatch_size)
            )
            for batch, mean, _ in iter_program_predictions(
                model,
                batches,
                device=device_value,
                maximum_batch_rows=config.minibatch_size,
                maximum_output_bytes=config.maximum_host_payload_bytes,
                include_reference=False,
            ):
                total -= float(
                    negative_binomial_log_prob(batch.counts, mean, model.dispersion)
                    .sum(dim=1)
                    .mean()
                ) * len(batch.counts)
            value = total / len(validation.counts)
        validation_trace.append(value)
        if value < best_loss - config.minimum_delta:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise ContractError("Program head did not produce a finite validation state.")
    model.load_state_dict(best_state)
    model.eval()
    return ProgramFitResult(
        model=model,
        training_loss=tuple(train_trace),
        validation_loss=tuple(validation_trace),
        best_epoch=best_epoch,
        stopped_early=len(train_trace) < config.max_epochs,
    )
