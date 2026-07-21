"""Production multi-time trajectory trainer for CREDO."""
from __future__ import annotations

import json
import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from ..config.schema import RunConfig
from ..data.core import MeasureKey, SparseTrajectoryProblem, TrajectoryProblem
from ..data.trajectory_view import TrajectoryView
from ..losses.counts import (
    CountBlock,
    FitnessBank,
    GroupedMultiTimeCountLikelihood,
    MultiTimeCountLikelihood,
    integrated_fitness_curve,
)
from ..losses.multitime import MultiTimeEndpointLoss, checkpoint_indices_for_taus, make_observed_tau_grid
from ..losses.regularizers import RolloutRegularizer
from ..losses.endpoint import EndpointGeometryMassLoss
from ..losses.weak_form import WeakFormLoss
from ..models.full_model import FullDynamicsModel
from ..models.weighted_sde import ParticleRollout, WeightedParticleSimulator
from .trainer import (
    EMA,
    WarmupCosineScheduler,
    _DIAGNOSTIC_KEYS,
    _diagnostics_from_rollout,
    _ess_gate_status,
    _history_dataframe,
    _uses_global_context_backend,
)
from .pruning import TrainingPruned
from .trajectory_batch import (
    TargetBalancedTrajectorySampler,
    TrajectoryContextBank,
    initialise_particles_from_trajectory,
)
from .trajectory_eval import rollout_metrics_by_key_time


@dataclass
class TrajectoryTrainingHistory:
    epochs: list[int] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)
    loss_end: list[float] = field(default_factory=list)
    loss_weak: list[float] = field(default_factory=list)
    loss_count: list[float] = field(default_factory=list)
    loss_reg: list[float] = field(default_factory=list)
    val_endpoint_loss: list[float] = field(default_factory=list)
    context_norm: list[float] = field(default_factory=list)
    q_entropy: list[float] = field(default_factory=list)
    freq_entropy: list[float] = field(default_factory=list)
    within_attention_entropy: list[float] = field(default_factory=list)
    group_attention_entropy: list[float] = field(default_factory=list)
    within_effective_keys: list[float] = field(default_factory=list)
    group_effective_keys: list[float] = field(default_factory=list)
    mass_log_range: list[float] = field(default_factory=list)
    state_to_mediator_effective_keys: list[float] = field(default_factory=list)
    local_to_global_mediator_effective_keys: list[float] = field(default_factory=list)
    mediator_to_group_effective_keys: list[float] = field(default_factory=list)
    edge_sparsity: list[float] = field(default_factory=list)
    effective_edge_mean: list[float] = field(default_factory=list)
    baseline_edge_mean: list[float] = field(default_factory=list)
    residual_edge_sparsity_loss: list[float] = field(default_factory=list)
    edge_entropy: list[float] = field(default_factory=list)
    control_edge_norm: list[float] = field(default_factory=list)
    mediator_orthogonality: list[float] = field(default_factory=list)
    residual_edge_abs_mean: list[float] = field(default_factory=list)
    residual_edge_signed_mean: list[float] = field(default_factory=list)
    mediator_usage_entropy: list[float] = field(default_factory=list)
    mediator_usage_min: list[float] = field(default_factory=list)
    mediator_usage_max: list[float] = field(default_factory=list)
    terminal_ess_frac_mean: list[float] = field(default_factory=list)
    terminal_ess_frac_min: list[float] = field(default_factory=list)
    min_ess_frac_mean: list[float] = field(default_factory=list)
    max_weight_frac_mean: list[float] = field(default_factory=list)
    logw_range_max: list[float] = field(default_factory=list)
    ess_gate_status: list[str] = field(default_factory=list)

    def append(self, epoch: int, metrics: dict[str, float]) -> None:
        self.epochs.append(int(epoch))
        self.loss_total.append(float(metrics.get("loss_total", math.nan)))
        self.loss_end.append(float(metrics.get("loss_end", math.nan)))
        self.loss_weak.append(float(metrics.get("loss_weak", math.nan)))
        self.loss_count.append(float(metrics.get("loss_count", math.nan)))
        self.loss_reg.append(float(metrics.get("loss_reg", math.nan)))
        self.val_endpoint_loss.append(float(metrics.get("val_endpoint_loss", math.nan)))
        for key in _DIAGNOSTIC_KEYS:
            getattr(self, key).append(float(metrics.get(key, math.nan)))
        self.terminal_ess_frac_mean.append(float(metrics.get("terminal_ess_frac_mean", math.nan)))
        self.terminal_ess_frac_min.append(float(metrics.get("terminal_ess_frac_min", math.nan)))
        self.min_ess_frac_mean.append(float(metrics.get("min_ess_frac_mean", math.nan)))
        self.max_weight_frac_mean.append(float(metrics.get("max_weight_frac_mean", math.nan)))
        self.logw_range_max.append(float(metrics.get("logw_range_max", math.nan)))
        self.ess_gate_status.append(str(metrics.get("ess_gate_status", "not_available")))

    def to_dataframe(self) -> pd.DataFrame:
        return _history_dataframe(self, {"epochs": "epoch"})


class TrajectoryTrainer:
    """Full-start multi-time trainer.

    This trainer is intentionally separate from the legacy endpoint
    :class:`credo.training.Trainer`.  It consumes ``TrajectoryProblem`` or
    ``SparseTrajectoryProblem`` and evaluates endpoint losses at observed
    downstream checkpoints on one continuous rollout.
    """

    def __init__(
        self,
        model: FullDynamicsModel,
        config: RunConfig,
        trajectory: TrajectoryProblem | SparseTrajectoryProblem,
        source_label: str | None = None,
        target_labels: list[str] | None = None,
        supported_measure_keys: list[MeasureKey] | None = None,
        *,
        validation_trajectory: TrajectoryProblem | SparseTrajectoryProblem | None = None,
        count_data: Optional[dict] = None,
        output_dir: str | Path | None = None,
        ema_decay: float = 0.995,
        warmup_epochs: int = 10,
        reporter: object | None = None,
    ) -> None:
        tc = config.training
        trc = config.trajectory_training
        config.validate_trajectory_contract(
            model_context_kind=getattr(model, "context_kind", None),
            model_ecological_growth=getattr(model.coeff_nets, "ecological_growth", None),
        )
        resolved_source_label = trc.source_label if source_label is None else source_label
        resolved_target_labels = trc.target_labels if target_labels is None else list(target_labels)
        if resolved_source_label != trc.source_label:
            raise ValueError(
                "TrajectoryTrainer source_label must match RunConfig.trajectory_training.source_label "
                f"({resolved_source_label!r} != {trc.source_label!r})."
            )
        if resolved_target_labels != trc.target_labels:
            raise ValueError(
                "TrajectoryTrainer target_labels must match RunConfig.trajectory_training.target_labels "
                f"({resolved_target_labels!r} != {trc.target_labels!r})."
            )
        resolved_output_dir = Path(config.output_dir if output_dir is None else output_dir)
        if output_dir is not None and resolved_output_dir.resolve() != Path(config.output_dir).resolve():
            raise ValueError(
                "TrajectoryTrainer output_dir must match RunConfig.output_dir "
                f"({str(resolved_output_dir)!r} != {config.output_dir!r})."
            )
        self.model = model
        self.config = config
        self.trajectory = trajectory
        self.validation_trajectory = validation_trajectory
        self.source_label = trc.source_label
        self.target_labels = list(trc.target_labels)
        self.count_data = count_data
        self.device = config.resolve_device()
        self.dtype = torch.float32
        self.output_dir = resolved_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ema_decay = float(ema_decay)
        self.warmup_epochs = int(warmup_epochs)
        # Optional search reporter (duck-typed: .report(epoch, metrics_mapping)
        # and .should_prune() -> bool). credo.training stays independent of
        # credo.search.
        self.reporter = reporter
        self._pruned_epoch: int | None = None

        if getattr(model, "context_kind", "mlp") == "causal_attention":
            raise NotImplementedError(
                "CEA trajectory training requires explicit causal-loss integration and tests."
            )
        self.evaluation_only_labels = list(trc.evaluation_only_labels)
        unknown_evaluation_labels = sorted(
            set(self.evaluation_only_labels) - set(self.target_labels)
        )
        if unknown_evaluation_labels:
            raise ValueError(
                "Evaluation-only labels are not trajectory targets: "
                f"{unknown_evaluation_labels}."
            )
        self.optimization_target_labels = [
            label for label in self.target_labels if label not in self.evaluation_only_labels
        ]
        if not self.optimization_target_labels:
            raise ValueError("At least one trajectory target must contribute to training loss.")

        self.view = TrajectoryView(
            trajectory=trajectory,
            source_label=self.source_label,
            target_labels=self.target_labels,
            measure_keys=supported_measure_keys,
            sparse_missing=trc.sparse_missing,
        )
        self.val_view = None
        if validation_trajectory is not None:
            model_embedding_ids = set(model.perturbation_ids)
            validation_probe = TrajectoryView(
                trajectory=validation_trajectory,
                source_label=self.source_label,
                target_labels=self.target_labels,
                sparse_missing=trc.sparse_missing,
            )
            validation_measure_keys = [
                key
                for key in validation_probe.source_keys
                if validation_probe.embedding_id(key) in model_embedding_ids
            ]
            if not validation_measure_keys:
                raise ValueError("Validation trajectory has no source keys with trained embeddings.")
            self.val_view = TrajectoryView(
                trajectory=validation_trajectory,
                source_label=self.source_label,
                target_labels=self.target_labels,
                measure_keys=validation_measure_keys,
                sparse_missing=trc.sparse_missing,
            )
        if self.val_view is not None:
            self._validate_view_time_axis(self.val_view)
        self.measure_keys = self.view.source_keys
        actual_key_mode = "sample_aware" if isinstance(self.measure_keys[0], tuple) else "pooled"
        if actual_key_mode != trc.key_mode:
            raise ValueError(
                "Trajectory measure-key mode does not match "
                f"RunConfig.trajectory_training.key_mode ({actual_key_mode!r} != {trc.key_mode!r})."
            )
        self.embedding_ids = self.view.embedding_id_list
        self.key_to_global_index = {key: idx for idx, key in enumerate(self.measure_keys)}
        context_views = [self.view]
        if self.val_view is not None:
            heldout_context_keys = [
                key for key in self.val_view.source_keys if key not in self.key_to_global_index
            ]
            if heldout_context_keys:
                context_views.append(
                    TrajectoryView(
                        trajectory=self.val_view.trajectory,
                        source_label=self.val_view.source_label,
                        target_labels=self.val_view.target_labels,
                        measure_keys=heldout_context_keys,
                        sparse_missing=self.val_view.sparse_missing,
                    )
                )
        self.context_measure_keys = [key for view in context_views for key in view.source_keys]
        self.context_key_to_global_index = {
            key: idx for idx, key in enumerate(self.context_measure_keys)
        }
        missing_embeddings = sorted(set(self.embedding_ids) - set(model.perturbation_ids))
        if missing_embeddings:
            raise KeyError(f"Model is missing embedding ids: {missing_embeddings[:10]}")
        self._validate_count_data()

        self.tau_grid = make_observed_tau_grid(
            self.view.observed_taus,
            steps_per_interval=trc.steps_per_interval,
            device=self.device,
            dtype=self.dtype,
        )
        self.checkpoint_indices = checkpoint_indices_for_taus(
            self.tau_grid,
            self.view.time_labels,
            self.view.observed_taus,
        )
        self.target_checkpoint_indices = {
            label: self.checkpoint_indices[label]
            for label in self.target_labels
        }
        self.optimization_checkpoint_indices = {
            label: self.target_checkpoint_indices[label]
            for label in self.optimization_target_labels
        }
        self.evaluation_checkpoint_indices = {
            label: self.target_checkpoint_indices[label]
            for label in self.evaluation_only_labels
        }

        self.precision = tc.precision
        self.autocast_enabled = self.device.startswith("cuda") and self.precision in {"fp16", "bf16"}
        self.compute_dtype = (
            torch.float16 if self.precision == "fp16"
            else torch.bfloat16 if self.precision == "bf16"
            else torch.float32
        )
        needs_history = (
            tc.lambda_weak > 0
            or tc.lambda_count > 0
            or tc.lambda_reg_net > 0
            or tc.lambda_reg_diffusion > 0
            or _uses_global_context_backend(model)
            or trc.context_protocol.startswith("grouped")
        )
        self.simulator = WeightedParticleSimulator(
            n_steps=max(1, len(self.tau_grid) - 1),
            store_history=needs_history,
        )
        self.endpoint_loss = MultiTimeEndpointLoss(
            EndpointGeometryMassLoss(
                eps=tc.sinkhorn_epsilon,
                tau=tc.sinkhorn_tau,
                max_iter=tc.sinkhorn_max_iter,
            ),
            time_weights=trc.endpoint_time_weights,
            reduction="mean",
            normalize_time_weights=trc.normalize_time_weights,
            fail_on_empty=True,
        )
        self.weak_loss = WeakFormLoss(
            n_test_functions=tc.n_test_functions,
            bandwidth=tc.test_function_bandwidth,
            latent_dim=config.latent.dim,
        )
        self.count_lik = MultiTimeCountLikelihood(time_weights=trc.endpoint_time_weights)
        self.grouped_count_lik = GroupedMultiTimeCountLikelihood(
            time_weights=trc.endpoint_time_weights
        )
        self.count_blocks: list[CountBlock] = []
        if isinstance(self.count_data, dict) and "blocks" in self.count_data:
            self.count_blocks = [
                block
                for block in self.count_data["blocks"]
                if block.time_label in self.optimization_checkpoint_indices
            ]
            if self.config.training.lambda_count > 0 and not self.count_blocks:
                raise ValueError(
                    "Grouped count data has no blocks at optimization target labels."
                )
        batching_requested = bool(trc.max_active_measure_keys)
        self.batch_sampler = (
            TargetBalancedTrajectorySampler(
                self.view,
                genes_per_batch=trc.genes_per_batch,
                controls_per_batch=trc.controls_per_batch,
                max_active_measure_keys=trc.max_active_measure_keys,
                seed=tc.seed,
            )
            if batching_requested
            else None
        )
        self.fitness_bank = (
            FitnessBank(
                self.optimization_target_labels,
                len(self.measure_keys),
                device=self.device,
                dtype=torch.float32,
            )
            if self.count_blocks and self.batch_sampler is not None
            else None
        )
        self.regularizer = RolloutRegularizer(
            lambda_diffusion=tc.lambda_reg_diffusion,
            lambda_drift=tc.lambda_reg_net,
            lambda_growth=tc.lambda_reg_net,
        )
        self.model.to(self.device)
        self.weak_loss.to(self.device)
        self.count_lik.to(self.device)
        self.grouped_count_lik.to(self.device)
        self.context_bank: TrajectoryContextBank | None = None
        if trc.context_protocol.startswith("grouped"):
            source_particles = min(8, max(1, self.config.simulation.n_particles))
            with torch.no_grad():
                initial_parts = [
                    initialise_particles_from_trajectory(
                        context_view.trajectory,
                        context_view.source_label,
                        context_view.source_keys,
                        n_particles=source_particles,
                        device=self.device,
                        dtype=self.dtype,
                        seed=tc.seed,
                    )
                    for context_view in context_views
                ]
                z0 = torch.cat([part[0] for part in initial_parts], dim=0)
                logw0 = torch.cat([part[1] for part in initial_parts], dim=0)
                log_m0 = torch.cat([part[2] for part in initial_parts], dim=0)
                initial_stats, _, _ = self.model.context_agg.summarize_groups(z0, logw0, log_m0)
            group_ids = [
                context_view.context_group(key)
                for context_view in context_views
                for key in context_view.source_keys
            ]
            group_order = {value: idx for idx, value in enumerate(sorted(set(group_ids)))}
            full_group_index = torch.tensor(
                [group_order[value] for value in group_ids],
                device=self.device,
                dtype=torch.long,
            )
            self.context_bank = TrajectoryContextBank(
                context_aggregator=self.model.context_agg,
                initial_statistics=initial_stats,
                context_group_index=full_group_index,
                n_steps=len(self.tau_grid) - 1,
            )
        self.history = TrajectoryTrainingHistory()
        self.ema = EMA(self.model, decay=ema_decay) if ema_decay > 0 else None
        self._best_val = math.inf
        self._best_checkpoint_path: Path | None = None

        self._configure_stage(trc.stage)

        self._write_manifests()

    def _validate_view_time_axis(self, view: TrajectoryView) -> None:
        if view.time_labels != self.view.time_labels:
            raise ValueError(
                "Validation trajectory time labels must match training time labels "
                f"({view.time_labels!r} != {self.view.time_labels!r})."
            )
        for label in self.view.time_labels:
            train_tau = float(self.view.trajectory.tau(label))
            val_tau = float(view.trajectory.tau(label))
            if not math.isclose(train_tau, val_tau, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(
                    f"Validation trajectory tau mismatch for {label!r}: "
                    f"{val_tau:.10g} != {train_tau:.10g}."
                )

    def _validate_count_data(self) -> None:
        if self.count_data is None or self.config.training.lambda_count <= 0:
            return
        if isinstance(self.count_data, dict) and "blocks" in self.count_data:
            blocks = list(self.count_data["blocks"])
            if not blocks or not all(isinstance(block, CountBlock) for block in blocks):
                raise TypeError("count_data['blocks'] must be a nonempty list of CountBlock objects.")
            n_keys = len(self.view.source_keys)
            for block in blocks:
                if int(block.key_indices.max()) >= n_keys:
                    raise ValueError("CountBlock key index is outside TrajectoryView source_keys.")
                if block.time_label not in self.target_labels:
                    raise ValueError(f"CountBlock has unknown target time {block.time_label!r}.")
            return
        if "key_order" not in self.count_data:
            raise ValueError("count_data must include key_order when lambda_count > 0.")
        key_level = self.count_data.get("key_level", "measure_key")
        if key_level != "measure_key":
            raise NotImplementedError(
                "TrajectoryTrainer count loss currently requires measure_key-level "
                "count_data. Aggregate embedding-level counts before passing them "
                "once duplicate embedding IDs are present."
            )
        key_order = list(self.count_data["key_order"])
        if key_order != list(self.view.source_keys):
            raise ValueError("count_data key_order must exactly match TrajectoryView source_keys.")

    def _configure_stage(self, stage: str) -> None:
        """Apply coarse continuation-stage parameter freezing."""
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        if stage == "joint":
            return
        if stage == "geometry":
            for parameter in self.model.coeff_nets.growth_head.parameters():
                parameter.requires_grad_(False)
            if self.model.embedding.growth_bias is not None:
                self.model.embedding.growth_bias.requires_grad_(False)
            if self.model.embedding.shared_growth_bias is not None:
                self.model.embedding.shared_growth_bias.requires_grad_(False)
            self.model.freeze_ecology()
            return
        if stage == "reaction":
            for parameter in self.model.coeff_nets.drift_head.parameters():
                parameter.requires_grad_(False)
            for parameter in self.model.coeff_nets.sigma_head.parameters():
                parameter.requires_grad_(False)
            self.model.freeze_embeddings()
            if hasattr(self.model.context_agg, "parameters"):
                for parameter in self.model.context_agg.parameters():
                    parameter.requires_grad_(False)
            return
        if stage == "context":
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            for parameter in self.model.context_agg.parameters():
                parameter.requires_grad_(True)
            for parameter in self.model.coeff_nets.growth_head.parameters():
                parameter.requires_grad_(True)
            return
        if stage == "control":
            self.model.freeze_embeddings()
            self.model.unfreeze_control_reference()
            self.model.freeze_ecology()
            return
        raise ValueError(f"Unknown trajectory stage {stage!r}.")

    def _write_manifests(self) -> None:
        from credo import __version__ as credo_version

        trajectory_config = {
            "credo_version": credo_version,
            "git_sha": self.config.git_sha,
            "source_label": self.source_label,
            "target_labels": self.target_labels,
            "optimization_target_labels": self.optimization_target_labels,
            "evaluation_only_labels": self.evaluation_only_labels,
            "physical_times": {
                label: self.trajectory.time_axis.physical(label)
                for label in self.view.time_labels
            },
            "normalized_taus": {
                label: self.trajectory.tau(label)
                for label in self.view.time_labels
            },
            "steps_per_interval": self.config.trajectory_training.steps_per_interval,
            "sparse_missing": self.config.trajectory_training.sparse_missing,
            "context_protocol": self.config.trajectory_training.context_protocol,
            "stage": self.config.trajectory_training.stage,
            "max_active_measure_keys": self.config.trajectory_training.max_active_measure_keys,
        }
        (self.output_dir / "trajectory_config.json").write_text(
            json.dumps(trajectory_config, indent=2),
            encoding="utf-8",
        )
        try:
            (self.output_dir / "run_config.json").write_text(
                self.config.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        manifest_rows = []
        for key in self.measure_keys:
            sample_id, perturbation_id = ("", str(key)) if isinstance(key, str) else key
            manifest_rows.append(
                {
                    "measure_key": str(key),
                    "sample_id": str(sample_id),
                    "perturbation_id": str(perturbation_id),
                    "embedding_id": self.view.embedding_id(key),
                    "guide_id": self.view.guide_id(key),
                    "target_gene": self.view.target_gene(key),
                    "context_group_id": self.view.context_group(key),
                    "is_control": key in set(self.view.control_measure_keys),
                    "has_source": key in self.trajectory.measures[self.source_label],
                }
            )
        pd.DataFrame(manifest_rows).to_csv(self.output_dir / "measure_key_manifest.csv", index=False)

        coverage_rows = []
        for label in self.target_labels:
            active = set(self.view.active_keys(label))
            for key in self.measure_keys:
                coverage_rows.append(
                    {
                        "time_label": label,
                        "measure_key": str(key),
                        "active": key in active,
                        "endpoint_role": (
                            "evaluation_only"
                            if label in self.evaluation_only_labels
                            else "optimization"
                        ),
                    }
                )
        pd.DataFrame(coverage_rows).to_csv(self.output_dir / "target_coverage_by_time.csv", index=False)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.training
        def _no_decay(name: str) -> bool:
            return name.endswith(".bias") or "norm" in name.lower() or "layernorm" in name.lower()

        grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
            ("net", False): [],
            ("net", True): [],
            ("embed", False): [],
            ("embed", True): [],
            ("transformer", False): [],
            ("transformer", True): [],
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if _uses_global_context_backend(self.model) and name.startswith("context_agg."):
                group = "transformer"
            elif "embedding" in name:
                group = "embed"
            else:
                group = "net"
            grouped[(group, _no_decay(name))].append(param)
        grouped[("net", False)].extend(p for p in self.count_lik.parameters() if p.requires_grad)
        grouped[("net", False)].extend(
            p for p in self.grouped_count_lik.parameters() if p.requires_grad
        )

        param_groups = []
        specs = {
            "net": (tc.lr_net, tc.weight_decay),
            "embed": (tc.lr_embed, tc.weight_decay),
            "transformer": (tc.lr_transformer, tc.transformer_weight_decay),
        }
        for group, (lr, decay) in specs.items():
            decay_params = grouped[(group, False)]
            no_decay_params = grouped[(group, True)]
            if decay_params:
                param_groups.append({"params": decay_params, "lr": lr, "weight_decay": decay})
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
        cls = torch.optim.AdamW if tc.optimizer == "adamw" else torch.optim.Adam
        return cls(param_groups, weight_decay=0.0)

    def _autocast_context(self):
        if self.autocast_enabled:
            return torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return nullcontext()

    def _rollout(
        self,
        view: TrajectoryView,
        *,
        n_particles: int,
        seed: int,
        training: bool,
    ) -> tuple[ParticleRollout, torch.Tensor, torch.Tensor, torch.Tensor]:
        rollout_dtype = self.compute_dtype if self.autocast_enabled else self.dtype
        z0, logw0, log_m0 = initialise_particles_from_trajectory(
            view.trajectory,
            view.source_label,
            view.source_keys,
            n_particles=n_particles,
            device=self.device,
            dtype=rollout_dtype,
            seed=seed,
        )
        with self._autocast_context():
            context_group_index = None
            if self.config.trajectory_training.context_protocol.startswith("grouped"):
                group_ids = [view.context_group(key) for key in view.source_keys]
                group_order = {group_id: idx for idx, group_id in enumerate(sorted(set(group_ids)))}
                context_group_index = torch.tensor(
                    [group_order[group_id] for group_id in group_ids],
                    device=self.device,
                    dtype=torch.long,
                )
            active_global_indices = None
            context_override = None
            if self.context_bank is not None and all(
                key in self.context_key_to_global_index for key in view.source_keys
            ):
                active_global_indices = torch.tensor(
                    [self.context_key_to_global_index[key] for key in view.source_keys],
                    device=self.device,
                    dtype=torch.long,
                )
                context_override = self.context_bank.override(
                    active_global_indices,
                    replace_active=(
                        self.config.trajectory_training.context_protocol
                        == "grouped_self_consistent"
                    ),
                )
            rollout = self.simulator.rollout(
                z0=z0,
                logw0=logw0,
                model=self.model,
                log_m0=log_m0,
                tau_start=view.trajectory.tau(view.source_label),
                tau_end=view.trajectory.tau(view.target_labels[-1]),
                tau_grid=self.tau_grid,
                perturbation_ids=[str(key) for key in view.source_keys],
                embedding_ids=[view.embedding_id(key) for key in view.source_keys],
                context_group_index=context_group_index,
                context_override=context_override,
            )
        return rollout, z0, logw0, log_m0

    def _loss_for_rollout(
        self,
        view: TrajectoryView,
        rollout: ParticleRollout,
        *,
        epoch: int,
        training: bool = True,
        active_key_indices: torch.Tensor | None = None,
        include_count: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], pd.DataFrame]:
        tc = self.config.training
        target_support, target_logw = view.target_tensors(
            device=self.device,
            dtype=torch.float32,
            labels=(self.target_labels if not training else self.optimization_target_labels),
            max_atoms=(
                self.config.trajectory_training.max_train_target_atoms
                if training and self.config.trajectory_training.max_train_target_atoms > 0
                else self.config.trajectory_training.max_eval_target_atoms
                if not training and self.config.trajectory_training.max_eval_target_atoms > 0
                else None
            ),
            seed=self.config.training.seed + int(epoch),
        )
        float_rollout = ParticleRollout(
            z_steps=rollout.z_steps.float(),
            logw_steps=rollout.logw_steps.float(),
            tau_steps=rollout.tau_steps.float(),
            log_m0=rollout.log_m0.float() if rollout.log_m0 is not None else None,
        )
        loss_end, endpoint_logs = self.endpoint_loss(
            rollout=float_rollout,
            checkpoint_indices=self.optimization_checkpoint_indices,
            target_support_by_time=target_support,
            target_logw_by_time=target_logw,
            prediction_keys=view.source_keys,
            embedding_ids=[view.embedding_id(key) for key in view.source_keys],
        )
        evaluation_endpoint_logs: dict[str, torch.Tensor] = {}
        if not training and self.evaluation_checkpoint_indices:
            _, evaluation_endpoint_logs = self.endpoint_loss(
                rollout=float_rollout,
                checkpoint_indices=self.evaluation_checkpoint_indices,
                target_support_by_time=target_support,
                target_logw_by_time=target_logw,
                prediction_keys=view.source_keys,
                embedding_ids=[view.embedding_id(key) for key in view.source_keys],
            )

        loss_weak = torch.tensor(0.0, device=self.device)
        if tc.lambda_weak > 0 and rollout.drift_steps is not None:
            loss_weak = self.weak_loss(
                z_steps=rollout.z_steps.float(),
                logw_steps=rollout.logw_steps.float(),
                drift_steps=rollout.drift_steps.float(),
                sigma_steps=rollout.sigma_steps.float(),
                growth_steps=rollout.growth_steps.float(),
                tau_steps=rollout.tau_steps.float(),
                refresh_centers=(epoch % 10 == 0),
            )

        loss_count = torch.tensor(0.0, device=self.device)
        count_logs: dict[str, torch.Tensor] = {}
        if (
            tc.lambda_count > 0
            and self.count_data is not None
            and rollout.growth_steps is not None
            and include_count
        ):
            if self.count_blocks:
                if active_key_indices is None:
                    active_key_indices = torch.tensor(
                        [self.key_to_global_index[key] for key in view.source_keys],
                        device=self.device,
                        dtype=torch.long,
                    )
                loss_count, count_logs = self.grouped_count_lik.forward_with_logs(
                    growth_steps=rollout.growth_steps.float(),
                    logw_steps=rollout.logw_steps.float(),
                    tau_steps=rollout.tau_steps.float(),
                    blocks=self.count_blocks,
                    checkpoint_indices=self.optimization_checkpoint_indices,
                    active_key_indices=active_key_indices,
                    fitness_bank=self.fitness_bank,
                )
            else:
                labels = set(self.optimization_checkpoint_indices)
                exposures = {
                    label: value.to(device=self.device, dtype=torch.float32)
                    for label, value in self.count_data["exposures"].items()
                    if label in labels
                }
                count_matrices = {
                    label: value.to(device=self.device, dtype=torch.float32)
                    for label, value in self.count_data["count_matrices"].items()
                    if label in labels
                }
                n_totals = {
                    label: value.to(device=self.device, dtype=torch.float32)
                    for label, value in self.count_data["n_totals"].items()
                    if label in labels
                }
                loss_count, count_logs = self.count_lik.forward_with_logs(
                    growth_steps=rollout.growth_steps.float(),
                    logw_steps=rollout.logw_steps.float(),
                    tau_steps=rollout.tau_steps.float(),
                    exposures=exposures,
                    count_matrices=count_matrices,
                    n_totals=n_totals,
                    checkpoint_indices=self.optimization_checkpoint_indices,
                )

        loss_reg = self.regularizer(
            drift_steps=rollout.drift_steps.float() if rollout.drift_steps is not None else None,
            sigma_steps=rollout.sigma_steps.float() if rollout.sigma_steps is not None else None,
            growth_steps=rollout.growth_steps.float() if rollout.growth_steps is not None else None,
        ).to(self.device)
        loss_reg = loss_reg + self.model.regularization(lambda_embed=tc.lambda_reg_embed)
        loss_reg = loss_reg + self.model.growth_bias_regularization(lambda_growth_bias=tc.lambda_reg_growth_bias)

        loss_total = (
            tc.lambda_end * loss_end
            + tc.lambda_weak * loss_weak
            + tc.lambda_count * loss_count
            + loss_reg
        )
        evaluation_logs = {
            key.replace("endpoint/", "evaluation_only_endpoint/", 1): value
            for key, value in evaluation_endpoint_logs.items()
        }
        logs = {**endpoint_logs, **evaluation_logs, **count_logs}
        metrics = {
            "loss_total": float(loss_total.detach().cpu()),
            "loss_end": float(loss_end.detach().cpu()),
            "loss_weak": float(loss_weak.detach().cpu()),
            "loss_count": float(loss_count.detach().cpu()),
            "loss_reg": float(loss_reg.detach().cpu()),
            **_diagnostics_from_rollout(rollout),
        }
        for label in self.evaluation_only_labels:
            value = evaluation_endpoint_logs.get(f"endpoint/{label}")
            if value is not None:
                metrics[f"evaluation_only_{label}_loss"] = float(value.detach().cpu())
        metrics["ess_gate_status"] = _ess_gate_status(metrics, tc)
        report_endpoint_logs = {**endpoint_logs, **evaluation_endpoint_logs}
        pred_df = rollout_metrics_by_key_time(
            view,
            rollout,
            self.target_checkpoint_indices,
            view.source_keys,
            endpoint_logs=report_endpoint_logs,
            time_labels=(
                self.target_labels if not training else self.optimization_target_labels
            ),
        )
        pred_df["endpoint_role"] = pred_df["time_label"].map(
            lambda label: "evaluation_only"
            if label in self.evaluation_only_labels
            else "optimization"
        )
        pred_df["used_for_training_loss"] = pred_df["endpoint_role"].eq("optimization")
        return loss_total, logs, metrics, pred_df

    def _view_for_keys(self, base_view: TrajectoryView, keys: list[MeasureKey]) -> TrajectoryView:
        return TrajectoryView(
            trajectory=base_view.trajectory,
            source_label=base_view.source_label,
            target_labels=base_view.target_labels,
            measure_keys=keys,
            sparse_missing=base_view.sparse_missing,
        )

    def _training_key_batches(self, epoch: int) -> list[list[MeasureKey]]:
        if self.batch_sampler is None:
            return [self.view.source_keys]
        batches = list(self.batch_sampler.batches(sweep=epoch - 1))
        requested = self.config.trajectory_training.steps_per_epoch
        if requested is None:
            return batches
        if not batches:
            raise ValueError("Target-balanced sampler produced no training batches.")
        return [batches[idx % len(batches)] for idx in range(requested)]

    def _evaluation_key_batches(self, view: TrajectoryView, epoch: int) -> list[list[MeasureKey]]:
        if self.batch_sampler is None or view is not self.view:
            cap = self.config.trajectory_training.max_active_measure_keys
            if cap and len(view.source_keys) > cap:
                return [view.source_keys[start:start + cap] for start in range(0, len(view.source_keys), cap)]
            return [view.source_keys]
        # Training batches repeat controls to anchor every target update. During
        # evaluation each key should be emitted exactly once.
        seen: set[MeasureKey] = set()
        batches: list[list[MeasureKey]] = []
        for candidate in self.batch_sampler.batches(sweep=epoch):
            unique = [key for key in candidate if key not in seen]
            if unique:
                batches.append(unique)
                seen.update(unique)
        remaining = [key for key in view.source_keys if key not in seen]
        if remaining:
            cap = self.config.trajectory_training.max_active_measure_keys or len(remaining)
            batches.extend(
                remaining[start:start + cap] for start in range(0, len(remaining), cap)
            )
        return batches

    @staticmethod
    def _mean_metrics(batch_metrics: list[dict[str, float]]) -> dict[str, float]:
        keys = set().union(*(metrics.keys() for metrics in batch_metrics))
        out: dict[str, float] = {}
        for key in keys:
            values = [metrics[key] for metrics in batch_metrics if isinstance(metrics.get(key), (int, float))]
            finite = [float(value) for value in values if math.isfinite(float(value))]
            out[key] = float(sum(finite) / len(finite)) if finite else math.nan
        statuses = [str(metrics.get("ess_gate_status", "not_available")) for metrics in batch_metrics]
        if "fail" in statuses:
            out["ess_gate_status"] = "fail"  # type: ignore[assignment]
        elif "warn" in statuses:
            out["ess_gate_status"] = "warn"  # type: ignore[assignment]
        else:
            out["ess_gate_status"] = statuses[0] if statuses else "not_available"  # type: ignore[assignment]
        return out

    def train(self) -> TrajectoryTrainingHistory:
        optimizer = self._build_optimizer()
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=max(1, self.config.training.epochs),
        )
        tc = self.config.training
        for epoch in range(1, tc.epochs + 1):
            self.model.train()
            batch_metrics: list[dict[str, float]] = []
            for batch_index, keys in enumerate(self._training_key_batches(epoch)):
                batch_view = self._view_for_keys(self.view, keys)
                active_indices = torch.tensor(
                    [self.key_to_global_index[key] for key in keys],
                    device=self.device,
                    dtype=torch.long,
                )
                rollout, _, _, _ = self._rollout(
                    batch_view,
                    n_particles=self.config.simulation.n_particles,
                    seed=tc.seed + epoch * 100_003 + batch_index,
                    training=True,
                )
                loss, _, metrics_batch, _ = self._loss_for_rollout(
                    batch_view,
                    rollout,
                    epoch=epoch,
                    training=True,
                    active_key_indices=active_indices,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
                optimizer.step()
                if self.fitness_bank is not None and rollout.growth_steps is not None:
                    self.fitness_bank.update(
                        integrated_fitness_curve(
                            rollout.growth_steps.float(),
                            rollout.logw_steps.float(),
                            rollout.tau_steps.float(),
                        ),
                        self.optimization_checkpoint_indices,
                        active_indices,
                    )
                if self.context_bank is not None:
                    context_indices = torch.tensor(
                        [self.context_key_to_global_index[key] for key in keys],
                        device=self.device,
                        dtype=torch.long,
                    )
                    self.context_bank.update(
                        rollout,
                        context_indices,
                        hard=(epoch % self.config.trajectory_training.context_bank_refresh == 0),
                    )
                batch_metrics.append(metrics_batch)
                if self.ema is not None:
                    self.ema.update()
            scheduler.step()
            metrics = self._mean_metrics(batch_metrics)

            should_evaluate = (
                epoch % self.config.trajectory_training.eval_every == 0
                or epoch == tc.epochs
            )
            val = self.evaluate(epoch=epoch, use_ema=False) if should_evaluate else None
            metrics["val_endpoint_loss"] = (
                val["metrics"].get("loss_end", math.nan) if val is not None else math.nan
            )
            metrics["validation_source"] = (
                val["metrics"].get("validation_source") if val is not None else "not_evaluated"
            )
            self.history.append(epoch, metrics)

            if should_evaluate and metrics["val_endpoint_loss"] < self._best_val:
                self._best_val = metrics["val_endpoint_loss"]
                self._best_checkpoint_path = self.save_checkpoint("checkpoint_best.pt", epoch=epoch)
                if self.ema is not None:
                    self.ema.apply_shadow()
                    self.save_checkpoint("checkpoint_best_ema.pt", epoch=epoch)
                    self.ema.restore()

            # Intermediate reporting / pruning for setting search (read-only).
            # On a prune request, persist the pruned checkpoint/history and raise
            # TrainingPruned so the trial is reported as pruned rather than scored
            # as a short completed run (and checkpoint_last is not written with a
            # misleading final epoch).
            if self.reporter is not None:
                self.reporter.report(epoch, metrics)
                if self.reporter.should_prune():
                    self._pruned_epoch = epoch
                    self.save_checkpoint("checkpoint_pruned.pt", epoch=epoch)
                    self.history.to_dataframe().to_csv(
                        self.output_dir / "training_history.csv", index=False
                    )
                    raise TrainingPruned(epoch)

            if epoch % max(1, tc.log_every) == 0 or epoch == tc.epochs:
                self.history.to_dataframe().to_csv(self.output_dir / "training_history.csv", index=False)

        self.save_checkpoint("checkpoint_last.pt", epoch=tc.epochs)
        self.history.to_dataframe().to_csv(self.output_dir / "training_history.csv", index=False)
        return self.history

    def _warn_train_self_eval_once(self) -> None:
        """Warn (once) when held-out validation was requested but is unavailable."""
        if getattr(self, "_warned_train_self_eval", False):
            return
        self._warned_train_self_eval = True
        warnings.warn(
            "TrajectoryTrainer received no validation trajectory. Validation metrics "
            "and best-checkpoint selection are computed on the TRAINING view "
            "(validation_source='train_self_eval'); reported 'val_*' values are "
            "NOT held out.",
            RuntimeWarning,
            stacklevel=3,
        )

    @torch.no_grad()
    def evaluate(self, *, epoch: int = 0, use_ema: bool = False) -> dict[str, object]:
        if self.val_view is not None:
            view = self.val_view
            validation_source = "held_out"
        else:
            view = self.view
            validation_source = "train_self_eval"
            self._warn_train_self_eval_once()
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        try:
            self.model.eval()
            metric_batches: list[dict[str, float]] = []
            prediction_batches: list[pd.DataFrame] = []
            for batch_index, keys in enumerate(self._evaluation_key_batches(view, epoch)):
                batch_view = self._view_for_keys(view, keys)
                rollout, _, _, _ = self._rollout(
                    batch_view,
                    n_particles=max(1, self.config.eval.n_eval_particles),
                    seed=self.config.training.seed + 100_000 + epoch * 1009 + batch_index,
                    training=False,
                )
                active_indices = None
                if all(key in self.key_to_global_index for key in keys):
                    active_indices = torch.tensor(
                        [self.key_to_global_index[key] for key in keys],
                        device=self.device,
                        dtype=torch.long,
                    )
                _, _, metrics_batch, pred_batch = self._loss_for_rollout(
                    batch_view,
                    rollout,
                    epoch=epoch,
                    training=False,
                    active_key_indices=active_indices,
                    include_count=view is self.view,
                )
                metric_batches.append(metrics_batch)
                prediction_batches.append(pred_batch)
            metrics = self._mean_metrics(metric_batches)
            pred_df = pd.concat(prediction_batches, ignore_index=True)
            metrics["validation_source"] = validation_source
            pred_df["validation_source"] = validation_source
            pred_df.to_csv(self.output_dir / "predicted_metrics_by_key_time.csv", index=False)
            if self.evaluation_only_labels:
                pred_df.loc[
                    pred_df["time_label"].isin(self.evaluation_only_labels)
                ].to_csv(
                    self.output_dir / "evaluation_only_metrics_by_key_time.csv",
                    index=False,
                )
            pd.DataFrame([{"epoch": epoch, **metrics}]).to_csv(
                self.output_dir / "validation_history.csv",
                mode="a",
                header=not (self.output_dir / "validation_history.csv").exists(),
                index=False,
            )
            return {"metrics": metrics, "predictions": pred_df}
        finally:
            if use_ema and self.ema is not None:
                self.ema.restore()

    def save_checkpoint(self, name: str, *, epoch: int) -> Path:
        path = self.output_dir / name
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "history": self.history.to_dataframe().to_dict(orient="list"),
            "source_label": self.source_label,
            "target_labels": self.target_labels,
            "optimization_target_labels": self.optimization_target_labels,
            "evaluation_only_labels": self.evaluation_only_labels,
            "measure_keys": [str(key) for key in self.measure_keys],
            "embedding_ids": self.embedding_ids,
            "count_likelihood_state_dict": self.count_lik.state_dict(),
            "grouped_count_likelihood_state_dict": self.grouped_count_lik.state_dict(),
        }
        if self.ema is not None:
            payload["ema_state_dict"] = self.ema.state_dict()
        if self.fitness_bank is not None:
            payload["fitness_bank"] = {
                "time_labels": self.fitness_bank.time_labels,
                "values": self.fitness_bank.values.detach().cpu(),
                "seen": self.fitness_bank.seen.detach().cpu(),
            }
        if self.context_bank is not None:
            payload["context_bank"] = {
                "measure_keys": [str(key) for key in self.context_measure_keys],
                "log_n_steps": self.context_bank.log_n_steps.detach().cpu(),
                "eta_steps": self.context_bank.eta_steps.detach().cpu(),
                "phi_steps": self.context_bank.phi_steps.detach().cpu(),
                "context_group_index": self.context_bank.context_group_index.detach().cpu(),
            }
        torch.save(payload, path)
        return path

    def load_auxiliary_state(self, payload: dict) -> None:
        """Restore detached banks and count dispersion for a continuation run."""
        if "count_likelihood_state_dict" in payload:
            self.count_lik.load_state_dict(payload["count_likelihood_state_dict"])
        if "grouped_count_likelihood_state_dict" in payload:
            self.grouped_count_lik.load_state_dict(payload["grouped_count_likelihood_state_dict"])

        fitness = payload.get("fitness_bank")
        if fitness is not None and self.fitness_bank is not None:
            if list(fitness.get("time_labels", [])) != self.fitness_bank.time_labels:
                raise ValueError("Checkpoint FitnessBank time labels do not match this trajectory.")
            for name in ("values", "seen"):
                source = torch.as_tensor(fitness[name])
                target = getattr(self.fitness_bank, name)
                if source.shape != target.shape:
                    raise ValueError(f"Checkpoint FitnessBank {name} shape does not match.")
                target.copy_(source.to(device=target.device, dtype=target.dtype))

        context = payload.get("context_bank")
        if context is not None and self.context_bank is not None:
            saved_keys = context.get("measure_keys")
            expected_keys = [str(key) for key in self.context_measure_keys]
            if saved_keys is not None and list(saved_keys) != expected_keys:
                raise ValueError("Checkpoint ContextBank measure-key order does not match.")
            for name in ("log_n_steps", "eta_steps", "phi_steps", "context_group_index"):
                source = torch.as_tensor(context[name])
                target = getattr(self.context_bank, name)
                if source.shape != target.shape:
                    raise ValueError(f"Checkpoint ContextBank {name} shape does not match.")
                target.copy_(source.to(device=target.device, dtype=target.dtype))


__all__ = ["TrajectoryTrainer", "TrajectoryTrainingHistory"]
