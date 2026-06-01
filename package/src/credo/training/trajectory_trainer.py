"""Production multi-time trajectory trainer for CREDO."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from ..config.schema import RunConfig
from ..data.core import MeasureKey, SparseTrajectoryProblem, TrajectoryProblem
from ..data.trajectory_view import TrajectoryView, embedding_id_for_measure_key
from ..losses.counts import MultiTimeCountLikelihood
from ..losses.multitime import MultiTimeEndpointLoss, checkpoint_indices_for_taus, make_observed_tau_grid
from ..losses.regularizers import RolloutRegularizer
from ..losses.uot import UOTLoss
from ..losses.weak_form import WeakFormLoss
from ..models.full_model import FullDynamicsModel
from ..models.weighted_sde import ParticleRollout, WeightedParticleSimulator
from .trainer import (
    EMA,
    WarmupCosineScheduler,
    _DIAGNOSTIC_KEYS,
    _diagnostics_from_rollout,
    _uses_global_context_backend,
)
from .trajectory_batch import initialise_particles_from_trajectory
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

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epoch": self.epochs,
                "loss_total": self.loss_total,
                "loss_end": self.loss_end,
                "loss_weak": self.loss_weak,
                "loss_count": self.loss_count,
                "loss_reg": self.loss_reg,
                "val_endpoint_loss": self.val_endpoint_loss,
                "context_norm": self.context_norm,
                "q_entropy": self.q_entropy,
                "freq_entropy": self.freq_entropy,
                "within_attention_entropy": self.within_attention_entropy,
                "group_attention_entropy": self.group_attention_entropy,
                "within_effective_keys": self.within_effective_keys,
                "group_effective_keys": self.group_effective_keys,
                "mass_log_range": self.mass_log_range,
                "state_to_mediator_effective_keys": self.state_to_mediator_effective_keys,
                "local_to_global_mediator_effective_keys": self.local_to_global_mediator_effective_keys,
                "mediator_to_group_effective_keys": self.mediator_to_group_effective_keys,
                "edge_sparsity": self.edge_sparsity,
                "effective_edge_mean": self.effective_edge_mean,
                "baseline_edge_mean": self.baseline_edge_mean,
                "residual_edge_sparsity_loss": self.residual_edge_sparsity_loss,
                "edge_entropy": self.edge_entropy,
                "control_edge_norm": self.control_edge_norm,
                "mediator_orthogonality": self.mediator_orthogonality,
                "residual_edge_abs_mean": self.residual_edge_abs_mean,
                "residual_edge_signed_mean": self.residual_edge_signed_mean,
                "mediator_usage_entropy": self.mediator_usage_entropy,
                "mediator_usage_min": self.mediator_usage_min,
                "mediator_usage_max": self.mediator_usage_max,
            }
        )


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
        source_label: str,
        target_labels: list[str],
        supported_measure_keys: list[MeasureKey] | None = None,
        *,
        validation_trajectory: TrajectoryProblem | SparseTrajectoryProblem | None = None,
        count_data: Optional[dict] = None,
        output_dir: str = "outputs",
        ema_decay: float = 0.995,
        warmup_epochs: int = 10,
    ) -> None:
        self.model = model
        self.config = config
        self.trajectory = trajectory
        self.validation_trajectory = validation_trajectory
        self.source_label = source_label
        self.target_labels = list(target_labels)
        self.count_data = count_data
        self.device = config.resolve_device()
        self.dtype = torch.float32
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ema_decay = float(ema_decay)
        self.warmup_epochs = int(warmup_epochs)

        tc = config.training
        trc = config.trajectory_training
        if trc.trajectory_mode != "full_start":
            raise NotImplementedError("TrajectoryTrainer supports trajectory_mode='full_start' only.")
        if trc.context_batch_mode != "all_keys":
            raise NotImplementedError("TrajectoryTrainer supports context_batch_mode='all_keys' only.")
        if trc.max_active_measure_keys:
            raise NotImplementedError("TrajectoryTrainer rolls out all active measure keys together.")
        if getattr(model, "context_kind", "mlp") == "causal_attention":
            raise NotImplementedError(
                "CEA trajectory training requires explicit causal-loss integration and tests."
            )

        self.view = TrajectoryView(
            trajectory=trajectory,
            source_label=source_label,
            target_labels=self.target_labels,
            measure_keys=supported_measure_keys,
            sparse_missing=trc.sparse_missing,
        )
        self.val_view = None
        if validation_trajectory is not None:
            model_embedding_ids = set(model.perturbation_ids)
            validation_measure_keys = [
                key
                for key in validation_trajectory.measures[source_label]
                if embedding_id_for_measure_key(key) in model_embedding_ids
            ]
            if not validation_measure_keys:
                raise ValueError("Validation trajectory has no source keys with trained embeddings.")
            self.val_view = TrajectoryView(
                trajectory=validation_trajectory,
                source_label=source_label,
                target_labels=self.target_labels,
                measure_keys=validation_measure_keys,
                sparse_missing=trc.sparse_missing,
            )
        if self.val_view is not None:
            self._validate_view_time_axis(self.val_view)
        self.measure_keys = self.view.source_keys
        self.embedding_ids = [embedding_id_for_measure_key(key) for key in self.measure_keys]
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
        )
        self.simulator = WeightedParticleSimulator(
            n_steps=max(1, len(self.tau_grid) - 1),
            store_history=needs_history,
        )
        self.endpoint_loss = MultiTimeEndpointLoss(
            UOTLoss(
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
        self.regularizer = RolloutRegularizer(
            lambda_diffusion=tc.lambda_reg_diffusion,
            lambda_drift=tc.lambda_reg_net,
            lambda_growth=tc.lambda_reg_net,
        )
        self.model.to(self.device)
        self.weak_loss.to(self.device)
        self.count_lik.to(self.device)
        self.history = TrajectoryTrainingHistory()
        self.ema = EMA(self.model, decay=ema_decay) if ema_decay > 0 else None
        self._best_val = math.inf
        self._best_checkpoint_path: Path | None = None

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

    def _write_manifests(self) -> None:
        from credo import __version__ as credo_version

        trajectory_config = {
            "credo_version": credo_version,
            "git_sha": self.config.git_sha,
            "source_label": self.source_label,
            "target_labels": self.target_labels,
            "physical_times": {
                label: self.trajectory.time_axis.physical(label)
                for label in self.view.time_labels
            },
            "normalized_taus": {
                label: self.trajectory.tau(label)
                for label in self.view.time_labels
            },
            "trajectory_mode": self.config.trajectory_training.trajectory_mode,
            "steps_per_interval": self.config.trajectory_training.steps_per_interval,
            "context_batch_mode": self.config.trajectory_training.context_batch_mode,
            "sparse_missing": self.config.trajectory_training.sparse_missing,
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
                    "embedding_id": embedding_id_for_measure_key(key),
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
            rollout = self.simulator.rollout(
                z0=z0,
                logw0=logw0,
                model=self.model,
                log_m0=log_m0,
                tau_start=view.trajectory.tau(view.source_label),
                tau_end=view.trajectory.tau(view.target_labels[-1]),
                tau_grid=self.tau_grid,
                perturbation_ids=[embedding_id_for_measure_key(key) for key in view.source_keys],
            )
        return rollout, z0, logw0, log_m0

    def _loss_for_rollout(
        self,
        view: TrajectoryView,
        rollout: ParticleRollout,
        *,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], pd.DataFrame]:
        tc = self.config.training
        target_support, target_logw = view.target_tensors(
            device=self.device,
            dtype=torch.float32,
        )
        loss_end, endpoint_logs = self.endpoint_loss(
            rollout=ParticleRollout(
                z_steps=rollout.z_steps.float(),
                logw_steps=rollout.logw_steps.float(),
                tau_steps=rollout.tau_steps.float(),
                log_m0=rollout.log_m0.float() if rollout.log_m0 is not None else None,
            ),
            checkpoint_indices=self.target_checkpoint_indices,
            target_support_by_time=target_support,
            target_logw_by_time=target_logw,
            prediction_keys=view.source_keys,
            embedding_ids=[embedding_id_for_measure_key(key) for key in view.source_keys],
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
        ):
            exposures = {
                label: value.to(device=self.device, dtype=torch.float32)
                for label, value in self.count_data["exposures"].items()
            }
            count_matrices = {
                label: value.to(device=self.device, dtype=torch.float32)
                for label, value in self.count_data["count_matrices"].items()
            }
            n_totals = {
                label: value.to(device=self.device, dtype=torch.float32)
                for label, value in self.count_data["n_totals"].items()
            }
            loss_count, count_logs = self.count_lik.forward_with_logs(
                growth_steps=rollout.growth_steps.float(),
                logw_steps=rollout.logw_steps.float(),
                tau_steps=rollout.tau_steps.float(),
                exposures=exposures,
                count_matrices=count_matrices,
                n_totals=n_totals,
                checkpoint_indices=self.target_checkpoint_indices,
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
        logs = {**endpoint_logs, **count_logs}
        metrics = {
            "loss_total": float(loss_total.detach().cpu()),
            "loss_end": float(loss_end.detach().cpu()),
            "loss_weak": float(loss_weak.detach().cpu()),
            "loss_count": float(loss_count.detach().cpu()),
            "loss_reg": float(loss_reg.detach().cpu()),
            **_diagnostics_from_rollout(rollout),
        }
        pred_df = rollout_metrics_by_key_time(
            view,
            rollout,
            self.target_checkpoint_indices,
            view.source_keys,
            endpoint_logs=endpoint_logs,
        )
        return loss_total, logs, metrics, pred_df

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
            rollout, _, _, _ = self._rollout(
                self.view,
                n_particles=self.config.simulation.n_particles,
                seed=tc.seed + epoch,
                training=True,
            )
            loss, _, metrics, _ = self._loss_for_rollout(self.view, rollout, epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            optimizer.step()
            scheduler.step()
            if self.ema is not None:
                self.ema.update()

            val = self.evaluate(epoch=epoch, use_ema=False)
            metrics["val_endpoint_loss"] = val["metrics"].get("loss_end", math.nan)
            self.history.append(epoch, metrics)

            if metrics["val_endpoint_loss"] < self._best_val:
                self._best_val = metrics["val_endpoint_loss"]
                self._best_checkpoint_path = self.save_checkpoint("checkpoint_best.pt", epoch=epoch)
                if self.ema is not None:
                    self.ema.apply_shadow()
                    self.save_checkpoint("checkpoint_best_ema.pt", epoch=epoch)
                    self.ema.restore()

            if epoch % max(1, tc.log_every) == 0 or epoch == tc.epochs:
                self.history.to_dataframe().to_csv(self.output_dir / "training_history.csv", index=False)

        self.save_checkpoint("checkpoint_last.pt", epoch=tc.epochs)
        self.history.to_dataframe().to_csv(self.output_dir / "training_history.csv", index=False)
        return self.history

    @torch.no_grad()
    def evaluate(self, *, epoch: int = 0, use_ema: bool = False) -> dict[str, object]:
        view = self.val_view or self.view
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        try:
            self.model.eval()
            rollout, _, _, _ = self._rollout(
                view,
                n_particles=max(1, self.config.eval.n_eval_particles),
                seed=self.config.training.seed + 100_000 + epoch,
                training=False,
            )
            _, _, metrics, pred_df = self._loss_for_rollout(view, rollout, epoch=epoch)
            pred_df.to_csv(self.output_dir / "predicted_metrics_by_key_time.csv", index=False)
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
            "measure_keys": [str(key) for key in self.measure_keys],
            "embedding_ids": self.embedding_ids,
        }
        if self.ema is not None:
            payload["ema_state_dict"] = self.ema.state_dict()
        torch.save(payload, path)
        return path


__all__ = ["TrajectoryTrainer", "TrajectoryTrainingHistory"]
