"""Pydantic configuration schemas for CREDO.

Every training run is reconstructable from one of these config objects.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    min_cells_p4: int = 20
    min_cells_p60: int = 20
    pooled_state: bool = True
    train_level: Literal["gene", "sgrna"] = "gene"
    min_total_mass: Optional[float] = None
    mass_value_col: Optional[str] = None
    mass_mode: Literal["auto", "count", "per_cell_contribution", "group_total"] = "auto"
    mass_scope: Literal["full_obs", "subset_only"] = "subset_only"


class VAEConfig(BaseModel):
    """Hyperparameters for the expression VAE latent backend."""
    hidden_dim: int = 512
    depth: int = 2
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    kl_weight: float = 1e-3
    kl_warmup_epochs: int = 20
    val_frac: float = 0.1
    early_stop_patience: int = 15
    grad_clip: float = 1.0
    seed: int = 0
    layer: Optional[str] = None
    use_raw: bool = False
    n_genes: int = 2000
    gene_mask_col: Optional[str] = None
    allow_empty_gene_mask_fallback: bool = False
    target_sum: float = 1e4
    strict_layer: bool = True
    strict_counts: bool = True
    expression_workers: int = 0
    expression_chunk_size: int = 1024
    batch_aware_hvg: bool = True
    hvg_batch_col: str = "Library"
    hvg_time_col: str = "Time point"
    hvg_min_cells_per_batch: int = 256
    allow_full_gene_scan: bool = False
    preload_dense_max_gb: float = 4.0
    reuse_artifact: bool = True
    use_amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"


class LatentConfig(BaseModel):
    source: Literal["pca", "vae"] = "pca"
    key: Optional[str] = "X_pca"
    dim: int = 16
    whiten: bool = True
    vae: VAEConfig = Field(default_factory=VAEConfig)

    @model_validator(mode="after")
    def _validate_backend(self) -> "LatentConfig":
        if self.source == "vae":
            if self.key not in (None, "", "X_vae"):
                raise ValueError(
                    "latent.source='vae' is incompatible with latent.key="
                    f"{self.key!r}. Use 'X_vae' or omit the key."
                )
            self.key = "X_vae"
        elif not self.key:
            raise ValueError("latent.key must be provided when latent.source='pca'.")
        return self


class ModelConfig(BaseModel):
    embedding_dim: int = 8
    n_programs: int = 8
    mediator_dim: int = 8
    hidden_dim: int = 128
    depth: int = 3
    activation_checkpointing: bool = False
    time_frequencies: int = 4
    sigma_min: float = 1e-3
    r_max: float = 3.0
    ecological_growth: bool = True
    use_growth_intercept: bool = True
    n_payoff_ranks: int = 4
    control_mode: Literal["anchored", "free", "soft_ref"] = "soft_ref"
    control_ref_penalty: float = 5e-4
    context_kind: Literal["mlp", "transformer", "causal_attention"] = "mlp"
    transformer_token_dim: int = 64
    transformer_heads: int = 4
    transformer_within_layers: int = 1
    transformer_cross_layers: int = 1
    transformer_inducing: int = 8
    transformer_dropout: float = 0.05
    mass_attention_temperature: float = 0.5
    transformer_growth_only: bool = True
    causal_token_dim: int = 64
    causal_heads: int = 4
    causal_n_mediators: int = 12
    causal_dropout: float = 0.05
    causal_mass_attention_temperature: float = 0.5
    causal_growth_only: bool = True
    causal_sparse_edges: bool = True
    causal_residual_policy: Literal["edges_only", "tokens_and_edges"] = "edges_only"

    @model_validator(mode="after")
    def _validate_context_backends(self) -> "ModelConfig":
        if self.transformer_token_dim < 1:
            raise ValueError("transformer_token_dim must be >= 1.")
        if self.transformer_heads < 1:
            raise ValueError("transformer_heads must be >= 1.")
        if self.transformer_token_dim % self.transformer_heads != 0:
            raise ValueError("transformer_token_dim must be divisible by transformer_heads.")
        if self.transformer_within_layers < 1:
            raise ValueError("transformer_within_layers must be >= 1.")
        if self.transformer_cross_layers < 1:
            raise ValueError("transformer_cross_layers must be >= 1.")
        if self.transformer_inducing < 1:
            raise ValueError("transformer_inducing must be >= 1.")
        if not 0.0 <= self.transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1).")
        if self.mass_attention_temperature < 0:
            raise ValueError("mass_attention_temperature must be >= 0.")
        if self.causal_token_dim < 1:
            raise ValueError("causal_token_dim must be >= 1.")
        if self.causal_heads < 1:
            raise ValueError("causal_heads must be >= 1.")
        if self.causal_token_dim % self.causal_heads != 0:
            raise ValueError("causal_token_dim must be divisible by causal_heads.")
        if self.causal_n_mediators < 1:
            raise ValueError("causal_n_mediators must be >= 1.")
        if not 0.0 <= self.causal_dropout < 1.0:
            raise ValueError("causal_dropout must be in [0, 1).")
        if self.causal_mass_attention_temperature < 0:
            raise ValueError("causal_mass_attention_temperature must be >= 0.")
        if self.context_kind == "causal_attention" and not self.causal_sparse_edges:
            raise ValueError(
                "causal_attention requires causal_sparse_edges=True. "
                "Dense mediator attention is not intervention-addressable CEA."
            )
        return self


class SimulationConfig(BaseModel):
    n_particles: int = 128
    n_steps: int = 24
    store_history: bool = False


class TrainingConfig(BaseModel):
    optimizer: Literal["adamw", "adam"] = "adamw"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    lr_net: float = 3e-4
    lr_embed: float = 1e-3
    lr_transformer: float = 5e-5
    lr_causal_attention: float = 5e-5
    weight_decay: float = 1e-6
    transformer_weight_decay: float = 1e-4
    causal_attention_weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lambda_end: float = 1.0
    lambda_count: float = 0.3
    lambda_weak: float = 0.1
    lambda_aux: float = 0.05
    lambda_reg_embed: float = 1e-4
    lambda_reg_growth_bias: float = 1e-4
    lambda_reg_net: float = 1e-4
    lambda_reg_diffusion: float = 1e-4
    lambda_causal_ctrl_edge: float = 1e-3
    lambda_causal_guide: float = 0.0
    lambda_causal_sparse: float = 1e-4
    lambda_causal_orth: float = 1e-4
    lambda_causal_ctx_smooth: float = 1e-4
    causal_loss_start_epoch: int = 100
    causal_loss_ramp_epochs: int = 200
    training_schedule: Literal["joint", "staged"] = "staged"
    stage_c_epochs: int = 150
    stage_d_epochs: int = 150
    max_active_perturbations: int = 0
    global_context_batching: Literal["full_context_cache", "error", "local_ablation"] = "full_context_cache"
    control_ref_warmup_epochs: int = 150
    seed: int = 0
    epochs: int = 300
    early_stop_patience: int = 50
    log_every: int = 10
    checkpoint_every: int = 50
    divergence_factor: float = 50.0
    divergence_patience: int = 2
    divergence_min_epochs: int = 25
    ess_warn_frac: float = 0.20
    ess_fail_frac: float = 0.05
    ess_claim_grade_min_frac: float = 0.10
    ess_max_weight_frac_fail: float = 0.50
    stage: Literal["A", "B", "C", "D", "E", "F", "all"] = "all"

    # Endpoint geometry-plus-log-mass loss parameters
    sinkhorn_epsilon: float = 0.1
    sinkhorn_tau: float = 1.0
    sinkhorn_max_iter: int = 100

    # Weak-form parameters
    n_test_functions: int = 32
    test_function_bandwidth: float = 1.0

    @model_validator(mode="after")
    def _validate_training_compatibility(self) -> "TrainingConfig":
        if self.max_active_perturbations < 0:
            raise ValueError("max_active_perturbations must be >= 0.")
        if self.global_context_batching == "local_ablation":
            raise ValueError(
                "global_context_batching='local_ablation' is reserved for explicit "
                "diagnostics and is not implemented in the claim-grade trainer."
            )
        if self.lr_transformer <= 0:
            raise ValueError("lr_transformer must be > 0.")
        if self.lr_causal_attention <= 0:
            raise ValueError("lr_causal_attention must be > 0.")
        if self.transformer_weight_decay < 0:
            raise ValueError("transformer_weight_decay must be >= 0.")
        if self.causal_attention_weight_decay < 0:
            raise ValueError("causal_attention_weight_decay must be >= 0.")
        causal_lambdas = {
            "lambda_causal_ctrl_edge": self.lambda_causal_ctrl_edge,
            "lambda_causal_guide": self.lambda_causal_guide,
            "lambda_causal_sparse": self.lambda_causal_sparse,
            "lambda_causal_orth": self.lambda_causal_orth,
            "lambda_causal_ctx_smooth": self.lambda_causal_ctx_smooth,
        }
        for name, value in causal_lambdas.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
        if self.causal_loss_start_epoch < 0:
            raise ValueError("causal_loss_start_epoch must be >= 0.")
        if self.causal_loss_ramp_epochs < 1:
            raise ValueError("causal_loss_ramp_epochs must be >= 1.")
        if self.divergence_factor <= 1:
            raise ValueError("divergence_factor must be > 1.")
        if self.divergence_patience < 1:
            raise ValueError("divergence_patience must be >= 1.")
        if self.divergence_min_epochs < 0:
            raise ValueError("divergence_min_epochs must be >= 0.")
        ess_thresholds = {
            "ess_warn_frac": self.ess_warn_frac,
            "ess_fail_frac": self.ess_fail_frac,
            "ess_claim_grade_min_frac": self.ess_claim_grade_min_frac,
            "ess_max_weight_frac_fail": self.ess_max_weight_frac_fail,
        }
        for name, value in ess_thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.ess_fail_frac > self.ess_warn_frac:
            raise ValueError("ess_fail_frac must be <= ess_warn_frac.")
        if self.ess_fail_frac > self.ess_claim_grade_min_frac:
            raise ValueError("ess_fail_frac must be <= ess_claim_grade_min_frac.")
        if self.ess_claim_grade_min_frac > self.ess_warn_frac:
            raise ValueError("ess_claim_grade_min_frac must be <= ess_warn_frac.")
        return self


class TrajectoryTrainingConfig(BaseModel):
    source_label: str = "90m"
    target_labels: list[str] = Field(default_factory=lambda: ["6h", "10h"])
    trajectory_mode: Literal["full_start", "full_plus_teacher", "pairwise"] = "full_start"
    steps_per_interval: int = 12
    endpoint_time_weights: dict[str, float] = Field(default_factory=dict)
    normalize_time_weights: bool = True
    teacher_forced_weight: float = 0.0
    max_active_measure_keys: int = 0
    context_batch_mode: Literal["all_keys", "batch_only"] = "all_keys"
    sparse_missing: Literal["mask", "error"] = "mask"
    key_mode: Literal["pooled", "sample_aware"] = "sample_aware"
    validation_source: Literal["train", "heldout", "all"] = "heldout"
    save_rollouts: bool = False
    save_particles_every: int = 0

    @model_validator(mode="after")
    def _validate_trajectory_training(self) -> "TrajectoryTrainingConfig":
        if not self.source_label:
            raise ValueError("source_label must not be empty.")
        if not self.target_labels:
            raise ValueError("target_labels must not be empty.")
        if self.steps_per_interval < 1:
            raise ValueError("steps_per_interval must be >= 1.")
        if self.teacher_forced_weight < 0:
            raise ValueError("teacher_forced_weight must be >= 0.")
        if self.trajectory_mode != "full_start" and self.teacher_forced_weight == 0:
            # Keep this permissive for CLI experimentation but explicit in config.
            pass
        if self.max_active_measure_keys < 0:
            raise ValueError("max_active_measure_keys must be >= 0.")
        return self


class EvalConfig(BaseModel):
    n_seeds: int = 3
    n_eval_particles: int = 384
    n_counterfactual_particles: int = 512


class RunConfig(BaseModel):
    """Top-level config that captures everything needed to reproduce a run."""
    run_id: str = "run"
    git_sha: Optional[str] = None
    data_id: Optional[str] = None
    latent_id: Optional[str] = None
    output_dir: str = "outputs"
    device: str = "auto"
    multi_gpu_devices: list[str] = Field(default_factory=list)

    data: DataConfig = Field(default_factory=DataConfig)
    latent: LatentConfig = Field(default_factory=LatentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    trajectory_training: TrajectoryTrainingConfig = Field(default_factory=TrajectoryTrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def _validate_run(self) -> "RunConfig":
        if self.training.lambda_count > 0 and len(self.multi_gpu_devices) > 1:
            raise ValueError(
                "lambda_count > 0 is not supported with the current multi-GPU single-model path."
            )
        return self

    def resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def resolve_training_devices(self) -> list[str]:
        import torch

        def _normalize(device: str) -> str:
            dev = str(device).strip()
            if not dev:
                return dev
            if dev.isdigit():
                return f"cuda:{dev}"
            if dev == "cuda":
                return "cuda:0"
            return dev

        if self.multi_gpu_devices:
            devices = [_normalize(device) for device in self.multi_gpu_devices if str(device).strip()]
            if not devices:
                return [self.resolve_device()]
            if not torch.cuda.is_available():
                return ["cpu"]
            count = torch.cuda.device_count()
            valid: list[str] = []
            for device in devices:
                if device.startswith("cuda:"):
                    try:
                        idx = int(device.split(":", 1)[1])
                    except ValueError:
                        continue
                    if idx < count:
                        valid.append(device)
                elif device == "cpu":
                    valid.append(device)
            return valid or [self.resolve_device()]
        return [self.resolve_device()]
