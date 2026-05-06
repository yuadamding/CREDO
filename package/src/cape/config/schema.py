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


class SimulationConfig(BaseModel):
    n_particles: int = 128
    n_steps: int = 24
    resample_ess_threshold: float = 0.5
    enable_resampling: bool = False
    store_history: bool = False


class TrainingConfig(BaseModel):
    optimizer: Literal["adamw", "adam"] = "adamw"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    lr_net: float = 3e-4
    lr_embed: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    lambda_end: float = 1.0
    lambda_count: float = 0.3
    lambda_weak: float = 0.1
    lambda_aux: float = 0.05
    lambda_reg_embed: float = 1e-4
    lambda_reg_growth_bias: float = 1e-4
    lambda_reg_net: float = 1e-4
    lambda_reg_diffusion: float = 1e-4
    training_schedule: Literal["joint", "staged"] = "staged"
    stage_c_epochs: int = 150
    stage_d_epochs: int = 150
    max_active_perturbations: int = 0
    control_ref_warmup_epochs: int = 150
    seed: int = 0
    epochs: int = 300
    early_stop_patience: int = 50
    log_every: int = 10
    checkpoint_every: int = 50
    stage: Literal["A", "B", "C", "D", "E", "F", "all"] = "all"

    # UOT parameters
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
        if self.lambda_count > 0 and self.max_active_perturbations > 0:
            raise ValueError(
                "lambda_count > 0 is incompatible with perturbation chunking "
                "(max_active_perturbations > 0)."
            )
        return self


class EvalConfig(BaseModel):
    n_seeds: int = 3
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
