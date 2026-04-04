"""Pydantic configuration schemas for cape.

Every training run is reconstructable from one of these config objects.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    min_cells_p4: int = 20
    min_cells_p60: int = 20
    pooled_state: bool = True
    train_level: Literal["gene", "sgrna"] = "gene"
    min_total_mass: Optional[float] = None


class LatentConfig(BaseModel):
    dim: int = 16
    whiten: bool = True


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
    ecological_drift: bool = False
    ecological_growth: bool = False
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
    lambda_reg_net: float = 1e-4
    lambda_reg_diffusion: float = 1e-4
    training_schedule: Literal["joint", "staged"] = "staged"
    stage_c_epochs: int = 150
    stage_d_epochs: int = 150
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
