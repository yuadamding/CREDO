from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
from torch import Tensor

from camfnd.data.contract import EndpointProblem, Key, PerturbSeqDynamicsData
from camfnd.models.full_coeff_nets import ControlAnchoredFullModel, FullCoefficientConfig
from camfnd.models.full_context_map import MeanFieldContextConfig
from camfnd.models.sinkhorn import normalized_geometry_loss, unbalanced_sinkhorn_divergence
from camfnd.simulation.full_joint_sim import FullJointSimulationResult, FullJointSimulator
from camfnd.simulation.single_screen_sim import LearnedSimulatorConfig


LossMode = Literal["uot", "normalized_only"]


@dataclass(frozen=True, slots=True)
class FullModelTrainConfig:
    embedding_dim: int = 4
    hidden_dim: int = 16
    depth: int = 1
    time_frequencies: int = 0
    context_dim: int = 1
    summary_dim: int = 8
    summary_hidden_dim: int = 16
    summary_depth: int = 1
    context_hidden_dim: int = 8
    context_depth: int = 1
    sigma_min: float = 0.02
    r_max: float = 2.0
    use_context: bool = True
    n_steps: int = 12
    particles_per_atom: int = 1
    seed: int = 29
    epochs: int = 60
    lr: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    epsilon: float = 0.08
    tau: float = 0.45
    sinkhorn_iters: int = 20
    reg_embedding: float = 1e-4
    reg_modulation: float = 1e-5
    reg_dispersion: float = 1e-5
    reg_network: float = 1e-6
    reg_context: float = 1e-6
    aux_mean_weight: float = 1.0
    aux_variance_weight: float = 5.0
    aux_mass_weight: float = 5.0
    aux_screen_delta_mean_weight: float = 10.0
    loss_mode: LossMode = "uot"
    device: str = "auto"
    dtype: str = "float64"

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self) -> None:
        for name in (
            "embedding_dim",
            "hidden_dim",
            "depth",
            "context_dim",
            "summary_dim",
            "summary_hidden_dim",
            "summary_depth",
            "context_hidden_dim",
            "context_depth",
            "n_steps",
            "particles_per_atom",
            "epochs",
            "sinkhorn_iters",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.time_frequencies < 0:
            raise ValueError("time_frequencies must be nonnegative.")
        if self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError("lr and grad_clip must be positive.")
        if self.epsilon <= 0 or self.tau <= 0:
            raise ValueError("epsilon and tau must be positive.")
        if self.sigma_min <= 0 or self.r_max <= 0:
            raise ValueError("sigma_min and r_max must be positive.")
        if self.loss_mode not in {"uot", "normalized_only"}:
            raise ValueError("Unsupported loss_mode.")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        for name in (
            "reg_embedding",
            "reg_modulation",
            "reg_dispersion",
            "reg_network",
            "reg_context",
            "aux_mean_weight",
            "aux_variance_weight",
            "aux_mass_weight",
            "aux_screen_delta_mean_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")

    def coefficient_config(self, *, latent_dim: int) -> FullCoefficientConfig:
        context_config = MeanFieldContextConfig(
            latent_dim=int(latent_dim),
            summary_dim=self.summary_dim,
            context_dim=self.context_dim,
            summary_hidden_dim=self.summary_hidden_dim,
            summary_depth=self.summary_depth,
            context_hidden_dim=self.context_hidden_dim,
            context_depth=self.context_depth,
            use_context=self.use_context,
        )
        return FullCoefficientConfig(
            latent_dim=int(latent_dim),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            time_frequencies=self.time_frequencies,
            context_dim=self.context_dim,
            sigma_min=self.sigma_min,
            r_max=self.r_max,
            use_context=self.use_context,
            context_config=context_config,
        )

    def simulator_config(self) -> LearnedSimulatorConfig:
        return LearnedSimulatorConfig(
            n_steps=self.n_steps,
            seed=self.seed,
            particles_per_atom=self.particles_per_atom,
            store_history=False,
            device=self.resolved_device,
            dtype=self.dtype,
        )


@dataclass(slots=True)
class FullModelTrainingResult:
    config: FullModelTrainConfig
    model: ControlAnchoredFullModel
    simulator: FullJointSimulator
    history: pd.DataFrame
    final_simulation: FullJointSimulationResult
    final_loss_table: pd.DataFrame


class FullEndpointLossComputer:
    def __init__(self, problem: EndpointProblem, config: FullModelTrainConfig) -> None:
        self.problem = problem
        self.config = config
        self.target_support: Dict[Key, Tensor] = {}
        self.target_weights: Dict[Key, Tensor] = {}
        self.target_mass: Dict[Key, Tensor] = {}
        self.target_mean: Dict[Key, Tensor] = {}
        self.target_var: Dict[Key, Tensor] = {}
        self.delta_mean_targets: Dict[str, Tensor] = {}
        dtype = config.simulator_config().torch_dtype
        device = torch.device(config.resolved_device)
        for key, measure in problem.terminal.items():
            support = torch.as_tensor(measure.support, dtype=dtype, device=device)
            weights = torch.as_tensor(measure.weights, dtype=dtype, device=device)
            norm = weights / weights.sum()
            mean = (norm[:, None] * support).sum(dim=0)
            centered = support - mean[None, :]
            var = torch.trace(centered.T @ (centered * norm[:, None]))
            self.target_support[key] = support
            self.target_weights[key] = weights
            self.target_mass[key] = weights.sum()
            self.target_mean[key] = mean
            self.target_var[key] = var

        sample_ids = sorted({key[0] for key in problem.keys if isinstance(key, tuple)})
        if len(sample_ids) == 2:
            s1, s2 = sample_ids
            for pid in problem.catalog.perturbation_ids:
                k1 = (s1, pid)
                k2 = (s2, pid)
                if k1 in self.target_mean and k2 in self.target_mean:
                    self.delta_mean_targets[pid] = self.target_mean[k2] - self.target_mean[k1]

    def endpoint_loss_for_key(self, pred_state, key: Key) -> Tensor:
        pred_support = pred_state.z
        pred_weights = pred_state.atom_weights()
        target_support = self.target_support[key]
        target_weights = self.target_weights[key]
        if self.config.loss_mode == "uot":
            return unbalanced_sinkhorn_divergence(
                pred_support,
                pred_weights,
                target_support,
                target_weights,
                epsilon=self.config.epsilon,
                tau=self.config.tau,
                max_iters=self.config.sinkhorn_iters,
            )
        return normalized_geometry_loss(
            pred_support,
            pred_weights,
            target_support,
            target_weights,
            epsilon=self.config.epsilon,
            tau=self.config.tau,
            max_iters=self.config.sinkhorn_iters,
        )

    def compute(self, terminal_particles: Dict[Key, object]) -> tuple[Tensor, Tensor, Tensor, pd.DataFrame]:
        rows = []
        endpoint_losses = []
        aux_losses = []
        pred_mean_by_key: Dict[Key, Tensor] = {}
        for key in self.problem.keys:
            particle_state = terminal_particles[key]
            endpoint_loss = self.endpoint_loss_for_key(particle_state, key)
            pred_mass = particle_state.total_mass()
            pred_mean = particle_state.mean()
            pred_var = particle_state.variance_trace()
            pred_mean_by_key[key] = pred_mean

            mass_aux = (pred_mass - self.target_mass[key]) ** 2
            mean_aux = ((pred_mean - self.target_mean[key]) ** 2).mean()
            var_aux = (pred_var - self.target_var[key]) ** 2
            aux_loss = (
                self.config.aux_mass_weight * mass_aux
                + self.config.aux_mean_weight * mean_aux
                + self.config.aux_variance_weight * var_aux
            )
            endpoint_losses.append(endpoint_loss)
            aux_losses.append(aux_loss)

            row = {
                "key": key,
                "sample_id": particle_state.sample_id,
                "perturbation_id": particle_state.perturbation_id,
                "endpoint_loss": float(endpoint_loss.detach().cpu()),
                "aux_loss": float(aux_loss.detach().cpu()),
                "pred_mass": float(pred_mass.detach().cpu()),
                "pred_var_trace": float(pred_var.detach().cpu()),
                "target_mass": float(self.target_mass[key].detach().cpu()),
                "target_var_trace": float(self.target_var[key].detach().cpu()),
            }
            pred_mean_cpu = pred_mean.detach().cpu().reshape(-1)
            target_mean_cpu = self.target_mean[key].detach().cpu().reshape(-1)
            for dim_idx, value in enumerate(pred_mean_cpu):
                row[f"pred_mean_{dim_idx}"] = float(value)
            for dim_idx, value in enumerate(target_mean_cpu):
                row[f"target_mean_{dim_idx}"] = float(value)
            rows.append(row)

        endpoint_total = torch.stack(endpoint_losses).mean()
        aux_total = torch.stack(aux_losses).mean()

        screen_delta_losses = []
        if self.delta_mean_targets and self.config.aux_screen_delta_mean_weight > 0:
            sample_ids = sorted({key[0] for key in self.problem.keys if isinstance(key, tuple)})
            if len(sample_ids) == 2:
                s1, s2 = sample_ids
                for pid, target_delta in self.delta_mean_targets.items():
                    k1 = (s1, pid)
                    k2 = (s2, pid)
                    if k1 in pred_mean_by_key and k2 in pred_mean_by_key:
                        pred_delta = pred_mean_by_key[k2] - pred_mean_by_key[k1]
                        screen_delta_losses.append(((pred_delta - target_delta) ** 2).mean())

        if screen_delta_losses:
            screen_delta_total = torch.stack(screen_delta_losses).mean()
        else:
            screen_delta_total = torch.zeros((), dtype=endpoint_total.dtype, device=endpoint_total.device)

        table = pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)
        return endpoint_total, aux_total, screen_delta_total, table


def train_full_model(
    dataset: PerturbSeqDynamicsData,
    *,
    config: Optional[FullModelTrainConfig] = None,
) -> FullModelTrainingResult:
    config = config or FullModelTrainConfig()
    config.validate()
    torch.set_default_dtype(config.simulator_config().torch_dtype)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    dataset.validate()
    problem = dataset.to_endpoint_problem(by_sample=True)
    if dataset.truth is not None and dataset.truth.simulator_config:
        problem.metadata = {**problem.metadata, "simulator_config": dict(dataset.truth.simulator_config)}

    latent_dim = int(problem.metadata.get("latent_dim", dataset.latent_dim))
    simulator = FullJointSimulator(problem, config.simulator_config())
    model = ControlAnchoredFullModel(problem.catalog, config.coefficient_config(latent_dim=latent_dim)).to(
        config.resolved_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_computer = FullEndpointLossComputer(problem, config)

    history_rows: List[dict] = []
    best_state = None
    best_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sim_result = simulator.run(model)
        endpoint_loss, aux_loss, screen_delta_loss, _ = loss_computer.compute(sim_result.terminal_particles)
        reg = model.regularization_terms()
        reg_total = (
            config.reg_embedding * reg["emb"]
            + config.reg_modulation * reg["mod"]
            + config.reg_dispersion * reg["disp"]
            + config.reg_network * reg["nn"]
            + config.reg_context * reg["context"]
        )
        total_loss = endpoint_loss + aux_loss + config.aux_screen_delta_mean_weight * screen_delta_loss + reg_total
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        total_loss_value = float(total_loss.detach().cpu())
        history_rows.append(
            {
                "epoch": epoch + 1,
                "total_loss": total_loss_value,
                "endpoint_loss": float(endpoint_loss.detach().cpu()),
                "aux_loss": float(aux_loss.detach().cpu()),
                "screen_delta_loss": float(screen_delta_loss.detach().cpu()),
                "reg_total": float(reg_total.detach().cpu()),
                "control_anchor_exact": bool(model.control_anchor_is_exact()),
            }
        )
        if total_loss_value < best_loss:
            best_loss = total_loss_value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    final_simulation = simulator.run(model)
    _, _, _, final_table = loss_computer.compute(final_simulation.terminal_particles)
    history = pd.DataFrame(history_rows)
    return FullModelTrainingResult(
        config=config,
        model=model,
        simulator=simulator,
        history=history,
        final_simulation=final_simulation,
        final_loss_table=final_table,
    )
