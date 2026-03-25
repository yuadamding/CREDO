from __future__ import annotations

"""Training for the single-screen control-anchored model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
from torch import Tensor

from camfnd.models.coeff_nets import Stage1CoefficientConfig, ControlAnchoredStage1Model
from camfnd.data.contract import EndpointProblem, Key, PerturbSeqDynamicsData
from camfnd.models.sinkhorn import normalized_geometry_loss, unbalanced_sinkhorn_divergence
from camfnd.simulation.single_screen_sim import LearnedSimulationResult, LearnedSimulatorConfig, LearnedStage1Simulator


LossMode = Literal["uot", "normalized_only"]


def _inv_softplus(y: float) -> float:
    y = float(max(y, 1e-8))
    return float(math.log(math.expm1(y)))


def _artanh_clipped(x: float, limit: float = 0.98) -> float:
    x = max(min(float(x), limit), -limit)
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


@dataclass(frozen=True, slots=True)
class Stage1TrainConfig:
    embedding_dim: int = 3
    hidden_dim: int = 16
    depth: int = 1
    time_frequencies: int = 0
    sigma_min: float = 0.02
    r_max: float = 2.0
    shared_diffusion: bool = False
    use_growth: bool = True
    n_steps: int = 16
    particles_per_atom: int = 1
    seed: int = 17
    epochs: int = 60
    lr: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    epsilon: float = 0.08
    tau: float = 0.45
    sinkhorn_iters: int = 20
    reg_embedding: float = 1e-4
    reg_modulation: float = 1e-5
    reg_dispersion: float = 1e-5
    reg_network: float = 1e-6
    aux_mean_weight: float = 1.0
    aux_variance_weight: float = 10.0
    aux_mass_weight: float = 5.0
    loss_mode: LossMode = "uot"
    device: str = "auto"
    dtype: str = "float64"

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive.")
        if self.epsilon <= 0 or self.tau <= 0:
            raise ValueError("epsilon and tau must be positive.")
        if self.sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive.")
        if self.loss_mode not in {"uot", "normalized_only"}:
            raise ValueError("Unsupported loss_mode.")
        for name in ("aux_mean_weight", "aux_variance_weight", "aux_mass_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")

    def coefficient_config(self) -> Stage1CoefficientConfig:
        return Stage1CoefficientConfig(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            time_frequencies=self.time_frequencies,
            sigma_min=self.sigma_min,
            r_max=self.r_max,
            shared_diffusion=self.shared_diffusion,
            use_growth=self.use_growth,
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
class Stage1TrainingResult:
    config: Stage1TrainConfig
    model: ControlAnchoredStage1Model
    simulator: LearnedStage1Simulator
    history: pd.DataFrame
    final_simulation: LearnedSimulationResult
    final_loss_table: pd.DataFrame


class EndpointLossComputer:
    def __init__(self, problem: EndpointProblem, config: Stage1TrainConfig) -> None:
        self.problem = problem
        self.config = config
        self.target_support: Dict[Key, Tensor] = {}
        self.target_weights: Dict[Key, Tensor] = {}
        self.target_mass: Dict[Key, Tensor] = {}
        self.target_mean: Dict[Key, Tensor] = {}
        self.target_var: Dict[Key, Tensor] = {}
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
            self.target_mean[key] = mean[0]
            self.target_var[key] = var

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

    def compute(self, terminal_particles: Dict[Key, object]) -> tuple[Tensor, Tensor, pd.DataFrame]:
        rows = []
        endpoint_losses = []
        aux_losses = []
        for key in self.problem.keys:
            particle_state = terminal_particles[key]
            endpoint_loss = self.endpoint_loss_for_key(particle_state, key)
            pred_mass = particle_state.total_mass()
            pred_mean = particle_state.mean()[0]
            pred_var = particle_state.variance_trace()
            mass_aux = (pred_mass - self.target_mass[key]) ** 2
            mean_aux = (pred_mean - self.target_mean[key]) ** 2
            var_aux = (pred_var - self.target_var[key]) ** 2
            aux_loss = (
                self.config.aux_mass_weight * mass_aux
                + self.config.aux_mean_weight * mean_aux
                + self.config.aux_variance_weight * var_aux
            )
            endpoint_losses.append(endpoint_loss)
            aux_losses.append(aux_loss)
            rows.append(
                {
                    "key": key,
                    "sample_id": particle_state.sample_id,
                    "perturbation_id": particle_state.perturbation_id,
                    "endpoint_loss": float(endpoint_loss.detach().cpu()),
                    "aux_loss": float(aux_loss.detach().cpu()),
                    "pred_mass": float(pred_mass.detach().cpu()),
                    "pred_mean_0": float(pred_mean.detach().cpu()),
                    "pred_var_trace": float(pred_var.detach().cpu()),
                    "target_mass": float(self.target_mass[key].detach().cpu()),
                    "target_mean_0": float(self.target_mean[key].detach().cpu()),
                    "target_var_trace": float(self.target_var[key].detach().cpu()),
                }
            )
        endpoint_total = torch.stack(endpoint_losses).mean()
        aux_total = torch.stack(aux_losses).mean()
        table = pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)
        return endpoint_total, aux_total, table


def initialize_model_from_stage1_moments(
    model: ControlAnchoredStage1Model,
    problem: EndpointProblem,
    config: Stage1TrainConfig,
) -> None:
    """Moment-based warm start for the Stage-I structured reference model."""

    if len(problem.keys) != len(problem.catalog.perturbation_ids):
        return

    kappa_target = 2.0
    if isinstance(problem.metadata, dict):
        maybe = problem.metadata.get("simulator_config", {}).get("kappa") if problem.metadata.get("simulator_config") else None
        if maybe is not None:
            kappa_target = float(maybe)

    with torch.no_grad():
        model.kappa_raw.fill_(_inv_softplus(kappa_target))

        control_key = next(key for key in problem.keys if (key[1] if isinstance(key, tuple) else key) == "ctrl")
        init_ctrl = problem.initial[control_key]
        term_ctrl = problem.terminal[control_key]
        m0_ctrl = float(init_ctrl.mean()[0])
        v0_ctrl = float(init_ctrl.variance_trace())
        m1_ctrl = float(term_ctrl.mean()[0])
        v1_ctrl = float(term_ctrl.variance_trace())
        exp_term = math.exp(-kappa_target)
        denom = max(1.0 - exp_term, 1e-6)
        theta_ctrl = (m1_ctrl - m0_ctrl * exp_term) / denom
        sigma_ctrl_sq = max((2.0 * kappa_target * (v1_ctrl - v0_ctrl * math.exp(-2.0 * kappa_target))) / max(1.0 - math.exp(-2.0 * kappa_target), 1e-6), 1e-8)
        sigma_ctrl = math.sqrt(max(sigma_ctrl_sq, 1e-8))
        rho_ctrl = math.log(term_ctrl.total_mass / init_ctrl.total_mass)

        model.theta_field.baseline.fill_(theta_ctrl)
        model.sigma_field.baseline.fill_(_inv_softplus(max(sigma_ctrl - config.sigma_min, 1e-6)))
        if config.use_growth and config.loss_mode == "uot":
            model.growth_field.baseline.fill_(_artanh_clipped(rho_ctrl / config.r_max))
        else:
            model.growth_field.baseline.zero_()

        model.theta_field.modulation.zero_()
        model.sigma_field.modulation.zero_()
        model.growth_field.modulation.zero_()

        for perturbation_id in model.embedding_store.non_control_ids:
            key = next(key for key in problem.keys if (key[1] if isinstance(key, tuple) else key) == perturbation_id)
            init_meas = problem.initial[key]
            term_meas = problem.terminal[key]
            m0 = float(init_meas.mean()[0])
            v0 = float(init_meas.variance_trace())
            m1 = float(term_meas.mean()[0])
            v1 = float(term_meas.variance_trace())
            theta = (m1 - m0 * exp_term) / denom
            sigma_sq = max((2.0 * kappa_target * (v1 - v0 * math.exp(-2.0 * kappa_target))) / max(1.0 - math.exp(-2.0 * kappa_target), 1e-6), 1e-8)
            sigma = math.sqrt(max(sigma_sq, 1e-8))
            rho = math.log(term_meas.total_mass / init_meas.total_mass)

            emb = model.embedding_store.forward_one(perturbation_id).detach().cpu()
            active_index = int(torch.argmax(torch.abs(emb)).item())
            model.theta_field.modulation[0, active_index] = theta - theta_ctrl
            if not config.shared_diffusion:
                model.sigma_field.modulation[0, active_index] = _inv_softplus(max(sigma - config.sigma_min, 1e-6)) - float(model.sigma_field.baseline[0, 0])
            if config.use_growth and config.loss_mode == "uot":
                target_raw = _artanh_clipped(rho / config.r_max)
                model.growth_field.modulation[0, active_index] = target_raw - float(model.growth_field.baseline[0, 0])


def train_stage1_model(
    dataset: PerturbSeqDynamicsData,
    *,
    config: Optional[Stage1TrainConfig] = None,
) -> Stage1TrainingResult:
    config = config or Stage1TrainConfig()
    config.validate()
    torch.set_default_dtype(config.simulator_config().torch_dtype)

    dataset.validate()
    problem = dataset.to_endpoint_problem(by_sample=True)
    if dataset.truth is not None and dataset.truth.simulator_config:
        problem.metadata = {**problem.metadata, "simulator_config": dict(dataset.truth.simulator_config)}

    simulator = LearnedStage1Simulator(problem, config.simulator_config())
    model = ControlAnchoredStage1Model(problem.catalog, config.coefficient_config()).to(config.resolved_device)
    initialize_model_from_stage1_moments(model, problem, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_computer = EndpointLossComputer(problem, config)

    history_rows: List[dict] = []
    best_state = None
    best_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sim_result = simulator.run(model)
        endpoint_loss, aux_loss, _ = loss_computer.compute(sim_result.terminal_particles)
        reg = model.regularization_terms()
        reg_total = (
            config.reg_embedding * reg["emb"]
            + config.reg_modulation * reg["mod"]
            + config.reg_dispersion * reg["disp"]
            + config.reg_network * reg["nn"]
        )
        total_loss = endpoint_loss + aux_loss + reg_total
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
                "reg_total": float(reg_total.detach().cpu()),
                "reg_emb": float(reg["emb"].detach().cpu()),
                "reg_mod": float(reg["mod"].detach().cpu()),
                "reg_disp": float(reg["disp"].detach().cpu()),
                "reg_nn": float(reg["nn"].detach().cpu()),
            }
        )
        if total_loss_value < best_loss:
            best_loss = total_loss_value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    final_simulation = simulator.run(model)
    _, _, final_table = loss_computer.compute(final_simulation.terminal_particles)
    history = pd.DataFrame(history_rows)
    return Stage1TrainingResult(
        config=config,
        model=model,
        simulator=simulator,
        history=history,
        final_simulation=final_simulation,
        final_loss_table=final_table,
    )


# Backward-compatible alias
train_step3_model = train_stage1_model

# Semantic aliases for the clearer software-facing API.
SingleScreenTrainConfig = Stage1TrainConfig
SingleScreenTrainingResult = Stage1TrainingResult
initialize_single_screen_model_from_moments = initialize_model_from_stage1_moments
train_single_screen_model = train_stage1_model
