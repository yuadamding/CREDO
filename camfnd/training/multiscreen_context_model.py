from __future__ import annotations

"""Training for the multi-screen context-aware model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch import Tensor

from camfnd.models.coeff_nets import Stage2CoefficientConfig, ControlAnchoredStage2Model
from camfnd.data.contract import EndpointProblem, Key, PerturbSeqDynamicsData
from camfnd.models.sinkhorn import normalized_geometry_loss, unbalanced_sinkhorn_divergence
from camfnd.simulation.single_screen_sim import LearnedSimulatorConfig
from camfnd.simulation.multiscreen_context_sim import LearnedStage2JointSimulator, LearnedStage2SimulationResult


LossMode = str  # 'uot' or 'normalized_only'


def _inv_softplus(y: float) -> float:
    y = float(max(y, 1e-8))
    return float(math.log(math.expm1(y)))


def _artanh_clipped(x: float, limit: float = 0.98) -> float:
    x = max(min(float(x), limit), -limit)
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


@dataclass(frozen=True, slots=True)
class Stage2TrainConfig:
    embedding_dim: int = 4
    sigma_min: float = 0.02
    r_max: float = 2.0
    shared_diffusion: bool = False
    use_growth: bool = True
    use_context: bool = True
    n_steps: int = 12
    particles_per_atom: int = 1
    seed: int = 17
    epochs: int = 40
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
    reg_context: float = 1e-6
    aux_mean_weight: float = 1.0
    aux_variance_weight: float = 5.0
    aux_mass_weight: float = 5.0
    aux_screen_delta_mean_weight: float = 10.0
    loss_mode: LossMode = 'uot'
    device: str = 'auto'
    dtype: str = 'float64'

    @property
    def resolved_device(self) -> str:
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device

    def validate(self) -> None:
        if self.epochs <= 0 or self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError('epochs, lr, and grad_clip must be positive.')
        if self.epsilon <= 0 or self.tau <= 0 or self.sinkhorn_iters <= 0:
            raise ValueError('epsilon, tau, and sinkhorn_iters must be positive.')
        if self.loss_mode not in {'uot', 'normalized_only'}:
            raise ValueError('Unsupported loss_mode.')
        for name in ('aux_mean_weight', 'aux_variance_weight', 'aux_mass_weight', 'aux_screen_delta_mean_weight'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative.')

    def coefficient_config(self) -> Stage2CoefficientConfig:
        return Stage2CoefficientConfig(
            embedding_dim=self.embedding_dim,
            sigma_min=self.sigma_min,
            r_max=self.r_max,
            shared_diffusion=self.shared_diffusion,
            use_growth=self.use_growth,
            use_context=self.use_context,
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
class Stage2TrainingResult:
    config: Stage2TrainConfig
    model: ControlAnchoredStage2Model
    simulator: LearnedStage2JointSimulator
    history: pd.DataFrame
    final_simulation: LearnedStage2SimulationResult
    final_loss_table: pd.DataFrame


class EndpointLossComputer:
    def __init__(self, problem: EndpointProblem, config: Stage2TrainConfig) -> None:
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
            self.target_mean[key] = mean[0]
            self.target_var[key] = var
        sample_ids = sorted({key[0] for key in problem.keys if isinstance(key, tuple)})
        if len(sample_ids) == 2:
            s1, s2 = sample_ids
            for pid in problem.catalog.perturbation_ids:
                k1 = (s1, pid)
                k2 = (s2, pid)
                self.delta_mean_targets[pid] = self.target_mean[k2] - self.target_mean[k1]

    def endpoint_loss_for_key(self, pred_state, key: Key) -> Tensor:
        pred_support = pred_state.z
        pred_weights = pred_state.atom_weights()
        target_support = self.target_support[key]
        target_weights = self.target_weights[key]
        if self.config.loss_mode == 'uot':
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
            pred_mean = particle_state.mean()[0]
            pred_var = particle_state.variance_trace()
            pred_mean_by_key[key] = pred_mean
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
            rows.append({
                'key': key,
                'sample_id': particle_state.sample_id,
                'perturbation_id': particle_state.perturbation_id,
                'endpoint_loss': float(endpoint_loss.detach().cpu()),
                'aux_loss': float(aux_loss.detach().cpu()),
                'pred_mass': float(pred_mass.detach().cpu()),
                'pred_mean_0': float(pred_mean.detach().cpu()),
                'pred_var_trace': float(pred_var.detach().cpu()),
                'target_mass': float(self.target_mass[key].detach().cpu()),
                'target_mean_0': float(self.target_mean[key].detach().cpu()),
                'target_var_trace': float(self.target_var[key].detach().cpu()),
            })
        endpoint_total = torch.stack(endpoint_losses).mean()
        aux_total = torch.stack(aux_losses).mean()

        screen_delta_losses = []
        sample_ids = sorted({key[0] for key in self.problem.keys if isinstance(key, tuple)})
        if len(sample_ids) == 2 and self.config.aux_screen_delta_mean_weight > 0:
            s1, s2 = sample_ids
            for pid in ('ctrl', 'drift', 'diff', 'react'):
                k1 = (s1, pid)
                k2 = (s2, pid)
                if k1 in pred_mean_by_key and k2 in pred_mean_by_key:
                    pred_delta = pred_mean_by_key[k2] - pred_mean_by_key[k1]
                    screen_delta_losses.append((pred_delta - self.delta_mean_targets[pid]) ** 2)
        screen_delta_total = torch.stack(screen_delta_losses).mean() if screen_delta_losses else torch.zeros((), dtype=endpoint_total.dtype, device=endpoint_total.device)
        table = pd.DataFrame(rows).sort_values(['sample_id', 'perturbation_id']).reset_index(drop=True)
        return endpoint_total, aux_total, screen_delta_total, table


def initialize_model_from_stage2_moments(
    model: ControlAnchoredStage2Model,
    problem: EndpointProblem,
    config: Stage2TrainConfig,
) -> None:
    sample_ids = sorted({key[0] for key in problem.keys if isinstance(key, tuple)})
    if len(sample_ids) != 2:
        return
    s1, s2 = sample_ids
    kappa_target = 2.0
    if isinstance(problem.metadata, dict):
        maybe = problem.metadata.get('simulator_config', {}).get('kappa') if problem.metadata.get('simulator_config') else None
        if maybe is not None:
            kappa_target = float(maybe)

    with torch.no_grad():
        model.kappa_raw.fill_(_inv_softplus(kappa_target))
        model.eta_raw.fill_(_inv_softplus(0.4))

        def pooled_stats(pid: str) -> tuple[float, float, float, float, float, float]:
            init_means, init_vars, term_means, term_vars, init_masses, term_masses = [], [], [], [], [], []
            for sid in sample_ids:
                init = problem.initial[(sid, pid)]
                term = problem.terminal[(sid, pid)]
                init_means.append(float(init.mean()[0]))
                init_vars.append(float(init.variance_trace()))
                term_means.append(float(term.mean()[0]))
                term_vars.append(float(term.variance_trace()))
                init_masses.append(float(init.total_mass))
                term_masses.append(float(term.total_mass))
            return (
                float(sum(init_means) / len(init_means)),
                float(sum(init_vars) / len(init_vars)),
                float(sum(term_means) / len(term_means)),
                float(sum(term_vars) / len(term_vars)),
                float(sum(init_masses) / len(init_masses)),
                float(sum(term_masses) / len(term_masses)),
            )

        m0_ctrl, v0_ctrl, m1_ctrl, v1_ctrl, M0_ctrl, M1_ctrl = pooled_stats('ctrl')
        exp_term = math.exp(-kappa_target)
        denom = max(1.0 - exp_term, 1e-6)
        theta_ctrl = (m1_ctrl - m0_ctrl * exp_term) / denom
        sigma_ctrl_sq = max((2.0 * kappa_target * (v1_ctrl - v0_ctrl * math.exp(-2.0 * kappa_target))) / max(1.0 - math.exp(-2.0 * kappa_target), 1e-6), 1e-8)
        sigma_ctrl = math.sqrt(max(sigma_ctrl_sq, 1e-8))
        rho_ctrl = math.log(M1_ctrl / max(M0_ctrl, 1e-8))

        model.theta_field.baseline.fill_(theta_ctrl)
        model.sigma_field.baseline.fill_(_inv_softplus(max(sigma_ctrl - config.sigma_min, 1e-6)))
        if config.use_growth and config.loss_mode == 'uot':
            model.growth_field.baseline.fill_(_artanh_clipped(rho_ctrl / config.r_max))
        else:
            model.growth_field.baseline.zero_()

        model.theta_field.modulation.zero_()
        model.sigma_field.modulation.zero_()
        model.growth_field.modulation.zero_()

        for perturbation_id in model.embedding_store.non_control_ids:
            m0, v0, m1, v1, M0, M1 = pooled_stats(perturbation_id)
            theta = (m1 - m0 * exp_term) / denom
            sigma_sq = max((2.0 * kappa_target * (v1 - v0 * math.exp(-2.0 * kappa_target))) / max(1.0 - math.exp(-2.0 * kappa_target), 1e-6), 1e-8)
            sigma = math.sqrt(max(sigma_sq, 1e-8))
            rho = math.log(M1 / max(M0, 1e-8))
            emb = model.embedding_store.forward_one(perturbation_id).detach().cpu()
            active_index = int(torch.argmax(torch.abs(emb)).item())
            model.theta_field.modulation[0, active_index] = theta - theta_ctrl
            if not config.shared_diffusion:
                model.sigma_field.modulation[0, active_index] = _inv_softplus(max(sigma - config.sigma_min, 1e-6)) - float(model.sigma_field.baseline[0, 0])
            if config.use_growth and config.loss_mode == 'uot':
                target_raw = _artanh_clipped(rho / config.r_max)
                model.growth_field.modulation[0, active_index] = target_raw - float(model.growth_field.baseline[0, 0])


def train_stage2_model(
    dataset: PerturbSeqDynamicsData,
    *,
    config: Optional[Stage2TrainConfig] = None,
) -> Stage2TrainingResult:
    config = config or Stage2TrainConfig()
    config.validate()
    torch.set_default_dtype(config.simulator_config().torch_dtype)

    dataset.validate()
    problem = dataset.to_endpoint_problem(by_sample=True)
    if dataset.truth is not None and dataset.truth.simulator_config:
        problem.metadata = {**problem.metadata, 'simulator_config': dict(dataset.truth.simulator_config)}

    simulator = LearnedStage2JointSimulator(problem, config.simulator_config())
    model = ControlAnchoredStage2Model(problem.catalog, config.coefficient_config()).to(config.resolved_device)
    initialize_model_from_stage2_moments(model, problem, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_computer = EndpointLossComputer(problem, config)

    history_rows: List[dict] = []
    best_state = None
    best_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sim_result = simulator.run(model)
        endpoint_loss, aux_loss, screen_delta_loss, _ = loss_computer.compute(sim_result.terminal_particles)
        reg = model.regularization_terms()
        reg_total = (
            config.reg_embedding * reg['emb']
            + config.reg_modulation * reg['mod']
            + config.reg_dispersion * reg['disp']
            + config.reg_network * reg['nn']
            + config.reg_context * reg['context']
        )
        total_loss = endpoint_loss + aux_loss + config.aux_screen_delta_mean_weight * screen_delta_loss + reg_total
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        total_loss_value = float(total_loss.detach().cpu())
        history_rows.append({
            'epoch': epoch + 1,
            'total_loss': total_loss_value,
            'endpoint_loss': float(endpoint_loss.detach().cpu()),
            'aux_loss': float(aux_loss.detach().cpu()),
            'screen_delta_loss': float(screen_delta_loss.detach().cpu()),
            'reg_total': float(reg_total.detach().cpu()),
            'eta': float(torch.nn.functional.softplus(model.eta_raw).detach().cpu()) if config.use_context else 0.0,
            'kappa': float(torch.nn.functional.softplus(model.kappa_raw).detach().cpu()),
        })
        if total_loss_value < best_loss:
            best_loss = total_loss_value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    final_simulation = simulator.run(model)
    _, _, _, final_table = loss_computer.compute(final_simulation.terminal_particles)
    history = pd.DataFrame(history_rows)
    return Stage2TrainingResult(
        config=config,
        model=model,
        simulator=simulator,
        history=history,
        final_simulation=final_simulation,
        final_loss_table=final_table,
    )


# Backward-compatible alias
train_step4_model = train_stage2_model

# Semantic aliases for the clearer software-facing API.
MultiscreenContextTrainConfig = Stage2TrainConfig
MultiscreenContextTrainingResult = Stage2TrainingResult
initialize_multiscreen_context_model_from_moments = initialize_model_from_stage2_moments
train_multiscreen_context_model = train_stage2_model
