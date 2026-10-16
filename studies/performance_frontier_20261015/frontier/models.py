"""New frontier generators; no data access, historical weights, or hidden RNG.

F0/F1 use the retained DIRECT equation and F4/F5 the retained eight-step DDR
equation. This is a configurable reimplementation, not reuse of fitted models.
F3/F5 add zero-initialized rank-four target codes with a learned Xavier-initialized
projection and supplied fixed fitting-support shrinkage. Controls and unseen
targets have exactly zero private continuation, including growth intercepts.

F2 is one *specific* new mixture: four independent nonlinear source/noise maps,
global learned component probabilities, and balanced deterministic component
allocation. With P particles and n_k assigned to component k, a particle gets
proposal correction P*pi[k]/n_k (4*pi[k] when P is divisible by four).
This is a stratified Monte Carlo mixture, not categorical straight-through
sampling or averaging component coordinates. Its global gate is not an arbitrary
source-conditioned mixture gate. Equal-weight sampled source particles and at
least four particles are required. With reaction disabled its mass multiplier is
exactly one by construction. Prefixes retain their original states but recompute
proposal counts and all weight/mass reductions, not truncated normalized weights.

All stochastic source draws and eight Gaussian increments are supplied by the
caller. No loss coefficients or scientific resource settings are selected here.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .configuration import Configuration, get_configuration


def mixture_component_counts(particles: int, components: int = 4) -> tuple[int, ...]:
    """Counts for assignment particle_index % K; scalar metadata only."""
    if (type(particles) is not int or type(components) is not int
            or components < 1 or particles < components):
        raise ValueError("At least one particle per mixture component is required")
    base, remainder = divmod(particles, components)
    return tuple(base + (component < remainder) for component in range(components))


def _mlp(inputs: int, width: int, depth: int) -> nn.Sequential:
    layers = []
    for _ in range(depth):
        layers.extend((nn.Linear(inputs, width, device="cpu", dtype=torch.float32), nn.Tanh()))
        inputs = width
    return nn.Sequential(*layers)


def time_features(tau: Tensor) -> Tensor:
    """Exact retained order: tau, sin(pi*tau), cos(pi*tau), ..., frequency 4."""
    tau = tau.reshape(())
    values = [tau]
    for frequency in range(1, 5):
        values.extend((torch.sin(frequency * torch.pi * tau),
                       torch.cos(frequency * torch.pi * tau)))
    return torch.stack(values)


def _modulated(hidden, embedding, reference, modulation, output_dim):
    # Retain the legacy effective-weight contraction order rather than forming
    # a laws x particles x outputs x embedding activation tensor.
    residual_weight = modulation.weight.reshape(output_dim, embedding.shape[-1], hidden.shape[-1])
    effective_weight = reference.weight[None] + torch.einsum("gr,orh->goh", embedding, residual_weight)
    output = torch.bmm(hidden, effective_weight.transpose(1, 2))
    if reference.bias is not None:
        output = output + reference.bias.reshape(1, 1, output_dim)
    if modulation.bias is not None:
        effective_bias = torch.einsum("gr,or->go", embedding, modulation.bias.reshape(output_dim, embedding.shape[-1]))
        output = output + effective_bias[:, None]
    return output


class _DirectMap(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, width, embedding = config.latent_dim, config.width, config.embedding_dim
        self.dim = dim
        self.bound = config.reaction_bound
        self.trunk = _mlp(2 * dim, width, config.depth)
        self.displacement_reference = nn.Linear(width, dim, device="cpu", dtype=torch.float32)
        self.displacement_modulation = nn.Linear(width, dim * embedding, device="cpu", dtype=torch.float32)
        self.weight_reference = nn.Linear(width, 1, device="cpu", dtype=torch.float32)
        self.weight_modulation = nn.Linear(width, embedding, device="cpu", dtype=torch.float32)

    def displacement(self, source, epsilon, embedding):
        hidden = self.trunk(torch.cat((source, epsilon), dim=-1))
        return _modulated(hidden, embedding, self.displacement_reference,
                          self.displacement_modulation, self.dim)

    def forward(self, source, epsilon, embedding, intercept, *, reaction_enabled):
        hidden = self.trunk(torch.cat((source, epsilon), dim=-1))
        displacement = _modulated(hidden, embedding, self.displacement_reference,
                                  self.displacement_modulation, self.dim)
        if reaction_enabled:
            raw = self.weight_reference(hidden).squeeze(-1)
            raw = raw + torch.einsum("gnr,gr->gn", self.weight_modulation(hidden), embedding)
            reaction = self.bound * torch.tanh(raw + intercept[:, None])
        else:
            reaction = source.new_zeros(source.shape[:2])
        return displacement, reaction


class _DDRFields(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, width, embedding = config.latent_dim, config.width, config.embedding_dim
        self.dim, self.floor, self.bound = dim, config.diffusion_floor, config.reaction_bound
        self.trunk = _mlp(dim + 9, width, config.depth)
        self.drift_reference = nn.Linear(width, dim, device="cpu", dtype=torch.float32)
        self.drift_residual = nn.Linear(width, dim * embedding, device="cpu", dtype=torch.float32)
        self.sigma_reference = nn.Linear(width, dim, device="cpu", dtype=torch.float32)
        self.sigma_residual = nn.Linear(width, dim * embedding, device="cpu", dtype=torch.float32)
        self.growth_reference = nn.Linear(width, 1, device="cpu", dtype=torch.float32)
        self.growth_residual = nn.Linear(width, embedding, device="cpu", dtype=torch.float32)

    def forward(self, states, tau, embedding, intercept, *, reaction_enabled):
        time = time_features(tau).reshape(1, 1, 9).expand(*states.shape[:2], 9)
        hidden = self.trunk(torch.cat((states, time), dim=-1))
        drift = _modulated(hidden, embedding, self.drift_reference, self.drift_residual, self.dim)
        sigma = functional.softplus(_modulated(hidden, embedding, self.sigma_reference,
                                               self.sigma_residual, self.dim)) + self.floor
        if reaction_enabled:
            raw = self.growth_reference(hidden).squeeze(-1)
            raw = raw + torch.einsum("gnr,gr->gn", self.growth_residual(hidden), embedding)
            reaction = self.bound * torch.tanh(raw + intercept[:, None])
        else:
            reaction = states.new_zeros(states.shape[:2])
        return drift, sigma, reaction


@dataclass
class Rollout:
    states: Tensor
    log_weights: Tensor
    log_mass_increment: Tensor
    reaction_log_increment: Tensor
    mixture_log_correction: Tensor
    component_ids: Tensor
    displacement: Tensor | None = None
    anchor_displacement: Tensor | None = None
    drift_steps: Tensor | None = None
    sigma_steps: Tensor | None = None
    reaction_steps: Tensor | None = None
    mixture_log_probabilities: Tensor | None = None
    reaction_enabled: bool = True
    configuration_id: str = ""

    @property
    def weights(self):
        return torch.softmax(self.log_weights, dim=-1)

    @property
    def mass_multiplier(self):
        return torch.exp(self.log_mass_increment)


class FrontierModel(nn.Module):
    """One registered configurable generator; guide IDs never enter the trunk.

    target_ids enumerate non-control fitting targets only. An unknown metadata ID
    maps to the exact-zero residual. support_factors are fixed values in [0,1],
    supplied from the fitting-only support rule; this class cannot certify their
    empirical provenance. Their determination remains a scientific contract item.
    """

    def __init__(self, configuration: str | Configuration, *, target_ids: Sequence[str] = (),
                 support_factors: Tensor | Sequence[float] | None = None,
                 control_ids: Sequence[str] = ("NTC",)):
        super().__init__()
        self.config = get_configuration(configuration) if isinstance(configuration, str) else configuration
        if not isinstance(self.config, Configuration):
            raise TypeError("A registered Configuration or identifier is required")
        registered = get_configuration(self.config.configuration_id)
        if self.config != registered:
            raise ValueError("Configuration differs from the registered recipe")
        ids, controls = tuple(target_ids), tuple(control_ids)
        if (any(not isinstance(x, str) or not x for x in (*ids, *controls))
                or len(set(ids)) != len(ids) or len(set(controls)) != len(controls)
                or not controls or set(ids) & set(controls)):
            raise ValueError("Distinct non-control target IDs and explicit control IDs are required")
        self.control_ids = controls
        self.target_ids = ids if self.config.target_residual else ()
        self._target_lookup = {key: index for index, key in enumerate(self.target_ids)}
        self.reference_embedding = nn.Parameter(torch.zeros(self.config.embedding_dim, device="cpu", dtype=torch.float32))
        if self.config.dynamics == "direct":
            self.components = nn.ModuleList(_DirectMap(self.config) for _ in range(self.config.mixture_components))
            if self.config.mixture_components > 1:
                self.mixture_logits = nn.Parameter(torch.zeros(self.config.mixture_components, device="cpu", dtype=torch.float32))
        else:
            self.fields = _DDRFields(self.config)
        # Private initialization follows the common network constructor so F0/F3
        # and F4/F5 have identical common tensors at a common fresh random seed.
        if self.config.target_residual:
            if not ids or support_factors is None:
                raise ValueError("Target families require a fitting target inventory and support factors")
            factors = torch.as_tensor(support_factors, dtype=torch.float32, device="cpu").detach().clone()
            if factors.shape != (len(ids),) or not bool(torch.isfinite(factors).all()) or bool(((factors < 0) | (factors > 1)).any()):
                raise ValueError("One finite fixed support factor in [0,1] per fitting target is required")
            self.register_buffer("support_factors", factors)
            self.target_codes = nn.Parameter(torch.zeros(len(ids), self.config.residual_rank, device="cpu", dtype=torch.float32))
            self.target_projection = nn.Parameter(torch.empty(self.config.residual_rank, self.config.embedding_dim, device="cpu", dtype=torch.float32))
            nn.init.xavier_uniform_(self.target_projection)
            self.target_growth_intercepts = nn.Parameter(torch.zeros(len(ids), device="cpu", dtype=torch.float32))
        elif support_factors is not None:
            raise ValueError("SHARED families must not receive private support tables")
        self.set_phase("state")

    def target_indices(self, target_ids: Sequence[str], *, device=None) -> Tensor:
        """-1 denotes control or unseen target; known noncontrols use 0..T-1."""
        if any(not isinstance(value, str) or not value for value in target_ids):
            raise ValueError("Target metadata must contain nonempty strings")
        return torch.tensor([self._target_lookup.get(value, -1) for value in target_ids],
                            device=device or self.reference_embedding.device, dtype=torch.long)

    def conditioning(self, indices: Tensor):
        """Tensor-only conditioning after checked metadata routing."""
        reference = self.reference_embedding[None].expand(len(indices), -1)
        if not self.config.target_residual:
            return reference, self.reference_embedding.new_zeros(len(indices))
        residuals = (self.target_codes @ self.target_projection) * self.support_factors[:, None]
        table = torch.cat((residuals.new_zeros(1, self.config.embedding_dim), residuals), dim=0)
        intercepts = self.target_growth_intercepts * self.support_factors
        intercepts = torch.cat((intercepts.new_zeros(1), intercepts))
        return reference + table.index_select(0, indices + 1), intercepts.index_select(0, indices + 1)

    def set_phase(self, phase: str):
        if phase not in ("state", "joint"):
            raise ValueError("Phase must be state or joint")
        self.phase = phase
        self.reaction_enabled = phase == "joint"
        for name, parameter in self.named_parameters():
            private_reaction = ("weight_reference" in name or "weight_modulation" in name
                                or "growth_reference" in name or "growth_residual" in name
                                or name == "target_growth_intercepts")
            parameter.requires_grad_(not private_reaction or self.reaction_enabled)
        return self

    def direct_primitive(self, source, epsilon, embedding, intercept, *, component=0):
        """Pure supplied-noise component equation, suitable for separate profiling."""
        if self.config.dynamics != "direct":
            raise ValueError("Not a direct generator")
        return self.components[component](source, epsilon, embedding, intercept,
                                          reaction_enabled=self.reaction_enabled)

    def ddr_primitive(self, states, tau, embedding, intercept):
        if self.config.dynamics != "ddr":
            raise ValueError("Not a DDR generator")
        return self.fields(states, tau, embedding, intercept, reaction_enabled=self.reaction_enabled)

    def _validate_inputs(self, source, noise, indices):
        if source.ndim != 3 or source.shape[-1] != self.config.latent_dim or min(source.shape[:2]) < 1:
            raise ValueError("Source must be a nonempty laws x particles x latent tensor")
        if source.dtype != torch.float32 or source.device != self.reference_embedding.device:
            raise ValueError("Source must be FP32 on the model device")
        if noise.shape != (self.config.integration_steps, *source.shape) or noise.dtype != source.dtype or noise.device != source.device:
            raise ValueError("Explicit eight-step Gaussian noise must match source device/dtype/shape")
        if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(noise).all()):
            raise ValueError("Source and noise must be finite")
        if indices.shape != (source.shape[0],) or indices.dtype != torch.long or indices.device != source.device:
            raise ValueError("One integer routing index per law on the source device is required")
        if self.config.target_residual and bool(((indices < -1) | (indices >= len(self.target_ids))).any()):
            raise ValueError("Routing index outside frozen target inventory")
        mixture_component_counts(source.shape[1], self.config.mixture_components)

    def rollout(self, source: Tensor, noise: Tensor, *, target_ids: Sequence[str] | None = None,
                indices: Tensor | None = None, compute_anchor: bool = True) -> Rollout:
        if indices is not None and target_ids is not None:
            raise ValueError("Supply either explicit routing indices or target metadata")
        if indices is None:
            if target_ids is None:
                if self.config.target_residual:
                    raise ValueError("Target families require explicit metadata or routing indices")
                indices = torch.full((source.shape[0],), -1, device=source.device, dtype=torch.long)
            else:
                indices = self.target_indices(target_ids, device=source.device)
        self._validate_inputs(source, noise, indices)
        embedding, intercept = self.conditioning(indices)
        particles, components = source.shape[1], self.config.mixture_components
        counts = mixture_component_counts(particles, components)
        component_ids = torch.arange(particles, device=source.device) % components
        correction = source.new_zeros(source.shape[:2])
        if self.config.dynamics == "direct":
            epsilon = noise.sum(dim=0) / math.sqrt(self.config.integration_steps)
            displacement = torch.zeros_like(source)
            reaction = source.new_zeros(source.shape[:2])
            anchor = torch.zeros_like(source) if compute_anchor else None
            log_pi = torch.log_softmax(self.mixture_logits, dim=0) if components > 1 else None
            for component in range(components):
                selected = torch.nonzero(component_ids == component, as_tuple=False).flatten()
                z, eps = source.index_select(1, selected), epsilon.index_select(1, selected)
                shift, rho = self.direct_primitive(z, eps, embedding, intercept, component=component)
                displacement = displacement.index_copy(1, selected, shift)
                reaction = reaction.index_copy(1, selected, rho)
                if compute_anchor:
                    shift0 = self.components[component].displacement(z, torch.zeros_like(eps), embedding)
                    anchor = anchor.index_copy(1, selected, shift0)
                if components > 1:
                    value = (math.log(particles) + log_pi[component] - math.log(counts[component])).expand(source.shape[0], len(selected))
                    correction = correction.index_copy(1, selected, value)
            states = source + displacement
            result = Rollout(states, reaction + correction - math.log(particles),
                             source.new_zeros(source.shape[0]), reaction, correction, component_ids,
                             displacement=displacement, anchor_displacement=anchor,
                             mixture_log_probabilities=log_pi,
                             reaction_enabled=self.reaction_enabled,
                             configuration_id=self.config.configuration_id)
        else:
            states, reaction = source, source.new_zeros(source.shape[:2])
            drifts, sigmas, reactions = [], [], []
            dt = source.new_tensor(1.0 / self.config.integration_steps)
            for step in range(self.config.integration_steps):
                tau = source.new_tensor(step / self.config.integration_steps)
                drift, sigma, growth = self.ddr_primitive(states, tau, embedding, intercept)
                states = states + drift * dt + sigma * dt.sqrt() * noise[step]
                reaction = reaction + growth * dt
                drifts.append(drift)
                sigmas.append(sigma)
                reactions.append(growth)
            result = Rollout(states, reaction - math.log(particles), source.new_zeros(source.shape[0]),
                             reaction, correction, component_ids, drift_steps=torch.stack(drifts),
                             sigma_steps=torch.stack(sigmas), reaction_steps=torch.stack(reactions),
                             reaction_enabled=self.reaction_enabled,
                             configuration_id=self.config.configuration_id)
        # The state-only mathematical multiplier is exactly 1, including F2's
        # stratified importance weights. Avoid manufacturing finite-roundoff mass
        # changes while retaining gradients through normalized F2 state weights.
        result.log_mass_increment = (torch.logsumexp(result.log_weights, dim=-1) if self.reaction_enabled
                                     else source.new_zeros(source.shape[0]))
        return result

    forward = rollout

    def prefix_rollout(self, result: Rollout, particles: int) -> Rollout:
        """Reuse a generated maximum-prefix population, with fresh reductions.

        No model or RNG call occurs. In particular, use the component probabilities
        and reaction phase stored with the rollout, not possibly changed current
        parameters/phase. Cross-model identity/custody is the publisher's contract.
        The caller must apply this before each full-catalog normalization and any
        seed-mixture construction. Noise-shape equivalence of separate rollouts is
        neither assumed nor required: generate once, then derive every prefix.
        """
        if not isinstance(result, Rollout) or result.configuration_id != self.config.configuration_id:
            raise ValueError("A rollout from the same registered configuration is required")
        if type(particles) is not int or not 1 <= particles <= result.states.shape[1]:
            raise ValueError("Prefix must be a positive integer within the generated population")
        components = self.config.mixture_components
        counts = mixture_component_counts(particles, components)
        states = result.states[:, :particles]
        reaction = result.reaction_log_increment[:, :particles]
        component_ids = result.component_ids[:particles]
        expected_ids = torch.arange(particles, device=states.device) % components
        if not torch.equal(component_ids, expected_ids):
            raise ValueError("Rollout component identities differ from the fixed round-robin proposal")
        correction = reaction.new_zeros(reaction.shape)
        if components > 1:
            log_pi = result.mixture_log_probabilities
            if log_pi is None or log_pi.shape != (components,):
                raise ValueError("Mixture rollout is missing its frozen component probabilities")
            count_tensor = log_pi.new_tensor(counts)
            per_component = math.log(particles) + log_pi - count_tensor.log()
            correction = per_component[component_ids][None].expand(states.shape[0], -1)
        log_weights = reaction + correction - math.log(particles)
        log_mass = (torch.logsumexp(log_weights, dim=-1) if result.reaction_enabled
                    else states.new_zeros(states.shape[0]))
        direct_slice = lambda value: None if value is None else value[:, :particles]
        step_slice = lambda value: None if value is None else value[:, :, :particles]
        return replace(result, states=states, reaction_log_increment=reaction,
                       mixture_log_correction=correction, component_ids=component_ids,
                       log_weights=log_weights, log_mass_increment=log_mass,
                       displacement=direct_slice(result.displacement),
                       anchor_displacement=direct_slice(result.anchor_displacement),
                       drift_steps=step_slice(result.drift_steps),
                       sigma_steps=step_slice(result.sigma_steps),
                       reaction_steps=step_slice(result.reaction_steps))

    def architecture(self):
        return dict(configuration=asdict(self.config), phase=self.phase,
                    target_ids=list(self.target_ids), control_ids=list(self.control_ids),
                    parameter_count=sum(p.numel() for p in self.parameters()),
                    active_parameter_count=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    ecological_context=False, source_particles="equal_weight_empirical_draws",
                    random_inputs="caller_supplied_eight_gaussian_increments",
                    mixture_gate="global_learned" if self.config.mixture_components > 1 else "none",
                    mixture_proposal="balanced_deterministic_particle_index_modulo_K",
                    mixture_proposal_correction="P*pi_k/n_k; counts_recomputed_at_each_prefix",
                    prefix_rule="reuse_maximum_rollout_states_recompute_proposal_weights_and_mass",
                    target_initialization="zero_codes_and_intercepts_xavier_projection" if self.config.target_residual else "none",
                    support_factors=self.support_factors.detach().cpu().tolist() if self.config.target_residual else [])


def build_model(configuration: str | Configuration, *, seed: int, **kwargs) -> FrontierModel:
    """Fresh deterministic CPU construction without changing caller RNG state."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("An explicit nonnegative 32-bit seed is required")
    with torch.random.fork_rng(devices=[]):
        # CPU initialization must not reseed a live CUDA worker's generators.
        torch.random.default_generator.manual_seed(seed)
        return FrontierModel(configuration, **kwargs)
