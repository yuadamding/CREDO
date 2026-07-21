"""Count likelihood for replicate-level guide abundance.

Model:  pi_{gs}(u) proportional to l_{g,b(s)} * exp(zeta_g(u))

where l_{g,b(s)} is the T0 exposure for perturbation g in batch b(s),
and zeta_g(u) is the integrated average fitness up to time u.

Likelihood: Dirichlet-multinomial or overdispersed multinomial.

The integrated fitness zeta_g is computed from the growth trajectory:
    zeta_g(tau) = integral_0^tau r_bar_g(s) ds
               ≈ sum_k r_bar_g(tau_k) * delta_tau_k
where r_bar_g = E_{p_g}[r_g] = sum_i w_norm_i r_i.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CountBlock:
    """One donor/context-group compositional count observation."""

    context_group_id: str
    time_label: str
    key_indices: torch.Tensor
    exposure: torch.Tensor
    counts: torch.Tensor
    n_total: torch.Tensor

    def __post_init__(self) -> None:
        key_indices = torch.as_tensor(self.key_indices, dtype=torch.long).reshape(-1)
        exposure = torch.as_tensor(self.exposure, dtype=torch.float32).reshape(-1)
        counts = torch.as_tensor(self.counts, dtype=torch.float32).reshape(-1)
        n_total = torch.as_tensor(self.n_total, dtype=torch.float32).reshape(())
        if not (len(key_indices) == len(exposure) == len(counts)):
            raise ValueError("CountBlock key_indices, exposure, and counts must have equal length.")
        if len(key_indices) < 1:
            raise ValueError("CountBlock must contain at least one key.")
        if len(torch.unique(key_indices)) != len(key_indices) or torch.any(key_indices < 0):
            raise ValueError("CountBlock key_indices must be unique and nonnegative.")
        if not torch.isfinite(exposure).all() or torch.any(exposure <= 0):
            raise ValueError("CountBlock exposure must be positive and finite.")
        _validate_count_matrix(counts.unsqueeze(0), require_integer=True)
        if not torch.isclose(counts.sum(), n_total, rtol=1e-5, atol=1e-5):
            raise ValueError("CountBlock n_total must equal counts.sum().")
        object.__setattr__(self, "key_indices", key_indices)
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "n_total", n_total)


class FitnessBank:
    """Detached genome-wide integrated-fitness estimates for batched count loss."""

    def __init__(
        self,
        time_labels: Iterable[str],
        n_keys: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        momentum: float = 0.9,
    ) -> None:
        labels = [str(label) for label in time_labels]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("FitnessBank time_labels must be nonempty and unique.")
        if n_keys < 1:
            raise ValueError("FitnessBank n_keys must be >= 1.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("FitnessBank momentum must be in [0, 1).")
        self.time_labels = labels
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        self.values = torch.zeros(len(labels), n_keys, device=device, dtype=dtype)
        self.seen = torch.zeros(len(labels), n_keys, device=device, dtype=torch.bool)
        self.momentum = float(momentum)

    @torch.no_grad()
    def update(
        self,
        zeta_curve: torch.Tensor,
        checkpoint_indices: Dict[str, int],
        active_key_indices: torch.Tensor,
    ) -> None:
        active = active_key_indices.to(device=self.values.device, dtype=torch.long)
        for label, bank_idx in self.label_to_index.items():
            if label not in checkpoint_indices:
                raise KeyError(f"Missing checkpoint for FitnessBank label {label!r}.")
            current = zeta_curve[checkpoint_indices[label]].detach().to(self.values)
            if current.shape != active.shape:
                raise ValueError("Active zeta length does not match active_key_indices.")
            old = self.values[bank_idx].index_select(0, active)
            was_seen = self.seen[bank_idx].index_select(0, active)
            blended = torch.where(
                was_seen,
                self.momentum * old + (1.0 - self.momentum) * current,
                current,
            )
            self.values[bank_idx].index_copy_(0, active, blended)
            self.seen[bank_idx].index_fill_(0, active, True)

    def compose(
        self,
        time_label: str,
        active_key_indices: torch.Tensor,
        active_zeta: torch.Tensor,
    ) -> torch.Tensor:
        bank_idx = self.label_to_index[str(time_label)]
        active = active_key_indices.to(device=self.values.device, dtype=torch.long)
        base = self.values[bank_idx].detach().clone().to(device=active_zeta.device, dtype=active_zeta.dtype)
        return base.index_copy(0, active.to(active_zeta.device), active_zeta)


def _as_float_tensor(x: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return x.to(device=device, dtype=dtype)


def _validate_count_matrix(count_matrix: torch.Tensor, *, require_integer: bool = False) -> None:
    if count_matrix.ndim != 2:
        raise ValueError(f"count_matrix must be 2D [samples, perturbations], got {count_matrix.ndim}D")
    if not torch.isfinite(count_matrix).all() or torch.any(count_matrix < 0):
        raise ValueError("count_matrix must be nonnegative and finite")
    if require_integer and not torch.allclose(count_matrix, count_matrix.round(), rtol=0.0, atol=1e-6):
        raise ValueError("count_matrix must contain integer-like counts")


def integrated_fitness(
    growth_steps: torch.Tensor,   # [K, G, N]
    logw_steps: torch.Tensor,     # [K+1, G, N]  (we use steps 0..K-1)
    tau_steps: torch.Tensor,      # [K+1]
) -> torch.Tensor:
    """Compute integrated average fitness zeta_g at tau=1.

    Returns [G] tensor.
    """
    return integrated_fitness_curve(growth_steps, logw_steps, tau_steps)[-1]


def integrated_fitness_curve(
    growth_steps: torch.Tensor,   # [K, G, N]
    logw_steps: torch.Tensor,     # [K+1, G, N]
    tau_steps: torch.Tensor,      # [K+1]
) -> torch.Tensor:
    """Compute cumulative integrated average fitness at every checkpoint.

    Returns ``zeta[k, g] = integral from tau_steps[0] to tau_steps[k]`` with
    shape ``[K+1, G]`` and supports non-uniform time grids.
    """
    K = growth_steps.shape[0]
    if logw_steps.shape[0] != K + 1 or tau_steps.shape[0] != K + 1:
        raise ValueError("growth_steps, logw_steps, and tau_steps have inconsistent lengths")

    logw_k = logw_steps[:K]
    logw_norm = logw_k - torch.logsumexp(logw_k, dim=-1, keepdim=True)
    w_norm = logw_norm.exp()
    r_bar = (w_norm * growth_steps).sum(-1)  # [K, G]

    dtau = tau_steps[1:] - tau_steps[:-1]
    increments = r_bar * dtau[:, None]
    zeta0 = torch.zeros(1, r_bar.shape[1], dtype=r_bar.dtype, device=r_bar.device)
    return torch.cat([zeta0, torch.cumsum(increments, dim=0)], dim=0)


def count_fractions_from_zeta(
    zeta: torch.Tensor,
    exposures: torch.Tensor,
    count_matrix: torch.Tensor,
) -> torch.Tensor:
    """Predict replicate perturbation fractions from exposure and fitness."""
    if not torch.is_floating_point(zeta):
        zeta = zeta.float()
    if zeta.ndim != 1:
        raise ValueError(f"zeta must be 1D [perturbations], got {zeta.ndim}D")
    count_matrix = _as_float_tensor(count_matrix, device=zeta.device, dtype=zeta.dtype)
    exposures = _as_float_tensor(exposures, device=zeta.device, dtype=zeta.dtype)
    _validate_count_matrix(count_matrix)
    if not torch.isfinite(exposures).all() or torch.any(exposures <= 0):
        raise ValueError("exposures must be positive and finite")
    G = zeta.shape[0]
    n_samples = count_matrix.shape[0]
    if count_matrix.shape[1] != G:
        raise ValueError(
            "count_matrix perturbation dimension must match zeta length "
            f"{G}, got {count_matrix.shape[1]}"
        )
    if exposures.ndim == 1:
        if exposures.shape[0] != G:
            raise ValueError(f"1D exposures must have length {G}, got {tuple(exposures.shape)}.")
        log_l = torch.log(exposures + 1e-30).unsqueeze(0)
        log_unnorm = log_l + zeta.unsqueeze(0)
        log_pi = log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)
        return log_pi.exp().expand(n_samples, -1)
    if exposures.ndim == 2:
        if exposures.shape != count_matrix.shape:
            raise ValueError(
                "2D exposures must match count_matrix shape "
                f"{tuple(count_matrix.shape)}, got {tuple(exposures.shape)}."
            )
        log_l = torch.log(exposures + 1e-30)
        log_unnorm = log_l + zeta.unsqueeze(0)
        log_pi = log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)
        return log_pi.exp()
    raise ValueError(f"exposures must be 1D or 2D, got {exposures.ndim}D tensor.")


def _validate_pi(pi: torch.Tensor, counts: torch.Tensor, *, strict_positive: bool = False) -> None:
    if pi.shape != counts.shape:
        raise ValueError(f"pi shape must match counts shape {tuple(counts.shape)}, got {tuple(pi.shape)}")
    if not torch.isfinite(pi).all() or torch.any(pi < 0):
        raise ValueError("pi must be nonnegative and finite")
    if strict_positive and torch.any(pi <= 0):
        raise ValueError("pi must be strictly positive for Dirichlet-multinomial likelihood")
    row_sums = pi.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-5):
        raise ValueError("pi rows must sum to 1")


class DirichletMultinomialLikelihood(nn.Module):
    """Dirichlet-multinomial log-likelihood for count data.

    Parameterization: concentration = phi * pi, where phi > 0 is overdispersion.
    """

    def __init__(self, log_phi: float = 0.0) -> None:
        super().__init__()
        self.log_phi = nn.Parameter(torch.tensor(log_phi))

    def forward(
        self,
        counts: torch.Tensor,     # [S, G]  observed counts per replicate and perturbation
        pi: torch.Tensor,         # [S, G]  predicted fractions (sum to 1 per replicate)
        n_total: torch.Tensor,    # [S]     total counts per replicate
    ) -> torch.Tensor:
        """Return negative log-likelihood (to minimise)."""
        if not torch.is_floating_point(counts):
            counts = counts.float()
        pi = pi.to(device=counts.device, dtype=counts.dtype)
        n_total = n_total.to(device=counts.device, dtype=counts.dtype)
        _validate_count_matrix(counts, require_integer=True)
        _validate_pi(pi, counts, strict_positive=True)
        observed_totals = counts.sum(dim=1)
        if not torch.allclose(observed_totals, n_total, rtol=1e-5, atol=1e-5):
            raise ValueError("n_total must equal counts.sum(dim=1) for Dirichlet-multinomial likelihood.")
        phi = self.log_phi.exp()
        alpha = phi * pi + 1e-8   # [S, G]  concentration parameters

        count_constant = torch.lgamma(n_total + 1.0) - torch.lgamma(counts + 1.0).sum(-1)
        concentration_terms = torch.lgamma(phi) - torch.lgamma(phi + n_total)
        category_terms = (torch.lgamma(alpha + counts) - torch.lgamma(alpha)).sum(-1)
        ll = count_constant + concentration_terms + category_terms

        return -ll.sum()


class CountLikelihood(nn.Module):
    """Full count likelihood integrating growth trajectories and T0 exposures.

    Parameters
    ----------
    use_dirichlet_multinomial: if False, use standard multinomial
    """

    def __init__(self, use_dirichlet_multinomial: bool = True) -> None:
        super().__init__()
        self.use_dm = use_dirichlet_multinomial
        if use_dirichlet_multinomial:
            self.dm_lik = DirichletMultinomialLikelihood()

    def forward(
        self,
        growth_steps: torch.Tensor,     # [K, G, N]
        logw_steps: torch.Tensor,       # [K+1, G, N]
        tau_steps: torch.Tensor,        # [K+1]
        exposures: torch.Tensor,        # [G] or [S, G]  T0 exposure per perturbation / replicate
        count_matrix: torch.Tensor,     # [S, G]  observed counts
        n_totals: torch.Tensor,         # [S]
    ) -> torch.Tensor:
        """Compute negative count log-likelihood."""
        zeta = integrated_fitness(growth_steps, logw_steps, tau_steps)  # [G]
        pi_rep = count_fractions_from_zeta(zeta, exposures, count_matrix)

        if self.use_dm:
            return self.dm_lik(count_matrix, pi_rep, n_totals)
        else:
            # Standard multinomial
            count_matrix = count_matrix.to(device=pi_rep.device, dtype=pi_rep.dtype)
            _validate_count_matrix(count_matrix, require_integer=True)
            log_lik = (count_matrix * torch.log(pi_rep + 1e-30)).sum(-1)
            return -log_lik.sum()


class MultiTimeCountLikelihood(nn.Module):
    """Count likelihood evaluated at multiple observed rollout checkpoints."""

    def __init__(
        self,
        use_dirichlet_multinomial: bool = True,
        time_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.use_dm = use_dirichlet_multinomial
        self.time_weights = time_weights or {}
        if use_dirichlet_multinomial:
            self.dm_lik = DirichletMultinomialLikelihood()

    def forward_with_logs(
        self,
        growth_steps: torch.Tensor,
        logw_steps: torch.Tensor,
        tau_steps: torch.Tensor,
        exposures: torch.Tensor | Dict[str, torch.Tensor],
        count_matrices: Dict[str, torch.Tensor],
        n_totals: Dict[str, torch.Tensor],
        checkpoint_indices: Dict[str, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zeta_curve = integrated_fitness_curve(growth_steps, logw_steps, tau_steps)
        total = torch.tensor(0.0, dtype=growth_steps.dtype, device=growth_steps.device)
        logs: Dict[str, torch.Tensor] = {}

        for time_label, count_matrix in count_matrices.items():
            if time_label not in checkpoint_indices:
                raise KeyError(f"Missing checkpoint index for count time label {time_label!r}")
            if time_label not in n_totals:
                raise KeyError(f"Missing n_totals for time label {time_label!r}")
            idx = checkpoint_indices[time_label]
            exposure_t = exposures[time_label] if isinstance(exposures, dict) else exposures
            pi_rep = count_fractions_from_zeta(zeta_curve[idx], exposure_t, count_matrix)
            if self.use_dm:
                loss_t = self.dm_lik(count_matrix, pi_rep, n_totals[time_label])
            else:
                count_matrix = count_matrix.to(device=pi_rep.device, dtype=pi_rep.dtype)
                _validate_count_matrix(count_matrix, require_integer=True)
                log_lik = (count_matrix * torch.log(pi_rep + 1e-30)).sum(-1)
                loss_t = -log_lik.sum()
            weight = float(self.time_weights.get(time_label, 1.0))
            total = total + weight * loss_t
            logs[f"counts/{time_label}"] = loss_t.detach()
            logs[f"counts/{time_label}/weight"] = torch.tensor(
                weight, dtype=growth_steps.dtype, device=growth_steps.device
            )
            logs[f"counts/{time_label}/n_samples"] = torch.tensor(
                count_matrix.shape[0], device=growth_steps.device
            )
            logs[f"counts/{time_label}/n_perturbations"] = torch.tensor(
                count_matrix.shape[1], device=growth_steps.device
            )
            logs[f"counts/{time_label}/n_total_sum"] = count_matrix.sum().detach()

        return total, logs

    def forward(
        self,
        growth_steps: torch.Tensor,
        logw_steps: torch.Tensor,
        tau_steps: torch.Tensor,
        exposures: torch.Tensor | Dict[str, torch.Tensor],
        count_matrices: Dict[str, torch.Tensor],
        n_totals: Dict[str, torch.Tensor],
        checkpoint_indices: Dict[str, int],
    ) -> torch.Tensor:
        total, _ = self.forward_with_logs(
            growth_steps=growth_steps,
            logw_steps=logw_steps,
            tau_steps=tau_steps,
            exposures=exposures,
            count_matrices=count_matrices,
            n_totals=n_totals,
            checkpoint_indices=checkpoint_indices,
        )
        return total


class GroupedMultiTimeCountLikelihood(nn.Module):
    """Donor/context-group count likelihood with full compositional denominators."""

    def __init__(
        self,
        use_dirichlet_multinomial: bool = True,
        time_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.use_dm = bool(use_dirichlet_multinomial)
        self.time_weights = time_weights or {}
        if self.use_dm:
            self.dm_lik = DirichletMultinomialLikelihood()

    def forward_with_logs(
        self,
        *,
        growth_steps: torch.Tensor,
        logw_steps: torch.Tensor,
        tau_steps: torch.Tensor,
        blocks: Iterable[CountBlock],
        checkpoint_indices: Dict[str, int],
        active_key_indices: Optional[torch.Tensor] = None,
        fitness_bank: Optional[FitnessBank] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zeta_curve = integrated_fitness_curve(growth_steps, logw_steps, tau_steps)
        if active_key_indices is None:
            active_key_indices = torch.arange(
                zeta_curve.shape[1], device=zeta_curve.device, dtype=torch.long
            )
        else:
            active_key_indices = active_key_indices.to(zeta_curve.device, torch.long)
        active_lookup = {
            int(global_idx): local_idx
            for local_idx, global_idx in enumerate(active_key_indices.tolist())
        }
        total = zeta_curve.new_zeros(())
        logs: Dict[str, torch.Tensor] = {}
        n_blocks = 0
        for block in blocks:
            if block.time_label not in checkpoint_indices:
                raise KeyError(f"Missing checkpoint index for count block {block.time_label!r}.")
            checkpoint = checkpoint_indices[block.time_label]
            active_zeta = zeta_curve[checkpoint]
            if fitness_bank is None:
                missing = [idx for idx in block.key_indices.tolist() if int(idx) not in active_lookup]
                if missing:
                    raise ValueError(
                        "Grouped count blocks reference inactive keys but no FitnessBank was supplied."
                    )
                local = torch.tensor(
                    [active_lookup[int(idx)] for idx in block.key_indices.tolist()],
                    device=zeta_curve.device,
                    dtype=torch.long,
                )
                zeta_block = active_zeta.index_select(0, local)
            else:
                full_zeta = fitness_bank.compose(
                    block.time_label,
                    active_key_indices,
                    active_zeta,
                )
                zeta_block = full_zeta.index_select(
                    0,
                    block.key_indices.to(device=zeta_curve.device, dtype=torch.long),
                )
            exposure = block.exposure.to(device=zeta_curve.device, dtype=zeta_curve.dtype)
            counts = block.counts.to(device=zeta_curve.device, dtype=zeta_curve.dtype).unsqueeze(0)
            n_total = block.n_total.to(device=zeta_curve.device, dtype=zeta_curve.dtype).reshape(1)
            pi = torch.softmax(torch.log(exposure) + zeta_block, dim=-1).unsqueeze(0)
            if self.use_dm:
                block_loss = self.dm_lik(counts, pi, n_total)
            else:
                block_loss = -(counts * torch.log(pi.clamp_min(1e-30))).sum()
            weight = float(self.time_weights.get(block.time_label, 1.0))
            total = total + weight * block_loss
            prefix = f"counts/{block.time_label}/{block.context_group_id}"
            logs[prefix] = block_loss.detach()
            logs[f"{prefix}/n_total"] = n_total.sum().detach()
            logs[f"{prefix}/n_keys"] = torch.tensor(len(block.key_indices), device=zeta_curve.device)
            n_blocks += 1
        if n_blocks == 0:
            raise ValueError("GroupedMultiTimeCountLikelihood received no CountBlock objects.")
        logs["counts/n_blocks"] = torch.tensor(n_blocks, device=zeta_curve.device)
        return total, logs

    def forward(self, **kwargs) -> torch.Tensor:
        total, _ = self.forward_with_logs(**kwargs)
        return total
