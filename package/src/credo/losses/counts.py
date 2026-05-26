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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _as_float_tensor(x: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return x.to(device=device, dtype=dtype)


def _validate_count_matrix(count_matrix: torch.Tensor) -> None:
    if count_matrix.ndim != 2:
        raise ValueError(f"count_matrix must be 2D [samples, perturbations], got {count_matrix.ndim}D")
    if not torch.isfinite(count_matrix).all() or torch.any(count_matrix < 0):
        raise ValueError("count_matrix must be nonnegative and finite")


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
    count_matrix = _as_float_tensor(count_matrix, device=zeta.device, dtype=zeta.dtype)
    exposures = _as_float_tensor(exposures, device=zeta.device, dtype=zeta.dtype)
    _validate_count_matrix(count_matrix)
    if not torch.isfinite(exposures).all() or torch.any(exposures <= 0):
        raise ValueError("exposures must be positive and finite")
    G = zeta.shape[0]
    n_samples = count_matrix.shape[0]
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


def _validate_pi(pi: torch.Tensor, counts: torch.Tensor) -> None:
    if pi.shape != counts.shape:
        raise ValueError(f"pi shape must match counts shape {tuple(counts.shape)}, got {tuple(pi.shape)}")
    if not torch.isfinite(pi).all() or torch.any(pi < 0):
        raise ValueError("pi must be nonnegative and finite")
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
        _validate_count_matrix(counts)
        _validate_pi(pi, counts)
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
            _validate_count_matrix(count_matrix)
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
                _validate_count_matrix(count_matrix)
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
