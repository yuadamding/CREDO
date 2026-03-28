"""Weak-form PINN residual loss.

For diagonal diffusion, the generator of the process applied to a test
function psi_m(z) is:

    L_g psi = grad(psi) . v_g  +  (1/2) sum_j sigma_{g,j}^2 d^2psi/dz_j^2
              + r_g * psi

We use Gaussian RBF test functions:
    psi_m(z) = exp(-||z - c_m||^2 / (2 h_m^2))

for which the gradients and Hessian diagonals have analytic forms.

The residual is:
    R_{g,m,k} = [E_hat[w_{k+1} psi_m(Z_{k+1})] - E_hat[w_k psi_m(Z_k)]] / dtau
                - E_hat[w_k L_g psi_m(Z_k)]

and the loss is sum_{g,m,k} R_{g,m,k}^2.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GaussianRBFTestFunctions:
    """Analytic Gaussian RBF test functions with exact first/second derivatives."""

    def __init__(
        self,
        centers: torch.Tensor,   # [M, d]
        bandwidth: float = 1.0,
    ) -> None:
        self.centers = centers   # [M, d]
        self.h = bandwidth

    @property
    def M(self) -> int:
        return len(self.centers)

    def psi(self, z: torch.Tensor) -> torch.Tensor:
        """z: [G, N, d] -> [G, N, M]."""
        # ||z - c_m||^2 for all m
        diff = z.unsqueeze(-2) - self.centers  # [G, N, M, d]
        sq_dist = (diff ** 2).sum(-1)          # [G, N, M]
        return torch.exp(-sq_dist / (2 * self.h ** 2))

    def grad_psi(self, z: torch.Tensor) -> torch.Tensor:
        """d psi_m / d z_j. Returns [G, N, M, d]."""
        diff = z.unsqueeze(-2) - self.centers  # [G, N, M, d]
        psi_val = self.psi(z).unsqueeze(-1)    # [G, N, M, 1]
        return -(1.0 / self.h ** 2) * diff * psi_val  # [G, N, M, d]

    def diag_hess_psi(self, z: torch.Tensor) -> torch.Tensor:
        """d^2 psi_m / d z_j^2 (diagonal of Hessian). Returns [G, N, M, d]."""
        diff = z.unsqueeze(-2) - self.centers   # [G, N, M, d]
        psi_val = self.psi(z).unsqueeze(-1)     # [G, N, M, 1]
        h2 = self.h ** 2
        return psi_val * ((diff / h2) ** 2 - 1.0 / h2)  # [G, N, M, d]


class WeakFormLoss(nn.Module):
    """Weak-form PINN residual loss over particle rollouts.

    Parameters
    ----------
    n_test_functions: M  (refreshed each epoch)
    bandwidth: h for Gaussian RBFs
    latent_dim: d
    """

    def __init__(
        self,
        n_test_functions: int = 32,
        bandwidth: float = 1.0,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.M = n_test_functions
        self.bandwidth = bandwidth
        self.latent_dim = latent_dim
        # Current test function centers (refreshed each forward call if requested)
        self.register_buffer("_centers", torch.zeros(n_test_functions, latent_dim))
        self._centers_initialized = False

    def refresh_test_functions(
        self,
        z_ref: torch.Tensor,  # [G, N, d] use to set center range
        scale: float = 2.0,
    ) -> None:
        """Sample new test function centers from a region covering the particles."""
        G, N, d = z_ref.shape
        z_flat = z_ref.detach().reshape(-1, d)
        z_min = z_flat.min(0).values
        z_max = z_flat.max(0).values
        centers = z_min + torch.rand(self.M, d, device=z_ref.device,
                                     dtype=z_ref.dtype) * (z_max - z_min)
        self._centers = centers
        self._centers_initialized = True

    def _get_test_fns(self, z_ref: torch.Tensor) -> GaussianRBFTestFunctions:
        if not self._centers_initialized:
            self.refresh_test_functions(z_ref)
        return GaussianRBFTestFunctions(
            centers=self._centers.to(z_ref.device, z_ref.dtype),
            bandwidth=self.bandwidth,
        )

    def forward(
        self,
        z_steps: torch.Tensor,       # [K+1, G, N, d]
        logw_steps: torch.Tensor,    # [K+1, G, N]
        drift_steps: torch.Tensor,   # [K, G, N, d]
        sigma_steps: torch.Tensor,   # [K, G, N, d]
        growth_steps: torch.Tensor,  # [K, G, N]
        tau_steps: torch.Tensor,     # [K+1]
        refresh_centers: bool = True,
    ) -> torch.Tensor:
        """Compute sum of squared residuals."""
        K_plus1, G, N, d = z_steps.shape
        K = K_plus1 - 1

        if refresh_centers:
            self.refresh_test_functions(z_steps[0])

        test_fns = self._get_test_fns(z_steps[0])
        M = test_fns.M

        # Compute weighted test-function expectations at each step
        # E_hat[w * psi_m] = sum_i w_i * psi_m(Z_i)  (unnormalized by N)
        # We use normalized log-weights for numerical stability
        # psi_vals: [K+1, G, N, M]
        psi_vals = torch.stack([test_fns.psi(z_steps[k]) for k in range(K + 1)], dim=0)

        # Normalized weights: w_norm_i = softmax of logw over N dim
        # [K+1, G, N] -> [K+1, G, N]
        logw_norm = logw_steps - torch.logsumexp(logw_steps, dim=-1, keepdim=True)
        w_norm = logw_norm.exp()  # [K+1, G, N]

        # Weighted test-fn expectation: [K+1, G, M]
        E_psi = torch.einsum("kgn, kgnm -> kgm", w_norm, psi_vals)

        # --- Generator values at step k ---
        # L_g psi = grad_psi . v + 0.5 * sum_j sigma_j^2 * diag_hess_j + r * psi
        residuals_sq = []
        dtau = (tau_steps[1:] - tau_steps[:-1]).unsqueeze(-1).unsqueeze(-1)  # [K, 1, 1]

        for k in range(K):
            zk = z_steps[k]           # [G, N, d]
            wk = w_norm[k]            # [G, N]

            grad_psi = test_fns.grad_psi(zk)        # [G, N, M, d]
            dhess_psi = test_fns.diag_hess_psi(zk)  # [G, N, M, d]
            psi_k = psi_vals[k]                      # [G, N, M]

            v = drift_steps[k]     # [G, N, d]
            s = sigma_steps[k]     # [G, N, d]
            r = growth_steps[k]    # [G, N]

            # Drift term: grad_psi . v  -> [G, N, M]
            drift_term = torch.einsum("gnmd, gnd -> gnm", grad_psi, v)
            # Diffusion term: 0.5 * sum_j sigma_j^2 * diag_hess_j -> [G, N, M]
            diff_term = 0.5 * torch.einsum("gnmd, gnd -> gnm", dhess_psi, s ** 2)
            # Growth term: r * psi -> [G, N, M]
            growth_term = r.unsqueeze(-1) * psi_k

            gen_psi = drift_term + diff_term + growth_term   # [G, N, M]

            # E_hat[w_k * L_g psi_m]
            E_gen = torch.einsum("gn, gnm -> gm", wk, gen_psi)  # [G, M]

            # Time derivative of E_hat[w * psi_m]
            dE_dtau = (E_psi[k + 1] - E_psi[k]) / dtau[k]  # [G, M]

            # Residual
            R = dE_dtau - E_gen   # [G, M]
            residuals_sq.append((R ** 2).mean())

        return torch.stack(residuals_sq).mean()
