"""Weak-form PINN residual loss.

For diagonal diffusion and normalized within-perturbation laws, the generator
of the process applied to a test function psi_m(z) is:

    L_g psi = grad(psi) . v_g  +  (1/2) sum_j sigma_{g,j}^2 d^2psi/dz_j^2
              + (r_g - E_g[r_g]) * psi

We use Gaussian RBF test functions:
    psi_m(z) = exp(-||z - c_m||^2 / (2 h_m^2))

for which the gradients and Hessian diagonals have analytic forms.

The residual is:
    R_{g,m,k} = [E_hat[w_{k+1} psi_m(Z_{k+1})] - E_hat[w_k psi_m(Z_k)]] / dtau
                - E_hat[w_k L_g psi_m(Z_k)]

and the loss is sum_{g,m,k} R_{g,m,k}^2.

The centering of the reaction term is intentional: total growth changes the
finite measure mass, while only relative growth reshapes the normalized
conditional law regularized here.

Memory-efficient implementation: avoids materializing [G, N, M, d] tensors.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GaussianRBFTestFunctions:
    """Analytic Gaussian RBF test functions with exact first/second derivatives.

    Memory-efficient: provides contracted operations that avoid materializing
    the full [G, N, M, d] gradient/Hessian tensors.
    """

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

    def _diff_and_psi(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z: [G, N, d] -> diff [G, N, M, d], psi [G, N, M]."""
        diff = z.unsqueeze(-2) - self.centers  # [G, N, M, d]
        sq_dist = (diff ** 2).sum(-1)          # [G, N, M]
        psi = torch.exp(-sq_dist / (2 * self.h ** 2))
        return diff, psi

    def psi(self, z: torch.Tensor) -> torch.Tensor:
        """z: [G, N, d] -> [G, N, M]."""
        _, psi = self._diff_and_psi(z)
        return psi

    def grad_psi(self, z: torch.Tensor) -> torch.Tensor:
        """Analytic gradient of psi with shape [G, N, M, d]."""
        diff, psi = self._diff_and_psi(z)
        return -(diff / (self.h ** 2)) * psi.unsqueeze(-1)

    def hess_diag_psi(self, z: torch.Tensor) -> torch.Tensor:
        """Analytic Hessian diagonal of psi with shape [G, N, M, d]."""
        diff, psi = self._diff_and_psi(z)
        h2 = self.h ** 2
        return psi.unsqueeze(-1) * ((diff ** 2) / (h2 ** 2) - (1.0 / h2))

    def generator_contracted(
        self,
        z: torch.Tensor,        # [G, N, d]
        v: torch.Tensor,        # [G, N, d]  drift
        sigma: torch.Tensor,    # [G, N, d]  diffusion std
        r: torch.Tensor,        # [G, N]     centered growth for normalized law
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute psi(z) and E_gen = w . (L_g psi) without [G,N,M,d] intermediates.

        Returns:
            psi: [G, N, M]
            gen_psi: [G, N, M]  where gen_psi[g,n,m] = (grad_psi . v + 0.5*sigma^2.hess_psi + r_centered*psi)
        """
        diff, psi = self._diff_and_psi(z)  # [G, N, M, d], [G, N, M]
        h2 = self.h ** 2

        # grad_psi . v = -(1/h^2) * psi * sum_j diff_j * v_j
        # = -(1/h^2) * psi * (diff . v)
        # diff: [G, N, M, d], v: [G, N, d] -> dot: [G, N, M]
        diff_dot_v = (diff * v.unsqueeze(-2)).sum(-1)  # [G, N, M]
        drift_term = -(1.0 / h2) * psi * diff_dot_v   # [G, N, M]

        # 0.5 * sigma^2 . diag_hess_psi
        # diag_hess_psi_j = psi * ((diff_j/h^2)^2 - 1/h^2)
        # contracted: 0.5 * sum_j sigma_j^2 * psi * ((diff_j/h^2)^2 - 1/h^2)
        # = 0.5 * psi * (sum_j sigma_j^2 * (diff_j^2/h^4 - 1/h^2))
        sigma_sq = sigma ** 2  # [G, N, d]
        diff_sq = diff ** 2    # [G, N, M, d]

        # sum_j sigma_j^2 * diff_j^2 / h^4  -> [G, N, M]
        term1 = (sigma_sq.unsqueeze(-2) * diff_sq).sum(-1) / (h2 ** 2)
        # sum_j sigma_j^2 / h^2  -> [G, N]
        term2 = sigma_sq.sum(-1) / h2  # [G, N]
        diff_term = 0.5 * psi * (term1 - term2.unsqueeze(-1))  # [G, N, M]

        # Delete diff to free memory before growth term
        del diff, diff_sq, diff_dot_v

        # Relative growth term: r_centered * psi -> [G, N, M].
        # The weak-form loss uses normalized weights, so r must be centered
        # before reaching this contracted generator.
        growth_term = r.unsqueeze(-1) * psi  # [G, N, M]

        gen_psi = drift_term + diff_term + growth_term  # [G, N, M]
        return psi, gen_psi


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
        """Sample new test function centers and adapt bandwidth to the data."""
        G, N, d = z_ref.shape
        z_flat = z_ref.detach().float().reshape(-1, d)
        z_min = z_flat.min(0).values
        z_max = z_flat.max(0).values
        centers = z_min + torch.rand(self.M, d, device=z_ref.device) * (z_max - z_min)
        self._centers = centers

        # Adapt bandwidth to the data scale so RBFs have non-vanishing values.
        n_sub = min(512, z_flat.shape[0])
        idx = torch.randperm(z_flat.shape[0], device=z_flat.device)[:n_sub]
        sub = z_flat[idx]  # [n_sub, d]
        sq_dists = ((sub.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)  # [n_sub, M]
        median_dist = sq_dists.median().sqrt().item()
        # Set bandwidth so that the RBF at median distance ≈ exp(-0.5) ≈ 0.6
        self._adaptive_bandwidth = max(median_dist, 1.0)
        self._centers_initialized = True

    def _get_test_fns(self, z_ref: torch.Tensor) -> GaussianRBFTestFunctions:
        if not self._centers_initialized:
            self.refresh_test_functions(z_ref)
        bw = getattr(self, "_adaptive_bandwidth", self.bandwidth)
        return GaussianRBFTestFunctions(
            centers=self._centers.to(z_ref.device, z_ref.dtype),
            bandwidth=bw,
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
        """Compute sum of squared residuals.

        All computation is done in float32 for numerical stability.
        """
        # Cast all inputs to float32 for numerical stability (fp16 rollouts)
        z_steps = z_steps.float()
        logw_steps = logw_steps.float()
        drift_steps = drift_steps.float()
        sigma_steps = sigma_steps.float()
        growth_steps = growth_steps.float()
        tau_steps = tau_steps.float()

        K_plus1, G, N, d = z_steps.shape
        K = K_plus1 - 1

        if refresh_centers:
            self.refresh_test_functions(z_steps[0])

        test_fns = self._get_test_fns(z_steps[0])

        # Normalized weights at each step: [K+1, G, N]
        logw_norm = logw_steps - torch.logsumexp(logw_steps, dim=-1, keepdim=True)
        w_norm = logw_norm.exp()

        # Compute E_psi at step 0 for the first residual
        psi_prev = test_fns.psi(z_steps[0])  # [G, N, M]
        E_psi_prev = torch.einsum("gn, gnm -> gm", w_norm[0], psi_prev)  # [G, M]

        residual_sum = torch.tensor(0.0, device=z_steps.device)

        for k in range(K):
            dtau_k = tau_steps[k + 1] - tau_steps[k]

            # The weak-form residual regularizes the normalized conditional
            # law for each perturbation. Center growth under the same
            # normalized weights; common growth should change mass only, not
            # the within-perturbation state distribution.
            growth_k = growth_steps[k]
            growth_mean_k = (w_norm[k] * growth_k).sum(dim=-1, keepdim=True)
            growth_centered_k = growth_k - growth_mean_k

            # Generator applied to test functions at step k
            # Uses memory-efficient contracted computation
            psi_k, gen_psi_k = test_fns.generator_contracted(
                z_steps[k], drift_steps[k], sigma_steps[k], growth_centered_k
            )

            # E_hat[w_k * L_g psi_m]
            E_gen = torch.einsum("gn, gnm -> gm", w_norm[k], gen_psi_k)  # [G, M]
            del gen_psi_k  # free immediately

            # E_psi at step k+1
            psi_next = test_fns.psi(z_steps[k + 1])  # [G, N, M]
            E_psi_next = torch.einsum("gn, gnm -> gm", w_norm[k + 1], psi_next)

            # Residual: dE/dtau - E[L_g psi]
            dE_dtau = (E_psi_next - E_psi_prev) / dtau_k  # [G, M]
            R = dE_dtau - E_gen   # [G, M]
            residual_sum = residual_sum + (R ** 2).mean()

            # Advance
            E_psi_prev = E_psi_next

        return residual_sum / K
