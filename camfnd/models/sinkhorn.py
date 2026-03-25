from __future__ import annotations

import torch
from torch import Tensor


def _ensure_column(x: Tensor) -> Tensor:
    if x.ndim != 2:
        raise ValueError("Support tensors must have shape [N, d].")
    return x


def pairwise_sqeuclidean(x: Tensor, y: Tensor) -> Tensor:
    x = _ensure_column(x)
    y = _ensure_column(y)
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).T
    return torch.clamp(x2 + y2 - 2.0 * x @ y.T, min=0.0)


def kl_div(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    return (p_safe * (torch.log(p_safe) - torch.log(q_safe)) - p_safe + q_safe).sum()


def unbalanced_ot_cost(
    x: Tensor,
    a: Tensor,
    y: Tensor,
    b: Tensor,
    *,
    epsilon: float = 0.05,
    tau: float = 0.5,
    max_iters: int = 150,
    tol: float = 1e-8,
) -> Tensor:
    """Entropic unbalanced OT cost for positive discrete measures.

    This implementation is intentionally compact and suitable for the small
    synthetic Stage-I benchmark used in Step 3.
    """

    x = _ensure_column(x)
    y = _ensure_column(y)
    a = a.reshape(-1)
    b = b.reshape(-1)
    if torch.any(a <= 0) or torch.any(b <= 0):
        raise ValueError("Discrete weights must be strictly positive for this implementation.")
    if epsilon <= 0 or tau <= 0:
        raise ValueError("epsilon and tau must be positive.")

    C = pairwise_sqeuclidean(x, y)
    K = torch.exp(-C / epsilon)
    rho = tau / (tau + epsilon)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(max_iters)):
        u_prev = u
        Kv = torch.clamp(K @ v, min=1e-12)
        u = (a / Kv) ** rho
        KTu = torch.clamp(K.T @ u, min=1e-12)
        v = (b / KTu) ** rho
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    P = torch.clamp(u[:, None] * K * v[None, :], min=1e-32)
    marg_x = P.sum(dim=1)
    marg_y = P.sum(dim=0)
    ref = a[:, None] * b[None, :]
    cost = (P * C).sum()
    cost = cost + epsilon * kl_div(P, ref)
    cost = cost + tau * kl_div(marg_x, a)
    cost = cost + tau * kl_div(marg_y, b)
    return cost


def unbalanced_sinkhorn_divergence(
    x: Tensor,
    a: Tensor,
    y: Tensor,
    b: Tensor,
    *,
    epsilon: float = 0.05,
    tau: float = 0.5,
    max_iters: int = 150,
    tol: float = 1e-8,
) -> Tensor:
    ab = unbalanced_ot_cost(x, a, y, b, epsilon=epsilon, tau=tau, max_iters=max_iters, tol=tol)
    aa = unbalanced_ot_cost(x, a, x, a, epsilon=epsilon, tau=tau, max_iters=max_iters, tol=tol)
    bb = unbalanced_ot_cost(y, b, y, b, epsilon=epsilon, tau=tau, max_iters=max_iters, tol=tol)
    return ab - 0.5 * aa - 0.5 * bb


def normalized_geometry_loss(
    x: Tensor,
    a: Tensor,
    y: Tensor,
    b: Tensor,
    *,
    epsilon: float = 0.05,
    tau: float = 0.5,
    max_iters: int = 150,
    tol: float = 1e-8,
) -> Tensor:
    a_norm = a / a.sum()
    b_norm = b / b.sum()
    return unbalanced_sinkhorn_divergence(x, a_norm, y, b_norm, epsilon=epsilon, tau=tau, max_iters=max_iters, tol=tol)
