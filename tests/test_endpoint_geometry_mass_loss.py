from __future__ import annotations

import torch
import pytest

from credo.losses.endpoint import EndpointGeometryMassLoss as PublicEndpointGeometryMassLoss
from credo.losses.uot import (
    EndpointGeometryMassLoss,
    UOTLoss,
    endpoint_geometry_mass_components,
    endpoint_geometry_mass_loss,
    sinkhorn_divergence,
    sinkhorn_divergence_components,
    sinkhorn_divergence_normalized,
)


pytestmark = pytest.mark.unit


def _measures():
    x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    y = torch.tensor([[0.1], [1.1], [2.1]], dtype=torch.float32)
    log_a = torch.log(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    log_b = torch.log(torch.tensor([1.5, 2.0, 4.0], dtype=torch.float32))
    return x, log_a, y, log_b


def test_uot_alias_matches_endpoint_geometry_mass_loss() -> None:
    x, log_a, y, log_b = _measures()
    new = endpoint_geometry_mass_loss(x, log_a, y, log_b, eps=0.2, tau=0.7, max_iter=40)
    old = sinkhorn_divergence(x, log_a, y, log_b, eps=0.2, tau=0.7, max_iter=40)

    assert torch.allclose(new, old)


def test_component_alias_includes_log_mass_terms() -> None:
    x, log_a, y, log_b = _measures()
    components = endpoint_geometry_mass_components(x, log_a, y, log_b, eps=0.2, tau=0.7, max_iter=40)
    alias_components = sinkhorn_divergence_components(x, log_a, y, log_b, eps=0.2, tau=0.7, max_iter=40)

    assert set(components) == {"geom", "mass", "log_mass_pred", "log_mass_target", "total"}
    for key in components:
        assert torch.allclose(components[key], alias_components[key])
    assert torch.allclose(components["total"], components["geom"] + components["mass"])


def test_uotloss_class_alias_matches_new_module() -> None:
    x, log_a, y, log_b = _measures()
    pred_z = x.unsqueeze(0)
    pred_logw = log_a.unsqueeze(0)
    target_support = {"p": y}
    target_logw = {"p": log_b}

    new_loss = EndpointGeometryMassLoss(eps=0.2, tau=0.7, max_iter=40, use_geomloss=False)
    with pytest.warns(DeprecationWarning, match="EndpointGeometryMassLoss"):
        old_loss = UOTLoss(eps=0.2, tau=0.7, max_iter=40, use_geomloss=False)
    new_total, new_components = new_loss.component_dict(pred_z, pred_logw, target_support, target_logw, ["p"])
    old_total, old_components = old_loss.component_dict(pred_z, pred_logw, target_support, target_logw, ["p"])

    assert torch.allclose(new_total, old_total)
    for key in new_components["p"]:
        assert torch.allclose(new_components["p"][key], old_components["p"][key])


def test_endpoint_module_is_public_import_home() -> None:
    assert PublicEndpointGeometryMassLoss is EndpointGeometryMassLoss


def test_geometry_backend_invariance() -> None:
    """geomloss and the manual log-domain fallback must agree on the geometry term.

    Before the cost-convention fix the manual fallback used the full squared
    Euclidean cost while geomloss (p=2) uses 0.5*||x-y||^2, so the two backends
    differed by a factor of ~2 and the reported divergence depended on whether
    geomloss was installed. They should now agree to within solver tolerance.
    """
    pytest.importorskip("geomloss")
    torch.manual_seed(0)
    x = torch.randn(48, 3)
    y = torch.randn(40, 3) + 0.4
    a = torch.softmax(torch.randn(48), dim=0)
    b = torch.softmax(torch.randn(40), dim=0)
    eps = 0.3

    geom_loss = EndpointGeometryMassLoss(eps=eps, use_geomloss=True, max_iter=500)
    fallback = EndpointGeometryMassLoss(eps=eps, use_geomloss=False, max_iter=500)
    assert geom_loss._geomloss_fn is not None  # geomloss actually active

    g_geomloss = geom_loss._geometry(x, a, y, b)
    g_fallback = fallback._geometry(x, a, y, b)

    assert float(g_geomloss) > 0 and float(g_fallback) > 0
    ratio = float(g_geomloss) / float(g_fallback)
    # Regression guard against the ~2x scale mismatch (would give ratio ~0.5 or ~2.0).
    assert 0.8 < ratio < 1.25, f"backend geometry mismatch, ratio={ratio:.3f}"
    assert torch.allclose(g_geomloss, g_fallback, rtol=0.15, atol=1e-3)


def test_sinkhorn_divergence_self_is_zero_and_nonnegative() -> None:
    """Debiasing must give self-divergence ~ 0 and a non-negative divergence."""
    torch.manual_seed(1)
    x = torch.randn(30, 4)
    a = torch.softmax(torch.randn(30), dim=0)

    self_div = sinkhorn_divergence_normalized(x, a, x, a, eps=0.5, max_iter=500)
    assert float(self_div) >= 0.0
    assert float(self_div) < 1e-4  # ~0 by construction of the debiased divergence

    y = torch.randn(28, 4) + 1.0
    b = torch.softmax(torch.randn(28), dim=0)
    div = sinkhorn_divergence_normalized(x, a, y, b, eps=0.5, max_iter=500)
    assert float(div) >= 0.0


def test_sinkhorn_divergence_increases_with_separation() -> None:
    """Geometry must grow as the target measure is translated away."""
    torch.manual_seed(2)
    x = torch.randn(32, 3)
    a = torch.softmax(torch.randn(32), dim=0)
    b = torch.softmax(torch.randn(32), dim=0)

    near = sinkhorn_divergence_normalized(x, a, x + 0.5, b, eps=0.5, max_iter=500)
    far = sinkhorn_divergence_normalized(x, a, x + 4.0, b, eps=0.5, max_iter=500)
    assert float(far) > float(near)


def test_missing_endpoint_target_fails_by_default() -> None:
    x, log_a, _, _ = _measures()
    loss_fn = EndpointGeometryMassLoss(eps=0.2, max_iter=20, use_geomloss=False)

    with pytest.raises(KeyError, match="Missing endpoint targets"):
        loss_fn.component_dict(
            pred_z=x.unsqueeze(0),
            pred_logw_abs=log_a.unsqueeze(0),
            target_support={},
            target_logw={},
            perturbation_ids=["missing"],
        )


def test_missing_endpoint_target_can_be_masked_for_sparse_batches() -> None:
    x, log_a, y, log_b = _measures()
    loss_fn = EndpointGeometryMassLoss(eps=0.2, max_iter=20, use_geomloss=False)

    total, components = loss_fn.component_dict(
        pred_z=torch.stack([x, x], dim=0),
        pred_logw_abs=torch.stack([log_a, log_a], dim=0),
        target_support={"present": y},
        target_logw={"present": log_b},
        perturbation_ids=["present", "missing"],
        fail_on_missing_target=False,
    )

    assert torch.isfinite(total)
    assert list(components) == ["present"]
