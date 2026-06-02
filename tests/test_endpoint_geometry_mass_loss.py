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
