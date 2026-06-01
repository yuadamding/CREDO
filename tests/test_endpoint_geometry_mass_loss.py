from __future__ import annotations

import torch

from credo.losses.uot import (
    EndpointGeometryMassLoss,
    UOTLoss,
    endpoint_geometry_mass_components,
    endpoint_geometry_mass_loss,
    sinkhorn_divergence,
    sinkhorn_divergence_components,
)


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
    old_loss = UOTLoss(eps=0.2, tau=0.7, max_iter=40, use_geomloss=False)
    new_total, new_components = new_loss.component_dict(pred_z, pred_logw, target_support, target_logw, ["p"])
    old_total, old_components = old_loss.component_dict(pred_z, pred_logw, target_support, target_logw, ["p"])

    assert torch.allclose(new_total, old_total)
    for key in new_components["p"]:
        assert torch.allclose(new_components["p"][key], old_components["p"][key])
