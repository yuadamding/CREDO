"""New-study structural variants; no cohort data or CUDA needed."""

import pytest
import torch

from credo.recipes.compact_sde_v3.model import CREDOModel
from credo.recipes.compact_sde_v3.recipe import CompactModelConfig


def make_model(**overrides):
    settings = dict(
        embedding_ids=("NTC", "target"),
        control_embedding_ids=("NTC",),
        latent_dim=4,
        hidden_dim=16,
        embedding_dim=3,
        context_mode="none",
    )
    settings.update(overrides)
    return CREDOModel(**settings)


@pytest.mark.parametrize("kind,depth", [("mlp", 2), ("mlp", 3), ("residual_mlp", 3)])
def test_drift_diffusion_reaction_are_trainable(kind, depth):
    torch.manual_seed(13)
    model = make_model(trunk_kind=kind, trunk_depth=depth)
    model.set_phase("mass")
    z = torch.randn(2, 7, 4)
    result = model(z, torch.tensor(0.4), ("NTC", "target"))
    assert result.drift.shape == result.sigma_diag.shape == z.shape
    assert result.growth.shape == (2, 7)
    assert torch.all(result.sigma_diag > 0)
    assert torch.all(result.growth.abs() <= model.growth_max)
    loss = result.drift.square().mean() + result.sigma_diag.mean() + result.growth.square().mean()
    loss.backward()
    for head in (model.drift_reference, model.sigma_reference, model.growth_reference):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    model.assert_soft_reference()
    assert torch.equal(model.residuals(("NTC",)), torch.zeros(1, 3))


@pytest.mark.parametrize("kind,depth", [("mlp", 2), ("mlp", 4), ("residual_mlp", 3)])
def test_constructor_and_weights_roundtrip(kind, depth):
    model = make_model(trunk_kind=kind, trunk_depth=depth)
    rebuilt = CREDOModel(**model.architecture())
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    z = torch.randn(2, 5, 4)
    a = model(z, torch.tensor(0.25), ("NTC", "target"))
    b = rebuilt(z, torch.tensor(0.25), ("NTC", "target"))
    for field in ("drift", "sigma_diag", "growth", "programs"):
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)


def test_released_default_constructor_shape_preserved():
    model = make_model()
    assert "trunk_kind" not in model.architecture()
    assert "trunk_depth" not in model.architecture()
    assert len(model.trunk) == 4


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_invalid_depth_rejected(depth):
    with pytest.raises(ValueError):
        make_model(trunk_depth=depth)
    with pytest.raises(ValueError):
        CompactModelConfig(trunk_depth=depth)


def test_unknown_trunk_rejected():
    with pytest.raises(ValueError, match="trunk_kind"):
        make_model(trunk_kind="unknown")


def test_context_disabled_even_if_supplied():
    model = make_model(trunk_kind="residual_mlp", trunk_depth=3)
    z = torch.randn(2, 5, 4)
    a = model(z, torch.tensor(0.4), ("NTC", "target"))
    b = model(z, torch.tensor(0.4), ("NTC", "target"), context=torch.randn(2, 8))
    torch.testing.assert_close(a.growth, b.growth, rtol=0, atol=0)
