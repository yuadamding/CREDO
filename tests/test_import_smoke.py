from __future__ import annotations


def test_credo_model_stack_imports() -> None:
    import cape
    import credo
    from cape.models.full_model import FullDynamicsModel
    from cape.training.trainer import Trainer

    assert cape.__version__ == "1.1.2"
    assert credo.__version__ == "1.1.2"
    assert FullDynamicsModel.__name__ == "FullDynamicsModel"
    assert Trainer.__name__ == "Trainer"
