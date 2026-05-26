from __future__ import annotations


def test_credo_model_stack_imports() -> None:
    import credo
    from credo.config import DataConfig, EvalConfig, RunConfig
    from credo.data import EndpointProblem as PublicEndpointProblem
    from credo.data.problems import EndpointProblem, TrajectoryProblem
    from credo.losses import MultiTimeEndpointLoss
    from credo.losses.trajectory import make_observed_tau_grid
    from credo.models import FullDynamicsModel as PublicFullDynamicsModel
    from credo.models.full_model import FullDynamicsModel
    from credo.models.particles import WeightedParticleSimulator
    from credo.training import Trainer as PublicTrainer
    from credo.training.trainer import Trainer

    assert credo.__version__ == "1.1.3"
    assert FullDynamicsModel.__name__ == "FullDynamicsModel"
    assert Trainer.__name__ == "Trainer"
    assert PublicFullDynamicsModel is FullDynamicsModel
    assert PublicTrainer is Trainer
    assert PublicEndpointProblem is EndpointProblem
    assert MultiTimeEndpointLoss.__name__ == "MultiTimeEndpointLoss"
    assert make_observed_tau_grid.__name__ == "make_observed_tau_grid"
    assert DataConfig.__name__ == "DataConfig"
    assert EvalConfig.__name__ == "EvalConfig"
    assert RunConfig.__name__ == "RunConfig"
